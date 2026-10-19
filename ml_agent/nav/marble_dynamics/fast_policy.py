"""The learned model's deterministic act for the planner, without the training-only work (2026-10-19, search speed-up
toward real-time play).

NavActorCritic.act(deterministic=True) also builds the critic, the range penalty, five distributions and their
log-probabilities, and spends ~100 small torch ops on gates and thresholds; per reply of the learned proposal's rollout
that cost 1.3 ms. deterministic_act returns the same six action values and next hidden state with ~0.45 ms: the layers
run as the same torch kernels on the same tensors (F.conv2d / F.linear / torch.gru_cell on the module weights), and
the heads' gates and thresholds run in Python with float32-exact arithmetic (a NumPy float32 scalar rounds like torch's
float32 kernels; torch compares a float32 tensor with a Python number in float32). sigmoid(x) > p is decided by the sign
of x - logit(p) beyond a 1e-3 margin (the float32 sigmoid is within a few ulps of 0.5 there), by the torch chain inside.
Bit-identical to model.act on every captured rollout (scratchpad bench_nav.py: 0 of 17 rollouts, 1,632 replies)."""
import math

import numpy as np
import torch
import torch.nn.functional as F

import nav.model as M

F32 = np.float32
_GUARD = 1e-3
_PARAMS = {}


def _params(model):
    p = _PARAMS.get(id(model))
    if p is not None and p['model'] is model:
        return p
    c = model.conv
    assert all(c[i].padding_mode == 'zeros' for i in (0, 2, 4))
    assert isinstance(c[1], torch.nn.ReLU) and isinstance(c[6], torch.nn.Flatten) and c[6].start_dim == 1 and c[6].end_dim == -1
    lin = lambda m: (m.weight, m.bias)
    p = dict(model=model, conv=[(c[i].weight, c[i].bias, c[i].stride, c[i].padding, c[i].dilation, c[i].groups) for i in (0, 2, 4)],
             fc=lin(c[7]), vec=lin(model.vec[0]), pre=lin(model.pre[0]),
             gru=(model.gru.weight_ih, model.gru.weight_hh, model.gru.bias_ih, model.gru.bias_hh),
             dir=(lin(model.dir_head[0]), lin(model.dir_head[2])), ss=(lin(model.ss_res[0]), lin(model.ss_res[2])),
             thr=(lin(model.throttle_head[0]), lin(model.throttle_head[2])), jump=(lin(model.jump_head[0]), lin(model.jump_head[2])),
             brake=(lin(model.brake_head[0]), lin(model.brake_head[2])), use=(lin(model.use_head[0]), lin(model.use_head[2])))
    _PARAMS[id(model)] = p
    return p


def _mlp2(x, layers):
    (w0, b0), (w1, b1) = layers
    return F.linear(torch.relu(F.linear(x, w0, b0)), w1, b1)


def _lt(v, thr):
    return float(v) < float(F32(thr))


def _gt(v, thr):
    return float(v) > float(F32(thr))


def _ge(v, thr):
    return float(v) >= float(F32(thr))


def _sig_gt(x, p, chain=None):
    """sigmoid(x) > p in float32 torch arithmetic, x a float32 value."""
    if 0.0 < p < 1.0:
        edge = math.log(p / (1.0 - p))
        if x - edge > _GUARD:
            return True
        if x - edge < -_GUARD:
            return False
    t = chain if chain is not None else torch.tensor([x], dtype=torch.float32)
    return bool((torch.sigmoid(t) > p)[0])


def supported(model):
    return isinstance(model, M.NavActorCritic) and not (M.SS_RES_APPROACH or M.SS_RES_HELD or model.JUMP_ON_PRIOR or not M.BRAKE_ENABLED)


def deterministic_act(model, crop_np, vec_np, h):
    """(model.act(crop, vec, h, deterministic=True)['action_game'][0].tolist(), its h_next) for one float32 crop
    (6, 32, 32) and observation vector, inside torch.inference_mode()."""
    if not supported(model):
        out = model.act(torch.from_numpy(crop_np).unsqueeze(0), torch.from_numpy(vec_np).unsqueeze(0), h, deterministic=True)
        return out['action_game'][0].tolist(), out['h_next']
    p = _params(model)
    vec = vec_np
    v_in = vec.copy(); v_in[M.VEC_USE_TRY] = vec[M.VEC_USE_TRY] * F32(0.0)     # vec * model._input_mask(vec)
    # core: the encoder and the recurrent cell
    x = torch.from_numpy(crop_np).unsqueeze(0)
    for (w, b, st, pd, dl, gr) in p['conv']:
        x = torch.relu(F.conv2d(x, w, b, st, pd, dl, gr))
    x = torch.relu(F.linear(x.flatten(1, -1), *p['fc']))
    x = torch.cat([x, torch.relu(F.linear(torch.from_numpy(v_in).unsqueeze(0), *p['vec']))], dim=1)
    h1 = torch.gru_cell(torch.relu(F.linear(x, *p['pre'])), h, *p['gru'])
    # heads: steering (with the gated Super Speed residual), throttle
    vin2 = v_in
    if M.DIR_GOAL_GAIN != 1.0:
        vin2 = v_in.copy(); vin2[0:2] = v_in[0:2] * F32(M.DIR_GOAL_GAIN)
    x_dir = torch.cat([h1, torch.from_numpy(vin2).unsqueeze(0)], dim=-1)
    g = _gt(vec[M.VEC_USE + 2], 0.5) or _lt(vec[M.VEC_USE + 5], M.SS_RES_RECENT_S / M.USE_T_SCALE)
    tanh_dir = torch.tanh(_mlp2(x_dir, p['dir']))
    res2 = None
    if g:
        res = _mlp2(x_dir, p['ss']) * M.SS_RES_SCALE
        mean_xy = tanh_dir + res[..., 0:2]; res2 = F32(res[0, 2].item())
    else:
        mean_xy = tanh_dir                                                      # + 0 x the gated residual
    thr = _mlp2(h1, p['thr']).squeeze(-1)
    throttle = model.THROTTLE_FLOOR + (1 - model.THROTTLE_FLOOR) * torch.sigmoid(thr)
    # jump: the gap prior (an edge toward the waypoint, ready, the landing approved), the soft ceiling, the exploration floor
    edge_u = F32(vec[M.VEC_GOAL_RAY_CLEAR]) * F32(vec[M.VEC_GOAL_DIST]) * F32(50.0)
    gp = (_lt(edge_u, M.JUMP_PRIOR_DIST) and _lt(vec[M.VEC_GOAL_RAY_CLEAR], 0.999) and _lt(vec[M.VEC_GOAL_RAY_BEYOND], M.JUMP_PRIOR_DROP)
          and _gt(vec[M.VEC_ON_FLOOR], 0.5) and _gt(F32(vec[M.VEC_SPEED]) * F32(20.0), M.JUMP_PRIOR_SPEED) and _gt(vec[M.VEC_GAP + 2], 0.5))
    jl = _mlp2(h1, p['jump']).squeeze(-1)
    jr = F32(jl.item()) - F32(M.JUMP_DAMP)
    if float(jr) > 0.0:
        xt = torch.tensor([float(jr)], dtype=torch.float32)
        jv = F32((xt + (M.JUMP_CAP * torch.tanh(xt / M.JUMP_CAP) - xt)).item())
    else:
        jv = jr                                                                 # x + (x - x) = x below the ceiling
    if gp:
        jv = jv + F32(M.JUMP_PRIOR)
    chain = None
    if M.JUMP_EXPLORE_P > 0.0 or M.JUMP_EXPLORE_BONUS != 0.0 or M.JUMP_FORCE_P > 0.0:
        near_drop = any(_lt(vec[10 + 2 * k], M.JUMP_EXPLORE_CLEAR_U / 20.0) and _lt(vec[11 + 2 * k], M.JUMP_EXPLORE_DROP) for k in range(16))
        safe = _gt(vec[M.VEC_ON_FLOOR], 0.5) and not near_drop
        rise = False
        if safe:
            moving = float(F32(vec[M.VEC_SPEED]) * F32(20.0)) > 1.0
            hx = float(vec[4] if moving else vec[0]); hy = float(vec[5] if moving else vec[1])
            q = math.atan2(hy, hx) / (2.0 * math.pi / 16)
            if abs(q - math.floor(q) - 0.5) > 1e-4:
                ridx = int(round(q)) % 16
            else:                                                               # at a rounding edge: the torch chain
                ridx = int((torch.round(torch.atan2(torch.tensor([hy], dtype=torch.float32), torch.tensor([hx], dtype=torch.float32))
                                        / (2.0 * math.pi / 16)).long() % 16)[0])
            for off in (-1, 0, 1):
                k = (ridx + off) % 16
                if _lt(vec[10 + 2 * k], M.JUMP_EXPLORE_RISE_U / 20.0) and _gt(vec[11 + 2 * k], M.JUMP_EXPLORE_RISE_DZ):
                    rise = True
        if rise and safe and M.JUMP_EXPLORE_BONUS != 0.0:
            jv = jv + F32(M.JUMP_EXPLORE_BONUS)
        if M.JUMP_EXPLORE_P > 0.0 and safe and not rise:
            jt = torch.tensor([float(jv)], dtype=torch.float32)
            pj_learn = torch.sigmoid(jt)
            pj = (pj_learn + M.JUMP_EXPLORE_P * (1.0 - pj_learn)).clamp(1e-6, 1.0 - 1e-6)
            chain = torch.log(pj) - torch.log1p(-pj)
            jv = F32(chain.item())
    j = 1.0 if _sig_gt(float(jv), M.JUMP_DET_THRESHOLD, chain) else 0.0
    # brake (the learned part only, as deterministic play reads it)
    bl = F32(_mlp2(h1, p['brake']).squeeze(-1).item()) - F32(1.0 if gp else 0.0) * F32(M.BRAKE_SUPPRESS)
    bl = min(max(bl, F32(-7.0)), F32(3.0))
    if res2 is not None:
        bl = bl + res2
    b = 1.0 if _sig_gt(float(bl), 0.5) else 0.0
    # use: only where the use prior allows one
    held = vec[M.VEC_POW:M.VEC_POW + 7]
    if not all(float(v) in (0.0, 1.0) for v in held):
        up = bool(model.use_prior(torch.from_numpy(vec).unsqueeze(0))[0] > 0.5)
    else:
        up = sum(float(v) for v in held) > 0.5 or _ge(vec[M.VEC_POW + 7], 0.2) or _gt(vec[M.VEC_POW + 8], 0.5)
    u = 0.0
    if up:
        aux = np.concatenate([vec[0:10], vec[M.VEC_POW:M.VEC_POW + 37]])
        raw = _mlp2(torch.cat([h1, torch.from_numpy(aux).unsqueeze(0)], dim=-1), p['use']).squeeze(-1)
        xu = float(F32(raw.item()) - F32(model.use_damp.item()))
        if xu > _GUARD:
            u = 1.0
        elif xu >= -_GUARD:
            x_use = raw - model.use_damp
            z_use = torch.where(x_use > 0, M.ceiling_st(x_use, M.USE_CAP), M.USE_LOGIT_CAP * torch.tanh(x_use / M.USE_LOGIT_CAP))
            u = 1.0 if bool((torch.sigmoid(z_use) > 0.5)[0]) else 0.0
    m = mean_xy[0].tolist()
    return [m[0], m[1], throttle.item(), j, b, u], h1
