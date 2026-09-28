"""Stage 3 M2 report for the flight head: path, takeoff, landing and safe-outcome accuracy by situation.

    python -m nav.learned_nav.eval_flight3        -> logs/learned_nav/stage3_flight_eval.json

Sets: validation blocks of the training maps, and each held-out development map. Slices (from the sample's inputs):
jump now (the jump key in the first reply) vs run-up first, flat vs slope start (> 5 deg), an exact edge within 3 u
straight ahead, and speed bands.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav import flight3 as F3                                       # noqa: E402
from nav.learned_nav import dynamics3 as D3                                     # noqa: E402

OUT = os.path.join(HERE, 'logs', 'learned_nav', 'stage3_flight_eval.json')


def auc(p, y):
    y = y.astype(bool)
    if y.all() or (~y).all():
        return None
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def evaluate(model, F, T, dev):
    import torch
    outs = {'path': [], 'take': [], 'landed': [], 'land': [], 'safe': []}
    with torch.no_grad():
        for b in range(0, len(F), 8192):
            o = model(torch.as_tensor(F[b:b + 8192].astype(np.float32), device=dev))
            for k in outs:
                outs[k].append(o[k].cpu().numpy())
    o = {k: np.concatenate(v) for k, v in outs.items()}
    H = F3.H
    path = T[:, :H * 3].reshape(-1, H, 3); mask = T[:, F3.OFF_MASK:F3.OFF_MASK + H]
    perr = np.linalg.norm(o['path'] - path, axis=2)
    take_t = T[:, F3.OFF_TAKE].astype(int); take_p = o['take'].argmax(1); take_p = np.where(take_p == H, -1, take_p)
    landed = T[:, F3.OFF_LANDED] > 0.5
    lerr = np.linalg.norm(o['land'] - T[:, F3.OFF_LP:F3.OFF_LP + 3], axis=1)
    safe = T[:, F3.OFF_SAFE] > 0.5; psafe = 1 / (1 + np.exp(-o['safe']))
    jump_now = F[:, F3.N_STATE + 5].astype(np.float32) > 0.5            # jump key of the first new reply (after pending)
    slope = np.degrees(np.arccos(np.clip(F[:, 9].astype(np.float32), -1, 1)))
    speed = F[:, 11].astype(np.float32) * 10
    ray0 = F[:, F3.N_STATE + F3.N_SEQ + 3 * len(F3.CF) * len(F3.CL) + 7].astype(np.float32) * 16    # the straight-ahead ray
    slices = {'all': np.ones(len(F), bool), 'jump_now': jump_now, 'run_up_first': ~jump_now, 'flat_start': slope <= 5,
              'slope_start': slope > 5, 'edge_ahead_3u': ray0 < 3, 'speed_0_6': speed < 6, 'speed_6+': speed >= 6}
    res = {}
    for k, m in slices.items():
        if m.sum() < 50:
            continue
        pm = perr[m]; mm = mask[m] > 0
        by_t = [float(np.median(pm[:, t][mm[:, t]])) if mm[:, t].any() else None for t in range(H)]
        res[k] = {'n': int(m.sum()), 'path_err_median_u': float(np.median(pm[mm])), 'path_err_p90_u': float(np.percentile(pm[mm], 90)),
                  'path_err_by_decision_median': by_t,
                  'takeoff_exact': float((take_p[m] == take_t[m]).mean()), 'takeoff_within_1': float((np.abs(take_p[m] - take_t[m]) <= 1).mean()),
                  'landed_auc': auc(1 / (1 + np.exp(-o['landed'][m])), landed[m]),
                  'landing_err_median_u': float(np.median(lerr[m][landed[m]])) if landed[m].any() else None,
                  'landing_err_p90_u': float(np.percentile(lerr[m][landed[m]], 90)) if landed[m].any() else None,
                  'safe_auc': auc(psafe[m], safe[m]), 'safe_rate': float(safe[m].mean()),
                  'safe_brier': float(((psafe[m] - safe[m]) ** 2).mean())}
    return res


def arbitration(fmodel, dev, maps=('KingOfTheMarble_Hunt_phys', 'Sprawl_Hunt_phys'), n_per_map=1000, seed=5):
    """The same validation jump samples predicted two ways: the flight head in one shot, and the step-model ensemble
    rolled forward on the recorded replies. Median path error by decision for both."""
    import torch
    from nav.learned_nav.geometry import Geometry
    from nav.learned_nav import eval3 as E3
    steps = D3.load_ensemble(dev)
    rng = np.random.default_rng(seed)
    out = {}
    for m in maps:
        d = np.load(os.path.join(F3.FEAT_DIR, f'flight_{m}.npz'))
        meta = json.loads(str(d['meta'])); val = np.nonzero(d['val'])[0]
        pick = rng.choice(val, min(n_per_map, len(val)), replace=False)
        shards = {fn: (S, mt) for fn, S, mt in D3.load_shards(m)}
        g = Geometry(m); eidx = D3.EdgeIndex(g)
        P, V, W, U, J, truth, mask, yaws, PREV = [], [], [], [], [], [], [], [], []
        for k in pick:
            fn, tr, dd = meta[k]
            if fn not in shards:
                continue
            S, mt = shards[fn]
            rows = S[S[:, D3.C_TRIAL] == tr]; m_ = mt[tr]
            if dd == 0:
                p0, v0, w0 = np.asarray(m_['start']['pos']), np.asarray(m_['start']['vel']), np.asarray(m_['start']['spin'])
                pend = np.array([rows[0][D3.C_RIGHT] - rows[0][D3.C_LEFT], rows[0][D3.C_FWD] - rows[0][D3.C_BACK], 0.0])
            else:
                r = rows[dd - 1]; p0, v0, w0 = r[D3.C_P].astype(float), r[D3.C_V].astype(float), r[D3.C_W].astype(float)
                pend = np.array([r[D3.C_RIGHT] - r[D3.C_LEFT], r[D3.C_FWD] - r[D3.C_BACK], r[D3.C_JUMP]])
            if dd >= 2:
                rp = rows[dd - 2]; prev = np.array([rp[D3.C_RIGHT] - rp[D3.C_LEFT], rp[D3.C_FWD] - rp[D3.C_BACK], rp[D3.C_JUMP]])
            else:
                prev = np.array([pend[0], pend[1], 0.0])
            PREV.append(prev)
            seq = [pend] + [np.array([r[D3.C_RIGHT] - r[D3.C_LEFT], r[D3.C_FWD] - r[D3.C_BACK], r[D3.C_JUMP]]) for r in rows[dd:dd + F3.H - 1]]
            seq = np.asarray(seq + [np.zeros(3)] * (F3.H - len(seq)))
            tp = np.zeros((F3.H, 3)); mk = np.zeros(F3.H)
            tt = rows[dd:dd + F3.H, D3.C_P].astype(float); tp[:len(tt)] = tt; mk[:len(tt)] = 1
            P.append(p0); V.append(v0); W.append(w0); U.append(seq[:, :2]); J.append(seq[:, 2]); truth.append(tp); mask.append(mk)
        P = np.asarray(P); V = np.asarray(V); W = np.asarray(W); U = np.asarray(U); J = np.asarray(J); truth = np.asarray(truth); mask = np.asarray(mask)
        roll = np.zeros_like(truth)
        PREV = np.asarray(PREV); Up, Jp = PREV[:, :2], PREV[:, 2]
        for t in range(F3.H):
            Fe, yaw = D3.features(g, eidx, P, V, W, U[:, t], J[:, t], Up, Jp)
            Up, Jp = U[:, t], J[:, t]
            mu = E3.predict(steps, Fe, dev)['mu']
            P, V, W = D3.apply_step(P, V, W, mu, yaw)
            roll[:, t] = P
        Fk = d['F'][pick[:len(truth)]].astype(np.float32)
        with torch.no_grad():
            fp = fmodel(torch.as_tensor(Fk, device=dev))['path'].cpu().numpy()
        # flight head path is in the start's heading frame relative to the start: to world
        p_start = truth[:, 0] * 0                                            # placeholder, replaced below
        errs_f, errs_r = [], []
        Tk = d['T'][pick[:len(truth)]]
        path_t = Tk[:, :F3.H * 3].reshape(-1, F3.H, 3)
        ef = np.linalg.norm(fp - path_t, axis=2)                             # both in the start frame
        # rollout error in world (frame-free)
        er = np.linalg.norm(roll - truth, axis=2)
        by_t = []
        for t in range(F3.H):
            mk_t = mask[:, t] > 0
            by_t.append({'decision': t + 1, 'flight_head': float(np.median(ef[mk_t, t])) if mk_t.any() else None,
                         'step_rollout': float(np.median(er[mk_t, t])) if mk_t.any() else None})
        out[m] = {'n': int(len(truth)), 'by_decision': by_t}
        print(m, 'arbitration at 4/8/16 decisions (flight head vs step rollout):',
              [(b['decision'], round(b['flight_head'], 3), round(b['step_rollout'], 3)) for b in by_t if b['decision'] in (4, 8, 16)], flush=True)
    return out


def main(name='flight3.pth'):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = F3.load(dev, name)
    out = {}
    F, T = F3.load_flight(D3.TRAIN_MAPS, 'val')
    out['validation'] = evaluate(model, F, T, dev)
    for m in D3.DEV_MAPS:
        p = os.path.join(F3.FEAT_DIR, f'flight_{m}.npz')
        if os.path.exists(p):
            d = np.load(p)
            out[m] = evaluate(model, d['F'], d['T'], dev)
    try:
        out['arbitration'] = arbitration(model, dev)
    except FileNotFoundError as e:
        print('arbitration skipped:', e)
    json.dump(out, open(OUT, 'w'), indent=1)
    for k, v in out.items():
        if k == 'arbitration':
            continue
        a = v['all']
        print(f'{k}: n {a["n"]}, path {a["path_err_median_u"]:.3f} u (p90 {a["path_err_p90_u"]:.2f}), takeoff exact {a["takeoff_exact"]:.3f}, '
              f'landing {a["landing_err_median_u"]} u, safe AUC {a["safe_auc"]}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'flight3.pth')
