"""Gem-side seeds (design section 6): launch states on the floor from which a jump is predicted to take the gem and
land safely. They guide the planner's approach (seed-aimed proposals and the approach cost); they are never
executed as they are: every route is re-predicted from the actual state (planner.py).

    python -m nav.learned_nav.seeds kotmjump_p1 -20.25 10.05 21.7      # -> datasets/learned_nav/seeds/<map>.npz

Sampling, from geometry and the model only: floor points within REACH of the gem (horizontally), heading toward it
within 60 deg, speed 2-14 u/s, rolling spin, full run-up input pending. For each, jump at decision 0-3 with no air
input or air input at 0, +-45, +-90 or 180 deg from the heading, rolled forward with the step ensemble
(planner.simulate); P(success) = P(pickup) from the closest pass x a safe continuation. A state is a seed if its
best candidate reaches P_SEED, judged as the planner judges (including the perturbed-start robustness check);
stored with that candidate.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav import dynamics3 as D3                                      # noqa: E402
from nav.learned_nav import planner as PL                                        # noqa: E402
from nav.learned_nav.geometry import Geometry                                    # noqa: E402

SEED_DIR = os.path.join(HERE, 'datasets', 'learned_nav', 'seeds')
N_STATES = 6000
REACH = 8.0
P_SEED = PL.P_JUMP
JUMPS = (0, 1, 2, 3)
AIRS = (np.nan, 0.0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2, math.pi)
BATCH = 6000


def sample_states(g, gem, rng, n):
    out = []
    tries = 0
    while len(out) < n and tries < 200 * n:
        tries += 1
        r = REACH * math.sqrt(rng.random()); a = rng.uniform(-math.pi, math.pi)
        x, y = gem[0] + r * math.cos(a), gem[1] + r * math.sin(a)
        z = g.floor_below(x, y, gem[2] + 0.5)
        if z is None or z < gem[2] - 3.0:
            continue
        nrm = g.normal_below(x, y, z + 0.01)
        if nrm is None or nrm[2] < 0.9:
            continue
        h = math.atan2(gem[1] - y, gem[0] - x) + math.radians(rng.uniform(-60, 60))
        sp = rng.uniform(2.0, 14.0)
        v = np.array([sp * math.cos(h), sp * math.sin(h), 0.0])
        out.append((x, y, z + PL.R + 0.01, v[0], v[1], z))
    return np.array(out)


def build(map_name, gem, seed=0, log=print):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = Geometry(map_name); eidx = D3.EdgeIndex(g); steps = D3.load_ensemble(dev)
    gem = np.asarray(gem, float)
    rng = np.random.default_rng(seed)
    S = sample_states(g, gem, rng, N_STATES)
    cands = [(j, a) for j in JUMPS for a in AIRS]
    nc = len(cands)
    t0 = time.time()
    best_p = np.zeros(len(S)); best_c = np.zeros(len(S), int)
    idx_all = np.repeat(np.arange(len(S)), nc); c_all = np.tile(np.arange(nc), len(S))
    def run(si, ci, var=None):
        """P(success) of candidates ci from states si; var = (kind, a) perturbs the start (planner.VARIANTS)."""
        out = []
        for b in range(0, len(si), BATCH):
            s_ = si[b:b + BATCH]; c_ = ci[b:b + BATCH]; n = len(s_)
            st = S[s_]
            P = st[:, 0:3].copy(); V = np.c_[st[:, 3:5], np.zeros(n)]
            W = np.c_[-V[:, 1], V[:, 0], np.zeros(n)] / PL.R
            hd = np.arctan2(V[:, 1], V[:, 0])
            run_u = PL.SQ2 * np.c_[np.cos(hd), np.sin(hd)]
            pr = PL.Programs(n)
            pr.ang[:] = hd[:, None]
            for k in range(n):
                j, a = cands[c_[k]]
                pr.jump[k, j] = 1
                if np.isnan(a):
                    pr.mode[k, j:] = PL.MODE_NONE
                else:
                    pr.ang[k, j:] = hd[k] + a
            pr.after[:] = PL.MODE_BRAKE
            if var is not None:
                kind, a = var
                if kind == 's':
                    V = V * np.array([a, a, 1.0]); W = W * a
                else:
                    c, s = math.cos(a), math.sin(a)
                    V = np.c_[c * V[:, 0] - s * V[:, 1], s * V[:, 0] + c * V[:, 1], V[:, 2]]
                    W = np.c_[c * W[:, 0] - s * W[:, 1], s * W[:, 0] + c * W[:, 1], W[:, 2]]
            sim = PL.simulate(steps, g, eidx, dev, P, V, W, run_u, np.zeros(n), run_u, np.zeros(n), pr, False, st[:, 5])
            out.append(PL.judge(sim, gem))
        return {k: np.concatenate([o[k] for o in out]) for k in ('p_succ', 'p_pick', 'safe', 'margin')}

    nom = run(idx_all, c_all)
    log(f'  nominal: {len(idx_all)} sequences, {time.time() - t0:.0f} s')
    # robustness (as the planner): candidates worth it again from the perturbed starts, mean safety x margin
    cand = np.nonzero(nom['p_succ'] >= 0.3)[0]
    pick_sum = np.zeros(len(cand)); safe_sum = np.zeros(len(cand))
    for var in PL.VARIANTS:
        r = run(idx_all[cand], c_all[cand], var)
        pick_sum += r['p_pick']; safe_sum += r['safe'] * (0.4 + 0.6 * r['margin'])
    ps_rob = np.minimum(nom['p_succ'][cand], 0.5 * (nom['p_pick'][cand] + pick_sum / PL.N_VAR) * safe_sum / PL.N_VAR)
    log(f'  robust: {len(cand)} candidates, {time.time() - t0:.0f} s')
    for k, i in enumerate(cand):
        s_ = idx_all[i]
        if ps_rob[k] > best_p[s_]:
            best_p[s_] = ps_rob[k]; best_c[s_] = c_all[i]
    keep = best_p >= P_SEED
    out = {'x': S[keep, 0], 'y': S[keep, 1], 'z': S[keep, 2], 'vx': S[keep, 3], 'vy': S[keep, 4],
           'j': np.array([cands[c][0] for c in best_c[keep]], int), 'air': np.array([cands[c][1] for c in best_c[keep]]),
           'p': best_p[keep]}
    os.makedirs(SEED_DIR, exist_ok=True)
    np.savez(os.path.join(SEED_DIR, f'{map_name}.npz'), gem=gem, **out)
    meta = {'map': map_name, 'gem': gem.tolist(), 'states': int(len(S)), 'seeds': int(keep.sum()),
            'p_seed': P_SEED, 'reach': REACH, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'speed_hist': np.histogram(np.hypot(out['vx'], out['vy']), bins=[0, 4, 6, 8, 10, 12, 14])[0].tolist(),
            'jump_hist': np.bincount(out['j'], minlength=4).tolist()}
    json.dump(meta, open(os.path.join(SEED_DIR, f'{map_name}.json'), 'w'), indent=1)
    log(f'{map_name}: {keep.sum()} seeds of {len(S)} states ({time.time() - t0:.0f} s)')
    return out


def load(map_name):
    p = os.path.join(SEED_DIR, f'{map_name}.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {k: z[k] for k in ('x', 'y', 'z', 'vx', 'vy', 'j', 'air', 'p')}


if __name__ == '__main__':
    m = sys.argv[1]
    build(m, [float(v) for v in sys.argv[2:5]], log=lambda s: print(s, flush=True))
