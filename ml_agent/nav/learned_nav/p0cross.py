"""Cross-check: the general stage 3 flight head (trained on the stage 3 maps, never on P0 data) choosing P0's jumps.

    python -m nav.learned_nav.p0cross            -> logs/learned_nav/p0cross.json

For every P0 start (the recorded, exact start state) the 46 P0 candidates are written as the reply sequence the P0
recorder sent (run-up held, the jump at decision j with air input a, air input a held after) and scored by the flight
head: P(pickup) from the predicted path's closest pass to the gem through a logistic curve fitted on P0 TRAINING
starts, times P(safe). The best candidate per held-out start is judged with P0's recorded outcome of that candidate
(every candidate was flown from every held-out start in P0). The stage 3 data includes KOTM, the same geometry as
the P0 hole, so this tests the objective and the controls transfer, not unseen geometry.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav import flight3 as F3                                       # noqa: E402
from nav.learned_nav.geometry import Geometry                                   # noqa: E402
from nav.learned_nav import dynamics as P0D                                     # noqa: E402
from nav.learned_nav import evaluate as P0E                                     # noqa: E402
from nav.learned_nav.record import CANDS, TARGET                                # noqa: E402

OUT = os.path.join(HERE, 'logs', 'learned_nav', 'p0cross.json')
SQ2 = math.sqrt(2.0)


def cand_sequence(h_req, h0, ci, stop_at=None):
    """(H+1, 2) world input vectors (pending first) and (H+1,) jump keys for P0 candidate ci. stop_at: the decision
    from which there is no input (P0 released every input on landing)."""
    run = SQ2 * np.array([math.cos(h0), math.sin(h0)])
    pend = SQ2 * np.array([math.cos(h_req), math.sin(h_req)])
    U = [pend]; J = [0.0]
    c = CANDS[ci]
    for i in range(F3.H):
        if c is None or i < c[0]:
            U.append(run); J.append(0.0)
        else:
            j, a = c
            v = np.zeros(2) if a == 0 else SQ2 * np.array([math.cos(h0 + math.radians((a - 1) * 45)), math.sin(h0 + math.radians((a - 1) * 45))])
            U.append(v); J.append(1.0 if i == j else 0.0)
    U = np.asarray(U); J = np.asarray(J)
    if stop_at is not None and c is not None:
        U[1 + stop_at:] = 0.0                                     # index 0 is the pending reply
    return U, J


def closest_pass(path_world, p0, gem):
    pts = np.vstack([p0[None, :], path_world])
    a, b = pts[:-1], pts[1:]
    ts = np.linspace(0, 1, 8)[None, :, None]
    q = a[:, None, :] + (b - a)[:, None, :] * ts
    return float(np.linalg.norm(q - gem[None, None, :], axis=2).min())


def score_starts(model, g, by, sids, dev):
    import torch
    out = {}
    feats = []; keys = []
    for sid in sids:
        r = next(iter(by[sid].values()))
        p0 = np.asarray(r['p0']); v0 = np.asarray(r['v0']); w0 = np.asarray(r['w0'])
        h0 = r['h0']; h_req = r['req']['heading']
        yaw = math.atan2(v0[1], v0[0]) if math.hypot(v0[0], v0[1]) > 0.5 else h0
        c, s = math.cos(yaw), math.sin(yaw)
        base = None
        for ci in range(len(CANDS)):
            U, J = cand_sequence(h_req, h0, ci)
            if base is None:
                base = F3.sample_features(g, p0, v0, w0, None, yaw, U, J)      # geometry once per start
            f = base.copy()
            uf, ul = c * U[:, 0] + s * U[:, 1], -s * U[:, 0] + c * U[:, 1]
            f[F3.N_STATE:F3.N_STATE + F3.N_SEQ] = np.c_[uf, ul, J][:F3.H].ravel()
            feats.append(f); keys.append((sid, ci, yaw, p0))
    X = np.asarray(feats, np.float32)

    def run(X):
        paths, safe, landt = [], [], []
        with torch.no_grad():
            for b in range(0, len(X), 4096):
                o = model(torch.as_tensor(X[b:b + 4096], device=dev))
                paths.append(o['path'].cpu().numpy()); safe.append(torch.sigmoid(o['safe']).cpu().numpy())
                landt.append(o['landt'].argmax(1).cpu().numpy() if 'landt' in o else np.full(len(X[b:b + 4096]), F3.H))
        return np.concatenate(paths), np.concatenate(safe), np.concatenate(landt)
    paths, safe, landt = run(X)
    # second pass: no input from the predicted landing decision on (P0's continuation), where a landing is predicted
    X2 = X.copy()
    for k, (sid, ci, yaw, p0) in enumerate(keys):
        if landt[k] < F3.H and CANDS[ci] is not None:
            r = next(iter(by[sid].values()))
            U, J = cand_sequence(r['req']['heading'], r['h0'], ci, stop_at=int(landt[k]) + 1)
            c, s = math.cos(yaw), math.sin(yaw)
            uf, ul = c * U[:, 0] + s * U[:, 1], -s * U[:, 0] + c * U[:, 1]
            X2[k, F3.N_STATE:F3.N_STATE + F3.N_SEQ] = np.c_[uf, ul, J][:F3.H].ravel()
    paths, safe, _ = run(X2)
    gem = np.asarray(TARGET)
    for k, (sid, ci, yaw, p0) in enumerate(keys):
        c, s = math.cos(yaw), math.sin(yaw)
        P = paths[k]
        world = np.c_[c * P[:, 0] - s * P[:, 1], s * P[:, 0] + c * P[:, 1], P[:, 2]] + p0
        out.setdefault(sid, np.zeros((len(CANDS), 2)))[ci] = (closest_pass(world, p0, gem), safe[k])
    return out


def main(model_name='flight3.pth'):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = F3.load(dev, model_name)
    g = Geometry('kotmjump_p0')
    by = P0D.load_trials()
    starts = {sid: next(iter(d.values())) for sid, d in by.items()}
    complete = [sid for sid, d in by.items() if len(d) == len(CANDS)]
    train = [sid for sid in complete if not starts[sid]['held_out']]
    ho = sorted(sid for sid in complete if starts[sid]['held_out'])
    rng = np.random.default_rng(3)
    fit = sorted(rng.choice(train, min(800, len(train)), replace=False))
    sc_fit = score_starts(model, g, by, fit, dev)
    dmin = np.concatenate([sc_fit[s][:, 0] for s in fit]); pick = np.concatenate([[float(by[s][ci]['pickup']) for ci in range(len(CANDS))] for s in fit])
    curve = P0D.fit_pass_curve(dmin, pick)
    sc = score_starts(model, g, by, ho, dev)
    out_tab = P0E.outcome_table(by, ho)
    feas = np.nanmax(out_tab['success'], axis=1) > 0
    picks = np.array([int(np.argmax(P0D.pass_prob(sc[s][:, 0], curve) * sc[s][:, 1])) for s in ho])
    got = out_tab['success'][np.arange(len(ho)), picks]
    k = int(np.nansum(got[feas])); n = int(feas.sum())
    p, lo, hi = P0E.wilson(k, n)
    # the pickup query itself, on every held-out flight: predicted pickup (>= 0.5) against P0's recorded pickup
    pp = np.concatenate([P0D.pass_prob(sc[s][:, 0], curve) for s in ho])
    pk = np.concatenate([[float(by[s][ci]['pickup']) for ci in range(len(CANDS))] for s in ho])
    pickup_q = {'n': int(len(pk)), 'positives': int(pk.sum()), 'false_pos': int(((pp >= 0.5) & (pk == 0)).sum()),
                'false_neg': int(((pp < 0.5) & (pk == 1)).sum()), 'auc': P0E.auc(pp, pk)}
    frozen = json.load(open(P0E.FROZEN))
    p0_learned = [frozen['picks'][s]['learned'] for s in ho]
    got0 = out_tab['success'][np.arange(len(ho)), p0_learned]
    k0 = int(np.nansum(got0[feas]))
    res = {'model': model_name, 'curve': curve, 'held_out_feasible': n, 'general_flight_head': {'success': k, 'rate': p, 'ci95': [lo, hi]},
           'p0_model_same_starts': {'success': k0, 'rate': k0 / n}, 'fit_starts': len(fit), 'pickup_query': pickup_q}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__' and not (len(sys.argv) > 1 and sys.argv[1] in ('rollout', 'combined')):
    main(sys.argv[1] if len(sys.argv) > 1 else 'flight3.pth')


# ------------------------------------------------------------------ the same test with step-model rollouts
def rollout_scores(steps, g, eidx, by, sids, dev, horizon=24):
    """For every start and candidate: roll the step-model ensemble forward decision by decision, flying the candidate
    exactly as the P0 recorder did (run-up, jump at j with air input a, the air input until the marble is back on a
    floor, then no input). Returns {sid: (46, 3)}: closest pass to the gem, ends safe (supported on a floor, not below
    the start floor by more than 2 u), predicted landing decision."""
    from nav.learned_nav import eval3 as E3
    from nav.learned_nav import dynamics3 as D3
    keys = [(sid, ci) for sid in sids for ci in range(len(CANDS))]
    st = {sid: next(iter(by[sid].values())) for sid in sids}
    P = np.array([st[s]['p0'] for s, _ in keys], dtype=np.float64)
    V = np.array([st[s]['v0'] for s, _ in keys], dtype=np.float64)
    W = np.array([st[s]['w0'] for s, _ in keys], dtype=np.float64)
    h0 = np.array([st[s]['h0'] for s, _ in keys]); hreq = np.array([st[s]['req']['heading'] for s, _ in keys])
    z0 = P[:, 2].copy()
    jt = np.array([CANDS[c][0] if CANDS[c] is not None else 999 for _, c in keys])
    air = np.array([CANDS[c][1] if CANDS[c] is not None else 0 for _, c in keys])
    run = SQ2 * np.c_[np.cos(h0), np.sin(h0)]
    ang = h0 + np.radians((air - 1) * 45.0)
    airv = np.where((air == 0)[:, None], 0.0, SQ2 * np.c_[np.cos(ang), np.sin(ang)])
    Up = SQ2 * np.c_[np.cos(hreq), np.sin(hreq)]; Jp = np.zeros(len(keys))
    gem = np.asarray(TARGET)
    path = [P.copy()]
    airborne = np.zeros(len(keys), bool); landed_at = np.full(len(keys), -1)
    sup_last = np.zeros(len(keys))
    for t in range(horizon):
        pre = t < jt
        U = np.where(pre[:, None], run, airv)
        U = np.where((landed_at >= 0)[:, None], 0.0, U)
        J = (t == jt).astype(float)
        F, yaw = D3.features(g, eidx, P, V, W, U, J, Up, Jp)
        pr = E3.predict(steps, F, dev)
        P, V, W = D3.apply_step(P, V, W, pr['mu'], yaw)
        sup = pr['p'][:, 0]
        lv, _ = D3.level_below(g, P[:, 0], P[:, 1], P[:, 2] + 0.3)
        gap = np.where(np.isfinite(lv), P[:, 2] - 0.19 - lv, 9.0)
        airborne |= (gap > 0.5) & (sup < 0.5)
        newly = airborne & (landed_at < 0) & (sup >= 0.5) & (gap < 0.35)
        landed_at = np.where(newly, t, landed_at)
        sup_last = sup
        Up, Jp = U, J
        path.append(P.copy())
    path = np.stack(path, 1)                                                    # (N, H+1, 3)
    a, b = path[:, :-1], path[:, 1:]
    ts = np.linspace(0, 1, 8)[None, None, :, None]
    q = a[:, :, None, :] + (b - a)[:, :, None, :] * ts
    dmin = np.linalg.norm(q - gem, axis=3).reshape(len(keys), -1).min(1)
    lv, _ = D3.level_below(g, P[:, 0], P[:, 1], P[:, 2] + 0.3)
    safe = (sup_last >= 0.5) & np.isfinite(lv) & (P[:, 2] > z0 - 2.0) & (~airborne | (landed_at >= 0))
    out = {}
    for k, (sid, ci) in enumerate(keys):
        out.setdefault(sid, np.zeros((len(CANDS), 3)))[ci] = (dmin[k], float(safe[k]), landed_at[k])
    return out


def main_rollout():
    import torch
    from nav.learned_nav import dynamics3 as D3
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    steps = D3.load_ensemble(dev)
    g = Geometry('kotmjump_p0'); eidx = D3.EdgeIndex(g)
    by = P0D.load_trials()
    starts = {sid: next(iter(d.values())) for sid, d in by.items()}
    complete = [sid for sid, d in by.items() if len(d) == len(CANDS)]
    train = [sid for sid in complete if not starts[sid]['held_out']]
    ho = sorted(sid for sid in complete if starts[sid]['held_out'])
    rng = np.random.default_rng(3)
    fit = sorted(rng.choice(train, min(800, len(train)), replace=False))
    sf = rollout_scores(steps, g, eidx, by, fit, dev)
    dmin = np.concatenate([sf[s][:, 0] for s in fit]); pick = np.concatenate([[float(by[s][ci]['pickup']) for ci in range(len(CANDS))] for s in fit])
    curve = P0D.fit_pass_curve(dmin, pick)
    safe_fit = np.concatenate([sf[s][:, 1] for s in fit]); safe_true = np.concatenate([[float(by[s][ci]['safe']) for ci in range(len(CANDS))] for s in fit])
    sc = rollout_scores(steps, g, eidx, by, ho, dev)
    out_tab = P0E.outcome_table(by, ho)
    feas = np.nanmax(out_tab['success'], axis=1) > 0
    score = {s: P0D.pass_prob(sc[s][:, 0], curve) * (0.05 + 0.95 * sc[s][:, 1]) for s in ho}
    picks = np.array([int(np.argmax(score[s])) for s in ho])
    got = out_tab['success'][np.arange(len(ho)), picks]
    k = int(np.nansum(got[feas])); n = int(feas.sum()); p, lo, hi = P0E.wilson(k, n)
    pp = np.concatenate([P0D.pass_prob(sc[s][:, 0], curve) for s in ho]); pk = np.concatenate([[float(by[s][ci]['pickup']) for ci in range(len(CANDS))] for s in ho])
    ss = np.concatenate([sc[s][:, 1] for s in ho]); st_ = np.concatenate([[float(by[s][ci]['safe']) for ci in range(len(CANDS))] for s in ho])
    res = {'method': 'step-model rollouts', 'curve': curve, 'held_out_feasible': n, 'success': k, 'rate': p, 'ci95': [lo, hi],
           'pickup_query': {'auc': P0E.auc(pp, pk), 'false_pos': int(((pp >= 0.5) & (pk == 0)).sum()), 'false_neg': int(((pp < 0.5) & (pk == 1)).sum()), 'positives': int(pk.sum())},
           'safe_accuracy_heldout': float((ss == st_).mean()), 'safe_accuracy_fit': float((safe_fit == safe_true).mean())}
    p = OUT.replace('.json', '_rollout.json')
    json.dump(res, open(p, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'rollout':
    main_rollout()


def main_combined(flight_name='flight3.pth'):
    """Predictor selection: the pickup from the step-model rollout's path, the safe landing from the flight head."""
    import torch
    from nav.learned_nav import dynamics3 as D3
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    steps = D3.load_ensemble(dev); fmodel = F3.load(dev, flight_name)
    g = Geometry('kotmjump_p0'); eidx = D3.EdgeIndex(g)
    by = P0D.load_trials()
    starts = {sid: next(iter(d.values())) for sid, d in by.items()}
    complete = [sid for sid, d in by.items() if len(d) == len(CANDS)]
    train = [sid for sid in complete if not starts[sid]['held_out']]
    ho = sorted(sid for sid in complete if starts[sid]['held_out'])
    rng = np.random.default_rng(3)
    fit = sorted(rng.choice(train, min(800, len(train)), replace=False))
    rf = rollout_scores(steps, g, eidx, by, fit, dev)
    curve = P0D.fit_pass_curve(np.concatenate([rf[s][:, 0] for s in fit]),
                               np.concatenate([[float(by[s][ci]['pickup']) for ci in range(len(CANDS))] for s in fit]))
    rh = rollout_scores(steps, g, eidx, by, ho, dev)
    fh = score_starts(fmodel, g, by, ho, dev)
    out_tab = P0E.outcome_table(by, ho)
    feas = np.nanmax(out_tab['success'], axis=1) > 0
    res = {}
    for tag, fn in (('rollout pickup x flight-head safe', lambda s: P0D.pass_prob(rh[s][:, 0], curve) * fh[s][:, 1]),):
        picks = np.array([int(np.argmax(fn(s))) for s in ho])
        got = out_tab['success'][np.arange(len(ho)), picks]
        k = int(np.nansum(got[feas])); n = int(feas.sum()); p, lo, hi = P0E.wilson(k, n)
        res[tag] = {'success': k, 'n': n, 'rate': p, 'ci95': [lo, hi]}
    json.dump(res, open(OUT.replace('.json', '_combined.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'combined':
    main_combined()
