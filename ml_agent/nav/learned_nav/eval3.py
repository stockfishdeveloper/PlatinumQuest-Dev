"""Stage 3 M2 report for the step model: one-step errors by situation, contact prediction, uncertainty, rollouts.

    python -m nav.learned_nav.eval3            -> logs/learned_nav/stage3_eval.json

Sets: 'val' = validation blocks of the training maps; one entry per held-out development map (never trained on).
Situations (from the recorded data): air (no support at the next step, marble bottom > 0.3 u above the floor),
rolling (support now and next, no collision), collision (a collision in the next step), near_edge (an exact edge
within 1 u), slope (floor under the marble steeper than 5 deg), and speed bands.
Rollouts: from sampled recorded states, 16 decisions (1.02 s) on the ensemble mean, feeding the model its own
predictions and the recorded replies; position error at 1, 2, 4, 8, 16 decisions, against the constant-velocity-with-
gravity baseline (ballistic: what a model without contact physics would say).
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav import dynamics3 as D                                     # noqa: E402
from nav.learned_nav.geometry import Geometry                                  # noqa: E402

OUT = os.path.join(HERE, 'logs', 'learned_nav', 'stage3_eval.json')
HORIZONS = (1, 2, 4, 8, 16)
N_ROLL = 1500


def predict(models, F, dev, modes=True):
    """Ensemble prediction. With contact-mode heads (dynamics3 mu_mode) the mean is the mode picked by the member's
    own collision probability (>= 0.5: the collision branch); 'mu_single' keeps the single blended mean."""
    import torch
    mus, lss, cls, nrm, singles = [], [], [], [], []
    with torch.no_grad():
        for m in models:
            outs = [[], [], [], [], []]
            for b in range(0, len(F), 65536):
                o = m(torch.as_tensor(F[b:b + 65536].astype(np.float32), device=dev))
                for k in range(4):
                    outs[k].append(o[k].cpu().numpy())
                outs[4].append(m.last_modes.cpu().numpy() if getattr(m, 'has_modes', False) else None)
            single = np.concatenate(outs[0]) * m.tstd + m.tmean
            singles.append(single)
            if modes and getattr(m, 'has_modes', False):
                md = np.concatenate(outs[4]) * m.tstd + m.tmean
                pcoll = 1 / (1 + np.exp(-np.concatenate(outs[2])[:, 1]))
                mus.append(np.where((pcoll >= 0.5)[:, None], md[:, 1], md[:, 0]))
            else:
                mus.append(single)
            lss.append(np.concatenate(outs[1]) + np.log(m.tstd))
            cls.append(np.concatenate(outs[2])); nrm.append(np.concatenate(outs[3]))
    mus = np.stack(mus); lss = np.stack(lss)
    return {'mu': mus.mean(0), 'mu_single': np.stack(singles).mean(0), 'std_alea': np.sqrt((np.exp(lss) ** 2).mean(0)), 'std_epi': mus.std(0),
            'p': 1 / (1 + np.exp(-np.stack(cls).mean(0))), 'normal': np.stack(nrm).mean(0)}


def auc(p, y):
    y = y.astype(bool)
    if y.all() or (~y).all():
        return None
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def one_step(models, F, T, dev):
    pr = predict(models, F, dev)
    mu = pr['mu']; y = T[:, :D.N_CONT]
    epos = np.linalg.norm(mu[:, 0:3] - y[:, 0:3], axis=1)
    evel = np.linalg.norm((mu[:, 3:6] - y[:, 3:6]) * 5, axis=1)
    espin = np.linalg.norm((mu[:, 6:9] - y[:, 6:9]) * 20, axis=1)
    std = np.sqrt(pr['std_alea'] ** 2 + pr['std_epi'] ** 2)
    within1 = (np.abs(mu - y) <= std).mean(0); within2 = (np.abs(mu - y) <= 2 * std).mean(0)
    sup_t = T[:, D.N_CONT]; coll_t = T[:, D.N_CONT + 1]
    nf = pr['normal']; nt = T[:, D.N_CONT + 3:D.N_CONT + 6]
    cosang = (nf * nt).sum(1) / np.maximum(1e-9, np.linalg.norm(nf, axis=1) * np.linalg.norm(nt, axis=1))
    nerr = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    gap = F[:, 11].astype(np.float32) * 3
    speed = F[:, 6].astype(np.float32) * 10
    edge_d = F[:, 15 + 243].astype(np.float32)
    slope = np.degrees(np.arccos(np.clip(F[:, 14].astype(np.float32), -1, 1)))
    slices = {'all': np.ones(len(F), bool), 'air': (sup_t == 0) & (gap > 0.3), 'rolling': (gap <= 0.05) & (sup_t == 1) & (coll_t == 0),
              'collision': coll_t == 1, 'near_edge': edge_d < 1.0, 'slope': (slope > 5) & (gap <= 0.05),
              'speed_0_4': speed < 4, 'speed_4_10': (speed >= 4) & (speed < 10), 'speed_10+': speed >= 10}
    res = {}
    for k, m in slices.items():
        if m.sum() < 50:
            continue
        epi = np.linalg.norm(pr['std_epi'][m][:, 0:3], axis=1)
        res[k] = {'n': int(m.sum()), 'pos_err_median_u': float(np.median(epos[m])), 'pos_err_p95_u': float(np.percentile(epos[m], 95)),
                  'vel_err_median': float(np.median(evel[m])), 'vel_err_p95': float(np.percentile(evel[m], 95)),
                  'spin_err_median': float(np.median(espin[m])),
                  'support_auc': auc(pr['p'][m, 0], sup_t[m]), 'collision_auc': auc(pr['p'][m, 1], coll_t[m]),
                  'collision_recall_at_0.5': float(((pr['p'][m, 1] >= 0.5) & (coll_t[m] == 1)).sum() / max(1, (coll_t[m] == 1).sum())),
                  'normal_err_median_deg': float(np.median(nerr[m][sup_t[m] == 1])) if (sup_t[m] == 1).any() else None,
                  'disagreement_vs_error_corr': float(np.corrcoef(epi, epos[m])[0, 1]) if len(epi) > 2 and epi.std() > 0 else None,
                  # systematic error: mean signed residual per component (frame: along, left, up), in u / u/s / rad/s
                  'bias_pos': (mu[m, 0:3] - y[m, 0:3]).mean(0).round(5).tolist(),
                  'bias_vel': ((mu[m, 3:6] - y[m, 3:6]) * 5).mean(0).round(4).tolist(),
                  'bias_spin': ((mu[m, 6:9] - y[m, 6:9]) * 20).mean(0).round(3).tolist()}
    res['calibration'] = {'within_1_std': within1.round(3).tolist(), 'within_2_std': within2.round(3).tolist(),
                          'ideal': [0.683, 0.954]}
    return res


def rollouts(models, m, split, dev, rng):
    """Open-loop rollouts on map m from sampled recorded states of the given split."""
    g = Geometry(m); eidx = D.EdgeIndex(g)
    starts = []
    for sidx, (fn, S, meta) in enumerate(D.load_shards(m)):
        tr = S[:, D.C_TRIAL].astype(np.int64)
        for t in np.unique(tr):
            rows = np.nonzero(tr == t)[0]
            if len(rows) < 2 + max(HORIZONS):
                continue
            if S[rows[:1 + max(HORIZONS)], D.C_OOB].any():
                continue
            i0 = rows[0] + int(rng.integers(0, len(rows) - max(HORIZONS)))
            p = S[i0, D.C_P]
            if split == 'val' and not D.val_block(p[0], p[1]):
                continue
            starts.append((S, i0))
    if not starts:
        return None
    sel = rng.choice(len(starts), min(N_ROLL, len(starts)), replace=False)
    starts = [starts[k] for k in sel]
    P = np.array([S[i, D.C_P] for S, i in starts], dtype=np.float64)
    V = np.array([S[i, D.C_V] for S, i in starts], dtype=np.float64)
    W = np.array([S[i, D.C_W] for S, i in starts], dtype=np.float64)
    air0 = np.array([S[i, D.C_SUPPORT] == 0 for S, i in starts])
    Pb = P.copy(); Vb = V.copy()
    err = {h: [] for h in HORIZONS}; errb = {h: [] for h in HORIZONS}
    def reply(S, r):
        return [S[r, D.C_RIGHT] - S[r, D.C_LEFT], S[r, D.C_FWD] - S[r, D.C_BACK]]
    Up = np.array([reply(S, i - 1) if i > 0 and S[i - 1, D.C_TRIAL] == S[i, D.C_TRIAL] else reply(S, i) for S, i in starts])
    Jp = np.array([S[i - 1, D.C_JUMP] if i > 0 and S[i - 1, D.C_TRIAL] == S[i, D.C_TRIAL] else 0.0 for S, i in starts])
    for step in range(max(HORIZONS)):
        U = np.array([reply(S, i + step) for S, i in starts])
        J = np.array([S[i + step, D.C_JUMP] for S, i in starts])
        F, yaw = D.features(g, eidx, P, V, W, U, J, Up, Jp)
        Up, Jp = U, J
        mu = predict(models, F, dev)['mu']
        P, V, W = D.apply_step(P, V, W, mu, yaw)
        Pb = Pb + Vb * 0.064 + np.array([0, 0, -0.5 * 20 * 0.064 ** 2]); Vb = Vb + np.array([0, 0, -20 * 0.064])
        h = step + 1
        if h in HORIZONS:
            truth = np.array([S[i + step + 1, D.C_P] for S, i in starts])
            err[h] = np.linalg.norm(P - truth, axis=1); errb[h] = np.linalg.norm(Pb - truth, axis=1)
    split = {k: {h: float(np.median(err[h][msk])) for h in HORIZONS} for k, msk in (('airborne_start', air0), ('floor_start', ~air0)) if msk.sum() > 20}
    return {'n': len(starts), 'n_airborne': int(air0.sum()), 'model_by_start': split,
            'model': {h: {'median': float(np.median(err[h])), 'p90': float(np.percentile(err[h], 90))} for h in HORIZONS},
            'ballistic': {h: {'median': float(np.median(errb[h])), 'p90': float(np.percentile(errb[h], 90))} for h in HORIZONS}}


def main():
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = D.load_ensemble(dev)
    rng = np.random.default_rng(7)
    t0 = time.time()
    out = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'one_step': {}, 'rollouts': {}}
    Fva, Tva = D.load_features(D.TRAIN_MAPS, 'val')
    out['one_step']['val'] = one_step(models, Fva, Tva, dev)
    print('val one-step:', json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in out['one_step']['val'].items() if k in ('all', 'air', 'rolling', 'collision')}), flush=True)
    for m in D.DEV_MAPS:
        p = os.path.join(D.FEAT_DIR, f'{m}.npz')
        if os.path.exists(p):
            d = np.load(p)
            out['one_step'][m] = one_step(models, d['F'], d['T'], dev)
            print(m, 'one-step all:', json.dumps(out['one_step'][m].get('all')), flush=True)
    for m in ['KingOfTheMarble_Hunt_phys', 'Sprawl_Hunt_phys', 'KingOfTheRing_Hunt_phys', 'VortexEffect_Hunt_phys']:
        r = rollouts(models, m, 'val', dev, rng)
        if r:
            out['rollouts'][f'{m} (validation blocks)'] = r
            print(m, 'rollout 16:', r['model'][16], 'ballistic', r['ballistic'][16], flush=True)
    for m in D.DEV_MAPS:
        r = rollouts(models, m, 'all', dev, rng)
        if r:
            out['rollouts'][f'{m} (held-out map)'] = r
            print(m, 'rollout 16:', r['model'][16], 'ballistic', r['ballistic'][16], flush=True)
    out['seconds'] = round(time.time() - t0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
