"""Stage 3b: a ranker that scores a root candidate by its predicted points over the next 20 s, trained on executed branch
outcomes (branch_bench.py --label-mode) and judged on the held-out pilot rounds by the pilot's own metric (the realized
gain of its argmax on the reserved futures). No map identity in the features: everything is relative to the marble.

usage (repo root, PYTHONPATH=ml_agent):
  python ml_agent/tools/sim_bench/ranker.py --train labels/run1/*.jsonl --eval pilot/run1/*.jsonl pilot/run2/*.jsonl --out ranker.pkl
"""
import argparse
import glob
import json
import math
import os
import pickle
import sys

import numpy as np

FEATURE_NAMES = []


def _rel(p, q):
    return [float(a - b) for a, b in zip(p, q)]


def candidate_vector(root, cand):
    """Numeric features of one candidate given the root context (dicts from branch_bench rows)."""
    f = cand.get('features') or {}
    rp = np.asarray(root['position'], dtype=float); rv = np.asarray(root['velocity'], dtype=float)
    ep = np.asarray(f.get('end_position') or root['position'], dtype=float); ev = np.asarray(f.get('end_velocity') or [0., 0., 0.], dtype=float)
    gems = np.asarray(root.get('gems') or [], dtype=float).reshape(-1, 4) if root.get('gems') else np.zeros((0, 4))
    speed_end = float(np.linalg.norm(ev[:2])); speed_root = float(np.linalg.norm(rv[:2]))
    if len(gems):
        d_end = np.linalg.norm(gems[:, :3] - ep, axis=1); d_root = np.linalg.norm(gems[:, :3] - rp, axis=1)
        k = int(np.argmin(d_end)); nearest_end = float(d_end[k]); nearest_val = float(gems[k, 3])
        sum_near = float(np.sum(gems[:, 3] / (1. + d_end)))           # value-weighted closeness of the remaining gems
        gain_dist = float(np.min(d_root) - nearest_end)                 # how much nearer the plan ends to a gem
        to_gem = gems[k, :3] - ep; n = float(np.linalg.norm(to_gem[:2]))
        align = float(np.dot(ev[:2], to_gem[:2]) / (n * speed_end)) if n > 1e-6 and speed_end > 1e-6 else 0.
        dz = float(gems[k, 2] - ep[2]); total_val = float(gems[:, 3].sum()); count = float(len(gems))
    else:
        nearest_end = 60.; nearest_val = 0.; sum_near = 0.; gain_dist = 0.; align = 0.; dz = 0.; total_val = 0.; count = 0.
    # trajectory features (rows saved from 10-30 18:00 on carry the forecast path every 4th reply): closest approach to the
    # nearest / most valuable remaining gems along the path, when it happens, and the path's progress toward them
    path = np.asarray(f.get('path') or [], dtype=float).reshape(-1, 3) if f.get('path') else None
    traj = [0.] * 8
    if path is not None and len(path) > 1 and len(gems):
        D = np.linalg.norm(gems[None, :, :3] - path[:, None, :], axis=2)          # (steps, gems)
        dmin = D.min(axis=0); tmin = D.argmin(axis=0) / max(1, len(path) - 1)       # fraction of the forecast
        order = np.argsort(dmin)
        k0 = order[0]; k1 = order[1] if len(order) > 1 else order[0]
        kv = int(np.argmax(gems[:, 3] / (1. + dmin)))
        traj = [float(dmin[k0]), float(tmin[k0]), float(dmin[k1]), float(tmin[k1]), float(dmin[kv]), float(gems[kv, 3]),
                float(np.sum(dmin < 0.8)),                                           # gems the path passes through
                float(D[0].min() - D[-1].min())]                                     # progress toward the nearest gem
    v = [
        float(f.get('cost') if f.get('cost') is not None and math.isfinite(f.get('cost')) else 0.),
        float(f.get('possible_pickup_interval') or 0), 1. if f.get('possible_pickup_interval') else 0.,
        float(f.get('minimum_gem_distance_u') or 0.),
        float(bool(f.get('terminal_support'))), float(bool(f.get('stable_continuation'))),
        float(f.get('maximum_tail_normal_speed_u_per_s') or 0.), float(f.get('continuation_s') or 0.),
        float(f.get('through_post_u') or 0.), float(f.get('evaluated_replies') or 0), float(f.get('replies') or 0),
        float(f.get('resources_consumed') or 0), float(f.get('inventory_consumed') or 0),
        float(bool(f.get('backup_supported'))), float(f.get('backup_drop_u') or 0.), float(f.get('backup_replies') or 0),
        float(f.get('max_speed') or 0.), float(f.get('blast_events') or 0), float(f.get('end_held') or 0),
        1. if cand.get('rejected') else 0.,
        speed_end, speed_root, float(ep[2] - rp[2]), float(np.linalg.norm((ep - rp)[:2])),
        nearest_end, nearest_val, sum_near, gain_dist, align, dz, total_val, count,
        float(root.get('held', 0)), float(root.get('held_type', 0)), float(root.get('blast', 0.)), float(root.get('restore_type', 0)),
        float(root.get('time_left_ms', 0)) / 1000.,
        float(len(root.get('powerups') or [])),
    ] + traj
    return v


FEATURE_NAMES[:] = ['cost', 'pickup_interval', 'has_pickup', 'min_gem_dist', 'terminal_support', 'stable', 'tail_normal_speed', 'continuation_s',
                    'through_post_u', 'evaluated_replies', 'replies', 'resources_consumed', 'inventory_consumed', 'backup_supported', 'backup_drop',
                    'backup_replies', 'max_speed', 'blast_events', 'end_held', 'gate_rejected', 'speed_end', 'speed_root', 'end_dz', 'end_dist_xy',
                    'nearest_gem_end', 'nearest_gem_value', 'value_closeness', 'gain_dist', 'align', 'gem_dz', 'gems_value', 'gems_count',
                    'held', 'held_type', 'blast', 'restore_type', 'time_left_s', 'powerups_n',
                    'traj_dmin0', 'traj_tmin0', 'traj_dmin1', 'traj_tmin1', 'traj_dmin_valuable', 'traj_valuable_value', 'traj_passes', 'traj_progress']


def target_value(run, target):
    """One run's label: 'points20' (default), 'points10', 'points5' (points within that many seconds of the root), 'firstpick'
    (minus the time to the first pickup, capped at the horizon), 'blend' (points10 + points20 / 2)."""
    if run is None:
        return None
    if target == 'points20':
        return float(run['points'])
    t0 = run['root_t_ms']
    within = lambda s: float(sum(p['value'] for p in run['picks'] if p['t_ms'] - t0 <= s * 1000))
    if target == 'points10':
        return within(10)
    if target == 'points5':
        return within(5)
    if target == 'firstpick':
        return -min([(p['t_ms'] - t0) / 1000. for p in run['picks']] or [20.])
    if target == 'blend':
        return within(10) + 0.5 * float(run['points'])
    raise ValueError(target)


REQUIRE_PATH = [False]


def rows_from(path, futures=None, target='points20'):
    """(X, y, groups, meta) from branch_bench rows: one row per branch (the baseline's chosen candidate included when its
    features were saved); y = mean points over `futures` (default: every future recorded)."""
    X = []; y = []; groups = []; meta = []
    for line in open(path, encoding='utf-8'):
        if not line.strip():
            continue
        r = json.loads(line)
        root = r.get('root_context')
        if not root:
            continue
        if REQUIRE_PATH[0] and not any((b.get('features') or {}).get('path') for b in r['arms']['branches'].values()):
            continue
        sid = (r['map'], r['pool'], str(r['seed']), r['decision'])
        def pts(runs):
            vals = [target_value(v, target) for f, v in runs.items() if v and (futures is None or int(f) in futures)]
            return float(np.mean(vals)) if vals else None
        if r.get('root_chosen_features'):
            p = pts(r['arms']['base'])
            if p is not None:
                X.append(candidate_vector(root, dict(features=r['root_chosen_features'], rejected=None))); y.append(p); groups.append(sid)
                meta.append(dict(sid=sid, name='base', kind='base', map=r['map'], bucket=r['bucket'], round=(r['pool'], str(r['seed']))))
        for name, b in r['arms']['branches'].items():
            p = pts(b['runs'])
            if p is None or not b.get('features'):
                continue
            X.append(candidate_vector(root, b)); y.append(p); groups.append(sid)
            meta.append(dict(sid=sid, name=name, kind=b['kind'], map=r['map'], bucket=r['bucket'], round=(r['pool'], str(r['seed']))))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float), groups, meta


def load_many(paths, futures=None, target='points20'):
    Xs = []; ys = []; gs = []; ms = []
    for p in paths:
        X, y, g, m = rows_from(p, futures, target)
        if len(X):
            Xs.append(X); ys.append(y); gs += g; ms += m
    return (np.concatenate(Xs) if Xs else np.zeros((0, len(FEATURE_NAMES)))), (np.concatenate(ys) if ys else np.zeros(0)), gs, ms


def relative_targets(y, groups):
    """Points minus the mean of the situation's own options: the ranker learns which option is better HERE, not how rich
    the situation is (situation difficulty is shared by every option and only adds noise to a within-situation ranking)."""
    y = np.asarray(y, dtype=float); out = y.copy(); idx = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    for g, ii in idx.items():
        out[ii] = y[ii] - y[ii].mean()
    return out


N_CAND = 32   # candidate-specific columns 0..31; root columns 32..37; trajectory columns 38..45 are candidate-specific too
CAND_COLS = list(range(32)) + list(range(38, 46))


def center_features(X, groups):
    """Candidate columns minus the situation's mean over its options: 'cost 10 u below the alternatives here', not an
    absolute cost that means nothing across situations. Root columns are shared by the options and stay as they are."""
    X = np.array(X, dtype=float, copy=True); idx = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    cols = [c for c in CAND_COLS if c < X.shape[1]]
    for g, ii in idx.items():
        X[np.ix_(ii, cols)] -= X[np.ix_(ii, cols)].mean(axis=0)
    return X


def fit(X, y, groups, kind='gbm', seed=0, relative=True, centered=True):
    from sklearn.ensemble import HistGradientBoostingRegressor
    if centered:
        X = center_features(X, groups)
    if relative:
        y = relative_targets(y, groups)
    if kind == 'gbm':   # in-sample 0.93 vs held-out 0.55 with 400 x 31 leaves (10-30): memorizes; kept for reference
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                                              l2_regularization=1., random_state=seed)
    elif kind == 'gbm_small':
        model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.03, max_leaf_nodes=6, min_samples_leaf=80,
                                              l2_regularization=10., random_state=seed)
    elif kind == 'ridge':
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(StandardScaler(), Ridge(alpha=30.))
    else:
        raise ValueError(kind)
    model.fit(X, y)
    return model


def evaluate(model, paths, include_gates=True, eval_target='points20', centered=True):
    """The pilot's metric on held-out rows: per situation, the argmax over {base, alternatives[, gate-rejected]} by the
    model's prediction on the PLANNING-free features, scored by the reserved futures (2000+); against the baseline's own
    choice. Also pairwise accuracy of model vs the planner's cost on branch pairs with different outcomes."""
    gains = []; base_pts = []; by_map = {}; pair_ok = pair_n = 0; cost_ok = 0; picked_kind = {}
    for p in paths:
        for line in open(p, encoding='utf-8'):
            if not line.strip():
                continue
            r = json.loads(line); root = r.get('root_context')
            if not root:
                continue
            reserved = [int(f) for f in r['arms']['base'] if int(f) >= 2000]
            res = lambda runs: float(np.mean([target_value(runs[str(f)], eval_target) for f in reserved if runs.get(str(f))]))
            base_R = res(r['arms']['base'])
            options = []
            if r.get('root_chosen_features'):
                options.append(('base', candidate_vector(root, dict(features=r['root_chosen_features'], rejected=None)), base_R, 'base', r['root_chosen_features'].get('cost')))
            else:
                options.append(('base', None, base_R, 'base', None))
            for name, b in r['arms']['branches'].items():
                if b['kind'] == 'gate' and not include_gates:
                    continue
                if not b.get('features'):
                    continue
                options.append((name, candidate_vector(root, b), res(b['runs']), b['kind'], b.get('cost')))
            vecs = [v for name, v, pts, kind, _ in options if v is not None]
            if vecs:
                M = np.asarray(vecs, dtype=float)
                if centered:
                    cols = [c for c in CAND_COLS if c < M.shape[1]]; M[:, cols] -= M[:, cols].mean(axis=0)
                preds = iter(model.predict(M))
            scored = [((next(preds) if v is not None else -1e9), name, pts, kind) for name, v, pts, kind, _ in options]
            if options[0][1] is None:
                scored[0] = (max(s[0] for s in scored[1:]) if len(scored) > 1 else 0., 'base', base_R, 'base')   # unknown base features: never prefer it
            best = max(scored, key=lambda s: s[0])
            gains.append(best[2] - base_R); base_pts.append(base_R); picked_kind[best[3]] = picked_kind.get(best[3], 0) + 1
            by_map.setdefault(r['map'], []).append((best[2], base_R))
            # pairwise against the cost
            br = [(v, pts, c) for name, v, pts, kind, c in options if v is not None and c is not None]
            if br:
                Mb = np.asarray([b[0] for b in br], dtype=float)
                if centered:
                    cols = [c for c in CAND_COLS if c < Mb.shape[1]]; Mb[:, cols] -= Mb[:, cols].mean(axis=0)
                pb = model.predict(Mb)
            for i in range(len(br)):
                for j in range(i + 1, len(br)):
                    if abs(br[i][1] - br[j][1]) < 1e-9:
                        continue
                    pair_n += 1
                    pair_ok += (pb[i] > pb[j]) == (br[i][1] > br[j][1])
                    cost_ok += (br[i][2] < br[j][2]) == (br[i][1] > br[j][1])
    ratio = float(np.mean([np.mean([a for a, b in v]) / np.mean([b for a, b in v]) for v in by_map.values()]))
    return dict(n=len(gains), mean_gain=float(np.mean(gains)), base=float(np.mean(base_pts)), ratio_equal_map=ratio,
                pairwise_model=pair_ok / max(1, pair_n), pairwise_cost=cost_ok / max(1, pair_n), pairs=pair_n, picked=picked_kind,
                per_map={m: round(np.mean([a for a, b in v]) / np.mean([b for a, b in v]), 3) for m, v in by_map.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', nargs='+', required=True); ap.add_argument('--eval', nargs='+', default=[])
    ap.add_argument('--out', default=''); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--absolute', action='store_true', help='regress raw points instead of within-situation relative points')
    ap.add_argument('--target', default='points20', help='training label: points20 | points10 | points5 | firstpick | blend')
    ap.add_argument('--eval-targets', nargs='*', default=['points20', 'points10'])
    ap.add_argument('--no-center', action='store_true', help='keep absolute candidate features (default: centered per situation)')
    ap.add_argument('--model', default='gbm_small', help='gbm | gbm_small | ridge')
    ap.add_argument('--require-path', action='store_true', help='train only on situations whose branches carry forecast paths')
    o = ap.parse_args()
    REQUIRE_PATH[0] = bool(o.require_path)
    train = [p for g in o.train for p in glob.glob(g)]; evals = [p for g in o.eval for p in glob.glob(g)]
    X, y, groups, meta = load_many(train, target=o.target)
    print('train rows %d from %d situations, %d files; target mean %.2f sd %.2f' % (len(X), len(set(groups)), len(train), y.mean() if len(y) else 0, y.std() if len(y) else 0))
    model = fit(X, y, groups, kind=o.model, seed=o.seed, relative=not o.absolute, centered=not o.no_center)
    # in-sample pairwise accuracy (overfitting / underfitting check)
    Xc = center_features(X, groups) if not o.no_center else X; pred = model.predict(Xc); idx = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    ok = n = 0
    for g, ii in idx.items():
        for a in range(len(ii)):
            for b in range(a + 1, len(ii)):
                i, j = ii[a], ii[b]
                if abs(y[i] - y[j]) < 1e-9:
                    continue
                n += 1; ok += (pred[i] > pred[j]) == (y[i] > y[j])
    print('in-sample pairwise accuracy %.3f over %d pairs' % (ok / max(1, n), n))
    if hasattr(model, 'feature_importances_'):
        pass
    if o.out:
        with open(o.out, 'wb') as f:
            pickle.dump(dict(model=model, features=FEATURE_NAMES), f)
    if evals:
        for et in o.eval_targets:
            for gates in (False, True):
                r = evaluate(model, evals, include_gates=gates, eval_target=et, centered=not o.no_center)
                print('train %s | eval %s%s: n %d, argmax ratio %.3f (mean gain %+.2f on base %.2f), pairwise model %.3f vs cost %.3f over %d pairs, picked %s' % (
                    o.target, et, ' +gates' if gates else '', r['n'], r['ratio_equal_map'], r['mean_gain'], r['base'], r['pairwise_model'], r['pairwise_cost'], r['pairs'], r['picked']))


if __name__ == '__main__':
    main()
