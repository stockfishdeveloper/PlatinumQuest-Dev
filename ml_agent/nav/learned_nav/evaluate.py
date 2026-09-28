"""P0b: train the predictor, freeze every arm on TRAINING starts, then evaluate on held-out starts.

    python -m nav.learned_nav.evaluate freeze      # train + select (validation inside the training starts), fit the
                                                   # baselines' one parameter each, write configs/learned_nav/p0_frozen.json
                                                   # and the arm trial list; nothing here reads a held-out outcome
    python -m nav.learned_nav.record run --port P --list datasets/learned_nav/p0/arms_list_K.json --out datasets/learned_nav/p0/arms_K.jsonl
    python -m nav.learned_nav.evaluate report      # oracle + arms on held-out starts -> logs/learned_nav/p0_results.json

Arms (KOTMJUMP_START_HERE.md), all choosing from the same 46 candidates:
* learned:      the candidate with the highest predicted chance of pickup plus a safe landing (the scoring variant,
                mirror augmentation and epoch count chosen on validation starts inside the training starts);
* hand-written: nav/physics.py (the existing landing predictor) where it applies (jump at 0..4 holding forward or
                no input): its arc from the jump point (constant rolling speed until the jump, as that predictor
                assumes) must pass within r_pick of the gem and jump_verdict must say 'floor'; the most central such
                pass wins; fallbacks: a 'floor' verdict with the most central pass, then the most central pass;
* fixed rule:   jump at the first decision (0..4) at which the marble will be within D of the lip when the impulse
                fires (lip along the start heading, constant speed), else at 4; hold forward;
* random:       a uniformly random candidate (seeded by the start id).
r_pick and D are each chosen on the training starts (the value giving the most successes there), like the learned
arm's own selection. The oracle (not an arm) is every candidate from every held-out start, recorded in the main
collection: a start is FEASIBLE if any candidate succeeded.
Pass: at least 100 feasible held-out starts, and the learned arm's 95 % Wilson interval on success (pickup plus
safe landing) over them clear of (entirely above) every baseline's.
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav import dynamics as D                                     # noqa: E402
from nav.learned_nav.record import (DATA_DIR, CANDS, N_AIR, TARGET, MAP, cand_index, lip_along, fine,   # noqa: E402
                                    floor_below, _sha)

CONFIG_DIR = os.path.join(HERE, 'configs', 'learned_nav')
FROZEN = os.path.join(CONFIG_DIR, 'p0_frozen.json')
ARMS_LIST = os.path.join(DATA_DIR, 'arms_list.json')
ARMS_OUT = os.path.join(DATA_DIR, 'arms.jsonl')
RESULTS_DIR = os.path.join(HERE, 'logs', 'learned_nav')
RESULTS = os.path.join(RESULTS_DIR, 'p0_results.json')
MODEL_PATH = os.path.join(D.MODEL_DIR, 'p0_flight.pth')
ARMS = ['learned', 'handwritten', 'fixed', 'random']
DT = 0.064
VAL_MOD = 5
MIN_FEASIBLE = 100


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


def is_val(sid, st):
    """Validation cells INSIDE the training starts (same cell scheme as the held-out split, another salt)."""
    from nav.learned_nav.record import split_cell
    bx, by, hb = split_cell(st['req']['x'], st['req']['y'], st['req']['heading'])
    key = f'val:{math.floor(bx)},{math.floor(by)},{math.floor(hb)}'.encode()
    return int(hashlib.md5(key).hexdigest(), 16) % VAL_MOD == 0


def outcome_table(by, sids):
    n = len(sids)
    out = {k: np.full((n, len(CANDS)), np.nan) for k in ('success', 'pickup', 'safe', 'oob', 'censored')}
    for i, sid in enumerate(sids):
        for ci, r in by[sid].items():
            out['success'][i, ci] = r['success']; out['pickup'][i, ci] = r['pickup']; out['safe'][i, ci] = r['safe']
            out['oob'][i, ci] = r['oob']; out['censored'][i, ci] = r['censored']
    return out


def start_arrays(by, sids):
    S = []; G = []
    for sid in sids:
        r = next(iter(by[sid].values()))
        S.append(D.start_features(r['p0'], r['v0'], r['w0'], r['h0'])); G.append(D.gem_rel(r['p0'], r['h0']))
    return np.asarray(S), np.asarray(G)


def pick_success(scores, succ, feasible):
    """Fraction of feasible starts whose argmax candidate succeeded."""
    picks = np.argmax(scores, axis=1)
    got = succ[np.arange(len(picks)), picks]
    got = np.nan_to_num(got)
    return float(got[feasible].mean()) if feasible.any() else float('nan'), picks


# ------------------------------------------------------------------ hand-written arm (nav/physics.py)
_TG = None


def terrain_grid():
    global _TG
    if _TG is None:
        from terrain_obs import TerrainMap
        from nav.terrain import TerrainGrid
        _TG = TerrainGrid(TerrainMap.resolve(MAP))
    return _TG


def hw_arc_dmin(xk, yk, z0, vx, vy, hold):
    """Closest pass of the physics.py arc (same integration as predict_landing) to the gem."""
    from nav import physics as PH
    sp = math.hypot(vx, vy)
    hx, hy = (vx / sp, vy / sp) if sp > 1e-6 else (1.0, 0.0)
    x = xk + hx * sp * PH.LAUNCH_DELAY; y = yk + hy * sp * PH.LAUNCH_DELAY
    a = PH.A_AIR if hold else 0.0
    s = 0.0; v = sp; z = z0; vz = PH.VZ0; best = 1e9; t = 0.0
    while t < 1.6 and z > z0 - 4.0:
        v += a * PH.SUB_DT; vz -= PH.G * PH.SUB_DT; s += v * PH.SUB_DT; z += vz * PH.SUB_DT; t += PH.SUB_DT
        d = math.sqrt((x + hx * s - TARGET[0]) ** 2 + (y + hy * s - TARGET[1]) ** 2 + (z - TARGET[2]) ** 2)
        best = min(best, d)
    return best


def hw_table(start_rec):
    """For the 10 applicable candidates: (ci, dmin, verdict_floor)."""
    from nav import physics as PH
    p0, v0 = start_rec['p0'], start_rec['v0']
    rows = []
    for j in range(5):
        xk = p0[0] + v0[0] * DT * (j + 1); yk = p0[1] + v0[1] * DT * (j + 1)
        has_floor = floor_below(xk, yk, p0[2]) is not None
        for a in (0, 1):
            hold = a == 1
            if not has_floor:
                rows.append((cand_index(j, a), 99.0, False))
                continue
            dmin = hw_arc_dmin(xk, yk, p0[2], v0[0], v0[1], hold)
            verdict, _ = PH.jump_verdict(terrain_grid(), xk, yk, p0[2], v0[0], v0[1], hold=hold)
            rows.append((cand_index(j, a), dmin, verdict == 'floor'))
    return rows


def hw_pick(rows, r_pick):
    good = [r for r in rows if r[1] <= r_pick and r[2]]
    if good:
        return min(good, key=lambda r: r[1])[0]
    safe = [r for r in rows if r[2]]
    if safe:
        return min(safe, key=lambda r: r[1])[0]
    return min(rows, key=lambda r: r[1])[0]


# ------------------------------------------------------------------ fixed rule
def fixed_pick(start_rec, d_lip):
    p0, v0, h0 = start_rec['p0'], start_rec['v0'], start_rec['h0']
    lip0 = lip_along(p0[0], p0[1], math.cos(h0), math.sin(h0))
    sp = math.hypot(v0[0], v0[1])
    for j in range(5):
        if lip0 - sp * (DT * (j + 1) + 0.032) <= d_lip:
            return cand_index(j, 1)
    return cand_index(4, 1)


def random_pick(sid):
    rng = np.random.default_rng(int(hashlib.md5((sid + ':random').encode()).hexdigest(), 16) % (2 ** 32))
    return int(rng.integers(len(CANDS)))


# ------------------------------------------------------------------ freeze
def freeze(epochs=60, log=print):
    import torch
    t0 = time.time()
    by = D.load_trials()
    starts = {sid: next(iter(d.values())) for sid, d in by.items()}
    complete = [sid for sid, d in by.items() if len(d) == len(CANDS)]
    train_sids = sorted(sid for sid in complete if not starts[sid]['held_out'])
    n_ho = sum(1 for sid in complete if starts[sid]['held_out'])
    va = [sid for sid in train_sids if is_val(sid, starts[sid])]
    tr = [sid for sid in train_sids if not is_val(sid, starts[sid])]
    log(f'{len(by)} starts recorded, {len(complete)} complete; training {len(train_sids)} ({len(tr)} fit, {len(va)} validation); '
        f'{n_ho} held out (not read)')
    data_tr0 = D.build(by, tr)
    S_va, G_va = start_arrays(by, va)
    out_va = outcome_table(by, va)
    feas_va = np.nanmax(out_va['success'], axis=1) > 0
    # the pass curve is fitted on the fit starts' own predictions (a subsample, for speed)
    rng = np.random.default_rng(0)
    sub = sorted(rng.choice(len(tr), size=min(300, len(tr)), replace=False))
    S_sub, G_sub = start_arrays(by, [tr[i] for i in sub])
    out_sub = outcome_table(by, [tr[i] for i in sub])

    def eval_fn(model):
        pv = D.predict(model, S_va, G_va)
        ps = D.predict(model, S_sub, G_sub)
        m = ~np.isnan(out_sub['pickup'])
        curve = D.fit_pass_curve(ps['dmin'][m], out_sub['pickup'][m])
        res = {}
        for name in D.VARIANTS:
            _, sc_ = D.score(pv, name, curve)
            sc, _ = pick_success(sc_, out_va['success'], feas_va)
            res[f'val_{name}'] = sc
        mm = ~np.isnan(out_va['safe'])
        ps_ = np.clip(pv['p_safe'][mm], 1e-4, 1 - 1e-4); ys = out_va['safe'][mm]
        res['val_safe_ll'] = float(-np.mean(ys * np.log(ps_) + (1 - ys) * np.log(1 - ps_)))
        return res

    log(f'validation: {int(feas_va.sum())} feasible of {len(va)} starts')
    best = None; hist_all = {}
    for mir in (False, True):
        data_tr = D.with_mirror(data_tr0) if mir else data_tr0
        log(f'--- mirror augmentation {mir}: {len(data_tr["si"])} training rows')
        model, hist = D.train(data_tr, epochs=epochs, eval_fn=eval_fn, log=log)
        hist_all[f'mirror_{mir}'] = hist
        for h in hist:
            for name in D.VARIANTS:
                k = f'val_{name}'
                if k in h and (best is None or h[k] > best[0] + 1e-9):
                    best = (h[k], name, h['epoch'], mir)
    val_score, variant, best_epoch, mir = best
    hist = hist_all
    log(f'selected: variant {variant}, mirror {mir}, {best_epoch} epochs (validation success {val_score:.3f} over '
        f'{int(feas_va.sum())} feasible validation starts)')
    # final fit on every training start with the selected settings
    data_all = D.build(by, train_sids)
    if mir:
        data_all = D.with_mirror(data_all)
    model, _ = D.train(data_all, epochs=best_epoch, log=log)
    curve = None
    if variant == 'pass':
        rng = np.random.default_rng(1)
        subs = sorted(rng.choice(len(train_sids), size=min(600, len(train_sids)), replace=False))
        S_s, G_s = start_arrays(by, [train_sids[i] for i in subs])
        o_s = outcome_table(by, [train_sids[i] for i in subs])
        ps = D.predict(model, S_s, G_s)
        m = ~np.isnan(o_s['pickup'])
        curve = D.fit_pass_curve(ps['dmin'][m], o_s['pickup'][m])
    meta = {'n_s': int(data_all['s_tab'].shape[1]), 'variant': variant, 'mirror': mir, 'epochs': best_epoch, 'curve': curve,
            'train_starts': len(train_sids), 'val_success': val_score}
    D.save(model, meta, MODEL_PATH)
    log(f'saved {MODEL_PATH}')

    # baselines' single parameters, on training starts
    out_tr = outcome_table(by, train_sids)
    feas_tr = np.nanmax(out_tr['success'], axis=1) > 0
    log('hand-written arm: physics.py verdicts on training starts ...')
    hw_rows = [hw_table(starts[sid]) for sid in train_sids]
    best_r = None
    for r_pick in np.arange(0.3, 1.51, 0.05):
        picks = [hw_pick(rows, r_pick) for rows in hw_rows]
        got = np.nan_to_num(out_tr['success'][np.arange(len(picks)), picks])
        sc = float(got[feas_tr].mean())
        if best_r is None or sc > best_r[1] + 1e-9:
            best_r = (float(round(r_pick, 2)), sc)
    best_d = None
    for d_lip in np.arange(0.0, 3.01, 0.25):
        picks = [fixed_pick(starts[sid], d_lip) for sid in train_sids]
        got = np.nan_to_num(out_tr['success'][np.arange(len(picks)), picks])
        sc = float(got[feas_tr].mean())
        if best_d is None or sc > best_d[1] + 1e-9:
            best_d = (float(d_lip), sc)
    log(f'hand-written r_pick {best_r[0]} (training success {best_r[1]:.3f}); fixed D {best_d[0]} (training success {best_d[1]:.3f})')

    # the arms' choices on held-out starts: computed now from the start states only, frozen with the settings
    ho = sorted(sid for sid in complete if starts[sid]['held_out'])
    S_ho, G_ho = start_arrays(by, ho)
    pr = D.predict(model, S_ho, G_ho)
    _, sc_ = D.score(pr, variant, curve)
    learned = np.argmax(sc_, axis=1)
    picks = {}
    for i, sid in enumerate(ho):
        st = starts[sid]
        picks[sid] = {'learned': int(learned[i]), 'handwritten': int(hw_pick(hw_table(st), best_r[0])),
                      'fixed': int(fixed_pick(st, best_d[0])), 'random': random_pick(sid)}
    os.makedirs(CONFIG_DIR, exist_ok=True)
    frozen = {'schema': 'p0_frozen_v1', 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
              'model': os.path.relpath(MODEL_PATH, HERE), 'model_sha1': _sha(MODEL_PATH), 'meta': meta,
              'handwritten_r_pick': best_r[0], 'handwritten_train_success': best_r[1],
              'fixed_d_lip': best_d[0], 'fixed_train_success': best_d[1], 'random_seed_rule': 'md5(start_id + ":random")',
              'held_out_starts': len(ho), 'pass_rule': {'min_feasible': MIN_FEASIBLE, 'interval': 'Wilson 95 %',
                                                         'rule': 'learned lower bound > every baseline upper bound'},
              'picks': picks, 'history': hist}
    json.dump(frozen, open(FROZEN, 'w'), indent=1)
    todo = []
    for sid in ho:
        st = starts[sid]
        s = {'id': sid, 'x': st['req']['x'], 'y': st['req']['y'], 'heading': st['req']['heading'], 'speed': st['req']['speed'],
             'held_out': True}
        for k, arm in enumerate(ARMS):
            todo.append([s, picks[sid][arm], k + 1])            # rep = arm number: each arm runs once per start
    json.dump(todo, open(ARMS_LIST, 'w'))
    log(f'frozen -> {FROZEN}; {len(todo)} arm trials -> {ARMS_LIST}; {time.time() - t0:.0f} s')


# ------------------------------------------------------------------ report
def calib(p, y, bins=10):
    p = np.asarray(p); y = np.asarray(y)
    edges = np.linspace(0, 1, bins + 1); rows = []; ece = 0.0
    for b in range(bins):
        m = (p >= edges[b]) & (p < edges[b + 1] if b < bins - 1 else p <= 1)
        if m.sum() == 0:
            continue
        rows.append({'lo': float(edges[b]), 'hi': float(edges[b + 1]), 'n': int(m.sum()), 'pred': float(p[m].mean()), 'actual': float(y[m].mean())})
        ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return {'bins': rows, 'ece': float(ece), 'brier': float(np.mean((p - y) ** 2))}


def auc(p, y):
    p = np.asarray(p); y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float('nan')
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def report(log=print):
    frozen = json.load(open(FROZEN))
    model, meta = D.load(MODEL_PATH)
    by = D.load_trials()
    picks = frozen['picks']
    ho = sorted(picks)
    starts = {sid: next(iter(by[sid].values())) for sid in ho}
    out = outcome_table(by, ho)
    feasible = np.nanmax(out['success'], axis=1) > 0
    n_feas = int(feasible.sum())
    # engine arm runs (each arm once per start), keyed by (start, arm)
    arm_runs = {}
    for fn in sorted(os.listdir(DATA_DIR)):
        if not (fn.startswith('arms') and fn.endswith('.jsonl')):
            continue
        for ln in open(os.path.join(DATA_DIR, fn)):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get('status') == 'ok' and r.get('schema') == 'p0_trial_v2':
                arm_runs[(r['start_id'], ARMS[r['rep'] - 1])] = r
    res = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'held_out_starts': len(ho), 'feasible': n_feas,
           'oracle_candidates_succeeding': float(np.nanmean(np.nansum(out['success'], axis=1)[feasible])) if n_feas else 0.0,
           'arms': {}, 'frozen': {k: frozen[k] for k in ('handwritten_r_pick', 'fixed_d_lip', 'meta', 'time')}}
    agree = 0; compared = 0
    for arm in ARMS:
        sel = np.array([picks[sid][arm] for sid in ho])
        look = out['success'][np.arange(len(ho)), sel]
        eng = np.array([float(arm_runs[(sid, arm)]['success']) if (sid, arm) in arm_runs else np.nan for sid in ho])
        for a_, b_ in zip(look, eng):
            if not np.isnan(b_):
                compared += 1; agree += int(a_ == b_)
        use = np.where(np.isnan(eng), look, eng)
        k = int(np.nansum(use[feasible])); n = n_feas
        p, lo, hi = wilson(k, n)
        pk = np.array([float(arm_runs[(sid, arm)]['pickup']) if (sid, arm) in arm_runs else out['pickup'][i, sel[i]] for i, sid in enumerate(ho)])
        oo = np.array([float(arm_runs[(sid, arm)]['oob']) if (sid, arm) in arm_runs else out['oob'][i, sel[i]] for i, sid in enumerate(ho)])
        res['arms'][arm] = {'success': k, 'n': n, 'rate': p, 'ci95': [lo, hi],
                            'success_all_heldout': float(np.nanmean(use)), 'pickup_rate_feasible': float(np.nanmean(pk[feasible])),
                            'oob_rate_feasible': float(np.nanmean(oo[feasible])), 'engine_runs': int((~np.isnan(eng)).sum())}
    res['engine_vs_oracle_agreement'] = {'compared': compared, 'agree': agree}
    L = res['arms']['learned']
    best_base = max((a for a in ARMS if a != 'learned'), key=lambda a: res['arms'][a]['ci95'][1])
    res['pass'] = bool(n_feas >= MIN_FEASIBLE and L['ci95'][0] > res['arms'][best_base]['ci95'][1])
    res['best_baseline'] = best_base

    # predictor quality on every held-out trial
    S, G = start_arrays(by, ho)
    pr = D.predict(model, S, G)
    p_pick, ps = D.score(pr, meta['variant'], meta['curve'])
    m = ~np.isnan(out['success'])
    res['predictor'] = {
        'pickup': {**calib(p_pick[m], out['pickup'][m]), 'auc': auc(p_pick[m], out['pickup'][m]),
                   'false_pos': int(((p_pick[m] >= 0.5) & (out['pickup'][m] == 0)).sum()),
                   'false_neg': int(((p_pick[m] < 0.5) & (out['pickup'][m] == 1)).sum()),
                   'positives': int(out['pickup'][m].sum()), 'n': int(m.sum())},
        'safe': {**calib(pr['p_safe'][m], out['safe'][m]), 'auc': auc(pr['p_safe'][m], out['safe'][m])},
        'success': {**calib(ps[m], out['success'][m]), 'auc': auc(ps[m], out['success'][m])}}
    # path and landing error
    errs = []; land_errs = []; path_by_t = [[] for _ in range(D.MAX_STEPS)]
    for i, sid in enumerate(ho):
        for ci, r in by[sid].items():
            P, M = D.path_target(r)
            e = np.linalg.norm(pr['path'][i, ci] - P, axis=1)
            last = r['landed'] + 8 if r['landed'] is not None else len(r['steps'])
            k = min(last, D.MAX_STEPS)
            mm = M[:k] > 0
            if mm.any():
                errs.append(float(e[:k][mm].mean()))
                for t in np.nonzero(mm)[0]:
                    path_by_t[t].append(float(e[t]))
            if r['land_p'] is not None:
                lp = D.frame(r['h0']) @ (np.asarray(r['land_p']) - np.asarray(r['p0']))
                land_errs.append(float(np.linalg.norm(pr['land'][i, ci] - lp)))
    res['predictor']['path_error_u'] = {'median': float(np.median(errs)), 'mean': float(np.mean(errs)),
                                        'p90': float(np.percentile(errs, 90)),
                                        'by_decision_median': [float(np.median(v)) if v else None for v in path_by_t]}
    res['predictor']['landing_error_u'] = {'median': float(np.median(land_errs)), 'mean': float(np.mean(land_errs)),
                                           'p90': float(np.percentile(land_errs, 90)), 'n': len(land_errs)}
    # per start, for the page
    per = []
    for i, sid in enumerate(ho):
        st = starts[sid]
        row = {'id': sid, 'x': st['p0'][0], 'y': st['p0'][1], 'h': st['h0'], 'speed': math.hypot(st['v0'][0], st['v0'][1]),
               'feasible': bool(feasible[i]), 'n_ok': int(np.nansum(out['success'][i]))}
        for arm in ARMS:
            ci = picks[sid][arm]
            r = arm_runs.get((sid, arm)) or by[sid][ci]
            row[arm] = {'cand': ci, 'success': bool(r['success']), 'pickup': bool(r['pickup']), 'oob': bool(r['oob'])}
        ci = picks[sid]['learned']
        r = by[sid][ci]
        F = D.frame(st['h0'])
        pred_w = (F.T @ pr['path'][i, ci].T).T + np.asarray(st['p0'])
        row['learned']['pred_success'] = float(ps[i, ci])
        row['learned']['pred_path'] = [[round(float(a), 2) for a in q] for q in pred_w[:30]]
        row['learned']['path'] = [[round(a, 2) for a in s_['p']] for s_ in r['steps'][:30]]
        per.append(row)
    res['starts'] = per
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(res, open(RESULTS, 'w'))
    log(f'held-out starts {len(ho)}, feasible {n_feas}')
    for arm in ARMS:
        a = res['arms'][arm]
        log(f'  {arm:12s} {a["success"]:4d}/{a["n"]} = {a["rate"]:.3f}  95% [{a["ci95"][0]:.3f}, {a["ci95"][1]:.3f}]  '
            f'pickup {a["pickup_rate_feasible"]:.3f}  oob {a["oob_rate_feasible"]:.3f}  engine runs {a["engine_runs"]}')
    log(f'engine vs oracle agreement: {agree}/{compared}')
    log(f'PASS: {res["pass"]} (best baseline {best_base})')
    pp = res['predictor']
    log(f'pickup AUC {pp["pickup"]["auc"]:.3f} FP {pp["pickup"]["false_pos"]} FN {pp["pickup"]["false_neg"]} of {pp["pickup"]["positives"]} positives; '
        f'safe AUC {pp["safe"]["auc"]:.3f}; success ECE {pp["success"]["ece"]:.3f}; path error median {pp["path_error_u"]["median"]:.2f} u; '
        f'landing error median {pp["landing_error_u"]["median"]:.2f} u')
    return res


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    if cmd == 'freeze':
        freeze()
    elif cmd == 'report':
        report()
