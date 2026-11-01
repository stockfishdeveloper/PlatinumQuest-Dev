"""Headroom pilot: branch-and-return from recorded situations (phase 2 of docs/IMPROVEMENT_LOOP_PLAN_2026-10-29.md).

A situation is a recorded decision D of a native round. The sim starts LEAD decisions earlier at the recorded state and
plays the baseline for LEAD decisions (a closed-loop prefix: the controller arrives warm, with its route, plan memory and
locks); the state reached is the root, the same for every arm (a reset-state benchmark with one declared protocol). At
the root:
  arm 1  baseline: the production controller continues for H seconds.
  arm 2  more work: the same controller with its search budget multiplied (3x, 10x).
  arm 3  existing candidates ranked by executed continuations: every search of the root decision is recorded; the best K
         valid candidates besides the chosen one (and G candidates the heuristic terminal gates rejected, for the gate audit)
         are executed open loop for their first M replies, then the baseline controller takes over for the rest of H.
  (arm 4, broader search, = arm 3 on a root search run with the candidate-widening settings; a later option.)
Futures: the prefix uses one spawn seed for every run; from the root on, the spawn and respawn generators are reseeded with
the future's seed, the same seed for every branch of one future (common random numbers). Planning futures P choose the
branch (argmax of the mean points over P); reserved futures R score the chosen branch and the baseline, never used to choose.
Reported per situation: points over H for every branch and future, the deployable gain (chosen branch minus baseline on R),
the clairvoyant bound (best branch on R minus baseline on R), the gate audit, time to the first pickup, falls, charged physics
steps per arm.

usage (repo root, PYTHONPATH=ml_agent):
  python ml_agent/tools/sim_bench/branch_bench.py --pool overnight_20261026/qr3_confirm_baseline --seed 916305284 --map KingOfTheMarble_Hunt
         --per-map 20 --out pilot.jsonl [--horizon-s 20 --lead 47 --k 4 --g 2 --planning 2 --reserved 2 --open-loop 16 --work 3 10]
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_world as W
import hunt_world as H
import full_round as FR

HEURISTIC_GATES = ('no_supported_continuation', 'no_braking_backup', 'unstable_continuation', 'ledge_drop', 'side_impact',
                   'cluster_penalty', 'no_progress', 'terminal_stop')   # refined after the first pilot run: see `rejections`


def classify(rows):
    """Bucket per decision index from the recorded round: 'respawn' (2 s after a fall), 'in_group' (between the first and
    last pickup of a group), 'approach' (after a group's last pickup until the next group's first), 'countdown'."""
    n = len(rows); bucket = ['approach'] * n
    pick = [float(r['outcome'].get('gem_delta') or 0) > 0 for r in rows]
    fell = [bool(r['outcome'].get('fell')) for r in rows]
    gemsets = [frozenset(i['id'] for i in r['before']['items'] if i.get('gem')) for r in rows]
    # group boundaries: the gem set grows
    first_pick = None
    for i in range(n):
        if i and len(gemsets[i] - gemsets[i - 1]) >= 1:
            first_pick = None
        if pick[i]:
            if first_pick is None:
                first_pick = i
            # inside a group from its first pickup until its last: mark later
    # second pass: in_group = from a group's first pickup to its last pickup
    i = 0
    while i < n:
        j = i
        while j + 1 < n and not (len(gemsets[j + 1] - gemsets[j]) >= 1):
            j += 1
        picks = [k for k in range(i, j + 1) if pick[k]]
        if len(picks) >= 2:
            for k in range(picks[0] + 1, picks[-1] + 1):
                bucket[k] = 'in_group'
        i = j + 1
    for i in range(n):
        if fell[i]:
            for k in range(i, min(n, i + 32)):
                bucket[k] = 'respawn'
        st = rows[i]['before']['native']['state']
        if st.get('mode', 0) != 0 or st.get('oob') or rows[i]['choice'].get('label') == 'countdown_neutral':
            bucket[i] = 'countdown'
    return bucket


def sample_situations(rows, per_map, lead, horizon_decisions, rng):
    """Decisions drawn uniformly in round time (stratified by bucket share), each with a playable start LEAD decisions
    earlier and at least `horizon_decisions` of round left."""
    bucket = classify(rows)
    ok = []
    for i in range(lead, len(rows) - 1):
        if bucket[i] == 'countdown':
            continue
        s = rows[i - lead]['before']['native']['state']
        if s.get('mode', 0) != 0 or s.get('oob') or rows[i - lead]['choice'].get('label') == 'countdown_neutral':
            continue
        if float(rows[i - lead]['raw_observation'][31]) < (lead + horizon_decisions) * 64 + 2000:
            continue
        ok.append(i)
    if not ok:
        return []
    chosen = sorted(rng.choice(ok, size=min(per_map, len(ok)), replace=False).tolist())
    return [(i, bucket[i]) for i in chosen]


_FEATURE_KEYS = ('cost', 'minimum_gem_distance_u', 'resources_consumed', 'inventory_consumed', 'terminal_support', 'stable_continuation',
                 'maximum_tail_normal_speed_u_per_s', 'continuation_s', 'possible_pickup_interval', 'through_post_u', 'evaluated_replies',
                 'requested_replies', 'goal_kind', 'powerup', 'detour_u', 'stopped_on_ambiguity')


def features(info, roll):
    """The planner's own judgement of a candidate (its info dict, JSON-safe parts) and the forecast's end state: the inputs
    of a learned ranker, saved beside the executed outcome."""
    f = {}
    for k in _FEATURE_KEYS:
        v = info.get(k)
        if isinstance(v, (bool, int, float, str)) or v is None:
            f[k] = v
    bb = info.get('braking_backup')
    if isinstance(bb, dict):
        f['backup_supported'] = bool(bb.get('supported')); f['backup_drop_u'] = float(bb.get('drop_u', 0.) or 0.); f['backup_replies'] = int(bb.get('replies', 0) or 0)
    ep = info.get('end_position')
    if ep is not None:
        f['end_position'] = [float(v) for v in ep]
    try:
        end = roll['states'][-1]
        f['end_velocity'] = [float(v) for v in end['velocity']]; f['end_held'] = int(end.get('held', 0)); f['end_z'] = float(end['position'][2])
        f['replies'] = len(roll.get('proposed') or [])
        f['max_speed'] = float(max(np.linalg.norm(np.asarray(s['velocity'])) for s in roll['states']))
        f['blast_events'] = int(sum(1 for c in roll.get('controls', []) for e in c.get('script_events', []) if e.get('kind') == 'blast_impulse'))
        # 10-30: the forecast's path itself (every 4th reply), for trajectory features at training time: hand features of the
        # end state alone did not rank branches on held-out data (pairwise 0.45-0.55 against the cost's 0.51)
        f['path'] = [[round(float(v), 3) for v in s['position']] for s in roll['states'][::4]]
        f['path_speed'] = [round(float(np.linalg.norm(s['velocity'])), 2) for s in roll['states'][::4]]
    except Exception:
        pass
    return f


def candidate_set(recorded, chosen_label, k, g):
    """The root decision's recorded searches -> (valid alternatives ranked by cost, gate-rejected ones), with their actions."""
    valid = []; rejected = []; seen = set(); reasons = {}
    for rec in recorded:
        for info, roll in rec['pairs']:
            if roll is None or not isinstance(info, dict):
                continue
            acts = roll.get('proposed') if isinstance(roll, dict) else None
            if not acts:
                continue
            key = (info.get('label'), tuple(rec['gem']))
            if key in seen:
                continue
            seen.add(key)
            c = dict(label=info.get('label'), gem=rec['gem'], cost=float(info.get('cost', math.inf)), rejected=info.get('rejected'),
                     actions=[list(map(float, a)) for a in acts], pickup=info.get('possible_pickup_interval'), features=features(info, roll))
            if info.get('rejected'):
                reasons[info['rejected']] = reasons.get(info['rejected'], 0) + 1
                if roll.get('failure') is None:
                    rejected.append(c)
            elif math.isfinite(c['cost']):
                valid.append(c)
    valid.sort(key=lambda c: c['cost']); rejected.sort(key=lambda c: c['cost'])
    # candidates whose first open-loop replies are the same are one branch (fast_16 / direct_fast_16 share their actions)
    def dedupe(cs, n=16):
        out = []; keys = set()
        for c in cs:
            key = tuple(tuple(round(v, 4) for v in a) for a in c['actions'][:n])
            if key in keys:
                continue
            keys.add(key); out.append(c)
        return out
    valid = dedupe(valid); rejected = dedupe(rejected)
    valid = [c for c in valid if c['label'] != chosen_label][:k]
    rejected = [c for c in rejected if c['rejected'] in HEURISTIC_GATES][:g]
    return valid, rejected, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True); ap.add_argument('--seed', required=True); ap.add_argument('--map', required=True)
    ap.add_argument('--per-map', type=int, default=20); ap.add_argument('--sample-seed', type=int, default=1)
    ap.add_argument('--out', required=True); ap.add_argument('--label', default='pilot')
    ap.add_argument('--horizon-s', type=float, default=20.); ap.add_argument('--lead', type=int, default=47)
    ap.add_argument('--k', type=int, default=4); ap.add_argument('--g', type=int, default=2); ap.add_argument('--open-loop', type=int, default=16)
    ap.add_argument('--planning', type=int, default=2); ap.add_argument('--reserved', type=int, default=2)
    ap.add_argument('--work', type=float, nargs='*', default=[3., 10.])
    ap.add_argument('--set', action='append', default=[]); ap.add_argument('--pflag', action='append', default=[])
    ap.add_argument('--budget', type=float, default=2.); ap.add_argument('--native-step-us', type=float, default=0.)
    ap.add_argument('--only', type=int, nargs='*', default=[], help='run only these recorded decisions')
    ap.add_argument('--normal-priority', action='store_true'); ap.add_argument('--trace', action='store_true')
    ap.add_argument('--label-mode', action='store_true', help='label factory: no work arms, every future reserved (use --planning 0), features saved')
    o = ap.parse_args()
    if o.label_mode:
        o.work = []
    if not o.normal_priority:
        import ctypes
        _k = ctypes.windll.kernel32; _k.GetCurrentProcess.restype = ctypes.c_void_p
        _k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]; _k.SetPriorityClass(_k.GetCurrentProcess(), 0x40)
    W.install_virtual_clock(); W.apply_settings(o.set)
    a = W.planner_args(o.map, budget_s=o.budget, pflags=o.pflag)
    terrain, engine = W.setup(a, o.map, native=True, native_step_us=o.native_step_us)
    W.apply_settings(o.set)
    mission = H.MissionHunt(o.map)
    rows = W.load_journal(W.journal_path(o.pool, o.map, o.seed))
    H_dec = int(o.horizon_s / .064); total = o.lead + H_dec
    rng = np.random.default_rng(o.sample_seed)
    sits = sample_situations(rows, o.per_map, o.lead, H_dec, rng)
    if o.only:
        idx = {r['decision']: i for i, r in enumerate(rows)}; b = classify(rows)
        sits = [(idx[d], b[idx[d]]) for d in o.only]
    planning = [1000 + i for i in range(o.planning)]; reserved = [2000 + i for i in range(o.reserved)]
    futures = planning + reserved
    prefix_seed = 500
    class O:   # the options play_round reads
        cap_s = 0.; trace = o.trace
    base_budget = float(a.search_budget_s)
    with open(o.out, 'a', encoding='utf-8') as out:
        for i, bucket in sits:
            t0 = time.process_time(); start = rows[i - o.lead]; dec = rows[i]['decision']
            res = dict(label=o.label, map=o.map, pool=o.pool, seed=o.seed, decision=dec, bucket=bucket, lead=o.lead, horizon_s=o.horizon_s,
                       planning=planning, reserved=reserved, arms={})
            # arm 1 on every future; the first run records the root's candidates
            a.search_budget_s = base_budget
            base = {}; recorded = None; chosen_label = None; steps = {}
            for f in futures:
                s0 = W.STEPS[0]
                r = FR.play_round(a, terrain, engine, mission, rows, prefix_seed, O, label='base', start=start, max_decisions=total,
                                  root_at=o.lead, record_root=(recorded is None), future_seed=f)
                steps['base', f] = W.STEPS[0] - s0
                base[f] = r['after_root']
                if recorded is None:
                    recorded = r['root_candidates']; chosen_label = r['labels'] and None
                    # the label chosen at the root is the chosen pair of the LAST recorded search of that decision
                    for rec in recorded[::-1]:
                        if rec['chosen'] is not None:
                            chosen_label = rec['chosen'][0].get('label')
                            res['root_chosen_features'] = features(rec['chosen'][0], rec['chosen'][1]) if rec['chosen'][1] is not None else None
                            break
            res['root_chosen'] = chosen_label; res['root_context'] = r.get('root_context')
            valid, rejected, reasons = candidate_set(recorded or [], chosen_label, o.k, o.g)
            res['rejections'] = reasons; res['n_valid'] = len(valid); res['n_rejected'] = len(rejected)
            # arm 3 branches (valid alternatives) and the gate audit (rejected ones)
            branches = {}
            for kind, cands in (('alt', valid), ('gate', rejected)):
                for ci, c in enumerate(cands):
                    name = '%s%d:%s' % (kind, ci, c['label'])
                    branches[name] = dict(kind=kind, label=c['label'], cost=c['cost'], rejected=c['rejected'], gem=c['gem'], runs={},
                                          features=c.get('features'), pickup=c.get('pickup'))
                    for f in futures:
                        s0 = W.STEPS[0]
                        r = FR.play_round(a, terrain, engine, mission, rows, prefix_seed, O, label=name, start=start, max_decisions=total,
                                          root_at=o.lead, root_actions=c['actions'][:o.open_loop], future_seed=f)
                        branches[name]['runs'][str(f)] = r['after_root']; steps[name, f] = W.STEPS[0] - s0
            # arm 2: more work, reserved futures only
            work = {}
            for mult in o.work:
                a.search_budget_s = base_budget * mult
                for f in reserved:
                    s0 = W.STEPS[0]
                    r = FR.play_round(a, terrain, engine, mission, rows, prefix_seed, O, label='work%g' % mult, start=start, max_decisions=total,
                                      root_at=o.lead, future_seed=f)
                    work['%g:%d' % (mult, f)] = r['after_root']; steps['work%g' % mult, f] = W.STEPS[0] - s0
            a.search_budget_s = base_budget
            # selection: planning futures choose, reserved futures score
            pts = lambda d: float(d['points']) if d else 0.
            base_P = np.mean([pts(base[f]) for f in planning]) if planning else float('nan'); base_R = np.mean([pts(base[f]) for f in reserved])
            options = {'base': base_P}
            for name, b in branches.items():
                if b['kind'] == 'alt':
                    options[name] = np.mean([pts(b['runs'][str(f)]) for f in planning]) if planning else float('-inf')
            sel = max(options, key=options.get) if planning else 'base'
            sel_R = base_R if sel == 'base' else np.mean([pts(branches[sel]['runs'][str(f)]) for f in reserved])
            alt_R = {name: np.mean([pts(b['runs'][str(f)]) for f in reserved]) for name, b in branches.items() if b['kind'] == 'alt'}
            gate_R = {name: np.mean([pts(b['runs'][str(f)]) for f in reserved]) for name, b in branches.items() if b['kind'] == 'gate'}
            res['arms'] = dict(base={str(f): base[f] for f in futures}, branches=branches, work=work)
            res['summary'] = dict(base_P=base_P, base_R=base_R, selected=sel, selected_R=sel_R, deployable_gain=sel_R - base_R,
                                  clairvoyant_alt=(max(alt_R.values()) if alt_R else base_R) - base_R,
                                  clairvoyant_all=(max(list(alt_R.values()) + list(gate_R.values()) + [base_R])) - base_R,
                                  gate_best=(max(gate_R.values()) - base_R) if gate_R else None,
                                  work={m: np.mean([pts(work['%g:%d' % (m, f)]) for f in reserved]) - base_R for m in o.work})
            res['steps'] = {'%s|%s' % k: v for k, v in steps.items()}; res['cpu_s'] = time.process_time() - t0
            out.write(json.dumps(res, default=float) + '\n'); out.flush()
            s = res['summary']
            print('%s %-14s dec %5d %-9s base P %.1f R %.1f | sel %-28s R %.1f gain %+.1f | clairvoyant alt %+.1f all %+.1f gate %s | work %s | valid %d rej %d | cpu %.0fs' % (
                o.label, o.map[:14], dec, bucket, base_P, base_R, sel[:28], sel_R, s['deployable_gain'], s['clairvoyant_alt'], s['clairvoyant_all'],
                '%+.1f' % s['gate_best'] if s['gate_best'] is not None else '-', {m: '%+.1f' % v for m, v in s['work'].items()},
                len(valid), len(rejected), res['cpu_s']), flush=True)


if __name__ == '__main__':
    main()
