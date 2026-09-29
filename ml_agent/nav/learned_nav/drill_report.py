"""Summary of stage 4 drill results (drills.py): the M3 gate numbers, by target, start category and failure type.

    python -m nav.learned_nav.drill_report test         # -> logs/learned_nav/drills/report_test.json
Gate (design M3): pickup plus continuation on at least 80 % of at least 200 feasible (fixture-safe) held-out starts,
balanced over the four targets; stalls are failures. Unsafe starts and abstentions are reported apart.
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav.drills import OUT_DIR                                      # noqa: E402
from nav.learned_nav.make_drill_map import DRILLS                               # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def rate(rows):
    k = sum(r['success'] for r in rows); n = len(rows)
    return {'success': k, 'n': n, 'rate': round(k / n, 4) if n else None, 'ci95': wilson(k, n)}


def main(set_name='test'):
    rows = []
    for m in DRILLS:
        p = os.path.join(OUT_DIR, f'{set_name}_{m}.jsonl')
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p)]
    safe = [r for r in rows if r['safe_start']]
    unsafe = [r for r in rows if not r['safe_start']]
    status = {}
    for r in safe:
        status[r['status']] = status.get(r['status'], 0) + 1
    # observed-state launch checks: predicted P(success) at each jump sent from the floor against its outcome
    # jump presses within LAUNCH_GAP decisions of each other are one launch (a press while the marble is already
    # leaving the floor does nothing; a real flight lasts longer); the launch counts as picked if any of its presses
    # was marked, and keeps the first press's P
    LAUNCH_GAP = 6
    launches = []
    for r in safe:
        grp = None
        for j in r['jumps']:
            if grp is not None and j['i'] - grp['last'] <= LAUNCH_GAP:
                grp['last'] = j['i']; grp['picked'] |= bool(j.get('picked'))
                continue
            grp = {'p_succ': j['p_succ'], 'last': j['i'], 'picked': bool(j.get('picked'))}
            launches.append(grp)
    bins = [(0.0, 0.35), (0.35, 0.45), (0.45, 0.6), (0.6, 0.75), (0.75, 0.9), (0.9, 1.01)]
    cal = []
    for lo, hi in bins:
        js = [j for j in launches if lo <= j['p_succ'] < hi]
        got = sum(1 for j in js if j['picked'])
        cal.append({'p_succ': [lo, hi], 'jumps': len(js), 'picked': got, 'rate': round(got / len(js), 3) if js else None})
    rep = {'set': set_name, 'starts': len(rows), 'safe_starts': len(safe), 'unsafe_starts': len(unsafe),
           'gate': rate(safe), 'gate_pass': bool(len(safe) >= 200 and safe and rate(safe)['rate'] >= 0.8),
           'by_target': {m: rate([r for r in safe if r['map'] == m]) for m in DRILLS},
           'by_category': {c: rate([r for r in safe if r['cat'] == c]) for c in ('rest', 'toward', 'side', 'away')},
           'status_safe_starts': status,
           'abstentions': sum(1 for r in safe if r['status'] == 'abstain'),
           'unsafe': rate(unsafe) | {'status': {s: sum(1 for r in unsafe if r['status'] == s) for s in {r['status'] for r in unsafe}}},
           'jumps_per_trial': round(sum(r['n_jumps'] for r in safe) / max(1, len(safe)), 2),
           'no_takeoff': sum(r['no_takeoff'] for r in safe),
           'time_to_pickup_s': sorted(round(0.064 * (r['t_pick'] + 1), 2) for r in safe if r['picked']),
           'launch_check_calibration': cal,
           'plan_ms_mean': round(sum(r['ms_mean'] for r in rows) / max(1, len(rows)), 1),
           'plan_ms_p95_max': round(max((r['ms_p95'] for r in rows), default=0.0), 1),
           'starts_sha1': sorted({(r['map'], r['starts_sha1']) for r in rows}), 'time': time.strftime('%Y-%m-%d %H:%M')}
    tp = rep['time_to_pickup_s']
    rep['time_to_pickup_median_s'] = tp[len(tp) // 2] if tp else None
    del rep['time_to_pickup_s']
    json.dump(rep, open(os.path.join(OUT_DIR, f'report_{set_name}.json'), 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'test')
