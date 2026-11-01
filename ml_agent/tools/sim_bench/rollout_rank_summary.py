"""Summarize rollout_rank.py output: pairwise agreement of the rollout score (and of the planner's cost) with the executed
10 / 20 s outcomes, and the realized gain of the rollout argmax over the baseline on the labels' futures.

usage: python ml_agent/tools/sim_bench/rollout_rank_summary.py rollout/*.jsonl
"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np


def exec_points(runs, horizon):
    vals = []
    for f, run in runs.items():
        if not run:
            continue
        if horizon >= 20:
            vals.append(float(run['points']))
        else:
            t0 = run['root_t_ms']; vals.append(float(sum(p['value'] for p in run['picks'] if p['t_ms'] - t0 <= horizon * 1000)))
    return float(np.mean(vals)) if vals else None


def main():
    rows = [json.loads(l) for g in sys.argv[1:] for p in glob.glob(g) for l in open(p, encoding='utf-8') if l.strip()]
    print('%d situations' % len(rows))
    for horizon in (10, 20):
        agree = defaultdict(lambda: [0, 0]); gains = []; base_pts = []; by_map = defaultdict(list); picked = defaultdict(int)
        for r in rows:
            opts = []
            base_e = exec_points(r['base_runs'], horizon)
            rs = r['rollout'].get(r['root_chosen']) if r.get('root_chosen') else None
            if rs and rs.get('ok'):
                opts.append(('base', rs['score'], rs.get('cost'), base_e, 'base'))
            for lab, b in r['branches'].items():
                s = r['rollout'].get(lab)
                if not s or not s.get('ok'):
                    continue
                e = exec_points(b['runs'], horizon)
                if e is None:
                    continue
                opts.append((lab, s['score'], s.get('cost'), e, b['kind']))
            if len(opts) < 2:
                continue
            for i in range(len(opts)):
                for j in range(i + 1, len(opts)):
                    a, b = opts[i], opts[j]
                    if abs(a[3] - b[3]) < 1e-9:
                        continue
                    truth = a[3] > b[3]
                    agree['rollout'][1] += 1; agree['rollout'][0] += ((a[1] > b[1]) == truth) if a[1] != b[1] else 0.5
                    if a[2] is not None and b[2] is not None:
                        agree['cost'][1] += 1; agree['cost'][0] += ((a[2] < b[2]) == truth) if a[2] != b[2] else 0.5
            if base_e is None:
                continue
            best = max(opts, key=lambda t: t[1])
            gains.append(best[3] - base_e); base_pts.append(base_e); by_map[r['map']].append((best[3], base_e)); picked[best[4]] += 1
        ratio = float(np.mean([np.mean([a for a, b in v]) / max(1e-9, np.mean([b for a, b in v])) for v in by_map.values()])) if by_map else float('nan')
        print('\nhorizon %d s: pairwise agreement rollout %.3f (%d pairs), cost %.3f (%d pairs); argmax equal-map ratio %.3f, mean gain %+.2f on base %.2f, n %d, picked %s' % (
            horizon, agree['rollout'][0] / max(1, agree['rollout'][1]), agree['rollout'][1], agree['cost'][0] / max(1, agree['cost'][1]), agree['cost'][1],
            ratio, float(np.mean(gains)) if gains else float('nan'), float(np.mean(base_pts)) if base_pts else float('nan'), len(gains), dict(picked)))
        print('   per map', {m: round(np.mean([a for a, b in v]) / max(1e-9, np.mean([b for a, b in v])), 3) for m, v in by_map.items()})


if __name__ == '__main__':
    main()
