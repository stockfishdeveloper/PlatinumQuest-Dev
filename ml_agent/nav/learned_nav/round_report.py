"""Summary of whole-round runs (rounds.py, hybrid.py, and the navigator's nav/real_run.py JSON) per arm.

    python -m nav.learned_nav.round_report hyb_current hyb_reset planner      # tags under logs/learned_nav/rounds/
Prints and writes logs/learned_nav/rounds/report.json: per arm the rounds, mean points (with the spread), gems floor /
floating, falls (and whose), falls within 3 s after a handback, handovers, planning time.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(HERE, 'logs', 'learned_nav', 'rounds')


def arm(tag, map_name='kotmjump'):
    p = os.path.join(DIR, f'{tag}_{map_name}.jsonl')
    rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    if not rows:
        return None
    pts = np.array([r['points'] for r in rows], float)
    out = {'rounds': len(rows), 'points_mean': round(float(pts.mean()), 2),
           'points_sd': round(float(pts.std(ddof=1)), 2) if len(rows) > 1 else None, 'points': pts.tolist(),
           'gems_floor_mean': round(float(np.mean([r['gems_floor'] for r in rows])), 2),
           'gems_float_mean': round(float(np.mean([r['gems_float'] for r in rows])), 2)}
    if 'falls_nav' in rows[0]:
        out.update({'falls_nav_mean': round(float(np.mean([r['falls_nav'] for r in rows])), 2),
                    'falls_planner_mean': round(float(np.mean([r['falls_planner'] for r in rows])), 2),
                    'falls_after_handback_total': int(sum(r['falls_after_handback'] for r in rows)),
                    'handovers_total': int(sum(r['handovers'] for r in rows)),
                    'planner_share': round(float(np.sum([r['planner_decisions'] for r in rows]) /
                                                 max(1, np.sum([r['decisions'] for r in rows]))), 3)})
    else:
        out['falls_mean'] = round(float(np.mean([r['falls'] for r in rows])), 2)
    out['plan_ms_mean'] = round(float(np.mean([r.get('plan_ms_mean', 0.0) for r in rows])), 0)
    return out


def navigator_baseline(map_name='kotmjump'):
    """The navigator alone: nav/real_run.py's result files for this map tagged stage5_baseline."""
    fs = glob.glob(os.path.join(HERE, 'logs', 'nav', f'real_run_{map_name}_*stage5_baseline.json'))
    if not fs:
        return None
    d = json.load(open(sorted(fs)[-1]))
    pts = np.array([r['points'] for r in d['rounds']], float)
    return {'rounds': len(pts), 'points_mean': round(float(pts.mean()), 2), 'points': pts.tolist(),
            'gems_mean': round(float(np.mean([r['gems'] for r in d['rounds']])), 2),
            'falls_mean': round(float(np.mean([r['falls'] for r in d['rounds']])), 2), 'checkpoint': d.get('ckpt')}


def main(tags):
    rep = {'navigator_alone': navigator_baseline()}
    for t in tags:
        rep[t] = arm(t)
    json.dump(rep, open(os.path.join(DIR, 'report.json'), 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
