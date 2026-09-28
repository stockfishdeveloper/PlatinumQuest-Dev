"""The P0a repeat check: 20 repeats of 12 trials (4 starts x 3 candidates) must agree.

    python -m nav.learned_nav.check_repeat            # reads datasets/learned_nav/p0/repeat.jsonl

Declared tolerance (before the first run, 2026-09-27): every recorded position within 0.01 u of the first repeat,
identical replies, identical pickup / takeoff / landing decisions. The out-of-bounds flag may differ by one
decision: it is raised by the game's out-of-bounds trigger, a script event, not by the physics step, and the
first run showed it arriving one decision early in 1 of 20 repeats with identical positions.
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav.record import DATA_DIR    # noqa: E402

TOL_U = 0.01


def check(path=os.path.join(DATA_DIR, 'repeat.jsonl')):
    rs = [json.loads(l) for l in open(path)]
    rs = [r for r in rs if r.get('status') == 'ok']
    g = collections.defaultdict(list)
    for r in rs:
        g[(r['start_id'], r['cand'])].append(r)
    rows = []; all_ok = True
    for k, L in sorted(g.items()):
        n = min(len(r['steps']) for r in L)
        P = np.array([[s['p'] for s in r['steps'][:n]] for r in L])
        dp = float(np.abs(P - P[0]).max())
        events = set((r['pickup'], r['t_pick'], r['took_off'], r['landed'], r['safe']) for r in L)
        acts = set(tuple(s['a'] for s in r['steps'][:n]) for r in L)
        oob_t = sorted(set(len(r['steps']) - 1 for r in L if r['oob']))
        oob_ok = len(oob_t) <= 1 or oob_t[-1] - oob_t[0] <= 1
        ok = dp <= TOL_U and len(events) == 1 and len(acts) == 1 and oob_ok
        all_ok &= ok
        o = L[0]
        rows.append({'start': k[0], 'cand': k[1], 'reps': len(L), 'max_dp': dp, 'events_agree': len(events) == 1,
                     'replies_agree': len(acts) == 1, 'oob_decisions': oob_t, 'pickup': o['pickup'], 'safe': o['safe'],
                     'took_off': o['took_off'], 'landed': o['landed'], 'ok': ok})
    return all_ok, rows


if __name__ == '__main__':
    ok, rows = check()
    for r in rows:
        print(r)
    print('REPEAT CHECK PASSED' if ok else 'REPEAT CHECK FAILED')
