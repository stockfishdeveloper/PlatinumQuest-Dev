"""Dry-run probes: the planner's jump plan vs the navigator's actual time from the same state."""
import json, sys, math
import numpy as np
ML = 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent/'
tag = sys.argv[1] if len(sys.argv) > 1 else 'kotm_dry'
rows = []
for line in open(ML + f'logs/learned_nav/rounds/{tag}_KingOfTheMarble_Hunt.jsonl'):
    r = json.loads(line)
    print('round points', r['points'], 'probes', r['probes'], 'falls', r['falls_nav'])
    L = r['legs']
    for p in r['probe_log']:
        m = [l for l in L if abs(l['gem'][0] - p['gem'][0]) < 0.5 and abs(l['gem'][1] - p['gem'][1]) < 0.5 and l['i0'] <= p['i'] <= l['pick_i']]
        nav = (m[0]['pick_i'] - p['i']) * 0.064 if m else None
        rows.append(p | {'nav': nav})
print('probes', len(rows), 'matched', sum(r['nav'] is not None for r in rows))
M = [r for r in rows if r['nav'] is not None]
tn = np.array([r['t_nav'] for r in M]); na = np.array([r['nav'] for r in M])
print('t_nav estimate vs actual: mean err %+.3f sd %.3f' % ((tn - na).mean(), (tn - na).std()))
J = [r for r in M if r['kind'] == 'pickup' and r['jump_at'] >= 0]
print('pickup-with-jump plans:', len(J))
for r in sorted(J, key=lambda r: r['nav'] - r['t_close'] * 0.064, reverse=True):
    tp = r['t_close'] * 0.064
    print('  (%6.1f,%5.1f) v %4.1f -> (%6.1f,%5.1f) straight %4.1f walk %4.1f  nav actual %.2f est %.2f  plan %.2f  gain %+.2f  p %.2f ok %s' % (
        r['x'], r['y'], r['speed'], r['gem'][0], r['gem'][1], r['straight'], r['walk'], r['nav'], r['t_nav'], tp, r['nav'] - tp, r.get('p_succ', r['p_best']), r['ok']))
