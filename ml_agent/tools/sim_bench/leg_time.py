"""Where does the round's time go? Per pickup-to-pickup leg: time, straight distance, path length, falls, labels.
usage (from ml_agent/dynamics_reports): python leg_time.py <pool>[+<pool>] ...
Buckets per leg: fall legs (a fall inside), long legs (> 8 s, no fall), normal legs. For normal legs: time split into
path/straight (detour) and speed. Reference speed 10 u/s.
"""
import gzip, json, glob, sys, os
import numpy as np
from collections import Counter, defaultdict

VREF = 10.
def legs_of(f):
    R = []
    for line in gzip.open(f, 'rt', encoding='utf-8'):
        try: R.append(json.loads(line))
        except Exception: pass
    R.sort(key=lambda d: d['decision'])
    out = []; cur = None
    for d in R:
        st = d['before']['native']['state']; p = np.asarray(st['position'], float); v = np.asarray(st['velocity'], float)
        mode_ok = st.get('mode', 0) == 0 and not st.get('oob')
        if cur is None:
            cur = dict(start=p, t=0, path=0., last=p, falls=0, labels=Counter(), slow=0, speeds=[])
        cur['t'] += 1
        if mode_ok:
            step = np.linalg.norm((p - cur['last'])[:2])
            if step < 3: cur['path'] += step
            cur['speeds'].append(float(np.linalg.norm(v[:2])))
            if np.linalg.norm(v[:2]) < 3: cur['slow'] += 1
        cur['last'] = p
        lab = d['choice'].get('label') or '?'
        fam = 'A2' if lab.startswith('A2') else 'hold' if lab == 'safe_hold' else 'jump' if 'jump' in lab else 'blast' if 'blast' in lab else 'through' if 'through' in lab else 'plan'
        cur['labels'][fam] += 1
        if d['outcome'].get('fell'): cur['falls'] += 1
        if float(d['outcome'].get('gem_delta') or 0) > 0:
            cur['end'] = p; out.append(cur); cur = None
    return out

for arg in sys.argv[1:]:
    tot = defaultdict(float); n = Counter(); worst = []
    for pool in arg.split('+'):
        for f in sorted(glob.glob(pool + '/*/7347019/physics_plan/1/decisions.jsonl.gz')):
            m = f.replace(os.sep, '/').split('/')[1][:10]
            for i, L in enumerate(legs_of(f)):
                T = L['t'] * .064; straight = float(np.linalg.norm((L['end'] - L['start'])[:2])); ideal = straight / VREF
                ex = T - ideal
                if L['falls']: k = 'fall'
                elif T > 8: k = 'long'
                else: k = 'normal'
                n[k] += 1; tot[k + '_time'] += T; tot[k + '_excess'] += ex
                if k == 'normal':
                    tot['n_path'] += L['path']; tot['n_straight'] += straight; tot['n_slow'] += L['slow'] * .064
                    sp = np.mean(L['speeds']) if L['speeds'] else 0; tot['n_speed_w'] += sp * T
                worst.append((ex, m, i, T, straight, L['path'], L['falls'], dict(L['labels'])))
    T = sum(tot[k + '_time'] for k in ('fall', 'long', 'normal'))
    print('== %s: legs %d, time %.0f s' % (arg[:40], sum(n.values()), T))
    for k in ('normal', 'long', 'fall'):
        print('  %-6s legs %4d  time %6.0f s (%4.1f%%)  excess over straight/%g %6.0f s (%4.1f%% of time)' % (k, n[k], tot[k + '_time'], 100 * tot[k + '_time'] / T, VREF, tot[k + '_excess'], 100 * tot[k + '_excess'] / T))
    if n['normal']:
        print('  normal legs: path/straight %.2f, mean speed %.2f u/s, time below 3 u/s %.0f s (%.1f%%)' % (tot['n_path'] / tot['n_straight'], tot['n_speed_w'] / tot['normal_time'], tot['n_slow'], 100 * tot['n_slow'] / tot['normal_time']))
    worst.sort(reverse=True)
    for w in worst[:12]:
        print('   worst: %+6.1f s %s leg %3d  T %5.1f straight %5.1f path %6.1f falls %d %s' % w)
