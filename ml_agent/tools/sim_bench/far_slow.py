"""Far legs: classify every playable decision by contact (air / slope / flat) and speed; for flat & slow, whether the
speed is rising (accelerating), the heading turning (> 20 deg in 0.25 s), or neither.
usage (from ml_agent/dynamics_reports): python far_slow.py <pool>[+<pool>]"""
import sys, glob, os, gzip, json, math
import numpy as np
from collections import Counter, defaultdict
for arg in sys.argv[1:]:
    C = defaultdict(Counter)
    for pool in arg.split('+'):
        for f in sorted(glob.glob(pool + '/*/7347019/physics_plan/1/decisions.jsonl.gz')):
            m = f.replace(os.sep, '/').split('/')[1][:10]
            R = []
            for line in gzip.open(f, 'rt', encoding='utf-8'):
                try: R.append(json.loads(line))
                except Exception: pass
            R.sort(key=lambda d: d['decision']); leg = []
            for d in R:
                leg.append(d)
                if float(d['outcome'].get('gem_delta') or 0) > 0 or d['outcome'].get('fell'):
                    fell = any(x['outcome'].get('fell') for x in leg)
                    p0 = np.asarray(leg[0]['before']['native']['state']['position']); p1 = np.asarray(d['before']['native']['state']['position'])
                    T = len(leg) * .064; straight = float(np.linalg.norm((p1 - p0)[:2]))
                    if not fell and T > 8 and straight >= 20 and float(d['outcome'].get('gem_delta') or 0) > 0:
                        sts = [x['before']['native']['state'] for x in leg]
                        for i, st in enumerate(sts):
                            if st.get('mode', 0) != 0 or st.get('oob') or (leg[i]['choice'].get('label') == 'countdown_neutral'): continue
                            v = np.asarray(st['velocity'], float); sp = float(np.linalg.norm(v[:2]))
                            n = np.asarray(st.get('best_contact_normal') or [0, 0, 0], float)
                            kind = 'air' if np.linalg.norm(n) < .5 else ('slope' if n[2] < .97 else 'flat')
                            band = 'fast' if sp >= 8 else 'slow'
                            key = kind + '_' + band
                            if kind == 'flat' and band == 'slow':
                                j = min(len(sts) - 1, i + 4); v2 = np.asarray(sts[j]['velocity'], float)[:2]; sp2 = float(np.linalg.norm(v2))
                                turn = 0.
                                if sp > .5 and sp2 > .5: turn = math.degrees(math.acos(max(-1, min(1, float(v[:2] @ v2) / (sp * sp2)))))
                                sub = 'turning' if turn > 20 else 'accelerating' if sp2 > sp + .5 else 'braking' if sp2 < sp - .5 else 'steady'
                                key += '_' + sub
                            C[m][key] += 1; C['ALL'][key] += 1
                    leg = []
    keys = ['flat_fast', 'slope_fast', 'air_fast', 'flat_slow_accelerating', 'flat_slow_turning', 'flat_slow_braking', 'flat_slow_steady', 'slope_slow', 'air_slow']
    print('== %s far legs: share of decisions (fast >= 8 u/s)' % arg[:40])
    print('%-11s %6s ' % ('map', 'sec') + ' '.join('%10s' % k.replace('flat_slow_', 'fs_').replace('_fast', '_F').replace('_slow', '_S') for k in keys))
    for m in sorted(C, key=lambda x: (x == 'ALL', x)):
        n = sum(C[m].values())
        print('%-11s %6.0f ' % (m, n * .064) + ' '.join('%10.2f' % (C[m][k] / n) for k in keys))
