"""Speed profile of far legs (> 8 s, >= 20 u straight, no fall): histogram of horizontal speed, airborne share, labels.
usage like leg_time.py"""
import sys, glob, os, gzip, json
import numpy as np
from collections import Counter, defaultdict
for arg in sys.argv[1:]:
    hist = defaultdict(lambda: np.zeros(8)); air = Counter(); labs = defaultdict(Counter); tt = Counter(); vz = Counter()
    for pool in arg.split('+'):
        for f in sorted(glob.glob(pool + '/*/7347019/physics_plan/1/decisions.jsonl.gz')):
            m = f.replace(os.sep, '/').split('/')[1][:10]
            R = []
            for line in gzip.open(f, 'rt', encoding='utf-8'):
                try: R.append(json.loads(line))
                except Exception: pass
            R.sort(key=lambda d: d['decision'])
            leg = []
            for d in R:
                leg.append(d)
                if float(d['outcome'].get('gem_delta') or 0) > 0 or d['outcome'].get('fell'):
                    fell = any(x['outcome'].get('fell') for x in leg)
                    p0 = np.asarray(leg[0]['before']['native']['state']['position']); p1 = np.asarray(d['before']['native']['state']['position'])
                    T = len(leg) * .064; straight = float(np.linalg.norm((p1 - p0)[:2]))
                    if not fell and T > 8 and straight >= 20 and float(d['outcome'].get('gem_delta') or 0) > 0:
                        for x in leg:
                            st = x['before']['native']['state']
                            if st.get('mode', 0) != 0 or st.get('oob'): continue
                            v = np.asarray(st['velocity'], float); sp = float(np.linalg.norm(v[:2]))
                            hist[m][min(7, int(sp // 2))] += 1; tt[m] += 1
                            # airborne: predicted contact normal missing is not logged in before; use |vz| > 1
                            if abs(v[2]) > 1.: air[m] += 1
                            lab = x['choice'].get('label') or '?'
                            fam = 'A2' if lab.startswith('A2') else 'hold' if lab == 'safe_hold' else lab.split('_')[0]
                            labs[m][fam] += 1
                    leg = []
    print('== %s far legs: share of playable decisions by horizontal speed band (u/s)' % arg[:40])
    print('%-11s %6s ' % ('map', 'sec') + ' '.join('%5s' % ('%d-%d' % (2 * i, 2 * i + 2) if i < 7 else '14+') for i in range(8)) + '  |vz|>1  top labels')
    H = np.zeros(8); TT = 0; A = 0; LL = Counter()
    for m in sorted(hist):
        h = hist[m] / max(1, tt[m]); H += hist[m]; TT += tt[m]; A += air[m]; LL.update(labs[m])
        print('%-11s %6.0f ' % (m, tt[m] * .064) + ' '.join('%5.2f' % x for x in h) + '  %5.2f  %s' % (air[m] / max(1, tt[m]), ' '.join('%s:%.2f' % (k, v / tt[m]) for k, v in labs[m].most_common(5))))
    print('%-11s %6.0f ' % ('ALL', TT * .064) + ' '.join('%5.2f' % x for x in H / TT) + '  %5.2f  %s' % (A / TT, ' '.join('%s:%.2f' % (k, v / TT) for k, v in LL.most_common(7))))
