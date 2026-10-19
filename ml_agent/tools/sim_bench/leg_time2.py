"""Long legs (> 8 s, no fall) split by straight distance; per map totals. usage like leg_time.py"""
import sys, glob, os
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leg_time import legs_of
for arg in sys.argv[1:]:
    per = defaultdict(lambda: defaultdict(float)); rounds = Counter()
    for pool in arg.split('+'):
        for f in sorted(glob.glob(pool + '/*/7347019/physics_plan/1/decisions.jsonl.gz')):
            m = f.replace(os.sep, '/').split('/')[1][:10]; rounds[m] += 1
            for L in legs_of(f):
                T = L['t'] * .064; straight = float(np.linalg.norm((L['end'] - L['start'])[:2]))
                dz = float(L['end'][2] - L['start'][2])
                if L['falls']: k = 'fall'
                elif T > 8 and straight < 20: k = 'stuck'
                elif T > 8: k = 'far'
                else: k = 'normal'
                per[m][k] += T; per[m][k + '_n'] += 1; per[m][k + '_ex'] += T - straight / 10.
                lab = L['labels']; tot = sum(lab.values())
                per[m][k + '_hold'] += lab.get('hold', 0) * .064; per[m][k + '_A2'] += lab.get('A2', 0) * .064
                if k == 'far': per[m]['far_dist'] += straight
    print('== %s  (seconds per round; stuck = >8 s leg to a gem < 20 u away; far = >8 s leg >= 20 u)' % arg[:50])
    print('%-11s %3s %7s %7s %7s %7s | %6s %6s | %8s' % ('map', 'n', 'normal', 'stuck', 'far', 'fall', 'stk_hd', 'stk_A2', 'far_dist'))
    for m in sorted(per):
        r = rounds[m]; p = per[m]
        print('%-11s %3d %7.1f %7.1f %7.1f %7.1f | %6.1f %6.1f | %8.1f' % (m, r, p['normal'] / r, p['stuck'] / r, p['far'] / r, p['fall'] / r,
              p['stuck_hold'] / r, p['stuck_A2'] / r, p['far_dist'] / max(1, p['far_n'])))
    allr = sum(rounds.values()); agg = defaultdict(float)
    for m in per:
        for k, v in per[m].items(): agg[k] += v
    print('%-11s %3d %7.1f %7.1f %7.1f %7.1f | %6.1f %6.1f | %8.1f' % ('ALL', allr, agg['normal'] / allr, agg['stuck'] / allr, agg['far'] / allr, agg['fall'] / allr,
          agg['stuck_hold'] / allr, agg['stuck_A2'] / allr, agg['far_dist'] / max(1, agg['far_n'])))
    print('far legs: n %d, mean T %.1f s, mean straight %.1f u, implied speed %.2f u/s' % (agg['far_n'], agg['far'] / max(1, agg['far_n']), agg['far_dist'] / max(1, agg['far_n']), agg['far_dist'] / max(1e-9, agg['far'])))
