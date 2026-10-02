"""Human gem order: at each pickup, was the next gem picked the nearest visible one (straight line)? Void legs vs all."""
import json, sys, math, glob
import numpy as np
sys.path.insert(0, 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent')
from nav.gems import visible_gems
from nav.learned_nav.geometry import Geometry
g = Geometry('KingOfTheMarble_Hunt')
ML = 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent/'
def over_void(path):
    return sum(1 for x, y, z in path if (g.floor_below(x, y, z) is None or g.floor_below(x, y, z) < z - 3.5))
from collections import Counter
rank_all, rank_void = Counter(), Counter(); nvis = Counter()
for f in sorted(glob.glob(ML + 'demos/demo_*.npz')):
    z = np.load(f, allow_pickle=True); o = z['obs_raw']; gd = z['gem_delta']; game = z['game']; oob = z['oob']
    tots = json.load(open(f.replace('.npz', '.json'))).get('game_totals_from_game', [])
    for gi in np.unique(game):
        if int(gi) >= len(tots) or tots[int(gi)] < 160:
            continue
        idx = np.nonzero(game == gi)[0]
        picks = [t for t in idx if gd[t] > 0]
        for t0, t1 in zip(picks, picks[1:]):
            if oob[t0:t1].any():
                continue
            vis = visible_gems(o[t0 + 1])            # after the pickup at t0
            if len(vis) < 2:
                continue
            p1 = o[t1, 0:3]
            nxt = min(vis, key=lambda q: math.hypot(q[0] - p1[0], q[1] - p1[1]))
            p0 = o[t0, 0:3]
            ds = sorted(vis, key=lambda q: math.hypot(q[0] - p0[0], q[1] - p0[1]))
            r = [k for k, q in enumerate(ds) if q is nxt][0]
            void = over_void(o[t0:t1:4, 0:3]) > 0
            rank_all[r] += 1; nvis[len(vis)] += 1
            if void:
                rank_void[r] += 1
print('rank of the chosen gem by straight distance (0 = nearest), strong human rounds')
print('  all legs ', sorted(rank_all.items()))
print('  void legs', sorted(rank_void.items()))
print('  visible gems after pickup', sorted(nvis.items()))
