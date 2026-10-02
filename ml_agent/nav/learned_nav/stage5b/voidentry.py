import json, sys, math, glob
import numpy as np
sys.path.insert(0, 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent')
from nav.gems import visible_gems
from nav.learned_nav.geometry import Geometry
g = Geometry('KingOfTheMarble_Hunt')
ML = 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent/'
def void_at(x, y, z):
    f = g.floor_below(x, y, z); return f is None or f < z - 3.5
rows = []
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
            seg = o[t0:t1 + 1]
            vv = [void_at(*q[0:3]) for q in seg[::2]]
            if not any(vv):
                continue
            k_void = 2 * vv.index(True)
            # takeoff: last tick before the void run with the marble going up (jump) -- approximate: the tick 10 before void entry
            p0, v0 = seg[0, 0:3], seg[0, 3:6]; p1 = seg[-1, 0:3]
            line = math.atan2(p1[1] - p0[1], p1[0] - p0[0]); hv = math.atan2(v0[1], v0[0])
            sp0 = math.hypot(v0[0], v0[1])
            # speed and heading at the void entry
            ve = seg[max(0, k_void - 3), 3:6]
            # distance from the pickup to the void entry
            dv = math.hypot(seg[k_void, 0] - p0[0], seg[k_void, 1] - p0[1])
            rows.append((sp0, math.degrees((hv - line + math.pi) % (2 * math.pi) - math.pi), math.hypot(ve[0], ve[1]), dv, (t1 - t0) * 0.016,
                         math.hypot(p1[0] - p0[0], p1[1] - p0[1])))
R = np.array(rows)
print('human void legs', len(R))
print('speed at the previous pickup: median %.1f  (p25 %.1f p75 %.1f)' % tuple(np.percentile(R[:, 0], [50, 25, 75])))
print('heading at the pickup vs the leg line (deg): median |angle| %.0f, share within 30 deg %.2f' % (np.median(np.abs(R[:, 1])), (np.abs(R[:, 1]) < 30).mean()))
print('speed entering the void: median %.1f (p25 %.1f p75 %.1f)' % tuple(np.percentile(R[:, 2], [50, 25, 75])))
print('distance pickup -> void entry: median %.1f (p25 %.1f p75 %.1f)' % tuple(np.percentile(R[:, 3], [50, 25, 75])))
print('leg time median %.2f, straight median %.1f' % (np.median(R[:, 4]), np.median(R[:, 5])))
for r in R[:15]:
    print('  v0 %.1f angle %+4.0f v_void %.1f d_void %.1f t %.2f straight %.1f' % tuple(r))
