import json, sys, math, glob, time
import numpy as np
sys.path.insert(0, 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent')
from nav.gems import visible_gems, choose
from nav.learned_nav.geometry import Geometry
from nav.learned_nav.route import Route
from nav.learned_nav.rounds import is_floating
g = Geometry('KingOfTheMarble_Hunt'); R = Route(g)
ML = 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent/'
for a, b in (((-27.2, 17.0), (-37.2, 27.0, 20.7)), ((-23.2, 13.0), (-13.2, 3.0, 20.7)), ((-27.2, 13.0), (-23.2, 17.0, 20.7))):
    print(a, b, 'gap', R.jump_gap(a[0], a[1], b), 'walk', round(R.walk(a[0], a[1], b), 2))
def over_void(path):
    return sum(1 for x, y, z in path if (g.floor_below(x, y, z) is None or g.floor_below(x, y, z) < z - 3.5))
n = 0; over = 0; over_match = 0; void_n = 0; void_hit = 0; ts = []
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
            ob = o[t0 + 1]; vis = visible_gems(ob)
            if len(vis) < 2:
                continue
            gt, _ = choose(vis, None, pos=ob[0:2], vel=ob[3:5])
            c0 = time.perf_counter()
            J, info = R.choose(ob[0:3], ob[3:6], vis, gt, floating=lambda q: is_floating(g, q[:3]))
            ts.append(time.perf_counter() - c0)
            p1 = o[t1, 0:3]
            nxt = min(vis, key=lambda q: math.hypot(q[0] - p1[0], q[1] - p1[1]))
            void = over_void(o[t0:t1:4, 0:3]) > 0
            n += 1; void_n += void
            if J is not None:
                over += 1; over_match += (J is nxt)
                void_hit += (J is nxt and void)
print('pickups', n, 'route overrides', over, 'of which the human took that gem', over_match, '| human void legs', void_n, 'overrides matching a human void leg', void_hit)
print('choose time ms: mean %.1f max %.1f' % (1000 * np.mean(ts), 1000 * np.max(ts)))
