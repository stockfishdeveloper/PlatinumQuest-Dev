"""Offline: the planner's first plan for a centre-gem -> corner-gem jump leg from pickup-like states."""
import sys, math, time
import numpy as np
sys.path.insert(0, 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent')
from nav.learned_nav.geometry import Geometry
from nav.learned_nav import planner as PL, drills as DR
g = Geometry('KingOfTheMarble_Hunt')
pl = PL.Planner(g, (0, 0, 0), seeds=None)
start = (-27.2, 17.0); gem = np.array([-37.2, 27.0, 20.7])
z0 = g.floor_below(start[0], start[1], 25.0)
print('floor at start', z0)
for tw in (0.015,):
    for hd_deg in (180, 135, 90):              # heading at the pickup: west, north-west (toward the corner), north
        for sp in (4.0, 7.7):
            h = math.radians(hd_deg)
            p = np.array([start[0], start[1], z0 + PL.R + 0.01]); v = np.array([sp * math.cos(h), sp * math.sin(h), 0.0])
            w = np.array([-v[1], v[0], 0.0]) / PL.R
            pl.set_target(gem, None); pl.floor_ref = DR.floor_ref_of(g, gem); pl.n_broad = PL.N_BROAD
            pl.time_w = tw; pl.approach_line = True; pl.land_steer = True
            u = PL.SQ2 * np.array([math.cos(h), math.sin(h)])
            res = []
            for rep in range(3):
                pl.reset(); pl.rng = np.random.default_rng(rep)
                js, info = pl.plan(p, v, w, None, u, 0.0, u, 0.0, False, False, pl.floor_ref)
                path = info['pred_path']; tc = info['t_close']
                spd = None
                if info['kind'] == 'pickup':
                    dp = path[min(tc, len(path) - 1)] - path[max(tc - 1, 0)]
                    spd = math.hypot(dp[0], dp[1]) / 0.064
                res.append((info['kind'], round(info.get('p_succ', 0), 2), tc, info['jump_at'], None if spd is None else round(spd, 1)))
            print('time_w %.3f heading %3d speed %.1f:' % (tw, hd_deg, sp), res)
