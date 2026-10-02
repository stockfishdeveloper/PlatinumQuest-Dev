import sys, math
import numpy as np
sys.path.insert(0, 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent')
from nav.learned_nav.geometry import Geometry
from nav.learned_nav import planner as PL, drills as DR
g = Geometry('KingOfTheMarble_Hunt')
pl = PL.Planner(g, (0, 0, 0), seeds=None)
start = (-27.2, 17.0); gem = np.array([-37.2, 27.0, 20.7])
z0 = g.floor_below(start[0], start[1], 25.0); fr = DR.floor_ref_of(g, gem)
def run(pr, p, v, w, u):
    n = len(pr)
    sim = PL.simulate(pl.steps, g, pl.eidx, pl.dev, np.tile(p, (n, 1)), np.tile(v, (n, 1)), np.tile(w, (n, 1)),
                      np.tile(u, (n, 1)), np.zeros(n), np.tile(u, (n, 1)), np.zeros(n), pr, False, fr)
    return sim, PL.judge(sim, gem)
for sp in (10.0, 12.0):
    h = math.radians(135)
    p = np.array([start[0], start[1], z0 + PL.R + 0.01]); v = np.array([sp * math.cos(h), sp * math.sin(h), 0.0])
    w = np.array([-v[1], v[0], 0.0]) / PL.R; u = PL.SQ2 * np.array([math.cos(h), math.sin(h)])
    pl.set_target(gem, None); pl.floor_ref = fr; pl.approach_line = True; pl.land_steer = False
    pr = pl._line(400, p, v, fr)
    sim, jd = run(pr, p, v, w, u)
    good = np.nonzero((jd['dmin'] < 0.6) & (sim['landed'] >= 0))[0][:20]
    print(f'speed {sp}: {len(good)} landed programs through the gem')
    tg = math.atan2(gem[1] - p[1], gem[0] - p[0])
    variants = [('brake', PL.MODE_BRAKE, 0, 49), ('none', PL.MODE_NONE, 0, 49)]
    for da in (-90, -45, 0, 45, 90):
        for an in (3, 6, 10):
            variants.append((f'dir{da:+d}/{an}', PL.MODE_DIR, da, an))
    for name, m, da, an in variants:
        q = pr.take(good)
        q.after[:] = m; q.after_ang[:] = tg + math.radians(da); q.after_n[:] = an
        s2, j2 = run(q, p, v, w, u)
        print('  %-10s safe %2d/%d  fell %2d  unsettled %2d  reair %2d  mean clear %.2f  best p %.2f' % (
            name, j2['safe'].sum(), len(good), j2['fell'].sum(), (~j2['settled']).sum(), s2['reair'].sum(),
            np.mean(np.minimum(j2['clear'], 3)), j2['p_succ'].max()))
