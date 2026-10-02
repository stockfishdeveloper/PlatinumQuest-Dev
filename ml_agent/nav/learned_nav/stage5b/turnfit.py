import json, sys, math
import numpy as np
sys.path.insert(0, 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent')
ML = 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent/'
from nav.learned_nav.geometry import Geometry
from nav.learned_nav import planner as PL, drills as DR
g = Geometry('KingOfTheMarble_Hunt')
cache = {}
def fine(a, b):
    k = (round(b[0], 1), round(b[1], 1))
    if k not in cache:
        cache[k] = PL.walk_distance_field(g, *PL.gem_disc(np.asarray(b)), DR.floor_ref_of(g, np.asarray(b, float)))
    return float(PL.field_at(g, cache[k], a[0], a[1]))
def first_dir(a, b):
    """initial direction of the walk from a to b (steepest descent over 1 u)"""
    k = (round(b[0], 1), round(b[1], 1)); fine(a, b)
    ang = np.linspace(-math.pi, math.pi, 32, endpoint=False)
    w = PL.field_at(g, cache[k], a[0] + np.cos(ang), a[1] + np.sin(ang))
    return ang[int(np.argmin(w))]
def last_dir(a, b):
    """final direction of the walk from a into b: reverse of the descent from b toward a"""
    k = (round(a[0], 1), round(a[1], 1)); fine(b, a)
    ang = np.linspace(-math.pi, math.pi, 32, endpoint=False)
    w = PL.field_at(g, cache[k], b[0] + 1.5 * np.cos(ang), b[1] + 1.5 * np.sin(ang))
    return ang[int(np.argmin(w))] + math.pi
X, T = [], []
for tag in ('kotm_navlegs', 'kotm_control', 'kotm_dry'):
    for line in open(ML + f'logs/learned_nav/rounds/{tag}_KingOfTheMarble_Hunt.jsonl'):
        L = json.loads(line).get('legs', [])
        for z, a, b in zip(L, L[1:], L[2:]):
            if a['i0'] - z['pick_i'] <= 1 and b['i0'] - a['pick_i'] <= 1:
                t = (b['pick_i'] - a['pick_i']) * 0.064
                w = fine(a['gem'], b['gem']); st = math.hypot(a['gem'][0] - b['gem'][0], a['gem'][1] - b['gem'][1])
                if not np.isfinite(w):
                    continue
                din = last_dir(z['gem'], a['gem']); dout = first_dir(a['gem'], b['gem'])
                turn = 1 - math.cos(dout - din)
                X.append((w, float(w - st >= 3.0), turn)); T.append(t)
X = np.array(X); T = np.array(T)
print('legs', len(T))
for cols, name in (((0,), 'walk'), ((0, 1), 'walk+detour'), ((0, 2), 'walk+turn'), ((0, 1, 2), 'walk+detour+turn')):
    A = np.c_[X[:, cols], np.ones(len(T))]
    c = np.linalg.lstsq(A, T, rcond=None)[0]; res = T - A @ c
    print('%-18s coef %s  resid sd %.3f' % (name, np.round(c, 3), res.std()))
