import json, sys, math, glob
import numpy as np
sys.path.insert(0, 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent')
ML = 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent/'
from nav.learned_nav.geometry import Geometry
from nav.learned_nav import planner as PL
from nav.learned_nav import drills as DR
g = Geometry('KingOfTheMarble_Hunt')

class W(PL.Planner):
    def __init__(self, g):   # geometry only, no models
        self.g = g; self.seeds = None; self.walk = None
cache = {}
def fine_walk(a, b):
    k = (round(b[0], 1), round(b[1], 1))
    if k not in cache:
        w = W(g); w.gem = np.asarray(b, float); fr = DR.floor_ref_of(g, np.asarray(b, float))
        w.walk_field(fr); cache[k] = (w, fr)
    w, fr = cache[k]
    return float(w.walk_at(np.array([a[0]]), np.array([a[1]]), fr)[0])

rows = []
for f in ['kotm_navlegs', 'kotm_control']:
    for line in open(ML + f'logs/learned_nav/rounds/{f}_KingOfTheMarble_Hunt.jsonl'):
        r = json.loads(line)
        L = r.get('legs', [])
        for a, b in zip(L, L[1:]):
            if b['i0'] - a['pick_i'] <= 1:
                t = (b['pick_i'] - a['pick_i']) * 0.064
                fw = fine_walk(a['gem'], b['gem'])
                st = math.hypot(a['gem'][0] - b['gem'][0], a['gem'][1] - b['gem'][1])
                rows.append((fw, st, b['walk0'], t))
R = np.array(rows); R = R[np.isfinite(R[:, 0])]
print('legs', len(R))
for name, col in (('fine walk', 0), ('straight', 1), ('coarse walk', 2)):
    A = np.c_[R[:, col], np.ones(len(R))]; c = np.linalg.lstsq(A, R[:, 3], rcond=None)[0]
    res = R[:, 3] - A @ c
    det = (R[:, 0] - R[:, 1]) >= 3
    print('%-12s t = %.4f x + %.3f  (%.1f u/s)  resid sd %.3f; detour legs mean resid %+.3f sd %.3f (n %d)' % (name, c[0], c[1], 1 / c[0], res.std(), res[det].mean(), res[det].std(), det.sum()))
np.save('C:/Users/doug/AppData/Local/Temp/claude/c--Users-doug-OneDrive-Documents-GitHub-PlatinumQuest-Dev/95f0bebd-02e6-445c-9156-b985ed352b96/scratchpad/walkfit.npy', R)
