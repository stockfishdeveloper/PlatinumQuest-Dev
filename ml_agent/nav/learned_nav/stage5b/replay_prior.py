import json, math, sys
import numpy as np
sys.path.insert(0, 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent')
from nav.learned_nav.geometry import Geometry
from nav.learned_nav import planner as PL, dynamics3 as D3
g = Geometry('kotmjump_p0')
pl = PL.Planner(g, (0, 0, 0), seeds=None, use_flight=False)
files = sys.argv[1:]
def replay(r, prior):
    S = np.array(r['states']); Rp = r['replies']; push = r['push']
    n = min(len(Rp), 22)
    acting = [push] + Rp[:-1]; prev = [push, push] + Rp[:-2]
    P, V, W = S[0:1, 0:3].copy(), S[0:1, 3:6].copy(), S[0:1, 6:9].copy()
    zs, err = [], []; after = np.zeros(1, bool)
    for k in range(n):
        U, J = PL.reply_vector(acting[k]); Up, Jp = PL.reply_vector(prev[k])
        Ja = np.array([J])
        F, yaw = D3.features(g, pl.eidx, P, V, W, U[None], Ja, Up[None], np.array([Jp]))
        mu, pc = pl.steps(F)
        fires = PL.jump_fires(g, P, V, Ja) if prior else np.zeros(1, bool)
        zb = P[:, 2].copy()
        P, V, W = D3.apply_step(P, V, W, mu, yaw)
        if prior:
            if after.any(): P[after, 2] = zb[after] + PL.JUMP_DZ[1]; V[after, 2] = PL.JUMP_VZ[1]
            if fires.any(): P[fires, 2] = zb[fires] + PL.JUMP_DZ[0]; V[fires, 2] = PL.JUMP_VZ[0]
            after = fires
        zs.append(P[0, 2]); err.append(np.linalg.norm(P[0] - S[k + 1, 0:3]))
    err += [np.nan] * (22 - len(err))
    return max(zs) - S[0, 2], err
E0, E1 = [], []
for f in files:
    for l in open(f):
        r = json.loads(l)
        a0, e0 = replay(r, False); a1, e1 = replay(r, True)
        E0.append(e0); E1.append(e1)
        print('v %4.1f j %d air %-6s | game fell %d | no prior: apex %.2f err@12 %.2f @16 %.2f | prior: apex %.2f err@12 %.2f @16 %.2f' % (
            r['speed'], r['jump'], r['air'], r['real']['fell'], a0, e0[11], e0[15], a1, e1[11], e1[15]))
E0 = np.array(E0); E1 = np.array(E1)
print('median path error by decision (no prior / prior):')
for k in (4, 8, 12, 16, 20):
    print('   %2d: %.2f / %.2f' % (k, np.nanmedian(E0[:, k - 1]), np.nanmedian(E1[:, k - 1])))
