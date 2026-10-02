"""Stage 3 recordings: model apex vs real apex after a takeoff, by speed, air input and edge distance (model replay from
4 decisions before the takeoff with the recorded replies)."""
import sys, glob, math, json
import numpy as np
sys.path.insert(0, 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent')
from nav.learned_nav.geometry import Geometry
from nav.learned_nav import planner as PL, dynamics3 as D3
m = sys.argv[1] if len(sys.argv) > 1 else 'KingOfTheMarble_Hunt_phys'
g = Geometry(m); eidx = D3.EdgeIndex(g)
steps = PL.FastEnsemble(D3.load_ensemble('cuda'), 'cuda')
files = sorted(glob.glob(f'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent/datasets/learned_nav/stage3/{m}/shard_*.npz'))
cases = []
for f in files[:6]:
    z = np.load(f, allow_pickle=True); S = z['steps']; F = list(z['fields'])
    c = {k: F.index(k) for k in ('trial', 'i', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'fwd', 'back', 'left', 'right', 'jump_key', 'cam_yaw', 'int_thr')}
    tr = S[:, c['trial']].astype(int)
    for t in np.unique(tr):
        rows = S[tr == t]
        rows = rows[np.argsort(rows[:, c['i']])]
        vz = rows[:, c['vz']]
        k = np.nonzero(vz > 4.0)[0]
        if not len(k) or k[0] < 4 or k[0] + 14 > len(rows):
            continue
        k0 = k[0]                     # first state with vz > 4 (after the takeoff)
        if rows[k0 - 1, c['vz']] > 1.0:
            continue
        cases.append((rows, c, k0))
    if len(cases) > 1500:
        break
print(m, 'takeoffs', len(cases))
B = len(cases); K = 14
start = np.array([r[k0 - 4] for r, c, k0 in cases])
c = cases[0][1]
P = start[:, [c['px'], c['py'], c['pz']]].astype(float); V = start[:, [c['vx'], c['vy'], c['vz']]].astype(float); W = start[:, [c['wx'], c['wy'], c['wz']]].astype(float)
def rep(row):
    js = (row[c['fwd']], row[c['back']], row[c['left']], row[c['right']], row[c['jump_key']], row[c['cam_yaw']])
    return PL.reply_vector(js)
zm = np.zeros((B, K)); zr = np.zeros((B, K))
for s in range(K):
    U = np.zeros((B, 2)); J = np.zeros(B); Up = np.zeros((B, 2)); Jp = np.zeros(B)
    for b, (r, cc, k0) in enumerate(cases):
        ks = k0 - 4 + s
        U[b], J[b] = rep(r[ks - 1]); Up[b], Jp[b] = rep(r[ks - 2])
        zr[b, s] = r[ks + 1, cc['pz']]
    Fe, yaw = D3.features(g, eidx, P, V, W, U, J, Up, Jp)
    mu, pc = steps(Fe)
    P, V, W = D3.apply_step(P, V, W, mu, yaw)
    zm[:, s] = P[:, 2]
z0 = start[:, c['pz']]
apex_r = zr.max(1) - z0; apex_m = zm.max(1) - z0
sp = np.hypot(start[:, c['vx']], start[:, c['vy']])
# edge distance at the takeoff state
tk = np.array([r[k0 - 1] for r, cc, k0 in cases])
E = eidx.nearest(tk[:, c['px']].astype(float), tk[:, c['py']].astype(float), tk[:, c['pz']].astype(float) - PL.R, np.ones(B), np.zeros(B)).reshape(B, D3.N_EDGE, 7)
risky = (E[..., 3] + E[..., 4]) > 0
ed = np.where(risky, E[..., 0], 99).min(1)
airthr = np.array([r[k0 + 1, cc['int_thr']] for r, cc, k0 in cases])
d = apex_m - apex_r
print('apex real median %.2f; model - real: median %+.3f, p10 %+.3f, share below -0.15: %.2f' % (np.median(apex_r), np.median(d), np.percentile(d, 10), (d < -0.15).mean()))
for lo, hi in ((0, 6), (6, 9), (9, 12), (12, 20)):
    for elo, ehi in ((0, 1.5), (1.5, 4), (4, 99)):
        mm = (sp >= lo) & (sp < hi) & (ed >= elo) & (ed < ehi)
        if mm.sum() >= 10:
            print('  speed %2d-%2d edge %3.1f-%4.1f: n %4d, apex err median %+.3f, share < -0.15 %.2f' % (lo, hi, elo, ehi, mm.sum(), np.median(d[mm]), (d[mm] < -0.15).mean()))
for a in (0.0, 0.5, 1.0):
    mm = np.isclose(airthr, a)
    if mm.sum() >= 10:
        print('  air throttle %.1f: n %4d, apex err median %+.3f, share < -0.15 %.2f' % (a, mm.sum(), np.median(d[mm]), (d[mm] < -0.15).mean()))

# teacher-forced vz around the takeoff for the worst cases: transitions k0-3 .. k0+1 (state k0 has vz > 4)
worst = np.argsort(d)[:6]
for b in worst:
    r, cc, k0 = cases[b]
    out = []
    for ks in range(k0 - 3, k0 + 2):
        row = r[ks]
        Pp = row[[cc['px'], cc['py'], cc['pz']]][None].astype(float); Vv = row[[cc['vx'], cc['vy'], cc['vz']]][None].astype(float); Ww = row[[cc['wx'], cc['wy'], cc['wz']]][None].astype(float)
        U, J = rep(r[ks - 1]); Upp, Jpp = rep(r[ks - 2])
        Fe, yaw = D3.features(g, eidx, Pp, Vv, Ww, U[None], np.array([J]), Upp[None], np.array([Jpp]))
        mu, pc = steps(Fe)
        P1, V1, W1 = D3.apply_step(Pp, Vv, Ww, mu, yaw)
        out.append('k%+d J%d Jp%d vz %.2f->real %.2f pred %.2f sup %.2f coll %.2f' % (ks - k0, J, Jpp, row[cc['vz']], r[ks + 1, cc['vz']], V1[0, 2], pc[0, 0], pc[0, 1]))
    print('case apex err %+.2f speed %.1f edge %.2f:' % (d[b], sp[b], ed[b]))
    for o in out:
        print('    ' + o)
