"""Watch "where will it land" at 1x on a map the models never trained on (viewing only; nothing is recorded).

    python -m nav.learned_nav.watch3 --port 9061 --map GemsAhoy_Hunt_phys
    then: marbleblast_mbx.exe -autotrain GemsAhoy_Hunt_phys -aiport 9061       (or nav\\learned_nav\\watch3.ps1)

Each round: a rolling start near an exact edge of the map, heading toward it. The step-model ensemble rolls every
candidate forward on the exact inputs the game will get (run-up, jump at decision j with an air direction, that input
until the marble is back on a floor, then no input); the flight head judges whether the landing is safe. The chosen
jump is one that crosses the edge and is predicted to land safely. A black gem marks the predicted landing point for a
second before the marble is placed and the jump is flown at real time; the console prints how far from the mark it
really landed.
"""
import argparse
import math
import os
import sys
import time

os.environ.setdefault('NAV_RENDER_EVERY', '1')

import numpy as np                                                               # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav import dynamics3 as D3                                      # noqa: E402
from nav.learned_nav import flight3 as F3                                        # noqa: E402
from nav.learned_nav import eval3 as E3                                          # noqa: E402
from nav.learned_nav.record3 import Recorder3, StartSampler                      # noqa: E402
from nav.learned_nav.geometry import VOID, DROP                                  # noqa: E402
from nav.protocol import NOOP_ACTION, RAW_POS                                    # noqa: E402

SQ2 = math.sqrt(2.0)
R = 0.19
JUMP_TICKS = range(5)
AIRS = [None] + [k * math.pi / 4 for k in range(8)]
H = 24


def rollout(steps, g, eidx, start, cands, dev):
    """Predicted path, landing row and safety for each candidate (j, air) from this start."""
    n = len(cands)
    P = np.tile(np.asarray(start['pos'], float), (n, 1)); V = np.tile(np.asarray(start['vel'], float), (n, 1))
    W = np.tile(np.asarray(start['spin'], float), (n, 1))
    h = start['heading']
    run = SQ2 * np.array([math.cos(h), math.sin(h)])
    airv = np.array([[0.0, 0.0] if a is None else SQ2 * np.array([math.cos(h + a), math.sin(h + a)]) for _, a in cands])
    jt = np.array([j for j, _ in cands])
    hold = np.where((jt == 0)[:, None], airv, run)                      # record3: the held input is reply 0 without its jump key
    z0 = P[:, 2].copy()
    airborne = np.zeros(n, bool); landed = np.full(n, -1); over_void = np.zeros(n, bool)
    path = []; Up = hold.copy(); Jp = np.zeros(n)
    sup = np.ones(n)
    for k in range(H):
        if k == 0:
            U = hold; J = np.zeros(n)
        else:
            r = k - 1                                                    # the reply sent at loop r
            U = np.where((r < jt)[:, None], run, airv)
            U = np.where(((landed >= 0) & (landed <= r - 1))[:, None], 0.0, U)
            J = (r == jt).astype(float)
        F, yaw = D3.features(g, eidx, P, V, W, U, J, Up, Jp)
        pr = E3.predict(steps, F, dev)
        P, V, W = D3.apply_step(P, V, W, pr['mu'], yaw)
        sup = pr['p'][:, 0]; last = pr['p'][:, 2]
        lv, _ = D3.level_below(g, P[:, 0], P[:, 1], P[:, 2] + 0.3)
        gap = np.where(np.isfinite(lv), P[:, 2] - R - lv, 9.0)
        over_void |= gap > 1.0
        newly_air = (~airborne) & (sup < 0.5) & (gap > 0.5)
        airborne |= newly_air
        land_now = airborne & (landed < 0) & (sup >= 0.5) & (last >= 0.5)
        landed = np.where(land_now, k, landed)
        Up, Jp = U, J
        path.append(P.copy())
    path = np.stack(path, 1)
    lv, _ = D3.level_below(g, P[:, 0], P[:, 1], P[:, 2] + 0.3)
    safe = (landed >= 0) & (sup >= 0.5) & np.isfinite(lv) & (P[:, 2] > z0 - 3.0)
    land_pt = np.array([path[i, landed[i]] if landed[i] >= 0 else path[i, -1] for i in range(n)])
    return path, landed, land_pt, safe, over_void, (hold, run, airv, jt)


def flight_safe(fmodel, g, start, cands, seqinfo, landed, dev):
    import torch
    hold, run, airv, jt = seqinfo
    p0 = np.asarray(start['pos']); v0 = np.asarray(start['vel']); w0 = np.asarray(start['spin'])
    yaw = math.atan2(v0[1], v0[0]) if math.hypot(v0[0], v0[1]) > 0.5 else start['heading']
    feats = []
    base = None
    c, s = math.cos(yaw), math.sin(yaw)
    for i in range(len(cands)):
        U = [hold[i]]; J = [0.0]
        for r in range(F3.H):
            if landed[i] >= 0 and r >= landed[i] + 1:
                U.append(np.zeros(2))
            else:
                U.append(run if r < jt[i] else airv[i])
            J.append(1.0 if r == jt[i] else 0.0)
        U = np.asarray(U); J = np.asarray(J)
        if base is None:
            base = F3.sample_features(g, p0, v0, w0, None, yaw, U, J)
        f = base.copy()
        f[F3.N_STATE:F3.N_STATE + F3.N_SEQ] = np.c_[c * U[:, 0] + s * U[:, 1], -s * U[:, 0] + c * U[:, 1], J][:F3.H].ravel()
        feats.append(f)
    with torch.no_grad():
        o = fmodel(torch.as_tensor(np.asarray(feats, np.float32), device=dev))
    return torch.sigmoid(o['safe']).cpu().numpy()


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--map', default='GemsAhoy_Hunt_phys')
    ap.add_argument('--n', type=int, default=200)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    steps = D3.load_ensemble(dev); fmodel = F3.load(dev, 'flight3.pth')
    rec = Recorder3(a.port, a.map, log=lambda m: print(m, flush=True))
    s = rec.s; g = rec.g; eidx = D3.EdgeIndex(g)
    s.env.set_speed(1); s.contact_on()
    clock = {'t': time.perf_counter()}
    raw_step = s.step

    def paced(js):
        clock['t'] = max(clock['t'] + 0.064, time.perf_counter())
        dt = clock['t'] - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        return raw_step(js)
    s.step = paced
    rng = np.random.default_rng(int(time.time()) % 100000)
    sampler = StartSampler(g, rng)
    cands = [(j, air) for j in JUMP_TICKS for air in AIRS]
    errs = []; on_pad = 0; done = 0
    while done < a.n:
        for _ in range(60):                                              # a start rolling toward an edge
            st = sampler.sample()
            if st['kind'] != 'floor' or not st['near_edge'] or st['speed'] < 6:
                continue
            p = st['pos']
            ray = g.edge_rays(p[0], p[1], p[2] - R - 0.01, [st['heading']], max_u=6.0)[0]
            if np.isfinite(ray[0]) and int(ray[1]) in (VOID, DROP):
                break
        else:
            continue
        path, landed, land_pt, safe, over_void, info = rollout(steps, g, eidx, st, cands, dev)
        psafe = flight_safe(fmodel, g, st, cands, info, landed, dev)
        dist = np.hypot(land_pt[:, 0] - st['pos'][0], land_pt[:, 1] - st['pos'][1])
        ok = safe & over_void & (landed >= 0) & (psafe >= 0.6) & (dist >= 3.0)
        if not ok.any():
            continue
        k = int(np.argmax(np.where(ok, psafe + 0.02 * dist, -1)))
        j, air = cands[k]
        mark = land_pt[k]
        s.control(f'MARK {mark[0]:.3f} {mark[1]:.3f} {mark[2]:.3f}')
        for _ in range(16):
            s.step(NOOP_ACTION)                                          # a second to see the mark
        prog = {'family': 'jump', 'thr': 1.0, 'len': 20, 'dir': 0.0, 'j': j, 'air': air, 'cont': 'none'}
        r = rec.trial(st, prog)
        done += 1
        if r is None:
            print(f'{done:3d}: the game did not take the start (skipped)', flush=True)
            continue
        meta, rows = r
        if meta['landed'] is not None:
            actual = rows[meta['landed'], 2:5].astype(float)
            e = float(np.linalg.norm(actual[:2] - mark[:2]))
            errs.append(e); on_pad += int(e <= 0.5)
            res = f'landed {e:.2f} u from the mark' + ('' if not meta['oob'] else ', then fell')
        else:
            res = 'fell before landing' if meta['oob'] else 'no landing seen'
        air_s = 'no air input' if air is None else f'air {int(round(math.degrees(air))) if math.degrees(air) <= 180 else int(round(math.degrees(air))) - 360:+d} deg'
        print(f'{done:3d}: {st["speed"]:4.1f} u/s, jump at decision {j}, {air_s}; predicted {dist[k]:.1f} u away, safe {psafe[k]:.2f} -> {res}'
              f'   [within 0.5 u: {on_pad}/{len(errs)}, median miss {np.median(errs) if errs else float("nan"):.2f} u]', flush=True)
        for _ in range(12):
            s.step(NOOP_ACTION)
        s.control('MARK off')


if __name__ == '__main__':
    main()
