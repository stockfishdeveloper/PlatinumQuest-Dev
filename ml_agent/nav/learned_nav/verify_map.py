"""Stage 3: verify a practice map's exported geometry in the game before it contributes any data (design 4.1).

    python -m nav.learned_nav.verify_map --port 8991 --map Sprawl_Hunt_phys      (then the game on that map)

Tests, with the engine's contact telemetry as the judge:
* floor: the marble placed just above a random floor point (along the exported normal) must, on its first supported
  decision, report a contact normal within NORMAL_DEG of the exported one, its centre R from the exported plane
  (within HEIGHT_TOL);
* ramp: on floor points steeper than 5 deg, released at rest, it must move within RAMP_DEG of straight downhill;
* lip: on exact edge segments (void or drop beyond), 0.3 u inside the marble is supported (at least 4 of 6
  decisions; it may roll, bevels slope), 0.35 u outside it falls (0.2 u or more within 6 decisions).
Declared pass rule (2026-09-27, before the first run): at least 95 % of floor tests, 95 % of lip tests and 90 % of
ramp tests agree. Results: datasets/learned_nav/geometry/<map>.verify.json.
"""
import argparse
import json
import math
import os
import time

import numpy as np

from nav.learned_nav.geometry import Geometry, GEOM_DIR, VOID, DROP
from nav.learned_nav.session import Session, RoundOver, R
from nav.protocol import RAW_POS, RAW_VEL, NOOP_ACTION

NORMAL_DEG = 3.0
HEIGHT_TOL = 0.04
RAMP_DEG = 10.0
N_FLOOR = 150
N_LIP = 80
PASS = {'floor': 0.95, 'lip': 0.95, 'ramp': 0.90}


def angle_deg(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    c = np.dot(a, b) / max(1e-9, np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def sample_floor(g, rng, n):
    """Top-level floor cells at least 0.8 u (along 8 directions) from any edge of their floor."""
    top = np.isfinite(g.heights[0])
    jj, ii = np.nonzero(top)
    out = []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        k = rng.integers(len(ii))
        x = g.xs[ii[k]] + rng.uniform(-0.09, 0.09); y = g.ys[jj[k]] + rng.uniform(-0.09, 0.09)
        z = g.floor_below(x, y, float(g.heights[0, jj[k], ii[k]]) + 0.01)
        nrm = g.normal_below(x, y, z + 0.01)
        if z is None or nrm is None or nrm[2] < math.cos(math.radians(45)):
            continue
        rays = g.edge_rays(x, y, z, np.radians(np.arange(0, 360, 45)), max_u=1.0)
        if np.isfinite(rays[:, 0]).any():
            continue
        out.append((x, y, z, nrm))
    return out


def sample_lips(g, rng, n):
    e = g.edges
    e = e[(e[:, 8] == VOID) | ((e[:, 8] == DROP) & (e[:, 9] > 0.6))]
    L = np.hypot(e[:, 3] - e[:, 0], e[:, 4] - e[:, 1])
    e = e[L > 0.8]; L = L[L > 0.8]
    out = []
    for _ in range(n * 20):
        if len(out) >= n or not len(e):
            break
        i = rng.choice(len(e), p=L / L.sum())
        u = rng.uniform(0.2, 0.8)
        p = e[i, 0:3] + u * (e[i, 3:6] - e[i, 0:3])
        o = e[i, 6:8]
        inside = (p[0] - o[0] * 0.3, p[1] - o[1] * 0.3)
        zin = g.floor_below(inside[0], inside[1], p[2] + 0.05)
        nin = g.normal_below(inside[0], inside[1], p[2] + 0.05)
        if zin is None or nin is None or abs(zin - p[2]) > 0.1 or nin[2] < 0.95:
            continue                                   # keep lips of flat-ish floor, where "inside" is clean
        out.append({'p': p, 'out': o, 'type': int(e[i, 8]), 'drop': float(e[i, 9])})
    return out


def run(port, map_name, seed=0, log=print):
    g = Geometry(map_name)
    s = Session(port, map_name, g, log=log)
    rng = np.random.default_rng(seed)
    res = {'floor': [], 'ramp': [], 'lip': []}
    t0 = time.time()
    for x, y, z, nrm in sample_floor(g, rng, N_FLOOR):
        centre = np.array([x, y, z]) + nrm * (R + 0.01)
        for attempt in range(3):
            try:
                o = s.place(centre)
                if o is None:
                    continue
                first = None; path = []
                for i in range(6):
                    msg, _ = s.step(NOOP_ACTION)
                    t = s.telemetry(); p = np.asarray(msg.obs[RAW_POS], dtype=np.float64)
                    path.append((p, np.asarray(msg.obs[RAW_VEL], dtype=np.float64)))
                    if first is None and t and t['support_steps'] > 0:
                        first = (i, np.array([t['nx'], t['ny'], t['nz']]), p)
                break
            except RoundOver:
                continue
        else:
            continue
        slope = math.degrees(math.acos(min(1.0, nrm[2])))
        if first is None:
            res['floor'].append({'x': x, 'y': y, 'ok': False, 'why': 'no support', 'slope': slope})
            continue
        i, tn, p = first
        dn = angle_deg(tn, nrm)
        # centre distance from the exported plane through the surface point (x, y, z)
        dist = float(np.dot(p - np.array([x, y, z]), nrm))
        ok = dn <= NORMAL_DEG and abs(dist - R) <= HEIGHT_TOL
        res['floor'].append({'x': x, 'y': y, 'slope': round(slope, 2), 'normal_err_deg': round(dn, 3),
                             'height_err': round(dist - R, 4), 'ok': bool(ok)})
        if slope > 5.0:
            v = path[-1][1]
            down = np.array([nrm[0], nrm[1]]); down /= max(1e-9, np.linalg.norm(down))     # gravity along the
                                                                                        # plane: nz * (nx, ny) horizontally
            vh = v[:2]
            if np.linalg.norm(vh) > 0.2:
                da = angle_deg(vh, down)
                res['ramp'].append({'x': x, 'y': y, 'slope': round(slope, 2), 'dir_err_deg': round(da, 2),
                                    'speed': round(float(np.linalg.norm(vh)), 2), 'ok': bool(da <= RAMP_DEG)})
            else:
                res['ramp'].append({'x': x, 'y': y, 'slope': round(slope, 2), 'ok': False, 'why': 'did not move'})
    for lip in sample_lips(g, rng, N_LIP):
        p, o = lip['p'], lip['out']
        rec = {'x': float(p[0]), 'y': float(p[1]), 'type': lip['type']}
        for side, off, z_expect_support in (('inside', -0.3, True), ('outside', 0.35, False)):
            c = np.array([p[0] + o[0] * off, p[1] + o[1] * off, p[2] + R + 0.01])
            got = None
            for attempt in range(3):
                try:
                    if s.place(c) is None:
                        continue
                    sup = 0; zs = []
                    for i in range(6):
                        msg, _ = s.step(NOOP_ACTION)
                        t = s.telemetry()
                        sup += int(t['support_steps'] > 0) if t else 0
                        zs.append(float(msg.obs[RAW_POS][2]))
                    got = (sup, zs[-1] - c[2])
                    break
                except RoundOver:
                    continue
            if got is None:
                rec[side] = None
                continue
            sup, dz = got
            rec[side] = {'support_decisions': sup, 'dz': round(dz, 3),
                         'ok': bool(sup >= 4 if z_expect_support else (dz < -0.2))}
        rec['ok'] = bool(rec.get('inside') and rec.get('outside') and rec['inside']['ok'] and rec['outside']['ok'])
        res['lip'].append(rec)
    summary = {}
    for k in ('floor', 'ramp', 'lip'):
        n = len(res[k]); ok = sum(r['ok'] for r in res[k])
        summary[k] = {'n': n, 'ok': ok, 'rate': round(ok / n, 3) if n else None, 'pass_at': PASS[k]}
    fl = [r for r in res['floor'] if 'normal_err_deg' in r]
    if fl:
        summary['floor']['normal_err_deg_p95'] = round(float(np.percentile([r['normal_err_deg'] for r in fl], 95)), 3)
        summary['floor']['height_err_p95'] = round(float(np.percentile([abs(r['height_err']) for r in fl], 95)), 4)
    verdict = all(summary[k]['rate'] is None or summary[k]['rate'] >= PASS[k] for k in PASS) and summary['floor']['n'] >= 50
    out = {'map': map_name, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'seconds': round(time.time() - t0),
           'pass': bool(verdict), 'summary': summary, 'tests': res,
           'rules': {'normal_deg': NORMAL_DEG, 'height_tol': HEIGHT_TOL, 'ramp_deg': RAMP_DEG, 'pass': PASS}}
    json.dump(out, open(os.path.join(GEOM_DIR, f'{map_name}.verify.json'), 'w'), indent=1,
              default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer) else str(o))
    log(f'{map_name}: {"PASS" if verdict else "FAIL"} {json.dumps(summary)}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--map', required=True)
    a = ap.parse_args()
    run(a.port, a.map)


if __name__ == '__main__':
    main()
