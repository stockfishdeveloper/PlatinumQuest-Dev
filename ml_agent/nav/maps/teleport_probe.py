"""Teleport proof of a generated block-cluster map (the verify-terrain rule, 2026-10-06).

Every Cube.dif box in the mission gives probe points: its top centre, two inset corners (0.3 u in along the box's
OWN axes, which checks the rotation direction on rectangles), and two points 0.4 u outside two edges at the top
height. Expected outcome from the terrain map: stay if the map's top floor at the point is within 0.15 u of the box
top, else fall. Trial: teleport at rest to (x, y, expected floor + R + 0.05), no input, watch up to 2 s; fell = a
drop of 0.3 u or more within 0.5 s, or out of bounds.

    python -m nav.maps.teleport_probe <map> <port>       (start this first; then one game: -autotrain <map> -aiport <port>)
Writes logs/nav/blockclusters/teleport_<map>.jsonl and prints a summary.
"""
import os, re, sys, json, math
import numpy as np
ML = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ML)
from terrain_obs import TerrainMap

CUSTOM = os.path.join(os.path.dirname(ML), 'Marble Blast Platinum', 'platinum', 'data', 'multiplayer', 'hunt', 'custom')
CUBE_OFF = (-0.06, 0.06, 0.12); CUBE_TOP = 2.12; CUBE_SIZE = 4.0
R_M = 0.19; LIFT = 0.05; EARLY = 8; HOLD = 32; DROP = 0.3


def boxes_of(mapname):
    t = open(os.path.join(CUSTOM, mapname + '.mcs')).read()
    out = []
    for m in re.finditer(r'new InteriorInstance\(\) \{ position = "([^"]+)"; rotation = "0 0 1 ([-\d.]+)"; scale = "([^"]+)"; '
                         r'interiorFile = "~/data/interiors_mbp/Cube.dif"', t):
        px, py, pz = (float(v) for v in m.group(1).split()); ang = -float(m.group(2))   # the file holds -ang (engine convention)
        sx, sy, sz = (float(v) for v in m.group(3).split())
        a = math.radians(ang); ca, sa = math.cos(a), math.sin(a)
        ox, oy, oz = CUBE_OFF[0] * sx, CUBE_OFF[1] * sy, CUBE_OFF[2] * sz
        cx, cy = px + ca * ox - sa * oy, py + sa * ox + ca * oy
        top = pz + CUBE_TOP * sz
        out.append((cx, cy, sx * CUBE_SIZE, sy * CUBE_SIZE, top, ang))
    return out


def trials_for(mapname, tm):
    pts = []
    for bi, (cx, cy, w, d, top, ang) in enumerate(boxes_of(mapname)):
        a = math.radians(ang); ca, sa = math.cos(a), math.sin(a)
        def at(u, v):   # box-local (u along width, v along depth) -> world
            return cx + ca * u - sa * v, cy + sa * u + ca * v
        cands = [('centre', at(0, 0)), ('corner', at(w / 2 - 0.3, d / 2 - 0.3)), ('corner', at(-w / 2 + 0.3, -d / 2 + 0.3)),
                 ('off_edge', at(w / 2 + 0.4, 0)), ('off_edge', at(0, -d / 2 - 0.4))]
        for cat, (x, y) in cands:
            h = tm.heights_at(np.array([x]), np.array([y]))[:, 0]
            h = h[np.isfinite(h)]
            if h.size == 0: continue
            floor = float(h.max())
            expect_stay = abs(floor - top) < 0.15
            pts.append(dict(box=bi, cat=cat, x=round(x, 3), y=round(y, 3), box_top=round(top, 3), floor_z=round(floor, 3),
                            expect='stay' if expect_stay else 'fall', place_z=round(top + R_M + LIFT, 3)))
    return pts


def trials_for_terrain(mapname, tm, n_interior=40, n_void=30, n_step=30, n_gem=25, n_lower=10, seed=7):
    """Trials for a REAL map (2026-10-09, the 10-map pool): sampled from the terrain map itself, no box list.
    interior: a floor cell whose 8 neighbours are floor at the same height -> stay.  void: 0.4 u past a floor edge
    over nothing -> fall.  step: 0.4 u past an edge over a cell >= 0.3 u lower, placed at the UPPER height -> fall
    (drop >= 0.3).  gem: every gem spawn's cell -> stay.  lower: a cell's second level (under an overhang) -> stay."""
    rng = np.random.default_rng(seed)
    H = tm.heights; top = H[0]; res = tm.res
    fl = np.isfinite(top); pts = []
    def add(cat, x, y, z_floor, expect):
        pts.append(dict(box=-1, cat=cat, x=round(float(x), 3), y=round(float(y), 3), box_top=round(float(z_floor), 3),
                        floor_z=round(float(z_floor), 3), expect=expect, place_z=round(float(z_floor) + R_M + LIFT, 3)))
    J, I = np.where(fl)
    pad = np.pad(top, 1, constant_values=np.nan)
    nb = np.stack([pad[1 + dj:pad.shape[0] - 1 + dj, 1 + di:pad.shape[1] - 1 + di] for dj in (-1, 0, 1) for di in (-1, 0, 1)])
    flat = fl & np.all(np.abs(nb - top) < 0.05, axis=0)
    jj, ii = np.where(flat)
    for k in rng.choice(len(jj), min(n_interior, len(jj)), replace=False):
        add('interior', tm.xs[ii[k]], tm.ys[jj[k]], top[jj[k], ii[k]], 'stay')
    # edges: floor cell with a 4-neighbour that is void (-> void trial) or >= 0.3 lower (-> step trial)
    void_c, step_c = [], []
    for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nbr = np.roll(np.roll(pad, -dj, 0), -di, 1)[1:-1, 1:-1]
        v = fl & ~np.isfinite(nbr); st = fl & np.isfinite(nbr) & (top - nbr >= 0.3)
        for arr, lst in ((v, void_c), (st, step_c)):
            a, b = np.where(arr)
            for k in range(len(a)):
                lst.append((a[k], b[k], dj, di))
    for lst, cat, n in ((void_c, 'void', n_void), (step_c, 'step', n_step)):
        if not lst: continue
        for k in rng.choice(len(lst), min(n, len(lst)), replace=False):
            j, i, dj, di = lst[k]
            x = tm.xs[i] + di * (res / 2 + 0.4); y = tm.ys[j] + dj * (res / 2 + 0.4)
            add(cat, x, y, top[j, i], 'fall')
    try:
        from nav.terrain import gem_spawn_points
        gems = gem_spawn_points(mapname)
    except Exception:
        gems = []
    if gems:
        for k in rng.choice(len(gems), min(n_gem, len(gems)), replace=False):
            gx, gy, gz = gems[k]
            h = tm.heights_at(np.array([gx]), np.array([gy]))[:, 0]; h = h[np.isfinite(h)]
            if h.size == 0: continue
            z = float(h[np.argmin(np.abs(h - gz))])
            add('gem', gx, gy, z, 'stay')
    if H.shape[0] > 1:
        lj, li = np.where(np.isfinite(H[1]))
        for k in rng.choice(len(lj), min(n_lower, len(lj)), replace=False):
            add('lower', tm.xs[li[k]], tm.ys[lj[k]], H[1][lj[k], li[k]], 'stay')
    return pts


def run(mapname, port):
    from nav.learned_nav.session import Session, RoundOver
    from nav.protocol import RAW_POS, RAW_VEL, NOOP_ACTION
    tm = TerrainMap(os.path.join(ML, 'terrain_maps', f'terrain_{mapname}.npz'))
    counts = os.environ.get('NAV_PROBE_COUNTS')          # "interior,void,step,gem,lower": a dense sweep (2026-10-09, Sprawl proof)
    if counts:
        ni, nv, ns, ng, nl = (int(v) for v in counts.split(','))
        trials = trials_for_terrain(mapname, tm, n_interior=ni, n_void=nv, n_step=ns, n_gem=ng, n_lower=nl, seed=11)
    else:
        trials = trials_for(mapname, tm) if os.path.exists(os.path.join(CUSTOM, mapname + '.mcs')) and boxes_of(mapname) else trials_for_terrain(mapname, tm)
    os.makedirs(os.path.join(ML, 'logs', 'nav', 'blockclusters'), exist_ok=True)
    out_path = os.path.join(ML, 'logs', 'nav', 'blockclusters', f'teleport_{mapname}{"_dense" if counts else ""}.jsonl')
    print(f'{len(trials)} trials on {mapname}; waiting for the game on port {port}', flush=True)

    class Geo:
        xs, ys, heights = tm.xs, tm.ys, tm.heights

    s = Session(port, mapname, Geo(), log=lambda m: print(m, flush=True))
    f = open(out_path, 'w')
    agree = disagree = 0
    for n, t in enumerate(trials):
        z0 = t['place_z']; rec = None
        for attempt in range(4):
            try:
                if s.place((t['x'], t['y'], z0)) is None:
                    continue
                zs, oob, fell_at = [], False, None
                for k in range(HOLD):
                    msg, info = s.step(NOOP_ACTION)
                    z = float(s.obs()[RAW_POS][2]); zs.append(round(z0 - z, 3))
                    if info['fell']: oob = True
                    if fell_at is None and (oob or z0 - z >= DROP): fell_at = k + 1
                    if fell_at is not None and (oob or z0 - z >= 1.0): break
                fell = fell_at is not None and fell_at <= EARLY
                rec = dict(t, fell=fell, fell_at=fell_at, oob=oob, drop=zs[-1] if zs else None, attempt=attempt)
                break
            except RoundOver:
                continue
        if rec is None:
            rec = dict(t, fell=None, error='not placed')
        else:
            ok = (rec['fell'] is False and t['expect'] == 'stay') or (rec['fell'] is True and t['expect'] == 'fall')
            rec['agree'] = ok; agree += ok; disagree += (not ok)
        f.write(json.dumps(rec) + '\n'); f.flush()
        print(f"{n + 1}/{len(trials)} box {t['box']:3d} {t['cat']:9s} ({t['x']:7.2f},{t['y']:7.2f}) expect {t['expect']:4s} -> "
              f"{'FELL' if rec.get('fell') else 'stayed' if rec.get('fell') is False else 'n/a'} {'' if rec.get('agree', True) else '  <-- DISAGREE'}", flush=True)
    f.close()
    print(f'DONE agree {agree} disagree {disagree} of {len(trials)}', flush=True)


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]))
