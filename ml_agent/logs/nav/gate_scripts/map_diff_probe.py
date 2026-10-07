"""Teleport probe of the KOTM terrain map change (2026-10-05, operator: "teleport the marble to the precise locations on
the map that changed ... and see if the marble falls or not").

Every point of the corrected map's 0.25 u grid where the old map (0.5 u grid, centre test) and the corrected map disagree:
  hidden_floor  old said no floor, corrected says floor  -> the marble placed at rest should STAY
  phantom_floor old said floor, corrected says no floor  -> the marble should FALL
plus controls: deep floor (1 u or more inside the real floor: must stay) and deep holes (1 u or more from any floor, inside
the arena outline: must fall). Each point also gets its exact signed distance to the real floor edge, from the mission's
own floor polygons (+ inside the floor, - outside; 0 = exactly on an edge line).
Trial: teleport at rest to (x, y, floor height + R + 0.05), no input, watch 32 decisions (2 s). Fell = 0.3 u or more
below the start within the first 8 decisions (0.5 s; an unsupported marble drops 2.5 u in that time, one rolling down a
ramp a few hundredths), or an out of bounds. Later drops are recorded separately (fell_by_2s).
Usage: python logs/nav/gate_scripts/map_diff_probe.py build
       python logs/nav/gate_scripts/map_diff_probe.py run <port> <part> <parts>   (then a game on KingOfTheMarble_Hunt)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from terrain_obs import TerrainMap

OLD = 'terrain_maps/terrain_KingOfTheMarble_Hunt_grid050_20261005.npz'
NEW = 'terrain_maps/terrain_KingOfTheMarble_Hunt_edgesafe025_diag.npz'
TRIALS = 'logs/nav/goal175/map_diff_trials_20261005.json'
R_M = 0.19
LIFT = 0.05
EARLY = 8          # decisions (0.5 s)
HOLD = 32          # decisions (2 s)
DROP = 0.3


def build():
    import shapely
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    import generate_terrain_map as G
    import hxDif
    old, new = TerrainMap(OLD), TerrainMap(NEW)
    X, Y = np.meshgrid(new.xs, new.ys)
    pts = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float64)
    oh = old.heights_at(pts[:, 0], pts[:, 1])
    nh = new.heights.reshape(new.K, -1)
    of, nf = np.isfinite(oh).any(axis=0), np.isfinite(nh).any(axis=0)
    mis = G.resolve_mission('KingOfTheMarble_Hunt')
    z_min, z_max = G.parse_in_bounds_z(mis)
    surfaces = []
    for dif_path, pos, rot, scl in G.parse_interiors(mis):
        surfaces += G.transform_surfaces(G.extract_surfaces(hxDif.Dif.Load(dif_path)), pos, rot, scl)
    polys = []
    for v, n in surfaces:                        # the same floor filter as generate_terrain_map.build_height_stack
        if n[2] <= G.FLOOR_NORMAL_Z or len(v) < 3:
            continue
        z = float(np.mean([p[2] for p in v]))
        if (z_min is not None and z < z_min - 0.5) or (z_max is not None and z > z_max + 0.5):
            continue
        p = Polygon([(p[0], p[1]) for p in v])
        p = p if p.is_valid else p.buffer(0)
        if p.area > 1e-9:
            polys.append(p)
    floor = unary_union(polys)
    P = shapely.points(pts)
    d = shapely.distance(floor.boundary, P)
    sd = np.where(shapely.contains(floor, P), d, -d)
    hull = shapely.contains(floor.convex_hull, P)
    cats = {'hidden_floor': ~of & nf, 'phantom_floor': of & ~nf,
            'control_floor': of & nf & (sd >= 1.0), 'control_hole': ~of & ~nf & (sd <= -1.0) & hull}
    rng = np.random.default_rng(20261005)
    trials = []
    for c, m in cats.items():
        idx = np.where(m)[0]
        if c.startswith('control'):
            idx = np.sort(rng.choice(idx, size=min(60, len(idx)), replace=False))
        for i in idx:
            if c == 'phantom_floor':
                h = float(oh[0, i])
            elif c == 'control_hole':
                h = 20.65
            else:
                h = float(nh[0, i])
            trials.append({'cat': c, 'x': round(float(pts[i, 0]), 4), 'y': round(float(pts[i, 1]), 4),
                           'floor_z': round(h, 3), 'edge_sd': round(float(sd[i]), 4)})
    order = rng.permutation(len(trials))            # mix categories over the parts and over time
    trials = [trials[k] for k in order]
    with open(TRIALS, 'w') as f:
        json.dump(trials, f)
    for c in cats:
        print(c, sum(t['cat'] == c for t in trials))
    print('saved', TRIALS, len(trials))


def run(port, part, parts):
    from nav.learned_nav.session import Session, RoundOver
    from nav.protocol import RAW_POS, RAW_VEL, NOOP_ACTION
    new = TerrainMap(NEW)
    with open(TRIALS) as f:
        trials = json.load(f)[part::parts]
    out_path = f'logs/nav/goal175/map_diff_probe_20261005_part{part}.jsonl'
    done = set()
    if os.path.exists(out_path):                     # resume after a crash
        with open(out_path) as f:
            done = {(r['x'], r['y'], r['cat']) for r in map(json.loads, f)}

    class Geo:                                       # what Session needs: grid bounds and heights
        xs, ys, heights = new.xs, new.ys, new.heights

    s = Session(port, 'KingOfTheMarble_Hunt', Geo(), log=lambda m: print(m, flush=True))
    f = open(out_path, 'a')
    for n, t in enumerate(trials):
        if (t['x'], t['y'], t['cat']) in done:
            continue
        z0 = t['floor_z'] + R_M + LIFT
        rec = None
        for attempt in range(4):
            try:
                if s.place((t['x'], t['y'], z0)) is None:
                    continue
                zs, oob, fell_at = [], False, None
                for k in range(HOLD):
                    msg, info = s.step(NOOP_ACTION)
                    z = float(s.obs()[RAW_POS][2]); zs.append(round(z0 - z, 3))
                    if info['fell']:
                        oob = True
                    if fell_at is None and (oob or z0 - z >= DROP):
                        fell_at = k + 1
                    if fell_at is not None and (oob or z0 - z >= 1.0):
                        break
                p = s.obs()[RAW_POS]; v = s.obs()[RAW_VEL]
                rec = dict(t, fell=fell_at is not None and fell_at <= EARLY, fell_by_2s=fell_at is not None,
                           fell_at=fell_at, oob=oob, drop=zs, moved=round(float(np.hypot(p[0] - t['x'], p[1] - t['y'])), 3),
                           speed=round(float(np.linalg.norm(v)), 3), attempt=attempt)
                break
            except RoundOver:
                continue
        if rec is None:
            rec = dict(t, fell=None, error='not placed')
        f.write(json.dumps(rec) + '\n'); f.flush()
        print(f"{n + 1}/{len(trials)} {t['cat']:14s} ({t['x']:7.2f},{t['y']:7.2f}) edge {t['edge_sd']:+.2f} -> "
              f"{'FELL' if rec.get('fell') else ('fell late' if rec.get('fell_by_2s') else 'stayed')}", flush=True)
    f.close()
    print('DONE', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build()
    else:
        run(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
