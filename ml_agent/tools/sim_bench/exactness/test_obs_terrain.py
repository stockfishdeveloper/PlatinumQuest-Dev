"""Bit-exactness of nav/native/obs_terrain (obs_native.dll) against the NumPy terrain queries (TerrainGrid.crop,
floor_height, edge_rays, gap_along) on random states of the given maps: near real floor samples (levels, edges, holes),
exactly on a level or midway between two (tie cases), and anywhere incl. off the grid; gem deltas and headings as
Python floats and NumPy float64 (the two promotion paths). Prints per query the number of differing answers (must be
0) and the time per call. usage (repo root, PYTHONPATH=ml_agent):
python ml_agent/tools/sim_bench/exactness/test_obs_terrain.py <n states per map> <map> [<map> ...]"""
import sys, math, time
import numpy as np
from nav.terrain import TerrainGrid, TerrainMap
from nav.native.obs_terrain import NativeTerrainQueries


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
                and np.array_equal(a.view(np.uint32) if a.dtype == np.float32 else a, b.view(np.uint32) if b.dtype == np.float32 else b))
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(same(u, v) for u, v in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) == type(b) and a == b and (a != 0 or math.copysign(1, a) == math.copysign(1, b))


n = int(sys.argv[1])
for m in sys.argv[2:]:
    t = TerrainGrid(TerrainMap.resolve(m)); q = NativeTerrainQueries(t)
    rng = np.random.default_rng(7)
    fin = np.argwhere(np.isfinite(t.heights))
    bad = dict(crop=0, floor=0, edge=0, gap=0); tp = dict.fromkeys(bad, 0.); tn = dict.fromkeys(bad, 0.)

    def check(key, f_np, f_nat, *args):
        t0 = time.perf_counter(); A = f_np(*args); t1 = time.perf_counter(); B = f_nat(*args); t2 = time.perf_counter()
        tp[key] += t1 - t0; tn[key] += t2 - t1
        if not same(A, B):
            bad[key] += 1
            if bad[key] <= 2:
                print('  DIFF', key, args[:3])

    for trial in range(n):
        r = rng.random()
        if r < 0.7:
            k, j, i = fin[rng.integers(len(fin))]
            x = float(t.x0 + i * t.res + rng.uniform(-1.5, 1.5)); y = float(t.y0 + j * t.res + rng.uniform(-1.5, 1.5))
            z = float(np.float32(t.heights[k, j, i] + rng.choice([0.19, 0.0, rng.uniform(-3, 3), 0.5, 1.0])))
        elif r < 0.85:
            k, j, i = fin[rng.integers(len(fin))]
            x = float(t.x0 + i * t.res); y = float(t.y0 + j * t.res)
            col = t.heights[:, j, i]; f = col[np.isfinite(col)]
            z = float(f[0]) if len(f) < 2 or rng.random() < .5 else float(np.float32((float(f[0]) + float(f[1])) / 2))
        else:
            x = float(rng.uniform(t.x0 - 30, t.x0 + t.W * t.res + 30)); y = float(rng.uniform(t.y0 - 30, t.y0 + t.H * t.res + 30))
            z = float(np.float32(rng.uniform(-20, 40)))
        check('crop', t.crop, q.crop, x, y, z)
        check('floor', t.floor_height, q.floor_height, x, y, z)
        gx = x + rng.uniform(-25, 25); gy = y + rng.uniform(-25, 25)
        if rng.random() < .5:
            gx, gy = np.float64(gx), np.float64(gy)
        if rng.random() < .05:
            gx, gy = x, y + 1e-4
        check('edge', t.edge_rays, q.edge_rays, x, y, z, [(gx - x, gy - y, 0.0, True)])
        ang = rng.uniform(-math.pi, math.pi); ux, uy = math.cos(ang), math.sin(ang)
        if rng.random() < .5:
            d = math.hypot(np.float64(ux * 7.3), np.float64(uy * 7.3)); ux, uy = np.float64(ux * 7.3) / d, np.float64(uy * 7.3) / d
        check('gap', t.gap_along, q.gap_along, x, y, z, ux, uy)
    print('%s (K %d): %d states; differing answers %s; NumPy / native us a call %s' % (
        m, t.K, n, bad, {k: '%.0f / %.1f' % (1e6 * tp[k] / n, 1e6 * tn[k] / n) for k in bad}), flush=True)
