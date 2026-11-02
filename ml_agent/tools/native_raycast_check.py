"""Compiled CollisionMap.raycast (nav/native marble_sim.cpp ms_raycast, ABI 15) vs the NumPy original, on random rays.

usage (repo root, PYTHONPATH=ml_agent): python ml_agent/tools/native_raycast_check.py [--n 20000] MAP [MAP ...]

Rays: (1) the support probe of search.terminal_support (0.05 above a point down to radius+0.10 below it) at random
positions over the map, (2) random segments 0.2 to 12 u long in both cull modes, (3) exact ties: vertical rays through
triangle vertices and edge midpoints, where several triangles share the hit (the first-minimal np.argmin rule).
"""
import argparse
import time
from pathlib import Path

import numpy as np

from nav.collision.experimental_sweep import CachedCompiledSweepMap
from nav.collision.mesh import CollisionMap
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.static_cache import load_queries
from nav.native.marble_sim import NativeSim

CACHE = Path(r'C:\Users\doug\AppData\Local\Temp\pq_checks\fast_nav_static_cache_20261013')
DLL = Path(r'C:\Users\doug\AppData\Local\Temp\pq_checks\compiled_sweep_20261012\sweep_kernel.dll')


def main():
    p = argparse.ArgumentParser(); p.add_argument('maps', nargs='+'); p.add_argument('--n', type=int, default=20000)
    a = p.parse_args()
    rng = np.random.default_rng(11)
    for name in a.maps:
        mesh = CachedCompiledSweepMap(ML / 'collision_maps' / (name + '.npz'), DLL); load_queries(mesh, DLL, CACHE)
        engine = GeneralPhysics(mesh); sim = NativeSim(engine)
        lo, hi = mesh.bounds[0], mesh.bounds[1]
        rays = []
        for _ in range(a.n):
            q = lo + (hi - lo) * rng.random(3); r = float(rng.choice([.2, .25, .3, .675]))
            rays.append((q + [0, 0, .05], q - [0, 0, r + .10], True))
        for _ in range(a.n):
            q = lo + (hi - lo) * rng.random(3); d = rng.standard_normal(3); d *= rng.uniform(.2, 12.) / np.linalg.norm(d)
            rays.append((q, q + d, bool(rng.random() < .7)))
        tri = mesh.triangles
        for _ in range(a.n // 2):
            i = int(rng.integers(len(tri))); j = int(rng.integers(3))
            v = tri[i, j] if rng.random() < .5 else (tri[i, j] + tri[i, (j + 1) % 3]) * .5
            rays.append((v + [0, 0, 1.], v - [0, 0, 1.], True))
        bad = 0; hits = 0; t_py = 0.; t_cc = 0.
        for s, e, cull in rays:
            t0 = time.perf_counter(); ref = CollisionMap._raycast_py(mesh, s, e, cull); t1 = time.perf_counter()
            mesh._native_ray = sim
            got = mesh.raycast(s, e, cull); t2 = time.perf_counter()
            mesh._native_ray = None
            t_py += t1 - t0; t_cc += t2 - t1
            hits += ref is not None
            if ref != got:
                bad += 1
                if bad <= 5:
                    print('MISMATCH', name, s.tolist(), e.tolist(), cull, '\n  py', ref, '\n  cc', got)
        print('%s: %d rays (%d hits) mismatches %d; python %.1f us, compiled %.1f us per ray'
              % (name, len(rays), hits, bad, 1e6 * t_py / len(rays), 1e6 * t_cc / len(rays)))


if __name__ == '__main__':
    main()
