"""Bit-exactness of nav/native/route_native (RoutePlanner.distance and navigate through the compiled routing) against the
Python methods on random positions of a map's walking terrain (as the planner builds it).
usage (repo root, PYTHONPATH=ml_agent): python test_route_native.py <map> <n goals> <n positions per goal>"""
import sys, time, math, copy, importlib.util
m, ngoal, npos = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sys.argv = ['x', '--map', m, '--variant', 'x', '--pool', 'planner_v84_start0_20261017', '--per-map', '1', '--native', '--real-clock', '--out', 'NUL']
spec = importlib.util.spec_from_file_location('sb_rt', 'ml_agent/tools/sim_bench/sim_bench.py'); SB = importlib.util.module_from_spec(spec); spec.loader.exec_module(SB)
import numpy as np
import nav.routing as RT
import nav.native.route_native as RN
a = SB.planner_args(m); terrain, engine = SB.setup(a, m)
geo = a.geometry_terrain
assert RT.NATIVE_DISTANCE is not None, 'attach() did not install the compiled distance'
rng = np.random.default_rng(3)
nodes = np.argwhere(np.isfinite(geo.walk_z))
bad_d = bad_n = 0; nd = nn = 0; td = [0., 0.]; tn = [0., 0.]; fb = 0
for g in range(ngoal):
    k, j, i = nodes[rng.integers(len(nodes))]
    gem = np.array([float(geo.wxs[i]) + rng.uniform(-.3, .3), float(geo.wys[j]) + rng.uniform(-.3, .3), float(geo.walk_z[k, j, i]) + .3])
    route = RT.RoutePlanner(geo); route.navigate(gem, gem)
    for q in range(npos):
        k2, j2, i2 = nodes[rng.integers(len(nodes))]
        x = float(geo.wxs[i2]) + rng.uniform(-2, 2); y = float(geo.wys[j2]) + rng.uniform(-2, 2); z = float(geo.walk_z[k2, j2, i2]) + rng.uniform(-.5, 1.)
        if rng.random() < .1:
            x += rng.uniform(-40, 40)                       # off the walk grid: the straight-line fallback
        kind = rng.integers(3)
        pos = (x, y, z) if kind == 0 else (np.array([x, y, z]) if kind == 1 else [np.float32(x), np.float32(y), np.float32(z)])
        if kind == 2:
            fb += 1
        # distance
        RT.NATIVE_DISTANCE = None; t0 = time.perf_counter(); A = route.distance(pos); t1 = time.perf_counter()
        RT.NATIVE_DISTANCE = RN.distance; B = route.distance(pos); t2 = time.perf_counter()
        td[0] += t1 - t0; td[1] += t2 - t1; nd += 1
        if not (type(A) == type(B) and (A == B or (math.isnan(A) and math.isnan(B)))):
            bad_d += 1
            if bad_d <= 3:
                print('distance DIFF', pos, A, B)
        # navigate (the learned proposal's per-reply call), from a copy of the route with its waypoint memory
        r1 = copy.copy(route); r2 = copy.copy(route); r3 = copy.copy(route)
        hook = RT.NATIVE_NAVIGATE; RT.NATIVE_NAVIGATE = None
        t0 = time.perf_counter(); A = RT.RoutePlanner.navigate(r1, np.asarray(pos, dtype=float), gem); t1 = time.perf_counter()
        RT.NATIVE_NAVIGATE = hook
        B = RN.navigate(engine.native_sim, r2, np.asarray(pos, dtype=float), gem); t2 = time.perf_counter()
        C = r3.navigate(np.asarray(pos, dtype=float), gem)          # the planner-wide hook
        if repr(C) != repr(A) or repr(r3.waypoint) != repr(r1.waypoint):
            bad_n += 1; print('hook DIFF', pos, A, C)
        tn[0] += t1 - t0; tn[1] += t2 - t1; nn += 1
        if not (repr(A) == repr(B) and repr(r1.waypoint) == repr(r2.waypoint)):
            bad_n += 1
            if bad_n <= 3:
                print('navigate DIFF', pos, A, B, r1.waypoint, r2.waypoint)
print('%s: distance %d calls, %d differ (Python %.1f us, native %.1f us, %d float32 inputs sent to Python); navigate %d calls, %d differ (Python %.0f us, native %.1f us)' % (
    m, nd, bad_d, 1e6 * td[0] / nd, 1e6 * td[1] / nd, fb, nn, bad_n, 1e6 * tn[0] / nn, 1e6 * tn[1] / nn))
