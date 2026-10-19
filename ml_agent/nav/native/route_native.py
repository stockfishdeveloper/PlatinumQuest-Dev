"""RoutePlanner.navigate through the compiled routing (2026-10-19, search speed-up toward real-time play).

controllers.inc already holds line-for-line ports of nav/routing.py's line_clear and path_waypoint (the compiled
SequenceProposal follows its route with them). navigate(sim, route, pos, gem) is RoutePlanner.navigate on those ports:
the same answers and the same waypoint memory on the route object, ~25 us instead of ~390 us per call inside the learned
model's rollout. Cases the port does not cover (a goal field to (re)compute, a terrain without ground clearance, which
the compiled line_clear always checks, or input types it does not model) run the Python method."""
import math

import numpy as np

import nav.routing as RT
from nav.native.marble_sim import _ptr

_BUF = {}
_SIMS = []          # the compiled simulators of this process (nav.native.attach); route terrains registered with them


def register(sim):
    """Called by nav.native.attach: RoutePlanner.distance on terrains registered with this simulator runs in C++."""
    if not any(s is sim for s in _SIMS):
        _SIMS.append(sim)
    RT.NATIVE_DISTANCE = distance
    RT.NATIVE_NAVIGATE = navigate_hook


def _handle(terrain):
    for sim in _SIMS:
        entry = sim.terrains.get(id(terrain))
        if entry is not None and entry[1]['object'] is terrain:
            return sim.lib, entry[0]
    return None


def distance(route, pos):
    """RoutePlanner.distance(pos) through the compiled dist_at (controllers.inc, a line-for-line port of
    TerrainGrid.dist_at), or None for the Python method: a terrain not registered with a simulator, a field that is not
    float32, or coordinate types whose NumPy promotion the port does not model (x and y must both be NumPy float64, which
    computes the walk cell in float64, or both Python floats, which compute it in float32)."""
    field = route.field
    if field.dtype != np.float32 or not field.flags.c_contiguous:
        return None
    h = _handle(route.terrain)
    if h is None:
        return None
    x, y, z = pos[0], pos[1], pos[2]
    tx, ty = type(x), type(y)
    if tx is np.float64 and ty is np.float64:
        strong = 1
    elif tx is float and ty is float:            # (a float32 x would make the off-grid fallback's hypot float32)
        strong = 0
    else:
        return None
    lib, th = h
    key = route.key
    if z is None:
        return lib.ms_dist_at(th, _ptr(field), float(x), float(y), float(key[0]), float(key[1]), 0, 0.0, strong)
    return lib.ms_dist_at(th, _ptr(field), float(x), float(y), float(key[0]), float(key[1]), 1, float(z), strong)


def navigate(sim, route, pos, gem):
    """RoutePlanner.navigate for the learned proposal's rollout (the engine's simulator given explicitly)."""
    if sim is not None:
        r = _navigate(sim.lib, lambda: sim.terrain(route.terrain), route, pos, gem)
        if r is not None:
            return r
    return route.navigate(pos, gem)


def navigate_hook(route, pos, gem):
    """nav.routing.NATIVE_NAVIGATE: RoutePlanner.navigate on a terrain registered with a simulator, or None."""
    h = _handle(route.terrain)
    if h is None:
        return None
    return _navigate(h[0], lambda: h[1], route, pos, gem)


def _navigate(lib, handle, route, pos, gem):
    terrain = route.terrain
    field = route.field
    wp = route.waypoint
    if (field is None or getattr(terrain, 'ground_clearance', None) is None or not isinstance(gem, np.ndarray)
            or gem.dtype != np.float64 or gem.ndim != 1 or len(gem) < 3 or field.dtype != np.float32 or not field.flags.c_contiguous
            or RT.goal_key(gem) != route.key or RT.LOOKAHEAD_U != 12.0 or RT.WAYPOINT_HOLD_U != 2.5 or RT.GEM_SNAP_U != 1.5
            or (wp is not None and not all(type(v) is float for v in wp[:3]))):
        return None
    th = handle()
    x, y, z = map(float, pos[:3])
    gx, gy, gz = float(gem[0]), float(gem[1]), float(gem[2])
    # a line toward the gem samples NumPy floats (strong), a line toward a stored waypoint plain floats (weak)
    if lib.ms_line_clear(th, x, y, gx, gy, 1, z, 1, gz, 1, 1):
        route.waypoint = None
        return tuple(gem[:3]), False
    if (wp is not None and math.hypot(wp[0] - x, wp[1] - y) > RT.WAYPOINT_HOLD_U
            and lib.ms_line_clear(th, x, y, wp[0], wp[1], 1, z, 1, wp[2], 0, 0)):
        return wp, True
    bufs = _BUF.get('b')
    if bufs is None:
        bufs = _BUF['b'] = (np.empty(3), np.empty(3))
    g, out = bufs; g[0] = gx; g[1] = gy; g[2] = gz
    if lib.ms_path_waypoint(th, _ptr(field), x, y, _ptr(g), 1, z, _ptr(out)):
        wp = (float(out[0]), float(out[1]), float(out[2]))
        route.waypoint = wp
        return wp, True
    route.waypoint = None
    return tuple(gem[:3]), False
