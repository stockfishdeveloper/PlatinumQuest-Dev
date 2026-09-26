"""Measured marble physics shared by the terrain graph, the observation and the model (2026-09-23).

Everything here comes from physics/jump_envelope.json, written by `python -m nav.measure_jump`
(the game as the oracle: teleport onto a flat strip at speed v, jump, record apex / flight / range).
It describes the MARBLE, not a map, so it holds on every map with the same marble datablock.

KOTM measurement 2026-09-23 (step 16 ms, jump held 2 ticks, forward held through the flight):
    apex 1.33-1.35 u and flight 0.75-0.77 s at every speed (a jump adds ~7.3 u/s vertically),
    range(v) = 0.75 v + 2.0 u  (v = 5.05 -> 5.95, 9.52 -> 8.85, 13.29 -> 11.71, 17.16 -> 14.67).
Two identical trials per condition: the simulation is deterministic under the fixed step.
"""
import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVELOPE_PATH = os.path.join(_HERE, 'physics', 'jump_envelope.json')

# fallbacks (the KOTM measurement above) if the json is missing
_DEFAULT_SPEED = np.array([0.0, 5.05, 5.61, 7.72, 9.52, 11.39, 13.29, 15.22, 17.16])
_DEFAULT_RANGE = np.array([1.81, 5.95, 6.39, 7.28, 8.85, 10.26, 11.71, 13.18, 14.67])
JUMP_APEX = 1.33            # u: the most a jump can rise (landing must be lower than this above the lip)
JUMP_FLIGHT_S = 0.76        # s: same-height flight time
LANDING_MARGIN = 1.0        # 0.5 -> 1.0 on 2026-09-24 (HANDOFF 28.34): approved jumps succeeded only 66-71 % at 0.5
                            # (forced-jump eval 117 pts / 56 falls). With 1.0 the marble centre must clear the far lip
                            # by a full unit: the 7 u holes drop out at cruise speed, corner cuts and bays stay in.
CRUISE_SPEED = 11.0         # 8 -> 11 on 2026-09-24 21:40 (HANDOFF 28.36): the speed the terrain graph assumes when
                            # admitting a jump edge. The human demo of 21:28 made 17 gap jumps (all landed) at a mean
                            # takeoff speed of 10.7 u/s crossing 9-15 u; at 8 u/s the graph admitted <= 6.5 u and held
                            # an edge for 2 of the 17. The field must value the cut at the speed a marble CAN carry;
                            # the crossable flag (speed-conditioned) still decides when it actually may jump.
MIN_JUMP_GAP = 3.0          # u: gaps narrower than this are rolled around, never jumped (the 2 u centre hole made the
                            # policy 'jump for the sake of jumping', operator 2026-09-24). KOTM-ONLY HACK, per the
                            # operator: NOT a long-term lower bound. Remove (0.0) or replace with a detour-based test
                            # before training other maps, where short gaps may be the jumps that matter.


def _load():
    try:
        d = json.load(open(ENVELOPE_PATH))
        rows = sorted(d.get('hold', []), key=lambda r: r['speed'])
        rows = [r for r in rows if r['range'] > 0]
        if len(rows) >= 3:
            sp = np.array([r['speed'] for r in rows]); rg = np.array([r['range'] for r in rows])
            apex = float(np.median([r['apex'] for r in rows])); fl = float(np.median([r['flight_s'] for r in rows]))
            return sp, rg, apex, fl
    except (OSError, ValueError, KeyError):
        pass
    return _DEFAULT_SPEED, _DEFAULT_RANGE, JUMP_APEX, JUMP_FLIGHT_S


_SPEED, _RANGE, JUMP_APEX, JUMP_FLIGHT_S = _load()
_SLOPE = float((_RANGE[-1] - _RANGE[-3]) / (_SPEED[-1] - _SPEED[-3]))    # extrapolation beyond the table


def jump_range(speed):
    """Horizontal distance (u) a jump covers from takeoff speed `speed` (u/s), forward held, landing
    at the takeoff height. Scalar or array."""
    s = np.asarray(speed, dtype=np.float64)
    r = np.interp(s, _SPEED, _RANGE)
    over = s > _SPEED[-1]
    if np.any(over):
        r = np.where(over, _RANGE[-1] + (s - _SPEED[-1]) * _SLOPE, r)
    return float(r) if np.ndim(speed) == 0 else r


def crossable(gap, speed, margin=LANDING_MARGIN):
    """True if a gap of `gap` u (lip to far floor) is within a jump from takeoff speed `speed`."""
    return np.asarray(gap) + margin <= jump_range(speed)


def speed_for_gap(gap, margin=LANDING_MARGIN):
    """Takeoff speed (u/s) needed to clear `gap` u with the landing margin."""
    need = float(gap) + margin
    return float(np.interp(need, _RANGE, _SPEED)) if need <= _RANGE[-1] else float(_SPEED[-1] + (need - _RANGE[-1]) / _SLOPE)


MAX_JUMP_GAP = float(jump_range(CRUISE_SPEED) - LANDING_MARGIN)   # the widest gap the graph admits (~7.5 u)


# ----------------------------------------------------------------------------- landing predictor (2026-09-25)
# Fitted to the flat-strip measurements above (physics/jump_envelope.json, hold-forward rows): the marble's
# centre follows z = z0 + VZ0 t - G/2 t^2 and, with forward held along the heading, s = v t + A_AIR/2 t^2.
# Residuals of the fit: range within 0.2 u at 9-17 u/s, 0.6 u at 5 u/s.
VZ0 = 7.40              # u/s: effective vertical launch. marble.cs jumpImpulse is 7.5; the engine subtracts the
                        # velocity already leaving the surface (~0.1 for a rolling marble). Per-tick capture 2026-09-25
                        # (logs/physics/arcs_*.csv): apex 1.336-1.346 u measured vs 1.340 simulated, same-height
                        # flight 0.736 s, hold-forward ranges within 0.09 u at 8-12 u/s. The impulse fires 2 ticks
                        # (32 ms) after the jump key is first seen.
G = 20.0                # u/s^2: gravity (marble.cs)
A_AIR = 6.8             # u/s^2: airAcceleration (engine default 5.0) x the move magnitude the joystick actually
                        # delivers on the camera diagonal (measured 6.8 from the per-tick capture; sqrt(2) x 5 would
                        # be 7.07). Hold-forward ranges then match within 0.02-0.09 u at 8-12 u/s.
LAUNCH_DELAY = 0.032    # s: the impulse fires two 16 ms ticks after the jump key is first seen (per-tick capture);
                        # the marble rolls on at its speed meanwhile, so the arc starts that far along the heading
R_MARBLE = 0.25         # u: rest height of the marble centre above the floor (trace: 0.20-0.29)
DT = 0.016              # s: terrain sampling step
SUB_DT = 0.008          # s: the engine's physics sub-step (marble.cc advancePhysics)
T_MAX = 2.5             # s: give up (void) after this
HIT_TOL = 0.45          # u: a floor level within [centre - R, centre + HIT_TOL] on the way down is a landing
WALL_TOL = 0.15         # u: if the centre is more than R + this below a floor level we reach, we hit its face (wall)


def _fine_for(terrain):
    """The 0.1 u raster for the terrain's map (nav/fine_terrain), or None."""
    name = getattr(terrain, 'name', None) or getattr(terrain, 'map_name', None)
    if not name:
        return None
    try:
        from nav.fine_terrain import get
        return get(str(name))
    except Exception:
        return None


def predict_landing(terrain, x, y, z, vx, vy, hold=True, jump=True):
    """Simulate a jump (or a drop, jump=False) from the current state on `terrain` (TerrainGrid or
    TerrainMap: needs heights_at and, if present, walkable_at). Returns a dict:
      verdict: 'floor' (lands on walkable floor), 'edge' (lands on floor the walk grid rejects),
               'wall' (meets floor from the side or below before landing), 'void' (no floor within T_MAX)
      t: flight time, x, y, z: landing point, dist: horizontal distance flown, crossed_void: True if the
      arc passed over cells with no floor at all, drop: landing height minus takeoff height.
    Heading = current velocity (forward held along it when hold=True). Pure marble physics + height map:
    nothing map-specific."""
    sp = math.hypot(vx, vy)
    hx, hy = (vx / sp, vy / sp) if sp > 1e-6 else (0.0, 0.0)
    z0 = float(z); vz = VZ0 if jump else 0.0
    a = A_AIR if hold else 0.0
    if jump:                                   # the key-to-impulse latency: the marble is further along when it launches
        x = x + hx * sp * LAUNCH_DELAY; y = y + hy * sp * LAUNCH_DELAY
    # ENGINE-EXACT integration (HANDOFF 28.40): marble.cc advancePhysics steps 8 ms at a time, velocity first
    # (mVelocity = A * dt + mVelocity) then position (testMove moves by velocity * dt). The arc is generated
    # the same way here, then sampled every DT for the terrain test. The jump impulse is applied before the
    # first step (applyContactForces runs inside the same step as the first gravity update).
    n_sub = int(round(DT / SUB_DT))
    ts = np.arange(1, int(T_MAX / DT) + 1) * DT
    ss = np.empty(len(ts)); zs = np.empty(len(ts)); vzs = np.empty(len(ts))
    s_ = 0.0; v_ = sp; z_ = z0; vz_ = vz
    for n in range(len(ts)):
        for _ in range(n_sub):
            v_ += a * SUB_DT; vz_ -= G * SUB_DT
            s_ += v_ * SUB_DT; z_ += vz_ * SUB_DT
        ss[n] = s_; zs[n] = z_; vzs[n] = vz_
    pxs = x + hx * ss; pys = y + hy * ss
    fine = _fine_for(terrain)
    H = (fine if fine is not None else terrain).heights_at(pxs, pys)   # (K, N): 0.1 u raster when available
    crossed = False; prev_levels_above = False
    for n in range(len(ts)):
        t = float(ts[n]); s = float(ss[n]); px = float(pxs[n]); py = float(pys[n])
        pz = float(zs[n]); vzt = float(vzs[n])
        h = H[:, n]
        levels = h[np.isfinite(h)]
        if len(levels) == 0:
            crossed = True; prev_levels_above = False
            continue
        below = levels[levels <= pz + HIT_TOL]
        above = levels[levels > pz + HIT_TOL]
        # a level that is above the marble now but was not above it (or absent) a step ago: we ran into it
        if len(above) and not prev_levels_above and t > 2 * DT:
            lv = float(above.min())
            if lv - pz < 3.0:                       # a lip / wall within reach, not a ceiling far overhead
                return {'verdict': 'wall', 't': t, 'x': px, 'y': py, 'z': pz, 'dist': s, 'crossed_void': crossed, 'drop': lv - z0}
        prev_levels_above = len(above) > 0
        if len(below) and vzt < 0.0:
            lv = float(below.max())
            if pz - R_MARBLE < lv - WALL_TOL:      # centre already below the floor level we reach: that is the
                return {'verdict': 'wall', 't': t, 'x': px, 'y': py, 'z': pz, 'dist': s, 'crossed_void': crossed, 'drop': lv - z0}   # face of the far lip
            if pz - R_MARBLE <= lv:                 # underside meets floor while descending: landing
                walk = getattr(terrain, 'walkable_at', None)
                ok = walk(px, py) if walk else True
                return {'verdict': 'floor' if ok else 'edge', 't': t, 'x': px, 'y': py, 'z': lv + R_MARBLE, 'dist': s,
                        'crossed_void': crossed, 'drop': lv + R_MARBLE - z0}
    return {'verdict': 'void', 't': t, 'x': px, 'y': py, 'z': pz, 'dist': sp * t + 0.5 * a * t * t, 'crossed_void': True, 'drop': float('nan')}


LATERAL_TOL = 1.0       # u: the landing must hold when the flight drifts this far to either side (air control)


def _floor_present(terrain, x, y):
    h = terrain.heights_at([x], [y])[:, 0]
    return bool(np.isfinite(h).any())


def _lands(terrain, p, hx, hy):
    """A landing counts if the landing point or the floor 1 u further along the heading is walkable (a
    landing on a bevel / edge cell that rolls onto the floor is a landing)."""
    if p['verdict'] == 'floor':
        return True
    if p['verdict'] == 'edge':
        walk = getattr(terrain, 'walkable_at', None)
        return bool(walk and walk(p['x'] + hx, p['y'] + hy))
    return False


def jump_verdict(terrain, x, y, z, vx, vy, hold=True):
    """The approval test (2026-09-25, HANDOFF 28.39). 'floor' only if the arc from a point that has floor
    under it lands on floor, and so do the same arcs shifted LATERAL_TOL to either side (air-control
    drift). hold=True: forward held through the flight (what a human does; the policy must learn to).
    hold=False: pure ballistic, the marble lands whatever it does in the air. Returns (verdict, landing
    dict of the centre arc). Verdicts: 'floor', 'edge', 'wall', 'void', 'nocontact'."""
    if not _floor_present(terrain, x, y):
        return 'nocontact', None
    sp = math.hypot(vx, vy)
    if sp < 1e-3:
        return 'void', None
    hx, hy = vx / sp, vy / sp
    lx, ly = -hy, hx
    centre = predict_landing(terrain, x, y, z, vx, vy, hold=hold)
    if not _lands(terrain, centre, hx, hy):
        return centre['verdict'], centre
    for side in (-LATERAL_TOL, LATERAL_TOL):
        p = predict_landing(terrain, x + lx * side, y + ly * side, z, vx, vy, hold=hold)
        if not _lands(terrain, p, hx, hy):
            return p['verdict'], centre
    return 'floor', centre
