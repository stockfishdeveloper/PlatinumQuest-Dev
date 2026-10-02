"""Analytic "jump now" check from the engine's measured constants (log 36): microseconds, no learned model, so it can
run inside the training workers (operator 10-02: "train the navigator and planner together") and at real time.

The key sent now fires two decisions later (log 36.2). From the state then: the fire step adds JUMP_DZ / JUMP_VZ,
the flight is ballistic under gravity plus AIR_ACCEL x the input (sqrt 2 toward the gem, or none), the landing is
where the bottom meets the floor level at that xy. Verdict: the takeoff point has floor under it, nothing stands in
the way (no level above the marble on the path), the landing floor is at the gem's level and the landing point has
CLEAR_U of floor around it, the path passes within PICK_U of the gem (in the air or in a short straight roll after
the landing), and the marble can stop on the floor ahead after a slide landing (A_STOP). Validated on the centre
drill (log 37.7).
"""
import math

import numpy as np

from nav.learned_nav import dynamics3 as D3

R = 0.19
DT = 0.064
G = 20.0
AIR_ACCEL = 5.0
SQ2 = math.sqrt(2.0)
JUMP_DZ = 0.4268
JUMP_VZ = 6.108
KEY_LAG = 2                  # decisions between the key sent and the fire
PICK_U = 1.0                 # within this of the gem (3-D) = taken
CLEAR_U = 0.5                # floor required this far around the landing point
A_STOP = 13.0                # u/s^2 braking after a landing
STOP_MARGIN = 0.5
REACH_U = 12.0               # a landing this close to the gem over continuous floor at its level counts (the navigator rolls on; 4 u rejected 61 of 90 planner crossings that landed short and rolled)
MAX_FLIGHT = 24              # decisions


def level(g, x, y, ztop):
    lv, above = D3.level_below(g, np.asarray([x]), np.asarray([y]), np.asarray([ztop]))
    return (float(lv[0]) if np.isfinite(lv[0]) else None), bool(above[0])


def floor_ahead(g, x, y, ux, uy, floor_ref, reach=10.0):
    """Distance along (ux, uy) to the first point without floor at floor_ref (reach if none)."""
    dq = np.arange(0.25, reach + 0.01, 0.25)
    lv, _ = D3.level_below(g, x + dq * ux, y + dq * uy, np.full(len(dq), floor_ref + 1.0))
    void = ~(np.isfinite(lv) & (lv >= floor_ref - 0.6))
    return float(dq[int(np.argmax(void))]) if void.any() else reach


def jump_ok(g, p, v, gem, floor_ref):
    """Either flight works: no air input (a bounce landing, the speed kept low) or the input toward the gem (a slide
    landing, faster but reaching further). The trained navigator chooses its own air input and landing."""
    return jump_now(g, p, v, gem, floor_ref, 'none') or jump_now(g, p, v, gem, floor_ref, 'gem')


def jump_now(g, p, v, gem, floor_ref, air='gem'):
    """True if pressing the jump key NOW takes the gem and lands safely by the analytic flight. p, v: the marble's
    position and velocity; gem: (x, y, z); floor_ref: the gem's floor level. air: 'gem' (input toward the gem in
    the air) or 'none'."""
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    vx, vy = float(v[0]), float(v[1])
    sp = math.hypot(vx, vy)
    if sp < 1.0:
        return False
    # the takeoff point: two decisions of rolling at the current velocity
    tx, ty = px + vx * DT * KEY_LAG, py + vy * DT * KEY_LAG
    lv0, above0 = level(g, tx, ty, pz + 0.3)
    if lv0 is None or abs(lv0 - floor_ref) > 0.6 or above0:
        return False
    z = lv0 + R + JUMP_DZ; vz = JUMP_VZ
    x, y = tx + vx * DT, ty + vy * DT                       # the fire step's horizontal travel
    gx, gy, gz = float(gem[0]), float(gem[1]), float(gem[2])
    if air == 'gem':
        d = math.hypot(gx - x, gy - y)
        ax, ay = (AIR_ACCEL * SQ2 * (gx - x) / d, AIR_ACCEL * SQ2 * (gy - y) / d) if d > 1e-6 else (0.0, 0.0)
    else:
        ax, ay = 0.0, 0.0
    dmin = math.sqrt((x - gx) ** 2 + (y - gy) ** 2 + (z - gz) ** 2)
    landed = None
    for _ in range(MAX_FLIGHT):
        x1 = x + vx * DT + 0.5 * ax * DT * DT; y1 = y + vy * DT + 0.5 * ay * DT * DT
        z1 = z + vz * DT - 0.5 * G * DT * DT
        vx += ax * DT; vy += ay * DT; vz -= G * DT
        lv, _ = level(g, x1, y1, max(z, z1) + R)             # the highest level under the marble's top
        if lv is not None and lv > z1 + R - 0.05 and lv > floor_ref + 0.6:
            return False                                     # a wall or lip across the marble's height band
        if lv is not None and z1 - R <= lv:
            if abs(lv - floor_ref) > 0.6:
                return False                                 # lands on another level
            z1 = lv + R
            landed = (x1, y1)
        x, y, z = x1, y1, z1
        dmin = min(dmin, math.sqrt((x - gx) ** 2 + (y - gy) ** 2 + (z - gz) ** 2))
        if landed is not None:
            break
    if landed is None:
        return False
    # floor around the landing point
    for a in (0.0, 1.571, 3.142, 4.712):
        lvq, _ = level(g, x + CLEAR_U * math.cos(a), y + CLEAR_U * math.sin(a), z + 0.3)
        if lvq is None or abs(lvq - floor_ref) > 0.6:
            return False
    # slide landing (input held): the speed is kept, folded into the floor plane
    sp_h = math.hypot(vx, vy)
    if air == 'gem':
        total = math.sqrt(vx * vx + vy * vy + vz * vz)
        if sp_h > 1e-6:
            vx, vy = vx / sp_h * total, vy / sp_h * total; sp_h = total
    ux, uy = (vx / sp_h, vy / sp_h) if sp_h > 1e-6 else (1.0, 0.0)
    # the gem taken in the air, or REACHABLE from the landing: within REACH_U over floor at the gem's level all the
    # way (the navigator steers after the landing; a straight roll missed 69 of 90 planner crossings, log 37.7)
    if dmin > PICK_U:
        dg = math.hypot(gx - x, gy - y)
        if dg > REACH_U:
            return False
        for t in np.linspace(0.0, 1.0, max(2, int(dg / 0.25) + 1)):
            lvq, _ = level(g, x + t * (gx - x), y + t * (gy - y), z + 0.3)
            if lvq is None or abs(lvq - floor_ref) > 0.6:
                return False
    # stopping room on the floor ahead
    room = floor_ahead(g, x, y, ux, uy, floor_ref)
    return sp_h * sp_h / (2.0 * A_STOP) + STOP_MARGIN <= room
