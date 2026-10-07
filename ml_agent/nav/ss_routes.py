"""Experimental first-turn route costs for one held Super Speed.

Uses current motion and terrain, with the existing kick approval. No map identity,
coordinates, item-placement knowledge, or assumed future powerup pickups. This
only estimates the initial turn cost; actual game-point gates judge its utility.
"""
import math

from nav.obs import (ON_FLOOR_VZ, ON_FLOOR_DZ, SS_KEY_LAG_S,
                     ss_kick_result, ss_kick_ok)
from nav.protocol import RAW_POW_HELD


def first_turn_costs(terrain, raw, gems, minimum_distance=8.0):
    if len(raw) <= RAW_POW_HELD or int(raw[RAW_POW_HELD]) != 2:
        return {}
    x, y, z, vx, vy, vz = map(float, raw[:6])
    speed = math.hypot(vx, vy)
    if (speed <= 1.0 or abs(vz) >= ON_FLOOR_VZ
            or abs(z - terrain.floor_z(x, y, z)) >= ON_FLOOR_DZ):
        return {}
    costs = {}
    for gem in gems:
        dx, dy = gem[0] - x, gem[1] - y
        distance = math.hypot(dx, dy)
        if distance < minimum_distance or abs(gem[2] - z) > ON_FLOOR_DZ:
            continue
        ux, uy = dx / distance, dy / distance
        _, (rvx, rvy), resolved_speed = ss_kick_result(vx, vy, ux, uy)
        # Pure braking with a still-negative result cannot execute this turn.
        if rvx * ux + rvy * uy <= 0:
            continue
        # No following-gem shortcut here: require braking room along this first leg.
        if ss_kick_ok(terrain, x, y, z, rvx, rvy, resolved_speed,
                      distance, ux, uy):
            costs[gem] = speed * SS_KEY_LAG_S
    return costs
