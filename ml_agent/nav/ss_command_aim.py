"""Opt-in command-time Super Speed aiming diagnostic on supported level floor.

Forced-use traces show the kick follows the previous observation's aim. Predict
that pending interval, then solve for the desired observable velocity after the
kick. Constant recent acceleration is an approximation, checked in game gates.
"""
import math

from nav.obs import (ON_FLOOR_VZ, ON_FLOOR_DZ, SS_DV, SS_DECEL,
                     ss_kick_result, ss_kick_ok, level_run)


def predict_aim(terrain, raw, previous, goal, next_goal=None, dt=.064):
    if previous is None or dt <= 0:
        return None
    x, y, z, vx, vy, vz = map(float, raw[:6])
    if (abs(vz) >= ON_FLOOR_VZ or abs(float(previous[5])) >= ON_FLOOR_VZ
            or abs(z - terrain.floor_z(x, y, z)) >= ON_FLOOR_DZ):
        return None
    dvx, dvy = vx - float(previous[3]), vy - float(previous[4])
    if math.hypot(dvx, dvy) > 2.0:  # discontinuity or unmodelled collision/impulse
        return None
    pvx, pvy = vx + dvx, vy + dvy
    px, py = x + (vx + .5 * dvx) * dt, y + (vy + .5 * dvy) * dt
    speed = math.hypot(vx, vy)
    if speed > 1e-6:
        required = speed * dt + .8
        if level_run(terrain, x, y, z, vx / speed, vy / speed, required + .5) <= required:
            return None
    dx, dy = goal[0] - px, goal[1] - py
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return None
    ux, uy = dx / distance, dy / distance
    # The engine impulse is 25; observation arrives after one interval of floor
    # loss. In 32 development fires the measured net kick was 24.104 +/- .0001.
    effective_dv = SS_DV - SS_DECEL * dt
    aim, (rvx, rvy), resolved = ss_kick_result(pvx, pvy, ux, uy, dv=effective_dv)
    nrel = (next_goal[0] - px, next_goal[1] - py) if next_goal is not None else None
    if rvx * ux + rvy * uy <= 0 or not ss_kick_ok(
            terrain, px, py, z, rvx, rvy, resolved, distance, ux, uy, nrel):
        return None
    return {'aim': aim, 'position': (px, py), 'velocity': (pvx, pvy),
            'result_velocity': (rvx, rvy), 'effective_dv': effective_dv, 'lead_s': dt}
