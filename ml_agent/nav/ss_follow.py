"""Inference-only velocity-tracking diagnostic for the first pickup after a kick."""
import math

from nav.obs import VEC_NEXT, WAYPOINT_DIST_SCALE, SS_DECEL, SS_KEY_LAG_S


def follow_action(action, raw, vec):
    if vec[8] <= .5:
        return action, False
    ux, uy = map(float, vec[:2])
    distance = float(vec[2]) * WAYPOINT_DIST_SCALE
    vx, vy = float(raw[3]), float(raw[4])
    speed = math.hypot(vx, vy)
    if distance <= .5 or speed <= 8.0 or len(action) > 5 and action[5] > .5:
        return action, False
    arrival_speed = 6.0
    if vec[VEC_NEXT + 4] > .5:
        nx = float(vec[VEC_NEXT]) * float(vec[VEC_NEXT + 2]) * WAYPOINT_DIST_SCALE - ux * distance
        ny = float(vec[VEC_NEXT + 1]) * float(vec[VEC_NEXT + 2]) * WAYPOINT_DIST_SCALE - uy * distance
        nd = math.hypot(nx, ny)
        if nd > 1e-6:
            cosine = max(-1., min(1., (nx * ux + ny * uy) / nd))
            arrival_speed = 14.0 - 8.0 * math.sqrt((1.0 - cosine) / 2.0)
    # Reserve the distance traveled while a command takes effect and the pickup
    # radius before computing a braking envelope from the measured deceleration.
    room = max(0., distance - speed * SS_KEY_LAG_S - .8)
    reference = math.sqrt(arrival_speed ** 2 + 2.0 * SS_DECEL * room)
    ax, ay = reference * ux - vx, reference * uy - vy
    if math.hypot(ax, ay) < .1:
        return action, False
    out = list(action)
    out[0], out[1] = ax, ay
    out[2], out[3], out[4] = 1., 0., 0.
    return out, True
