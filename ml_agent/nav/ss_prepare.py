"""Inference-only approach experiment; not part of the PPO action distribution.

Keep approaching a gem on a clear floor when a held Super Speed can make the
following turn. This tests whether premature ordinary braking prevents useful
kicks. Actual firing still uses the existing, current-state approval mask.
"""
import math

from nav.obs import (VEC_POW, VEC_NEXT, WAYPOINT_DIST_SCALE, SS_KEY_LAG_S,
                     SS_RUN_MARGIN, level_run)


def prepare_action(action, terrain, raw, vec, max_distance=18.0, minimum_next=8.0):
    if len(action) > 5 and action[5] > .5:
        return action, False
    if vec[VEC_POW + 1] <= .5 or vec[8] <= .5 or vec[VEC_POW + 27] <= .5:
        return action, False
    distance = float(vec[2]) * WAYPOINT_DIST_SCALE
    if not .5 < distance <= max_distance or vec[VEC_NEXT + 4] <= .5:
        return action, False
    ux, uy = map(float, vec[:2])
    nx = float(vec[VEC_NEXT]) * float(vec[VEC_NEXT + 2]) * WAYPOINT_DIST_SCALE - ux * distance
    ny = float(vec[VEC_NEXT + 1]) * float(vec[VEC_NEXT + 2]) * WAYPOINT_DIST_SCALE - uy * distance
    if math.hypot(nx, ny) < minimum_next:
        return action, False
    x, y, z, vx, vy = map(float, raw[:5])
    speed = math.hypot(vx, vy)
    along = vx * ux + vy * uy
    if speed <= 1.0 or along < .866 * speed:
        return action, False
    required = distance + SS_KEY_LAG_S * speed + SS_RUN_MARGIN
    lip, _, _ = terrain.gap_along(x, y, z, ux, uy, max_u=required + 1.0)
    if min(lip, level_run(terrain, x, y, z, ux, uy, required + 1.0)) <= required:
        return action, False
    # Maintain forward drive and damp sideways motion. The following kick is
    # neither scheduled nor assumed successful; normal approval is re-evaluated.
    out = list(action)
    out[0] = ux - 2.0 * (vx - along * ux) / speed
    out[1] = uy - 2.0 * (vy - along * uy) / speed
    out[2] = 1.0
    out[3] = 0.0
    out[4] = 0.0
    return out, True
