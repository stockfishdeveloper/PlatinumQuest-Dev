"""Policy action -> game joystick reply (no torch import: the instance workers use this)."""
import math
import os

# VIEWING ONLY. The camera yaw is normally set so the commanded direction lands on the camera
# diagonal, where the per-axis clamp allows a move of length sqrt(2) instead of 1. That tracks
# the ~38 deg/decision command jitter, so the camera swings ~15 times a second and a human
# cannot follow what the marble is doing. NAV_CAM_LOCK=1 pins the camera and falls back to the
# infinity norm, which reaches the corners of a WORLD-aligned square: mean applied magnitude
# 1.141 instead of 1.414, i.e. the 82.1-point configuration rather than the 91.8-point one.
# Same policy and same decisions, just without the overspeed. Do NOT set this for training.
CAM_LOCK = os.environ.get("NAV_CAM_LOCK", "0") == "1"


def action_to_joystick(dx, dy, throttle, jump=0, brake=0, vx=0.0, vy=0.0):
    """(dx, dy) world direction (+x right/east, +y forward/north), throttle [0,1]; brake overrides
    the direction to anti-velocity at full throttle.

    Returns (fwd, back, left, right, jump, cam_yaw). The camera yaw is chosen so the commanded
    direction lands on the CAMERA diagonal, where the engine's per-axis cap allows a vector of
    length sqrt(2) instead of 1. A human does this with the mouse: their yaw sweeps the full circle
    and their applied force averages 1.23-1.36 in every world direction, while a yaw pinned to 0
    gives 1.414 only near world diagonals and 1.0 on the axes (agent mean was 1.141).
    `format_action(*joystick)` passes the sixth element straight through as cam_yaw.
    """
    if brake > 0.5:
        speed_xy = math.sqrt(vx * vx + vy * vy)
        if speed_xy > 0.1:
            dx, dy, throttle = -vx / speed_xy, -vy / speed_xy, 1.0
    else:
        # INFINITY norm, not the 2-norm (fixed 2026-09-20). The game takes per-axis inputs, so the
        # reachable set is a SQUARE: a keyboard player holding two keys sends a vector of length
        # 1.414, and does so on 76.6 % of active frames. Normalising to the 2-norm confined the
        # policy to the inscribed circle and threw away up to 41 % of the available force on every
        # diagonal. That is the 1.41x speed gap against the human almost exactly, and it is why
        # raising THROTTLE_FLOOR bought only ~4 %: throttle could only scale a vector that had
        # already been clipped. The commanded DIRECTION is identical under both scalings, so this
        # changes only how hard the marble is pushed, never where. Neither axis exceeds 1.0.
        # 2-norm here, NOT the infinity norm. The inf-norm existed to reach the corners of a
        # WORLD-aligned square; with the camera rotating to meet the commanded direction every
        # direction is already a corner, so inf-norming first and then scaling by sqrt(2) would
        # overshoot the per-axis cap (it produced camera axes of 1.414, caught by the assertion
        # in this patch's self-test).
        n = math.sqrt(dx * dx + dy * dy)
        if n > 1e-6:
            dx, dy = dx / n, dy / n
    # Put the commanded direction on the camera diagonal. mlAgent.cs rotates world -> camera as
    # camera_angle = world_angle + yaw, so yaw = pi/4 - phi lands it at 45 deg, and a world
    # magnitude of sqrt(2)*throttle makes each camera axis exactly `throttle` -- at the cap, never
    # over it. Braking keeps the unit vector: it wants anti-velocity, not a diagonal boost.
    if brake > 0.5:
        cam_yaw = 0.0
        mx, my = dx * throttle, dy * throttle
    else:
        if CAM_LOCK:
            cam_yaw = 0.0
            n = max(abs(dx), abs(dy))          # infinity norm: corners of the world-aligned square
            if n > 1e-6:
                dx, dy = dx / n, dy / n
            mx, my = dx * throttle, dy * throttle
        else:
            cam_yaw = math.pi / 4.0 - math.atan2(dy, dx)
            m = math.sqrt(2.0) * throttle
            mx, my = dx * m, dy * m
    return (round(max(my, 0.0), 6), round(max(-my, 0.0), 6), round(max(-mx, 0.0), 6), round(max(mx, 0.0), 6),
            int(jump), round(cam_yaw, 6))
