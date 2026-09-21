"""Policy action -> game joystick reply (no torch import: the instance workers use this)."""
import math


def action_to_joystick(dx, dy, throttle, jump=0, brake=0, vx=0.0, vy=0.0):
    """(dx, dy) world direction (+x right/east, +y forward/north), throttle [0,1]; brake overrides
    the direction to anti-velocity at full throttle. Returns (fwd, back, left, right, jump)."""
    if brake > 0.5:
        speed_xy = math.sqrt(vx * vx + vy * vy)
        if speed_xy > 0.1:
            dx, dy, throttle = -vx / speed_xy, -vy / speed_xy, 1.0
    else:
        n = math.sqrt(dx * dx + dy * dy)
        if n > 1e-6:
            dx, dy = dx / n, dy / n
    mx, my = dx * throttle, dy * throttle
    return (round(max(my, 0.0), 6), round(max(-my, 0.0), 6), round(max(-mx, 0.0), 6), round(max(mx, 0.0), 6), int(jump))
