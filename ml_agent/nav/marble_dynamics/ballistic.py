"""Ballistic landing projection: finish an airborne candidate's flight with no input.

A 24-reply search horizon rejects every candidate that is still in the air at
its end, which rules out Super Jumps, long blasts and most gap jumps. Instead,
when a candidate ends airborne, project the remaining flight with NO input:

1. coarse: step the free-flight parabola and raycast each segment against the
   collision mesh; reject at once if the path leaves the mission volume or drops
   below every surface (microseconds per candidate);
2. exact: for a plausible landing, continue the real engine with no-input
   replies, in chunks, until the marble lands and settles (or fails), capped.

The accepted plan is the searched prefix plus that no-input tail, which is
exactly what gets sent, so "safe with no input" is the plan itself, not a
heuristic about air control. Every decision still revalidates the remaining
plan from the newly observed state.
"""
import numpy as np

from nav.protocol import NOOP_ACTION
from nav.marble_dynamics.bridge import reply, predict_replies
from nav.marble_dynamics.viability import resume_forecast, concatenate
from nav.marble_dynamics.search import terminal_support

MAX_FLIGHT_REPLIES = 64          # 4.1 s of no-input flight (Super Jump needs ~2.5 s)
HELICOPTER_FLIGHT_REPLIES = 96   # 6.1 s
CHUNK = 8
SETTLE = 4


def coarse_landing(mesh, state, bounds=None, max_s=4.5, dt=0.064):
    """Free-flight parabola against the mesh. Returns dict(hit, reason, ...)."""
    p = np.asarray(state['position'], dtype=float)
    v = np.asarray(state['velocity'], dtype=float)
    scale = .25 if state['effects'][5][0] else 1.
    g = np.asarray(state['gravity_direction'], dtype=float) * float(state['parameters']['gravity']) * scale
    r = float(state['radius'])
    floor = float(np.min(mesh.triangles[:, :, 2]))
    t = 0.
    while t < max_s:
        q = p + v * dt + 0.5 * g * dt * dt
        step = q - p
        length = float(np.linalg.norm(step))
        if length > 1e-9:
            hit = mesh.raycast(p, q + step / length * r, cull_backfaces=True)
            if hit is not None:
                normal = np.asarray(hit['normal'], dtype=float)
                return dict(hit=True, t=t, normal=normal.tolist(), floor_like=bool(normal[2] > .4), point=q.tolist())
        p, v, t = q, v + g * dt, t + dt
        if bounds is not None:
            lo, hi = bounds
            if np.any(p + r < lo) or np.any(p - r > hi):
                return dict(hit=False, reason='leaves_bounds', t=t)
        if p[2] < floor - 2 * r and v[2] < 0:
            return dict(hit=False, reason='below_all_surfaces', t=t)
    return dict(hit=False, reason='no_landing_within_horizon', t=t)


class NoInput:
    """No-input replies; the camera yaw is held so nothing turns."""
    def __init__(self, steps, yaw=0.):
        self.steps = steps
        self.yaw = float(yaw)

    def __call__(self, state, index):
        action = reply(NOOP_ACTION)
        action[5] = self.yaw
        return action


def project_landing(engine, dynamics, rollout, max_replies=MAX_FLIGHT_REPLIES, settle=SETTLE):
    """Continue `rollout` with no input until it lands and settles, fails, or hits the cap."""
    current = rollout
    yaw = rollout['proposed'][-1][5] if rollout['proposed'] else 0.
    total = 0
    while total < max_replies:
        resumed = resume_forecast(dynamics, current)
        chunk = predict_replies(engine, resumed, NoInput(CHUNK, yaw), stop_on_ambiguity=True)
        current = concatenate(current, chunk)
        total += CHUNK
        if current['failure'] or current['ambiguous_intervals']:
            return current, 'prediction_failure' if current['failure'] else 'contact_ambiguity'
        supported, stable, _ = terminal_support(engine.mesh, current['states'][-settle:])
        if stable:
            return current, None
    return current, 'no_landing_within_cap'


def ballistic_continuation(engine, dynamics, rollout, bounds=None):
    """Coarse filter then exact no-input projection. Returns (rollout or None, reason, coarse)."""
    end = rollout['states'][-1]
    coarse = coarse_landing(engine.mesh, end, bounds)
    if not coarse['hit']:
        return None, 'coarse:' + coarse['reason'], coarse
    cap = HELICOPTER_FLIGHT_REPLIES if end['effects'][5][0] else MAX_FLIGHT_REPLIES
    try:
        landed, reason = project_landing(engine, dynamics, rollout, cap)
    except ValueError as exc:
        return None, 'unsupported_projection:' + str(exc)[:60], coarse
    if reason is not None:
        return None, 'exact:' + reason, coarse
    return landed, None, coarse
