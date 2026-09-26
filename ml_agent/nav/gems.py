"""Real-gem target selection shared by real rounds (nav/real_run.py) and real-gem TRAINING
(nav/vec_worker.py, NAV_REAL_GEMS=1). Torch-free on purpose: the workers must stay torch-free.

visible_gems(raw)  -> [(world_x, world_y, world_z, value, dist), ...] nearest first
choose(gems, current, pos, vel) -> (target, next_target)
    Sticky: keep the current target while it is still on the list unless another is much closer
    (SWITCH_GAIN). MOMENTUM_K adds a turn-around cost K * speed * (1 - cos(angle between the
    marble's velocity and the bearing to the gem)), the +8-14 point ordering term of HANDOFF 28.7.
"""
import math
import os

import numpy as np

from nav.protocol import RAW_GEMS

ABSENT = -500.0                # RAW_GEMS pads missing slots with value/dist <= -500
STICKY_TOL = 0.75              # u: a gem within this of the current target counts as the same gem
SWITCH_GAIN = 0.60             # only abandon the current target for one at most this fraction of its cost
MOMENTUM_K = float(os.environ.get('NAV_MOMENTUM_K', '1.6'))   # s; 0 = pure nearest (HANDOFF 28.7)
VALUE_WEIGHT = float(os.environ.get('NAV_VALUE_WEIGHT', '0'))  # >0 prefers higher-value gems


def visible_gems(raw):
    """Real gems as (world_x, world_y, world_z, value, dist), nearest first, absent slots dropped."""
    px, py, pz = float(raw[0]), float(raw[1]), float(raw[2])
    g = np.asarray(raw[RAW_GEMS], dtype=np.float64).reshape(5, 5)
    out = []
    for dx, dy, dz, value, dist in g:
        if value <= ABSENT or dist <= ABSENT:
            continue
        out.append((px + dx, py + dy, pz + dz, float(value), float(dist)))
    out.sort(key=lambda t: t[4])
    return out


TOUR_TURN_U = float(os.environ.get('NAV_TOUR_TURN_U', '6.0'))   # u charged for a full reversal at a gem
                               # (scaled by sin(half the turn angle)): turning at ~7 u/s with ~13 u/s^2 of
                               # grip costs about 2 v^2 sin(phi/2) / a = 7.5 u of rolling time; 6 is a round
                               # figure below that. Only used by plan_tour.
TOUR_SWITCH = 0.90             # plan_tour keeps the current target unless another first gem gives a plan at
                               # most this fraction of the best plan that starts with the current target


def plan_tour(gems, current, pos, vel, dist=None, momentum_k=None, turn_u=None, switch=None):
    """WHOLE-SPAWN order (2026-09-26, HANDOFF 28.50; evaluation-only via NAV_TOUR in nav/real_run.py).
    choose() ranks gems one at a time; this scores every order of the visible gems (at most 5! = 120):
    leg distances + the momentum cost of the first turn (as in choose) + turn_u * sin(phi / 2) for the turn
    at each gem. Returns (target, next_target) of the best order, sticky to `current`.
    dist(from_xy, gem) -> distance; None = straight line (needs no terrain map)."""
    if not gems:
        return None, None
    k = MOMENTUM_K if momentum_k is None else momentum_k
    tu = TOUR_TURN_U if turn_u is None else turn_u
    sw = TOUR_SWITCH if switch is None else switch
    px, py = float(pos[0]), float(pos[1]); vx, vy = float(vel[0]), float(vel[1])
    speed = math.hypot(vx, vy)
    if dist is None:
        def dist(a, g):
            return math.hypot(g[0] - a[0], g[1] - a[1])

    def turn(ax, ay, bx, by):
        na, nb = math.hypot(ax, ay), math.hypot(bx, by)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        c = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
        return math.sin(math.acos(c) / 2.0)

    def cost(order):
        c = dist((px, py), order[0])
        dx, dy = order[0][0] - px, order[0][1] - py
        if speed > 1.0 and k > 0:
            d = math.hypot(dx, dy)
            if d > 1e-6:
                c += k * speed * (1.0 - (dx * vx + dy * vy) / (d * speed))
        for i in range(1, len(order)):
            a, b = order[i - 1], order[i]
            c += dist((a[0], a[1]), b)
            ex, ey = b[0] - a[0], b[1] - a[1]
            c += tu * turn(dx, dy, ex, ey)
            dx, dy = ex, ey
        return c

    import itertools
    plans = [(cost(p), p) for p in itertools.permutations(gems)]
    best_c, best = min(plans, key=lambda t: t[0])
    if current is not None:
        held = [g for g in gems if math.hypot(g[0] - current[0], g[1] - current[1]) < STICKY_TOL and abs(g[2] - current[2]) < 2.0]
        if held and best[0] is not held[0]:
            hc, hp = min(((c, p) for c, p in plans if p[0] is held[0]), key=lambda t: t[0])
            if best_c >= sw * hc:
                best = hp
    return best[0], (best[1] if len(best) > 1 else None)


def choose(gems, current, pos=None, vel=None, terrain=None, mfield=None, momentum_k=None):
    """Pick the gem to chase, and the one after it for the next-gem observation block."""
    if not gems:
        return None, None
    k = MOMENTUM_K if momentum_k is None else momentum_k
    speed = 0.0
    if k > 0 and pos is not None and vel is not None:
        speed = math.hypot(float(vel[0]), float(vel[1]))

    def cost(g):
        c = g[4]
        if terrain is not None and mfield is not None and pos is not None:
            c = terrain.dist_at(mfield, g[0], g[1], (float(pos[0]), float(pos[1])))
        if VALUE_WEIGHT > 0 and g[3] > 0:
            c = c / (g[3] ** VALUE_WEIGHT)
        if speed > 1.0:
            dx, dy = g[0] - float(pos[0]), g[1] - float(pos[1])
            d = math.hypot(dx, dy)
            if d > 1e-6:
                cos_t = (dx * float(vel[0]) + dy * float(vel[1])) / (d * speed)
                c += k * speed * (1.0 - cos_t)
        return c

    ranked = sorted(gems, key=cost)
    best = ranked[0]
    nxt = ranked[1] if len(ranked) > 1 else None
    if current is None:
        return best, nxt
    still = [g for g in gems if math.hypot(g[0] - current[0], g[1] - current[1]) < STICKY_TOL
             and abs(g[2] - current[2]) < 2.0]
    if not still:
        return best, nxt
    held = still[0]
    if cost(best) < SWITCH_GAIN * cost(held):
        return best, nxt
    other = [g for g in ranked if g is not held]
    return held, (other[0] if other else None)
