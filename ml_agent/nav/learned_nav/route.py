"""Jump-aware gem order for the hybrid (stage 5b): when to send the planner on a jump leg the navigator's chooser
would not take.

The human's 163-177 KOTM rounds cross a hole on about 16 legs a round, and 29 of 41 of those go to the SECOND-nearest
gem: the order is chosen around the jump (the navigator's greedy chooser takes such legs about once a round, and a
walking tour chooser at inference made the navigator slower, log 28.50-28.54). So the navigator keeps its own greedy
target, and this module only says when an order that STARTS with a jump leg beats the best order that starts with
the greedy target by ROUTE_GAIN_S; the planner then drives that whole leg (hybrid.py).

Leg times (every order of the visible gems, at most MAX_GEMS):
* walking: the navigator's time, fitted on its KOTM legs 2026-09-29 (689 legs, residual sd 0.29 s):
  NAV_T0 + NAV_K x walking distance (the planner's walking field) + NAV_DETOUR_S when the walk is DETOUR_U or more
  longer than the straight line + NAV_TURN_S x (1 - cos) of the turn at the start of the leg;
* jumping (floor gems only): the straight line from the start to the gem crosses one run of void no wider than the
  measured jump envelope allows (nav/physics.MAX_JUMP_GAP), with RUNUP_U of floor before it and LAND_U after, and no
  raised floor on the way: JUMP_K x straight distance + the same turn cost + JUMP_EXTRA_S.
Nothing here names a map: geometry, the measured envelope and the navigator's measured pace only.
"""
import itertools
import math

import numpy as np

from nav.learned_nav import planner as PL
from nav.learned_nav import dynamics3 as D3
from nav.learned_nav.drills import floor_ref_of
from nav.physics import MAX_JUMP_GAP

NAV_T0, NAV_K, NAV_DETOUR_S, NAV_TURN_S = 0.211, 0.090, 0.384, 0.253
DETOUR_U = 3.0
PICK_SPEED = 7.7             # u/s: the navigator's usual speed at a pickup (the turn cost of the fit is at that speed)
JUMP_K = 0.039               # s/u along the straight line of a jump leg. Stage 6b crossing drill (v1, 26 KOTM legs from
JUMP_EXTRA_S = 1.12          # the navigator's real pickup states): t = 1.12 + 0.039 x straight + 1.47 x (1 - cos turn),
JUMP_TURN_S = 1.47           # residual sd 0.34 s. Aligned starts (within 30 deg) take 1.66-1.73 s (the human's 1.66),
                             # 30-60 deg 2.3 s; before the drill the guess was 0.09 x straight + 0.3 with the walking turn cost
RUNUP_U = 2.0                # u of floor on the line before the void (a run-up)
LAND_U = 1.0                 # u of floor on the line after the void, up to the gem
STEP_U = 0.25                # line sampling
ROUTE_GAIN_S = 0.3           # a jump-first order must beat the greedy target's best order by this much
ORDER_SWITCH_S = 0.3         # Route.order keeps the current target unless another first gem's best order beats it by this
ORDER_JUMPS = False          # Route.order: legs may be jumped where the line allows (the oracle / consult takes them).
                             # OFF: g6v (on) 139.2, u/gem 11.09 against greedy 10.99: orders built on jumps the navigator walks
SETUP_EXTRA_S = 0.4          # a setup order (the planner drives to the greedy gem so as to leave it aligned toward the jump
                             # gem): the jump leg is costed as aligned, the shaped approach costs this much extra
MAX_GEMS = 5                 # orders of at most this many visible gems (5! = 120)


class Route:
    def __init__(self, g):
        self.g = g
        self.fields = {}         # gem key -> (walking-distance field to the gem, floor_ref)
        self.pairs = {}          # (gem a, gem b) -> (walk, straight, walk start direction, jump line direction or None)

    @staticmethod
    def key(gem):
        return (round(float(gem[0]), 1), round(float(gem[1]), 1), round(float(gem[2]), 1))

    def field(self, gem):
        k = self.key(gem)
        if k not in self.fields:
            gem3 = np.asarray(gem[:3], float)
            fr = floor_ref_of(self.g, gem3)
            self.fields[k] = (PL.walk_distance_field(self.g, *PL.gem_disc(gem3), fr), fr)
        return self.fields[k]

    def walk(self, x, y, gem):
        """Walking distance from (x, y); within WALK_CLEAR of an edge (off the field) the nearest field cell within
        1 u plus the distance to it (the first version returned inf there and wrongly ruled out the greedy target)."""
        D, _ = self.field(gem)
        w = float(PL.field_at(self.g, D, x, y))
        if np.isfinite(w):
            return w
        r = np.array([0.25, 0.5, 0.75, 1.0]); a = np.linspace(-math.pi, math.pi, 16, endpoint=False)
        rr, aa = np.meshgrid(r, a)
        vals = PL.field_at(self.g, D, x + rr.ravel() * np.cos(aa.ravel()), y + rr.ravel() * np.sin(aa.ravel())) + rr.ravel()
        return float(np.min(vals))

    def out_dir(self, x, y, gem, r=1.0):
        """Direction the walk from (x, y) to the gem starts in (steepest descent of its field over r u)."""
        D, _ = self.field(gem)
        ang = np.linspace(-math.pi, math.pi, 32, endpoint=False)
        w = PL.field_at(self.g, D, x + r * np.cos(ang), y + r * np.sin(ang))
        if not np.isfinite(w).any():
            return math.atan2(gem[1] - y, gem[0] - x)
        return float(ang[int(np.nanargmin(np.where(np.isfinite(w), w, np.nan)))])

    def jump_gap(self, x, y, gem):
        """Void crossed by the straight line from (x, y) to a floor gem if a jump can take it (one void run no wider
        than MAX_JUMP_GAP, RUNUP_U of floor before, LAND_U after, no raised floor on the line); 0.0 if the line stays
        on the floor; None otherwise."""
        _, fr = self.field(gem)
        d = math.hypot(gem[0] - x, gem[1] - y)
        n = max(2, int(d / STEP_U) + 1)
        t = np.linspace(0.0, 1.0, n)
        xs = x + t * (gem[0] - x); ys = y + t * (gem[1] - y)
        lv, above = D3.level_below(self.g, xs, ys, np.full(n, fr + 1.0))
        if above.any():
            return None
        floor = np.isfinite(lv) & (lv >= fr - 0.6)
        if floor.all():
            return 0.0
        void = np.nonzero(~floor)[0]
        first, last = int(void[0]), int(void[-1])
        if (~floor[first:last + 1]).sum() != last - first + 1:
            return None                                          # more than one void run
        step = d / (n - 1)
        gap = (last - first + 1) * step
        if gap > MAX_JUMP_GAP or first * step < RUNUP_U or (n - 1 - last) * step < LAND_U:
            return None
        return gap

    def legs(self, x, y, heading, speed, gem):
        """(walking time, jumping time or inf, exit direction of each) for one leg from (x, y) moving along heading."""
        w = self.walk(x, y, gem)
        st = math.hypot(gem[0] - x, gem[1] - y)
        scale = speed / PICK_SPEED
        if np.isfinite(w):
            d0 = self.out_dir(x, y, gem)
            t_w = (NAV_T0 + NAV_K * w + (NAV_DETOUR_S if w - st >= DETOUR_U else 0.0)
                   + NAV_TURN_S * (1.0 - math.cos(d0 - heading)) * scale)
        else:
            t_w = math.inf
        line = math.atan2(gem[1] - y, gem[0] - x)
        gap = self.jump_gap(x, y, gem)
        t_j = math.inf
        if gap is not None and gap > 0.0:
            t_j = JUMP_K * st + JUMP_TURN_S * (1.0 - math.cos(line - heading)) + JUMP_EXTRA_S
        return t_w, t_j, line

    def order(self, p, v, gems, current, floating=lambda gem: False):
        """WHOLE-SPAWN order (operator 10-01, log 37): the best order of the visible gems by the fitted leg times,
        every leg walked or jumped (whichever is faster where the line allows a jump), the first leg from the
        marble's state (speed and heading), the turn at each gem charged. Sticky: the current target's best order is
        kept unless the best order beats it by ORDER_SWITCH_S. Returns (target, next, info)."""
        gems = [q for q in gems[:MAX_GEMS]]
        if not gems:
            return None, None, None
        px, py = float(p[0]), float(p[1])
        speed = math.hypot(float(v[0]), float(v[1]))
        heading = math.atan2(float(v[1]), float(v[0])) if speed > 0.5 else 0.0
        sp0 = speed if speed > 0.5 else 0.0
        first = {}
        for k, q in enumerate(gems):
            first[k] = (math.inf, math.inf, 0.0) if floating(q) else self.legs(px, py, heading, sp0, q)
        for a, qa in enumerate(gems):
            for b, qb in enumerate(gems):
                pk = (self.key(qa), self.key(qb))
                if a != b and pk not in self.pairs:
                    w = self.walk(qa[0], qa[1], qb)
                    st = math.hypot(qb[0] - qa[0], qb[1] - qa[1])
                    d0 = self.out_dir(qa[0], qa[1], qb)
                    gap = None if floating(qb) else self.jump_gap(qa[0], qa[1], qb)
                    line = math.atan2(qb[1] - qa[1], qb[0] - qa[0])
                    self.pairs[pk] = (w, st, d0, line if (gap is not None and gap > 0.0) else None)

        def leg_t(a, b, heading_in):
            w, st, d0, line = self.pairs[(self.key(gems[a]), self.key(gems[b]))]
            t_w = (NAV_T0 + NAV_K * w + (NAV_DETOUR_S if w - st >= DETOUR_U else 0.0)
                   + NAV_TURN_S * (1.0 - math.cos(d0 - heading_in))) if np.isfinite(w) else math.inf
            t_j = math.inf
            if ORDER_JUMPS and line is not None:
                t_j = JUMP_K * st + JUMP_TURN_S * (1.0 - math.cos(line - heading_in)) + JUMP_EXTRA_S
            return min(t_w, t_j)

        best = {}                                                  # first gem -> (time, order)
        for order in itertools.permutations(range(len(gems))):
            f = order[0]
            t_w, t_j, h0 = first[f]
            t = min(t_w, t_j if ORDER_JUMPS else math.inf)
            if not np.isfinite(t):
                continue
            h_in = h0
            for a, b in zip(order, order[1:]):
                t += leg_t(a, b, h_in)
                h_in = math.atan2(gems[b][1] - gems[a][1], gems[b][0] - gems[a][0])
            if f not in best or t < best[f][0]:
                best[f] = (t, order)
        if not best:
            return None, None, None
        fb = min(best, key=lambda k: best[k][0])
        pick = fb
        if current is not None:
            held = next((k for k, q in enumerate(gems) if math.hypot(q[0] - current[0], q[1] - current[1]) < 0.75), None)
            if held is not None and held in best and best[fb][0] >= best[held][0] - ORDER_SWITCH_S:
                pick = held
        t, order = best[pick]
        info = {'t_order': round(t, 3), 'order': [list(map(float, gems[k][:2])) for k in order]}
        return gems[order[0]], (gems[order[1]] if len(order) > 1 else None), info

    def choose(self, p, v, gems, greedy, floating=lambda gem: False):
        """A gem to take by a jump leg first when that beats the greedy target by ROUTE_GAIN_S, else None.
        gems: visible gems (x, y, z, ...); greedy: the navigator's target. Returns (gem, info); info['setup'] names
        the gem to jump to AFTER the greedy gem when that order beats the greedy order (see below)."""
        gems = [q for q in gems[:MAX_GEMS]]
        if len(gems) < 2 or greedy is None:
            return None, None
        px, py = float(p[0]), float(p[1])
        speed = math.hypot(float(v[0]), float(v[1]))
        heading = math.atan2(float(v[1]), float(v[0])) if speed > 0.5 else 0.0
        sp0 = speed if speed > 0.5 else 0.0
        first = {}
        for k, q in enumerate(gems):
            if floating(q):
                first[k] = (math.inf, math.inf, 0.0)
            else:
                first[k] = self.legs(px, py, heading, sp0, q)
        for a, qa in enumerate(gems):
            for b, qb in enumerate(gems):
                pk = (self.key(qa), self.key(qb))
                if a != b and pk not in self.pairs:
                    # gem to gem: walking distance, its start direction, and the jump line if one is possible
                    w = self.walk(qa[0], qa[1], qb)
                    st = math.hypot(qb[0] - qa[0], qb[1] - qa[1])
                    d0 = self.out_dir(qa[0], qa[1], qb)
                    gap = None if floating(qb) else self.jump_gap(qa[0], qa[1], qb)
                    line = math.atan2(qb[1] - qa[1], qb[0] - qa[0])
                    self.pairs[pk] = (w, st, d0, line if (gap is not None and gap > 0.0) else None)

        def leg_t(a, b, heading_in, jumps):
            """(time, jumped) gem a -> gem b arriving along heading_in; jumps: a jump leg allowed."""
            w, st, d0, line = self.pairs[(self.key(gems[a]), self.key(gems[b]))]
            t_w = (NAV_T0 + NAV_K * w + (NAV_DETOUR_S if w - st >= DETOUR_U else 0.0)
                   + NAV_TURN_S * (1.0 - math.cos(d0 - heading_in))) if np.isfinite(w) else math.inf
            t_j = math.inf
            if jumps and line is not None:
                t_j = JUMP_K * st + JUMP_TURN_S * (1.0 - math.cos(line - heading_in)) + JUMP_EXTRA_S
            return (t_j, True) if t_j < t_w else (t_w, False)

        gi = next((k for k, q in enumerate(gems) if abs(q[0] - greedy[0]) < 0.5 and abs(q[1] - greedy[1]) < 0.5), None)
        if gi is None:
            return None, None
        # the greedy baseline walks every leg (the navigator never jumps on its own); a jump-first order jumps its
        # first leg from the marble's state; a setup order walks to the greedy gem and jumps its SECOND leg (the human
        # arrives aligned because the order is chosen around the jump, log 33.3): the navigator drives to the greedy
        # gem with that gem as its next-gem hint, and the planner is consulted at the pickup
        best_g, best_j, best_j_first, best_s, best_s_gem = math.inf, math.inf, None, math.inf, None
        for order in itertools.permutations(range(len(gems))):
            f = order[0]
            t_w, t_j, _ = first[f]
            h0 = math.atan2(gems[f][1] - py, gems[f][0] - px)
            if f == gi and np.isfinite(t_w):
                t = t_w; h_in = h0
                for a, b in zip(order, order[1:]):
                    t += leg_t(a, b, h_in, False)[0]
                    h_in = math.atan2(gems[b][1] - gems[a][1], gems[b][0] - gems[a][0])
                best_g = min(best_g, t)
                if len(order) > 1:
                    t = t_w + SETUP_EXTRA_S; h_in = h0; jumped2 = False
                    for n, (a, b) in enumerate(zip(order, order[1:])):
                        if n == 0:
                            line = self.pairs[(self.key(gems[a]), self.key(gems[b]))][3]
                            if line is None:
                                break                                    # no crossing to jump: not a setup order
                            dt, jj = leg_t(a, b, line, True)             # aligned: the approach is shaped for it
                        else:
                            dt, jj = leg_t(a, b, h_in, False)
                        t += dt; jumped2 |= (n == 0 and jj)
                        h_in = math.atan2(gems[b][1] - gems[a][1], gems[b][0] - gems[a][0])
                    else:
                        if jumped2 and t < best_s:
                            best_s, best_s_gem = t, order[1]
            if np.isfinite(t_j):
                t = t_j; h_in = h0
                for a, b in zip(order, order[1:]):
                    t += leg_t(a, b, h_in, False)[0]
                    h_in = math.atan2(gems[b][1] - gems[a][1], gems[b][0] - gems[a][0])
                if t < best_j:
                    best_j, best_j_first = t, f
        info = {'t_greedy': round(best_g, 3), 't_jump': round(best_j, 3), 't_setup': round(best_s, 3)}
        if best_j_first is not None and best_j < best_g - ROUTE_GAIN_S and best_j <= best_s:
            return gems[best_j_first], info
        if best_s_gem is not None and best_s < best_g - ROUTE_GAIN_S:
            info['setup'] = gems[best_s_gem]
        return None, info
