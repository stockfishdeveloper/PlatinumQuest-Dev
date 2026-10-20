"""Wall bounces (2026-10-19, operator: "the marble never seems to use walls to hit against to turn back on itself to get
other gems ... it can be very beneficial to bounce off of a wall which just reverses its momentum ... can we utilize wall
hits?").

Why the planner never did it: every candidate it forecast steered at the gem or along a walking route (routes keep clear
of walls), and after a gem the 'through' plans turn toward the next gem at once, so no plan ever carried its speed into a
wall. Nothing rejected bounces: the forecast is the game's own contact code and the score is time to the gems.

The engine (marble.cc, wall contact; MarbleData maxDotSlide 0.5, bounceRestitution 0.5): a hit steeper than about 30
degrees to the wall keeps the velocity along the wall and sends the part into it back at half strength (times the
material's restitution, 1 for ordinary walls); a shallower hit slides along the wall at full speed. A wall ahead of a
marble that must turn around therefore turns it at once, where braking and accelerating back takes well over a second.

Bounce plans toward the current target G, offered when the marble moves at speed away from it (the target more than
MIN_ANGLE degrees off the velocity): drive at full speed into a wall point W chosen so that the rebound heads for G (a
bank shot with the engine's restitution), and after the rebound go to G as the ordinary route plan does (through it to the
tour's next gem when there is one). Walls are found with horizontal rays in the map's collision mesh (any map, no per-map
data). Variants: the aim point shifted along the wall, and 'bounce_ahead' (keep going straight until something turns the
marble around). The exact forecast judges every candidate with the usual gates. The cluster ranking (search.follow_time)
also tries bounce plans from a pickup, so an approach that sets up a quick bounce to the next gem gets the credit.
"""
import copy
import math

import numpy as np

import nav.marble_dynamics.search as S
from nav.marble_dynamics.gem_demo import Proposal

RANGE_U = 12.          # walls within this distance ahead of the marble
RAYS = 15              # horizontal rays across the forward half-plane
FAN_DEG = 70.          # rays within this angle of the velocity
RESTITUTION = .5       # MarbleData bounceRestitution x an ordinary wall's material restitution (1)
MAX_DOT_SLIDE = .5     # MarbleData maxDotSlide: shallower hits slide along the wall instead of bouncing
WALLS = 2              # walls kept (shortest marble -> wall -> target path first)
SHIFTS_U = (0., -.75, .75)   # aim points along the wall around the bank-shot point
TOP = 11.              # speed into the wall (gem_demo.TARGET_SPEED_CAP)
AHEAD_TOPS = (11., 14.)
MIN_SPEED = 4.         # u/s: slower marbles turn by themselves
MIN_ANGLE = 90.        # degrees between the velocity and the target direction
STEPS = 64             # forecast length (replies): the target's pickup must fall inside it


def _horizontal(v):
    v = np.asarray(v, dtype=float).copy(); v[2] = 0.
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def find_walls(mesh, position, velocity, target, z):
    """Walls ahead of a marble at `position` moving with `velocity` that can bank it toward `target`:
    [(path length, wall point W, wall normal n)], W at height z by the restitution-adjusted mirror rule, shortest first."""
    P = np.asarray(position[:3], dtype=float).copy(); P[2] = z
    G = np.asarray(target[:3], dtype=float).copy(); G[2] = z
    heading = _horizontal(velocity)
    if heading is None:
        return []
    base = math.atan2(heading[1], heading[0]); found = {}
    for k in range(RAYS):
        a = base + math.radians(FAN_DEG) * (2. * k / (RAYS - 1) - 1.)
        hit = mesh.raycast(P, P + RANGE_U * np.array([math.cos(a), math.sin(a), 0.]), cull_backfaces=True)
        if hit is None:
            continue
        n = np.asarray(hit['normal'], dtype=float)
        if abs(n[2]) > .35:
            continue                      # floor, ramp or ceiling, not a wall
        n = _horizontal(n)
        Q = np.asarray(hit['point'], dtype=float)
        key = (round(math.degrees(math.atan2(n[1], n[0])) / 10.), round(float(Q @ n) / .5))
        found.setdefault(key, (Q, n))
    out = []
    for Q, n in found.values():
        hP = float((P - Q) @ n); hG = float((G - Q) @ n)
        if hP < .6 or hG < .3:
            continue                      # the marble at the wall, or the target behind it
        u = np.array([-n[1], n[0], 0.])
        pU = float((P - Q) @ u); gU = float((G - Q) @ u)
        # the along-wall speed is kept, the normal part comes back at RESTITUTION: the hit point W (along the wall)
        # satisfies hG (w - pU) = e hP (gU - w)
        w = (hG * pU + RESTITUTION * hP * gU) / (hG + RESTITUTION * hP)
        W = Q + w * u; W[2] = z
        into = _horizontal(W - P)
        if into is None or -float(into @ n) < MAX_DOT_SLIDE:
            continue                      # too shallow: a slide, not a bounce
        check = mesh.raycast(P, W - .5 * n, cull_backfaces=True)
        if check is None or np.linalg.norm(np.asarray(check['point']) - W) > .3:
            continue                      # something in the way, or the wall does not reach W
        back = mesh.raycast(W + .45 * n, G, cull_backfaces=True)
        if back is not None and back['t'] < .97:
            continue                      # the rebound line to the target is blocked
        out.append((float(np.linalg.norm(W - P) + np.linalg.norm(G - W)), W, n))
    out.sort(key=lambda r: r[0])
    return out[:WALLS]


class BounceProposal:
    """At full speed into the wall point `wall` (normal `normal`), or straight ahead when wall is None, until the
    marble rebounds; then the ordinary route plan to the target (search.SequenceProposal, through it to the next gem
    when `through` is given). With `via` (a gem), first at full speed through that gem (the through plans' pass test)."""

    def __init__(self, route, gem, steps, wall=None, normal=None, top=TOP, through=None, via=None, direct=True, via_item=None):
        self.gem = np.asarray(gem, dtype=float); self.steps = int(steps)
        self.wall = None if wall is None else np.asarray(wall, dtype=float)
        self.normal = None if normal is None else np.asarray(normal, dtype=float)
        self.top = float(top)
        self.via = None if via is None else np.asarray(via, dtype=float)
        # the via gem counts as passed when the marble's path enters its pickup box (the game's test: the item's world box
        # against the marble swept over the step, expanded by radius + 0.2 u). The through plans' 0.6 u centre test kept
        # chasing a gem collected ~1 u off-centre: the marble circled it (Cragmire 10-19 decision 2153).
        self.via_bounds = None if via_item is None else np.asarray(via_item['bounds'], dtype=float)
        # after the rebound: straight at the target keeping the speed (direct, the default: a walking route around a
        # step looped the marble back into the wall, Cragmire 10-19 decision 2153), or along the walking route
        self.after = S.SequenceProposal(route, gem, steps, through=through, direct=direct)
        self.bounced = False; self.into = False; self.heading = None; self.passed = via is None; self.previous = None

    def __call__(self, state, index):
        if index == 0:
            self.bounced = False; self.into = False; self.heading = None; self.passed = self.via is None; self.previous = None
            self.after.passed = False; self.after.previous = None
        p = np.asarray(state['position'], dtype=float); v = np.asarray(state['velocity'], dtype=float)
        if not self.passed and self.via_bounds is not None:
            a = self.previous if self.previous is not None else p
            r = float(state.get('radius', .2)) + .2
            lo, hi = self.via_bounds[:3], self.via_bounds[3:]
            self.passed = bool(np.all(np.maximum(a, p) + r >= lo) and np.all(np.minimum(a, p) - r <= hi))
            self.previous = p
            if not self.passed:
                action = Proposal(self.via, 0., speed_distance=1e3, min_speed=self.top, top=self.top)(state, 0)
                action[8] = 0
                return action
        if not self.passed:
            if self.previous is not None:
                a = self.previous; ab = p - a
                t = float(np.clip(((self.via[:3] - a) @ ab) / max(float(ab @ ab), 1e-12), 0., 1.))
                self.passed = float(np.linalg.norm(a + t * ab - self.via[:3])) < S.THROUGH_SWITCH_U
            else:
                self.passed = float(np.linalg.norm(p - self.via[:3])) < S.THROUGH_SWITCH_U
            self.previous = p
            if not self.passed:
                action = Proposal(self.via, 0., speed_distance=1e3, min_speed=self.top, top=self.top)(state, 0)
                action[8] = 0
                return action
        if not self.bounced:
            if self.wall is not None:
                vn = float(v @ self.normal); d = float((p - self.wall) @ self.normal)
                if vn < -.5:
                    self.into = True
                if (self.into and vn > .2) or d < .3:
                    self.bounced = True
            else:
                if self.heading is None:
                    self.heading = _horizontal(v)
                if self.heading is None or float(v[:2] @ self.heading[:2]) < -.5:
                    self.bounced = True
        if self.bounced:
            return self.after(state, index)
        goal = self.wall if self.wall is not None else p + 30. * self.heading
        action = Proposal(goal, 0., speed_distance=1e3, min_speed=self.top, top=self.top)(state, 0)
        action[8] = 0
        return action


def through_specs(engine, route, dynamics, gem, through, steps):
    """(label, BounceProposal) candidates that drive at full speed through the target `gem` into a wall behind it,
    banked toward the tour's next gem through[1], when that gem lies behind the target along the approach."""
    state = dynamics['native']['state']
    p = np.asarray(state['position'], dtype=float); A = np.asarray(gem[:3], dtype=float); B = np.asarray(through[1][:3], dtype=float)
    if abs(float(A[2] - p[2])) > 1.5 or abs(float(B[2] - A[2])) > 1.5:
        return []
    approach = _horizontal(A - p)
    back = _horizontal(B - A)
    if approach is None or back is None or float(approach @ back) > 0.:
        return []                         # the next gem is ahead: the through plans carry on to it
    item = next((i for i in dynamics.get('items', []) if i.get('gem') and np.linalg.norm(np.asarray(i['position']) - A) < .1), None)
    specs = []
    for i, (_, W, n) in enumerate(find_walls(engine.mesh, A, approach, B, float(p[2]))):
        u = np.array([-n[1], n[0], 0.])
        for shift, top, direct in [(s, TOP, True) for s in SHIFTS_U] + [(0., TOP, False), (0., max(AHEAD_TOPS), True)]:
            specs.append(('bounce_through_wall%d_%+.2f_%d%s' % (i, shift, int(top), '' if direct else '_route'),
                          BounceProposal(through[0], B, steps, wall=W + shift * u, normal=n, top=top, via=A, direct=direct,
                                         via_item=item)))
    return specs


def bounce_specs(engine, route, dynamics, gem, through, steps):
    """(label, BounceProposal) candidates toward the target `gem` when the marble moves at speed away from it, and
    (through_specs) through the target into a wall behind it toward the next gem."""
    state = dynamics['native']['state']
    if state.get('mode', 0) != 0:
        return []
    p = np.asarray(state['position'], dtype=float); v = np.asarray(state['velocity'], dtype=float)
    G = np.asarray(gem[:3], dtype=float)
    lead = through_specs(engine, route, dynamics, gem, through, steps) if through is not None else []
    if abs(float(G[2] - p[2])) > 1.5 or float(np.linalg.norm(v[:2])) < MIN_SPEED:
        return lead                       # raised gems (the climb families) or a slow marble
    aim = np.asarray(copy.copy(route).navigate(p, G)[0], dtype=float)[:2] - p[:2]   # (a copy: the route's waypoint memory stays)
    if float(np.linalg.norm(aim)) < 1e-6:
        return lead
    cos = float(v[:2] @ aim) / (float(np.linalg.norm(v[:2])) * float(np.linalg.norm(aim)))
    if math.degrees(math.acos(max(-1., min(1., cos)))) < MIN_ANGLE:
        return lead                       # the target is ahead: nothing to turn around for
    specs = lead
    heading = _horizontal(v)
    hit = engine.mesh.raycast(p, p + RANGE_U * heading, cull_backfaces=True)
    if hit is not None and abs(float(hit['normal'][2])) <= .35:
        # 'straight ahead' only toward a wall (without one the marble rolls on and off whatever edge comes next)
        specs += [('bounce_ahead_' + str(int(top)), BounceProposal(route, G, steps, top=top, through=through)) for top in AHEAD_TOPS]
    for i, (_, W, n) in enumerate(find_walls(engine.mesh, p, v, G, float(p[2]))):
        u = np.array([-n[1], n[0], 0.])
        for shift, direct in [(s, True) for s in SHIFTS_U] + [(0., False)]:
            specs.append(('bounce_wall%d_%+.2f%s' % (i, shift, '' if direct else '_route'),
                          BounceProposal(route, G, steps, wall=W + shift * u, normal=n, through=through, direct=direct)))
    return specs
