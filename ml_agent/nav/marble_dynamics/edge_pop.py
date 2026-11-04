"""Edge pops (2026-11-03/04, operator's TreasureBox demonstration, generalized): the marble comes DOWN onto an exposed floor edge
with its bottom at the edge's height while still carrying speed; the edge's reflection of the forward-and-down velocity plus the
spin-friction kick turn the run speed into a vertical pop (14 u/s up from a 14.5 u/s run: 4.3 u of rise with 5 u/s kept
forward). Nothing is pre-determined: the edges come from the collision mesh (edge_hits.mesh_lips), the approach is whatever
the marble is doing now (rolling, jumping or flying in), and the compiled forecast judges every candidate from the live state.

The landing must be within about 0.1 u of edge height, so no single plan survives a change of start state: every decision the
search forecasts a SPREAD over aim offsets along the edge x speed caps at the edge x jump replies (and no jump at all, for a
marble already in the air or rolling off a drop) and the usual cost picks the earliest that collects or lands up there. Before
the press the receding horizon may swap the plan for a better one; from the press through the contact it is held (PlanMemory).
Game at 1x (11-03, the same sweep on the exact simulator): 10 of 11 tries took TreasureBox's box-top gem 2.18-2.43 s after a
rest start. Port: nav/native/controllers.inc sub_rule kind 10 (one extra rule on the sequence plan: the speed cap at the edge)."""
import math
import numpy as np
from nav.marble_dynamics.search import SequenceProposal as _SequenceProposal, cached_route
from nav.marble_dynamics.gem_demo import Proposal
from nav.marble_dynamics.edge_hits import JUMP_IMPULSE, GRAVITY, DECISION_S

POP_CAPS_FAR = (13., 14.5)                        # speed caps held at the edge (u/s) while the edge is still far: a coarse grid shows the opportunity ...
POP_CAPS_NEAR = (12., 12.5, 13., 13.5, 14., 14.5, 15.)   # ... and the fine one (0.5 steps) once the jump is due within POP_NEAR_REPLIES (the live run: on one
                                                  # live start 13.5 collected where 13 and 14 both missed
POP_NEAR_REPLIES = 6                              # the descent estimate's jump reply at most this far ahead = near
POP_OFFSETS_FAR_U = (0., -.75, .75)               # aim offsets along the edge from the target's projection while it is far (which part of the edge pops the marble up
                                                  # toward the target is for the forecast to find: TreasureBox's working lanes lie 1 u west
                                                  # of the straight line to the gem, which runs in the box wall's shadow)
POP_OFFSETS_NEAR_U = (0.,)                        # near the edge: the sticky lane only (the budget goes to caps and timings) ...
POP_HEADING_POINT = True                          # ... (where the heading meets the edge), the line point when the marble is slow
POP_END_MARGIN_U = .35                            # aim points stay this far from the segment's ends
POP_AIM_BEYOND_U = 8.                             # the steering aims this far PAST the edge point along the edge's normal, so the run meets the
                                                  # edge nearly square from wherever it starts (the pop needs a square hit; the pickup box stays
                                                  # on the edge point)
POP_MAX_OBLIQUE_DEG = 8.                          # the approach line may be at most this far off square to the edge (the pop needs a square hit)
POP_DELAYS_FAR = (-1, 0, 1)                       # jump replies around the descent estimate while far ...
POP_DELAYS_NEAR = (-2, -1, 0, 1, 2, 3)            # ... and near (a 5 x 5 grid: 25/28 and fewer fast pops in the bed; this 7 x 6: 27/28) (the arrival estimate is about a reply coarse; a press acts a reply later)
POP_FORWARD_U = 6.                                # the pop carries the marble about this far past the edge (5 u/s kept forward, 1.4 s flight) ...
POP_BEHIND_U = .3                                 # ... so the target must lie ahead of the edge (at most this far behind it) ...
POP_LATERAL_U = 3.5                               # ... and within this of the approach line beyond the edge
POP_NO_JUMP = True                                # the no-press candidate too (already airborne, or rolling off a drop onto the edge)
POP_LOCK_AFTER_CONTACT = 3                        # replies past the forecast edge contact the plan stays held (at least) ...
POP_FLIGHT_LOCK_REPLIES = 22                      # ... and through the pop's flight when no pickup is forecast (1.4 s)
POP_RISE_MIN_U = -3.                              # edges this far below the marble still count (a gap's far edge, a drop onto a ledge lip)
POP_RISE_MAX_U = 1.25                             # ... up to a jump's reach above it
POP_DETOUR_RATIO = None                           # trigger without a climb (None = off): the walking route is this many times the straight line ...
                                                  # (ON it kept the family active over most of Vortex/Cragmire and starved the search: ten-map pairs r1 -3.45)
POP_DETOUR_SLACK_U = 3.                           # ... plus this (a gap or a wall between: an edge on the way may pop the marble across)
POP_MAX_LIPS = 1                                  # edges forecast per decision, best reach estimate first (the search budget)
POP_AHEAD_MAX_U = 14.                             # an edge farther ahead than this is not forecast yet (the run-up it needs is shorter)
POP_FLIGHT_REPLIES = 26                           # forecast horizon past the edge contact (the pop's flight and its landing, 1.7 s)
POP_UP_PER_RUN = 14. / 14.5                       # the pop's upward speed per u/s of run speed at the edge (TreasureBox tick trace: 14.5 in, 14 up)
POP_REACH_SLACK_U = .5                            # the estimated peak may fall this short of the target and still be offered
POP_LIP_SCAN = 8                                  # lips looked at (edge_hits.candidates, nearest first) before the reach ranking
POP_APPROACH_JUMP = False                         # the approach carries the best-guess jump (True) or runs without one (False, bed v2/v5)
POP_APPROACH_SPEED = 14.                          # far stage: the approach plan runs the route to the aim point at this speed, no jump, and
POP_PROMISE_U = 8.                                # ... is priced by the pop's promise: the route from the estimated landing to the gem + this
POP_LANDING_AHEAD_U = 3.                          # the estimated landing: this far past the edge along the approach, at the target's height
POP_APPROACH_MAX_DECISIONS = 70                   # an approach chosen this long (4.5 s) without a pop hit gives the edges up for this target ...
POP_FAILED_DECISIONS = 150                        # ... for this long (10 s)
POP_LEVEL_SLACK_U = .6                            # a pop plan ending at least this close below the target's height ...
POP_LEVEL_NEAR_U = 5.                             # ... and this near it horizontally is priced like a near pickup (search.edge_climb):
POP_LEVEL_COST_U = -60.                           # this + the distance left + the time (a pickup plan is -100 + time)
RADIUS = .19


def pop_failed(engine, target):
    """True while the edges toward this target are given up (the approach ran POP_APPROACH_MAX_DECISIONS without a hit)."""
    key = tuple(round(float(c), 1) for c in target[:3]); until = getattr(engine, 'pop_failed', {}).get(key, -1)
    return until > int(getattr(engine, 'decision_index', 0))


def note_pop_choice(engine, label, target, info=None):
    """Called by the controller with the chosen plan's label: tracks how long an approach has been running without a pop hit, and
    keeps the approach's aim point sticky (info['pop_aim'])."""
    key = tuple(round(float(c), 1) for c in target[:3]); di = int(getattr(engine, 'decision_index', 0))
    if '_pop_' in str(label) and info is not None and info.get('pop_aim') is not None:
        engine.pop_sticky_aim = (tuple(info['pop_aim'][0]), list(info['pop_aim'][1]))
    elif '_pop_' not in str(label):
        engine.pop_sticky_aim = None
    if '_pop_approach' in str(label):
        since = getattr(engine, 'pop_approach_since', None)
        if since is None or since[0] != key:
            engine.pop_approach_since = since = (key, di)
        if di - since[1] > POP_APPROACH_MAX_DECISIONS:
            failed = getattr(engine, 'pop_failed', None)
            if failed is None: failed = engine.pop_failed = {}
            failed[key] = di + POP_FAILED_DECISIONS; engine.pop_approach_since = None
    elif '_pop_' in str(label):
        engine.pop_approach_since = None   # a pop plan proper (a hit): the approach did its job


def pop_reach(lip, position, target, cap=POP_CAPS_NEAR[-1]):
    """(ok, score) of an edge for the target: the pop's estimated peak (lip height + (POP_UP_PER_RUN cap)^2 / 2g) must reach the
    target's height within POP_REACH_SLACK_U; the score is the lip-to-target distance (closer edges first)."""
    q = np.asarray(lip['point'], dtype=float); t = np.asarray(target[:3], dtype=float); p = np.asarray(position[:3], dtype=float)
    up = POP_UP_PER_RUN * float(cap); peak = float(q[2]) + up * up / (2. * GRAVITY)
    ok = peak >= float(t[2]) - POP_REACH_SLACK_U
    # the marble must be able to come down onto the edge: its bottom at edge height needs a rise within a jump's apex (or an edge
    # below it, reached from the air or by rolling off)
    ok = ok and float(lip.get('rise', 0.)) + RADIUS <= JUMP_IMPULSE * JUMP_IMPULSE / (2. * GRAVITY) + .05
    # the pop carries the marble on past the edge along the approach: the target must lie ahead of the edge, near that line
    d = q[:2] - p[:2]; n = float(np.linalg.norm(d))
    if n > 1e-6:
        u = d / n; r = t[:2] - q[:2]; fwd = float(r @ u); lat = abs(float(r[0] * u[1] - r[1] * u[0]))
        ok = ok and fwd >= -POP_BEHIND_U and fwd <= POP_FORWARD_U + POP_LATERAL_U and lat <= POP_LATERAL_U
    return ok, float(np.linalg.norm(t[:2] - q[:2]))


def pop_trigger(route, position, target):
    """The target needs height (edge_hits.climb_needed) or lies across a walking detour (a gap, a wall): a pop may be the way."""
    from nav.marble_dynamics.edge_hits import climb_needed
    if climb_needed(position, target):
        return True
    if route is None or POP_DETOUR_RATIO is None:
        return False
    try:
        d = float(route.distance(position))
    except Exception:
        return False
    straight = float(np.linalg.norm(np.asarray(target[:2], dtype=float) - np.asarray(position[:2], dtype=float)))
    return (not math.isfinite(d)) or d > POP_DETOUR_RATIO * straight + POP_DETOUR_SLACK_U


def pop_candidates(engine, position, target, max_lips=POP_MAX_LIPS, route=None):
    """Edges worth a pop toward the target, best first: edge_hits.candidates (lips on the way, any orientation, at or below the
    marble's level too) ranked by pop_reach; [] unless pop_trigger."""
    from nav.marble_dynamics.edge_hits import candidates
    if pop_failed(engine, target) or not pop_trigger(route, position, target):
        return []
    lips = candidates(engine, position, target, max_lips=POP_LIP_SCAN, rise_min=POP_RISE_MIN_U, rise_max=POP_RISE_MAX_U, need_climb=False)
    ranked = []
    for lip in lips:
        if float(lip.get('ahead', 0.)) > POP_AHEAD_MAX_U:
            continue
        ok, score = pop_reach(lip, position, target)
        if ok:
            ranked.append((score, lip))
    ranked.sort(key=lambda x: x[0])
    return [lip for _, lip in ranked[:max_lips]]


ROLL_ACC_POP = ((8., 14.), (11., 11.), (13.5, 8.), (15., 6.))   # full-throttle floor acceleration (u/s^2) up to each speed: exact engine from
                                                              # rest on TreasureBox 11-04 (0.9 u/s a reply to 8 u/s, 0.7 to 11, 0.5 to 13.5); a reply of lag first


def arrival_time(ahead, speed, top=None):
    """Seconds a full-throttle run from `speed` needs to cover `ahead` u (ROLL_ACC_POP), the speed capped at `top`."""
    cap = float(ROLL_ACC_POP[-1][0] if top is None else min(top, ROLL_ACC_POP[-1][0]))
    v = max(0., float(speed)); s = 0.; t = DECISION_S if v < .5 else 0.; dt = .016
    while s < ahead and t < 10.:
        a = next((acc for lim, acc in ROLL_ACC_POP if v < lim), 0.)
        v = min(cap, v + a * dt); s += v * dt; t += dt
    return t


def descent_reply(ahead, speed, rise, cap):
    """The jump reply that brings the marble DOWN onto the edge with its bottom at the edge's height (the jump arc's second crossing
    of rise + radius, timed from the capped full-throttle arrival); None when a jump cannot reach the edge."""
    h = max(0., float(rise) + RADIUS)
    disc = JUMP_IMPULSE * JUMP_IMPULSE - 2. * GRAVITY * h
    if disc < 0:
        return None
    t_down = (JUMP_IMPULSE + math.sqrt(disc)) / GRAVITY
    return int(round((arrival_time(ahead, speed, cap) - t_down) / DECISION_S)) - 1


def descent_replies(ahead, speed, rise, cap, around=POP_DELAYS_NEAR):
    j = descent_reply(ahead, speed, rise, cap)
    return set() if j is None else {max(0, j + k) for k in around}


class EdgePopProposal(_SequenceProposal):
    """A direct-aim sequence at the edge point that holds `cap` u/s (full throttle up to it, no braking for the aim point) with
    the jump at reply `jump` (-1: none), then the ordinary continuation to the gem once the edge is passed (the through box).
    One extra rule on the sequence plan: compiled as sub_rule kind 10 (nav/native/controllers.inc)."""

    def __init__(self, lroute, aim, steps, through, cap, jump):
        super().__init__(lroute, np.asarray(aim, dtype=float), steps, jump=int(jump), direct=True, through=through)
        self.cap = float(cap)

    def __call__(self, state, index):
        action = super().__call__(state, index)
        if not self.passed:
            action = Proposal(self.gem, 0., jump_at=self.jump - index, speed_distance=1e3, min_speed=self.cap, top=self.cap)(state, 0)
            action[8] = 0
        return action


def edge_extent(engine, lip):
    """The whole straight edge the candidate's mesh segment belongs to: collinear lip segments (edge_hits.mesh_lips) chained end to
    end at the same height, as (a, b) endpoints. The mesh splits one ledge edge into 2 u pieces; aim points may use all of it."""
    from nav.marble_dynamics.edge_hits import mesh_lips
    seg = lip.get('seg')
    if seg is None:
        return None
    key = tuple(round(float(x), 3) for x in seg[0]) + tuple(round(float(x), 3) for x in seg[1])
    cache = getattr(engine, '_pop_extents', None)
    if cache is None:
        cache = engine._pop_extents = {}
    if key in cache:
        return cache[key]
    a = np.asarray(seg[0], dtype=float); b = np.asarray(seg[1], dtype=float)
    segs, _ = mesh_lips(engine)
    if not len(segs):
        cache[key] = (a, b); return cache[key]
    u = b - a; L = float(np.linalg.norm(u))
    if L < 1e-6:
        cache[key] = (a, b); return cache[key]
    u = u / L
    for _ in range(64):
        grown = False
        for end in (0, 1):
            e = b if end else a
            for s2 in segs:
                for q, r in ((s2[0], s2[1]), (s2[1], s2[0])):
                    if float(np.linalg.norm(q - e)) > 1e-3 or abs(float(r[2] - e[2])) > 1e-3:
                        continue
                    d = r - e; n = float(np.linalg.norm(d))
                    if n < 1e-6 or abs(float(np.cross(d / n, u)[2])) > 1e-3 or (float((d / n) @ u) < 0) != bool(end == 0):
                        continue
                    if end: b = r
                    else: a = r
                    grown = True; break
                if grown: break
            if grown: break
        if not grown: break
    cache[key] = (a, b)   # the mesh does not change: computed once per segment (14 ms a decision in Python loops before)
    return cache[key]


def pop_specs(engine, route, dynamics, lip, k, steps, ctx, speed, near_allowed=True):
    """(label, EdgePopProposal) over the spread for one edge candidate (edge_hits.candidates dict with point, rise, tangent, seg)."""
    st = dynamics['native']['state']; p = np.asarray(st['position'], dtype=float)
    lp = np.asarray(lip['point'], dtype=float); tangent = np.asarray(lip.get('tangent') or [0., 0., 0.], dtype=float)
    ahead0 = float(np.linalg.norm(lp[:2] - p[:2]))
    j0 = descent_reply(ahead0, speed, lip['rise'], POP_CAPS_NEAR[-1]); est = steps if j0 is None else max(0, j0)
    near = near_allowed and est <= POP_NEAR_REPLIES   # the fine sweep only for the best-ranked edge (the search budget)
    caps = POP_CAPS_NEAR if near else POP_CAPS_FAR; around = POP_DELAYS_NEAR if near else POP_DELAYS_FAR
    v = np.asarray(st['velocity'], dtype=float); airborne = int(st.get('ground_ms', 0)) == 0 or abs(float(v[2])) > 1.
    no_jump = POP_NO_JUMP and (airborne or float(lip.get('rise', 0.)) <= 0.)   # a rolling marble below the edge must jump to come down on it
    out = []
    extent = edge_extent(engine, lip)
    aims = aim_points(lip, p, v, POP_OFFSETS_NEAR_U if near else POP_OFFSETS_FAR_U, extent, target=ctx[1])
    # a sticky aim: the aim point the approach was last chosen for (note_pop_choice) stays the first base point while the marble heads
    # for this edge, so the approach does not zig-zag between the line point and the heading point as the marble moves
    sticky = getattr(engine, 'pop_sticky_aim', None); lkey = tuple(round(float(x), 2) for x in lp)
    if sticky is not None and sticky[0] == lkey:
        sa = np.asarray(sticky[1], dtype=float)
        aims = [sa] + [a for a in aims if np.linalg.norm(a - sa) >= .1]
    if not aims:
        return out   # an edge shorter than twice the end margin, or the lanes all outside it (Sprawl round: IndexError)
    if near:
        base0 = aims[0]; aims = [base0] + [base0 + tangent * off for off in POP_OFFSETS_NEAR_U if off != 0.]
        aims = [np.round(a * 4.) / 4. for a in aims]   # (outside the edge's extent the forecast simply fails them)
    gem = np.asarray(ctx[1], dtype=float); nroute = ctx[0]
    for ai, aim in enumerate(aims):
        # the steering target: beyond the edge along its outward normal (from the marble's side)
        nrm = np.array([-tangent[1], tangent[0], 0.]) if float(np.linalg.norm(tangent[:2])) > .5 else None
        if nrm is not None:
            if float(nrm[:2] @ (aim[:2] - p[:2])) < 0.:
                nrm = -nrm
            far = np.round((aim + nrm * POP_AIM_BEYOND_U) * 4.) / 4.
        else:
            far = aim
        if ai == 0:
            # the approach (both stages: near the edge it keeps the marble driving while the sweep looks for the exact hit each
            # decision): the route to the aim point at POP_APPROACH_SPEED, no jump, forecast up to the estimated jump reply; priced
            # by the pop's promise (the route from the estimated landing up there to the gem + POP_PROMISE_U + the approach time),
            # so that heading for the edge at speed beats walking round while no exact hit is in the spread yet
            lroute = cached_route(route.terrain, far)   # the route key must match the steering target (compiled controller)
            d = aim[:2] - p[:2]; n_ = float(np.linalg.norm(d)); u = d / n_ if n_ > 1e-6 else np.array([0., 1.])
            land = np.array([aim[0] + u[0] * POP_LANDING_AHEAD_U, aim[1] + u[1] * POP_LANDING_AHEAD_U, float(gem[2])])
            rest = float(np.linalg.norm(gem[:2] - land[:2]))   # the flight steers at the gem: the straight line (the walking route from a
                                                               # point off the ledge went round and lost the approach to accelerate_then_brake)
            # the approach is the best-guess pop itself (the jump at the descent estimate for POP_APPROACH_SPEED): it stays a valid
            # plan through the jump window (a no-jump run was rejected 4 u before the step and the staging took over, bed 18/26)
            # while the sweep looks for the exact hit each decision; when none comes the best guess is what gets executed
            ahead = float(np.linalg.norm(aim[:2] - p[:2]))
            ja = descent_reply(ahead, speed, lip['rise'], POP_APPROACH_SPEED); ja = 0 if ja is None or ja < 0 else int(ja)
            n_app = min(steps, int(round(arrival_time(ahead, speed, POP_APPROACH_SPEED) / DECISION_S)) + POP_FLIGHT_REPLIES)
            n_app = max(n_app, ja + 2, 4)
            if POP_APPROACH_JUMP:
                spec = EdgePopProposal(lroute, far, n_app, ctx, POP_APPROACH_SPEED, ja if not airborne else -1)
            else:
                # the no-jump run up to the estimated jump reply (bed v2: 23/28 vs 7/28 with the best-guess jump, which executed
                # a miss whenever the sweep had not hit yet)
                spec = EdgePopProposal(lroute, far, max(4, min(steps, est)), ctx, POP_APPROACH_SPEED, -1)
            spec.box = [float(aim[0]) - 1., float(aim[1]) - 1., float(aim[2]) - .6, float(aim[0]) + 1., float(aim[1]) + 1., float(aim[2]) + 2.5]
            spec._edge_lip_z = float(lp[2]); spec._edge_lip = [float(x) for x in lp]
            spec._pop_promise = rest + POP_PROMISE_U + max(ja, 1) * DECISION_S * 8.   # the time to the jump (the flight is in POP_PROMISE_U;
                                                                                    # charging the whole forecast lost it to wall plans, bed v4-v6)
            spec._pop_aim = (lkey, [float(x) for x in aim])
            out.append(('edge_hit_%d_pop_approach_%g_%g' % (k, aim[0], aim[1]), spec))
        lroute = cached_route(route.terrain, far)   # the route key must match the steering target (compiled controller)
        box = [float(aim[0]) - 1., float(aim[1]) - 1., float(aim[2]) - .6, float(aim[0]) + 1., float(aim[1]) + 1., float(aim[2]) + 2.5]
        ahead = float(np.linalg.norm(aim[:2] - p[:2]))
        for cap in sorted(caps, key=lambda c: abs(c - 13.5)):   # the likeliest caps first: a deadline trims the fringe
            delays = sorted(descent_replies(ahead, speed, lip['rise'], cap, around), key=lambda j: abs(j - est))
            if no_jump:
                delays.append(-1)
            for j in delays:
                if j >= steps:
                    continue
                n = steps if j < 0 else min(steps, int(round(arrival_time(ahead, speed, cap) / DECISION_S)) + POP_FLIGHT_REPLIES)
                spec = EdgePopProposal(lroute, far, max(n, 12), ctx, cap, j)
                spec.box = box; spec._edge_lip_z = float(lp[2]); spec._edge_lip = [float(x) for x in lp]; spec._pop_aim = (lkey, [float(x) for x in aim])
                out.append(('edge_hit_%d_pop_%g_%g_%g_%d' % (k, aim[0], aim[1], cap, j), spec))
    return out


def aim_points(lip, p, v, offsets, extent=None, target=None):
    """Aim points on the edge: the segment point nearest the line to the target (lip['point']) and, moving, the point where the
    marble's heading meets the edge; each with `offsets` along the edge, kept POP_END_MARGIN_U inside the edge (`extent`: the whole
    straight edge, else the mesh segment), 0.25 u snapped, deduplicated, the base points first."""
    lp = np.asarray(lip['point'], dtype=float); seg = lip.get('seg') if extent is None else extent
    tangent = np.asarray(lip.get('tangent') or [0., 0., 0.], dtype=float)
    # the lanes are laid out from the TARGET: its projection onto the edge line (clamped into the edge) and offsets along the edge
    # from there (the pop's flight drifts about 1.3 u toward the target, so the lane must end within ~1.5 u of it); the marble
    # approaches its lane from wherever it is (the sticky aim keeps the lane across decisions). This is the live controller's lane
    # set (TreasureBox: x 87.4-88.0 for the gem at x 89), derived from the edge and the target, nothing map-specific.
    bases = []
    if seg is not None and float(np.linalg.norm(tangent[:2])) > .5 and target is not None:
        a = np.asarray(seg[0], dtype=float); b = np.asarray(seg[1], dtype=float); ab = b - a; L = float(np.linalg.norm(ab))
        if L > 1e-6:
            s_t = float((np.asarray(target[:2], dtype=float) - a[:2]) @ ab[:2]) / (L * L)
            lo, hi = POP_END_MARGIN_U / L, 1. - POP_END_MARGIN_U / L
            if lo <= hi:
                bases.append(a + ab * float(min(max(s_t, lo), hi)))
    bases.append(lp)
    if seg is not None and POP_HEADING_POINT and False:
        a = np.asarray(seg[0], dtype=float); b = np.asarray(seg[1], dtype=float); ab = b - a
        sp = float(np.linalg.norm(v[:2]))
        if sp > 2. and float(np.linalg.norm(ab[:2])) > 1e-6:
            # p + t h = a + s ab (horizontal): solve the 2x2 system
            h = v[:2] / sp; m = np.array([[h[0], -ab[0]], [h[1], -ab[1]]])
            det = float(np.linalg.det(m))
            if abs(det) > 1e-6:
                t, s_ = np.linalg.solve(m, a[:2] - p[:2])
                if t > 0.:
                    L = float(np.linalg.norm(ab)); s_ = float(np.clip(s_, POP_END_MARGIN_U / L, 1. - POP_END_MARGIN_U / L))
                    bases.append(a + ab * s_)
    out = []
    for off in sorted(offsets, key=abs):
        for base in bases:
            q = base + tangent * off
            if seg is not None:
                a = np.asarray(seg[0], dtype=float); b = np.asarray(seg[1], dtype=float); ab = b - a; L = float(np.linalg.norm(ab))
                if L > 1e-6:
                    lo, hi = POP_END_MARGIN_U / L, 1. - POP_END_MARGIN_U / L
                    if lo > hi:
                        continue   # an edge shorter than twice the margin
                    s_ = float((q - a) @ ab) / (L * L)
                    if off == 0.:
                        s_ = min(max(s_, lo), hi)   # a base point is kept: moved inside the margin
                    elif s_ < lo or s_ > hi:
                        continue
                    q = a + ab * s_
            q = np.round(q * 4.) / 4.   # 0.25 u snap: the same aim gives the same route key across decisions
            if not any(np.linalg.norm(q - o) < .1 for o in out):
                out.append(q)
    return out

