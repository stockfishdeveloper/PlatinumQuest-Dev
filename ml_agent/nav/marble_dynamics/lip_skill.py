"""Lip skill (2026-11-03): the standalone edge-hit skill as a planner candidate family.

The skill (tools/edge_skill: a goal-conditioned proposer plus a contact-conditioned feedback controller, verified
416/416 on 1,044 sealed native starts, see dynamics_reports/edge_skill_20261103/REPORT.md) climbs onto a ledge by
clipping its lip with a timed jump, lands on the platform behind the lip and hands the marble over at a requested exit
velocity, without Blast or powerups. Here it is offered to the planner as an ordinary candidate: the skill solves from
the actual observed state on its own private simulator, and the checked reply list it returns (up to the supported
arrival on the platform) is forecast again by the production simulator with the real items, followed by the ordinary
route-following continuation (LandingContinuation), assessed and ranked by the usual gates and costs. Nothing here
names a map: lips come from the collision mesh (edge_hits.mesh_lips), the platform behind a lip is measured by raycasts,
and the skill declines what it cannot execute.

Why a candidate family and not a committed feedback executor: every accepted native trial of the sealed suites replayed
its forecast exactly (per-reply error 0.0), so the open-loop reply list is the executed plan; the planner's own
retained-plan revalidation and its 0.05 u prediction gate replace the laboratory's feedback step, and a cleared plan
simply solves again from the new state (the skill accepts airborne, jump-committed starts).
"""
import copy
import math
import time

import numpy as np

RADIUS = .189750001
LIP_SKILL = True            # code default (matched control: --no-lip-skill -> engine.lip_skill False)
STRICT = False              # raise skill errors instead of counting them (set during tests)
MAX_LIPS = 2                # lips solved per search, nearest first
BUDGET = 48                 # proposer candidates when little search time remains (the sealed evaluation's budget)
BUDGET_TIERS = ((.4, 192), (.2, 96))   # (remaining search budget s, candidates): live states need deeper search than the laboratory's
MIN_REMAINING_S = .06       # skip the skill when less of the search budget remains
CAPTURE_DIR = ''            # when set (--set lip_skill.CAPTURE_DIR=...), every live solve's case and state are appended there (retraining data)
RISE_MIN_U = .35            # lip height above the approach floor: the trained envelope
RISE_MAX_U = 1.5
TARGET_RISE_U = .3          # the target must be at least this much above the marble's floor (a climb)
AHEAD_MIN_U = 1.5
AHEAD_MAX_U = 16.
LATERAL_MAX_U = 4.
LIP_MIN_LEN_U = 2.
APPROACH_COS = .6           # the marble's heading (when moving) and the target direction within ~53 deg of the lip normal: the trained approach angles
AIR_MAX_RISE_U = 1.6        # an airborne start only from an ordinary jump: this high above the floor at most ...
AIR_MAX_VZ = 8.             # ... and this vertical speed at most (falls from above are outside the envelope)
WINDOW_U = 10.              # the cell's lip window centred near the approach line (trained lateral range)
FLOOR_TOL_U = .06           # the approach floor must be level within this (flat run-up)
DECLINE_DECISIONS = 16      # a declined lip is not solved again for this many decisions ...
DECLINE_MOVE_U = 1.5        # ... unless the marble moved this far
EXTRA_REPLIES = 8           # at least this much route following after the handoff
EXIT_WINDOW_TICKS = 8       # the skill's own contract: supported arrival at the goal held this long (the planner judges the rest)
HORIZON_REPLIES = 64        # the skill forecast's length for live solves (4.1 s; the laboratory used 128)
WARM_LIPS = 8               # accepted parameters remembered per lip (warm start of the next solve)
SKILL_TICK_STEPS = .04      # virtual-clock charge per simulated skill tick, in production physics steps (~4 us vs 100 us)
TIME_RATE_U_S = 8.          # a longer forecast without a pickup is charged like the powerup rule (search.py 'use')

_ENGINES = {}
_PROPOSER = []
_MODEL = 'lip_skill_proposer_v6a.pt'


def model_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[2] / 'physics' / _MODEL


def skill_engine(engine):
    """A private physics engine on the same mesh with the skill's compiled controller (the production simulator and its
    bound clearance/pool are untouched)."""
    key = str(getattr(engine.mesh, 'path', id(engine.mesh)))
    e = _ENGINES.get(key)
    if e is None:
        from nav.collision.mesh import CollisionMap
        from nav.marble_dynamics.physics import GeneralPhysics
        from tools.edge_skill import compiled
        compiled.build()
        e = GeneralPhysics(CollisionMap(engine.mesh.path))
        compiled.attach(e)
        e.bridge_physics_ms = 16; e.planning_deadline = None
        _ENGINES[key] = e
    return e


def proposer():
    if not _PROPOSER:
        from tools.edge_skill.learn import Proposer
        _PROPOSER.append(Proposer(model_path()))
    return _PROPOSER[0]


def _rotation(yaw):
    return np.array([[math.cos(yaw), -math.sin(yaw), 0.], [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])


def _floor(mesh, p, above=.4, below=6.):
    return mesh.raycast(np.asarray(p, dtype=float) + [0., 0., above], np.asarray(p, dtype=float) - [0., 0., below], cull_backfaces=True)


def platform_geometry(mesh, centre, forward, tangent, lip_z):
    """Depth and width of the level platform behind the lip (None when there is none): the real-fixture probes."""
    def floor(x, y=0.):
        return _floor(mesh, centre + forward * x + tangent * y)
    inside = floor(.45); outside = floor(-.45)
    if inside is None or abs(float(inside['point'][2]) - lip_z) > .025: return None
    if outside is not None and float(outside['point'][2]) > lip_z - .25: return None
    depth = None
    for possible in (3.5, 5., 7., 9.):
        hits = [floor(x, y) for x in np.linspace(.35, possible - .25, 6) for y in (-.6, 0., .6)]
        if all(h is not None and abs(float(h['point'][2]) - lip_z) < .025 for h in hits): depth = possible
        else: break
    if depth is None: return None
    width = 1.8
    for possible in (2.5, 3.5, 5.):
        hits = [floor(x, y) for x in (.4, depth / 2, depth - .3) for y in (-possible / 2 + .1, possible / 2 - .1)]
        if all(h is not None and abs(float(h['point'][2]) - lip_z) < .025 for h in hits): width = possible
        else: break
    return dict(depth=float(depth), width=float(width))


def opportunities(engine, dynamics, gem, max_lips=MAX_LIPS):
    """Lip cells on the way to a higher target, nearest first: dicts(cell, case fields) ready for the proposer."""
    from nav.marble_dynamics.edge_hits import mesh_lips
    st = dynamics['native']['state']; p = np.asarray(st['position'], dtype=float); v = np.asarray(st['velocity'], dtype=float)
    mesh = engine.mesh; t = np.asarray(gem[:3], dtype=float)
    under = _floor(mesh, p, above=.05, below=4.)
    if under is None: return []
    floor_z = float(under['point'][2])
    if float(t[2]) < floor_z + RADIUS + TARGET_RISE_U: return []
    grounded = int(st.get('ground_ms', 0)) > 0 and p[2] - RADIUS - floor_z < .08
    if not grounded and (p[2] - RADIUS - floor_z > AIR_MAX_RISE_U or abs(float(v[2])) > AIR_MAX_VZ): return []
    speed = float(np.linalg.norm(v[:2])); heading = v[:2] / speed if speed > 2. else None
    d = t[:2] - p[:2]; L = float(np.linalg.norm(d))
    if L < 1e-6: return []
    u = d / L; left = np.array([-u[1], u[0]])
    segs, normals = mesh_lips(engine)
    if not len(segs): return []
    out = []; order = []; funnel = _stats(engine).setdefault('funnel', {})
    def drop(key):
        funnel[key] = funnel.get(key, 0) + 1
    for i, (seg, n) in enumerate(zip(segs, normals)):
        if n[2] < .995 or abs(seg[0, 2] - seg[1, 2]) > .01: drop('tilted'); continue
        ab = seg[1] - seg[0]; length = float(np.linalg.norm(ab))
        if length < LIP_MIN_LEN_U: drop('short'); continue
        lip_z = float(seg[0, 2]); rise = lip_z - floor_z
        if not RISE_MIN_U <= rise <= RISE_MAX_U: drop('rise'); continue
        if lip_z > float(t[2]) + .3: drop('above_target'); continue
        # the segment point closest to the approach line
        abh = ab[:2]; lat_a = float((seg[0, :2] - p[:2]) @ left); lat_ab = float(abh @ left)
        s = float(np.clip(-lat_a / lat_ab, 0., 1.)) if abs(lat_ab) > 1e-9 else .5
        q = seg[0] + ab * s
        ahead = float((q[:2] - p[:2]) @ u); lateral = abs(float((q[:2] - p[:2]) @ left))
        if not AHEAD_MIN_U <= ahead <= min(AHEAD_MAX_U, L + 2.): drop('ahead'); continue
        if lateral > LATERAL_MAX_U: drop('lateral'); continue
        tangent = ab / length; side = np.array([tangent[1], -tangent[0], 0.])
        forward = side if float((q[:2] - p[:2]) @ side[:2]) > 0 else -side   # into the platform, away from the marble
        if float(forward[:2] @ u) < APPROACH_COS: drop('target_angle'); continue       # the target is not beyond this lip
        if heading is not None and float(forward[:2] @ heading) < APPROACH_COS: drop('heading'); continue   # moving along the lip, not into it
        order.append((ahead, i, q, forward, tangent, length, s, lip_z))
    order.sort(key=lambda x: x[0])
    for ahead, i, q, forward, tangent, length, s, lip_z in order:
        # a lip window of WINDOW_U centred near the approach line, inside the segment
        half = min(WINDOW_U, length) / 2.
        c = s * length; c = min(max(c, half), length - half)
        centre = segs[i][0] + tangent * c
        geo = platform_geometry(mesh, centre, forward, tangent, lip_z)
        if geo is None: drop('platform'); continue
        # a level run-up from the marble to the lip foot (the trained approach floor)
        foot = centre - forward * .3; ok = True
        for f in np.linspace(.08, 1., 8):
            h = _floor(mesh, p * (1 - f) + foot * f, above=.4, below=4.)
            if h is None or abs(float(h['point'][2]) - floor_z) > FLOOR_TOL_U: ok = False; break
        if not ok: drop('runup'); continue
        yaw = math.atan2(forward[1], forward[0]); rot = _rotation(yaw)
        origin = np.array([centre[0], centre[1], floor_z])
        height = lip_z - floor_z; depth = geo['depth']; pw = geo['width']
        cell = dict(id='lip_%d' % i, index=int(i), map=str(getattr(mesh, 'path', '')), split='live', origin=origin.tolist(),
                    yaw_deg=math.degrees(yaw), lip_z=height, lip_depth=depth, width=float(2 * half),
                    landing_z=height, landing_x=0., landing_end=depth, landing_center=[depth / 2, 0.],
                    landing_size=[depth, pw], landing_yaw=0., family='high', friction=1., restitution=1.)
        g = (t - origin) @ rot                                            # the target in the cell frame
        on_platform = (abs(g[2] - height - RADIUS) < .6 and .3 < g[0] < depth - .3 and abs(g[1]) < pw / 2 - .3)
        speed = 5. if depth >= 5 else 3.
        if on_platform:
            goal = [float(np.clip(g[0], .6, depth - .5)), float(np.clip(g[1], -pw / 2 + .5, pw / 2 - .5)), height + RADIUS]
            direction = np.array([1., 0.])
        else:
            direction = g[:2] / max(float(np.linalg.norm(g[:2])), 1e-6)
            ang = float(np.clip(math.atan2(direction[1], direction[0]), -math.pi / 4, math.pi / 4))
            direction = np.array([math.cos(ang), math.sin(ang)])
            gx = min(2., depth - 1.); goal = [gx, float(np.clip(direction[1] * gx, -pw / 2 + .5, pw / 2 - .5)), height + RADIUS]
        cell['goal_local'] = goal; cell['exit_speed'] = speed
        pl = (p - origin) @ rot; vl = v @ rot
        kind = 'ground' if grounded else ('air_rising' if vl[2] > .6 else 'air_descending')
        case = dict(id='live_' + cell['id'], cell=cell, position=pl.tolist(), velocity=vl.tolist(), spin=(np.asarray(st['spin'], dtype=float) @ rot).tolist(),
                    goal=goal, exit_velocity=(direction * speed).tolist(), goal_radius=.9, airborne=not grounded,
                    start_kind=kind, start_floor=0., seed=int(i) * 7919 + 11, split='live', required_contact_mode=None, stable_inputs=True,
                    exit_window_ticks=EXIT_WINDOW_TICKS, require_backup=False, horizon_replies=HORIZON_REPLIES)
        out.append(dict(case=case, lip=[float(x) for x in centre], rise=height, ahead=ahead, index=int(i)))
        if len(out) >= max_lips: break
    return out


def solve_live(engine, opp, d, budget, di):
    """The proposer's candidates plus the lip's last accepted parameters, solved and audited on the private simulator."""
    from tools.edge_skill.execution import solve
    case = opp['case']; se = skill_engine(engine); pr = proposer()
    warm = getattr(engine, 'lip_skill_warm', None)
    if warm is None: warm = engine.lip_skill_warm = {}
    initial = list(warm.get(opp['index'], [])) + pr.params(case, d)
    candidate = solve(se, case, d, budget=budget, seed=case['seed'] + di, initial_params=initial, max_alternatives=1, accept_early=True)
    if candidate['result']['success']:
        warm[opp['index']] = [candidate['params']]
        while len(warm) > WARM_LIPS: warm.pop(next(iter(warm)))
    return candidate


def _stats(engine):
    s = getattr(engine, 'lip_skill_stats', None)
    if s is None: s = engine.lip_skill_stats = {}
    return s


def _count(engine, key, n=1):
    s = _stats(engine); s[key] = s.get(key, 0) + n


def skill_pairs(engine, route, dynamics, gem, steps, target_item=None):
    """(info, rollout) pairs of the skill's checked plans, forecast by the production simulator; [] when it declines."""
    from nav.marble_dynamics.bridge import predict_replies
    from nav.marble_dynamics.search import assess, LandingContinuation
    st = dynamics['native']['state']
    if any(e[0] for e in st.get('effects', [])) or (dynamics.get('inventory') or {}).get('mega_active'):
        return []                     # the skill is unpowered: active effects change the physics it was checked on
    deadline = getattr(engine, 'planning_deadline', None)
    remaining = (deadline - time.perf_counter()) if deadline else 1.
    if remaining < MIN_REMAINING_S:
        _count(engine, 'skipped_budget'); return []
    try:
        found = opportunities(engine, dynamics, gem)
    except Exception as exc:
        if STRICT: raise
        _count(engine, 'error'); engine.lip_skill_error = repr(exc); return []
    if not found: return []
    _count(engine, 'offered', len(found))
    di = int(getattr(engine, 'decision_index', 0)); p = np.asarray(st['position'], dtype=float)
    recent = [r for r in getattr(engine, 'lip_skill_recent', []) if di - r[0] < DECLINE_DECISIONS]
    engine.lip_skill_recent = recent
    pairs = []; hook = getattr(engine, 'native_step_hook', None); virtual = getattr(engine, 'virtual_step_cost_s', None) is not None
    for k, opp in enumerate(found):
        if any(r[2] == opp['index'] and np.linalg.norm(p - r[1]) < DECLINE_MOVE_U for r in recent):
            _count(engine, 'skipped_recent'); continue
        remaining = (deadline - time.perf_counter()) if deadline else 1.
        if remaining < MIN_REMAINING_S:
            _count(engine, 'skipped_budget'); break
        budget = next((b for limit, b in BUDGET_TIERS if remaining >= limit), BUDGET)
        t0 = time.perf_counter()
        try:
            candidate = solve_live(engine, opp, copy.copy(dynamics), budget, di)
        except Exception as exc:
            if STRICT: raise
            _count(engine, 'error'); engine.lip_skill_error = repr(exc); continue
        wall = time.perf_counter() - t0
        if hook is not None and virtual:
            hook(int(candidate['stats'].get('candidates', budget) * HORIZON_REPLIES * 4 * SKILL_TICK_STEPS))   # charge the skill's work to the virtual clock
        if CAPTURE_DIR:
            import os, pickle
            os.makedirs(CAPTURE_DIR, exist_ok=True)
            with open(os.path.join(CAPTURE_DIR, 'solves_%d.pkl' % os.getpid()), 'ab') as f:
                pickle.dump(dict(case=copy.deepcopy(opp['case']), d=copy.deepcopy(dynamics), gem=[float(x) for x in gem[:3]], success=bool(candidate['result']['success']),
                                 params=candidate['params'], budget=budget, map=str(getattr(engine.mesh, 'path', ''))), f)
        if not candidate['result']['success']:
            _count(engine, 'declined'); recent.append((di, p.copy(), opp['index'])); continue
        _count(engine, 'solved')
        result = candidate['result']; reach_reply = int(math.ceil((result['reach_tick'] or 0) / 4.)) + 1   # hand over on the platform at the goal
        prefix = [list(a) for a in candidate['rollout']['proposed'][:reach_reply]]
        if not prefix: continue
        extra = max(EXTRA_REPLIES, steps - len(prefix))
        proposal = LandingContinuation(prefix, route, gem, extra)
        label = 'lip_skill_%d' % k
        rollout = predict_replies(engine, dynamics, proposal, stop_on_ambiguity=True)
        info = assess(engine, route, dynamics, gem, rollout, label, target_item)
        info.update(evaluated_replies=len(rollout['controls']), requested_replies=rollout['requested_replies'],
                    stopped_on_ambiguity=rollout['stopped_on_ambiguity'], proposal_replies=len(prefix))
        if rollout['requested_replies'] > steps and info['rejected'] is None and info['possible_pickup_interval'] is None:
            penalty = (rollout['requested_replies'] - steps) * .064 * float(getattr(engine, 'ballistic_time_rate', TIME_RATE_U_S) or TIME_RATE_U_S)
            info['cost'] += penalty; info['time_penalty_u'] = penalty
        info['lip_skill'] = dict(lip=opp['lip'], rise=round(opp['rise'], 3), ahead=round(opp['ahead'], 2), start_kind=opp['case']['start_kind'],
                                 reach_reply=reach_reply, lip_tick=result.get('lip_tick'), contact_mode=result.get('contact_mode'),
                                 candidates=candidate['stats'].get('candidates'), planning_ms=round(wall * 1000, 1), budget=budget,
                                 skill_score=round(float(result['score']), 3))
        if info['rejected'] is None: _count(engine, 'admitted')
        else: _count(engine, 'rejected_' + str(info['rejected']).split(':')[0])
        pairs.append((info, rollout))
    engine.lip_skill_recent = recent
    return pairs
