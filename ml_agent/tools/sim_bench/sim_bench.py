"""Offline closed-loop benchmark of the physics planner (no game).

From recorded journal states (the decision right after a pickup = the start of a leg), the real ContinuousController
plays against the planner's own engine port: each 64 ms reply is advanced with predict_replies, gems are collected by the
native item query (possible_pickup), no new gem group ever spawns. An episode ends after --pickups pickups, a fall (outside
the InBounds volume), a prediction failure, or --cap-s seconds. Deterministic: the search deadline runs on a virtual clock
that advances --step-cost-ms per 16 ms physics step (calibrate with --real-clock).

usage (repo root, PYTHONPATH=ml_agent):
  python sim_bench.py --pool planner_v23_start2_20261015 --map Sprawl_Hunt --per-map 8 --variant base --out x.jsonl
  variant switches: --align W (direction-aware end value, prototype), --set module.NAME=value (e.g. gem_demo.TARGET_SPEED_CAP=13.5)
"""
import sys, os, json, gzip, copy, time, math, argparse
import numpy as np
from pathlib import Path
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument('--pool', action='append', required=True)
ap.add_argument('--map', required=True)
ap.add_argument('--per-map', type=int, default=8)
ap.add_argument('--offset', type=int, default=0)
ap.add_argument('--start-seed', default='7347019')   # 2026-10-25: the recorded round (seed) the starts come from
ap.add_argument('--pickups', type=int, default=2)
ap.add_argument('--endgame', type=float, default=0.)   # 2026-10-17: one episode per recorded round, from the first decision with
                                                        # at most this many seconds left until the round ends; score = points
ap.add_argument('--next-group', action='store_true')   # 2026-10-19: with --whole-group, the group that spawned next in the recorded round appears
                                                       # when this one is cleared (the game's spawn does not depend on the marble); the
                                                       # episode ends at the new group's first pickup (t_s then includes the trip)
ap.add_argument('--whole-group', action='store_true')   # 2026-10-17: run until every gem visible at the start is collected (the
                                                        # current group): a few pickups reward greedy orders that leave a far gem last
ap.add_argument('--cap-s', type=float, default=20.)
ap.add_argument('--variant', default='base')
ap.add_argument('--align', type=float, default=0.)
ap.add_argument('--align-accel', type=float, default=14.)
ap.add_argument('--set', action='append', default=[])
ap.add_argument('--out', required=True)
ap.add_argument('--step-cost-ms', type=float, default=0.6)
ap.add_argument('--real-clock', action='store_true')
ap.add_argument('--budget', type=float, default=2.)
ap.add_argument('--max-episodes', type=int, default=0)
ap.add_argument('--slice', default='')   # 'A:B': only the chosen starts A..B-1 (split one map's episodes over processes, same starts)
ap.add_argument('--far-min', type=float, default=0.)
ap.add_argument('--far-max', type=float, default=1e9)
ap.add_argument('--fall-legs', action='store_true')   # only legs whose recorded run fell before the next pickup
ap.add_argument('--stuck-legs', action='store_true')
ap.add_argument('--pflag', action='append', default=[])
ap.add_argument('--trace', action='store_true')
ap.add_argument('--lead', type=int, default=0)
ap.add_argument('--edge-gems', type=float, default=0.)   # >0: starts lead decisions before pickups of gems within this walk-grid distance of a drop   # start this many decisions before the selected leg start (through its pickup)
ap.add_argument('--at-decision', type=int, default=0)   # 2026-10-25: start exactly at this recorded decision (mid-leg too)
ap.add_argument('--only-decision', type=int, default=0)   # extra planner launch flags, e.g. --pflag=--fallback-cooldown=0  # only legs whose recorded first pickup took > 8 s for a gem < 20 u away
ap.add_argument('--normal-priority', action='store_true')
ap.add_argument('--rise-min', type=float, default=None)   # 2026-10-20: only starts whose first pickup is this much higher than the start
ap.add_argument('--held', type=int, default=-1)   # only starts where the marble holds this powerup (2 = Super Speed)
ap.add_argument('--module', action='append', default=[])
ap.add_argument('--search-workers', type=int, default=0)   # 2026-10-19: candidates evaluated by this many worker processes
                                                          # (nav/marble_dynamics/parallel_search.py; use with --real-clock)
ap.add_argument('--native', action='store_true')   # compiled forecasts (nav/native; the game's planner since 2026-10-16 v41)
ap.add_argument('--native-step-us', type=float, default=0.)   # virtual clock per physics step with --native (0 = NATIVE_STEP_US)
o = ap.parse_args()
if not o.normal_priority:
    import ctypes   # idle priority: never take CPU from running game pools (their search budget is wall clock)
    _k = ctypes.windll.kernel32; _k.GetCurrentProcess.restype = ctypes.c_void_p
    _k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    if not _k.SetPriorityClass(_k.GetCurrentProcess(), 0x40):
        raise SystemExit('could not set idle priority')

VC = [1000.0]; STEPS = [0]
if not o.real_clock:
    time.perf_counter = lambda: VC[0]

ap2 = argparse.ArgumentParser(add_help=False)
ap2.add_argument('--module', action='append', default=[])   # name=path: load a modified copy of a planner module (e.g. search=x.py)
for spec in ap2.parse_known_args()[0].module:
    import importlib.util
    _name, _path = spec.split('=', 1); _full = 'nav.marble_dynamics.' + _name
    _spec = importlib.util.spec_from_file_location(_full, _path); _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_full] = _mod; _spec.loader.exec_module(_mod)
import nav.marble_dynamics.reliability_eval as RE
import nav.marble_dynamics.gem_demo as GD
import nav.marble_dynamics.search as S
from nav.marble_dynamics.bridge import predict_replies, translated
from nav.marble_dynamics.navigator import project_observation
from nav.marble_dynamics.fast_nav import fully_outside_inbounds
from nav.protocol import RAW_GEMS, RAW_SCORE, RAW_GEMS_REMAINING, RAW_POW_ITEMS
from nav.routing import path_waypoint

ML = Path('ml_agent')
NATIVE_STEP_US = 100.   # --native: virtual clock per physics step. 2026-10-16 calibration under pool load (open loop, 250
                        # decisions each): Cube 61, Sprawl 108, Bowl 138 us per step (Python work per forecast dominates)
FLAGS = ('--random-starts --only-branch physics_plan --full-round --horizon 24 --commit 4 --reuse-plan --geometry-route '
         '--resource-acquisition --recharge-continuation --landing-search --proposal-samples 20 --query-cache --clearance-cache '
         '--global-goals --physics-frame-ms 16 --fallback-cooldown 2 --search-budget-s %g --pickup-landing-continuation 40 '
         '--component-crossing --ballistic-landing --use-horizon 48 --use-margin-s 1').split()
CACHE = r'C:\Users\doug\AppData\Local\Temp\pq_checks\fast_nav_static_cache_20261013'
DLL = r'C:\Users\doug\AppData\Local\Temp\pq_checks\compiled_sweep_20261012\sweep_kernel.dll'


class _Got(Exception):
    pass


def planner_args(name):
    import argparse as _ap
    orig = _ap.ArgumentParser.parse_args

    def cap(self, args=None, namespace=None):
        raise _Got(orig(self, args, namespace))
    _ap.ArgumentParser.parse_args = cap
    saved = sys.argv
    sys.argv = ['reliability_eval', '--maps', name, '--seeds', '7347019'] + [f % o.budget if '%' in f else f for f in FLAGS] + \
        ['--cache-dir', CACHE, '--sweep-dll', DLL, '--output', 'ml_agent/dynamics_reports/_sim_unused'] +         [x for f in o.pflag for x in f.split('=', 1)]
    try:
        RE.main()
    except _Got as g:
        a = g.args[0]
    finally:
        _ap.ArgumentParser.parse_args = orig; sys.argv = saved
    return a


def setup(a, name):
    """Engine/terrain exactly as reliability_eval.main() builds them for one map."""
    from nav.terrain import TerrainMap
    from nav.marble_dynamics.static_cache import cached_terrain, cached_walking, load_queries
    from nav.collision.experimental_sweep import CachedCompiledSweepMap
    from nav.marble_dynamics.physics import GeneralPhysics
    terrain = cached_terrain(TerrainMap.resolve(name), a.cache_dir)
    mesh = CachedCompiledSweepMap(ML / 'collision_maps' / (name + '.npz'), a.sweep_dll); load_queries(mesh, a.sweep_dll, a.cache_dir)
    engine = GeneralPhysics(mesh)
    engine.bridge_physics_ms = a.physics_frame_ms
    engine.terminal_stop_replies = a.terminal_stop_replies; engine.terminal_stop_scope = a.terminal_stop_scope
    engine.landing_replies = getattr(a, 'landing_continuation', 0); engine.pickup_landing_replies = getattr(a, 'pickup_landing_continuation', 0)
    engine.use_horizon = getattr(a, 'use_horizon', 0); engine.use_first = getattr(a, 'use_first', False)
    engine.blast_value_u = float(getattr(a, 'blast_value_u', 10.)); engine.blast_floor_u = float(getattr(a, 'blast_floor_u', 0.))
    GD.END_VELOCITY_WEIGHT = float(getattr(a, 'end_velocity_weight', .12))
    GD.THROTTLE_GAIN = float(getattr(a, 'throttle_gain', 4.)); GD.THROTTLE_CAP = float(getattr(a, 'throttle_cap', 1.))
    GD.BRAKE_DECEL = float(getattr(a, 'brake_decel', 7.)); GD.TARGET_SPEED_CAP = float(getattr(a, 'speed_cap', 11.))
    S.ROUTE_SPEED_LAW = not getattr(a, 'waypoint_brake_law', False); S.PICKUP_MIN_SPEED = float(getattr(a, 'pickup_min_speed', 6.))
    S.FAST_PROPOSALS = not getattr(a, 'legacy_fast_proposals', False)
    S.THROUGH_GEM = not getattr(a, 'no_through_gem', False); S.BRAKING_BACKUP = not getattr(a, 'no_braking_backup', False)
    # as reliability_eval.main() (v70: corner plans on by default via RE.CORNER_DEFAULT; before 18:55 10-17 the sim left them off)
    S.CORNER_PROPOSALS = bool(getattr(RE, 'CORNER_DEFAULT', False)) or bool(getattr(a, 'corner_proposals', False))
    engine.detour_blast_ratio = float(getattr(a, 'detour_blast_ratio', 1.5) or 0.) or math.inf
    engine.use_margin_u = float(getattr(a, 'use_margin_s', 0.) or 0.) * float(getattr(engine, 'ballistic_time_rate', 8.) or 8.)
    engine.interleave_candidates = getattr(a, 'interleave_candidates', False)
    engine.recheck_risky_plans = getattr(a, 'recheck_risky_plans', False)
    engine.mega_enabled = getattr(a, 'mega_proposals', False)   # 2026-10-20 Mega use proposals
    engine.pickup_stop = getattr(a, 'pickup_stop', False)
    engine.edge_hits = getattr(a, 'edge_hits', False)
    engine.wall_rebounds = getattr(a, 'wall_rebounds', True)
    engine.pad_hops = getattr(a, 'pad_hops', False)
    engine.resource_cap = getattr(a, 'resource_cap', True)
    engine.time_price = getattr(a, 'time_price', False)
    from nav.marble_dynamics import gem_demo as _gd; _gd.set_time_price(engine.time_price)
    if engine.time_price: engine.ballistic_time_rate = _gd.TIME_PRICE_U_S
    engine.pad_arrival = getattr(a, 'pad_arrival', True)
    a.geometry_terrain = cached_walking(terrain, mesh, a.sweep_dll, a.cache_dir, memo=a.clearance_cache)
    engine.ballistic_landing = getattr(a, 'ballistic_landing', False); engine.ballistic_reserve_s = getattr(a, 'ballistic_reserve_s', 0.)
    engine.opportunistic_powerups = getattr(a, 'opportunistic_powerups', 0.)
    engine.mission_bounds = None
    try:
        from nav.marble_dynamics.fast_nav import mission_inbounds
        engine.mission_bounds = mission_inbounds(json.loads((ML / 'collision_maps' / (name + '.json')).read_text()))
    except (ValueError, OSError):
        engine.mission_bounds = None
    engine.crossings = None
    if getattr(a, 'component_crossing', False):
        from nav.marble_dynamics.crossings import Crossings
        engine.crossings = Crossings(a.geometry_terrain, mesh=engine.mesh, climbs=getattr(a, 'climb_crossings', False))
    real_step = engine.step

    def step(*args, **kw):
        STEPS[0] += 1; VC[0] += o.step_cost_ms / 1000.
        return real_step(*args, **kw)
    engine.step = step
    if o.native:
        # the compiled forecast never calls engine.step: it reports its physics steps through native_step_hook, and with
        # virtual_step_cost_s set it stops a forecast at the virtual deadline itself (max_steps)
        from nav.native import attach
        attach(engine); engine.native_sim.bind_clearance(a.geometry_terrain)
        cost = (o.native_step_us or NATIVE_STEP_US) / 1e6

        def hook(n):
            STEPS[0] += n
            if not o.real_clock:
                VC[0] += n * cost
        engine.native_step_hook = hook
        engine.virtual_step_cost_s = None if o.real_clock else cost
    return terrain, engine


# ---------------------------------------------------------------- variant: direction-aware end value (prototype v38)
_orig_cost = GD.candidate_cost


def end_loss(route_terrain, field, goal, state, vc, accel):
    """Distance-equivalent time lost reaching cruise speed vc along the route direction from this end velocity:
    0.5 * (vc - v_par)+ * |dv| / a with dv = (vc - v_par, -v_perp) (constant acceleration a toward the cruise velocity)."""
    p = np.asarray(state['position'], dtype=float); v = np.asarray(state['velocity'], dtype=float)[:2]
    if field is not None:
        wp, _ = path_waypoint(route_terrain, float(p[0]), float(p[1]), field, goal, mz=float(p[2]))
    else:
        wp = goal
    u = np.asarray(wp[:2], dtype=float) - p[:2]; n = float(np.linalg.norm(u))
    if n < 1e-6:
        return 0.
    u /= n; vpar = float(v @ u); vperp = float(np.linalg.norm(v - vpar * u))
    deficit = max(0., vc - vpar)
    return .5 * deficit * math.hypot(deficit, vperp) / accel


def aligned_cost(route, states, gem, routed, item, after=None):
    cost, close, pickup = _orig_cost(route, states, gem, routed, item, after)
    if pickup is not None and after is not None and item is not None:
        nroute, ngem, leg = after; j = min(len(states) - 1, GD.THROUGH_EVAL_REPLIES)
        if pickup <= j:
            s = states[j]
            cost += -GD.END_VELOCITY_WEIGHT * float(np.linalg.norm(s['velocity'])) + o.align * end_loss(
                nroute.terrain, nroute.field, np.asarray(ngem[:3], dtype=float), s, GD.TARGET_SPEED_CAP, o.align_accel)
        return cost, close, pickup
    if pickup is not None:
        return cost, close, pickup
    end = states[-1]
    cost += -GD.END_VELOCITY_WEIGHT * float(np.linalg.norm(end['velocity'])) + o.align * end_loss(
        route.terrain, route.field if routed else None, np.asarray(gem[:3], dtype=float), end, GD.TARGET_SPEED_CAP, o.align_accel)
    return cost, close, pickup


# ---------------------------------------------------------------- sim world
def resume(dynamics, rollout):
    """viability.resume_forecast without the ambiguity refusal (the sim accepts the engine's own choice)."""
    result = copy.deepcopy(dynamics)
    pending = list(dynamics['bridge_action']); move = copy.deepcopy(dynamics['native']['pending_move'])
    hold = int(dynamics['pow_yaw_hold']); aim = float(dynamics['pow_yaw'])
    for action in rollout['proposed']:
        move, _, hold, aim = translated(pending, hold, aim)
        pending = list(action)
    state = copy.deepcopy(rollout['states'][-1])
    result['native']['state'] = state
    result['native']['frame_clock_ms'] = int(dynamics['native']['frame_clock_ms']) + 64 * len(rollout['proposed'])
    result['native']['pending_move'] = move
    result['bridge_action'] = pending; result['pow_yaw_hold'] = hold; result['pow_yaw'] = aim
    for source, key in [('blast_context', 'blast'), ('inventory_context', 'inventory')]:
        if source in state:
            result[key] = copy.deepcopy(state[source])
    return result


class Msg:
    def __init__(self, obs, dynamics):
        self.obs = obs; self.dynamics = dynamics


def fam(label):
    label = label or '?'
    if label.startswith('A2'):
        return 'A2'
    if label == 'safe_hold':
        return 'hold'
    if label.startswith('through_steer'):
        return 'tsteer'
    if 'through_corner' in label:
        return 'corner'
    if label.startswith('route_corner'):
        return 'rcorner'
    if label.startswith('edge_hit'):
        return 'edge'     # 2026-10-20 edge-hit plans (nav/marble_dynamics/edge_hits.py)
    if label.startswith('bounce'):
        return 'bounce'   # 2026-10-19 wall-bounce plans (nav/marble_dynamics/wall_bounce.py)
    for k in ('through', 'blast', 'jump', 'use', 'steering', 'throttle', 'direct', 'route', 'frozen', 'crossing', 'landing', 'recharge'):
        if k in label:
            return k
    return 'other'


def episode(a, terrain, engine, rec, nxt=None):
    d = copy.deepcopy(rec['before']); raw0 = np.asarray(rec['raw_observation'], dtype=np.float32); p0 = raw0[:3].copy()
    t_base = 0; points_base = 0.; picks_base = 0; t_spawn = None   # --next-group: the observation is rebased when the next group appears
    ctrl = RE.ContinuousController(terrain, engine, a.checkpoint, a, 'physics_plan')
    if getattr(engine, 'parallel', None) is not None:
        engine.parallel.bind(dict(terrain=terrain, geometry=a.geometry_terrain, mesh=engine.mesh, sim=engine.native_sim,
                                  model=ctrl.nav.model, crossings=engine.crossings, contact_order=getattr(engine, 'contact_order', None)))
    raw = raw0.copy(); msg = Msg(raw, d)
    picks = []; points = 0.; removed = []; pw_removed = []; labels = Counter(); why = 'cap'; t = 0; cap = int(o.cap_s / .064)
    tick_v = []; drops = 0; drop_watch = None   # 2026-10-23: executed side impacts and ledge drops (search.side_impacts rule)
    hist = []; impact_by = Counter()   # (label family, predicted side impacts) of the chosen plans; executed impacts by the plan 2 decisions back
    target = sum(1 for it in d.get('items', []) if it.get('gem')) if o.whole_group else o.pickups
    if o.endgame:
        cap = max(1, int(float(raw0[31]) / 64.)); target = 10 ** 6   # run until the round's clock reaches zero
    speeds = []; zfloor = None; blasts = 0
    try:
        lo = engine.mesh.bounds_min if hasattr(engine.mesh, 'bounds_min') else None
    except Exception:
        lo = None
    while t < cap:
        remaining = max(1, int(float(raw[31]) / 64.))
        before = raw.copy()
        action, info = ctrl.action(msg, remaining)
        labels[fam(info.get('label'))] += 1
        hist.append((fam(info.get('label')), int(((info.get('cluster_penalty') or {}).get('side_impacts')) or 0)))
        if o.trace:
            st0 = d['native']['state']; et = getattr(terrain, 'edge_dist', None)
            _cx = getattr(getattr(engine, 'crossings', None), 'active', None)
            _ph = getattr(engine, 'last_crossing_phase', None)
            _lc = getattr(engine, 'last_crossing', None)
            if _lc and _lc.get('climb2'):   # 2026-10-25 climb skill: the staged climb, active or not
                _ph = '%s:%s(%s,%s)m%.2f/%.2f' % (_ph, _lc['kind'], round(_lc['launch'][0], 1), round(_lc['launch'][1], 1), float(_lc['needed_meter']),
                                                 float((d.get('blast') or {}).get('value', 0.)))
                _cx = None
            elif _lc and _ph:
                _ph = '%s:%s' % (_ph, 'gap' if _lc.get('gap_route') else ('wall' if _lc.get('wall') else ('climb' if _lc.get('climb') else 'old')))
            print('   t %5.2f %-26s reason=%-22s v=%5.2f pos=%s z=%.2f pk=%s ph=%s%s' % (t * .064, str(info.get('label'))[:26], str(info.get('reason'))[:22],
                  float(np.linalg.norm(np.asarray(st0['velocity'])[:2])), np.round(st0['position'][:2], 1).tolist(), st0['position'][2], info.get('possible_pickup_interval'),
                  _ph, '' if not _cx else ' gap=%s launch=%s' % (round(float(_cx.get('gap_u', 0)), 1), np.round(_cx['launch'][:2], 1).tolist())),
                  info.get('cluster_penalty') or '', flush=True)
        engine.planning_deadline = None
        try:
            roll = predict_replies(engine, d, [action])
        except ValueError as exc:
            why = 'sim_error:' + str(exc)[:50]; break
        if roll['failure']:
            why = 'sim_failure'; break
        blasts += sum(1 for c in roll.get('controls', []) for e in c.get('script_events', []) if e.get('kind') == 'blast_impulse')
        nd = resume(d, roll); st = nd['native']['state']
        _P, _V = S._tick_arrays(roll['native_states'])
        if _V is not None:
            seg = np.asarray(([tick_v[-1]] if tick_v else []) + _V[(1 if tick_v else 0):].tolist(), dtype=float)
            tick_v.extend(_V[(1 if tick_v else 0):].tolist())
            if len(seg) > 1:
                h_ = np.hypot(seg[:, 0], seg[:, 1]); hit_ = ((h_[:-1] - h_[1:]) > S.BUMP_TICK_DV) & (np.hypot(seg[1:, 0] - seg[:-1, 0], seg[1:, 1] - seg[:-1, 1]) > np.abs(seg[1:, 2] - seg[:-1, 2]))
                if hit_.any():
                    src = hist[-3] if len(hist) >= 3 else hist[0]
                    impact_by[src[0] + (' (predicted)' if src[1] else ' (unpredicted)')] += 1
        gained = 0.; keep = []
        for it in nd.get('items', []):
            # the world's pickups are the game's own per-tick test, whatever the planner's switch (10-16: the reply box
            # gave false pickups, so the sim before 10-16 09:40 rewarded plans that skim gems at speed)
            if it.get('gem') and GD.tick_pickup(roll['native_states'], it) is not None:
                gained += float(it.get('value') or 1.); removed.append(np.asarray(it['position'], dtype=float)); picks.append((t + 1) * .064)
            elif (not it.get('gem') and 1 <= int(it.get('powerup') or 0) <= 5 and int(st['held']) == 0
                  and GD.tick_pickup(roll['native_states'], it) is not None):
                # 2026-10-17: the world collects powerups too (an empty hand takes one it touches). Before, the sim never
                # did, so the planner's powerup detour (v58) chased a powerup 1 u away for the rest of the episode.
                st['held'] = int(it['powerup']); pw_removed.append(np.asarray(it['position'], dtype=float))
                for inv in (nd.get('inventory'), st.get('inventory_context')):
                    if isinstance(inv, dict) and 'held_type' in inv: inv['held_type'] = int(it['powerup'])
            else:
                keep.append(it)
        nd['items'] = keep; points += gained
        if gained > 0:
            mz = float(st['position'][2]); mp_ = np.asarray(st['position'][:2], dtype=float)
            lv = [float(it['position'][2]) for it in keep if it.get('gem') and np.linalg.norm(np.asarray(it['position'][:2], dtype=float) - mp_) < 20.
                  and mz - 1.5 <= float(it['position'][2]) <= mz + 1.]
            drop_watch = max(lv) if lv else None
        elif drop_watch is not None and float(st['position'][2]) < drop_watch - S.LEDGE_DROP_DZ:
            drops += 1; drop_watch = None
        fell = bool(engine.mission_bounds is not None and fully_outside_inbounds(st, engine.mission_bounds))
        raw = project_observation(raw0, p0, st, t + 1 - t_base)
        g = raw[RAW_GEMS].reshape(5, 5); pos = np.asarray(st['position'], dtype=float)
        for slot in g:
            if slot[3] > -500 and any(np.linalg.norm(pos + slot[:3] - q) < .1 for q in removed):
                slot[3] = -999.; slot[4] = -999.
        raw[RAW_GEMS] = g.reshape(-1)
        if pw_removed:
            pi = raw[RAW_POW_ITEMS].reshape(3, 5)
            for slot in pi:
                if slot[0] > 0 and any(np.linalg.norm(pos + slot[1:4] - q) < .1 for q in pw_removed):
                    slot[:] = -999.
            raw[RAW_POW_ITEMS] = pi.reshape(-1)
        raw[RAW_SCORE] = raw0[RAW_SCORE] + points - points_base; raw[RAW_GEMS_REMAINING] = raw0[RAW_GEMS_REMAINING] - (len(picks) - picks_base)
        if nxt and t_spawn is None and not any(it.get('gem') for it in nd['items']):
            # the group is cleared: the recorded next group appears (nearest 5 in the observation's gem slots)
            nd['items'] = list(nd['items']) + [copy.deepcopy(it) for it in nxt]
            rows = sorted(((np.asarray(it['position'], dtype=float) - pos).tolist() + [float(it.get('value') or 1.)] for it in nxt),
                          key=lambda r: float(np.linalg.norm(r[:3])))[:5]
            g = np.full((5, 5), -999., dtype=np.float32)
            for k, r in enumerate(rows): g[k] = r[:3] + [r[3], float(np.linalg.norm(r[:3]))]
            raw[RAW_GEMS] = g.reshape(-1); raw[RAW_GEMS_REMAINING] = len(nxt)
            raw0 = raw.copy(); p0 = pos.copy(); t_base = t + 1; points_base = points; picks_base = len(picks); removed = []; t_spawn = t + 1
        speeds.append(float(np.linalg.norm(np.asarray(st['velocity'])[:2])))
        outcome = dict(fell=fell, gem_delta=gained, round_ended=False, reconnected=False)
        msg = Msg(raw, nd); ctrl.observe(before, msg, outcome); d = nd; t += 1
        if fell:
            why = 'fell'; break
        if t_spawn is not None:
            if len(picks) > picks_base:
                why = 'trip_done'; break
            continue
        if len(picks) >= target and not nxt:
            why = 'done'; break
        if not any(it.get('gem') for it in nd['items']):
            why = 'done' if o.whole_group else 'no_gems'; break
    # 2026-10-19: where the episode ended and the gems at its start (the expected trip to the next group, scratchpad cmp_lastgem.py)
    V_ = np.asarray(tick_v, dtype=float); impacts = 0
    if len(V_) > 2:
        h = np.hypot(V_[:, 0], V_[:, 1]); dh = h[:-1] - h[1:]
        dvh = np.hypot(V_[1:, 0] - V_[:-1, 0], V_[1:, 1] - V_[:-1, 1]); dvz = np.abs(V_[1:, 2] - V_[:-1, 2])
        hit = (dh > S.BUMP_TICK_DV) & (dvh > dvz); impacts = int(np.count_nonzero(hit[1:] & ~hit[:-1]) + int(bool(hit[0])))
    return dict(t_s=t * .064, picks=picks, points=points, why=why, labels=dict(labels), side_impacts=impacts, ledge_drops=drops, impact_by=dict(impact_by), mean_speed=float(np.mean(speeds)) if speeds else 0.,
                fallback=int(ctrl.counts.get('fallback_decisions', 0)), safe_holds=int(ctrl.counts.get('safe_holds', 0)),
                searches=int(ctrl.counts.get('full_searches', 0)), blasts=blasts,
                end=[float(x) for x in d['native']['state']['position']], end_vel=[float(x) for x in d['native']['state']['velocity']],
                gems0=[[float(x) for x in it['position']] for it in rec['before'].get('items', []) if it.get('gem')],
                lastgem_tours=int(ctrl.counts.get('lastgem_tours', 0)), next_group_aim=int(ctrl.counts.get('next_group_aim', 0)),
                t_spawn=t_spawn * .064 if t_spawn is not None else None, t_trip=(t - t_spawn) * .064 if t_spawn is not None else None)


EDGE_TERRAIN = [None]
NEXT_ITEMS = {}


def edge_starts(pool, name):
    f = ML / 'dynamics_reports' / pool / name / (o.start_seed + '/physics_plan/1/decisions.jsonl.gz')
    R = []
    for line in gzip.open(f, 'rt', encoding='utf-8'):
        try:
            R.append(json.loads(line))
        except Exception:
            pass
    R.sort(key=lambda d: d['decision']); out = []; T = EDGE_TERRAIN[0]
    for k, d in enumerate(R):
        if float(d['outcome'].get('gem_delta') or 0) <= 0 or k - o.lead < 1:
            continue
        p = d['before']['native']['state']['position']
        j, i = T.cell_of(p[0], p[1]); lv = T.level_at(j, i, float(p[2]) - .19)
        if lv < 0 or float(T.edge_dist[lv, j, i]) >= o.edge_gems:
            continue
        s0 = R[k - o.lead]; st = s0['before']['native']['state']
        if st.get('mode', 0) != 0 or st.get('oob') or s0['choice'].get('label') == 'countdown_neutral':
            continue
        if any(R[j2]['outcome'].get('fell') or float(R[j2]['outcome'].get('gem_delta') or 0) > 0 for j2 in range(k - o.lead, k)):
            continue
        fell_after = any(R[j2]['outcome'].get('fell') for j2 in range(k, min(len(R), k + 60)))
        out.append((s0, None, 1.0 if fell_after else 0.0))
    return out


def endgame_starts(pool, name):
    f = ML / 'dynamics_reports' / pool / name / (o.start_seed + '/physics_plan/1/decisions.jsonl.gz')
    if not f.exists():
        return []
    R = []
    try:
        for line in gzip.open(f, 'rt', encoding='utf-8'):
            try:
                R.append(json.loads(line))
            except Exception:
                pass
    except EOFError:
        pass
    R.sort(key=lambda d: d['decision'])
    for i, d in enumerate(R):
        st = d['before']['native']['state']
        if float(d['raw_observation'][31]) <= o.endgame * 1000. and st.get('mode', 0) == 0 and not st.get('oob'):
            got = sum(float(R[k]['outcome'].get('gem_delta') or 0) for k in range(i, len(R)))
            return [(d, got, 0.)]
    return []


def at_decision_start(pool, name):
    """--at-decision N: the recorded round's decision N itself (any decision, not only a leg start), when playable."""
    f = ML / 'dynamics_reports' / pool / name / (o.start_seed + '/physics_plan/1/decisions.jsonl.gz')
    for line in gzip.open(f, 'rt', encoding='utf-8'):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d['decision'] == o.at_decision:
            st = d['before']['native']['state']
            return [] if st.get('mode', 0) != 0 or st.get('oob') else [(d, None, 0.0)]
    return []


def starts(pool, name):
    if o.at_decision:
        return at_decision_start(pool, name)
    if o.endgame:
        return endgame_starts(pool, name)
    if o.edge_gems:
        return edge_starts(pool, name)
    f = ML / 'dynamics_reports' / pool / name / (o.start_seed + '/physics_plan/1/decisions.jsonl.gz')
    R = []
    for line in gzip.open(f, 'rt', encoding='utf-8'):
        try:
            R.append(json.loads(line))
        except Exception:
            pass
    R.sort(key=lambda d: d['decision']); out = []
    for i in range(1, len(R) - 1):
        prev, d = R[i - 1], R[i]
        if float(prev['outcome'].get('gem_delta') or 0) <= 0 or prev['outcome'].get('fell'):
            continue
        st = d['before']['native']['state']
        if st.get('mode', 0) != 0 or st.get('oob') or (d['choice'].get('label') == 'countdown_neutral'):
            continue
        gems = [it for it in d['before'].get('items', []) if it.get('gem')]
        if len(gems) < (2 if o.whole_group else o.pickups):
            continue
        # recorded time to the next o.pickups pickups (for reference) and the straight distance to the next pickup
        k = i; got = 0; tt = None; nxt = None; fell_at = None
        while k < len(R):
            if R[k]['outcome'].get('fell'):
                fell_at = k; break
            if float(R[k]['outcome'].get('gem_delta') or 0) > 0:
                got += 1
                if nxt is None:
                    nxt = np.asarray(R[k]['before']['native']['state']['position'], dtype=float)
                if got >= (len(gems) if o.whole_group else o.pickups):
                    tt = (k - i + 1) * .064; break
            k += 1
        if o.fall_legs:
            if fell_at is None or got > 0 or (fell_at - i) * .064 > 20:
                continue
            q = np.asarray(R[fell_at]['before']['native']['state']['position'], dtype=float)
            dl = R[i - o.lead] if o.lead and i - o.lead >= 0 else d
            if o.lead and (dl['before']['native']['state'].get('mode', 0) != 0 or dl['before']['native']['state'].get('oob')
                           or any(R[j]['outcome'].get('fell') for j in range(i - o.lead, i))):
                continue
            out.append((dl, (fell_at - i + 1) * .064, float(np.linalg.norm((q - np.asarray(st['position']))[:2]))))
            continue
        far = None if nxt is None else float(np.linalg.norm((nxt - np.asarray(st['position']))[:2]))
        if o.rise_min is not None and (nxt is None or float(nxt[2]-st['position'][2]) < o.rise_min):
            continue
        if o.next_group:
            prevg = None; found = None
            for k2 in range(i, min(len(R), i + 3000)):
                gk = {tuple(np.round(it['position'], 2)) for it in R[k2]['before'].get('items', []) if it.get('gem')}
                if prevg is not None and len(gk - prevg) >= 2:
                    found = [it for it in R[k2]['before'].get('items', []) if it.get('gem') and tuple(np.round(it['position'], 2)) not in prevg]; break
                prevg = gk
            if not found:
                continue
            NEXT_ITEMS[(pool, d['decision'])] = found
        if o.stuck_legs:
            k1 = next((j for j in range(i, len(R)) if float(R[j]['outcome'].get('gem_delta') or 0) > 0 or R[j]['outcome'].get('fell')), None)
            if k1 is None or R[k1]['outcome'].get('fell') or (k1 - i + 1) * .064 <= 8 or far is None or far >= 20:
                continue
        out.append((d, tt, far))
    return out


def main():
    for kv in o.set:
        key, val = kv.split('=', 1); mod, attr = key.split('.')
        import nav.marble_dynamics.geometry_route as GR, nav.marble_dynamics.crossings as CRX, nav.marble_dynamics.gap_links as GLX, nav.marble_dynamics.climbs as CLX
        module = {'gem_demo': GD, 'search': S, 'reliability_eval': RE, 'geometry_route': GR, 'crossings': CRX, 'gap_links': GLX, 'climbs': CLX}[mod]
        setattr(module, attr, type(getattr(module, attr))(val) if not isinstance(getattr(module, attr), bool) else val in ('1', 'True', 'true'))
    a = planner_args(o.map)
    terrain, engine = setup(a, o.map)
    EDGE_TERRAIN[0] = a.geometry_terrain
    for kv in o.set:   # re-apply after setup() (it resets the gem_demo/search globals from the planner args)
        key, val = kv.split('=', 1); mod, attr = key.split('.')
        import nav.marble_dynamics.geometry_route as GR, nav.marble_dynamics.crossings as CRX, nav.marble_dynamics.gap_links as GLX, nav.marble_dynamics.climbs as CLX
        module = {'gem_demo': GD, 'search': S, 'reliability_eval': RE, 'geometry_route': GR, 'crossings': CRX, 'gap_links': GLX, 'climbs': CLX}[mod]
        setattr(module, attr, type(getattr(module, attr))(val) if not isinstance(getattr(module, attr), bool) else val in ('1', 'True', 'true'))
    if o.search_workers:
        from nav.marble_dynamics.parallel_search import ParallelEvaluator, worker_init
        overrides = [tuple(sp.split('=', 1)) for sp in ap2.parse_known_args()[0].module]
        engine.parallel = ParallelEvaluator(o.search_workers, worker_init(a, o.map, overrides))
    if o.align:
        S.candidate_cost = aligned_cost
    cand = []
    for pool in o.pool:
        cand += [(pool,) + c for c in starts(pool, o.map) if c[2] is not None and o.far_min <= c[2] < o.far_max]
    if o.held >= 0:
        # 2026-10-20: a held Mega is inventory held_type 6 (the native 'held' stays 0 for it)
        cand = [c for c in cand if int(c[1]['before']['native']['state'].get('held', 0)) == o.held
                or int((c[1]['before'].get('inventory') or {}).get('held_type', -1)) == o.held]
    step = max(1, len(cand) // o.per_map); chosen = cand[o.offset % step::step][:o.per_map]
    if o.only_decision:
        chosen = [c for c in cand if c[1]['decision'] == o.only_decision][:1]
    if o.max_episodes:
        chosen = chosen[:o.max_episodes]
    if o.slice:
        lo, hi = (int(x) for x in o.slice.split(':'))
        chosen = chosen[lo:hi]
    with open(o.out, 'a', encoding='utf-8') as out:
        for pool, rec, rec_t, far in chosen:
            s0 = STEPS[0]; w0 = time.perf_counter() if o.real_clock else None; r0 = time.process_time()
            res = episode(a, terrain, engine, rec, NEXT_ITEMS.get((pool, rec['decision'])) if o.next_group else None)
            res.update(variant=o.variant, map=o.map, pool=pool, decision=rec['decision'], recorded_t=rec_t, far=far,
                       steps=STEPS[0] - s0, cpu_s=time.process_time() - r0)
            if o.real_clock:
                res['wall_s'] = time.perf_counter() - w0
            out.write(json.dumps(res) + '\n'); out.flush()
            print('%s %s dec %5d far %5s rec %5s -> t %5.1f picks %s why %s fb %d sh %d speed %.1f cpu %.0fs' % (
                o.variant, o.map[:10], rec['decision'], '%.0f' % far if far is not None else '-', '%.1f' % rec_t if rec_t else '-',
                res['t_s'], ['%.1f' % p for p in res['picks']], res['why'], res['fallback'], res['safe_holds'], res['mean_speed'], res['cpu_s']), flush=True)


if __name__ == '__main__':
    main()
