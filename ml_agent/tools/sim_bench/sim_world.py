"""The offline simulator's world: planner arguments, engine/terrain setup and the bridge-state resume, factored out of
sim_bench.py (2026-10-29) so the full-round driver (full_round.py), the conformance tool (conformance.py) and the leg
benchmark share one definition.

Flags: by default the PRODUCTION planner flags (nav.marble_dynamics.realtime.FLAGS, horizon 48, as the native campaigns
and real-time play run them). sim_bench.py before 10-29 used its own list with --horizon 24 (the pre-v45 horizon).

Virtual clock: with native forecasts every physics step charges NATIVE_STEP_US to time.perf_counter(), which this module
replaces when `install_virtual_clock()` is called; the search deadline then depends only on the work done, so an episode
is deterministic (A/A on 2026-10-29: four Sprawl legs identical to the step).
"""
import copy
import json
import math
import os
import sys
import time

import numpy as np

ML = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ML not in sys.path:
    sys.path.insert(0, ML)

VC = [1000.0]        # the virtual clock (seconds)
STEPS = [0]          # physics steps charged so far
NATIVE_STEP_US = 100.   # --native: virtual clock per physics step. 2026-10-16 calibration under pool load (open loop, 250
                        # decisions each): Cube 61, Sprawl 108, Bowl 138 us per step (Python work per forecast dominates)
CACHE = r'C:\Users\doug\AppData\Local\Temp\pq_checks\fast_nav_static_cache_20261013'
DLL = r'C:\Users\doug\AppData\Local\Temp\pq_checks\compiled_sweep_20261012\sweep_kernel.dll'
_real_perf_counter = time.perf_counter


def install_virtual_clock():
    time.perf_counter = lambda: VC[0]


def production_flags(budget_s=2.):
    """realtime.FLAGS without its '--seeds N' prefix, the search budget set to budget_s (the virtual budget)."""
    from nav.marble_dynamics import realtime as rt
    flags = list(rt.FLAGS[2:])
    for i, f in enumerate(flags):
        if f == '--search-budget-s':
            flags[i + 1] = '%g' % budget_s
    return flags


LEGACY_FLAGS = ('--random-starts --only-branch physics_plan --full-round --horizon 24 --commit 4 --reuse-plan --geometry-route '
                '--resource-acquisition --recharge-continuation --landing-search --proposal-samples 20 --query-cache --clearance-cache '
                '--global-goals --physics-frame-ms 16 --fallback-cooldown 2 --search-budget-s %g --pickup-landing-continuation 40 '
                '--component-crossing --ballistic-landing --use-horizon 48 --use-margin-s 1')


class _Got(Exception):
    pass


def planner_args(name, budget_s=2., pflags=(), legacy=False):
    """The reliability_eval argument namespace for one map, exactly as main() would parse it."""
    import argparse as _ap
    import nav.marble_dynamics.reliability_eval as RE
    orig = _ap.ArgumentParser.parse_args

    def cap(self, args=None, namespace=None):
        raise _Got(orig(self, args, namespace))
    _ap.ArgumentParser.parse_args = cap
    saved = sys.argv
    flags = (LEGACY_FLAGS % budget_s).split() if legacy else production_flags(budget_s)
    if '--cache-dir' not in flags:
        flags += ['--cache-dir', CACHE, '--sweep-dll', DLL]
    sys.argv = ['reliability_eval', '--maps', name, '--seeds', '7347019'] + flags + \
        ['--output', 'ml_agent/dynamics_reports/_sim_unused'] + [x for f in pflags for x in f.split('=', 1)]
    try:
        RE.main()
    except _Got as g:
        a = g.args[0]
    finally:
        _ap.ArgumentParser.parse_args = orig; sys.argv = saved
    return a


def setup(a, name, native=True, step_cost_ms=0.6, native_step_us=0., real_clock=False):
    """Engine/terrain exactly as reliability_eval.main() builds them for one map (the settings block mirrored)."""
    import nav.marble_dynamics.reliability_eval as RE
    import nav.marble_dynamics.gem_demo as GD
    import nav.marble_dynamics.search as S
    from nav.terrain import TerrainMap
    from nav.marble_dynamics.static_cache import cached_terrain, cached_walking, load_queries
    from nav.collision.experimental_sweep import CachedCompiledSweepMap
    from nav.marble_dynamics.physics import GeneralPhysics
    terrain = cached_terrain(TerrainMap.resolve(name), a.cache_dir)
    mesh = CachedCompiledSweepMap(os.path.join(ML, 'collision_maps', name + '.npz'), a.sweep_dll); load_queries(mesh, a.sweep_dll, a.cache_dir)
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
    S.POWERUP_THROUGH = bool(getattr(a, 'powerup_through', True))
    S.LAND_SHORT = bool(getattr(a, 'land_short', False))
    S.POW_LAUNCH_ROLL = bool(getattr(a, 'roll_launches', False))
    S.LAUNCH_COMPLETE = bool(getattr(a, 'launch_completion', False) or S.LAUNCH_COMPLETE) and not getattr(a, 'no_launch_completion', False)
    S.FAST_SUPPORTED = bool(getattr(a, 'fast_supported', False))
    S.THROUGH_GEM = not getattr(a, 'no_through_gem', False); S.BRAKING_BACKUP = not getattr(a, 'no_braking_backup', False)
    S.CORNER_PROPOSALS = bool(getattr(RE, 'CORNER_DEFAULT', False)) or bool(getattr(a, 'corner_proposals', False))
    engine.detour_blast_ratio = float(getattr(a, 'detour_blast_ratio', 1.5) or 0.) or math.inf
    engine.use_margin_s = float(getattr(a, 'use_margin_s', 0.) or 0.)
    engine.use_margin_u = engine.use_margin_s * float(getattr(engine, 'ballistic_time_rate', 8.) or 8.)
    engine.interleave_candidates = getattr(a, 'interleave_candidates', False)
    engine.recheck_risky_plans = getattr(a, 'recheck_risky_plans', False)
    engine.mega_enabled = getattr(a, 'mega_proposals', False)
    engine.pickup_stop = getattr(a, 'pickup_stop', False)
    engine.edge_hits = getattr(a, 'edge_hits', False)
    engine.lip_skill = getattr(a, 'lip_skill', True)
    engine.wall_rebounds = getattr(a, 'wall_rebounds', True)
    engine.pad_hops = getattr(a, 'pad_hops', False)
    engine.resource_cap = getattr(a, 'resource_cap', True)
    engine.time_price = getattr(a, 'time_price', False)
    GD.set_time_price(engine.time_price)
    if engine.time_price:
        engine.ballistic_time_rate = GD.TIME_PRICE_U_S
    engine.pad_arrival = getattr(a, 'pad_arrival', True)
    a.geometry_terrain = cached_walking(terrain, mesh, a.sweep_dll, a.cache_dir, memo=a.clearance_cache)
    engine.ballistic_landing = getattr(a, 'ballistic_landing', False); engine.ballistic_reserve_s = getattr(a, 'ballistic_reserve_s', 0.)
    engine.opportunistic_powerups = getattr(a, 'opportunistic_powerups', 0.)
    engine.checked_powerup_detours = getattr(a, 'checked_powerup_detours', False)
    engine.mission_bounds = None
    try:
        from nav.marble_dynamics.fast_nav import mission_inbounds
        engine.mission_bounds = mission_inbounds(json.loads(open(os.path.join(ML, 'collision_maps', name + '.json')).read()))
    except (ValueError, OSError):
        engine.mission_bounds = None
    engine.crossings = None
    if getattr(a, 'component_crossing', False):
        from nav.marble_dynamics.crossings import Crossings
        engine.crossings = Crossings(a.geometry_terrain, mesh=engine.mesh, climbs=getattr(a, 'climb_crossings', False))
    real_step = engine.step

    def step(*args, **kw):
        STEPS[0] += 1; VC[0] += step_cost_ms / 1000.
        return real_step(*args, **kw)
    engine.step = step
    if native:
        from nav.native import attach
        attach(engine); engine.native_sim.bind_clearance(a.geometry_terrain)
        cost = (native_step_us or NATIVE_STEP_US) / 1e6

        def hook(n):
            STEPS[0] += n
            if not real_clock:
                VC[0] += n * cost
        engine.native_step_hook = hook
        engine.virtual_step_cost_s = None if real_clock else cost
    return terrain, engine


def apply_settings(settings):
    """--set module.NAME=value on the planner modules (after setup(), which resets some of them from the args)."""
    import nav.marble_dynamics.gem_demo as GD, nav.marble_dynamics.search as S, nav.marble_dynamics.reliability_eval as RE
    import nav.marble_dynamics.geometry_route as GR, nav.marble_dynamics.crossings as CRX, nav.marble_dynamics.gap_links as GLX
    import nav.marble_dynamics.climbs as CLX, nav.terrain as TRN, nav.marble_dynamics.resources as RSX, nav.marble_dynamics.lip_skill as LSX
    modules = {'gem_demo': GD, 'search': S, 'reliability_eval': RE, 'geometry_route': GR, 'crossings': CRX, 'gap_links': GLX,
               'climbs': CLX, 'terrain': TRN, 'resources': RSX, 'lip_skill': LSX}
    for kv in settings:
        key, val = kv.split('=', 1); mod, attr = key.split('.')
        module = modules[mod]; cur = getattr(module, attr)
        setattr(module, attr, (val in ('1', 'True', 'true')) if isinstance(cur, bool) else type(cur)(val))


def resume(dynamics, rollout):
    """viability.resume_forecast without the ambiguity refusal (the sim accepts the engine's own choice)."""
    from nav.marble_dynamics.bridge import translated
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


def load_journal(path):
    import gzip
    rows = []
    for line in gzip.open(path, 'rt', encoding='utf-8'):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    rows.sort(key=lambda d: d['decision'])
    return rows


def journal_path(pool, name, seed):
    return os.path.join(ML, 'dynamics_reports', pool, name, str(seed), 'physics_plan', '1', 'decisions.jsonl.gz')
