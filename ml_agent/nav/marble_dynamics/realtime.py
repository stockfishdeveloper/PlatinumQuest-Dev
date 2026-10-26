"""Real-time planner play: the physics planner drives a game started the normal way, at normal speed (2026-10-19,
operator: "my main goal is for us to be able to play these maps (or any map in the future) via the play > chose map
option in the game i.e. in realtime").

How a round runs. The game's bridge (client/scripts/ai/mlAgent.cs) connects to this server on port 8888 whenever a Hunt
round starts: Play > Hunt > map, or a -autotrain launch. The server asks the game for 16 ms physics frames with one
observation per 64 ms decision, lockstep, the exact marble state (DYNAMICS), 1x speed and real-time frames: the game
(marbleblast_realtime.exe, REALTIMESTEP) runs each 16 ms frame on its own 16 ms of wall clock, so the round plays at
normal speed with every frame drawn on time (~62 a second). With an exe that lacks it, the server paces its replies to
64 ms of wall clock instead (PacedEnv; the picture then moves in 64 ms bursts). The planner is the pool planner
unchanged (reliability_eval.ContinuousController with the v86 flag set, parallel candidate evaluation).

Plan-ahead. A search takes ~0.15 s (p99 ~0.6 s) while the game wants a reply every 64 ms, so the planner runs up to LEAD
decisions ahead of the game in its own thread. For each decision it builds the observation the game will send then:
an exact compiled forecast, from the newest real observation, of the replies already committed for the decisions in
between (position, velocity, spin, timers, pending inputs, Blast meter, held powerup), with the items those moves will
collect removed (the game's own per-tick pickup test). It asks the planner for that decision's reply and commits it. The
game thread sends committed replies on time and holds the game (lockstep) only when the planner has not reached the
decision yet (the first search of a round, after a fall, a search longer than the lead). What a forecast cannot know
(a new gem group, a fall, a rescue) reaches the planner in the next real observation and affects the moves it commits
from then on: up to LEAD decisions of reaction delay (0.32 s at LEAD 5).

Information: only what the game shows this player (its own marble's state, the items on the map, the round clock);
nothing about future spawns.

usage (repo root): PYTHONPATH=ml_agent python -m nav.marble_dynamics.realtime     (then start a Hunt round in the game)
"""
import argparse
import copy
import json
import math
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

LEAD = 5                  # decisions the planner may run ahead of the game (5 x 64 ms = 0.32 s)
SEARCH_LEAD = 2           # optional searches (a better plan than the one being followed) only this far ahead or more
ORDER_CANDIDATES_AHEAD = 3   # first targets tried in a cluster when ahead (reliability_eval.ORDER_CANDIDATES); 1 when not
MIN_BUDGET_S = .05        # the shortest search budget (a search the plan needs, with the game about to reach it)
SEARCH_WORKERS = 8        # candidate-evaluation worker processes per map (parallel_search; 16 measured no faster)
PORT = 8888               # the bridge's default port: a game started from the menu connects here
DECISION_S = .064
STALL_REPORT_S = .25      # a hold longer than this is printed
PLAN_FAIL_LIMIT_S = 4.    # fail before the bridge's 5 s live lockstep timeout releases unchecked motion
CHECKED_TAIL = False      # opt-in until paired native real-time checks pass
CHECKED_RESERVE = 8       # at most 512 ms of already checked continuation
ML = Path(__file__).resolve().parents[2]
OUTPUT = ML / 'dynamics_reports' / 'realtime'
# the v86 pool planner (scratchpad launch_pool.sh flag set + --horizon 48, dynamics_reports/planner_v86_*/prepared.json)
FLAGS = ['--seeds', '7347019', '--only-branch', 'physics_plan', '--full-round', '--horizon', '48', '--commit', '4',
         '--reuse-plan', '--geometry-route', '--resource-acquisition', '--recharge-continuation', '--landing-search',
         '--proposal-samples', '20', '--query-cache', '--clearance-cache', '--global-goals', '--physics-frame-ms', '16',
         '--fallback-cooldown', '2', '--search-budget-s', '2', '--pickup-landing-continuation', '40',
         '--component-crossing', '--ballistic-landing', '--use-horizon', '48', '--use-margin-s', '1',
         '--cache-dir', r'C:\Users\doug\AppData\Local\Temp\pq_checks\fast_nav_static_cache_20261013',
         '--sweep-dll', r'C:\Users\doug\AppData\Local\Temp\pq_checks\compiled_sweep_20261012\sweep_kernel.dll']


def log(text):
    print(time.strftime('%H:%M:%S ') + text, flush=True)


# ------------------------------------------------------------------------------------------------------ planner world
class _Parsed(Exception):
    pass


def planner_args(name):
    """reliability_eval's argument namespace for one map with the production flag set (its parser, stopped right after
    parsing)."""
    import nav.marble_dynamics.reliability_eval as RE
    original = argparse.ArgumentParser.parse_args

    def parsed(self, args=None, namespace=None):
        raise _Parsed(original(self, args, namespace))
    argparse.ArgumentParser.parse_args = parsed
    saved = sys.argv
    sys.argv = ['reliability_eval', '--maps', name, '--output', str(OUTPUT / '_args')] + FLAGS
    try:
        RE.main()
        raise RuntimeError('reliability_eval.main did not parse its arguments')
    except _Parsed as got:
        return got.args[0]
    finally:
        argparse.ArgumentParser.parse_args = original
        sys.argv = saved


class World:
    """One map's planner: terrain, collision mesh, compiled simulator, walking terrain, crossings and the candidate
    worker pool, built exactly as reliability_eval.main() builds them for a pool round."""

    def __init__(self, name):
        import nav.marble_dynamics.reliability_eval as RE
        import nav.marble_dynamics.gem_demo as GD
        import nav.marble_dynamics.search as S
        from nav.terrain import TerrainMap
        from nav.marble_dynamics.static_cache import cached_terrain, cached_walking, load_queries
        from nav.collision.experimental_sweep import CachedCompiledSweepMap
        from nav.marble_dynamics.physics import GeneralPhysics
        from nav.marble_dynamics.navigator import FrozenNavigator
        from nav.native import attach
        t0 = time.perf_counter()
        a = planner_args(name)
        self.name = name; self.args = a
        terrain = cached_terrain(TerrainMap.resolve(name), a.cache_dir)
        mesh = CachedCompiledSweepMap(ML / 'collision_maps' / (name + '.npz'), a.sweep_dll)
        load_queries(mesh, a.sweep_dll, a.cache_dir)
        engine = GeneralPhysics(mesh)
        attach(engine)
        engine.bridge_physics_ms = a.physics_frame_ms
        engine.terminal_stop_replies = a.terminal_stop_replies
        engine.terminal_stop_scope = a.terminal_stop_scope
        engine.landing_replies = getattr(a, 'landing_continuation', 0)
        engine.pickup_landing_replies = getattr(a, 'pickup_landing_continuation', 0)
        engine.use_horizon = getattr(a, 'use_horizon', 0)
        engine.use_first = getattr(a, 'use_first', False)
        engine.blast_value_u = float(getattr(a, 'blast_value_u', 10.))
        engine.blast_floor_u = float(getattr(a, 'blast_floor_u', 0.))
        GD.END_VELOCITY_WEIGHT = float(getattr(a, 'end_velocity_weight', .12))
        GD.THROTTLE_GAIN = float(getattr(a, 'throttle_gain', 4.)); GD.THROTTLE_CAP = float(getattr(a, 'throttle_cap', 1.))
        GD.BRAKE_DECEL = float(getattr(a, 'brake_decel', 7.))
        GD.TARGET_SPEED_CAP = float(getattr(a, 'speed_cap', 11.))
        S.ROUTE_SPEED_LAW = not getattr(a, 'waypoint_brake_law', False)
        S.PICKUP_MIN_SPEED = float(getattr(a, 'pickup_min_speed', 6.))
        S.FAST_PROPOSALS = not getattr(a, 'legacy_fast_proposals', False)
        S.THROUGH_GEM = not getattr(a, 'no_through_gem', False)
        S.THROUGH_BOX = bool(getattr(a, 'through_pickup_ticks', False))
        S.REFINE_CANDIDATES = int(getattr(a, 'refine_candidates', 0))
        S.OPEN_TOP = 13.5 if getattr(a, 'open_top', False) else 0.
        S.RAISED_BLAST_FIRST = bool(getattr(a, 'raised_blast_first', False))
        S.CORNER_PROPOSALS = RE.CORNER_DEFAULT or bool(getattr(a, 'corner_proposals', False))
        S.BRAKING_BACKUP = not getattr(a, 'no_braking_backup', False)
        RE.SPAWN_ORDER = bool(getattr(a, 'spawn_order', RE.SPAWN_ORDER)); S.SS_RIDE = bool(getattr(a, 'ss_ride', False)); GD.TIME_COST = bool(getattr(a, 'time_cost', False))
        S.LATE_PAD_JUMPS = bool(getattr(a, 'late_pad_jumps', S.LATE_PAD_JUMPS))
        from nav.marble_dynamics import crossings as _crossings_mod
        _crossings_mod.GAP_ROUTES = bool(getattr(a, 'gap_routes', _crossings_mod.GAP_ROUTES))
        from nav.marble_dynamics import climbs as _climbs_mod
        _climbs_mod.CLIMB_SKILL = bool(getattr(a, 'climb_skill', _climbs_mod.CLIMB_SKILL))
        S.CORNER_SLOW = bool(getattr(a, 'slow_corners', S.CORNER_SLOW))
        S.TL_JUMP_FOLLOW = bool(getattr(a, 'tl_jump_follow', S.TL_JUMP_FOLLOW))
        engine.detour_blast_ratio = float(getattr(a, 'detour_blast_ratio', 1.5) or 0.) or math.inf
        engine.use_margin_u = float(getattr(a, 'use_margin_s', 0.) or 0.) * float(getattr(engine, 'ballistic_time_rate', 8.) or 8.)
        engine.interleave_candidates = getattr(a, 'interleave_candidates', False)
        engine.recheck_risky_plans = getattr(a, 'recheck_risky_plans', False)
        engine.mega_enabled = getattr(a, 'mega_proposals', False)
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
        engine.native_sim.bind_clearance(a.geometry_terrain)
        engine.ballistic_landing = getattr(a, 'ballistic_landing', False)
        engine.ballistic_reserve_s = getattr(a, 'ballistic_reserve_s', 0.)
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
            if _crossings_mod.GAP_ROUTES:
                engine.crossings.gap_links()   # 2026-10-25: built here (0.5-2 s), not at the first decision of a round
        FrozenNavigator(terrain, a.checkpoint)
        self.terrain = terrain; self.engine = engine
        engine.parallel = None
        if SEARCH_WORKERS:
            from nav.marble_dynamics.parallel_search import ParallelEvaluator, worker_init
            engine.parallel = ParallelEvaluator(SEARCH_WORKERS, worker_init(a, name))
        log('%s: planner ready in %.1f s (%d search workers)' % (name, time.perf_counter() - t0, SEARCH_WORKERS))

    def controller(self):
        from nav.marble_dynamics.reliability_eval import ContinuousController
        c = ContinuousController(self.terrain, self.engine, self.args.checkpoint, self.args, 'physics_plan')
        e = self.engine
        if e.parallel is not None:
            e.parallel.bind(dict(terrain=self.terrain, geometry=self.args.geometry_terrain, mesh=e.mesh, sim=e.native_sim,
                                 model=c.nav.model, crossings=e.crossings, contact_order=getattr(e, 'contact_order', None)))
        return c

    def close(self):
        if self.engine.parallel is not None:
            self.engine.parallel.close()
            self.engine.parallel = None


class WorldManager:
    """One active map and worker pool. Static geometry caches remain on disk."""
    def __init__(self, factory=World):
        self.factory = factory
        self.world = None

    def get(self, name):
        if self.world is not None and self.world.name == name:
            return self.world
        self.close()                    # free old workers BEFORE starting another eight
        self.world = self.factory(name)
        return self.world

    def close(self):
        if self.world is not None:
            old, self.world = self.world, None
            old.close()


# ----------------------------------------------------------------------------------------------- predicted observation
class Msg:
    """What ContinuousController reads from a game message: the raw observation and the DYNAMICS context."""

    def __init__(self, obs, dynamics, tick=0):
        self.obs = obs; self.dynamics = dynamics; self.tick = tick


def predict(engine, actual, replies):
    """The observation the game will send len(replies) decisions after `actual` if it receives `replies` in between:
    (Msg, {k: gem points collected in the k-th interval}). Exact compiled forecast of the bridge and physics; items the
    forecast collects (the game's per-tick test) are gone; nothing new appears. Raises ValueError when the forecast
    cannot continue (out of bounds, a mode the compiled simulator does not model)."""
    from nav.marble_dynamics.script_forecast import forecast_items
    from nav.marble_dynamics.viability import resume_forecast
    from nav.protocol import (RAW_POS, RAW_VEL, RAW_SPIN, RAW_GEMS, RAW_TIME_LEFT, RAW_TIME_REMAINING, RAW_SCORE,
                              RAW_GEMS_REMAINING, RAW_POW_HELD, RAW_POW_BLAST, RAW_POW_SPECIAL, RAW_POW_ITEMS)
    from nav.obs import POW_RESPAWN_S
    n = len(replies)
    dynamics = actual.dynamics
    saved = getattr(engine, 'planning_deadline', None); engine.planning_deadline = None
    try:
        forecast = forecast_items(engine, dynamics, [list(r) for r in replies])
    finally:
        engine.planning_deadline = saved
    if forecast['failure'] or len(forecast['controls']) != n:
        raise ValueError('forecast of the committed replies stopped: ' + str(forecast['failure']))
    predicted = resume_forecast(dynamics, forecast)
    state = predicted['native']['state']
    items = forecast['items']; picked = Counter()
    picked_events = forecast['item_events']
    for event in picked_events:
        if event['item']['gem']:
            picked[event['interval']] += float(event['item']['value'])
    predicted['items'] = items
    raw = np.array(actual.obs, dtype=np.float32, copy=True)
    p = np.asarray(state['position'], dtype=float); p0 = np.asarray(dynamics['native']['state']['position'], dtype=float)
    raw[RAW_POS] = p; raw[RAW_VEL] = state['velocity']; raw[RAW_SPIN] = state['spin']
    held = int(state['held'])
    if (state.get('inventory_context') or {}).get('held_type') == 6:
        held = 6
    raw[RAW_POW_HELD] = held
    blast = state.get('blast_context')
    if blast is not None:
        raw[RAW_POW_BLAST] = blast['value']; raw[RAW_POW_SPECIAL] = int(blast['special'])
    raw[43:46] = [state['effects'][i][1] / 1000. if state['effects'][i][0] else 0. for i in (3, 4, 5)]
    raw[RAW_TIME_LEFT] -= n * 64; raw[RAW_TIME_REMAINING] += n * 64
    gained = sum(picked.values()); raw[RAW_SCORE] += gained; raw[RAW_GEMS_REMAINING] -= gained
    # the observer's gem slots: the first five gems in object order, nearest first, -999 padding (observer.cs)
    gems = [i for i in items if i['gem']][:5]
    slots = []
    for g in gems:
        rel = np.asarray(g['position'], dtype=float) - p
        slots.append([rel[0], rel[1], rel[2], float(g['value']), float(np.linalg.norm(rel))])
    slots.sort(key=lambda s: s[4])
    slots += [[-999.] * 5] * (5 - len(slots))
    raw[RAW_GEMS] = np.asarray(slots, dtype=np.float32).ravel()
    # powerup slots: positions relative to the marble; respawn timers run on; collected items start respawning
    pw = raw[RAW_POW_ITEMS].reshape(3, 5)
    for slot in pw:
        if slot[0] > 0:
            slot[1:4] += (p0 - p).astype(np.float32)
            if slot[4] > 0:
                slot[4] = max(0., float(slot[4]) - n * DECISION_S)
            for event in picked_events:
                item = event['item']
                if not item['gem'] and np.linalg.norm(np.asarray(item['position'], dtype=float) - (p + slot[1:4])) < .1:
                    elapsed = n * DECISION_S - (event['tick'] * .016 if event['tick'] is not None else (event['interval']+1) * DECISION_S)
                    slot[4] = max(0., POW_RESPAWN_S - elapsed)
    raw[RAW_POW_ITEMS] = pw.ravel()
    return Msg(raw, predicted, int(actual.tick) + n), dict(picked)


# ------------------------------------------------------------------------------------------------------- the planner
class AheadPlanner(threading.Thread):
    """Runs the planner up to LEAD decisions ahead of the game on predicted observations; commits one reply per
    decision. The game thread feeds real observations (see) and takes committed replies (reply_for)."""

    def __init__(self, world, controller, lead=LEAD):
        super().__init__(daemon=True)
        self.world = world; self.engine = world.engine; self.controller = controller; self.lead = int(lead)
        self.cv = threading.Condition()
        self.latest = None            # (decision, Msg, outcome of the step into it)
        self.committed = {}           # decision -> reply
        self.sent = {}                # decision -> reply actually sent (the forecast's inputs)
        self.checked = {}             # decision -> (reply, expected state, preceding bridge reply)
        self.generation = 0           # an in-flight result cannot survive a restart or checked fallback
        self.next = None              # next decision to plan; None until the round's first playable decision
        self.reset = False
        self.running = True
        self.error = None
        self.prev = None              # the observation the planner saw at the previous decision
        self.tail = []; self.tail_from = None   # legacy fallback retained while checked-tail is experimental
        self.stats = Counter(); self.leads = []; self.latencies = []; self.labels = Counter()
        self.chosen = {}; self.last_label = '?'   # 2026-10-20: plan label per committed decision, journaled by play_round
        # The stuck check's route distance, computed HERE for each new real observation. Route queries reach the
        # compiled simulator (line_clear -> ground clearance -> marble_sim.segment), which one thread at a time may use:
        # the game thread's own route query during a forecast on this thread corrupted the heap (10-19 13:24 and 17:24,
        # 0xc0000374; faulthandler showed both stacks).
        from nav.routing import RoutePlanner
        self.progress_route = RoutePlanner(controller.progress_route.terrain)
        self.progress = (None, None, float('inf'))   # (real decision, goal key, route distance to the goal)

    # game thread -----------------------------------------------------------------------------------------------
    def see(self, decision, msg, outcome):
        with self.cv:
            self.latest = (decision, msg, outcome)
            if self.next is None:
                self.next = decision
            self.cv.notify_all()

    def restart(self, decision):
        """Plans committed past `decision` were made for a world that changed (fall, rescue): drop them and plan again
        from the real observation at `decision`."""
        with self.cv:
            self.committed = {d: r for d, r in self.committed.items() if d < decision}
            self.checked.clear(); self.generation += 1
            self.next = decision; self.reset = True
            self.cv.notify_all()

    def reply_for(self, decision):
        """The committed reply for `decision`, waiting (the game holds in lockstep) until the planner has made it."""
        t0 = time.perf_counter()
        with self.cv:
            while decision not in self.committed:
                if not self.running:
                    raise RuntimeError('planner stopped before committing a reply')
                if self.error is not None:
                    raise RuntimeError('planner thread failed:\n' + self.error)
                if CHECKED_TAIL and self._checked_reply(decision):
                    break
                if time.perf_counter() - t0 > PLAN_FAIL_LIMIT_S:
                    raise RuntimeError('no reply for decision %d after %.0f s' % (decision, PLAN_FAIL_LIMIT_S))
                self.cv.wait(.05)
            r = self.committed.pop(decision)
            self.sent[decision] = r
            self.last_label = self.chosen.pop(decision, '?')
            for d in [d for d in self.chosen if d < decision - 4 * self.lead - 8]:
                del self.chosen[d]
            for d in [d for d in self.sent if d < decision - 4 * self.lead - 8]:
                del self.sent[d]
        held = time.perf_counter() - t0
        return r, held

    def _checked_reply(self, decision):
        """Game thread: numeric comparison only, never touches the native simulator.

        Publish a previously checked continuation only if the actual state and
        preceding sent reply still agree. Invalidate any concurrent calculation
        for this decision; it must resume from the action actually sent.
        """
        entry = self.checked.get(decision)
        if entry is None or self.latest is None or self.latest[0] != decision:
            return False
        action, expected, previous, bridge = entry
        dynamics = self.latest[1].dynamics
        actual = dynamics['native']['state']
        from nav.marble_dynamics.reliability_eval import pending_reply_matches
        if (not np.all(np.isfinite(dynamics['bridge_action'])) or not np.all(np.isfinite(previous))
                or actual['oob'] or actual['mode'] != 0 or not pending_reply_matches(dynamics['bridge_action'], previous)
                or (previous[6] and abs(float(dynamics['bridge_action'][7])-float(previous[7])) > 2e-5)):
            self.checked.clear(); self.stats['checked_tail_mismatch'] += 1
            return False
        move = dynamics['native'].get('pending_move')
        if (move is None or move.keys() != bridge['pending_move'].keys()
                or any(not np.isfinite(float(move[k])) or not np.isfinite(float(v)) or abs(float(move[k])-float(v)) > 2e-5
                       for k,v in bridge['pending_move'].items())
                or dynamics.get('pow_yaw_hold') != bridge['pow_yaw_hold']
                or (bridge['pow_yaw_hold'] and not np.isclose(dynamics.get('pow_yaw', float('nan')),
                                                             bridge['pow_yaw'], atol=2e-5, rtol=0))):
            self.checked.clear(); self.stats['checked_tail_mismatch'] += 1
            return False
        for key, tolerance in (('position', .001), ('velocity', .01), ('spin', .1), ('radius', 1e-6), ('gravity_direction', 1e-7), ('yaw', 1e-6)):
            av, ev = np.asarray(actual[key]), np.asarray(expected[key])
            if not np.all(np.isfinite(av)) or not np.all(np.isfinite(ev)) or np.max(np.abs(av-ev)) > tolerance:
                self.checked.clear(); self.stats['checked_tail_mismatch'] += 1
                return False
        same_parameters = (actual['parameters'].keys() == expected['parameters'].keys() and all(
            np.allclose(actual['parameters'][k], expected['parameters'][k], atol=1e-6, rtol=1e-7)
            for k in actual['parameters']))
        if (actual['held'] != expected['held'] or actual['effects'] != expected['effects'] or not same_parameters):
            self.checked.clear(); self.stats['checked_tail_mismatch'] += 1
            return False
        for key in ('inventory', 'blast'):
            exp = expected.get(key + '_context')
            obs = dynamics.get(key)
            if (exp is None) != (obs is None):
                self.checked.clear(); self.stats['checked_tail_mismatch'] += 1
                return False
            if exp is not None and obs is not None:
                fields = ('held_type', 'mega_active', 'permanent_mega') if key == 'inventory' else tuple(set(exp) | set(obs))
                if (any(k not in exp or k not in obs or not np.isfinite(float(exp[k]))
                        or not np.isfinite(float(obs[k])) or abs(float(exp[k])-float(obs[k])) > 2e-5 for k in fields)
                        or (key == 'inventory' and (exp.get('restore_type', 0) or obs.get('restore_type', 0)))):
                    self.checked.clear(); self.stats['checked_tail_mismatch'] += 1
                    return False
        self.committed[decision] = list(action)
        self.next = decision+1
        self.generation += 1; self.reset = 'checked_continuation'
        self.stats['checked_tail_replies'] += 1
        self.cv.notify_all()
        return True

    def progress_for(self):
        """(goal key, route distance) from the newest real observation the planner thread has measured."""
        with self.cv:
            return self.progress[1], self.progress[2]

    def _measure_progress(self, decision, actual):
        from nav.marble_dynamics.reliability_eval import goal_key
        if self.progress[0] == decision:
            return
        goal = self.controller.task_goal; distance = float('inf'); pos = actual.obs[:3]
        if goal is not None:
            self.progress_route.navigate(pos, goal[:3])
            distance = self.progress_route.distance(pos)
        with self.cv:
            self.progress = (decision, goal_key(goal) if goal is not None else None, distance)

    def stop(self):
        """Stop and WAIT for the thread: a planner thread still finishing a search when the next round's planner starts
        would share the world's engine and search workers with it."""
        with self.cv:
            self.running = False
            self.cv.notify_all()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(30.)
            if self.is_alive():
                raise RuntimeError('planner thread did not stop; refusing to reuse its simulator')

    # planner thread ---------------------------------------------------------------------------------------------
    def run(self):
        try:
            self._run()
        except Exception:
            with self.cv:
                self.error = traceback.format_exc()
                self.cv.notify_all()

    def budget(self, lead):
        """The planning budget for a decision `lead` decisions ahead of the game. The pool planner searches again on
        every decision while it follows a plan that only gets closer (reliability_eval.RESEARCH_EVERY 1), tries three
        first targets in a cluster (ORDER_CANDIDATES) and has a 2 s budget plus extensions: 78 ms a decision on average
        with 8 workers, more than the 64 ms real time allows (the first real-time round held the game 34 s in 80 s).
        Here the optional work happens only while the planner is at least SEARCH_LEAD decisions ahead; otherwise it
        follows its current plan (every plan it follows passed the physics checks) and a needed search (pickup, a plan
        that stopped holding) gets the time left before the game reaches that decision."""
        import nav.marble_dynamics.reliability_eval as RE
        import nav.marble_dynamics.search as S
        ahead = lead >= SEARCH_LEAD
        RE.RESEARCH_EVERY = 1 if ahead else 10 ** 6
        RE.ORDER_CANDIDATES = ORDER_CANDIDATES_AHEAD if ahead else 1
        RE.ORDER_EXTRA_S = 0.; S.CLUSTER_EXTRA_S = 0.; S.EXTRA_JUMP_S = .05
        self.world.args.search_budget_s = min(2., max(MIN_BUDGET_S, lead * DECISION_S - .02))
        self.world.args.stall_budget_s = self.world.args.search_budget_s
        self.stats['ahead' if ahead else 'following'] += 1

    def _run(self):
        from nav.protocol import NOOP_ACTION, RAW_TIME_LEFT
        from nav.marble_dynamics.bridge import reply
        while True:
            with self.cv:
                while self.running and not (self.latest is not None and self.next is not None
                                            and self.next - self.latest[0] <= self.lead):
                    self.cv.wait(.1)
                if not self.running:
                    return
                if self.reset:
                    checked_rebase = self.reset == 'checked_continuation'
                    self.reset = False
                    self.prev = None; self.tail = []; self.tail_from = None
                    self.controller.invalidate('realtime_checked_rebase' if checked_rebase else 'realtime_restart',
                                               reset_actor=True, preserve_task=checked_rebase)
                    self.stats['checked_rebases' if checked_rebase else 'restarts'] += 1
                v = self.next; a, actual, outcome = self.latest
                generation = self.generation
                if v < a:                                # the game passed this decision while we were busy
                    self.stats['skipped_late'] += a - v
                    self.next = v = a
                replies = [self.sent.get(d, self.committed.get(d)) for d in range(a, v)]
            self._measure_progress(a, actual)
            t0 = time.perf_counter()
            if any(r is None for r in replies):
                self.restart(a); continue
            if v == a:
                msg, picked, step = actual, {}, outcome
            else:
                try:
                    msg, picked = predict(self.engine, actual, replies)
                except ValueError:
                    self.stats['forecast_stopped'] += 1
                    with self.cv:                        # nothing to plan from yet: wait for the next real observation
                        while self.running and self.latest[0] == a and not self.reset:
                            self.cv.wait(.05)
                    continue
                step = dict(fell=False, round_ended=False, reconnected=False, gem_delta=picked.get(v - a - 1, 0.))
            if self.prev is not None:
                self.controller.observe(self.prev.obs, msg, step)
            remaining = max(1, int(float(msg.obs[RAW_TIME_LEFT]) / 1000. / DECISION_S))
            self.budget(v - a)
            action, info = self.controller.action(msg, remaining)
            label = info.get('label', '?')
            if label == 'neutral' and v > a:
                if CHECKED_TAIL:
                    # Wait for the real next group. Only a state-matched checked
                    # continuation may carry the marble through this boundary.
                    self.stats['waiting_for_new_gems'] += 1
                    with self.cv:
                        while self.running and self.latest[0] == a and not self.reset:
                            self.cv.wait(.05)
                    continue
                k = v - self.tail_from if self.tail_from is not None else -1
                if 0 <= k < len(self.tail):
                    action = list(self.tail[k]); label = 'plan_past_group'
                else:
                    last = self.committed.get(v - 1) or self.sent.get(v - 1)
                    if last is not None:
                        action = list(last); label = 'hold_until_new_gems'
            elif label != 'neutral' and self.controller.memory.actions:
                self.tail = [list(r) for r in self.controller.memory.remaining[1:]]; self.tail_from = v + 1
            self.prev = msg
            with self.cv:
                if self.next != v or self.generation != generation:
                    continue
                self.checked = {d: r for d, r in self.checked.items() if a <= d < v}
                forecast = getattr(self.controller, 'last_rollout', None)
                if CHECKED_TAIL and forecast is not None and list(action) == list(forecast['proposed'][0]):
                    from nav.marble_dynamics.bridge import translated
                    pending = msg.dynamics['bridge_action']
                    hold, aim = msg.dynamics['pow_yaw_hold'], msg.dynamics['pow_yaw']
                    for k in range(1, min(CHECKED_RESERVE+1, len(forecast['controls']))):
                        move, _, hold, aim = translated(pending, hold, aim)
                        pending = forecast['proposed'][k-1]
                        bridge = dict(pending_move=move, pow_yaw_hold=hold, pow_yaw=aim)
                        self.checked[v+k] = (list(forecast['proposed'][k]), copy.deepcopy(forecast['states'][k]),
                                             list(pending), bridge)
                self.committed[v] = action; self.chosen[v] = label; self.next = v + 1
                self.leads.append(v - a); self.latencies.append(time.perf_counter() - t0); self.labels[label] += 1
                self.cv.notify_all()


# ---------------------------------------------------------------------------------------------------- one real round
def setup_round(env, world_name=None):
    """Control words for a real-time round: real-time frames first, then 16 ms physics frames, a 64 ms decision per
    observation, lockstep, every frame drawn, 1x, exact marble state. Ends with any observation backlog skipped."""
    from nav.marble_dynamics.record import command
    env.python_paced = False
    paced = engine_pacing(env)
    env.speed = 1
    env.control('SPEED 1')
    env.control('FIXEDSTEP 16 64')
    env.control('LOCKSTEP 1')
    env.control('RENDEREVERY 1')
    env.control('VIEWYAW 0')
    command(env, 'DYNAMICSWAIT 600000', 'dynamics_wait')   # hold the game while a map's planner loads
    env.set_game_rules(False)
    command(env, 'DYNAMICSLIVE 1', 'dynamics_live')
    env.drain()
    env.python_paced = not paced


def engine_pacing(env):
    """REALTIMESTEP 1: the game itself runs each 16 ms frame on its own 16 ms of wall clock (marbleblast_realtime.exe,
    2026-10-19), from this control word on. True when this exe has it; otherwise the replies are paced here (PacedEnv)
    once the setup is done."""
    from nav.protocol import NOOP_ACTION
    env.debug = None
    env.control('REALTIMESTEP 1')
    for _ in range(32):
        env.drain()                  # the acknowledgement can sit behind stale observations
        if env.debug and env.debug[0] == 'realtime_step':
            return len(env.debug) > 1 and env.debug[1] == '1'
        if env.round_ended or env.reconnected:
            break
        env.step(NOOP_ACTION, repeat=1)
    log('this game exe has no real-time frames (REALTIMESTEP: start the game with play_realtime.ps1, '
        'marbleblast_realtime.exe); replies are paced here instead, so the picture moves in 64 ms steps')
    return False


def catch_up(env):
    """Skip the observations already waiting when we are about to reply, so the reply answers the newest one. In
    lockstep the game sends the next observation only after our reply, so a backlog means the game ran without us: a
    lockstep timeout, or (10-19) the frames it ran unpaced between two setup words while the server paced its replies,
    which left the server answering observations ~50 decisions old. Non-observation lines are handled as usual. Returns
    (observations skipped, their points, whether one of them fell); the newest becomes the pending observation."""
    import select
    from nav.protocol import parse_message
    skipped = 0; gems = 0.; fell = False
    while True:
        i = env.buf.find(b'\n')
        if i < 0:
            try:
                ready = select.select([env.conn], [], [], 0.0)[0]
                data = env.conn.recv(65536) if ready else None
            except OSError:
                data = None
            if not data:
                break                # nothing waiting (or a disconnect, which the next step notices)
            env.buf += data
            continue
        line, env.buf = env.buf[:i + 1], env.buf[i + 1:]
        m = parse_message(line.decode('utf-8', 'replace'))
        if m.kind == 'obs':
            skipped += 1; gems += max(0., float(m.gem_delta)); fell |= bool(m.oob); env.ticks += 1; env.msg = m
            continue
        env.recent_lines = (env.recent_lines + ['%d:%s:%s' % (env.ticks, m.kind, m.raw[:160])])[-20:]
        if m.kind == 'end':
            env.round_ended = True; env.last_round_score = -m.gem_delta
        elif m.kind == 'debug':
            env.debug = m.fields
        elif m.kind == 'stats':
            env.stats = m.fields
    return skipped, gems, fell


def wait_context(env, limit=400):
    from nav.protocol import NOOP_ACTION
    for _ in range(limit):
        d = env.msg.dynamics
        if isinstance(d, dict) and isinstance(d.get('native', {}).get('state'), dict) and d.get('game_context', {}).get('available'):
            return True
        env.step(NOOP_ACTION, repeat=1)
        if env.round_ended or env.reconnected:
            return False
    return False


def play_round(env, world, journal_folder):
    """One Hunt round in real time. Decisions are the game's observation ticks (control words and rescue waits consume
    ticks too). Returns a summary dict."""
    from nav.protocol import NOOP_ACTION
    from nav.marble_dynamics.bridge import reply
    from nav.marble_dynamics.reliability_eval import visible_gems, goal_key
    from nav.recovery import ProgressMonitor, simulation_time
    from nav.marble_dynamics.record import command
    controller = world.controller()
    planner = AheadPlanner(world, controller)
    planner.start()
    duration = float(env.info.get('round_ms', 0.)) / 1000.
    journal_folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    journal = (journal_folder / ('%s_%s.jsonl' % (world.name, stamp))).open('w', encoding='utf-8', buffering=1)
    monitor = ProgressMonitor(8.)
    points = 0.; falls = 0; rescues = 0; holds = []; started = time.perf_counter(); decisions = 0; backlog = 0
    playing = False; outcome = dict(fell=False, gem_delta=0., round_ended=False, reconnected=False)
    stop = 'round_end'
    command(env, 'DYNAMICSWAIT 5000', 'dynamics_wait')    # during play the game never holds longer than this
    try:
        while True:
            msg = env.msg; tick = int(msg.tick)
            countdown = duration > 0 and env.time_left_s() >= duration
            if not countdown and not playing:
                playing = True
                st = ((msg.dynamics if isinstance(msg.dynamics, dict) else {}).get('native') or {}).get('state') or {}
                spin = float(np.linalg.norm(st.get('spin', [0., 0., 0.]))) if st else 0.
                log('%s: GO at tick %d (speed %.2f u/s, spin %.1f rad/s from the pre-spin)' % (
                    world.name, tick, float(np.linalg.norm(np.asarray(msg.obs[3:5], dtype=float))), spin))
            held_s = 0.
            if playing:
                planner.see(tick, msg, outcome)
                action, held_s = planner.reply_for(tick)
                if held_s > .02:
                    holds.append(held_s)
                    if held_s > STALL_REPORT_S:
                        log('  held the game %.2f s at tick %d' % (held_s, tick))
            else:
                # Ready/Set: spin the marble up on its pad toward the first gem (ContinuousController.prespin; the planner
                # thread is idle until GO, so this thread may use the planner's route and collision code here)
                action, _ = controller.prespin(msg)
            # (1x pacing: the game runs each frame on its own slot of wall clock; without REALTIMESTEP every reply waits
            # for its 64 ms slot in PacedEnv._send)
            msg, outcome = env.step(action[:6], repeat=1, use_pow=int(action[6]), pow_yaw=action[7] if action[6] else None,
                                    use_blast=int(action[8]))
            skipped, skipped_points, skipped_fell = catch_up(env)
            if skipped:
                backlog += skipped
                log('  skipped %d stale observations at tick %s (the game ran ahead of the server)' % (skipped, env.msg.tick))
                outcome = dict(outcome, gem_delta=float(outcome['gem_delta']) + skipped_points,
                               fell=bool(outcome['fell'] or skipped_fell), round_ended=outcome['round_ended'] or env.round_ended)
            decisions += 1
            points += max(0., float(outcome['gem_delta']))
            if decisions % 400 == 0 and playing:
                recent = planner.latencies[-400:] or [0.]; lead = planner.leads[-400:] or [0]
                log('  %s: %d points, %.0f s left, held %d times (%.1f s), planner mean %.0f ms ahead %.1f decisions, %s' % (
                    world.name, points, env.time_left_s(), len(holds), sum(holds), 1000 * float(np.mean(recent)),
                    float(np.mean(lead)), dict(planner.stats)))
            journal.write(json.dumps(dict(tick=tick, action=list(map(float, action)), held_s=round(held_s, 4),
                                          gem_delta=outcome['gem_delta'], fell=outcome['fell'],
                                          time_left_s=env.time_left_s(),
                                          pos=[round(float(x), 3) for x in np.asarray(msg.obs[0:6], dtype=float)],   # 2026-10-21: position + velocity for the audits
                                          label=planner.last_label if playing else 'countdown')) + '\n')
            if outcome['round_ended'] or outcome['reconnected']:
                stop = 'round_end' if outcome['round_ended'] else 'reconnected'
                break
            if not playing:
                continue
            if outcome['fell']:
                # the legal out-of-bounds click (as in the pools), then plan again from the respawn
                falls += 1
                env.control('OOBCLICK')
                points += max(0., float(env.control_info['gem_delta']))
                planner.restart(int(env.msg.tick))
                outcome = dict(fell=False, gem_delta=0., round_ended=False, reconnected=False)
                if env.control_info['round_ended'] or env.control_info['reconnected']:
                    stop = 'round_end' if env.control_info['round_ended'] else 'reconnected'
                    break
                continue
            # this (game) thread never calls route or physics code: the planner thread measures the route distance
            raw = env.msg.obs
            key, distance = planner.progress_for()
            idle = monitor.observe(simulation_time(env), key, distance,
                                   picked=outcome['gem_delta'], fell=outcome['fell'], gems=[goal_key(g) for g in visible_gems(raw)])
            if idle >= monitor.window_s - 1e-9 and visible_gems(raw):
                # 8 s without progress: the legal quick respawn (as in the pools), then plan again
                context = env.msg.dynamics.get('game_context', {})
                if not (context.get('available') and context.get('can_quick_respawn') is False):
                    monitor.rescue_attempts += 1
                    accepted = env.quick_rescue()
                    if accepted:
                        rescues += 1
                        monitor.rescued(simulation_time(env), float('inf'))
                    else:
                        monitor.last_progress = simulation_time(env)
                    planner.restart(int(env.msg.tick))
                    points += max(0., float(env.control_info['gem_delta']))
                    outcome = dict(fell=False, gem_delta=0., round_ended=False, reconnected=False)
                    if env.control_info['round_ended'] or env.control_info['reconnected']:
                        stop = 'round_end' if env.control_info['round_ended'] else 'reconnected'
                        break
    finally:
        planner.stop()
        journal.close()
    leads = np.asarray(planner.leads) if planner.leads else np.zeros(1)
    lat = np.asarray(planner.latencies) if planner.latencies else np.zeros(1)
    summary = dict(map=world.name, stop=stop, points=points, score=env.last_round_score, falls=falls, rescues=rescues,
                   rescue_attempts=monitor.rescue_attempts, accepted_rescues=rescues,
                   decisions=decisions, skipped=backlog, wall_s=round(time.perf_counter() - started, 1), holds=len(holds),
                   held_s=round(float(sum(holds)), 2), max_hold_s=round(max(holds, default=0.), 2),
                   mean_lead=round(float(leads.mean()), 2), p50_plan_s=round(float(np.median(lat)), 3),
                   p99_plan_s=round(float(np.percentile(lat, 99)), 3), planner=dict(planner.stats),
                   labels=dict(planner.labels.most_common(8)), counts=dict(controller.counts.most_common(12)))
    log('ROUND ' + json.dumps(summary))
    from nav.collision.mesh import atomic_json
    atomic_json(journal_folder / ('%s_%s.summary.json' % (world.name, stamp)), summary)
    return summary


# --------------------------------------------------------------------------------------------------------- the server
def serve(port=PORT):
    import os
    os.environ.setdefault('NAV_RENDER_EVERY', '1')
    sys.setswitchinterval(.001)                    # the game thread gets the interpreter within ~1 ms of a reply being due
    import faulthandler
    faulthandler.enable(all_threads=True)          # a native crash prints every thread's Python stack to stderr
    from nav.env import HuntEnv

    class PacedEnv(HuntEnv):
        """python_paced (only with a game exe that lacks real-time frames; set at the end of setup_round): every reply
        waits for its 64 ms slot of wall clock; in lockstep each reply releases 64 ms of game time, so the round runs at
        1x, and after a hold (the planner behind, a map loading) the clock restarts: the game never runs fast to catch
        up. Otherwise (marbleblast_realtime.exe) the game holds each frame until its time and replies go out at once.
        Never paced before setup_round knows which: replies paced here while the game ran fixed frames unpaced between
        FIXEDSTEP and LOCKSTEP let it send ~50 observations ahead (10-19)."""
        clock = None
        python_paced = False

        def _send(self, line):
            if not self.python_paced:
                self.clock = None
                HuntEnv._send(self, line)
                return
            now = time.perf_counter()
            due = None if self.clock is None else self.clock + DECISION_S
            if due is not None and now < due:
                time.sleep(due - now); self.clock = due
            else:
                self.clock = now
            HuntEnv._send(self, line)

    worlds = WorldManager()
    env = PacedEnv(port=port, speed=1, log=log)
    log('waiting for a Hunt round on port %d (start the game, then Play > Hunt > a map)' % port)
    env.connect()
    env.drain()
    try:
        while True:
            conn0 = env.connections
            name = env.info.get('mission')
            if not name:
                env.request_info(); name = env.info.get('mission')
            setup_round(env)
            if env.connections != conn0:
                # 2026-10-25: the game reconnected during the setup (a new game process, possibly on another map): the name
                # read above came from the old game (TreasureBox planned a whole Fractured round: 22 points, 12 falls);
                # the new game may also have missed part of the setup, so it is sent again
                env.request_info(); name = env.info.get('mission') or name
                setup_round(env)
            try:
                world = worlds.get(name)
            except Exception:
                log('%s: cannot plan this map:\n%s' % (name, traceback.format_exc()))
                raise
            if not wait_context(env):
                log('no live marble state; waiting for the next round')
            else:
                play_round(env, world, OUTPUT / time.strftime('%Y%m%d'))
            env.wait_new_round()
    except KeyboardInterrupt:
        pass
    finally:
        worlds.close()
        env.close()


if __name__ == '__main__':
    serve()
