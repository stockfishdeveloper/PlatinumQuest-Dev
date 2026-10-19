"""Parallel candidate evaluation for the physics search (2026-10-19, operator: "speed up the search using whatever levers
you need", "you are allowed to use more cores"; the goal is real-time play).

search() forecasts and scores 30-60 candidate plans per decision, and each candidate is independent of the others (every
proposal carries its own copy of the route). Profile 10-19 (Cragmire / Sprawl, real clock): ~90 % of a search is Python
around the compiled forecasts (state unpacking, scoring, routing, the learned model's per-reply rollout), so threads would
not help; separate processes do. A pool of worker processes, each holding its own copy of the map, the collision mesh,
the compiled simulator and the learned model, evaluates the candidates; the parent merges the results in the serial
order, so with no deadline search() returns exactly what the serial loop returns.

Big shared objects travel by reference: a pickler hook names them and each worker resolves the names to its own
copies. Large arrays (route fields, 2-3 MB each) are sent to a worker once and cached there under a content key.
Nothing here reads information a player could not have: the workers see exactly the inputs the serial search sees.

usage: pool = ParallelEvaluator(n, init) where init = worker_init(args, map_name, overrides); engine.parallel = pool.
"""
import io
import math
import os
import pickle
import queue
import sys
import threading
import time
import traceback
import zlib
import multiprocessing as mp
from multiprocessing.connection import wait as _wait

import numpy as np

BIG_ARRAY = 64 * 1024                 # bytes: arrays at least this big go through the per-worker cache
SLOW_PROPOSALS = ('NavigatorProposal', 'TrackLine', 'LineProposal')   # dispatched first (per-reply Python controllers)
ENGINE_SKIP = {'mesh', 'native_sim', 'contact_order', 'crossings', 'polygons', 'polygon_ids', 'native_step_hook', 'step',
               'parallel', 'safe_hold', 'order_stats', 'order_pick', 'planning_deadline', '_speculated'}
SPECULATE_PROJECTIONS = 'coarse'      # landing work of airborne candidates run ahead in the workers: 'coarse' = the coarse
                                      # parabola only; 'full' = also the exact projection and its score (exact too, but the
                                      # parent then unpickles a fully built landed rollout per airborne candidate, since
                                      # viability.concatenate materializes the states: Cragmire 10-19 episode 1 94 -> 113 s);
                                      # None = off
GLOBAL_MODULES = ('nav.marble_dynamics.search', 'nav.marble_dynamics.gem_demo', 'nav.marble_dynamics.reliability_eval',
                  'nav.marble_dynamics.racing_line', 'nav.marble_dynamics.crossings', 'nav.marble_dynamics.ballistic',
                  'nav.marble_dynamics.viability', 'nav.marble_dynamics.landing', 'nav.marble_dynamics.blast',
                  'nav.marble_dynamics.resources', 'nav.marble_dynamics.bridge')
_PLAIN = (bool, int, float, str, type(None))


def _plain(v, depth=0):
    if isinstance(v, _PLAIN):
        return True
    if depth > 3:
        return False
    if isinstance(v, (tuple, list)):
        return all(_plain(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, _PLAIN) and _plain(x, depth + 1) for k, x in v.items())
    return False


def module_globals():
    """UPPER_CASE plain module constants of the planner modules (the values in force in this process, --set included)."""
    out = {}
    for name in GLOBAL_MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        out[name] = {k: v for k, v in vars(mod).items() if k.isupper() and not k.startswith('_') and _plain(v)}
    return out


# ------------------------------------------------------------------------------------------------- reference pickling
class _Keys:
    """Content keys of the big arrays referenced in one search (id -> key, valid while the arrays are alive)."""
    def __init__(self):
        self.by_id = {}; self.arrays = {}

    def key(self, a):
        k = self.by_id.get(id(a))
        if k is None:
            c = np.ascontiguousarray(a)
            k = ('arr', c.shape, c.dtype.str, zlib.crc32(memoryview(c).cast('B')))
            self.by_id[id(a)] = k; self.arrays[k] = c
        return k


_RESOLVE = [None, None]       # the receiving process's (refs by name, arrays by key) while ref_loads runs


def _ref(name):
    return _RESOLVE[0][name]


def _arr(key):
    return _RESOLVE[1][key]


class RefPickler(pickle.Pickler):
    """Bound objects travel as their names, big arrays as content keys. reducer_override (not persistent_id) because
    the C pickler skips it for plain ints, floats, strings and containers: persistent_id ran ~120 times per candidate
    (10.5 M calls, 6.7 s of the parent's 114 s in a Cragmire episode, 10-19)."""
    def __init__(self, f, refs, keys):
        super().__init__(f, protocol=pickle.HIGHEST_PROTOCOL)
        self._refs = refs; self._keys = keys

    def reducer_override(self, obj):
        name = self._refs.get(id(obj))
        if name is not None:
            return _ref, (name,)
        if isinstance(obj, np.ndarray) and obj.nbytes >= BIG_ARRAY and obj.dtype != object:
            return _arr, (self._keys.key(obj),)
        return NotImplemented


def ref_dumps(obj, refs, keys):
    f = io.BytesIO(); RefPickler(f, refs, keys).dump(obj); return f.getvalue()


def ref_loads(data, refs, arrays):
    _RESOLVE[0] = refs; _RESOLVE[1] = arrays
    try:
        return pickle.loads(data)
    finally:
        _RESOLVE[0] = _RESOLVE[1] = None


# --------------------------------------------------------------------------------------------- one candidate (worker)
def evaluate_candidate(engine, route, dynamics, gem, label, proposal, target_item, steps, landing_replies, ballistic,
                       brake_replies):
    """The serial search loop's body for one candidate (search.search), without the deadline: the main forecast and
    score, the landing continuation, the brake tail. Returns a dict the parent merges in the serial order."""
    from nav.marble_dynamics.bridge import predict_replies
    from nav.marble_dynamics.search import assess, LandingContinuation, BrakeContinuation
    try:
        rollout = predict_replies(engine, dynamics, proposal, stop_on_ambiguity=True)
    except ValueError as exc:
        return dict(kind='unsupported', pair=(dict(label=label, rejected='unsupported_proposal', failure=str(exc), cost=math.inf), None))
    info = assess(engine, route, dynamics, gem, rollout, label, target_item)
    info.update(evaluated_replies=len(rollout['controls']), requested_replies=rollout['requested_replies'],
                stopped_on_ambiguity=rollout['stopped_on_ambiguity'])
    if 'use' in label and rollout['requested_replies'] > steps and info['rejected'] is None and info['possible_pickup_interval'] is None:
        penalty = (rollout['requested_replies'] - steps) * .064 * float(getattr(engine, 'ballistic_time_rate', 8.) or 8.)
        info['cost'] += penalty; info['time_penalty_u'] = penalty
    out = dict(kind='ok', main=(info, rollout), air_main=bool(ballistic and info['rejected'] == 'no_supported_continuation'),
               landing=None, air_landing=False, brake=None)
    pickup_landing = int(getattr(engine, 'pickup_landing_replies', 0) or 0)
    collects = info.get('possible_pickup_interval') is not None
    extend = (landing_replies or (pickup_landing and collects)) and info['rejected'] == 'no_supported_continuation'
    if extend:
        replies = landing_replies or pickup_landing
        landed = predict_replies(engine, dynamics, LandingContinuation(rollout['proposed'], route, gem, replies), stop_on_ambiguity=True)
        checked = assess(engine, route, dynamics, gem, landed, label + '_landing', target_item)
        checked.update(proposal_replies=len(rollout['proposed']), landing_replies=replies, pickup_only_landing=not landing_replies,
                       evaluated_replies=len(landed['controls']), requested_replies=landed['requested_replies'],
                       stopped_on_ambiguity=landed['stopped_on_ambiguity'], forecast_s=(len(landed['states']) - 1) * .064)
        out['landing'] = (checked, landed)
        out['air_landing'] = bool(ballistic and collects and checked['rejected'] == 'no_supported_continuation')
    if SPECULATE_PROJECTIONS:
        # search.project_airborne's per-candidate work done ahead: the coarse landing and, for a floor-like one, the exact
        # projection and its score. Deterministic, so the parent's later selection (the best few by landing distance)
        # reads exactly the values it would have computed itself.
        full = SPECULATE_PROJECTIONS == 'full'
        if out['air_main']:
            out['proj_main'] = speculate_projection(engine, route, dynamics, gem, target_item, label, rollout, full)
        if out['air_landing']:
            out['proj_landing'] = speculate_projection(engine, route, dynamics, gem, target_item, label + '_landing', out['landing'][1], full)
    if brake_replies and info['rejected'] == 'no_supported_continuation':
        extended = predict_replies(engine, dynamics, BrakeContinuation(rollout['proposed'], gem, brake_replies), stop_on_ambiguity=True)
        checked = assess(engine, route, dynamics, gem, extended, label + '_brake_tail', target_item)
        checked.update(proposal_replies=len(rollout['proposed']), braking_replies=brake_replies,
                       evaluated_replies=len(extended['controls']), requested_replies=extended['requested_replies'],
                       stopped_on_ambiguity=extended['stopped_on_ambiguity'], forecast_s=(len(extended['states']) - 1) * .064)
        out['brake'] = (checked, extended)
    return out


def speculate_projection(engine, route, dynamics, gem, target_item, label, rollout, full=False):
    from nav.marble_dynamics.ballistic import coarse_landing
    from nav.marble_dynamics.search import project_exact
    coarse = coarse_landing(engine.mesh, rollout['states'][-1], getattr(engine, 'mission_bounds', None))
    exact = (project_exact(engine, route, dynamics, gem, target_item, label, rollout)
             if full and coarse['hit'] and coarse['floor_like'] else None)
    return dict(coarse=coarse, exact=exact)


# -------------------------------------------------------------------------------------------------- worker process
def worker_init(args, map_name, overrides=(), setup=None):
    """What a worker needs to build the same planner world as the parent: the planner args, the map, the module
    override files the parent loaded (tools/sim_bench --module) and optionally an importable setup function
    'module:function' (default: build_engine below, the copy of reliability_eval.main()'s per-map setup)."""
    a = {}
    for k, v in vars(args).items():
        if k == 'geometry_terrain':
            continue
        try:
            pickle.dumps(v); a[k] = v
        except Exception:
            pass
    return dict(args=a, map=map_name, overrides=list(overrides), setup=setup, globals=module_globals(), sys_path=list(sys.path))


def build_engine(a, name):
    """reliability_eval.main()'s per-map setup (terrain, mesh, compiled simulator, walking terrain, crossings, bounds)."""
    import json
    from nav.terrain import TerrainMap, TerrainGrid
    from nav.marble_dynamics.record import ML
    from nav.marble_dynamics.physics import GeneralPhysics
    from nav.collision.experimental_sweep import CompiledSweepMap, CachedCompiledSweepMap
    from nav.collision.mesh import CollisionMap
    from nav.marble_dynamics.query_cache import CachedCollisionMap
    from nav.marble_dynamics.geometry_route import walking_terrain
    from nav.marble_dynamics.ground_cache import MemoGroundClearance
    if a.cache_dir:
        from nav.marble_dynamics.static_cache import cached_terrain, cached_walking, load_queries
        terrain = cached_terrain(TerrainMap.resolve(name), a.cache_dir)
    else:
        terrain = TerrainGrid(TerrainMap.resolve(name))
    path = ML / 'collision_maps' / (name + '.npz')
    mesh = (CachedCompiledSweepMap if a.query_cache else CompiledSweepMap)(path, a.sweep_dll) if a.sweep_dll else \
        (CachedCollisionMap if a.query_cache else CollisionMap)(path)
    if a.cache_dir:
        load_queries(mesh, a.sweep_dll, a.cache_dir)
    engine = GeneralPhysics(mesh)
    from nav.native import attach
    attach(engine)
    geometry = None
    if a.geometry_route:
        geometry = cached_walking(terrain, mesh, a.sweep_dll, a.cache_dir, memo=a.clearance_cache) if a.cache_dir else walking_terrain(terrain, engine.mesh)
        if a.clearance_cache and not a.cache_dir:
            geometry.ground_clearance = MemoGroundClearance(geometry.ground_clearance)
        engine.native_sim.bind_clearance(geometry)
    engine.crossings = None
    if getattr(a, 'component_crossing', False):
        from nav.marble_dynamics.crossings import Crossings
        engine.crossings = Crossings(geometry if a.geometry_route else terrain, mesh=engine.mesh, climbs=getattr(a, 'climb_crossings', False))
    try:
        from nav.marble_dynamics.fast_nav import mission_inbounds
        engine.mission_bounds = mission_inbounds(json.loads((ML / 'collision_maps' / (name + '.json')).read_text()))
    except (ValueError, OSError):
        engine.mission_bounds = None
    return terrain, geometry, engine


def _load_overrides(overrides):
    import importlib.util
    for name, path in overrides:
        full = 'nav.marble_dynamics.' + name
        spec = importlib.util.spec_from_file_location(full, path); mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod; spec.loader.exec_module(mod)


def _worker_main(cmd, res, init):
    try:
        sys.path[:0] = [p for p in init['sys_path'] if p not in sys.path]
        _load_overrides(init['overrides'])
        import argparse
        a = argparse.Namespace(**init['args'])
        if init.get('setup'):
            import importlib
            mod, fn = init['setup'].split(':'); terrain, geometry, engine = getattr(importlib.import_module(mod), fn)(a, init['map'])
        else:
            terrain, geometry, engine = build_engine(a, init['map'])
        a.geometry_terrain = geometry
        from nav.marble_dynamics.navigator import FrozenNavigator
        model = FrozenNavigator(terrain, a.checkpoint).model
        for name, values in init['globals'].items():
            mod = sys.modules.get(name)
            if mod is None:
                __import__(name); mod = sys.modules[name]
            for k, v in values.items():
                setattr(mod, k, v)
        engine.planning_deadline = None
        refs = dict(terrain=terrain, geometry=geometry, mesh=engine.mesh, sim=engine.native_sim, model=model,
                    crossings=engine.crossings, contact_order=getattr(engine, 'contact_order', None))
        refs = {k: v for k, v in refs.items() if v is not None}
        arrays = {}; ctx = None; ctx_id = None
        res.send_bytes(pickle.dumps(('ready', os.getpid())))
        while True:
            msg = pickle.loads(cmd.recv_bytes())
            kind = msg[0]
            if kind == 'stop':
                break
            if kind == 'arrays':
                arrays.update(msg[1]); continue
            if kind == 'ctx':
                ctx_id = msg[1]; ctx = ref_loads(msg[2], refs, arrays)
                for k, v in ctx['engine'].items():
                    setattr(engine, k, v)
                engine.planning_deadline = None
                continue
            if kind == 'cand':
                _, sid, idx, data = msg
                try:
                    if sid != ctx_id:
                        raise RuntimeError('candidate for an unknown search context')
                    t_c = time.perf_counter()
                    label, proposal = ref_loads(data, refs, arrays)
                    r = evaluate_candidate(engine, ctx['route'], ctx['dynamics'], ctx['gem'], label, proposal, ctx['target_item'],
                                           ctx['steps'], ctx['landing_replies'], ctx['ballistic'], ctx['brake_replies'])
                    r['_t'] = time.perf_counter() - t_c   # worker seconds (timing statistics only)
                    res.send_bytes(pickle.dumps(('res', sid, idx, r), protocol=pickle.HIGHEST_PROTOCOL))
                except Exception:
                    res.send_bytes(pickle.dumps(('err', sid, idx, traceback.format_exc())))
    except Exception:
        try:
            res.send_bytes(pickle.dumps(('fatal', traceback.format_exc())))
        except Exception:
            pass


# --------------------------------------------------------------------------------------------------- parent side
class _Sender(threading.Thread):
    """Writes one worker's command pipe. A Windows pipe holds 8 KB, so a larger message blocks its writer until the
    worker reads it; a worker busy with a candidate reads nothing until it has written that result, which waits for the
    parent to read. With the parent's main thread doing both, a big candidate sent to a busy worker deadlocked the pool
    (King of the Marble, 10-19). The main thread now only queues commands and reads results."""

    def __init__(self, conn):
        super().__init__(daemon=True)
        self.conn = conn; self.q = queue.SimpleQueue(); self.error = None

    def run(self):
        while True:
            data = self.q.get()
            if data is None:
                return
            try:
                self.conn.send_bytes(data)
            except Exception as exc:          # a dead worker: reported at the next send
                self.error = exc
                return

    def send(self, data):
        if self.error is not None:
            raise RuntimeError('parallel search worker pipe failed: %r' % (self.error,))
        self.q.put(data)


class ParallelEvaluator:
    """A pool of candidate-evaluation workers for one map. evaluate() returns the per-candidate results in order."""

    def __init__(self, n, init, timeout_s=600.):
        ctx = mp.get_context('spawn')
        self.workers = []; self.conns = []; self.senders = []; self.sent = []; self.pids = []
        # Spawned children re-run the parent's __main__ script unless it is hidden (tools/sim_bench parses argv and
        # swaps the clock at import): workers need only this module.
        main = sys.modules.get('__main__'); saved = (getattr(main, '__spec__', None), getattr(main, '__file__', None))
        try:
            if main is not None:
                main.__spec__ = None
                if hasattr(main, '__file__'):
                    del main.__file__
            for _ in range(int(n)):
                cmd_r, cmd_w = ctx.Pipe(duplex=False)        # parent -> worker
                res_r, res_w = ctx.Pipe(duplex=False)        # worker -> parent
                p = ctx.Process(target=_worker_main, args=(cmd_r, res_w, init), daemon=True)
                p.start(); cmd_r.close(); res_w.close()
                sender = _Sender(cmd_w); sender.start()
                self.workers.append(p); self.conns.append(res_r); self.senders.append(sender); self.sent.append(set())
        finally:
            if main is not None:
                main.__spec__ = saved[0]
                if saved[1] is not None:
                    main.__file__ = saved[1]
        t0 = time.perf_counter()
        for c in self.conns:
            if not c.poll(max(1., timeout_s - (time.perf_counter() - t0))):
                raise RuntimeError('parallel search worker did not start')
            msg = pickle.loads(c.recv_bytes())
            if msg[0] != 'ready':
                raise RuntimeError('parallel search worker failed:\n' + str(msg[1]))
            self.pids.append(msg[1])
        self.search_id = 0; self.refs = {}
        self.stats = dict(searches=0, candidates=0, wall_s=0., work_s=0., longest_s=0., dispatch_s=0.)

    def bind(self, refs):
        """Parent objects that travel by name: {name: object} (terrain, geometry, mesh, sim, model, crossings)."""
        self.refs = {id(v): k for k, v in refs.items() if v is not None}

    def close(self):
        for s_ in self.senders:
            try:
                s_.send(pickle.dumps(('stop',))); s_.q.put(None)
            except Exception:
                pass
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def evaluate(self, engine, route, dynamics, gem, specs, target_item, steps, landing_replies, ballistic, brake_replies,
                 deadline=None):
        """Results of evaluate_candidate for every (label, proposal) in specs, in order; None for a candidate whose
        result did not arrive before `deadline` (time.perf_counter() value; None = wait for all)."""
        t0 = time.perf_counter(); self.search_id += 1; sid = self.search_id
        keys = _Keys()
        state = {k: v for k, v in engine.__dict__.items() if k not in ENGINE_SKIP and not callable(v)}
        ctx = dict(engine=state, route=route, dynamics=dynamics, gem=gem, target_item=target_item, steps=steps,
                   landing_replies=landing_replies, ballistic=ballistic, brake_replies=brake_replies)
        ctx_bytes = ref_dumps(ctx, self.refs, keys); t_pickled = None
        cand_bytes = [ref_dumps(sp, self.refs, keys) for sp in specs]
        for i, s_ in enumerate(self.senders):
            missing = {k: a for k, a in keys.arrays.items() if k not in self.sent[i]}
            if missing:
                s_.send(pickle.dumps(('arrays', missing), protocol=pickle.HIGHEST_PROTOCOL)); self.sent[i].update(missing)
            s_.send(pickle.dumps(('ctx', sid, ctx_bytes), protocol=pickle.HIGHEST_PROTOCOL))
        # dispatch order (the merge keeps the serial order): the slow candidates first, so they do not finish alone at the
        # end of the search. The learned model's proposal runs per reply in Python (~10x a compiled candidate).
        slow = [k for k, sp in enumerate(specs) if type(sp[1]).__name__ in SLOW_PROPOSALS]
        order = slow + [k for k in range(len(specs)) if k not in set(slow)]
        results = [None] * len(specs); todo = order[::-1]; busy = {}
        # two candidates in flight per worker hide the pipe latency; the next one is sent as each result comes back
        for _ in range(2):
            for i, s_ in enumerate(self.senders):
                if todo:
                    k = todo.pop(); s_.send(pickle.dumps(('cand', sid, k, cand_bytes[k]), protocol=pickle.HIGHEST_PROTOCOL))
                    busy.setdefault(i, []).append(k)
        pending = sum(len(v) for v in busy.values()); t_sent = time.perf_counter()
        conn_index = {id(c): i for i, c in enumerate(self.conns)}
        while pending:
            timeout = None if deadline is None else max(0., deadline - time.perf_counter())
            ready = _wait(self.conns, timeout)
            if not ready:
                break
            for c in ready:
                msg = pickle.loads(c.recv_bytes()); i = conn_index[id(c)]
                if msg[0] == 'fatal':
                    raise RuntimeError('parallel search worker died:\n' + msg[1])
                _, rsid, k, payload = msg
                if rsid != sid:
                    continue
                pending -= 1; busy[i].remove(k)
                if msg[0] == 'err':
                    raise RuntimeError('parallel candidate %r failed:\n%s' % (specs[k][0], payload))
                results[k] = payload
                if todo and (deadline is None or time.perf_counter() < deadline):
                    j = todo.pop(); self.senders[i].send(pickle.dumps(('cand', sid, j, cand_bytes[j]), protocol=pickle.HIGHEST_PROTOCOL))
                    busy[i].append(j); pending += 1
        # results that missed the deadline (at most two per worker) arrive later and are skipped by their search id
        self.stats['searches'] += 1; self.stats['candidates'] += len(specs); self.stats['wall_s'] += time.perf_counter() - t0
        self.stats['dispatch_s'] += t_sent - t0
        times = [r.get('_t', 0.) for r in results if r is not None]
        if times:
            self.stats['work_s'] += sum(times); self.stats['longest_s'] += max(times)
        return results
