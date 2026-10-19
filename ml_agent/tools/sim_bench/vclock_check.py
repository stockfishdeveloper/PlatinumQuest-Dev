"""Candidates judged within a 2 s budget: real clock vs the sim's virtual clock (0.85 ms per physics step).
usage (repo root, PYTHONPATH=ml_agent): python vclock_check.py <pool> <map> <decision> [...]"""
import gzip, json, copy, sys, time
import numpy as np
from pathlib import Path
from nav.terrain import TerrainMap
from nav.routing import RoutePlanner, goal_key
from nav.marble_dynamics.static_cache import cached_terrain, cached_walking, load_queries
from nav.collision.experimental_sweep import CachedCompiledSweepMap
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.physics import GeneralPhysics
import nav.marble_dynamics.search as S
from nav.marble_dynamics.crossings import Crossings
from nav.marble_dynamics.navigator import FrozenNavigator
from nav.marble_dynamics.reliability_eval import tour
CACHE = Path(r'C:\Users\doug\AppData\Local\Temp\pq_checks\fast_nav_static_cache_20261013')
DLL = Path(r'C:\Users\doug\AppData\Local\Temp\pq_checks\compiled_sweep_20261012\sweep_kernel.dll')
pool, name, decs = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
terrain = cached_terrain(TerrainMap.resolve(name), CACHE)
mesh = CachedCompiledSweepMap(ML / 'collision_maps' / (name + '.npz'), DLL); load_queries(mesh, DLL, CACHE)
walk = cached_walking(terrain, mesh, DLL, CACHE)
engine = GeneralPhysics(mesh); engine.bridge_physics_ms = 16; engine.use_horizon = 48; engine.use_margin_u = 8.
engine.ballistic_landing = True; engine.pickup_landing_replies = 40; engine.detour_blast_ratio = 1.5
engine.crossings = Crossings(walk, mesh=mesh); engine.blast_value_u = 10.; engine.use_completed_candidates_on_timeout = True
nav = FrozenNavigator(terrain, ML / 'models/nav/real_maps_20261010_A2/nav_latest.pth')
VC = [1000.]; STEPS = [0]; real_step = engine.step
def step(*a, **k):
    STEPS[0] += 1; VC[0] += .00085; return real_step(*a, **k)
engine.step = step
R = {}
for l in gzip.open(Path('ml_agent/dynamics_reports') / pool / name / '7347019/physics_plan/1/decisions.jsonl.gz', 'rt', encoding='utf-8'):
    try: d = json.loads(l); R[d['decision']] = d
    except Exception: pass
real_pc = time.perf_counter
for k in decs:
    d = R[k]; base = d['before']; raw = np.asarray(d['raw_observation'], np.float32); dest = d['choice']['goal_context']['destination']
    tgt, nxt, _ = tour(terrain, {}, raw, list(dest) + [0, 0])
    engine.next_for = {goal_key(tgt): nxt} if tgt is not None and nxt is not None and goal_key(tgt) == goal_key(dest) else {}
    for mode in ('real', 'virtual'):
        if mode == 'virtual': time.perf_counter = lambda: VC[0]
        else: time.perf_counter = real_pc
        s0 = STEPS[0]; w0 = real_pc()
        engine.planning_deadline = time.perf_counter() + 2.
        chosen, pairs = S.search(engine, RoutePlanner(walk), copy.deepcopy(base), raw, np.asarray(dest[:3]), nav, 24, brake_replies=0)
        time.perf_counter = real_pc
        print('dec %d %-7s: %2d candidates, %5d steps, wall %.2f s, chosen %s' % (k, mode, len(pairs), STEPS[0] - s0, real_pc() - w0, chosen[0]['label'] if chosen else None), flush=True)
