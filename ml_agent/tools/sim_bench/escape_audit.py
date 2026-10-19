"""Stuck periods: at states sampled from recorded no-pickup stretches (> 8 s, target gem < 20 u away), does a 2 s
(virtual clock) search find a pickup plan? base search.py vs a variant module.
usage (repo root, PYTHONPATH=ml_agent): python escape_audit.py <variant search.py> <pool> <map> [every N decisions] [max states]"""
import gzip, json, copy, sys, time, ctypes, importlib.util
import numpy as np
from pathlib import Path
from collections import Counter
_k = ctypes.windll.kernel32; _k.GetCurrentProcess.restype = ctypes.c_void_p; _k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_k.SetPriorityClass(_k.GetCurrentProcess(), 0x40)
variant_path, pool, name = sys.argv[1], sys.argv[2], sys.argv[3]
every = int(sys.argv[4]) if len(sys.argv) > 4 else 20; cap = int(sys.argv[5]) if len(sys.argv) > 5 else 40
import nav.marble_dynamics.search as BASE
spec = importlib.util.spec_from_file_location('variant_search', variant_path); VAR = importlib.util.module_from_spec(spec); spec.loader.exec_module(VAR)
VAR.RAISED_BLAST_FIRST = True if hasattr(VAR, 'RAISED_BLAST_FIRST') else None
from nav.terrain import TerrainMap
from nav.routing import RoutePlanner, goal_key
from nav.marble_dynamics.static_cache import cached_terrain, cached_walking, load_queries
from nav.collision.experimental_sweep import CachedCompiledSweepMap
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.physics import GeneralPhysics
from nav.marble_dynamics.crossings import Crossings
from nav.marble_dynamics.navigator import FrozenNavigator
from nav.marble_dynamics.reliability_eval import tour
CACHE = Path(r'C:\Users\doug\AppData\Local\Temp\pq_checks\fast_nav_static_cache_20261013')
DLL = Path(r'C:\Users\doug\AppData\Local\Temp\pq_checks\compiled_sweep_20261012\sweep_kernel.dll')
terrain = cached_terrain(TerrainMap.resolve(name), CACHE)
mesh = CachedCompiledSweepMap(ML / 'collision_maps' / (name + '.npz'), DLL); load_queries(mesh, DLL, CACHE)
walk = cached_walking(terrain, mesh, DLL, CACHE)
engine = GeneralPhysics(mesh); engine.bridge_physics_ms = 16; engine.use_horizon = 48; engine.use_margin_u = 8.
engine.ballistic_landing = True; engine.pickup_landing_replies = 40; engine.detour_blast_ratio = 1.5
engine.crossings = Crossings(walk, mesh=mesh); engine.blast_value_u = 10.; engine.use_completed_candidates_on_timeout = True
nav = FrozenNavigator(terrain, ML / 'models/nav/real_maps_20261010_A2/nav_latest.pth')
VC = [1000.]; real_step = engine.step
def step(*a, **k):
    VC[0] += .00085; return real_step(*a, **k)
engine.step = step
R = [json.loads(l) for l in gzip.open(Path('ml_agent/dynamics_reports') / pool / name / '7347019/physics_plan/1/decisions.jsonl.gz', 'rt', encoding='utf-8') if l.strip()]
R.sort(key=lambda d: d['decision'])
# no-pickup stretches > 8 s without a fall
picks = [i for i, d in enumerate(R) if float(d['outcome'].get('gem_delta') or 0) > 0 or d['outcome'].get('fell')]
states = []
for a, b in zip([0] + picks, picks + [len(R) - 1]):
    if (b - a) * .064 <= 8: continue
    for i in range(a + 1, b, every):
        d = R[i]; st = d['before']['native']['state']; gc = d['choice'].get('goal_context') or {}
        if st['mode'] != 0 or st['oob'] or gc.get('kind') != 'gem' or d['choice'].get('label') == 'countdown_neutral': continue
        if np.linalg.norm(np.asarray(gc['destination'][:2]) - np.asarray(st['position'][:2])) > 20: continue
        states.append(d)
states = states[:cap]
real_pc = time.perf_counter; c = Counter()
for d in states:
    base = d['before']; raw = np.asarray(d['raw_observation'], np.float32); dest = d['choice']['goal_context']['destination']
    tgt, nxt, _ = tour(terrain, {}, raw, list(dest) + [0, 0])
    engine.next_for = {goal_key(tgt): nxt} if tgt is not None and nxt is not None and goal_key(tgt) == goal_key(dest) else {}
    res = {}
    for label, mod in (('base', BASE), ('variant', VAR)):
        time.perf_counter = lambda: VC[0]; engine.planning_deadline = VC[0] + 2.
        try:
            chosen, pairs = mod.search(engine, RoutePlanner(walk), copy.deepcopy(base), raw, np.asarray(dest[:3]), nav, 24, brake_replies=0)
        finally:
            time.perf_counter = real_pc
        res[label] = chosen is not None and chosen[0].get('possible_pickup_interval') is not None
    c['states'] += 1; c['base_pickup'] += res['base']; c['variant_pickup'] += res['variant']
    c['gained'] += res['variant'] and not res['base']; c['lost'] += res['base'] and not res['variant']
print('%s %s: %s' % (name, pool[8:20], dict(c)), flush=True)
