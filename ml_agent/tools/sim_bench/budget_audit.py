"""Which candidate families does the 2 s deadline starve? At sampled recorded states: search with the virtual 2 s budget
(0.85 ms per physics step) vs unbudgeted; report how often the unbudgeted best is a pickup plan the budgeted search missed,
and that plan's family. usage (repo root, PYTHONPATH=ml_agent): python budget_audit.py <pool> <map> [N]"""
import gzip, json, copy, sys, time, ctypes
import numpy as np
from pathlib import Path
from collections import Counter
_k = ctypes.windll.kernel32; _k.GetCurrentProcess.restype = ctypes.c_void_p; _k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_k.SetPriorityClass(_k.GetCurrentProcess(), 0x40)
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
pool, name = sys.argv[1], sys.argv[2]; per = int(sys.argv[3]) if len(sys.argv) > 3 else 20
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
real_pc = time.perf_counter
R = [json.loads(l) for l in gzip.open(Path('ml_agent/dynamics_reports') / pool / name / '7347019/physics_plan/1/decisions.jsonl.gz', 'rt', encoding='utf-8') if l.strip()]
R = [d for d in R if (d['choice'].get('goal_context') or {}).get('kind') == 'gem' and d['before']['native']['state']['mode'] == 0
     and not d['before']['native']['state']['oob'] and 'cost' in d['choice']]
R.sort(key=lambda d: d['decision']); stepn = max(1, len(R) // per)
fam = lambda l: 'none' if not l else ('blast' if 'blast' in l else 'use' if 'use_' in l else 'jump' if 'jump' in l else 'through' if 'through' in l else 'roll')
c = Counter(); missed = Counter(); diffs = []
for d in R[::stepn][:per]:
    base = d['before']; raw = np.asarray(d['raw_observation'], np.float32); dest = d['choice']['goal_context']['destination']
    tgt, nxt, _ = tour(terrain, {}, raw, list(dest) + [0, 0])
    engine.next_for = {goal_key(tgt): nxt} if tgt is not None and nxt is not None and goal_key(tgt) == goal_key(dest) else {}
    out = {}
    for mode in ('budget', 'full'):
        time.perf_counter = (lambda: VC[0]) if mode == 'budget' else real_pc
        engine.planning_deadline = time.perf_counter() + 2. if mode == 'budget' else None
        chosen, pairs = S.search(engine, RoutePlanner(walk), copy.deepcopy(base), raw, np.asarray(dest[:3]), nav, 24, brake_replies=0)
        time.perf_counter = real_pc
        out[mode] = (chosen[0]['label'] if chosen else None, chosen[0]['cost'] if chosen else None, chosen[0].get('possible_pickup_interval') if chosen else None, len(pairs))
    c['states'] += 1
    b, f = out['budget'], out['full']
    if f[2] is not None and b[2] is None:
        c['pickup_missed'] += 1; missed[fam(f[0])] += 1
    if b[1] is not None and f[1] is not None: diffs.append(b[1] - f[1])
    c['same_choice'] += b[0] == f[0]
print('%s %s: %s, missed-pickup families %s, mean cost gap budget-full %.2f u, candidates budget/full %.0f/%.0f' % (
    name, pool[8:20], dict(c), dict(missed), float(np.mean(diffs)) if diffs else 0, 0, 0), flush=True)
