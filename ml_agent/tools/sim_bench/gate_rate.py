"""How often does the terminal gate reject candidates? Unbudgeted searches at sampled recorded states (repo root, PYTHONPATH=ml_agent).
usage: python gate_rate.py <pool> <map> [per-map N] [hover]"""
import gzip, json, copy, sys, time, ctypes
import numpy as np
from pathlib import Path
from collections import Counter
_k = ctypes.windll.kernel32; _k.GetCurrentProcess.restype = ctypes.c_void_p; _k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_k.SetPriorityClass(_k.GetCurrentProcess(), 0x40)
if len(sys.argv) > 4 and sys.argv[4] == 'hover':
    import importlib.util
    spec = importlib.util.spec_from_file_location('nav.marble_dynamics.search', sys.argv[0].replace('gate_rate.py', 'search_hover.py'))
    m = importlib.util.module_from_spec(spec); sys.modules['nav.marble_dynamics.search'] = m; spec.loader.exec_module(m)
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
pool, name = sys.argv[1], sys.argv[2]; per = int(sys.argv[3]) if len(sys.argv) > 3 else 10
terrain = cached_terrain(TerrainMap.resolve(name), CACHE)
mesh = CachedCompiledSweepMap(ML / 'collision_maps' / (name + '.npz'), DLL); load_queries(mesh, DLL, CACHE)
walk = cached_walking(terrain, mesh, DLL, CACHE)
engine = GeneralPhysics(mesh); engine.bridge_physics_ms = 16; engine.use_horizon = 48; engine.use_margin_u = 8.
engine.ballistic_landing = True; engine.pickup_landing_replies = 40; engine.detour_blast_ratio = 1.5
engine.crossings = Crossings(walk, mesh=mesh); engine.blast_value_u = 10.; engine.use_completed_candidates_on_timeout = True
nav = FrozenNavigator(terrain, ML / 'models/nav/real_maps_20261010_A2/nav_latest.pth')
R = [json.loads(l) for l in gzip.open(Path('ml_agent/dynamics_reports') / pool / name / '7347019/physics_plan/1/decisions.jsonl.gz', 'rt', encoding='utf-8') if l.strip()]
R = [d for d in R if (d['choice'].get('goal_context') or {}).get('kind') == 'gem' and d['before']['native']['state']['mode'] == 0
     and not d['before']['native']['state']['oob'] and 'cost' in d['choice']]
R.sort(key=lambda d: d['decision']); step = max(1, len(R) // per)
tot = Counter(); allrej = 0; n = 0
for d in R[::step][:per]:
    base = d['before']; raw = np.asarray(d['raw_observation'], np.float32); dest = d['choice']['goal_context']['destination']
    tgt, nxt, _ = tour(terrain, {}, raw, list(dest) + [0, 0])
    engine.next_for = {goal_key(tgt): nxt} if tgt is not None and nxt is not None and goal_key(tgt) == goal_key(dest) else {}
    engine.planning_deadline = None
    chosen, pairs = S.search(engine, RoutePlanner(walk), copy.deepcopy(base), raw, np.asarray(dest[:3]), nav, 24, brake_replies=0)
    rej = Counter(p[0]['rejected'] for p in pairs); tot.update(rej); n += 1
    valid = rej.get(None, 0); allrej += valid == 0
print('%s %s %s: %d searches, candidates %d, rejected share by reason %s, searches with no valid candidate %d' % (
    name, pool[8:20], sys.argv[4] if len(sys.argv) > 4 else 'strict', n, sum(tot.values()),
    {str(k)[:26]: round(v / sum(tot.values()), 3) for k, v in tot.most_common()}, allrej), flush=True)
