"""At recorded 'planning_time_budget' fallback states: what would pick() find with 2 s (virtual) vs unbudgeted?
Counts: useful plan found with 2 s / only unbudgeted / never; and for 'never': candidate rejection reasons.
usage (repo root, PYTHONPATH=ml_agent): python fallback_audit.py <pool> <map> [N]"""
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
from nav.marble_dynamics.reliability_eval import tour, pick
from nav.marble_dynamics.search import PlanMemory
from nav.marble_dynamics.global_goals import observed_gems
from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded
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
R = [d for d in R if (d['choice'].get('label') or '').startswith('A2') and d['choice'].get('reason') == 'planning_time_budget'
     and d['before']['native']['state']['mode'] == 0 and not d['before']['native']['state']['oob']]
R.sort(key=lambda d: d['decision']); stepn = max(1, len(R) // per)
c = Counter(); rej = Counter(); speeds = []
for d in R[::stepn][:per]:
    dyn = d['before']; raw = np.asarray(d['raw_observation'], np.float32)
    dest = (d['choice'].get('goal_context') or {}).get('destination')
    if not dest: continue
    tgt, nxt, gems = tour(terrain, {}, raw, list(dest) + [0, 0])
    gems = observed_gems(dyn, raw) or gems
    current = list(dest) + [0., -1.] if tgt is None else tgt
    engine.next_for = {goal_key(tgt): nxt} if tgt is not None and nxt is not None else {}
    res = {}
    for mode in ('budget', 'full'):
        time.perf_counter = (lambda: VC[0]) if mode == 'budget' else real_pc
        engine.planning_deadline = time.perf_counter() + 2. if mode == 'budget' else None
        try:
            chosen, pairs, _ = pick(engine, RoutePlanner(walk), copy.deepcopy(dyn), raw, gems, current, nav, 24, PlanMemory())
        except PlanningBudgetExceeded:
            chosen, pairs = None, []
        finally:
            time.perf_counter = real_pc; engine.planning_deadline = None
        res[mode] = chosen
        if mode == 'full' and chosen is None:
            rej.update(str(p.get('rejected'))[:28] for p in pairs)
    c['states'] += 1
    k = 'budget_found' if res['budget'] else 'full_only' if res['full'] else 'none'
    c[k] += 1; speeds.append(float(np.linalg.norm(dyn['native']['state']['velocity'][:2])))
print('%s %s: %s speed median %.1f | rejections when none: %s' % (name, pool[8:20], dict(c), float(np.median(speeds)) if speeds else 0, dict(rej.most_common(5))), flush=True)
