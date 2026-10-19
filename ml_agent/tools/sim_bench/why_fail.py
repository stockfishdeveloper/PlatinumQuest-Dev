"""Re-run the search at recorded decisions (unbudgeted and 2 s) and list every candidate's rejection.
usage (repo root, PYTHONPATH=ml_agent): python why_fail.py <pool> <map> <decision> [<decision> ...]
"""
import gzip, json, copy, sys, time
import numpy as np
from pathlib import Path
from collections import Counter
from nav.terrain import TerrainMap
from nav.routing import RoutePlanner
from nav.marble_dynamics.static_cache import cached_terrain, cached_walking, load_queries
from nav.collision.experimental_sweep import CachedCompiledSweepMap
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.physics import GeneralPhysics
import nav.marble_dynamics.search as S
from nav.marble_dynamics.crossings import Crossings
from nav.marble_dynamics.navigator import FrozenNavigator
from nav.marble_dynamics.reliability_eval import tour
from nav.routing import goal_key

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
R = {}
with gzip.open(Path('ml_agent/dynamics_reports') / pool / name / '7347019/physics_plan/1/decisions.jsonl.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        try:
            d = json.loads(line); R[d['decision']] = d
        except Exception: pass
for k in decs:
    d = R[k]; base = d['before']; raw = np.asarray(d['raw_observation'], np.float32)
    dest = (d['choice'].get('goal_context') or {}).get('destination')
    st = base['native']['state']
    tgt, nxt, _ = tour(terrain, {}, raw, list(dest) + [0, 0])
    engine.next_for = {goal_key(tgt): nxt} if tgt is not None and nxt is not None and goal_key(tgt) == goal_key(dest) else {}
    print('   tour next gem', None if not engine.next_for else [round(float(v), 1) for v in nxt[:3]])
    print('=== decision', k, 'label', d['choice'].get('label'), d['choice'].get('reason'), 'pos', np.round(st['position'], 2).tolist(),
          'v', np.round(st['velocity'], 2).tolist(), 'goal', dest)
    for budget in (0., 2.):
        route = RoutePlanner(walk)
        engine.planning_deadline = (time.perf_counter() + budget) if budget else None
        t0 = time.perf_counter()
        try:
            chosen, pairs = S.search(engine, route, copy.deepcopy(base), raw, np.asarray(dest[:3]), nav, 24, brake_replies=0)
        except Exception as exc:
            print('  budget', budget, 'raised', repr(exc)[:120]); continue
        dt = time.perf_counter() - t0
        rej = Counter(p[0]['rejected'] for p in pairs)
        print('  budget %.0f: %.1fs, %d candidates, chosen %s, rejections %s' % (budget, dt, len(pairs), chosen[0]['label'] if chosen else None, dict(rej)))
        if budget == 0.:
            for info, roll in pairs:
                end = roll['states'][-1] if roll else None
                print('    %-34s %-28s cost=%7.1f pk=%-4s end=%s v=%s' % (info['label'][:34], str(info['rejected'])[:28], info['cost'] if np.isfinite(info['cost']) else 999,
                      info.get('possible_pickup_interval'), None if end is None else np.round(end['position'], 1).tolist(),
                      None if end is None else round(float(np.linalg.norm(end['velocity'])), 1)))
