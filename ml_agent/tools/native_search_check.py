"""The planner's full search with the Python forecast vs the compiled forecast (nav/native), on recorded states.

usage (repo root, PYTHONPATH=ml_agent): python ml_agent/tools/native_search_check.py [--per-round 8] <round folder> ...
Every candidate must match: label, verdict, cost, pickup interval and every predicted 64 ms state (position, velocity,
spin). Prints the time of both searches (unbudgeted, so both evaluate every candidate).
"""
import argparse
import copy
import gzip
import json
import time
from pathlib import Path

import numpy as np

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
from nav.native import attach

CACHE = Path(r'C:\Users\doug\AppData\Local\Temp\pq_checks\fast_nav_static_cache_20261013')
DLL = Path(r'C:\Users\doug\AppData\Local\Temp\pq_checks\compiled_sweep_20261012\sweep_kernel.dll')


def setup(name):
    terrain = cached_terrain(TerrainMap.resolve(name), CACHE)
    mesh = CachedCompiledSweepMap(ML / 'collision_maps' / (name + '.npz'), DLL); load_queries(mesh, DLL, CACHE)
    walk = cached_walking(terrain, mesh, DLL, CACHE)
    engine = GeneralPhysics(mesh); engine.bridge_physics_ms = 16; engine.use_horizon = 48; engine.use_margin_u = 8.
    engine.ballistic_landing = True; engine.pickup_landing_replies = 40; engine.detour_blast_ratio = 1.5
    engine.crossings = Crossings(walk, mesh=mesh); engine.blast_value_u = 10.; engine.use_completed_candidates_on_timeout = True
    nav = FrozenNavigator(terrain, ML / 'models/nav/real_maps_20261010_A2/nav_latest.pth')
    sim = attach(engine); engine.native_sim = None
    return terrain, walk, engine, nav, sim


def run(engine, walk, base, raw, dest, nav, budget=None):
    route = RoutePlanner(walk); engine.planning_deadline = None if budget is None else time.perf_counter() + budget
    t0 = time.perf_counter()
    chosen, pairs = S.search(engine, route, copy.deepcopy(base), raw, np.asarray(dest[:3]), nav, 24, brake_replies=0)
    return time.perf_counter() - t0, chosen, pairs


def same_rollout(a, b):
    if (a is None) != (b is None):
        return False
    if a is None:
        return True
    if len(a['states']) != len(b['states']) or a['failure'] != b['failure'] or a['ambiguous_intervals'] != b['ambiguous_intervals']:
        return False
    for x, y in zip(a['states'], b['states']):
        for k in ('position', 'velocity', 'spin'):
            if list(x[k]) != list(y[k]):
                return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('rounds', nargs='+', type=Path)
    p.add_argument('--per-round', type=int, default=8)
    p.add_argument('--budget', type=float, default=None, help='a real planning deadline for the compiled search (game conditions)')
    a = p.parse_args()
    cache = {}; tot = dict(searches=0, candidates=0, identical=0, t_py=0., t_c=0.)
    for folder in a.rounds:
        name = folder.resolve().parts[-4]
        if name not in cache:
            cache[name] = setup(name)
        terrain, walk, engine, nav, sim = cache[name]
        R = []
        with gzip.open(folder / 'decisions.jsonl.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    R.append(json.loads(line))
                except ValueError:
                    pass
        R.sort(key=lambda d: d['decision'])
        picks = [d for d in R if d['before']['native']['state'].get('mode', 0) == 0 and not d['before']['native']['state'].get('oob')
                 and (d['choice'].get('goal_context') or {}).get('destination')]
        picks = picks[::max(1, len(picks) // a.per_round)][:a.per_round]
        for d in picks:
            base = d['before']; raw = np.asarray(d['raw_observation'], np.float32); dest = d['choice']['goal_context']['destination']
            tgt, nxt, _ = tour(terrain, {}, raw, list(dest) + [0, 0])
            engine.next_for = {goal_key(tgt): nxt} if tgt is not None and nxt is not None and goal_key(tgt) == goal_key(dest) else {}
            engine.native_sim = None
            t_py, c_py, p_py = run(engine, walk, base, raw, dest, nav)
            engine.native_sim = sim
            t_c, c_c, p_c = run(engine, walk, base, raw, dest, nav, a.budget)
            engine.native_sim = None
            ident = 0
            for (ia, ra), (ib, rb) in zip(p_py, p_c):
                keys = ('label', 'rejected', 'cost', 'possible_pickup_interval')
                if all(ia.get(k) == ib.get(k) or (k == 'cost' and not np.isfinite(ia.get(k, 0)) and not np.isfinite(ib.get(k, 0))) for k in keys) and same_rollout(ra, rb):
                    ident += 1
                elif tot['searches'] < 40:
                    print('   DIFF', name, d['decision'], ia.get('label'), ia.get('rejected'), ib.get('rejected'), ia.get('cost'), ib.get('cost'))
            same_choice = (c_py is None and c_c is None) or (c_py is not None and c_c is not None and c_py[0]['label'] == c_c[0]['label'] and c_py[0]['cost'] == c_c[0]['cost'])
            tot['searches'] += 1; tot['candidates'] += max(len(p_py), len(p_c)); tot['identical'] += ident if len(p_py) == len(p_c) else 0
            tot['t_py'] += t_py; tot['t_c'] += t_c
            print('%-22s dec %5d: %2d/%2d candidates identical, same choice %s | search %.2f s python, %.2f s compiled (%.1fx)'
                  % (name, d['decision'], ident, len(p_py), same_choice, t_py, t_c, t_py / max(t_c, 1e-9)), flush=True)
    print('TOTAL', tot, 'speed-up %.1fx' % (tot['t_py'] / max(tot['t_c'], 1e-9)))


if __name__ == '__main__':
    main()
