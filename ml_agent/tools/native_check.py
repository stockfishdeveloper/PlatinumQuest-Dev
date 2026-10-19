"""Compiled physics (nav/native) vs the Python port (GeneralPhysics.step) on recorded native 16 ms transitions.

usage (repo root, PYTHONPATH=ml_agent): python ml_agent/tools/native_check.py [--stride 5] [--per-round 3000] <round folder> ...
A round folder holds native.jsonl.gz (pool journals: <pool>/<map>/7347019/physics_plan/1). Reports, per map: steps
compared, bit-identical results (position, velocity, spin, yaw, held, effects, best contact normal, event kinds),
the largest difference, and both simulators' one-step error against the game.
"""
import argparse
import gzip
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from nav.marble_dynamics.static_cache import load_queries
from nav.collision.experimental_sweep import CachedCompiledSweepMap
from nav.marble_dynamics.record import ML
from nav.marble_dynamics.physics import GeneralPhysics
from nav.native.marble_sim import NativeSim

CACHE = Path(r'C:\Users\doug\AppData\Local\Temp\pq_checks\fast_nav_static_cache_20261013')
DLL = Path(r'C:\Users\doug\AppData\Local\Temp\pq_checks\compiled_sweep_20261012\sweep_kernel.dll')


def kinds(events):
    return [e['kind'] for e in events if e['kind'] != 'diagnostic']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('rounds', nargs='+', type=Path)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--per-round', type=int, default=3000)
    p.add_argument('--show', type=int, default=5)
    a = p.parse_args()
    engines = {}
    total = Counter(); timing = defaultdict(float); shown = 0
    for folder in a.rounds:
        name = folder.resolve().parts[-4]
        if name not in engines:
            mesh = CachedCompiledSweepMap(ML / 'collision_maps' / (name + '.npz'), DLL); load_queries(mesh, DLL, CACHE)
            engine = GeneralPhysics(mesh)
            engines[name] = (engine, NativeSim(engine))
        engine, sim = engines[name]
        rows = []
        with gzip.open(folder / 'native.jsonl.gz', 'rt') as f:
            for line in f:
                r = json.loads(line)
                if r.get('kind') == 'transition':
                    rows.append(r)
        c = Counter(); worst = 0.; err_py = []; err_c = []
        for r in rows[::a.stride][:a.per_round]:
            b = r['before']
            if b.get('mode', 0) != 0 or b.get('oob') or r['after'].get('mode', 0) != 0:
                continue
            before = dict(b); before['clock_ms'] = r['sim_ms']
            t0 = time.perf_counter()
            try:
                ps, pe = engine.step(before, r['move'], r['dt_ms']); perr = None
            except ValueError as exc:
                perr = str(exc)
            t1 = time.perf_counter()
            try:
                cs, ce = sim.step(before, r['move'], r['dt_ms']); cerr = None
            except ValueError as exc:
                cerr = str(exc)
            t2 = time.perf_counter()
            timing['python'] += t1 - t0; timing['native'] += t2 - t1
            c['steps'] += 1
            if perr or cerr:
                c['both_failed' if perr and cerr else 'one_failed'] += 1
                if bool(perr) != bool(cerr) and shown < a.show:
                    print('  FAILURE MISMATCH', name, r['sim_ms'], perr, cerr); shown += 1
                continue
            same = (ps['position'] == cs['position'] and ps['velocity'] == cs['velocity'] and ps['spin'] == cs['spin']
                    and ps['yaw'] == cs['yaw'] and ps['held'] == cs['held']
                    and [list(e) for e in ps['effects']] == cs['effects']
                    and ps.get('best_contact_normal') == cs.get('best_contact_normal'))
            same_events = kinds(pe) == kinds(ce)
            d = max(float(np.max(np.abs(np.asarray(ps[k]) - cs[k]))) for k in ('position', 'velocity', 'spin'))
            worst = max(worst, d)
            c['identical'] += same; c['identical_events'] += same_events
            if not same or not same_events:
                c['different'] += 1
                if shown < a.show:
                    print('  DIFF', name, 'sim_ms', r['sim_ms'], 'max |d| %.3g' % d, 'events py', kinds(pe)[:8], 'c', kinds(ce)[:8]); shown += 1
            after = r['after']
            err_py.append(float(np.linalg.norm(np.asarray(ps['position']) - after['position'])))
            err_c.append(float(np.linalg.norm(np.asarray(cs['position']) - after['position'])))
        total.update(c)
        q = lambda x: '%.2e / %.2e' % (np.percentile(x, 99), max(x)) if x else '-'
        print('%-22s steps %5d identical %5d (events %5d) differ %3d failed %d | max |py-c| %.3g | vs game p99/max: python %s, compiled %s'
              % (name, c['steps'], c['identical'], c['identical_events'], c['different'], c['both_failed'] + c['one_failed'], worst, q(err_py), q(err_c)), flush=True)
    print('TOTAL', dict(total))
    if total['steps']:
        print('time per step: python %.1f us, compiled %.1f us (incl. Python packing), speed-up %.1fx'
              % (timing['python'] / total['steps'] * 1e6, timing['native'] / total['steps'] * 1e6, timing['python'] / max(timing['native'], 1e-12)))


if __name__ == '__main__':
    main()
