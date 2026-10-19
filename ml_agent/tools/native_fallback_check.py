"""The fallback searches (settle.choose_settled, phased_search.explore, recharge.choose_recharge) and landing.py's
support-point continuations with the Python forecast vs the compiled one (nav/native ABI 6 runs the Settle / Aim /
Recharge / landing.LandingContinuation controllers), on recorded states.

usage (repo root, PYTHONPATH=ml_agent): python ml_agent/tools/native_fallback_check.py [--per-round 6] <round folder> ...
Every candidate must match: label, verdict, cost, pickup interval, the proposed actions and every predicted 64 ms state.
"""
import argparse
import copy
import gzip
import json
import time
from pathlib import Path

import numpy as np

from nav.routing import RoutePlanner
from nav.marble_dynamics.settle import choose_settled
from nav.marble_dynamics.phased_search import explore
from nav.marble_dynamics.recharge import choose_recharge
from nav.marble_dynamics.landing import LandingContinuation, landing_points
from nav.marble_dynamics.bridge import predict_replies
from nav.marble_dynamics.search import assess

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_search_check as N


def same(pa, pb):
    if len(pa) != len(pb):
        return 0, max(len(pa), len(pb))
    ok = 0
    for (ia, ra), (ib, rb) in zip(pa, pb):
        keys = ('label', 'rejected', 'cost', 'possible_pickup_interval')
        match = all(ia.get(k) == ib.get(k) or (k == 'cost' and not np.isfinite(ia.get(k, 0)) and not np.isfinite(ib.get(k, 0))) for k in keys)
        match = match and N.same_rollout(ra, rb) and [list(map(float, a)) for a in ra['proposed']] == [list(map(float, a)) for a in rb['proposed']]
        ok += int(match)
        if not match:
            print('      DIFF', ia.get('label'), ia.get('rejected'), ib.get('rejected'), ia.get('cost'), ib.get('cost'))
    return ok, len(pa)


def landing_pairs(engine, route, base, gem, source):
    """landing.alternatives' continuations, from the prefixes of the given candidates (any verdict)."""
    out = []
    for info, rollout in source[:4]:
        if rollout is None or rollout['failure'] or len(rollout['proposed']) < 4:
            continue
        count = min(8, len(rollout['proposed']) - 1); prefix = rollout['proposed'][:count]
        for index, point in enumerate(landing_points(engine.mesh, rollout['states'][count], 3)):
            changed = predict_replies(engine, copy.deepcopy(base), LandingContinuation(prefix, point['position'], 80), stop_on_ambiguity=True)
            out.append((assess(engine, route, base, gem, changed, info['label'] + '_landing_' + str(index)), changed))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('rounds', nargs='+', type=Path)
    p.add_argument('--per-round', type=int, default=6)
    a = p.parse_args()
    cache = {}; tot = dict(candidates=0, identical=0, t_py=0., t_c=0.)
    for folder in a.rounds:
        name = folder.resolve().parts[-4]
        if name not in cache:
            cache[name] = N.setup(name)
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
            base = d['before']; gem = np.asarray(d['choice']['goal_context']['destination'][:3]); raw = d['raw_observation']
            line = '%-22s dec %5d:' % (name, d['decision'])
            for label, run in (('settle', lambda r: choose_settled(engine, r, copy.deepcopy(base), gem)),
                               ('aim', lambda r: explore(engine, r, copy.deepcopy(base), gem, horizon=80, samples=12, seed=997)),
                               ('recharge', lambda r: choose_recharge(engine, r, copy.deepcopy(base), gem, 1000, float(raw[31])))):
                engine.planning_deadline = None
                engine.native_sim = None; t0 = time.perf_counter(); _, pa = run(RoutePlanner(walk)); t1 = time.perf_counter()
                engine.native_sim = sim; _, pb = run(RoutePlanner(walk)); t2 = time.perf_counter(); engine.native_sim = None
                ok, n = same(pa, pb)
                tot['candidates'] += n; tot['identical'] += ok; tot['t_py'] += t1 - t0; tot['t_c'] += t2 - t1
                line += ' %s %d/%d (%.2f s vs %.2f s) |' % (label, ok, n, t1 - t0, t2 - t1)
                if label == 'settle':
                    source = pa
            route = RoutePlanner(walk); route.navigate(base['native']['state']['position'], gem)
            engine.native_sim = None; t0 = time.perf_counter(); pa = landing_pairs(engine, route, base, gem, source); t1 = time.perf_counter()
            engine.native_sim = sim; pb = landing_pairs(engine, route, base, gem, source); t2 = time.perf_counter(); engine.native_sim = None
            ok, n = same(pa, pb); tot['candidates'] += n; tot['identical'] += ok; tot['t_py'] += t1 - t0; tot['t_c'] += t2 - t1
            line += ' landing %d/%d (%.2f s vs %.2f s) |' % (ok, n, t1 - t0, t2 - t1)
            print(line, flush=True)
    print('TOTAL', tot, 'speed-up %.1fx' % (tot['t_py'] / max(tot['t_c'], 1e-9)))


if __name__ == '__main__':
    main()
