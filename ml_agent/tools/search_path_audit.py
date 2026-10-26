"""Compare serial search with its worker evaluator, without scheduling or deadlines.

This isolates evaluator drift; it does not measure multiprocessing overhead or
full-round points. Uses recorded native states and production world setup.
Run from the repository root with PYTHONPATH=ml_agent.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from nav.marble_dynamics import realtime as RT
from nav.marble_dynamics import search as S
from nav.marble_dynamics.parallel_search import evaluate_candidate
from nav.routing import RoutePlanner, goal_key


class InlineWorkers:
    """Execute the actual worker function locally, removing IPC and timing."""

    def __init__(self):
        self.spec_counts = Counter()

    def evaluate(self, engine, route, dynamics, gem, specs, target_item, steps,
                 landing_replies, ballistic, brake_replies, deadline=None):
        self.spec_counts = Counter(label for label, _ in specs)
        return [evaluate_candidate(engine, route, dynamics, gem, label, proposal,
                                   target_item, steps, landing_replies, ballistic,
                                   brake_replies) for label, proposal in specs]


def candidates_by_label(pairs):
    return {info['label']: dict(rejected=info.get('rejected'),
                               cost=info['cost'] if math.isfinite(info.get('cost', math.inf)) else None,
                               replies=info.get('evaluated_replies'),
                               pickup=info.get('possible_pickup_interval'))
            for info, _ in pairs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--maps', nargs='+', required=True)
    p.add_argument('--seed', default='7347019')
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--roll-launches', action='store_true',
                   help='Also inspect the opt-in no-jump launch family')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    RT.SEARCH_WORKERS = 0
    if a.roll_launches:
        S.POW_LAUNCH_ROLL = True
    report = dict(contract='Same native state, no deadline, no navigator proposal; '
                           'actual worker evaluator executed inline. Not a points test.',
                  pool=str(a.pool), seed=a.seed, roll_launches=a.roll_launches,
                  python=sys.version, sources={}, samples=[])
    for mod in (S, sys.modules[evaluate_candidate.__module__], RT):
        src = Path(mod.__file__)
        report['sources'][src.name] = hashlib.sha256(src.read_bytes()).hexdigest()
    for name in a.maps:
        path = a.pool / name / a.seed / 'physics_plan/1/decisions.jsonl.gz'
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        eligible = []
        for row in rows:
            st = row['before']['native']['state']
            choice = row.get('choice') or {}
            if st['oob'] or st['mode'] != 0:
                continue
            if not choice.get('goal_context', {}).get('destination'):
                continue
            if a.roll_launches and st['held'] != 2:
                continue
            if st['held'] == 2 or str(choice.get('label', '')).startswith('edge_hit'):
                eligible.append(row)
        if not eligible:
            continue
        indices = np.linspace(0, len(eligible) - 1, min(a.samples, len(eligible)), dtype=int)
        world = RT.World(name)
        try:
            engine = world.engine
            engine.planning_deadline = None
            for index in indices:
                row = eligible[index]
                context = row['choice']['goal_context']
                gem = np.asarray(context['destination'], dtype=float)
                nxt = context.get('next_goal')
                engine.next_for = {goal_key(gem): nxt} if nxt is not None else {}
                item = next((i for i in row['before']['items']
                             if i['id'] == context.get('item_id')), None)
                results = []
                workers = InlineWorkers()
                for worker in (None, workers):
                    engine.parallel = worker
                    engine._through_cache = None
                    route = RoutePlanner(world.args.geometry_terrain)
                    chosen, pairs = S.search(engine, route, copy.deepcopy(row['before']),
                                             row['raw_observation'], gem, None,
                                             horizon=world.args.horizon,
                                             target_item=item, learned=False)
                    results.append((chosen[0]['label'] if chosen else None,
                                    candidates_by_label(pairs)))
                serial, parallel = results
                missing = sorted(set(serial[1]) - set(parallel[1]))
                extra = sorted(set(parallel[1]) - set(serial[1]))
                changed = {label: dict(serial=serial[1][label], worker=parallel[1][label])
                           for label in sorted(set(serial[1]) & set(parallel[1]))
                           if serial[1][label] != parallel[1][label]}
                result = dict(map=name, decision=row['decision'],
                              serial_choice=serial[0], worker_choice=parallel[0],
                              serial_count=len(serial[1]), worker_count=len(parallel[1]),
                              missing_from_worker=missing, extra_in_worker=extra,
                              duplicate_specs={k: v for k, v in workers.spec_counts.items() if v > 1},
                              changed=changed)
                report['samples'].append(result)
                print('%s decision %d: %s / %s; missing %d, extra %d, changed %d' %
                      (name, row['decision'], serial[0], parallel[0], len(missing),
                       len(extra), len(changed)), flush=True)
        finally:
            world.engine.parallel = None
            world.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
