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
        self.delegate = None

    def evaluate(self, engine, route, dynamics, gem, specs, target_item, steps,
                 landing_replies, ballistic, brake_replies, deadline=None):
        self.spec_counts = Counter(label for label, _ in specs)
        if self.delegate is not None:
            return self.delegate.evaluate(engine, route, dynamics, gem, specs, target_item, steps,
                                          landing_replies, ballistic, brake_replies, deadline=deadline)
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
    p.add_argument('--powerup-targets', action='store_true',
                   help='Check recorded powerup approaches with an observed gem as their onward destination')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--planner-flag', action='append', default=[])
    p.add_argument('--workers', type=int, default=0, help='Use actual multiprocessing workers instead of inline evaluation')
    p.add_argument('--assert-equal', action='store_true')
    a = p.parse_args()
    RT.SEARCH_WORKERS = 0
    RT.FLAGS = RT.FLAGS + a.planner_flag
    if a.roll_launches:
        RT.FLAGS = RT.FLAGS + ['--roll-launches']
    if a.powerup_targets:
        RT.FLAGS = RT.FLAGS + ['--powerup-through']
    report = dict(contract='Same native state, no deadline, no navigator proposal; '
                           'actual worker evaluator executed inline. Not a points test.',
                  pool=str(a.pool), seed=a.seed, roll_launches=a.roll_launches,
                  powerup_targets=a.powerup_targets,
                  workers=a.workers, planner_flags=a.planner_flag,
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
            if a.powerup_targets:
                item_id=choice['goal_context'].get('item_id')
                if any(i['id']==item_id and not i.get('gem') and i.get('powerup')
                       for i in row['before'].get('items',[])):
                    eligible.append(row)
                continue
            if a.roll_launches and st['held'] != 2:
                continue
            if st['held'] == 2 or str(choice.get('label', '')).startswith('edge_hit'):
                eligible.append(row)
        if not eligible:
            continue
        indices = np.linspace(0, len(eligible) - 1, min(a.samples, len(eligible)), dtype=int)
        world = RT.World(name)
        workers = InlineWorkers()
        if a.workers:
            from nav.marble_dynamics.parallel_search import ParallelEvaluator, worker_init
            workers.delegate = ParallelEvaluator(a.workers, worker_init(world.args, name))
            workers.delegate.bind(dict(terrain=world.terrain, geometry=world.args.geometry_terrain,
                                       mesh=world.engine.mesh, sim=world.engine.native_sim,
                                       crossings=world.engine.crossings,
                                       contact_order=getattr(world.engine,'contact_order',None)))
        try:
            engine = world.engine
            engine.planning_deadline = None
            for index in indices:
                row = eligible[index]
                context = row['choice']['goal_context']
                gem = np.asarray(context['destination'], dtype=float)
                nxt = context.get('next_goal')
                if a.powerup_targets and nxt is None:
                    gems=[i for i in row['before']['items'] if i.get('gem')]
                    if not gems:continue
                    nxt=min(gems,key=lambda i:np.linalg.norm(np.asarray(i['position'])-gem))['position']
                engine.next_for = {goal_key(gem): nxt} if nxt is not None else {}
                item = next((i for i in row['before']['items']
                             if i['id'] == context.get('item_id')), None)
                results = []
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
                              target_kind='powerup' if a.powerup_targets else 'recorded goal',
                              onward=nxt,
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
            if workers.delegate is not None:workers.delegate.close()
            world.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    if a.assert_equal and not report['samples']:
        raise SystemExit('No eligible recorded states; agreement was not tested')
    if a.assert_equal and any(r['missing_from_worker'] or r['extra_in_worker'] or r['changed']
                              or r['duplicate_specs'] or r['serial_choice'] != r['worker_choice']
                              for r in report['samples']):
        raise SystemExit('Search path mismatch; see audit report')


if __name__ == '__main__':
    main()
