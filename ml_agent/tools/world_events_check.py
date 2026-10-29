"""Check benchmark pickup semantics against recorded states and both physics paths."""
import argparse
import copy
import gzip
import json
from pathlib import Path

from nav.marble_dynamics import realtime as RT
from tools.sim_bench.world_events import advance_world, consumed_powerups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    RT.SEARCH_WORKERS = 0
    world = RT.World('Concentric_Hunt')
    engine = world.engine
    path = args.pool/'Concentric_Hunt/7347019/physics_plan/1/decisions.jsonl.gz'
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        records = (json.loads(line)['before'] for line in f)
        original = next(d for d in records if d['native']['state']['mode'] == 0
                        and not d['native']['state']['oob'])
    native = engine.native_sim
    rows = []
    try:
        for physics in ('compiled', 'python'):
            for held, pickup in ((1, 2), (6, 2), (2, 6), (2, 2)):
                d = copy.deepcopy(original)
                state = d['native']['state']
                state['held'] = held if held <= 5 else 0
                d['inventory']['held_type'] = held
                state['inventory_context'] = copy.deepcopy(d['inventory'])
                # No queued use: this check isolates pickup and inventory replacement.
                d['bridge_action'][6] = 0
                d['native']['pending_move']['use'] = False
                p = state['position']
                d['items'] = [dict(id=987654321, gem=False, powerup=pickup,
                                  position=list(p), bounds=[x-.01 for x in p]+[x+.01 for x in p])]
                engine.native_sim = native if physics == 'compiled' else None
                engine.mega_enabled = False
                roll = advance_world(engine, d, [0]*9)
                assert roll['failure'] is None, roll['failure']
                final = roll['states'][-1]
                expected = pickup if pickup != held else held
                assert final['inventory_context']['held_type'] == expected
                assert int(final['held']) == (expected if expected <= 5 else 0)
                assert bool(consumed_powerups(roll)) == (held != pickup)
                assert engine.mega_enabled is False
                assert engine.native_sim is (native if physics == 'compiled' else None)
                rows.append(dict(physics=physics, held=held, pickup=pickup,
                                 final_held=expected, consumed=sorted(consumed_powerups(roll))))
    finally:
        engine.native_sim = native
        world.close()
    args.out.write_text(json.dumps(dict(checks=rows, passed=len(rows)), indent=2), encoding='utf-8')
    print('Passed', len(rows), 'world pickup checks')


if __name__ == '__main__':
    main()
