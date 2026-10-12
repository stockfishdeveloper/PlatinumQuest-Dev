"""Lock geometry-selected development/holdout candidates before engine screening."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from nav.replay import load_starts
from nav.protocol import RAW_POW_HELD


def select(rows, per_resource=2):
    buckets = defaultdict(list)
    for s in rows:
        held = int(s['raw'][RAW_POW_HELD])
        if held not in (1, 2, 5) or len(s.get('hist', [])) != 32:
            continue
        pos = s['world']['pose'][:3]
        distance = math.dist(pos, s['goal'])
        dz = s['goal'][2] - pos[2]
        if not 3 <= distance <= 40:
            continue
        # A candidate geometric opportunity, not a claim of causal benefit.
        category = 'rise' if dz >= 2 else 'far' if distance >= 15 else 'near'
        buckets[s['mission'], held].append((category, distance, s))
    selected, annotations = [], {}
    for bucket, candidates in sorted(buckets.items()):
        candidates.sort(key=lambda x: ({'rise': 0, 'far': 1, 'near': 2}[x[0]], x[1], x[2]['id']))
        chosen, groups, categories = [], set(), set()
        while candidates and len(chosen) < per_resource:
            # Diversity across source rounds and approach categories takes priority.
            i = min(range(len(candidates)), key=lambda j: (
                candidates[j][2]['split_group'] in groups, candidates[j][0] in categories, j))
            category, distance, s = candidates.pop(i)
            chosen.append(s); groups.add(s['split_group']); categories.add(category)
            annotations[s['id']] = dict(category=category, distance=distance,
                                       target_rise=s['goal'][2] - s['world']['pose'][2])
        selected.extend(chosen)
    return selected, annotations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dev', action='append', required=True)
    p.add_argument('--holdout', action='append', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--per-resource', type=int, default=6,
                   help='Per-map/resource candidate cap; six permits three use and three save cases')
    p.add_argument('--recordings', help='Assemble fresh normal-play JSONL into round-partitioned pools')
    args = p.parse_args()
    if args.per_resource <= 0:
        p.error('per-resource must be positive')
    if args.recordings:
        from nav.escape_practice import assemble
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/real_maps_20261009.json').read_text())
        assemble(args.recordings, config['maps'])
        args.dev.append(str(Path(args.recordings) / 'starts_dev.json'))
        args.holdout.append(str(Path(args.recordings) / 'starts_eval.json'))
    def load(paths):
        rows = {}
        for path in paths:
            for s in load_starts(path):
                if s['id'] in rows and rows[s['id']] != s:
                    raise ValueError('conflicting duplicate fixture')
                rows[s['id']] = s
        return list(rows.values())
    dev, holdout = load(args.dev), load(args.holdout)
    overlap = {s['split_group'] for s in dev} & {s['split_group'] for s in holdout}
    if overlap or {s['id'] for s in dev} & {s['id'] for s in holdout}:
        raise ValueError('development/holdout source overlap')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(method='geometry only; no outcome selection; source-round partitions preserved',
                    inputs={str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                            for path in args.dev + args.holdout})
    for name, rows in [('dev', dev), ('holdout', holdout)]:
        chosen, annotations = select(rows, args.per_resource)
        if not chosen:
            raise ValueError('empty selected pool: ' + name)
        (out / f'{name}_candidates.json').write_text(json.dumps(chosen, indent=1))
        manifest[name] = dict(n=len(chosen), source_rounds=len({s['split_group'] for s in chosen}),
                              maps=dict(Counter(s['mission'] for s in chosen)), annotations=annotations)
    (out / 'selection.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: {m: v for m,v in manifest[k].items() if m != 'annotations'}
                      for k in ('dev','holdout')}, indent=2))


if __name__ == '__main__':
    main()
