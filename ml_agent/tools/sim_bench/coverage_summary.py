"""Summarize a full-round coverage batch (full_round.py rows): every arm against the baseline, paired by (map, spawn seed).
Reports the mean paired delta with its standard error, the equal-map ratio, per-map deltas, falls, mean speed and the
fallback / hold counts, and a cluster bootstrap by map of the equal-map ratio.

usage: python ml_agent/tools/sim_bench/coverage_summary.py coverage/batch1 [--base base] [--boot 2000]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load(folder):
    rows = defaultdict(dict)   # arm -> (map, spawn_seed) -> row
    for f in glob.glob(os.path.join(folder, '*.jsonl')):
        for line in open(f, encoding='utf-8'):
            if line.strip():
                r = json.loads(line); rows[r['label']][(r['map'], int(r['spawn_seed']))] = r
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('--base', default='base'); ap.add_argument('--boot', type=int, default=2000)
    o = ap.parse_args(); rows = load(o.folder); base = rows.get(o.base, {})
    rng = np.random.default_rng(0)
    print('baseline rounds: %d, mean %.1f points, falls %.2f a round' % (len(base), np.mean([r['points'] for r in base.values()]) if base else 0,
                                                                        np.mean([r['falls'] for r in base.values()]) if base else 0))
    for arm, arows in sorted(rows.items()):
        if arm == o.base:
            continue
        keys = sorted(set(arows) & set(base))
        if not keys:
            continue
        d = np.asarray([arows[k]['points'] - base[k]['points'] for k in keys], dtype=float)
        by_map = defaultdict(list)
        for k in keys:
            by_map[k[0]].append((arows[k]['points'], base[k]['points']))
        ratio = np.mean([np.mean([a for a, b in v]) / max(1e-9, np.mean([b for a, b in v])) for v in by_map.values()])
        maps = list(by_map); draws = []
        for _ in range(o.boot):
            pick = rng.choice(len(maps), size=len(maps), replace=True)
            draws.append(np.mean([np.mean([a for a, b in by_map[maps[i]]]) / max(1e-9, np.mean([b for a, b in by_map[maps[i]]])) for i in pick]))
        falls = np.mean([arows[k]['falls'] for k in keys]) - np.mean([base[k]['falls'] for k in keys])
        speed = np.mean([arows[k]['mean_speed'] for k in keys]) - np.mean([base[k]['mean_speed'] for k in keys])
        fb = np.mean([arows[k]['fallback'] for k in keys]) - np.mean([base[k]['fallback'] for k in keys])
        print('\n%-16s n=%3d  delta %+5.2f +- %4.2f a round  equal-map ratio %.3f [%.3f, %.3f]  falls %+.2f  speed %+.2f  fallback decisions %+.0f' % (
            arm, len(keys), d.mean(), d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0., ratio, np.percentile(draws, 2.5), np.percentile(draws, 97.5), falls, speed, fb))
        print('   per map: ' + ', '.join('%s %+.1f' % (m[:10], np.mean([a - b for a, b in v])) for m, v in sorted(by_map.items())))


if __name__ == '__main__':
    main()
