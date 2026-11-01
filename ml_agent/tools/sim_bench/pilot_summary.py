"""Aggregate branch_bench.py output: per bucket and map, points over the common horizon by arm, as ratios of equal-map
means, with uncertainty clustered by source round (map, pool, seed). Clairvoyant bounds are reported apart from the
deployable selector's gain; clear-time diagnostics stay separate from points.

usage: python ml_agent/tools/sim_bench/pilot_summary.py pilot.jsonl [more.jsonl ...] [--boot 2000]
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np


def load(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding='utf-8'):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def arm_points(r):
    s = r['summary']; out = {'base': s['base_R'], 'selected': s['selected_R'], 'clairvoyant_alt': s['base_R'] + s['clairvoyant_alt'],
                             'clairvoyant_all': s['base_R'] + s['clairvoyant_all']}
    for m, v in s['work'].items():
        out['work%s' % m] = s['base_R'] + v
    if s.get('gate_best') is not None:
        out['gate_best'] = s['base_R'] + s['gate_best']
    return out


def cluster_ratio(rows, arm, rng, boot):
    """Ratio of equal-map means arm / base with a cluster bootstrap over source rounds."""
    by_map = defaultdict(lambda: defaultdict(list))   # map -> cluster -> [(arm, base)]
    for r in rows:
        p = arm_points(r)
        if arm not in p:
            continue
        by_map[r['map']][(r['pool'], r['seed'])].append((p[arm], p['base']))

    def stat(sample):
        ratios = []
        for m, clusters in sample.items():
            a = [x for c in clusters.values() for x, _ in c]; b = [y for c in clusters.values() for _, y in c]
            if b and np.mean(b) > 0:
                ratios.append(np.mean(a) / np.mean(b))
        return float(np.mean(ratios)) if ratios else float('nan')
    point = stat(by_map)
    draws = []
    for _ in range(boot):
        sample = {}
        for m, clusters in by_map.items():
            keys = list(clusters); pick = rng.choice(len(keys), size=len(keys), replace=True)
            sample[m] = {i: clusters[keys[k]] for i, k in enumerate(pick)}
        draws.append(stat(sample))
    return point, float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('files', nargs='+'); ap.add_argument('--boot', type=int, default=2000)
    o = ap.parse_args(); rows = load(o.files); rng = np.random.default_rng(0)
    print('%d situations, %d maps, %d source rounds' % (len(rows), len({r['map'] for r in rows}), len({(r['map'], r['pool'], r['seed']) for r in rows})))
    arms = ['selected', 'clairvoyant_alt', 'clairvoyant_all', 'gate_best'] + sorted({k for r in rows for k in arm_points(r) if k.startswith('work')})
    for bucket in ['all'] + sorted({r['bucket'] for r in rows}):
        sub = rows if bucket == 'all' else [r for r in rows if r['bucket'] == bucket]
        base = np.mean([r['summary']['base_R'] for r in sub])
        print('\n[%s] n=%d  base points over the horizon: %.2f' % (bucket, len(sub), base))
        for arm in arms:
            pt, lo, hi = cluster_ratio(sub, arm, rng, o.boot)
            gain = np.mean([arm_points(r)[arm] - r['summary']['base_R'] for r in sub if arm in arm_points(r)])
            print('   %-16s ratio %.3f  [%.3f, %.3f]  mean gain %+.2f pts' % (arm, pt, lo, hi, gain))
    # diagnostics: first-pickup time after the root, falls
    def first_pick(d):
        return (d['picks'][0]['t_ms'] - d['root_t_ms']) / 1000. if d and d['picks'] else None
    fp_base = [first_pick(v) for r in rows for v in r['arms']['base'].values()]
    print('\nfirst pickup after the root (baseline): median %.2f s, none within the horizon %d of %d' % (
        np.median([x for x in fp_base if x is not None]) if any(x is not None for x in fp_base) else float('nan'),
        sum(x is None for x in fp_base), len(fp_base)))
    rej = defaultdict(int)
    for r in rows:
        for k, v in r.get('rejections', {}).items():
            rej[k] += v
    print('root rejection reasons:', dict(sorted(rej.items(), key=lambda kv: -kv[1])[:12]))
    steps = defaultdict(list)
    for r in rows:
        for k, v in r['steps'].items():
            steps[k.split('|')[0].split(':')[0]].append(v)
    print('charged physics steps per run:', {k: int(np.mean(v)) for k, v in steps.items()})


if __name__ == '__main__':
    main()
