"""Run deterministic ten-map traversal screens with bounded subprocess ownership.

Each variant gets the same recorded starts and planner horizon. This orchestrator
only writes reports; native score pairs must run separately at matched CPU load.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

MAPS = ('KingOfTheMarble_Hunt', 'Sprawl_Hunt', 'Bowl_Hunt', 'CubeIsle_Hunt',
        'Duplex_Hunt', 'Cragmire_Hunt', 'FracturedIslands_Hunt', 'VortexEffect_Hunt',
        'TreasureBox_Hunt', 'Concentric_Hunt')
COMPLETE = ['--powerup-through', '--land-short', '--roll-launches', '--launch-completion']
VARIANTS = dict(base=[], short=['--land-short'], through=['--land-short', '--powerup-through'],
                complete=COMPLETE, fast=['--fast-supported'], full=COMPLETE+['--fast-supported'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--per-map', type=int, default=4)
    p.add_argument('--maps', nargs='+', default=MAPS)
    p.add_argument('--variants', nargs='+')
    p.add_argument('--config', type=Path, help='Variant JSON: label -> {pflags: [...], args: [...]}')
    p.add_argument('--pool', action='append')
    p.add_argument('--extra', action='append', default=[], help='Additional sim arguments, one token per option')
    a = p.parse_args()
    variants = json.loads(a.config.read_text()) if a.config else VARIANTS
    a.variants = a.variants or list(variants)
    if any(v not in variants for v in a.variants):p.error('Unknown variant')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'variants.json').write_text(json.dumps(variants,indent=2),encoding='utf-8')
    pools = a.pool or ['handoff_safety_20261020/cp3_baseline', 'handoff_safety_20261020/cp3_candidate']
    env = dict(os.environ, PYTHONPATH='ml_agent', PYTHONDONTWRITEBYTECODE='1')
    jobs = [(name, variant) for name in a.maps for variant in a.variants]
    status = dict(started=time.time(), jobs=len(jobs), completed=[], failed=[])

    def run(job):
        name, variant = job
        out = a.out / ('%s.%s.jsonl' % (variant, name))
        cmd = [sys.executable, '-u', 'ml_agent/tools/sim_bench/sim_bench.py',
               '--map', name, '--variant', variant, '--out', str(out), '--per-map', str(a.per_map),
               '--whole-group', '--next-group', '--native', '--cap-s', '40', '--pflag=--horizon=48']
        for pool in pools:
            cmd += ['--pool', pool]
        settings = variants[variant]
        if isinstance(settings,list):settings=dict(pflags=settings)
        cmd += ['--pflag=' + flag for flag in settings.get('pflags',[])] + settings.get('args',[]) + a.extra
        started = time.time()
        with out.with_suffix('.log').open('w', encoding='utf-8') as log:
            r = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        episodes = sum(1 for _ in out.open()) if out.exists() else 0
        return dict(map=name, variant=variant, returncode=r.returncode,
                    episodes=episodes, elapsed_s=round(time.time()-started, 1))

    def save():
        temporary = a.out / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temporary.replace(a.out / 'status.json')

    save()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(pending):
            result = future.result()
            status['completed'].append(result)
            if result['returncode'] or not result['episodes']:
                status['failed'].append(result)
            save()
            print(json.dumps(result), flush=True)
    status['finished'] = time.time()
    save()
    if status['failed']:
        raise SystemExit('Failed jobs: %d; inspect status.json and per-job logs' % len(status['failed']))


if __name__ == '__main__':
    main()
