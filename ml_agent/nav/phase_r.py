"""Read-only checkpoint baselines. This module never imports or runs the trainer.

python -m nav.phase_r                       # show the preparation configuration
python -m nav.phase_r --baseline --rounds 4 --instances 8
python -m nav.phase_r --baseline --maps Cragmire_Hunt --profiles assisted --rounds 1
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/real_maps_20261009.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hidden_process(args, **kwargs):
    if os.name == 'nt':
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        kwargs['startupinfo'] = si
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(args, **kwargs)


def stop(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=15)


def wait_ready(proc, stdout, timeout=120):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if proc.poll() is not None:
            raise RuntimeError(f'runner exited during startup: {proc.returncode}')
        if stdout.exists() and 'warm-up: model ready' in stdout.read_text(errors='replace'):
            return
        time.sleep(.25)
    raise TimeoutError('runner did not finish its preconnection model/map warm-up')


def evaluate(job, slot, output, rounds, seed):
    label, checkpoint, mission, profile_name, profile = job
    tag = f'phase_r_{label}_{profile_name}'
    stem = f'{label}_{profile_name}_{mission}'
    stdout, stderr = output / (stem + '.stdout.txt'), output / (stem + '.stderr.txt')
    env = os.environ.copy()
    env.update(NAV_PORT=str(9200 + slot), NAV_MAP=mission, NAV_CKPT=str(checkpoint),
               NAV_ROUNDS=str(rounds), NAV_TAG=tag, NAV_TOUR='walk', NAV_DIRECT_GEM='0',
               NAV_ESCAPE_POWERUPS='1', NAV_COMPETITIVE=str(int(profile['competitive'])),
               NAV_RESCUE_S=str(profile['rescue_s']), NAV_STUCK_S='0',
               NAV_JUMP_EXPLORE_P='0', NAV_JUMP_EXPLORE_BONUS='0', NAV_JUMP_FORCE_P='0',
               NAV_JUMP_DET_THRESHOLD='0.5', NAV_JUMP_ON_PRIOR='0', NAV_SAMPLE='0', NAV_NO_USE='0',
               NAV_FORCE_USE='0', NAV_WATCH='0', NAV_VIEW_SUBSTEPS='1', NAV_SPEED='3',
               NAV_RENDER_EVERY='100', NAV_TORCH_THREADS='1', CUDA_VISIBLE_DEVICES='-1',
               NAV_NO_SAVE='1', NAV_OBS_MS='64', NAV_TRACE=str(output / (stem + '.csv')),
               NAV_EVAL_OUTPUT_DIR=str(output), PYTHONUNBUFFERED='1')
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['NAV_ESCAPE_RECORD_DIR'] = str(output / 'escape_practice') if not profile['competitive'] else ''
    for name in ('NAV_SS_ROUTE', 'NAV_SS_TOUR', 'NAV_SS_PREP', 'NAV_SS_FOLLOW', 'NAV_THROUGH_GEM',
                 'NAV_SS_COMMAND_AIM', 'NAV_SS_YAW_FIX', 'NAV_GEO_ORDER'):
        env[name] = '0'
    result = dict(checkpoint=label, map=mission, profile=profile_name, status='failed',
                  stdout=str(stdout), stderr=str(stderr))
    runner = game = None
    start = time.monotonic()
    try:
        with stdout.open('w') as out, stderr.open('w') as err:
            runner = hidden_process([sys.executable, '-u', '-m', 'nav.real_run'], cwd=ROOT,
                                    env=env, stdout=out, stderr=err)
            wait_ready(runner, stdout)
            exe = ROOT.parent / 'Marble Blast Platinum/marbleblast_mbx.exe'
            game = hidden_process([str(exe), '-autotrain', mission, '-aiport', str(9200 + slot),
                                   '-offline', '-aicompetitive', str(int(profile['competitive'])),
                                   '-aiseed', str(seed)], cwd=exe.parent,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            code = runner.wait(timeout=1200)
            if code:
                raise RuntimeError(f'runner exit {code}: {stderr.read_text(errors="replace")[-1500:]}')
        path = output / f'real_run_{mission}_{checkpoint.stem}_{tag}.json'
        data = json.loads(path.read_text())
        rows = data['rounds']
        if len(rows) != rounds or not all(r['completed'] and r['engine_points'] is not None for r in rows):
            raise RuntimeError('incomplete rounds or missing final engine scores')
        if any(abs(r['points'] - r['engine_points']) > .01 for r in rows):
            raise RuntimeError('reward-derived points disagree with final engine score')
        metrics = ('points', 'points_per_min', 'falls', 'stalled_s', 'rescues', 'powerup_escapes',
                   'failed_escape_uses', 'group_completions', 'pathed_pct')
        result.update(status='ok', artifact=str(path), update=data.get('update'),
                      n=len(rows), mean={k: statistics.mean(r[k] for r in rows) for k in metrics},
                      points_se=statistics.stdev(r['engine_points'] for r in rows) / len(rows) ** .5 if len(rows) > 1 else None)
    except Exception as exc:
        result['reason'] = str(exc)
    finally:
        stop(game); stop(runner)
    result['wall_seconds'] = round(time.monotonic() - start, 2)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--instances', type=int, default=8)
    parser.add_argument('--maps', nargs='+')
    parser.add_argument('--profiles', nargs='+', default=['assisted', 'competitive'])
    parser.add_argument('--checkpoints', nargs='+')
    parser.add_argument('--checkpoint-file', type=Path, help='Evaluate this isolated candidate without editing the shared config')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.instances <= 8 or args.rounds < 1:
        parser.error('use 1..8 instances and at least one round')
    config = json.loads(CONFIG.read_text())
    if args.checkpoint_file:
        config['checkpoints']['candidate'] = str(args.checkpoint_file.resolve())
    labels = args.checkpoints or (['candidate'] if args.checkpoint_file else ['base', 'clusters'])
    maps = args.maps or config['maps']
    for mission in maps:
        if not (ROOT / f'terrain_maps/terrain_{mission}.npz').exists():
            parser.error(f'missing terrain for {mission}')
    checkpoints = {label: ROOT / config['checkpoints'][label] for label in labels}
    protected = {str(path): digest(path) for path in checkpoints.values()}
    jobs = [(label, checkpoint, mission, profile, config['profiles'][profile])
            for profile in args.profiles for mission in maps for label, checkpoint in checkpoints.items()]
    if not args.baseline:
        print(json.dumps(dict(config=config, jobs=len(jobs), rounds=args.rounds,
                             note='Preparation only. Pass --baseline for evaluations; no training command exists here.'), indent=2))
        return
    # Refuse to overlap the eight evaluation slots with another local game workload.
    if os.name == 'nt':
        check = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                                "@(Get-Process marbleblast_mbx -ErrorAction SilentlyContinue).Count"],
                               capture_output=True, text=True, check=True)
        if int(check.stdout.strip() or '0'):
            parser.error('close existing game instances before the baseline sweep')
    output = args.output or Path(tempfile.gettempdir()) / 'pq_checks' / ('phase_r_' + time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    print(f'BASELINE OUTPUT {output}', flush=True)
    results = []
    started = time.monotonic()
    # Reuse a port only after its job closes both processes. Long maps cannot idle other slots.
    slots = queue.Queue()
    for slot in range(args.instances):
        slots.put(slot)
    def run_job(job):
        slot = slots.get()
        try:
            return evaluate(job, slot, output, args.rounds, 10090 + maps.index(job[2]))
        finally:
            slots.put(slot)
    with ThreadPoolExecutor(max_workers=args.instances) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            (output / 'summary.json').write_text(json.dumps(dict(results=results), indent=2))
    unchanged = all(digest(Path(path)) == h for path, h in protected.items())
    report = dict(results=results, config=config, checkpoint_sha256=protected,
                  checkpoints_unchanged=unchanged, wall_seconds=round(time.monotonic() - started, 2))
    if (output / 'escape_practice').exists():
        from nav.escape_practice import assemble
        report['escape_practice'] = assemble(output / 'escape_practice', config['maps'])
    (output / 'summary.json').write_text(json.dumps(report, indent=2))
    print(f'FINISHED {len(results)} jobs in {report["wall_seconds"]:.1f}s; checkpoints unchanged={unchanged}', flush=True)
    if not unchanged or any(r['status'] != 'ok' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
