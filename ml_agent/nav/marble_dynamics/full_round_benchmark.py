"""Prepare or run a sealed full-round A2 reference, without importing a trainer.

The existing real_run controller handles native OOB and the assisted 8s rescue.
This reference uses its original 64ms physics/64ms decisions. It must not be
pooled with the experimental 16ms first-fall planning demonstrations.
Default invocation writes preparation only; --run-a2 runs read-only evaluation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import queue

from nav.collision.mesh import atomic_json
from nav.phase_r import evaluate
from nav.source_archive import snapshot_sources
from nav.marble_dynamics.record import ML, sha


EXPECTED_SETTINGS = dict(decision_ms=64, view_substeps=1, watch=False,
    oob_click=True, tour='walk', smooth=1.0, stuck_s=0.0, competitive=False,
    rescue_s=8.0, direct_gem=False, sample=False, no_use=False,
    force_use=False, ss_yaw_fix=False)


def checked_rounds(data, rounds):
    """Incomplete prefixes, first-fall runs and mismatched rules are not scores."""
    settings = data.get('evaluation_settings')
    if settings != EXPECTED_SETTINGS:
        raise ValueError('Full-round reference settings mismatch: ' + str(settings))
    rows = data.get('rounds', [])
    if len(rows) != rounds:
        raise ValueError('Missing requested complete rounds')
    for row in rows:
        if row.get('completed') is not True:
            raise ValueError('A reconnect/partial round is not a completed round')
        for key in ('engine_points', 'points', 'initial_score', 'round_seconds',
                    'start_time_left_s', 'played_seconds'):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Missing/nonfinite round field: ' + key)
        if row['initial_score'] != 0 or row['engine_points'] < 0:
            raise ValueError('Reference must start with zero score')
        if abs(row['points'] - row['engine_points']) > .01:
            raise ValueError('Accumulated pickups disagree with native final score')
        duration = row['round_seconds']
        if duration <= 0 or not duration - .256 <= row['start_time_left_s'] <= duration + .01:
            raise ValueError('Late/incomplete start is not a full-round reference')
        # REPLAYWORLD.clock_ms is getSimTime(), not the Hunt countdown. Extra
        # simulation/control/end-state intervals can make it exceed round_ms.
        # Require coverage, a full start and the native end score; don't invent
        # equality between these two independently recorded clocks.
        if row['played_seconds'] < duration - .256:
            raise ValueError('Observed play does not cover the expected round duration')
    return rows


def source_files(maps):
    # real_run imports flat nav modules and terrain_obs; keep code readable later.
    paths = list((ML / 'nav').glob('*.py'))
    paths += [Path(__file__), ML / 'nav/source_archive.py', ML / 'terrain_obs.py',
        ML / 'nav/collision/mesh.py', ML / 'nav/marble_dynamics/record.py',
        ML / 'configs/real_maps_20261009.json',
        ML.parent / 'Marble Blast Platinum/marbleblast_mbx.exe']
    ai = ML.parent / 'Marble Blast Platinum/platinum/client/scripts/ai'
    paths += list(ai.glob('*.cs'))
    for name in maps:
        paths += [ML / ('terrain_maps/terrain_' + name + '.npz'),
                  ML / ('collision_maps/' + name + '.npz')]
    return list(dict.fromkeys(p.resolve() for p in paths))


def prepare(output, maps, seeds, rounds, instances=1):
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = ML / 'models/nav/real_maps_20261010_A2/nav_latest.pth'
    paths = source_files(maps) + [checkpoint.resolve()]
    sources = {str(p): sha(p) for p in paths}
    snapshot_sources(output, [p for p in paths if p.suffix.lower() in ('.py', '.cs', '.json')])
    protocol = dict(schema='full_round_a2_reference_v1', maps=maps, seeds=seeds,
        rounds_per_seed=rounds, planned_rounds=len(maps)*len(seeds)*rounds,
        evaluation_instances=instances,
        checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha(checkpoint),
        settings=EXPECTED_SETTINGS, physics_frame_ms=64, decision_ms=64,
        seed_contract='Fresh native game for every map/seed; same seed list for later candidates. A supplied RNG seed does not certify identical observed starting worlds or countdown history. Archive actual start contexts; label exact pairing only after matching those states. Never select starts by outcome.',
        score_contract='Native end-message score after complete original-duration rounds. Continue through native falls; assisted rescue costs native round time. Missing/partial/timeout results invalidate the declared sweep.',
        selection='Equal weight per map, points primary; falls, rescues, power-use outcomes and wall-time secondary. Preserve every seed and every failure.',
        comparison_contract='Legacy-cadence A2 reference only. Candidate must repeat the same full-round protocol before comparison. A 16ms physics controller requires its own matched A2 reference; first-fall prefixes cannot be pooled.',
        clock_contract='Hunt round duration and native end message define the scored round. Replay getSimTime() elapsed is separately diagnostic and can exceed the countdown duration; it is not forced to equal the Hunt timer.',
        stage='reference_preparation', evaluation_started=False,
        student_fitting_started=False, policy_training_started=False,
        candidate_controller_ready=False, source_hashes=sources)
    atomic_json(output / 'protocol.json', protocol)
    return protocol


def run_reference(output, protocol):
    if os.name == 'nt':
        existing = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
            "@(Get-Process marbleblast_mbx -ErrorAction SilentlyContinue).Count"],
            check=True, capture_output=True, text=True)
        if int(existing.stdout.strip() or '0'):
            raise RuntimeError('An existing mbx game occupies the evaluation workload')
    # Prevent accidental inherited experiments from changing the sealed rules.
    # Parent-process changes are temporary and only affect owned evaluation jobs.
    inherited = {k: v for k, v in os.environ.items() if k.startswith('NAV_')}
    for key in inherited:
        del os.environ[key]
    os.environ.update(NAV_H_RESET='never', NAV_NO_NEXT='0', NAV_USE_EPS='0',
        NAV_OOB_CLICK='1', NAV_SMOOTH='1.0', NAV_FORCE_THROTTLE='0')
    jobs = []
    started = time.monotonic()
    slots = queue.Queue()
    for slot in range(protocol['evaluation_instances']):
        slots.put(slot)

    def job(name, seed):
        slot = slots.get()
        try:
            if (output / 'STOP').exists():
                raise RuntimeError('Operator stop: declared benchmark incomplete')
            if any(sha(Path(p)) != value for p, value in protocol['source_hashes'].items()):
                raise RuntimeError('Source/checkpoint changed after declaration')
            dest = output / name / str(seed)
            dest.mkdir(parents=True, exist_ok=False)
            atomic_json(dest / 'declared.json', dict(map=name, seed=seed, port=9200+slot,
                protocol_sha256=sha(output / 'protocol.json'), training=False))
            result = evaluate(('A2', Path(protocol['checkpoint']), name, 'assisted',
                dict(competitive=False, rescue_s=8.0)), slot, dest,
                protocol['rounds_per_seed'], seed)
            result.update(seed=seed, score_valid=False)
            if result['status'] == 'ok':
                try:
                    data = json.loads(Path(result['artifact']).read_text())
                    rows = checked_rounds(data, protocol['rounds_per_seed'])
                    result.update(score_valid=True,
                        native_round_points=[r['engine_points'] for r in rows],
                        artifact_sha256=sha(result['artifact']))
                except ValueError as exc:
                    result.update(status='invalid', reason=str(exc))
            if any(sha(Path(p)) != value for p, value in protocol['source_hashes'].items()):
                result.update(status='invalid', score_valid=False,
                    reason='Source/checkpoint changed during evaluation')
            return result
        finally:
            slots.put(slot)

    try:
        with ThreadPoolExecutor(max_workers=protocol['evaluation_instances']) as pool:
            futures = [pool.submit(job, name, seed) for name in protocol['maps']
                for seed in protocol['seeds']]
            for future in as_completed(futures):
                jobs.append(future.result())
                atomic_json(output / 'progress.json', dict(jobs=jobs,
                    evaluation_started=True, complete=False, training=False))
        complete = all(j['score_valid'] for j in jobs)
        per_map = {name: statistics.mean(p for j in jobs if j['map'] == name
            for p in j.get('native_round_points', [])) for name in protocol['maps']} if complete else None
        report = dict(jobs=jobs, complete=complete, valid_score_sweep=complete,
            per_map_mean_points=per_map,
            equal_map_mean_points=statistics.mean(per_map.values()) if complete else None,
            elapsed_wall_s=time.monotonic()-started, training=False,
            candidate_comparison=False, protocol_sha256=sha(output / 'protocol.json'))
        atomic_json(output / 'summary.json', report)
        if not complete:
            raise RuntimeError('At least one round is invalid; no aggregate score published')
        return report
    finally:
        for key in list(os.environ):
            if key.startswith('NAV_'):
                del os.environ[key]
        os.environ.update(inherited)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int, default=[481012, 481013, 481014])
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--instances', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-a2', action='store_true')
    args = p.parse_args()
    # phase_r starts its child with cwd=ml_agent, so every child artifact path
    # must be absolute even when this CLI was invoked from the repository root.
    args.output = args.output.resolve()
    config = json.loads((ML / 'configs/real_maps_20261009.json').read_text())
    maps = args.maps or config['maps']
    if args.rounds < 1 or not 1 <= args.instances <= 2 or len(set(maps)) != len(maps) or len(set(args.seeds)) != len(args.seeds):
        p.error('Positive rounds,1..2 instances and distinct maps/seeds required')
    if any(name not in config['maps'] for name in maps):
        p.error('Map must belong to the selected ten-map pool')
    protocol = prepare(args.output, maps, args.seeds, args.rounds, args.instances)
    print(json.dumps(dict(output=str(args.output), planned_rounds=protocol['planned_rounds'],
        evaluation_started=args.run_a2, training=False)), flush=True)
    if args.run_a2:
        result = run_reference(args.output, protocol)
        print(json.dumps(dict(complete=result['complete'],
            per_map_mean_points=result['per_map_mean_points'],
            wall_s=result['elapsed_wall_s'], training=False)), flush=True)


if __name__ == '__main__':
    main()
