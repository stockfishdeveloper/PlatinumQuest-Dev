"""Reproducible hidden native rounds for the menu-play planner.

Run maps sequentially: concurrent games distort real-time latency. The first
unscored native round warms the bridge, matching reliability_eval. The measured
round uses ordinary gem RNG and no restores, grants or teleports. Evidence and
failures are retained; only this tool's own game and planner workers are closed.

PYTHONPATH=ml_agent python ml_agent/tools/realtime_bench.py --maps Sprawl_Hunt \
    --seed 7347019 --output ml_agent/dynamics_reports/realtime_planner_20261019/live_base
"""
import argparse
from contextlib import contextmanager
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from nav.collision.mesh import atomic_json
from nav.env import HuntEnv
from nav.marble_dynamics import realtime as rt
from nav.marble_dynamics.record import ROOT, ML, command, sha
from nav.marble_dynamics.reliability_eval import start_layout
from nav.marble_dynamics.static_cache import guarded_launch
from nav.protocol import NOOP_ACTION


def source_identity():
    paths = list((ML / 'nav').rglob('*.py'))
    paths += list((ML / 'nav/native').glob('*.cpp')) + list((ML / 'nav/native').glob('*.inc'))
    paths += [Path(__file__), ROOT / 'Marble Blast Platinum/marbleblast_realtime.exe',
              ROOT / 'Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs',
              ROOT / 'Marble Blast Platinum/platinum/client/scripts/physics.cs',
              ROOT / 'Marble Blast Platinum/platinum/server/scripts/game.cs']
    from nav.native.marble_sim import DEFAULT_DLL
    paths.append(DEFAULT_DLL)
    paths.append(Path(rt.FLAGS[rt.FLAGS.index('--sweep-dll') + 1]))
    return {str(p.resolve()): sha(p) for p in paths}


def freeze_sources(output):
    """Keep the exact tested source and native binaries as verified evidence."""
    from nav.source_archive import snapshot_sources
    sources = source_identity()
    snapshot_sources(output, sources, root=Path(os.path.commonpath(list(sources))))
    if any(sha(path) != expected for path, expected in sources.items()):
        raise RuntimeError('Sources changed while preparing native benchmark')
    return sources


class BenchEnv(HuntEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rescue_attempts = self.accepted_rescue_count = 0
        self.trace = None

    def step(self, *args, **kwargs):
        before = self.msg
        t = time.perf_counter()
        msg, info = super().step(*args, **kwargs)
        if self.trace is not None and before is not None:
            self.trace.write(json.dumps(dict(tick=before.tick, raw=before.obs.tolist(),
                dynamics=before.dynamics, args=args, kwargs=kwargs, outcome=info,
                step_wall_s=time.perf_counter()-t), separators=(',', ':')) + '\n')
        return msg, info

    def quick_rescue(self):
        self.rescue_attempts += 1
        accepted = super().quick_rescue()
        self.accepted_rescue_count += int(bool(accepted))
        return accepted


@contextmanager
def native_game(name, seed, port, folder):
    env = BenchEnv(port, speed=3, log=rt.log)
    env.obs_ms = 64
    env.srv.settimeout(70)
    exe = ROOT / 'Marble Blast Platinum/marbleblast_realtime.exe'
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    proc = None
    with (folder / 'game.log').open('wb') as log:
        try:
            proc = guarded_launch([str(exe), '-autotrain', name, '-aiport', str(port),
                '-offline', '-aicompetitive', '0', '-aiseed', str(seed), '-airandomstart'],
                8, cwd=str(exe.parent), startupinfo=startup, stdout=log, stderr=log)
            env.connect()
            env.conn.settimeout(35)
            if env.info.get('mission') != name:
                raise RuntimeError('Wrong native map: ' + str(env.info))
            env.control('SPINWAIT 4')
            command(env, 'DYNAMICSWAIT 600000', 'dynamics_wait')
            env.control('FIXEDSTEP 64')
            env.control('CONTACT 1')
            env.step(NOOP_ACTION, repeat=400)
            env.set_game_rules(False)
            while not env.round_ended:
                _, outcome = env.step(NOOP_ACTION, repeat=1)
                if outcome['reconnected']:
                    raise RuntimeError('Reconnect during native warmup')
            env.wait_new_round()
            yield env, proc
        finally:
            env.close()
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps', nargs='+', default=['Sprawl_Hunt', 'Cragmire_Hunt', 'TreasureBox_Hunt'])
    p.add_argument('--seed', type=int, default=7347019)
    p.add_argument('--port', type=int, default=9380)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--search-workers', type=int, default=8)
    p.add_argument('--lead', type=int, default=5)
    p.add_argument('--checked-tail', action='store_true', help='Experimental state-matched checked plan continuation')
    p.add_argument('--legacy-fast-proposals', action='store_true', help='Matched control for the repaired native fast-controller bug')
    p.add_argument('--trace', action='store_true', help='Full before-observation trace; adds measured I/O overhead')
    p.add_argument('--mega-probe', action='store_true', help='Unscored granted-Mega mechanics trials, not scored rounds')
    p.add_argument('--restore-probe', action='store_true', help='Unscored saved-powerup respawn timing measurement')
    p.add_argument('--competitive', action='store_true', help='Mega probe: measure the competitive duration')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 0 < a.seed < 2147383647 or not 0 <= a.search_workers <= 16 or a.rounds < 1:
        p.error('Invalid seed, worker count, or round count')
    a.output.mkdir(parents=True, exist_ok=False)
    if a.restore_probe:
        restore_probe(a)
        return
    if a.mega_probe:
        mega_probe(a)
        return
    sys.setswitchinterval(.001)  # match realtime.serve's game/planner thread scheduling
    rt.CHECKED_TAIL = a.checked_tail
    rt.SEARCH_WORKERS = a.search_workers
    if a.legacy_fast_proposals:
        rt.FLAGS = rt.FLAGS + ['--legacy-fast-proposals']
    # AheadPlanner's default is bound at import; make the benchmark override explicit.
    rt.AheadPlanner.__init__.__defaults__ = (a.lead,)
    sources = freeze_sources(a.output)
    from nav.marble_dynamics.parallel_search import module_globals
    atomic_json(a.output / 'prepared.json', dict(args=vars(a) | {'output': str(a.output)},
        sources=sources, planner_flags=rt.FLAGS, globals=module_globals(),
        runtime=dict(thread_switch_interval_s=sys.getswitchinterval(),
                     hold_reporting_threshold_s=.02, physics_frame_ms=16, decision_ms=64),
        contract='Sequential native real-time rounds; ordinary randomized openings; no scored world restores'))
    reports = []
    started = time.perf_counter()
    try:
        for name in a.maps:
            folder = a.output / name
            folder.mkdir()
            world = rt.World(name)
            try:
                with native_game(name, a.seed, a.port, folder) as (env, proc):
                    for ri in range(a.rounds):
                        rt.setup_round(env)
                        if env.python_paced:
                            raise RuntimeError('This benchmark requires native real-time pacing')
                        env.replay_world = None
                        env.control('REPLAYCAPTURE 1')
                        if not rt.wait_context(env):
                            raise RuntimeError('Missing initial native context')
                        for _ in range(32):
                            if env.replay_world is not None:
                                break
                            _, outcome = env.step(NOOP_ACTION, repeat=1)
                            if outcome['round_ended'] or outcome['reconnected']:
                                raise RuntimeError('Native round changed before opening RNG capture')
                        if env.replay_world is None:
                            raise RuntimeError('Missing native opening RNG capture')
                        opening = start_layout(env.msg.dynamics, env.replay_world)
                        expected = a.seed + opening['eval_round'] - 1
                        if opening['engine_seed'] != expected or not opening['random_opening']:
                            raise RuntimeError('Native opening seed audit failed')
                        initial = dict(raw=env.msg.obs.tolist(), dynamics=env.msg.dynamics,
                            layout=opening, process_id=proc.pid)
                        atomic_json(folder / ('initial_%d.json' % ri), initial)
                        env.rescue_attempts = env.accepted_rescue_count = 0
                        if a.trace:
                            env.trace = gzip.open(folder / ('observations_%d.jsonl.gz' % ri), 'wt', encoding='utf-8')
                        try:
                            report = rt.play_round(env, world, folder)
                        finally:
                            if env.trace is not None:
                                env.trace.close()
                                env.trace = None
                        if (report['rescue_attempts'] != env.rescue_attempts or
                                report['accepted_rescues'] != env.accepted_rescue_count):
                            raise RuntimeError('Round and connection rescue counts disagree')
                        report.update(round=ri, opening=opening,
                            source_unchanged=all(sha(k) == v for k, v in sources.items()))
                        reports.append(report)
                        atomic_json(a.output / 'summary.json', dict(rounds=reports, complete=False))
                        if report['stop'] != 'round_end' or not report['source_unchanged']:
                            raise RuntimeError('Incomplete round or source mutation')
                        if ri + 1 < a.rounds:
                            env.wait_new_round()
            finally:
                world.close()
        atomic_json(a.output / 'summary.json', dict(rounds=reports, complete=True,
            elapsed_wall_s=time.perf_counter()-started))
    except BaseException:
        atomic_json(a.output / 'failure.json', dict(traceback=traceback.format_exc(),
            completed_rounds=len(reports), elapsed_wall_s=time.perf_counter()-started))
        raise


def mega_probe(args):
    """Measure script transitions at the production 16 ms/64 ms bridge cadence.

    Fixtures deliberately grant/teleport for mechanics tests; their scores are
    never included in planner evaluations. Record exact native ticks and bridge
    observations so activation, parameter changes and expiry can be replayed.
    """
    import copy
    import numpy as np
    from nav.marble_dynamics.record import game, reset, archive, RUNTIME
    sources = freeze_sources(args.output)
    atomic_json(args.output / 'prepared.json', dict(sources=sources, maps=args.maps,
        competitive=args.competitive, physics_frame_ms=16, decision_ms=64,
        contract='Unscored mechanics fixtures: grants and teleports, no policy training'))
    results = []
    for name in args.maps:
        old = ML / 'datasets/marble_dynamics/timers_20261012' / name / 'manifest.json'
        manifest = json.loads(old.read_text(encoding='utf-8'))
        base = next(t['spec'] for t in manifest['trials'] if t['spec']['powerup'] == 'MegaMarbleItem')
        with game(name, args.port, obs_ms=64, seed=args.seed, process_cap=8) as (env, proc):
            env.control('FIXEDSTEP 16 64')
            command(env, 'DYNAMICSLIVE 1', 'dynamics_live')
            env.set_game_rules(args.competitive)
            for kind in ('stationary', 'rolling', 'airborne'):
                folder = args.output / name / kind
                folder.mkdir(parents=True)
                spec = copy.deepcopy(base)
                if kind == 'stationary':
                    spec['velocity'] = [0., 0., 0.]
                    spec['spin'] = [0., 0., 0.]
                elif kind == 'airborne':
                    spec['position'][2] += 6.
                    spec['velocity'] = [4., 0., 0.]
                    spec['spin'] = [0., 0., 0.]
                reset(env, spec)
                native = RUNTIME / ('mega64_%s_%s_%s.jsonl' % (name, kind, int(args.competitive)))
                command(env, 'DYNAMICSRECORD ' + native.as_posix(), 'dynamics_record')
                use_at = 0 if kind == 'airborne' else 8
                with gzip.open(folder / 'observations.jsonl.gz', 'wt', encoding='utf-8') as trace:
                    for i in range(220):
                        before = env.msg
                        action = [0., 0., 0., 0., 0, 0., int(i == use_at), 0., 0]
                        msg, outcome = env.step(action[:6], repeat=1, use_pow=action[6], pow_yaw=0. if action[6] else None)
                        trace.write(json.dumps(dict(index=i, tick=before.tick, raw=before.obs.tolist(),
                            dynamics=before.dynamics, action=action, outcome=outcome), separators=(',', ':')) + '\n')
                        if outcome['round_ended'] or outcome['reconnected']:
                            raise RuntimeError('Native round changed inside Mega measurement')
                        if outcome['fell']:
                            break
                command(env, 'DYNAMICSRECORD off', 'dynamics_record')
                archive(native, folder / 'native.jsonl.gz')
                with native.open(encoding='utf-8') as f:
                    rows = [r for r in map(json.loads, f) if r.get('kind') == 'transition']
                transitions = []
                for i, (x, y) in enumerate(zip(rows, rows[1:])):
                    changes = {}
                    for key in ('radius', 'velocity', 'position', 'spin'):
                        delta = np.asarray(y['before'][key])-np.asarray(x['after'][key])
                        if np.max(np.abs(delta)) > 1e-5:
                            changes[key] = dict(before=x['after'][key], after=y['before'][key])
                    if x['after']['parameters'] != y['before']['parameters']:
                        changes['parameters'] = {k:[x['after']['parameters'][k],v]
                            for k,v in y['before']['parameters'].items() if x['after']['parameters'][k] != v}
                    if changes:
                        transitions.append(dict(tick=i+1, sim_ms=y['sim_ms'], changes=changes))
                result = dict(map=name, kind=kind, competitive=args.competitive, transitions=transitions,
                    native_ticks=len(rows), spec=spec, fell=bool(outcome['fell']), native_sha256=sha(native))
                atomic_json(folder / 'report.json', result)
                results.append(result)
                print(json.dumps(result), flush=True)
    atomic_json(args.output / 'summary.json', dict(trials=results, complete=True,
        source_unchanged=all(sha(k) == v for k,v in sources.items())))


def restore_probe(args):
    from nav.marble_dynamics.record import game, archive, RUNTIME
    sources = freeze_sources(args.output)
    reports = []
    for name in args.maps:
        folder = args.output / name
        folder.mkdir()
        with game(name, args.port, obs_ms=64, seed=args.seed, process_cap=8) as (env, proc):
            env.control('FIXEDSTEP 16 64')
            command(env, 'DYNAMICSLIVE 1', 'dynamics_live')
            env.control('GIVEPOW SuperJumpItem'); env.step(NOOP_ACTION, repeat=8)
            native = RUNTIME / ('restore64_' + name + '.jsonl')
            command(env, 'DYNAMICSRECORD ' + native.as_posix(), 'dynamics_record')
            env.set_game_rules(False)
            if not env.quick_rescue():
                raise RuntimeError('Legal quick respawn was refused in restore probe')
            with gzip.open(folder / 'observations.jsonl.gz', 'wt', encoding='utf-8') as trace:
                for i in range(100):
                    before = env.msg
                    action = [0.,0.,0.,0.,0,0.,int(i==50),0.,0]
                    _, outcome = env.step(action[:6], repeat=1, use_pow=action[6], pow_yaw=0. if action[6] else None)
                    trace.write(json.dumps(dict(index=i,tick=before.tick,raw=before.obs.tolist(),
                        dynamics=before.dynamics,action=action,outcome=outcome),separators=(',',':'))+'\n')
                    if outcome['round_ended'] or outcome['reconnected']:
                        raise RuntimeError('Round changed in restore measurement')
            command(env, 'DYNAMICSRECORD off', 'dynamics_record')
            archive(native, folder / 'native.jsonl.gz')
            reports.append(dict(map=name,native_sha256=sha(native)))
    atomic_json(args.output / 'summary.json',dict(complete=True,reports=reports,
        source_unchanged=all(sha(k)==v for k,v in sources.items()),
        contract='Unscored deliberate grant and legal respawn; no policy training'))


if __name__ == '__main__':
    main()
