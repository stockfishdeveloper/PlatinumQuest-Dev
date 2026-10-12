"""Finite real-game preparation probe, no model/optimizer. Diagnostic teleports are explicit.

python -m nav.preflight_r
"""
import json
import argparse
import os
from pathlib import Path
import tempfile
import math

# Set before importing worker modules, in this isolated diagnostic process.
os.environ.update(NAV_DRILL_PLAN='none', NAV_REAL_GEMS='1', NAV_TOUR='walk',
                  NAV_DIRECT_GEM='0', NAV_ESCAPE_POWERUPS='1', NAV_COMPETITIVE='0',
                  NAV_RESCUE_S='8', NAV_SPEED='1', NAV_RENDER_EVERY='100', NAV_LIVE_RECORD='0')
from nav.vec_worker import InstanceWorker
from nav.phase_r import ROOT, hidden_process, stop
from nav.protocol import NOOP_ACTION, RAW_POW_HELD, RAW_POW_BLAST
from nav.routing import RoutePlanner, line_clear
from nav.recovery import simulation_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--starts', type=Path, help='Also verify one normally recorded escape start in the engine')
    parser.add_argument('--output', type=Path, help='Save separately from earlier preparation evidence')
    parser.add_argument('--reconnects', type=int, default=1, help='Finite alternating Sprawl/Cragmire reconnection checks')
    args = parser.parse_args()
    if not 1 <= args.reconnects <= 10:
        parser.error('--reconnects must be between 1 and 10')
    worker = InstanceWorker(0, 9210, 123, 1., 1.)
    worker.load_map('Cragmire_Hunt')  # graph ready before launching the game
    env = worker.env
    game = None
    report = dict(map='Cragmire_Hunt', speed=1, diagnostic_teleports=True, training=False)
    try:
        exe = ROOT.parent / 'Marble Blast Platinum/marbleblast_mbx.exe'
        game = hidden_process([str(exe), '-autotrain', 'Cragmire_Hunt', '-aiport', '9210',
                               '-offline', '-aicompetitive', '0', '-aiseed', '10100'], cwd=exe.parent)
        worker.connect()
        countdown_checks = 0
        while env.time_left_s() >= env.info['round_ms'] / 1000.0:
            worker.step([0., 0., 0., 0., 0., 0., 0., 0.])
            if env.time_left_s() >= env.info['round_ms'] / 1000.0:
                assert worker.recovery.last_progress is None
                countdown_checks += 1
            if countdown_checks > 100:
                raise RuntimeError('Countdown did not finish')
        report['countdown_excluded_checks'] = countdown_checks
        env.control('REPLAYCAPTURE 1')
        for _ in range(300):
            if env.replay_world and env.replay_world['supported']:
                break
            env.step(NOOP_ACTION, repeat=1)
        else:
            raise RuntimeError('No supported world after GO')
        env.teleport(1, -2, 13.19, settle_ticks=8)
        route = RoutePlanner(worker.terrain)
        low = (1., -2., 13.); upper = (1., -2., 21.)
        route.navigate(env.pos(), low)
        low_distance = route.distance(env.pos())
        wp, routed = route.navigate(env.pos(), upper)
        report['under_deck'] = dict(position=env.pos().tolist(), low_distance=low_distance,
                                    upper_distance=route.distance(env.pos()), waypoint=wp, routed=routed)
        assert low_distance < 1 and routed and route.distance(env.pos()) > 30
        assert line_clear(worker.terrain, *env.pos()[:2], *wp[:2], az=env.pos()[2], bz=wp[2])
        # Stationary real worker transitions must trigger rescue after game-time progress timeout.
        worker.goal = upper; worker.segs.retarget(env.pos()[0], env.pos()[1], upper, None, z=env.pos()[2])
        worker.crop, worker.vec, worker.on_floor = worker._build()
        worker.recovery.reset()
        initial_s = simulation_time(env)
        rescued = None
        for _ in range(180):
            worker.on_floor = False  # terrain misses some wedged contacts; rescue cannot depend on this estimate
            reply = worker.step([0., 0., 0., 0., 0., 0., 0., 0.])
            if worker.recovery.rescues:
                rescued = dict(elapsed_s=simulation_time(env) - initial_s,
                               transition_steps=reply['duration_steps'], reward=reply['r'],
                               done=reply['done'], new_segment=reply['new_segment'],
                               position=env.pos().tolist(), metrics=worker.recovery.stats())
                break
        report['worker_rescue'] = rescued
        assert rescued is not None and rescued['elapsed_s'] >= 8
        assert rescued['new_segment'] and not rescued['done'] and math.isfinite(rescued['reward'])
        report['cooldown_refused'] = not env.quick_rescue()
        assert report['cooldown_refused']
        env.set_game_rules(True)
        report['competitive_refused'] = not env.quick_rescue()
        # Also challenge the engine gate directly, beyond the Python refusal.
        env.rescue_accepted = None; env.control('QUICKRESPAWN')
        for _ in range(16):
            if env.rescue_accepted is not None:
                break
            env.step(NOOP_ACTION, repeat=1)
        report['competitive_engine_refused'] = env.rescue_accepted is False
        assert report['competitive_engine_refused']
        env.set_game_rules(False)
        # A real map pickup, rather than GIVEPOW: verify the existing action reaches game physics.
        sj = max((row for row in env.replay_catalog if not row[0] and 'Jump' in row[1]), key=lambda row: row[2][2])
        sx, sy, sz = sj[2]
        floor = worker.terrain.floor_z(sx, sy, sz)
        env.teleport(sx, sy, max(sz, floor + .2), settle_ticks=8)
        for _ in range(12):
            if int(env.msg.obs[RAW_POW_HELD]) == 1:
                break
            env.step(NOOP_ACTION, repeat=1)
        assert int(env.msg.obs[RAW_POW_HELD]) == 1, 'normal Super Jump pickup failed'
        worker.crop, worker.vec, worker.on_floor = worker._build()
        z0 = float(env.pos()[2]); peak = z0; vertical_speed = 0.; fell = False
        for _ in range(40):
            use_sj = float(int(env.msg.obs[RAW_POW_HELD]) == 1)
            reply = worker.step([0., 0., 0., 0., 0., use_sj, 0., 0.])
            if reply.get('fell'):
                fell = True
                break
            peak = max(peak, float(env.pos()[2]))
            vertical_speed = max(vertical_speed, float(env.vel()[2]))
        report['super_jump'] = dict(normal_pickup=sj, rise_u=peak-z0,
                                    held_after=int(env.msg.obs[RAW_POW_HELD]), vertical_speed=vertical_speed, fell=fell)
        assert peak - z0 > 2 and vertical_speed > 10 and int(env.msg.obs[RAW_POW_HELD]) != 1
        # Blast charges through ordinary game time. No meter/inventory is fabricated.
        env.teleport(1, -2, 13.19, settle_ticks=8)
        for _ in range(1000):
            if float(env.msg.obs[RAW_POW_BLAST]) >= .3 and int(env.msg.obs[RAW_POW_HELD]) == 0:
                break
            env.step(NOOP_ACTION, repeat=1)
        charge = float(env.msg.obs[RAW_POW_BLAST])
        worker.crop, worker.vec, worker.on_floor = worker._build()
        z0 = float(env.pos()[2]); peak = z0; minimum = charge
        for _ in range(25):
            worker.step([0., 0., 0., 0., 0., 1., 0., 0.])
            peak = max(peak, float(env.pos()[2])); minimum = min(minimum, float(env.msg.obs[RAW_POW_BLAST]))
        report['blast'] = dict(charge_before=charge, minimum_after=minimum, rise_u=peak-z0)
        assert minimum < charge - .1 and peak-z0 > .2
        report['reconnections'] = []
        for index in range(args.reconnects):
            mission = 'Sprawl_Hunt' if index % 2 == 0 else 'Cragmire_Hunt'
            stop(game)
            game = hidden_process([str(exe), '-autotrain', mission, '-aiport', '9210',
                                   '-offline', '-aicompetitive', '0', '-aiseed', str(10101 + index)], cwd=exe.parent)
            reply = worker.step([0., 0., 0., 0., 0., 0., 0., 0.])
            report['reconnected_map'] = dict(mission=worker.mission, game_rules=env.game_rules,
                                           telemetry=env.replay_world is not None, map_rounds=worker.map_rounds,
                                           new_segment=reply['new_segment'], skip=reply['skip'])
            report['reconnections'].append(report['reconnected_map'])
            assert worker.mission == mission and env.replay_world is not None
            assert env.game_rules == {'competitive': False} and reply['new_segment'] and reply['skip']
        if args.starts:
            from nav.replay import load_starts
            start = load_starts(args.starts)[0]
            stop(game)
            game = hidden_process([str(exe), '-autotrain', start['mission'], '-aiport', '9210',
                                   '-offline', '-aicompetitive', '0', '-aiseed', '10102'], cwd=exe.parent)
            worker.step([0., 0., 0., 0., 0., 0., 0., 0.])
            for _ in range(300):
                if env.replay_world and env.replay_world['supported']:
                    break
                env.step(NOOP_ACTION, repeat=1)
            else:
                raise RuntimeError('Replay mission did not reach a supported state')
            worker.begin_replay(start)  # verifies catalog, pose, inventory, items, RNG, clock and score
            report['restored_practice'] = dict(id=start['id'], mission=start['mission'],
                                             resource=start.get('resource'), warm_observations=len(worker.warm),
                                             held=env.replay_world['held'], status='ok')
        report['status'] = 'ok'
    except Exception as exc:
        report.update(status='failed', reason=repr(exc), worker_logs=worker.drain_log())
        raise
    finally:
        env.close(); stop(game)
        path = args.output or Path(tempfile.gettempdir()) / 'pq_checks/phase_r_preflight_20261009.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        print(f'PREFLIGHT {path}', flush=True)


if __name__ == '__main__':
    main()
