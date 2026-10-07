"""Read-only engine replay probe. No optimizer, trainer, or checkpoint writes.

Run against one separately launched -autotrain game on --port.
"""
import argparse
import copy
import json
from pathlib import Path

from nav.env import HuntEnv
from nav.gems import visible_gems
from nav.protocol import NOOP_ACTION
from nav.replay import restore


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=9978)
    p.add_argument('--out', default='logs/nav/replay_probe.json')
    p.add_argument('--powerup', action='store_true', help='collect a real Super Speed and probe inventory/timer/kick replay')
    args = p.parse_args()
    env = HuntEnv(args.port)
    try:
        env.connect(); env.control('REPLAYCAPTURE 1')
        for _ in range(300):
            if (env.replay_world and env.replay_world['supported'] and visible_gems(env.msg.obs)
                    and env.time_left_s() < 176 and abs(env.vel()[2]) < 0.3):
                break
            env.step(NOOP_ACTION)
        if not env.replay_world:
            raise RuntimeError('no replay telemetry (check script compilation)')
        if args.powerup:
            item = next(c for c in env.replay_catalog if not c[0] and 'SuperSpeed' in c[1])
            env.teleport(*item[2], spin=(0, 0, 0))
            for _ in range(30):
                if 'SuperSpeed' in env.replay_world['held']:
                    break
                env.step(NOOP_ACTION)
            else:
                raise RuntimeError('probe did not collect the real Super Speed item')
        gems = visible_gems(env.msg.obs)
        s = {'schema': 2, 'id': 'probe:0', 'split_group': 'probe', 'mission': env.info['mission'],
             'catalog': env.replay_catalog, 'world': copy.deepcopy(env.replay_world),
             'raw': env.msg.obs.tolist(), 'goal': list(gems[0][:3]),
             'next': list(gems[1][:3]) if len(gems) > 1 else None, 'hist': []}
        print('fixture', s['world'], flush=True)
        trials = []
        for _ in range(3):
            t0 = restore(env, s)
            path = []
            for decision in range(64 if args.powerup else 16):
                msg, info = env.step(NOOP_ACTION, use_pow=int(args.powerup and decision < 3), pow_yaw=0.0)
                path.append({'pose': msg.obs[:6].tolist(), 'points': info['gem_delta'],
                             'held': env.replay_world['held'], 'items': env.replay_world['items'],
                             'clock_ms': env.replay_world['clock_ms'] - t0})
            trials.append(path)
        error = max(abs(a - b) for t in trials[1:] for ra, rb in zip(trials[0], t)
                    for a, b in zip(ra['pose'], rb['pose']))
        out = {'max_repeat_error': error, 'trials': trials, 'fixture': s}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f'REPLAY PROBE max repeated trajectory error={error:.6f}', flush=True)
        if error > 0.02 or any(t != trials[0] for t in trials[1:]):
            raise RuntimeError('replay is not repeatable')
        if args.powerup and all(row['held'] != 'none' for row in trials[0]):
            raise RuntimeError('probe never fired the held powerup')
    finally:
        env.close()


if __name__ == '__main__':
    main()
