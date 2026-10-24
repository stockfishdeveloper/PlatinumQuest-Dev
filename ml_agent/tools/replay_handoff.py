"""Replay a recorded pool round bit-exactly up to a decision, then hand the marble to the human and record the demo.

The planner pools are seeded and run the game in lockstep, so a pool journal (decisions.jsonl.gz) replays exactly
(nav/marble_dynamics/replay_watch.py). This tool replays the recorded controls to --stop-at, then answers the game
with "RECORD": the game switches to recording mode (mlAgent.cs enableRecording: the human's inputs drive the marble,
the view pin is released, the inputs are appended to every message) and the rest of the round is recorded exactly as
record_demos.py records a demo (ml_agent/demos/demo_<stamp>.npz). The gems, the clock and the marble's state at the
hand-off are the recorded round's. Usage (repo root):
  PYTHONPATH=ml_agent python ml_agent/tools/replay_handoff.py --journal <pool>/<map>/<seed>/physics_plan/1 --stop-at 2290 [--speed 6]
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
from nav.protocol import NOOP_ACTION, parse_message
from nav.marble_dynamics.record import game, command
from nav.marble_dynamics.reliability_eval import await_full_round_context
from nav.marble_dynamics.replay_watch import load_round
import record_demos as RD


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--journal', type=Path, required=True)
    p.add_argument('--stop-at', type=int, required=True, help='hand the marble over after this many recorded decisions')
    p.add_argument('--port', type=int, default=9080)
    p.add_argument('--speed', type=float, default=4.0, help='replay pacing before the hand-off (4 = four times real time)')
    p.add_argument('--snap-u', type=float, default=0.2)
    p.add_argument('--out', default=str(ML / 'demos'))
    a = p.parse_args()
    rows, initial, report = load_round(a.journal)
    parts = a.journal.resolve().parts
    name = parts[-4]
    native_seed = int(report.get('native_seed') or rows[0]['before']['native'].get('seed') or parts[-3])
    stop = min(a.stop_at, len(rows) - 1)
    t_left = float(rows[stop]['raw_observation'][31]) / 1000.
    print(f'replaying {name} seed {native_seed} to decision {stop} ({t_left:.0f} s left on the clock), then your controls', flush=True)
    rec = RD.DemoRecorder(RD.resolve_terrain_path(name, False), a.out)
    with game(name, a.port, obs_ms=64, seed=native_seed, process_cap=8, random_start=True, watch=True, freecam=False) as (env, proc):
        env.set_game_rules(False)
        while not env.round_ended:
            _, outcome = env.step(NOOP_ACTION, repeat=1)
            if outcome['reconnected']:
                raise RuntimeError('Reconnect during pre-round warmup')
        env.wait_new_round()
        env.control('FIXEDSTEP 16 64')
        env.speed = 1
        env.control('SPEED 1')
        env.control('RENDEREVERY 1')
        command(env, 'DYNAMICSLIVE 1', 'dynamics_live')
        env.control('REPLAYCAPTURE 1')
        neutral = await_full_round_context(env)
        want = int(initial.get('startup_neutral_decisions', neutral))
        for _ in range(max(0, want - neutral)):
            env.step(NOOP_ACTION, repeat=1)
        next_tick = time.perf_counter(); snaps = 0; worst = 0.
        for i, row in enumerate(rows):
            if i >= stop:
                break
            live_state = env.msg.dynamics['native']['state']
            live = np.asarray(live_state['position'], dtype=float)
            rec_state = row['before']['native']['state']
            recp = np.asarray(rec_state['position'], dtype=float)
            drift = float(np.linalg.norm(live - recp)); worst = max(worst, drift)
            if (a.snap_u and drift > a.snap_u and not live_state.get('oob') and not rec_state.get('oob')
                    and live_state.get('mode', 0) == 0 and rec_state.get('mode', 0) == 0):
                target = rows[i + 1]['before']['native']['state'] if i + 1 < len(rows) else rec_state
                if target.get('oob') or target.get('mode', 0) != 0:
                    target = rec_state
                words = [*target['position'], *target['velocity'], *target.get('spin', [0., 0., 0.])]
                env.control('TELEPORT ' + ' '.join(format(float(v), '.9g') for v in words)); snaps += 1
            else:
                action = row['action']
                env.step(action[:6], repeat=1, use_pow=int(action[6]), pow_yaw=action[7] if action[6] else None,
                         use_blast=int(action[8]) if len(action) > 8 else 0)
            if i % 400 == 0:
                print(f'  decision {i}: drift {drift:.2f} u (worst {worst:.2f}, snaps {snaps})', flush=True)
            next_tick += 0.064 / max(a.speed, 1e-3)
            slack = next_tick - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_tick = time.perf_counter()
        st = env.msg.dynamics['native']['state']
        print(f'HAND-OFF at decision {stop}: marble at {np.round(st["position"], 1).tolist()}, {float(np.linalg.norm(st["velocity"])):.1f} u/s, '
              f'{float(env.msg.obs[31]) / 1000.:.0f} s left. The marble is yours: play to the end of the round.', flush=True)
        # recording mode: answer every message with RECORD and record the human's inputs (record_demos.DemoRecorder)
        # leave the lockstep (the fixed-step build takes its move from the bridge action each step and ignores the
        # keyboard: first hand-off, 24 s at 0.2 u/s with the operator's inputs recorded but unused); normal engine
        # timing again, then the recording handshake
        env.control('FIXEDSTEP 0')
        env.control('LOCKSTEP 0')
        env._send('RECORD')
        ticks = 0
        try:
            while True:
                line = env._readline()
                if line is None:
                    break
                msg = parse_message(line)
                if msg.kind == 'obs' or line.startswith('['):
                    try:
                        rec.handle(line.strip())
                    except Exception as exc:
                        rec.malformed += 1
                        if rec.malformed <= 5:
                            print(f'  bad message ({exc}): {line[:100]}', flush=True)
                    ticks += 1
                    if ticks % 1500 == 0:
                        rec.report(final=False)
                    parts_ = line.split('|')
                    if len(parts_) >= 4 and parts_[0] == '[]' and int(float(parts_[3])):
                        print('round over', flush=True)
                        break
                env._send('RECORD')
        except KeyboardInterrupt:
            pass
        finally:
            rec.save(); rec.report(final=True)


if __name__ == '__main__':
    main()
