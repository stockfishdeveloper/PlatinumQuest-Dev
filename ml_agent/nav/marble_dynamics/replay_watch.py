"""Operator viewing: replay a recorded full round at 1x with the free camera.

The planner pools run the game in lockstep, so a live viewing freezes for up to
the search budget at every full search. This replays the exact controls a pool
journal recorded (decisions.jsonl.gz) into a fresh game on the same map, native
seed and random start, one 64 ms reply per 64 ms of wall clock, with no planner
in the loop. The game is deterministic in lockstep, so the marble should retrace
the recorded round; the drift between the live and recorded positions is printed
every 200 decisions, and when it grows past --resync-u the recorded native state
can be restored when the journal carries it (viewing only, never used for scoring).

Usage (repo root):
  PYTHONPATH=ml_agent python -m nav.marble_dynamics.replay_watch --journal <pool>/<map>/<seed>/physics_plan/1 [--port 9080]
"""
import argparse
import copy
import gzip
import json
import time
from pathlib import Path

import numpy as np

from nav.protocol import NOOP_ACTION
from nav.marble_dynamics.record import game, command
from nav.marble_dynamics.reliability_eval import await_full_round_context
from nav.marble_dynamics.commit_demo import restore


def load_round(folder):
    folder = Path(folder)
    rows = []
    with gzip.open(folder / 'decisions.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    initial = json.loads((folder / 'initial_context.json').read_text(encoding='utf-8'))
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8')) if (folder / 'report.json').exists() else {}
    return rows, initial, report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--journal', type=Path, required=True, help='round folder holding decisions.jsonl.gz and initial_context.json')
    p.add_argument('--port', type=int, default=9080)
    p.add_argument('--physics-frame-ms', type=int, default=16)
    p.add_argument('--resync-u', type=float, default=0.0, help='REPLAYRESTORE the recorded pose when the live marble drifts further than this (0 = never; it adds idle steps and refuses once gems differ, 2026-10-16)')
    p.add_argument('--snap-u', type=float, default=0.2, help='TELEPORT the live marble to the recorded position, velocity and spin when it drifts further than this (0 = never). Viewing only.')
    p.add_argument('--speed', type=float, default=1.0, help='wall-clock pacing factor (1 = real time)')
    p.add_argument('--headless', action='store_true', help='diagnostics: hidden game window, no free camera, pool render settings (no SPEED 1 / RENDEREVERY 1)')
    p.add_argument('--stop-at', type=int, default=0, help='diagnostics: stop after this many decisions (0 = whole round)')
    a = p.parse_args()
    rows, initial, report = load_round(a.journal)
    parts = a.journal.resolve().parts
    name = parts[-4]
    native_seed = int(report.get('native_seed') or rows[0]['before']['native'].get('seed') or parts[-3])
    print(f'replaying {name} seed {native_seed}: {len(rows)} decisions, recorded points {report.get("actual_points")}', flush=True)
    with game(name, a.port, obs_ms=64, seed=native_seed, process_cap=8, random_start=True, watch=not a.headless) as (env, proc):
        env.set_game_rules(False)
        while not env.round_ended:
            _, outcome = env.step(NOOP_ACTION, repeat=1)
            if outcome['reconnected']:
                raise RuntimeError('Reconnect during pre-round warmup')
        env.wait_new_round()
        env.control('FIXEDSTEP ' + str(a.physics_frame_ms) + ' 64')
        if not a.headless:
            env.speed = 1
            env.control('SPEED 1')
            env.control('RENDEREVERY 1')
        command(env, 'DYNAMICSLIVE 1', 'dynamics_live')
        env.replay_world = None
        env.control('REPLAYCAPTURE 1')
        neutral = await_full_round_context(env)
        want = int(initial.get('startup_neutral_decisions', neutral))
        for _ in range(max(0, want - neutral)):
            env.step(NOOP_ACTION, repeat=1)
        started = time.perf_counter()
        next_tick = started
        points = 0.0
        resyncs = 0
        worst = 0.0
        snaps = 0
        rec_points = 0.0
        for i, row in enumerate(rows):
            if a.stop_at and i >= a.stop_at:
                break
            action = row['action']
            live_state = env.msg.dynamics['native']['state']
            live = np.asarray(live_state['position'], dtype=float)
            rec_state = row['before']['native']['state']
            rec = np.asarray(rec_state['position'], dtype=float)
            drift = float(np.linalg.norm(live - rec))
            if (a.snap_u and drift > a.snap_u and not live_state.get('oob') and not rec_state.get('oob')
                    and live_state.get('mode', 0) == 0 and rec_state.get('mode', 0) == 0):
                # 2026-10-16: a 0.05-0.08 u difference appears 40-60 s into a Cube Isle round and grows by
                # chaos; REPLAYRESTORE needed idle steps and refused once a gem differed. TELEPORT sets the
                # recorded position, velocity and spin in place of this decision's action (viewing only).
                # the marble does not move during the decision that carries the TELEPORT (a cascade of 0.6 u
                # snaps every decision at ~9 u/s, 2026-10-16): place it where the recording is one decision later
                target = rows[i + 1]['before']['native']['state'] if i + 1 < len(rows) else rec_state
                if target.get('oob') or target.get('mode', 0) != 0:
                    target = rec_state
                words = [*target['position'], *target['velocity'], *target.get('spin', [0., 0., 0.])]
                env.control('TELEPORT ' + ' '.join(format(float(v), '.9g') for v in words))
                snaps += 1
                if snaps <= 5 or snaps % 25 == 0:
                    print(f'decision {i + 1}: drift {drift:.2f} u, teleported to the recorded state (snaps {snaps})', flush=True)
                points += max(0.0, float(env.control_info['gem_delta']))
                next_tick += 0.064 / max(a.speed, 1e-3)
                slack = next_tick - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_tick = time.perf_counter()
                continue
            if drift > 0.05 and worst <= 0.05:
                print(f'FIRST DIVERGENCE at decision {i + 1}: live {live.round(2).tolist()} recorded {rec.round(2).tolist()}; previous recorded outcome {rows[i - 1]["outcome"] if i else None}; recorded choice {row["choice"].get("label")}', flush=True)
            worst = max(worst, drift)
            if a.resync_u and drift > a.resync_u and isinstance(env.replay_world, dict) and env.replay_world.get('supported'):
                # Lockstep replay is deterministic only most of the time (a
                # 0.05 u difference at decision 510 of the v16 Sprawl s1 round
                # grew into a fall and four minutes of nonsense). Snap the live
                # marble back to the recorded pose; items and gem groups stay
                # as they are in the live world (viewing only).
                world = copy.deepcopy(env.replay_world)
                rec_state = row['before']['native']['state']
                pose = list(world['pose'])
                pose[0:3] = [float(v) for v in rec_state['position']]
                pose[3:6] = [float(v) for v in rec_state['velocity']]
                if len(pose) >= 9:
                    pose[6:9] = [float(v) for v in rec_state['spin']]
                world['pose'] = pose
                try:
                    restore(env, world)
                    resyncs += 1
                    print(f'decision {i + 1}: drift {drift:.2f} u, resynced to the recorded pose', flush=True)
                except Exception as exc:
                    print(f'decision {i + 1}: resync failed ({str(exc)[:80]}); continuing without it', flush=True)
                    a.resync_u = 0.0
            _, outcome = env.step(action[:6], repeat=1, use_pow=int(action[6]),
                                  pow_yaw=action[7] if action[6] else None, use_blast=int(action[8]))
            next_tick += 0.064 / max(a.speed, 1e-3)
            slack = next_tick - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_tick = time.perf_counter()
            points += max(0.0, float(outcome['gem_delta']))
            if outcome['round_ended'] or outcome['reconnected']:
                print(f'round ended at decision {i + 1}; points {points:.0f}', flush=True)
                break
            if outcome['fell'] or row['outcome'].get('fell'):
                print(f'decision {i + 1}: fell live={outcome["fell"]} recorded={row["outcome"].get("fell")}; sending OOBCLICK', flush=True)
                env.control('OOBCLICK')
                points += max(0.0, float(env.control_info['gem_delta']))
            rec_points = rec_points + max(0.0, float(row['outcome'].get('gem_delta') or 0.0)) if i else max(0.0, float(row['outcome'].get('gem_delta') or 0.0))
            if (i + 1) % 200 == 0:
                print(f'decision {i + 1}: points {points:.0f} (recorded {rec_points:.0f}), drift {drift:.2f} u, worst {worst:.2f} u, resyncs {resyncs}, snaps {snaps}, time left {env.time_left_s():.0f} s', flush=True)
        print(f'done: points {points:.0f} vs recorded {report.get("actual_points")}, worst drift {worst:.2f} u, resyncs {resyncs}, snaps {snaps}', flush=True)
        for _ in range(8):
            env.step(NOOP_ACTION, repeat=1)


if __name__ == '__main__':
    main()
