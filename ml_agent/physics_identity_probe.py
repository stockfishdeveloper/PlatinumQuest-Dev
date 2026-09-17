"""
Are the marble physics identical at 3x and 10x game speed?

    python physics_identity_probe.py            (game launched with -autotrain <Mission>)

Connects to one game instance on any Hunt map. The start pose is the marble's
own spawn position on the first message (so no map knowledge is needed). For
each trial it sets the speed, teleports the marble back to that pose with zero
velocity, waits until it is at rest, runs a fixed back-and-forth key script
(20 s of game time: forward/back/left/right pairs, diagonals, jumps), then sends no keys until
the marble is at rest again and records where it stopped. Trials alternate
3x / 10x so that both speeds see the same machine conditions. A respawn
(position jump > 5 u in one tick) marks the trial as fallen.

Output: the rest position of every trial, the spread WITHIN each speed (the
noise floor of the async bridge: the reply to tick t may land on tick t or t+1)
and the distance BETWEEN the 3x and 10x means. Physics is "identical" if the
between-speed distance is no larger than the within-speed spread. Also compares
the whole trajectory tick by tick against the first 3x trial.
Results: logs/physics_identity_<port>.json
"""
import os
import sys
import json
import time
import socket
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = [3, 10, 3, 10]                 # alternating trials
NOOP = '0,0,0,0,0,0.000000,0'
START = None                           # set from the marble's spawn position on the first message
REST_POSE = [None]                     # rested pose of trial 1, reused as the exact start of every trial
# (ticks, fwd, back, left, right, jump). 20 s of game time (1250 ticks at
# 16 ms): opposite-direction pairs of equal length so the marble wanders but
# stays near the start, with a jump at the start of every third segment.
# Lengths come from a fixed seed, so every run uses the same script.
SCRIPT_TICKS_TARGET = 1250


def make_script(seed=7):
    import random
    rng = random.Random(seed)
    pairs = [((1, 0, 0, 0), (0, 1, 0, 0)), ((0, 0, 1, 0), (0, 0, 0, 1)),
             ((1, 0, 1, 0), (0, 1, 0, 1)), ((1, 0, 0, 1), (0, 1, 1, 0))]
    out, total, k = [], 0, 0
    while total < SCRIPT_TICKS_TARGET:
        a, b = pairs[k % len(pairs)]; n = rng.randint(20, 55)
        for d in (a, b):
            jump = 1 if k % 3 == 0 else 0
            if jump:
                out.append((4, *d, 1)); out.append((n - 4, *d, 0))
            else:
                out.append((n, *d, 0))
            total += n; k += 1
    return out


SCRIPT = make_script() if not os.environ.get('PROBE_SHORT') else [(35, 1, 0, 0, 0, 0), (45, 0, 1, 0, 0, 0), (30, 0, 0, 1, 0, 0), (40, 0, 0, 0, 1, 0)]
SCRIPT_TICKS = sum(s[0] for s in SCRIPT)
REST_SPEED = 0.005                     # |v| below this for REST_TICKS consecutive ticks = at rest
REST_TICKS = 40
REST_CAP = 1500                        # give up waiting for rest after this many ticks
SETTLE_AFTER_SPEED = 60


def script_cmd(i):
    for n, f, b, l, r, j in SCRIPT:
        if i < n:
            return f'{f},{b},{l},{r},{j},0.000000,0'
        i -= n
    return NOOP


class Probe:
    def __init__(self):
        self.trials = []
        self.log_lines = []

    def log(self, s):
        print(s, flush=True); self.log_lines.append(f"[{datetime.now():%H:%M:%S}] {s}")

    def run(self, conn, remaining):
        f = conn.makefile('r'); tick = ['']; stats = ['?']

        def recv():
            while True:
                line = f.readline()
                if not line:
                    raise ConnectionError('game disconnected')
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                if parts[0] == 'STATS':            # answer to the STATS control reply
                    stats[0] = parts[1] if len(parts) > 1 else '?'
                    continue
                try:
                    obs = json.loads(parts[0])
                except ValueError:
                    obs = []
                tick[0] = parts[-1].strip() if len(parts) >= 5 and parts[-1].strip().isdigit() else ''
                if len(obs) >= 35:
                    return obs
                send(NOOP)

        def send(reply):
            # action replies carry the tick they answer (fixed action delay in the game)
            if reply[:1].isdigit() and tick[0]:
                reply += f',t{tick[0]}'
            conn.sendall((reply + '\n').encode())

        def wait_running(min_time_left_s):
            while True:
                obs = recv()
                if obs[31] > min_time_left_s * 1000:
                    return obs
                send(NOOP)

        def wait_rest(obs):
            """Send no keys until the marble has been (nearly) still for REST_TICKS ticks."""
            still = 0
            for k in range(REST_CAP):
                v = float(np.linalg.norm(obs[3:6]))
                still = still + 1 if v < REST_SPEED else 0
                if still >= REST_TICKS:
                    return obs, k
                send(NOOP); obs = recv()
            return obs, REST_CAP

        global START
        while remaining:
            speed = remaining[0]
            idx = len(self.trials)
            self.log(f"=== trial {idx + 1}/{len(PLAN)}: {speed}x ===")
            obs = wait_running(90)                       # enough round time for the whole trial even at 3x
            if START is None:
                START = (round(float(obs[0]), 3), round(float(obs[1]), 3), round(float(obs[2]), 3) + 0.3)
                self.log(f"   start pose = marble spawn position {START}")
            send(f'SPEED {speed}')
            if os.environ.get('PROBE_DELAY') and idx == 0:
                send(f"DELAY {os.environ['PROBE_DELAY']}")
            for _ in range(SETTLE_AFTER_SPEED):
                obs = recv(); send(NOOP)
            send(f'TELEPORT {START[0]} {START[1]} {START[2]} 0 0 0')
            obs = recv()
            obs, _ = wait_rest(obs)
            # Re-teleport to the rested pose with exactly zero velocity so every
            # trial starts from the same floating-point state (settling from
            # 0.3 u above the floor leaves ~1e-4 u differences that a chaotic
            # 20 s script amplifies to metres).
            if REST_POSE[0] is None:
                REST_POSE[0] = tuple(round(float(v), 4) for v in obs[0:3])
            rp = REST_POSE[0]
            send(f'TELEPORT {rp[0]} {rp[1]} {rp[2]} 0 0 0'); obs = recv()
            for _ in range(20):
                send(NOOP); obs = recv()
            start = [round(float(v), 4) for v in obs[0:3]]
            traj = []; t0 = time.perf_counter(); fell = False; prev_pos = np.array(obs[0:3])
            for i in range(SCRIPT_TICKS):
                send(script_cmd(i)); obs = recv()
                traj.append(list(obs[0:6]))
                fell |= bool(np.linalg.norm(np.array(obs[0:3]) - prev_pos) > 5); prev_pos = np.array(obs[0:3])
            send(NOOP); obs = recv()
            obs, rest_ticks = wait_rest(obs)
            fell |= bool(np.linalg.norm(np.array(obs[0:3]) - np.array(START)) > 40)
            wall = time.perf_counter() - t0
            end = [round(float(v), 4) for v in obs[0:3]]
            # fixed-delay counters since the last STATS (on-time, late, held, delay, tick)
            stats[0] = '?'; send('STATS')
            for _ in range(5):
                obs = recv(); send(NOOP)
            trial = {'speed': speed, 'start': start, 'end': end, 'ticks_to_rest': rest_ticks,
                     'rest_reached': rest_ticks < REST_CAP, 'fell': fell, 'script_wall_s': round(wall, 2),
                     'traj': np.array(traj)[:, :3].round(4).tolist()}
            self.trials.append(trial)
            trial['stats'] = stats[0]
            self.log(f"   start {start}  ->  rest at {end}  ({'FELL/RESPAWNED, ' if fell else ''}rest after {rest_ticks} free ticks, wall {wall:.1f}s)"
                     f"  replies on-time,late,held,delay,tick = {stats[0]}")
            remaining.pop(0)
        send('SPEED 3')


def summarize(trials):
    print("\nSUMMARY")
    by = {}
    for t in trials:
        by.setdefault(t['speed'], []).append(t)
    means = {}
    for spd, ts in sorted(by.items()):
        ends = np.array([t['end'] for t in ts])
        means[spd] = ends.mean(axis=0)
        spread = max((np.linalg.norm(a - b) for a in ends for b in ends), default=0.0)
        print(f"  {spd:>3}x: rest positions " + '; '.join(str(t['end']) for t in ts) + f"  | within-speed max spread {spread:.4f} u")
    if len(means) == 2:
        (s1, m1), (s2, m2) = sorted(means.items())
        print(f"  between {s1}x and {s2}x means: {np.linalg.norm(m1 - m2):.4f} u")
    ref = next((t for t in trials if t['speed'] == PLAN[0]), None)
    if ref is not None:
        r = np.array(ref['traj'])
        for t in trials[1:]:
            a = np.array(t['traj']); n = min(len(a), len(r))
            dev = np.linalg.norm(a[:n] - r[:n], axis=1)
            print(f"  trajectory vs trial 1 ({ref['speed']}x): trial at {t['speed']}x -> max dev {dev.max():.4f} u at tick {int(dev.argmax())}, mean {dev.mean():.4f} u")


def main():
    probe = Probe()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = int(os.environ.get('PROBE_PORT', '8888'))
    srv.bind(('127.0.0.1', port)); srv.listen(1)
    probe.log(f"physics_identity_probe: waiting for the game on {port} (plan {PLAN})")
    remaining = list(PLAN)
    try:
        while remaining:
            conn, addr = srv.accept(); probe.log(f"game connected {addr}")
            try:
                probe.run(conn, remaining)
            except (ConnectionError, OSError) as e:
                probe.log(f"connection ended: {e} (resuming, remaining {remaining})")
            finally:
                conn.close()
    finally:
        srv.close()
        os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
        with open(os.path.join(HERE, 'logs', f'physics_identity_{port}.json'), 'w') as fh:
            json.dump({'when': datetime.now().isoformat(), 'script': SCRIPT, 'trials': probe.trials, 'log': probe.log_lines}, fh, indent=1)
        summarize(probe.trials)


if __name__ == '__main__':
    main()
