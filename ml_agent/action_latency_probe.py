"""
How many ticks pass between the trainer sending a key and the marble reacting,
at each game speed?

    python action_latency_probe.py          (game running any Hunt map, connects to 8888)

For each speed: teleport to the spawn pose, wait for rest, then send NOOP for a
random number of ticks (so the command lands at every phase of the wall-clock
frame), send FORWARD and count the messages until the marble's speed exceeds a
threshold. Repeats REPS times per speed and prints the latency distribution.
Replies echo the observation's tick, so with the game's fixed action delay K
(mlAgent.cs $MLAgent::ActionDelay) the expected latency is K+1 at every speed;
without the delay it was 2 ticks at 1x, 2-3 at 3x, 3-6 at 10x. Results: logs/action_latency_<port>.json
"""
import os
import time
import json
import random
import socket
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPEEDS = [3, 10]
REPS = 4
NOOP = '0,0,0,0,0,0.000000,0'
FWD = '1,0,0,0,0,0.000000,0'
MOVE_V = 0.05          # speed that counts as "reacted"
REST_V = 0.005
MAX_WAIT = 200


def main():
    port = int(os.environ.get('PROBE_PORT', '8888'))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', port)); srv.listen(1)
    print(f"action_latency_probe: waiting on {port}", flush=True)
    conn, _ = srv.accept(); f = conn.makefile('r'); tick = ['']

    def recv():
        while True:
            line = f.readline()
            if not line:
                raise ConnectionError('game disconnected')
            parts = line.strip().split('|')
            try:
                obs = json.loads(parts[0])
            except ValueError:
                obs = []
            tick[0] = parts[-1].strip() if len(parts) >= 5 and parts[-1].strip().isdigit() else ''
            if len(obs) >= 35:
                return obs
            send(NOOP)

    def send(r):
        # action replies carry the tick they answer (fixed action delay in the game)
        if r[:1].isdigit() and tick[0]:
            r += f',t{tick[0]}'
        conn.sendall((r + '\n').encode())

    def wait_rest(obs):
        still = 0
        for _ in range(1500):
            still = still + 1 if np.linalg.norm(obs[3:6]) < REST_V else 0
            if still >= 30:
                return obs
            send(NOOP); obs = recv()
        return obs

    rng = random.Random(3)
    results = {}
    obs = recv()
    start = (obs[0], obs[1], obs[2] + 0.3)
    print(f"start pose {tuple(round(v, 2) for v in start)}", flush=True)
    try:
        for speed in SPEEDS:
            send(f'SPEED {speed}'); obs = recv()
            for _ in range(30):
                send(NOOP); obs = recv()
            t0 = time.perf_counter()
            for _ in range(60):
                send(NOOP); obs = recv()
            rate = 60 / (time.perf_counter() - t0)       # confirms the speed change took effect
            print(f"  {speed:>3}x: {rate:.0f} ticks/s wall ({rate * 0.016:.1f}x real time)", flush=True)
            lats = []
            for rep in range(REPS):
                send(f'TELEPORT {start[0]} {start[1]} {start[2]} 0 0 0'); obs = recv()
                obs = wait_rest(obs)
                for _ in range(rng.randint(0, 25)):
                    send(NOOP); obs = recv()
                # command sent in reply to message 0; message k is the state k ticks later
                send(FWD); k = 0; lat = None
                while k < MAX_WAIT:
                    obs = recv(); k += 1
                    if np.linalg.norm(obs[3:6]) > MOVE_V:
                        lat = k; break
                    send(FWD)
                send(NOOP)
                lats.append(lat)
            lats_ok = [l for l in lats if l is not None]
            results[speed] = lats
            print(f"  {speed:>3}x: latency ticks {lats}  | mean {np.mean(lats_ok):.2f}, min {min(lats_ok)}, max {max(lats_ok)}", flush=True)
    finally:
        try:
            send('SPEED 3')
        except OSError:
            pass
        conn.close(); srv.close()
        os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
        with open(os.path.join(HERE, 'logs', f'action_latency_{port}.json'), 'w') as fh:
            json.dump({'when': datetime.now().isoformat(), 'results': results}, fh, indent=1)


if __name__ == '__main__':
    main()
