"""
Does every scripted action tick reach the physics at every game speed?

    python action_granularity_probe.py      (game running any Hunt map, connects to 8888)

From rest, sends FORWARD for exactly one tick every 20 ticks (PULSES times),
then no keys, and measures the peak speed and the displacement after the
pulses. If the engine ran a wall-clock frame's script updates before that
frame's physics ticks, single-tick actions would be overwritten before physics
saw them at high time scales and the displacement would shrink with speed.
Repeats REPS times per speed (exact start state via a second teleport).
"""
import os
import json
import socket
import numpy as np

SPEEDS = [3, 10]
REPS = 2
PULSES = 6
GAP = 20
NOOP = '0,0,0,0,0,0.000000,0'
FWD = '1,0,0,0,0,0.000000,0'
BACK = '0,1,0,0,0,0.000000,0'
REST_V = 0.005


def main():
    port = int(os.environ.get('PROBE_PORT', '8888'))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', port)); srv.listen(1)
    print(f"action_granularity_probe: waiting on {port}", flush=True)
    conn, _ = srv.accept(); f = conn.makefile('r'); tick = ['']

    def recv():
        while True:
            line = f.readline()
            if not line:
                raise ConnectionError('game disconnected')
            parts = line.strip().split('|')
            if parts[0] == 'STATS':
                continue
            try:
                obs = json.loads(parts[0])
            except ValueError:
                obs = []
            tick[0] = parts[-1].strip() if len(parts) >= 5 and parts[-1].strip().isdigit() else ''
            if len(obs) >= 35:
                return obs
            send(NOOP)

    def send(r):
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

    obs = recv()
    start = (obs[0], obs[1], obs[2] + 0.3)
    send(f'TELEPORT {start[0]} {start[1]} {start[2]} 0 0 0'); obs = recv(); obs = wait_rest(obs)
    rest = tuple(round(float(v), 4) for v in obs[0:3])
    print(f"start pose {rest}", flush=True)
    try:
        for speed in SPEEDS:
            send(f'SPEED {speed}'); obs = recv()
            for _ in range(30):
                send(NOOP); obs = recv()
            for rep in range(REPS):
                send(f'TELEPORT {rest[0]} {rest[1]} {rest[2]} 0 0 0'); obs = recv()
                for _ in range(20):
                    send(NOOP); obs = recv()
                p0 = np.array(obs[0:3]); vmax = 0.0
                for i in range(PULSES * GAP + 60):
                    send(FWD if i % GAP == 0 and i < PULSES * GAP else NOOP)
                    obs = recv(); vmax = max(vmax, float(np.linalg.norm(obs[3:6])))
                disp = float(np.linalg.norm(np.array(obs[0:3]) - p0))
                print(f"  {speed:>3}x rep {rep + 1}: peak speed {vmax:.4f}  displacement after pulses {disp:.4f} u  end {[round(float(v), 4) for v in obs[0:3]]}", flush=True)
                obs = wait_rest(obs)
            # Alternating FWD/BACK every tick: with true interleaving the marble
            # jitters in place; if only the last script update of a wall frame
            # reaches physics, one direction dominates and the marble drifts.
            for rep in range(REPS):
                send(f'TELEPORT {rest[0]} {rest[1]} {rest[2]} 0 0 0'); obs = recv()
                for _ in range(20):
                    send(NOOP); obs = recv()
                p0 = np.array(obs[0:3]); vmax = 0.0
                for i in range(200):
                    send(FWD if i % 2 == 0 else BACK)
                    obs = recv(); vmax = max(vmax, float(np.linalg.norm(obs[3:6])))
                disp = np.array(obs[0:3]) - p0
                print(f"  {speed:>3}x alternating F/B rep {rep + 1}: peak speed {vmax:.3f}  displacement {disp.round(3).tolist()}", flush=True)
                obs = wait_rest(obs)
    finally:
        try:
            send('SPEED 3')
        except OSError:
            pass
        conn.close(); srv.close()


if __name__ == '__main__':
    main()
