"""
One-off check: replay a recorded human demo through the game.

    python replay_demo.py                       # newest file in demos/
    python replay_demo.py demos/demo_XXXX.npz

Run this instead of the trainer (restart the game first so it picks up the
updated mlAgent.cs), then host the map and start the round. At GO this script:
  1. sets the game speed to 1x                     ("SPEED 1" control reply)
  2. teleports the marble to the demo's first recorded position and velocity
                                                    ("TELEPORT x y z vx vy vz")
  3. feeds the recorded actions back, one per 16 ms tick, converted with the
     exact joystick mapping the trainer uses, camera locked at yaw 0 like
     training (so joystick right/forward = world +x/+y, the recorded frame)
  4. prints how far the live marble is from the recorded position every second

If the recorded actions and frame are right, the marble retraces the route you
drove for a while before the physics drifts apart (small differences compound;
gems and the respawn points are random too). A wrong frame convention shows up
immediately as the marble heading off in a different direction. The replay
restarts from the top every time a new round begins.
"""
import os
import sys
import glob
import json
import socket

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_ppo import Actor

TICK = 0.016
NOOP = '0,0,0,0,0,0.000000,0'


class Replay:
    def __init__(self, path):
        a = np.load(path, allow_pickle=False)
        self.act, raw, self.use_pow, game = a['action'], a['obs_raw'], a['use_pow'], a['game']
        # replay the first recorded round only
        n = int(np.searchsorted(game, 1)) if game.max() > 0 else len(self.act)
        self.n = min(n, len(self.act))
        self.pos = raw[:, 0:3]
        self.vel = raw[:, 3:6]
        self.p0, self.v0 = raw[0, 0:3], raw[0, 3:6]
        # Respawns (out-of-bounds) move the marble to a random spawn point, so the
        # live game would desync there. Wherever the recorded position jumps
        # between ticks, teleport the live marble to the recorded post-jump state
        # instead of sending that tick's action. Key = action index k (tick i),
        # applied so the jump shows up at demo tick i+1 like in the recording.
        jumps = np.linalg.norm(np.diff(self.pos[:self.n], axis=0), axis=1)
        self.resync = {int(i): (self.pos[i + 1], self.vel[i + 1]) for i in np.where(jumps > 2.0)[0]}
        self.reset()

    @staticmethod
    def teleport(p, v):
        return f'TELEPORT {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}'

    def reset(self):
        self.m = 0           # message index within the current round
        self.drift = []      # (demo tick, distance)
        self.finished = False

    def reply(self, obs):
        """Return the reply for message m (observed at tick m; applied at tick m+1)."""
        m = self.m
        self.m += 1
        if m == 0:
            return 'SPEED 1'
        if m == 1:
            return self.teleport(self.p0, self.v0)
        # obs at message m corresponds to demo tick kk (teleport took effect at m == 3)
        kk = m - 3
        if 0 <= kk < self.n:
            d = float(np.linalg.norm(np.array(obs[0:3]) - self.pos[kk]))
            self.drift.append((kk, d))
            if kk % 62 == 0:
                ak = self.act[kk]
                print(f"  t={kk * TICK:5.1f}s  live {np.round(obs[0:3], 1)}  demo {np.round(self.pos[kk], 1)}"
                      f"  drift {d:5.2f} u  | demo action dx={ak[0]:+.2f} dy={ak[1]:+.2f} thr={ak[2]:.0f} jump={int(ak[3])}")
        elif kk == self.n and not self.finished:
            self.finished = True
            print("Demo finished (marble released).")
            summarise(self.drift)
        k = m - 2            # demo tick whose action is applied next
        if k in self.resync:
            p, v = self.resync[k]
            print(f"  t={k * TICK:5.1f}s  respawn in the recording: teleporting to {np.round(p, 1)}")
            return self.teleport(p, v)
        if k < self.n:
            dx, dy, thr, jump, _brake = self.act[k]
            f, b, l, r, j = Actor.action_to_joystick(float(dx), float(dy), float(thr), int(jump), 0)
            return f'{f},{b},{l},{r},{j},0.000000,{int(self.use_pow[k])}'
        return NOOP


def summarise(drift):
    if not drift:
        return
    d = np.array([x[1] for x in drift]); t = np.array([x[0] for x in drift]) * TICK
    print("Drift between live marble and recorded path:")
    for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 60), (60, 600)):
        mask = (t >= lo) & (t < hi)
        if mask.any():
            print(f"  {lo:>3}-{hi:<3} s: mean {d[mask].mean():5.2f} u  max {d[mask].max():5.2f} u")


def serve(rp, host='127.0.0.1', port=8888):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.settimeout(1.0)
    srv.bind((host, port)); srv.listen(1)
    try:
        while True:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            print(f"Game connected from {addr}; replay starts at GO.")
            rp.reset()
            buf = ''
            try:
                while True:
                    data = conn.recv(8192).decode('utf-8')
                    if not data:
                        print("Game disconnected."); break
                    buf += data
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('|')
                        try:
                            obs = json.loads(parts[0])
                            done = len(parts) > 3 and parts[3].strip() == '1'
                        except ValueError:
                            obs, done = [], False
                        if len(obs) < 6 or done:
                            reply = NOOP
                            if rp.m > 0:
                                print(f"Round ended after {rp.m} ticks; the replay restarts at the next GO.")
                                if not rp.finished:
                                    summarise(rp.drift)
                                rp.reset()
                        else:
                            reply = rp.reply(obs)
                        conn.sendall((reply + '\n').encode())
            except (ConnectionResetError, ConnectionAbortedError, OSError) as e:
                print(f"Connection lost ({e}).")
            finally:
                conn.close()
                if rp.m > 0 and not rp.finished:
                    summarise(rp.drift)
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        files = sorted(glob.glob(os.path.join(HERE, 'demos', 'demo_*.npz')))
        if not files:
            print("No demos found in demos/."); return
        path = files[-1]
    rp = Replay(path)
    print(f"Replaying {os.path.basename(path)}: {rp.n} ticks ({rp.n * TICK:.1f} s) of the first recorded round")
    print(f"Start position {np.round(rp.p0, 2)} velocity {np.round(rp.v0, 2)}; "
          f"{len(rp.resync)} respawn(s) in the recording will be re-synced by teleport")
    print("Host the map and start the round. Ctrl+C to stop.")
    port = int(os.environ.get('REPLAY_PORT', '8888'))
    serve(rp, port=port)


if __name__ == '__main__':
    main()
