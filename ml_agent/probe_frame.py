"""
One-off probe: does a world-frame command move the marble in that world direction?

    python probe_frame.py          (instead of the trainer; then host the map)

Each cycle: wait until the marble is at rest on the floor, command a pure world
direction (+x, +y, -x, -y in turn) for ~1 s through the normal action reply,
measure the direction the marble actually moves in (world frame, from the
observation), then teleport it below the map so it goes out of bounds and
respawns at a random spawn point (each spawn carries a different camera yaw).
Prints, per cycle: spawn position, expected spawn yaw (from the mission), the
commanded direction, the measured direction, and the error. Ctrl+C to stop.
Also written to logs/probe_frame.log.
"""
import os
import sys
import json
import math
import socket
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NOOP = '0,0,0,0,0,0.000000,0'
DIRS = [((1, 0), '+x'), ((0, 1), '+y'), ((-1, 0), '-x'), ((0, -1), '-y')]
PUSH_TICKS = 60
SETTLE_TICKS = 40
MAX_CYCLES = 16

# King of the Marble spawn rows/columns and their SpawnTrigger rotations (deg)
def expected_spawn_yaw(x, y):
    if y > 30: return 180
    if y < 0: return 0
    if x < -40: return 90
    if x > -10: return -90
    return None


def cmd(dx, dy):
    f = max(dy, 0); b = max(-dy, 0); r = max(dx, 0); l = max(-dx, 0)
    return f'{f},{b},{l},{r},0,0.000000,0'


def main():
    logf = open(os.path.join(HERE, 'logs', 'probe_frame.log'), 'a')
    def log(s):
        print(s); logf.write(f"[{datetime.now():%H:%M:%S}] {s}\n"); logf.flush()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', 8888)); srv.listen(1)
    log("probe_frame: waiting for the game (host the map and start the round)")
    conn, addr = srv.accept(); log(f"game connected {addr}")
    buf = ''
    state = 'settle'; ticks = 0; cycle = 0; spawn = None; vel_hist = []
    results = []
    try:
        while cycle < MAX_CYCLES:
            data = conn.recv(8192).decode('utf-8')
            if not data:
                log("game disconnected"); break
            buf += data
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                if not line.strip():
                    continue
                try:
                    obs = json.loads(line.split('|')[0])
                except ValueError:
                    obs = []
                if len(obs) < 6:
                    conn.sendall((NOOP + '\n').encode()); continue
                pos = np.array(obs[0:3]); vel = np.array(obs[3:6]); speed = float(np.linalg.norm(vel[:2]))
                reply = NOOP
                if state == 'settle':
                    # at rest on the floor for SETTLE_TICKS ticks -> start a push
                    ticks = ticks + 1 if (speed < 0.3 and abs(vel[2]) < 0.5 and pos[2] > 18) else 0
                    if ticks >= SETTLE_TICKS:
                        spawn = pos.copy(); state = 'push'; ticks = 0; vel_hist = []
                        d, name = DIRS[cycle % 4]
                        log(f"cycle {cycle + 1}: at rest at {np.round(spawn, 1)} (expected spawn yaw {expected_spawn_yaw(spawn[0], spawn[1])} deg); commanding world {name}")
                elif state == 'push':
                    d, name = DIRS[cycle % 4]
                    reply = cmd(*d); ticks += 1; vel_hist.append(vel[:2].copy())
                    if ticks >= PUSH_TICKS:
                        v = np.mean(vel_hist[-10:], axis=0)
                        ang_c = math.degrees(math.atan2(d[1], d[0])); ang_m = math.degrees(math.atan2(v[1], v[0]))
                        err = (ang_m - ang_c + 180) % 360 - 180
                        moved = pos - spawn
                        log(f"   measured velocity {np.round(v, 2)} (|v|={np.linalg.norm(v):.1f}), moved {np.round(moved[:2], 1)}: "
                            f"commanded {ang_c:.0f} deg, measured {ang_m:.0f} deg, ERROR {err:+.0f} deg")
                        results.append((expected_spawn_yaw(spawn[0], spawn[1]), name, err))
                        state = 'kick'; ticks = 0
                elif state == 'kick':
                    # send the marble out of bounds -> respawn at a random spawn point
                    reply = f'TELEPORT {pos[0]:.2f} {pos[1]:.2f} 0.0 0 0 0'
                    state = 'respawn'; ticks = 0
                elif state == 'respawn':
                    ticks += 1
                    if pos[2] > 18 and ticks > 30:
                        state = 'settle'; ticks = 0; cycle += 1
                    elif ticks > 600:
                        log("   no respawn seen in 10 s; retrying the kick"); state = 'kick'
                conn.sendall((reply + '\n').encode())
    except KeyboardInterrupt:
        pass
    finally:
        conn.close(); srv.close()
        if results:
            log("SUMMARY (expected spawn yaw, command, error):")
            for r in results:
                log(f"   yaw {r[0]!s:>5}  cmd {r[1]}  error {r[2]:+.0f} deg")
        logf.close()


if __name__ == '__main__':
    main()
