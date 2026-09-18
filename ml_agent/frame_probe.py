"""After which respawns is the control frame wrong?

    python frame_probe.py       (game on FlatIslands_Hunt or the flat map; trainer stopped)

For each trial: teleport to a start, measure the command->acceleration angle for a +x
command (should be ~0 deg), then drive off the east edge; wait for the respawn
(natural OOB respawn, or a forced RESPAWN control on odd trials), settle, and measure
the angle again. Prints the angle before/after each respawn. A ~180 deg angle after a
respawn = the frame flip that made the navigator drive away from its waypoints
(found 2026-09-17: command/acceleration cosine -0.8 in the segments after some falls).
"""
import os
import math
import numpy as np
from nav.env import HuntEnv
from nav.protocol import NOOP_ACTION

RIGHT = (0.0, 0.0, 0.0, 1.0, 0)      # +x in the world frame (fwd, back, left, right, jump)
FWD = (1.0, 0.0, 0.0, 0.0, 0)        # +y


def measure(env, js, ticks=24):
    """Angle (deg) between the commanded world direction and the resulting acceleration."""
    v0 = np.array(env.vel()[:2], dtype=float)
    for _ in range(ticks):
        env.step(js, repeat=1)
    v1 = np.array(env.vel()[:2], dtype=float)
    a = v1 - v0
    cmd = np.array([js[3] - js[2], js[0] - js[1]])
    if np.linalg.norm(a) < 0.2:
        return None
    c = float(np.dot(cmd, a) / (np.linalg.norm(cmd) * np.linalg.norm(a)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def settle(env, max_ticks=120):
    for _ in range(max_ticks):
        v = env.vel(); p = env.pos()
        if abs(float(v[2])) < 0.05 and float(np.hypot(v[0], v[1])) < 0.5 and p[2] < 1.5:
            return True
        env.step(NOOP_ACTION, repeat=1)
    return False


def wait_respawn(env, max_ticks=600):
    for i in range(max_ticks):
        p = env.pos()
        if p[2] > -1.0 and abs(p[0]) < 4 and abs(p[1]) < 4:
            return i
        env.step(NOOP_ACTION, repeat=1)
    return None


def main():
    env = HuntEnv(int(os.environ.get('PROBE_PORT', '8888')), speed=3, log=lambda s: None); env.connect()
    print('map', env.info['mission'], flush=True)
    for trial in range(10):
        forced = trial % 2 == 1
        env.teleport(6.0, 0.0, 0.98, settle_ticks=10); settle(env)
        a0 = measure(env, RIGHT); env.teleport(6.0, 0.0, 0.98, settle_ticks=10); settle(env)
        # drive off the east edge of the centre island
        for _ in range(80):
            env.step(RIGHT, repeat=1)
            if env.pos()[2] < -2.0:
                break
        if forced:
            env.control('RESPAWN')
        n = wait_respawn(env)
        settle(env)
        a1 = measure(env, RIGHT); settle(env)
        a2 = measure(env, FWD)
        print(f"trial {trial}: {'forced' if forced else 'natural'} respawn (after {n} ticks) | angle before {a0 if a0 is None else round(a0)} deg | after: +x cmd {a1 if a1 is None else round(a1)} deg, +y cmd {a2 if a2 is None else round(a2)} deg", flush=True)
        settle(env)
    env.close()


if __name__ == '__main__':
    main()
