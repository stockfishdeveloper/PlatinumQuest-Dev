"""Does a 64 ms decision give the same thrust as holding the key for four 16 ms ticks?

THE QUESTION (user, 2026-09-21): the agent sends one move per 64 ms simulation step, while a human
holds the key down across every 16 ms tick. If the engine applies the agent's move as a single
impulse per step rather than for the whole 64 ms, the agent would be permanently down on thrust and
no amount of reward tuning would fix it.

WHY IT IS WORTH TESTING. Measured on FlatGemTraining, acceleration while thrusting along travel is
IDENTICAL to the human's below 6 u/s (13.3-14.0 against 13.4-14.1) but falls progressively behind
above it (4.05 against 5.95 at 15-16 u/s). Acceleration is proportional to
(desiredVelocity - currentVelocity), so a faster decay means a LOWER desiredVelocity: the agent's
curve extrapolates to a terminal speed near 17 u/s, the human's near 21.

THE TEST. Park the marble at rest, then hold ONE constant input with no policy in the loop and
record the speed curve until it stops rising. Run it at several fixed step sizes. If a 64 ms step
delivers the same thrust as four 16 ms ticks, the terminal speed is the same at every step size and
the user's hypothesis is refuted. If the 64 ms terminal speed is lower, it is confirmed and the fix
is in the interface, not the reward.

Also runs a single-axis trial, which should terminate at 1/sqrt(2) of the diagonal if the two-key
square is behaving.

    python step_size_probe.py            (game already running with -autotrain <Mission> -aiport N)
    NAV_PORT=8920 python step_size_probe.py
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from nav.env import HuntEnv                       # noqa: E402
from nav.protocol import NOOP_ACTION              # noqa: E402

PORT = int(os.environ.get('NAV_PORT', '8920'))
HOLD_S = 6.0                                      # seconds of simulated time to hold the input
STEPS_MS = [64, 32, 16]                           # step sizes to compare


def settle(env, ticks=40):
    for _ in range(ticks):
        env.step(NOOP_ACTION, repeat=1)


def speed(env):
    o = env.msg.obs
    return math.hypot(float(o[3]), float(o[4]))


def run_trial(env, step_ms, keys, label, start_xyz):
    """Hold `keys` from rest at the map centre and fit acceleration against speed.

    The first version of this probe held the input for 6 s, which ran the marble off the 125 u
    platform: peaks of 40-45 u/s and a diagonal SLOWER than a single axis, i.e. falls and respawns
    in the numbers. Now it teleports to the centre, holds only until the marble leaves the floor or
    the room runs out, and never uses a sample that is not on the floor.

    Acceleration is proportional to (desiredVelocity - v), so fitting accel against v and reading
    off the zero crossing gives the TARGET SPEED for that step size without needing to reach it.
    """
    env.control(f'FIXEDSTEP {step_ms}')
    env.teleport(start_xyz[0], start_xyz[1], start_xyz[2], 0.0, 0.0, 0.0, settle_ticks=4)
    settle(env, int(600 / step_ms) + 4)
    js = (keys[0], keys[1], keys[2], keys[3], 0, 0.0)
    dt = step_ms / 1000.0
    pts = []
    prev = speed(env)
    n = int(3500 / step_ms)                  # 3.5 s of sim time, well inside the platform
    for i in range(n):
        env.step(js, repeat=1)
        o = env.msg.obs
        # on the floor = not falling and not below where we started. The raw observation has no
        # on-floor flag (ObsBuilder derives it), so derive it the same way here.
        on_floor = abs(float(o[5])) < 1.0 and float(o[2]) > start_xyz[2] - 1.0
        v = speed(env)
        if on_floor and v > 0.5:
            pts.append((prev, (v - prev) / dt))
        prev = v
        if not on_floor:
            break
    if len(pts) < 8:
        print('  %-22s step %3d ms: TOO FEW ON-FLOOR SAMPLES (%d)' % (label, step_ms, len(pts)))
        return None
    vs = np.array([p[0] for p in pts]); ac = np.array([p[1] for p in pts])
    keep = (vs > 1.0) & (ac > 0.2)           # the linear part, before it flattens into noise
    if keep.sum() < 6:
        keep = vs > 0.5
    k, c = np.polyfit(vs[keep], ac[keep], 1)  # accel = k*v + c, zero at v = -c/k
    target = (-c / k) if k < -1e-6 else float('nan')
    print('  %-22s step %3d ms: %3d samples, reached %5.2f u/s, fitted TARGET SPEED %6.2f u/s'
          % (label, step_ms, len(pts), vs.max(), target))
    return {'label': label, 'step_ms': step_ms, 'n': len(pts), 'reached': float(vs.max()),
            'target': float(target), 'slope': float(k), 'intercept': float(c)}


def main():
    env = HuntEnv(PORT, speed=1)
    print(f'[probe] listening on {PORT}; start the game with -aiport {PORT}')
    env.connect()
    env.control('LOCKSTEP 1')
    env.control('RENDEREVERY 1')
    print('[probe] connected on %s' % env.info.get('mission', '?'))
    # start from the middle of the platform so a 3.5 s hold cannot reach an edge
    o = env.msg.obs
    start = (float(o[0]), float(o[1]), float(o[2]))
    print('[probe] start pose %.1f %.1f %.1f' % start)
    results = []
    for step_ms in STEPS_MS:
        results.append(run_trial(env, step_ms, (1.0, 0.0, 0.0, 1.0), 'two keys (diagonal)', start))
    for step_ms in STEPS_MS:
        results.append(run_trial(env, step_ms, (1.0, 0.0, 0.0, 0.0), 'one key (single axis)', start))
    results = [r for r in results if r]

    print()
    print('VERDICT')
    diag = {r['step_ms']: r['target'] for r in results if 'two' in r['label']}
    axis = {r['step_ms']: r['target'] for r in results if 'one' in r['label']}
    base = diag.get(16)
    print('  TARGET SPEED (where acceleration reaches zero), two keys held:')
    for ms in STEPS_MS:
        if ms in diag and base and base == base:
            print('    %3d ms step: %6.2f u/s   (%.1f%% of the 16 ms result)' % (ms, diag[ms], 100 * diag[ms] / base))
    print('  diagonal / single axis at each step size (1.414 = the two-key square is intact):')
    for ms in STEPS_MS:
        if ms in diag and ms in axis and axis[ms] > 0:
            print('    %3d ms step: %.3f' % (ms, diag[ms] / axis[ms]))
    print()
    if base and base == base and 64 in diag:
        if abs(diag[64] - base) / base < 0.05:
            print('  => a 64 ms decision delivers the SAME thrust as four 16 ms ticks. The agent is not')
            print('     losing speed to impulse-versus-hold; the hypothesis is refuted.')
        else:
            print('  => the 64 ms step reaches a DIFFERENT target speed (%.1f%% of 16 ms). The decision' % (100*diag[64]/base))
            print('     interval itself is changing the physics, and the fix is in the interface.')
    out = os.path.join(HERE, 'logs', 'nav', 'step_size_probe.json')
    json.dump(results, open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
