"""Measure the marble's JUMP ENVELOPE with the game as the oracle (2026-09-23, PHYSICS_SKILLS_DESIGN 3).

For each approach speed: TELEPORT onto a long flat walkway with that velocity, let the marble touch
down, press jump for two 16 ms ticks while holding forward, then record every tick until the marble is
back at floor height. Outputs per speed: speed at takeoff, apex height, flight time, and RANGE (u of
horizontal travel between takeoff and touching the same-height floor again). Two variants: holding
forward through the flight (airAcceleration acts) and hands off.

Writes logs/physics/jump_envelope_<map>.csv (raw per-trial rows) and physics/jump_envelope.json (the
fitted table that nav/physics.py loads). Run with the trainer STOPPED:

    set NAV_OBS_MS=16 ; python -m nav.measure_jump            (then launch the game on port 8920:
    marbleblast_mbx.exe -autotrain KingOfTheMarble_Hunt -aiport 8920)
"""
import json
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault('NAV_OBS_MS', '16')
os.environ.setdefault('NAV_RENDER_EVERY', '100')
from nav.env import HuntEnv, OBS_MS                               # noqa: E402
from nav.joystick import action_to_joystick                       # noqa: E402
from nav.protocol import NOOP_ACTION                              # noqa: E402
from nav.terrain import TerrainGrid                               # noqa: E402
from terrain_obs import TerrainMap                                # noqa: E402

MAP = os.environ.get('NAV_MAP', 'KingOfTheMarble_Hunt')
PORT = int(os.environ.get('NAV_PORT', '8920'))
# a straight FLAT run (the north walkway at y = 28 is a ramp): KOTM's east walkway x = -12, y 0..30, z 20.65
START = (-12.0, 1.0)
HEADING = (0.0, 1.0)
SPEEDS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
TRIALS = 2
TICK_S = OBS_MS / 1000.0
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_trial(env, z_floor, speed, hold_forward):
    hx, hy = HEADING
    env.teleport(START[0], START[1], z_floor + 0.6, hx * speed, hy * speed, 0.0, settle_ticks=1)
    # touch down: the marble is dropped 0.6 u above the floor with the requested velocity; wait until
    # it has stopped bouncing (|vz| < 0.05 for 3 consecutive ticks, at least 8 ticks after the drop)
    still = 0; z_hist = []
    for k in range(int(1.5 / TICK_S)):
        o = env.msg.obs
        z_hist.append(float(o[2]))
        still = still + 1 if (abs(float(o[5])) < 0.05 and k >= 8) else 0
        if still >= 3:
            break
        js = action_to_joystick(hx, hy, 1.0 if speed > 0 else 0.0, 0, 0, float(o[3]), float(o[4]))
        env.step(js, repeat=1)
    o = env.msg.obs
    x0, y0, z0 = float(o[0]), float(o[1]), float(o[2])
    v0 = math.hypot(float(o[3]), float(o[4]))
    if still < 3 or abs(z0 - z_floor) > 1.0:
        return None
    # jump: trigger held for 2 ticks, forward held (or not) through the flight
    rows = []
    apex = z0; t = 0.0; landed = False; airborne = False
    for k in range(int(3.0 / TICK_S)):
        o = env.msg.obs
        thr = 1.0 if hold_forward else 0.0
        js = action_to_joystick(hx, hy, thr, 1 if k < 2 else 0, 0, float(o[3]), float(o[4]))
        msg, info = env.step(js, repeat=1)
        o = msg.obs; t += TICK_S
        z = float(o[2]); apex = max(apex, z)
        rows.append((t, float(o[0]), float(o[1]), z, float(o[3]), float(o[4]), float(o[5])))
        if z > z0 + 0.15:
            airborne = True
        if airborne and z <= z0 + 0.08 and float(o[5]) <= 0.0:
            landed = True
            break
        if info['fell'] or info['round_ended']:
            break
    if not rows:
        return None
    xl, yl = rows[-1][1], rows[-1][2]
    rng = (xl - x0) * hx + (yl - y0) * hy
    return {'speed_cmd': speed, 'speed_takeoff': round(v0, 2), 'hold_forward': int(hold_forward), 'apex': round(apex - z0, 3),
            'flight_s': round(t, 3), 'range': round(rng, 2), 'landed': int(landed), 'airborne': int(airborne), 'z_rest': round(z0, 3)}


def main():
    t = TerrainGrid(TerrainMap.resolve(MAP))
    j, i = t.cell_of(*START)
    z_floor = float(t.walk_top[j, i])
    print(f'{MAP}: start {START} floor z {z_floor:.2f}; step {OBS_MS} ms; waiting for the game on port {PORT}', flush=True)
    env = HuntEnv(port=PORT, speed=1, action_repeat=1)
    env.connect(); env.request_info(); env.set_speed(1)
    for _ in range(5):
        env.step(NOOP_ACTION, repeat=1)
    results = []
    for hold in (True, False):
        for s in SPEEDS:
            for n in range(TRIALS):
                r = run_trial(env, z_floor, s, hold)
                if r is None:
                    print(f'  speed {s} hold {int(hold)}: no touchdown, skipped', flush=True)
                    continue
                results.append(r)
                print(f'  speed {s:4.1f} (takeoff {r["speed_takeoff"]:5.2f}) hold {int(hold)}: apex {r["apex"]:.2f} u, flight {r["flight_s"]:.2f} s, range {r["range"]:.2f} u, landed {r["landed"]}', flush=True)
    os.makedirs(os.path.join(HERE, 'logs', 'physics'), exist_ok=True)
    os.makedirs(os.path.join(HERE, 'physics'), exist_ok=True)
    csv = os.path.join(HERE, 'logs', 'physics', f'jump_envelope_{MAP}.csv')
    with open(csv, 'w') as f:
        f.write('speed_cmd,speed_takeoff,hold_forward,apex,flight_s,range,landed,airborne\n')
        for r in results:
            f.write(','.join(str(r[k]) for k in ('speed_cmd', 'speed_takeoff', 'hold_forward', 'apex', 'flight_s', 'range', 'landed', 'airborne')) + '\n')
    # table: per hold variant, mean over trials of (takeoff speed, range, apex, flight) by commanded speed
    table = {'map': MAP, 'obs_ms': OBS_MS, 'hold': [], 'nohold': []}
    for hold, key in ((1, 'hold'), (0, 'nohold')):
        for s in SPEEDS:
            rs = [r for r in results if r['hold_forward'] == hold and r['speed_cmd'] == s and r['landed']]
            if rs:
                table[key].append({'speed': round(float(np.mean([r['speed_takeoff'] for r in rs])), 2),
                                   'range': round(float(np.mean([r['range'] for r in rs])), 2),
                                   'apex': round(float(np.mean([r['apex'] for r in rs])), 3),
                                   'flight_s': round(float(np.mean([r['flight_s'] for r in rs])), 3)})
    out = os.path.join(HERE, 'physics', 'jump_envelope.json')
    json.dump(table, open(out, 'w'), indent=1)
    print('wrote', csv, 'and', out, flush=True)
    env.close()


if __name__ == '__main__':
    main()
