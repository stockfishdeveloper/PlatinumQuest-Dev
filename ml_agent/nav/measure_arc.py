"""Per-tick jump trajectories from the game, to verify nav/physics.predict_landing to the centimetre
(2026-09-25, HANDOFF 28.40). Trainer STOPPED, one game on port 8920 launched after this binds:

    set NAV_OBS_MS=16 ; python -m nav.measure_arc
    marbleblast_mbx.exe -autotrain KingOfTheMarble_Hunt -aiport 8920

Cases: on the flat east walkway (x = -12, y 1..30, floor 20.65), at 6 / 8 / 10 / 12 u/s: forward held,
hands off, left held; plus a drop (no jump) off the north edge of the centre block at 8 u/s.
Writes logs/physics/arcs_<map>.csv with every 16 ms sample: case, speed, t, x, y, z, vx, vy, vz.
"""
import math
import os

os.environ.setdefault('NAV_OBS_MS', '16')
os.environ.setdefault('NAV_RENDER_EVERY', '100')
from nav.env import HuntEnv, OBS_MS                               # noqa: E402
from nav.joystick import action_to_joystick                       # noqa: E402
from nav.protocol import NOOP_ACTION                              # noqa: E402

MAP = os.environ.get('NAV_MAP', 'KingOfTheMarble_Hunt')
PORT = int(os.environ.get('NAV_PORT', '8920'))
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICK = OBS_MS / 1000.0
FLAT = {'start': (-12.0, 1.0), 'z': 20.65, 'heading': (0.0, 1.0)}
DROP = {'start': (-26.4, 22.0), 'z': 20.65, 'heading': (0.0, 1.0)}     # north edge of the block at y ~ 24.5


def settle(env, hx, hy, speed, z_floor):
    still = 0
    for k in range(int(1.5 / TICK)):
        o = env.msg.obs
        still = still + 1 if (abs(float(o[5])) < 0.05 and k >= 8) else 0
        if still >= 3:
            return True
        env.step(action_to_joystick(hx, hy, 1.0 if speed > 0 else 0.0, 0, 0, float(o[3]), float(o[4])), repeat=1)
    return False


def run_case(env, case, speed, mode, z_floor, out):
    hx, hy = case['heading']
    env.teleport(case['start'][0], case['start'][1], z_floor + 0.6, hx * speed, hy * speed, 0.0, settle_ticks=1)
    if not settle(env, hx, hy, speed, z_floor):
        print(f'  {mode} v{speed}: no touchdown', flush=True); return
    jump = mode != 'drop'
    for k in range(int(2.5 / TICK)):
        o = env.msg.obs
        if mode == 'hold' or mode == 'drop':
            js = action_to_joystick(hx, hy, 1.0, 1 if (jump and k < 2) else 0, 0, float(o[3]), float(o[4]))
        elif mode == 'free':
            js = action_to_joystick(hx, hy, 0.0, 1 if k < 2 else 0, 0, float(o[3]), float(o[4]))
        else:   # 'left': full stick to the left of the heading
            js = action_to_joystick(-hy, hx, 1.0, 1 if k < 2 else 0, 0, float(o[3]), float(o[4]))
        msg, info = env.step(js, repeat=1)
        o = msg.obs
        out.write(f'{mode},{speed},{(k + 1) * TICK:.3f},{o[0]:.4f},{o[1]:.4f},{o[2]:.4f},{o[3]:.4f},{o[4]:.4f},{o[5]:.4f}\n')
        if info['fell'] or info['round_ended']:
            break
    print(f'  {mode} v{speed}: recorded', flush=True)


def main():
    os.makedirs(os.path.join(HERE, 'logs', 'physics'), exist_ok=True)
    path = os.path.join(HERE, 'logs', 'physics', f'arcs_{MAP}.csv')
    env = HuntEnv(port=PORT, speed=1, action_repeat=1)
    print(f'waiting for the game on port {PORT}', flush=True)
    env.connect(); env.request_info(); env.set_speed(1)
    for _ in range(5):
        env.step(NOOP_ACTION, repeat=1)
    with open(path, 'w') as out:
        out.write('mode,speed,t,x,y,z,vx,vy,vz\n')
        for speed in (6.0, 8.0, 10.0, 12.0):
            for mode in ('hold', 'free', 'left'):
                run_case(env, FLAT, speed, mode, FLAT['z'], out)
        run_case(env, DROP, 8.0, 'drop', DROP['z'], out)
        run_case(env, DROP, 8.0, 'hold', DROP['z'], out)
    print('wrote', path, flush=True)
    env.close()


if __name__ == '__main__':
    main()
