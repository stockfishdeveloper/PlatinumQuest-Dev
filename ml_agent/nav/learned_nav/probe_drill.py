"""Stage 0 checks for the P0 drill map (KOTMJUMP_START_HERE.md): run once against a fresh game.

    python -m nav.learned_nav.probe_drill --port 8941
    (then start: marbleblast_mbx.exe -autotrain kotmjump_p0 -aiport 8941)

1. the map: mission name, round length, the single target gem up;
2. spin teleport: the same rolling start with and without the spin words; with spin the marble keeps its speed;
3. pickup + recovery + GEMRESET: teleport onto the gem, expect a pickup, the fall, the respawn, then the gem back;
4. hole lips: teleport onto floor points around the hole (should stay up) and into the hole (should fall),
   against the terrain map.
"""
import argparse
import math
import time

from nav.learned_nav.record import Recorder, TARGET, R, floor_below
from nav.protocol import RAW_POS, RAW_VEL, RAW_SPIN, NOOP_ACTION, format_teleport
from nav.joystick import action_to_joystick


def fmt(v):
    return '(' + ', '.join(f'{float(a):.2f}' for a in v) + ')'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    a = ap.parse_args()
    rec = Recorder(a.port)
    env = rec.env
    rec.ready()
    o = rec.obs()
    print(f'1. mission {env.info.get("mission")}, round {rec.round_s:.0f} s, time left {env.time_left_s():.0f} s, '
          f'pos {fmt(o[RAW_POS])}, gem up {rec.gem_present()}')

    # 2. spin teleport
    for spin in (None, 'rolling'):
        x, y, vx, vy = -30.25, 1.0, 0.0, 8.0
        run = action_to_joystick(0.0, 1.0, 1.0, 0, 0, vx, vy)
        rec.step(run)
        z = floor_below(x, y, 21.0) + R + 0.01
        w = None if spin is None else (-vy / R, vx / R, 0.0)
        rec.control(format_teleport(x, y, z, vx, vy, 0.0, spin=w))
        rows = []
        for i in range(6):
            rec.step(run)
            ob = rec.obs()
            rows.append(f'v {fmt(ob[RAW_VEL])} w {fmt(ob[RAW_SPIN])}')
        print(f'2. spin={spin}: requested v (0, 8, 0), rolling w ({-vy / R:.1f}, 0, 0)')
        for i, r in enumerate(rows):
            print(f'     step {i + 1}: {r}')
        rec.ready()

    # 3. pickup, fall, recovery, GEMRESET
    rec.ready()
    rec.control(format_teleport(TARGET[0], TARGET[1], TARGET[2], 0, 0, 0))
    got = 0.0; fell_at = None
    for i in range(80):
        msg, info = rec.step(NOOP_ACTION)
        got += info['gem_delta']
        if info['fell'] and fell_at is None:
            fell_at = i
            break
    print(f'3. teleport onto the gem: gem_delta total {got}, fell at step {fell_at}, gem up after pickup {rec.gem_present()}')
    t = time.perf_counter()
    rec.ready()
    print(f'   recovered and GEMRESET: pos {fmt(rec.obs()[RAW_POS])}, gem up {rec.gem_present()} '
          f'({time.perf_counter() - t:.1f} s)')

    # 4. lips: expect what the terrain map says
    pts = [(-34.0, 10.05), (-33.0, 10.05), (-26.0, 10.05), (-27.0, 10.05), (-30.25, 6.6), (-30.25, 7.2),
           (-27.5, 12.3), (-30.25, 13.4), (-30.25, 14.1), (-25.0, 9.0), (-23.0, 9.0), (-22.5, 15.3)]
    for x, y in pts:
        expect = floor_below(x, y, 21.0) is not None
        rec.ready()
        fb = floor_below(x, y, 21.0)
        z = (fb if fb is not None else 20.65) + R + 0.01
        rec.control(format_teleport(x, y, z, 0, 0, 0))
        fell = False
        for i in range(25):
            msg, info = rec.step(NOOP_ACTION)
            if info['fell']:
                fell = True
                break
        pz = float(rec.obs()[RAW_POS][2])
        stayed = (not fell) and pz > 20.5
        tag = 'ok' if stayed == expect else 'MISMATCH'
        print(f'4. ({x:.2f}, {y:.2f}) terrain floor {fb}: expect {"floor" if expect else "hole"}, '
              f'got {"floor" if stayed else "fell"} (z {pz:.2f}) {tag}')
    print('done')


if __name__ == '__main__':
    main()
