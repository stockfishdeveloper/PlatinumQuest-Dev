"""Stage 3 check of the engine's contact telemetry (CONTACT control word, Marble::getContactTelemetry).

    python -m nav.learned_nav.probe_contact --port 8972      then: marbleblast_mbx.exe -autotrain kotmjump_p0 -aiport 8972

Prints the 13 telemetry numbers per decision for: resting on the floor, falling from mid-air onto the floor, rolling
off the target hole's south lip, and a jump across it. Expected: on the floor every sub-step has a supporting contact
with normal (0, 0, 1); in the air none; a landing shows a collision and the approach speed.
"""
import argparse
import math

from nav.learned_nav.record import Recorder, R, floor_below, TARGET
from nav.protocol import RAW_POS, RAW_VEL, NOOP_ACTION, CONTACT_FIELDS, format_teleport
from nav.joystick import action_to_joystick


def show(tag, rec, js, n):
    for i in range(n):
        msg, info = rec.step(js)
        e = msg.extra
        p = msg.obs[RAW_POS]; v = msg.obs[RAW_VEL]
        if e is None:
            print(f'{tag} {i}: NO TELEMETRY (old observer.cs.dso or old exe?)')
            return
        d = dict(zip(CONTACT_FIELDS, e))
        print(f'{tag} {i:2d}: z {p[2]:6.3f} vz {v[2]:6.2f} | steps {int(d["sub_steps"])} contact {int(d["contact_steps"])} '
              f'support {int(d["support_steps"])} coll {int(d["collisions"])} n {d["nx"]:+.3f},{d["ny"]:+.3f},{d["nz"]:+.3f} '
              f'approach {d["max_approach"]:.2f} fric {d["friction"]:.2f} rest {d["restitution"]:.2f} last {int(d["last_step_contact"])}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    a = ap.parse_args()
    rec = Recorder(a.port)
    rec.ready()
    rec.control('CONTACT 1')
    x, y = -30.25, 2.0
    z = floor_below(x, y, 21.0) + R + 0.01
    rec.control(format_teleport(x, y, z, 0, 0, 0))
    show('rest', rec, NOOP_ACTION, 5)
    rec.control(format_teleport(x, y, z + 1.5, 0, 0, 0))
    show('drop', rec, NOOP_ACTION, 10)
    run = action_to_joystick(0.0, 1.0, 1.0, 0, 0, 0.0, 8.0)
    rec.step(run)
    rec.control(format_teleport(x, 4.0, z, 0, 8.0, 0, spin=(-8.0 / R, 0, 0)))
    show('rolloff', rec, run, 10)
    rec.ready()
    rec.step(run)
    rec.control(format_teleport(x, 1.0, z, 0, 11.0, 0, spin=(-11.0 / R, 0, 0)))
    show('run', rec, run, 3)
    jump = action_to_joystick(0.0, 1.0, 1.0, 1, 0, 0.0, 11.0)
    show('jump', rec, jump, 1)
    show('fly', rec, run, 16)
    rec.control('CONTACT 0')
    msg, _ = rec.step(NOOP_ACTION)
    print('after CONTACT 0: extra =', msg.extra, '(expected None)')


if __name__ == '__main__':
    main()
