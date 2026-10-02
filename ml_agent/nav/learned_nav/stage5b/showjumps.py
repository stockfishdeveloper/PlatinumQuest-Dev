"""Show the planner's KOTM hole crossings at 1x (viewing only).

    python nav/learned_nav/stage5b/showjumps.py --port 9611 --map kotmjump_p0 --loops 3
    (then: marbleblast_mbx.exe -autotrain kotmjump_p0 -aiport 9611)

Phase 1 (fast, lockstep): from pickup-like states at each of the four centre gems, the planner (shortcut settings:
time weight, straight-line jump proposals, after-landing steering, measured jump takeoff) flies across the big hole to
the corner gem spot, and every reply it sends is recorded. Phase 2 (1x, every frame drawn): each crossing that took the
corner spot is replayed from the same teleported start with the same replies, paced to 64 ms a decision; the game is
deterministic, so this is the planner's own run at real speed. kotmjump_p0 has KOTM's geometry (no gem at the corners:
the target is where KOTM's corner gem spawns).
"""
import argparse
import math
import os
import sys
import time

os.environ.setdefault('NAV_RENDER_EVERY', '1')
ML = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ML)
import numpy as np                                                               # noqa: E402
from nav.learned_nav.geometry import Geometry                                    # noqa: E402
from nav.learned_nav.session import Session, R, rolling_spin                     # noqa: E402
from nav.learned_nav import planner as PL, drills as DR                          # noqa: E402
from nav.protocol import RAW_POS, RAW_VEL, RAW_SPIN                              # noqa: E402
from nav.joystick import action_to_joystick                                      # noqa: E402

CENTRE = [((-27.2, 17.0), (-37.2, 27.0, 20.7), 135.0), ((-23.2, 17.0), (-13.2, 27.0, 20.7), 45.0),
          ((-27.2, 13.0), (-37.2, 3.0, 20.7), -135.0), ((-23.2, 13.0), (-13.2, 3.0, 20.7), -45.0)]
STARTS = ((7.5, 0.0), (9.0, 15.0), (8.0, -15.0))      # speed, heading offset from the diagonal (deg)


def log(m):
    print(time.strftime('[%H:%M:%S] ') + m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--map', default='kotmjump_p0')
    ap.add_argument('--loops', type=int, default=3)
    a = ap.parse_args()
    g = Geometry(a.map)
    pl = PL.Planner(g, (0, 0, 0), seeds=None)
    s = Session(a.port, a.map, g, log=log)
    shows = []
    log('phase 1: planning the crossings (fast)')
    for (cx, cy), gem, diag in CENTRE:
        gem = np.asarray(gem); fr = DR.floor_ref_of(g, gem)
        z0 = g.floor_below(cx, cy, 25.0)
        for sp, off in STARTS:
            h = math.radians(diag + off)
            pos = (cx, cy, z0 + R + 0.01); vel = (sp * math.cos(h), sp * math.sin(h), 0.0)
            spin = rolling_spin(vel, (0, 0, 1))
            push = tuple(action_to_joystick(math.cos(h), math.sin(h), 1.0, 0, 0, 0.0, 0.0))
            o = s.place(pos, vel, spin, hold=push)
            if o is None:
                continue
            start = np.asarray(o[RAW_POS], float).copy()
            pl.set_target(gem, None); pl.floor_ref = fr; pl.n_broad = PL.N_BROAD
            pl.time_w = 0.015; pl.approach_line = True; pl.land_steer = True
            Uc, Jc = PL.reply_vector(push); Up, Jp = Uc.copy(), Jc
            replies, t_pick, fell, after = [], None, False, 0
            for d in range(80):
                ob = s.obs(); p = np.asarray(ob[RAW_POS], float); v = np.asarray(ob[RAW_VEL], float); w = np.asarray(ob[RAW_SPIN], float)
                tel = s.env.msg.extra
                lv = g.floor_below(p[0], p[1], p[2]); gap = (p[2] - R - lv) if lv is not None else 9.0
                airborne = not (tel is not None and tel[2] > 0) and gap > 0.5
                if t_pick is None and np.linalg.norm(p - gem) < 0.9:
                    t_pick = d
                if t_pick is not None:
                    after += 1
                    if after > 16:
                        break
                js, info = pl.plan(p, v, w, tel, Uc, Jc, Up, Jp, airborne, t_pick is not None, fr)
                js = tuple(float(q) for q in js[:4]) + (int(js[4]), float(js[5]) if len(js) > 5 else 0.0)
                replies.append(js)
                _, si = s.step(js)
                Up, Jp = Uc, Jc; Uc, Jc = PL.reply_vector(js)
                if si['fell']:
                    fell = True
                    break
            ok = t_pick is not None and not fell
            log('  centre (%.1f, %.1f) -> corner (%.1f, %.1f), %.1f u/s: %s%s' % (
                cx, cy, gem[0], gem[1], sp, 'took the corner spot' if ok else 'missed', ' in %.2f s' % (t_pick * 0.064) if ok else ''))
            if ok:
                shows.append({'pos': pos, 'vel': vel, 'spin': spin, 'push': push, 'replies': replies, 'start': start,
                              'label': 'centre (%.1f, %.1f) -> corner (%.1f, %.1f) at %.1f u/s, gem at %.2f s' % (
                                  cx, cy, gem[0], gem[1], sp, t_pick * 0.064)})
            if fell:
                s.recover()
    log('phase 2: %d crossings at 1x' % len(shows))
    s.env.set_speed(1); s.contact_on()
    stop = tuple(action_to_joystick(0.0, 0.0, 1.0, 0, 1, 0.0, 0.0))
    for loop in range(a.loops):
        for k, sh in enumerate(shows):
            o = s.place(sh['pos'], sh['vel'], sh['spin'], hold=sh['push'])
            if o is None:
                continue
            dev = float(np.linalg.norm(np.asarray(o[RAW_POS], float) - sh['start']))
            log('  [%d/%d] %s%s' % (k + 1, len(shows), sh['label'], '' if dev < 1e-3 else ' (start differs by %.4f u)' % dev))
            t = time.perf_counter()
            for js in sh['replies'] + [stop] * 20:
                t = max(t + 0.064, time.perf_counter())
                time.sleep(max(0.0, t - time.perf_counter()))
                v = s.obs()[RAW_VEL]
                if js is stop:
                    js = tuple(action_to_joystick(0.0, 0.0, 1.0, 0, 1, float(v[0]), float(v[1])))
                _, si = s.step(js)
                if si['fell']:
                    log('    fell in the replay')
                    s.recover()
                    break
            time.sleep(0.8)
    log('done')


if __name__ == '__main__':
    main()
