"""Stage 6b: how fast can the marble physically turn? An engine measurement over the input space (no human data).

    python -m nav.learned_nav.turndrill --port 9651 --map kotmjump_p0          (then the game on that port)
    python -m nav.learned_nav.turndrill --analyze [--model]                    (offline: per program, per target turn)

Start: the most open floor spot of the map (largest radius of same-level floor), the marble rolling at SPEED0 u/s
along +x with rolling spin. Programs (each held N_DEC decisions, one input per decision, as the planner sends them):
* steer: full throttle in the direction at OFFSET degrees from the initial velocity (30..180), held;
* brake k, then steer at +90 (k = 2, 4, 6 decisions of brake);
* oversteer: full throttle at +150 for k decisions (2, 4, 6), then at +90;
* jump then steer at +90 (a jump at decision 0, air input at +90).
Recorded per decision: position, velocity heading (deg, relative to the initial one), speed. The analysis reports,
for target turns of 60 / 90 / 120 degrees, the first decision at which the heading has turned that far, the speed
then, and the ground covered: the physical cost of a turn, to compare with the crossing drill's fitted turn cost
(planner 1.47 s per (1 - cos), human 0.33) and to check the step model's turn prediction (--model).
Results: logs/learned_nav/turn/turndrill_<map>.jsonl.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav.geometry import Geometry                                     # noqa: E402
from nav.learned_nav.session import Session, R, rolling_spin                     # noqa: E402
from nav.protocol import RAW_POS, RAW_VEL, RAW_SPIN                              # noqa: E402
from nav.joystick import action_to_joystick                                      # noqa: E402

OUT_DIR = os.path.join(HERE, 'logs', 'learned_nav', 'turn')
SPEED0 = 8.0
N_DEC = 24                   # 1.5 s
REPEATS = 2


def programs():
    """name -> list of (dir_angle_rel_deg or None for brake, jump) per decision; None angle + brake."""
    out = {}
    for off in (30, 60, 90, 120, 150, 180):
        out[f'steer{off}'] = [(off, 0, 0)] * N_DEC
    for k in (2, 4, 6):
        out[f'brake{k}_steer90'] = [(None, 1, 0)] * k + [(90, 0, 0)] * (N_DEC - k)
    for k in (2, 4, 6):
        out[f'over150x{k}_steer90'] = [(150, 0, 0)] * k + [(90, 0, 0)] * (N_DEC - k)
    for k in (2, 4):
        out[f'over180x{k}_steer90'] = [(180, 0, 0)] * k + [(90, 0, 0)] * (N_DEC - k)
    out['jump_steer90'] = [(90, 0, 1)] + [(90, 0, 0)] * (N_DEC - 1)
    out['brake2_jump_steer90'] = [(None, 1, 0)] * 2 + [(90, 0, 1)] + [(90, 0, 0)] * (N_DEC - 3)
    return out


def open_spot(g):
    """Floor point with the largest radius of same-level floor around it (raster search)."""
    best = (0.0, None, None)
    xs = np.arange(g.xs.min() + 2, g.xs.max() - 2, 1.0); ys = np.arange(g.ys.min() + 2, g.ys.max() - 2, 1.0)
    ang = np.linspace(-math.pi, math.pi, 24, endpoint=False)
    for x in xs:
        for y in ys:
            z0 = g.floor_below(x, y, 30.0)
            if z0 is None:
                continue
            r = 0.5
            while r < 12.0:
                ok = True
                for a in ang:
                    z = g.floor_below(x + r * math.cos(a), y + r * math.sin(a), z0 + 0.5)
                    if z is None or abs(z - z0) > 0.3:
                        ok = False; break
                if not ok:
                    break
                r += 0.5
            if r > best[0]:
                best = (r, (float(x), float(y)), z0)
    return best


def run(a):
    g = Geometry(a.map)
    r_open, (x0, y0), z0 = open_spot(g)
    log = lambda m: print(m, flush=True)
    log(f'open spot ({x0:.1f}, {y0:.1f}) floor {z0:.2f}, radius {r_open:.1f} u')
    s = Session(a.port, a.map, g, log=log)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = open(os.path.join(OUT_DIR, f'turndrill_{a.map}.jsonl'), 'a')
    progs = programs()
    for name, prog in progs.items():
        for rep in range(REPEATS):
            # start so that the turn (to the left, +y) stays inside the open disc: begin at the -x side, heading +x
            pos = (x0 - 0.5 * r_open, y0 - 0.3 * r_open, z0 + R + 0.01); vel = (SPEED0, 0.0, 0.0)
            push = tuple(action_to_joystick(1.0, 0.0, 1.0, 0, 0, 0.0, 0.0))
            o = s.place(pos, vel, rolling_spin(vel, (0, 0, 1)), hold=push)
            if o is None:
                log(f'{name}: teleport refused'); continue
            rec = []; fell = False
            for d in range(N_DEC + 8):
                ob = s.obs(); p = np.asarray(ob[RAW_POS], float); v = np.asarray(ob[RAW_VEL], float)
                spd = math.hypot(v[0], v[1]); hd = math.degrees(math.atan2(v[1], v[0])) if spd > 0.3 else 0.0
                rec.append([round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3), round(spd, 3), round(hd, 1)])
                ang, brake, jump = prog[d] if d < len(prog) else (90, 0, 0)
                if brake:
                    js = action_to_joystick(0.0, 0.0, 1.0, jump, 1, v[0], v[1])
                else:
                    th = math.radians(ang)
                    js = action_to_joystick(math.cos(th), math.sin(th), 1.0, jump, 0, v[0], v[1])
                msg, sinfo = s.step(tuple(float(q) for q in js[:4]) + (int(js[4]), float(js[5]) if len(js) > 5 else 0.0))
                if sinfo['fell']:
                    fell = True; break
            row = {'program': name, 'rep': rep, 'fell': fell, 'rec': rec, 'start': [pos[0], pos[1], pos[2]], 'speed0': SPEED0}
            out.write(json.dumps(row) + '\n'); out.flush()
            hd = [q[4] for q in rec]
            log('%-22s rep %d: fell %d, heading after 8/16/24 decisions %5.0f %5.0f %5.0f deg, speed %4.1f %4.1f %4.1f' % (
                name, rep, fell, hd[min(8, len(hd) - 1)], hd[min(16, len(hd) - 1)], hd[min(24, len(hd) - 1)],
                rec[min(8, len(rec) - 1)][3], rec[min(16, len(rec) - 1)][3], rec[min(24, len(rec) - 1)][3]))
            if fell:
                s.recover()
    log('done')


def analyze(a):
    rows = [json.loads(l) for l in open(os.path.join(OUT_DIR, f'turndrill_{a.map}.jsonl'))]
    print('%-22s %s' % ('program', '  '.join('turn %3d: dec  speed  dist' % t for t in (60, 90, 120))))
    seen = {}
    for r in rows:
        seen.setdefault(r['program'], []).append(r)
    for name, rs in seen.items():
        cells = []
        for T in (60, 90, 120):
            vals = []
            for r in rs:
                rec = r['rec']
                hit = next((d for d, q in enumerate(rec) if q[4] >= T - 5), None)
                if hit is not None:
                    dist = math.hypot(rec[hit][0] - rec[0][0], rec[hit][1] - rec[0][1])
                    vals.append((hit, rec[hit][3], dist))
            if vals:
                m = np.mean(vals, axis=0); cells.append('%11.1f %5.1f %5.1f' % tuple(m))
            else:
                cells.append('%11s %5s %5s' % ('-', '-', '-'))
        print('%-22s %s  fell %d/%d' % (name, '  '.join(cells), sum(r['fell'] for r in rs), len(rs)))


def model_replay(a):
    """The same programs through the planner's step model from the recorded start state: heading and speed at
    decisions 8 / 16 / 24, game vs model."""
    from nav.learned_nav import planner as PL
    from nav.learned_nav.session import rolling_spin
    from nav.joystick import action_to_joystick
    g = Geometry(a.map)
    pl = PL.Planner(g, (0.0, 0.0, 0.0), seeds=None, use_flight=False)
    rows = [json.loads(l) for l in open(os.path.join(OUT_DIR, f'turndrill_{a.map}.jsonl'))]
    seen = {}
    for r in rows:
        seen.setdefault(r['program'], r)
    progs = programs()
    names = list(seen)
    n = len(names)
    pr = PL.Programs(n)
    for i, name in enumerate(names):
        for d, (ang, brake, jump) in enumerate(progs[name][:PL.H]):
            if brake:
                pr.mode[i, d] = PL.MODE_BRAKE
            else:
                pr.mode[i, d] = PL.MODE_DIR; pr.ang[i, d] = math.radians(ang)
            pr.jump[i, d] = jump
        pr.after[i] = PL.MODE_DIR; pr.after_ang[i] = math.radians(90); pr.after_n[i] = PL.H + 1
    r0 = seen[names[0]]
    P = np.tile(np.array(r0['rec'][0][:3], float), (n, 1)); V = np.tile(np.array([SPEED0, 0.0, 0.0]), (n, 1))
    W = np.tile(np.array(rolling_spin((SPEED0, 0.0, 0.0), (0, 0, 1)), float), (n, 1))
    push = tuple(action_to_joystick(1.0, 0.0, 1.0, 0, 0, 0.0, 0.0))
    Uc, Jc = PL.reply_vector(push)
    sim = PL.simulate(pl.steps, g, pl.eidx, pl.dev, P, V, W, np.tile(Uc, (n, 1)), np.full(n, Jc), np.tile(Uc, (n, 1)), np.full(n, Jc), pr, False, float(r0['rec'][0][2]) - R)
    print('%-22s %s' % ('program', '   game heading/speed @8 @16 @24        model heading/speed @8 @16 @24'))
    for i, name in enumerate(names):
        rec = seen[name]['rec']
        gm = ['%4.0f/%4.1f' % (rec[min(k, len(rec) - 1)][4], rec[min(k, len(rec) - 1)][3]) for k in (8, 16, 24)]
        md = []
        for k in (8, 16, 24):
            v = sim['vel'][i, k]; md.append('%4.0f/%4.1f' % (math.degrees(math.atan2(v[1], v[0])), math.hypot(v[0], v[1])))
        print('%-22s %s   %s' % (name, ' '.join(gm), ' '.join(md)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int)
    ap.add_argument('--map', default='kotmjump_p0')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--model', action='store_true')
    a = ap.parse_args()
    if a.model:
        model_replay(a)
    elif a.analyze:
        analyze(a)
    else:
        run(a)


if __name__ == '__main__':
    main()
