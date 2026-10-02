"""Closed-loop planner test of the KOTM centre-gem -> corner-gem crossing (kotmjump_p0 geometry = KOTM).

    python cornerdrill.py --port P --map kotmjump_p0
Teleport to a pickup-like state at the centre gem moving roughly toward the corner; the planner (shortcut settings:
time weight, straight-line proposals, after-landing steering, jump prior) plans to the corner point. Success: passes
within 0.9 u of it (the pickup radius is ~1 u) and stays on the map for 16 decisions after.
"""
import argparse, json, math, os, sys, time
import numpy as np
ML = 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent'
sys.path.insert(0, ML)
from nav.learned_nav.geometry import Geometry
from nav.learned_nav.session import Session, R, rolling_spin
from nav.learned_nav import planner as PL, drills as DR
from nav.protocol import RAW_POS, RAW_VEL, RAW_SPIN, NOOP_ACTION
from nav.joystick import action_to_joystick

ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int); ap.add_argument('--map')
a = ap.parse_args()
out = open(os.path.join(ML, 'logs', 'learned_nav', 'cornerdrill.jsonl'), 'a')
g = Geometry(a.map)
pl = PL.Planner(g, (0, 0, 0), seeds=None)
gem = np.array([-37.2, 27.0, 20.7]); fr = DR.floor_ref_of(g, gem)
s = Session(a.port, a.map, g, log=lambda m: print(m, flush=True))
z0 = g.floor_below(-27.2, 17.0, 25.0)
rng = np.random.default_rng(1)
for trial in range(24):
    sp = float(rng.uniform(6.0, 10.0)); h = math.radians(135.0 + rng.uniform(-35, 35))
    pos = (-27.2 + rng.uniform(-0.3, 0.3), 17.0 + rng.uniform(-0.3, 0.3), z0 + R + 0.01)
    vel = (sp * math.cos(h), sp * math.sin(h), 0.0)
    push = tuple(action_to_joystick(math.cos(h), math.sin(h), 1.0, 0, 0, 0.0, 0.0))
    o = s.place(pos, vel, rolling_spin(vel, (0, 0, 1)), hold=push)
    if o is None:
        print('teleport refused', flush=True); continue
    pl.set_target(gem, None); pl.floor_ref = fr; pl.n_broad = PL.N_BROAD
    pl.time_w = 0.015; pl.approach_line = True; pl.land_steer = True
    Uc, Jc = PL.reply_vector(push); Up, Jp = Uc.copy(), Jc
    path = []; dmin = 9.0; t_pick = None; fell = False; kinds = []; jumps = 0; after = 0
    for d in range(80):
        ob = s.obs(); p = np.asarray(ob[RAW_POS], float); v = np.asarray(ob[RAW_VEL], float); w = np.asarray(ob[RAW_SPIN], float)
        tel = s.env.msg.extra
        lv = g.floor_below(p[0], p[1], p[2]); gap = (p[2] - R - lv) if lv is not None else 9.0
        sup = bool(tel is not None and tel[2] > 0); airborne = (not sup) and gap > 0.5
        dd = float(np.linalg.norm(p - gem))
        if dd < dmin:
            dmin = dd
        if t_pick is None and dd < 0.9:
            t_pick = d
        if t_pick is not None:
            after += 1
            if after > 16:
                break
        js, info = pl.plan(p, v, w, tel, Uc, Jc, Up, Jp, airborne, t_pick is not None, fr)
        kinds.append(info['kind'][0] if info['kind'] else '-')
        jumps += int(js[4])
        msg, sinfo = s.step(tuple(float(q) for q in js[:4]) + (int(js[4]), float(js[5]) if len(js) > 5 else 0.0))
        Up, Jp = Uc, Jc; Uc, Jc = PL.reply_vector(js)
        path.append([round(float(q), 2) for q in p])
        if sinfo['fell']:
            fell = True; break
    ok = t_pick is not None and not fell
    row = {'trial': trial, 'speed': round(sp, 2), 'heading': round(math.degrees(h), 1), 'ok': ok, 't_pick': t_pick,
           'dmin': round(dmin, 2), 'fell': fell, 'jumps': jumps, 'kinds': ''.join(kinds), 'path': path}
    out.write(json.dumps(row) + '\n'); out.flush()
    print('trial %2d speed %.1f heading %.0f: ok %d t_pick %s dmin %.2f fell %d jumps %d kinds %s' % (
        trial, sp, math.degrees(h), ok, t_pick, dmin, fell, jumps, ''.join(kinds)), flush=True)
    if fell:
        s.recover()
print('done', flush=True)
