"""In-game check of the KOTM centre-to-corner diagonal jump (open loop) against the step model's prediction.

    python diagjump.py --port P --map kotmjump_p0
Teleport to a centre-gem pickup state moving toward the corner gem, fly a straight-line jump program, brake after the
landing; record the real path and the model's prediction of the same program from the observed start.
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
out = open(os.path.join(ML, 'logs', 'learned_nav', 'diagjump3.jsonl'), 'a')
g = Geometry(a.map)
pl = PL.Planner(g, (0, 0, 0), seeds=None)
gem = np.array([-37.2, 27.0, 20.7]); fr = DR.floor_ref_of(g, gem)
s = Session(a.port, a.map, g, log=lambda m: print(m, flush=True))
h = math.radians(135.0); tg = h
z0 = g.floor_below(-27.2, 17.0, 25.0)
push = tuple(action_to_joystick(math.cos(h), math.sin(h), 1.0, 0, 0, 0.0, 0.0))
trial = 0
for sp in (9.0, 10.0, 11.0):
    for j in (0, 1):
        for air in ('toward', 'none', 'brake', 'half'):
            # start 0.6 u behind the centre gem so the 2 settle decisions bring the marble to about the gem
            back = 0.3
            pos = (-27.2 - back * math.cos(h), 17.0 - back * math.sin(h), z0 + R + 0.01)
            vel = (sp * math.cos(h), sp * math.sin(h), 0.0)
            o = s.place(pos, vel, rolling_spin(vel, (0, 0, 1)), hold=push)
            if o is None:
                print('teleport refused', sp, j, air, flush=True); continue
            p0 = np.asarray(o[RAW_POS], float); v0 = np.asarray(o[RAW_VEL], float); w0 = np.asarray(o[RAW_SPIN], float)
            pr = PL.Programs(1); pr.ang[:] = tg; pr.jump[0, j] = 1
            if air == 'none':
                pr.mode[0, j + 1:] = PL.MODE_NONE
            elif air == 'brake':
                pr.mode[0, j + 1:] = PL.MODE_BRAKE
            elif air == 'half':
                pr.thr[0, j + 1:] = 0.5
            pr.after[0] = PL.MODE_BRAKE; pr.mode[:, PL.H - PL.TAIL:] = PL.MODE_BRAKE
            U = PL.SQ2 * np.array([math.cos(h), math.sin(h)])
            sim = PL.simulate(pl.steps, g, pl.eidx, pl.dev, p0[None], v0[None], w0[None], U[None], np.zeros(1), U[None],
                              np.zeros(1), pr, False, fr)
            jd = PL.judge(sim, gem)
            # execute
            states = [np.r_[p0, v0, w0].tolist()]; replies = []; tels = []
            real = [p0.tolist()]; landed = -1; airborne = False; reair = False; fell = False
            for d in range(PL.H):
                ob = s.obs(); v = np.asarray(ob[RAW_VEL], float); pz = float(ob[RAW_POS][2]); px, py = float(ob[RAW_POS][0]), float(ob[RAW_POS][1])
                tel = s.env.msg.extra; sup = bool(tel is not None and tel[2] > 0)
                lv = g.floor_below(px, py, pz); gap = (pz - R - lv) if lv is not None else 9.0
                if not sup and gap > 0.5:
                    if landed >= 0:
                        reair = True
                    airborne = True
                elif airborne and sup and landed < 0:
                    landed = d
                if landed >= 0:
                    js = tuple(action_to_joystick(0.0, 0.0, 1.0, 0, 1, float(v[0]), float(v[1])))
                else:
                    m = int(pr.mode[0, d])
                    if m == PL.MODE_NONE:
                        js = tuple(action_to_joystick(0.0, 0.0, 0.0, int(pr.jump[0, d]), 0, 0.0, 0.0))
                    elif m == PL.MODE_BRAKE:
                        js = tuple(action_to_joystick(0.0, 0.0, 1.0, int(pr.jump[0, d]), 1, float(v[0]), float(v[1])))
                    else:
                        js = tuple(action_to_joystick(math.cos(tg), math.sin(tg), float(pr.thr[0, d]), int(pr.jump[0, d]), 0, float(v[0]), float(v[1])))
                replies.append([float(c) for c in js]); tels.append(list(tel) if tel is not None else None)
                msg, info = s.step(js)
                states.append(np.r_[msg.obs[RAW_POS][0:3], msg.obs[RAW_VEL][0:3], msg.obs[RAW_SPIN][0:3]].astype(float).tolist())
                real.append([float(c) for c in msg.obs[RAW_POS][0:3]])
                if info['fell']:
                    fell = True; break
            R_ = np.array(real)
            dmin_real = float(np.min(np.linalg.norm(R_[:, :2] - gem[:2], axis=1)))
            row = {'states': states, 'replies': replies, 'push': list(push), 'trial': trial, 'speed': sp, 'jump': j, 'air': air, 'start': p0.round(2).tolist(), 'v0': round(float(np.hypot(v0[0], v0[1])), 2),
                   'real': {'fell': fell, 'landed': landed, 'reair': reair, 'dmin': round(dmin_real, 2),
                            'land_xy': R_[landed, :2].round(2).tolist() if landed >= 0 else None, 'end': R_[-1].round(2).tolist()},
                   'model': {'fell': bool(jd['fell'][0]), 'landed': int(sim['landed'][0]), 'reair': bool(sim['reair'][0]),
                             'dmin': round(float(jd['dmin'][0]), 2), 'safe': bool(jd['safe'][0]),
                             'land_xy': sim['path'][0, sim['landed'][0], :2].round(2).tolist() if sim['landed'][0] >= 0 else None,
                             'end': sim['path'][0, -1].round(2).tolist()}}
            out.write(json.dumps(row) + '\n'); out.flush()
            print(json.dumps(row), flush=True)
            trial += 1
            if fell:
                s.recover()
print('done', flush=True)
