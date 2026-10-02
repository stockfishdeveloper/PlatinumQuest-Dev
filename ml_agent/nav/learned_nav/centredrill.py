"""Stage 6c: a physics check of crossings INTO a narrow landing (KOTM centre block) with FIXED programs: no planner.
The operator (10-01): the hybrid never jumps from the ring into the centre; "fix the physics". This drill measures
the engine and the step model on the same inputs.

    python -m nav.learned_nav.centredrill --port 9661 --map kotmjump_p0 [--shard k/n]   (the game on that port)
    python -m nav.learned_nav.centredrill --analyze                                   (offline: model vs game)

Starts: rays from the centre gem every 22.5 deg; where the ray crosses a void of >= 2 u, the marble starts RUNUP u
beyond the far lip on the floor, rolling at SPEEDS u/s straight at the gem (rolling spin). Program: full throttle
toward the gem; the jump key is pressed at the first decision where the void begins within LIP_U along the velocity
(the reply acts one decision later, the planner's rule); in the air one of AIRS: no input / the line to the gem /
60 deg past it; after the landing (support regained) brake. Recorded per decision: position, velocity, the input
sent (mode, direction, jump), support. Outcome: picked (within PICK_U of the gem in 3-D), fell, never took off.
Analysis: the same inputs through the step ensemble (planner.simulate) from the recorded start: p_succ, the
predicted landing point and closest pass against the game's. A table per (direction, speed, lip, air) and the
confusion of model p_succ against the game's outcome locates where the model is wrong.
Results: logs/learned_nav/centre/<map>.jsonl (resumable by trial key; run 1 = 12 decisions after the pickup, archived as <map>_run1.jsonl).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav.geometry import Geometry                                     # noqa: E402
from nav.learned_nav.session import Session, R, rolling_spin                     # noqa: E402
from nav.learned_nav import dynamics3 as D3                                      # noqa: E402
from nav.protocol import RAW_POS, RAW_VEL, RAW_SPIN                              # noqa: E402
from nav.joystick import action_to_joystick                                      # noqa: E402

OUT_DIR = os.path.join(HERE, 'logs', 'learned_nav', 'centre')
GEOMETRY = 'KingOfTheMarble_Hunt'
GEM = (-23.2, 13.0, 20.7)
DIRS = [22.5 * k for k in range(16)]
RUNUP = 4.0
SPEEDS = (9.0, 11.0, 13.0)
LIPS = (1.0, 2.0, 3.0)
AIRS = ('none', 'gem', 'past60')
PICK_U = 0.9
N_DEC = 64
AFTER_PICK = 24              # decisions observed after the pickup (the judge's CONT_WIN): a fall in them = not ok
MIN_VOID = 2.0


def ray_profile(g, gem, deg, zref):
    """Segments (r0, level or None) along the ray from the gem."""
    a = math.radians(deg); last = 'x'; segs = []
    for r in np.arange(0.0, 30.0, 0.25):
        z = g.floor_below(gem[0] + r * math.cos(a), gem[1] + r * math.sin(a), zref + 1.5)
        lvl = None if (z is None or z < zref - 0.6) else round(float(z), 2)
        key = lvl if lvl is None else 'f'
        if key != last:
            segs.append((float(r), lvl)); last = key
    return segs


def starts(g):
    out = []
    zref = g.floor_below(GEM[0], GEM[1], GEM[2] + 1.0)
    for deg in DIRS:
        segs = ray_profile(g, GEM, deg, zref)
        far = None
        for k, (r0, lvl) in enumerate(segs):
            if lvl is None and k + 1 < len(segs):
                width = segs[k + 1][0] - r0
                if width >= MIN_VOID:
                    far = segs[k + 1][0]; near = r0; break
        if far is None:
            continue
        a = math.radians(deg)
        r = far + RUNUP
        x, y = GEM[0] + r * math.cos(a), GEM[1] + r * math.sin(a)
        z = g.floor_below(x, y, zref + 1.5)
        if z is None or abs(z - zref) > 0.6:
            continue
        for sp in SPEEDS:
            vx, vy = -sp * math.cos(a), -sp * math.sin(a)
            out.append({'deg': deg, 'speed': sp, 'p': [x, y, z + R + 0.01], 'v': [vx, vy, 0.0],
                        'w': list(rolling_spin((vx, vy, 0.0), (0, 0, 1))), 'void': [near, far], 'straight': r})
    return out, zref


def js_of(mode, ang, jump, v):
    if mode == 'brake':
        js = action_to_joystick(0.0, 0.0, 1.0, jump, 1, v[0], v[1])
    elif mode == 'none':
        js = action_to_joystick(0.0, 0.0, 0.0, jump, 0, v[0], v[1])
    else:
        js = action_to_joystick(math.cos(ang), math.sin(ang), 1.0, jump, 0, v[0], v[1])
    return tuple(float(q) for q in js[:4]) + (int(js[4]), float(js[5]) if len(js) > 5 else 0.0)


def run(a):
    g = Geometry(GEOMETRY)
    sts, zref = starts(g)
    trials = [(s, lip, air) for s in sts for lip in LIPS for air in AIRS]
    if a.shard:
        k, n = (int(q) for q in a.shard.split('/'))
        trials = trials[k::n]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'{a.map}.jsonl')
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line); done.add((r['deg'], r['speed'], r['lip'], r['air']))
    trials = [t for t in trials if (t[0]['deg'], t[0]['speed'], t[1], t[2]) not in done]
    log = lambda m: print(m, flush=True)
    log(f'centredrill: {len(sts)} starts, {len(trials)} trials to run ({len(done)} done)')
    s = Session(a.port, a.map, g, log=log)
    out = open(out_path, 'a')
    gem = np.asarray(GEM, float)
    for st, lip, air in trials:
        p0 = st['p']; v0 = st['v']
        tg = math.atan2(gem[1] - p0[1], gem[0] - p0[0])
        push = js_of('dir', tg, 0, v0)
        o = s.place(tuple(p0), tuple(v0), tuple(st['w']), hold=push)
        if o is None:
            log(f'{st["deg"]} {st["speed"]} {lip} {air}: teleport refused'); continue
        rec = []; jumped_at = None; took_off = None; landed_at = None; dmin = 9.0; t_pick = None; fell = False
        for d in range(N_DEC):
            ob = s.obs(); p = np.asarray(ob[RAW_POS], float); v = np.asarray(ob[RAW_VEL], float)
            tel = s.env.msg.extra
            sup = bool(tel is not None and tel[2] > 0)
            lv = g.floor_below(p[0], p[1], p[2]); gap = (p[2] - R - lv) if lv is not None else 9.0
            airborne = (not sup) and gap > 0.5
            dd3 = float(np.linalg.norm(p - gem)); dmin = min(dmin, dd3)
            if t_pick is None and dd3 < PICK_U:
                t_pick = d
            if jumped_at is not None and took_off is None and airborne:
                took_off = d
            if took_off is not None and landed_at is None and sup and d > took_off:
                landed_at = d
            spd = math.hypot(v[0], v[1])
            jump = 0
            if jumped_at is None and sup and spd > 1.0:
                ux, uy = v[0] / spd, v[1] / spd
                dq = np.arange(0.25, lip + 0.01, 0.25)
                lvs, _ = D3.level_below(g, p[0] + dq * ux, p[1] + dq * uy, np.full(len(dq), p[2] + 0.3))
                void = ~(np.isfinite(lvs) & (lvs >= zref - 0.6))
                if void.any() and not void[0]:
                    jump = 1; jumped_at = d
            if landed_at is not None:
                mode, ang = 'brake', 0.0
            elif jumped_at is not None and d > jumped_at:
                if air == 'none':
                    mode, ang = 'none', 0.0
                elif air == 'gem':
                    mode, ang = 'dir', math.atan2(gem[1] - p[1], gem[0] - p[0])
                else:
                    mode, ang = 'dir', tg + math.radians(60)
            else:
                mode, ang = 'dir', tg
            rec.append([round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3),
                        round(float(v[0]), 3), round(float(v[1]), 3), round(float(v[2]), 3),
                        mode, round(ang, 4), jump, int(sup), int(airborne)])
            js = js_of(mode, ang, jump, v)
            msg, sinfo = s.step(js)
            if sinfo['fell']:
                fell = True; break
            if t_pick is not None and d > t_pick + AFTER_PICK:
                break
        # fallen = below the start floor by 0.5 u at the end (the session's fall flag needs the out-of-bounds trigger)
        if rec and rec[-1][2] - R < zref - 0.5:
            fell = True
        ok = t_pick is not None and not fell
        row = {'deg': st['deg'], 'speed': st['speed'], 'lip': lip, 'air': air, 'p': p0, 'v': v0, 'w': st['w'],
               'void': st['void'], 'straight': st['straight'], 'ok': ok, 'fell': fell, 't_pick': t_pick, 'dmin': round(dmin, 3),
               'jumped_at': jumped_at, 'took_off': took_off, 'landed_at': landed_at, 'rec': rec}
        out.write(json.dumps(row) + '\n'); out.flush()
        log('%5.1f deg %4.1f u/s lip %.1f air %-6s: %s jump@%s off@%s land@%s pick@%s dmin %.2f' % (
            st['deg'], st['speed'], lip, air, 'OK  ' if ok else ('FELL' if fell else 'miss'), jumped_at, took_off, landed_at, t_pick, dmin))
        if fell:
            s.recover()
    log('done')


def analyze(a):
    from nav.learned_nav import planner as PL
    g = Geometry(GEOMETRY)
    rows = [json.loads(l) for l in open(a.file or os.path.join(OUT_DIR, f'{a.map}.jsonl'))]
    rows = [r for r in rows if r['jumped_at'] is not None]
    pl = PL.Planner(g, GEM, seeds=None)
    gem = np.asarray(GEM, float)
    fr = g.floor_below(GEM[0], GEM[1], GEM[2] + 1.0)
    n = len(rows)
    pr = PL.Programs(n)
    P = np.zeros((n, 3)); V = np.zeros((n, 3)); W = np.zeros((n, 3))
    for i, r in enumerate(rows):
        rec = r['rec']
        P[i] = rec[0][:3]; V[i] = rec[0][3:6]; W[i] = r['w']
        for d in range(PL.H):
            q = rec[min(d, len(rec) - 1)]
            mode, ang, jump = q[6], q[7], q[8]
            if d >= len(rec):
                mode, jump = 'brake', 0
            pr.mode[i, d] = {'dir': PL.MODE_DIR, 'brake': PL.MODE_BRAKE, 'none': PL.MODE_NONE}[mode]
            pr.ang[i, d] = ang; pr.jump[i, d] = jump
        pr.after[i] = PL.MODE_BRAKE; pr.after_n[i] = PL.H + 1
    tg = np.arctan2(gem[1] - P[:, 1], gem[0] - P[:, 0])
    Uc = np.zeros((n, 2)); Jc = np.zeros(n)
    for i in range(n):
        u, j = PL.reply_vector(js_of('dir', float(tg[i]), 0, V[i]))
        Uc[i] = u; Jc[i] = j
    sim = PL.simulate(pl.steps, g, pl.eidx, pl.dev, P, V, W, Uc, Jc, Uc.copy(), Jc.copy(), pr, False, fr)
    jd = PL.judge(sim, gem)
    print('%-6s %5s %4s %-6s | game: %-4s jump land(x,y)      | model: p_succ p_pick land(x,y)      dmin   | landing err along/lateral' % (
        'deg', 'speed', 'lip', 'air', 'ok'))
    conf = {'ok_hi': 0, 'ok_lo': 0, 'bad_hi': 0, 'bad_lo': 0}
    errs = []
    for i, r in enumerate(rows):
        rec = r['rec']
        gl = rec[r['landed_at']][:2] if r['landed_at'] is not None else None
        ml = int(sim['landed'][i]); mlp = sim['path'][i, ml, :2] if ml >= 0 else None
        ps = float(jd['p_succ'][i]); pp = float(jd['p_pick'][i])
        key = ('ok' if r['ok'] else 'bad') + ('_hi' if ps >= 0.5 else '_lo'); conf[key] += 1
        e = ''
        if gl is not None and mlp is not None:
            d = mlp - np.asarray(gl); ux, uy = math.cos(tg[i]), math.sin(tg[i])
            al, la = float(d[0] * ux + d[1] * uy), float(-d[0] * uy + d[1] * ux); errs.append((al, la)); e = '%+.2f %+.2f' % (al, la)
        dm = float(np.min(np.linalg.norm(sim['path'][i] - gem, axis=1)))
        print('%-6.1f %5.1f %4.1f %-6s | %-4s %4s %-16s | %6.2f %6.2f %-16s %5.2f | %s' % (
            r['deg'], r['speed'], r['lip'], r['air'], 'OK' if r['ok'] else ('FELL' if r['fell'] else 'miss'), r['jumped_at'],
            '' if gl is None else '(%.1f,%.1f)@%d' % (gl[0], gl[1], r['landed_at']), ps, pp,
            '' if mlp is None else '(%.1f,%.1f)@%d' % (mlp[0], mlp[1], ml), dm, e))
    print('confusion: game ok & model p>=0.5 %d, game ok & model p<0.5 %d (FALSE NEGATIVES), game bad & p>=0.5 %d (false positives), bad & low %d' % (
        conf['ok_hi'], conf['ok_lo'], conf['bad_hi'], conf['bad_lo']))
    if errs:
        E = np.array(errs); print('model landing error vs game (n %d): along %+.2f (sd %.2f), lateral %+.2f (sd %.2f), mean |lateral| %.2f' % (
            len(E), E[:, 0].mean(), E[:, 0].std(), E[:, 1].mean(), E[:, 1].std(), np.abs(E[:, 1]).mean()))
    # per factor
    for fac in ('speed', 'lip', 'air', 'deg'):
        vals = sorted(set(r[fac] for r in rows))
        cells = []
        for vv in vals:
            idx = [i for i, r in enumerate(rows) if r[fac] == vv]
            ok = sum(rows[i]['ok'] for i in idx); hi = sum(float(jd['p_succ'][i]) >= 0.5 for i in idx)
            cells.append('%s: game %d/%d model %d' % (vv, ok, len(idx), hi))
        print(fac + ': ' + ' | '.join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int)
    ap.add_argument('--map', default='kotmjump_p0')
    ap.add_argument('--shard', default='')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--file', default='')
    a = ap.parse_args()
    if a.analyze:
        analyze(a)
    else:
        run(a)


if __name__ == '__main__':
    main()
