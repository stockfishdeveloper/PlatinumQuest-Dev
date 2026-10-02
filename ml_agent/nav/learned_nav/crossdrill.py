"""Stage 6b: the crossing leg from REAL pickup states (cross_starts.py), planner closed-loop in the game.

    python -m nav.learned_nav.crossdrill --port 9621 --map kotmjump_p0 --set dev --tag v0 [--planner nav.learned_nav.planner_v8]
    (then: marbleblast_mbx.exe -autotrain kotmjump_p0 -aiport 9621)
    python -m nav.learned_nav.crossdrill --report v0 v1 [--set dev]      (paired comparison, exact McNemar, times)

A trial: teleport to the recorded state right after a pickup at gem A (position, velocity, spin; the input held is a
push along the velocity, standing in for the navigator's last reply), aim the planner at gem B across the hole with
the shortcut settings (time weight, straight-line proposals, after-landing steering), one planner decision per 64 ms
until the marble passes within PICK_U of B (the pickup radius), falls, or TIMEOUT_DEC. Success = pickup and AFTER
decisions on the map afterwards. Recorded per trial: decisions to the pickup (the leg time), the decision of the first
pickup plan, the jump decision and the speed at it, the kinds sequence (a approach / p pickup / c continue), and every
decision's state. The navigator's fitted time for the same leg (route.Route.legs, walking) is stored for comparison.
Results: logs/learned_nav/cross/<tag>_<set>.jsonl (resumable: done start ids are skipped).
"""
import argparse
import importlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav.geometry import Geometry                                     # noqa: E402
from nav.learned_nav.session import Session, R                                   # noqa: E402
from nav.learned_nav import drills as DR                                         # noqa: E402
from nav.learned_nav import route as RT                                          # noqa: E402
from nav.protocol import RAW_POS, RAW_VEL, RAW_SPIN                              # noqa: E402
from nav.joystick import action_to_joystick                                      # noqa: E402

STARTS_DIR = os.path.join(HERE, 'datasets', 'learned_nav', 'cross')
OUT_DIR = os.path.join(HERE, 'logs', 'learned_nav', 'cross')
PICK_U = 0.9                 # within this of the gem centre = picked (the pickup radius is ~1 u; planner PICK_R0 0.85)
TIMEOUT_DEC = 70             # 4.5 s (the navigator walks these legs in ~2.2 s)
AFTER = 24                   # decisions on the map after the pickup (16 until log 36.3; a run-off after the pickup
                             # registers as a fall by the floor-level check below, the game's trigger comes later)
SHORT_TIME_W = 0.015         # the hybrid's shortcut settings (hybrid.SHORT_TIME_W)


def load_starts(set_name):
    return json.load(open(os.path.join(STARTS_DIR, f'starts_{set_name}.json')))


def run(a):
    PL = importlib.import_module(a.planner)
    g = Geometry(a.map)
    pl = PL.Planner(g, (0.0, 0.0, 0.0), seeds=None)
    rt = RT.Route(g)
    starts = load_starts(a.set)
    starts = [s for s in starts if abs(s['angle']) <= a.max_angle]   # the human's crossings: median 30 deg, p75 ~60
    if a.ids:
        want = set(int(x) for x in a.ids.split(','))
        starts = [s for s in starts if s['id'] in want]
    starts = starts[a.offset::a.stride][:a.limit] if a.limit else starts[a.offset::a.stride]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'{a.tag}_{a.set}.jsonl')
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            done.add(json.loads(line)['id'])
    starts = [s for s in starts if s['id'] not in done]
    log = lambda m: print(m, flush=True)
    log(f'crossdrill: planner {a.planner}, set {a.set}, {len(starts)} starts to run ({len(done)} done), map {a.map}')
    s = Session(a.port, a.map, g, log=log)
    out = open(out_path, 'a')
    fr_cache = {}
    n_ok = n_fell = 0; times = []
    for st in starts:
        gem = np.asarray(st['gem_b'], float)
        gk = tuple(np.round(gem, 1))
        if gk not in fr_cache:
            fr_cache[gk] = DR.floor_ref_of(g, gem)
        fr = fr_cache[gk]
        p0 = st['p']; v0 = st['v']; w0 = st['w']
        sp = math.hypot(v0[0], v0[1])
        h = math.atan2(v0[1], v0[0]) if sp > 0.5 else math.atan2(gem[1] - p0[1], gem[0] - p0[0])
        push = tuple(action_to_joystick(math.cos(h), math.sin(h), 1.0, 0, 0, 0.0, 0.0))
        o = s.place(tuple(p0), tuple(v0), tuple(w0), hold=push)
        if o is None:
            log(f'start {st["id"]}: teleport refused'); continue
        pl.set_target(gem, None); pl.floor_ref = fr; pl.n_broad = PL.N_BROAD
        pl.time_w = SHORT_TIME_W; pl.approach_line = True; pl.land_steer = True
        for kv in (a.variant.split(',') if a.variant else []):        # experiment settings, e.g. psat=0.6,time_w=0.03
            k, v = kv.split('='); setattr(pl, k, float(v))
        Uc, Jc = PL.reply_vector(push); Up, Jp = Uc.copy(), Jc
        t_nav = rt.legs(p0[0], p0[1], h, sp, gem)[0]
        rec = []; kinds = []; dmin = 9.0; t_pick = None; fell = False; after = 0
        jump_at = None; jump_speed = None; first_plan = None; ms = []
        t0 = time.time()
        for d in range(TIMEOUT_DEC + AFTER):
            ob = s.obs(); p = np.asarray(ob[RAW_POS], float); v = np.asarray(ob[RAW_VEL], float); w = np.asarray(ob[RAW_SPIN], float)
            tel = s.env.msg.extra
            lv = g.floor_below(p[0], p[1], p[2]); gap = (p[2] - R - lv) if lv is not None else 9.0
            sup = bool(tel is not None and tel[2] > 0); airborne = (not sup) and gap > 0.5
            dd = float(math.hypot(p[0] - gem[0], p[1] - gem[1]))
            dd3 = float(np.linalg.norm(p - gem))
            dmin = min(dmin, dd3)
            if t_pick is None and dd3 < PICK_U:
                t_pick = d
            if t_pick is not None:
                after += 1
                if after > AFTER:
                    break
            elif d >= TIMEOUT_DEC:
                break
            js, info = pl.plan(p, v, w, tel, Uc, Jc, Up, Jp, airborne, t_pick is not None, fr)
            ms.append(info['ms'])
            kind = info['kind'][0] if info['kind'] else '-'
            kinds.append(kind)
            if first_plan is None and kind == 'p':
                first_plan = d
            spd = float(math.hypot(v[0], v[1]))
            if js[4] and jump_at is None and not airborne:
                jump_at = d; jump_speed = round(spd, 2)
            rec.append([round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2), round(spd, 2), kind,
                        int(info.get('tag', -1)), int(js[4]), round(float(info.get('p_succ', 0.0)), 2), int(airborne),
                        round(dd, 2)])
            msg, sinfo = s.step(tuple(float(q) for q in js[:4]) + (int(js[4]), float(js[5]) if len(js) > 5 else 0.0))
            Up, Jp = Uc, Jc; Uc, Jc = PL.reply_vector(js)
            if sinfo['fell']:
                fell = True; break
        if rec and rec[-1][2] - R < fr - 0.5:
            fell = True                                            # below the gem's floor at the end: fallen
        ok = t_pick is not None and not fell
        row = {'id': st['id'], 'set': a.set, 'tag': a.tag, 'planner': a.planner, 'gem_a': st['gem_a'], 'gem_b': st['gem_b'],
               'speed': st['speed'], 'angle': st['angle'], 'straight': st['straight'], 'ok': ok, 't_pick': t_pick,
               't_s': None if t_pick is None else round(t_pick * 0.064, 3), 'dmin': round(dmin, 2), 'fell': fell,
               'timeout': t_pick is None and not fell, 'jump_at': jump_at, 'jump_speed': jump_speed,
               'first_plan': first_plan, 'kinds': ''.join(kinds), 't_nav': round(t_nav, 3),
               'ms': round(float(np.mean(ms)), 0) if ms else None, 'wall': round(time.time() - t0, 1), 'rec': rec}
        out.write(json.dumps(row) + '\n'); out.flush()
        n_ok += int(ok); n_fell += int(fell)
        if ok:
            times.append(t_pick * 0.064)
        log('start %3d A %s -> B %s speed %4.1f angle %+4.0f: ok %d t %s fell %d jump_at %s @ %s first_plan %s '
            'kinds %s (t_nav %.2f) [ok %d fell %d of %d, median t %.2f]' % (
                st['id'], st['gem_a'][:2], st['gem_b'][:2], st['speed'], st['angle'], ok,
                '%.2f' % (t_pick * 0.064) if t_pick is not None else '-', fell, jump_at, jump_speed, first_plan,
                ''.join(kinds), t_nav, n_ok, n_fell, n_ok + n_fell + (len(times) - n_ok) + 0, np.median(times) if times else 0.0))
        if fell:
            s.recover()
    log('done: ok %d, fell %d, of %d; median leg time %.2f s' % (n_ok, n_fell, len(starts), np.median(times) if times else 0.0))


def summarize(rows):
    n = len(rows); ok = [r for r in rows if r['ok']]
    fell = sum(r['fell'] for r in rows); to = sum(r['timeout'] for r in rows)
    t = np.array([r['t_s'] for r in ok]) if ok else np.array([])
    tn = np.array([r['t_nav'] for r in rows])
    # expected time per start with a failure costed at the navigator's walking time plus a respawn (3.5 s)
    exp = np.array([r['t_s'] if r['ok'] else r['t_nav'] + 3.5 for r in rows])
    return {'n': n, 'ok': len(ok), 'fell': fell, 'timeout': to, 't_median': float(np.median(t)) if len(t) else None,
            't_mean': float(t.mean()) if len(t) else None, 't_nav_mean': float(tn.mean()), 'exp_mean': float(exp.mean()),
            'jump_speed_median': float(np.median([r['jump_speed'] for r in rows if r['jump_speed'] is not None] or [0])),
            'first_plan_median': float(np.median([r['first_plan'] for r in rows if r['first_plan'] is not None] or [-1]))}


def phases(r):
    """Per-trial phases from the decision record: run-up (to the jump), flight (airborne), landing to pickup."""
    rec = r['rec']
    air = [q[8] for q in rec]
    jump = r['jump_at']
    t_land = None
    if jump is not None:
        for i in range(jump + 1, len(rec)):
            if air[i]:
                for k in range(i + 1, len(rec)):
                    if not air[k]:
                        t_land = k; break
                break
    out = {'jump': jump, 'jump_speed': r['jump_speed'], 'land': t_land, 'pick': r['t_pick'],
           'land_speed': rec[t_land][3] if t_land is not None else None,
           'land_dd': rec[t_land][9] if t_land is not None else None,
           'first_plan': r['first_plan'], 'n_approach': r['kinds'].count('a'),
           'peak_speed': max(q[3] for q in rec) if rec else None}
    return out


def analyze(a):
    sets = [a.set] if a.set else ['dev', 'test']
    for set_name in sets:
        for tag in a.analyze:
            f = os.path.join(OUT_DIR, f'{tag}_{set_name}.jsonl')
            if not os.path.exists(f):
                continue
            rows = [json.loads(l) for l in open(f)]
            print(f'== {tag} {set_name}: {json.dumps(summarize(rows))}')
            ok = [r for r in rows if r['ok']]
            P = [phases(r) for r in ok]
            med = lambda k: np.median([p[k] for p in P if p[k] is not None]) if any(p[k] is not None for p in P) else float('nan')
            print('  ok trials %d: median jump decision %.0f (speed %.1f, first pickup plan at %.0f, approach decisions %.0f), '
                  'landing at %.0f (speed %.1f, %.1f u from the gem), pickup at %.0f; peak speed %.1f' % (
                      len(P), med('jump'), med('jump_speed'), med('first_plan'), med('n_approach'), med('land'), med('land_speed'),
                      med('land_dd'), med('pick'), med('peak_speed')))
            for r in [r for r in rows if not r['ok']]:
                ph = phases(r)
                print('  FAIL id %3d A %s -> B %s speed %4.1f angle %+4.0f: fell %d timeout %d jump %s @ %s land %s dd %s first_plan %s dmin %.2f kinds %s' % (
                    r['id'], r['gem_a'][:2], r['gem_b'][:2], r['speed'], r['angle'], r['fell'], r['timeout'], ph['jump'], ph['jump_speed'],
                    ph['land'], ph['land_dd'], ph['first_plan'], r['dmin'], r['kinds'][:60]))
            # time split by angle
            for lo, hi in ((0, 30), (30, 60)):
                sub = [r for r in rows if lo <= abs(r['angle']) < hi]
                if sub:
                    t = [r['t_s'] for r in sub if r['ok']]
                    print('  |angle| %2d-%2d: %d starts, ok %d, fell %d, median t %.2f, t_nav %.2f' % (
                        lo, hi, len(sub), sum(r['ok'] for r in sub), sum(r['fell'] for r in sub), np.median(t) if t else float('nan'),
                        np.mean([r['t_nav'] for r in sub])))


def report(a):
    from scipy.stats import binomtest
    sets = [a.set] if a.set else ['dev', 'test']
    for set_name in sets:
        data = {}
        for tag in a.report:
            f = os.path.join(OUT_DIR, f'{tag}_{set_name}.jsonl')
            if os.path.exists(f):
                data[tag] = {json.loads(l)['id']: json.loads(l) for l in open(f)}
        if not data:
            continue
        print(f'== {set_name}')
        for tag, rows in data.items():
            print('  %-12s %s' % (tag, json.dumps(summarize(list(rows.values())))))
        tags = list(data)
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                A, B = data[tags[i]], data[tags[j]]
                ids = sorted(set(A) & set(B))
                if not ids:
                    continue
                a_only = sum(A[k]['ok'] and not B[k]['ok'] for k in ids); b_only = sum(B[k]['ok'] and not A[k]['ok'] for k in ids)
                p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
                both = [k for k in ids if A[k]['ok'] and B[k]['ok']]
                dt = np.mean([B[k]['t_s'] - A[k]['t_s'] for k in both]) if both else float('nan')
                ea = np.mean([A[k]['t_s'] if A[k]['ok'] else A[k]['t_nav'] + 3.5 for k in ids])
                eb = np.mean([B[k]['t_s'] if B[k]['ok'] else B[k]['t_nav'] + 3.5 for k in ids])
                print('  %s vs %s on %d shared starts: ok %d vs %d (only-A %d, only-B %d, McNemar p %.3f); '
                      'time B - A on both-ok %+.2f s; expected time %.2f vs %.2f s' % (
                          tags[i], tags[j], len(ids), sum(A[k]['ok'] for k in ids), sum(B[k]['ok'] for k in ids),
                          a_only, b_only, p, dt, ea, eb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int)
    ap.add_argument('--map', default='kotmjump_p0')
    ap.add_argument('--set', default=None)
    ap.add_argument('--tag', default='v0')
    ap.add_argument('--planner', default='nav.learned_nav.planner')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--ids', default='')
    ap.add_argument('--max-angle', type=float, default=60.0)
    ap.add_argument('--variant', default='')
    ap.add_argument('--report', nargs='*')
    ap.add_argument('--analyze', nargs='*')
    a = ap.parse_args()
    if a.report:
        report(a)
    elif a.analyze:
        analyze(a)
    else:
        if a.set is None:
            a.set = 'dev'
        run(a)


if __name__ == '__main__':
    main()
