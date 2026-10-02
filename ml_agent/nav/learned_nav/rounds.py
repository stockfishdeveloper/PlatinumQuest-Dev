"""Stage 5 (design M4): whole rounds with the planner alone, every gem group of the real map.

    python -m nav.learned_nav.rounds --port 9301 --map kotmjump --rounds 2 --tag planner
    (then: marbleblast_mbx.exe -autotrain kotmjump -aiport 9301)

Each decision: the gems in view (the observation's gem list), an order for them, the planner (planner.py) aimed at
the first. The order: every permutation of the current gems scored by the time-to-gem guide leg by leg (each leg from
the previous gem, arriving along the leg), plus FLOAT_SETUP seconds for a floating gem (no floor under it: it needs a
run-up and a jump); re-chosen after every pickup. A floating gem gets its seeds (seeds.for_gem: built from geometry and
the model for that position); a floor gem is approached by walking distance. After a floating pickup the planner keeps
the marble safe until it has landed (continuation), then turns to the next gem. Out of bounds: the game's respawn,
then on. Nothing names a map: gems come from the game, geometry from the export.
Per round (logs/learned_nav/rounds/<tag>_<map>.jsonl): points, gems (floor / floating), falls, groups cleared with
their times, the longest time without a pickup, planning time.
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav.geometry import Geometry                                     # noqa: E402
from nav.learned_nav.session import Session, RoundOver, R                         # noqa: E402
from nav.learned_nav import planner as PL                                        # noqa: E402
from nav.learned_nav import seeds as SEEDS                                       # noqa: E402
from nav.learned_nav import drills as DR                                         # noqa: E402
from nav.gems import visible_gems                                                # noqa: E402
from nav.protocol import RAW_POS, RAW_VEL, RAW_SPIN, NOOP_ACTION                 # noqa: E402
from nav.joystick import action_to_joystick                                      # noqa: E402

OUT_DIR = os.path.join(HERE, 'logs', 'learned_nav', 'rounds')
FLOAT_SETUP = 1.5            # s added to a floating gem's leg in the order search (run-up and jump)
FLOAT_DZ = 1.5               # a gem with no floor within this below it floats
CONT_MAX = 40                # decisions of continuation after a floating pickup at most
LAND_HOLD = 4                # ... ending once the marble has been on a floor this many decisions
FLOOR_BROAD = 160            # broad proposals per decision toward a floor gem (planner.N_BROAD for a floating one)
ARRIVE_SPEED = 6.0           # u/s assumed when a leg of the order starts at a gem
TURN_S = 0.9                 # s per (1 - cos turn) at the start of a leg, scaled by speed / ARRIVE_SPEED (turn drill)
LEG_TIME_W = 0.015           # planner.time_w on every leg (the hybrid's shortcut value; the default 0.004 let a braking
LEG_P_SAT = 0.6              # turn at P 0.95 beat a fast turn at 0.8 a second quicker) and the ranking saturation


def is_floating(g, gem):
    z = g.floor_below(gem[0], gem[1], gem[2] + 0.05)
    return z is None or gem[2] - z > FLOAT_DZ


def choose_order(p, v, gems, floating, guide):
    """Permutation of gems (list of xyz) with the least guide time; legs chained from the current state."""
    k = len(gems)
    perms = list(itertools.permutations(range(k)))
    X, Y, VX, VY, GX, GY, which = [], [], [], [], [], [], []
    turn = np.zeros(len(perms))
    for pi, perm in enumerate(perms):
        pos = np.asarray(p[:2], float); vel = np.asarray(v[:2], float)
        for leg, i in enumerate(perm):
            gx, gy = gems[i][0], gems[i][1]
            X.append(pos[0]); Y.append(pos[1]); VX.append(vel[0]); VY.append(vel[1]); GX.append(gx); GY.append(gy)
            which.append((pi, i))
            d = np.array([gx, gy]) - pos
            # the turn at the start of the leg, costed from the turn drill (log 34.8: 90 deg ~0.9 s at speed, more
            # beyond); the guide alone sent the marble backwards on a third of the legs (log 34.12)
            sp = float(np.linalg.norm(vel)); dn = float(np.linalg.norm(d))
            if sp > 0.5 and dn > 1e-6:
                turn[pi] += TURN_S * (1.0 - float(np.dot(vel, d)) / (sp * dn)) * min(1.0, sp / ARRIVE_SPEED)
            vel = d / max(1e-6, dn) * ARRIVE_SPEED; pos = np.array([gx, gy])
    n = len(X)
    t = guide(np.array(X), np.array(Y), np.array(VX), np.array(VY), np.zeros(n), np.ones(n), np.array(GX), np.array(GY))
    tot = turn.copy()
    for (pi, i), ti in zip(which, t):
        tot[pi] += max(0.0, float(ti)) + (FLOAT_SETUP if floating[i] else 0.0)
    b = int(np.argmin(tot))
    return list(perms[b]), float(tot[b])


def same_gem(a, b, tol=0.3):
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol and abs(a[2] - b[2]) < tol


def play(port, map_name, n_rounds, tag, log):
    g = Geometry(map_name)
    s = Session(port, map_name, g, log=log)
    planner = PL.Planner(g, (0.0, 0.0, 0.0), seeds=None)
    guide = planner.guide
    if guide is None:
        from nav.learned_nav import guide as GD
        guide = GD.Guide(planner.dev)
    seed_cache = {}
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'{tag}_{map_name}.jsonl')
    done = 0
    while done < n_rounds:
        st = {'map': map_name, 'tag': tag, 'gems_floor': 0, 'gems_float': 0, 'falls': 0, 'groups': [], 'points': 0.0,
              'decisions': 0, 'ms': [], 'float_attempts': 0, 'longest_gap_s': 0.0, 'jumps': 0,
              'legs': [], 'trace': []}      # stage 6b: every leg (gem, decisions, pickup speed) and a per-decision trace
        try:
            s.ready()
            target = None; target_float = False; order_gems = []
            leg_i0 = 0; leg_gem = None
            cont = 0; landed_n = 0; last_pick_i = 0; group_key = None; group_t0 = 0
            Uc, Jc = PL.reply_vector(NOOP_ACTION + (0.0,)); Up, Jp = Uc.copy(), Jc
            msg = s.env.msg
            i = 0
            while True:
                ob = msg.obs
                p = np.asarray(ob[RAW_POS], float); v = np.asarray(ob[RAW_VEL], float); w = np.asarray(ob[RAW_SPIN], float)
                tel = msg.extra
                lv = g.floor_below(p[0], p[1], p[2])
                gap = (p[2] - R - lv) if lv is not None else 9.0
                sup = bool(tel is not None and tel[2] > 0)
                airborne = (not sup) and gap > 0.5
                gems = [q[:3] for q in visible_gems(ob)]
                key = tuple(sorted((round(q[0], 1), round(q[1], 1)) for q in gems))
                if gems and group_key is None:
                    group_key = key; group_t0 = i
                if cont > 0:                                           # after a floating pickup: land first
                    cont -= 1
                    landed_n = landed_n + 1 if (sup and gap < 0.35) else 0
                    if landed_n >= LAND_HOLD:
                        cont = 0
                if cont == 0 and gems and (target is None or not any(same_gem(target, q) for q in gems)):
                    floating = [is_floating(g, q) for q in gems]
                    order, _ = choose_order(p, v, gems, floating, guide)
                    target = np.asarray(gems[order[0]], float); target_float = floating[order[0]]
                    kk = tuple(np.round(target, 2))
                    if target_float and kk not in seed_cache:
                        seed_cache[kk] = SEEDS.for_gem(map_name, target)
                    planner.set_target(target, seed_cache.get(kk) if target_float else None)
                    # a floor gem needs no jump search: a smaller budget (a round is ~2800 decisions)
                    planner.n_broad = PL.N_BROAD if target_float else FLOOR_BROAD
                    planner.floor_ref = DR.floor_ref_of(g, target)
                    planner.time_w = LEG_TIME_W; planner.p_sat = LEG_P_SAT      # stage 6b: time decides among trusted plans
                    # leave this gem aligned toward the order's next gem (planner.exit_dir; stage 6b, log 34.12: legs that
                    # start turned away take 2.5-3 s against 1.15-1.25 s aligned, and the turn itself costs ~1 s beyond 90 deg)
                    planner.exit_dir = None
                    if len(order) > 1 and not target_float and not floating[order[1]]:
                        nxt = np.asarray(gems[order[1]], float); uu = nxt[:2] - target[:2]
                        if np.linalg.norm(uu) > 1e-6:
                            uu = uu / np.linalg.norm(uu); planner.exit_dir = (float(uu[0]), float(uu[1]))
                    st['float_attempts'] += int(target_float)
                    leg_i0 = i; leg_gem = [round(float(c), 2) for c in target]
                if target is None and cont == 0:                   # no gem in view yet: brake and wait
                    js = tuple(action_to_joystick(0.0, 0.0, 1.0, 0, 1, v[0], v[1]))
                else:                                              # a target, or the continuation after a pickup
                    js, info = planner.plan(p, v, w, tel, Uc, Jc, Up, Jp, airborne, cont > 0, planner.floor_ref)
                    js = tuple(float(a) for a in js[:4]) + (int(js[4]), float(js[5]))
                    st['ms'].append(info['ms'])
                    st['jumps'] += int(js[4] and not airborne)
                    st['trace'].append([i, round(float(p[0]), 2), round(float(p[1]), 2), round(math.hypot(v[0], v[1]), 2),
                                        (info['kind'] or '-')[0], int(info.get('tag', -1)), int(js[4]), 0,
                                        round(float(np.linalg.norm(np.asarray(target[:2]) - p[:2])), 2) if target is not None else -1.0])
                msg, sinfo = s.step(js)
                Up, Jp = Uc, Jc
                Uc, Jc = PL.reply_vector(js)
                i += 1
                if sinfo['gem_delta'] > 0:
                    st['points'] += float(sinfo['gem_delta'])
                    # which gem went: the one of the previous view that is no longer there (not always the target)
                    new = [q[:3] for q in visible_gems(msg.obs)]
                    gone = [q for q in gems if not any(same_gem(q, r) for r in new)]
                    if gone and is_floating(g, gone[0]):
                        st['gems_float'] += 1; cont = CONT_MAX; landed_n = 0
                    else:
                        st['gems_floor'] += 1
                    st['longest_gap_s'] = max(st['longest_gap_s'], (i - last_pick_i) * 0.064); last_pick_i = i
                    st['legs'].append({'gem': leg_gem, 'i0': leg_i0, 'pick_i': i, 't': round((i - leg_i0) * 0.064, 3),
                                       'pick_speed': round(math.hypot(msg.obs[RAW_VEL][0], msg.obs[RAW_VEL][1]), 2),
                                       'floating': bool(gone and is_floating(g, gone[0]))})
                    target = None
                if sinfo['fell']:
                    st['falls'] += 1
                    s.recover()
                    msg = s.env.msg; target = None; cont = 0
                    Uc, Jc = PL.reply_vector(NOOP_ACTION + (0.0,)); Up, Jp = Uc.copy(), Jc
                # a new group: more gems in view than before (positions are rebuilt from the marble's position and
                # the offset, so comparing them rounded flickers: the first version counted 12 groups in 13 s)
                if len(visible_gems(msg.obs)) > len(gems) and group_key is not None:
                    st['groups'].append(round((i - group_t0) * 0.064, 2)); group_t0 = i
                st['decisions'] = i
                if i % 200 == 0:
                    log(f'  t {i * 0.064:.0f} s: points {st["points"]:.0f}, floor {st["gems_floor"]}, floating '
                        f'{st["gems_float"]}, falls {st["falls"]}, groups {len(st["groups"])}')
        except RoundOver:
            pass
        st['score_end'] = st['points']
        ms = st.pop('ms')
        st['plan_ms_mean'] = float(np.mean(ms)) if ms else 0.0
        st['plan_ms_p95'] = float(np.percentile(ms, 95)) if ms else 0.0
        st['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(out, 'a') as f:
            f.write(json.dumps(st) + '\n')
        done += 1
        log(f'round {done}: points {st["points"]:.0f}, gems floor {st["gems_floor"]} floating {st["gems_float"]}, '
            f'falls {st["falls"]}, groups cleared {len(st["groups"])} {st["groups"]}, longest gap {st["longest_gap_s"]:.1f} s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--map', default='kotmjump')
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--tag', default='planner')
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    log_f = open(os.path.join(OUT_DIR, f'{a.tag}_{a.map}.log'), 'a', buffering=1)

    def log(m):
        line = f'[{time.strftime("%H:%M:%S")}] {m}'
        print(line, flush=True); log_f.write(line + '\n')
    play(a.port, a.map, a.rounds, a.tag, log)


if __name__ == '__main__':
    main()
