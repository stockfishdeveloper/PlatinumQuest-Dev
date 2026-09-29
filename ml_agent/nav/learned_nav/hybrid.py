"""Stage 5 (design M4): the PPO navigator drives, the planner takes over for floating gems and hands back.

    python -m nav.learned_nav.hybrid --port 9311 --map kotmjump --rounds 2 --memory current --tag hyb_current
    (then: marbleblast_mbx.exe -autotrain kotmjump -aiport 9311)

Ownership. The navigator (nav_latest.pth, played exactly as nav/real_run.py plays it: its gem chooser, the gem as the
goal, the next gem, the stuck-breaker, mean actions) drives while its target has a floor under it. When its target is
a FLOATING gem (no floor within 1.5 u below) the planner (planner.py) owns the marble: approach, run-up, jump, and after
the pickup the landing, until the marble has been on a floor LAND_HOLD decisions; then control goes back. Both always
see the real last reply (the game applies each reply one decision late).
Memory at handback (the design's comparison): 'current' keeps running the navigator on every observation while the
planner drives, its actions discarded, so its recurrent state is current when it takes over; 'reset' starts it from
a fresh state at handback.
Per round (logs/learned_nav/rounds/<tag>_<map>.jsonl): points, gems floor / floating, falls with their owner, falls
within 3 s after a handback, handovers, groups cleared, planning time.
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
from nav.learned_nav.session import Session, RoundOver, R                         # noqa: E402
from nav.learned_nav import planner as PL                                        # noqa: E402
from nav.learned_nav import seeds as SEEDS                                       # noqa: E402
from nav.learned_nav import drills as DR                                         # noqa: E402
from nav.learned_nav.rounds import is_floating, same_gem, OUT_DIR, CONT_MAX, LAND_HOLD   # noqa: E402
from nav.gems import visible_gems                                                # noqa: E402
from nav.protocol import RAW_POS, RAW_VEL, RAW_SPIN, NOOP_ACTION                 # noqa: E402
from nav.joystick import action_to_joystick as js_of                             # noqa: E402

AFTER_HANDBACK_S = 3.0       # a fall this soon after the navigator takes back over counts against the handover
STUCK_S, STUCK_U, STUCK_DETOUR = 3.0, 1.0, 24     # nav/real_run.py's stuck-breaker (defaults there)


def play(port, map_name, n_rounds, memory, tag, log):
    import torch
    from terrain_obs import TerrainMap
    from nav.terrain import TerrainGrid, CROP_SHAPE
    from nav.obs import ObsBuilder, NAV_OBS_VERSION, VEC_DIM
    from nav.model import NavActorCritic, action_to_joystick
    from nav.real_run import choose
    g = Geometry(map_name)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(os.path.join(HERE, 'models', 'nav', 'nav_latest.pth'), map_location=dev, weights_only=False)
    assert ck.get('obs_version') == NAV_OBS_VERSION
    model = NavActorCritic().to(dev); model.load_state_dict(ck['model']); model.eval()
    terrain = TerrainGrid(TerrainMap.resolve(map_name)); obs_b = ObsBuilder(terrain)
    planner = PL.Planner(g, (0.0, 0.0, 0.0), seeds=None)
    with torch.no_grad():
        model.act(torch.zeros(1, *CROP_SHAPE, device=dev), torch.zeros(1, VEC_DIM, device=dev),
                  model.initial_state(1, dev), deterministic=True)
    s = Session(port, map_name, g, log=log)
    log(f'hybrid: navigator update {ck.get("update")}, memory at handback "{memory}", map {map_name}')
    seed_cache = {}
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'{tag}_{map_name}.jsonl')
    done = 0
    while done < n_rounds:
        st = {'map': map_name, 'tag': tag, 'memory': memory, 'navigator_update': ck.get('update'), 'points': 0.0,
              'gems_floor': 0, 'gems_float': 0, 'falls_nav': 0, 'falls_planner': 0, 'falls_after_handback': 0,
              'handovers': 0, 'groups': [], 'decisions': 0, 'planner_decisions': 0, 'ms': [], 'stuck_breaks': 0}
        try:
            s.ready()
            obs_b.reset(); h = model.initial_state(1, dev)
            target = None; owner = 'nav'; cont = 0; landed_n = 0; handback_i = -10 ** 9
            ptarget = None; group_key = None; group_t0 = 0
            pos_hist = []; stuck_until = -1
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
                vis = visible_gems(ob)
                key = tuple(sorted((round(q[0], 1), round(q[1], 1)) for q in vis))
                if vis and group_key is None:
                    group_key = key; group_t0 = i
                target, nxt = choose(vis, target, pos=ob[0:2], vel=ob[3:5], terrain=terrain, mfield=None)
                # ---- ownership
                if owner == 'nav' and target is not None and is_floating(g, target[:3]):
                    owner = 'planner'; st['handovers'] += 1
                    ptarget = np.asarray(target[:3], float)
                    kk = tuple(np.round(ptarget, 2))
                    if kk not in seed_cache:
                        seed_cache[kk] = SEEDS.for_gem(map_name, ptarget)
                    planner.set_target(ptarget, seed_cache[kk])
                    planner.floor_ref = DR.floor_ref_of(g, ptarget)
                if owner == 'planner' and cont == 0 and ptarget is not None and not any(same_gem(ptarget, q[:3]) for q in vis):
                    ptarget = None                                     # the gem went without our pickup (should not happen)
                    owner = 'nav'; handback_i = i
                if owner == 'planner' and cont > 0:
                    cont -= 1
                    landed_n = landed_n + 1 if (sup and gap < 0.35) else 0
                    if landed_n >= LAND_HOLD or cont == 0:
                        owner = 'nav'; cont = 0; handback_i = i; ptarget = None
                        if memory == 'reset':
                            obs_b.reset(); h = model.initial_state(1, dev)
                # ---- the navigator (always run in 'current' memory mode; only when it owns otherwise)
                js_nav = None
                if owner == 'nav' or memory == 'current':
                    if target is not None:
                        goal = (target[0], target[1], target[2])
                        ngoal = (nxt[0], nxt[1], nxt[2]) if nxt else None
                        mx, my = float(p[0]), float(p[1]); pos_hist.append((mx, my))
                        nwin = int(round(STUCK_S / 0.064))
                        if (owner == 'nav' and i >= stuck_until and len(pos_hist) > nwin
                                and math.hypot(mx - pos_hist[-1 - nwin][0], my - pos_hist[-1 - nwin][1]) < STUCK_U):
                            obs_b.reset(); h = model.initial_state(1, dev); stuck_until = i + STUCK_DETOUR
                            st['stuck_breaks'] += 1
                        crop, vec, _ = obs_b.build(ob, goal, ngoal)
                        with torch.no_grad():
                            o = model.act(torch.as_tensor(crop, device=dev).unsqueeze(0), torch.as_tensor(vec, device=dev).unsqueeze(0),
                                          h, deterministic=not (i < stuck_until))
                        h = o['h_next']
                        a = o['action_game'][0].tolist()
                        js_nav = tuple(action_to_joystick(a[0], a[1], a[2], a[3], a[4], float(v[0]), float(v[1])))
                    else:
                        js_nav = tuple(js_of(0.0, 0.0, 1.0, 0, 1, v[0], v[1]))     # no gem in view: brake and wait
                if owner == 'planner':
                    js, info = planner.plan(p, v, w, tel, Uc, Jc, Up, Jp, airborne, cont > 0, planner.floor_ref)
                    st['ms'].append(info['ms']); st['planner_decisions'] += 1
                else:
                    js = js_nav
                js = tuple(float(a) for a in js[:4]) + (int(js[4]), float(js[5]) if len(js) > 5 else 0.0)
                msg, sinfo = s.step(js)
                Up, Jp = Uc, Jc
                Uc, Jc = PL.reply_vector(js)
                i += 1
                if sinfo['gem_delta'] > 0:
                    st['points'] += float(sinfo['gem_delta'])
                    new = [q[:3] for q in visible_gems(msg.obs)]
                    gone = [q[:3] for q in vis if not any(same_gem(q[:3], r) for r in new)]
                    if gone and is_floating(g, gone[0]):
                        st['gems_float'] += 1
                        if owner == 'planner':
                            cont = CONT_MAX; landed_n = 0
                    else:
                        st['gems_floor'] += 1
                    target = None
                if sinfo['fell']:
                    st['falls_planner' if owner == 'planner' else 'falls_nav'] += 1
                    if owner == 'nav' and (i - handback_i) * 0.064 <= AFTER_HANDBACK_S:
                        st['falls_after_handback'] += 1
                    s.recover(); msg = s.env.msg
                    owner = 'nav'; cont = 0; ptarget = None; target = None
                    obs_b.reset(); h = model.initial_state(1, dev); pos_hist = []
                    Uc, Jc = PL.reply_vector(NOOP_ACTION + (0.0,)); Up, Jp = Uc.copy(), Jc
                # a new group: more gems in view than before (positions are rebuilt from the marble's position and
                # the offset, so comparing them rounded flickers: the first version counted 12 groups in 13 s)
                if len(visible_gems(msg.obs)) > len(vis) and group_key is not None:
                    st['groups'].append(round((i - group_t0) * 0.064, 2)); group_t0 = i
                st['decisions'] = i
                if i % 300 == 0:
                    log(f'  t {i * 0.064:.0f} s: points {st["points"]:.0f}, floor {st["gems_floor"]}, floating '
                        f'{st["gems_float"]}, falls nav {st["falls_nav"]} planner {st["falls_planner"]}, '
                        f'groups {len(st["groups"])}, owner {owner}')
        except RoundOver:
            pass
        ms = st.pop('ms')
        st['plan_ms_mean'] = float(np.mean(ms)) if ms else 0.0
        st['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(out, 'a') as f:
            f.write(json.dumps(st) + '\n')
        done += 1
        log(f'round {done}: points {st["points"]:.0f}, gems floor {st["gems_floor"]} floating {st["gems_float"]}, '
            f'falls nav {st["falls_nav"]} planner {st["falls_planner"]} (after handback {st["falls_after_handback"]}), '
            f'handovers {st["handovers"]}, groups {len(st["groups"])} {st["groups"]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--map', default='kotmjump')
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--memory', choices=('current', 'reset'), default='current')
    ap.add_argument('--tag', default='hybrid')
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    log_f = open(os.path.join(OUT_DIR, f'{a.tag}_{a.map}.log'), 'a', buffering=1)

    def log(m):
        line = f'[{time.strftime("%H:%M:%S")}] {m}'
        print(line, flush=True); log_f.write(line + '\n')
    play(a.port, a.map, a.rounds, a.memory, a.tag, log)


if __name__ == '__main__':
    main()
