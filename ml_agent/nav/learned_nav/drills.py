"""Stage 4 drills (design M3): the planner (planner.py) closed-loop in the game, one floating gem per drill map.

    python -m nav.learned_nav.drills --port 9201 --map kotmjump_p1 --set dev
    (then: marbleblast_mbx.exe -autotrain kotmjump_p1 -aiport 9201)

Starts (fixed per map and set, written once to logs/learned_nav/drills/starts_<set>_<map>.json and then frozen): on
the gem's floor level, 4-18 u from the gem horizontally, at least 1 u from any void or drop edge; 30 % at rest,
otherwise 2-12 u/s with rolling spin, heading toward the gem (within 40 deg) for 35 % and anywhere for the rest, so
most starts need a turn and a run-up. No input held.
Fixture (before any planner trial, engine only): each start is also run with three simple controls (brake; steer 90
deg left then brake; right then brake) for 3 s. A start none of them keeps on the map is UNSAFE: still run, reported
apart, outside the gate's denominator. Feasibility of the task itself from a safe start: the floor is one connected
level, and a pickup plus continuation from that floor is demonstrated in the engine for each target.
A trial: place the start, then one planner decision per 64 ms decision until success, out of bounds or TIMEOUT_DEC
(stalls count as failures). Success = the engine's pickup (score change; the drill map has only the target gem),
then a landing and CONT decisions on the map (P0's continuation rule), ending over a floor.
Results: logs/learned_nav/drills/<set>_<map>.jsonl (resumable: done starts are skipped).
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav.geometry import Geometry, VOID, DROP, WALL                 # noqa: E402
from nav.learned_nav.session import Session, RoundOver, R, rolling_spin          # noqa: E402
from nav.learned_nav import planner as PL                                       # noqa: E402
from nav.learned_nav import seeds as SEEDS                                      # noqa: E402
from nav.learned_nav.make_drill_map import DRILLS                               # noqa: E402
from nav.protocol import RAW_POS, RAW_VEL, RAW_SPIN, NOOP_ACTION                # noqa: E402
from nav.joystick import action_to_joystick                                     # noqa: E402
from nav.gems import visible_gems                                               # noqa: E402

OUT_DIR = os.path.join(HERE, 'logs', 'learned_nav', 'drills')
N_STARTS = {'dev': 30, 'test': 64, 'test5': 64,  # test: ~15 % of starts are unsafe (dev fixture), and the gate needs >= 200
                                               # safe; test5: stage 5's own frozen set (a new hash, never seen)
            'devb': 60, 'val': 30, 'test6': 64}  # 2026-09-29: devb = the larger tuning set, val = checked only for a
                                               # candidate that did well on devb (dev overfitted in stage 5), test6 = the next gate
D_MIN, D_MAX = 4.0, 18.0
CLEAR = 1.0
TIMEOUT_DEC = 188            # 12 s to take the gem
CONT = 8                     # decisions on the map after the landing (0.5 s)
EXTRA_DEC = 40               # room for the continuation after a late pickup
FIX_DEC = 47                 # fixture controls: 3 s
AIRBORNE_DZ = 0.5
LEVEL_DZ = 0.5               # centre this far below the start level: dropped off it


def gem_of(map_name):
    return np.array([float(v) for v in DRILLS[map_name].split()])


def floor_ref_of(g, gem):
    zs = []
    for a in np.linspace(-math.pi, math.pi, 36, endpoint=False):
        for r in (5.0, 6.0, 7.0):
            z = g.floor_below(gem[0] + r * math.cos(a), gem[1] + r * math.sin(a), gem[2] + 0.5)
            if z is not None:
                zs.append(z)
    return float(np.median(zs))


def make_starts(map_name, set_name, g):
    gem = gem_of(map_name); fref = floor_ref_of(g, gem)
    seed = int(hashlib.md5(f'drills:{map_name}:{set_name}'.encode()).hexdigest(), 16) % (2 ** 31)
    rng = np.random.default_rng(seed)
    out = []
    ang16 = np.linspace(-math.pi, math.pi, 16, endpoint=False)
    while len(out) < N_STARTS[set_name]:
        r = math.sqrt(rng.uniform(D_MIN ** 2, D_MAX ** 2)); a = rng.uniform(-math.pi, math.pi)
        x, y = gem[0] + r * math.cos(a), gem[1] + r * math.sin(a)
        z = g.floor_below(x, y, fref + 0.5)
        if z is None or abs(z - fref) > 0.3:
            continue
        n = g.normal_below(x, y, z + 0.01)
        if n is None or n[2] < 0.95:
            continue
        rays = g.edge_rays(x, y, z, ang16, max_u=CLEAR)
        hit = np.isfinite(rays[:, 0])
        if (hit & np.isin(rays[:, 1], (VOID, DROP))).any() or (hit & (rays[:, 1] == WALL) & (rays[:, 0] < 0.4)).any():
            continue
        tg = math.atan2(gem[1] - y, gem[0] - x)
        if rng.random() < 0.3:
            sp, h, cat = 0.0, tg, 'rest'
        else:
            sp = rng.uniform(2.0, 12.0)
            if rng.random() < 0.35:
                h = tg + math.radians(rng.uniform(-40, 40))
            else:
                h = rng.uniform(-math.pi, math.pi)
            off = abs((h - tg + math.pi) % (2 * math.pi) - math.pi)
            cat = 'toward' if off <= math.radians(45) else ('away' if off >= math.radians(135) else 'side')
        v = np.array([sp * math.cos(h), sp * math.sin(h), 0.0])
        out.append({'id': len(out), 'pos': [x, y, z + R + 0.01], 'vel': v.tolist(), 'spin': list(rolling_spin(v, n)),
                    'heading': h, 'speed': sp, 'dist': r, 'cat': cat, 'floor': z})
    return {'map': map_name, 'set': set_name, 'gem': gem.tolist(), 'floor_ref': fref, 'seed': seed, 'starts': out,
            'time': time.strftime('%Y-%m-%d %H:%M:%S')}


def starts_file(map_name, set_name):
    return os.path.join(OUT_DIR, f'starts_{set_name}_{map_name}.json')


class Drill:
    def __init__(self, port, map_name, log=print):
        self.map = map_name; self.log = log
        self.g = Geometry(map_name)
        self.gem = gem_of(map_name)
        self.s = Session(port, map_name, self.g, log=log)

    # -- game helpers
    def gem_present(self):
        for q in visible_gems(self.s.obs()):
            if math.hypot(q[0] - self.gem[0], q[1] - self.gem[1]) < 0.6 and abs(q[2] - self.gem[2]) < 0.6:
                return True
        return False

    def ensure_gem(self):
        if self.gem_present():
            return
        self.s.control('GEMRESET')
        for _ in range(30):
            self.s.step(NOOP_ACTION)
            if self.gem_present():
                return
        raise RuntimeError('target gem did not respawn')

    def place(self, st):
        self.s.ready()
        self.ensure_gem()
        return self.s.place(st['pos'], st['vel'], st['spin'], hold=NOOP_ACTION)

    def state(self, msg):
        ob = msg.obs
        p = np.asarray(ob[RAW_POS], np.float64); v = np.asarray(ob[RAW_VEL], np.float64); w = np.asarray(ob[RAW_SPIN], np.float64)
        tel = msg.extra
        lv = self.g.floor_below(p[0], p[1], p[2])
        gap = (p[2] - R - lv) if lv is not None else 9.0
        sup = bool(tel is not None and tel[2] > 0)
        last = bool(tel is not None and tel[12] > 0)
        return p, v, w, tel, gap, sup, last

    # -- fixture: can simple controls keep the marble on the map?
    def fixture(self, st):
        res = {}
        for name in ('brake', 'left', 'right'):
            o = self.place(st)
            if o is None:
                res[name] = 'bad_teleport'
                continue
            h = st['heading']; ok = True
            for i in range(FIX_DEC):
                v = np.asarray(self.s.obs()[RAW_VEL], np.float64)
                if name == 'brake' or i >= 20:
                    js = action_to_joystick(0.0, 0.0, 1.0, 0, 1, v[0], v[1])
                else:
                    a = h + (math.pi / 2 if name == 'left' else -math.pi / 2)
                    js = action_to_joystick(math.cos(a), math.sin(a), 1.0, 0, 0, v[0], v[1])
                msg, info = self.s.step(js)
                if info['fell']:
                    ok = False
                    break
            if ok:
                p, v, w, tel, gap, sup, last = self.state(self.s.env.msg)
                ok = gap < 1.5
            res[name] = 'safe' if ok else 'fell'
            if ok:
                break                                   # one safe control is enough
        return any(v == 'safe' for v in res.values()), res

    # -- one planner trial
    def trial(self, st, planner, fref):
        o = self.place(st)
        if o is None:
            return {'status': 'bad_teleport'}
        planner.reset()
        Uc, Jc = PL.reply_vector(NOOP_ACTION + (0.0,)); Up, Jp = Uc.copy(), Jc
        msg = self.s.env.msg
        picked = False; t_pick = None; landed_after = None; oob = False
        jumps = []; kinds = {}; ms = []; traj = []; airborne_pick = False; replies = []
        took_off = None; last_jump = None; no_takeoff = 0; bounces = 0
        status = None
        for i in range(TIMEOUT_DEC + EXTRA_DEC):
            if i >= TIMEOUT_DEC and not picked:
                break
            p, v, w, tel, gap, sup, last = self.state(msg)
            airborne = (not sup) and gap > AIRBORNE_DZ
            js, info = planner.plan(p, v, w, tel, Uc, Jc, Up, Jp, airborne, picked, fref)
            js = tuple(float(a) for a in js[:4]) + (int(js[4]), float(js[5]))
            replies.append(js)
            kinds[info['kind']] = kinds.get(info['kind'], 0) + 1; ms.append(info['ms'])
            if js[4] and not airborne:
                jumps.append({'i': i, 'p_succ': round(info.get('p_succ', 0.0), 3), 'dmin': round(info['dmin'], 3),
                              'flight_best': round(info.get('flight_best', -1.0), 3), 'kind': info['kind'],
                              'p': [round(float(a), 3) for a in p], 'v': [round(float(a), 3) for a in v],
                              'pred': info['pred_path'][:, :].tolist()[:30]})
                last_jump = i
            msg, st_info = self.s.step(js)
            Up, Jp = Uc, Jc
            Uc, Jc = PL.reply_vector(js)
            p2, v2, w2, tel2, gap2, sup2, last2 = self.state(msg)
            pp = info['pred_path']
            # per decision: actual position, jump key, plan kind, best P(success), chosen plan's safe / fell /
            # clearance / landing index, and its predicted position 1 and 5 decisions ahead (to compare with traj)
            traj.append([round(float(a), 3) for a in p2] + [int(js[4]), info['kind'][0], round(info.get('p_best', -1.0), 3),
                         int(info['safe']), int(info['fell']), round(info['clear'], 2), info['landed'],
                         [round(float(a), 3) for a in pp[1]], [round(float(a), 3) for a in pp[5]]])
            if last_jump is not None and took_off is None and not sup2 and gap2 > AIRBORNE_DZ:
                took_off = i
            if last_jump is not None and i - last_jump == 4 and (took_off is None or took_off < last_jump):
                no_takeoff += 1
            if st_info['gem_delta'] > 0 and not picked:
                picked = True; t_pick = i; airborne_pick = not sup2
                if jumps:
                    jumps[-1]['picked'] = True
            if st_info['fell']:
                oob = True
                break
            # the landing after the pickup: supported ON a floor (a lip hit is contact too, but not a landing); a
            # bounce back into the air starts the count again
            if picked and landed_after is None and sup2 and gap2 < 0.35 and (not airborne_pick or i > t_pick):
                landed_after = i
            elif picked and landed_after is not None and not sup2 and gap2 > AIRBORNE_DZ:
                landed_after = None; bounces += 1
            if picked and landed_after is not None and i - landed_after >= CONT:
                # P0's rule: on the map and over a floor; M3 also wants the start level back (a pocket under a
                # hole's lip is on the map but stuck)
                if gap2 >= 1.5:
                    status = 'picked_no_floor'
                elif p2[2] < fref + R - LEVEL_DZ:
                    status = 'picked_dropped'
                else:
                    status = 'success'
                break
        p_end = self.state(msg)[0]
        if status is None:
            if picked:
                status = 'picked_fell' if oob else 'picked_no_continuation'
            elif oob:
                status = 'fell_after_jump' if jumps else 'fell_rolling'
            elif p_end[2] < fref + R - LEVEL_DZ:
                status = 'dropped'
            else:
                status = 'missed' if jumps else 'abstain'
        return {'status': status, 'success': status == 'success', 'picked': picked, 't_pick': t_pick, 'oob': oob,
                'decisions': len(traj), 'jumps': jumps, 'n_jumps': len(jumps), 'no_takeoff': no_takeoff, 'kinds': kinds,
                'bounces_after_pickup': bounces, 'replies': replies,     # exact replies: a trial can be replayed
                'ms_mean': float(np.mean(ms)) if ms else 0.0, 'ms_p95': float(np.percentile(ms, 95)) if ms else 0.0,
                'traj': traj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--map', required=True)
    ap.add_argument('--set', required=True, choices=tuple(N_STARTS))
    ap.add_argument('--planner', default='planner')        # a planner module in nav/learned_nav (e.g. planner_v8)
    ap.add_argument('--tag', default='')                   # results go to <set>_<tag>_<map>.jsonl
    ap.add_argument('--limit', type=int, default=0)        # only the first N starts of the set (0 = all)
    a = ap.parse_args()
    global PL
    if a.planner != 'planner':
        import importlib
        PL = importlib.import_module(f'nav.learned_nav.{a.planner}')
    stem = f'{a.set}_{a.tag}_{a.map}' if a.tag else f'{a.set}_{a.map}'
    os.makedirs(OUT_DIR, exist_ok=True)
    log_f = open(os.path.join(OUT_DIR, f'{stem}.log'), 'a', buffering=1)

    def log(m):
        line = f'[{time.strftime("%H:%M:%S")}] {m}'
        print(line, flush=True); log_f.write(line + '\n')

    g = Geometry(a.map)
    sf = starts_file(a.map, a.set)
    if os.path.exists(sf):
        S = json.load(open(sf))
    else:
        S = make_starts(a.map, a.set, g)
        json.dump(S, open(sf, 'w'), indent=1)
    d = Drill(a.port, a.map, log=log)
    d.s.control('CONTACT 1')
    # fixture first, for every start, before any planner trial on this set
    if not all('safe' in st for st in S['starts']):
        log(f'fixture: {len(S["starts"])} starts')
        for st in S['starts']:
            if 'safe' in st:
                continue
            for _ in range(3):
                try:
                    st['safe'], st['fixture'] = d.fixture(st)
                    break
                except RoundOver:
                    continue
            json.dump(S, open(sf, 'w'), indent=1)
        log(f'fixture done: {sum(st["safe"] for st in S["starts"])} safe of {len(S["starts"])}')
    sha = hashlib.sha1(open(sf, 'rb').read()).hexdigest()[:12]
    log(f'starts {os.path.basename(sf)} sha1 {sha}')
    seeds = SEEDS.load(a.map, getattr(PL, 'SEED_VERSION', None))
    sha_of = lambda p: hashlib.sha1(open(p, 'rb').read()).hexdigest()[:12] if os.path.exists(p) else None
    prov = {'planner_py': sha_of(PL.__file__), 'drills_py': sha_of(__file__),
            'seeds_npz': sha_of(os.path.join(SEEDS.SEED_DIR, getattr(PL, 'SEED_VERSION', ''), f'{a.map}.npz')), 'use_guide': PL.USE_GUIDE,
            'p_go': PL.P_GO, 'p_jump': PL.P_JUMP, 'planner_module': a.planner}
    log(f'seeds: {0 if seeds is None else len(seeds["x"])}; provenance {json.dumps(prov)}')
    planner = PL.Planner(g, d.gem, seeds=seeds, rng=np.random.default_rng(12345))
    out = os.path.join(OUT_DIR, f'{stem}.jsonl')
    done = set()
    if os.path.exists(out):
        for line in open(out):
            done.add(json.loads(line)['id'])
    n_ok = n_run = 0
    for st in (S['starts'][:a.limit] if a.limit else S['starts']):
        if st['id'] in done:
            continue
        r = None
        for _ in range(3):
            try:
                r = d.trial(st, planner, S['floor_ref'])
                if r['status'] != 'bad_teleport':
                    break
            except RoundOver:
                continue
        if r is None:
            continue
        r.update({'id': st['id'], 'safe_start': st['safe'], 'cat': st['cat'], 'speed': st['speed'], 'dist': st['dist'],
                  'starts_sha1': sha, 'map': a.map, 'set': a.set, 'provenance': prov})
        with open(out, 'a') as f:
            f.write(json.dumps(r) + '\n')
        if 'success' not in r:                                  # the game refused the start three times: no trial
            log(f'start {st["id"]:3d}: teleport refused three times, skipped')
            continue
        n_run += 1; n_ok += int(r['success'])
        log(f'start {st["id"]:3d} ({st["cat"]}, {st["speed"]:.1f} u/s, {st["dist"]:.1f} u, safe {st["safe"]}): {r["status"]}'
            f' in {r["decisions"]} decisions, jumps {r["n_jumps"]}, plan {r["ms_mean"]:.0f} ms  [{n_ok}/{n_run}]')
    log('finished')


if __name__ == '__main__':
    main()
