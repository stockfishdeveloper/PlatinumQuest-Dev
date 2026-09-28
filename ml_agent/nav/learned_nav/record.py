"""P0a flight recorder (KOTMJUMP_START_HERE.md): jump trials at the kotmjump_p0 drill hole, recorded from the game.

    python -m nav.learned_nav.record plan --starts 1400 --shards 6          # sample starts, write trial shards
    python -m nav.learned_nav.record run --port 8941 --shard 0              # run one shard against one game
    python -m nav.learned_nav.record repeat --port 8941                     # the 20-repeat check

Start Python first (it binds the port), then the game:  marbleblast_mbx.exe -autotrain kotmjump_p0 -aiport <port>

A trial:
1. hold the run-up input (full throttle along the requested heading) for one decision, so the held input is
   right when the teleport lands (a control word leaves the previous input in force);
2. TELEPORT onto the floor with the requested velocity and ROLLING spin (w = (-vy, vx, 0) / r), then SETTLE more
   run-up decisions. The observation after that is the trial's START: the recorded state (measured: exactly the
   teleported state; the first physics step comes after it). Requested values are kept as metadata only;
3. the candidate's controls, one reply per 64 ms decision: run-up until the jump decision, the jump press with
   the air input, the air input held until landing, then NO input for CONT_STEPS decisions. A reply sent at
   decision i acts on the physics between decisions i+1 and i+2 (bridge delay, mlAgent.cs with ActionDelay 0);
   every recorded step carries its decision index and the reply exactly as serialized;
4. outcomes from the game: pickup (score change; the drill map has only the target gem), out of bounds, and
   takeoff / landing from the recorded trajectory against the 0.1 u floor raster.
Candidates: index 0 = no jump (run-up held); 1 + 9 j + a = jump at decision j (0..4) holding air input a
(0 none, 1..8 = heading + (a-1) x 45 deg, counter-clockwise). The same 46 for every arm.
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
from nav.env import HuntEnv, OBS_MS, TRAINING_MODE                         # noqa: E402
from nav.protocol import RAW_POS, RAW_VEL, RAW_SPIN, NOOP_ACTION, format_teleport, format_action  # noqa: E402
from nav.joystick import action_to_joystick                                  # noqa: E402
from nav.gems import visible_gems                                            # noqa: E402

MAP = 'kotmjump_p0'
TARGET = (-30.25, 10.05, 21.7)
FLOOR_Z = 20.65
# The target hole on the 0.1 u raster (checked in the game 2026-09-27, probe_drill): x -33.75..-26.78,
# y 6.58..11.58, plus a northern bay on the west half up to y 13.58. The 0.5 u grid is 0.3 u off at the south lip.
HOLE_BOX = (-33.8, -26.75, 6.55, 13.6)
R = 0.19                               # marble radius (measured 2026-09-26)
SETTLE = 2
JUMP_TICKS = (0, 1, 2, 3, 4)
N_AIR = 9
MAX_STEPS = 45                         # 2.9 s: the window of a trial that never leaves the floor, and the predicted path
MAX_TOTAL = 75                         # 4.8 s: hard cap; a flight still undecided here is CENSORED (counted unsafe)
CONT_STEPS = 8                         # 0.5 s of continuation after landing
AIRBORNE_DZ = 0.5                      # centre this far above the floor under it = airborne
LAND_DZ = 0.35                         # back within this of a floor while not rising fast = landed
SCHEMA = 'p0_trial_v2'                 # v2: flights run to landing + 0.5 s (cap MAX_TOTAL), censored flag, end_over_floor
DATA_DIR = os.path.join(HERE, 'datasets', 'learned_nav', 'p0')
HELD_OUT_MOD = 4                       # one region x heading cell in four is held out
BLOCK_U = 3.0                          # split cells: 3 u position blocks ...
HBIN_DEG = 20.0                        # ... x 20 deg bins of heading relative to the direction to the gem
GUARD_U = 0.25                         # starts this close to a block edge ...
GUARD_DEG = 2.0                        # ... or to a heading-bin edge are not used (keeps neighbours apart)


def candidates():
    return [None] + [(j, a) for j in JUMP_TICKS for a in range(N_AIR)]


CANDS = candidates()


def cand_index(j, a):
    return 1 + j * N_AIR + a


# ------------------------------------------------------------------ geometry (0.1 u raster)
_FINE = None


def fine():
    global _FINE
    if _FINE is None:
        from nav.fine_terrain import get
        _FINE = get(MAP)
    return _FINE


def floor_below(x, y, z, fz=None):
    """Highest floor level at (x, y) not above z + 0.3, or None."""
    h = (fz or fine()).heights_at([x], [y])[:, 0]
    h = h[np.isfinite(h) & (h <= z + 0.3)]
    return float(h.max()) if len(h) else None


def lip_along(x, y, hx, hy, max_u=16.0, step=0.05):
    """Distance along (hx, hy) to the first point with no floor at the start's level, or inf."""
    ds = np.arange(1, int(max_u / step) + 1) * step
    h = fine().heights_at(x + hx * ds, y + hy * ds)
    ok = np.any(np.isfinite(h) & (np.abs(h - FLOOR_Z) < 0.3), axis=0)
    bad = np.nonzero(~ok)[0]
    return float(ds[bad[0]]) if len(bad) else math.inf


def in_hole_box(x, y):
    return HOLE_BOX[0] < x < HOLE_BOX[1] and HOLE_BOX[2] < y < HOLE_BOX[3]


# ------------------------------------------------------------------ starts and the split
def rel_heading(x, y, heading):
    to_gem = math.atan2(TARGET[1] - y, TARGET[0] - x)
    return (heading - to_gem + math.pi) % (2 * math.pi) - math.pi


def split_cell(x, y, heading):
    bx = (x - TARGET[0]) / BLOCK_U; by = (y - TARGET[1]) / BLOCK_U
    hb = math.degrees(rel_heading(x, y, heading)) / HBIN_DEG
    return bx, by, hb


def held_out_of(x, y, heading):
    """(held_out, guarded). The split is BY REGION AND HEADING, never by trial: 3 u position blocks x 20 deg
    heading bins (heading relative to the direction to the gem), one cell in four held out (md5 of the cell).
    guarded = within GUARD_U / GUARD_DEG of a cell edge: such starts are skipped, so a training start and a
    held-out start in different cells are at least 0.5 u or 4 deg apart."""
    bx, by, hb = split_cell(x, y, heading)
    key = f'{math.floor(bx)},{math.floor(by)},{math.floor(hb)}'.encode()
    ho = int(hashlib.md5(key).hexdigest(), 16) % HELD_OUT_MOD == 0
    fx = bx - math.floor(bx); fy = by - math.floor(by); fh = hb - math.floor(hb)
    gu = GUARD_U / BLOCK_U; gd = GUARD_DEG / HBIN_DEG
    guarded = min(fx, 1 - fx) < gu or min(fy, 1 - fy) < gu or min(fh, 1 - fh) < gd
    return ho, guarded


def sample_starts(n, seed, speed_min=0.0):
    """Rolling starts on the floor around the target hole: 0-15 u/s, heading within 60 deg of the direction
    to the gem, the lip along the heading 0.3-6 u ahead, and the
    first void along the heading is the target hole. Requested values; the recorded start is what the game did."""
    rng = np.random.default_rng(seed)
    out = []; tries = 0
    while len(out) < n:
        tries += 1
        x = rng.uniform(TARGET[0] - 11, TARGET[0] + 11)
        y = rng.uniform(TARGET[1] - 11, TARGET[1] + 11)
        if in_hole_box(x, y):
            continue
        fb = floor_below(x, y, 21.0)
        if fb is None or abs(fb - FLOOR_Z) > 0.05:
            continue
        clear = True
        for ang in range(0, 360, 45):
            a = math.radians(ang)
            if floor_below(x + 0.6 * math.cos(a), y + 0.6 * math.sin(a), 21.0) is None:
                clear = False
                break
        if not clear:
            continue
        to_gem = math.atan2(TARGET[1] - y, TARGET[0] - x)
        heading = to_gem + math.radians(rng.uniform(-60, 60))
        heading = (heading + math.pi) % (2 * math.pi) - math.pi
        ho, guarded = held_out_of(x, y, heading)
        if guarded:
            continue
        hx, hy = math.cos(heading), math.sin(heading)
        speed = float(rng.uniform(speed_min, 15.0))
        lip = lip_along(x, y, hx, hy)
        if not (math.isfinite(lip) and 0.3 <= lip <= 6.0):
            continue
        lx, ly = x + hx * (lip + 0.2), y + hy * (lip + 0.2)
        if not in_hole_box(lx, ly):
            continue
        out.append({'id': f's{seed}_{len(out):05d}', 'x': round(x, 3), 'y': round(y, 3), 'heading': round(heading, 4),
                    'speed': round(speed, 3), 'lip_u': round(lip, 3), 'held_out': ho})
    return out


# ------------------------------------------------------------------ provenance
def _sha(path):
    try:
        return hashlib.sha1(open(path, 'rb').read()).hexdigest()[:12]
    except OSError:
        return None


def provenance():
    pq = os.path.join(HERE, '..', 'Marble Blast Platinum')
    exe = os.path.join(pq, 'marbleblast_mbx.exe')
    return {'schema': 'p0_manifest_v1', 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'map': MAP,
            'engine_exe_sha1': _sha(exe),
            'engine_exe_mtime': time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(exe))) if os.path.exists(exe) else None,
            'mlAgent_cs_sha1': _sha(os.path.join(pq, 'platinum', 'client', 'scripts', 'ai', 'mlAgent.cs')),
            'observer_cs_sha1': _sha(os.path.join(pq, 'platinum', 'client', 'scripts', 'ai', 'observer.cs')),
            'mission_sha1': _sha(os.path.join(pq, 'platinum', 'data', 'multiplayer', 'hunt', 'custom', MAP + '.mcs')),
            'record_py_sha1': _sha(os.path.abspath(__file__)),
            'settings': {'obs_ms': OBS_MS, 'training_mode': TRAINING_MODE, 'settle': SETTLE, 'max_steps': MAX_STEPS,
                         'max_total': MAX_TOTAL,
                         'cont_steps': CONT_STEPS, 'jump_ticks': list(JUMP_TICKS), 'n_air': N_AIR,
                         'airborne_dz': AIRBORNE_DZ, 'land_dz': LAND_DZ, 'r': R, 'target': TARGET}}


# ------------------------------------------------------------------ the game side
class Recorder:
    def __init__(self, port, log=print):
        self.log = log
        self.env = HuntEnv(port, speed=3, log=log)
        self.env.connect()
        self.round_s = float(self.env.info.get('round_ms', 0.0)) / 1000.0
        mission = self.env.info.get('mission', '')
        if mission != MAP:
            raise SystemExit(f'the game is on "{mission}", not {MAP}')
        self.log(f'connected: {mission}, round {self.round_s:.0f} s')
        from terrain_obs import TerrainMap
        self.coarse = TerrainMap(TerrainMap.resolve(MAP))

    def obs(self):
        return self.env.msg.obs

    def step(self, js):
        msg, info = self.env.step(js, repeat=1)
        if info['round_ended'] or info['reconnected']:
            self.log('round ended / reconnected: waiting for the next round')
            self.env.wait_new_round()
            raise RuntimeError('round_over')
        return msg, info

    def control(self, word):
        self.env.control(word)
        if self.env.round_ended or self.env.reconnected:
            self.env.wait_new_round()
            raise RuntimeError('round_over')

    def in_countdown(self):
        return self.round_s > 0 and self.env.time_left_s() >= self.round_s - 0.5

    def gem_present(self):
        for g in visible_gems(self.obs()):
            if math.hypot(g[0] - TARGET[0], g[1] - TARGET[1]) < 0.6 and abs(g[2] - TARGET[2]) < 0.6:
                return True
        return False

    def on_map(self):
        x, y, z = (float(v) for v in self.obs()[RAW_POS])
        return self.coarse.contains(x, y) and z > 19.0

    def ready(self):
        """Countdown over, marble on the map, target gem present."""
        n = 0
        while self.in_countdown():
            self.step(NOOP_ACTION)
            n += 1
            if n > 400:
                raise RuntimeError('countdown did not end')
        if not self.on_map():
            self.control('OOBCLICK')
            for i in range(200):
                self.step(NOOP_ACTION)
                if self.on_map() and i >= 20:
                    break
                if i in (60, 120):
                    self.control('RESPAWN')
            if not self.on_map():
                raise RuntimeError('marble could not be put back on the map')
        if not self.gem_present():
            self.control('GEMRESET')
            for _ in range(30):
                self.step(NOOP_ACTION)
                if self.gem_present():
                    break
            if not self.gem_present():
                raise RuntimeError('target gem did not respawn')

    def run(self, start, ci):
        cand = CANDS[ci]
        self.ready()
        x, y, h, sp = start['x'], start['y'], start['heading'], start['speed']
        hx, hy = math.cos(h), math.sin(h)
        vx, vy = sp * hx, sp * hy
        run_up = action_to_joystick(hx, hy, 1.0, 0, 0, vx, vy)
        self.step(run_up)                                     # the held input when the teleport lands
        fz = floor_below(x, y, 21.0)
        self.control(format_teleport(x, y, fz + R + 0.01, vx, vy, 0.0, spin=(-vy / R, vx / R, 0.0)))
        for _ in range(SETTLE):
            self.step(run_up)
        o = self.obs()
        p0 = [float(v) for v in o[RAW_POS]]; v0 = [float(v) for v in o[RAW_VEL]]; w0 = [float(v) for v in o[RAW_SPIN]]
        # Measured 2026-09-27 (repeat check): the observation after the settle decisions is the teleported state
        # itself (position, velocity and spin exactly as requested); the first physics step follows it. So the
        # check is tight, and it also catches a teleport the game ignored (shortly after a respawn).
        speed0 = math.hypot(v0[0], v0[1])
        if (math.hypot(p0[0] - x, p0[1] - y) > 0.05 or abs(p0[2] - (fz + R + 0.01)) > 0.05
                or math.hypot(v0[0] - vx, v0[1] - vy) > 0.1):
            return {'schema': SCHEMA, 'start_id': start['id'], 'cand': ci, 'status': 'bad_teleport', 'p0': p0, 'v0': v0}
        # the frame: the actual heading at the start; below 1 u/s, the run-up direction
        h0 = math.atan2(v0[1], v0[0]) if speed0 >= 1.0 else h
        c0, s0 = math.cos(h0), math.sin(h0)
        run = action_to_joystick(c0, s0, 1.0, 0, 0, v0[0], v0[1])

        def air(a, jump):
            if a == 0:
                return action_to_joystick(0.0, 0.0, 0.0, jump, 0, 0.0, 0.0)
            ang = h0 + math.radians((a - 1) * 45.0)
            return action_to_joystick(math.cos(ang), math.sin(ang), 1.0, jump, 0, 0.0, 0.0)

        steps = []
        airborne = False; took_off = None; landed = None; oob = False; picked = 0.0; t_pick = None
        for i in range(MAX_TOTAL):
            if cand is None:
                js = run if landed is None else NOOP_ACTION
            else:
                j, a = cand
                if landed is not None:
                    js = NOOP_ACTION
                elif i < j:
                    js = run
                elif i == j:
                    js = air(a, 1)
                else:
                    js = air(a, 0)
            wire = format_action(*js)
            msg, info = self.step(js)
            ob = msg.obs
            px, py, pz = (float(v) for v in ob[RAW_POS])
            vz = float(ob[RAW_VEL][2])
            fb = floor_below(px, py, pz)
            gd = float(info['gem_delta'])
            if gd > 0 and t_pick is None:
                t_pick = i
            picked += gd
            fell = bool(info['fell'])
            steps.append({'i': i, 'a': wire, 'p': [round(px, 4), round(py, 4), round(pz, 4)],
                          'v': [round(float(v), 4) for v in ob[RAW_VEL]], 'w': [round(float(v), 3) for v in ob[RAW_SPIN]],
                          'fz': None if fb is None else round(fb, 3), 'gd': gd, 'oob': int(fell), 'tick': msg.tick})
            above = (pz - fb) if fb is not None else 99.0
            if not airborne and above > AIRBORNE_DZ:
                airborne = True; took_off = i
            if airborne and landed is None and fb is not None and above < R + LAND_DZ and vz < 2.5:
                landed = i
            if fell:
                oob = True
                break
            if landed is not None and i - landed >= CONT_STEPS:
                break
            if took_off is None and i + 1 >= MAX_STEPS:
                break
        # The trial must end in a decided state. Edge hits are real (2026-09-27 repeat check: a marble clipping the far
        # edge of a hole left it at vz +11 to +12.7 u/s, more than a jump), so a flight can go on well past the first
        # contact: it runs until it has landed and rolled CONT_STEPS decisions, or out of bounds, or the cap.
        last = steps[-1]
        censored = (not oob) and ((took_off is not None and landed is None)
                                  or (landed is not None and last['i'] - landed < CONT_STEPS))
        end_over_floor = last['fz'] is not None and last['p'][2] - last['fz'] < 1.5
        rec = {'schema': SCHEMA, 'status': 'ok', 'start_id': start['id'], 'held_out': start['held_out'], 'cand': ci,
               'jump_tick': None if cand is None else cand[0], 'air': None if cand is None else cand[1],
               'req': {k: start[k] for k in ('x', 'y', 'heading', 'speed')},
               'p0': [round(v, 4) for v in p0], 'v0': [round(v, 4) for v in v0], 'w0': [round(v, 3) for v in w0],
               'h0': round(h0, 5), 'steps': steps, 'pickup': picked > 0, 't_pick': t_pick, 'oob': oob,
               'took_off': took_off, 'landed': landed,
               'land_p': steps[landed]['p'] if landed is not None else None,
               'land_v': steps[landed]['v'] if landed is not None else None}
        # safe: never out of bounds, not censored, a flight that started landed (then 0.5 s without falling),
        # and the marble ends over floor
        rec['censored'] = censored
        rec['end_over_floor'] = end_over_floor
        rec['safe'] = (not oob) and (not censored) and (took_off is None or landed is not None) and end_over_floor
        rec['success'] = bool(rec['pickup'] and rec['safe'])
        return rec


def run_list(port, todo, out_path, log, tag=''):
    rec = Recorder(port, log=log)
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get('status') == 'ok':
                done.add((r['start_id'], r['cand'], r.get('rep', 0)))
    out = open(out_path, 'a', buffering=1)
    if not done:
        man = provenance(); man['tag'] = tag; man['port'] = port; man['n_todo'] = len(todo)
        out.write(json.dumps(man) + '\n')
    t0 = time.perf_counter(); n = 0; bad = 0
    for item in todo:
        s, c = item[0], item[1]
        rep = item[2] if len(item) > 2 else 0
        if (s['id'], c, rep) in done:
            continue
        r = None
        for attempt in range(4):
            try:
                r = rec.run(s, c)
            except RuntimeError as e:
                log(f'trial {s["id"]}/{c} attempt {attempt}: {e}')
                r = None
                continue
            if r['status'] == 'ok':
                break
            bad += 1
        if r is None or r['status'] != 'ok':
            log(f'trial {s["id"]}/{c}: gave up ({None if r is None else r["status"]})')
            if r is not None:
                out.write(json.dumps(r) + '\n')
            continue
        r['rep'] = rep
        out.write(json.dumps(r) + '\n'); n += 1
        if n % 200 == 0:
            el = time.perf_counter() - t0
            log(f'{n} trials in {el:.0f} s ({n / el:.2f}/s), {bad} teleport retries')
    log(f'finished: {n} trials in {time.perf_counter() - t0:.0f} s, {bad} teleport retries')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['plan', 'run', 'repeat'])
    ap.add_argument('--port', type=int)
    ap.add_argument('--shard', type=int)
    ap.add_argument('--starts', type=int, default=1400)
    ap.add_argument('--shards', type=int, default=6)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--speed-min', type=float, default=0.0)
    ap.add_argument('--batch', default='', help="batch name: '' = the first batch (shard_k / trials_k), 'b' = shard_bk / trials_bk")
    ap.add_argument('--list', help='run: a JSON list of [start, cand] instead of a shard')
    ap.add_argument('--out', help='run: output path instead of the shard default')
    a = ap.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    if a.cmd == 'plan':
        starts = sample_starts(a.starts, a.seed, a.speed_min)
        json.dump(starts, open(os.path.join(DATA_DIR, f'starts{a.batch}.json'), 'w'))
        ho = sum(s['held_out'] for s in starts)
        todo = [[s, c] for s in starts for c in range(len(CANDS))]
        rng = np.random.default_rng(a.seed)
        order = rng.permutation(len(todo))
        for k in range(a.shards):
            part = [todo[i] for i in order[k::a.shards]]
            # keep a start's candidates together where possible: sort the shard by start, then candidate
            part.sort(key=lambda t: (t[0]['id'], t[1]))
            json.dump(part, open(os.path.join(DATA_DIR, f'shard_{a.batch}{k}.json'), 'w'))
        print(f'{len(starts)} starts ({ho} held out), {len(todo)} trials in {a.shards} shards -> {DATA_DIR}')
        return
    out_path = a.out or (os.path.join(DATA_DIR, 'repeat.jsonl') if a.cmd == 'repeat'
                         else os.path.join(DATA_DIR, f'trials_{a.batch}{a.shard}.jsonl'))
    log_f = open(out_path + '.log', 'a', buffering=1)

    def log(m):
        line = f'[{time.strftime("%H:%M:%S")}] {m}'
        print(line, flush=True); log_f.write(line + '\n')

    if a.cmd == 'repeat':
        starts = sample_starts(6, 777)[:4]
        todo = []
        for s in starts:
            for c in (0, cand_index(2, 1), cand_index(0, 1)):    # no jump; jump at 2 hold forward; jump at 0 hold forward
                for rep in range(20):
                    todo.append([s, c, rep])
        run_list(a.port, todo, out_path, log, tag='repeat')
    else:
        todo = json.load(open(a.list)) if a.list else json.load(open(os.path.join(DATA_DIR, f'shard_{a.batch}{a.shard}.json')))
        run_list(a.port, todo, out_path, log, tag=f'shard {a.shard}' if a.list is None else os.path.basename(a.list))


if __name__ == '__main__':
    main()
