"""Stage 3 recorder (design section 4, M1): short controlled trials on any verified practice map.

    python -m nav.learned_nav.record3 --port 9001 --map Sprawl_Hunt_phys --shard 0 --trials 20000
    (then: marbleblast_mbx.exe -autotrain Sprawl_Hunt_phys -aiport 9001)

A trial: place the marble in a sampled physical state (position on or above a surface, velocity along the surface,
spin), then fly a sampled control sequence on the deployed 64 ms decision grid, recording every decision until the
sequence ends: a jump runs to its landing plus CONT decisions; anything leaving the map ends at out of bounds.
The start state is the observation after the teleport (measured: exactly the requested state). Recorded per
decision: position, velocity, spin, the engine's contact telemetry (13 numbers), the reply exactly as the joystick
adapter built it (fwd, back, left, right, jump, camera yaw) and the intent behind it (world direction, throttle,
jump, brake), gem delta, out of bounds. Shards: datasets/learned_nav/stage3/<map>/shard_<k>_<n>.npz, each with a
provenance manifest. A map is recorded only if its geometry passed the in-game verification (verify_map).
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
from nav.learned_nav.geometry import Geometry, GEOM_DIR, VOID, DROP, WALL          # noqa: E402
from nav.learned_nav.session import Session, RoundOver, R, rolling_spin            # noqa: E402
from nav.protocol import RAW_POS, RAW_VEL, RAW_SPIN, NOOP_ACTION, CONTACT_FIELDS   # noqa: E402
from nav.joystick import action_to_joystick                                        # noqa: E402
from nav.env import OBS_MS                                                         # noqa: E402

DATA_DIR = os.path.join(HERE, 'datasets', 'learned_nav', 'stage3')
SCHEMA = 'stage3_trial_v1'
SHARD_TRIALS = 2000
MAX_DEC = 60              # 3.8 s cap on any trial
ROLL_DEC = (8, 24)        # rolling-family trial length range, decisions
CONT = 8                  # decisions after a landing
AIRBORNE_DZ = 0.5
FAMILIES = ('roll', 'switch', 'brake', 'none', 'jump', 'jump_switch')
FAMILY_P = (0.20, 0.15, 0.08, 0.07, 0.35, 0.15)
STEP_FIELDS = ('trial', 'i', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz') + CONTACT_FIELDS + \
              ('fwd', 'back', 'left', 'right', 'jump_key', 'cam_yaw', 'int_dx', 'int_dy', 'int_thr', 'int_jump', 'int_brake',
               'gem_delta', 'oob', 'floor_z')


def _sha(p):
    try:
        return hashlib.sha1(open(p, 'rb').read()).hexdigest()[:12]
    except OSError:
        return None


def provenance(map_name):
    pq = os.path.join(HERE, '..', 'Marble Blast Platinum')
    ai = os.path.join(pq, 'platinum', 'client', 'scripts', 'ai')
    gj = json.load(open(os.path.join(GEOM_DIR, f'{map_name}.json')))
    return {'schema': SCHEMA, 'map': map_name, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'engine_exe_sha1': _sha(os.path.join(pq, 'marbleblast_mbx.exe')),
            'mlAgent_cs_sha1': _sha(os.path.join(ai, 'mlAgent.cs')), 'observer_cs_sha1': _sha(os.path.join(ai, 'observer.cs')),
            'geometry': {'schema': gj['schema'], 'hashes': gj['hashes'], 'time': gj['time']},
            'record3_py_sha1': _sha(os.path.abspath(__file__)),
            'settings': {'obs_ms': OBS_MS, 'max_dec': MAX_DEC, 'roll_dec': ROLL_DEC, 'cont': CONT, 'families': FAMILIES,
                         'family_p': FAMILY_P, 'r': R}}


# ------------------------------------------------------------------ start states
class StartSampler:
    def __init__(self, g, rng):
        self.g = g; self.rng = rng
        top = np.isfinite(g.heights[0])
        # floor cells with room above (no second level within 2.5 u overhead)
        clear = ~(np.isfinite(g.heights[1]) & ((g.heights[0] - g.heights[1]) < 2.5))
        self.jj, self.ii = np.nonzero(top & clear)
        e = g.edges
        self.edges = e
        self.elen = np.hypot(e[:, 3] - e[:, 0], e[:, 4] - e[:, 1])

    def _surface(self, x, y, zhint):
        z = self.g.floor_below(x, y, zhint)
        if z is None:
            return None
        n = self.g.normal_below(x, y, z + 0.01)
        if n is None or n[2] < 0.55:                       # steeper than ~57 deg: skip
            return None
        return z, n

    def sample(self):
        rng = self.rng; g = self.g
        for _ in range(200):
            near_edge = rng.random() < 0.5 and len(self.edges)
            if near_edge:
                k = rng.choice(len(self.edges), p=self.elen / self.elen.sum())
                s = self.edges[k]
                u = rng.uniform(0.05, 0.95)
                p = s[0:3] + u * (s[3:6] - s[0:3])
                d_in = rng.uniform(0.3, 6.0)
                x, y = p[0] - s[6] * d_in, p[1] - s[7] * d_in
                zh = p[2] + 1.0
            else:
                k = rng.integers(len(self.ii))
                x = g.xs[self.ii[k]] + rng.uniform(-0.1, 0.1); y = g.ys[self.jj[k]] + rng.uniform(-0.1, 0.1)
                zh = float(g.heights[0, self.jj[k], self.ii[k]]) + 0.05
            sf = self._surface(x, y, zh)
            if sf is None:
                continue
            z, n = sf
            # heading: toward the edge's outward normal (within 70 deg) for most near-edge starts, else uniform
            if near_edge and rng.random() < 0.7:
                base = math.atan2(s[7], s[6]); h = base + math.radians(rng.uniform(-70, 70))
            else:
                h = rng.uniform(-math.pi, math.pi)
            speed = float(rng.choice([rng.uniform(0, 4), rng.uniform(4, 16)], p=[0.25, 0.75]))
            d = np.array([math.cos(h), math.sin(h), 0.0])
            d = d - n * np.dot(d, n); d /= max(1e-9, np.linalg.norm(d))          # along the surface
            airborne = rng.random() < 0.10
            if airborne:
                c = np.array([x, y, z]) + n * R + np.array([0, 0, rng.uniform(0.3, 3.0)])
                v = d * speed * rng.uniform(0.3, 1.0); v[2] += rng.uniform(-6, 6)
                w = np.asarray(rolling_spin(v * np.array([1, 1, 0]), (0, 0, 1))) * rng.uniform(0.0, 1.2)
                kind = 'air'
            else:
                c = np.array([x, y, z]) + n * (R + 0.01)
                v = d * speed
                w = np.asarray(rolling_spin(v, n))
                kind = 'floor'
                if rng.random() < 0.15:                                          # slipping: spin off the rolling value
                    w = w * rng.uniform(0.0, 1.5) + rng.normal(0, 8, 3)
                    kind = 'slip'
            # keep starts away from walls the marble would be placed into
            if kind != 'air' and np.isfinite(g.edge_rays(x, y, z, [0.0, math.pi / 2, math.pi, -math.pi / 2], max_u=0.35)[:, 0]).any():
                continue
            return {'pos': c.tolist(), 'vel': v.tolist(), 'spin': w.tolist(), 'heading': h, 'speed': speed,
                    'normal': n.tolist(), 'kind': kind, 'near_edge': bool(near_edge)}
        raise RuntimeError('could not sample a start')


# ------------------------------------------------------------------ control sequences
def sample_controls(rng):
    """A control program: family and parameters. Directions are relative to the start heading (radians)."""
    fam = str(rng.choice(FAMILIES, p=FAMILY_P))
    ang = lambda: math.radians(45.0 * rng.integers(8))
    p = {'family': fam, 'thr': float(rng.choice([1.0, 1.0, 1.0, 0.5])), 'len': int(rng.integers(*ROLL_DEC))}
    if fam == 'roll':
        p['dir'] = ang()
    elif fam == 'switch':
        p['dir'] = ang(); p['dir2'] = ang(); p['k'] = int(rng.integers(2, p['len'] - 2))
    elif fam in ('jump', 'jump_switch'):
        p['dir'] = 0.0 if rng.random() < 0.6 else ang()               # run-up
        p['j'] = int(rng.integers(0, 7))                               # jump decision
        p['air'] = None if rng.random() < 0.15 else ang()              # air input (None = no input)
        p['cont'] = str(rng.choice(['none', 'brake', 'air']))          # after landing
        if fam == 'jump_switch':
            p['air2'] = None if rng.random() < 0.2 else ang(); p['k2'] = int(rng.integers(2, 9))
    return p


def intent(p, i, h0, vel, airborne, landed_at):
    """(dx, dy, throttle, jump, brake) in world frame for decision i of program p."""
    def world(rel):
        a = h0 + rel
        return math.cos(a), math.sin(a)
    fam = p['family']; thr = p['thr']
    if fam == 'none':
        return (0.0, 0.0, 0.0, 0, 0)
    if fam == 'brake':
        return (0.0, 0.0, 1.0, 0, 1)
    if fam == 'roll':
        return (*world(p['dir']), thr, 0, 0)
    if fam == 'switch':
        return (*world(p['dir'] if i < p['k'] else p['dir2']), thr, 0, 0)
    # jumps
    j = p['j']
    if landed_at is not None:
        c = p['cont']
        if c == 'none':
            return (0.0, 0.0, 0.0, 0, 0)
        if c == 'brake':
            return (0.0, 0.0, 1.0, 0, 1)
        a = p['air'] if p['air'] is not None else p['dir']
        return (*world(a), thr, 0, 0)
    if i < j:
        return (*world(p['dir']), thr, 0, 0)
    air = p['air']
    if fam == 'jump_switch' and i >= j + p['k2']:
        air = p['air2']
    jump = 1 if i == j else 0
    if air is None:
        return (0.0, 0.0, 0.0, jump, 0)
    return (*world(air), 1.0, jump, 0)


# ------------------------------------------------------------------ the recorder
class Recorder3:
    def __init__(self, port, map_name, log=print):
        self.g = Geometry(map_name)
        self.map = map_name
        self.log = log
        self.s = Session(port, map_name, self.g, log=log)

    def trial(self, start, prog):
        s = self.s; g = self.g
        h0 = start['heading']
        v_start = np.asarray(start['vel'])
        dx0, dy0, thr0, j0, b0 = intent(prog, 0, h0, v_start, False, None)
        hold = action_to_joystick(dx0, dy0, thr0, 0, b0, v_start[0], v_start[1]) if (thr0 or b0) else NOOP_ACTION
        o = s.place(start['pos'], start['vel'], start['spin'], hold=hold)
        if o is None:
            return None
        rows = []
        airborne = start['kind'] == 'air'; took_off = 0 if airborne else None; landed = None; oob = False
        vel = np.asarray(o[RAW_VEL], dtype=np.float64)
        n_dec = MAX_DEC if prog['family'] in ('jump', 'jump_switch') or airborne else prog['len']
        for i in range(n_dec):
            it = intent(prog, i, h0, vel, airborne, landed)
            dx, dy, thr, jmp, brk = it
            js = action_to_joystick(dx, dy, thr, jmp, brk, vel[0], vel[1]) if (thr > 0 or brk or jmp) else NOOP_ACTION
            if len(js) == 5:
                js = tuple(js) + (0.0,)
            msg, info = s.step(js)
            ob = msg.obs
            p = np.asarray(ob[RAW_POS], dtype=np.float64); vel = np.asarray(ob[RAW_VEL], dtype=np.float64)
            w = ob[RAW_SPIN]
            tel = msg.extra if msg.extra is not None else np.full(len(CONTACT_FIELDS), np.nan)
            fz = g.floor_below(p[0], p[1], p[2])
            fell = bool(info['fell'])
            if rows and np.linalg.norm(p - np.asarray(rows[-1][2:5])) > 2.5:
                s.oob_pending = True                                    # a respawn landed mid-trial: discard it
                return None
            rows.append([0, i, *p, *vel, *w, *tel, *[float(v) for v in js[:6]], dx, dy, thr, jmp, brk,
                         float(info['gem_delta']), int(fell), np.nan if fz is None else fz])
            support = tel[2] > 0 if np.isfinite(tel[2]) else False
            if not airborne and not support and (fz is None or p[2] - fz > AIRBORNE_DZ):
                airborne = True; took_off = i
            elif airborne and landed is None and support and tel[12] > 0:
                landed = i
            if fell:
                oob = True
                break
            if landed is not None and i - landed >= CONT and (prog['family'] in ('jump', 'jump_switch') or start['kind'] == 'air'):
                break
            if prog['family'] in ('jump', 'jump_switch') and took_off is None and i >= prog['j'] + 10:
                break                                                   # the jump never left the floor
        meta = {'map': self.map, 'start': start, 'prog': prog, 'h0': h0, 'n': len(rows), 'oob': oob,
                'took_off': took_off, 'landed': landed,
                'censored': bool(airborne and landed is None and not oob and len(rows) >= MAX_DEC)}
        return meta, np.asarray(rows, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--map', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--trials', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=None)
    a = ap.parse_args()
    ver = os.path.join(GEOM_DIR, f'{a.map}.verify.json')
    if not (os.path.exists(ver) and json.load(open(ver)).get('pass')):
        raise SystemExit(f'{a.map}: geometry not verified in the game (run verify_map first)')
    out_dir = os.path.join(DATA_DIR, a.map); os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, f'shard_{a.shard}.log'), 'a', buffering=1)

    def log(m):
        line = f'[{time.strftime("%H:%M:%S")}] {m}'
        print(line, flush=True); log_f.write(line + '\n')

    done = sum(int(f.split('_')[-1].split('.')[0]) for f in os.listdir(out_dir)
               if f.startswith(f'shard_{a.shard}_') and f.endswith('.npz'))
    seed = a.seed if a.seed is not None else int(hashlib.md5(f'{a.map}:{a.shard}:{done}'.encode()).hexdigest(), 16) % (2 ** 31)
    rng = np.random.default_rng(seed)
    rec = Recorder3(a.port, a.map, log=log)
    sampler = StartSampler(rec.g, rng)
    man = provenance(a.map); man['shard'] = a.shard; man['seed'] = seed
    metas, chunks = [], []
    part = len([f for f in os.listdir(out_dir) if f.startswith(f'shard_{a.shard}_') and f.endswith('.npz')])
    t0 = time.time(); n = done; fails = 0
    while n < a.trials:
        start = sampler.sample(); prog = sample_controls(rng)
        try:
            r = rec.trial(start, prog)
        except RoundOver:
            continue
        except RuntimeError as e:
            log(f'trial failed: {e}'); fails += 1
            if fails > 50:
                raise
            continue
        if r is None:
            fails += 1
            continue
        meta, rows = r
        rows[:, 0] = len(metas)
        metas.append(meta); chunks.append(rows); n += 1
        if len(metas) >= SHARD_TRIALS or n >= a.trials:
            fn = os.path.join(out_dir, f'shard_{a.shard}_{part:03d}_{len(metas)}.npz')
            with open(fn + '.partial', 'wb') as f:
                np.savez_compressed(f, steps=np.concatenate(chunks), fields=np.array(STEP_FIELDS),
                                    meta=np.array(json.dumps(metas)), manifest=np.array(json.dumps(man)))
            os.replace(fn + '.partial', fn)
            part += 1
            el = time.time() - t0
            log(f'{n} trials ({(n - done) / el:.1f}/s), wrote {os.path.basename(fn)}; teleport retries {fails}')
            metas, chunks = [], []
    log(f'finished {a.map} shard {a.shard}: {n} trials')


if __name__ == '__main__':
    main()
