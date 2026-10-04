"""Powerup bridge check and measurements (POWERUP_PLAN phases 1-2, 2026-10-02).

    python -m nav.learned_nav.powdrill --port 9671 --map KingOfTheMarble_Hunt --test bridge
    python -m nav.learned_nav.powdrill --port 9671 --map KingOfTheMarble_Hunt --test blast

'bridge': read the V7 powerup block, drive over a Super Speed and a Super Jump, read the held type back, fire the
Super Speed along a chosen yaw and the Super Jump, print the velocity before and after; fire the regular blast when
the meter allows. 'blast': sample the blast meter from BLAST_REQUIRED to 1 in steps and record the vertical speed
each blast gives (the height curve), then the picked-up Blast.
Results: logs/learned_nav/powerups/<test>_<map>.jsonl
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
from nav.learned_nav.session import Session, R                                   # noqa: E402
from nav.protocol import (RAW_POS, RAW_VEL, RAW_POW_HELD, RAW_POW_BLAST, RAW_POW_SPECIAL, RAW_POW_MEGA,   # noqa: E402
                          RAW_POW_MEGA_LEFT, RAW_POW_ITEMS, RAW_POW_BOUNCE_LEFT, RAW_POW_SHOCK_LEFT, RAW_POW_HELI_LEFT,
                          POW_NAMES, BLAST_REQUIRED, format_action, NOOP_ACTION)
from nav.joystick import action_to_joystick                                      # noqa: E402

OUT_DIR = os.path.join(HERE, 'logs', 'learned_nav', 'powerups')
GEOMETRY = {'KingOfTheMarble_Hunt': 'KingOfTheMarble_Hunt', 'kotmjump_p0': 'KingOfTheMarble_Hunt',
            'ParkourPeaks_Hunt': 'ParkourPeaks_Hunt_phys', 'Promontory_Hunt': 'Promontory_Hunt'}
MCS_DIRS = os.path.join(HERE, '..', 'Marble Blast Platinum', 'platinum', 'data', 'multiplayer', 'hunt')
ITEM_TYPES = {'SuperJumpItem': 'super_jump', 'SuperSpeedItem': 'super_speed', 'SuperBounceItem': 'super_bounce',
              'ShockAbsorberItem': 'shock_absorber', 'HelicopterItem': 'helicopter', 'MegaMarbleItem': 'mega', 'BlastItem': 'blast'}


def mission_items(map_name):
    """{type: [(x, y, z), ...]} from the mission file: each `new Item(...) { ... }` block's dataBlock and position,
    whatever their order inside the block."""
    import glob, re
    out = {}
    for f in glob.glob(os.path.join(MCS_DIRS, '*', map_name + '.mcs')):
        txt = open(f, encoding='utf-8', errors='replace').read()
        for blk in re.finditer(r'new Item\s*\([^)]*\)\s*\{(.*?)\};', txt, re.S):
            body = blk.group(1)
            db = re.search(r'dataBlock\s*=\s*"([A-Za-z]+Item)(?:_MBU|_PQ)?"', body)
            ps = re.search(r'position\s*=\s*"([-\d.e]+) ([-\d.e]+) ([-\d.e]+)"', body)
            if db and ps and ITEM_TYPES.get(db.group(1)):
                out.setdefault(ITEM_TYPES[db.group(1)], []).append((float(ps.group(1)), float(ps.group(2)), float(ps.group(3))))
    return out
KOTM_ITEMS = {'super_speed': (-31.2, 3.0), 'super_jump': (-19.2, 3.0), 'blast': (-31.2, 27.0), 'mega': (-25.2, 15.0)}   # KingOfTheMarble_Hunt.mcs


def pow_block(ob):
    items = np.asarray(ob[RAW_POW_ITEMS], float).reshape(3, 5)
    return {'held': POW_NAMES[int(ob[RAW_POW_HELD])] if 0 <= int(ob[RAW_POW_HELD]) <= 7 else int(ob[RAW_POW_HELD]),
            'blast': round(float(ob[RAW_POW_BLAST]), 3), 'special': int(ob[RAW_POW_SPECIAL]),
            'mega': int(ob[RAW_POW_MEGA]), 'mega_left': round(float(ob[RAW_POW_MEGA_LEFT]), 2),
            'items': [(POW_NAMES[int(t)] if 1 <= int(t) <= 7 else None, round(x, 1), round(y, 1), round(z, 1), round(l, 1))
                      for t, x, y, z, l in items if t > -500]}


def yaw_for(dx, dy):
    """Camera yaw that faces world direction (dx, dy): forward = (sin yaw, cos yaw) (observer.cs)."""
    return math.atan2(dx, dy)


def hold(s, js, n):
    last = None
    for _ in range(n):
        last, info = s.step(js)
        if info['fell']:
            s.recover(); break
    return last


class Drill:
    def __init__(self, port, map_name, log):
        try:
            self.g = Geometry(GEOMETRY.get(map_name, map_name))
        except Exception as e:                                   # no geometry export: teleports use the item heights
            log(f'no geometry for {map_name} ({e}); using mission item heights'); self.g = None
        self.s = Session(port, map_name, self.g, log=log)
        self.log = log
        self.map = map_name
        self.items = mission_items(map_name)
        log('mission items: ' + ', '.join(f'{k} x{len(v)}' for k, v in self.items.items()))
        os.makedirs(OUT_DIR, exist_ok=True)

    def floor(self, x, y, default):
        if self.g is None:
            return default
        z = self.g.floor_below(x, y, default + 5.0)
        return z if z is not None else default

    def obs(self):
        return self.s.obs()

    def raw(self, js, use_pow=0, pow_yaw=None, use_blast=0):
        """One decision with the raw bridge words (the session's step() carries only the joystick)."""
        env = self.s.env
        env._send(format_action(*js, use_pow=use_pow, tick=env.msg.tick, pow_yaw=pow_yaw, use_blast=use_blast))
        m = env._recv_obs(); env.msg = m
        return m.obs

    def drive_to(self, x, y, max_dec=80):
        """Roll to within 0.6 u of (x, y) on the floor (full throttle toward it, braking near)."""
        for _ in range(max_dec):
            ob = self.obs(); p = ob[RAW_POS]; v = ob[RAW_VEL]
            dx, dy = x - p[0], y - p[1]; d = math.hypot(dx, dy)
            if d < 0.6:
                return True
            sp = math.hypot(v[0], v[1])
            if d < 2.5 and sp > 4.0:
                js = action_to_joystick(0, 0, 1.0, 0, 1, v[0], v[1])
            else:
                js = action_to_joystick(dx / d, dy / d, 1.0 if d > 2.0 else 0.4, 0, 0, v[0], v[1])
            _, info = self.s.step(tuple(float(q) for q in js[:4]) + (int(js[4]), float(js[5])))
            if info['fell']:
                self.s.recover(); return False
        return False

    def test_bridge(self):
        ob = self.obs(); self.log('start: ' + json.dumps(pow_block(ob)))
        rows = []
        # find the nearest super speed and super jump items from the observation block (camera-relative -> world:
        # ForceYaw is 0 in autotrain, so camera == world)
        for want, (wx, wy) in (('super_speed', KOTM_ITEMS['super_speed']), ('super_jump', KOTM_ITEMS['super_jump'])):
            # teleport 3 u beside the item (its world position is known for KOTM), roll over it
            z = self.g.floor_below(wx - 3.0, wy, 25.0)
            self.s.place((wx - 3.0, wy, (z if z is not None else 20.65) + R + 0.01), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            ob = self.obs(); blk = pow_block(ob)
            self.log(f'{want}: beside it, block = {blk}')
            ok = self.drive_to(wx, wy)
            ob = self.obs(); blk = pow_block(ob)
            self.log(f'  arrived {ok}: held = {blk["held"]}')
            if blk['held'] != want:
                rows.append({'test': want, 'picked': False}); continue
            # stop, then fire: super speed along +x world (yaw = atan2(1, 0)); super jump from rest
            hold(self.s, tuple(float(q) for q in action_to_joystick(0, 0, 1.0, 0, 1, ob[RAW_VEL][0], ob[RAW_VEL][1])[:4]) + (0, 0.0), 12)
            ob = self.obs(); v0 = np.asarray(ob[RAW_VEL], float); p0 = np.asarray(ob[RAW_POS], float)
            yaw = yaw_for(1.0, 0.0) if want == 'super_speed' else None
            trace = []
            ob = self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0), use_pow=1, pow_yaw=yaw)
            for k in range(14):
                v = np.asarray(ob[RAW_VEL], float); p = np.asarray(ob[RAW_POS], float)
                trace.append([round(float(c), 3) for c in (p[0], p[1], p[2], v[0], v[1], v[2])])
                ob = self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0))
            blk2 = pow_block(ob)
            self.log(f'  fired {want}: v before {np.round(v0, 2)}; held now {blk2["held"]}')
            self.log('   vx per decision: ' + ' '.join('%.2f' % t[3] for t in trace))
            self.log('   vz per decision: ' + ' '.join('%.2f' % t[5] for t in trace))
            self.log('   z  per decision: ' + ' '.join('%.2f' % t[2] for t in trace))
            rows.append({'test': want, 'picked': True, 'v0': v0.tolist(), 'p0': p0.tolist(), 'yaw': yaw, 'trace': trace})
        # the regular blast
        hold(self.s, (0.0, 0.0, 0.0, 0.0, 0, 0.0), 30)
        ob = self.obs(); blk = pow_block(ob); self.log(f'blast meter {blk["blast"]} (needs {BLAST_REQUIRED})')
        waited = 0
        while blk['blast'] < BLAST_REQUIRED and waited < 400:
            ob = self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0)); blk = pow_block(ob); waited += 1
        v0 = np.asarray(ob[RAW_VEL], float); meter = blk['blast']
        ob = self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0), use_blast=1)
        trace = []
        for k in range(14):
            v = np.asarray(ob[RAW_VEL], float); p = np.asarray(ob[RAW_POS], float)
            trace.append([round(float(c), 3) for c in (p[0], p[1], p[2], v[0], v[1], v[2])])
            ob = self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0))
        blk2 = pow_block(ob)
        self.log(f'blast at meter {meter}: vz before {v0[2]:.2f}; meter after {blk2["blast"]}')
        self.log('   vz per decision: ' + ' '.join('%.2f' % t[5] for t in trace))
        self.log('   z  per decision: ' + ' '.join('%.2f' % t[2] for t in trace))
        rows.append({'test': 'blast', 'meter': meter, 'v0': v0.tolist(), 'trace': trace, 'meter_after': blk2['blast']})
        with open(os.path.join(OUT_DIR, 'bridge.jsonl'), 'a') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        self.log('done')


    # ------------------------------------------------------------------ phase 2 measurements
    def radius(self):
        env = self.s.env; env.debug = None; env.control('RADIUS')
        for _ in range(12):
            if env.debug is not None:
                break
            self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0))
        d = {}
        for f in (env.debug or []):
            if '=' in f:
                k, v = f.split('=', 1); d[k] = v
        return d

    def still(self, n=20):
        hold(self.s, (0.0, 0.0, 0.0, 0.0, 0, 0.0), n)
        return self.obs()

    def stand_at(self, x, y, vel=(0.0, 0.0, 0.0), z=None):
        zf = self.floor(x, y, 20.65 if z is None else z - 0.2) if z is None else z - 0.2
        rad = float(self.radius().get('radius', R)) if pow_block(self.obs())['mega'] else R   # the mega sits higher
        self.s.place((x, y, zf + rad + 0.01), tuple(vel), (0.0, 0.0, 0.0))
        return self.still(8)

    def pick(self, name, which=0):
        if name in self.items:
            wx, wy, wz = self.items[name][which]
        else:
            wx, wy = KOTM_ITEMS[name]; wz = None
        # wait for the item to be there (7 s respawn after a pickup): its slot in the block, else 7.5 s
        for _ in range(130):
            ob = self.obs(); p = ob[RAW_POS]; blk = pow_block(ob)
            it = next((q for q in blk['items'] if q[0] == name and abs(p[0] + q[1] - wx) < 1.5 and abs(p[1] + q[2] - wy) < 1.5), None)
            if it is not None and it[4] <= 0.0:
                break
            self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0))
        if wz is not None:
            # any map: drop onto the item itself (overlap = pickup), no driving needed
            self.s.place((wx, wy, wz + 0.3), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        else:
            self.stand_at(wx, wy - 2.5)
            self.drive_to(wx, wy + 1.0)       # roll through the item (the mega sits 0.45 u up at (-25.2, 15))
        ob = self.still(12); blk = pow_block(ob)
        if name == 'blast':
            return blk['special'] == 1
        if blk['held'] != name:
            self.log(f'  pickup of {name} failed: held {blk["held"]}')
            return False
        return True

    def trace_after(self, use_pow=0, pow_yaw=None, use_blast=0, n=16, js=(0.0, 0.0, 0.0, 0.0, 0, 0.0)):
        ob = self.raw(js, use_pow=use_pow, pow_yaw=pow_yaw, use_blast=use_blast)
        out = []
        for _ in range(n):
            v = np.asarray(ob[RAW_VEL], float); p = np.asarray(ob[RAW_POS], float)
            out.append([round(float(c), 3) for c in (p[0], p[1], p[2], v[0], v[1], v[2])])
            ob = self.raw(js)
        return out

    def test_speed(self):
        """Super Speed direction: aim (yaw held with no use) for AIM decisions, then use; four yaws."""
        rows = []
        for aim in (0, 2):
            for yaw_deg in (0, 90, 180, -90):
                if not self.pick('super_speed'):
                    continue
                self.stand_at(-36.5, 9.0)                       # open floor west, room in every direction
                yaw = math.radians(yaw_deg)
                for _ in range(aim):
                    self.raw((0.0, 0.0, 0.0, 0.0, 0, yaw))
                tr = self.trace_after(use_pow=1, pow_yaw=yaw, n=6)
                dv = [tr[k][3:5] for k in range(4)]
                mags = [math.hypot(a, b) for a, b in dv]
                k = int(np.argmax(mags)); boost = mags[k]
                ang = math.degrees(math.atan2(dv[k][1], dv[k][0]))
                self.log(f'super speed: aim {aim} dec, yaw {yaw_deg:4d} deg -> boost {boost:.1f} u/s at world angle {ang:.0f} deg (atan2(vy, vx)) on decision {k}')
                rows.append({'test': 'speed', 'aim': aim, 'yaw_deg': yaw_deg, 'boost': boost, 'world_angle_deg': ang, 'decision': k, 'trace': tr})
                self.still(20)
        return rows

    def test_blast(self):
        """The regular blast: meter value -> vertical speed, standing on the floor; then the picked-up Blast."""
        rows = []
        self.wait_mega_off()
        self.stand_at(-36.5, 9.0)
        for tgt in (0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            ob = self.obs(); m = float(ob[RAW_POW_BLAST]); n = 0
            while m < tgt - 0.002 and n < 500:
                ob = self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0)); m = float(ob[RAW_POW_BLAST]); n += 1
            tr = self.trace_after(use_blast=1, n=14)
            vz = [t[5] for t in tr]; z = [t[2] for t in tr]
            self.log(f'blast at meter {m:.3f}: vz max {max(vz):.2f} on decision {int(np.argmax(vz))}, apex +{max(z) - z[0]:.2f} u; expected 10*sqrt(m) = {10 * math.sqrt(m):.2f}')
            rows.append({'test': 'blast', 'meter': m, 'vz_max': max(vz), 'apex': max(z) - z[0], 'trace': tr})
            self.still(24)
        if self.pick('blast'):
            self.stand_at(-36.5, 9.0); ob = self.obs(); blk = pow_block(ob)
            tr = self.trace_after(use_blast=1, n=14)
            vz = [t[5] for t in tr]; z = [t[2] for t in tr]
            self.log(f'PICKUP blast (special {blk["special"]}, meter {blk["blast"]}): vz max {max(vz):.2f}, apex +{max(z) - z[0]:.2f} u; expected 10*1.03 = 10.3')
            rows.append({'test': 'blast_pickup', 'meter': blk['blast'], 'special': blk['special'], 'vz_max': max(vz), 'apex': max(z) - z[0], 'trace': tr})
        return rows

    def wait_mega_off(self):
        n = 0
        while pow_block(self.obs())['mega'] == 1 and n < 400:
            self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0)); n += 1
        return n

    def test_mega(self):
        """Mega marble: radius, duration, floor speed, jump height, jump-then-mega boost."""
        rows = []
        self.wait_mega_off()
        r0 = self.radius(); self.log(f'normal: {r0}')
        jsf = action_to_joystick(0.0, 1.0, 1.0, 0, 0, 0.0, 0.0)   # +y: 26 u of floor along x = -36
        js = tuple(float(q) for q in jsf[:4]) + (0, float(jsf[5]))
        self.stand_at(-36.0, 2.0)
        tr = self.trace_after(js=js, n=30); sp_n = [math.hypot(t[3], t[4]) for t in tr]
        jsb = action_to_joystick(0.0, 0.0, 1.0, 0, 1, tr[-1][3], tr[-1][4]); jsb = tuple(float(q) for q in jsb[:4]) + (0, float(jsb[5]))
        trb = self.trace_after(js=jsb, n=20); sp_b = [math.hypot(t[3], t[4]) for t in trb]
        self.log(f'normal braking from {sp_b[0]:.1f}: ' + ' '.join('%.1f' % v for v in sp_b[:12]))
        self.stand_at(-36.5, 9.0); trj = self.trace_after(js=(0.0, 0.0, 0.0, 0.0, 1, 0.0), n=16); apex_n = max(t[2] for t in trj) - trj[0][2]
        self.log(f'normal: speed after 1 s {sp_n[15]:.2f}, 2 s {sp_n[29]:.2f} u/s; jump apex +{apex_n:.2f} u')
        if not self.pick('mega'):
            return rows
        self.stand_at(-36.5, 9.0)
        tr0 = self.trace_after(use_pow=1, n=8)
        ob = self.obs(); blk = pow_block(ob); rm = self.radius()
        self.log(f'mega used: block {blk}; radius {rm}; vz on use {[t[5] for t in tr0[:5]]}; z {[t[2] for t in tr0[:8]]}')
        self.stand_at(-36.0, 2.0); self.still(16)             # the activation pops the marble up: let it land first
        tr = self.trace_after(js=js, n=30); sp_m = [math.hypot(t[3], t[4]) for t in tr]
        # braking from the end of that run
        jsb = action_to_joystick(0.0, 0.0, 1.0, 0, 1, tr[-1][3], tr[-1][4]); jsb = tuple(float(q) for q in jsb[:4]) + (0, float(jsb[5]))
        trb = self.trace_after(js=jsb, n=20); sp_b = [math.hypot(t[3], t[4]) for t in trb]
        self.log(f'mega braking from {sp_b[0]:.1f}: ' + ' '.join('%.1f' % v for v in sp_b[:12]))
        self.stand_at(-36.5, 9.0); trj = self.trace_after(js=(0.0, 0.0, 0.0, 0.0, 1, 0.0), n=16); apex_m = max(t[2] for t in trj) - trj[0][2]
        blk = pow_block(self.obs())
        self.log(f'mega: speed after 1 s {sp_m[15]:.2f}, 2 s {sp_m[29]:.2f} u/s; jump apex +{apex_m:.2f} u; mega_left now {blk["mega_left"]}')
        n = 0
        while pow_block(self.obs())['mega'] == 1 and n < 400:
            self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0)); n += 1
        self.log(f'mega ended after ~{n * 0.064:.1f} s more; radius back to {self.radius().get("radius")}')
        rows.append({'test': 'mega', 'radius_normal': r0, 'radius_mega': rm, 'speed_normal': sp_n, 'speed_mega': sp_m, 'apex_normal': apex_n, 'apex_mega': apex_m})
        for delay in (1, 2, 3, 4):
            if not self.pick('mega'):
                break
            self.stand_at(-36.5, 9.0)
            ob = self.raw((0.0, 0.0, 0.0, 0.0, 1, 0.0))
            for _ in range(delay - 1):
                ob = self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0))
            tr = self.trace_after(use_pow=1, n=20); zs = [t[2] for t in tr]
            self.log(f'jump then mega after {delay} dec: apex +{max(zs) - tr[0][2]:.2f} u, vz peak {max(t[5] for t in tr):.2f} (plain jump {apex_n:.2f}, mega jump {apex_m:.2f}); vz {[t[5] for t in tr[:6]]}')
            rows.append({'test': 'jump_mega', 'delay': delay, 'apex': max(zs) - tr[0][2], 'trace': tr})
            n = 0
            while pow_block(self.obs())['mega'] == 1 and n < 400:
                self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0)); n += 1
        return rows


    def test_yaw(self):
        """Which way of setting the camera yaw sticks (diagnostic, YAWSET word): read back after 1 and 4 decisions."""
        self.stand_at(-36.5, 9.0)
        for mode in (0, 1, 2, 3):
            self.s.env.control('YAWSET 0 3'); self.still(4)
            self.s.env.control(f'YAWSET 1.0 {mode}')
            self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0))
            r1 = self.radius()
            for _ in range(3):
                self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0))
            r4 = self.radius()
            self.log(f'YAWSET 1.0 mode {mode}: after 1 dec {r1}; after 4 dec {r4}')
        # the action word 6 (steering yaw) as the drill sends it: yaw 1.0 for 3 decisions, then read
        for _ in range(3):
            self.raw((0.0, 0.0, 0.0, 0.0, 0, 1.0))
        self.log(f'word-6 yaw 1.0 x3: {self.radius()}')
        # fire the super speed after each mode, yaw 0 (expect +y if yaw 0 means forward = +y) and yaw 1.5708
        for mode in (1, 2):
            for yaw in (0.0, 1.5708):
                if not self.pick('super_speed'):
                    continue
                self.stand_at(-36.5, 9.0)
                self.s.env.control(f'YAWSET {yaw} {mode}')
                self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0))
                r = self.radius()
                tr = self.trace_after(use_pow=1, pow_yaw=yaw, n=6)
                dv = [tr[k][3:5] for k in range(4)]; mags = [math.hypot(a, b) for a, b in dv]; k = int(np.argmax(mags))
                self.log(f'mode {mode} yaw {yaw}: yaws before fire {r}; boost {mags[k]:.1f} u/s at world angle {math.degrees(math.atan2(dv[k][1], dv[k][0])):.0f} deg')
                self.still(20)
        return []


    def test_aimlag(self):
        """How many decisions the steering yaw (word 6) must be held before a Super Speed fires along it."""
        rows = []
        for aim in (1, 2, 3, 4, 6):
            if not self.pick('super_speed'):
                continue
            self.stand_at(-36.5, 9.0)
            yaw = 1.5708                                    # expect world -x? (+y is yaw 0): measured below
            for _ in range(aim):
                self.raw((0.0, 0.0, 0.0, 0.0, 0, yaw))
            r = self.radius()
            tr = self.trace_after(use_pow=1, pow_yaw=yaw, n=6, js=(0.0, 0.0, 0.0, 0.0, 0, yaw))
            dv = [tr[k][3:5] for k in range(4)]; mags = [math.hypot(a, b) for a, b in dv]; k = int(np.argmax(mags))
            ang = math.degrees(math.atan2(dv[k][1], dv[k][0]))
            self.log(f'aim {aim} dec at yaw {yaw}: client {r.get("camyaw")} server {r.get("srvyaw")} -> boost {mags[k]:.1f} u/s at world angle {ang:.0f} deg')
            rows.append({'test': 'aimlag', 'aim': aim, 'yaw': yaw, 'boost': mags[k], 'world_angle_deg': ang})
            self.still(20)
        # the mega use, held 2 decisions, with the block polled for 12 decisions
        if self.pick('mega'):
            self.stand_at(-36.5, 9.0)
            ob = self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0), use_pow=1)
            ob = self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0), use_pow=1)
            seq = []
            for _ in range(12):
                blk = pow_block(ob); seq.append((blk['held'], blk['mega'], blk['mega_left']))
                ob = self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0))
            self.log(f'mega use held 2 dec: {seq}; radius {self.radius().get("radius")}')
        return rows


    def test_mega2(self):
        """Mega on the floor, clean: acceleration from rest, braking, the groove and the strip; normal first."""
        rows = []
        self.wait_mega_off()
        jsf = action_to_joystick(0.0, 1.0, 1.0, 0, 0, 0.0, 0.0); js = tuple(float(q) for q in jsf[:4]) + (0, float(jsf[5]))
        def run_from(label):
            self.stand_at(-36.0, 2.0); self.still(10)
            tr = self.trace_after(js=js, n=36); sp = [math.hypot(t[3], t[4]) for t in tr]
            self.log(f'{label}: speed by decision ' + ' '.join('%.1f' % v for v in sp[::3]))
            return sp
        def groove(label, speed):
            # roll east along y = 13 from x = -30 at `speed` across the groove (r 2.25-3.5 from the centre gem) onto the block
            self.stand_at(-30.0, 13.0, vel=(speed, 0.0, 0.0));
            jse = action_to_joystick(1.0, 0.0, 1.0, 0, 0, speed, 0.0); jse = tuple(float(q) for q in jse[:4]) + (0, float(jse[5]))
            tr = self.trace_after(js=jse, n=24)
            zs = [t[2] for t in tr]; xs = [t[0] for t in tr]; sp = [math.hypot(t[3], t[4]) for t in tr]
            self.log(f'{label} groove at {speed} u/s: z min {min(zs):.2f} (floor+rad {zs[0]:.2f}), z max {max(zs):.2f}, x end {xs[-1]:.1f}, speed end {sp[-1]:.1f}, vz peak {max(t[5] for t in tr):.2f}')
            return tr
        sp_n = run_from('normal accel'); g_n = [groove('normal', v) for v in (6.0, 10.0)]
        if not self.pick('mega'):
            return rows
        self.stand_at(-36.5, 9.0); self.trace_after(use_pow=1, n=4); self.still(24)
        sp_m = run_from('mega accel'); g_m = [groove('mega', v) for v in (6.0, 10.0)]
        blk = pow_block(self.obs()); self.log(f'mega left {blk["mega_left"]}')
        rows.append({'test': 'mega2', 'speed_normal': sp_n, 'speed_mega': sp_m, 'groove_normal': g_n, 'groove_mega': g_m})
        self.wait_mega_off()
        return rows

    def test_speed2(self):
        """Super Speed at speed and in the air: does the boost add 25 to the velocity, along the yaw, regardless?"""
        rows = []
        yaw = 1.5708                                                   # world +x
        for label, vel, air in (('from rest', (0.0, 0.0, 0.0), False), ('at 10 u/s along +x', (10.0, 0.0, 0.0), False),
                                ('at 10 u/s along -x', (-10.0, 0.0, 0.0), False), ('in the air (after a jump)', (0.0, 0.0, 0.0), True)):
            if not self.pick('super_speed'):
                continue
            self.stand_at(-36.5, 9.0, vel=vel)
            js = (0.0, 0.0, 0.0, 0.0, 0, yaw)
            if air:
                self.raw((0.0, 0.0, 0.0, 0.0, 1, yaw)); self.raw(js); self.raw(js)    # the jump fires 2 decisions after the key
            else:
                self.raw(js)
            ob = self.obs(); v0 = np.asarray(ob[RAW_VEL], float)
            tr = self.trace_after(use_pow=1, pow_yaw=yaw, n=6, js=js)
            dv = [(tr[k][3] - v0[0], tr[k][4] - v0[1]) for k in range(4)]; mags = [math.hypot(a, b) for a, b in dv]; k = int(np.argmax(mags))
            self.log(f'super speed {label}: v0 {np.round(v0, 1)} z0 {ob[RAW_POS][2]:.2f} -> dv {mags[k]:.1f} u/s at world angle {math.degrees(math.atan2(dv[k][1], dv[k][0])):.0f} deg; v after {tr[k][3]:.1f},{tr[k][4]:.1f},{tr[k][5]:.1f}')
            rows.append({'test': 'speed2', 'label': label, 'v0': v0.tolist(), 'dv': mags[k], 'angle': math.degrees(math.atan2(dv[k][1], dv[k][0])), 'trace': tr})
            self.still(24)
        return rows


    def test_effects(self):
        """Helicopter (and any timed effect): one measurement per use, within the 5 s, each from a fresh pickup:
        (a) a jump: apex and hang time, (b) a jump then 1.5 s of full throttle +x in the air: air control,
        (c) a drop from 3 u: fall speed and restitution. The plain marble first at the same spot."""
        rows = []
        name = next((n for n in ('helicopter', 'super_bounce', 'shock_absorber') if n in self.items), None)
        if name is None:
            self.log('no helicopter / bounce / shock item on this map'); return rows
        # the flattest instance: floor at the item's level 3 u around it (Promontory's items sit on slopes and ledges)
        def flat(it):
            x0, y0, z0 = it
            if self.g is None:
                return 0
            offs = []
            for dx, dy in ((0, 0), (1.5, 0), (-1.5, 0), (0, 1.5), (0, -1.5), (3, 0), (-3, 0), (0, 3), (0, -3)):
                q = self.g.floor_below(x0 + dx, y0 + dy, z0 + 1.0)
                if q is None:
                    return -9.0
                offs.append(q - z0)
            return -(max(offs) - min(offs))                       # 0 = perfectly flat around the item (items float ~0.5 u up)
        order = sorted(range(len(self.items[name])), key=lambda i: -flat(self.items[name][i]))
        x, y, z = self.items[name][order[0]]
        self.log(f'{name}: using item #{order[0]} at {(x, y, z)} (floor spread {-flat((x, y, z)):.2f} u)')
        jsf = action_to_joystick(1.0, 0.0, 1.0, 0, 0, 0.0, 0.0); jsx = tuple(float(q) for q in jsf[:4]) + (0, float(jsf[5]))
        def jump(label):
            self.stand_at(x, y, z=z); trj = self.trace_after(js=(0.0, 0.0, 0.0, 0.0, 1, 0.0), n=60)
            zj = [t[2] for t in trj]; apex = max(zj) - zj[0]; hang = sum(1 for t in trj if t[2] > zj[0] + 0.05) * 0.064
            self.log(f'{label} jump: apex +{apex:.2f} u, hang {hang:.2f} s, vz {[round(t[5], 1) for t in trj[2:8]]}; heli left {self.obs()[RAW_POW_HELI_LEFT]:.1f}')
            return {'apex': apex, 'hang': hang, 'trace': trj}
        def aircontrol(label):
            self.stand_at(x, y, z=z)
            self.raw((0.0, 0.0, 0.0, 0.0, 1, 0.0)); self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0)); self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0))
            tra = self.trace_after(js=jsx, n=24); vx = [t[3] for t in tra]; zs = [t[2] for t in tra]
            self.log(f'{label} air control: vx after 0.5 / 1.0 / 1.5 s of throttle {vx[7]:.1f} / {vx[15]:.1f} / {vx[23]:.1f}; in the air {sum(1 for q in zs if q > zs[0] - 0.05) * 0.064:.2f} s')
            return {'vx': vx, 'trace': tra}
        def drop(label):
            self.stand_at(x, y, z=z); zf = float(self.obs()[RAW_POS][2])
            self.s.place((x, y, zf + 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            tr = self.trace_after(n=40); vz = [t[5] for t in tr]
            k = next((i for i in range(1, len(vz)) if vz[i] > vz[i - 1] + 2.0), None)
            rest = (vz[k] / -vz[k - 1]) if k is not None and vz[k - 1] < -1 else None
            self.log(f'{label} drop 3 u: fastest fall {min(vz):.2f} u/s, restitution {rest if rest is None else round(rest, 2)}')
            return {'restitution': rest, 'vz_min': min(vz), 'trace': tr}
        base = {'jump': jump('plain'), 'air': aircontrol('plain'), 'drop': drop('plain')}
        rows.append({'test': 'effects', 'which': 'plain', **base})
        res = {}
        for k, fn in (('jump', jump), ('air', aircontrol), ('drop', drop)):
            which = order[len(res) % len(order)]
            if not self.pick(name, which):
                continue
            self.stand_at(x, y, z=z); self.trace_after(use_pow=1, n=3)
            ob = self.obs(); self.log(f'{name} used (item #{which}): heli left {ob[RAW_POW_HELI_LEFT]:.2f} bounce {ob[RAW_POW_BOUNCE_LEFT]:.2f} shock {ob[RAW_POW_SHOCK_LEFT]:.2f}')
            res[k] = fn(name)
            n = 0
            while max(self.obs()[RAW_POW_BOUNCE_LEFT], self.obs()[RAW_POW_SHOCK_LEFT], self.obs()[RAW_POW_HELI_LEFT]) > 0 and n < 200:
                self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0)); n += 1
        rows.append({'test': 'effects', 'which': name, **res})
        return rows

    def test_bounce(self):
        """Super Bounce and Shock Absorber step by step: held after the pickup, the active flags and schedules after
        the use, the drop restitution, with the RADIUS debug line at every step."""
        rows = []
        for name in ('super_bounce', 'shock_absorber'):
            if name not in self.items:
                continue
            for which in range(min(2, len(self.items[name]))):
                x, y, z = self.items[name][which]
                ok = self.pick(name, which)
                r0 = self.radius(); self.log(f'{name} #{which} at {(x, y, z)}: pickup {ok}; {r0.get("held")} act {r0.get("act")} sched {r0.get("sched")}')
                if not ok:
                    continue
                self.stand_at(x, y, z=z)
                self.trace_after(use_pow=1, n=3)
                r1 = self.radius(); self.log(f'   after use: held {r1.get("held")} act {r1.get("act")} sched {r1.get("sched")}; obs left bounce {self.obs()[RAW_POW_BOUNCE_LEFT]:.2f} shock {self.obs()[RAW_POW_SHOCK_LEFT]:.2f}')
                zf = float(self.obs()[RAW_POS][2])
                self.s.place((x, y, zf + 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
                tr = self.trace_after(n=30); vz = [t[5] for t in tr]
                k = next((i for i in range(1, len(vz)) if vz[i] > vz[i - 1] + 2.0), None)
                rest = (vz[k] / -vz[k - 1]) if k is not None and vz[k - 1] < -1 else None
                self.log(f'   drop: restitution {rest if rest is None else round(rest, 2)} (vz {vz[k - 1] if k else None} -> {vz[k] if k else None}); {self.radius().get("act")} {self.radius().get("sched")}')
                rows.append({'test': 'bounce', 'name': name, 'which': which, 'restitution': rest, 'drop': tr})
                n = 0
                while max(self.obs()[RAW_POW_BOUNCE_LEFT], self.obs()[RAW_POW_SHOCK_LEFT]) > 0 and n < 200:
                    self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0)); n += 1
        return rows


    def test_launch(self):
        """Spin launch: a spinning marble that lands on a high-friction (tarmac) floor, plain and as a mega activated in
        the air after a jump (operator: launches the marble forward VERY fast). Spin about +y rolls along +x."""
        rows = []
        import collections
        # a flat tarmac spot from the geometry (material friction_high)
        tris, norms, mats, names = self.g.tris, self.g.norms, self.g.mats, list(self.g.mat_names)
        hi = [i for i, n in enumerate(names) if 'friction' in str(n).lower() and 'high' in str(n).lower()]
        sel = np.isin(mats, hi) & (norms[:, 2] > 0.95)
        if not sel.any():
            self.log('no flat high-friction floor on this map'); return rows
        c = tris[sel].mean(axis=1)
        cl = collections.Counter((round(float(x) / 6) * 6, round(float(y) / 6) * 6, round(float(z), 1)) for x, y, z in c)
        (tx, ty, tz), ncount = cl.most_common(1)[0]
        zf = self.g.floor_below(tx, ty, tz + 1.0)
        tz = zf if zf is not None else tz
        self.log(f'tarmac spot ({tx}, {ty}, {tz:.2f}), {ncount} triangles in the cell')
        def launch(label, spin, mega):
            # start spinning in place on the tarmac, jump, (mega in the air), land, read the speed
            if mega:
                # the nearest mega item to the spot
                k = int(np.argmin([math.hypot(x - tx, y - ty) for x, y, z in self.items['mega']]))
                if not self.pick('mega', k):
                    return None
            self.s.place((tx, ty, tz + R + 0.01), (0.0, 0.0, 0.0), (0.0, spin, 0.0))
            ob = self.raw((0.0, 0.0, 0.0, 0.0, 1, 0.0))               # jump key
            ob = self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0))
            ob = self.raw((0.0, 0.0, 0.0, 0.0, 0, 0.0), use_pow=1 if mega else 0)   # fires 2 decisions later, in the air
            tr = self.trace_after(n=40)
            sp = [math.hypot(t[3], t[4]) for t in tr]; zs = [t[2] for t in tr]
            self.log(f'{label}: spin {spin} rad/s, mega {mega}: peak speed {max(sp):.1f} u/s at decision {int(np.argmax(sp))}, speed at 2.5 s {sp[-1]:.1f}, z max {max(zs) - zs[0]:.2f}; vx by decision ' + ' '.join('%.1f' % t[3] for t in tr[::4]))
            blk = pow_block(self.obs())
            if blk['mega']:
                self.wait_mega_off()
            return {'spin': spin, 'mega': mega, 'peak': max(sp), 'trace': tr}
        for spin in (90.0, 150.0, 250.0, 400.0):
            for mega in (False, True):
                r = launch('tarmac', spin, mega)
                if r:
                    rows.append({'test': 'launch', **r})
        return rows


    def test_spin(self):
        """How fast can the marble spin? (a) rolling at full throttle on the floor for 2.5 s, (b) in the air after a
        jump holding a direction, (c) in the air after a jump holding the OPPOSITE direction to the roll (spin up
        against the velocity), (d) the same with the helicopter if the map has one. Spin read from the observation."""
        rows = []
        from nav.protocol import RAW_SPIN
        x, y = (-36.0, 2.0) if self.map.startswith('King') else tuple(self.items['super_speed'][0][:2])
        z = None if self.map.startswith('King') else self.items['super_speed'][0][2]
        jsf = action_to_joystick(0.0, 1.0, 1.0, 0, 0, 0.0, 0.0); jsy = tuple(float(q) for q in jsf[:4]) + (0, float(jsf[5]))
        jsb = action_to_joystick(0.0, -1.0, 1.0, 0, 0, 0.0, 0.0); jsyb = tuple(float(q) for q in jsb[:4]) + (0, float(jsb[5]))
        def spin_of(ob):
            w = np.asarray(ob[RAW_SPIN], float); return float(np.linalg.norm(w)), [round(float(c), 1) for c in w]
        # (a) floor
        self.stand_at(x, y, z=z)
        env = self.s.env; ws = []
        ob = self.raw(jsy)
        for _ in range(40):
            ws.append(spin_of(ob)[0]); ob = self.raw(jsy)
        sp = math.hypot(ob[RAW_VEL][0], ob[RAW_VEL][1])
        self.log(f'floor, full throttle 2.5 s: speed {sp:.1f} u/s, spin {ws[-1]:.0f} rad/s (speed/spin = {sp / max(ws[-1], 1e-6):.3f}, R 0.19); spin by 0.5 s: ' + ' '.join('%.0f' % w for w in ws[7::8]))
        rows.append({'test': 'spin', 'case': 'floor', 'spin': ws, 'speed': sp})
        # (b) in the air, holding the roll direction; (c) holding the opposite
        for label, js in (('air, same direction', jsy), ('air, opposite direction', jsyb)):
            self.stand_at(x, y, z=z, vel=(0.0, 8.0, 0.0))
            ob = self.raw((0.0, 0.0, 0.0, 0.0, 1, 0.0)); ob = self.raw(js); ob = self.raw(js)
            w0 = spin_of(ob); ws = []; zs = []
            for _ in range(14):
                ob = self.raw(js); ws.append(spin_of(ob)); zs.append(float(ob[RAW_POS][2]))
            self.log(f'{label}: spin at takeoff {w0[0]:.0f} {w0[1]} -> after 0.9 s in the air {ws[-1][0]:.0f} {ws[-1][1]}; z {[round(q, 2) for q in zs[::3]]}')
            rows.append({'test': 'spin', 'case': label, 'spin0': w0, 'spin': ws})
        return rows


    def test_redirect(self):
        """V9 redirect (log 40.12): roll along the ring at 8 u/s with a Super Speed held, fire with the aim that
        ss_redirect_aim solves for a next gem at 90 deg, and read the velocity 3-8 decisions later: expect it along the
        line to the next gem at ~23 u/s. Both turns; also the straight run (aim = heading) as a control."""
        from nav.obs import ss_redirect_aim
        rows = []
        x, y = -36.0, 14.0                                   # on the west ring, floor 20.65, rolling south toward (-36, 2)
        for label, vel, nxt in (('south, next east', (0.0, -8.0), (1.0, 0.0)), ('south, next west', (0.0, -8.0), (-1.0, 0.0)),
                                ('south, straight run', (0.0, -8.0), (0.0, -1.0))):
            if not self.pick('super_speed'):
                continue
            aim, lam = ss_redirect_aim(vel[0], vel[1], nxt[0], nxt[1])
            yaw = math.atan2(aim[0], aim[1])
            self.stand_at(x, y, vel=(vel[0], vel[1], 0.0))
            self.s.place((x, y, self.floor(x, y, 20.65) + R + 0.01), (vel[0], vel[1], 0.0), (0.0, 0.0, 0.0))
            tr = self.trace_after(use_pow=1, pow_yaw=yaw, n=12)
            vs = [(math.hypot(t[3], t[4]), math.degrees(math.atan2(t[4], t[3]))) for t in tr]
            want = math.degrees(math.atan2(nxt[1], nxt[0]))
            self.log(f'{label}: aim ({aim[0]:.2f}, {aim[1]:.2f}) yaw {yaw:.3f}, predicted {lam:.1f} u/s at {want:.0f} deg; '
                     + 'speed/angle by decision: ' + ' '.join(f'{sp:.1f}@{ang:.0f}' for sp, ang in vs[::2])
                     + f'; pos end ({tr[-1][0]:.1f}, {tr[-1][1]:.1f}, {tr[-1][2]:.1f})')
            rows.append({'test': 'redirect', 'case': label, 'aim': aim, 'trace': tr})
            self.still(30)
        return rows


    def test_brake(self):
        """Braking from a Super Speed's speed on the floor: no input vs the full brake (anti-velocity at full throttle,
        as action_to_joystick's brake) vs holding forward. Deceleration per decision at 25 -> 10 u/s. (log 40.18)"""
        rows = []
        x, y = -36.0, 2.5                                    # bottom of the west ring, floor runs east for 22+ u
        for label, mode in (('no input', 0), ('full brake', 1), ('hold forward', 2)):
            self.s.place((x, y, self.floor(x, y, 20.65) + R + 0.01), (25.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            tr = []
            ob = self.obs()
            for _ in range(28):
                v = np.asarray(ob[RAW_VEL], float); p = np.asarray(ob[RAW_POS], float)
                tr.append((float(p[0]), float(p[1]), float(p[2]), float(v[0]), float(v[1])))
                if mode == 0:
                    js = (0.0, 0.0, 0.0, 0.0, 0, 0.0)
                elif mode == 1:
                    j = action_to_joystick(0.0, 0.0, 1.0, 0, 1, float(v[0]), float(v[1])); js = tuple(float(q) for q in j[:4]) + (0, float(j[5]))
                else:
                    j = action_to_joystick(1.0, 0.0, 1.0, 0, 0, float(v[0]), float(v[1])); js = tuple(float(q) for q in j[:4]) + (0, float(j[5]))
                ob = self.raw(js)
            sp = [math.hypot(t[3], t[4]) for t in tr]
            # decel between the decisions where speed is between 22 and 10 u/s
            ks = [k for k in range(1, len(sp)) if 10.0 <= sp[k] <= 22.0 and sp[k - 1] > sp[k]]
            dec = (sp[ks[0] - 1] - sp[ks[-1]]) / (0.064 * (ks[-1] - ks[0] + 1)) if len(ks) >= 2 else float('nan')
            dist = math.hypot(tr[-1][0] - tr[0][0], tr[-1][1] - tr[0][1])
            self.log(f'{label}: speed by decision ' + ' '.join(f'{q:.1f}' for q in sp[::3]) + f'; decel {dec:.1f} u/s^2 (22->10 u/s); travelled {dist:.1f} u in {len(tr) * 0.064:.1f} s; z end {tr[-1][2]:.2f}')
            rows.append({'test': 'brake', 'case': label, 'decel': dec, 'speeds': sp})
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--map', default='KingOfTheMarble_Hunt')
    ap.add_argument('--test', default='bridge')
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)
    d = Drill(a.port, a.map, log)
    rows = []
    if a.test == 'bridge':
        d.test_bridge()
    for t in a.test.split(','):
        if t == 'speed':
            rows += d.test_speed()
        elif t == 'blast':
            rows += d.test_blast()
        elif t == 'mega':
            rows += d.test_mega()
        elif t == 'yaw':
            rows += d.test_yaw()
        elif t == 'aimlag':
            rows += d.test_aimlag()
        elif t == 'mega2':
            rows += d.test_mega2()
        elif t == 'speed2':
            rows += d.test_speed2()
        elif t == 'effects':
            rows += d.test_effects()
        elif t == 'bounce':
            rows += d.test_bounce()
        elif t == 'launch':
            rows += d.test_launch()
        elif t == 'spin':
            rows += d.test_spin()
        elif t == 'redirect':
            rows += d.test_redirect()
        elif t == 'brake':
            rows += d.test_brake()
    if rows:
        with open(os.path.join(OUT_DIR, 'measure.jsonl'), 'a') as f:
            for r in rows:
                f.write(json.dumps(r) + chr(10))
    log('done')


if __name__ == '__main__':
    main()
