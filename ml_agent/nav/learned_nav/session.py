"""One game connection for stage 3 work on any practice map: contact telemetry on, recovery, state teleports.

The game must be on the expected map (`-autotrain <map>`); Python binds the port first (nav_ready.ps1).
Timing (measured in P0, 2026-09-27): a reply sent at decision i acts between decisions i+1 and i+2; after a
TELEPORT control word and SETTLE more decisions the observation is exactly the teleported state, and the first
physics step follows it. A control word leaves the previous input held, so the input wanted at the start is sent
one decision before the teleport.
"""
import math

import numpy as np

from nav.env import HuntEnv
from nav.protocol import RAW_POS, RAW_VEL, RAW_SPIN, NOOP_ACTION, CONTACT_FIELDS, format_teleport

R = 0.19
SETTLE = 2
SPIN_WAIT_MS = 600000        # lockstep spin-wait (see contact_on); the engine advances anyway after AI::WaitTimeoutMs


class RoundOver(RuntimeError):
    pass


class Session:
    def __init__(self, port, expect_map, geometry, log=print):
        self.log = log
        self.geo = geometry
        self.env = HuntEnv(port, speed=3, log=log)
        self.env.connect()
        self.round_s = float(self.env.info.get('round_ms', 0.0)) / 1000.0
        mission = self.env.info.get('mission', '')
        if mission != expect_map:
            raise SystemExit(f'the game is on "{mission}", not {expect_map}')
        self.zmin = float(np.nanmin(geometry.heights))
        self.oob_pending = False       # an out of bounds was seen: the game will respawn the marble (after a delay)
        self.contact_on()
        self.log(f'connected: {mission}, round {self.round_s:.0f} s')

    def contact_on(self):
        self.env.control('CONTACT 1')
        # lockstep waits spin instead of sleeping (SPINWAIT, mlAgent.cs): under a locked or occluded session each
        # 1 ms engine sleep cost ~200 ms (probe 2026-09-29: replies over 3 ms made a decision 40-260 ms of wall time)
        self.env.control(f'SPINWAIT {SPIN_WAIT_MS}')

    # -- stepping
    def obs(self):
        return self.env.msg.obs

    def telemetry(self):
        e = self.env.msg.extra
        return dict(zip(CONTACT_FIELDS, e)) if e is not None else None

    def step(self, js):
        msg, info = self.env.step(js, repeat=1)
        if info['fell']:
            self.oob_pending = True
        if info['round_ended'] or info['reconnected']:
            self.log('round ended / reconnected: waiting for the next round')
            self.env.wait_new_round()
            self.contact_on()
            raise RoundOver('round over')
        return msg, info

    def control(self, word):
        self.env.control(word)
        if self.env.round_ended or self.env.reconnected:
            self.env.wait_new_round()
            self.contact_on()
            raise RoundOver('round over')

    def in_countdown(self):
        return self.round_s > 0 and self.env.time_left_s() >= self.round_s - 0.5

    def on_map(self):
        x, y, z = (float(v) for v in self.obs()[RAW_POS])
        g = self.geo
        return g.xs[0] <= x <= g.xs[-1] and g.ys[0] <= y <= g.ys[-1] and z > self.zmin - 1.0

    def ready(self):
        """Countdown over and the marble on the map. After an out of bounds the game schedules a respawn that
        would otherwise land in the middle of the next trial (measured 2026-09-27: 55 of 300 KOTM trials), so
        the quick respawn is triggered here and awaited: the marble must visibly move to a spawn point."""
        n = 0
        while self.in_countdown():
            self.step(NOOP_ACTION)
            n += 1
            if n > 400:
                raise RuntimeError('countdown did not end')
        if self.oob_pending or not self.on_map():
            self.recover()

    def recover(self):
        p_before = np.asarray(self.obs()[RAW_POS], dtype=np.float64)
        self.control('OOBCLICK')
        moved_at = None
        for i in range(240):
            self.step(NOOP_ACTION)
            p = np.asarray(self.obs()[RAW_POS], dtype=np.float64)
            if moved_at is None and np.linalg.norm(p - p_before) > 2.0 and self.on_map():
                moved_at = i
            p_before = p if moved_at is None else p_before
            if moved_at is not None and i - moved_at >= 12:
                break
            if i in (60, 120, 180) and moved_at is None:
                # the game may have respawned the marble by itself already: then it rests on the map, supported
                t = self.telemetry()
                if self.on_map() and t is not None and t['support_steps'] > 0 and np.linalg.norm(self.obs()[RAW_VEL]) < 0.5:
                    moved_at = i - 12
                    break
                self.control('RESPAWN')
        self.oob_pending = False
        if moved_at is None or not self.on_map():
            # soft failure: the next trial teleports anyway, and a late respawn inside a trial is caught by the
            # recorder's jump guard (trial discarded, recovery requested again)
            p = self.obs()[RAW_POS]
            self.log(f'recover: respawn not confirmed (pos {float(p[0]):.1f} {float(p[1]):.1f} {float(p[2]):.1f}, '
                     f'on map {self.on_map()}, telemetry {self.telemetry()})')

    def place(self, pos, vel=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0), hold=NOOP_ACTION, settle_js=None):
        """Teleport to an exact state with `hold` as the held input; returns the start observation (the
        teleported state) or None if the game did not take the teleport."""
        self.ready()
        self.step(hold)
        self.control(format_teleport(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], spin=spin))
        for _ in range(SETTLE):
            self.step(settle_js if settle_js is not None else hold)
        o = self.obs()
        p0 = np.asarray(o[RAW_POS], dtype=np.float64); v0 = np.asarray(o[RAW_VEL], dtype=np.float64)
        if np.linalg.norm(p0 - np.asarray(pos)) > 0.05 or np.linalg.norm(v0 - np.asarray(vel)) > 0.1:
            return None
        return o


def rolling_spin(vel, normal):
    """Spin of a marble rolling without slip on a surface with this normal: w = n x v / r."""
    n = np.asarray(normal, dtype=np.float64); v = np.asarray(vel, dtype=np.float64)
    return tuple(np.cross(n, v) / R)
