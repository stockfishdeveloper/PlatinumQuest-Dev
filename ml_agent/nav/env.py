"""One game connection as an environment: the game connects to us, sends an
observation every 16 ms of sim time, and we answer each one. A decision is
ACTION_REPEAT observations (4 x 16 ms = 64 ms of sim time) with the same
joystick reply.

    python -m nav.env --smoke        connects to a running game (any hunt map), steps 300 decisions

Rounds end by themselves (the game sends []|-score|0|1, disconnects and reconnects
a few seconds later on the next round); step() reports that as round_ended so
the caller can start a fresh segment. Control words (SPEED, TELEPORT, INFO,
STATS) consume one 16 ms message each (the game applies no keys that tick).
"""
import os
import sys
import time
import socket
import select
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav.protocol import (GameMessage, parse_message, format_action, format_teleport,   # noqa: E402
                          NOOP_ACTION, RAW_POS, RAW_VEL, RAW_TIME_LEFT)

ACTION_REPEAT = 4
TICK_S = 0.016
# Built-engine training mode (2026-09-18, marbleblast_mbx.exe): the game advances exactly OBS_MS
# of simulation per observation and stands still until we reply (lockstep), so one observation
# is one decision, nothing is ever stale, and the sim runs as fast as the round trip allows
# (~70x per instance). With the shipped exe these control words are ignored and the old
# 16 ms / 4-message decision applies.
TRAINING_MODE = True
OBS_MS = int(os.environ.get('NAV_OBS_MS', '64'))   # ms of sim advanced per decision. 64 is the
                              # trained default; NAV_OBS_MS lets an EVAL run at a finer step
                              # without retraining, to test whether control bandwidth is the
                              # constraint on arrival speed. NOTE: TIME, GEM_SPEED_REF,
                              # TIMEOUT_PER_GEM and the airborne counters are all denominated
                              # in DECISIONS, so changing this for TRAINING requires rescaling
                              # them all. For eval only it is safe.
RENDER_EVERY = 100
NEW_ROUND_MIN_LEFT_S = 200.0     # fallback when the round length is unknown (5-min rounds)
NEW_ROUND_FRACTION = 0.8         # clock >= this fraction of the round length = the next round has started


class HuntEnv:
    def __init__(self, port=8888, speed=3, action_repeat=ACTION_REPEAT, log=print):
        self.port = port
        self.speed = speed
        self.repeat = 1 if TRAINING_MODE else action_repeat
        self.obs_ms = OBS_MS if TRAINING_MODE else 16
        self.log = log
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(('127.0.0.1', port)); self.srv.listen(1)
        self.conn = None; self.buf = b''
        self.reads = 0; self.stale_reads = 0   # lines served from the buffer without waiting = we were late
        self.msg = None                   # the observation awaiting our reply
        self.info = {}                    # from the INFO control word
        self.stats = None
        self.debug = None
        self.recent_lines = []            # last non-observation lines, for diagnostics
        self.round_ended = False
        self.reconnected = False
        self.ticks = 0
        self._t0 = time.perf_counter(); self._ticks0 = 0
        self.connections = 0

    # ------------------------------------------------------------------ socket
    def _accept(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
        self.log(f'[env:{self.port}] waiting for the game')
        self.conn, addr = self.srv.accept()
        self.buf = b''
        self.connections += 1
        self.reconnected = self.connections > 1
        self.log(f'[env:{self.port}] game connected ({self.connections})')

    def _send(self, line):
        try:
            self.conn.sendall((line + '\n').encode())
        except OSError as e:
            # the game went away mid-step (crash or the game loop's proactive restart):
            # wait for the next instance; the reply is lost, the caller sees `reconnected`
            self.log(f'[env:{self.port}] send failed ({e}); waiting for the game to come back')
            self._accept()

    def _readline(self):
        waited = False
        while True:
            i = self.buf.find(b'\n')
            if i >= 0:
                line, self.buf = self.buf[:i + 1], self.buf[i + 1:]
                self.reads += 1
                if not waited:
                    self.stale_reads += 1
                return line.decode('utf-8', 'replace')
            try:
                data = self.conn.recv(65536)
            except OSError:
                data = b''
            if data:
                self.buf += data; waited = True
                continue
            self.log(f'[env:{self.port}] game disconnected')
            self._accept()

    def lag_pct(self):
        """Share of observations since the last call that were already waiting in the socket when we
        asked for them (0 = we answer every game frame as it happens; high = we run behind)."""
        r, st = self.reads, self.stale_reads
        self.reads = self.stale_reads = 0
        return 100.0 * st / r if r else 0.0

    def drain(self):
        """Discard the observations that piled up while we were busy. The game never waits for a
        reply: it sends one observation per frame and keeps the last keys pressed, so after any
        pause on our side (a PPO update = ~12 s of sim at 3x) the socket holds hundreds of stale
        observations. Answering those one by one drove the marble from observations seconds in
        the past (the 'wrong direction' segments, ignored TELEPORTs and refused RESPAWNs of
        2026-09-17). Returns the number of observations dropped; the newest one becomes the
        pending observation. Non-observation lines in the backlog are processed as usual
        (a round end sets round_ended)."""
        n = 0; last = None
        while True:
            i = self.buf.find(b'\n')
            if i >= 0:
                line, self.buf = self.buf[:i + 1], self.buf[i + 1:]
                m = parse_message(line.decode('utf-8', 'replace'))
                if m.kind == 'obs':
                    n += 1; last = m; self.ticks += 1
                else:
                    self.recent_lines.append(f'{self.ticks}:{m.kind}:{m.raw[:160]}')
                    self.recent_lines = self.recent_lines[-20:]
                    if m.kind == 'end':
                        self.round_ended = True
                    elif m.kind == 'info':
                        self.info = {'mission': m.fields[0] if m.fields else '',
                                     'round_ms': float(m.fields[1]) if len(m.fields) > 1 else 0.0,
                                     'time_scale': float(m.fields[2]) if len(m.fields) > 2 else 0.0}
                    elif m.kind == 'stats':
                        self.stats = m.fields
                    elif m.kind == 'debug':
                        self.debug = m.fields
                continue
            ready, _, _ = select.select([self.conn], [], [], 0.0)
            if not ready:
                break
            try:
                data = self.conn.recv(65536)
            except OSError:
                data = b''
            if not data:
                self.log(f'[env:{self.port}] game disconnected (found while draining)')
                self._accept(); last = self._recv_obs(); self.reconnected = True
                break
            self.buf += data
        if last is not None:
            self.msg = last
        return n

    def _recv_obs(self, reply_to_extra=NOOP_ACTION):
        """Next 'obs' message; stats/info are stored; a round-end message sets round_ended."""
        while True:
            m = parse_message(self._readline())
            if m.kind == 'obs':
                self.ticks += 1
                return m
            self.recent_lines.append(f'{self.ticks}:{m.kind}:{m.raw[:160]}')
            self.recent_lines = self.recent_lines[-20:]
            if m.kind == 'stats':
                self.stats = m.fields
            elif m.kind == 'debug':
                self.debug = m.fields
            elif m.kind == 'info':
                self.info = {'mission': m.fields[0] if m.fields else '',
                             'round_ms': float(m.fields[1]) if len(m.fields) > 1 else 0.0,
                             'time_scale': float(m.fields[2]) if len(m.fields) > 2 else 0.0}
            elif m.kind == 'end':
                self.round_ended = True
                self._send(format_action(*reply_to_extra, tick=m.tick))
            else:
                self._send(format_action(*reply_to_extra, tick=m.tick))

    # ------------------------------------------------------------------ lifecycle
    def connect(self):
        self._accept()
        self.msg = self._recv_obs()
        self.reconnected = False
        self.request_info()
        self.set_speed(self.speed)
        return self.msg

    def request_info(self, wait_ticks=30):
        self.info = {}
        self.control('INFO')
        for _ in range(wait_ticks):
            if self.info:
                break
            self.step(NOOP_ACTION, repeat=1)
        return self.info

    def control(self, word):
        """Send a control word as the reply to the pending observation; the next observation
        becomes the pending one."""
        self._send(word)
        self.msg = self._recv_obs()

    def request_debug(self, wait_ticks=24):
        """Ask the game for frame diagnostics (DEBUG control); returns the fields or None."""
        self.debug = None
        self.control('DEBUG')
        for _ in range(wait_ticks):
            if self.debug is not None:
                break
            self.step(NOOP_ACTION, repeat=1)
        return self.debug

    def set_speed(self, n):
        self.speed = n
        self.control(f'SPEED {n:g}')
        if TRAINING_MODE:
            # on every (re)connect path: fixed step, lockstep, render 1 frame in RENDER_EVERY
            self.control(f'FIXEDSTEP {OBS_MS}')
            self.control('LOCKSTEP 1')
            self.control(f'RENDEREVERY {RENDER_EVERY}')

    def mark(self, x, y, z):
        """Show the current waypoint in the game (a small start pad); cosmetic."""
        self.control(f'MARK {x:.3f} {y:.3f} {z:.3f}')

    def mark_off(self):
        """Hide the waypoint marker (between gem groups nothing may be drawn off a gem)."""
        self.control('MARK off')

    def teleport(self, x, y, z, vx=0.0, vy=0.0, vz=0.0, settle_ticks=2):
        self.control(format_teleport(x, y, z, vx, vy, vz))
        for _ in range(settle_ticks):
            self.step(NOOP_ACTION, repeat=1)
        return self.msg

    # ------------------------------------------------------------------ stepping
    def step(self, joystick, use_pow=0, repeat=None):
        """Reply `joystick` = (fwd, back, left, right, jump) to the pending observation and hold it
        for `repeat` observations. Returns (last_msg, info) with info: fell (any oob in the
        window), gem_delta, round_ended, reconnected, ticks."""
        repeat = self.repeat if repeat is None else repeat
        self.round_ended = False; self.reconnected = False
        fell = False; gems = 0.0
        m = self.msg
        for i in range(repeat):
            self._send(format_action(*joystick, use_pow=use_pow, tick=m.tick))
            m = self._recv_obs()
            fell |= bool(m.oob); gems += m.gem_delta
            # NOTE: m.done is the old trainer's 15000-step episode cap, not the round end;
            # the round end is the 'end' message (round_ended).
            if self.round_ended or self.reconnected:
                break
        self.msg = m
        return m, {'fell': fell, 'gem_delta': gems, 'round_ended': self.round_ended,
                   'reconnected': self.reconnected, 'ticks': self.ticks}

    def step_async(self, joystick, use_pow=0):
        """Send the reply to the pending observation without waiting for the next one (vector env:
        send to every instance first, then collect; the lockstepped games run in parallel)."""
        self.round_ended = False; self.reconnected = False
        self._send(format_action(*joystick, use_pow=use_pow, tick=self.msg.tick))

    def step_wait(self):
        """Second half of step_async: the next observation and the same info dict as step()."""
        m = self._recv_obs()
        self.msg = m
        return m, {'fell': bool(m.oob), 'gem_delta': m.gem_delta, 'round_ended': self.round_ended,
                   'reconnected': self.reconnected, 'ticks': self.ticks}

    def time_left_s(self):
        return float(self.msg.obs[RAW_TIME_LEFT]) / 1000.0 if self.msg is not None and self.msg.obs is not None else 0.0

    elapsed_s = time_left_s      # old name; the field counts DOWN

    def wait_new_round(self, max_ticks=30000):
        """After the round-end message the game shows the results, restarts the level (same
        socket; a reconnect can also happen) and resumes sending observations when the new
        round's timer runs. OOB respawns are disabled in the end state, so nothing useful can
        happen until then. Replies NOOP until the first observation of the new round."""
        n = 0
        prev = self.time_left_s()
        t0 = time.perf_counter()
        # a fresh round shows (nearly) the whole round length on the clock; rounds are 3 or 5 min
        round_s = float(self.info.get('round_ms', 0.0)) / 1000.0
        fresh = round_s * NEW_ROUND_FRACTION if round_s > 0 else NEW_ROUND_MIN_LEFT_S
        self.log(f'[env:{self.port}] round over (time left {prev:.0f}s of {round_s:.0f}s); waiting for the next round')
        # The observation right after the end message usually already belongs to the new
        # round (clock back at the full round length); otherwise wait for the clock to jump up.
        while n < max_ticks and not (prev >= fresh and not self.round_ended):
            self.msg, _ = self.step(NOOP_ACTION, repeat=1)
            n += 1
            if self.reconnected:
                break
            t = self.time_left_s()
            if t > prev + 60.0 or t >= fresh:
                if not self.round_ended:
                    break
            prev = t
        self.log(f'[env:{self.port}] next round after {n} ticks / {time.perf_counter() - t0:.0f} s wall '
                 f'(time left {self.time_left_s():.0f}s, reconnected {self.reconnected})')
        self.round_ended = False
        self.request_info()
        self.set_speed(self.speed)
        return self.msg

    def rtf(self):
        """Real-time factor since the last call (sim seconds per wall second)."""
        now = time.perf_counter()
        dt = now - self._t0; dticks = self.ticks - self._ticks0
        self._t0, self._ticks0 = now, self.ticks
        return (dticks * self.obs_ms / 1000.0 / dt) if dt > 0 else 0.0

    def pos(self):
        return self.msg.obs[RAW_POS]

    def vel(self):
        return self.msg.obs[RAW_VEL]

    def close(self):
        try:
            if self.conn:
                self.conn.close()
        finally:
            self.srv.close()


def _smoke():
    env = HuntEnv(port=int(os.environ.get('PROBE_PORT', '8888')))
    env.connect()
    print('info:', env.info)
    env.rtf()
    t0 = time.perf_counter()
    for i in range(300):
        m, inf = env.step(NOOP_ACTION)
        if inf['round_ended']:
            print('round ended at decision', i)
    print(f'300 decisions in {time.perf_counter() - t0:.1f} s, rtf {env.rtf():.2f}x, pos {np.round(env.pos(), 2)}')
    env.close()


if __name__ == '__main__':
    if '--smoke' in sys.argv:
        _smoke()
    else:
        print(__doc__)
