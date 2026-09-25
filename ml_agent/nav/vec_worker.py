"""One game instance per process: the env, its map, observation building and the segment
logic run here; the trainer process does the batched policy, the rollouts and PPO.

Started by the trainer as `python -m nav.vec_worker <idx> <game port> <seed> <arrive_r> <arrive_dz>
<control port>` (a plain subprocess, not multiprocessing: a spawned child would re-import the
trainer module and with it torch, 380 MB per worker) and talks to it over a local
multiprocessing.connection socket.

Protocol over the pipe (trainer -> worker):
    ('act', a_game)          step once with this action (5 floats: dx, dy, throttle, jump, brake)
    ('arrive', r, dz)        the arrival ratchet moved
    ('stop',)
worker -> trainer, one reply per command, a dict:
    ready:   {'crop', 'vec', 'mission', 'log'}                       (first message, after connecting)
    step:    {'skip': False, 'r', 'done', 'outcome', 'crop', 'vec', 'new_segment', 'fell', 'trace',
              'stats' (only when a segment ended), 'mission', 'log', 'flips'}
             {'skip': True, 'truncate': k, 'crop', 'vec', 'new_segment': True, ...}
                                       the step produced no transition (frame flip / round end):
                                       drop the last k rollout steps of this worker and reset its state
Everything the worker wants printed goes into 'log' (a list of strings) because only the trainer
owns the log file.
"""
import os
import sys
import math
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrain_obs import TerrainMap                                                  # noqa: E402
from nav.protocol import RAW_VEL                                                    # noqa: E402
from nav.env import HuntEnv, OBS_MS                                                         # noqa: E402
from nav.terrain import TerrainGrid                                                 # noqa: E402
from nav.obs import ObsBuilder, VEC_GAP                                             # noqa: E402
from nav.joystick import action_to_joystick                                         # noqa: E402
from nav.waypoints import SegmentManager, RoundOver                                 # noqa: E402
from nav.gems import visible_gems, choose, STICKY_TOL                               # noqa: E402

# Frame check (2026-09-17): a segment whose marble accelerates against its commands (cosine
# < FRAME_FLIP_COS over the first FRAME_CHECK_N rolling decisions) is corrupted game state;
# force a respawn, drop those decisions from the rollout and start over. (The original cause,
# stale observations after an update, is gone in lockstep; this stays as a safety net.)
FRAME_CHECK_N = 12
FRAME_FLIP_COS = -0.3
GAME_SPEED = int(os.environ.get('NAV_SPEED', '3'))
REAL_GEMS = os.environ.get('NAV_REAL_GEMS', '1') == '1'   # 2026-09-23 (HANDOFF 28.25): train on the GAME's gems.
                              # Goals come from nav/gems.choose (the same chooser real rounds use), the game's
                              # gem_delta is the arrival, there are no teleports and no episode end at a pickup,
                              # and every finished round logs its real score: training IS the real game now.
TRAIN_WATCH = os.environ.get('NAV_TRAIN_WATCH', '0') == '1'   # pace every decision to real
                              # time so a human can watch TRAINING (not the eval). Lockstep
                              # means the sim advances as fast as we reply, so without this
                              # the marble is a blur whatever the speed setting says.
# EMA weight on the commanded DIRECTION, applied before it reaches the game so the policy trains
# against the smoothed dynamics. 1.0 = raw output, which is where this should stay.
# STALE RATIONALE: this cited HANDOFF section 5b, "the raw policy flips heading 39.5 deg per
# decision and sustains thrust for 0.06 s, which caps speed at ~4 u/s". No longer true. After
# DIR_GOAL_GAIN = 30 the policy holds a heading inside 20 deg for a median 0.90 s, LONGER than the
# human's 0.74 s, and FlatGem real rounds run at 10.08 u/s. See HANDOFF section 23.
ACTION_SMOOTH = float(os.environ.get('NAV_ACTION_SMOOTH', '1.0'))   # 1.0 = OFF. This low-passed
                               # the commanded direction so the policy could not twitch. It bought
                               # speed (3.51 -> 5.26 u/s) but it is a crutch: imposed from outside,
                               # tuned per situation, and at 0.15 it was too sluggish to hook a gem.
                               # Replaced by GEM_SPEED_BONUS in waypoints.py, which pays for
                               # reaching gems sooner and leaves the technique to the policy.


class InstanceWorker:
    def __init__(self, idx, port, seed, arrive_r, arrive_dz):
        self.idx = idx; self.port = port
        self.rng = np.random.default_rng(seed)
        self.arrive_r, self.arrive_dz = arrive_r, arrive_dz
        self.lines = []
        self.env = HuntEnv(port, speed=GAME_SPEED, log=self.log)
        self.mission = ''; self.terrain = None; self.obs_b = None; self.segs = None
        self.goal = None; self.crop = None; self.vec = None; self.on_floor = True
        self.seg_steps = 0; self.fc_num = 0.0; self.fc_den = 0.0
        self.ep_reward = 0.0; self.seg_count = 0; self.frame_flips = 0; self.steps = 0
        self.smooth_dir = None            # EMA state for the commanded direction
        self.real = False; self.target = None            # real-gem mode state
        self.round_points = 0.0; self.round_gems = 0; self.round_falls = 0
        self.t0 = time.perf_counter()
        self.prof = {'game': 0.0, 'obs': 0.0, 'begin': 0.0}   # wall seconds since the last reply

    def log(self, s):
        self.lines.append(f'[{self.idx}] {s}')

    def drain_log(self):
        out, self.lines = self.lines, []
        return out

    # ------------------------------------------------------------------ setup
    def load_map(self, name):
        t = TerrainGrid(TerrainMap.resolve(name))
        self.mission = name; self.terrain = t; self.obs_b = ObsBuilder(t)
        self.segs = SegmentManager(t, self.rng, self.log, arrive_r=self.arrive_r, arrive_dz=self.arrive_dz)
        self.real = REAL_GEMS and bool(getattr(t, 'gem_spawns', None)); self.segs.real_mode = self.real
        self.log(f'real-gem training mode: {self.real}')
        self.log(f'mission {name}: terrain {t.W}x{t.H} @ {t.res} u, walkable {int(t.walkable.sum())} cells')

    def sync_map(self):
        if not self.env.info.get('mission'):
            self.env.request_info()
        if self.env.info.get('mission') and self.env.info['mission'] != self.mission:
            self.load_map(self.env.info['mission'])
            self.log(f'MAP {self.mission}')

    def connect(self):
        self.env.connect()
        for attempt in range(10):
            if self.env.info.get('mission'):
                break
            self.log(f'no INFO answer yet (attempt {attempt + 1}); asking again')
            self.env.request_info(wait_ticks=60)
        if not self.env.info.get('mission'):
            raise RuntimeError(f'instance {self.idx} never answered INFO; is mlAgent.cs up to date (delete the .dso)?')
        self.load_map(self.env.info['mission'])
        self.start_segment()

    def _round_over(self):
        """Every path that crosses a round end goes through here: log the finished round's real
        score once, reset the counters. (Before 2026-09-23 17:00 the RoundOver paths inside
        start_segment/recover skipped both, so a round ending mid-respawn merged into the next.)"""
        if self.real and self.env.round_ended:
            self.log(f'GAME map={self.mission} points={self.round_points:.0f} gems={self.round_gems} falls={self.round_falls} rtf={self.env.rtf():.1f}')
        self.round_points = 0.0; self.round_gems = 0; self.round_falls = 0

    def start_segment(self):
        tb = time.perf_counter()
        while True:
            try:
                if self.real:
                    vis = []
                    for _ in range(60):               # a gem is normally visible at once; wait through a spawn gap
                        vis = visible_gems(self.env.msg.obs)
                        if vis:
                            break
                        self.segs._step_checked(self.env, 1)
                    if vis:
                        o = self.env.msg.obs
                        tgt, nxt = choose(vis, None, o[0:2], o[3:5])
                        self.goal = self.segs.begin(self.env, real_goal=tgt[:3], real_next=(nxt[:3] if nxt else None))
                        self.target = tgt
                        break
                self.goal = self.segs.begin(self.env)
                break
            except RoundOver:
                self.segs.abandon()
                self._round_over()
                if self.env.round_ended:
                    self.env.wait_new_round()
                else:
                    self.env.request_info(); self.env.set_speed(GAME_SPEED)
                self.sync_map()
                self.log(f'new round (game connection {self.env.connections})')
        self.obs_b.reset()
        self.smooth_dir = None            # the marble was just teleported: old heading is meaningless
        self.seg_steps = 0; self.fc_num = self.fc_den = 0.0
        self.crop, self.vec, self.on_floor = self.obs_b.build(self.env.msg.obs, self.goal, self.segs.next_goal())
        self.prof['begin'] += time.perf_counter() - tb

    # ------------------------------------------------------------------ one decision
    def step(self, a_game):
        vel = self.env.msg.obs[RAW_VEL]
        a = [float(v) for v in a_game]
        if ACTION_SMOOTH < 1.0:
            n = math.hypot(a[0], a[1])
            if n > 1e-6:
                ux, uy = a[0] / n, a[1] / n
                # accumulate UNNORMALISED, normalise only the output: renormalising the state
                # every step makes an exact 180 deg reversal a fixed point (opposite unit
                # vectors cancel) and the heading could never flip.
                if self.smooth_dir is None:
                    self.smooth_dir = (ux, uy)
                else:
                    self.smooth_dir = ((1.0 - ACTION_SMOOTH) * self.smooth_dir[0] + ACTION_SMOOTH * ux,
                                       (1.0 - ACTION_SMOOTH) * self.smooth_dir[1] + ACTION_SMOOTH * uy)
                sn = math.hypot(*self.smooth_dir)
                a[0], a[1] = (self.smooth_dir[0] / sn, self.smooth_dir[1] / sn) if sn > 1e-6 else (ux, uy)
        js = action_to_joystick(a[0], a[1], a[2], a[3], a[4], float(vel[0]), float(vel[1]))
        v_before = (float(vel[0]), float(vel[1])); prev_obs = self.env.msg.obs
        tg = time.perf_counter()
        if TRAIN_WATCH:
            # hold each decision to OBS_MS of wall clock, the same pacing real_run uses in WATCH
            nt = getattr(self, '_next_tick', None)
            if nt is None:
                nt = self._next_tick = time.perf_counter()
            self._next_tick = nt + OBS_MS / 1000.0
            slack = self._next_tick - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                self._next_tick = time.perf_counter()
        msg, info = self.env.step(js)
        self.prof['game'] += time.perf_counter() - tg
        if self.real:
            self.round_points += float(info['gem_delta'])
            if info['gem_delta'] > 0:
                self.round_gems += 1
            if info['fell']:
                self.round_falls += 1
        rolling = self.on_floor and abs(float(prev_obs[2]) - self.terrain.floor_z(float(prev_obs[0]), float(prev_obs[1]), float(prev_obs[2]))) < 0.3
        if rolling and not info['fell'] and not (info['round_ended'] or info['reconnected']):
            cx, cy = js[3] - js[2], js[0] - js[1]
            ax, ay = float(msg.obs[3]) - v_before[0], float(msg.obs[4]) - v_before[1]
            nc, na = math.hypot(cx, cy), math.hypot(ax, ay)
            if nc > 0.3 and na > 0.05:
                self.fc_num += (cx * ax + cy * ay) / (nc * na); self.fc_den += 1.0
        self.seg_steps += 1
        if self.seg_steps == FRAME_CHECK_N and self.fc_den >= 5 and self.fc_num / self.fc_den < FRAME_FLIP_COS:
            self.frame_flips += 1
            self.log(f'FRAME FLIP: command/acceleration cosine {self.fc_num / self.fc_den:.2f} over the first {FRAME_CHECK_N} decisions at '
                     f'{np.round(msg.obs[:3], 1)}; forcing a respawn and restarting the segment (flips on this instance {self.frame_flips})')
            k = self.seg_steps - 1
            self.segs.abandon(); self.ep_reward = 0.0
            self.env.control('RESPAWN')
            self.start_segment()
            return self._reply(skip=True, truncate=k)
        if info['round_ended'] or info['reconnected']:
            self.segs.abandon()
            self._round_over()
            if info['round_ended']:
                self.env.wait_new_round()
            else:
                self.env.request_info(); self.env.set_speed(GAME_SPEED)
            self.sync_map()
            self.log(f'new round (game connection {self.env.connections}); rtf {self.env.rtf():.1f}')
            self.ep_reward = 0.0
            self.start_segment()
            return self._reply(skip=True, truncate=0)
        airborne = not self.on_floor
        r, done, outcome = self.segs.step(msg.obs[:3], info['fell'], airborne, info['round_ended'], self.env.time_left_s(),
                                          braked=a[4] > 0.5, airborne_decisions=self.obs_b.airborne,
                                          jumped=a[3] > 0.5, vel=(float(msg.obs[3]), float(msg.obs[4])),
                                          approved=bool(self.vec is not None and self.vec[VEC_GAP + 2] > 0.5),   # 28.34
                                          gap_ratio=float(self.vec[VEC_GAP + 3]) if self.vec is not None else 0.0,   # 28.37 run-up credit
                                          lip_u=float(self.vec[VEC_GAP]) * 20.0 if self.vec is not None else float('inf'),
                                          # elements 5-6 are the policy's MEAN direction when the
                                          # trainer supplies it. Charging TURN_COST on the SAMPLE
                                          # billed the policy for its own exploration noise (~17 deg
                                          # of the ~34 deg at std 0.21), which it could only reduce
                                          # by shrinking log_std -- the one thing the entropy
                                          # controller exists to prevent. Falls back to the sample
                                          # for eval paths that send a bare 5-vector.
                                          cmd_dir=(a[5], a[6]) if len(a) > 6 else (a[0], a[1]),
                                          picked=float(info['gem_delta']) if self.real else 0.0)
        if self.segs.take_respawn():
            # fell mid-group: get back on the map, keep the same gems, carry on
            try:
                self.segs.recover(self.env)
            except RoundOver:
                self.segs.abandon()
                self._round_over()
                if self.env.round_ended:
                    self.env.wait_new_round()
                else:
                    self.env.request_info(); self.env.set_speed(GAME_SPEED)
                self.sync_map()
                self.ep_reward = 0.0
                self.start_segment()
                return self._reply(skip=True, truncate=0)
            self.obs_b.reset()                    # the marble teleported: frame history is stale
        mk = self.segs.take_mark()
        if mk is not None:
            self.env.mark(mk[0], mk[1], mk[2])      # show the next gem of the group
            # ...and retarget the OBSERVATION onto it. Without this the segment manager chases the
            # new gem (reward, arrival, prev_d) while the policy still sees the one it just
            # collected, sitting at its own feet: it orbits a dead waypoint until the timeout.
            self.goal = self.segs.seg.goal
        self.ep_reward += r
        o = msg.obs
        trace = (f'{self.seg_count},{self.env.time_left_s():.2f},{o[0]:.2f},{o[1]:.2f},{o[2]:.2f},{o[3]:.2f},{o[4]:.2f},{o[5]:.2f},'
                 f'{int(self.on_floor)},{js[0]:.2f},{js[1]:.2f},{js[2]:.2f},{js[3]:.2f},{js[4]},{int(a[4] > 0.5)},{int(info["fell"])},'
                 f'{r:.2f},{int(done)},{outcome or ""},{self.segs.seg.prev_d:.1f},{self.goal[0]:.1f},{self.goal[1]:.1f}')
        self.steps += 1
        stats = None; ep = None
        if done:
            self.seg_count += 1; ep = self.ep_reward; self.ep_reward = 0.0
            stats = self.segs.stats()
            self.start_segment()
        else:
            if self.real:
                vis = visible_gems(msg.obs)
                if vis:
                    tgt, nxt = choose(vis, self.target, msg.obs[0:2], msg.obs[3:5])
                    g = self.segs.seg.goal
                    want = (nxt[:3] if nxt else None)
                    if tgt is not None and math.hypot(tgt[0] - g[0], tgt[1] - g[1]) > STICKY_TOL:
                        self.segs.retarget(float(msg.obs[0]), float(msg.obs[1]), tgt[:3], want)
                        self.goal = self.segs.seg.goal
                        self.env.mark(self.goal[0], self.goal[1], self.goal[2])
                    elif tgt is not None:
                        nn = self.segs.seg.real_next
                        if (nn is None) != (want is None) or (want is not None and math.hypot(want[0] - nn[0], want[1] - nn[1]) > STICKY_TOL):
                            self.segs.set_next(float(msg.obs[0]), float(msg.obs[1]), want)
                    self.target = tgt
            tg = time.perf_counter()
            self.crop, self.vec, self.on_floor = self.obs_b.build(self.env.msg.obs, self.goal, self.segs.next_goal())
            self.prof['obs'] += time.perf_counter() - tg
        return self._reply(skip=False, r=r, done=done, outcome=outcome, new_segment=bool(done), fell=info['fell'],
                           trace=trace, stats=stats, ep_reward=ep)

    def _reply(self, **kw):
        kw.update({'crop': self.crop, 'vec': self.vec, 'mission': self.mission, 'log': self.drain_log(),
                   'flips': self.frame_flips, 'seg_count': self.seg_count, 'prof': self.prof})
        self.prof = {'game': 0.0, 'obs': 0.0, 'begin': 0.0}
        if kw.get('skip'):
            kw['new_segment'] = True
        return kw


def worker_main(conn, idx, port, seed, arrive_r, arrive_dz):
    w = InstanceWorker(idx, port, seed, arrive_r, arrive_dz)
    try:
        w.connect()
        conn.send({'ready': True, 'crop': w.crop, 'vec': w.vec, 'mission': w.mission, 'log': w.drain_log()})
        while True:
            cmd = conn.recv()
            if cmd[0] == 'act':
                conn.send(w.step(cmd[1]))
            elif cmd[0] == 'arrive':
                w.arrive_r, w.arrive_dz = cmd[1], cmd[2]
                w.segs.arrive_r, w.segs.arrive_dz = cmd[1], cmd[2]
                w.segs.history = []
            elif cmd[0] == 'stop':
                break
    except (EOFError, ConnectionResetError, BrokenPipeError):
        pass                                     # the trainer went away (restart / stop): just exit
    except Exception as e:                       # the trainer restarts everything on a worker failure
        import traceback
        try:
            conn.send({'error': f'{e}\n{traceback.format_exc()}', 'log': w.drain_log()})
        except Exception:
            pass
    finally:
        try:
            w.env.close()
        except Exception:
            pass


if __name__ == '__main__':
    from multiprocessing.connection import Client
    idx, game_port, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    arrive_r, arrive_dz, ctl_port = float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
    conn = Client(('127.0.0.1', ctl_port), authkey=b'nav')
    worker_main(conn, idx, game_port, seed, arrive_r, arrive_dz)
