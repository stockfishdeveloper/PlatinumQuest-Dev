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
import uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrain_obs import TerrainMap                                                  # noqa: E402
from nav.protocol import RAW_VEL, RAW_POW_HELD, RAW_POW_BLAST, RAW_POW_SPECIAL, BLAST_REQUIRED, RAW_SPIN, RAW_POS   # noqa: E402
from nav.env import HuntEnv, OBS_MS                                                         # noqa: E402
from nav.replay import SCHEMA, WINDOW_MS, APPROACH_MS, validate_start, load_starts, restore as restore_replay
from nav.terrain import TerrainGrid                                                 # noqa: E402
from nav.obs import ObsBuilder, VEC_GAP, VEC_POW, ss_aim, ss_redirect_window, WAYPOINT_DIST_SCALE, fill_use_state, USE_T_SCALE   # noqa: E402
USE_T_SCALE_WARM = 9.0        # 40.40: 'no recent fire' in the warm-up observations
import json                                                                         # noqa: E402
# 40.26 SUPER SPEED CURRICULUM, Stage 1: these instances do not play rounds; each segment teleports the marble INTO a
# recorded post-kick state from the agent's own play (nav/ss_drill_starts.py) and the targets are the gem the kick was
# aimed at and the gem after it (synthetic arrival, the same shaped reward). The other instances play ordinary rounds.
DRILL_INSTANCES = os.environ.get('NAV_DRILL_INSTANCES', '0,1,2,3')   # 'none' = no drills
DRILL_STARTS = os.environ.get('NAV_DRILL_STARTS', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                               'datasets', 'ss_drill', 'starts_dev.json'))
DRILL_MAX_SPEED = 25.0        # u/s after the kick: only the kicks the approval allows (solver <= 24, the game ~1.7 lower)
DRILL_SPEED_JITTER = 0.10     # engine variations of the recorded start: speed x U(1 -/+ this)
DRILL_ANGLE_JITTER = 4.0      # deg on the heading
DRILL_POS_JITTER = 0.15       # u
DRILL_GEM_DZ = 0.2            # target height above the floor
DRILL_EVAL = os.environ.get('NAV_DRILL_EVAL', '0') == '1'   # evaluation (nav/ss_drill_eval.py): starts in order, no variations
# 40.27 Stage 2: a drill starts PRE_DEC (5) decisions before a recorded kick with a Super Speed granted (bridge GIVEPOW),
# the gem about to be picked up, the kick's gem and the one after as targets; the POLICY decides whether and when to use it
DRILL_PLAN = os.environ.get('NAV_DRILL_PLAN', '0:4,1:4,2:4,3:4')   # real-game windows plus ordinary points-only rounds
DRILL2_STARTS = os.environ.get('NAV_DRILL2_STARTS', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                                 'datasets', 'ss_drill', 'starts2_dev.json'))
DRILL3_STARTS = os.environ.get('NAV_DRILL3_STARTS', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                                 'datasets', 'ss_drill', 'starts3_dev.json'))   # 40.27 Stage 3:
                              # the approach: 20 decisions (1.3 s) before the kick, with the pickups on the way in as targets
DRILL_NO_USE = os.environ.get('NAV_DRILL_NO_USE', '0') == '1'  # matched comparison: the use bit is ignored
SS_DATABLOCK = 'SuperSpeedItem_MBU'                          # granted at a stage 2 start (falls back to SuperSpeedItem)
# Real Hunt snapshots 1.5 s before a fire, including the original target and prior observations.
# Stage 4 restores the game state and plays a fixed simulation-time window through real pickups.
LIVE_RECORD = os.environ.get('NAV_LIVE_RECORD', '1') == '1'
LIVE_HIST = 32                # raw observations kept before the pre state (2 s)
LIVE_PRE_DEC = int(round(APPROACH_MS / OBS_MS))  # 1.5 s before a fire; preserve the approach
LIVE_LOOKBACKS = tuple(sorted({int(n) for n in os.environ.get('NAV_LIVE_LOOKBACKS', str(LIVE_PRE_DEC)).split(',')}, reverse=True))
if not LIVE_LOOKBACKS or min(LIVE_LOOKBACKS) < 3 or max(LIVE_LOOKBACKS) > 128:
    raise ValueError('NAV_LIVE_LOOKBACKS must contain decision counts between 3 and 128')
DRILL4_STARTS = os.environ.get('NAV_DRILL4_STARTS', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                                 'datasets', 'ss_drill', 'starts_replay_dev.json'))


def drill_stage_of(idx):
    """40.27: the drill stage of an instance (0 = ordinary rounds)."""
    if DRILL_PLAN:
        for item in DRILL_PLAN.split(','):
            i, _, st = item.partition(':')
            if i.strip() == str(idx):
                return int(st or 1)
        return 0
    return 1 if (DRILL_INSTANCES != 'none' and str(idx) in DRILL_INSTANCES.split(',')) else 0
from nav.joystick import action_to_joystick                                         # noqa: E402
from nav.waypoints import SegmentManager, RoundOver                                 # noqa: E402
from nav.gems import visible_gems, choose, plan_tour, STICKY_TOL                    # noqa: E402
TOUR = os.environ.get('NAV_TOUR', '').replace('greedy', '')   # 2026-09-26 (HANDOFF 28.51-28.54, operator): the gem
                              # chooser for real-gem training, same default as nav/real_run.py. '' or 'greedy' = choose()
                              # (DEFAULT again since 28.54: training with the planner gave no gain, 28.52); 'walk' = the
                              # whole-spawn planner (nav/gems.plan_tour, walk-only Dijkstra); 'euc' = straight-line.

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
        self.round_uses = {}; self.prev_held = 0           # powerup use decisions / fires this round (log 40)
        self.ss_fire_step = -10**9                           # step of the last Super Speed fire (falls within 3 s -> 'fss')
        self.redirect_sent = -10**9; self.redirect_open = []  # redirect uses: did the gem get picked within 12 decisions of the fire?
        self.fire_pending = None; self.fire_bucket = None    # 40.20: the kick's resultant speed (3 decisions after the fire) in
                                                              # buckets A < 18, B 18-24, C > 24 u/s: 'ksA..C' fires, 'kfA..C' falls within 3 s
        self.pk = {'on': [0.0, 0], 'off': [0.0, 0], 'none': [0.0, 0]}   # pickup speed sums by redirect flag (log 40.13)
        self.rw = {'fire': [0.0, 0], 'all': [0.0, 0]}      # reward per decision within 30 decisions of a Super Speed fire vs all (40.18)
        # 40.26 use state (V10 observation): the use bits sent the last two decisions, a kick pending, its latched aim
        self.use_sent = [False, False]; self.pending_since = None; self.latched_aim = None
        self.drill_stage = drill_stage_of(idx); self.drill = self.drill_stage > 0
        self.hist = []                 # 40.40: the last LIVE_HIST (raw obs, goal, next goal) of this instance's play
        self.live_file = None; self.warm = None; self.final_obs = None
        self.live_written = 0; self.live_rejected = 0
        self.session_id = uuid.uuid4().hex
        self.round_id = 0
        self.replay_start_ms = None
        self.recovery_pending = False
        self.replay_window_ms = WINDOW_MS
        self.last_drill_result = None
        self.eval_starts = None
        self.drill_starts = []; self.drill_cur = None
        self.drill_agg = {'n': 0, 'hit1': 0, 'hit2': 0, 'fell': 0, 't1': 0.0, 't2': 0.0, 'ret': 0.0}
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
        if self.drill:
            if self.drill_stage != 4 and not DRILL_EVAL:
                raise ValueError('synthetic stages 1-3 are diagnostics only; points training uses stage 4')
            src = {2: DRILL2_STARTS, 3: DRILL3_STARTS, 4: DRILL4_STARTS}.get(self.drill_stage, DRILL_STARTS)
            if self.drill_stage == 4:
                if DRILL_EVAL:
                    self.drill_starts = self.eval_starts or []
                else:
                    self.drill_starts = load_starts(src)
                self.real = True; self.segs.real_mode = True
                self.segs.drill_mode = True; self.segs.drill_window = True
                self.tour_fields = {}
                self.log(f'real-game replay: {len(self.drill_starts)} fixtures, game points, {WINDOW_MS} ms')
                return
            try:
                allst = json.load(open(src))
            except Exception as e:
                allst = []; self.log(f'drill starts {src} not readable: {e}')
            self.drill_starts = [s for s in allst if s.get('mission', name) == name and s['speed_after'] <= DRILL_MAX_SPEED]
            if self.drill_starts:
                self.real = False; self.segs.real_mode = False; self.segs.drill_mode = True
                self.segs.drill_window = self.drill_stage == 4          # 40.40: a fixed window, falls continue
                self.log(f'SUPER SPEED DRILL instance, stage {self.drill_stage}: {len(self.drill_starts)} starts from '
                         f'{os.path.basename(src)}{" (no use)" if DRILL_NO_USE else ""}')
            else:
                self.drill = False; self.drill_stage = 0; self.segs.drill_mode = False
                self.log(f'no drill starts for {name}: ordinary rounds')
        self.log(f'real-gem training mode: {self.real}')
        self.log(f'mission {name}: terrain {t.W}x{t.H} @ {t.res} u, walkable {int(t.walkable.sum())} cells')
        self.tour_fields = {}                 # walk-only Dijkstra field per gem position (TOUR == 'walk'), per map
        self.log(f'gem chooser: {"whole-spawn planner (" + TOUR + ")" if TOUR else "greedy"}')

    def walk_dist(self, a, g):
        key = (round(g[0], 1), round(g[1], 1))
        f = self.tour_fields.get(key)
        if f is None:
            f = self.tour_fields[key] = self.terrain.goal_field(g[0], g[1], jumps=False)
        return self.terrain.dist_at(f, float(a[0]), float(a[1]), (g[0], g[1]))

    def pick(self, vis, current, pos, vel):
        """Target and next gem: the whole-spawn planner by default (TOUR), else the greedy chooser."""
        if TOUR:
            return plan_tour(vis, current, pos, vel, dist=(self.walk_dist if TOUR == 'walk' else None))
        return choose(vis, current, pos, vel)

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
        if LIVE_RECORD or self.drill_stage == 4:
            self.env.control('REPLAYCAPTURE 1')
            for _ in range(8):
                if self.env.replay_world is not None:
                    break
                self.env.step((0, 0, 0, 0, 0), repeat=1)
            if self.env.replay_world is None:
                raise RuntimeError('trainingReplay.cs did not send world telemetry')
        if self.drill_stage == 4:
            # Initial countdown events can override a restored state after the acknowledgment.
            for _ in range(300):
                if self.env.replay_world['supported']:
                    break
                self.env.step((0, 0, 0, 0, 0), repeat=1)
            else:
                raise RuntimeError('mission never reached a supported replay state after GO')
        if self.drill_stage == 4 and DRILL_EVAL:
            return                       # evaluator explicitly starts each named fixture once
        self.start_segment()

    def _round_over(self):
        """Every path that crosses a round end goes through here: log the finished round's real
        score once, reset the counters. (Before 2026-09-23 17:00 the RoundOver paths inside
        start_segment/recover skipped both, so a round ending mid-respawn merged into the next.)"""
        if self.real and self.env.round_ended:
            uses = ','.join(f'{k}:{v}' for k, v in sorted(self.round_uses.items())) or '-'
            pk = ' '.join(f'pk{k}={v[0] / v[1]:.1f}/{v[1]}' for k, v in self.pk.items() if v[1] > 0)
            pk += ' ' + ' '.join(f'rw{k}={v[0] / v[1]:.3f}/{v[1]}' for k, v in self.rw.items() if v[1] > 0)
            self.log(f'GAME map={self.mission} points={self.round_points:.0f} gems={self.round_gems} falls={self.round_falls} rtf={self.env.rtf():.1f} pow={uses} {pk}')
        self.round_points = 0.0; self.round_gems = 0; self.round_falls = 0; self.round_uses = {}
        self.pk = {'on': [0.0, 0], 'off': [0.0, 0], 'none': [0.0, 0]}
        self.rw = {'fire': [0.0, 0], 'all': [0.0, 0]}
        self.round_id += 1
        self.hist = []

    def start_segment(self):
        tb = time.perf_counter()
        self.recovery_pending = False
        while True:
            try:
                if self.drill:
                    g = self._begin_drill()
                    if g is not None:
                        self.goal = g
                        break
                if self.real:
                    vis = []
                    for _ in range(60):               # a gem is normally visible at once; wait through a spawn gap
                        vis = visible_gems(self.env.msg.obs)
                        if vis:
                            break
                        self.segs._step_checked(self.env, 1)
                    if vis:
                        o = self.env.msg.obs
                        tgt, nxt = self.pick(vis, None, o[0:2], o[3:5])
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
        self.crop, self.vec, self.on_floor = self._build()
        self.prof['begin'] += time.perf_counter() - tb

    def _build(self):
        """The observation for the next decision: obs.build plus the V10 use state (40.26). In a drill the Super Speed
        has just been spent, so the held slot (and the blast meter) read empty whatever the game holds."""
        raw = self.env.msg.obs
        if self.drill_stage == 1 and len(raw) > RAW_POW_HELD:
            raw = np.array(raw, dtype=np.float64)
            raw[RAW_POW_HELD] = 0.0; raw[RAW_POW_BLAST] = 0.0; raw[RAW_POW_SPECIAL] = 0.0
        crop, vec, on_floor = self.obs_b.build(raw, self.goal, self.segs.next_goal())
        fill_use_state(vec, self.use_sent[0], self.use_sent[1], self.pending_since is not None, self.latched_aim,
                       (self.steps - self.ss_fire_step) * OBS_MS / 1000.0)
        if self.real and not self.drill and LIVE_RECORD:
            ng = self.segs.next_goal()
            self.hist.append({'raw': [float(q) for q in self.env.msg.obs], 'goal': list(self.goal),
                              'next': list(ng) if ng is not None else None,
                              'world': self.env.replay_world,
                              'use': [self.use_sent[0], self.use_sent[1], self.pending_since is not None,
                                      self.latched_aim, (self.steps - self.ss_fire_step) * OBS_MS / 1000.0]})
            if len(self.hist) > LIVE_HIST + max(LIVE_LOOKBACKS) + 2:
                del self.hist[0]
        return crop, vec, on_floor

    def _begin_drill(self):
        """40.26 Stage 1: teleport into a recorded post-kick state (engine variations of speed, heading, position);
        targets: the gem the kick was aimed at, then the gem after it. None if no start took."""
        if self.drill_stage == 4:
            s = self.drill_starts[int(self.rng.integers(len(self.drill_starts)))]
            return self.begin_replay(s)
        key = 'pre' if self.drill_stage >= 2 else 'post'
        self.warm = None
        for _ in range(5):
            if DRILL_EVAL:
                self.drill_i = getattr(self, 'drill_i', -1) + 1
                s = self.drill_starts[self.drill_i % len(self.drill_starts)]
                x, y, z, vx, vy, vz, wx, wy, wz = s[key]
            else:
                s = self.drill_starts[int(self.rng.integers(len(self.drill_starts)))]
                x, y, z, vx, vy, vz, wx, wy, wz = s[key]
                k = 1.0 + self.rng.uniform(-DRILL_SPEED_JITTER, DRILL_SPEED_JITTER)
                ang = math.radians(self.rng.uniform(-DRILL_ANGLE_JITTER, DRILL_ANGLE_JITTER))
                ca, sa = math.cos(ang), math.sin(ang)
                vx, vy = k * (vx * ca - vy * sa), k * (vx * sa + vy * ca)
                x += self.rng.uniform(-DRILL_POS_JITTER, DRILL_POS_JITTER); y += self.rng.uniform(-DRILL_POS_JITTER, DRILL_POS_JITTER)
            z += 0.02
            pts = [s['target'], s['next']]
            if self.drill_stage >= 2:
                if s.get('pre_goals'):
                    pts = [list(p) for p in s['pre_goals']] + [s['next']]   # the pickups on the way in, the kick's gem, the next
                elif math.hypot(s['pre_target'][0] - s['target'][0], s['pre_target'][1] - s['target'][1]) > 0.5:
                    pts = [s['pre_target']] + pts                           # the pickup that comes before the kick
            goals = [(gx, gy, self.terrain.floor_z(gx, gy, z) + DRILL_GEM_DZ) for gx, gy in pts]
            if self.drill_stage >= 2:
                # 40.27: a Super Speed held from the start (the server's own pickup path; lands during begin_drill's step)
                self.env.control(f'GIVEPOW {SS_DATABLOCK}')
            g = self.segs.begin_drill(self.env, (x, y, z, vx, vy, vz, wx, wy, wz), goals)
            if g is not None:
                self.drill_cur = {'t0': self.steps, 'n1': None, 'n2': None, 'v0': math.hypot(vx, vy), 'ret': 0.0,
                                  'ngoals': len(goals), 'fire': None, 'id': s.get('id', ''), 'falls': 0, 'picks': []}
                # stage 1: the start is the first decision after the kick; stage 2: no kick yet (V10 use state)
                self.ss_fire_step = (self.steps - 1) if self.drill_stage == 1 else -10**9
                self.pending_since = None; self.latched_aim = None
                self.use_sent = [False, False]
                return g
        return None

    def begin_replay(self, s):
        validate_start(s, self.replay_window_ms)
        self.replay_start_ms = restore_replay(self.env, s, self.replay_window_ms)
        self.recovery_pending = False
        self.segs.abandon()
        self.segs.pending_respawn = False; self.segs.pending_mark = None
        self.goal = tuple(s['goal'])
        self.segs.begin_replay(self.env.pos(), self.goal, s.get('next'))
        self.target = (*self.goal, 1.0, 0.0)  # chooser preserves the recorded first target
        self.obs_b.reset(); self.smooth_dir = None
        self.use_sent = [False, False]; self.pending_since = None; self.latched_aim = None
        self.ss_fire_step = -10**9; self.redirect_sent = -10**9
        self.redirect_open = []; self.fire_pending = None
        self.seg_steps = 0; self.ep_reward = 0.0
        self.drill_cur = {'id': s['id'], 't0': self.steps, 'n1': None, 'n2': None,
                          'v0': math.hypot(*s['world']['pose'][3:5]), 'ret': 0.0,
                          'ngoals': 0, 'fire': None, 'falls': 0, 'picks': [], 'points': 0.0}
        self.warm = []
        ob = ObsBuilder(self.terrain)
        for row in s.get('hist', []):
            c, v, _ = ob.build(np.asarray(row['raw']), tuple(row['goal']), row.get('next'))
            fill_use_state(v, *row.get('use', [False, False, False, None, USE_T_SCALE_WARM]))
            self.warm.append((c, v))
        self.crop, self.vec, self.on_floor = self._build()
        return self.goal

    def _write_live(self, rec):
        """Append a restorable approach fixture, even when the later manoeuvre fails."""
        if self.live_file is None:
            d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'nav')
            self.live_file = open(os.path.join(d, f'live_starts_{time.strftime("%Y%m%d_%H%M%S")}_{self.idx}.jsonl'), 'a', encoding='utf-8')
        self.live_file.write(json.dumps(rec) + '\n'); self.live_file.flush()
        self.live_written += 1

    def _record_approach(self):
        for lookback in LIVE_LOOKBACKS:
            self._record_approach_at(lookback)

    def _record_approach_at(self, lookback):
        if len(self.hist) < lookback + 2:
            return
        k = len(self.hist) - lookback - 1
        row = self.hist[k]
        if row['world'] is None:
            return
        group = f'{self.session_id}:{self.round_id}'
        s = {'schema': SCHEMA, 'id': f'{group}:{row["world"]["clock_ms"]}', 'split_group': group,
             'mission': self.mission, 'catalog': self.env.replay_catalog, 'world': row['world'],
             'raw': row['raw'], 'goal': row['goal'], 'next': row['next'],
             'hist': [{key: value for key, value in h.items() if key != 'world'} for h in self.hist[:k][-LIVE_HIST:]],
             'approach_ms': self.env.replay_world['clock_ms'] - row['world']['clock_ms'],
             'lookback_decisions': lookback,
             'behavior': os.environ.get('NAV_COLLECTION_BEHAVIOR', 'policy')}
        try:
            validate_start(s)
        except ValueError:
            self.live_rejected += 1
            return  # unsupported active effects/round tail; never fake a replacement state
        self._write_live(s)

    def _drill_done(self, outcome):
        c = self.drill_cur
        if c is None:
            return
        if self.drill_stage == 4:
            elapsed = self.env.replay_world['clock_ms'] - self.replay_start_ms
            self.last_drill_result = {'id': c['id'], 'status': 'ok', 'points': c['points'],
                                     'falls': c['falls'], 'pickup_times_ms': c['picks'],
                                     'fired': c['fire'] is not None, 'elapsed_ms': elapsed,
                                     'outcome': outcome}
            self.log('WINDOW ' + json.dumps(self.last_drill_result, separators=(',', ':')))
            self.drill_cur = None
            return
        a = self.drill_agg
        a['n'] += 1; a['ret'] += c['ret']
        if c['n1'] is not None:
            a['hit1'] += 1; a['t1'] += c['n1'] * OBS_MS / 1000.0
        if c['n2'] is not None:
            a['hit2'] += 1; a['t2'] += (c['n2'] - c['n1']) * OBS_MS / 1000.0
        if outcome == 'fell':
            a['fell'] += 1
        pre = 'DRILL' if self.drill_stage == 1 else f'DRILL{self.drill_stage}'
        fire = '' if self.drill_stage == 1 else (f'fired={int(c["fire"] is not None)} tf={(c["fire"] or 0) * OBS_MS / 1000.0:.2f} '
                                                 f'pk={c["ngoals"] - 2} ')
        extra = f' id={c["id"]} picks={len(c["picks"])} tpicks={",".join(f"{t * OBS_MS / 1000.0:.2f}" for t in c["picks"])} falls={c["falls"]}' if self.drill_stage == 4 else ''
        self.log(f'{pre} v0={c["v0"]:.1f} {fire}hit1={int(c["n1"] is not None)} t1={(c["n1"] or 0) * OBS_MS / 1000.0:.2f} '
                 f'hit2={int(c["n2"] is not None)} t2={((c["n2"] - c["n1"]) if c["n2"] else 0) * OBS_MS / 1000.0:.2f} '
                 f'out={outcome} ret={c["ret"]:.1f}{extra}')
        if a['n'] % 100 == 0:
            self.log(f'DRILLSUM n={a["n"]} hit1={100.0 * a["hit1"] / a["n"]:.0f}% hit2={100.0 * a["hit2"] / a["n"]:.0f}% '
                     f'fell={100.0 * a["fell"] / a["n"]:.0f}% t1={a["t1"] / max(a["hit1"], 1):.2f}s t2={a["t2"] / max(a["hit2"], 1):.2f}s '
                     f'ret={a["ret"] / a["n"]:.1f}')
            self.drill_agg = {'n': 0, 'hit1': 0, 'hit2': 0, 'fell': 0, 't1': 0.0, 't2': 0.0, 'ret': 0.0}
        self.drill_cur = None

    # ------------------------------------------------------------------ one decision
    def step(self, a_game):
        clock_before = self.env.replay_world['clock_ms'] if self.env.replay_world else None
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
        # the use bit (action element 5, POWERUP_PLAN phase 4): fire the held powerup, else the blast meter when it
        # is usable; a Super Speed goes along the commanded direction (the bridge latches that yaw over the key lag)
        use_pow = use_blast = 0; pow_yaw = None
        # 40.18: how often is a Super Speed use APPROVED (held, on the floor, waypoint >= 8 u, kick survivable)? 'ap2'
        if (self.vec is not None and self.vec[VEC_POW + 1] > 0.5 and self.vec[VEC_POW + 26] > 0.5 and self.vec[8] > 0.5
                and self.vec[2] * WAYPOINT_DIST_SCALE >= 8.0):
            self.round_uses['ap2'] = self.round_uses.get('ap2', 0) + 1
        held = int(prev_obs[RAW_POW_HELD]) if len(prev_obs) > RAW_POW_HELD else 0
        sent = False
        # 40.26: Super Speed only (the other powerups and the blast are not fired in this experiment); a drill instance
        # fires nothing (its kick already happened)
        if len(a) >= 8 and a[5] > 0.5 and held == 2 and self.drill_stage != 1 and not DRILL_NO_USE:
            use_pow = 1
            key = 'u2'
            ax, ay = ss_aim(self.vec)             # the physics aim (the kick onto the line to the waypoint)
            pow_yaw = math.atan2(ax, ay)
            # 40.15: 'r2' = a turn (velocity more than 30 deg off the waypoint line), 'u2' = a run
            spd = math.hypot(self.vec[4], self.vec[5])
            if spd > 1e-6 and (self.vec[4] * self.vec[0] + self.vec[5] * self.vec[1]) / spd < 0.866:
                key = 'r2'; self.redirect_sent = self.steps
            self.round_uses[key] = self.round_uses.get(key, 0) + 1
            sent = True
            if self.pending_since is None:
                self.pending_since = self.steps
            self.latched_aim = (ax, ay)           # the bridge re-latches the yaw on every use word
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
        if self.real and self.recovery_pending:
            # Recovery occupies an ordinary timed transition, with its own score delta.
            self.recovery_pending = False
            self.env.round_ended = False; self.env.reconnected = False
            self.env.control('OOBCLICK')
            msg = self.env.msg
            info = {'fell': bool(msg.oob), 'gem_delta': msg.gem_delta,
                    'round_ended': self.env.round_ended, 'reconnected': self.env.reconnected}
            sent = False
        else:
            msg, info = self.env.step(js, use_pow=use_pow, pow_yaw=pow_yaw, use_blast=use_blast)
        self.prof['game'] += time.perf_counter() - tg
        held_now = int(msg.obs[RAW_POW_HELD]) if len(msg.obs) > RAW_POW_HELD else 0
        self.use_sent = [sent, self.use_sent[0]]                   # 40.26 V10 use state
        if held > 0 and held_now == 0 and not info['fell'] and self.drill_stage != 1:   # the held powerup went: fired (or lost to a respawn)
            if self.drill_stage >= 2 and held == 2 and self.drill_cur is not None and self.drill_cur['fire'] is None:
                self.drill_cur['fire'] = self.steps - self.drill_cur['t0'] + 1
            self.round_uses[f'f{held}'] = self.round_uses.get(f'f{held}', 0) + 1
            if held == 2:
                self.ss_fire_step = self.steps; self.fire_pending = self.steps
                self.pending_since = None; self.latched_aim = None
                if self.real and not self.drill and LIVE_RECORD:
                    self._record_approach()
                if self.steps - self.redirect_sent <= 3:
                    self.redirect_open.append([self.steps, self.round_gems])
        if self.pending_since is not None and (self.steps - self.pending_since > 3 or held_now != 2):
            self.pending_since = None; self.latched_aim = None    # no kick came (or the item went another way)
        if self.redirect_open and self.steps - self.redirect_open[0][0] >= 24:   # 1.5 s: the waypoint after a turn is >= 8 u away
            _, g0 = self.redirect_open.pop(0)
            k = 'rhit' if self.round_gems > g0 else 'rmiss'
            self.round_uses[k] = self.round_uses.get(k, 0) + 1
        if self.fire_pending is not None and self.steps - self.fire_pending >= 3:
            sp3 = math.hypot(float(msg.obs[3]), float(msg.obs[4]))
            self.fire_bucket = 'A' if sp3 < 18.0 else ('B' if sp3 <= 24.0 else 'C')
            self.round_uses['ks' + self.fire_bucket] = self.round_uses.get('ks' + self.fire_bucket, 0) + 1
            self.fire_pending = None
        if info['fell'] and self.steps - self.ss_fire_step <= 47:   # a fall within 3 s of a Super Speed fire
            self.round_uses['fss'] = self.round_uses.get('fss', 0) + 1
            if self.fire_bucket is not None:
                self.round_uses['kf' + self.fire_bucket] = self.round_uses.get('kf' + self.fire_bucket, 0) + 1
        if self.real and info['gem_delta'] > 0:
            # pickup speed by the redirect flag: on (Super Speed held, redirect possible), off (held, not possible), none
            grp = ('on' if self.vec[VEC_POW + 27] > 0.5 else 'off') if held == 2 else 'none'
            self.pk[grp][0] += math.hypot(v_before[0], v_before[1]); self.pk[grp][1] += 1
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
        if (not self.drill and self.seg_steps == FRAME_CHECK_N and self.fc_den >= 5
                and self.fc_num / self.fc_den < FRAME_FLIP_COS):   # 40.26: a drill starts at 15-24 u/s, decelerating whatever the input
            self.frame_flips += 1
            self.log(f'FRAME FLIP: command/acceleration cosine {self.fc_num / self.fc_den:.2f} over the first {FRAME_CHECK_N} decisions at '
                     f'{np.round(msg.obs[:3], 1)}; forcing a respawn and restarting the segment (flips on this instance {self.frame_flips})')
            k = self.seg_steps - 1
            self.segs.abandon(); self.ep_reward = 0.0
            self.env.control('RESPAWN')
            self.start_segment()
            return self._reply(skip=True, truncate=k)
        if info['round_ended'] or info['reconnected']:
            if self.drill_stage == 4 and DRILL_EVAL:
                raise RuntimeError('round ended or connection changed during named replay')
            ended = bool(info['round_ended'])
            final_reward = float(info['gem_delta'])
            ep = self.ep_reward + final_reward
            round_stats = None
            if ended and self.real and not self.drill and self.segs.seg is not None:
                # 2026-10-05 22:40 (operator: the speed, falls and seconds-per-pickup cards were empty with every game on
                # full rounds): a real round is one whole segment; record it so SegmentManager.stats() reports it
                self.segs.seg.outcome = 'round'
                self.segs._record(self.segs.seg, 'round')
                round_stats = self.segs.stats()
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
            return self._reply(skip=not ended, truncate=0, r=final_reward, done=ended, outcome='round',
                               new_segment=True, trace='', stats=round_stats, ep_reward=ep, duration_steps=1.0)
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
                                          cmd_dir=(a[6], a[7]) if len(a) > 7 else (a[0], a[1]),   # after the use bit (6 action elements)
                                          picked=float(info['gem_delta']) if self.real else 0.0)
        if self.drill and self.drill_cur is not None and self.segs.seg is not None:   # 40.26: drill outcome bookkeeping
            if r >= 1.0 - 1e-6 and self.drill_stage == 4 and self.segs.drill_window:   # a pickup in the window (points reward)
                self.drill_cur['picks'].append(self.env.replay_world['clock_ms'] - self.replay_start_ms)
                self.drill_cur['points'] += r
            if info['fell']:
                self.drill_cur['falls'] += 1
            c = self.segs.seg.collected - (self.drill_cur['ngoals'] - 2)   # the kick's gem and the one after (stage 2 may
            if c >= 1 and self.drill_cur['n1'] is None:                    # have the pickup before them as a first target)
                self.drill_cur['n1'] = self.steps - self.drill_cur['t0'] + 1
            if c >= 2 and self.drill_cur['n2'] is None:
                self.drill_cur['n2'] = self.steps - self.drill_cur['t0'] + 1
            self.drill_cur['ret'] += r
        need_respawn = self.segs.take_respawn()
        if need_respawn and self.real:
            self.recovery_pending = True
            self.obs_b.reset()
        elif need_respawn:
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
            if not self.real:
                self.env.mark(mk[0], mk[1], mk[2])
            # ...and retarget the OBSERVATION onto it. Without this the segment manager chases the
            # new gem (reward, arrival, prev_d) while the policy still sees the one it just
            # collected, sitting at its own feet: it orbits a dead waypoint until the timeout.
            self.goal = self.segs.seg.goal
        self.ep_reward += r
        if self.real:
            self.rw['all'][0] += r; self.rw['all'][1] += 1
            if 0 <= self.steps - self.ss_fire_step < 30:
                self.rw['fire'][0] += r; self.rw['fire'][1] += 1
        o = msg.obs
        if self.drill_stage == 4:
            done = self.env.replay_world['clock_ms'] - self.replay_start_ms >= self.replay_window_ms
            outcome = 'window' if done else None
        trace = (f'{self.seg_count},{self.env.time_left_s():.2f},{o[0]:.2f},{o[1]:.2f},{o[2]:.2f},{o[3]:.2f},{o[4]:.2f},{o[5]:.2f},'
                 f'{int(self.on_floor)},{js[0]:.2f},{js[1]:.2f},{js[2]:.2f},{js[3]:.2f},{js[4]},{int(a[4] > 0.5)},{int(info["fell"])},'
                 f'{r:.2f},{int(done)},{outcome or ""},{self.segs.seg.prev_d:.1f},{self.goal[0]:.1f},{self.goal[1]:.1f}')
        self.steps += 1
        duration_steps = ((self.env.replay_world['clock_ms'] - clock_before) / OBS_MS
                          if clock_before is not None else 1.0)
        if self.real:
            vis = visible_gems(msg.obs)
            if vis:
                tgt, nxt = self.pick(vis, self.target, msg.obs[0:2], msg.obs[3:5])
                g = self.segs.seg.goal
                want = (nxt[:3] if nxt else None)
                if tgt is not None and math.hypot(tgt[0] - g[0], tgt[1] - g[1]) > STICKY_TOL:
                    self.segs.retarget(float(msg.obs[0]), float(msg.obs[1]), tgt[:3], want)
                    self.goal = self.segs.seg.goal
                    # Cosmetic MARK controls advance the simulation; omit them in scored transitions.
                elif tgt is not None:
                    nn = self.segs.seg.real_next
                    if (nn is None) != (want is None) or (want is not None and math.hypot(want[0] - nn[0], want[1] - nn[1]) > STICKY_TOL):
                        self.segs.set_next(float(msg.obs[0]), float(msg.obs[1]), want)
                self.target = tgt
        stats = None; ep = None
        if done:
            if self.drill_stage == 4:
                self.segs.seg.outcome = outcome
                self.segs._record(self.segs.seg, outcome)
            self.seg_count += 1; ep = self.ep_reward; self.ep_reward = 0.0
            stats = self.segs.stats()
            self.final_obs = None
            if self.drill:
                fc, fv, _ = self._build()              # 40.40: the drill's final observation (the trainer bootstraps its value)
                self.final_obs = (fc, fv)
                self._drill_done(outcome)
            if not DRILL_EVAL or self.drill_stage != 4:
                self.start_segment()
        else:
            tg = time.perf_counter()
            self.crop, self.vec, self.on_floor = self._build()
            self.prof['obs'] += time.perf_counter() - tg
        return self._reply(skip=False, r=r, done=done, outcome=outcome, new_segment=bool(done), fell=info['fell'],
                           trace=trace, stats=stats, ep_reward=ep,
                           duration_steps=duration_steps,
                           truncated=bool(done and outcome == 'window'))

    def _reply(self, **kw):
        kw.update({'crop': self.crop, 'vec': self.vec, 'mission': self.mission, 'log': self.drain_log(),
                   'flips': self.frame_flips, 'seg_count': self.seg_count, 'prof': self.prof,
                   'drill': bool(self.drill)})            # 40.33: the trainer bootstraps a drill's end (train_nav v_cont)
        if kw.get('done') and self.final_obs is not None:   # 40.40: the ended drill's final observation
            kw['final_crop'], kw['final_vec'] = self.final_obs; self.final_obs = None
        if self.warm is not None:                             # 40.40: GRU warm-up observations for a live start
            kw['warm'] = self.warm; self.warm = None
        self.prof = {'game': 0.0, 'obs': 0.0, 'begin': 0.0}
        if kw.get('skip'):
            kw['new_segment'] = True
        return kw


def worker_main(conn, idx, port, seed, arrive_r, arrive_dz):
    w = InstanceWorker(idx, port, seed, arrive_r, arrive_dz)
    try:
        w.connect()
        conn.send({'ready': True, **w._reply()})
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
