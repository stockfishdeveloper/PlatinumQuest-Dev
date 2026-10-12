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
from nav.obs import ObsBuilder, VEC_GAP, VEC_POW, ss_aim, ss_redirect_window, WAYPOINT_DIST_SCALE, fill_use_state, USE_T_SCALE, VEC_USE_TRY   # noqa: E402
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
PRACTICE_MODE = os.environ.get('NAV_PRACTICE_MODE', 'judgment')
if PRACTICE_MODE not in ('handling', 'judgment'):
    raise ValueError('NAV_PRACTICE_MODE must be handling or judgment')
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
from nav.routing import RoutePlanner, goal_key
from nav.recovery import (ProgressMonitor, simulation_time, potential_term,
                          legacy_progress_term, shaping_configuration)
from nav.ppo_recurrent import GAMMA
REWARD_CONFIGURATION = shaping_configuration(GAMMA)
SHAPING_MODE = REWARD_CONFIGURATION['mode']
SHAPING_GAMMA = REWARD_CONFIGURATION['gamma']
POTENTIAL_K = REWARD_CONFIGURATION['k']
POTENTIAL_GROUND_U = REWARD_CONFIGURATION['floor_hold_u']
FALL_CHARGE = bool(REWARD_CONFIGURATION.get('fall_charge', False))   # 40.95 (operator 10-11: "You can charge every fall")
USE_TRY_P = float(os.environ.get('NAV_USE_TRY_P', '0.25'))      # 40.95: share of pickups / blast charges drawn as TRY items
from nav.model import ESCAPE_POWERUPS
DIRECT_GEM = os.environ.get('NAV_DIRECT_GEM', '0') == '1'
COMPETITIVE = os.environ.get('NAV_COMPETITIVE', '0') == '1'
RESCUE_S = float(os.environ.get('NAV_RESCUE_S', '0'))
if COMPETITIVE and RESCUE_S > 0:
    raise ValueError('stuck rescue is unavailable in competitive Hunt')
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
        self.route = None
        self.phi_prev = None; self.phi_key = None       # potential shaping state (recovery.potential_term)
        self.round_shaping = 0.0; self.shape_frozen = 0
        self.fall_charge = None; self.round_fall_charge = 0.0   # 40.95: a fall awaiting its charge; the round's total
        self.ground_ref = None; self.try_respawn_step = -10 ** 9   # 40.97
        self._reset_try()
        self.recovery = ProgressMonitor(window_s=RESCUE_S or 8.0)
        self.pathed = False
        self.map_rounds = 0
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
        self.last_use_request = None
        self.practice_recorder = None
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
        if os.environ.get('NAV_PRACTICE_RECORD_DIR'):
            from nav.escape_practice import EscapeRecorder
            self.practice_recorder = EscapeRecorder(os.environ['NAV_PRACTICE_RECORD_DIR'],
                                                     f'{self.session_id}_{self.idx}', name)
        self.segs = SegmentManager(t, self.rng, self.log, arrive_r=self.arrive_r, arrive_dz=self.arrive_dz)
        self.real = REAL_GEMS and bool(getattr(t, 'gem_spawns', None)); self.segs.real_mode = self.real
        self.route = RoutePlanner(t)
        self.recovery.reset()
        self.map_rounds = 0
        self._rotation_state()
        self.drill_stage = drill_stage_of(self.idx); self.drill = self.drill_stage > 0
        if self.drill:
            if self.drill_stage != 4 and not DRILL_EVAL:
                raise ValueError('synthetic stages 1-3 are diagnostics only; points training uses stage 4')
            src = {2: DRILL2_STARTS, 3: DRILL3_STARTS, 4: DRILL4_STARTS}.get(self.drill_stage, DRILL_STARTS)
            if self.drill_stage == 4:
                if DRILL_EVAL:
                    self.drill_starts = self.eval_starts or []
                else:
                    self.drill_starts = [s for s in load_starts(src) if s['mission'] == name]
                if not self.drill_starts and not DRILL_EVAL:
                    self.log(f'no replay starts for {name}: ordinary real-map rounds')
                    self.drill = False; self.drill_stage = 0
                else:
                    self.real = True; self.segs.real_mode = True
                    self.segs.drill_mode = True; self.segs.drill_window = True
                    self.tour_fields = {}
                    self.log(f'real-game replay: {len(self.drill_starts)} fixtures, game points, {WINDOW_MS} ms')
                    return
            allst = []
            if self.drill:
                try:
                    with open(src) as file:
                        allst = json.load(file)
                except Exception as e:
                    self.log(f'drill starts {src} not readable: {e}')
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
        self.route = RoutePlanner(t)
        self.recovery.reset()
        self.map_rounds = 0
        self._rotation_state()
        self.log(f'gem chooser: {"whole-spawn planner (" + TOUR + ")" if TOUR else "greedy"}')

    def walk_dist(self, a, g):
        key = (round(g[0], 1), round(g[1], 1), round(g[2], 1))
        f = self.tour_fields.get(key)
        if f is None:
            f = self.tour_fields[key] = self.terrain.goal_field(g[0], g[1], jumps=False, z=g[2])
        return self.terrain.dist_at(f, float(a[0]), float(a[1]), (g[0], g[1]), z=(a[2] if len(a) > 2 else None))

    def pick(self, vis, current, pos, vel):
        """Target and next gem: the whole-spawn planner by default (TOUR), else the greedy chooser."""
        if TOUR:
            return plan_tour(vis, current, pos, vel, dist=(self.walk_dist if TOUR == 'walk' else None))   # pos carries z (height-aware tour)
        return choose(vis, current, pos, vel)

    def sync_map(self):
        if not self.env.info.get('mission'):
            self.env.request_info()
        if self.env.info.get('mission') and self.env.info['mission'] != self.mission:
            self.load_map(self.env.info['mission'])
            self.log(f'MAP {self.mission}')

    def configure_reconnected_game(self):
        self.env.request_info(); self.env.set_speed(GAME_SPEED)
        self.env.set_game_rules(COMPETITIVE)
        if LIVE_RECORD or self.drill_stage == 4 or RESCUE_S > 0:
            for attempt in range(4):           # 2026-10-10 02:15: block B lost a restart to this after a map rotation; ask again, never die
                self.env.control('REPLAYCAPTURE 1')
                for _ in range(64):
                    if self.env.replay_world is not None:
                        break
                    self.env.step((0, 0, 0, 0, 0), repeat=1)
                if self.env.replay_world is not None:
                    break
            if self.env.replay_world is None:
                self.log('WARNING: reconnected game sent no replay telemetry; progress timing falls back to the round clock')

    def connect(self):
        self.env.connect()
        self.env.set_game_rules(COMPETITIVE)
        for attempt in range(10):
            if self.env.info.get('mission'):
                break
            self.log(f'no INFO answer yet (attempt {attempt + 1}); asking again')
            self.env.request_info(wait_ticks=60)
        if not self.env.info.get('mission'):
            raise RuntimeError(f'instance {self.idx} never answered INFO; is mlAgent.cs up to date (delete the .dso)?')
        self.load_map(self.env.info['mission'])
        if LIVE_RECORD or self.drill_stage == 4 or RESCUE_S > 0:
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

    def _rotation_state(self):
        directory = os.environ.get('NAV_ROTATION_STATE_DIR')
        if not directory:
            return
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f'instance_{self.idx}.json')
        tmp = path + f'.{os.getpid()}.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(dict(mission=self.mission, rounds=self.map_rounds, worker_pid=os.getpid()), f)
            os.replace(tmp, path)
        except OSError as e:                      # 2026-10-09 21:50: a locked state file (OneDrive) must never kill a worker
            self.log(f'rotation state not written ({e}); the game loop keeps the current map')
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _round_over(self):
        """Every path that crosses a round end goes through here: log the finished round's real
        score once, reset the counters. (Before 2026-09-23 17:00 the RoundOver paths inside
        start_segment/recover skipped both, so a round ending mid-respawn merged into the next.)"""
        if self.real and self.env.round_ended:
            self.map_rounds += 1
            self._rotation_state()
            uses = ','.join(f'{k}:{v}' for k, v in sorted(self.round_uses.items())) or '-'
            pk = ' '.join(f'pk{k}={v[0] / v[1]:.1f}/{v[1]}' for k, v in self.pk.items() if v[1] > 0)
            pk += ' ' + ' '.join(f'rw{k}={v[0] / v[1]:.3f}/{v[1]}' for k, v in self.rw.items() if v[1] > 0)
            recovery = ' '.join(f'{k}={v}' for k, v in self.recovery.stats().items())
            self.log(f'GAME map={self.mission} points={self.round_points:.0f} gems={self.round_gems} falls={self.round_falls} rtf={self.env.rtf():.1f} pow={uses} {pk} {recovery} shape={self.round_shaping:.2f} shape_frozen={self.shape_frozen} fall_charge={self.round_fall_charge:.2f} '
                     f'try_items={self.try_counts[0]}/{self.try_counts[1]}')
        self.recovery.reset()
        self.phi_prev = None; self.phi_key = None; self.round_shaping = 0.0; self.shape_frozen = 0
        self.fall_charge = None; self.round_fall_charge = 0.0; self.ground_ref = None
        self._reset_try()
        if self.route is not None:
            self.route.reset()
        self.round_points = 0.0; self.round_gems = 0; self.round_falls = 0; self.round_uses = {}
        self.pk = {'on': [0.0, 0], 'off': [0.0, 0], 'none': [0.0, 0]}
        self.rw = {'fire': [0.0, 0], 'all': [0.0, 0]}
        self.round_id += 1
        self.hist = []
        if self.practice_recorder is not None:
            self.practice_recorder.history.clear()

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
                        tgt, nxt = self.pick(vis, None, o[0:3], o[3:5])
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
                    self.configure_reconnected_game()
                self.sync_map()
                self.log(f'new round (game connection {self.env.connections})')
        self.obs_b.reset()
        self.smooth_dir = None            # the marble was just teleported: old heading is meaningless
        self.seg_steps = 0; self.fc_num = self.fc_den = 0.0
        self.crop, self.vec, self.on_floor = self._build()
        self.prof['begin'] += time.perf_counter() - tb

    def _reset_try(self):
        """40.95: per-item TRY state (held type, pickup and blast-charge sequence numbers, the usable item's identity and
        its drawn flag); try_counts = [TRY items, all usable items] this round."""
        self.try_held = 0; self.try_seq = 0; self.try_meter = 0.0; self.try_special = False; self.try_blast_seq = 0
        self.try_last_type = 0; self.try_zero_step = -10 ** 9
        self.try_ident = None; self.try_flag = False; self.try_counts = [0, 0]; self.try_seen = {}

    def _try_flag(self, raw):
        """1.0 if the powerup or blast charge usable now was drawn as a TRY item: one draw (USE_TRY_P) when a new item
        becomes usable (a pickup, or a blast charge after the previous blast went off). A held slot that blinks to 0
        and back to the same type within 2 s (a respawn) is the same item."""
        if len(raw) <= RAW_POW_SPECIAL:
            return 0.0
        held = int(raw[RAW_POW_HELD]); meter = float(raw[RAW_POW_BLAST]); special = float(raw[RAW_POW_SPECIAL]) > 0.5
        if held != self.try_held:
            blink = (held == self.try_last_type and self.steps - self.try_zero_step <= 30
                     and self.steps - getattr(self, 'try_respawn_step', -10 ** 9) <= 60)   # 40.97: only after a respawn
            if held > 0 and not blink:
                self.try_seq += 1
            if held == 0 and self.try_held > 0:
                self.try_last_type = self.try_held; self.try_zero_step = self.steps
            self.try_held = held
        if meter < self.try_meter - 0.05 or (self.try_special and not special):
            self.try_blast_seq += 1                       # a blast went off: the next charge is a new item
        self.try_meter = meter; self.try_special = special
        if held > 0:
            ident = ('item', held, self.try_seq)
        elif meter >= BLAST_REQUIRED or special:
            ident = ('blast', self.try_blast_seq)
        else:
            ident = None
        if ident != self.try_ident:
            self.try_ident = ident
            if ident is None:
                self.try_flag = False
            elif ident in self.try_seen:                  # the same item again (e.g. after a held-slot blink)
                self.try_flag = self.try_seen[ident]
            else:
                self.try_flag = bool(self.rng.random() < USE_TRY_P)
                self.try_seen[ident] = self.try_flag
                self.try_counts[0] += int(self.try_flag); self.try_counts[1] += 1
        return 1.0 if self.try_flag else 0.0

    def _build(self):
        """The observation for the next decision: obs.build plus the V10 use state (40.26). In a drill the Super Speed
        has just been spent, so the held slot (and the blast meter) read empty whatever the game holds."""
        raw = self.env.msg.obs
        if self.drill_stage == 1 and len(raw) > RAW_POW_HELD:
            raw = np.array(raw, dtype=np.float64)
            raw[RAW_POW_HELD] = 0.0; raw[RAW_POW_BLAST] = 0.0; raw[RAW_POW_SPECIAL] = 0.0
        nav_goal = self.goal
        self.pathed = False
        if self.real and (not self.drill or self.drill_stage == 4) and not DIRECT_GEM:
            nav_goal, self.pathed = self.route.navigate(raw[:3], self.goal)
        crop, vec, on_floor = self.obs_b.build(raw, nav_goal, self.segs.next_goal(), gem_goal=self.goal)
        fill_use_state(vec, self.use_sent[0], self.use_sent[1], self.pending_since is not None, self.latched_aim,
                       (self.steps - self.ss_fire_step) * OBS_MS / 1000.0)
        vec[VEC_USE_TRY] = self._try_flag(raw) if (self.real and not self.drill) else 0.0   # 40.95 (not a network input)
        recorder = getattr(self, 'practice_recorder', None)
        if recorder is not None and self.real and not self.drill:
            recorder.observe(self.env, self.goal, nav_goal, self.segs.next_goal(), vec,
                             [*self.use_sent, self.pending_since is not None, self.latched_aim,
                              (self.steps - self.ss_fire_step) * OBS_MS / 1000.0], self.round_id)
        if self.real and not self.drill and LIVE_RECORD:
            ng = self.segs.next_goal()
            self.hist.append({'raw': [float(q) for q in self.env.msg.obs], 'goal': list(nav_goal), 'gem_goal': list(self.goal),
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

    def begin_replay(self, s, handling=None):
        validate_start(s, self.replay_window_ms)
        self.replay_start_ms = restore_replay(self.env, s, self.replay_window_ms)
        self.recovery_pending = False
        self.route.reset(); self.recovery.reset()
        self.phi_prev = None; self.phi_key = None; self.fall_charge = None; self.ground_ref = None
        self.segs.abandon()
        self.segs.pending_respawn = False; self.segs.pending_mark = None
        self.goal = tuple(s['goal'])
        self.segs.begin_replay(self.env.pos(), self.goal, s.get('next'))
        self.target = (*self.goal, 1.0, 0.0)  # chooser preserves the recorded first target
        self.obs_b.reset(); self.smooth_dir = None
        self.use_sent = [False, False]; self.pending_since = None; self.latched_aim = None
        self.last_use_request = None
        self.ss_fire_step = -10**9; self.redirect_sent = -10**9
        self.redirect_open = []; self.fire_pending = None
        self.seg_steps = 0; self.ep_reward = 0.0
        self.drill_cur = {'id': s['id'], 't0': self.steps, 'n1': None, 'n2': None,
                          'v0': math.hypot(*s['world']['pose'][3:5]), 'ret': 0.0,
                          'ngoals': 0, 'fire': None, 'falls': 0, 'picks': [], 'points': 0.0,
                          'activation_events': [], 'use_requests': {}}
        self.warm = []
        ob = ObsBuilder(self.terrain)
        for row in s.get('hist', []):
            c, v, _ = ob.build(np.asarray(row['raw']), tuple(row['goal']), row.get('next'), gem_goal=row.get('gem_goal'))
            fill_use_state(v, *row.get('use', [False, False, False, None, USE_T_SCALE_WARM]))
            self.warm.append((c, v))
        self.crop, self.vec, self.on_floor = self._build()
        self.drill_cur['mode'] = 'judgment'
        self.drill_cur['resource'] = int(self.env.msg.obs[RAW_POW_HELD])
        self.drill_cur['reset_activation_ms'] = 0
        activate = handling if handling is not None else PRACTICE_MODE == 'handling'
        if activate:
            self._activate_handling_reset()
        return self.goal

    def _activate_handling_reset(self):
        """Consume recorded SJ/SS/Helicopter/blast before requesting a PPO action."""
        from nav.protocol import NOOP_ACTION
        held = int(self.env.msg.obs[RAW_POW_HELD])
        charge = float(self.env.msg.obs[RAW_POW_BLAST])
        if held not in (1, 2, 5) and not (held == 0 and charge >= BLAST_REQUIRED):
            raise ValueError('handling fixture needs recorded SJ/SS/Helicopter or charged blast')
        clock = self.env.replay_world['clock_ms']
        next_goal = self.segs.next_goal()
        setup_points = 0.0
        consumed = False
        for _ in range(8):
            self.warm.append((self.crop.copy(), self.vec.copy()))
            msg, info = self.env.step(NOOP_ACTION, use_pow=int(held > 0),
                                      use_blast=int(held == 0), repeat=1)
            if info['fell'] or info['round_ended'] or info['reconnected']:
                raise RuntimeError('handling reset activation crossed a fall or game boundary')
            setup_points += float(info['gem_delta'])
            self.use_sent = [True, self.use_sent[0]]
            self.crop, self.vec, self.on_floor = self._build()
            consumed = (int(msg.obs[RAW_POW_HELD]) != held if held else
                        float(msg.obs[RAW_POW_BLAST]) < charge - .05)
            if consumed:
                break
        if not consumed:
            raise RuntimeError('handling reset activation was not confirmed by engine inventory')
        # Release the key; none of these reset transitions enter the rollout or its reward.
        self.warm.append((self.crop.copy(), self.vec.copy()))
        _, info = self.env.step(NOOP_ACTION, repeat=1)
        if info['fell'] or info['round_ended'] or info['reconnected']:
            raise RuntimeError('handling reset release crossed a fall or game boundary')
        setup_points += float(info['gem_delta'])
        self.use_sent = [False, True]
        self.replay_start_ms = self.env.replay_world['clock_ms']
        self.drill_cur.update(mode='handling', reset_activation_ms=self.replay_start_ms - clock,
                              reset_points=setup_points)
        self.segs.begin_replay(self.env.pos(), self.goal, next_goal)
        self.crop, self.vec, self.on_floor = self._build()

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
        # Reconnect clears current telemetry before the new mission is configured.
        # Never pair an old history row with a missing/new connection catalog.
        world = self.env.replay_world
        if world is None or self.env.replay_catalog is None or self.env.reconnected:
            return
        if len(self.hist) < lookback + 2:
            return
        k = len(self.hist) - lookback - 1
        row = self.hist[k]
        if row['world'] is None or world['clock_ms'] < row['world']['clock_ms']:
            return
        group = f'{self.session_id}:{self.round_id}'
        s = {'schema': SCHEMA, 'id': f'{group}:{row["world"]["clock_ms"]}', 'split_group': group,
             'mission': self.mission, 'catalog': self.env.replay_catalog, 'world': row['world'],
             'raw': row['raw'], 'goal': row.get('gem_goal', row['goal']), 'next': row['next'],
             'hist': [{key: value for key, value in h.items() if key != 'world'} for h in self.hist[:k][-LIVE_HIST:]],
             'approach_ms': world['clock_ms'] - row['world']['clock_ms'],
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
                                     'mode': c.get('mode', 'judgment'), 'resource': c.get('resource'),
                                     'reset_activation_ms': c.get('reset_activation_ms', 0),
                                     'reset_points': c.get('reset_points', 0),
                                     'activation_events': c.get('activation_events', []),
                                     'use_requests': c.get('use_requests', {}),
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

    def _phi(self, pos):
        """-(route distance) at pos for the current gem, or None where it is undefined (no route, or no floor level
        within POTENTIAL_GROUND_U of the marble height: over a void, high in a flight) so the caller keeps the last."""
        if self.route is None or self.goal is None:
            return None
        if self.route.key != goal_key(self.goal):
            self.route.navigate(pos, self.goal)
        x, y, z = (float(v) for v in pos[:3])
        j, i = self.terrain.cell_of(x, y)
        k = self.terrain.level_at(j, i, z)
        if k < 0 or abs(float(self.terrain.walk_z[k, j, i]) - z) > POTENTIAL_GROUND_U:
            return None
        d = self.route.distance((x, y, z))
        return -float(d) if (math.isfinite(d) and d < 1e6) else None

    def _shaping(self, pos_after, duration_steps, terminal, key_before):
        """The potential term for this transition; updates the shaping state. 0 unless points mode with POTENTIAL_K > 0."""
        if POTENTIAL_K <= 0.0 or not (self.real and (not self.drill or
                                                   (self.drill_stage == 4 and SHAPING_MODE == 'discounted'))):
            return 0.0
        key_after = goal_key(self.goal) if self.goal is not None else None
        changed = key_after != key_before
        phi_after = None if terminal else self._phi(pos_after)
        if phi_after is None and not terminal:
            self.shape_frozen += 1
        # 2026-10-11 (operator: "You can charge every fall"; log 40.95): in the legacy mode a target switch zeroes the
        # term and re-bases the potential, so a fall followed by a new target at the respawn (27 % of A2's falls) was
        # never charged while the progress made before it had been paid. Now a fall is charged against the gem it was
        # chasing (its route field at the fall) as soon as the marble is back on the floor, whatever the new target.
        if SHAPING_MODE != 'discounted' and getattr(self, 'fall_charge', None) is not None:
            if terminal:
                self.fall_charge = None                       # round over: the ordinary terminal handling below
            elif phi_after is None:
                return 0.0                                    # still in the air / over the void: the charge waits
            else:
                ogoal, ofield = self.fall_charge
                self.fall_charge = None
                x, y, z = (float(v) for v in pos_after[:3])
                d_old = self.terrain.dist_at(ofield, x, y, ogoal[:2], z=z)
                term = (POTENTIAL_K * (-float(d_old) - self.phi_prev)
                        if (self.phi_prev is not None and math.isfinite(d_old) and d_old < 1e6) else 0.0)
                self.round_fall_charge = getattr(self, 'round_fall_charge', 0.0) + term
                self.phi_prev = phi_after; self.phi_key = key_after
                self._ground_ref(phi_after, key_after)
                self.round_shaping += term
                return term
        if SHAPING_MODE == 'discounted':
            term = potential_term(self.phi_prev, phi_after, duration_steps, SHAPING_GAMMA, POTENTIAL_K,
                                  terminal=terminal)
        else:
            term = legacy_progress_term(self.phi_prev, phi_after, POTENTIAL_K,
                                        terminal=terminal, goal_changed=changed)
        if terminal:
            self.phi_prev = None; self.phi_key = None
        elif changed and SHAPING_MODE != 'discounted':
            self.phi_prev = self._phi(pos_after); self.phi_key = key_after
        else:
            if phi_after is not None:
                self.phi_prev = phi_after
            self.phi_key = key_after
        if not terminal and phi_after is not None:
            self._ground_ref(phi_after, key_after)
        elif terminal:
            self.ground_ref = None
        self.round_shaping += term
        return term

    def _ground_ref(self, phi, key):
        """40.97 (review): the gem, its route field and the potential at the last decision on a route floor. A fall is
        charged against this, so a target switch while airborne, before the out-of-bounds step, cannot skip the charge."""
        route = getattr(self, 'route', None)
        if self.goal is None or route is None or route.key != key:
            self.ground_ref = None
            return
        self.ground_ref = (tuple(self.goal[:3]), route.field, float(phi))

    def finish_round_transition(self, info, reward, ep=None, duration_steps=1.0):
        if self.drill_stage == 4 and DRILL_EVAL:
            raise RuntimeError('round ended or connection changed during named replay')
        ended = bool(info['round_ended'])
        ep = self.ep_reward + reward if ep is None else ep
        round_stats = None
        if ended and self.real and not self.drill and self.segs.seg is not None:
            self.segs.seg.outcome = 'round'
            self.segs._record(self.segs.seg, 'round')
            round_stats = self.segs.stats()
        self.segs.abandon()
        self._round_over()
        if ended:
            self.env.wait_new_round()
        else:
            self.configure_reconnected_game()
        self.sync_map()
        self.log(f'new round (game connection {self.env.connections}); rtf {self.env.rtf():.1f}')
        self.ep_reward = 0.0
        self.start_segment()
        return self._reply(skip=not ended, truncate=0, r=reward, done=ended, outcome='round',
                           new_segment=True, trace='', stats=round_stats, ep_reward=ep,
                           duration_steps=duration_steps)

    # ------------------------------------------------------------------ one decision
    def _transition_duration(self, clock_before, time_before):
        world = self.env.replay_world
        if clock_before is not None and world is not None and world['clock_ms'] >= clock_before:
            return max(1.0, (world['clock_ms'] - clock_before) / OBS_MS)
        if self.env.reconnected:
            return 1.0  # Connection-boundary transitions are skipped, not discounted across game clocks.
        return max(1.0, (simulation_time(self.env) - time_before) * 1000 / OBS_MS)

    def step(self, a_game):
        clock_before = self.env.replay_world['clock_ms'] if self.env.replay_world else None
        time_before = simulation_time(self.env)
        key_before = goal_key(self.goal) if self.goal is not None else None
        if POTENTIAL_K > 0.0 and self.real and (not self.drill or
                                               (self.drill_stage == 4 and SHAPING_MODE == 'discounted')):
            if self.phi_prev is None or (SHAPING_MODE != 'discounted' and self.phi_key != key_before
                                         and getattr(self, 'fall_charge', None) is None):   # 40.95: a pending fall keeps its potential
                self.phi_prev = self._phi(self.env.msg.obs[:3])
                if self.phi_prev is None and SHAPING_MODE == 'discounted':
                    self.phi_prev = 0.0
                self.phi_key = key_before
        # 40.95: the gem and route field this decision starts from, for charging a fall against them (see _shaping)
        goal_before = self.goal
        route = getattr(self, 'route', None)
        field_before = route.field if (route is not None and route.key == key_before) else None
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
        # 2026-10-09 20:45 (operator): every held powerup (1 Super Jump, 3 Super Bounce, 4 Shock Absorber, 5 Helicopter,
        # 6 Mega Marble, 7 Blast pickup) is used with the ordinary use key whenever the policy says so; the blast when
        # nothing is held. Super Speed (2) keeps its aimed dispatch above.
        if len(a) >= 6 and a[5] > .5 and self.drill_stage != 1 and not DRILL_NO_USE and held != 2:
            if held > 0:
                use_pow = 1; sent = True
                self.round_uses[f'u{held}'] = self.round_uses.get(f'u{held}', 0) + 1
            else:
                use_blast = 1; sent = True
        if sent:
            self.last_use_request = (held, self.steps)
            if self.drill_cur is not None and self.drill_stage == 4:
                requests = self.drill_cur['use_requests']
                requests[str(held)] = requests.get(str(held), 0) + 1
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
            self.last_use_request = None
            self.env.round_ended = False; self.env.reconnected = False
            self.env.control('OOBCLICK')
            if self.practice_recorder is not None:
                self.practice_recorder.history.clear()
            msg = self.env.msg
            info = {'fell': bool(msg.oob), 'gem_delta': msg.gem_delta,
                    'round_ended': self.env.round_ended, 'reconnected': self.env.reconnected}
            sent = False
        else:
            msg, info = self.env.step(js, use_pow=use_pow, pow_yaw=pow_yaw, use_blast=use_blast)
        self.prof['game'] += time.perf_counter() - tg
        if info['round_ended'] or info['reconnected']:
            # The observation can already belong to another game. Handle the
            # boundary before inventory accounting, recording, or old-map routing.
            duration = self._transition_duration(clock_before, time_before)
            term = (self._shaping(msg.obs[:3], duration, True, key_before)
                    if info['round_ended'] else 0.0)
            self.round_points += float(info['gem_delta'])
            return self.finish_round_transition(info, float(info['gem_delta']) + term,
                                                duration_steps=duration)
        if info['fell']:
            self.try_respawn_step = self.steps                          # 40.97: a held-slot blink follows a respawn
        gr = getattr(self, 'ground_ref', None)
        if FALL_CHARGE and info['fell'] and getattr(self, 'fall_charge', None) is None and gr is not None and gr[1] is not None:
            # 40.95/40.97: charged once back on the floor, against the gem chased at the last floor contact
            self.fall_charge = (gr[0], gr[1]); self.phi_prev = gr[2]
        held_now = int(msg.obs[RAW_POW_HELD]) if len(msg.obs) > RAW_POW_HELD else 0
        recent_request = getattr(self, 'last_use_request', None)
        requested = (recent_request is not None and 0 <= self.steps - recent_request[1] <= 8)
        blast_fired = (requested and recent_request[0] == 0 and not info['fell']
                       and float(msg.obs[RAW_POW_BLAST]) < float(prev_obs[RAW_POW_BLAST]) - .05)
        if blast_fired:
            self.round_uses['blast'] = self.round_uses.get('blast', 0) + 1
            if self.drill_cur is not None and self.drill_stage == 4:
                self.drill_cur['activation_events'].append(dict(resource=0, time_ms=self.env.replay_world['clock_ms'] - self.replay_start_ms))
            if self.drill_cur is not None and self.drill_cur['fire'] is None:
                self.drill_cur['fire'] = self.steps - self.drill_cur['t0'] + 1
        if not info['fell'] and ((held == 1 and held_now == 0 and requested and recent_request[0] == 1) or blast_fired):
            self.recovery.escape_use(simulation_time(self.env))
        if info['fell']:
            self.last_use_request = None
        self.use_sent = [sent, self.use_sent[0]]                   # 40.26 V10 use state
        if held > 0 and held_now == 0 and not info['fell'] and self.drill_stage != 1 and requested and recent_request[0] == held:
            if self.drill_stage >= 2 and requested and recent_request[0] == held and self.drill_cur is not None and self.drill_cur['fire'] is None:
                self.drill_cur['fire'] = self.steps - self.drill_cur['t0'] + 1
            if self.drill_stage == 4 and requested and recent_request[0] == held and self.drill_cur is not None:
                self.drill_cur['activation_events'].append(dict(resource=held, time_ms=self.env.replay_world['clock_ms'] - self.replay_start_ms))
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
        duration_steps = self._transition_duration(clock_before, time_before)
        if self.real:
            vis = visible_gems(msg.obs)
            if vis:
                tgt, nxt = self.pick(vis, self.target, msg.obs[0:3], msg.obs[3:5])
                g = self.segs.seg.goal
                want = (nxt[:3] if nxt else None)
                if tgt is not None and (math.hypot(tgt[0] - g[0], tgt[1] - g[1]) > STICKY_TOL or abs(tgt[2] - g[2]) > .25):
                    self.segs.retarget(float(msg.obs[0]), float(msg.obs[1]), tgt[:3], want, z=float(msg.obs[2]))
                    self.goal = self.segs.seg.goal
                    # Cosmetic MARK controls advance the simulation; omit them in scored transitions.
                elif tgt is not None:
                    nn = self.segs.seg.real_next
                    if (nn is None) != (want is None) or (want is not None and math.hypot(want[0] - nn[0], want[1] - nn[1]) > STICKY_TOL):
                        self.segs.set_next(float(msg.obs[0]), float(msg.obs[1]), want, z=float(msg.obs[2]))
                self.target = tgt
        rescued = False
        if self.real and not self.drill and not done and self.env.round_started():
            self.route.navigate(msg.obs[:3], self.goal)
            now = simulation_time(self.env)
            keys = [goal_key(g) for g in visible_gems(msg.obs)]
            idle = self.recovery.observe(now, goal_key(self.goal), self.route.distance(msg.obs[:3]),
                                         picked=info['gem_delta'], fell=info['fell'], gems=keys,
                                         groups_completed=(self.env.replay_world or {}).get('group_completions'))
            if RESCUE_S > 0 and idle >= RESCUE_S and keys and not info['fell']:
                self.recovery.rescue_attempts += 1
                rescued = self.env.quick_rescue()
                extra = float(self.env.control_info['gem_delta'])
                r += extra; self.round_points += extra; self.round_gems += int(extra > 0)
                self.ep_reward += extra
                for key in ('fell', 'round_ended', 'reconnected'):
                    info[key] |= self.env.control_info[key]
                if info['round_ended'] or info['reconnected']:
                    duration_steps = self._transition_duration(clock_before, time_before)
                    term = (self._shaping(self.env.msg.obs[:3], duration_steps, True, key_before)
                            if info['round_ended'] else 0.0)
                    r += term; self.ep_reward += term
                    return self.finish_round_transition(info, r, ep=self.ep_reward,
                                                        duration_steps=duration_steps)
                if rescued:
                    self.ground_ref = None; self.try_respawn_step = self.steps   # 40.97: a teleport, not a fall
                    self.last_use_request = None
                    if self.practice_recorder is not None:
                        self.practice_recorder.history.clear()
                    self.route.reset()
                    self.route.navigate(self.env.msg.obs[:3], self.goal)
                    self.recovery.rescued(simulation_time(self.env), self.route.distance(self.env.msg.obs[:3]))
                    self.obs_b.reset(); self.smooth_dir = None
                    self.pending_since = None; self.latched_aim = None; self.use_sent = [False, False]
                    self.segs.seg.last_pos = np.asarray(self.env.msg.obs[:3], dtype=float)
                    self.segs.seg.prev_d = self.route.distance(self.env.msg.obs[:3])
                    self.hist = []  # a replay approach must not bridge the rescue discontinuity
                else:
                    self.recovery.last_progress = simulation_time(self.env)
                why = next((l.split('RESCUE|', 1)[1] for l in reversed(self.env.recent_lines) if 'RESCUE|' in l), '')
                self.log(f'RESCUE map={self.mission} accepted={int(rescued)} idle_s={idle:.2f} game_s={simulation_time(self.env):.3f} game_reply={why}')
        # Controls/recovery are part of this transition, not dropped or charged as one tick.
        duration_steps = self._transition_duration(clock_before, time_before)
        term = self._shaping(self.env.msg.obs[:3], duration_steps, bool(info['round_ended']), key_before)
        r += term; self.ep_reward += term
        if info['round_ended'] or info['reconnected']:
            return self.finish_round_transition(info, r, ep=self.ep_reward, duration_steps=duration_steps)
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
        return self._reply(skip=False, r=r, done=done, outcome=outcome, new_segment=bool(done or rescued), fell=info['fell'],
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
