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
from nav.env import HuntEnv                                                         # noqa: E402
from nav.terrain import TerrainGrid                                                 # noqa: E402
from nav.obs import ObsBuilder                                                      # noqa: E402
from nav.joystick import action_to_joystick                                         # noqa: E402
from nav.waypoints import SegmentManager, RoundOver                                 # noqa: E402

# Frame check (2026-09-17): a segment whose marble accelerates against its commands (cosine
# < FRAME_FLIP_COS over the first FRAME_CHECK_N rolling decisions) is corrupted game state;
# force a respawn, drop those decisions from the rollout and start over. (The original cause,
# stale observations after an update, is gone in lockstep; this stays as a safety net.)
FRAME_CHECK_N = 12
FRAME_FLIP_COS = -0.3
GAME_SPEED = 3


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

    def start_segment(self):
        tb = time.perf_counter()
        while True:
            try:
                self.goal = self.segs.begin(self.env)
                break
            except RoundOver:
                self.segs.abandon()
                if self.env.round_ended:
                    self.env.wait_new_round()
                else:
                    self.env.request_info(); self.env.set_speed(GAME_SPEED)
                self.sync_map()
                self.log(f'new round (game connection {self.env.connections})')
        self.obs_b.reset()
        self.seg_steps = 0; self.fc_num = self.fc_den = 0.0
        self.crop, self.vec, self.on_floor = self.obs_b.build(self.env.msg.obs, self.goal)
        self.prof['begin'] += time.perf_counter() - tb

    # ------------------------------------------------------------------ one decision
    def step(self, a_game):
        vel = self.env.msg.obs[RAW_VEL]
        a = [float(v) for v in a_game]
        js = action_to_joystick(a[0], a[1], a[2], a[3], a[4], float(vel[0]), float(vel[1]))
        v_before = (float(vel[0]), float(vel[1])); prev_obs = self.env.msg.obs
        tg = time.perf_counter()
        msg, info = self.env.step(js)
        self.prof['game'] += time.perf_counter() - tg
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
                                          jumped=a[3] > 0.5)
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
            tg = time.perf_counter()
            self.crop, self.vec, self.on_floor = self.obs_b.build(self.env.msg.obs, self.goal)
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
