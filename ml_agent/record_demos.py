"""
Record human demonstrations for behaviour cloning.

Run this INSTEAD of train_ppo.py while you play:

    python record_demos.py                  # listens on 127.0.0.1:8888 like the trainer
    python record_demos.py --inspect demos/demo_20260914_120000.npz   # summarise a file

Then just host the map (King of the Marble by default) and play. Nothing to
type in the game: this server answers every message with "RECORD", and the
game script switches itself into recording mode (you control the marble at
1x, observations pinned to the agent's frame, your inputs appended) as soon
as it sees that reply. Rounds auto-restart; play as many as you like.
Ctrl+C here when done.

What is recorded, per 16 ms game tick, from the message
    obs_json|gemDelta|oob|done|forward,backward,left,right,jump,usePowerup,cameraYaw
  obs_raw   (35)  the raw observation, in the agent's FIXED frame (yaw 0 = world),
                  because MLAgent::enableRecording() pins the observer's yaw
  obs_model (161) exactly what the policy network sees: normalized obs + tick
                  frame history + terrain sample, built with the same code as
                  training (play.normalize_obs, terrain_obs.TerrainMap)
  action    (5)   the agent's action format: dx, dy (unit vector of the human's
                  movement intent in the fixed frame), throttle (0/1), jump,
                  brake. The human steers with keys + camera; their key vector
                  is rotated from their camera into the fixed frame using the
                  recorded camera yaw (observer.cs convention:
                  camera right = (cos yaw, -sin yaw), forward = (sin yaw, cos yaw)).
                  brake is DERIVED: 1 when the intent opposes the velocity.
  has_move  (1)   1 when any movement key was held (dx, dy meaningful)
  use_pow   (1)   1 while the use-powerup input is held (for a future powerup action)
  yaw, inputs_raw, gem_delta, oob, done, tick, game     bookkeeping

Live self-checks (printed every 10 s and in the final summary) tell you whether
the frame conversion is right: the human's intent direction must line up with
the marble's acceleration over the next 8 ticks. "frame check" reads ~0.85 on
real play and the "mirrored" alternative ~0.0; anything else prints
CHECK FRAME CONVENTION.

Output: demos/demo_<timestamp>.npz (rewritten every 30 s and on exit) plus a
.json summary next to it.
"""
import os
import sys
import json
import math
import time
import socket
import argparse
from collections import deque
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from play import normalize_obs                       # same normalization as the trainer
from terrain_obs import TerrainMap
from train_ppo import resolve_terrain_path, DEFAULT_TERRAIN_MAP

OBS_BASE = 35
FRAME_HISTORY_COUNT = 4
FRAME_SKIP = 8
FRAME_HISTORY_DIMS = 6
HIST_LEN = FRAME_HISTORY_COUNT * FRAME_SKIP + 1
BRAKE_MIN_SPEED = 2.0      # derived-brake label needs some speed to be meaningful
BRAKE_COS = -0.7           # intent within ~45 deg of anti-velocity


def intent_to_world(fwd, back, left, right, yaw):
    """Human key vector in their camera frame -> unit direction in the fixed frame."""
    r = float(right) - float(left)
    f = float(fwd) - float(back)
    wx = r * math.cos(yaw) + f * math.sin(yaw)
    wy = -r * math.sin(yaw) + f * math.cos(yaw)
    n = math.hypot(wx, wy)
    if n < 1e-6:
        return 0.0, 0.0, 0.0
    return wx / n, wy / n, min(1.0, n)


class DemoRecorder:
    def __init__(self, terrain_path, out_dir):
        self.terrain = TerrainMap(terrain_path) if terrain_path else None
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(out_dir, f'demo_{self.stamp}.npz')
        self.frame_history = deque(maxlen=HIST_LEN)
        self.rows = {k: [] for k in ('obs_raw', 'obs_model', 'action', 'has_move', 'use_pow', 'yaw',
                                     'inputs_raw', 'gem_delta', 'oob', 'done', 'tick', 'game')}
        self.tick = 0
        self.game = 0
        self.game_totals = []          # totals reported by the game at each round end
        self.t0 = time.time()
        self.last_save = time.time()
        self.last_report = time.time()
        self.malformed = 0
        self.pre_record_ticks = 0      # ticks received before the game switched to recording mode
        self.recording_started = False

    # ------------------------------------------------------------------ per tick
    def handle(self, line):
        parts = line.split('|')
        if len(parts) < 4:
            self.malformed += 1
            return
        obs_json, gem_delta, oob, done = parts[0], float(parts[1]), int(float(parts[2])), int(float(parts[3]))
        obs = json.loads(obs_json)
        if len(obs) == 0 and done:            # game end: []|-total|0|1
            self.game_totals.append(int(abs(gem_delta)))
            self.game += 1
            self.frame_history.clear()
            return
        if len(obs) != OBS_BASE:
            self.malformed += 1
            return
        if len(parts) < 5:
            self.pre_record_ticks += 1        # handshake in flight: game not yet in recording mode
            if self.pre_record_ticks == 50:
                print("  WARNING: 50 ticks without the human-input block. Is the game script up to date (mlAgent.cs handshake)?")
            return
        if not self.recording_started:
            self.recording_started = True
            print(f"  [{datetime.now():%H:%M:%S}] game switched to recording mode after {self.pre_record_ticks} tick(s); recording.")
        inp = [float(v) for v in parts[4].split(',')]
        fwd, back, left, right, jump, use_pow, yaw = inp[:7]

        raw = np.array(obs, dtype=np.float32)
        dx, dy, mag = intent_to_world(fwd, back, left, right, yaw)
        has_move = 1.0 if mag > 0 else 0.0
        vx, vy = float(raw[3]), float(raw[4])
        speed = math.hypot(vx, vy)
        brake = 0.0
        if has_move and speed > BRAKE_MIN_SPEED and (dx * vx + dy * vy) / speed < BRAKE_COS:
            brake = 1.0
        action = np.array([dx, dy, 1.0 if has_move else 0.0, 1.0 if jump > 0.5 else 0.0, brake], dtype=np.float32)

        norm = normalize_obs(raw.copy())
        self.frame_history.append(norm[0:6].copy())
        hist = []
        for i in range(1, FRAME_HISTORY_COUNT + 1):
            idx = len(self.frame_history) - 1 - i * FRAME_SKIP
            hist.append(self.frame_history[idx] if idx >= 0 else np.zeros(FRAME_HISTORY_DIMS, dtype=np.float32))
        terrain = self.terrain.observe(raw) if self.terrain else TerrainMap.flat_observe()   # point samples + edge rays
        obs_model = np.concatenate([norm] + hist + [terrain]).astype(np.float32)

        r = self.rows
        r['obs_raw'].append(raw); r['obs_model'].append(obs_model); r['action'].append(action)
        r['has_move'].append(has_move); r['use_pow'].append(1.0 if use_pow > 0.5 else 0.0); r['yaw'].append(yaw)
        r['inputs_raw'].append(np.array(inp[:6], dtype=np.float32))
        r['gem_delta'].append(gem_delta); r['oob'].append(oob); r['done'].append(done)
        r['tick'].append(self.tick); r['game'].append(self.game)
        self.tick += 1
        if done:
            self.frame_history.clear()

        now = time.time()
        if now - self.last_report > 10:
            self.report(final=False); self.last_report = now
        if now - self.last_save > 30:
            self.save(); self.last_save = now

    # ------------------------------------------------------------------ checks
    def arrays(self):
        return {k: np.array(v) for k, v in self.rows.items()}

    def stats(self, a):
        s = self._stats(a)
        # plain Python types only: numpy scalars are not JSON serialisable
        def py(v):
            if isinstance(v, (np.floating,)): return float(v)
            if isinstance(v, (np.integer,)): return int(v)
            if isinstance(v, (list, tuple)): return [py(x) for x in v]
            return v
        return {k: py(v) for k, v in s.items()}

    def _stats(self, a):
        n = len(a['tick'])
        if n < 50:
            return {'ticks': n}
        raw = a['obs_raw']; act = a['action']
        vxy = raw[:, 3:5]; speed = np.linalg.norm(vxy, axis=1)
        gem = a['gem_delta'] > 0
        s = {
            'ticks': int(n), 'seconds_of_play': round(n * 0.016, 1),
            'games_completed': int(len(self.game_totals)), 'game_totals_from_game': self.game_totals,
            'gem_points': int(a['gem_delta'].sum()), 'pickups': int(gem.sum()),
            'oob': int(a['oob'].sum()),
            'move_input_pct': round(100 * a['has_move'].mean(), 1),
            'jump_pct': round(100 * act[:, 3].mean(), 2),
            'use_powerup_pct': round(100 * a['use_pow'].mean(), 2),
            'derived_brake_pct': round(100 * act[:, 4].mean(), 2),
            'mean_speed': round(float(speed.mean()), 2),
            'p90_speed': round(float(np.percentile(speed, 90)), 2),
            'pickup_speed': round(float(speed[gem].mean()), 2) if gem.any() else None,
            'ticks_per_pickup': round(n / max(1, gem.sum()), 1),
            'malformed_messages': self.malformed,
            'pre_record_ticks': self.pre_record_ticks,
        }
        # Frame check. A human pushes the marble where it is going, so the intent
        # direction (rotated into the fixed frame) must line up with the velocity
        # direction while moving. If the yaw convention were wrong, the same data
        # rotated with the mirrored yaw would line up instead. Secondary check:
        # intent vs the velocity CHANGE over the next 8 ticks (acceleration).
        ir = a['inputs_raw']; yaw = a['yaw']
        mirrored = np.array([intent_to_world(f, b, l, r_, -y)[:2] for (f, b, l, r_, *_), y in zip(ir, yaw)], dtype=np.float32)
        mv = (a['has_move'] > 0) & (speed > 3.0)
        if mv.sum() > 100:
            vdir = vxy[mv] / np.linalg.norm(vxy[mv], axis=1, keepdims=True)
            s['frame_check_cos'] = round(float((act[mv, 0:2] * vdir).sum(1).mean()), 3)
            s['frame_check_cos_mirrored_yaw'] = round(float((mirrored[mv] * vdir).sum(1).mean()), 3)
            s['frame_check_samples'] = int(mv.sum())
        if n > 20:
            dv = vxy[8:] - vxy[:-8]
            m = (a['has_move'][:-8] > 0) & (np.linalg.norm(dv, axis=1) > 0.3)
            if m.sum() > 100:
                v = dv[m] / np.linalg.norm(dv[m], axis=1, keepdims=True)
                s['frame_check_accel_cos'] = round(float((act[:-8][m, 0:2] * v).sum(1).mean()), 3)
                s['frame_check_accel_cos_mirrored_yaw'] = round(float((mirrored[:-8][m] * v).sum(1).mean()), 3)
        # terrain sanity: on the floor (not falling), the 2u samples should mostly be present
        if self.terrain is not None:
            grounded = raw[:, 5] > -8.0
            present2 = a['obs_model'][:, 59:][:, [ (k * 4 + 0) * 2 for k in range(8)]]
            s['terrain_present_at_2u_pct_when_grounded'] = round(100 * float(present2[grounded].mean()), 1)
            s['pos_range_x'] = [round(float(raw[:, 0].min()), 1), round(float(raw[:, 0].max()), 1)]
            s['pos_range_y'] = [round(float(raw[:, 1].min()), 1), round(float(raw[:, 1].max()), 1)]
            s['terrain_map_x'] = [round(float(self.terrain.xs[0]), 1), round(float(self.terrain.xs[-1]), 1)]
            s['terrain_map_y'] = [round(float(self.terrain.ys[0]), 1), round(float(self.terrain.ys[-1]), 1)]
        return s

    def report(self, final):
        a = self.arrays()
        try:
            s = self.stats(a)
        except Exception as e:
            print(f"  [{datetime.now():%H:%M:%S}] {len(a['tick']):,} ticks (summary error: {e})"); return
        if 'seconds_of_play' not in s:
            print(f"  [{datetime.now():%H:%M:%S}] {s['ticks']} ticks so far..."); return
        # Verdict uses the ACCELERATION check: a human's key direction lines up
        # with how the velocity changes (measured 0.86 on real play, ~0.01 for a
        # wrong yaw convention). Intent vs current velocity direction is only
        # ~0.3 for a human because of turning, braking and inertia.
        fc = s.get('frame_check_accel_cos'); fm = s.get('frame_check_accel_cos_mirrored_yaw')
        verdict = ''
        if fc is not None:
            verdict = 'OK' if (fc > 0.5 and fc > fm + 0.3) else 'CHECK FRAME CONVENTION'
        print(f"  [{datetime.now():%H:%M:%S}] {s['ticks']:,} ticks ({s['seconds_of_play']:.0f}s) | games {s['games_completed']} "
              f"| {s['gem_points']} pts / {s['pickups']} pickups ({s['ticks_per_pickup']} ticks each) | OOB {s['oob']} "
              f"| move {s['move_input_pct']}% jump {s['jump_pct']}% brake {s['derived_brake_pct']}% "
              f"| speed mean {s['mean_speed']} p90 {s['p90_speed']} pickup {s['pickup_speed']} "
              f"| frame check {fc} vs mirrored {fm} {verdict}"
              + (f" | terrain@2u {s.get('terrain_present_at_2u_pct_when_grounded')}%" if 'terrain_present_at_2u_pct_when_grounded' in s else '')
              + (f" | malformed {s['malformed_messages']}" if s['malformed_messages'] else ''))
        if final:
            print(json.dumps(s, indent=2))

    def save(self):
        a = self.arrays()
        if len(a['tick']) == 0:
            return
        # Atomic, interrupt-proof save. The 2026-09-14 session lost 22 minutes of
        # play: a Ctrl+C landed during the final np.savez_compressed, numpy closed
        # the half-written archive cleanly (one 4 KB member) and it had already
        # replaced the good file. Now: write to a temp file with SIGINT deferred,
        # then os.replace() it over the old one only if it is complete.
        import signal
        tmp = self.path + '.tmp'
        deferred = []
        prev = signal.signal(signal.SIGINT, lambda sig, frame: deferred.append(sig))
        try:
            with open(tmp, 'wb') as f:
                np.savez_compressed(f, **a,
                                    terrain_map=np.array(self.terrain.path if self.terrain else ''),
                                    obs_layout=np.array('35 raw normalized | 24 frame history (t-8,16,24,32 ticks) | 64 terrain | 38 edge rays'),
                                    action_layout=np.array('dx, dy (fixed frame unit vector), throttle, jump, brake(derived)'))
            with np.load(tmp, allow_pickle=False) as chk:          # verify before replacing anything
                if len(chk['tick']) != len(a['tick']):
                    raise IOError(f"temp archive has {len(chk['tick'])} ticks, expected {len(a['tick'])}")
            os.replace(tmp, self.path)
        except BaseException as e:
            print(f"  SAVE FAILED ({type(e).__name__}: {e}); the previous file is untouched")
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        finally:
            signal.signal(signal.SIGINT, prev)
            if deferred:
                print("  (Ctrl+C received during save; finishing the save first)")
        try:
            with open(self.path.replace('.npz', '.json'), 'w') as f:
                json.dump(self.stats(a), f, indent=2)
        except Exception as e:      # a summary bug must never cost the recording
            print(f"  (summary not written: {e})")
        print(f"  saved {self.path} ({len(a['tick']):,} ticks)")


def inspect(path):
    a = dict(np.load(path, allow_pickle=False))
    rec = DemoRecorder(str(a.get('terrain_map', '')) or None, os.path.dirname(path) or '.')
    rec.rows = {k: list(a[k]) for k in rec.rows if k in a}
    rec.game_totals = []
    print(json.dumps(rec.stats({k: np.array(v) for k, v in rec.rows.items()}), indent=2))
    act = a['action']
    print("\naction sample (dx, dy, throttle, jump, brake) at 5 random moving ticks:")
    idx = np.where(a['has_move'] > 0)[0]
    for i in np.random.default_rng(0).choice(idx, size=min(5, len(idx)), replace=False):
        print("  ", np.round(act[i], 3), "yaw", round(float(a['yaw'][i]), 2), "keys F,B,L,R:", a['inputs_raw'][i][:4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8888)
    ap.add_argument('--terrain', default=None, help=f'terrain map name/path (default {DEFAULT_TERRAIN_MAP})')
    ap.add_argument('--no-terrain', action='store_true')
    ap.add_argument('--out', default=os.path.join(HERE, 'demos'))
    ap.add_argument('--inspect', default=None, help='summarise an existing .npz and exit')
    args = ap.parse_args()
    if args.inspect:
        inspect(args.inspect); return

    terrain_path = resolve_terrain_path(args.terrain, args.no_terrain)
    rec = DemoRecorder(terrain_path, args.out)
    print(f"Recording demos to {rec.path}")
    print(f"Terrain map: {terrain_path}")
    print(f"Listening on {args.host}:{args.port}. Now host the map and play; recording starts by itself at GO. Ctrl+C to stop.")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.settimeout(1.0)
    srv.bind((args.host, args.port)); srv.listen(1)
    try:
        while True:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            print(f"Game connected from {addr}")
            buf = ''
            try:
                while True:
                    data = conn.recv(8192).decode('utf-8')
                    if not data:
                        print("Game disconnected; waiting for it again (Ctrl+C to finish).")
                        break
                    buf += data
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec.handle(line)
                        except Exception as e:
                            rec.malformed += 1
                            if rec.malformed <= 5:
                                print(f"  bad message ({e}): {line[:120]}")
                        conn.sendall(b'RECORD\n')          # handshake: the game switches to recording mode on seeing this
            except (ConnectionResetError, ConnectionAbortedError, OSError) as e:
                print(f"Connection lost ({e}); waiting for the game again (Ctrl+C to finish).")
            finally:
                conn.close()
                rec.frame_history.clear()
                rec.recording_started = False
                rec.save()
    except KeyboardInterrupt:
        pass
    finally:
        rec.save()
        rec.report(final=True)
        srv.close()


if __name__ == '__main__':
    main()
