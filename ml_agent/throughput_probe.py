"""
Throughput and physics-integrity probe for the game bridge.

    python throughput_probe.py [speeds...]      default: 1 3 6 10 15 20 30 45

Run instead of the trainer, then start the game (or let run_game_loop.ps1 /
-autotrain do it). For each requested time scale it:
  1. sets the scale with the "SPEED n" control reply and lets it settle,
  2. measures for MEASURE_TICKS ticks: wall time per tick (=> real-time factor),
     the hunt-timer step per message (must be 16 ms: 32 = a dropped tick, 0 =
     a duplicated observation), game-process CPU, and reply latency,
  3. runs a fixed scripted maneuver from a fixed teleport pose and records the
     trajectory, then compares it with the 1x trajectory (max deviation).
Physics integrity = 16 ms steps, no drops/duplicates, and identical
trajectories at every scale. Results go to logs/throughput_probe.json and the
console. Ctrl+C aborts.
"""
import os
import sys
import json
import time
import socket
import statistics
from datetime import datetime

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

HERE = os.path.dirname(os.path.abspath(__file__))
SPEEDS = [float(a) for a in sys.argv[1:]] or [1, 3, 6, 10, 15, 20, 30, 45]
SETTLE_TICKS = 90
MEASURE_TICKS = 500
NOOP = '0,0,0,0,0,0.000000,0'
POSE = (-25.0, 27.0, 21.4)           # middle of the north band (floor 20.8 from x=-47 to -3), above the floor
REST_TICKS = 60
# scripted maneuver: (ticks, fwd, back, left, right, jump)
SCRIPT = [(50, 0, 0, 0, 1, 0), (40, 0, 1, 0, 0, 0), (60, 0, 0, 1, 0, 0), (6, 0, 0, 1, 0, 1), (24, 1, 0, 0, 0, 0)]   # stays on the band
SCRIPT_TICKS = sum(s[0] for s in SCRIPT)


def script_cmd(i):
    for n, f, b, l, r, j in SCRIPT:
        if i < n:
            return f'{f},{b},{l},{r},{j},0.000000,0'
        i -= n
    return NOOP


def game_proc():
    if psutil is None:
        return None
    for p in psutil.process_iter(['name']):
        if (p.info['name'] or '').lower().startswith('marbleblast'):
            return p
    return None


class Probe:
    def __init__(self):
        self.results = {}
        self.ref_traj = None
        self.log_lines = []

    def log(self, s):
        print(s, flush=True); self.log_lines.append(f"[{datetime.now():%H:%M:%S}] {s}")

    def run(self, conn, remaining):
        f = conn.makefile('r')
        def recv():
            while True:
                line = f.readline()
                if not line:
                    raise ConnectionError('game disconnected')
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                try:
                    obs = json.loads(parts[0])
                except ValueError:
                    obs = []
                return obs, parts
        def send(reply):
            conn.sendall((reply + '\n').encode())
        def wait_running(min_time_left_s):
            """Skip round ends / dead zones; return the first obs with enough time left."""
            while True:
                obs, parts = recv()
                if len(obs) >= 35 and obs[31] > min_time_left_s * 1000:
                    return obs
                send(NOOP)
        proc = game_proc()
        maxfps = os.environ.get('PROBE_MAXFPS')
        if maxfps is not None and not getattr(self, '_maxfps_sent', False):
            recv(); send(f'MAXFPS {maxfps}'); self._maxfps_sent = True
            self.log(f"sent MAXFPS {maxfps}")
        while remaining:
            speed = remaining[0]
            self.log(f"=== speed {speed:g}x ===")
            obs = wait_running(60)
            send(f'SPEED {speed:g}')
            for _ in range(SETTLE_TICKS):
                obs, _ = recv(); send(NOOP)
            # --- measurement window ---
            if proc: proc.cpu_percent(None)
            t0 = time.perf_counter(); last_wall = t0; timer_steps = []; gaps = []; lat = []; states = []
            prev_timer = obs[31]
            for k in range(MEASURE_TICKS):
                obs, parts = recv()
                now = time.perf_counter(); gaps.append(now - last_wall); last_wall = now
                if len(obs) >= 35:
                    timer_steps.append(prev_timer - obs[31]); prev_timer = obs[31]; states.append(obs[0:6])
                ts = time.perf_counter(); send(NOOP); lat.append(time.perf_counter() - ts)
            wall = time.perf_counter() - t0
            cpu = proc.cpu_percent(None) if proc else None
            steps = np.array(timer_steps)
            st = np.array(states); moving = np.linalg.norm(st[:-1, 3:6], axis=1) > 0.3
            dup = int(np.sum((np.abs(np.diff(st, axis=0)).max(axis=1) < 1e-9) & moving))
            rtf = MEASURE_TICKS * 0.016 / wall
            res = {
                'requested': speed, 'rtf': round(rtf, 2), 'ticks_per_s': round(MEASURE_TICKS / wall, 1),
                'timer_step_ms_median': float(np.median(steps)) if len(steps) else None,
                'timer_steps_not_16': int(np.sum(np.abs(steps - 16) > 0.5)) if len(steps) else None,
                'timer_steps_32_or_more': int(np.sum(steps >= 31.5)) if len(steps) else None,
                'timer_steps_zero': int(np.sum(steps < 0.5)) if len(steps) else None,
                'gap_ms_p50': round(1000 * statistics.median(gaps), 2), 'gap_ms_p99': round(1000 * float(np.percentile(gaps, 99)), 2),
                'gap_ms_max': round(1000 * max(gaps), 1),
                'game_cpu_pct': round(cpu, 1) if cpu is not None else None,
                'dup_states_while_moving': dup, 'moving_msgs': int(moving.sum()),
            }
            self.log(f"   rtf {res['rtf']}x ({res['ticks_per_s']} ticks/s wall) | timer step median {res['timer_step_ms_median']} ms, "
                     f"not-16: {res['timer_steps_not_16']} (>=32: {res['timer_steps_32_or_more']}, 0: {res['timer_steps_zero']}) "
                     f"| dup states {dup}/{int(moving.sum())} moving msgs | gap p50 {res['gap_ms_p50']} p99 {res['gap_ms_p99']} max {res['gap_ms_max']} ms | game CPU {res['game_cpu_pct']}%")
            # --- determinism test ---
            obs = wait_running(30)
            send(f'TELEPORT {POSE[0]} {POSE[1]} {POSE[2]} 0 0 0')
            for _ in range(REST_TICKS):
                obs, _ = recv(); send(NOOP)
            start = np.array(obs[0:3]); traj = []
            for i in range(SCRIPT_TICKS + 10):
                obs, _ = recv()
                traj.append(obs[0:3] + obs[3:6])
                send(script_cmd(i))
            traj = np.array(traj)
            res['start_pose'] = [round(float(v), 3) for v in start]
            res['end_pos'] = [round(float(v), 3) for v in traj[-1, :3]]
            if self.ref_traj is None:
                self.ref_traj = traj; res['traj_dev_max'] = 0.0; res['traj_dev_mean'] = 0.0
                self.log(f"   reference trajectory recorded (start {res['start_pose']}, end {res['end_pos']})")
            else:
                n = min(len(traj), len(self.ref_traj))
                dev = np.linalg.norm(traj[:n, :3] - self.ref_traj[:n, :3], axis=1)
                res['traj_dev_max'] = round(float(dev.max()), 4); res['traj_dev_mean'] = round(float(dev.mean()), 4)
                res['traj_dev_at_tick'] = int(dev.argmax())
                self.log(f"   trajectory vs 1x: max deviation {res['traj_dev_max']} u (tick {res['traj_dev_at_tick']}), mean {res['traj_dev_mean']} u; end {res['end_pos']}")
            res['trajectory'] = traj[:, :3].round(4).tolist()
            self.results[f'{speed:g}'] = res
            remaining.pop(0)
        send('SPEED 3')
        self.log("done; game speed set back to 3x")


def main():
    probe = Probe()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = int(os.environ.get('PROBE_PORT', '8888'))
    srv.bind(('127.0.0.1', port)); srv.listen(1)
    probe.log(f"throughput_probe: waiting for the game on {port} (speeds {SPEEDS})")
    # Rounds end (and the game reconnects) in the middle of a sweep at high
    # speeds, so accept new connections and resume with the remaining speeds.
    remaining = list(SPEEDS)
    try:
        while remaining:
            conn, addr = srv.accept(); probe.log(f"game connected {addr}")
            try:
                probe.run(conn, remaining)
            except (ConnectionError, OSError) as e:
                probe.log(f"connection ended: {e} (resuming with speeds {remaining})")
            finally:
                conn.close()
    finally:
        srv.close()
        os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
        with open(os.path.join(HERE, 'logs', f'throughput_probe_{port}.json'), 'w') as fh:
            json.dump({'when': datetime.now().isoformat(), 'results': probe.results, 'log': probe.log_lines}, fh, indent=1)
        print("\nSUMMARY  requested -> measured rtf | ticks/s | bad timer steps | traj dev max | game CPU")
        for k, r in probe.results.items():
            print(f"   {float(k):5.1f}x -> {r['rtf']:5.2f}x | {r['ticks_per_s']:7.1f} | dup {r.get('dup_states_while_moving', '?')} | {r.get('traj_dev_max', 0):7.4f} u | {r['game_cpu_pct']}%")


if __name__ == '__main__':
    main()
