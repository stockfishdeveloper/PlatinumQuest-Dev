"""How much faster does the built engine's AI training mode run, and is the physics unchanged?

    python fixedstep_probe.py          (built engine launched with -autotrain FlatGemTraining_Hunt)

Runs the physics_identity_probe key script (20 s of game time) under several timing modes, from
the same rest pose each time, and reports for each: sim speed (game seconds per wall second),
physics ticks per second, the rest position, and the trajectory deviation from mode 0.

  mode 0  normal timing at 3x                      (what the shipped exe does today: the baseline)
  mode 1  FIXEDSTEP 32 + LOCKSTEP, render every frame
  mode 2  FIXEDSTEP 32 + LOCKSTEP, render 1 in 10
  mode 3  FIXEDSTEP 32 + LOCKSTEP, render 1 in 100
  mode 4  mode 3 again                             (determinism: must match mode 3 exactly)
  mode 5  FIXEDSTEP 32, no lockstep, render 1 in 100 (raw sim ceiling; replies race the sim)

Integrity: modes 1-4 must agree with each other to 0.000 u (one action per tick, no timing
jitter) and with mode 0 to within the async bridge's own noise (physics_identity_probe measured
~3 u between 3x and 10x for the same script). Results: logs/fixedstep_probe.json
"""
import os
import sys
import json
import time
import socket
import numpy as np
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from physics_identity_probe import SCRIPT, NOOP, REST_SPEED, REST_TICKS, REST_CAP  # noqa: E402

# PROBE_QUANT=<ms>: round every key-script segment to a multiple of this many ms so the key
# changes fall on the same game times for every step size (a fair physics comparison between
# 32/64/128 ms steps; without it the segment lengths round differently per step size).
QUANT = int(os.environ.get('PROBE_QUANT', '0'))
if QUANT:
    SCRIPT = [(max(1, round(n * 16 / QUANT)) * QUANT // 16,) + tuple(rest) for n, *rest in SCRIPT]

MODES = [
    ('normal 3x (baseline)',            ['FIXEDSTEP 0', 'LOCKSTEP 0', 'RENDEREVERY 1', 'SPEED 3'], 16),
    ('fixed 32 + lockstep, render 1/1', ['FIXEDSTEP 32', 'LOCKSTEP 1', 'RENDEREVERY 1'], 32),
    ('fixed 32 + lockstep, render 1/10', ['FIXEDSTEP 32', 'LOCKSTEP 1', 'RENDEREVERY 10'], 32),
    ('fixed 32 + lockstep, render 1/100', ['FIXEDSTEP 32', 'LOCKSTEP 1', 'RENDEREVERY 100'], 32),
    ('fixed 32 + lockstep, render 1/100 (repeat)', ['FIXEDSTEP 32', 'LOCKSTEP 1', 'RENDEREVERY 100'], 32),
    ('fixed 32, NO lockstep, render 1/100', ['FIXEDSTEP 32', 'LOCKSTEP 0', 'RENDEREVERY 100'], 32),
    ('fixed 64 (2 ticks/obs) + lockstep, render 1/100', ['FIXEDSTEP 64', 'LOCKSTEP 1', 'RENDEREVERY 100'], 64),
    ('fixed 128 (4 ticks/obs) + lockstep, render 1/100', ['FIXEDSTEP 128', 'LOCKSTEP 1', 'RENDEREVERY 100'], 128),
]
TICK_MS = 32
# PROBE_MODES=1,6 : run only these mode indices (the first one listed is the comparison reference)
if os.environ.get('PROBE_MODES'):
    MODES = [] if os.environ['PROBE_MODES'] == 'none' else [MODES[int(i)] for i in os.environ['PROBE_MODES'].split(',')]


def main():
    port = int(os.environ.get('PROBE_PORT', '8888'))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', port)); srv.listen(1)
    print(f'fixedstep_probe: waiting for the game on {port}', flush=True)
    conn, _ = srv.accept(); f = conn.makefile('r')
    tick = ['']

    def recv():
        while True:
            line = f.readline()
            if not line:
                raise ConnectionError('game disconnected')
            parts = line.strip().split('|')
            if not parts[0].startswith('['):
                continue                     # STATS/INFO/DEBUG replies
            try:
                obs = json.loads(parts[0])
            except ValueError:
                obs = []
            tick[0] = parts[-1].strip() if len(parts) >= 5 and parts[-1].strip().isdigit() else ''
            if len(obs) >= 35:
                return obs
            send(NOOP)                       # round-end message etc.

    def send(reply):
        if reply[:1].isdigit() and tick[0]:
            reply += f',t{tick[0]}'
        conn.sendall((reply + '\n').encode())

    def control(word):
        send(word); return recv()

    def wait_rest(obs):
        still = 0
        for k in range(REST_CAP):
            v = float(np.linalg.norm(obs[3:6]))
            still = still + 1 if v < REST_SPEED else 0
            if still >= REST_TICKS:
                return obs, k
            send(NOOP); obs = recv()
        return obs, REST_CAP

    obs = recv()
    start = [round(float(v), 4) for v in obs[:3]]
    print(f'connected; start pose {start}', flush=True)
    results = []
    for mi, (name, words, step_ms) in enumerate(MODES):
        for w in words:
            obs = control(w)
        obs = control(f'TELEPORT {start[0]} {start[1]} {start[2] + 0.5} 0 0 0')
        for _ in range(int(1000 / step_ms)):        # 1 s to settle
            send(NOOP); obs = recv()
        obs, _ = wait_rest(obs)
        pose = [round(float(v), 4) for v in obs[:3]]
        # run the key script by sim time
        traj = []; sim_ms = 0; n_obs = 0
        t0 = time.perf_counter()
        for n16, fw, bk, lf, rt, jp in SCRIPT:
            n = max(1, round(n16 * 16 / step_ms))
            for _ in range(n):
                send(f'{fw},{bk},{lf},{rt},{jp},0.000000,0'); obs = recv()
                sim_ms += step_ms; n_obs += 1
                traj.append([sim_ms] + [round(float(v), 4) for v in obs[:3]])
        wall = time.perf_counter() - t0
        obs, rest_ticks = wait_rest(obs)
        end = [round(float(v), 4) for v in obs[:3]]
        r = {'mode': mi, 'name': name, 'words': words, 'step_ms': step_ms, 'start': pose, 'end': end,
             'sim_s': sim_ms / 1000.0, 'wall_s': round(wall, 3), 'speed': round(sim_ms / 1000.0 / wall, 2),
             'ticks_per_s': round(sim_ms / TICK_MS / wall, 1), 'obs': n_obs, 'traj': traj, 'rest_ticks': rest_ticks}
        results.append(r)
        print(f'mode {mi} [{name}]: {r["sim_s"]:.1f} s of game time in {wall:.2f} s wall -> {r["speed"]}x, '
              f'{r["ticks_per_s"]} physics ticks/s ({n_obs} observations); rest at {end}', flush=True)
    # PROBE_SUSTAIN=<s>: after the modes, step continuously for that many wall seconds in
    # fixed 64 + lockstep mode, starting at PROBE_START_AT (epoch seconds) so several probes
    # can be made to overlap: aggregate throughput = the sum over instances.
    sustain = float(os.environ.get('PROBE_SUSTAIN', '0'))
    if sustain > 0:
        for w in ['FIXEDSTEP 64', 'LOCKSTEP 1', 'RENDEREVERY 100']:
            obs = control(w)
        # rendezvous: PROBE_SYNC_DIR = a directory; this probe drops ready_<port> there and waits
        # for a file named 'go' (the launcher writes it once every probe is ready)
        sync_dir = os.environ.get('PROBE_SYNC_DIR')
        if sync_dir:
            open(os.path.join(sync_dir, f'ready_{port}'), 'w').close()
            while not os.path.exists(os.path.join(sync_dir, 'go')):
                time.sleep(0.05)
        start_at = float(os.environ.get('PROBE_START_AT', '0'))
        while time.time() < start_at:
            time.sleep(0.05)
        t0 = time.perf_counter(); n = 0; keys = ['1,0,0,0,0,0.000000,0', '0,1,0,0,0,0.000000,0', '0,0,1,0,0,0.000000,0', '0,0,0,1,0,0.000000,0']
        while time.perf_counter() - t0 < sustain:
            send(keys[(n // 20) % 4]); obs = recv(); n += 1
        wall = time.perf_counter() - t0
        print(f'SUSTAINED: {n} decisions (64 ms each, 2 ticks) in {wall:.1f} s wall -> {n * 0.064 / wall:.1f}x, '
              f'{n * 2 / wall:.0f} physics ticks/s, {n / wall:.0f} decisions/s', flush=True)
    for w in ['FIXEDSTEP 0', 'LOCKSTEP 0', 'RENDEREVERY 1', 'SPEED 3']:
        control(w)
    conn.close(); srv.close()

    if not results:
        return
    print('\nINTEGRITY (rest position distance / trajectory deviation sampled every 32 ms of game time)')
    def sampled(tr):
        d = {int(t): np.array(p) for t, *p in tr}
        return d
    ref = sampled(results[0]['traj'])
    for r in results:
        cur = sampled(r['traj'])
        keys = sorted(set(ref) & set(cur))
        dev = np.array([np.linalg.norm(ref[k] - cur[k]) for k in keys]) if keys else np.zeros(1)
        print(f'  mode {r["mode"]} vs mode 0: rest {np.linalg.norm(np.array(r["end"]) - np.array(results[0]["end"])):.3f} u apart; '
              f'trajectory max dev {dev.max():.3f} u, mean {dev.mean():.3f} u over {len(keys)} samples')
    ls = [r for r in results if 'LOCKSTEP 1' in r['words']]
    for a in ls[1:]:
        ca, cb = sampled(a['traj']), sampled(ls[0]['traj'])
        keys = sorted(set(ca) & set(cb))
        dev = np.array([np.linalg.norm(ca[k] - cb[k]) for k in keys])
        print(f'  lockstep mode {a["mode"]} vs lockstep mode {ls[0]["mode"]}: rest {np.linalg.norm(np.array(a["end"]) - np.array(ls[0]["end"])):.4f} u apart; '
              f'trajectory max dev {dev.max():.4f} u (must be 0)')
    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    with open(os.path.join(HERE, 'logs', 'fixedstep_probe.json'), 'w') as fh:
        json.dump({'when': datetime.now().isoformat(), 'results': results}, fh)


if __name__ == '__main__':
    main()
