"""Stage 3 overnight collection: a queue of recorder jobs over the verified maps, SLOTS games at a time.

    python -m nav.learned_nav.collect3            # runs until the queue is done; resumable (shards on disk count)

Each job is one (map, shard) of JOB_TRIALS trials: Python recorder first (it binds the port), then the game on that
map. A job that exits cleanly is done; a crashed or stalled one (no shard log progress for STALL_S) is killed and
requeued, and resumes from its written shards. Disk guard (operator, 2026-09-27): stop before the stage 3 data
exceeds DATA_CAP_GB or the free space falls below MIN_FREE_GB. Log: logs/learned_nav/collect3.log.
"""
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PQ = os.path.normpath(os.path.join(HERE, '..', 'Marble Blast Platinum'))
EXE = os.path.join(PQ, 'marbleblast_mbx.exe')
PY = sys.executable
DATA = os.path.join(HERE, 'datasets', 'learned_nav', 'stage3')
LOG = os.path.join(HERE, 'logs', 'learned_nav', 'collect3.log')
SLOTS = 8
PORT0 = 9101
JOB_TRIALS = 20000
STALL_S = 600
DATA_CAP_GB = 20.0
MIN_FREE_GB = 25.0
TRAIN = {'KingOfTheMarble_Hunt_phys': 500000}
for m in ('VortexEffect_Hunt_phys', 'GemsInTheRoad_Hunt_phys', 'Tilo_Hunt_phys', 'BasinHill_Hunt_phys',
          'ParkourPeaks_Hunt_phys', 'Duplex_Hunt_phys', 'Cragmire_Hunt_phys', 'Sprawl_Hunt_phys',
          'KingOfTheRing_Hunt_phys', 'MaximoCenter_Hunt_phys'):
    TRAIN[m] = 180000
DEV = {'GemsAhoy_Hunt_phys': 60000, 'Acropolis2_Hunt_phys': 60000}


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def done_trials(m, shard):
    d = os.path.join(DATA, m)
    if not os.path.isdir(d):
        return 0
    return sum(int(f.split('_')[-1].split('.')[0]) for f in os.listdir(d) if f.startswith(f'shard_{shard}_') and f.endswith('.npz'))


def data_gb():
    tot = 0
    for root, _, files in os.walk(DATA):
        tot += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return tot / 1e9


def listening(port):
    out = subprocess.run(['netstat', '-ano', '-p', 'TCP'], capture_output=True, text=True).stdout
    return re.search(r'127\.0\.0\.1:%d\s+\S+\s+LISTENING' % port, out) is not None


def kill(p):
    if p is not None and p.poll() is None:
        subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True)


def jobs():
    q = []
    order = list(TRAIN.items()) + list(DEV.items())
    for m, total in order:
        for k in range((total + JOB_TRIALS - 1) // JOB_TRIALS):
            if done_trials(m, k) < JOB_TRIALS:
                q.append((m, k))
    # interleave maps so every map gets data early
    by = {}
    for m, k in q:
        by.setdefault(m, []).append((m, k))
    out = []
    while any(by.values()):
        for m in list(by):
            if by[m]:
                out.append(by[m].pop(0))
    return out


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    queue = jobs()
    log(f'collect3: {len(queue)} jobs of {JOB_TRIALS} trials, {SLOTS} slots')
    slots = [None] * SLOTS
    retries = {}
    while queue or any(slots):
        free = shutil.disk_usage(HERE).free / 1e9
        size = data_gb()
        stop = free < MIN_FREE_GB or size > DATA_CAP_GB
        for i in range(SLOTS):
            s = slots[i]
            if s is not None:
                rc = s['py'].poll()
                logf = os.path.join(DATA, s['map'], f'shard_{s["shard"]}.log')
                last = os.path.getmtime(logf) if os.path.exists(logf) else s['t0']
                stalled = time.time() - max(last, s['t0']) > STALL_S
                if rc is None and not stalled:
                    continue
                kill(s['py']); kill(s['game'])
                if rc == 0:
                    log(f'slot {i}: {s["map"]} shard {s["shard"]} done ({done_trials(s["map"], s["shard"])} trials)')
                else:
                    key = (s['map'], s['shard'])
                    retries[key] = retries.get(key, 0) + 1
                    log(f'slot {i}: {s["map"]} shard {s["shard"]} {"stalled" if stalled else f"exited {rc}"}; '
                        f'retry {retries[key]} ({done_trials(*key)} trials on disk)')
                    if retries[key] <= 5:
                        queue.append(key)
                    else:
                        log(f'slot {i}: giving up on {key}')
                slots[i] = None
                time.sleep(2)
            if slots[i] is None and queue and not stop:
                m, k = queue.pop(0)
                port = PORT0 + i
                out = open(os.path.join(HERE, 'logs', 'learned_nav', f'collect3_slot{i}.txt'), 'a')
                py = subprocess.Popen([PY, '-u', '-m', 'nav.learned_nav.record3', '--port', str(port), '--map', m,
                                       '--shard', str(k), '--trials', str(JOB_TRIALS)], cwd=HERE, stdout=out,
                                      stderr=subprocess.STDOUT, creationflags=0x08000000)
                t = time.time()
                while not listening(port) and time.time() - t < 60 and py.poll() is None:
                    time.sleep(0.3)
                if py.poll() is not None:
                    log(f'slot {i}: recorder for {m} shard {k} exited at start ({py.returncode})')
                    queue.append((m, k)); continue
                game = subprocess.Popen([EXE, '-autotrain', m, '-aiport', str(port)], cwd=PQ)
                slots[i] = {'map': m, 'shard': k, 'py': py, 'game': game, 't0': time.time()}
                log(f'slot {i}: started {m} shard {k} on port {port} ({len(queue)} jobs queued)')
        if stop and not any(slots):
            log(f'collect3: stopped by the disk guard (free {free:.1f} GB, data {size:.1f} GB)')
            break
        time.sleep(5)
    log(f'collect3: finished; data {data_gb():.2f} GB')


if __name__ == '__main__':
    main()
