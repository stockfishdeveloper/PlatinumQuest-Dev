"""Jump-progress series for the dashboard (2026-09-24, HANDOFF 28.34), torch-free.

Tails the trainer's decision traces (logs/nav/trace_*.csv) and aggregates, per bucket of updates, for the
KOTM instances (0-6 under the 7/1 split):
  approved takeoffs per instance-minute   (jump pressed on the floor while the gap prior was on)
  approved-takeoff success %              (the marble crossed a void and landed vs fell)
  void crossings per instance-minute      (landed after passing over non-walkable cells) with / without a jump
  falls within 1.5 s of an approved takeoff, per instance-minute
"""
import collections
import csv
import glob
import os
import threading
import time

from terrain_obs import TerrainMap
from nav.terrain import TerrainGrid

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(HERE, 'logs', 'nav')
STEPS_PER_BUCKET = 204800        # trace steps per point (~25 updates, ~8 min); the dashboard maps step -> update from the NAV log
KOTM_INSTS = {str(i) for i in range(7)}
DECISION_S = 0.064
WINDOW = 24                      # decisions after a takeoff that count as its outcome / its fall
HISTORY_HOURS = 24               # older trace files are ignored (every trace ever written is tens of millions of rows)


class JumpProgress:
    def __init__(self, map_name='KingOfTheMarble_Hunt'):
        self.t = TerrainGrid(TerrainMap.resolve(map_name))
        self.lock = threading.Lock()
        self.buckets = collections.OrderedDict()     # bucket -> counters
        self.file = None; self.pos = 0; self.inst = {}
        self.rows_seen = 0
        self._cache = []

    def _b(self, step):
        b = (int(step) // STEPS_PER_BUCKET) * STEPS_PER_BUCKET
        if b not in self.buckets:
            self.buckets[b] = {'dec': 0, 'appr': 0, 'appr_ok': 0, 'appr_fell': 0, 'cross_j': 0, 'cross_nj': 0, 'fall_appr': 0}
        return self.buckets[b]

    def _feed(self, r):
        inst = r['inst']
        if inst not in KOTM_INSTS:
            return
        st = self.inst.setdefault(inst, {'k': 0, 'over': False, 'take': -10**9, 'appr': False, 'take_b': None, 'jump_take': False, 'prev_j': 0})
        st['k'] += 1; k = st['k']
        try:
            step = int(float(r['step'])); x = float(r['x']); y = float(r['y'])
            onf = float(r['on_floor']) > 0.5; j = int(float(r['jump'])); oob = int(float(r['oob']))
            gp = float(r['gp'] or 0) > 0.5
        except (ValueError, KeyError):
            return
        b = self._b(step); b['dec'] += 1
        if j and not st['prev_j'] and onf:
            st['take'] = k; st['appr'] = gp; st['take_b'] = b; st['resolved'] = False
            if gp:
                b['appr'] += 1
        st['prev_j'] = j
        void = not self.t.walkable_at(x, y)
        recent = (k - st['take']) <= WINDOW
        if oob:
            if recent and st['appr'] and not st.get('resolved', True):
                st['take_b']['appr_fell'] += 1; st['take_b']['fall_appr'] += 1; st['resolved'] = True
            st['over'] = False
            return
        if not onf and void and not st['over']:
            st['over'] = True; st['jump_take'] = recent
        elif st['over'] and onf and not void:
            st['over'] = False
            if st['jump_take']:
                b['cross_j'] += 1
                if st['appr'] and not st.get('resolved', True):
                    st['take_b']['appr_ok'] += 1; st['resolved'] = True
            else:
                b['cross_nj'] += 1

    def _newest(self):
        fs = sorted(glob.glob(os.path.join(LOG_DIR, 'trace_*.csv')), key=os.path.getmtime)
        return fs[-1] if fs else None

    def _load_all_history(self):
        cutoff = time.time() - HISTORY_HOURS * 3600
        fs = sorted((f for f in glob.glob(os.path.join(LOG_DIR, 'trace_*.csv')) if os.path.getmtime(f) >= cutoff), key=os.path.getmtime)
        for f in fs[:-1]:
            try:
                with open(f, newline='') as fh:
                    for r in csv.DictReader(fh):
                        self._feed(r); self.rows_seen += 1
            except OSError:
                pass
            self.inst = {}

    def poll(self):
        f = self._newest()
        if f is None:
            return
        if f != self.file:
            self.file = f; self.pos = 0; self.inst = {}; self.header = None
        try:
            with open(f, newline='') as fh:
                fh.seek(self.pos)
                if self.pos == 0:
                    self.header = fh.readline().rstrip('\r\n').split(',')
                    self.pos = fh.tell()
                chunk = fh.read()
                if not chunk:
                    return
                lines = chunk.split('\n')
                complete = lines[:-1] if not chunk.endswith('\n') else lines[:-1]
                tail = lines[-1] if not chunk.endswith('\n') else ''
                self.pos = fh.tell() - len(tail.encode('utf-8'))
        except OSError:
            return
        with self.lock:
            for line in complete:
                if not line:
                    continue
                vals = line.rstrip(chr(13)).split(',')
                if len(vals) != len(self.header):
                    continue
                self._feed(dict(zip(self.header, vals))); self.rows_seen += 1

    def run(self, interval=10.0):
        with self.lock:
            self._load_all_history()
        while True:
            try:
                self.poll()
            except Exception as e:      # keep the thread alive; the dashboard shows what it has
                print('jump_progress:', e, flush=True)
            time.sleep(interval)

    def series(self):
        # never block the web server: while the history load holds the lock, serve the last result
        if not self.lock.acquire(timeout=0.2):
            return self._cache
        try:
            out = []
            for b, c in self.buckets.items():
                mins = c['dec'] * DECISION_S / 60.0
                if mins < 0.5:
                    continue
                n = c['appr_ok'] + c['appr_fell']
                out.append({'step': b, 'appr_per_min': round(c['appr'] / mins, 3), 'appr_success': (round(100.0 * c['appr_ok'] / n, 1) if n else None),
                            'appr_n': n, 'cross_j_per_min': round(c['cross_j'] / mins, 3), 'cross_nj_per_min': round(c['cross_nj'] / mins, 3),
                            'fall_appr_per_min': round(c['fall_appr'] / mins, 3), 'minutes': round(mins, 1)})
            self._cache = out
            return out
        finally:
            self.lock.release()


def start_background(map_name='KingOfTheMarble_Hunt'):
    jp = JumpProgress(map_name)
    th = threading.Thread(target=jp.run, daemon=True); th.start()
    return jp
