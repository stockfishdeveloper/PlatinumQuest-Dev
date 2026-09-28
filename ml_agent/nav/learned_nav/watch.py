"""Watch the P0 learned jump chooser at 1x on the drill map (viewing only; nothing is recorded for training).

    python -m nav.learned_nav.watch --port 8961                 # held-out feasible starts from the P0 evaluation
    python -m nav.learned_nav.watch --port 8961 --fresh         # brand-new random starts (never recorded)
    then: marbleblast_mbx.exe -autotrain kotmjump_p0 -aiport 8961

Each start: the marble is teleported onto the floor, rolling; the model scores all 46 candidates from that state and
the best one is flown, at real time (one 64 ms decision per 64 ms of wall time, every frame drawn). The physics step
is the recorded one (64 ms), so what you see is what the evaluation measured. A one-second pause between starts.
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault('NAV_RENDER_EVERY', '1')          # draw every frame (read by nav.env at import)

import numpy as np                                                               # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav import dynamics as D                                        # noqa: E402
from nav.learned_nav.record import Recorder, CANDS, R, floor_below, sample_starts  # noqa: E402
from nav.learned_nav.evaluate import MODEL_PATH, RESULTS                         # noqa: E402
from nav.protocol import NOOP_ACTION                                             # noqa: E402


def describe(ci):
    c = CANDS[ci]
    if c is None:
        return 'no jump'
    j, a = c
    t = (a - 1) * 45
    air = 'no air input' if a == 0 else f'air {t - 360 if t > 180 else t:+d} deg from the heading'
    return f'jump at decision {j}, {air}'


class PacedRecorder(Recorder):
    def __init__(self, port, log=print):
        super().__init__(port, log=log)
        self.env.set_speed(1)
        self.next_t = time.perf_counter()

    def step(self, js):
        self.next_t = max(self.next_t + 0.064, time.perf_counter())
        dt = self.next_t - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        return super().step(js)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--fresh', action='store_true')
    ap.add_argument('--n', type=int, default=200)
    a = ap.parse_args()
    model, meta = D.load(MODEL_PATH)
    if a.fresh:
        starts = sample_starts(a.n, int(time.time()) % 100000)
    else:
        res = json.load(open(RESULTS))
        ids = [s['id'] for s in res['starts'] if s['feasible']]
        pool = {}
        for fn in os.listdir(D.DATA_DIR):
            if fn.startswith('starts') and fn.endswith('.json'):
                pool.update({s['id']: s for s in json.load(open(os.path.join(D.DATA_DIR, fn)))})
        starts = [pool[i] for i in ids][:a.n]
    rec = PacedRecorder(a.port)
    won = 0
    for k, st in enumerate(starts):
        x, y, h, sp = st['x'], st['y'], st['heading'], st['speed']
        vx, vy = sp * math.cos(h), sp * math.sin(h)
        p0 = [x, y, floor_below(x, y, 21.0) + R + 0.01]
        f = D.start_features(p0, [vx, vy, 0.0], [-vy / R, vx / R, 0.0], h)
        g = D.gem_rel(p0, h)
        pr = D.predict(model, f[None], g[None])
        score = D.score(pr, meta['variant'], meta['curve'])[1][0]
        ci = int(np.argmax(score))
        r = rec.run(st, ci)
        won += int(r.get('success', False))
        print(f'{k + 1:3d} {st["id"]}: {sp:4.1f} u/s, {describe(ci)}; predicted {score[ci]:.2f} -> '
              f'{"PICKUP + SAFE" if r.get("success") else ("pickup, fell" if r.get("pickup") else ("fell" if r.get("oob") else "missed"))}'
              f'  ({won}/{k + 1})', flush=True)
        for _ in range(16):                       # a second on the floor (or respawning) between starts
            rec.step(NOOP_ACTION)


if __name__ == '__main__':
    main()
