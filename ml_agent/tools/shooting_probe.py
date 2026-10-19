"""Random-shooting probe: can ANY action sequence reach the target gem from a recorded stuck state?

usage (repo root, PYTHONPATH=ml_agent): python ml_agent/tools/shooting_probe.py clips.json [--clips 1,2] [--n 3000] [--replies 96]
Samples piecewise-constant world-frame stick directions (a new direction every 4-16 replies), random jump replies and
Blast replies, forecasts each with the compiled bridge (ms_forecast, fixed action list) and tests the game's per-tick
pickup on the target gem. Reports how many sequences collect it, the earliest pickup, and the time per forecast.
"""
import argparse
import ctypes
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np

from nav.joystick import action_to_joystick
from nav.marble_dynamics.bridge import reply
from nav.native.forecast import Q_SIZE, R_SIZE, _blast_array, _inventory_array, _move_array
from nav.native.marble_sim import S_SIZE, _ptr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_search_check as N


def random_plan(rng, n, start_v):
    actions = []; i = 0
    while i < n:
        hold = int(rng.integers(4, 17)); ang = float(rng.uniform(-math.pi, math.pi)); thr = float(rng.choice([1., 1., .6, .3, 0.]))
        brake = rng.random() < .1
        for _ in range(hold):
            if i >= n:
                break
            js = action_to_joystick(math.cos(ang), math.sin(ang), thr, 0, 1 if brake else 0, start_v[0], start_v[1])
            a = reply(js, 0, math.atan2(math.cos(ang), math.sin(ang)))
            actions.append(a); i += 1
    for _ in range(int(rng.integers(0, 4))):
        actions[int(rng.integers(0, n))][4] = 1
    if rng.random() < .5:
        actions[int(rng.integers(0, n))][8] = 1
    return actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('clips', type=Path); p.add_argument('--clips', dest='which', default='1,2,3,5,9')
    p.add_argument('--n', type=int, default=3000); p.add_argument('--replies', type=int, default=96)
    a = p.parse_args()
    clips = json.loads(a.clips.read_text(encoding='utf-8')); rng = np.random.default_rng(3)
    for c in [int(x) for x in a.which.split(',')]:
        clip = clips[c - 1]; folder = Path(clip['journal']); name = folder.parts[-4]
        terrain, walk, engine, nav, sim = N.setup(name)
        R = {}
        with gzip.open(folder / 'decisions.jsonl.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line); R[d['decision']] = d
                except ValueError:
                    pass
        k = [x for x in sorted(R) if clip['start'] + 60 <= x][0]
        d = R[k]; base = d['before']; st = base['native']['state']; dest = d['choice']['goal_context']['destination']
        item = next(i for i in base['items'] if i.get('gem') and np.linalg.norm(np.asarray(i['position']) - dest[:3]) < .1)
        lo = np.asarray(item['bounds'][:3]); hi = np.asarray(item['bounds'][3:])
        native = base['native']; count = a.replies
        bridge = np.zeros(18); _move_array(native['pending_move'], bridge[0:6]); bridge[6:15] = base['bridge_action']
        bridge[15] = int(base['pow_yaw_hold']); bridge[16] = float(base['pow_yaw']); bridge[17] = int(native['frame_clock_ms'])
        state = dict(native['state']); start = np.empty(S_SIZE); sim.pack(state, start)
        blast = _blast_array(base.get('blast')); inv = _inventory_array(base.get('inventory'))
        rows = np.empty((4 * count + 1, S_SIZE)); replies = np.zeros((count, R_SIZE)); bo = np.zeros(1); blo = np.zeros(13)
        status = (ctypes.c_int * 8)(); q = np.zeros(Q_SIZE); consts = np.zeros(13)
        hits = []; t0 = time.perf_counter(); fails = 0
        for s in range(a.n):
            plan = np.ascontiguousarray(np.asarray(random_plan(rng, count, st['velocity']), dtype=float))
            sim.lib.ms_forecast(sim.handle, None, _ptr(start), _ptr(bridge), _ptr(blast), _ptr(inv), count, 0, count, _ptr(plan), 0, _ptr(q),
                                None, None, _ptr(consts), 0., -1, _ptr(rows), _ptr(replies), _ptr(bo), _ptr(blo), status)
            n = status[1]
            if status[2] >= 0:
                fails += 1
            pos = rows[:n, 0:3]; r = rows[1:n, 9:10] + .2
            p0, p1 = pos[:-1], pos[1:]
            ok = np.flatnonzero(np.all(np.maximum(p0, p1) + r >= lo, axis=1) & np.all(np.minimum(p0, p1) - r <= hi, axis=1))
            if len(ok):
                hits.append(int(ok[0]) // 4 + 1)
        dt = time.perf_counter() - t0
        g = np.asarray(dest[:3]); pp = np.asarray(st['position'])
        print('clip %d %s | dec %d: gem %.1f u away %+.1f u up, blast meter %.2f | %d of %d random %d-reply sequences collect it%s | %.2f ms per forecast (%d failed)'
              % (c, clip['title'][:40], k, np.linalg.norm((g - pp)[:2]), g[2] - pp[2], float((base.get('blast') or {}).get('value', 0)),
                 len(hits), a.n, count, (', earliest %.1f s' % (min(hits) * .064)) if hits else '', dt / a.n * 1000, fails), flush=True)


if __name__ == '__main__':
    main()
