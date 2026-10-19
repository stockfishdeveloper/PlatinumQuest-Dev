"""Pickups: turn angle between the velocity at the pickup and the straight line to the next pickup, the speed at the
pickup, the slowest speed in the next second, and the time to the next pickup. usage (ml_agent/dynamics_reports): python turn_audit.py <pool>[+...]"""
import sys, glob, os, gzip, json, math
import numpy as np
from collections import defaultdict
for arg in sys.argv[1:]:
    bins = defaultdict(lambda: [0, 0., 0., 0., 0.])
    for pool in arg.split('+'):
        for f in sorted(glob.glob(pool + '/*/7347019/physics_plan/1/decisions.jsonl.gz')):
            R = []
            for line in gzip.open(f, 'rt', encoding='utf-8'):
                try: R.append(json.loads(line))
                except Exception: pass
            R.sort(key=lambda d: d['decision'])
            picks = [i for i, d in enumerate(R) if float(d['outcome'].get('gem_delta') or 0) > 0 and not d['outcome'].get('fell')]
            for a, b in zip(picks, picks[1:]):
                if any(R[k]['outcome'].get('fell') for k in range(a, b + 1)): continue
                st = R[a]['before']['native']['state']; v = np.asarray(st['velocity'], float)[:2]; sp = float(np.linalg.norm(v))
                p = np.asarray(st['position'], float)[:2]; q = np.asarray(R[b]['before']['native']['state']['position'], float)[:2]
                dq = q - p; dist = float(np.linalg.norm(dq))
                if sp < 1 or dist < 2 or dist > 20: continue
                ang = math.degrees(math.acos(max(-1., min(1., float(v @ dq) / (sp * dist)))))
                mn = min(float(np.linalg.norm(np.asarray(R[k]['before']['native']['state']['velocity'], float)[:2])) for k in range(a + 1, min(b, a + 16) + 1))
                T = (b - a) * .064
                k = int(ang // 30) * 30
                x = bins[k]; x[0] += 1; x[1] += sp; x[2] += mn; x[3] += T; x[4] += dist
    print('== %s pickups followed by a leg of 2-20 u (no fall)' % arg[:40])
    n = sum(x[0] for x in bins.values())
    for k in sorted(bins):
        x = bins[k]
        print('  turn %3d-%3d deg: %5d (%4.1f%%)  speed at pickup %.1f  slowest in next 1 s %.1f  next leg %.1f u in %.2f s (%.1f u/s)' % (
            k, k + 30, x[0], 100 * x[0] / n, x[1] / x[0], x[2] / x[0], x[4] / x[0], x[3] / x[0], x[4] / x[3]))
