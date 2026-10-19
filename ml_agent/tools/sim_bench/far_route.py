"""Far legs vs the walk-grid route: route length at the leg start, actual path length, time; time lost vs route/11.
usage (repo root, PYTHONPATH=ml_agent): python far_route.py <pool>[+<pool>] [map ...]"""
import sys, glob, os, gzip, json
import numpy as np
from pathlib import Path
from collections import defaultdict
from nav.terrain import TerrainMap
from nav.routing import RoutePlanner
from nav.marble_dynamics.static_cache import cached_terrain
CACHE = Path(r'C:\Users\doug\AppData\Local\Temp\pq_checks\fast_nav_static_cache_20261013')
R0 = 'ml_agent/dynamics_reports/'
pools = sys.argv[1].split('+'); maps = sys.argv[2:]
tot = defaultdict(float)
for name in maps:
    terrain = cached_terrain(TerrainMap.resolve(name), CACHE)
    agg = defaultdict(float); n = 0
    for pool in pools:
        f = R0 + pool + '/' + name + '/7347019/physics_plan/1/decisions.jsonl.gz'
        if not os.path.exists(f): continue
        R = []
        for line in gzip.open(f, 'rt', encoding='utf-8'):
            try: R.append(json.loads(line))
            except Exception: pass
        R.sort(key=lambda d: d['decision'])
        leg = []
        for d in R:
            leg.append(d)
            if float(d['outcome'].get('gem_delta') or 0) > 0 or d['outcome'].get('fell'):
                fell = any(x['outcome'].get('fell') for x in leg)
                st0 = leg[0]['before']['native']['state']; p0 = np.asarray(st0['position'], float)
                p1 = np.asarray(d['before']['native']['state']['position'], float)
                T = len(leg) * .064; straight = float(np.linalg.norm((p1 - p0)[:2]))
                if not fell and T > 8 and straight >= 20 and float(d['outcome'].get('gem_delta') or 0) > 0:
                    rp = RoutePlanner(terrain); rp.navigate(p0, p1); rd = float(rp.distance(p0))
                    path = 0.; last = p0; slow = 0
                    for x in leg:
                        q = np.asarray(x['before']['native']['state']['position'], float); s = float(np.linalg.norm((q - last)[:2]))
                        if s < 3: path += s
                        last = q
                    if np.isfinite(rd):
                        n += 1; agg['T'] += T; agg['route'] += rd; agg['path'] += path; agg['straight'] += straight
                        agg['lost'] += T - rd / 11.
                leg = []
    if n:
        print('%-22s far legs %3d | mean T %5.1f s  route %5.1f u  path %5.1f u  straight %5.1f u | path/route %.2f  path speed %.2f u/s  route speed %.2f u/s  lost vs route/11: %.1f s a leg'
              % (name, n, agg['T'] / n, agg['route'] / n, agg['path'] / n, agg['straight'] / n, agg['path'] / agg['route'], agg['path'] / agg['T'], agg['route'] / agg['T'], agg['lost'] / n))
        for k, v in agg.items(): tot[k] += v
        tot['n'] += n
n = tot['n']
print('ALL far legs %d | mean T %.1f s route %.1f path %.1f | path/route %.2f path speed %.2f route speed %.2f lost %.1f s a leg' % (n, tot['T'] / n, tot['route'] / n, tot['path'] / n, tot['path'] / tot['route'], tot['path'] / tot['T'], tot['route'] / tot['T'], tot['lost'] / n))
