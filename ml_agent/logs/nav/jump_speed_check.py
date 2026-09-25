"""Overnight check (HANDOFF 28.37): approved-jump rate, success, median takeoff speed and crossing length in
16-minute buckets of the newest training trace, KOTM instances only. Usage: python logs/nav/jump_speed_check.py"""
import csv, collections, glob, math, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from terrain_obs import TerrainMap
from nav.terrain import TerrainGrid
t = TerrainGrid(TerrainMap.resolve('KingOfTheMarble_Hunt'))
f = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trace_*.csv')), key=os.path.getmtime)[-1]
rows = list(csv.DictReader(open(f, newline='')))
by = collections.defaultdict(list)
for r in rows:
    if r['inst'] in {str(i) for i in range(7)}:
        by[r['inst']].append(r)
B = 409600; agg = collections.OrderedDict()
for inst, rs in by.items():
    prev = 0
    for k, r in enumerate(rs):
        b = int(float(r['step'])) // B * B; a = agg.setdefault(b, {'dec': 0, 'ok': 0, 'fell': 0, 'sp': [], 'len': []})
        a['dec'] += 1
        j = int(float(r['jump'])); onf = float(r['on_floor']) > 0.5
        if j and not prev and onf and float(r['gp'] or 0) > 0.5:
            x0, y0 = float(r['x']), float(r['y']); v0 = math.hypot(float(r['vx']), float(r['vy']))
            saw = False; out = None
            for q in rs[k + 1:k + 45]:
                x, y = float(q['x']), float(q['y'])
                if int(float(q['oob'])):
                    out = 1; break
                if not t.walkable_at(x, y):
                    saw = True
                elif saw and float(q['on_floor']) > 0.5:
                    out = 0; a['len'].append(math.hypot(x - x0, y - y0)); break
            if out == 0:
                a['ok'] += 1; a['sp'].append(v0)
            elif out == 1:
                a['fell'] += 1
        prev = j
print(os.path.basename(f))
print('bucket(~16 min)  approved/min  success  takeoff speed med  crossing med   n')
for b, a in agg.items():
    mins = a['dec'] * 0.064 / 60; n = a['ok'] + a['fell']
    if mins < 20:
        continue
    print(f'{b:>12d}     {n/mins:5.2f}       {100*a["ok"]/max(1,n):4.0f}%     {statistics.median(a["sp"]) if a["sp"] else 0:5.1f}            {statistics.median(a["len"]) if a["len"] else 0:5.1f}     {n}')
