# Kicks in real_run traces (speed jump >= 10 u/s in one decision) and what followed: fall within 3 s, gems within 1.5 s.
# python kick_falls.py trace1.csv [trace2.csv ...]
import csv, sys, math
rows = []
for f in sys.argv[1:]:
    rows = list(csv.DictReader(open(f)))
    ks = []
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if a['round'] != b['round']:
            continue
        if float(b['speed']) - float(a['speed']) >= 10.0:
            fall = any(int(rows[j]['fell']) for j in range(i, min(i + 47, len(rows))) if rows[j]['round'] == b['round'])
            g15 = sum(int(rows[j]['gem']) for j in range(i, min(i + 24, len(rows))) if rows[j]['round'] == b['round'])
            vx, vy = float(b['vx']), float(b['vy']); tx, ty = float(b['tx']) - float(b['x']), float(b['ty']) - float(b['y'])
            ang = math.degrees(math.acos(max(-1, min(1, (vx * tx + vy * ty) / max(1e-6, math.hypot(vx, vy) * math.hypot(tx, ty))))))
            ks.append((float(a['speed']), float(b['speed']), float(a['tdist']), ang, fall, g15))
    n = len(ks); nf = sum(k[4] for k in ks)
    print(f'{f}: kicks {n}, fall within 3 s {nf} ({100.0 * nf / max(n, 1):.0f} %)')
    for lo, hi in ((0, 22), (22, 26), (26, 99)):
        sub = [k for k in ks if lo <= k[1] < hi]
        if sub:
            print(f'  speed after {lo}-{hi}: n {len(sub)}, falls {sum(k[4] for k in sub)}, gems 1.5 s {sum(k[5] for k in sub) / len(sub):.2f}, '
                  f'dist {sum(k[2] for k in sub) / len(sub):.1f}, angle {sum(k[3] for k in sub) / len(sub):.0f}')
    for lo, hi in ((0, 8), (8, 14), (14, 99)):
        sub = [k for k in ks if lo <= k[2] < hi]
        if sub:
            print(f'  target dist {lo}-{hi}: n {len(sub)}, falls {sum(k[4] for k in sub)}, gems 1.5 s {sum(k[5] for k in sub) / len(sub):.2f}')
