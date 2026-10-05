# Per kick (speed jump >= 10 u/s): decisions to the first gem, speed there, the turn between the velocity and the new
# target right after that gem, then a fall (decisions after the kick, speed before it, gem picked before?).
import csv, sys, math
for f in sys.argv[1:]:
    rows = list(csv.DictReader(open(f)))
    out = []
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if a['round'] != b['round'] or float(b['speed']) - float(a['speed']) < 10.0:
            continue
        g_at = None; fall_at = None
        for j in range(i, min(i + 47, len(rows))):
            if rows[j]['round'] != b['round']:
                break
            if g_at is None and int(rows[j]['gem']) > 0:
                g_at = j
            if int(rows[j]['fell']):
                fall_at = j; break
        turn = None; sp_g = None
        if g_at is not None and g_at + 1 < len(rows):
            r = rows[g_at + 1]; vx, vy = float(r['vx']), float(r['vy']); tx, ty = float(r['tx']) - float(r['x']), float(r['ty']) - float(r['y'])
            sp_g = math.hypot(vx, vy)
            turn = math.degrees(math.acos(max(-1, min(1, (vx * tx + vy * ty) / max(1e-6, sp_g * math.hypot(tx, ty))))))
        out.append((i, float(b['speed']), g_at - i if g_at else None, sp_g, turn, fall_at - i if fall_at else None,
                    float(rows[fall_at - 1]['speed']) if fall_at else None, float(rows[fall_at - 1]['tdist']) if fall_at else None))
    fl = [o for o in out if o[5] is not None]; ok = [o for o in out if o[5] is None]
    print(f'{f}: kicks {len(out)}, falls {len(fl)}')
    for name, grp in (('fell', fl), ('ok', ok)):
        g = [o for o in grp if o[2] is not None]
        if g:
            print(f'  {name}: gem first {len(g)}/{len(grp)}, dec to gem {sum(o[2] for o in g) / len(g):.1f}, speed at gem {sum(o[3] for o in g) / len(g):.1f}, '
                  f'turn to next {sum(o[4] for o in g) / len(g):.0f} deg')
    for o in fl:
        print(f'    kick sp {o[1]:.1f} gem@{o[2]} sp {o[3] if o[3] is None else round(o[3], 1)} turn {o[4] if o[4] is None else round(o[4])} | fall@{o[5]} sp {o[6]:.1f} tdist {o[7]:.1f}')
