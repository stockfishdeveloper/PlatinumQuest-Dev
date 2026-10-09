"""Scan teleports along a box's own u and v axes to find where the game's box really is.
    python -m nav.maps.box_scan <map> <port> <box_index> [<box_index> ...]
"""
import os, sys, json, math
import numpy as np
ML = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ML)
from terrain_obs import TerrainMap
from nav.maps.teleport_probe import boxes_of, R_M, LIFT

STEPS = np.arange(-2.0, 2.01, 0.2)


def run(mapname, port, idxs):
    from nav.learned_nav.session import Session, RoundOver
    from nav.protocol import RAW_POS, NOOP_ACTION
    tm = TerrainMap(os.path.join(ML, 'terrain_maps', f'terrain_{mapname}.npz'))
    bx = boxes_of(mapname)

    class Geo:
        xs, ys, heights = tm.xs, tm.ys, tm.heights
    s = Session(port, mapname, Geo(), log=lambda m: print(m, flush=True))
    out = open(os.path.join(ML, 'logs', 'nav', 'blockclusters', f'boxscan_{mapname}.jsonl'), 'a')
    for bi in idxs:
        cx, cy, w, d, top, ang = bx[bi]
        a = math.radians(ang); ca, sa = math.cos(a), math.sin(a)
        print(f'box {bi}: centre ({cx:.2f},{cy:.2f}) {w:.1f}x{d:.1f} top {top:.2f} ang {ang:.1f}', flush=True)
        for axis in ('u', 'v'):
            row = []
            for t in STEPS:
                x, y = (cx + ca * t, cy + sa * t) if axis == 'u' else (cx - sa * t, cy + ca * t)
                z0 = top + R_M + LIFT
                res = '?'
                for attempt in range(3):
                    try:
                        if s.place((x, y, z0)) is None: continue
                        zf = None
                        for k in range(10):
                            s.step(NOOP_ACTION)
                            zf = float(s.obs()[RAW_POS][2])
                        drop = z0 - zf
                        res = 'B' if drop < 0.15 else ('f' if drop < 0.6 else 'F')   # on the box top / 0.5 lower / the floor
                        out.write(json.dumps(dict(box=bi, axis=axis, t=round(float(t), 2), x=round(x, 3), y=round(y, 3), drop=round(drop, 3))) + '\n')
                        break
                    except RoundOver:
                        continue
                row.append(res)
            print(f'  {axis}: ' + ''.join(row) + '   (t from -2.0 to +2.0 by 0.2; B = box top, f = half drop, F = floor)', flush=True)
    out.close(); print('DONE', flush=True)


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]), [int(v) for v in sys.argv[3:]])
