"""Every jump a HUMAN made in a recorded demo, and what it bought (2026-09-24, HANDOFF 28.36).

For each jump-key press while on the floor: position, heading, takeoff speed, whether the flight crossed
non-walkable cells (a gap jump) and how long the crossing was, the height change, and an estimate of the
walking detour it saved (walk-only Dijkstra distance from takeoff to landing vs the straight flight).
Also reports, at the SAME takeoff cells, whether the training graph has a jump edge (would the reward
field ever route across) and whether the physics flag would have approved it at that speed.

    python -m nav.demo_jumps demos/demo_20260924_212808.npz
"""
import math
import os
import sys

import numpy as np

from terrain_obs import TerrainMap
from nav.terrain import TerrainGrid, JUMP_DROP, JUMP_RISE
from nav.physics import crossable, jump_range

TICK_S = 0.016


def main():
    path = sys.argv[1]
    d = np.load(path, allow_pickle=False)
    raw = d['obs_raw']; inp = d['inputs_raw']; game = d['game'] if 'game' in d else np.zeros(len(raw))
    oob = d['oob'] if 'oob' in d else np.zeros(len(raw))
    t = TerrainGrid(TerrainMap.resolve('KingOfTheMarble_Hunt'))
    print(f'{os.path.basename(path)}: {len(raw)} ticks ({len(raw)*TICK_S/60:.1f} min), games {int(game.max())+1 if len(game) else 0}')
    jump = inp[:, 4] > 0.5
    pos = raw[:, 0:3]; vel = raw[:, 3:6]
    # on floor: near a floor level and not moving vertically (same rule as nav/obs.py)
    fz = np.array([t.floor_z(float(x), float(y), float(z)) for x, y, z in pos])
    on_floor = (np.abs(vel[:, 2]) < 0.3) & (np.abs(pos[:, 2] - fz) < 0.6)
    # the observation on the press tick already carries the jump impulse (vz ~ +7.4), so test the tick before
    onsets = np.nonzero(jump[1:] & ~jump[:-1] & on_floor[:-1])[0] + 1
    print(f'jump presses on the floor: {len(onsets)} ({len(onsets)/(len(raw)*TICK_S/60):.2f} per minute)')
    rows = []
    for k in onsets:
        x, y, z = (float(v) for v in pos[k - 1]); vx, vy = float(vel[k - 1, 0]), float(vel[k - 1, 1])   # state before the impulse
        sp = math.hypot(vx, vy)
        # follow the flight: until back on the floor (or oob) within 2 s
        crossed = False; landed = None; fell = False; maxz = z
        for q in range(k + 3, min(len(raw), k + int(2.0 / TICK_S))):
            qx, qy, qz = (float(v) for v in pos[q]); maxz = max(maxz, qz)
            if oob[q] > 0.5 or (q > 0 and game[q] != game[k]):
                fell = True; break
            if not t.walkable_at(qx, qy):
                crossed = True
            elif crossed and on_floor[q]:
                landed = (qx, qy, qz); break
            elif not crossed and on_floor[q] and q > k + 12:
                break
        kind = 'GAP' if crossed and landed else ('FELL' if crossed and fell else 'hop')
        detour = None; gap_len = None; approved = None; graph_edge = None
        if landed:
            gap_len = math.hypot(landed[0] - x, landed[1] - y)
            try:
                f = t.goal_field(landed[0], landed[1], jumps=False)
                walk = t.dist_at(f, x, y, (landed[0], landed[1]))
                detour = walk - gap_len if math.isfinite(walk) else None
            except Exception:
                detour = None
            if sp > 0.5:
                hx, hy = vx / sp, vy / sp
                lip, gap, ldz = t.gap_along(x, y, z, hx, hy)
                approved = bool(math.isfinite(gap) and -JUMP_DROP <= ldz <= JUMP_RISE and crossable(lip + gap, sp) and t.walkable_at(x, y))
            j0, i0 = t.cell_of(x, y); j1, i1 = t.cell_of(landed[0], landed[1])
            graph_edge = any((a, b, c, e) == (j0, i0, j1, i1) or (abs(a - j0) <= 1 and abs(b - i0) <= 1 and abs(c - j1) <= 1 and abs(e - i1) <= 1)
                             for (a, b, c, e, g) in t.jump_edge_cells)
        rows.append((k, kind, x, y, z, sp, gap_len, detour, maxz - z, approved, graph_edge))
    print('\n tick    kind   at (x, y)        speed  crossed  walk-detour-saved  rise   physics-approved  graph-edge')
    for (k, kind, x, y, z, sp, gl, det, rise, appr, ge) in rows:
        print(f'{k:6d}  {kind:5s}  ({x:6.1f},{y:6.1f})  {sp:5.1f}  '
              f'{"%5.1f u" % gl if gl is not None else "   -  "}  {"%6.1f u" % det if det is not None else "    -   "}     {rise:4.2f}   '
              f'{appr if appr is not None else "-"!s:6}          {ge if ge is not None else "-"}')
    gaps = [r for r in rows if r[1] == 'GAP']
    if gaps:
        print(f'\nGAP jumps: {len(gaps)}; mean crossing {np.mean([r[6] for r in gaps]):.1f} u, mean detour saved '
              f'{np.mean([r[7] for r in gaps if r[7] is not None]):.1f} u, takeoff speed {np.mean([r[5] for r in gaps]):.1f} u/s; '
              f'physics would approve {sum(1 for r in gaps if r[9])}/{len(gaps)}; graph has an edge for {sum(1 for r in gaps if r[10])}/{len(gaps)}')
        np.save(os.path.join(os.path.dirname(path), 'human_gap_jumps.npy'), np.array([(r[2], r[3], r[5], r[6]) for r in gaps]))


if __name__ == '__main__':
    main()
