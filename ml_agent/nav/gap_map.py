"""Gap map: what the navigator's TERRAIN side knows about every gap on a map (2026-09-23).

Two panels, written to logs/nav/gap_map_<map>.png:

  LEFT  "reward graph": walkable cells (grey), edge cells (white) and every JUMP EDGE the walk graph
        contains (green lines). These are the only gaps the PROGRESS reward field can ever route
        across: 8 compass directions from an edge cell, at most JUMP_GAP u wide, landing within
        [-JUMP_DROP, +JUMP_RISE]. Anything else is "go around".
  RIGHT "true gap per edge cell": from every edge cell, ray-march the fine 0.5 u height map in 16
        headings (the same 16 the observation's edge rays use) across the void until floor appears
        again within RAY_RANGE u with an acceptable landing height. Colour = the SHORTEST such gap:
        green <= JUMP_GAP (the graph can see it), yellow JUMP_GAP..2x, red wider, black = no floor
        within RAY_RANGE in any heading that leaves this cell over the void. The policy's crop shows the full shape; this panel shows
        what is crossable versus what the graph admits.

    python -m nav.gap_map KingOfTheMarble_Hunt
"""
import math
import os
import sys

import numpy as np

from terrain_obs import TerrainMap, RAY_RANGE, RAY_STEP, RAY_HEADINGS
from nav.terrain import TerrainGrid, JUMP_GAP, JUMP_DROP, JUMP_RISE
from nav.physics import MAX_JUMP_GAP


def jump_edges(t):
    """Re-run the graph's jump-edge rule and return [(j0, i0, j1, i1), ...] (walk cells)."""
    out = []
    max_cells = int(round(JUMP_GAP / t.walk_res))
    for (j, i) in np.argwhere(t.edge):
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            for k in range(2, max_cells + 1):
                jj, ii = j + dy * k, i + dx * k
                if not t.in_walk_grid(jj, ii):
                    break
                if t.walkable[jj, ii]:
                    if not any(t.walkable[j + dy * m, i + dx * m] for m in range(1, k)):
                        dz = t.walk_top[jj, ii] - t.walk_top[j, i]
                        if -JUMP_DROP <= dz <= JUMP_RISE:
                            out.append((j, i, jj, ii))
                    break
    return out


def true_gaps(t):
    """For every edge cell: shortest crossable gap (u) over 16 headings on the fine grid, or inf."""
    fine = t.heights                        # (K, H, W) at t.res
    present = np.isfinite(fine).any(0)
    H, W = present.shape
    res = t.res
    heads = [(math.cos(2 * math.pi * k / RAY_HEADINGS), math.sin(2 * math.pi * k / RAY_HEADINGS)) for k in range(RAY_HEADINGS)]
    steps = int(round(RAY_RANGE / RAY_STEP))
    result = {}
    for (j, i) in np.argwhere(t.edge):
        x = t.wxs[i]; y = t.wys[j]; z0 = t.walk_top[j, i]
        best = math.inf
        for cx, cy in heads:
            in_void = False; d_void = None
            for s in range(1, steps + 1):
                d = s * RAY_STEP
                fx = (x + cx * d - t.x0) / res; fy = (y + cy * d - t.y0) / res
                ii = int(round(fx)); jj = int(round(fy))
                if not (0 <= ii < W and 0 <= jj < H):
                    break
                if not present[jj, ii]:
                    if not in_void:
                        in_void = True; d_void = d
                    continue
                if not in_void:
                    if d > 1.5:
                        break              # this heading leaves the cell over floor: not a gap of THIS edge
                    continue
                if in_void:
                    zs = fine[:, jj, ii]; zs = zs[np.isfinite(zs)]
                    if len(zs) and np.any((zs - z0 >= -JUMP_DROP) & (zs - z0 <= JUMP_RISE)):
                        best = min(best, d - d_void + RAY_STEP)
                    break
        result[(j, i)] = best
    return result


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'KingOfTheMarble_Hunt'
    t = TerrainGrid(TerrainMap.resolve(name))
    je = [(j, i, jj, ii) for (j, i, jj, ii, g) in t.jump_edge_cells]   # the graph's OWN edges (measured rule)
    gaps = true_gaps(t)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ext = [t.wxs[0] - 0.5, t.wxs[-1] + 0.5, t.wys[0] - 0.5, t.wys[-1] + 0.5]
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    for a in ax:
        base = np.zeros(t.walkable.shape + (3,)) + 0.08
        base[t.walkable] = 0.35
        base[t.edge] = 0.9
        a.imshow(base, origin='lower', extent=ext, interpolation='nearest')
        a.set_xlabel('x (u)'); a.set_ylabel('y (u)'); a.set_aspect('equal')
    for (j, i, jj, ii) in je:
        ax[0].plot([t.wxs[i], t.wxs[ii]], [t.wys[j], t.wys[jj]], color='#3fb950', lw=1.2, alpha=0.9)
    ax[0].set_title(f'{name}: walk graph JUMP EDGES (green): {len(je)} directed, 16 headings, measured range <= {MAX_JUMP_GAP:.1f} u\n'
                    f'the PROGRESS reward can only route across these gaps')
    xs, ys, cs = [], [], []
    n_ok = n_mid = n_far = n_none = 0
    for (j, i), g in gaps.items():
        xs.append(t.wxs[i]); ys.append(t.wys[j])
        if g <= JUMP_GAP:
            cs.append('#3fb950'); n_ok += 1
        elif g <= 2 * JUMP_GAP:
            cs.append('#f0c040'); n_mid += 1
        elif math.isfinite(g):
            cs.append('#f85149'); n_far += 1
        else:
            cs.append('#000000'); n_none += 1
    ax[1].scatter(xs, ys, c=cs, s=14, marker='s', linewidths=0)
    ax[1].set_title(f'shortest TRUE gap from each edge cell, 16 headings, 0.5 u ray march\n'
                    f'green <= {JUMP_GAP:.0f} u: {n_ok}   yellow {JUMP_GAP:.0f}-{2*JUMP_GAP:.0f} u: {n_mid}   red > {2*JUMP_GAP:.0f} u: {n_far}   black: no floor within {RAY_RANGE:.0f} u: {n_none}')
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'nav', f'gap_map_{name}.png')
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print('wrote', out)
    print(f'edge cells {len(gaps)}: crossable <= {JUMP_GAP} u {n_ok}, {JUMP_GAP}-{2*JUMP_GAP} u {n_mid}, wider {n_far}, none {n_none}; graph jump edges {len(je)}')
    # per-gap-width histogram of the shortest crossing
    fin = sorted(g for g in gaps.values() if math.isfinite(g))
    if fin:
        hist = {}
        for g in fin:
            hist[round(g)] = hist.get(round(g), 0) + 1
        print('shortest-gap histogram (u: edge cells):', dict(sorted(hist.items())))


if __name__ == '__main__':
    main()
