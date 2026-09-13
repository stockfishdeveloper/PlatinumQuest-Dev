"""Build the local-terrain height stack for a map from its .dif geometry.

    python generate_terrain_map.py KingOfTheMarble_Hunt
    python generate_terrain_map.py Prophetic_Hunt --grid-res 0.5

Writes:
    terrain_maps/terrain_<name>.npz   height stack loaded by terrain_obs.TerrainMap
    terrain_maps/terrain_<name>.png   check image: top-floor height map plus the
                                      64-dim sample star at a few spawn/gem positions

Method (map-agnostic, no hand tuning):
  1. Find the mission file (.mcs/.mis) by name; read EVERY InteriorInstance
     (file + world position) and the InBoundsTrigger volume.
  2. Parse each .dif (vendored hxDif), keep upward-facing surfaces
     (normal_z > 0.5): floors, ramps, bevels.
  3. Rasterize onto one 2-D grid. For every cell inside a floor polygon,
     evaluate that polygon's plane at the cell centre, so ramps get their true
     height and multi-level maps get one height per level (up to K per cell).
  4. Drop anything outside the InBoundsTrigger's z-range: geometry below the
     out-of-bounds plane is decorative and must not read as a landing floor.
"""
import os
import re
import sys
import glob
import math
import argparse
import numpy as np
from matplotlib.path import Path as MplPath

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hxDif
from generate_map_topdown import extract_surfaces, parse_mission_items
from terrain_obs import TerrainMap

FLOOR_NORMAL_Z = 0.5
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PLATINUM_ROOT = os.path.join(REPO_ROOT, 'Marble Blast Platinum', 'platinum')
DATA_ROOT = os.path.join(PLATINUM_ROOT, 'data')


# ---------------------------------------------------------------------------
# Mission file parsing
# ---------------------------------------------------------------------------

def resolve_mission(name):
    """Find the mission file for a map name. Exact stem match wins, else the
    shortest path containing the name. Also accepts a path."""
    if os.path.isfile(name):
        return name
    cands = []
    for ext in ('.mcs', '.mis'):
        cands += glob.glob(os.path.join(DATA_ROOT, '**', f'*{name}*{ext}'), recursive=True)
    if not cands:
        raise FileNotFoundError(f'No .mcs/.mis matching "{name}" under {DATA_ROOT}')
    exact = [p for p in cands if os.path.splitext(os.path.basename(p))[0].lower() == name.lower()]
    if exact:
        cands = exact
    return sorted(cands, key=lambda p: (len(p), p))[0]


def _parse_vec(s):
    return tuple(float(v) for v in s.split())


def parse_interiors(mis_path):
    """All (dif_path, world_position) pairs placed by the mission."""
    text = open(mis_path, 'r', errors='replace').read()
    out = []
    for m in re.finditer(r'new InteriorInstance\(\)\s*\{([^}]+)\}', text):
        block = m.group(1)
        f = re.search(r'interiorFile\s*=\s*"([^"]+)"', block)
        if not f:
            continue
        pos_m = re.search(r'position\s*=\s*"([^"]+)"', block)
        rot_m = re.search(r'rotation\s*=\s*"([^"]+)"', block)
        scl_m = re.search(r'scale\s*=\s*"([^"]+)"', block)
        pos = _parse_vec(pos_m.group(1)) if pos_m else (0.0, 0.0, 0.0)
        rot = _parse_vec(rot_m.group(1)) if rot_m else (1.0, 0.0, 0.0, 0.0)
        scl = _parse_vec(scl_m.group(1)) if scl_m else (1.0, 1.0, 1.0)
        rel = f.group(1)
        if rel.startswith('~/'):
            dif = os.path.join(PLATINUM_ROOT, rel[2:])
        else:
            dif = os.path.join(os.path.dirname(mis_path), rel)
        dif = os.path.normpath(dif)
        if abs(rot[3]) > 1e-6 or any(abs(s - 1.0) > 1e-6 for s in scl):
            print(f'  WARNING: interior {os.path.basename(dif)} has rotation {rot} / scale {scl}; '
                  f'only translation is applied')
        if not os.path.exists(dif):
            print(f'  WARNING: interior file not found, skipped: {dif}')
            continue
        out.append((dif, pos))
    if not out:
        raise RuntimeError(f'No InteriorInstance with an existing .dif in {mis_path}')
    return out


def parse_in_bounds_z(mis_path):
    """(z_min, z_max) of the InBoundsTrigger volume, or (None, None)."""
    text = open(mis_path, 'r', errors='replace').read()
    for m in re.finditer(r'new Trigger\([^)]*\)\s*\{([^}]+)\}', text):
        block = m.group(1)
        if 'InBoundsTrigger' not in block:
            continue
        pos = _parse_vec(re.search(r'position\s*=\s*"([^"]+)"', block).group(1))
        scl_m = re.search(r'scale\s*=\s*"([^"]+)"', block)
        scl = _parse_vec(scl_m.group(1)) if scl_m else (1.0, 1.0, 1.0)
        poly_m = re.search(r'polyhedron\s*=\s*"([^"]+)"', block)
        poly = _parse_vec(poly_m.group(1)) if poly_m else (0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 1)
        o = np.array(poly[0:3]); v1 = np.array(poly[3:6]); v2 = np.array(poly[6:9]); v3 = np.array(poly[9:12])
        corners = []
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    p = o + a * v1 + b * v2 + c * v3
                    corners.append(pos[2] + p[2] * scl[2])
        return min(corners), max(corners)
    return None, None


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def build_height_stack(surfaces, grid_res, z_min, z_max, k_max):
    floors = [(v, n) for v, n in surfaces if n[2] > FLOOR_NORMAL_Z and len(v) >= 3]
    if not floors:
        raise RuntimeError('No floor surfaces found')

    pts = np.array([p for v, _ in floors for p in v], dtype=np.float64)
    pad = 2 * grid_res
    x_min, x_max = pts[:, 0].min() - pad, pts[:, 0].max() + pad
    y_min, y_max = pts[:, 1].min() - pad, pts[:, 1].max() + pad
    xs = np.arange(x_min, x_max + grid_res, grid_res)
    ys = np.arange(y_min, y_max + grid_res, grid_res)
    W, H = len(xs), len(ys)

    cell_lists = [[[] for _ in range(W)] for _ in range(H)]
    n_kept = n_dropped_bounds = 0
    for verts, normal in floors:
        vx = np.array([p[0] for p in verts]); vy = np.array([p[1] for p in verts])
        i0 = max(0, int(np.floor((vx.min() - xs[0]) / grid_res)) - 1)
        i1 = min(W - 1, int(np.ceil((vx.max() - xs[0]) / grid_res)) + 1)
        j0 = max(0, int(np.floor((vy.min() - ys[0]) / grid_res)) - 1)
        j1 = min(H - 1, int(np.ceil((vy.max() - ys[0]) / grid_res)) + 1)
        if i1 < i0 or j1 < j0:
            continue
        sub_x = xs[i0:i1 + 1]; sub_y = ys[j0:j1 + 1]
        gx, gy = np.meshgrid(sub_x, sub_y)
        cells = np.column_stack([gx.ravel(), gy.ravel()])
        inside = MplPath(list(zip(vx, vy))).contains_points(cells)
        if not inside.any():
            continue
        nx, ny, nz = normal
        v0 = verts[0]
        # plane through v0 with this normal: z = v0z - (nx (x - v0x) + ny (y - v0y)) / nz
        zs = v0[2] - (nx * (cells[:, 0] - v0[0]) + ny * (cells[:, 1] - v0[1])) / nz
        for (cx, cy), z, ok in zip(cells, zs, inside):
            if not ok:
                continue
            if (z_min is not None and z < z_min - 0.5) or (z_max is not None and z > z_max + 0.5):
                n_dropped_bounds += 1
                continue
            i = int(round((cx - xs[0]) / grid_res)); j = int(round((cy - ys[0]) / grid_res))
            cell_lists[j][i].append(float(z))
            n_kept += 1

    heights = np.full((k_max, H, W), np.nan, dtype=np.float32)
    max_used = 0
    n_walk = 0
    for j in range(H):
        for i in range(W):
            hs = cell_lists[j][i]
            if not hs:
                continue
            hs = sorted(set(round(h, 2) for h in hs), reverse=True)
            merged = []
            for h in hs:                      # merge heights within 0.25 (same surface, mesh seams)
                if not merged or merged[-1] - h > 0.25:
                    merged.append(h)
            max_used = max(max_used, len(merged))
            heights[:min(k_max, len(merged)), j, i] = merged[:k_max]
            n_walk += 1
    stats = dict(cells=W * H, walkable=n_walk, max_levels_used=max_used,
                 samples_kept=n_kept, samples_dropped_out_of_bounds=n_dropped_bounds)
    return xs.astype(np.float32), ys.astype(np.float32), heights, stats


# ---------------------------------------------------------------------------
# Check image
# ---------------------------------------------------------------------------

def render_check_image(tm, items, out_path, mapname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    top = tm.heights[0]
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor('#1a1a1a')
    extent = [tm.xs[0] - tm.res / 2, tm.xs[-1] + tm.res / 2, tm.ys[0] - tm.res / 2, tm.ys[-1] + tm.res / 2]
    for ax in axes:
        ax.set_facecolor('#1a1a1a')
        ax.imshow(top, origin='lower', extent=extent, cmap='viridis', interpolation='nearest')
        ax.tick_params(colors='#cccccc')
        ax.set_xlabel('X (world)', color='#dddddd'); ax.set_ylabel('Y (world)', color='#dddddd')
    im = axes[0].images[0]
    cb = fig.colorbar(im, ax=axes[0], fraction=0.04)
    cb.set_label('top floor z', color='#dddddd'); cb.ax.tick_params(colors='#cccccc')
    axes[0].set_title(f'Top floor height  -  {mapname}  ({tm.K}-level stack, {tm.res}u cells)', color='#dddddd')

    gems = items.get('gems_red', []) + items.get('gems_yellow', [])
    spawns = items.get('spawns', [])
    for gx, gy, _ in gems:
        axes[0].plot(gx, gy, 'D', color='#ff5555', ms=6, mec='white', mew=0.5)
    for sx, sy, _ in spawns:
        axes[0].plot(sx, sy, 's', color='#66ccff', ms=6, mec='white', mew=0.5)

    # Sample stars at a few positions: up to 3 spawns and 3 gems, marble on the top floor there
    probes = [(x, y) for x, y, _ in spawns[:3]] + [(x, y) for x, y, _ in gems[:3]]
    ax = axes[1]
    ax.set_title('Sample star at probe positions: green flat, orange lower, blue higher, red void '
                 '(marker size ~ |height difference|)', color='#dddddd')
    for (px, py) in probes:
        h = tm.heights_at(np.array([px]), np.array([py]))[:, 0]
        h = h[np.isfinite(h)]
        if len(h) == 0:
            continue
        pz = float(h[0]) + 0.2
        vec = tm.sample(px, py, pz, 0.0)
        pts = tm.sample_points(px, py, 0.0)
        ax.plot(px, py, 'o', color='white', ms=8, mec='black')
        for k in range(len(pts)):
            p, dz = vec[2 * k], vec[2 * k + 1]
            if p < 0.5:
                col, ms = '#ff3333', 4
            elif abs(dz) < 0.03:
                col, ms = '#33dd55', 4
            elif dz > 0:
                col, ms = '#4488ff', 3 + 8 * min(1.0, abs(dz))
            else:
                col, ms = '#ffaa22', 3 + 8 * min(1.0, abs(dz))
            ax.plot([px, pts[k, 0]], [py, pts[k, 1]], '-', color=col, lw=0.6, alpha=0.5)
            ax.plot(pts[k, 0], pts[k, 1], 'o', color=col, ms=ms)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, facecolor='#1a1a1a')
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mapname', help='map name (e.g. KingOfTheMarble_Hunt) or path to a .mcs/.mis')
    ap.add_argument('--grid-res', type=float, default=0.5)
    ap.add_argument('--k', type=int, default=4, help='max floor levels stored per cell')
    ap.add_argument('--z-min', type=float, default=None, help='override the in-bounds z floor')
    ap.add_argument('--z-max', type=float, default=None, help='override the in-bounds z ceiling')
    args = ap.parse_args()

    mis_path = resolve_mission(args.mapname)
    mapname = os.path.splitext(os.path.basename(mis_path))[0]
    print(f'Mission: {mis_path}')

    z_min, z_max = parse_in_bounds_z(mis_path)
    if args.z_min is not None: z_min = args.z_min
    if args.z_max is not None: z_max = args.z_max
    print(f'In-bounds z range: {z_min} .. {z_max}' if z_min is not None else
          'In-bounds trigger not found: no z filtering (pass --z-min if the map has decorative geometry below)')

    surfaces = []
    for dif_path, pos in parse_interiors(mis_path):
        dif = hxDif.Dif.Load(dif_path)
        s = extract_surfaces(dif, interior_offset=pos)
        print(f'  interior {os.path.basename(dif_path)} at {pos}: {len(s)} surfaces')
        surfaces += s

    xs, ys, heights, stats = build_height_stack(surfaces, args.grid_res, z_min, z_max, args.k)
    print(f'Grid {len(xs)}x{len(ys)} cells at {args.grid_res}u: {stats["walkable"]:,} with floor, '
          f'max levels in a cell {stats["max_levels_used"]} (stored {args.k}), '
          f'{stats["samples_dropped_out_of_bounds"]:,} samples dropped as out of bounds')
    if stats['max_levels_used'] > args.k:
        print(f'  NOTE: some cells have more than {args.k} floor levels; raise --k if this map needs them')

    out_dir = os.path.join(HERE, 'terrain_maps')
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f'terrain_{mapname}.npz')
    np.savez_compressed(npz_path, xs=xs, ys=ys, heights=heights, grid_res=np.float32(args.grid_res),
                        map_name=np.array(mapname),
                        z_bounds=np.array([z_min if z_min is not None else -1e9,
                                           z_max if z_max is not None else 1e9], dtype=np.float32))
    print(f'Saved {npz_path}')

    tm = TerrainMap(npz_path)
    try:
        items = parse_mission_items(mis_path)
    except Exception as e:
        print(f'  (mission items not parsed: {e})')
        items = {}
    png_path = os.path.join(out_dir, f'terrain_{mapname}.png')
    render_check_image(tm, items, png_path, mapname)
    print(f'Saved {png_path}')

    # Text dump of one probe so the encoding can be eyeballed in the console
    probes = items.get('spawns', [])[:1] or items.get('gems_red', [])[:1]
    for px, py, _ in probes:
        h = tm.heights_at(np.array([px]), np.array([py]))[:, 0]
        h = h[np.isfinite(h)]
        if len(h):
            pz = float(h[0]) + 0.2
            print(f'\nSample at ({px:.1f}, {py:.1f}, z={pz:.1f}):')
            print(tm.describe_sample(tm.sample(px, py, pz)))


if __name__ == '__main__':
    main()
