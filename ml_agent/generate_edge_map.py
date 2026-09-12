"""Map-agnostic edge detector for the void-avoidance reward shaping.

Given any .dif (Marble Blast interior), this script:
  1. Parses all floor surfaces (normal_z > 0.5) at any elevation
  2. Clusters them by Z to handle multi-level maps
  3. Rasterizes each level onto a 2D grid
  4. Finds "edge cells" — walkable cells with at least one non-walkable
     neighbor (8-connectivity). These are the boundary points where the
     marble could fall off.
  5. Outputs:
     - edges_<MAPNAME>.npy : numpy array of (x, y, z) edge point coords for
       loading at training time
     - edges_<MAPNAME>.html : interactive 3D Plotly visualization where the
       user can rotate and verify edges match the actual map geometry

Usage:
    python generate_edge_map.py prophetic
    python generate_edge_map.py KingOfTheMarble
    python generate_edge_map.py <mapname>   # also accepts paths

Will resolve .dif/.mis files automatically by searching for the map name.
"""
import os
import sys
import math
import glob
import numpy as np
import argparse
from matplotlib.path import Path as MplPath

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hxDif
from generate_map_topdown import extract_surfaces, parse_interior_position


def find_map_files(name):
    """Resolve mapname -> (mcs_or_mis_path, dif_path, interior_world_pos)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_root = os.path.join(repo_root, 'Marble Blast Platinum', 'platinum', 'data')

    # Find mission file
    mis_candidates = []
    for ext in ('.mcs', '.mis'):
        for p in glob.glob(os.path.join(search_root, '**', f'*{name}*{ext}'),
                           recursive=True):
            mis_candidates.append(p)
    if not mis_candidates:
        raise FileNotFoundError(f'No .mcs/.mis matching "{name}" found under {search_root}')
    # Prefer hunt variants if user said the bare name
    mis_path = sorted(mis_candidates, key=lambda p: (len(p), p))[0]
    print(f'Mission file: {mis_path}')

    # Parse interiorFile + position from mission
    with open(mis_path, 'r') as f:
        text = f.read()
    import re
    interior_file = None
    interior_pos = (0.0, 0.0, 0.0)
    for m in re.finditer(r'new InteriorInstance\(\)\s*\{([^}]+)\}', text):
        block = m.group(1)
        if_m = re.search(r'interiorFile\s*=\s*\"([^\"]+)\"', block)
        pos_m = re.search(r'position\s*=\s*\"([^\"]+)\"', block)
        if if_m:
            # First non-trigger InteriorInstance — usually the main level.
            interior_file = if_m.group(1)
            if pos_m:
                interior_pos = tuple(float(v) for v in pos_m.group(1).split())
            break
    if not interior_file:
        raise RuntimeError(f'No interiorFile found in {mis_path}')

    # Resolve ~/ — in MBP this is the platinum/ userMods root, parent of search_root (which is .../platinum/data)
    platinum_root = os.path.dirname(search_root)
    if interior_file.startswith('~/'):
        dif_path = os.path.join(platinum_root, interior_file[2:])
    else:
        dif_path = os.path.join(os.path.dirname(mis_path), interior_file)
    dif_path = os.path.normpath(dif_path)
    if not os.path.exists(dif_path):
        raise FileNotFoundError(f'Resolved .dif not found: {dif_path}')

    return mis_path, dif_path, interior_pos


def cluster_by_z(floors, z_tolerance=0.5):
    """Group floor surfaces by their mean Z, clustering surfaces within
    z_tolerance into the same elevation level. Returns list of
    (representative_z, [(verts, normal), ...])."""
    by_z = []
    for verts, normal in floors:
        mz = float(np.mean([v[2] for v in verts]))
        matched = False
        for i, (lvl_z, group) in enumerate(by_z):
            if abs(lvl_z - mz) <= z_tolerance:
                group.append((verts, normal))
                # Update representative z as running mean
                new_z = (lvl_z * (len(group) - 1) + mz) / len(group)
                by_z[i] = (new_z, group)
                matched = True
                break
        if not matched:
            by_z.append((mz, [(verts, normal)]))
    return by_z


def rasterize_level(surfaces_at_level, grid_res=0.5):
    """Rasterize floors at a single Z-level onto a 2D grid.

    Returns (xs, ys, walkable_mask)."""
    if not surfaces_at_level:
        return np.array([]), np.array([]), np.array([[]], dtype=bool)
    all_xy = np.array([(v[0], v[1]) for verts, _ in surfaces_at_level for v in verts])
    x_min, x_max = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
    y_min, y_max = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
    # Pad slightly so boundary cells are clearly outside
    pad = grid_res * 2
    xs = np.arange(x_min - pad, x_max + pad + grid_res, grid_res)
    ys = np.arange(y_min - pad, y_max + pad + grid_res, grid_res)

    walkable = np.zeros((len(ys), len(xs)), dtype=bool)
    cell_pts = np.array([(x, y) for y in ys for x in xs])
    for verts, _ in surfaces_at_level:
        xy = [(v[0], v[1]) for v in verts]
        if len(xy) < 3:
            continue
        path = MplPath(xy)
        inside = path.contains_points(cell_pts).reshape(len(ys), len(xs))
        walkable |= inside

    return xs, ys, walkable


def detect_edge_cells(walkable_mask):
    """For each walkable cell, check if any of its 8 neighbors is NOT walkable.
    Returns a boolean mask of edge cells (same shape as walkable_mask)."""
    if walkable_mask.size == 0:
        return np.zeros_like(walkable_mask)
    edges = np.zeros_like(walkable_mask)
    H, W = walkable_mask.shape
    for j in range(H):
        for i in range(W):
            if not walkable_mask[j, i]:
                continue
            # Check 8 neighbors
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    if dj == 0 and di == 0:
                        continue
                    nj, ni = j + dj, i + di
                    if nj < 0 or nj >= H or ni < 0 or ni >= W:
                        edges[j, i] = True
                        break
                    if not walkable_mask[nj, ni]:
                        edges[j, i] = True
                        break
                if edges[j, i]:
                    break
    return edges


def build_edge_map(dif_path, interior_pos, grid_res=0.5, z_tolerance=0.5,
                   playable_z_min=None):
    """Top-level: load .dif, extract floors, cluster by Z, detect edges per level.
    Returns:
        edge_points: (N, 3) numpy array of (x, y, z) edge point world coords
        levels: list of dicts with per-level data for rendering
    """
    print(f'Loading {dif_path} ...')
    dif = hxDif.Dif.Load(dif_path)
    print(f'  Version {dif.difVersion}, {len(dif.interiors)} interior(s)')

    surfaces = extract_surfaces(dif, interior_offset=interior_pos)
    print(f'  {len(surfaces)} total surfaces')

    # Filter to floor surfaces (upward-facing)
    floors = [(v, n) for v, n in surfaces if n[2] > 0.5]
    if playable_z_min is not None:
        floors = [(v, n) for v, n in floors
                  if np.mean([p[2] for p in v]) >= playable_z_min]
    print(f'  {len(floors)} floor surfaces (normal_z > 0.5)')

    # Cluster by elevation
    levels = cluster_by_z(floors, z_tolerance=z_tolerance)
    levels.sort(key=lambda L: L[0])
    print(f'  {len(levels)} distinct Z-levels detected:')
    for z, group in levels:
        print(f'    z={z:6.2f}  ({len(group)} surfaces)')

    # Process each level
    all_edges = []
    level_data = []
    for z, surfaces_at_level in levels:
        xs, ys, walkable = rasterize_level(surfaces_at_level, grid_res=grid_res)
        edges = detect_edge_cells(walkable)
        edge_xy = []
        for j in range(edges.shape[0]):
            for i in range(edges.shape[1]):
                if edges[j, i]:
                    edge_xy.append((xs[i], ys[j], z))
        edge_xy_arr = np.array(edge_xy, dtype=np.float32) if edge_xy else np.zeros((0, 3))
        all_edges.append(edge_xy_arr)
        level_data.append({
            'z': z,
            'surfaces': surfaces_at_level,
            'xs': xs,
            'ys': ys,
            'walkable': walkable,
            'edges': edges,
            'edge_pts': edge_xy_arr,
        })

    edge_pts = np.concatenate(all_edges) if all_edges else np.zeros((0, 3))
    print(f'  TOTAL edge points: {len(edge_pts):,}')
    return edge_pts, level_data


def render_3d_matplotlib(level_data, mapname, edge_pts, out_path):
    """Matplotlib 3D static rendering — for display in chat and quick verification.
    Plotly HTML is the interactive version; this is a one-off snapshot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')

    # Floor polygons
    polys = []
    for lvl in level_data:
        for verts, _ in lvl['surfaces']:
            if len(verts) < 3:
                continue
            polys.append(verts)
    coll = Poly3DCollection(polys, alpha=0.35, facecolor='#88ccff',
                            edgecolor='#666666', linewidth=0.2)
    ax.add_collection3d(coll)

    # Edge points
    if len(edge_pts) > 0:
        ax.scatter(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2] + 0.15,
                   c='red', s=4, alpha=0.85,
                   label=f'Edge points ({len(edge_pts):,})')

    # Set bounds
    all_xy = np.array([(v[0], v[1]) for lvl in level_data
                       for verts, _ in lvl['surfaces'] for v in verts])
    all_z = np.array([v[2] for lvl in level_data
                      for verts, _ in lvl['surfaces'] for v in verts])
    if len(all_xy):
        ax.set_xlim(all_xy[:, 0].min(), all_xy[:, 0].max())
        ax.set_ylim(all_xy[:, 1].min(), all_xy[:, 1].max())
        ax.set_zlim(all_z.min(), all_z.max() + 2)

    ax.set_xlabel('X', color='#dddddd')
    ax.set_ylabel('Y', color='#dddddd')
    ax.set_zlabel('Z', color='#dddddd')
    ax.set_title(f'Edge map — {mapname}\n{len(edge_pts):,} edge points across {len(level_data)} Z-levels',
                 color='#dddddd', fontsize=12)
    ax.tick_params(colors='#cccccc')
    if len(edge_pts):
        ax.legend(loc='upper right', facecolor='#222222', edgecolor='#666666',
                  labelcolor='#dddddd', fontsize=9)
    # Better viewing angle
    ax.view_init(elev=35, azim=-60)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor='#1a1a1a')
    plt.close()
    print(f'  Static 3D PNG: {out_path}')


def render_3d_plotly(level_data, mapname, edge_pts, out_path):
    """Interactive 3D Plotly visualization for user verification.

    - Floor surfaces: light blue translucent
    - Edge points: red dots (the boundary cells the reward shaping will use)
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Floor surfaces as Mesh3d (one per surface, triangulate as fan from v0)
    # For performance group them by level
    for lvl in level_data:
        for verts, _normal in lvl['surfaces']:
            if len(verts) < 3:
                continue
            xs_v = [v[0] for v in verts]
            ys_v = [v[1] for v in verts]
            zs_v = [v[2] for v in verts]
            # Fan triangulation
            i_idx, j_idx, k_idx = [], [], []
            for n in range(1, len(verts) - 1):
                i_idx.append(0); j_idx.append(n); k_idx.append(n + 1)
            fig.add_trace(go.Mesh3d(
                x=xs_v, y=ys_v, z=zs_v,
                i=i_idx, j=j_idx, k=k_idx,
                color='lightblue', opacity=0.35,
                flatshading=True, showlegend=False,
                hoverinfo='skip',
            ))

    # Edge points as scatter
    if len(edge_pts) > 0:
        fig.add_trace(go.Scatter3d(
            x=edge_pts[:, 0], y=edge_pts[:, 1], z=edge_pts[:, 2] + 0.15,
            mode='markers',
            marker=dict(size=2.5, color='red', opacity=0.85),
            name=f'Edge points ({len(edge_pts):,})',
            hovertemplate='Edge<br>x=%{x:.1f}, y=%{y:.1f}, z=%{z:.1f}<extra></extra>',
        ))

    fig.update_layout(
        title=f'Edge map — {mapname} ({len(edge_pts):,} edge points across {len(level_data)} Z-level(s))',
        scene=dict(
            xaxis=dict(title='X', backgroundcolor='#1a1a1a', color='#dddddd', gridcolor='#444'),
            yaxis=dict(title='Y', backgroundcolor='#1a1a1a', color='#dddddd', gridcolor='#444'),
            zaxis=dict(title='Z', backgroundcolor='#1a1a1a', color='#dddddd', gridcolor='#444'),
            aspectmode='data',
            bgcolor='#1a1a1a',
        ),
        paper_bgcolor='#1a1a1a',
        font=dict(color='#dddddd'),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f'  3D visualization: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mapname', help='Map name to search for (e.g., "prophetic", "KingOfTheMarble")')
    parser.add_argument('--grid-res', type=float, default=0.5,
                        help='Rasterization resolution in world units (default 0.5)')
    parser.add_argument('--z-tolerance', type=float, default=0.5,
                        help='Z clustering tolerance — surfaces within this Z range are treated as one level (default 0.5)')
    parser.add_argument('--playable-z-min', type=float, default=None,
                        help='Filter out floors below this Z (e.g., underground decorative geometry)')
    args = parser.parse_args()

    mis_path, dif_path, interior_pos = find_map_files(args.mapname)
    print(f'Interior at world position: {interior_pos}\n')

    edge_pts, level_data = build_edge_map(
        dif_path, interior_pos,
        grid_res=args.grid_res, z_tolerance=args.z_tolerance,
        playable_z_min=args.playable_z_min,
    )

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, 'edge_maps')
    os.makedirs(out_dir, exist_ok=True)
    mapname = os.path.basename(args.mapname).split('.')[0]

    np_path = os.path.join(out_dir, f'edges_{mapname}.npy')
    np.save(np_path, edge_pts)
    print(f'  Saved edge points: {np_path}')

    html_path = os.path.join(out_dir, f'edges_{mapname}.html')
    render_3d_plotly(level_data, mapname, edge_pts, html_path)

    png_path = os.path.join(out_dir, f'edges_{mapname}.png')
    render_3d_matplotlib(level_data, mapname, edge_pts, png_path)


if __name__ == '__main__':
    main()
