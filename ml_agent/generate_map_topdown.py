"""Render an accurate top-down view of a Torque .dif interior, plus the gem,
spawn, and powerup positions from the matching Hunt mission file.

Usage:
    python generate_map_topdown.py

Output: map_topdown.html (Plotly figure) + map_topdown.png (matplotlib).

Uses RandomityGuy's hxDif parser (vendored as ml_agent/hxDif.py) which handles
MBP-extended .dif version 44.
"""
import os
import sys
import math
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib import cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hxDif


# ---------------------------------------------------------------------------
# .dif parsing helpers
# ---------------------------------------------------------------------------

def fix_indices(indices):
    """Convert Torque tri-strip-style winding to polygon-fan order.

    Verbatim from RandomityGuy's io_dif/import_dif.py; required because
    .dif stores windings as a sequence that must be unscrambled.
    """
    n = len(indices)
    new_indices = [0] * n
    for i in range(n):
        if i >= 2:
            if i % 2 == 0:
                new_indices[n - 1 - (i - 2) // 2] = indices[i]
            else:
                new_indices[(i + 1) // 2] = indices[i]
        else:
            new_indices[i] = indices[i]
    return new_indices


def extract_surfaces(dif, interior_offset=(0.0, 0.0, 0.0)):
    """Extract every surface as a list of (vertices_xyz, normal_xyz).

    interior_offset is the position the .mcs places this interior at in
    world space — needed to align with gem world coords.
    """
    ox, oy, oz = interior_offset
    out = []
    for interior in dif.interiors:
        points = interior.points
        windings = interior.windings
        normals = interior.normals
        planes = interior.planes
        surfaces = interior.surfaces
        for s in surfaces:
            idx = windings[s.windingStart : s.windingStart + s.windingCount]
            idx = fix_indices(idx)
            idx = idx[::-1]  # reverse for correct normal direction
            verts = []
            for i in idx:
                p = points[i]
                verts.append((p.x + ox, p.y + oy, p.z + oz))
            # Surface normal
            plane = planes[s.planeIndex]
            n = normals[plane.normalIndex]
            nx, ny, nz = n.x, n.y, n.z
            if getattr(s, 'planeFlipped', False):
                nx, ny, nz = -nx, -ny, -nz
            out.append((verts, (nx, ny, nz)))
    return out


# ---------------------------------------------------------------------------
# Mission file parsing — extract gems, spawn points, powerups
# ---------------------------------------------------------------------------

def parse_mission_items(mcs_path):
    """Extract gem, spawn, and powerup XY positions from a .mcs file."""
    with open(mcs_path, 'r') as f:
        text = f.read()

    items = {'gems_red': [], 'gems_yellow': [], 'spawns': [],
             'powerups': []}

    # Item blocks (gems and powerups)
    for m in re.finditer(r'new Item\(\)\s*\{([^}]+)\}', text):
        block = m.group(1)
        pos_m = re.search(r'position\s*=\s*\"([^\"]+)\"', block)
        db_m = re.search(r'dataBlock\s*=\s*\"([^\"]+)\"', block)
        if not (pos_m and db_m):
            continue
        x, y, z = [float(v) for v in pos_m.group(1).split()]
        db = db_m.group(1)
        if db == 'GemItemRed_MBU':
            items['gems_red'].append((x, y, z))
        elif db == 'GemItemYellow_MBU':
            items['gems_yellow'].append((x, y, z))
        elif 'Item' in db:  # SuperJump, SuperSpeed, Blast, MegaMarble
            items['powerups'].append((x, y, z, db))

    # SpawnTrigger blocks
    for m in re.finditer(r'new Trigger\(\w*\)\s*\{([^}]+)\}|new Trigger\(\)\s*\{([^}]+)\}', text):
        block = m.group(1) or m.group(2)
        if 'SpawnTrigger' not in block:
            continue
        pos_m = re.search(r'position\s*=\s*\"([^\"]+)\"', block)
        if pos_m:
            x, y, z = [float(v) for v in pos_m.group(1).split()]
            items['spawns'].append((x, y, z))

    return items


def parse_interior_position(mcs_path, interior_filename):
    """Find the world position where the .mcs places this .dif interior."""
    with open(mcs_path, 'r') as f:
        text = f.read()
    # Look for an InteriorInstance whose interiorFile matches
    for m in re.finditer(r'new InteriorInstance\(\)\s*\{([^}]+)\}', text):
        block = m.group(1)
        if interior_filename in block:
            pos_m = re.search(r'position\s*=\s*\"([^\"]+)\"', block)
            if pos_m:
                return tuple(float(v) for v in pos_m.group(1).split())
    return (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_topdown(surfaces, items, out_path, label='Map',
                   floor_normal_threshold=0.5,
                   playable_z_min=None):
    """Render the top-down view. Floors (normal_z > threshold) shown filled
    and colored by elevation; walls (|normal_z| < threshold) shown as thin
    outlines so they appear as edges in the plan view.

    playable_z_min: if set, only include surfaces whose mean Z is above this
    threshold. Useful to filter out underground decorative geometry.
    """
    if not surfaces:
        print('No surfaces to render')
        return

    # Optional Z filter — drop surfaces below the play area
    if playable_z_min is not None:
        before = len(surfaces)
        surfaces = [s for s in surfaces
                    if np.mean([v[2] for v in s[0]]) >= playable_z_min]
        print(f'Z-filter (z >= {playable_z_min}): kept {len(surfaces)} of {before} surfaces')

    # Get bounds (after filtering)
    all_x, all_y, all_z = [], [], []
    for verts, _ in surfaces:
        for x, y, z in verts:
            all_x.append(x); all_y.append(y); all_z.append(z)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min, z_max = min(all_z), max(all_z)
    print(f'Geometry bounds: x[{x_min:.1f}, {x_max:.1f}]  '
          f'y[{y_min:.1f}, {y_max:.1f}]  z[{z_min:.1f}, {z_max:.1f}]')

    # Pad
    pad = 3.0
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a1a')

    # Separate floors and walls; sort floors low-to-high so high ones overlay
    floors = []
    walls = []
    for verts, normal in surfaces:
        if abs(normal[2]) > floor_normal_threshold:
            floors.append((verts, normal))
        else:
            walls.append((verts, normal))

    # Sort floors by mean z so highest z is drawn on top
    floors.sort(key=lambda s: np.mean([v[2] for v in s[0]]))

    # Color floors by mean z
    cmap = matplotlib.colormaps['viridis']
    z_range = max(z_max - z_min, 1.0)
    floor_polys = []
    floor_colors = []
    for verts, normal in floors:
        xy = [(v[0], v[1]) for v in verts]
        if len(xy) < 3:
            continue
        floor_polys.append(MplPolygon(xy, closed=True))
        mean_z = np.mean([v[2] for v in verts])
        t = (mean_z - z_min) / z_range
        floor_colors.append(cmap(t))

    floor_collection = PatchCollection(floor_polys, facecolors=floor_colors,
                                       edgecolors='#444444', linewidths=0.3,
                                       alpha=0.85)
    ax.add_collection(floor_collection)

    # Walls as outlines so we can see their plan-view profile
    wall_polys = []
    for verts, normal in walls:
        xy = [(v[0], v[1]) for v in verts]
        if len(xy) < 2:
            continue
        wall_polys.append(MplPolygon(xy, closed=True))
    wall_collection = PatchCollection(wall_polys, facecolors='none',
                                      edgecolors='#888888', linewidths=0.4,
                                      alpha=0.5)
    ax.add_collection(wall_collection)

    # Items overlay
    if items.get('gems_red'):
        xy = np.array([(p[0], p[1]) for p in items['gems_red']])
        ax.scatter(xy[:, 0], xy[:, 1], c='#ff4444', s=120, marker='D',
                   edgecolors='white', linewidths=1.5, label=f'Red gem (1pt) [{len(xy)}]', zorder=10)
    if items.get('gems_yellow'):
        xy = np.array([(p[0], p[1]) for p in items['gems_yellow']])
        ax.scatter(xy[:, 0], xy[:, 1], c='#ffdd44', s=160, marker='D',
                   edgecolors='white', linewidths=1.5, label=f'Yellow gem (2pt) [{len(xy)}]', zorder=10)
    if items.get('spawns'):
        xy = np.array([(p[0], p[1]) for p in items['spawns']])
        ax.scatter(xy[:, 0], xy[:, 1], c='#44ddff', s=80, marker='s',
                   edgecolors='white', linewidths=1.0, label=f'Spawn [{len(xy)}]', zorder=9)
    if items.get('powerups'):
        # Group by type for separate markers
        groups = {}
        for x, y, z, db in items['powerups']:
            key = db.replace('Item_MBU', '').replace('Item', '')
            groups.setdefault(key, []).append((x, y))
        markers = {'SuperJump': '^', 'SuperSpeed': '>',
                   'Blast': '*', 'MegaMarble': 'P'}
        colors = {'SuperJump': '#88ff88', 'SuperSpeed': '#ff88ff',
                  'Blast': '#ff8844', 'MegaMarble': '#dd44ff'}
        for key, pts in groups.items():
            xy = np.array(pts)
            ax.scatter(xy[:, 0], xy[:, 1], c=colors.get(key, '#cccccc'),
                       s=100, marker=markers.get(key, 'o'),
                       edgecolors='white', linewidths=1.0,
                       label=f'{key} [{len(xy)}]', zorder=9)

    # Colorbar for elevation
    sm = cm.ScalarMappable(cmap=cmap,
                           norm=plt.Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Elevation (z, world units)', color='#cccccc')
    cb.ax.yaxis.set_tick_params(color='#cccccc')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#cccccc')

    ax.set_title(f'{label} — top-down (floors colored by elevation, walls as outlines)',
                 color='#dddddd', fontsize=12)
    ax.set_xlabel('X (world units)', color='#cccccc')
    ax.set_ylabel('Y (world units)', color='#cccccc')
    ax.tick_params(colors='#cccccc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#666666')
    leg = ax.legend(loc='upper right', framealpha=0.85, facecolor='#222222',
                    edgecolor='#666666', labelcolor='#dddddd')
    ax.grid(True, alpha=0.15, color='#666666')

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, facecolor='#1a1a1a')
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)

    mcs_path = os.path.join(repo_root, 'Marble Blast Platinum', 'platinum',
                            'data', 'multiplayer', 'hunt', 'beginner',
                            'KingOfTheMarble_Hunt.mcs')
    dif_path = os.path.join(repo_root, 'Marble Blast Platinum', 'platinum',
                            'data', 'multiplayer', 'interiors', 'mbu',
                            'KingOfTheMarble.dif')

    print(f'Loading mission: {mcs_path}')
    print(f'Loading interior: {dif_path}')

    items = parse_mission_items(mcs_path)
    interior_pos = parse_interior_position(mcs_path, 'KingOfTheMarble.dif')
    print(f'Interior placed at world position: {interior_pos}')
    print(f'Items found:')
    print(f'  Red gems: {len(items["gems_red"])}')
    print(f'  Yellow gems: {len(items["gems_yellow"])}')
    print(f'  Spawn points: {len(items["spawns"])}')
    print(f'  Powerups: {len(items["powerups"])}')

    print('Parsing .dif...')
    dif = hxDif.Dif.Load(dif_path)
    print(f'  Version: {dif.difVersion}')
    print(f'  Interiors: {len(dif.interiors)}')

    surfaces = extract_surfaces(dif, interior_offset=interior_pos)
    print(f'  Total surfaces extracted: {len(surfaces)}')

    # Marble plays at z~20.4. Filter out everything below ~15 (underground
    # decorative bases of the corner pillars, foundation pieces, etc.).
    out_path = os.path.join(here, 'map_topdown.png')
    render_topdown(surfaces, items, out_path,
                   label='King of the Marble (Hunt)',
                   playable_z_min=15.0)


if __name__ == '__main__':
    main()
