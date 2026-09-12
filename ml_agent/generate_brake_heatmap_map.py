"""Map-aware brake heatmap.

Probes the model's brake action probability at every walkable XY cell on a
real map (geometry from .dif). Only cells that lie on a floor surface
(normal_z > 0.5, at playable elevation) are probed — no point asking the
model about midair states.

Currently hardcoded for KingOfTheMarble_Hunt. Velocity at every cell is
fixed at FULL_SPEED east (+X), matching the convention from the original
flat-map brake heatmap. Current gem in the obs = nearest of the 12 real
gem positions on this map.
"""
import os
import sys
import math
import re
import numpy as np
import torch
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path as MplPath
from matplotlib import cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hxDif
from train_ppo import Actor
from generate_map_topdown import (extract_surfaces, parse_mission_items,
                                   parse_interior_position)

# ---------------------------------------------------------------------------
# Probe config
# ---------------------------------------------------------------------------

FULL_SPEED = 12.0
VEL_X = FULL_SPEED
VEL_Y = 0.0

GRID_RES = 1.0       # cell size in world units (smaller = finer, slower)
PLAYABLE_Z_MIN = 15.0  # filter out underground decorative geometry
FLOOR_NORMAL_Z = 0.5   # surfaces with normal_z above this count as floors

# Map files
MCS_NAME = 'KingOfTheMarble_Hunt.mcs'
DIF_NAME = 'KingOfTheMarble.dif'


# ---------------------------------------------------------------------------
# Obs builder (matches train_ppo.py / observer.cs convention)
# ---------------------------------------------------------------------------

def normalize_obs(obs):
    obs = obs.copy().astype(np.float32)
    obs[0:3] /= 100.0
    obs[3:6] /= 20.0
    for i in range(5):
        b = 6 + i*5
        if obs[b+4] < -500:
            obs[b:b+3] = 0.0
            obs[b+3] = 0.0
            obs[b+4] = 1.0
        else:
            if obs[b+4] > 0.01:
                obs[b:b+3] /= obs[b+4]
            else:
                obs[b:b+3] = 0.0
            obs[b+3] /= 5.0
            obs[b+4] /= 100.0
    obs[31] /= 300000.0
    obs[32] /= 300000.0
    obs[33] /= 100.0
    obs[34] /= 50.0
    return np.clip(obs, -2.0, 2.0)


def make_obs(marble_xyz, marble_vel, gem_xyz):
    obs = np.zeros(59, dtype=np.float32)
    mx, my, mz = marble_xyz
    vx, vy = marble_vel
    obs[0] = mx; obs[1] = my; obs[2] = mz
    obs[3] = vx; obs[4] = vy; obs[5] = 0
    gx, gy, gz = gem_xyz
    obs[6] = gx - mx; obs[7] = gy - my; obs[8] = gz - mz
    obs[9] = 1
    obs[10] = math.sqrt((gx-mx)**2 + (gy-my)**2 + (gz-mz)**2)
    for i in range(1, 5):
        obs[6 + i*5 + 4] = -999  # sentinel: no gem
    obs[31] = 150000
    obs[32] = 150000
    obs[33] = 30
    obs[34] = 3
    for i in range(4):
        obs[35 + i*6 + 0] = mx / 100.0
        obs[35 + i*6 + 1] = my / 100.0
        obs[35 + i*6 + 2] = mz / 100.0
        obs[35 + i*6 + 3] = vx / 20.0
        obs[35 + i*6 + 4] = vy / 20.0
    return normalize_obs(obs)


# ---------------------------------------------------------------------------
# Walkable-cell rasterization
# ---------------------------------------------------------------------------

def rasterize_floors(surfaces, grid_res, playable_z_min, normal_z_min):
    """Build per-cell (x, y, floor_z) for cells that lie on a floor surface.

    Returns:
        xs, ys: 1-D arrays of cell centers (full grid)
        floor_z: 2-D array (len(ys), len(xs)) — NaN if cell is not walkable,
                 else the elevation of the floor at that cell
        floor_polys_xy: list of (vertices, mean_z) for rendering the shape
    """
    floors = []
    for verts, normal in surfaces:
        if normal[2] <= normal_z_min:
            continue
        mean_z = np.mean([v[2] for v in verts])
        if mean_z < playable_z_min:
            continue
        xy = [(v[0], v[1]) for v in verts]
        if len(xy) < 3:
            continue
        floors.append((xy, mean_z))

    if not floors:
        raise RuntimeError('No floor surfaces found')

    # Bounds
    all_xy = np.array([p for poly, _ in floors for p in poly])
    x_min = math.floor(all_xy[:, 0].min())
    x_max = math.ceil(all_xy[:, 0].max())
    y_min = math.floor(all_xy[:, 1].min())
    y_max = math.ceil(all_xy[:, 1].max())

    xs = np.arange(x_min + grid_res / 2, x_max, grid_res)
    ys = np.arange(y_min + grid_res / 2, y_max, grid_res)
    floor_z = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)

    # For each floor polygon, mark cells inside it. Higher floors win
    # (we want the marble standing on the highest surface beneath it).
    cell_pts = np.array([(x, y) for y in ys for x in xs])
    for poly, mean_z in floors:
        path = MplPath(poly)
        inside = path.contains_points(cell_pts)
        inside_grid = inside.reshape(len(ys), len(xs))
        # Higher-z floor takes precedence
        update = inside_grid & (np.isnan(floor_z) | (mean_z > floor_z))
        floor_z[update] = mean_z

    return xs, ys, floor_z, floors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)

    mcs_path = os.path.join(repo_root, 'Marble Blast Platinum', 'platinum',
                            'data', 'multiplayer', 'hunt', 'beginner',
                            MCS_NAME)
    dif_path = os.path.join(repo_root, 'Marble Blast Platinum', 'platinum',
                            'data', 'multiplayer', 'interiors', 'mbu',
                            DIF_NAME)

    # Load checkpoint
    ckpts = sorted(glob.glob(os.path.join(here, 'models', 'checkpoints', 'update_*.pth')),
                   key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))
    if not ckpts:
        print('No checkpoints found')
        sys.exit(1)
    latest = ckpts[-1]
    label = os.path.basename(latest).replace('.pth', '')
    print(f'Probing checkpoint: {latest}')

    model = Actor(obs_dim=59)
    cp = torch.load(latest, weights_only=False, map_location='cpu')
    model.load_state_dict(cp['actor_state_dict'])
    model.eval()

    # Parse map
    print('Parsing .dif and .mcs ...')
    items = parse_mission_items(mcs_path)
    interior_pos = parse_interior_position(mcs_path, DIF_NAME)
    dif = hxDif.Dif.Load(dif_path)
    surfaces = extract_surfaces(dif, interior_offset=interior_pos)
    print(f'  Surfaces: {len(surfaces)}, interior at world {interior_pos}')

    # Rasterize floors
    print(f'Rasterizing floors at {GRID_RES}-unit resolution ...')
    xs, ys, floor_z, floors = rasterize_floors(
        surfaces, GRID_RES, PLAYABLE_Z_MIN, FLOOR_NORMAL_Z)
    walkable_mask = ~np.isnan(floor_z)
    n_walk = int(walkable_mask.sum())
    print(f'  Grid: {len(xs)} x {len(ys)} cells, '
          f'{n_walk} walkable ({100*n_walk/walkable_mask.size:.1f}%)')

    # All real gems on this map (red + yellow)
    all_gems = items['gems_red'] + items['gems_yellow']
    print(f'  Real gems on map: {len(all_gems)}')

    # Build obs for every walkable cell. Velocity = full speed east.
    # Current gem = nearest of the 12 real gems to that cell.
    walk_cells = []
    obs_batch = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if not walkable_mask[j, i]:
                continue
            z = floor_z[j, i] + 0.3
            # Nearest real gem
            best_d = float('inf')
            best_gem = None
            for gx, gy, gz in all_gems:
                d = (gx - x) ** 2 + (gy - y) ** 2 + (gz - z) ** 2
                if d < best_d:
                    best_d = d; best_gem = (gx, gy, gz)
            walk_cells.append((j, i))
            obs_batch.append(make_obs((float(x), float(y), float(z)),
                                       (VEL_X, VEL_Y), best_gem))

    obs_batch = np.array(obs_batch, dtype=np.float32)
    print(f'Probing {len(obs_batch):,} walkable cells ...')
    t = torch.from_numpy(obs_batch)
    with torch.no_grad():
        _, _, _, brake_logits = model(t)
        brake_probs = torch.sigmoid(brake_logits).cpu().numpy().squeeze(-1)

    brake_grid = np.full(walkable_mask.shape, np.nan, dtype=np.float32)
    for (j, i), p in zip(walk_cells, brake_probs):
        brake_grid[j, i] = p

    print(f'\nBrake probability stats (walkable cells, vel=({VEL_X:+.0f},0)):')
    walk_probs = brake_grid[walkable_mask]
    print(f'  min={walk_probs.min()*100:.2f}%  max={walk_probs.max()*100:.2f}%  '
          f'mean={walk_probs.mean()*100:.2f}%  std={walk_probs.std()*100:.2f}pp')

    # West-of-gem vs east-of-gem analysis
    west_mask = np.zeros_like(brake_grid, dtype=bool)
    east_mask = np.zeros_like(brake_grid, dtype=bool)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if not walkable_mask[j, i]:
                continue
            for gx, gy, _gz in all_gems:
                dx_to_gem = gx - x
                dy_to_gem = gy - y
                d = math.sqrt(dx_to_gem**2 + dy_to_gem**2)
                if d < 1.0 or d > 8.0: continue
                if abs(dy_to_gem) > 3.0: continue
                if dx_to_gem > 0:
                    west_mask[j, i] = True
                elif dx_to_gem < 0:
                    east_mask[j, i] = True
    if west_mask.any() and east_mask.any():
        wa = brake_grid[west_mask].mean()
        ea = brake_grid[east_mask].mean()
        print(f'  West-of-gem (heading toward, narrow cone): {wa*100:.2f}% (n={west_mask.sum()})')
        print(f'  East-of-gem  (heading away,  narrow cone): {ea*100:.2f}% (n={east_mask.sum()})')
        print(f'  W-E discrimination: {(wa-ea)*100:+.2f}pp (positive = direction-aware brake)')

    # ---------------------------------------------------------------
    # Render
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 12))
    ax.set_facecolor('#1a1a1a')

    # 1) Brake probability overlay (the colored walkable cells ARE the map shape)
    cmap = matplotlib.colormaps['inferno']
    p_min = float(walk_probs.min())
    p_max = float(walk_probs.max())
    extent = (xs[0] - GRID_RES/2, xs[-1] + GRID_RES/2,
              ys[0] - GRID_RES/2, ys[-1] + GRID_RES/2)
    masked = np.ma.array(brake_grid, mask=~walkable_mask)
    im = ax.imshow(masked, cmap=cmap, origin='lower', extent=extent,
                   vmin=p_min, vmax=p_max, alpha=1.0,
                   interpolation='nearest', zorder=1)

    # 2) Floor polygon outlines for crisp map shape (no fill, drawn on top)
    floor_polys = [MplPolygon(poly, closed=True) for poly, _ in floors]
    floor_collection = PatchCollection(floor_polys,
                                       facecolors='none',
                                       edgecolors='#888888',
                                       linewidths=0.5, alpha=0.6, zorder=2)
    ax.add_collection(floor_collection)

    # 3) Items overlay (gems, spawns, powerups)
    if items['gems_red']:
        xy = np.array([(p[0], p[1]) for p in items['gems_red']])
        ax.scatter(xy[:, 0], xy[:, 1], c='#ff4444', s=100, marker='D',
                   edgecolors='white', linewidths=1.2,
                   label=f'Red gem [{len(xy)}]', zorder=20)
    if items['gems_yellow']:
        xy = np.array([(p[0], p[1]) for p in items['gems_yellow']])
        ax.scatter(xy[:, 0], xy[:, 1], c='#ffdd44', s=130, marker='D',
                   edgecolors='white', linewidths=1.2,
                   label=f'Yellow gem [{len(xy)}]', zorder=20)
    if items['spawns']:
        xy = np.array([(p[0], p[1]) for p in items['spawns']])
        ax.scatter(xy[:, 0], xy[:, 1], c='#44ddff', s=60, marker='s',
                   edgecolors='white', linewidths=0.8,
                   label=f'Spawn [{len(xy)}]', zorder=19, alpha=0.7)

    # Velocity-direction indicator: small east-pointing arrows
    arrow_step = 8
    ax_w = xs[-1] - xs[0]
    arrow_len = ax_w * 0.04
    for ax_i in np.arange(xs[0] + arrow_step, xs[-1], arrow_step):
        for ay_i in np.arange(ys[0] + arrow_step, ys[-1], arrow_step):
            # Only show arrow if in/near walkable area
            jx = int(round(ay_i - ys[0]))
            ix = int(round(ax_i - xs[0]))
            if 0 <= jx < len(ys) and 0 <= ix < len(xs):
                if walkable_mask[jx, ix]:
                    ax.annotate('', xy=(ax_i + arrow_len, ay_i),
                                xytext=(ax_i, ay_i),
                                arrowprops=dict(arrowstyle='->',
                                                color='#ff5555',
                                                alpha=0.6, lw=0.8),
                                zorder=8)

    ax.set_aspect('equal')
    ax.set_title(f'Brake Probability — {label} on KingOfTheMarble | '
                 f'velocity=({VEL_X:.0f}, 0) (full speed east) | '
                 f'min={p_min*100:.1f}% max={p_max*100:.1f}% '
                 f'mean={walk_probs.mean()*100:.1f}%',
                 color='#dddddd', fontsize=11)
    ax.set_xlabel('X (world units)', color='#cccccc')
    ax.set_ylabel('Y (world units)', color='#cccccc')
    ax.tick_params(colors='#cccccc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#666666')

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Brake probability', color='#cccccc')
    cb.ax.yaxis.set_tick_params(color='#cccccc')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#cccccc')

    leg = ax.legend(loc='upper right', framealpha=0.85,
                    facecolor='#222222', edgecolor='#666666',
                    labelcolor='#dddddd', fontsize=9)

    out_path = os.path.join(here, 'brake_heatmap_kotm.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, facecolor='#1a1a1a')
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
