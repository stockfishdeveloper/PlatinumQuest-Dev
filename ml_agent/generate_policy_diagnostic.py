"""Diagnostic: does the model perceive the holes on KOTM?

For every walkable cell on the map, probes:
  1. Actor's mean direction (where the policy WANTS to go)
  2. Critic's V(s) (how "good" the model thinks this state is)

Renders side-by-side:
  - Left: vector field of preferred direction, overlaid on map shape
  - Right: V(s) heatmap

If arrows at cells near voids point INTO the void: model is oblivious to holes.
If V(s) drops sharply at near-void cells: model perceives danger.

If both: model knows about holes, ignoring them (reward problem).
If neither: model is genuinely blind to map geometry (obs problem).
If only V(s) low: critic sees danger but policy hasn't aligned.
If only actor avoiding: visual artifact, not really avoiding.

Velocity at each probe is set to 0 (testing "what would you do here standing
still") so the result reflects the model's positional knowledge alone, not
reactive responses to velocity.
"""
import os
import sys
import math
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
from train_ppo import Actor, Critic
from generate_map_topdown import (extract_surfaces, parse_mission_items,
                                   parse_interior_position)
from generate_brake_heatmap_map import (rasterize_floors, normalize_obs, make_obs)


# Probe config
GRID_RES = 1.0
PLAYABLE_Z_MIN = 15.0
FLOOR_NORMAL_Z = 0.5

# Velocity for probe: zero by default (testing positional knowledge)
PROBE_VX = 0.0
PROBE_VY = 0.0

# Map files
MCS_NAME = 'KingOfTheMarble_Hunt.mcs'
DIF_NAME = 'KingOfTheMarble.dif'


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)

    mcs_path = os.path.join(repo_root, 'Marble Blast Platinum', 'platinum',
                            'data', 'multiplayer', 'hunt', 'beginner', MCS_NAME)
    dif_path = os.path.join(repo_root, 'Marble Blast Platinum', 'platinum',
                            'data', 'multiplayer', 'interiors', 'mbu', DIF_NAME)

    # Latest checkpoint
    ckpts = sorted(glob.glob(os.path.join(here, 'models', 'checkpoints', 'update_*.pth')),
                   key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))
    latest = ckpts[-1]
    label = os.path.basename(latest).replace('.pth', '')
    print(f'Probing checkpoint: {latest}')

    actor = Actor(obs_dim=59)
    critic = Critic(obs_dim=59)
    cp = torch.load(latest, weights_only=False, map_location='cpu')
    actor.load_state_dict(cp['actor_state_dict'])
    critic.load_state_dict(cp['critic_state_dict'], strict=False)  # older checkpoints lack the PopArt buffers
    actor.eval()
    critic.eval()

    # Parse map
    items = parse_mission_items(mcs_path)
    interior_pos = parse_interior_position(mcs_path, DIF_NAME)
    dif = hxDif.Dif.Load(dif_path)
    surfaces = extract_surfaces(dif, interior_offset=interior_pos)
    print(f'Surfaces: {len(surfaces)}')

    # Walkable cells
    xs, ys, floor_z, floors = rasterize_floors(surfaces, GRID_RES, PLAYABLE_Z_MIN, FLOOR_NORMAL_Z)
    walkable_mask = ~np.isnan(floor_z)
    n_walk = int(walkable_mask.sum())
    print(f'Walkable cells: {n_walk}')

    all_gems = items['gems_red'] + items['gems_yellow']

    # Build batched obs: marble at each walkable cell, stationary, nearest gem
    walk_cells = []
    obs_batch = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if not walkable_mask[j, i]:
                continue
            z = floor_z[j, i] + 0.3
            best_d = float('inf')
            best_gem = None
            for gx, gy, gz in all_gems:
                d = (gx - x) ** 2 + (gy - y) ** 2 + (gz - z) ** 2
                if d < best_d:
                    best_d = d; best_gem = (gx, gy, gz)
            walk_cells.append((j, i))
            obs_batch.append(make_obs((float(x), float(y), float(z)),
                                       (PROBE_VX, PROBE_VY), best_gem))

    obs_batch = np.array(obs_batch, dtype=np.float32)
    print(f'Probing {len(obs_batch):,} cells...')
    t = torch.from_numpy(obs_batch)
    with torch.no_grad():
        mean_xy, _throttle_logit, _jump_logit, _brake_logit = actor(t)
        values = critic(t).squeeze(-1).cpu().numpy()
        mean_xy_np = mean_xy.cpu().numpy()  # (N, 2) — dx, dy preferred direction

    # Build per-cell grids
    dir_grid_x = np.full(walkable_mask.shape, np.nan, dtype=np.float32)
    dir_grid_y = np.full(walkable_mask.shape, np.nan, dtype=np.float32)
    value_grid = np.full(walkable_mask.shape, np.nan, dtype=np.float32)
    for idx, (j, i) in enumerate(walk_cells):
        dir_grid_x[j, i] = mean_xy_np[idx, 0]
        dir_grid_y[j, i] = mean_xy_np[idx, 1]
        value_grid[j, i] = values[idx]

    # Direction policy uses camera-relative frame. mean_xy[0] is "right" and
    # mean_xy[1] is "forward" in CAMERA space. For visualization on the world
    # map, we need to UN-rotate: assume camera yaw = 0 for the probe (we built
    # the obs that way), so camera frame == world frame.
    # Per game/observer.cs convention: forward = (sin(yaw), cos(yaw)),
    # right = (cos(yaw), -sin(yaw)). At yaw=0: forward = (0, 1), right = (1, 0).
    # So mean_xy[0] = right (world +X), mean_xy[1] = forward (world +Y). ✓

    print(f'\nValue stats: min={value_grid[walkable_mask].min():.2f}  '
          f'max={value_grid[walkable_mask].max():.2f}  '
          f'mean={value_grid[walkable_mask].mean():.2f}')
    print(f'Direction magnitude stats: mean={np.linalg.norm(mean_xy_np, axis=1).mean():.3f}')

    # ---------- Render ----------
    fig, axes = plt.subplots(1, 2, figsize=(22, 11))

    grid_step = GRID_RES
    extent = (xs[0] - grid_step/2, xs[-1] + grid_step/2,
              ys[0] - grid_step/2, ys[-1] + grid_step/2)

    for ax in axes:
        ax.set_facecolor('#1a1a1a')
        ax.set_aspect('equal')

    # ===== LEFT: Direction vector field =====
    ax = axes[0]

    # Floor shape as light fill so voids are obvious
    floor_polys = [MplPolygon(poly, closed=True) for poly, _ in floors]
    floor_collection = PatchCollection(floor_polys, facecolors='#2a2a2a',
                                       edgecolors='#888888', linewidths=0.4,
                                       alpha=0.9, zorder=1)
    ax.add_collection(floor_collection)

    # Quiver of preferred directions. Show every other cell for clarity.
    skip = 2
    qx, qy, qu, qv = [], [], [], []
    for j in range(0, len(ys), skip):
        for i in range(0, len(xs), skip):
            if not walkable_mask[j, i]:
                continue
            qx.append(xs[i]); qy.append(ys[j])
            # Normalize direction so arrow length is consistent
            dx = dir_grid_x[j, i]; dy = dir_grid_y[j, i]
            mag = math.sqrt(dx*dx + dy*dy)
            if mag > 1e-6:
                qu.append(dx / mag); qv.append(dy / mag)
            else:
                qu.append(0); qv.append(0)
    ax.quiver(qx, qy, qu, qv, color='#ffaa44', alpha=0.85,
              scale=40, width=0.003, zorder=3, headwidth=4, headlength=5)

    # Gems and spawns
    if items['gems_red']:
        xy = np.array([(p[0], p[1]) for p in items['gems_red']])
        ax.scatter(xy[:, 0], xy[:, 1], c='#ff4444', s=100, marker='D',
                   edgecolors='white', linewidths=1.0, label=f'Red gem [{len(xy)}]', zorder=20)
    if items['gems_yellow']:
        xy = np.array([(p[0], p[1]) for p in items['gems_yellow']])
        ax.scatter(xy[:, 0], xy[:, 1], c='#ffdd44', s=130, marker='D',
                   edgecolors='white', linewidths=1.0, label=f'Yellow gem [{len(xy)}]', zorder=20)
    if items['spawns']:
        xy = np.array([(p[0], p[1]) for p in items['spawns']])
        ax.scatter(xy[:, 0], xy[:, 1], c='#44ddff', s=50, marker='s',
                   edgecolors='white', linewidths=0.6, alpha=0.7,
                   label=f'Spawn [{len(xy)}]', zorder=19)

    ax.set_title(f'Actor mean direction (where the policy WANTS to go) — {label}\n'
                 f'Arrows pointing INTO black voids = model oblivious to holes',
                 color='#dddddd', fontsize=11)
    ax.set_xlabel('X (world)', color='#cccccc')
    ax.set_ylabel('Y (world)', color='#cccccc')
    ax.tick_params(colors='#cccccc')
    for spine in ax.spines.values(): spine.set_edgecolor('#666666')
    ax.legend(loc='upper right', framealpha=0.85, facecolor='#222222',
              edgecolor='#666666', labelcolor='#dddddd', fontsize=9)

    # ===== RIGHT: Critic V(s) heatmap =====
    ax = axes[1]

    masked_v = np.ma.array(value_grid, mask=~walkable_mask)
    v_min = value_grid[walkable_mask].min()
    v_max = value_grid[walkable_mask].max()
    cmap = matplotlib.colormaps['plasma']
    im = ax.imshow(masked_v, cmap=cmap, origin='lower', extent=extent,
                   vmin=v_min, vmax=v_max, interpolation='nearest', zorder=1)

    # Map outline on top
    floor_polys2 = [MplPolygon(poly, closed=True) for poly, _ in floors]
    fc2 = PatchCollection(floor_polys2, facecolors='none',
                          edgecolors='#aaaaaa', linewidths=0.4, alpha=0.6, zorder=2)
    ax.add_collection(fc2)

    # Same items overlay
    if items['gems_red']:
        xy = np.array([(p[0], p[1]) for p in items['gems_red']])
        ax.scatter(xy[:, 0], xy[:, 1], c='#ff4444', s=100, marker='D',
                   edgecolors='white', linewidths=1.0, zorder=20)
    if items['gems_yellow']:
        xy = np.array([(p[0], p[1]) for p in items['gems_yellow']])
        ax.scatter(xy[:, 0], xy[:, 1], c='#ffdd44', s=130, marker='D',
                   edgecolors='white', linewidths=1.0, zorder=20)

    ax.set_title(f'Critic V(s) — what the model "thinks" each cell is worth\n'
                 f'Dark zones near voids = critic learned the danger',
                 color='#dddddd', fontsize=11)
    ax.set_xlabel('X (world)', color='#cccccc')
    ax.set_ylabel('Y (world)', color='#cccccc')
    ax.tick_params(colors='#cccccc')
    for spine in ax.spines.values(): spine.set_edgecolor('#666666')

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('V(s)', color='#cccccc')
    cb.ax.yaxis.set_tick_params(color='#cccccc')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#cccccc')

    fig.patch.set_facecolor('#1a1a1a')
    out_path = os.path.join(here, 'policy_diagnostic.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor='#1a1a1a')
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
