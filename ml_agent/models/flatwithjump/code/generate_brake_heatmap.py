"""Generate a brake-probability heatmap for the latest checkpoint.

Methodology: at every cell on the map, marble is moving FULL SPEED EAST
(+X direction). The "current gem" in the obs is the gem nearest to that
cell. We then ask the model: "Would you brake here?"

If the model has learned the right rule, the heatmap should light up
WEST of each gem (marble about to overshoot it heading east) and stay
dark EAST of each gem (marble already past, no need to brake).

If brake is direction-blind, the heatmap will look uniform.
"""
import torch
import numpy as np
import math
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ppo import Actor

# Map geometry (FlatWithJump_Hunt.mcs) — 4 symmetric platforms + 4 cardinal ground gems
PLATFORMS = [(10, 10), (-10, 10), (-10, -10), (10, -10)]   # corner platforms (gems on top at z=1.6)
GROUND_GEMS = [(17, 0), (-17, 0), (0, 17), (0, -17)]       # cardinal ground gems (z=0.6)
ALL_GEMS = [(x, y, 1.6) for x, y in PLATFORMS] + [(x, y, 0.6) for x, y in GROUND_GEMS]

# Velocity: full speed east. Marble's typical max speed on flat ground is ~12.
FULL_SPEED = 12.0
VEL_X = FULL_SPEED
VEL_Y = 0.0

# Grid resolution.
N_POINTS = 100
GRID_MIN = -25.0
GRID_MAX = 25.0
CELL_PX = max(4, int(round(600 / N_POINTS)))


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


def make_obs(marble_xy, marble_vel, gem_xyz):
    obs = np.zeros(59, dtype=np.float32)
    mx, my = marble_xy
    mz = 0.3
    vx, vy = marble_vel
    obs[0] = mx; obs[1] = my; obs[2] = mz
    obs[3] = vx; obs[4] = vy; obs[5] = 0
    gx, gy, gz = gem_xyz
    obs[6] = gx - mx; obs[7] = gy - my; obs[8] = gz - mz
    obs[9] = 1
    obs[10] = math.sqrt((gx-mx)**2 + (gy-my)**2 + (gz-mz)**2)
    for i in range(1, 5):
        obs[6 + i*5 + 4] = -999  # sentinel for absent gems
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


def color_for_prob(p, scale):
    """Greyscale, normalized so p=scale is white."""
    v = int(round(min(1.0, p / scale) * 255)) if scale > 0 else 0
    v = max(0, min(255, v))
    return f"rgb({v},{v},{v})"


def nearest_gem(x, y):
    best = None
    best_d = float('inf')
    for gem in ALL_GEMS:
        gx, gy, gz = gem
        d = (gx - x) ** 2 + (gy - y) ** 2
        if d < best_d:
            best_d = d
            best = gem
    return best


def main():
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'checkpoints')
    pattern = os.path.join(ckpt_dir, 'update_*.pth')
    ckpts = sorted(glob.glob(pattern), key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0]))
    if not ckpts:
        print(f'No checkpoints found in {ckpt_dir}')
        sys.exit(1)
    latest = ckpts[-1]
    label = os.path.basename(latest).replace('.pth', '')
    print(f'Probing checkpoint: {latest}')
    print(f'Velocity at every cell: ({VEL_X:+.1f}, {VEL_Y:+.1f}) — full speed east')

    model = Actor(obs_dim=59)
    cp = torch.load(latest, weights_only=False, map_location='cpu')
    model.load_state_dict(cp['actor_state_dict'])
    model.eval()

    xs = np.linspace(GRID_MIN, GRID_MAX, N_POINTS)
    ys = np.linspace(GRID_MIN, GRID_MAX, N_POINTS)
    grid_step = (GRID_MAX - GRID_MIN) / (N_POINTS - 1) if N_POINTS > 1 else 1.0
    n_cells = len(xs) * len(ys)
    print(f'Grid: {len(xs)} x {len(ys)} = {n_cells} cells, 1 probe each = {n_cells:,} probes')

    all_obs = np.zeros((n_cells, 59), dtype=np.float32)
    idx = 0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            gem = nearest_gem(float(x), float(y))
            all_obs[idx] = make_obs((float(x), float(y)), (VEL_X, VEL_Y), gem)
            idx += 1

    print(f'  Running batched forward pass over {n_cells:,} observations...')
    t = torch.from_numpy(all_obs)
    with torch.no_grad():
        _, _, _, brake_logits = model(t)
        brake_probs = torch.sigmoid(brake_logits).cpu().numpy().squeeze(-1)

    grid = brake_probs.reshape(len(ys), len(xs))

    print(f'\nBrake probability stats (velocity fixed at +X full speed):')
    print(f'  min={grid.min()*100:.2f}%  max={grid.max()*100:.2f}%  mean={grid.mean()*100:.2f}%  std={grid.std()*100:.2f}pp')

    # West-of-gem vs east-of-gem analysis: marble heading east, so cells
    # WEST of a gem are heading TOWARD it (should brake), cells EAST of a
    # gem are heading AWAY from it (should NOT brake).
    west_mask = np.zeros_like(grid, dtype=bool)
    east_mask = np.zeros_like(grid, dtype=bool)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            for gx, gy, _ in ALL_GEMS:
                dx_to_gem = gx - float(x)
                dy_to_gem = gy - float(y)
                d = math.sqrt(dx_to_gem * dx_to_gem + dy_to_gem * dy_to_gem)
                if d < 1.0 or d > 8.0:
                    continue
                # Within 8u of gem AND in the +/-22.5° lateral cone of approach
                if abs(dy_to_gem) > 3.0:
                    continue
                if dx_to_gem > 0:    # gem is to the east of marble -> marble heading toward gem
                    west_mask[j, i] = True
                elif dx_to_gem < 0:  # gem is to the west of marble -> marble heading away
                    east_mask[j, i] = True
    if west_mask.any():
        west_avg = grid[west_mask].mean()
        print(f'  Cells WEST of a gem (heading toward, within 8u, narrow cone): {west_avg*100:.2f}% (n={west_mask.sum()})')
    if east_mask.any():
        east_avg = grid[east_mask].mean()
        print(f'  Cells EAST of a gem (heading away, within 8u, narrow cone): {east_avg*100:.2f}% (n={east_mask.sum()})')
    if west_mask.any() and east_mask.any():
        print(f'  West - East discrimination: {(west_avg - east_avg)*100:+.2f}pp '
              f'(positive = model has direction-aware brake)')

    cell_px = CELL_PX
    grid_w = len(xs) * cell_px
    grid_h = len(ys) * cell_px

    def world_to_px(x, y):
        col = (x - GRID_MIN) / grid_step
        row = (GRID_MAX - y) / grid_step
        cx = col * cell_px + cell_px / 2.0
        cy = row * cell_px + cell_px / 2.0
        return cx, cy

    # Color scale: stretch to grid max so any spatial structure is visible.
    color_scale = max(grid.max(), 0.01)

    html_parts = []
    html_parts.append('<!DOCTYPE html><html><head><meta charset="utf-8"><style>')
    html_parts.append('body{font-family:Consolas,monospace;background:#0d1117;color:#e6edf3;margin:0;padding:20px}')
    html_parts.append('h2{margin:0 0 8px}')
    html_parts.append('p{margin:4px 0;color:#7d8590;font-size:0.85rem}')
    html_parts.append('.heatmap-wrap{position:relative;display:inline-block;margin:12px auto;border:1px solid #30363d}')
    html_parts.append('table{border-collapse:collapse}')
    html_parts.append(f'td{{width:{cell_px}px;height:{cell_px}px;font-size:0;padding:0;border:0}}')
    html_parts.append('.overlay{position:absolute;top:0;left:0;pointer-events:none}')
    html_parts.append('.legend{display:flex;justify-content:center;align-items:center;gap:6px;margin:12px 0;flex-wrap:wrap}')
    html_parts.append('.legend-bar{display:inline-block;width:300px;height:18px;'
                      'background:linear-gradient(to right,'
                      ' rgb(0,0,0) 0%, rgb(64,64,64) 25%, rgb(128,128,128) 50%, rgb(192,192,192) 75%, rgb(255,255,255) 100%);'
                      'border:1px solid #30363d;vertical-align:middle}')
    html_parts.append('.legend-label{font-size:0.75rem;color:#7d8590}')
    html_parts.append('.legend-marker{display:inline-flex;align-items:center;gap:6px;margin-left:18px;font-size:0.8rem}')
    html_parts.append('.stats{text-align:center;font-size:0.85rem;color:#e6edf3;margin:8px 0}')
    html_parts.append('.stats span{display:inline-block;margin:0 12px}')
    html_parts.append('.center{text-align:center}')
    html_parts.append('</style></head><body>')

    html_parts.append(f'<h2 class="center">Brake Probability Heatmap &mdash; {label}</h2>')
    html_parts.append(f'<p class="center">Velocity at every cell = ({VEL_X:+.0f}, {VEL_Y:+.0f}) (full speed EAST). '
                      f'Current gem = the gem nearest to each cell. Color scale: 0% (black) &rarr; '
                      f'{color_scale*100:.2f}% (white).</p>')
    html_parts.append('<div class="legend">')
    html_parts.append('<span class="legend-label">0%</span>')
    html_parts.append('<span class="legend-bar"></span>')
    html_parts.append(f'<span class="legend-label">{color_scale*100:.2f}%</span>')
    html_parts.append('<span class="legend-marker">'
                      '<svg width="14" height="14"><rect x="1" y="1" width="12" height="12" stroke="#58a6ff" stroke-width="2" fill="none"/></svg>'
                      ' platform</span>')
    html_parts.append('<span class="legend-marker">'
                      '<svg width="14" height="14"><polygon points="7,2 12,7 7,12 2,7" fill="#f0883e"/></svg>'
                      ' elevated gem</span>')
    html_parts.append('<span class="legend-marker">'
                      '<svg width="14" height="14"><polygon points="7,2 12,7 7,12 2,7" fill="#f0c040"/></svg>'
                      ' ground gem</span>')
    html_parts.append('<span class="legend-marker">'
                      '<svg width="32" height="14"><line x1="2" y1="7" x2="28" y2="7" stroke="#ff5555" stroke-width="2"/>'
                      '<polygon points="22,3 30,7 22,11" fill="#ff5555"/></svg>'
                      ' marble velocity (east)</span>')
    html_parts.append('</div>')

    html_parts.append(f'<div class="stats">'
                      f'<span><b>Min:</b> {grid.min()*100:.2f}%</span>'
                      f'<span><b>Max:</b> {grid.max()*100:.2f}%</span>'
                      f'<span><b>Mean:</b> {grid.mean()*100:.2f}%</span>'
                      f'<span><b>Std:</b> {grid.std()*100:.2f}pp</span>')
    if west_mask.any() and east_mask.any():
        html_parts.append(f'<span><b>West-of-gem:</b> {west_avg*100:.2f}%</span>'
                          f'<span><b>East-of-gem:</b> {east_avg*100:.2f}%</span>'
                          f'<span><b>W&minus;E gap:</b> {(west_avg-east_avg)*100:+.2f}pp</span>')
    html_parts.append('</div>')

    html_parts.append('<div class="center">')
    html_parts.append('<div class="heatmap-wrap">')

    html_parts.append('<table>')
    for j in range(len(ys) - 1, -1, -1):
        y = ys[j]
        html_parts.append('<tr>')
        for i, x in enumerate(xs):
            p = grid[j, i]
            color = color_for_prob(p, color_scale)
            html_parts.append(f'<td style="background:{color}" title="({x:+.0f},{y:+.0f}) {p*100:.2f}%"></td>')
        html_parts.append('</tr>')
    html_parts.append('</table>')

    # Overlay: platforms, gem icons, plus tiny eastward arrows in a sparse grid
    # so the velocity direction is visible at a glance.
    html_parts.append(f'<svg class="overlay" width="{grid_w}" height="{grid_h}" viewBox="0 0 {grid_w} {grid_h}">')

    plat_size_world = 2.0
    plat_size_px = plat_size_world / grid_step * cell_px
    for px, py in PLATFORMS:
        cx, cy = world_to_px(px, py)
        x0 = cx - plat_size_px / 2.0
        y0 = cy - plat_size_px / 2.0
        html_parts.append(
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{plat_size_px:.1f}" height="{plat_size_px:.1f}" '
            f'fill="none" stroke="#58a6ff" stroke-width="2"/>'
        )

    gem_radius_px = 5.0
    for px, py in PLATFORMS:
        cx, cy = world_to_px(px, py)
        html_parts.append(
            f'<polygon points="{cx:.1f},{cy-gem_radius_px:.1f} '
            f'{cx+gem_radius_px:.1f},{cy:.1f} '
            f'{cx:.1f},{cy+gem_radius_px:.1f} '
            f'{cx-gem_radius_px:.1f},{cy:.1f}" '
            f'fill="#f0883e" stroke="#0d1117" stroke-width="0.7"/>'
        )
    for gx, gy in GROUND_GEMS:
        cx, cy = world_to_px(gx, gy)
        html_parts.append(
            f'<polygon points="{cx:.1f},{cy-gem_radius_px:.1f} '
            f'{cx+gem_radius_px:.1f},{cy:.1f} '
            f'{cx:.1f},{cy+gem_radius_px:.1f} '
            f'{cx-gem_radius_px:.1f},{cy:.1f}" '
            f'fill="#f0c040" stroke="#0d1117" stroke-width="0.7"/>'
        )

    # Sparse east-arrow grid showing the velocity direction.
    arrow_step = 5  # world units between arrow tails
    arrow_len_px = grid_step_arrow_len = 3.0 / grid_step * cell_px
    for ax in range(int(GRID_MIN) + arrow_step, int(GRID_MAX), arrow_step):
        for ay in range(int(GRID_MIN) + arrow_step, int(GRID_MAX), arrow_step):
            cx, cy = world_to_px(ax, ay)
            x_end = cx + arrow_len_px
            html_parts.append(
                f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x_end:.1f}" y2="{cy:.1f}" '
                f'stroke="#ff5555" stroke-width="1" stroke-opacity="0.55"/>'
                f'<polygon points="{x_end:.1f},{cy-2.5:.1f} {x_end+4:.1f},{cy:.1f} {x_end:.1f},{cy+2.5:.1f}" '
                f'fill="#ff5555" fill-opacity="0.55"/>'
            )

    html_parts.append('</svg>')
    html_parts.append('</div>')
    html_parts.append('</div>')

    html_parts.append('<p class="center">Marble is moving east everywhere (red arrows). '
                      'If brake is direction-aware, you should see bright spots WEST of each gem '
                      '(marble about to overshoot) and dark zones EAST of each gem (marble already past).</p>')
    html_parts.append('</body></html>')

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brake_heatmap.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
