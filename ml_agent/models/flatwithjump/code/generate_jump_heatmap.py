"""Generate a jump-probability heatmap for the latest checkpoint.

For each XY cell on the map, samples the model's jump probability across
multiple possible "current gem" scenarios and averages, then renders as
a continuous greyscale grid in jump_heatmap.html. Platform locations
marked with a cyan border.
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

# Grid resolution. N_POINTS = number of evenly-spaced sample points per axis.
# Total probes = N_POINTS * N_POINTS * len(ALL_GEMS).
# 51 → 51x51 grid (~21k probes, fast); 100 → 100x100 (~80k probes, ~2-3s on CPU).
N_POINTS = 100
GRID_MIN = -25.0
GRID_MAX = 25.0
# Cell size in pixels — scaled so 51x51 -> 612px and 100x100 -> 600px (roughly constant canvas).
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
    """Build a single observation. Marble at marble_xy with z=0.3, velocity given."""
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
    # IMPORTANT: frame history (obs[35:59]) must be in NORMALIZED scale.
    # normalize_obs() only operates on obs[0:35]. During real training, frame
    # history is built from already-normalized values, so it's pre-scaled.
    # Pre-divide here to match (pos/100, vel/20).
    for i in range(4):
        obs[35 + i*6 + 0] = mx / 100.0
        obs[35 + i*6 + 1] = my / 100.0
        obs[35 + i*6 + 2] = mz / 100.0
        obs[35 + i*6 + 3] = vx / 20.0
        obs[35 + i*6 + 4] = vy / 20.0
    return normalize_obs(obs)


def jump_prob(model, obs):
    t = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        # Actor.forward returns 4 values (added brake_logit). Discard brake here.
        _, _, jump_logit, _ = model(t)
        return torch.sigmoid(jump_logit).item()


def avg_jump_prob_at(model, mx, my):
    """For marble at (mx, my), average jump probability across each possible
    'current gem' choice. Velocity is set to a small magnitude pointed AT each
    gem (since the model is more likely to jump when approaching, this gives
    a representative probe of 'what would the model do here, on average')."""
    probs = []
    for gem in ALL_GEMS:
        gx, gy, gz = gem
        # Velocity pointed roughly at the gem at moderate speed
        dx, dy = gx - mx, gy - my
        d = math.sqrt(dx*dx + dy*dy)
        if d > 0.01:
            vx = (dx / d) * 6.0
            vy = (dy / d) * 6.0
        else:
            vx, vy = 0, 0
        obs = make_obs((mx, my), (vx, vy), gem)
        probs.append(jump_prob(model, obs))
    return float(np.mean(probs))


def color_for_prob(p):
    """Continuous greyscale: 0% = pure black, 100% = pure white."""
    v = int(round(p * 255))
    v = max(0, min(255, v))
    return f"rgb({v},{v},{v})"


def is_platform_cell(x, y):
    """Marble at this XY would be near a platform."""
    for px, py in PLATFORMS:
        if abs(x - px) <= 1.0 and abs(y - py) <= 1.0:
            return True
    return False


def is_groundgem_cell(x, y):
    for gx, gy in GROUND_GEMS:
        if abs(x - gx) <= 1.0 and abs(y - gy) <= 1.0:
            return True
    return False


def main():
    # Find latest checkpoint
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'checkpoints')
    pattern = os.path.join(ckpt_dir, 'update_*.pth')
    ckpts = sorted(glob.glob(pattern), key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0]))
    if not ckpts:
        print(f'No checkpoints found in {ckpt_dir}')
        sys.exit(1)
    latest = ckpts[-1]
    label = os.path.basename(latest).replace('.pth', '')
    print(f'Probing checkpoint: {latest}')

    model = Actor(obs_dim=59)
    cp = torch.load(latest, weights_only=False, map_location='cpu')
    model.load_state_dict(cp['actor_state_dict'])
    model.eval()

    # Build grid using N_POINTS evenly spaced samples per axis
    xs = np.linspace(GRID_MIN, GRID_MAX, N_POINTS)
    ys = np.linspace(GRID_MIN, GRID_MAX, N_POINTS)
    grid_step = (GRID_MAX - GRID_MIN) / (N_POINTS - 1) if N_POINTS > 1 else 1.0
    n_cells = len(xs) * len(ys)
    n_gems = len(ALL_GEMS)
    print(f'Grid: {len(xs)} x {len(ys)} = {n_cells} cells, {n_gems} gem scenarios each = {n_cells*n_gems:,} probes')

    # Build all observations as one big array, then run a single batched
    # forward pass. ~80k probes in ~2s instead of cell-by-cell forever.
    all_obs = np.zeros((n_cells * n_gems, 59), dtype=np.float32)
    idx = 0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            for gem in ALL_GEMS:
                gx, gy, _gz = gem
                dx = gx - x
                dy = gy - y
                d = math.sqrt(dx*dx + dy*dy)
                if d > 0.01:
                    vx = (dx / d) * 6.0
                    vy = (dy / d) * 6.0
                else:
                    vx, vy = 0.0, 0.0
                all_obs[idx] = make_obs((float(x), float(y)), (vx, vy), gem)
                idx += 1

    print(f'  Running batched forward pass over {n_cells*n_gems:,} observations...')
    t = torch.from_numpy(all_obs)
    with torch.no_grad():
        _, _, jump_logits, _ = model(t)
        jump_probs = torch.sigmoid(jump_logits).cpu().numpy().squeeze(-1)

    # Average across gem scenarios per cell, reshape to grid
    jump_probs = jump_probs.reshape(n_cells, n_gems).mean(axis=1)
    grid = jump_probs.reshape(len(ys), len(xs))

    # Statistics
    print(f'\nJump probability stats across grid:')
    print(f'  min={grid.min()*100:.2f}%  max={grid.max()*100:.2f}%  mean={grid.mean()*100:.2f}%  std={grid.std()*100:.2f}pp')
    plat_mask = np.zeros_like(grid, dtype=bool)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if is_platform_cell(float(x), float(y)):
                plat_mask[j, i] = True
    plat_avg = grid[plat_mask].mean() if plat_mask.any() else 0
    nonplat_avg = grid[~plat_mask].mean()
    print(f'  Avg over platform cells: {plat_avg*100:.2f}%')
    print(f'  Avg over non-platform cells: {nonplat_avg*100:.2f}%')
    print(f'  Discrimination (platform - non-platform): {(plat_avg - nonplat_avg)*100:+.2f}pp')

    # Render HTML
    cell_px = CELL_PX
    grid_w = len(xs) * cell_px
    grid_h = len(ys) * cell_px

    # Pixel coordinate from world coords (y is flipped: +Y at top).
    def world_to_px(x, y):
        col = (x - GRID_MIN) / grid_step                # 0 = leftmost cell
        row = (GRID_MAX - y) / grid_step                # 0 = topmost cell (y=+25)
        cx = col * cell_px + cell_px / 2.0
        cy = row * cell_px + cell_px / 2.0
        return cx, cy

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

    html_parts.append(f'<h2 class="center">Jump Probability Heatmap — {label}</h2>')
    html_parts.append('<p class="center">Each cell = avg jump probability if marble is at that XY across all 8 gem scenarios. '
                      'Velocity pointed toward each gem.</p>')
    html_parts.append('<div class="legend">')
    html_parts.append('<span class="legend-label">0%</span>')
    html_parts.append('<span class="legend-bar"></span>')
    html_parts.append('<span class="legend-label">100%</span>')
    # Inline SVG legend markers
    html_parts.append('<span class="legend-marker">'
                      '<svg width="14" height="14"><rect x="1" y="1" width="12" height="12" stroke="#58a6ff" stroke-width="2" fill="none"/></svg>'
                      ' platform</span>')
    html_parts.append('<span class="legend-marker">'
                      '<svg width="14" height="14"><polygon points="7,2 12,7 7,12 2,7" fill="#f0883e"/></svg>'
                      ' elevated gem</span>')
    html_parts.append('<span class="legend-marker">'
                      '<svg width="14" height="14"><polygon points="7,2 12,7 7,12 2,7" fill="#f0c040"/></svg>'
                      ' ground gem</span>')
    html_parts.append('</div>')

    html_parts.append(f'<div class="stats">'
                      f'<span><b>Min:</b> {grid.min()*100:.1f}%</span>'
                      f'<span><b>Max:</b> {grid.max()*100:.1f}%</span>'
                      f'<span><b>Mean:</b> {grid.mean()*100:.1f}%</span>'
                      f'<span><b>Platform avg:</b> {plat_avg*100:.1f}%</span>'
                      f'<span><b>Non-platform avg:</b> {nonplat_avg*100:.1f}%</span>'
                      f'<span><b>Gap:</b> {(plat_avg-nonplat_avg)*100:+.2f}pp</span>'
                      f'</div>')

    # Wrap table + SVG overlay so platform boxes / gem icons sit on top of the heatmap.
    html_parts.append('<div class="center">')
    html_parts.append('<div class="heatmap-wrap">')

    # Heatmap grid table (no per-cell outlines anymore — that's done by the overlay).
    html_parts.append('<table>')
    for j in range(len(ys) - 1, -1, -1):  # reverse so y=+25 is at top
        y = ys[j]
        html_parts.append('<tr>')
        for i, x in enumerate(xs):
            p = grid[j, i]
            color = color_for_prob(p)
            html_parts.append(f'<td style="background:{color}" title="({x:+.0f},{y:+.0f}) {p*100:.1f}%"></td>')
        html_parts.append('</tr>')
    html_parts.append('</table>')

    # SVG overlay: platform boundary boxes + gem icons.
    html_parts.append(f'<svg class="overlay" width="{grid_w}" height="{grid_h}" viewBox="0 0 {grid_w} {grid_h}">')

    # Platform boundary boxes (~2x2 world units, scaled to pixels).
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

    # Gem icons (diamond shape, ~10x10 px).
    gem_radius_px = 5.0
    # Elevated gems (on top of platforms) — orange diamond
    for px, py in PLATFORMS:
        cx, cy = world_to_px(px, py)
        html_parts.append(
            f'<polygon points="{cx:.1f},{cy-gem_radius_px:.1f} '
            f'{cx+gem_radius_px:.1f},{cy:.1f} '
            f'{cx:.1f},{cy+gem_radius_px:.1f} '
            f'{cx-gem_radius_px:.1f},{cy:.1f}" '
            f'fill="#f0883e" stroke="#0d1117" stroke-width="0.7"/>'
        )
    # Ground gems — gold diamond
    for gx, gy in GROUND_GEMS:
        cx, cy = world_to_px(gx, gy)
        html_parts.append(
            f'<polygon points="{cx:.1f},{cy-gem_radius_px:.1f} '
            f'{cx+gem_radius_px:.1f},{cy:.1f} '
            f'{cx:.1f},{cy+gem_radius_px:.1f} '
            f'{cx-gem_radius_px:.1f},{cy:.1f}" '
            f'fill="#f0c040" stroke="#0d1117" stroke-width="0.7"/>'
        )

    html_parts.append('</svg>')
    html_parts.append('</div>')   # heatmap-wrap
    html_parts.append('</div>')   # center

    html_parts.append('<p class="center">Hover any cell to see exact (x,y) and jump probability.</p>')
    html_parts.append('</body></html>')

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jump_heatmap.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
