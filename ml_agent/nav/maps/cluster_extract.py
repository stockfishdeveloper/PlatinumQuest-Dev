"""Find tight multi-height gem clusters on Hunt maps and pull out the raised 'box' geometry under them.

Output: a JSON of clusters (gems + fitted boxes) and per-cluster check images.
"""
import os, re, sys, json, math
import numpy as np
ML = r'C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent'
sys.path.insert(0, ML)
import hxDif
from generate_map_topdown import extract_surfaces
from generate_terrain_map import resolve_mission, parse_interiors, transform_surfaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Rectangle
from matplotlib.path import Path as MplPath

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clusters')
os.makedirs(OUT, exist_ok=True)

LINK_XY = 4.0      # gems closer than this (horizontal) are one cluster
LINK_DZ = 3.5
MIN_GEMS = 4
MAX_SPAN = 14.0    # footprint diagonal cap: bigger = not 'tight'
MARGIN = 1.0       # around the gem footprint when picking geometry


def load_gems(mis):
    t = open(mis, errors='ignore').read()
    gems = []
    for m in re.finditer(r'new Item\(\w*\)\s*\{([^}]+)\}', t):
        b = m.group(1)
        db = re.search(r'dataBlock\s*=\s*"([^"]+)"', b); p = re.search(r'position\s*=\s*"([^"]+)"', b)
        if db and p and db.group(1).lower().startswith('gemitem'):
            x, y, z = [float(v) for v in p.group(1).split()[:3]]
            gems.append((x, y, z, db.group(1)))
    return gems


def load_surfaces(mis):
    surfs = []
    for dif_path, pos, rot, scl in parse_interiors(mis):
        d = hxDif.Dif.Load(dif_path)
        s = extract_surfaces(d)
        surfs += transform_surfaces(s, pos, rot, scl)
    return surfs


def cluster_gems(gems):
    n = len(gems); parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            a, b = gems[i], gems[j]
            if math.hypot(a[0] - b[0], a[1] - b[1]) < LINK_XY and abs(a[2] - b[2]) < LINK_DZ:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(gems[i])
    out = []
    for g in groups.values():
        if len(g) < MIN_GEMS: continue
        xs = [p[0] for p in g]; ys = [p[1] for p in g]; zs = [p[2] for p in g]
        span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        levels = sorted(set(round(z / 0.4) for z in zs))
        if span > MAX_SPAN or len(levels) < 2: continue
        out.append(g)
    return out


def poly_area(v):
    x = np.array([p[0] for p in v]); y = np.array([p[1] for p in v])
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def analyse_cluster(g, surfs):
    xs = [p[0] for p in g]; ys = [p[1] for p in g]; zs = [p[2] for p in g]
    x0, x1, y0, y1 = min(xs) - MARGIN, max(xs) + MARGIN, min(ys) - MARGIN, max(ys) + MARGIN
    zlo, zhi = min(zs) - 3.0, max(zs) + 1.0
    tops = []      # upward polygons inside the footprint
    for verts, n in surfs:
        if n[2] < 0.85: continue
        vz = [p[2] for p in verts]
        if max(vz) < zlo or min(vz) > zhi: continue
        vx = [p[0] for p in verts]; vy = [p[1] for p in verts]
        if max(vx) < x0 or min(vx) > x1 or max(vy) < y0 or min(vy) > y1: continue
        tops.append((verts, float(np.mean(vz)), poly_area(verts)))
    if not tops: return None
    # the floor a gem sits on: the highest upward polygon under the gem, within 1.2 u below it
    def floor_under(x, y, z):
        best = None
        for verts, zc, a in tops:
            if zc > z + 0.1 or zc < z - 1.5: continue
            if MplPath([(p[0], p[1]) for p in verts]).contains_point((x, y), radius=0.05):
                if best is None or zc > best: best = zc
        return best
    gem_floor = [floor_under(x, y, z) for x, y, z, _ in g]
    base = min(zc for v, zc, a in tops)   # lowest floor in the footprint = the ground the boxes stand on
    # boxes = upward polygons above the base; fit each to its bbox
    boxes = []
    for verts, zc, a in tops:
        if zc - base < 0.3: continue
        vx = [p[0] for p in verts]; vy = [p[1] for p in verts]
        bx0, bx1, by0, by1 = min(vx), max(vx), min(vy), max(vy)
        barea = (bx1 - bx0) * (by1 - by0)
        if barea < 0.2: continue
        boxes.append({'x0': bx0, 'x1': bx1, 'y0': by0, 'y1': by1, 'top': zc, 'rect_fill': a / barea if barea else 0,
                      'area': a})
    return {'base': base, 'tops': tops, 'boxes': boxes, 'gem_floor': gem_floor,
            'bbox': (x0, x1, y0, y1)}


def draw(name, ci, g, info):
    x0, x1, y0, y1 = info['bbox']
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    zmin = info['base']; zmax = max(zc for v, zc, a in info['tops'])
    cm = plt.get_cmap('viridis')
    for verts, zc, a in sorted(info['tops'], key=lambda t: t[1]):
        c = cm((zc - zmin) / max(zmax - zmin, 0.1))
        ax[0].add_patch(MplPoly([(p[0], p[1]) for p in verts], closed=True, fc=c, ec='k', lw=0.3))
    for verts, zc, a in sorted(info['tops'], key=lambda t: t[1]):
        if zc - info['base'] < 0.3: continue
    for b in info['boxes']:
        c = cm((b['top'] - zmin) / max(zmax - zmin, 0.1))
        ax[1].add_patch(Rectangle((b['x0'], b['y0']), b['x1'] - b['x0'], b['y1'] - b['y0'], fc=c, ec='k', lw=0.5))
    for a in ax:
        for x, y, z, db in g:
            a.plot(x, y, 'o', mfc='red' if 'Red' in db else ('yellow' if 'Yellow' in db else 'cyan'), mec='k', ms=9)
            a.text(x, y, f'{z:.1f}', fontsize=7, ha='left', va='bottom')
        a.set_xlim(x0 - 2, x1 + 2); a.set_ylim(y0 - 2, y1 + 2); a.set_aspect('equal')
    ax[0].set_title(f'{name} cluster {ci}: real floor polygons (color = height)')
    ax[1].set_title(f'box fit (base z {info["base"]:.2f})')
    fig.savefig(os.path.join(OUT, f'{name}_c{ci:02d}.png'), dpi=90); plt.close(fig)


def main():
    result = {}
    for name in sys.argv[1:]:
        mis = resolve_mission(name)
        gems = load_gems(mis); surfs = load_surfaces(mis)
        cl = cluster_gems(gems)
        print(f'== {name}: {len(gems)} gems, {len(surfs)} surfaces, {len(cl)} tight multi-height clusters')
        rows = []
        for ci, g in enumerate(sorted(cl, key=lambda g: -len(g))):
            info = analyse_cluster(g, surfs)
            if info is None: print(f'  c{ci}: no floor found'); continue
            zs = sorted(set(round(p[2], 1) for p in g))
            nb = len(info['boxes']); rect = np.mean([b['rect_fill'] for b in info['boxes']]) if nb else 0
            heights = sorted(set(round(b['top'] - info['base'], 2) for b in info['boxes']))
            missing = sum(1 for f in info['gem_floor'] if f is None)
            print(f'  c{ci:02d}: {len(g)} gems at {[round(p[0],1) for p in g][:1]}... z-levels {zs}  boxes {nb} rect-fill {rect:.2f} '
                  f'box heights over base {heights[:6]}  gems w/o floor {missing}')
            draw(name, ci, g, info)
            rows.append({'gems': g, 'base': info['base'], 'boxes': info['boxes'], 'bbox': info['bbox'],
                         'gem_floor': info['gem_floor']})
        result[name] = rows
    json.dump(result, open(os.path.join(OUT, 'clusters.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
