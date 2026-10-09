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

# Minimum upward component of a surface normal for it to count as FLOOR.
# 0.5 admitted anything up to 60 deg from horizontal, which kept the 45 deg BEVELS that
# chamfer the edges of holes. Rasterised, a bevel becomes an innocent-looking height 0.5 u
# below the floor, the walk grid calls it walkable, and the marble slides straight into the
# hole -- measured at the centre of KingOfTheMarble, where the true 2x2 void read as a 2x1
# void with two false-floor cells on its lower edge (2026-09-19).
# 0.85 is ~32 deg, matching nav/terrain.py MAX_SLOPE = 0.6 (~31 deg): we no longer STORE as
# floor anything the navigator would refuse to WALK on.
FLOOR_NORMAL_Z = float(os.environ.get('NAV_FLOOR_NORMAL_Z', '0.85'))
EDGE_EPS = 0.02               # 2026-10-05: a grid point counts as floor if it or one of 8 points EDGE_EPS away (axes and diagonals)
                              # lies inside a floor polygon. The grid lined up with the level's tile edges, so many points sat
                              # exactly ON an edge, where the inside test flips with the edge's direction: hole edges came out
                              # floor on some sides and void on others (KOTM: the marble resting on floor at (-24.43, 16.17)
                              # beside the 2 x 2 hole at x -26.2..-24.2, y 14..16 read as no floor and the driver stalled).
DEFAULT_GRID_RES = 0.25       # 2026-10-05: was 0.5; lookups use the nearest grid point, so 0.25 halves the edge error
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
        if not os.path.exists(dif):
            print(f'  WARNING: interior file not found, skipped: {dif}')
            continue
        out.append((dif, pos, rot, scl))
    if not out:
        raise RuntimeError(f'No InteriorInstance with an existing .dif in {mis_path}')
    return out


def _axis_angle_matrix(rot):
    """3x3 rotation matrix for a Torque "x y z angle_deg" axis-angle, in the ENGINE's convention.

    2026-10-07 (nav/maps/teleport_probe.py + box_scan.py on BlockClusters_v0_Hunt): the engine turns an interior
    the opposite way to the textbook axis-angle matrix (Torque builds the transposed matrix). Boxes rotated ~45 deg
    off a right angle came out 2.6 u wide along the predicted axes instead of 2.0, and all three clusters whose
    doubled angle sat 35-50 deg off a right angle disagreed with the game while every cluster near a multiple of
    90 agreed. Never showed before: KOTM's interior has rotation "1 0 0 0". So the matrix is returned transposed."""
    ax = np.array(rot[0:3], dtype=np.float64)
    n = np.linalg.norm(ax)
    ang = math.radians(rot[3])
    if n < 1e-9 or abs(ang) < 1e-9:
        return np.eye(3)
    x, y, z = ax / n
    c, s = math.cos(ang), math.sin(ang)
    C = 1.0 - c
    M = np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                  [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                  [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])
    return M.T          # the engine's (transposed) convention, see the docstring


def transform_surfaces(surfaces, pos, rot, scl):
    """Apply the InteriorInstance transform (scale, then rotate, then translate)
    to surfaces extracted at the origin. Normals are recomputed from the
    transformed vertices (non-uniform scale changes slopes), keeping the
    original facing."""
    R = _axis_angle_matrix(rot)
    S = np.array(scl, dtype=np.float64)
    T = np.array(pos, dtype=np.float64)
    out = []
    for surf in surfaces:
        verts, n = surf[0], surf[1]
        v = (np.array(verts, dtype=np.float64) * S) @ R.T + T
        n_rot = R @ np.array(n, dtype=np.float64)
        if len(v) >= 3:
            # Newell normal of the transformed polygon
            nn = np.zeros(3)
            for i in range(len(v)):
                a, b = v[i], v[(i + 1) % len(v)]
                nn += np.cross(a, b)
            if np.linalg.norm(nn) > 1e-9:
                nn /= np.linalg.norm(nn)
                if np.dot(nn, n_rot) < 0:
                    nn = -nn
                n_rot = nn
        out.append(([tuple(p) for p in v], tuple(n_rot)) + tuple(surf[2:]))
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

# ---------------------------------------------------------------------------------------------------------------
# 2026-10-09 (real-map pool, Phase R): which slopes are FLOOR is physics, per surface, not a fixed 32 degrees.
# marble.cc: traction on a contact is capped at staticFriction (1.1, marble datablock) x the surface's friction x the
# normal force, and the push is capped by angularAcceleration (75 rad/s^2) x radius (0.2 u) = 15 u/s^2. Holding or
# climbing a slope of angle a needs g sin a of tangential force against g cos a of normal force (g = 20), so a surface
# is climbable when tan a <= 1.1 mu and sin a <= 15/20. The game maps textures to friction in platinum/data/init.cs
# (addMaterialMapping; DefaultMaterial 1.0, friction_high 1.5, friction_low 0.2, ice 0.03, tarmac 0.35). Operator
# 10-09: "the marble can climb ramps above 45 degrees on its own power" (default material: 47.7 deg by this rule),
# Bowl's 51 deg friction_high wall is climbable, Cube Isle's 32 deg ramps are floor.
# MEASURED 10-09 15:10 (climb_test.py): the marble pushed up Bowl's 51 deg friction_high wall from rest and reached
# its top (-7 -> -1.9), so the push cap (sin a <= 15/20 = 48.6 deg) is NOT a limit the engine enforces; only the
# friction rule stays: tan a <= 1.1 mu (default material 47.7 deg, friction_high 58.8 deg, friction_low 12.4 deg).
STATIC_FRICTION = 1.1          # marble datablock staticFriction
PUSH_ACCEL = 1e9               # (measured: not a limit) angularAcceleration x radius would be 15 u/s^2
GRAVITY = 20.0
CLIMB_MARGIN = 1.0
DEFAULT_FRICTION = 1.0
_FRICTION_TABLE = None


def friction_table():
    """{texture base name (lower) -> friction} from platinum/data/init.cs (MaterialProperty + addMaterialMapping)."""
    global _FRICTION_TABLE
    if _FRICTION_TABLE is not None:
        return _FRICTION_TABLE
    import re
    path = os.path.join(os.path.dirname(HERE), 'Marble Blast Platinum', 'platinum', 'data', 'init.cs')
    props, table = {}, {}
    try:
        txt = open(path, encoding='latin-1').read()
        for m in re.finditer(r'new MaterialProperty\((\w+)\)\s*\{(.*?)\};', txt, re.S):
            f = re.search(r'friction\s*=\s*([-\d.]+)', m.group(2))
            if f:
                props[m.group(1)] = float(f.group(1))
        for m in re.finditer(r'addMaterialMapping\(\s*"([^"]*)"\s*,\s*(\w+)\s*\)', txt):
            if m.group(2) in props:
                table[m.group(1).lower()] = props[m.group(2)]
    except OSError:
        pass
    _FRICTION_TABLE = table
    return table


def material_friction(name):
    """Friction of a DIF texture name (Shaders/friction_low_shadow -> friction_low_shadow -> 0.2); default 1."""
    base = os.path.basename(str(name)).lower()
    f = friction_table().get(base)
    if f is None or f < 0:           # RandomForceMaterial is -1: treat as default
        return DEFAULT_FRICTION
    return f


def climbable(normal, friction):
    """True if a surface with this upward normal can be rolled on / climbed (see the physics note above)."""
    nz = float(normal[2])
    if nz <= 0.0:
        return False
    sin_a = math.sqrt(max(0.0, 1.0 - nz * nz))
    tan_a = sin_a / nz
    return tan_a <= CLIMB_MARGIN * STATIC_FRICTION * friction and sin_a <= CLIMB_MARGIN * PUSH_ACCEL / GRAVITY


def surf_parts(sf):
    """(verts, normal, friction) for a 2-tuple (default friction) or 3-tuple (verts, normal, material) surface."""
    if len(sf) >= 3:
        return sf[0], sf[1], material_friction(sf[2])
    return sf[0], sf[1], DEFAULT_FRICTION


HEADROOM = 0.75               # 2026-10-09 09:45: a floor level with ANOTHER FLOOR LEVEL less than this above it is dropped too (the
                              # marble is 0.38 u across; a 0.5 u crawlspace is not a floor). Found on ExampleMission (a real map):
                              # the raised tiles have no underside faces, so COVER_CLEARANCE saw no ceiling and the stack kept
                              # 9.5 / 9.0 / 8.5 at one cell: the rays followed the 8.5 floor UNDER the boxes, no edge, no rise,
                              # the policy never jumped there (stall logits above 0 on 0.8 % vs 7 % on the generated maps).
COVER_CLEARANCE = 0.45        # 2026-10-08 00:40: a floor level with a DOWNWARD-facing surface (a ceiling, the underside of a
                              # box or a solid brush) less than this above it has no room for the marble (diameter 0.38) and
                              # is dropped. Found on the generated block-cluster maps: the 90 x 90 platform's top polygon ran
                              # on underneath every Cube.dif box, so the stack kept a phantom floor under each box, the edge
                              # rays walking the floor level never saw a box side as an edge (no rise, no edge at all), and the
                              # walk grid routed straight through boxes. Real maps have no floor polygon under a raised tile;
                              # this rule makes the generated maps read the same way. Map-agnostic: it only uses geometry.


SOLID_TOP_NORMAL_Z = 0.995     # a box top is level (within ~6 deg); ramps and tilted slabs are never solids


def solid_footprints(surfaces):
    """2026-10-09 10:00: boxes FUSED into a floor (no underside face) still leave the floor polygon running underneath them in the
    stack. Each upward polygon (a box top) whose VERTICAL faces share two or more of its vertices is a box: those faces reach
    down to the box bottom. Returns [(top_polygon_xy, z_top, z_bottom)]; build_height_stack drops floor levels inside that
    footprint between z_bottom and z_top (they are inside the solid). Pure geometry, any map."""
    def key(p):
        return (round(p[0], 2), round(p[1], 2), round(p[2], 2))
    surfaces = [(sf[0], sf[1]) for sf in surfaces]
    walls = [(v, n) for v, n in surfaces if abs(n[2]) < 0.3 and len(v) >= 3]
    wall_keys = [set(key(p) for p in v) for v, n in walls]
    out = []
    for v, n in surfaces:
        # 2026-10-09 13:40: boxes only (flat tops). A tilted slab (Sprawl 18 deg, Vortex Effect 14 deg ramps) shares
        # vertices with its own side faces too, and with the top taken as the MEAN vertex height half of the slab's own
        # surface lay "inside the solid" and was deleted (teleport probes: floor the game has, the map lacked).
        if n[2] <= SOLID_TOP_NORMAL_Z or len(v) < 3:
            continue
        tk = set(key(p) for p in v)
        z_top = float(np.mean([p[2] for p in v]))
        z_bot = None
        for wv, wk in zip(walls, wall_keys):
            if len(tk & wk) >= 2:
                zb = min(p[2] for p in wv[0])
                if zb < z_top - 0.2:
                    z_bot = zb if z_bot is None else min(z_bot, zb)
        if z_bot is not None:
            out.append(([(p[0], p[1]) for p in v], z_top, float(z_bot)))
    return out


def build_height_stack(surfaces, grid_res, z_min, z_max, k_max):
    parts = [surf_parts(sf) for sf in surfaces]
    floors = [(v, n, mu) for v, n, mu in parts if len(v) >= 3 and climbable(n, mu)]
    ceilings = [(v, n, mu) for v, n, mu in parts if n[2] < -FLOOR_NORMAL_Z and len(v) >= 3]
    solids = solid_footprints(surfaces)
    if not floors:
        raise RuntimeError('No floor surfaces found')

    pts = np.array([p for v, _, _ in floors for p in v], dtype=np.float64)
    pad = 2 * grid_res
    x_min, x_max = pts[:, 0].min() - pad, pts[:, 0].max() + pad
    y_min, y_max = pts[:, 1].min() - pad, pts[:, 1].max() + pad
    xs = np.arange(x_min, x_max + grid_res, grid_res)
    ys = np.arange(y_min, y_max + grid_res, grid_res)
    W, H = len(xs), len(ys)

    cell_lists = [[[] for _ in range(W)] for _ in range(H)]
    ceil_lists = [[[] for _ in range(W)] for _ in range(H)]
    n_kept = n_dropped_bounds = 0
    for verts, normal, mu, target in [(v, n, mu, cell_lists) for v, n, mu in floors] + [(v, n, mu, ceil_lists) for v, n, mu in ceilings]:
        nz = float(normal[2]); tan_a = math.sqrt(max(0.0, 1.0 - nz * nz)) / abs(nz) if abs(nz) > 1e-6 else 99.0
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
        path = MplPath(list(zip(vx, vy)))
        inside = path.contains_points(cells)
        # edge-safe (see EDGE_EPS); the diagonal offsets matter at polygon CORNERS, where a point lies on two edges and all
        # four axis offsets stay on an edge (2026-10-05 22:05: 8 corner points broke KOTM's symmetry, e.g. (-28.7, 11.5))
        for ox, oy in ((EDGE_EPS, 0.0), (-EDGE_EPS, 0.0), (0.0, EDGE_EPS), (0.0, -EDGE_EPS),
                       (EDGE_EPS, EDGE_EPS), (EDGE_EPS, -EDGE_EPS), (-EDGE_EPS, EDGE_EPS), (-EDGE_EPS, -EDGE_EPS)):
            inside |= path.contains_points(cells + np.array([ox, oy]))
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
            target[j][i].append((float(z), tan_a, mu) if target is cell_lists else float(z))
            n_kept += 1

    solid_lists = [[[] for _ in range(W)] for _ in range(H)]
    for poly, z_top, z_bot in solids:
        vx = np.array([q[0] for q in poly]); vy = np.array([q[1] for q in poly])
        i0 = max(0, int(np.floor((vx.min() - xs[0]) / grid_res)) - 1); i1 = min(W - 1, int(np.ceil((vx.max() - xs[0]) / grid_res)) + 1)
        j0 = max(0, int(np.floor((vy.min() - ys[0]) / grid_res)) - 1); j1 = min(H - 1, int(np.ceil((vy.max() - ys[0]) / grid_res)) + 1)
        if i1 < i0 or j1 < j0:
            continue
        gx, gy = np.meshgrid(xs[i0:i1 + 1], ys[j0:j1 + 1])
        cells = np.column_stack([gx.ravel(), gy.ravel()])
        inside = MplPath(list(zip(vx, vy))).contains_points(cells, radius=0.01)
        for (cx, cy), ok in zip(cells, inside):
            if ok:
                i = int(round((cx - xs[0]) / grid_res)); j = int(round((cy - ys[0]) / grid_res))
                solid_lists[j][i].append((z_bot, z_top))
    heights = np.full((k_max, H, W), np.nan, dtype=np.float32)
    slopes = np.full((k_max, H, W), np.nan, dtype=np.float32)      # rise/run of the surface at each level
    frictions = np.full((k_max, H, W), np.nan, dtype=np.float32)   # the game's friction for its texture
    max_used = 0
    n_walk = 0
    n_covered = 0
    for j in range(H):
        for i in range(W):
            hs = cell_lists[j][i]
            if not hs:
                continue
            sl = solid_lists[j][i]
            if sl:                            # inside a fused box (between its wall bottoms and its top): not a floor
                before = len(hs)
                hs = [h for h in hs if not any(zb - 0.05 < h[0] < zt - 0.05 for zb, zt in sl)]
                n_covered += before - len(hs)
                if not hs:
                    continue
            cs = ceil_lists[j][i]
            if cs:                            # COVER_CLEARANCE: a floor just under a ceiling is solid-covered, not a floor
                before = len(hs)
                hs = [h for h in hs if not any(-0.05 <= c - h[0] < COVER_CLEARANCE for c in cs)]
                n_covered += before - len(hs)
                if not hs:
                    continue
            hs = sorted(hs, key=lambda t: -t[0])
            merged = []                       # [(z, max slope, min friction)]: merge heights within 0.25 (same surface, mesh seams)
            for z, tan_a, mu in hs:
                z = round(z, 2)
                if merged and merged[-1][0] - z <= 0.25:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], tan_a), min(merged[-1][2], mu))
                else:
                    merged.append((z, tan_a, mu))
            # HEADROOM: a level with another level less than HEADROOM above it is covered (top-first order, so the
            # higher level is already in `kept` when the lower one is tested)
            kept = []
            for lv in merged:
                if kept and kept[-1][0] - lv[0] < HEADROOM:
                    n_covered += 1
                    continue
                kept.append(lv)
            merged = kept
            max_used = max(max_used, len(merged))
            for k, (z, tan_a, mu) in enumerate(merged[:k_max]):
                heights[k, j, i] = z; slopes[k, j, i] = tan_a; frictions[k, j, i] = mu
            n_walk += 1
    stats = dict(cells=W * H, walkable=n_walk, max_levels_used=max_used,
                 samples_kept=n_kept, samples_dropped_out_of_bounds=n_dropped_bounds, samples_dropped_covered=n_covered,
                 slopes=slopes, frictions=frictions)
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
    tmp = out_path + '.tmp.png'
    plt.savefig(tmp, dpi=110, facecolor='#1a1a1a')
    os.replace(tmp, out_path)       # 2026-10-09: OneDrive refuses in-place overwrites (Errno 22)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mapname', help='map name (e.g. KingOfTheMarble_Hunt) or path to a .mcs/.mis')
    ap.add_argument('--grid-res', type=float, default=DEFAULT_GRID_RES)
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
    for dif_path, pos, rot, scl in parse_interiors(mis_path):
        dif = hxDif.Dif.Load(dif_path)
        s = transform_surfaces(extract_surfaces(dif, with_material=True), pos, rot, scl)
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
    # 2026-10-09: write beside and replace. Overwriting a file OneDrive holds fails with Errno 22 and left a STALE map
    # (Sprawl kept its pre-fix contents while the build printed the new counts).
    tmp_path = npz_path + '.tmp.npz'
    np.savez_compressed(tmp_path, xs=xs, ys=ys, heights=heights, slope=stats['slopes'], friction=stats['frictions'],
                        grid_res=np.float32(args.grid_res),
                        map_name=np.array(mapname),
                        z_bounds=np.array([z_min if z_min is not None else -1e9,
                                           z_max if z_max is not None else 1e9], dtype=np.float32))
    os.replace(tmp_path, npz_path)
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
