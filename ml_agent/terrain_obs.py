"""Local terrain observation for the ML agent.

Gives the policy a view of the floor geometry around the marble, sampled from
a height stack precomputed from the map's .dif by generate_terrain_map.py
(one command per map, no hand tuning, no TorqueScript changes).

Why: without it the agent can only avoid holes by memorizing their XY
coordinates, which is slow to learn and never transfers to another map. With
it, "void two units to my right" is a local pattern that means the same thing
on every map, so what the agent learns from falling carries over.

No value judgment is attached to any sample. A drop to a lower floor reads as
"floor present, some distance below"; a void reads as "no floor"; a wall or
raised platform reads as "floor above". The agent learns from OOB outcomes
which patterns are dangerous, so intentional drops on multi-level maps are
not penalized by construction.

Layout (TERRAIN_DIM = 64):
    for each of 8 directions (E, NE, N, NW, W, SW, S, SE in the marble's frame)
        for each range in RANGES (2, 5, 10, 20 world units)
            [present, dz]
    present: 1.0 if any floor exists at that point (any level, within the map's
             in-bounds z range); 0.0 for void / off the map.
    dz:      (floor_z - marble_z) / DZ_SCALE clipped to [-1, 1], using the floor
             level whose height is closest to the marble's own (what the marble
             would meet first if it rolled there). A gentle ramp reads as a small
             signed value that grows with range; a wall or raised platform as a
             large positive; a drop as negative; a chasm with a floor far below
             as -1 with present=1, which is distinct from void (present=0, dz=-1).

The frame is the camera frame used by observer.cs (x = camera right,
y = camera forward). With the camera yaw locked at 0 this is the world frame;
sample() accepts the yaw so nothing here has to change if the camera ever
rotates again.

Edge rays (EDGE_DIM = 38, added 2026-09-15), appended after the 64 dims above:
    for each of RAY_HEADINGS (16) headings, every 22.5 deg starting at +x:
        [edge_dist, edge_dz]
    for each of the GEM_RAYS (3) nearest gems in the observation:
        [clear, edge_dz]
    A ray walks from the marble along the heading in RAY_STEP (0.5 u) steps,
    following the floor it is on (a level within FOLLOW_TOL of the current
    height, so ramps are followed) until the floor ends. edge_dist is that
    distance / RAY_RANGE (1.0 if the floor continues for the full range);
    edge_dz is what lies beyond the edge relative to the current floor, /
    DZ_SCALE clipped to [-1, 1]: -1 for void (OOB), a smaller negative for a
    drop to a lower floor, positive for a wall or raised platform; 0 when no
    edge was found. The gem rays walk the line to each gem up to the gem's
    own distance: clear = edge distance / gem distance, so 1.0 means the
    floor runs all the way to the gem and 0.15 means it ends after 15% of the
    way (a gap between us). Absent gems read [0, 0].
    Why: the 8x4 point samples cannot express "the floor ends 3 u ahead on the
    line to that gem" for a 4-unit hole (it falls between sample points), so
    neither PPO nor cloning from human demos could generalize hole avoidance.
    Map-independent: it is the same lookup on any map's height stack.
"""
import os
import math
import numpy as np

# Sample geometry
DIRECTIONS = [(math.cos(k * math.pi / 4.0), math.sin(k * math.pi / 4.0)) for k in range(8)]
RANGES = (2.0, 5.0, 10.0, 20.0)
CHANNELS = 2
TERRAIN_DIM = len(DIRECTIONS) * len(RANGES) * CHANNELS   # 64

# Encoding
DZ_SCALE = 10.0    # relative-height normalization: +/-10 world units map to +/-1

# (32, 2) offsets in the marble's frame, ordered direction-major, range-minor
_OFFSETS = np.array([[dx * r, dy * r] for (dx, dy) in DIRECTIONS for r in RANGES],
                    dtype=np.float32)

# Edge rays
RAY_HEADINGS = 16
RAY_RANGE = 20.0
RAY_STEP = 0.5
RAY_STEPS = int(round(RAY_RANGE / RAY_STEP))     # 40
FOLLOW_TOL = 1.0      # floor continuity per step (follows ramps up to ~63 deg)
GEM_RAYS = 3
EDGE_DIM = RAY_HEADINGS * 2 + GEM_RAYS * 2       # 38
PERCEPTION_DIM = TERRAIN_DIM + EDGE_DIM          # 102: what observe() returns
_RAY_DIRS = np.array([[math.cos(k * 2 * math.pi / RAY_HEADINGS), math.sin(k * 2 * math.pi / RAY_HEADINGS)]
                      for k in range(RAY_HEADINGS)], dtype=np.float32)
_RAY_S = (np.arange(RAY_STEPS, dtype=np.float32) + 1.0) * RAY_STEP     # 0.5 .. 20.0
GEM_SLOTS = 5          # raw obs layout: 5 gems x [dx, dy, dz, value, dist] from index 6
GEM_BASE = 6


def gem_rels_from_raw(raw):
    """(dx, dy, dz, present) for the first GEM_RAYS gem slots of a raw 35-dim obs."""
    out = []
    for i in range(GEM_RAYS):
        b = GEM_BASE + i * GEM_SLOTS
        present = float(raw[b + 4]) > -500.0
        out.append((float(raw[b]), float(raw[b + 1]), float(raw[b + 2]), present))
    return out


class TerrainMap:
    """Height stack lookup. Loads terrain_maps/terrain_<map>.npz."""

    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.path = path
        self.xs = d['xs'].astype(np.float32)              # (W,) cell centre x
        self.ys = d['ys'].astype(np.float32)              # (H,) cell centre y
        self.heights = d['heights'].astype(np.float32)    # (K, H, W), NaN = no floor; sorted top-first
        self.res = float(d['grid_res'])
        self.x0 = float(self.xs[0])
        self.y0 = float(self.ys[0])
        self.K, self.H, self.W = self.heights.shape
        self.name = str(d['map_name']) if 'map_name' in d.files else os.path.basename(path)
        self.z_bounds = tuple(float(v) for v in d['z_bounds']) if 'z_bounds' in d.files else None

    def contains(self, x, y):
        """True if (x, y) lies within the map's grid footprint."""
        return (self.xs[0] - self.res <= x <= self.xs[-1] + self.res
                and self.ys[0] - self.res <= y <= self.ys[-1] + self.res)

    # ------------------------------------------------------------------ resolution
    @staticmethod
    def resolve(name_or_path):
        """Accept a path, a map name ("KingOfTheMarble_Hunt") or a file stem."""
        if os.path.exists(name_or_path):
            return name_or_path
        base = os.path.basename(name_or_path)
        if base.endswith('.npz'):
            base = base[:-4]
        if base.startswith('terrain_'):
            base = base[len('terrain_'):]
        here = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(here, 'terrain_maps', f'terrain_{base}.npz')
        if os.path.exists(p):
            return p
        raise FileNotFoundError(
            f'No terrain map for "{name_or_path}". Generate it with:\n'
            f'    python generate_terrain_map.py {base}\n'
            f'(expected {p})')

    @staticmethod
    def flat_sample():
        """The sample a marble would see on an infinite flat floor: every point
        present, dz = 0. Used with --no-terrain and to pad old-format probes."""
        v = np.zeros(TERRAIN_DIM, dtype=np.float32)
        v[0::2] = 1.0
        return v

    @staticmethod
    def flat_edges():
        """Edge rays on an infinite flat floor: no edge on any heading, every
        gem line clear (gem rays read [1, 0] even for absent gems here)."""
        v = np.zeros(EDGE_DIM, dtype=np.float32)
        v[0::2] = 1.0
        return v

    @staticmethod
    def flat_observe():
        return np.concatenate([TerrainMap.flat_sample(), TerrainMap.flat_edges()])

    # ------------------------------------------------------------------ lookup
    def heights_at(self, xs, ys):
        """Heights (K, N) at world XY arrays; NaN where no floor / outside the grid."""
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        i = np.rint((xs - self.x0) / self.res).astype(np.int64)
        j = np.rint((ys - self.y0) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.W) & (j >= 0) & (j < self.H)
        out = np.full((self.K, len(xs)), np.nan, dtype=np.float32)
        if ok.any():
            out[:, ok] = self.heights[:, j[ok], i[ok]]
        return out

    def sample_points(self, x, y, yaw=0.0):
        """World XY of the 32 sample points for a marble at (x, y) with camera yaw."""
        if yaw != 0.0:
            c, s = math.cos(yaw), math.sin(yaw)
            # observer.cs: camera right = (cos, -sin), forward = (sin, cos) in world
            lx, ly = _OFFSETS[:, 0], _OFFSETS[:, 1]
            off = np.stack([lx * c + ly * s, -lx * s + ly * c], axis=1)
        else:
            off = _OFFSETS
        return off + np.array([x, y], dtype=np.float32)

    def sample(self, x, y, z, yaw=0.0):
        """The 64-dim terrain observation for a marble centred at world (x, y, z)."""
        pts = self.sample_points(x, y, yaw)
        h = self.heights_at(pts[:, 0], pts[:, 1])                      # (K, 32), NaN = no floor
        rel = h - z
        dist = np.where(np.isnan(rel), np.inf, np.abs(rel))
        nearest = dist.argmin(axis=0)                                   # level closest to my height
        present = np.isfinite(dist.min(axis=0))
        rel_nearest = rel[nearest, np.arange(rel.shape[1])]
        dz = np.where(present, np.clip(np.nan_to_num(rel_nearest) / DZ_SCALE, -1.0, 1.0), -1.0)
        out = np.empty(TERRAIN_DIM, dtype=np.float32)
        out[0::2] = present.astype(np.float32)
        out[1::2] = dz.astype(np.float32)
        return out

    # ------------------------------------------------------------------ edge rays
    def _march(self, x, y, z0, dirs, max_steps):
        """Walk rays from (x, y) at floor height z0 along unit `dirs` (R, 2).
        max_steps (R,) caps each ray. Returns (edge_dist, edge_dz): edge_dist in
        world units (inf if the floor continues for the whole ray), edge_dz the
        relative height of what lies beyond the edge (/DZ_SCALE, clipped; -1
        for void; 0 if no edge)."""
        R = len(dirs)
        pts = (np.asarray([x, y], dtype=np.float32)[None, None, :]
               + dirs[:, None, :] * _RAY_S[None, :, None])                # (R, S, 2)
        h = self.heights_at(pts[:, :, 0].ravel(), pts[:, :, 1].ravel())  # (K, R*S)
        h = h.reshape(self.K, R, RAY_STEPS)
        z_cur = np.full(R, float(z0), dtype=np.float32)
        edge_dist = np.full(R, np.inf, dtype=np.float32)
        edge_dz = np.zeros(R, dtype=np.float32)
        active = max_steps > 0
        ar = np.arange(R)
        for sidx in range(RAY_STEPS):
            if not active.any():
                break
            col = h[:, :, sidx]                                            # (K, R)
            diff = np.where(np.isnan(col), np.inf, np.abs(col - z_cur[None, :]))
            k = diff.argmin(axis=0)
            best = diff[k, ar]
            level = col[k, ar]
            cont = best <= FOLLOW_TOL
            hit = active & ~cont
            if hit.any():
                edge_dist[hit] = _RAY_S[sidx]
                beyond = np.isfinite(best)
                dz = np.where(beyond, np.clip(np.nan_to_num(level - z_cur) / DZ_SCALE, -1.0, 1.0), -1.0)
                edge_dz[hit] = dz[hit]
                active = active & ~hit
            z_cur = np.where(active & cont, level, z_cur)
            active = active & (sidx + 1 < max_steps)
        return edge_dist, edge_dz

    def floor_height(self, x, y, z):
        """Height of the floor level nearest to z at (x, y); z itself if none."""
        h = self.heights_at([x], [y])[:, 0]
        if np.all(np.isnan(h)):
            return float(z)
        return float(h[np.nanargmin(np.abs(h - z))])

    def edge_rays(self, x, y, z, gem_rels):
        """The EDGE_DIM-dim edge observation for a marble at world (x, y, z).
        gem_rels: list of (dx, dy, dz, present) for the GEM_RAYS nearest gems
        (see gem_rels_from_raw)."""
        z0 = self.floor_height(x, y, z)
        dirs = np.zeros((RAY_HEADINGS + GEM_RAYS, 2), dtype=np.float32)
        dirs[:RAY_HEADINGS] = _RAY_DIRS
        max_steps = np.full(RAY_HEADINGS + GEM_RAYS, RAY_STEPS, dtype=np.int64)
        gem_d = np.zeros(GEM_RAYS, dtype=np.float32)
        for i in range(GEM_RAYS):
            dx, dy, dz, present = gem_rels[i] if i < len(gem_rels) else (0.0, 0.0, 0.0, False)
            d = math.hypot(dx, dy)
            if present and d > 1e-3:
                dirs[RAY_HEADINGS + i] = (dx / d, dy / d)
                gem_d[i] = d
                max_steps[RAY_HEADINGS + i] = min(RAY_STEPS, int(math.ceil(d / RAY_STEP)))
            else:
                dirs[RAY_HEADINGS + i] = (1.0, 0.0)
                max_steps[RAY_HEADINGS + i] = 0
        dist, dz = self._march(x, y, z0, dirs, max_steps)
        out = np.zeros(EDGE_DIM, dtype=np.float32)
        hd = dist[:RAY_HEADINGS]
        out[0:RAY_HEADINGS * 2:2] = np.where(np.isfinite(hd), hd / RAY_RANGE, 1.0)
        out[1:RAY_HEADINGS * 2:2] = dz[:RAY_HEADINGS]
        for i in range(GEM_RAYS):
            j = RAY_HEADINGS * 2 + i * 2
            if gem_d[i] > 0:
                gd = dist[RAY_HEADINGS + i]
                out[j] = min(1.0, gd / gem_d[i]) if np.isfinite(gd) else 1.0
                out[j + 1] = dz[RAY_HEADINGS + i]
        return out

    def observe(self, raw):
        """The full PERCEPTION_DIM (64 + 38) block for a raw 35-dim game obs in
        the world frame: point samples + edge rays."""
        x, y, z = float(raw[0]), float(raw[1]), float(raw[2])
        return np.concatenate([self.sample(x, y, z, 0.0), self.edge_rays(x, y, z, gem_rels_from_raw(raw))])

    def describe_edges(self, vec):
        """Human-readable dump of an edge-ray vector."""
        rows = []
        for k in range(RAY_HEADINGS):
            d, dz = vec[2 * k], vec[2 * k + 1]
            ang = k * 360 // RAY_HEADINGS
            tag = 'clear' if d >= 0.999 else f'{d * RAY_RANGE:4.1f}u ' + ('void' if dz <= -0.999 else f'dz{dz * DZ_SCALE:+.0f}')
            rows.append(f'{ang:3d}deg {tag}')
        for i in range(GEM_RAYS):
            c, dz = vec[RAY_HEADINGS * 2 + 2 * i], vec[RAY_HEADINGS * 2 + 2 * i + 1]
            rows.append(f'gem{i + 1}: clear {c:.2f}' + ('' if c >= 0.999 else (' void' if dz <= -0.999 else f' dz{dz * DZ_SCALE:+.0f}')))
        return '\n'.join(rows)

    def describe_sample(self, vec):
        """Human-readable dump of a sample vector (for logs / diagnostics)."""
        names = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
        rows = []
        for di, name in enumerate(names):
            cells = []
            for ri, r in enumerate(RANGES):
                k = (di * len(RANGES) + ri) * CHANNELS
                p, dz = vec[k], vec[k + 1]
                if p < 0.5:
                    tag = 'void'
                elif abs(dz) < 0.03:
                    tag = 'flat'
                else:
                    tag = f'{dz * DZ_SCALE:+.1f}'
                cells.append(f'{int(r):>2}u:{tag:<5}')
            rows.append(f'{name:>2} ' + ' '.join(cells))
        return '\n'.join(rows)
