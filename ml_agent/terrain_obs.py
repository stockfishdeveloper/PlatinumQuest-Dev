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
