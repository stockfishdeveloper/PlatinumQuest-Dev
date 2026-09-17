"""Map-agnostic terrain queries for the navigator: local height crops, a walkable
grid, goal-distance fields (Dijkstra) and start/goal sampling.

Everything comes from the height stack that generate_terrain_map.py builds
from the map's .dif files (terrain_maps/terrain_<map>.npz); nothing here is
tuned per map.

    python -m nav.terrain --check FlatGemTraining_Hunt     renders the walk grid, a goal field and a crop
"""
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrain_obs import TerrainMap, DZ_SCALE  # noqa: E402

CROP_CELLS = 32
FINE_RES = 0.5        # 16 u each way
COARSE_RES = 2.0      # 64 u each way
CROP_CHANNELS = 3     # per crop: nearest-level relative height, floor present, second-level relative height
CROP_SHAPE = (2 * CROP_CHANNELS, CROP_CELLS, CROP_CELLS)

WALK_RES = 1.0
MAX_SLOPE = 0.6       # rise/run above which a cell is not walkable (31 deg); a marble teleported onto a
                      # 45-deg edge bevel slid straight off (King of the Marble, 2026-09-17)
EDGE_COST = 2.0       # path cost multiplier for cells bordering a drop
DROP_EDGE = 2.0       # a neighbour more than this far below counts as a drop (edge)


def _crop_offsets(res):
    c = (np.arange(CROP_CELLS, dtype=np.float32) - (CROP_CELLS - 1) / 2.0) * res
    gx, gy = np.meshgrid(c, c)                    # gy varies along rows (north up in the array = +y)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


class TerrainGrid(TerrainMap):
    """TerrainMap plus crops, walkability, distance fields and sampling."""

    def __init__(self, path, walk_res=WALK_RES):
        super().__init__(path)
        self._fine = _crop_offsets(FINE_RES)
        self._coarse = _crop_offsets(COARSE_RES)
        self.walk_res = float(walk_res)
        self.z_floor_min = float(np.nanmin(self.heights))
        self._build_walk_grid()

    # ------------------------------------------------------------------ crops
    def _crop_one(self, x, y, z, offsets):
        pts = offsets + np.array([x, y], dtype=np.float32)
        h = self.heights_at(pts[:, 0], pts[:, 1])                       # (K, N), NaN = no floor
        rel = h - z
        dist = np.where(np.isnan(rel), np.inf, np.abs(rel))
        order = np.argsort(dist, axis=0)
        n = rel.shape[1]
        ar = np.arange(n)
        first = rel[order[0], ar]
        present = np.isfinite(first)
        ch0 = np.where(present, np.clip(np.nan_to_num(first) / DZ_SCALE, -1.0, 1.0), -1.0)
        if self.K > 1:
            second = rel[order[1], ar]
            ch2 = np.where(np.isfinite(second), np.clip(np.nan_to_num(second) / DZ_SCALE, -1.0, 1.0), 0.0)
        else:
            ch2 = np.zeros(n, dtype=np.float32)
        out = np.stack([ch0, present.astype(np.float32), ch2]).astype(np.float32)
        return out.reshape(CROP_CHANNELS, CROP_CELLS, CROP_CELLS)

    def crop(self, x, y, z):
        """(6, 32, 32): fine crop (0.5 u cells) then coarse crop (2 u cells), world-axis aligned,
        row index increasing with +y, column index with +x."""
        return np.concatenate([self._crop_one(x, y, z, self._fine), self._crop_one(x, y, z, self._coarse)], axis=0)

    # ------------------------------------------------------------------ walk grid
    def _build_walk_grid(self):
        step = max(1, int(round(self.walk_res / self.res)))
        top = self.heights[0][::step, ::step]                    # top floor level per walk cell
        self.wxs = self.xs[::step]
        self.wys = self.ys[::step]
        self.wH, self.wW = top.shape
        present = np.isfinite(top)
        filled = np.where(present, top, np.nan)
        # slope from finite neighbours only
        gy, gx = np.gradient(np.nan_to_num(filled, nan=0.0), self.walk_res)
        slope = np.hypot(gx, gy)
        # drops: any 8-neighbour missing or more than DROP_EDGE below
        pad = np.pad(filled, 1, constant_values=np.nan)
        edge = np.zeros_like(present)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = pad[1 + dy:1 + dy + self.wH, 1 + dx:1 + dx + self.wW]
                edge |= present & (np.isnan(nb) | (filled - nb > DROP_EDGE))
        self.walk_top = filled
        self.walkable = present & (slope < MAX_SLOPE)
        self.edge = edge & self.walkable
        # interior: walkable, not at a drop, and every 8-neighbour walkable too (starts and
        # goals go here, so a marble is never placed on a bevel or half a cell from the void)
        padw = np.pad(self.walkable, 1, constant_values=False)
        inner = self.walkable.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                inner &= padw[1 + dy:1 + dy + self.wH, 1 + dx:1 + dx + self.wW]
        self.interior = inner & ~self.edge
        self._graph = None

    def cell_of(self, x, y):
        i = int(round((x - self.wxs[0]) / self.walk_res))
        j = int(round((y - self.wys[0]) / self.walk_res))
        return j, i

    def in_walk_grid(self, j, i):
        return 0 <= j < self.wH and 0 <= i < self.wW

    def _build_graph(self):
        from scipy.sparse import coo_matrix
        H, W = self.wH, self.wW
        idx = np.full((H, W), -1, dtype=np.int64)
        nodes = np.argwhere(self.walkable)
        idx[nodes[:, 0], nodes[:, 1]] = np.arange(len(nodes))
        rows, cols, w = [], [], []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = nodes
            b = nodes + np.array([dy, dx])
            ok = (b[:, 0] >= 0) & (b[:, 0] < H) & (b[:, 1] >= 0) & (b[:, 1] < W)
            a, b = a[ok], b[ok]
            ib = idx[b[:, 0], b[:, 1]]
            ok = ib >= 0
            a, b, ib = a[ok], b[ok], ib[ok]
            ia = idx[a[:, 0], a[:, 1]]
            d = self.walk_res * math.hypot(dx, dy)
            dz = np.abs(self.walk_top[b[:, 0], b[:, 1]] - self.walk_top[a[:, 0], a[:, 1]])
            cost = d * (1.0 + dz / d) * np.where(self.edge[a[:, 0], a[:, 1]] | self.edge[b[:, 0], b[:, 1]], EDGE_COST, 1.0)
            rows.append(ia); cols.append(ib); w.append(cost)
        rows = np.concatenate(rows); cols = np.concatenate(cols); w = np.concatenate(w)
        n = len(nodes)
        g = coo_matrix((np.concatenate([w, w]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))), shape=(n, n)).tocsr()
        self._graph = g
        self._node_idx = idx
        self._nodes = nodes

    def goal_field(self, x, y):
        """Walking distance from every walk cell to the goal at world (x, y); inf where unreachable.
        Returns an (H, W) array. The goal snaps to the nearest walkable cell within 3 u."""
        from scipy.sparse.csgraph import dijkstra
        if self._graph is None:
            self._build_graph()
        j, i = self.cell_of(x, y)
        src = self._nearest_walkable(j, i, radius=3)
        field = np.full((self.wH, self.wW), np.inf, dtype=np.float32)
        if src is None:
            return field
        d = dijkstra(self._graph, directed=False, indices=[self._node_idx[src]])[0]
        field[self._nodes[:, 0], self._nodes[:, 1]] = d
        return field

    def _nearest_walkable(self, j, i, radius=3):
        best, bd = None, 1e9
        for dj in range(-radius, radius + 1):
            for di in range(-radius, radius + 1):
                jj, ii = j + dj, i + di
                if self.in_walk_grid(jj, ii) and self.walkable[jj, ii]:
                    d = dj * dj + di * di
                    if d < bd:
                        best, bd = (jj, ii), d
        return best

    def dist_at(self, field, x, y, goal_xy):
        """Path distance to the goal from world (x, y); falls back to straight-line + 5 u off the walkable grid."""
        j, i = self.cell_of(x, y)
        if self.in_walk_grid(j, i) and np.isfinite(field[j, i]):
            return float(field[j, i])
        c = self._nearest_walkable(j, i, radius=2)
        if c is not None and np.isfinite(field[c]):
            return float(field[c]) + self.walk_res
        return float(math.hypot(x - goal_xy[0], y - goal_xy[1])) + 5.0

    # ------------------------------------------------------------------ sampling
    def _interior_cells(self):
        return np.argwhere(self.interior)

    def sample_start(self, rng):
        cells = self._interior_cells()
        j, i = cells[rng.integers(len(cells))]
        return float(self.wxs[i]), float(self.wys[j]), float(self.walk_top[j, i])

    def sample_goal(self, x, y, rng, dmin=8.0, dmax=40.0, tries=50):
        """A walkable, non-edge cell between dmin and dmax (straight line) from (x, y) that is
        reachable (finite path). Returns (gx, gy, gz, path_len) or None."""
        field = self.goal_field(x, y)          # distances from the marble to everything
        cells = self._interior_cells()
        cx = self.wxs[cells[:, 1]]; cy = self.wys[cells[:, 0]]
        d = np.hypot(cx - x, cy - y)
        ok = (d >= dmin) & (d <= dmax) & np.isfinite(field[cells[:, 0], cells[:, 1]])
        if not ok.any():
            ok = (d >= 2.0) & np.isfinite(field[cells[:, 0], cells[:, 1]])
            if not ok.any():
                return None
        cand = cells[ok]
        j, i = cand[rng.integers(len(cand))]
        return float(self.wxs[i]), float(self.wys[j]), float(self.walk_top[j, i]), float(field[j, i])

    def floor_z(self, x, y, z):
        return self.floor_height(x, y, z)


def _check(name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = TerrainGrid(TerrainMap.resolve(name))
    rng = np.random.default_rng(0)
    sx, sy, sz = t.sample_start(rng)
    g = t.sample_goal(sx, sy, rng)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(np.where(t.walkable, 1.0, 0.0) + np.where(t.edge, 1.0, 0.0), origin='lower',
                 extent=[t.wxs[0], t.wxs[-1], t.wys[0], t.wys[-1]]); ax[0].set_title('walkable (1) / edge (2)')
    if g is not None:
        gx, gy, gz, pl = g
        f = t.goal_field(gx, gy)
        ax[1].imshow(np.where(np.isfinite(f), f, np.nan), origin='lower', extent=[t.wxs[0], t.wxs[-1], t.wys[0], t.wys[-1]])
        ax[1].plot([sx, gx], [sy, gy], 'r.-'); ax[1].set_title(f'goal field; start->goal path {pl:.1f} u')
    c = t.crop(sx, sy, sz + 0.5)
    ax[2].imshow(c[0], origin='lower', vmin=-1, vmax=1); ax[2].set_title('fine crop: relative height at start')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'terrain_maps', f'check_{name}.png')
    fig.savefig(out, dpi=80); print('wrote', os.path.normpath(out))
    print(f'walk grid {t.wH}x{t.wW} at {t.walk_res} u: walkable {int(t.walkable.sum())}, edge {int(t.edge.sum())}; start {(sx, sy, sz)}; goal {g}')


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--check':
        _check(sys.argv[2])
    else:
        print(__doc__)
