"""Map-agnostic terrain queries for the navigator: local height crops, a walkable
grid, goal-distance fields (Dijkstra) and start/goal sampling.

Everything comes from the height stack that generate_terrain_map.py builds
from the map's .dif files (terrain_maps/terrain_<map>.npz); nothing here is
tuned per map.

    python -m nav.terrain --check FlatGemTraining_Hunt     renders the walk grid, a goal field and a crop
"""
import io
import os
import sys
import math
import warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrain_obs import TerrainMap, DZ_SCALE  # noqa: E402

# Fraction of goals drawn from ALL walkable ground rather than interior-only cells. Real gems
# spawn anywhere walkable (41 % of KOTM's walkable cells are edge cells, 15 % on islands),
# so 1.0 reproduces the real distribution. See HANDOFF_NAV_TRAINING.md section 4b.
EDGE_GOAL_P = 1.0

# Draw goals from the map's REAL gem spawn points rather than arbitrary walkable cells, so the
# training task cannot validate itself against terrain-map errors (see HANDOFF 5d).
GOALS_FROM_SPAWNS = os.environ.get('NAV_GOALS_FROM_SPAWNS', '1') == '1'
_SPAWN_CACHE = {}


def gem_spawn_points(map_name):
    """World (x, y, z) of every gem spawn Item in the mission -- the only places a real gem can
    appear. These are the Items huntGems.cs scans (dataBlock "GemItem*")."""
    if map_name in _SPAWN_CACHE:
        return _SPAWN_CACHE[map_name]
    import re
    pts = []
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from generate_terrain_map import resolve_mission
        txt = io.open(resolve_mission(map_name), encoding='utf-8', errors='ignore').read()
        for m in re.finditer(r'new\s+Item\s*\([^)]*\)\s*\{(.*?)\};', txt, re.S):
            body = m.group(1)
            db = re.search(r'dataBlock\s*=\s*"([^"]+)"', body)
            pp = re.search(r'position\s*=\s*"([^"]+)"', body)
            if db and pp and db.group(1).lower().startswith('gemitem'):
                x, y, z = (float(v) for v in pp.group(1).split()[:3])
                pts.append((x, y, z))
    except Exception as e:
        sys.stderr.write(f'gem_spawn_points({map_name!r}) FAILED: {e}\n')
        pts = []
    if not pts:
        # Loud on purpose: a silent fall back to grid-cell goals reintroduces exactly the
        # self-validating loop this rule exists to break (HANDOFF 5d).
        sys.stderr.write(f'WARNING no gem spawn points for {map_name!r}; goals fall back to grid cells\n')
    _SPAWN_CACHE[map_name] = pts
    return pts

CROP_CELLS = 32
FINE_RES = 0.5        # 16 u each way
COARSE_RES = 2.0      # 64 u each way
CROP_CHANNELS = 3     # per crop: nearest-level relative height, floor present, second-level relative height
CROP_SHAPE = (2 * CROP_CHANNELS, CROP_CELLS, CROP_CELLS)

WALK_RES = 1.0
MAX_SLOPE = 0.6       # rise/run above which a cell is not walkable (31 deg); a marble teleported onto a
                      # 45-deg edge bevel slid straight off (King of the Marble, 2026-09-17)
EDGE_COST = 2.0       # path cost multiplier for cells bordering a drop. NOTE (2026-09-21): on a
                      # map with ~3 u walkways this ring is every cell, so it is close to a
                      # constant and gives the policy no early "turn here" direction. See the
                      # rejected inflation note below and HANDOFF section 18.
# TRIED AND REJECTED 2026-09-21: graded costmap inflation (cost x3 at a lip decaying to x1 over
# 3 u) to make the routing field turn the marble early. DEGENERATE ON THIS GEOMETRY: KOTM's
# walkways are ~3 u wide, so distance-to-nearest-lip is 0.00 for essentially every walkable cell.
# Inflation multiplied every edge by the same factor and changed no gradient DIRECTION at all.
# self.edge_dist is kept below because it is a useful measurement, just not a useful cost.
DROP_EDGE = 2.0       # a neighbour more than this far below counts as a drop (edge)
JUMP_GOAL_P = 0.33    # share of spawn-goal draws that prefer a goal whose shortest route uses a jump
JUMP_SAVE_MIN = 1.0   # u the jump route must save over the walk-only route to count (HANDOFF 28.13)
JUMP_GAP = 4.0        # (superseded 2026-09-23, HANDOFF 28.27) the old 8-direction rule admitted gaps up to
                      # this wide. Jump edges now come from nav/physics.py: any of 16 headings, gap up to
                      # physics.MAX_JUMP_GAP (measured range at CRUISE_SPEED minus the landing margin).
JUMP_HEADINGS = 16    # headings searched from every edge cell (the observation's 16 ray headings)
JUMP_DROP = 6.0       # landing may be this far below the take-off cell ...
JUMP_RISE = 1.5       # ... or this far above
JUMP_COST = 1.2       # 1.5 -> 1.2 on 2026-09-23 (HANDOFF 28.27): a jump edge is priced as its measured flight
                      # (0.76 s, about the rolling time for the same distance) plus a landing loss. At 1.5 the
                      # hypotenuse of a 7+7 u corner (10 u x 1.5 = 15) still lost to walking the legs (14).
                      # (superseded) 3.0 -> 1.5 on 2026-09-22 (HANDOFF 28.13). Path cost multiplier for a jump edge (per
                      # unit of gap). A hop is priced near its TIME cost (a 3 u hop takes about as long
                      # as rolling 4.5 u); the fall risk is priced by FALL in the reward, not here. At 3.0
                      # (and doubled by the duplicate-edge bug above) no KOTM route ever used a jump, so
                      # neither PROGRESS nor the jump curriculum could ever reward one.


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
        self._build_graph()          # walk graph, jump edges and floor components, once per map
        # the real gem spawn pool for this map, snapped to walk cells (empty if unavailable)
        self.gem_spawns = []
        if GOALS_FROM_SPAWNS:
            for (gx, gy, gz) in gem_spawn_points(getattr(self, 'name', '') or ''):
                j, i = self.cell_of(gx, gy)
                if not self.in_walk_grid(j, i):
                    continue
                if not self.walkable[j, i]:
                    c = self._nearest_walkable(j, i, radius=2)
                    if c is None:
                        continue
                    j, i = c
                # Keep the gem's TRUE mission position as the goal. The cell (j, i) is only for
                # reachability and Dijkstra lookups. Snapping the goal to the walk-cell centre and
                # replacing z with the floor height put every waypoint 0.71 u horizontally and
                # 0.05 u vertically off the gem -- and with a 0.65 u pickup radius the marble
                # could sit exactly on its waypoint without touching the gem.
                self.gem_spawns.append((float(gx), float(gy), float(gz), j, i))

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
        # Aggregate the step x step block of fine cells instead of sampling ONE of them
        # (fixed 2026-09-19): `heights[0][::step, ::step]` threw away floor that existed half a
        # cell away, marking 174 KOTM walk cells empty when their block did contain floor.
        h0 = self.heights[0]
        # pad (not truncate) up to a whole number of blocks so the last row/column of the map
        # is not silently dropped
        pj = (-h0.shape[0]) % step
        pi = (-h0.shape[1]) % step
        if pj or pi:
            h0 = np.pad(h0, ((0, pj), (0, pi)), constant_values=np.nan)
        blocks = h0.reshape(h0.shape[0] // step, step, h0.shape[1] // step, step)
        # How a 1 u walk cell summarises its step x step block of 0.5 u height cells.
        # NAV_WALK_AGG: 'sub' (the original single sample), 'majority' (nanmax, but only if most
        # sub-cells are floored), 'any' (nanmax over any floored sub-cell). Kept switchable
        # because this is being A/B tested against real-round score, which is the only metric
        # that has told the truth about terrain changes.
        agg = os.environ.get('NAV_WALK_AGG', 'sub')
        if agg == 'sub':
            top = blocks[:, 0, :, 0]
        else:
            finite = np.isfinite(blocks).sum(axis=(1, 3))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN blocks are expected
                top = np.nanmax(blocks, axis=(1, 3))
            if agg == 'majority':
                top = np.where(finite * 2 >= step * step, top, np.nan)
        self.wxs = self.xs[::step]
        self.wys = self.ys[::step]
        self.wH, self.wW = top.shape
        present = np.isfinite(top)
        filled = np.where(present, top, np.nan)
        # Slope from finite neighbours only. nan_to_num(nan=0.0) was WRONG (fixed 2026-09-19):
        # a missing cell became z = 0 while these floors sit at z ~ 20, so every gap read as a
        # 20-unit cliff and the gradient at its 8 neighbours blew past MAX_SLOPE = 0.6. One
        # missing cell therefore made all of its neighbours non-walkable, turning small
        # rasterisation gaps into large PHANTOM HOLES the policy refused to cross -- verified by
        # teleporting the marble into them: 4 of 6 "holes" were solid ground.
        # Filling each gap with its nearest finite height contributes no artificial gradient;
        # `present` still masks the genuinely missing cells, and the edge test below is
        # unaffected because it reads `filled`, which keeps its NaNs.
        if present.any() and not present.all():
            from scipy.ndimage import distance_transform_edt
            _, (ij, ii) = distance_transform_edt(~present, return_indices=True)
            slope_src = filled[ij, ii]
        else:
            slope_src = np.nan_to_num(filled, nan=0.0)
        gy, gx = np.gradient(slope_src, self.walk_res)
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
        # Graded inflation: distance from every cell to the nearest lip, turned into a routing
        # cost multiplier. distance_transform_edt measures distance to the nearest ZERO, so feed
        # it the complement of the lip mask. Non-walkable cells get 0 distance too, which makes
        # the band reach round the far side of a thin wall as well -- harmless, it is all void.
        from scipy.ndimage import distance_transform_edt
        lip = self.edge | (~self.walkable)
        dist_u = distance_transform_edt(~lip) * self.walk_res
        self.edge_dist = dist_u.astype(np.float32)
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

    def _measured_jump_edges(self):
        """[(j, i, jj, ii, gap_u), ...]: takeoff walk cell, landing walk cell, gap length."""
        from nav.physics import MAX_JUMP_GAP
        present = np.isfinite(self.heights).any(0)
        H, W = present.shape
        heads = [(math.cos(2 * math.pi * k / JUMP_HEADINGS), math.sin(2 * math.pi * k / JUMP_HEADINGS)) for k in range(JUMP_HEADINGS)]
        step = self.res
        n_steps = int(math.ceil((MAX_JUMP_GAP + 1.5) / step))
        out = []
        for (j, i) in np.argwhere(self.edge):
            x = float(self.wxs[i]); y = float(self.wys[j]); z0 = float(self.walk_top[j, i])
            for cx, cy in heads:
                in_void = False; d_void = 0.0
                for s in range(1, n_steps + 1):
                    d = s * step
                    ii = int(round((x + cx * d - self.x0) / step)); jj = int(round((y + cy * d - self.y0) / step))
                    if not (0 <= ii < W and 0 <= jj < H):
                        break
                    if not present[jj, ii]:
                        if not in_void:
                            in_void = True; d_void = d
                        continue
                    if not in_void:
                        if d > 1.5:
                            break                  # floor continues: this heading has no lip here
                        continue
                    gap = d - d_void              # void samples span [d_void, d): their extent is d - d_void
                    if gap > MAX_JUMP_GAP:
                        break
                    zs = self.heights[:, jj, ii]; zs = zs[np.isfinite(zs)]
                    if len(zs) and np.any((zs - z0 >= -JUMP_DROP) & (zs - z0 <= JUMP_RISE)):
                        lj, li = self.cell_of(x + cx * (d + 0.5), y + cy * (d + 0.5))
                        if self.in_walk_grid(lj, li) and self.walkable[lj, li] and (lj, li) != (j, i):
                            out.append((j, i, lj, li, float(gap)))
                    break
        return out

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
        # Jump edges (2026-09-17): from every edge cell, look across the gap in the 8
        # directions for the first walkable cell within JUMP_GAP; the cells between must be
        # non-walkable (a real gap) and the landing height within [-JUMP_DROP, +JUMP_RISE].
        # Without these, crossing a gap was never "progress" and goals were never sampled
        # across one, so the policy learned to brake at edges instead of jumping.
        # MEASURED jump edges (2026-09-23, HANDOFF 28.27; the 8-direction / 4 u rule is above in the
        # comment history). From every edge cell, march the FINE height map in JUMP_HEADINGS headings:
        # the void must begin within 1.5 u (it is this cell's own lip), and the first floor after it
        # within physics.MAX_JUMP_GAP u with a height change in [-JUMP_DROP, +JUMP_RISE] is a landing.
        # MAX_JUMP_GAP is the measured range at CRUISE_SPEED minus LANDING_MARGIN (~7 u on KOTM's
        # marble), so a 7 x 7 hole is admitted straight across and every corner cut is admitted too.
        jr, jc, jw = [], [], []
        self.jump_edge_cells = []
        for (j, i, jj, ii, gap) in self._measured_jump_edges():
            jr.append(idx[j, i]); jc.append(idx[jj, ii]); jw.append(gap * JUMP_COST)
            self.jump_edge_cells.append((j, i, jj, ii, gap))
        self.jump_edges = len(jr)
        rows = np.concatenate(rows); cols = np.concatenate(cols); w = np.concatenate(w)
        n = len(nodes)
        # connected components WITHOUT jump edges: "same island" test for goal sampling
        from scipy.sparse.csgraph import connected_components
        comp = np.full((H, W), -1, dtype=np.int64)
        self._graph_nojump = None
        if n:
            g0 = coo_matrix((np.concatenate([w, w]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))), shape=(n, n)).tocsr()
            _, labels = connected_components(g0, directed=False)
            comp[nodes[:, 0], nodes[:, 1]] = labels
            self._graph_nojump = g0       # walk-only graph: goal_field(..., jumps=False) measures the detour a jump saves
        self.component = comp
        if jr:
            # DEDUPLICATE (fixed 2026-09-22, HANDOFF 28.13): every gap is found from BOTH of its edge cells,
            # so each jump edge was appended twice and coo_matrix SUMMED the duplicates: a 3 u hop over
            # KOTM's centre hole cost 18 instead of 9, and no shortest path on KOTM ever used a jump.
            best = {}
            for a, b, c in zip(jr, jc, jw):
                key = (min(a, b), max(a, b))
                if key not in best or c < best[key]:
                    best[key] = c
            jr = [k[0] for k in best]; jc = [k[1] for k in best]; jw = [best[k] for k in best]
            self.jump_edges = len(jr)
            rows = np.concatenate([rows, np.array(jr)]); cols = np.concatenate([cols, np.array(jc)])
            w = np.concatenate([w, np.array(jw, dtype=np.float64)])
        g = coo_matrix((np.concatenate([w, w]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))), shape=(n, n)).tocsr()
        self._graph = g
        self._node_idx = idx
        self._nodes = nodes

    def goal_field(self, x, y, jumps=True):
        """Walking distance from every walk cell to the goal at world (x, y); inf where unreachable.
        Returns an (H, W) array. The goal snaps to the nearest walkable cell within 3 u.
        `jumps=False` uses the walk-only graph (no jump edges): comparing the two fields tells
        whether the shortest route to a cell uses a jump, and how much the jump saves."""
        from scipy.sparse.csgraph import dijkstra
        if self._graph is None:
            self._build_graph()
        j, i = self.cell_of(x, y)
        src = self._nearest_walkable(j, i, radius=3)
        field = np.full((self.wH, self.wW), np.inf, dtype=np.float32)
        if src is None:
            return field
        graph = self._graph if (jumps or self._graph_nojump is None) else self._graph_nojump
        d = dijkstra(graph, directed=False, indices=[self._node_idx[src]])[0]
        field[self._nodes[:, 0], self._nodes[:, 1]] = d
        return field

    def walkable_at(self, x, y):
        """True if the walk cell under world (x, y) is walkable floor (False over a hole or off the map)."""
        j, i = self.cell_of(x, y)
        return bool(self.in_walk_grid(j, i) and self.walkable[j, i])

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

    def _sample_spawn_goal(self, x, y, rng, dmin, dmax, field):
        """A reachable REAL gem spawn point between dmin and dmax of (x, y), or None.
        Widens to any reachable spawn point before giving up, because a map may have only a
        handful (KOTM 12, JumpOnly 3) and the distance band can exclude them all."""
        cand = []
        for (gx, gy, gz, j, i) in self.gem_spawns:
            if not np.isfinite(field[j, i]):
                continue
            d = math.hypot(gx - x, gy - y)
            if d < 1.0:
                continue                       # already standing on it
            cand.append((gx, gy, gz, float(field[j, i]), d, j, i))
        if not cand:
            return None
        band = [c for c in cand if dmin <= c[4] <= dmax]
        pick = band if band else cand
        # JUMP CURRICULUM (2026-09-22, HANDOFF 28.13): with probability JUMP_GOAL_P prefer a spawn whose
        # shortest route from here uses a jump edge and saves at least JUMP_SAVE_MIN u over walking.
        # The policy had jumping trained out of it (JUMP_DAMP 3, 0.1 % of real-round decisions) and a
        # hop over KOTM's centre hole is 5.7 u against 12.5 u round it. Practice needs the task.
        if JUMP_GOAL_P > 0.0 and rng.random() < JUMP_GOAL_P:
            nojump = self.goal_field(x, y, jumps=False)
            saves = [c for c in pick if (nojump[c[5], c[6]] if len(c) > 6 else np.inf) - c[3] >= JUMP_SAVE_MIN]
            if saves:
                pick = saves
        gx, gy, gz, path_len, _ = pick[rng.integers(len(pick))][:5]
        return (gx, gy, gz, path_len)

    def _goal_cells(self, rng):
        """Cells a GOAL may occupy. Real gems spawn anywhere walkable, including hard against a
        drop, so with probability EDGE_GOAL_P draw from the whole walkable set rather than the
        interior-only set. Training on interior cells alone taught the policy to avoid exactly
        the places gems appear (see HANDOFF 4b)."""
        if EDGE_GOAL_P >= 1.0 or rng.random() < EDGE_GOAL_P:
            return np.argwhere(self.walkable)
        return np.argwhere(self.interior)

    def sample_start(self, rng):
        cells = self._interior_cells()
        j, i = cells[rng.integers(len(cells))]
        return float(self.wxs[i]), float(self.wys[j]), float(self.walk_top[j, i])

    def sample_goal(self, x, y, rng, dmin=8.0, dmax=40.0, tries=50, cross_gap_p=0.5):
        """A walkable cell between dmin and dmax (straight line) from (x, y) that is
        reachable (finite path, jump edges included). With probability 1 - cross_gap_p the goal
        is restricted to the marble's own connected floor (no gap to cross), so half the
        segments train plain locomotion and half train gap crossing.
        Returns (gx, gy, gz, path_len) or None."""
        field = self.goal_field(x, y)          # distances from the marble to everything
        if GOALS_FROM_SPAWNS and getattr(self, 'gem_spawns', None):
            g = self._sample_spawn_goal(x, y, rng, dmin, dmax, field)
            if g is not None:
                return g
            # else fall through: no spawn point fits this distance band, use the grid
        cells = self._goal_cells(rng)
        cx = self.wxs[cells[:, 1]]; cy = self.wys[cells[:, 0]]
        d = np.hypot(cx - x, cy - y)
        ok = (d >= dmin) & (d <= dmax) & np.isfinite(field[cells[:, 0], cells[:, 1]])
        if rng.random() >= cross_gap_p:
            j0, i0 = self.cell_of(x, y)
            src = self._nearest_walkable(j0, i0, radius=3)
            if src is not None and self.component[src] >= 0:
                same = self.component[cells[:, 0], cells[:, 1]] == self.component[src]
                if (ok & same).any():
                    ok = ok & same
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
