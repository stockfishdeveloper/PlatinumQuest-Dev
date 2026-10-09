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
WALK_LEVELS = 4       # 2026-10-09 (Phase R): floor levels kept per walk cell (a deck over a floor, a ramp under a bridge)
LEVEL_MERGE = 0.5     # u: fine samples of one walk cell closer than this in height are the same level (a 1 u ramp cell
                      # spans up to ~0.9 u at the 42 deg climb limit)
STEP_UP = 1.5         # u: the most a neighbouring level may be ABOVE and still connect (a hop onto a box, a ramp's rise
                      # over one 1 u cell; the same JUMP_RISE the gap jumps land on)
DOWN_MAX = 6.0        # u: the most a neighbouring level may be BELOW and still connect (= JUMP_DROP: a drop the marble
                      # survives); the graph is directed, so a wall connects downward only
STEEP_SLOPE = 0.7     # rise/run (35 deg): a cell steeper than this that has a drop beside it is a SLIDE, not a route
                      # (2026-10-09 14:25: KOTM's 45 deg hole bevels became floor under the friction rule and the
                      # planner cut across them diagonally, 147.9 -> 141.2 and falls 3.7 -> 4.3 a round). Ramps are
                      # 20-32 deg; a steep slope away from any drop (Bowl's walls) stays walkable but costs more.
MAX_SLOPE = 0.6       # (superseded 2026-10-09) rise/run above which a cell was not walkable; which slopes are floor is
                      # now decided per surface by the builder (generate_terrain_map.climbable: engine traction x the
                      # texture's friction). Kept for the legacy tools that import it.
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
                k = self.level_at(j, i, gz)
                if k < 0 or abs(float(self.walk_z[k, j, i]) - gz) > 1.5:
                    c = self._nearest_walkable(j, i, radius=2, z=gz)
                    if c is None:
                        continue
                    k, j, i = c
                # Keep the gem's TRUE mission position as the goal. The cell (j, i) is only for
                # reachability and Dijkstra lookups. Snapping the goal to the walk-cell centre and
                # replacing z with the floor height put every waypoint 0.71 u horizontally and
                # 0.05 u vertically off the gem -- and with a 0.65 u pickup radius the marble
                # could sit exactly on its waypoint without touching the gem.
                self.gem_spawns.append((float(gx), float(gy), float(gz), k, j, i))

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
    # ------------------------------------------------------------------ walk grid (multi-level, 2026-10-09)
    # Phase R (real-map pool): a walk cell may hold several floor LEVELS (a deck above a floor, a ramp under a
    # bridge: Cragmire's tower, Treasure Box's rooms). Nodes are (level, j, i). Two neighbouring nodes connect when
    # the height change between them is one the marble can make: up to STEP_UP u up (a hop onto a box, a ramp's rise
    # over one cell) and down to DOWN_MAX u (a drop it survives); the graph is DIRECTED, so a 2 u wall connects
    # downward only. Which slopes are floor at all is decided per surface by the builder (generate_terrain_map
    # climbable(): the engine's traction model and the texture's friction), so every stored level is rollable.
    # Nothing here is map-specific.
    def _build_walk_grid(self):
        step = max(1, int(round(self.walk_res / self.res)))
        K, Hf, Wf = self.heights.shape
        pj = (-Hf) % step; pi = (-Wf) % step
        h = self.heights; sl = self.slope; mu = self.friction
        if pj or pi:
            h = np.pad(h, ((0, 0), (0, pj), (0, pi)), constant_values=np.nan)
            sl = np.pad(sl, ((0, 0), (0, pj), (0, pi)), constant_values=np.nan)
            mu = np.pad(mu, ((0, 0), (0, pj), (0, pi)), constant_values=np.nan)
        Hb, Wb = h.shape[1] // step, h.shape[2] // step
        # (Hb, Wb, K*step*step): every fine sample of the block, all levels
        hb = h.reshape(K, Hb, step, Wb, step).transpose(1, 3, 0, 2, 4).reshape(Hb, Wb, -1)
        sb = sl.reshape(K, Hb, step, Wb, step).transpose(1, 3, 0, 2, 4).reshape(Hb, Wb, -1)
        mb = mu.reshape(K, Hb, step, Wb, step).transpose(1, 3, 0, 2, 4).reshape(Hb, Wb, -1)
        self.wxs = self.xs[::step]
        self.wys = self.ys[::step]
        self.wH, self.wW = Hb, Wb
        KW = self.KW = WALK_LEVELS
        walk_z = np.full((KW, Hb, Wb), np.nan, dtype=np.float32)
        walk_slope = np.zeros((KW, Hb, Wb), dtype=np.float32)
        walk_mu = np.ones((KW, Hb, Wb), dtype=np.float32)
        any_floor = np.isfinite(hb).any(axis=2)
        for j, i in np.argwhere(any_floor):
            zs = hb[j, i]; ok = np.isfinite(zs)
            zs = zs[ok]; ss = sb[j, i][ok]; ms = mb[j, i][ok]
            order = np.argsort(-zs)
            zs, ss, ms = zs[order], ss[order], ms[order]
            # cluster samples within LEVEL_MERGE of each other (top first); a cluster is one level
            levels = []
            for z, sv, mv in zip(zs, ss, ms):
                if levels and levels[-1][0] - z <= LEVEL_MERGE:
                    levels[-1][3].append(z); levels[-1][1] = max(levels[-1][1], sv); levels[-1][2] = min(levels[-1][2], mv)
                else:
                    levels.append([z, sv, mv, [z]])
            for k, lv in enumerate(levels[:KW]):
                # the level's height is the MEDIAN of its samples (a ramp cell reads as its middle)
                walk_z[k, j, i] = float(np.median(lv[3])); walk_slope[k, j, i] = lv[1]; walk_mu[k, j, i] = lv[2]
        self.walk_z = walk_z; self.walk_slope = walk_slope; self.walk_mu = walk_mu
        self.walkable = np.isfinite(walk_z)                     # (KW, H, W)
        self.walk_top = walk_z[0]                               # legacy 2-D view: the top level of every cell
        # support: for each node and each of the 8 directions, is there a neighbour level the marble can step/roll to
        # (dz in [-DROP_EDGE, +STEP_UP])? A direction without one is a lip (a drop > DROP_EDGE, or nothing).
        edge = np.zeros_like(self.walkable)
        padz = np.pad(walk_z, ((0, 0), (1, 1), (1, 1)), constant_values=np.nan)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = padz[:, 1 + dy:1 + dy + Hb, 1 + dx:1 + dx + Wb]          # (KW, H, W) neighbour levels
                supported = np.zeros_like(self.walkable)
                for ka in range(KW):
                    za = walk_z[ka]
                    dz = nb - za[None]                                       # (KW, H, W): each neighbour level vs level ka
                    supported[ka] = np.isfinite(za) & np.any(np.isfinite(dz) & (dz >= -DROP_EDGE) & (dz <= STEP_UP), axis=0)
                edge |= self.walkable & ~supported
        # steep cells beside a lip are slides into the drop: not walkable at all (see STEEP_SLOPE)
        slide = self.walkable & (walk_slope > STEEP_SLOPE) & edge
        if slide.any():
            self.walkable &= ~slide
            walk_z[slide] = np.nan
            self.walk_z = walk_z
            edge &= self.walkable
        self.edge = edge
        from scipy.ndimage import distance_transform_edt
        self.edge_dist = np.zeros((KW, Hb, Wb), dtype=np.float32)
        inner = self.walkable.copy()
        padw = np.pad(self.walkable, ((0, 0), (1, 1), (1, 1)), constant_values=False)
        for k in range(KW):
            lip = self.edge[k] | ~self.walkable[k]
            self.edge_dist[k] = (distance_transform_edt(~lip) * self.walk_res).astype(np.float32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    inner[k] &= padw[k, 1 + dy:1 + dy + Hb, 1 + dx:1 + dx + Wb]
        # interior: walkable, not at a lip, and every 8-neighbour cell has floor at this level too (starts and goals
        # go here, so a marble is never placed on a bevel or half a cell from the void)
        self.interior = inner & ~self.edge
        self._graph = None

    def cell_of(self, x, y):
        i = int(round((x - self.wxs[0]) / self.walk_res))
        j = int(round((y - self.wys[0]) / self.walk_res))
        return j, i

    def in_walk_grid(self, j, i):
        return 0 <= j < self.wH and 0 <= i < self.wW

    def level_at(self, j, i, z=None):
        """Level index at walk cell (j, i) nearest to z (the top level when z is None); -1 if the cell has no floor."""
        if not self.in_walk_grid(j, i):
            return -1
        zs = self.walk_z[:, j, i]
        if not np.isfinite(zs).any():
            return -1
        if z is None:
            return 0
        d = np.abs(np.where(np.isfinite(zs), zs, 1e9) - float(z))
        return int(np.argmin(d))

    def _measured_jump_edges(self):
        """[(k, j, i, kk, jj, ii, gap_u), ...]: takeoff node, landing node, gap length. Marched on the FINE height stack
        from every lip node in JUMP_HEADINGS headings: 'floor' at a sample means a level within [-JUMP_DROP, +JUMP_RISE]
        of the takeoff height (a deck far below is void for this purpose), the void must begin within 1.5 u, and the
        first floor after it within physics.MAX_JUMP_GAP u is the landing."""
        from nav.physics import MAX_JUMP_GAP, MIN_JUMP_GAP
        Hs, Ws = self.heights.shape[1], self.heights.shape[2]
        heads = [(math.cos(2 * math.pi * k / JUMP_HEADINGS), math.sin(2 * math.pi * k / JUMP_HEADINGS)) for k in range(JUMP_HEADINGS)]
        step = self.res
        n_steps = int(math.ceil((MAX_JUMP_GAP + 1.5) / step))
        out = []
        for (k, j, i) in np.argwhere(self.edge):
            x = float(self.wxs[i]); y = float(self.wys[j]); z0 = float(self.walk_z[k, j, i])
            for cx, cy in heads:
                in_void = False; d_void = 0.0
                for sidx in range(1, n_steps + 1):
                    d = sidx * step
                    ii = int(round((x + cx * d - self.x0) / step)); jj = int(round((y + cy * d - self.y0) / step))
                    if not (0 <= ii < Ws and 0 <= jj < Hs):
                        break
                    zs = self.heights[:, jj, ii]; zs = zs[np.isfinite(zs)]
                    floor_here = len(zs) > 0 and bool(np.any((zs - z0 >= -JUMP_DROP) & (zs - z0 <= JUMP_RISE)))
                    if not floor_here:
                        if not in_void:
                            in_void = True; d_void = d
                        continue
                    if not in_void:
                        if d > 1.5:
                            break                  # floor continues: this heading has no lip here
                        continue
                    gap = d - d_void              # void samples span [d_void, d): their extent is d - d_void
                    if gap > MAX_JUMP_GAP or gap < MIN_JUMP_GAP:
                        break
                    lj, li = self.cell_of(x + cx * (d + 0.5), y + cy * (d + 0.5))
                    lz = float(zs[np.argmin(np.abs(zs - z0))])
                    lk = self.level_at(lj, li, lz)
                    if lk >= 0 and abs(float(self.walk_z[lk, lj, li]) - lz) <= 1.0 and (lk, lj, li) != (k, j, i):
                        out.append((k, j, i, lk, lj, li, float(gap)))
                    break
        return out

    def _build_graph(self):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        KW, H, W = self.walkable.shape
        idx = np.full((KW, H, W), -1, dtype=np.int64)
        nodes = np.argwhere(self.walkable)                      # rows (k, j, i)
        n = len(nodes)
        idx[nodes[:, 0], nodes[:, 1], nodes[:, 2]] = np.arange(n)
        rows, cols, w = [], [], []
        za = self.walk_z[nodes[:, 0], nodes[:, 1], nodes[:, 2]]
        ea = self.edge[nodes[:, 0], nodes[:, 1], nodes[:, 2]]
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            bj = nodes[:, 1] + dy; bi = nodes[:, 2] + dx
            ok = (bj >= 0) & (bj < H) & (bi >= 0) & (bi < W)
            d = self.walk_res * math.hypot(dx, dy)
            for kb in range(KW):
                sel = np.where(ok)[0]
                if not len(sel):
                    continue
                ib = idx[kb, bj[sel], bi[sel]]
                zb = self.walk_z[kb, bj[sel], bi[sel]]
                dz = zb - za[sel]
                good = (ib >= 0) & np.isfinite(dz) & (dz >= -DOWN_MAX) & (dz <= STEP_UP)
                if not good.any():
                    continue
                a = sel[good]; b = ib[good]; dzg = np.abs(dz[good])
                lip = ea[a] | self.edge[kb, bj[a], bi[a]]
                steep = np.maximum(self.walk_slope[nodes[a, 0], nodes[a, 1], nodes[a, 2]], self.walk_slope[kb, bj[a], bi[a]])
                cost = d * (1.0 + dzg / d) * (1.0 + steep) * np.where(lip, EDGE_COST, 1.0)   # slopes cost more than flat
                rows.append(a); cols.append(b); w.append(cost)
        rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        w = np.concatenate(w) if w else np.zeros(0, np.float64)
        # walk-only graph (directed): goal_field(..., jumps=False) measures the detour a jump saves; its UNDIRECTED
        # connected components are the "same island" test for goal sampling
        comp = np.full((KW, H, W), -1, dtype=np.int64)
        self._graph_nojump = None
        if n:
            g0 = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
            _, labels = connected_components(g0, directed=True, connection='weak')
            comp[nodes[:, 0], nodes[:, 1], nodes[:, 2]] = labels
            self._graph_nojump = g0
        self.component = comp
        # jump edges (measured rule, HANDOFF 28.27), directed takeoff -> landing, deduplicated (28.13)
        best = {}
        self.jump_edge_cells = []
        for (k, j, i, kk, jj, ii, gap) in self._measured_jump_edges():
            a, b = idx[k, j, i], idx[kk, jj, ii]
            if a < 0 or b < 0:
                continue
            c = gap * JUMP_COST
            if (a, b) not in best or c < best[(a, b)]:
                best[(a, b)] = c
                self.jump_edge_cells.append((k, j, i, kk, jj, ii, gap))
        self.jump_edges = len(best)
        if best:
            jr = np.array([k_[0] for k_ in best]); jc = np.array([k_[1] for k_ in best]); jw = np.array([best[k_] for k_ in best], dtype=np.float64)
            rows = np.concatenate([rows, jr]); cols = np.concatenate([cols, jc]); w = np.concatenate([w, jw])
        self._graph = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr() if n else None
        self._node_idx = idx
        self._nodes = nodes

    def goal_field(self, x, y, jumps=True, z=None):
        """Path distance from every node to the goal at world (x, y[, z]); inf where the goal cannot be reached from
        it. Returns a (KW, H, W) array. The goal snaps to the nearest walkable node within 3 u (the level nearest z,
        or the top level when z is None). `jumps=False` uses the walk-only graph."""
        from scipy.sparse.csgraph import dijkstra
        if self._graph is None:
            self._build_graph()
        j, i = self.cell_of(x, y)
        src = self._nearest_walkable(j, i, radius=3, z=z)
        field = np.full((self.KW, self.wH, self.wW), np.inf, dtype=np.float32)
        if src is None or self._graph is None:
            return field
        graph = self._graph if (jumps or self._graph_nojump is None) else self._graph_nojump
        # distance TO the goal along directed edges = distance FROM the goal on the transposed graph
        d = dijkstra(graph.T.tocsr(), directed=True, indices=[self._node_idx[src]])[0]
        field[self._nodes[:, 0], self._nodes[:, 1], self._nodes[:, 2]] = d
        return field

    def walkable_at(self, x, y, z=None):
        """True if the walk cell under world (x, y) has floor (at the level nearest z when given)."""
        j, i = self.cell_of(x, y)
        return self.level_at(j, i, z) >= 0

    def _nearest_walkable(self, j, i, radius=3, z=None):
        """Nearest node (k, jj, ii) to walk cell (j, i): by cell distance first, then by height difference from z
        (the top level is preferred when z is None). None if nothing walkable within the radius."""
        best, bd = None, (1e9, 1e9)
        for dj in range(-radius, radius + 1):
            for di in range(-radius, radius + 1):
                jj, ii = j + dj, i + di
                if not self.in_walk_grid(jj, ii):
                    continue
                zs = self.walk_z[:, jj, ii]
                ks = np.where(np.isfinite(zs))[0]
                if not len(ks):
                    continue
                k = ks[0] if z is None else int(ks[np.argmin(np.abs(zs[ks] - float(z)))])
                key = (dj * dj + di * di, 0.0 if z is None else abs(float(zs[k]) - float(z)))
                if key < bd:
                    best, bd = (int(k), jj, ii), key
        return best

    def dist_at(self, field, x, y, goal_xy, z=None):
        """Path distance to the goal from world (x, y[, z]); falls back to straight-line + 5 u off the walkable grid."""
        j, i = self.cell_of(x, y)
        k = self.level_at(j, i, z)
        if k >= 0 and np.isfinite(field[k, j, i]):
            return float(field[k, j, i])
        c = self._nearest_walkable(j, i, radius=2, z=z)
        if c is not None and np.isfinite(field[c]):
            return float(field[c]) + self.walk_res
        return float(math.hypot(x - goal_xy[0], y - goal_xy[1])) + 5.0

    # ------------------------------------------------------------------ sampling
    def _interior_cells(self):
        return np.argwhere(self.interior)                      # rows (k, j, i)

    def _sample_spawn_goal(self, x, y, rng, dmin, dmax, field, z=None):
        """A reachable REAL gem spawn point between dmin and dmax of (x, y), or None.
        Widens to any reachable spawn point before giving up, because a map may have only a
        handful (KOTM 12, JumpOnly 3) and the distance band can exclude them all."""
        cand = []
        for (gx, gy, gz, k, j, i) in self.gem_spawns:
            if not np.isfinite(field[k, j, i]):
                continue
            d = math.hypot(gx - x, gy - y)
            if d < 1.0:
                continue                       # already standing on it
            cand.append((gx, gy, gz, float(field[k, j, i]), d, k, j, i))
        if not cand:
            return None
        band = [c for c in cand if dmin <= c[4] <= dmax]
        pick = band if band else cand
        # JUMP CURRICULUM (2026-09-22, HANDOFF 28.13): with probability JUMP_GOAL_P prefer a spawn whose
        # shortest route from here uses a jump edge and saves at least JUMP_SAVE_MIN u over walking.
        if JUMP_GOAL_P > 0.0 and rng.random() < JUMP_GOAL_P:
            nojump = self.goal_field(x, y, jumps=False, z=z)
            saves = [c for c in pick if nojump[c[5], c[6], c[7]] - c[3] >= JUMP_SAVE_MIN]
            if saves:
                pick = saves
        gx, gy, gz, path_len, _ = pick[rng.integers(len(pick))][:5]
        return (gx, gy, gz, path_len)

    def _goal_cells(self, rng):
        """Nodes a GOAL may occupy. Real gems spawn anywhere walkable, including hard against a
        drop, so with probability EDGE_GOAL_P draw from the whole walkable set rather than the
        interior-only set. Training on interior cells alone taught the policy to avoid exactly
        the places gems appear (see HANDOFF 4b)."""
        if EDGE_GOAL_P >= 1.0 or rng.random() < EDGE_GOAL_P:
            return np.argwhere(self.walkable)
        return np.argwhere(self.interior)

    def sample_start(self, rng):
        cells = self._interior_cells()
        k, j, i = cells[rng.integers(len(cells))]
        return float(self.wxs[i]), float(self.wys[j]), float(self.walk_z[k, j, i])

    def sample_goal(self, x, y, rng, dmin=8.0, dmax=40.0, tries=50, cross_gap_p=0.5, z=None):
        """A walkable node between dmin and dmax (straight line) from (x, y[, z]) that is reachable (finite path, jump
        edges included). With probability 1 - cross_gap_p the goal is restricted to the marble's own connected floor
        (no gap to cross), so half the segments train plain locomotion and half train gap crossing.
        Returns (gx, gy, gz, path_len) or None."""
        field = self.goal_field(x, y, z=z)          # distances from the marble to everything
        if GOALS_FROM_SPAWNS and getattr(self, 'gem_spawns', None):
            g = self._sample_spawn_goal(x, y, rng, dmin, dmax, field, z=z)
            if g is not None:
                return g
            # else fall through: no spawn point fits this distance band, use the grid
        cells = self._goal_cells(rng)
        cx = self.wxs[cells[:, 2]]; cy = self.wys[cells[:, 1]]
        d = np.hypot(cx - x, cy - y)
        fv = field[cells[:, 0], cells[:, 1], cells[:, 2]]
        ok = (d >= dmin) & (d <= dmax) & np.isfinite(fv)
        if rng.random() >= cross_gap_p:
            j0, i0 = self.cell_of(x, y)
            src = self._nearest_walkable(j0, i0, radius=3, z=z)
            if src is not None and self.component[src] >= 0:
                same = self.component[cells[:, 0], cells[:, 1], cells[:, 2]] == self.component[src]
                if (ok & same).any():
                    ok = ok & same
        if not ok.any():
            ok = (d >= 2.0) & np.isfinite(fv)
            if not ok.any():
                return None
        cand = cells[ok]
        k, j, i = cand[rng.integers(len(cand))]
        return float(self.wxs[i]), float(self.wys[j]), float(self.walk_z[k, j, i]), float(field[k, j, i])

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
