"""Crossings between walking components: route to the narrowest gap, then jump it.

A gem on another walking component has no floor route. Instead of aiming
straight at it (and driving off the nearest edge), find a chain of crossings
between components (each a closest pair of walkable cells with a gap the
marble can jump), route to a staging point behind the launch cell so the run-up
points across the gap, and once lined up make the landing cell the goal so the
ordinary jump/blast candidates, continued to their landing, can cross. Nothing
here is map specific: components and gaps come from the walking graph and the
collision geometry, and the physics search still decides every control.
"""
import heapq
import math

import numpy as np

MAX_GAP_U = 14.0         # operator 10-14: every Fractured gap is crossable with jump + blast; the physics forecast decides
MAX_RISE_U = 6.5         # a jump (1.4 u) plus a full Blast rose 6-7 u in the 10-14 journals; the physics forecast decides
CLIMB_RISE_U = 1.5       # above this rise (STEP_UP) the crossing is a climb: it needs no hole and may need Blast charge
JUMP_RISE_U = 2.5        # rise a running jump clears on its own (10-14 journals: plain jumps rose 2-3 u); above it Blast is needed
BLAST_RISE_U = 4.5       # extra rise a full meter adds on top of that (jump + full Blast measured 6-7 u)
RUNUP_U = 5.0            # staging point this far behind the launch cell, along the crossing direction
CROSS_RADIUS_U = 6.5     # within this distance of the launch cell (and lined up) the landing cell is the goal
ALIGN_COS = 0.6          # velocity must point within ~53 degrees of the crossing direction, or be slow
SLOW_U_S = 2.0           # below this speed alignment is not required (a run-up starts from here)
MIN_COMPONENT_NODES = 3  # tiny slivers are only considered when they hold the goal


def needed_meter(rise_u):
    """Blast meter fraction a climb of `rise_u` needs (0 when a jump alone does it)."""
    return float(min(1.0, max(0.0, (rise_u - JUMP_RISE_U) / BLAST_RISE_U)))


class Crossings:
    def __init__(self, terrain, mesh=None, climbs=False):
        """climbs=False (default, the v7 behaviour): only gap crossings between
        walking components at about the same height, no Blast 'charge' phase.
        climbs=True (v8-v10, opt-in): also stage climbs onto higher levels with
        strong (directed) components; lost 10-20 points on Concentric/Cube in
        every pool of 2026-10-14 (HANDOFF log 20:02 and 21:05)."""
        self.terrain = terrain
        self.mesh = mesh
        self.climbs = bool(climbs)
        nodes = terrain._nodes
        self.points = np.column_stack((terrain.wxs[nodes[:, 2]], terrain.wys[nodes[:, 1]],
                                       terrain.walk_z[nodes[:, 0], nodes[:, 1], nodes[:, 2]])).astype(float)
        self.components = np.asarray(terrain.component[nodes[:, 0], nodes[:, 1], nodes[:, 2]])
        graph = getattr(terrain, '_graph', None)
        if self.climbs and graph is not None and len(nodes) > 0:
            # Directed reachability: the walking graph's drop edges are one way,
            # so weak components merge a pit with the ledge above it (the marble
            # can fall in but not walk out). Strong components keep them apart.
            from scipy.sparse.csgraph import connected_components
            _, labels = connected_components(graph, directed=True, connection='strong')
            self.components = np.asarray(labels)
            self.component_grid = np.full_like(terrain.component, -1)
            self.component_grid[nodes[:, 0], nodes[:, 1], nodes[:, 2]] = labels
        else:
            self.component_grid = terrain.component
        self.sizes = {int(c): int(n) for c, n in zip(*np.unique(self.components[self.components >= 0], return_counts=True))}
        self._trees = {}
        self._interior_trees = {}
        interior = getattr(terrain, 'interior', None)
        self.interior = (np.asarray(interior[nodes[:, 0], nodes[:, 1], nodes[:, 2]], dtype=bool)
                         if interior is not None else np.ones(len(nodes), dtype=bool))
        self._edges = {}
        self._paths = {}
        self.available = hasattr(terrain, 'component') and len(nodes) > 0
        self.failed = {}          # (source, target) -> decision count until which the crossing is skipped
        self.decision = 0
        self.cross_brakes = 0

    def component_at(self, position, radius=3):
        j, i = self.terrain.cell_of(position[0], position[1])
        # Under a ledge or against a wall the nearest walkable cell can be a few
        # cells away; widen the search before giving up.
        c = None
        for r in (radius, 2 * radius, 4 * radius):
            c = self.terrain._nearest_walkable(j, i, radius=r, z=position[2])
            if c is not None:
                break
        return None if c is None else int(self.component_grid[c])

    def node_near(self, position, radius=6, dz=1.5):
        """Nearest walk node at the marble's height (within dz), or None."""
        t = self.terrain
        j, i = t.cell_of(position[0], position[1])
        best, bd = None, None
        for dj in range(-radius, radius + 1):
            for di in range(-radius, radius + 1):
                jj, ii = j + dj, i + di
                if not t.in_walk_grid(jj, ii):
                    continue
                zs = t.walk_z[:, jj, ii]
                for k in np.where(np.isfinite(zs))[0]:
                    if abs(float(zs[k]) - float(position[2])) > dz:
                        continue
                    key = (dj * dj + di * di, abs(float(zs[k]) - float(position[2])))
                    if bd is None or key < bd:
                        best, bd = (int(k), jj, ii), key
        return best

    def reachable(self, route, position, goal):
        """True when the route's walking field reaches the goal from the marble's own level.

        Walking components say nothing about one-way paths (up a ramp whose only
        way back is a drop), so the directed goal field is the authority: a goal
        with a finite path distance needs no staged crossing.
        """
        route.navigate(position, goal[:3])
        field = getattr(route, 'field', None)
        if field is None:
            return False
        node = self.node_near(position)
        return node is not None and bool(np.isfinite(field[node]))

    def _tree(self, component):
        from scipy.spatial import cKDTree
        if component not in self._trees:
            mask = self.components == component
            self._trees[component] = (cKDTree(self.points[mask][:, :2]), self.points[mask])
        return self._trees[component]

    def _interior_tree(self, component):
        from scipy.spatial import cKDTree
        if component not in self._interior_trees:
            mask = (self.components == component) & self.interior
            if mask.sum() < MIN_COMPONENT_NODES:
                mask = self.components == component
            self._interior_trees[component] = (cKDTree(self.points[mask][:, :2]), self.points[mask])
        return self._interior_trees[component]

    def _crossing(self, source, target):
        """Closest jumpable (launch, land) pair from `source` to `target`, or None.

        A landing more than CLIMB_RISE_U above the launch is a climb: its landing
        must be an interior cell of the upper level at least 1 u away in plan, so
        the launch stands beside the ledge (never under an overhang that shares
        the cell) and the run-up direction points onto the top.
        """
        _, a = self._tree(source)
        tree, b = self._tree(target)
        d, idx = tree.query(a[:, :2])
        rise = b[idx][:, 2] - a[:, 2]
        ok = (d <= MAX_GAP_U) & (rise <= CLIMB_RISE_U)
        score = np.where(ok, d + 2.0 * np.maximum(0.0, rise) + 0.5 * np.maximum(0.0, -rise), np.inf)
        itree, ib = self._interior_tree(target)
        di, iidx = itree.query(a[:, :2])
        irise = ib[iidx][:, 2] - a[:, 2]
        iok = (di >= 1.0) & (di <= MAX_GAP_U) & (irise > CLIMB_RISE_U) & (irise <= MAX_RISE_U)
        if not self.climbs:
            iok[:] = False
        iscore = np.where(iok, di + 2.0 * irise, np.inf)
        if not ok.any() and not iok.any():
            return None
        o = int(np.argmin(score))
        oi = None
        for cand in np.argsort(iscore)[:40]:
            if not np.isfinite(iscore[cand]):
                break
            # A climb needs headroom above the launch cell: under an overhang the
            # marble hits the ceiling before it clears the ledge.
            if self._headroom(a[cand], float(ib[iidx[cand]][2]) + 1.0):
                oi = int(cand)
                break
        if oi is not None and iscore[oi] < score[o]:
            o, launch, land, gap, up = oi, a[oi].tolist(), ib[iidx[oi]].tolist(), float(di[oi]), float(irise[oi])
        elif np.isfinite(score[o]):
            launch, land, gap, up = a[o].tolist(), b[idx[o]].tolist(), float(d[o]), float(rise[o])
        else:
            return None
        direction = np.asarray(land[:2]) - np.asarray(launch[:2])
        norm = float(np.linalg.norm(direction))
        direction = (direction / norm).tolist() if norm > 1e-6 else [1.0, 0.0]
        staging = self._staging(source, launch, direction)
        # Aim past the far edge: a goal on the edge cell itself pulls the marble
        # back toward the drop after it has landed.
        inside = self._inside(target, land, direction)
        return dict(launch=launch, land=land, land_goal=inside, gap_u=gap, rise_u=up,
                    direction=direction, staging=staging, source=source, target=target,
                    climb=bool(self.climbs and up > CLIMB_RISE_U),
                    needed_meter=needed_meter(up) if self.climbs else 0.0)

    def _headroom(self, launch, top_z):
        if self.mesh is None:
            return True
        a = np.array([launch[0], launch[1], launch[2] + 0.3])
        b = np.array([launch[0], launch[1], top_z])
        return self.mesh.raycast(a, b, cull_backfaces=True) is None

    def braked(self):
        """The controller braked in a cross phase with no checked candidate.

        A few seconds of that means this crossing does not work from here (the
        forecast rejects every candidate): drop it for a while so the planner
        tries another crossing or another gem instead of waiting at the edge.
        """
        self.cross_brakes += 1
        active = getattr(self, 'active', None)
        if active is not None and self.cross_brakes >= 32:
            self.failed[(active['source'], active['target'])] = self.decision + 470
            self.active = None
            self.cross_brakes = 0

    def _inside(self, component, land, direction, depth=2.5):
        _, pts = self._tree(component)
        want = np.asarray(land[:2]) + depth * np.asarray(direction)
        d = np.linalg.norm(pts[:, :2] - want, axis=1)
        o = int(np.argmin(d))
        return pts[o].tolist() if d[o] <= 2.0 else list(land)

    def _staging(self, component, launch, direction):
        """A walkable cell of the same component about RUNUP_U behind the launch cell."""
        _, pts = self._tree(component)
        want = np.asarray(launch[:2]) - RUNUP_U * np.asarray(direction)
        d = np.linalg.norm(pts[:, :2] - want, axis=1)
        o = int(np.argmin(d))
        if d[o] > 2.5:
            return list(launch)
        return pts[o].tolist()

    def _edges_from(self, source, extra=()):
        if source not in self._edges:
            targets = [c for c, n in self.sizes.items() if n >= MIN_COMPONENT_NODES and c != source]
            self._edges[source] = {t: x for t in targets for x in [self._crossing(source, t)] if x is not None}
        edges = self._edges[source]
        checked = self._edges.setdefault(('checked', source), set())
        for t in extra:
            if t != source and t not in edges and t not in checked:
                checked.add(t)
                x = self._crossing(source, t)
                if x is not None:
                    edges[t] = x
        return edges

    def first_hop(self, source, target):
        """First crossing on the shortest chain of jumpable gaps from source to target, or None."""
        key = (source, target, tuple(sorted(k for k, until in self.failed.items() if until > self.decision)))
        if key in self._paths:
            return self._paths[key]
        best = {source: 0.0}
        first = {}
        heap = [(0.0, source)]
        result = None
        while heap:
            cost, c = heapq.heappop(heap)
            if c == target:
                result = first.get(c)
                break
            if cost > best.get(c, math.inf):
                continue
            for t, x in self._edges_from(c, extra=(target,)).items():
                if self.failed.get((c, t), -1) > self.decision:
                    continue
                nc = cost + x['gap_u'] + 1.0
                if nc < best.get(t, math.inf):
                    best[t] = nc
                    first[t] = first.get(c, x)
                    heapq.heappush(heap, (nc, t))
        self._paths[key] = result
        return result

    def line_has_hole(self, position, goal, step=0.5, depth=6.0):
        """True when the straight line to the goal passes over a point with no floor nearby.

        Walking components are also split by walls and steps; those are handled by
        the ordinary search. Only a real drop justifies a staged crossing.
        """
        mesh = getattr(self, 'mesh', None)
        if mesh is None:
            return True
        a = np.asarray(position[:3], dtype=float)
        b = np.asarray(goal[:3], dtype=float)
        d = b - a
        length = float(np.linalg.norm(d[:2]))
        if length < 1e-6:
            return False
        key = (tuple(np.round(a[:2] * 2)), tuple(np.round(b[:2] * 2)))
        cache = self.__dict__.setdefault('_hole_cache', {})
        if key in cache:
            return cache[key]
        n = max(2, int(length / step))
        hole = False
        for i in range(1, n):
            p = a + d * (i / n)
            hit = mesh.raycast(np.array([p[0], p[1], max(a[2], b[2]) + depth]), np.array([p[0], p[1], min(a[2], b[2]) - depth]), cull_backfaces=True)
            if hit is None:
                hole = True
                break
        if len(cache) > 4096:
            cache.clear()
        cache[key] = hole
        return hole

    def stage(self, position, goal, velocity=None, meter=None, route=None):
        """Return (goal to pursue now, phase, crossing).

        phase is None (same component or no jumpable chain), 'approach' (route to
        the staging point behind the launch cell), 'cross' (lined up: the landing
        cell is the goal and airborne candidates may finish their flight) or
        'charge' (a climb whose rise needs more Blast meter than `meter` holds:
        wait at the staging point and keep the meter).
        """
        if not self.available:
            return goal, None, None
        self.decision += 1
        value = goal[3] if len(goal) > 3 else 1.0
        extra = goal[4] if len(goal) > 4 else 0.0
        # Keep an active crossing while the marble is still on its launch side
        # and near the launch cell: edge/slope cells can belong to tiny walking
        # components, and recomputing there would drop a plan mid run-up.
        active = getattr(self, 'active', None)
        if active is not None:
            u = np.asarray(active['direction'])
            rel = np.asarray(position[:2]) - np.asarray(active['launch'][:2])
            if math.hypot(rel[0], rel[1]) <= CROSS_RADIUS_U + 1.0 and float(rel @ u) <= 1.0:
                if meter is not None and meter < active.get('needed_meter', 0.0):
                    s = active['staging']
                    self.cross_brakes = 0
                    return (s[0], s[1], s[2], value, extra), 'charge', active
                land = active.get('land_goal', active['land'])
                return (land[0], land[1], land[2], value, extra), 'cross', active
            self.active = None
        self.cross_brakes = 0
        if route is not None and self.reachable(route, position, goal):
            return goal, None, None
        here = self.component_at(position)
        there = self.component_at(goal[:3])
        if here is None or there is None or here == there:
            return goal, None, None
        crossing = self.first_hop(here, there)
        if crossing is None:
            return goal, None, None
        # Walls and steps inside one level are the ordinary search's business;
        # only a real drop or a climb to a higher level justifies staging.
        if not crossing.get('climb') and not self.line_has_hole(position, goal):
            return goal, None, None
        launch = crossing['launch']
        u = np.asarray(crossing['direction'])
        rel = np.asarray(position[:2]) - np.asarray(launch[:2])
        along = float(rel @ u)                       # >0 means already past the launch cell
        near = math.hypot(rel[0], rel[1]) <= CROSS_RADIUS_U and along <= 0.5
        aligned = True
        if velocity is not None:
            v = np.asarray(velocity[:2], dtype=float)
            speed = float(np.linalg.norm(v))
            aligned = speed <= SLOW_U_S or float(v @ u) / max(speed, 1e-6) >= ALIGN_COS
        s = crossing['staging']
        if meter is not None and meter < crossing.get('needed_meter', 0.0):
            # Not enough Blast for the rise: go to (or hold at) the run-up point
            # and let the meter charge; the search must not spend it meanwhile.
            self.active = crossing if near else None
            return (s[0], s[1], s[2], value, extra), 'charge', crossing
        if near and aligned:
            self.active = crossing
            land = crossing.get('land_goal', crossing['land'])
            return (land[0], land[1], land[2], value, extra), 'cross', crossing
        return (s[0], s[1], s[2], value, extra), 'approach', crossing
