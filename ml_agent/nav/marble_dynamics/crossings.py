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
WALL_CLIMBS = False      # 2026-10-22 (parked 09:50, untested in a pool; see docs/POINTS_PLAN_2026-10-22.md): wall-climb crossings (run at a perpendicular face, jump just before it, rebound up onto the level); walls chosen along the way, the Blast reserved meanwhile (run at a perpendicular face, jump just before it, rebound up onto the level)
WALL_MIN_RISE_U = 1.5    # the upper level at least this far above the run-up floor ...
WALL_MAX_RISE_U = 5.5    # ... and at most this (demo: 4.0 u with a 2 u step face; the rebound kick scales with speed)
WALL_FACE_ABOVE_U = 1.2  # the face top at least this far above the run-up floor (a wall, not a curb)
WALL_LAND_U = 5.0        # the rebound lands about this far back from the face
WALL_LAND_TOL_U = 2.6    # an interior cell of the upper level within this of the landing estimate (the deck's interior ring sits 2 u in from its edge)
WALL_RUNUP_U = 14.0      # staging point this far back from the face (the demo's run started 14 u out)
WALL_LANE_HEADROOM_U = 0.8   # clearance above the run-up lane (under an overhanging deck; the marble is 0.4 u across)
WALL_RUN_SPEED_U_S = 12.0    # time estimate for the wall way: distance to the staging point + run-up at this speed ...
WALL_FLIGHT_S = 1.5          # ... plus the jump, rebound and landing
WALK_CLIMB_SPEED_U_S = 7.0   # the walking route to a higher goal (ramps, turns) at this speed; the wall way is staged when it is quicker


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
            return self._wall_crossing(source, target)   # 2026-10-22: a wall climb where no gap or Blast climb exists
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

    def _wall_faces(self):
        """Vertical mesh faces (|normal z| < 0.35) with their horizontal normals, z ranges and centroids, once."""
        if getattr(self, '_wall_face_cache', None) is None:
            mesh = self.mesh
            if mesh is None or not hasattr(mesh, 'triangles'):
                self._wall_face_cache = (np.zeros((0, 3, 3)), np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 3)))
            else:
                tri = np.asarray(mesh.triangles, dtype=float); nrm = np.asarray(mesh.normals, dtype=float)
                keep = np.abs(nrm[:, 2]) < 0.35
                tri = tri[keep]; nrm = nrm[keep]
                zr = np.column_stack((tri[:, :, 2].min(axis=1), tri[:, :, 2].max(axis=1)))
                self._wall_face_cache = (tri, nrm, zr, tri.mean(axis=1))
        return self._wall_face_cache

    def _floor_z(self, x, y, z_from, depth):
        hit = self.mesh.raycast(np.array([x, y, z_from], dtype=float), np.array([x, y, z_from - depth], dtype=float), cull_backfaces=True)
        if hit is None or float(hit['normal'][2]) < 0.75:
            return None
        return float(hit['point'][2])

    def _wall_crossing(self, source, target, zs=None, zt=None, position=None, velocity=None, goal_xy=None):
        """Wall-climb crossing from the run-up level `zs` of `source` onto the level `zt` of `target`, or None (the best
        face by run-up length). Levels default to the components' median heights; within ONE weak component (a deck
        joined to the floor by a ramp) the caller passes the marble's and the goal's heights."""
        if not WALL_CLIMBS or self.mesh is None:
            return None
        from scipy.spatial import cKDTree
        _, a_all = self._tree(source)
        _, ib_all = self._interior_tree(target)
        if len(a_all) == 0 or len(ib_all) == 0:
            return None
        zs = float(np.median(a_all[:, 2])) if zs is None else float(zs)
        zt = float(np.median(ib_all[:, 2])) if zt is None else float(zt)
        a = a_all[np.abs(a_all[:, 2] - zs) <= 0.8]
        if goal_xy is not None:
            # the upper LEVEL around the goal, whatever components it is split into (gems sit on pads that are
            # components of their own): every walk cell at the goal's height within 12 u of it
            g2 = np.asarray(goal_xy[:2], dtype=float)
            sel = (np.abs(self.points[:, 2] - zt) <= 0.8) & (np.linalg.norm(self.points[:, :2] - g2, axis=1) <= 12.0)
            ib = self.points[sel]
        else:
            ib = ib_all[np.abs(ib_all[:, 2] - zt) <= 0.8]
        if len(a) < MIN_COMPONENT_NODES or len(ib) < MIN_COMPONENT_NODES:
            return None
        itree = cKDTree(ib[:, :2]); stree = cKDTree(a[:, :2]); spts = a
        rise = zt - zs
        if not (WALL_MIN_RISE_U <= rise <= WALL_MAX_RISE_U):
            return None
        tri, nrm, zr, cen = self._wall_faces()
        if len(tri) == 0:
            return None
        # faces standing on the run-up level: bottom near the floor, top at least WALL_FACE_ABOVE_U above it
        ok = (zr[:, 0] <= zs + 0.6) & (zr[:, 1] >= zs + WALL_FACE_ABOVE_U)
        # ... within reach of the target component in plan (the landing is 5 u from the face)
        dists, _ = itree.query(cen[:, :2], k=1)
        ok &= dists <= WALL_LAND_U + 3.0
        best = None
        for i in np.nonzero(ok)[0]:
            n = nrm[i][:2]; L = float(np.linalg.norm(n))
            if L < 1e-6:
                continue
            n = n / L
            q = np.array([cen[i][0], cen[i][1], zs + 0.2])
            land = q[:2] + WALL_LAND_U * n
            dl, il = itree.query(land)
            if dl > WALL_LAND_TOL_U or abs(float(ib[il][2]) - zt) > 1.0:
                continue                              # the rebound must come down on the upper level
            # the run-up lane: walkable floor of the source component at the run-up level, with headroom
            lane_ok = True; launch = None; staging = None
            for sdist in (2.5, 5.0, 8.0, 11.0, WALL_RUNUP_U):
                pt = q[:2] + sdist * n
                d0, i0 = stree.query(pt)
                if d0 > 1.6 or abs(float(spts[i0][2]) - zs) > 0.8:
                    if sdist <= 8.0:
                        lane_ok = False
                    break
                if self.mesh.raycast(np.array([pt[0], pt[1], zs + 0.3]), np.array([pt[0], pt[1], zs + 0.3 + WALL_LANE_HEADROOM_U]), cull_backfaces=True) is not None:
                    lane_ok = False; break
                if sdist == 2.5:
                    launch = spts[i0].tolist()
                staging = spts[i0].tolist()
            if not lane_ok or launch is None or staging is None:
                continue
            runup = float(np.linalg.norm(np.asarray(staging[:2]) - q[:2]))
            if runup < 8.0:
                continue                              # too short a run for the speed the climb needs
            if position is not None:
                # 2026-10-22: the wall along the way: time to reach the staging point and run in, a second more when the
                # staging point lies behind the marble's direction of travel (the demo's wall was straight ahead).
                # Already inside the run-up lane (within 1.5 u of its line, 6-15 u from the face): the run starts here.
                pq = np.asarray(position[:2], dtype=float) - q[:2]
                along_in = float(pq @ n); lateral = abs(float(pq[0] * n[1] - pq[1] * n[0]))
                turn = 0.0
                if velocity is not None:
                    v = np.asarray(velocity[:2], dtype=float); sp = float(np.linalg.norm(v))
                    if sp > 3.0:
                        c = float(v @ (-n)) / sp            # the run-up heads INTO the face: -n
                        turn = 0.0 if c > 0.7 else (1.0 if c > -0.3 else 2.5)   # a 45+ degree turn costs a second, a reversal 2.5
                if lateral <= 1.5 and 6.0 <= along_in <= runup + 1.0:
                    score = along_in / WALL_RUN_SPEED_U_S + turn
                    staging = [float(position[0]), float(position[1]), float(zs + 0.2)]; runup = along_in
                else:
                    rel = np.asarray(staging[:2]) - np.asarray(position[:2], dtype=float); dist = float(np.linalg.norm(rel))
                    score = (dist + runup) / WALL_RUN_SPEED_U_S + turn
                    if velocity is not None and dist > 1e-6:
                        v = np.asarray(velocity[:2], dtype=float); sp = float(np.linalg.norm(v))
                        if sp > 3.0 and float(v @ rel) / (sp * dist) < 0.0:
                            score += 1.0
            else:
                score = -runup + 0.3 * float(dl)
            if best is None or score < best[0]:
                direction = (-n).tolist()             # the run-up heads INTO the face
                best = (score, dict(launch=launch, land=[float(land[0]), float(land[1]), float(ib[il][2])],
                                    land_goal=[float(land[0] + 1.5 * n[0]), float(land[1] + 1.5 * n[1]), float(ib[il][2])],
                                    gap_u=WALL_LAND_U, rise_u=rise, direction=direction, staging=staging,
                                    source=source, target=target, climb=True, needed_meter=0.0, wall=True,
                                    wall_point=[float(q[0]), float(q[1]), float(q[2])], runup_u=runup))
        return None if best is None else best[1]

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
            if math.hypot(rel[0], rel[1]) <= (active.get('runup_u', 0.0) + 1.0 if active.get('wall') else CROSS_RADIUS_U) + 1.0 and float(rel @ u) <= 1.0:
                if meter is not None and meter < active.get('needed_meter', 0.0):
                    s = active['staging']
                    self.cross_brakes = 0
                    return (s[0], s[1], s[2], value, extra), 'charge', active
                land = active.get('land_goal', active['land'])
                return (land[0], land[1], land[2], value, extra), 'cross', active
            self.active = None
        self.cross_brakes = 0
        here = self.component_at(position)
        there = self.component_at(goal[:3])
        if route is not None and self.reachable(route, position, goal):
            # 2026-10-22: reachable, but a higher goal behind a long detour may be quicker up a wall
            if not (WALL_CLIMBS and here is not None and there is not None
                    and float(goal[2]) - float(position[2]) >= WALL_MIN_RISE_U):
                return goal, None, None
            try:
                walk = float(route.distance(position))
            except Exception:
                walk = math.inf
            key = (here, there, round(float(position[2]), 0), round(float(goal[2]), 0), int(position[0] // 6), int(position[1] // 6))   # coarse position: the wall along the way
            cache = self.__dict__.setdefault('_wall_cache', {})
            if key not in cache:
                if len(cache) > 256:
                    cache.clear()
                cache[key] = self._wall_crossing(here, there, zs=float(position[2]) - 0.2, zt=float(goal[2]), position=position, velocity=velocity, goal_xy=goal[:2])
            wall = cache[key]
            if wall is None:
                return goal, None, None
            # time estimate: the wall way = to the staging point + the run-up + the flight, at run speed; the walk way
            # at the slower speed a climbing route manages (ramps, turns)
            to_staging = float(np.linalg.norm(np.asarray(wall['staging'][:2]) - np.asarray(position[:2], dtype=float)))
            t_wall = (to_staging + wall.get('runup_u', WALL_RUNUP_U)) / WALL_RUN_SPEED_U_S + WALL_FLIGHT_S
            t_walk = walk / WALK_CLIMB_SPEED_U_S if math.isfinite(walk) else math.inf
            if t_walk <= t_wall:
                return goal, None, None
            crossing = wall
        else:
            crossing = None
        if crossing is None:
            if here is None or there is None or here == there:
                return goal, None, None
            if WALL_CLIMBS and float(goal[2]) - float(position[2]) >= WALL_MIN_RISE_U:
                # 2026-10-22: no walking route up: a wall climb onto the goal's level, before the gap/Blast chain
                key = ('w', here, round(float(position[2]), 0), round(float(goal[2]), 0), int(position[0] // 6), int(position[1] // 6), int(goal[0] // 4), int(goal[1] // 4))
                cache = self.__dict__.setdefault('_wall_cache', {})
                if key not in cache:
                    if len(cache) > 256:
                        cache.clear()
                    cache[key] = self._wall_crossing(here, there, zs=float(position[2]) - 0.2, zt=float(goal[2]), position=position, velocity=velocity, goal_xy=goal[:2])
                crossing = cache[key]
            if crossing is None:
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
        near = math.hypot(rel[0], rel[1]) <= (crossing.get('runup_u', 0.0) + 1.0 if crossing.get('wall') else CROSS_RADIUS_U) and along <= 0.5   # a wall climb crosses from its staging point: the whole run-up is the approach
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
