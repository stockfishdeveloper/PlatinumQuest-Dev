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
GAP_ROUTES = True        # 2026-10-25 (operator, Fractured overlay: "the marble never crosses east to west or north to south ... only at a
                         # specific pathway"): routes may cross gaps (gap_links: a running jump, or a jump + Blast, priced by time). When
                         # the quickest route to the goal crosses a gap, that gap is staged (run-up, line up, landing as the goal) instead
                         # of walking around to a bridge or to the one narrowest gap per pair of walking components (control --no-gap-routes)
GAP_SPEED_U_S = 11.0     # route cost units per second for both ways (physics.CRUISE_SPEED); the Blast recharge wait is added on top
GAP_MIN_SAVE_S = 1.5     # the gap way must be at least this much quicker than the walking way
GAP_STAGE_MIN_U = 3.0    # narrower gaps are left to the search (pad hops inside a cluster need no run-up) ...
GAP_FAR_U = 0.0          # ... unless the goal is this far away (0 = off; see gap_crossing)
GAP_BLAST_MIN_METER = 0.0    # a Blast gap is considered from any meter level (the recharge wait is priced); the meter must be live
GAP_RUNUP_MIN_U = 2.0    # run-up point between these distances behind the launch (gap_links.runup_needed of the speed the gap needs;
GAP_RUNUP_MAX_U = 8.0    # sim: a 10 u gap entered at 1-4 u/s braked at the edge for 2 s with no valid flight)
GAP_SPEED_SLACK_U_S = 0.5    # the cross phase starts at the needed speed less this ...
GAP_RUNUP_SLACK_U = 0.75     # ... or this close to the full run-up behind the launch
GAP_RUN_RANGE_U = 20.0       # with search.GAP_RUN the cross phase (run-and-jump plans, kept only when they land) starts anywhere
GAP_RUN_LATERAL_U = 2.0      # within this distance behind the launch and inside a cone around the crossing line (this half-width
GAP_RUN_CONE = 0.7           # at the launch, plus this much per unit behind it); elsewhere the marble goes to the run-up point
GAP_GIVE_UP_CALLS = 400      # a gap crossing staged for this many stage() calls in a row (~2 per decision) is dropped for a while
GAP_COMMIT_PAST_U = 1.5      # past the lip, an active gap crossing stays in its cross phase until this far beyond the far edge
GAP_SMALL_MAX_RISE_U = 0.5   # with search.GAP_RUN a gap narrower than GAP_STAGE_MIN_U is staged when it rises at most this much
ROUTE_CLIMBS = True          # 2026-10-25 (with geometry_route.CLIMB_EDGES): when the route to the goal goes up a climb edge within
ROUTE_CLIMB_U = 20.0         # this much route length, the climb skill stages a climb onto that edge's upper floor (a gem behind a ledge)


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
        self.gaps = None          # GAP_ROUTES: gap_links.GapLinks of this terrain, built on first use
        self.climb_kit = None     # climbs.ClimbSkill of this terrain, built on first use (self.climbs is the old opt-in climb flag)

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
        if getattr(self.terrain, 'climb_edges', None):
            field = self.walk_field(goal)   # the route field also climbs (geometry_route.CLIMB_EDGES): walking is the walk-only graph
        node = self.node_near(position)
        return node is not None and bool(np.isfinite(field[node]))

    def walk_field(self, goal):
        """Walk-only distance field to the goal (the walking graph without climb edges)."""
        return self.terrain.goal_field(float(goal[0]), float(goal[1]), jumps=False, z=float(goal[2]))

    def route_climb_edge(self, position, goal):
        """The upper node (x, y, z) of the first climb edge (terrain.climb_edges) on the route from the marble toward the goal within
        ROUTE_CLIMB_U of route length (the route field's downhill path, as routing.path_waypoint follows it), else None."""
        t = self.terrain
        edges = getattr(t, 'climb_edges', None)
        graph = getattr(t, '_graph', None)
        if not edges or graph is None:
            return None
        field = t.goal_field(float(goal[0]), float(goal[1]), z=float(goal[2]))
        j, i = t.cell_of(position[0], position[1])
        k = t.level_at(j, i, position[2])
        if k < 0 or not np.isfinite(field[k, j, i]):
            c = t._nearest_walkable(j, i, radius=3, z=position[2])
            if c is None or not np.isfinite(field[c]):
                return None
            k, j, i = c
        start = float(field[k, j, i])
        for _ in range(int(ROUTE_CLIMB_U * 3 / t.walk_res) + 8):
            node = int(t._node_idx[k, j, i])
            lo, hi = graph.indptr[node:node + 2]
            ids = graph.indices[lo:hi]
            if not len(ids):
                return None
            cells = t._nodes[ids]
            values = field[cells[:, 0], cells[:, 1], cells[:, 2]]
            q = int(np.argmin(graph.data[lo:hi] + values))
            if not np.isfinite(values[q]) or values[q] >= field[k, j, i]:
                return None
            nxt = int(ids[q])
            kk, jj, ii = map(int, cells[q])
            if (node, nxt) in edges:
                return (float(t.wxs[ii]), float(t.wys[jj]), float(t.walk_z[kk, jj, ii]))
            k, j, i = kk, jj, ii
            if start - float(field[k, j, i]) > ROUTE_CLIMB_U or field[k, j, i] <= 1e-5:
                return None
        return None

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
            if active.get('gap_route'):
                self.failed[('gap',) + tuple(active['edge'])] = self.decision + 470   # this gap edge, not the component pair
            else:
                self.failed[(active['source'], active['target'])] = self.decision + 470
            self.active = None
            self.cross_brakes = 0

    def _inside(self, component, land, direction, depth=2.5):
        _, pts = self._tree(component)
        want = np.asarray(land[:2]) + depth * np.asarray(direction)
        d = np.linalg.norm(pts[:, :2] - want, axis=1)
        o = int(np.argmin(d))
        return pts[o].tolist() if d[o] <= 2.0 else list(land)

    def _staging(self, component, launch, direction, runup=None):
        """A walkable cell of the same component about RUNUP_U (or `runup`) behind the launch cell."""
        _, pts = self._tree(component)
        want = np.asarray(launch[:2]) - (RUNUP_U if runup is None else float(runup)) * np.asarray(direction)
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

    def gap_links(self):
        """GAP_ROUTES: the terrain's gap edges (built once, ~0.3 s), or None when unavailable."""
        if self.gaps is None:
            try:
                from nav.marble_dynamics.gap_links import GapLinks
                self.gaps = GapLinks(self.terrain, clearance=getattr(self.terrain, 'ground_clearance', None))
            except Exception:
                self.gaps = False
        return self.gaps or None

    def climb_skill(self):
        """climbs.ClimbSkill of this terrain (built on first use), or None."""
        if getattr(self, 'climb_kit', None) is None:
            try:
                from nav.marble_dynamics.climbs import ClimbSkill
                self.climb_kit = ClimbSkill(self.terrain, self.mesh, self._wall_faces()) if self.mesh is not None else False
            except Exception:
                self.climb_kit = False
        return self.climb_kit or None

    def climb_crossing(self, position, velocity, goal, route, meter):
        """climbs.CLIMB_SKILL: the best climb up to a goal on a higher place (open jump + Blast, or wall hit + jump + Blast timed to
        the meter) when it beats walking round, as a crossing dict (climb2=True); else None."""
        from nav.marble_dynamics import climbs as _climbs
        if not _climbs.CLIMB_SKILL:
            return None
        skill = self.climb_skill()
        if skill is None:
            return None
        failed_until = self.failed.get(('climb',) + tuple(round(float(c), 1) for c in goal[:3]), -1)
        if failed_until > self.decision:
            return None
        # the cheap test first (no route work, no change to the shared route's state): a goal on the marble's own level, or one
        # out of the skill's range, is no climb
        p = np.asarray(position, dtype=float)
        zm = skill.floor_at(p[0], p[1], p[2] - _climbs.RADIUS)
        if (zm is None or float(goal[2]) - zm < _climbs.CLIMB_MIN_RISE_U
                or math.hypot(float(goal[0]) - p[0], float(goal[1]) - p[1]) > _climbs.CLIMB_RANGE_U):
            return None
        walk = math.inf
        if getattr(self.terrain, 'climb_edges', None):
            # the route field climbs too (geometry_route.CLIMB_EDGES): the walking time comes from the walk-only graph
            node = self.node_near(position)
            if node is not None:
                d = float(self.walk_field(goal)[node])
                walk = d if math.isfinite(d) else math.inf
        elif route is not None:
            import copy as _copy
            r2 = _copy.copy(route)   # a copy: the shared route's held waypoint must not change when no climb is staged
            try:
                if self.reachable(r2, position, goal):
                    walk = float(r2.distance(position))
            except Exception:
                walk = math.inf
        gk = tuple(round(float(c), 1) for c in goal[:3])
        prefer = self.__dict__.setdefault('_climb_choice', {}).get(gk)
        try:
            climb = skill.plan(position, velocity, goal, meter, walk, prefer=prefer)
        except Exception:
            return None
        if climb is not None:
            self._climb_choice[gk] = (climb['kind'], tuple(climb['launch'][:2]))   # sticky: see climbs.CLIMB_SWITCH_S
            if len(self._climb_choice) > 64:
                self._climb_choice.pop(next(iter(self._climb_choice)))
        return climb

    def route_climb(self, position, velocity, goal, route, meter):
        """ROUTE_CLIMBS: a climb (climb_crossing) onto the upper floor of the route's next climb edge, staged for the goal (goal0) with
        the edge's upper node as the climb's own goal; None when the route has no climb edge near or the climb skill declines it."""
        upper = self.route_climb_edge(position, goal)
        if upper is None:
            return None
        climb = self.climb_crossing(position, velocity, upper, route, meter)
        if climb is not None:
            climb['goal0'] = list(goal[:3]); climb['climb_goal'] = list(upper); climb['route_climb'] = True
        return climb

    def _climb_stage(self, climb, position, velocity, meter, goal, value, extra):
        """Phase of a climb: 'cross' in the run lane when the meter will be enough by the take-off (the climb plans then run in,
        jump, Blast and land up there), else 'approach' to the run start (the marble waits there for the meter)."""
        climb['goal'] = list(goal[:3]); climb.setdefault('goal0', list(goal[:3]))
        key = ('climb',) + tuple(round(float(c), 1) for c in goal[:3])
        self._climb_calls = self._climb_calls + 1 if getattr(self, '_climb_key', None) == key else 1
        self._climb_key = key
        if self._climb_calls > GAP_GIVE_UP_CALLS:
            self.failed[key] = self.decision + 470   # staged too long without getting up: leave it for a while
            self._climb_key = None
            return goal, None, None
        in_lane, enough = self.climb_skill().ready(climb, position, velocity, meter)
        if in_lane and enough:
            self.active = climb
            land = climb['land_goal']
            return (land[0], land[1], land[2], value, extra), 'cross', climb
        s = climb['staging']
        return (s[0], s[1], s[2], value, extra), 'approach', climb

    def gap_crossing(self, position, goal, route, meter):
        """GAP_ROUTES: a crossing dict for the first gap (GAP_STAGE_MIN_U or wider) on the quickest route to the goal when that
        route, with the Blast recharge wait, beats the walking route by GAP_MIN_SAVE_S; else None."""
        G = self.gap_links()
        if G is None:
            return None
        start = G.node_near(position)
        if start is None:
            return None
        blast = meter is not None and meter >= GAP_BLAST_MIN_METER
        cost, pred = G.route_from(start, blast=blast)
        target, rest = G.target_node(goal, cost)
        if target is None:
            return None
        # GAP_FAR_U small-gap staging on trips (every gap on the route) was tried: Fractured sim trips fell or stalled in the
        # cross phase of 1-2 u gaps (decision 643: fell at 4.3 s after 24 A2 decisions; 1182: 19.7 vs 14.7 s), so it stays 0
        far = GAP_FAR_U > 0. and math.hypot(float(goal[0]) - float(position[0]), float(goal[1]) - float(position[1])) >= GAP_FAR_U
        first = G.path_first_gap(cost, pred, start, target, min_gap=0.0 if far else GAP_STAGE_MIN_U)
        if first is None and self._gap_run_on():
            # with the run-and-jump plans a small gap on the way is staged too when it is level or a drop (sim Cragmire: after a
            # run over a 4 u gap the old narrowest-gap crossing of a 1 u gap braked at its edge for 6 s); raised pads stay with
            # the ordinary search
            first = G.path_first_gap(cost, pred, start, target, min_gap=0.0)
            if first is not None and float(first[0]['dz']) > GAP_SMALL_MAX_RISE_U:
                first = None
        if first is None:
            return None
        e, walked = first
        if self.failed.get(('gap', e['a'], e['b']), -1) > self.decision:
            return None
        from nav.marble_dynamics.gap_links import BLAST_CHARGE_S
        wait = max(0.0, (e['meter'] - float(meter or 0.0)) * BLAST_CHARGE_S - walked / GAP_SPEED_U_S) if e['meter'] > 0.0 else 0.0
        t_gap = (float(cost[target]) + rest) / GAP_SPEED_U_S + wait
        walk = math.inf
        if route is not None:
            try:
                # route.distance falls back to straight line + 5 u where walking cannot reach the goal: only a reachable
                # goal has a walking way to compare with (otherwise the alternative is a crossing anyway)
                if self.reachable(route, position, goal):
                    walk = float(route.distance(position))
            except Exception:
                walk = math.inf
        t_walk = walk / GAP_SPEED_U_S if math.isfinite(walk) else math.inf
        if not t_gap + GAP_MIN_SAVE_S <= t_walk:
            return None
        a, b = e['a'], e['b']
        launch = G.points[a].tolist(); land = G.points[b].tolist()
        d = np.asarray(land[:2]) - np.asarray(launch[:2]); n = float(np.linalg.norm(d))
        u = (d / n) if n > 1e-6 else np.asarray(e['dir'], dtype=float)
        src = int(self.components[a]); dst = int(self.components[b])
        # the run-up the gap needs (gap_links: the speed at which the flight, with the edge's Blast, clears it, rolled up to
        # from rest) on the edge's checked straight lane behind the launch: the staging point stands on that lane
        u = np.asarray(e['dir'], dtype=float)
        v_need = float(e.get('v_need', 0.0)); runup = float(e.get('runup', RUNUP_U))
        staging = [launch[0] - u[0] * runup, launch[1] - u[1] * runup, launch[2]]
        inside = self._inside(dst, land, u.tolist()) if dst >= 0 else list(land)
        return dict(launch=launch, land=land, land_goal=inside, gap_u=float(e['gap']), rise_u=float(e['dz']), direction=u.tolist(),
                    staging=staging, source=src, target=dst, climb=False, needed_meter=float(e['meter']),
                    blast=bool(e['meter'] > 0.0), gap_route=True, edge=(a, b), saves_s=round(t_walk - t_gap, 2), wait_s=round(wait, 2),
                    v_need=round(float(v_need), 2), runup_u=round(runup, 2))

    def _cross_radius(self, crossing):
        """How near the launch the cross phase may start: a wall climb's whole run-up, a gap's run-up point, else CROSS_RADIUS_U."""
        if crossing.get('wall'):
            return crossing.get('runup_u', 0.0) + 1.0
        if crossing.get('gap_route'):
            return max(CROSS_RADIUS_U, crossing.get('runup_u', 0.0) + 1.5)
        return CROSS_RADIUS_U

    def _gap_near(self, crossing, position, slack=0.0):
        """A gap route's cross phase may start here: with search.GAP_RUN, anywhere GAP_RUN_RANGE_U behind the launch inside the
        cone around the crossing line (the run-and-jump plans line up by themselves and only a landing one is kept); else within
        _cross_radius of the launch. `slack` widens the region (an active crossing is kept a little beyond where it starts)."""
        from nav.marble_dynamics import search as _S
        u = np.asarray(crossing['direction'], dtype=float)
        rel = np.asarray(position[:2], dtype=float) - np.asarray(crossing['launch'][:2], dtype=float)
        along = float(rel @ u)
        if not _S.GAP_RUN:
            return math.hypot(rel[0], rel[1]) <= self._cross_radius(crossing) + slack and along <= 0.5 + slack
        lateral = abs(float(rel[0] * u[1] - rel[1] * u[0]))
        return (along <= 0.5 + slack and -along <= GAP_RUN_RANGE_U + slack
                and lateral <= GAP_RUN_LATERAL_U + slack + GAP_RUN_CONE * max(0.0, -along))

    def _gap_tick(self, crossing, goal):
        """Book-keeping each time a gap crossing is returned: the gem beyond goes on the crossing (the run-and-jump plans follow
        its route once landed); True when the crossing has been staged GAP_GIVE_UP_CALLS times in a row (it is then dropped for
        a while and the caller stages afresh)."""
        crossing['goal'] = list(goal[:3])
        key = ('gap',) + tuple(crossing['edge'])
        self._gap_calls = self._gap_calls + 1 if getattr(self, '_gap_key', None) == key else 1
        self._gap_key = key
        if self._gap_calls <= GAP_GIVE_UP_CALLS:
            return False
        self.failed[key] = self.decision + 470
        self.active = None
        self._gap_key = None
        return True

    @staticmethod
    def _gap_run_on():
        from nav.marble_dynamics import search as _S
        return bool(_S.GAP_RUN)

    def gap_ready(self, crossing, position, velocity):
        """A gap crossing may start (or keep) its run: rolling on along the crossing, the marble can still reach the speed the gap
        needs by the launch (v^2 + 2 a d >= v_need^2, a = gap_links.ROLL_ACC); else it goes back to the run-up point. True at the
        run-up point (that is how it is placed) and all along a proper run-up."""
        from nav.marble_dynamics.gap_links import ROLL_ACC
        u = np.asarray(crossing['direction'], dtype=float)
        along = float((np.asarray(position[:2], dtype=float) - np.asarray(crossing['launch'][:2], dtype=float)) @ u)
        v_along = max(0.0, float(np.asarray(velocity[:2], dtype=float) @ u)) if velocity is not None else 0.0
        reachable = math.sqrt(v_along * v_along + 2.0 * ROLL_ACC * (max(0.0, -along) + GAP_RUNUP_SLACK_U))
        return reachable >= float(crossing.get('v_need', 0.0)) - GAP_SPEED_SLACK_U_S

    def first_hop(self, source, target):
        """First crossing on the shortest chain of jumpable gaps from source to target, or None."""
        key = (source, target, tuple(sorted(k for k, until in self.failed.items() if until > self.decision and k[0] != 'gap')))
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
        if active is not None and active.get('climb2'):
            # a staged climb (climbs.py): committed while the run is under way or the marble is in the air short of the place
            g0 = active.get('goal0')
            same_goal = g0 is not None and math.hypot(float(goal[0]) - g0[0], float(goal[1]) - g0[1]) <= 1.0
            p = np.asarray(position, dtype=float)
            skill = self.climb_skill()
            up = skill is not None and skill.up_there(active, p)
            airborne = skill is not None and skill.airborne(p)   # no floor under the marble (z above the run floor held on the deck: sim 2084)
            if same_goal and not up:
                in_region, enough = skill.ready(active, position, velocity, meter)
                if airborne or (in_region and enough):
                    land = active['land_goal']
                    return (land[0], land[1], land[2], value, extra), 'cross', active
            self.active = None
            active = None
        if active is not None:
            u = np.asarray(active['direction'])
            rel = np.asarray(position[:2]) - np.asarray(active['launch'][:2])
            if active.get('gap_route'):
                along = float(rel @ u)
                lip = active.get('lip_along')
                edge = float((np.asarray(active['land'][:2]) - np.asarray(active['launch'][:2])) @ u)
                if lip is not None and float(lip) - 0.3 < along <= edge + GAP_COMMIT_PAST_U:
                    # past the lip: the run is committed until the marble is over the far side (re-staging from mid-air sent
                    # the search back toward the run-up point: nothing passed and the actor took the marble down, sim 1730)
                    land = active.get('land_goal', active['land'])
                    return (land[0], land[1], land[2], value, extra), 'cross', active
                # kept for the gem it was staged for only (the cross region is wide: another gem may need no crossing)
                g0 = active.get('goal0')
                keep = (along <= 1.0 and self._gap_near(active, position, slack=1.0)
                        and g0 is not None and math.hypot(float(goal[0]) - g0[0], float(goal[1]) - g0[1]) <= 1.0)
                if keep and self._gap_tick(active, goal):
                    return self.stage(position, goal, velocity=velocity, meter=meter, route=route)
            else:
                keep = math.hypot(rel[0], rel[1]) <= self._cross_radius(active) + 1.0 and float(rel @ u) <= 1.0
            if keep:
                if meter is not None and meter < active.get('needed_meter', 0.0):
                    s = active['staging']
                    self.cross_brakes = 0
                    return (s[0], s[1], s[2], value, extra), 'charge', active
                if active.get('gap_route') and not self._gap_run_on() and not self.gap_ready(active, position, velocity):
                    s = active['staging']   # too slow for this gap and too close to the launch: back to the run-up point
                    return (s[0], s[1], s[2], value, extra), 'approach', active
                land = active.get('land_goal', active['land'])
                return (land[0], land[1], land[2], value, extra), 'cross', active
            self.active = None
        self.cross_brakes = 0
        here = self.component_at(position)
        there = self.component_at(goal[:3])
        climb = self.climb_crossing(position, velocity, goal, route, meter)
        if climb is None and ROUTE_CLIMBS:
            climb = self.route_climb(position, velocity, goal, route, meter)
        if climb is not None:
            return self._climb_stage(climb, position, velocity, meter, climb.get('climb_goal', goal), value, extra)
        gap = self.gap_crossing(position, goal, route, meter) if GAP_ROUTES else None
        if gap is None:
            self._gap_key = None
        if gap is not None:
            crossing = gap
        elif route is not None and self.reachable(route, position, goal):
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
        # only a real drop or a climb to a higher level justifies staging (a gap edge is a real gap by construction).
        if not crossing.get('climb') and not crossing.get('gap_route') and not self.line_has_hole(position, goal):
            return goal, None, None
        if crossing.get('gap_route'):
            if self._gap_tick(crossing, goal):
                return self.stage(position, goal, velocity=velocity, meter=meter, route=route)
            crossing.setdefault('goal0', list(goal[:3]))   # the gem this crossing is staged for (see the active block)
        launch = crossing['launch']
        u = np.asarray(crossing['direction'])
        rel = np.asarray(position[:2]) - np.asarray(launch[:2])
        along = float(rel @ u)                       # >0 means already past the launch cell
        if crossing.get('gap_route'):
            near = self._gap_near(crossing, position)
        else:
            near = math.hypot(rel[0], rel[1]) <= self._cross_radius(crossing) and along <= 0.5   # a wall climb crosses from its staging point: the whole run-up is the approach
        aligned = True
        if velocity is not None and not crossing.get('gap_route'):   # a gap route's run-and-jump plans line up by themselves
            v = np.asarray(velocity[:2], dtype=float)
            speed = float(np.linalg.norm(v))
            aligned = speed <= SLOW_U_S or float(v @ u) / max(speed, 1e-6) >= ALIGN_COS
        s = crossing['staging']
        if meter is not None and meter < crossing.get('needed_meter', 0.0):
            # Not enough Blast for the rise: go to (or hold at) the run-up point
            # and let the meter charge; the search must not spend it meanwhile.
            self.active = crossing if near else None
            return (s[0], s[1], s[2], value, extra), 'charge', crossing
        if crossing.get('gap_route') and near and not self._gap_run_on() and not self.gap_ready(crossing, position, velocity):
            # a gap needs its run-up: go back to the run-up point (keep the crossing so it is not recomputed meanwhile)
            self.active = crossing
            return (s[0], s[1], s[2], value, extra), 'approach', crossing
        if near and aligned:
            self.active = crossing
            land = crossing.get('land_goal', crossing['land'])
            return (land[0], land[1], land[2], value, extra), 'cross', crossing
        return (s[0], s[1], s[2], value, extra), 'approach', crossing
