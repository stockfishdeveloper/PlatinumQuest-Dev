"""Gap links: jumps and jump + Blast flights across gaps, as extra edges of the walking route graph (2026-10-25).

Operator (Fractured Islands overlay of 10 rounds): "the marble never crosses east to west or north to south ... it only
travels between islands at a specific pathway ... i promise you the marble can cross ... sometimes the blast might need a
little recharge first but even that is still much faster than going the long way around."

The planner's route graph (geometry_route.walking_terrain) is walk-only: every gap is a detour to a bridge, and walking
components meet only at the ONE narrowest gap per pair (crossings._crossing). Here every gap a running jump, or a jump plus
the Blast, can clear becomes a directed edge (launch lip -> landing), found by marching the fine height stack from every lip
node in GAP_HEADINGS headings and kept only when the flight arc is clear of the collision geometry. Reach comes from the
game's own constants (jump impulse, gravity, air acceleration, Blast power x sqrt(meter), 25 s recharge): at an 11 u/s run-up
a jump carries ~9.7 u on the level, a jump + minimum Blast ~16.8 u, a full Blast ~27 u. Edges are priced by time (the flight
plus the Blast's value); the meter a crossing needs is kept per edge so the caller can price the recharge wait. Nothing is map
specific: gaps come from each map's geometry, reach from the physics, and the search still decides every control.
"""
import heapq
import math
from collections import OrderedDict

import numpy as np

JUMP_VZ = 7.5            # u/s: the marble's jumpImpulse
GRAVITY = 20.            # u/s^2
AIR_ACC = 5.             # u/s^2: airAcceleration (the stick pushes the marble forward in flight)
BLAST_POWER = 10.        # u/s at a full meter (impulse = power x sqrt(meter), straight up; nav/marble_dynamics/blast.py)
BLAST_MIN = .2           # the meter needed to fire at all
BLAST_CHARGE_S = 25.     # s from empty to full
BLAST_AFTER_S = .128     # the Blast fires two replies after the jump (the search's climb family: highest flight)
RUNUP_SPEED = 11.        # u/s: the run-up speed assumed for reach (physics.CRUISE_SPEED); the marble rolls up to 15
LANDING_MARGIN = 1.      # u: the landing must clear the far lip by this much (physics.LANDING_MARGIN)
MIN_GAP_U = .5           # narrower is not a gap (physics.MIN_JUMP_GAP)
MAX_GAP_U = 20.          # u: widest gap searched (a full Blast carries ~27 u on the level; long flights land less precisely)
MAX_RISE_U = 3.          # landing at most this far above the launch ...
MAX_DROP_U = 6.          # ... or this far below (terrain.JUMP_DROP)
GAP_HEADINGS = 16        # headings marched from every lip node (the observation's ray headings)
LIP_VOID_U = 1.5         # the void must begin within this of the lip node
LAND_DEPTH_U = 2.0       # a landing: floor at the landing level this far along the heading from the far edge ...
LAND_HALF_U = 0.5        # ... and on lines this far to either side (from LAND_SIDE_SLACK_U past the edge: a slanted edge),
LAND_SIDE_SLACK_U = 0.5  # within LAND_DZ_U of the landing height (sim Fractured: a platform corner clipped by the line passed as
LAND_DZ_U = 0.6          # the landing of a 3.5 u gap; every run over it fell)
JUMP_COST = 1.2          # route cost per unit of gap (terrain.JUMP_COST: a hop priced near its flight time)
JUMP_EDGE_U = 1.         # extra cost of any jump edge (the hop's time; geometry_route.JUMP_EDGE_COST_U)
BLAST_EDGE_U = 10.       # extra cost of an edge that spends the Blast (its value; engine.blast_value_u)
ARC_SEGMENTS = 8         # the flight arc is checked against the geometry as this many straight segments
FIELD_CACHE = 48
GOAL_NEAR_U = 4.         # a goal walking and gaps cannot enter (a gem on a pad, cut off from the walking graph) is routed to the
                         # reachable node within this distance of it; the search makes the last hop


def airtime(vz, dz):
    """Seconds until a flight launched at vertical speed vz comes down to dz above the launch (None if it never gets there)."""
    disc = vz * vz - 2. * GRAVITY * dz
    if disc < 0.:
        return None
    return (vz + math.sqrt(disc)) / GRAVITY


def flight_time(dz, meter=0.):
    """Seconds in the air of a running jump (plus a Blast of `meter` fired BLAST_AFTER_S after it) landing dz above the launch
    (None if it cannot get that high)."""
    if meter > 0.:
        # rise to the Blast, then a second ballistic flight from there
        tb = BLAST_AFTER_S; zb = JUMP_VZ * tb - .5 * GRAVITY * tb * tb; vzb = JUMP_VZ - GRAVITY * tb + BLAST_POWER * math.sqrt(meter)
        t2 = airtime(vzb, dz - zb)
        return None if t2 is None else tb + t2
    return airtime(JUMP_VZ, dz)


def reach(dz, meter=0., v_h=RUNUP_SPEED):
    """Horizontal distance of that flight at run-up speed v_h (the stick adds AIR_ACC forward in flight)."""
    t = flight_time(dz, meter)
    if t is None:
        return 0.
    return v_h * t + .5 * AIR_ACC * t * t


ROLL_ACC = 14.           # u/s^2 the marble gains rolling (angular acceleration 75 x radius 0.19)
RUNUP_MAX_U = 8.         # the run-up lane behind a launch is searched this far (straight, on the launch's level, clear of geometry)
RUNUP_MIN_U = 1.5        # a launch with less clear run-up than this is not a launch
RUNUP_STEP_U = .5
SPEED_MAX = 13.          # u/s: the run-up speed counted on at most (the marble rolls up to 15; a straight lane rarely gives more)


def speed_needed(gap, dz, meter=0.):
    """Run-up speed (u/s) at which the flight clears `gap` + LANDING_MARGIN (inf if it cannot land that high)."""
    t = flight_time(dz, meter)
    if t is None or t <= 0.:
        return math.inf
    return max(0., (gap + LANDING_MARGIN - .5 * AIR_ACC * t * t) / t)


def runup_needed(speed):
    """Straight roll (u) to reach `speed` from rest, plus half a unit to line up."""
    return speed * speed / (2. * ROLL_ACC) + .5


def needed_meter(gap, dz, v_h=RUNUP_SPEED):
    """Blast meter a gap needs at run-up speed v_h: 0 for a plain jump, BLAST_MIN..1 with the Blast, None when out of reach."""
    want = gap + LANDING_MARGIN
    if reach(dz, 0., v_h) >= want:
        return 0.
    m = BLAST_MIN
    while m <= 1.0001:
        if reach(dz, m, v_h) >= want:
            return round(min(1., m), 3)
        m += .05
    return None


def flight_arc(launch, heading, meter, horizontal, dz):
    """Points of the flight from `launch` along `heading` over `horizontal` u (for the arc clearance check)."""
    vz0 = JUMP_VZ; pts = []
    # time to cover the horizontal distance with the run-up speed and forward air acceleration
    a, b, c = .5 * AIR_ACC, RUNUP_SPEED, -horizontal
    t_end = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    for s in range(ARC_SEGMENTS + 1):
        t = t_end * s / ARC_SEGMENTS
        d = RUNUP_SPEED * t + .5 * AIR_ACC * t * t
        if meter > 0. and t > BLAST_AFTER_S:
            tb = BLAST_AFTER_S; zb = vz0 * tb - .5 * GRAVITY * tb * tb; vzb = vz0 - GRAVITY * tb + BLAST_POWER * math.sqrt(meter)
            z = zb + vzb * (t - tb) - .5 * GRAVITY * (t - tb) ** 2
        else:
            z = vz0 * t - .5 * GRAVITY * t * t
        if s == ARC_SEGMENTS:
            z = max(z, dz)        # the last point stands on the landing floor
        pts.append(np.array([launch[0] + heading[0] * d, launch[1] + heading[1] * d, launch[2] + max(z, min(0., dz) if s else 0.)]))
    return pts


class GapLinks:
    """Gap edges of one walking terrain and the route fields of walking + gap edges."""

    def __init__(self, terrain, clearance=None):
        from scipy.sparse import coo_matrix
        t = terrain
        if t._graph is None:
            t._build_graph()
        self.terrain = t
        nodes = t._nodes
        self.n = len(nodes)
        self.points = np.column_stack((t.wxs[nodes[:, 2]], t.wys[nodes[:, 1]], t.walk_z[nodes[:, 0], nodes[:, 1], nodes[:, 2]])).astype(float)
        radius = getattr(clearance, 'radius', .18975)
        base = t._graph.tocsr()
        existing = set(zip(*base.nonzero()))
        self.edges = []                      # dict(a, b, gap, dz, meter, dir)
        self.edge_of = {}                    # (a, b) -> index into self.edges
        self.checked = self.blocked = 0
        rows, cols, data = [], [], []
        for (k, j, i, kk, jj, ii, gap, cx, cy) in self._candidates():
            a = int(t._node_idx[k, j, i]); b = int(t._node_idx[kk, jj, ii])
            if a < 0 or b < 0 or a == b or (a, b) in existing or (a, b) in self.edge_of:
                continue
            dz = float(self.points[b][2] - self.points[a][2])
            d2 = self.points[b][:2] - self.points[a][:2]; horizontal = float(np.hypot(*d2))
            if horizontal < 1e-6:
                continue
            u = d2 / horizontal
            # the straight run-up lane behind the launch decides the speed the marble can bring (sim: a run-up point picked
            # behind a wall left the marble bumping it for 12 s); the meter the gap needs is computed at that speed
            lane = self._runup_lane(a, u, clearance, radius)
            if lane < RUNUP_MIN_U:
                continue
            v_ach = min(SPEED_MAX, math.sqrt(2. * ROLL_ACC * max(0., lane - .5)))
            meter = needed_meter(gap, dz, v_ach)
            if meter is None:
                continue
            v_need = speed_needed(gap, dz, meter)
            runup = min(lane, max(RUNUP_MIN_U, runup_needed(v_need)))
            if clearance is not None:
                lp = self.points[a] + [0., 0., radius + .001]
                arc = flight_arc(lp, (float(u[0]), float(u[1])), meter, horizontal, dz)
                self.checked += 1
                if not all(clearance.segment(p, q) for p, q in zip(arc, arc[1:])):
                    self.blocked += 1
                    continue
            self.edge_of[(a, b)] = len(self.edges)
            self.edges.append(dict(a=a, b=b, gap=float(gap), dz=dz, meter=float(meter), dir=(float(u[0]), float(u[1])),
                                   lane=float(lane), runup=float(runup), v_need=float(v_need)))
            rows.append(a); cols.append(b)
            data.append(gap * JUMP_COST + (BLAST_EDGE_U if meter > 0. else JUMP_EDGE_U))
        extra = coo_matrix((data, (rows, cols)), shape=base.shape).tocsr()
        self.graph = (base + extra).tocsr()
        self.graph_t = self.graph.T.tocsr()
        # the same without the Blast edges (no Blast on this map, or the gem order's distances)
        jr = [n for n, e in enumerate(self.edges) if e['meter'] == 0.]
        jump_extra = coo_matrix(([data[n] for n in jr], ([rows[n] for n in jr], [cols[n] for n in jr])), shape=base.shape).tocsr()
        self.graph_jump = (base + jump_extra).tocsr()
        self.graph_jump_t = self.graph_jump.T.tocsr()
        self._fields = OrderedDict()

    def _candidates(self):
        """(k, j, i, kk, jj, ii, gap, cx, cy): lip node, landing node, gap length and heading, marched on the fine height stack
        (terrain._measured_jump_edges' rule, widened to MAX_GAP_U and MAX_RISE_U)."""
        t = self.terrain
        H = t.heights
        Hs, Ws = H.shape[1], H.shape[2]
        ang = 2 * np.pi * np.arange(GAP_HEADINGS) / GAP_HEADINGS
        heads = np.column_stack((np.cos(ang), np.sin(ang)))
        step = float(t.res); n_steps = int(math.ceil((MAX_GAP_U + LIP_VOID_U) / step))
        ds = step * np.arange(1, n_steps + 1)
        for (k, j, i) in np.argwhere(t.edge):
            if not np.isfinite(t.walk_z[k, j, i]):
                continue
            x = float(t.wxs[i]); y = float(t.wys[j]); z0 = float(t.walk_z[k, j, i])
            ii = np.rint((x + heads[:, :1] * ds - t.x0) / step).astype(int)        # (headings, steps)
            jj = np.rint((y + heads[:, 1:] * ds - t.y0) / step).astype(int)
            inb = (ii >= 0) & (ii < Ws) & (jj >= 0) & (jj < Hs)
            zs = H[:, np.clip(jj, 0, Hs - 1), np.clip(ii, 0, Ws - 1)]             # (levels, headings, steps)
            rel = zs - z0
            with np.errstate(invalid='ignore'):
                ok = np.isfinite(zs) & (rel >= -MAX_DROP_U) & (rel <= MAX_RISE_U)
            floor = ok.any(axis=0) & inb
            for h in range(GAP_HEADINGS):
                f = floor[h]
                void = np.flatnonzero(~f)
                if not len(void):
                    continue
                nv = int(void[0])
                if ds[nv] > LIP_VOID_U or not inb[h, nv]:
                    continue                      # floor continues past the lip, or the map ends there
                later = np.flatnonzero(f[nv:])
                if not len(later):
                    continue
                nf = nv + int(later[0])
                gap = float(ds[nf] - ds[nv])      # void samples span [d_void, d_floor)
                if gap > MAX_GAP_U or gap < MIN_GAP_U:
                    continue
                cx, cy = float(heads[h, 0]), float(heads[h, 1])
                dl = float(ds[nf]) + .5
                lj, li = t.cell_of(x + cx * dl, y + cy * dl)
                zl = zs[:, h, nf][ok[:, h, nf]]
                lz = float(zl[np.argmin(np.abs(zl - z0))])
                if not self._landing_ok(x, y, cx, cy, float(ds[nf]), lz):
                    continue
                lk = t.level_at(lj, li, lz)
                if lk >= 0 and abs(float(t.walk_z[lk, lj, li]) - lz) <= 1.0 and (lk, lj, li) != (k, j, i):
                    yield (int(k), int(j), int(i), int(lk), int(lj), int(li), gap, cx, cy)

    def _landing_ok(self, x, y, cx, cy, d0, lz):
        """Floor at height lz (within LAND_DZ_U) for LAND_DEPTH_U past the far edge at d0 on the heading (cx, cy) from (x, y),
        and LAND_HALF_U to either side from LAND_SIDE_SLACK_U past the edge (fine height stack)."""
        t = self.terrain; H = t.heights; Hs, Ws = H.shape[1], H.shape[2]; step = float(t.res)
        for lat, start in ((0., d0), (-LAND_HALF_U, d0 + LAND_SIDE_SLACK_U), (LAND_HALF_U, d0 + LAND_SIDE_SLACK_U)):
            s = start
            while s <= d0 + LAND_DEPTH_U + 1e-9:
                px = x + cx * s - cy * lat; py = y + cy * s + cx * lat
                ii = int(round((px - t.x0) / step)); jj = int(round((py - t.y0) / step))
                if not (0 <= ii < Ws and 0 <= jj < Hs):
                    return False
                zs = H[:, jj, ii]; zs = zs[np.isfinite(zs)]
                if not len(zs) or float(np.min(np.abs(zs - lz))) > LAND_DZ_U:
                    return False
                s += step
        return True

    def _runup_lane(self, a, u, clearance, radius):
        """Clear straight run-up (u) behind launch node `a` against direction u: floor at the launch's level every RUNUP_STEP_U
        (fine height stack, within 0.35 u) and a clear sweep over it, up to RUNUP_MAX_U."""
        t = self.terrain; H = t.heights; Hs, Ws = H.shape[1], H.shape[2]
        x0, y0, z0 = self.points[a]
        good = 0.; s = RUNUP_STEP_U
        while s <= RUNUP_MAX_U + 1e-9:
            x = x0 - u[0] * s; y = y0 - u[1] * s
            ii = int(round((x - t.x0) / t.res)); jj = int(round((y - t.y0) / t.res))
            if not (0 <= ii < Ws and 0 <= jj < Hs):
                break
            zs = H[:, jj, ii]; zs = zs[np.isfinite(zs)]
            if not len(zs) or float(np.min(np.abs(zs - z0))) > .35:
                break
            good = s; s += RUNUP_STEP_U
        if clearance is not None and good > 0.:
            top = np.array([x0, y0, z0 + radius + .001])
            while good >= RUNUP_MIN_U:
                start = top - [u[0] * good, u[1] * good, 0.]
                if clearance.segment(start, top):
                    break
                good = math.floor(good * .5 / RUNUP_STEP_U) * RUNUP_STEP_U
        return float(good)

    def node_near(self, position, radius=3, dz=1.5):
        """Walk node under the position at its height (within dz), or None."""
        t = self.terrain
        j, i = t.cell_of(position[0], position[1])
        c = t._nearest_walkable(j, i, radius=radius, z=position[2])
        if c is None or abs(float(t.walk_z[c]) - float(position[2])) > dz:
            return None
        return int(t._node_idx[c])

    def field(self, goal, blast=True):
        """Route cost from every node to the goal over walking + gap edges (inf where unreachable); blast=False leaves out the
        edges that need the Blast."""
        from scipy.sparse.csgraph import dijkstra
        t = self.terrain
        key = (round(float(goal[0]), 2), round(float(goal[1]), 2), round(float(goal[2]), 2), bool(blast))
        f = self._fields.get(key)
        if f is not None:
            self._fields.move_to_end(key)
            return f
        j, i = t.cell_of(goal[0], goal[1])
        c = t._nearest_walkable(j, i, radius=3, z=goal[2])
        if c is None:
            f = np.full(self.n, np.inf)
        else:
            f = dijkstra(self.graph_t if blast else self.graph_jump_t, directed=True, indices=[int(t._node_idx[c])])[0]
        self._fields[key] = f
        while len(self._fields) > FIELD_CACHE:
            self._fields.popitem(last=False)
        return f

    def first_gap(self, field, start, blast=True, min_gap=0.):
        """(edge, walk cost to its launch) of the first gap edge at least `min_gap` wide on the cheapest route from node `start`,
        or None (the route walks, or only narrower gaps lie on it)."""
        g = self.graph if blast else self.graph_jump; walked = 0.; u = int(start); seen = set()
        if not np.isfinite(field[u]):
            return None
        while field[u] > 1e-9 and u not in seen:
            seen.add(u)
            lo, hi = g.indptr[u], g.indptr[u + 1]
            if lo == hi:
                return None
            vs = g.indices[lo:hi]; ws = g.data[lo:hi]
            o = int(np.argmin(ws + field[vs])); v = int(vs[o])
            e = self.edge_of.get((u, v))
            if e is not None and self.edges[e]['gap'] >= min_gap:
                return self.edges[e], walked
            walked += float(ws[o]); u = v
        return None

    def route_from(self, start, blast=True):
        """(cost, predecessors) from node `start` to every node over walking + gap edges (one forward Dijkstra; the last start
        is kept, as several gems are staged from the same position in one decision)."""
        from scipy.sparse.csgraph import dijkstra
        key = (int(start), bool(blast))
        if getattr(self, '_from_key', None) != key:
            d, p = dijkstra(self.graph if blast else self.graph_jump, directed=True, indices=[int(start)], return_predecessors=True)
            self._from_key = key; self._from = (d[0], p[0])
        return self._from

    def target_node(self, goal, cost, near_u=GOAL_NEAR_U):
        """Node to route to for a goal: its own walk node when reachable, else (a gem on a pad that walking and gaps cannot enter:
        the search hops onto it) the reachable node near it with the least cost + JUMP_COST x the remaining straight distance."""
        t = self.terrain
        j, i = t.cell_of(goal[0], goal[1])
        c = t._nearest_walkable(j, i, radius=3, z=goal[2])
        if c is not None:
            n = int(t._node_idx[c])
            if np.isfinite(cost[n]):
                return n, 0.
        g = np.asarray(goal[:3], dtype=float)
        dxy = np.hypot(self.points[:, 0] - g[0], self.points[:, 1] - g[1])
        ok = (dxy <= near_u) & np.isfinite(cost) & (self.points[:, 2] <= g[2] + 1.) & (self.points[:, 2] >= g[2] - MAX_DROP_U)
        if not ok.any():
            return None, 0.
        score = np.where(ok, cost + JUMP_COST * dxy, np.inf)
        n = int(np.argmin(score))
        return n, float(JUMP_COST * dxy[n])

    def path_first_gap(self, cost, pred, start, target, min_gap=0.):
        """(edge, walk cost to its launch) of the first gap edge at least `min_gap` wide on the cheapest path start -> target, or None."""
        path = [int(target)]
        while path[-1] != int(start):
            p = int(pred[path[-1]])
            if p < 0:
                return None
            path.append(p)
            if len(path) > self.n:
                return None
        path.reverse()
        for u, v in zip(path, path[1:]):
            e = self.edge_of.get((u, v))
            if e is not None and self.edges[e]['gap'] >= min_gap:
                return self.edges[e], float(cost[u])
        return None

    def summary(self):
        m = np.array([e['meter'] for e in self.edges]) if self.edges else np.zeros(0)
        return dict(edges=len(self.edges), jump_edges=int((m == 0).sum()), blast_edges=int((m > 0).sum()),
                    arcs_checked=self.checked, arcs_blocked=self.blocked)
