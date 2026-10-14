"""Shared, height-aware navigation; gem selection and scoring stay outside this module."""
import math
import numpy as np

from nav.terrain import DROP_EDGE, STEP_UP, LEVEL_MERGE

LOOKAHEAD_U = 12.0
WAYPOINT_HOLD_U = 2.5
GEM_SNAP_U = 1.5


def goal_key(goal):
    return tuple(round(float(v), 3) for v in goal[:3])


def line_clear(terrain, ax, ay, bx, by, step=0.35, az=None, bz=None):
    """Follow connected floor heights, and require the destination's floor when given.

    A clear ground-level XY line to a gem on the deck above is not a shortcut.
    Floating gems resolve to their nearest floor, rather than requiring gem z == floor z.
    """
    n = max(int(math.ceil(math.hypot(bx - ax, by - ay) / step)), 1)
    z = az
    last = None
    clearance=getattr(terrain,'ground_clearance',None)
    floor_points=[] if clearance is not None else None
    for u in range(n + 1):
        x, y = ax + (bx - ax) * u / n, ay + (by - ay) * u / n
        j, i = terrain.cell_of(x, y)
        # Repeated samples in the same static walk cell cannot change its floor
        # after the first sample resolves z to that floor. Keep the exact sample
        # arithmetic/cell sequence and check every cell transition and endpoint.
        if last is not None and (j, i) == last[1:]:
            continue
        k = terrain.level_at(j, i, z)
        if k < 0:
            return False
        zz = float(terrain.walk_z[k, j, i])
        if z is not None and not (-DROP_EDGE <= zz - z <= STEP_UP):
            return False
        z = zz
        last = (k, j, i)
        if floor_points is not None:floor_points.append((x,y,zz))
    if bz is not None:
        k, j, i = last
        dest = terrain.level_at(j, i, bz)
        if dest < 0 or abs(float(terrain.walk_z[dest, j, i]) - z) > LEVEL_MERGE:
            return False
    if clearance is not None:
        if not floor_points or floor_points[-1][:2]!=(bx,by):floor_points.append((bx,by,z))
        return clearance(floor_points)
    return True


def snap_to_walkable(terrain, gx, gy, gz, radius=3):
    j, i = terrain.cell_of(gx, gy)
    k = terrain.level_at(j, i, gz)
    if k >= 0:
        return (gx, gy, gz), False
    c = terrain._nearest_walkable(j, i, radius=radius, z=gz)
    if c is None:
        return (gx, gy, gz), False
    kk, jj, ii = c
    return (float(terrain.wxs[ii]), float(terrain.wys[jj]),
            float(terrain.walk_z[kk, jj, ii])), True


def path_waypoint(terrain, mx, my, field, gem, lookahead=LOOKAHEAD_U, mz=None):
    """Follow actual directed graph edges, then string-pull within the same floor route."""
    j, i = terrain.cell_of(mx, my)
    k = terrain.level_at(j, i, mz)
    if k < 0 or not np.isfinite(field[k, j, i]):
        c = terrain._nearest_walkable(j, i, radius=3, z=mz)
        if c is None or not np.isfinite(field[c]):
            return tuple(gem[:3]), False
        k, j, i = c
    if line_clear(terrain, mx, my, gem[0], gem[1], az=mz, bz=gem[2]):
        return tuple(gem[:3]), False
    graph = terrain._graph
    if graph is None:
        return tuple(gem[:3]), False
    path = []
    travelled = 0.0
    for _ in range(int(lookahead * 3 / terrain.walk_res) + 8):
        node = int(terrain._node_idx[k, j, i])
        lo, hi = graph.indptr[node:node + 2]
        ids = graph.indices[lo:hi]
        if not len(ids):
            break
        cells = terrain._nodes[ids]
        values = field[cells[:, 0], cells[:, 1], cells[:, 2]]
        costs = graph.data[lo:hi] + values
        q = int(np.argmin(costs))
        if not np.isfinite(costs[q]) or values[q] >= field[k, j, i]:
            break
        kk, jj, ii = map(int, cells[q])
        travelled += math.hypot(ii - i, jj - j) * terrain.walk_res
        k, j, i = kk, jj, ii
        path.append((float(terrain.wxs[i]), float(terrain.wys[j]),
                     float(terrain.walk_z[k, j, i])))
        if field[k, j, i] <= 1e-5 or travelled >= lookahead * 3:
            break
    if not path:
        return tuple(gem[:3]), False
    pick = None
    for wp in path:
        if math.hypot(wp[0] - mx, wp[1] - my) > lookahead:
            break
        if line_clear(terrain, mx, my, wp[0], wp[1], az=mz, bz=wp[2]):
            pick = wp
    if pick is None:
        pick = path[0]  # a jump edge can deliberately point across a gap
    if (math.hypot(pick[0] - gem[0], pick[1] - gem[1]) <= GEM_SNAP_U
            and line_clear(terrain, pick[0], pick[1], gem[0], gem[1], az=pick[2], bz=gem[2])):
        return tuple(gem[:3]), False
    return pick, True


class RoutePlanner:
    def __init__(self, terrain):
        self.terrain = terrain
        self.reset()

    def reset(self):
        self.key = None
        self.field = None
        self.waypoint = None

    def navigate(self, pos, gem):
        key = goal_key(gem)
        if key != self.key:
            self.field = self.terrain.goal_field(gem[0], gem[1], z=gem[2])
            self.key, self.waypoint = key, None
        x, y, z = map(float, pos[:3])
        if line_clear(self.terrain, x, y, gem[0], gem[1], az=z, bz=gem[2]):
            self.waypoint = None
            return tuple(gem[:3]), False
        wp = self.waypoint
        if (wp is not None and math.hypot(wp[0] - x, wp[1] - y) > WAYPOINT_HOLD_U
                and line_clear(self.terrain, x, y, wp[0], wp[1], az=z, bz=wp[2])):
            return wp, True
        wp, routed = path_waypoint(self.terrain, x, y, self.field, gem, mz=z)
        self.waypoint = wp if routed else None
        return wp, routed

    def distance(self, pos):
        if self.field is None or self.key is None:
            return math.inf
        return self.terrain.dist_at(self.field, pos[0], pos[1], self.key[:2], z=pos[2])
