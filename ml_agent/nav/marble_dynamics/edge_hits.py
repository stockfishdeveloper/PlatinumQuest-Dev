"""Edge hits (2026-10-20, operator request): gain height by bouncing off the lip of a ledge.

The engine's bounce (marble.cc velocityCancel) reflects the velocity component along the contact normal with the
restitution and adds a spin-friction kick along the contact tangent. On the LIP of a ledge the normal points from the
edge to the marble centre, so part of the forward/downward speed comes back upward while most of the forward speed is
kept (118 natural lip/edge hits in the 10-20 baseline pools: lip hits kept 94 percent of horizontal speed; the compiled
simulator reproduced the kick exactly in 114 of 118 from 0.64 s ahead, 111 of 118 from 2.6 s ahead).

Nothing here names a map or a trick location: lips come from the collision mesh (floor polygon edges that no other
floor continues), candidates are the lips of ledges ABOVE the marble on the way to a higher target, and the proposals
are ordinary direct-aim sequences whose jump is timed so the rising marble clips the lip, then drive on to the
target. The usual gates judge them.

v2 (same evening): v1 offered lips below the marble and fired on far targets that were barely higher (Sprawl: a gem
85 u away, 1.6 u up); those plans were plain jumps. Now only lips 0.4-2.2 u above the marble count, the target must be
a real climb (near, or rising at least 5 percent of its distance), and the jump reply is computed from the lip
distance, the current speed and the jump's rise time instead of a fixed list.
"""
import math

import numpy as np
from nav.marble_dynamics.search import SequenceProposal as _SequenceProposal

FLOOR_NZ = .75          # a floor face: normal z above this
WALL_NZ = .35           # v7: the other side of a lip must be steeper than this (a wall) or absent
LIP_DROP_U = .5         # v7: and the floor 0.4 u beyond the edge must be at least this much lower (or absent)
LIP_MIN_LEN_U = .5      # ignore shorter lip segments
RISE_MIN_U = 1.         # the family is offered when the target is at least this much higher than the marble
TARGET_NEAR_U = 35.     # ... and either this near horizontally
TARGET_GRADE = .05      # ... or rising at least this fraction of its horizontal distance
AHEAD_MIN_U = 1.5       # lip point at least this far ahead horizontally
AHEAD_MAX_U = 16.       # ... and at most this far
LATERAL_MAX_U = 4.      # lip point at most this far off the straight line to the target
ABOVE_MIN_U = .4        # lip of a ledge above the marble: at least this much higher ...
ABOVE_MAX_U = 4.5       # v10 wall hits: hitting the wall EARLY in a jump adds the spin-friction kick (~6 u/s) to the rising speed:
                        # (7.5+6)^2/40 = 4.5 u (the operator's demo: 3-5 u climbs with a jump, +5 u/s kicks without)
MAX_LIPS = 2            # 2026-10-21: two lips x three timings x three lines = 18 forecasts at most
JUMP_IMPULSE = 7.5      # u/s, the marble datablock's jumpImpulse
GRAVITY = 20.
DEFAULT_SPEED = 10.     # assumed approach speed when the marble is slow now (the plan accelerates first)
DECISION_S = .064
WALL_EARLY_S = .10      # v10: jump this long before reaching the wall so the contact comes while still rising fast


def lips(triangles, normals, boundary_edges, probe=None):
    """Exposed floor lips: polygon-boundary edges of floor triangles whose other side is a wall or nothing.

    Returns an array (n,2,3) of segment endpoints and the (n,3) floor normals. An edge shared with another floor
    (a flat seam) or with a slope (normal z between WALL_NZ and FLOOR_NZ: a ramp, the Bowl's rim) is not a lip: v6
    spent 80 decisions on Bowl 7347019 jumping at slope seams (-6 points).
    """
    tri = np.asarray(triangles, dtype=float); nrm = np.asarray(normals, dtype=float)
    be = np.asarray(boundary_edges, dtype=bool)
    floor = nrm[:, 2] > FLOOR_NZ

    def key(p):
        return tuple(np.round(p, 3))
    # every polygon-boundary edge of every triangle -> the smallest |normal z| among its other sharers
    sharers = {}
    for i in range(len(tri)):
        for j in range(3):
            if not be[i, j]:
                continue
            k = frozenset((key(tri[i, j]), key(tri[i, (j + 1) % 3])))
            sharers.setdefault(k, []).append(i)
    segs = []; ns = []
    for i in np.nonzero(floor)[0]:
        for j in range(3):
            if not be[i, j]:
                continue
            a = tri[i, j]; b = tri[i, (j + 1) % 3]
            others = [o for o in sharers[frozenset((key(a), key(b)))] if o != i]
            if any(abs(nrm[o, 2]) >= WALL_NZ for o in others):
                continue          # a floor seam or a slope continues the surface
            if np.linalg.norm(b - a) < LIP_MIN_LEN_U:
                continue
            if probe is not None:
                c = tri[i, (j + 2) % 3]; mid = (a + b) / 2.; out = mid - c; out -= nrm[i] * float(out @ nrm[i])
                out /= max(float(np.linalg.norm(out)), 1e-9); p = mid + out * .4
                hit = probe(p + [0., 0., .3], p - [0., 0., 2.])
                if hit is not None and float(hit['point'][2]) > float(mid[2]) - LIP_DROP_U:
                    continue      # the floor continues beyond the edge: a seam, not a lip
            segs.append((a, b)); ns.append(nrm[i])
    return (np.asarray(segs, dtype=float).reshape(-1, 2, 3), np.asarray(ns, dtype=float).reshape(-1, 3))


def mesh_lips(engine):
    """Lips of the engine's collision mesh, computed once per mesh."""
    cache = getattr(engine, '_lips', None); mesh = engine.mesh
    if cache is not None and cache[0] is mesh:
        return cache[1]
    probe = (lambda a, b: mesh.raycast(a, b, cull_backfaces=True)) if hasattr(mesh, 'raycast') else None
    result = lips(mesh.triangles, mesh.normals, mesh.boundary_edges, probe=probe)
    engine._lips = (mesh, result)
    return result


def climb_needed(position, target):
    p = np.asarray(position, dtype=float); t = np.asarray(target[:3], dtype=float)
    rise = float(t[2] - p[2]); L = float(np.linalg.norm(t[:2] - p[:2]))
    return rise >= RISE_MIN_U and (L <= TARGET_NEAR_U or rise >= TARGET_GRADE * L)


def candidates(engine, position, target, max_lips=MAX_LIPS):
    """Lips of ledges above the marble on the way to a higher target: dicts(point, ahead, rise), nearest first."""
    p = np.asarray(position, dtype=float); t = np.asarray(target[:3], dtype=float)
    if not climb_needed(p, t):
        return []
    segs, _ = mesh_lips(engine)
    if not len(segs):
        return []
    d = t[:2] - p[:2]; L = float(np.linalg.norm(d))
    if L < 1e-6:
        return []
    u = d / L; left = np.array([-u[1], u[0]])
    a = segs[:, 0]; b = segs[:, 1]; ab = b - a; abh = ab[:, :2]
    abab = np.einsum('ij,ij->i', abh, abh)
    # the point of each segment closest to the approach line (horizontal plane): solve lateral offset = 0 along it
    lat_a = (a[:, :2] - p[:2]) @ left; lat_ab = abh @ left
    safe = np.where(np.abs(lat_ab) > 1e-9, lat_ab, 1.)
    s = np.where(np.abs(lat_ab) > 1e-9, np.clip(-lat_a / safe, 0., 1.), .5)
    pts = a + ab * s[:, None]
    ahead = (pts[:, :2] - p[:2]) @ u; lateral = np.abs((pts[:, :2] - p[:2]) @ left); rise = pts[:, 2] - p[2]
    along = np.abs(abh @ u) / np.sqrt(np.maximum(abab, 1e-12))
    ok = ((ahead >= AHEAD_MIN_U) & (ahead <= min(AHEAD_MAX_U, L + 2.)) & (lateral <= LATERAL_MAX_U)
          & (rise >= ABOVE_MIN_U) & (rise <= ABOVE_MAX_U) & (along < .8))
    idx = np.nonzero(ok)[0]
    if not len(idx):
        return []
    idx = idx[np.argsort(ahead[idx])]
    out = []
    for i in idx:
        q = pts[i]
        if any(np.linalg.norm(q - o['point']) < 1.5 for o in out):
            continue
        q = np.round(q * 4.) / 4.   # 0.25 u snap: the same lip gives the same goal key across decisions (route cache)
        out.append(dict(point=[float(x) for x in q], ahead=float(ahead[i]), rise=float(rise[i])))
        if len(out) >= max_lips:
            break
    return out


def rise_time(height):
    """Seconds after a jump until the marble has risen `height` u (None when the jump cannot reach it)."""
    disc = JUMP_IMPULSE * JUMP_IMPULSE - 2. * GRAVITY * height
    if disc < 0:
        return None
    return (JUMP_IMPULSE - math.sqrt(disc)) / GRAVITY


def jump_replies(ahead, speed, rise, radius=.19):
    """Jump replies for a lip: the marble should be at the lip's height when it arrives (and the neighbours).

    Above a jump's own reach two timings are offered (v10, from the engine's bounce code and the operator's demo):
    hit the wall EARLY in the jump, still rising fast, so the wall contact's spin-friction kick (bounceKineticFriction
    x contact friction x 5 x normal speed / 2r, ~6 u/s at 12 u/s) adds to the rising speed (4-5 u of height); and
    meet the wall near the apex (v5). The climb gate keeps whichever actually reaches the lip."""
    v = max(float(speed), 1e-6)
    v = v if v >= 4. else DEFAULT_SPEED
    t_lip = ahead / v
    t_up = rise_time(max(0., rise - radius * .5))
    out = set()
    if t_up is not None:
        j = int(round((t_lip - t_up) / DECISION_S)); out.update((max(0, j - 1), max(0, j), max(0, j + 1)))
    else:
        j = int(round((t_lip - JUMP_IMPULSE / GRAVITY) / DECISION_S)); out.update((max(0, j), max(0, j + 1)))
    if rise > 1.0:
        j = int(round((t_lip - WALL_EARLY_S) / DECISION_S)); out.update((max(0, j - 1), max(0, j)))
    return tuple(sorted(out))


WALL_TRIGGER_U = (0., 2.)      # low rebounds: no jump, or 2 u before the wall
WALL_CLIMB_TRIGGER_U = (2., 3.)  # climbs: jump 2-3 u before the wall (demo: 1.8 u at 16.8 u/s; sim: 2-3 u climbs, 1-1.5 u fails)
WALL_CLIMB_SPEEDS = (16., 20.)   # climbs need the fast run (the rebound kick scales with the speed into the wall)   # v11: jump when the lip point is this close horizontally (contact while rising fast)
WALL_FAST = 16.                    # v11: approach speed kept toward the wall (the demo hits at 12-16 u/s)


def wall_hit_proposal(sequence_proposal_class, lroute, lip, steps, through, trigger_u, fast=WALL_FAST):
    """A direct-aim sequence toward the lip point that jumps once, when the lip is `trigger_u` away horizontally
    (v11, from the operator's demo and the engine's bounce rule: the wall contact's spin-friction kick lands while
    the marble is still rising, whatever the approach speed). A subclass, so the compiled controller declines it and
    the per-reply forecast runs it. `sequence_proposal_class` is kept for the callers; the class is module-level
    (v22, 2026-10-21: the real-time server pickles candidates to its search workers; a local class crashed it)."""
    return WallHitProposal(lroute, lip, steps, through, trigger_u, fast)


class WallHitProposal(_SequenceProposal):
    def __init__(self, lroute, lip, steps, through, trigger_u, fast=WALL_FAST):
        super().__init__(lroute, np.asarray(lip, dtype=float), steps, jump=-1, direct=True, through=through, fast=fast)
        self.trigger_u = float(trigger_u); self.jumped = False; self.use = -1

    def __call__(self, state, index):
        if index == 0:
            self.jumped = False
        action = super().__call__(state, index)
        if not self.jumped and self.trigger_u > 0.:   # v13: trigger 0 = the plain no-jump reversal
            p = np.asarray(state['position'], dtype=float)
            if float(np.linalg.norm(self.gem[:2] - p[:2])) <= self.trigger_u:
                action = list(action); action[4] = 1; self.jumped = True
        return action


# ---- v12 wall rebounds (operator demo 2026-10-21, Concentric rows 29072 / 27452 / 32323): run INTO a wall at 13-16 u/s
# with the target behind or beside, jump a marble-length before contact, and the wall throws the marble back up and over
# onto the ledge: reversal at ~50-65 percent speed plus 2.5-3.5 u of height. Nothing map-specific: wall faces come from
# the mesh, the reflection of the approach must head toward the target, the compiled physics judges the rest.
WALL_NEAR_U = 1.0        # v17: wall point at least this far; was 2.5, which dropped every approach just before contact (wall3: 55 of 70 abandoned)
WALL_FAR_U = 16.         # ... and at most this far (2026-10-22 demo: the run-up to the x 118 step started 14 u out)
WALL_CLIMB_FACE_U = 1.2  # 2026-10-22: for a climb the face top must be at least this far above the marble centre (demo wall: 1.6)
WALL_CLIMB_RISE_U = 1.5  # 2026-10-22: a target at least this much higher is a CLIMB: the wall may lie beyond the target (run under it, rebound back up)
WALL_MAX_RISE_U = 5.5    # target at most this much higher (2026-10-22 demo: a 16 u/s run at a wall + jump right before it climbed 4.7 u: kick +5.4 u/s on top of the jump)
WALL_FAST_ONLY_RISE_U = 3.   # above this rise only the fast approach (16 u/s) is offered: the rebound kick scales with the speed into the wall
WALL_MIN_RISE_U = -.5    # ... and not much lower (a rebound down is a plain roll)
WALL_REFLECT_DEG = 70.   # the reflected approach must head within this of the target
WALL_TARGET_BEHIND_DEG = 50.  # only when the target is NOT ahead (a straight run would do)
WALL_MAX = 3             # 2026-10-22: up to three walls for a CLIMB (the demo's wall ranked third: the sim climbs off it, not off the two nearer faces); the search keeps one for low rebounds
WALL_FLIGHT_U = 5.       # v12b: the rebound lands about this far back from the wall point
WALL_MIN_SAVING_U = 0.   # v12b: and that landing must not lengthen the walking route (v13: was 8, which left only climbs)
WALL_SPEEDS = (11., 16.) # v13: approach speeds offered (the flight back scales with the speed in)


def wall_faces(engine):
    """Vertical faces of the collision mesh (|normal z| < WALL_NZ), once per mesh: (n,3,3) triangles and (n,3) normals."""
    cache = getattr(engine, '_wall_faces', None); mesh = engine.mesh
    if cache is not None and cache[0] is mesh:
        return cache[1]
    nrm = np.asarray(mesh.normals, dtype=float); tri = np.asarray(mesh.triangles, dtype=float)
    keep = np.abs(nrm[:, 2]) < WALL_NZ
    result = (tri[keep], nrm[keep], np.nonzero(keep)[0])
    engine._wall_faces = (mesh, result)
    return result


WALL_LAND_PROBES_U = (2.5, 5.)   # v15: flight distances probed for floor behind the rebound
WALL_LAND_DROP_U = 1.2           # v15: floor may be at most this far below the start level


def _landing_floor_z(mesh, q, r, z0):
    """Height of the floor under the rebound's landing estimate (the long flight), or None."""
    land = np.array([q[0] + r[0] * WALL_LAND_PROBES_U[-1], q[1] + r[1] * WALL_LAND_PROBES_U[-1], z0 + 8.])
    hit = mesh.raycast(land, land - [0., 0., 8. + WALL_LAND_DROP_U], cull_backfaces=True)
    return None if hit is None or hit['normal'][2] <= FLOOR_NZ else float(hit['point'][2])


def _landing_floor(mesh, q, r, z0):
    """True when a floor (upward normal) lies under the rebound's estimated landing for one of the probed flights."""
    for flight in WALL_LAND_PROBES_U:
        land = np.array([q[0] + r[0] * flight, q[1] + r[1] * flight, z0 + 4.])
        hit = mesh.raycast(land, land - [0., 0., 4. + WALL_LAND_DROP_U], cull_backfaces=True)
        if hit is not None and hit['normal'][2] > FLOOR_NZ:
            return True
    return False


WALL_MIN_APPROACH_U_S = 4.   # v19: the marble must already carry this much speed toward the wall (a wall redirects momentum; it is never a detour)


def wall_candidates(engine, position, target, max_walls=WALL_MAX, route=None, landing_route=None, velocity=None):
    """Wall points to run into so the rebound heads for a higher (or level) target that is not straight ahead."""
    p = np.asarray(position, dtype=float); t = np.asarray(target[:3], dtype=float)
    rise = float(t[2] - p[2])
    if not (WALL_MIN_RISE_U <= rise <= WALL_MAX_RISE_U):
        return []
    tri, nrm, idx = wall_faces(engine)
    if not len(tri):
        return []
    mesh = engine.mesh
    near = mesh.query_aabb(p - WALL_FAR_U, p + WALL_FAR_U)
    sel = np.isin(idx, near)
    if not sel.any():
        return []
    tri = tri[sel]; nrm = nrm[sel]; ids = idx[sel]
    cl = mesh.closest_points(p, ids)
    g = t[:2] - p[:2]; gL = float(np.linalg.norm(g))
    if gL < 1e-6:
        return []
    gu = g / gL
    out = []
    here = float(route.distance(p)) if route is not None else None
    order = np.argsort(np.linalg.norm(cl[:, :2] - p[:2], axis=1))
    for i in order:
        q = cl[i]; n = nrm[i]
        d = q[:2] - p[:2]; L = float(np.linalg.norm(d))
        if not (WALL_NEAR_U <= L <= WALL_FAR_U):
            continue
        zs = tri[i][:, 2]
        if float(zs.max()) < p[2] + .5:
            continue   # the face must RISE above the marble (a face below its floor is the edge of its own level: probe 1166 jumped off it)
        if rise >= WALL_CLIMB_RISE_U and float(zs.max()) < p[2] + WALL_CLIMB_FACE_U:
            continue   # 2026-10-22: a climb needs a wall, not a curb (the 0.6 u faces at x 106 ranked first and rebounded along the floor)
        q = np.array([q[0], q[1], float(p[2])])   # aim at the marble's own height on that face
        a = d / L
        if velocity is not None and float(np.dot(np.asarray(velocity[:2], dtype=float), a)) < WALL_MIN_APPROACH_U_S:
            continue   # v19: four pairs: 122 of 240 wall plans started moving AWAY from the wall (3.9 pts/10 s vs 5.1 toward)
        if float(np.dot(a, n[:2])) >= -.3:           # the face must face the marble
            continue
        if rise < WALL_CLIMB_RISE_U and float(np.dot(a, gu)) > math.cos(math.radians(WALL_TARGET_BEHIND_DEG)):
            continue                                 # level target ahead: a straight run does it (a climb may use the wall beyond it)
        nh = n[:2] / max(float(np.linalg.norm(n[:2])), 1e-6)
        r = a - 2. * float(np.dot(a, nh)) * nh       # horizontal reflection of the approach
        tg = t[:2] - q[:2]; tgL = float(np.linalg.norm(tg))
        if tgL < 1e-6:
            continue
        if float(np.dot(r, tg / tgL)) < math.cos(math.radians(WALL_REFLECT_DEG)):
            continue
        qq = np.round(q * 4.) / 4.
        if any(np.linalg.norm(qq - o['point']) < 2. for o in out):
            continue
        if not _landing_floor(mesh, q, r, p[2]):
            continue   # v15: nothing to land on behind the rebound (void): the forecast would only fall
        if here is not None and math.isfinite(here):
            # v12b: the rebound must land somewhere that shortens the walk to the target (Concentric probe: valid
            # rebounds landed on the ring 46 u from the gem, the far way round; the lower run was right to win)
            land = np.array([q[0] + r[0] * WALL_FLIGHT_U, q[1] + r[1] * WALL_FLIGHT_U, t[2]])
            rr = landing_route if landing_route is not None else route
            there = float(rr.distance(land))
            if not math.isfinite(there) or there > here - WALL_MIN_SAVING_U:
                continue
        # 2026-10-22 ranking: does the rebound land at the TARGET's height (the climb) and head for it? (demo: the right wall
        # was the 5th nearest; the four nearer faces rebounded along the floor)
        land_z = _landing_floor_z(mesh, q, r, p[2])
        height_match = 0. if land_z is None else max(0., 1. - abs(land_z - (t[2] - .2)) / 1.5)
        align = float(np.dot(r, tg / tgL))
        out.append(dict(point=[float(x) for x in qq], ahead=L, rise=rise, normal=[float(x) for x in n],
                        score=2. * height_match + align - .02 * L))
    out.sort(key=lambda w: -w['score'])
    return out[:max_walls]

