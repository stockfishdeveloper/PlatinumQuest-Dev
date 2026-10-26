"""Climb skill (operator 2026-10-25): getting up to a higher place.

When the target sits higher than a plain jump reaches:
  - with a Blast meter high enough for a jump + Blast to reach that height, go to a spot by the place's edge with nothing
    overhead, run in, jump + Blast and land up there ('open' climb);
  - with the meter too low, run at a wall whose rebound heads at the place, jump just before it and Blast: the wall's kick adds
    height (TreasureBox's 2 u pit wall: about 4 u of rise with no Blast at all), and start the run when the meter will be high
    enough by the time the marble reaches the wall ('wall' climb).
Whichever gets up there soonest, counting the meter's charge, is staged, unless walking round is quicker. Any map: the place from
the walking terrain, walls and overhead blockers from the collision mesh, heights from the engine's jump and Blast numbers and a
measured wall-hit table. The plans (ClimbProposal, climb_specs) are physics-checked: the search keeps only those whose forecast
lands up there (search.apply_crossing_landing).
"""
import math

import numpy as np

CLIMB_SKILL = True
CLIMB_MIX = True            # the run phase searches the climb plans AND the ordinary plans for the gem itself (a direct jump + Blast
                            # that collects it wins on its pickup; any other plan counts only once it stands up there). Sim Vortex 2834:
                            # climb plans alone overshot a block gem the direct jump + Blast takes at 1.3 s (+8 s)
CLIMB_MIN_RISE_U = 2.6     # a target this far above the marble's floor may be climbed to (lower platforms are the hop / edge-hit
                            # families' job: sim 2026-10-25, TreasureBox's 2-4 u cluster platforms staged as climbs cost 1.4 s an episode) ...
CLIMB_RANGE_U = 25.0        # ... when it lies within this (straight) of the marble
CLIMB_SAVE_S = 0.8          # a climb is staged when it gets up there this much sooner than walking round
CLIMB_SWITCH_S = 1.0        # the climb chosen before for the same target is kept unless another is this much quicker
RUN_REGION_EXTRA_U = 6.0    # a climb's run may be tried from this much farther out than its run start ...
RUN_REGION_LATERAL_U = 2.5  # ... within this of its line at the edge / wall ...
RUN_REGION_CONE = 0.3       # ... widening this much per unit back (the plans line up by themselves; the forecast decides)
READY_METER_SLACK = 0.1     # the climb plans are tried once the meter is within this of the estimate (the forecast decides)
CLIMB_WALK_SPEED_U_S = 7.0  # the walking way up (ramps, stairs, turns) at this speed
CLIMB_RUN_SPEED_U_S = 10.0  # time estimate of the roll to a run start
JUMP_VZ = 7.5; GRAVITY = 20.; BLAST_POWER = 10.; BLAST_MIN = .2; BLAST_CHARGE_S = 25.; RADIUS = .19
OPEN_DELAY_S = .064         # the open climb's Blast fires about this long after the jump
OPEN_CLEAR_U = .5           # ... and must clear the place's edge by this much
# Wall hit + jump (+ Blast): rise above the run floor reached by most runs, per meter (TreasureBox east pit wall, a 2 u face, 4-10 u
# runs at full speed, jump 0.1-1.5 u before contact, Blast 0-4 replies after; median peaks of 45-225 forecast runs a level)
WALL_RISE_TABLE = ((0.0, 4.1), (0.2, 5.3), (0.3, 5.7), (0.4, 6.2), (0.5, 6.7), (0.75, 7.8), (1.0, 8.4))
WALL_RISE_MARGIN_U = .6     # ... less this (a run the forecast finds, not the median one)
WALL_FACE_MIN_U = 1.2       # a wall: its face rises at least this far above the run floor
WALL_RUN_MIN_U = 6.0        # a wall's run-in lane at least this long ...
WALL_RUN_MAX_U = 11.0       # ... used up to this
WALL_REACH_U = (1.5, 8.0)   # the place must lie this far in front of the wall (the rebound's reach) ...
WALL_LATERAL_U = 2.5        # ... within this of the run line
WALL_NEAR_U = 14.0          # walls searched within this of the target
OPEN_SPOT_OUT_U = (1.5, 2.5)   # open launch spots this far out from the place's edge ...
OPEN_RUNIN_U = 2.5          # ... with a straight run-in this long behind them
HEADROOM_U = .8             # run-in lanes are clear this high above their floor (they may pass under a deck)
PLACE_RADIUS_U = 12.0       # the place around the target: walk cells this far ...
PLACE_CELL_U = 0.5          # ... sampled on a lattice this fine
EDGE_NORMAL_U = 1.25        # an edge's outward normal: from the place's cells this close to it
PLACE_ROLL_U_S = 6.0        # time estimate of the roll on the place from the landing to the target
PLACE_BELOW_U = 1.0         # ... from this far below the target's level (a block or pad a hop above a deck: the deck counts) ...
PLACE_ABOVE_U = .8          # ... to this far above it
PLACE_STEP_U = .6           # neighbouring cells of the place differ in height by at most this
LEVEL_TOL_U = .6            # a floor 'at' a height: within this of it
LEAD_S = .08                # jump presses lead the take-off point by this much travel time (a reply acts ~1.25 replies later)
OPEN_JUMP_U = (2.0, 2.8, 3.6)   # open climbs: jump this far before the place's edge ...
OPEN_BLAST_AFTER = (1, 2, 3)    # ... Blast this many replies after the jump press
WALL_TRIG_U = (0.3, 0.6, 1.0)   # wall climbs: jump this far before the face ...
WALL_BLAST_AFTER = (0, 1, 2)    # ... Blast this many replies after the press (and the no-Blast run)


def open_meter(rise):
    """Meter a jump + Blast (OPEN_DELAY_S after the jump) needs to rise `rise` + OPEN_CLEAR_U; None when a full meter is short."""
    t = OPEN_DELAY_S
    h = JUMP_VZ * t - .5 * GRAVITY * t * t
    vz = JUMP_VZ - GRAVITY * t
    need = max(0.0, rise + OPEN_CLEAR_U - h)
    dv = math.sqrt(2.0 * GRAVITY * need) - vz
    if dv <= 0.0:
        return BLAST_MIN
    m = (dv / BLAST_POWER) ** 2
    return None if m > 1.0 else max(BLAST_MIN, m)


def wall_meter(rise):
    """Least meter for a wall hit + jump (+ Blast) to rise `rise` (WALL_RISE_TABLE less WALL_RISE_MARGIN_U): 0 = no Blast needed;
    None when out of reach."""
    for m, r in WALL_RISE_TABLE:
        if r - WALL_RISE_MARGIN_U >= rise:
            return m
    return None


class ClimbSkill:
    """Climb options of one map (walking terrain + collision mesh); places and options are cached per target and floor level."""

    def __init__(self, walk, mesh, faces):
        self.walk = walk
        self.mesh = mesh
        self.faces = faces          # crossings.Crossings._wall_faces(): (triangles, normals, z ranges, centroids)
        self._places = {}
        self._options = {}

    # ------------------------------------------------------------------ geometry
    def floor_at(self, x, y, z, tol=LEVEL_TOL_U):
        """Height of the walk level at (x, y) within `tol` of z, else None."""
        w = self.walk
        j, i = w.cell_of(float(x), float(y))
        if not w.in_walk_grid(j, i):
            return None
        zs = w.walk_z[:, j, i]
        ok = np.isfinite(zs) & (np.abs(zs - float(z)) <= tol)
        if not ok.any():
            return None
        return float(zs[ok][np.argmin(np.abs(zs[ok] - float(z)))])

    def clear(self, a, b):
        return self.mesh.raycast(np.asarray(a, dtype=float), np.asarray(b, dtype=float), cull_backfaces=False) is None

    def place(self, target):
        """The place the target stands on: dict(z, pts (n, 3), edges [(point, outward normal)]) or None; cached."""
        key = (round(float(target[0]), 1), round(float(target[1]), 1), round(float(target[2]), 1))
        if key in self._places:
            return self._places[key]
        w = self.walk
        j0, i0 = w.cell_of(float(target[0]), float(target[1]))
        res = None
        if w.in_walk_grid(j0, i0):
            zs = w.walk_z[:, j0, i0]
            ok = np.isfinite(zs) & (zs <= float(target[2]) + 0.3) & (zs >= float(target[2]) - 1.5)
            if ok.any():
                k0 = int(np.flatnonzero(ok)[np.argmax(zs[ok])])
                z0 = float(zs[k0])
                lo, hi = z0 - PLACE_BELOW_U, z0 + PLACE_ABOVE_U
                rad = int(PLACE_RADIUS_U / w.walk_res)
                st = max(1, int(round(PLACE_CELL_U / w.walk_res)))   # a coarser lattice than the walk grid
                seen = {(k0, j0, i0)}
                stack = [(k0, j0, i0)]
                pts = []; edges = []
                while stack:
                    k, j, i = stack.pop()
                    z = float(w.walk_z[k, j, i])
                    pts.append((float(w.wxs[i]), float(w.wys[j]), z))
                    missing = np.zeros(2)
                    for dj in (-st, 0, st):
                        for di in (-st, 0, st):
                            if dj == 0 and di == 0:
                                continue
                            jj, ii = j + dj, i + di
                            found = False
                            if w.in_walk_grid(jj, ii) and abs(jj - j0) <= rad and abs(ii - i0) <= rad:
                                zz = w.walk_z[:, jj, ii]
                                for kk in np.flatnonzero(np.isfinite(zz)):
                                    zk = float(zz[kk])
                                    if lo <= zk <= hi and abs(zk - z) <= PLACE_STEP_U:
                                        found = True
                                        if (kk, jj, ii) not in seen:
                                            seen.add((kk, jj, ii)); stack.append((int(kk), jj, ii))
                                        break
                            if not found and (dj == 0 or di == 0):
                                missing += (np.sign(di), np.sign(dj))
                    n = float(np.hypot(missing[0], missing[1]))
                    if n > 0.0:
                        edges.append((np.array([float(w.wxs[i]), float(w.wys[j]), z]), missing / n))
                if len(pts) >= 8:
                    P = np.asarray(pts)
                    from scipy.spatial import cKDTree
                    tree = cKDTree(P[:, :2])
                    # edge points about 1 u apart, their outward normals smoothed over EDGE_NORMAL_U: the mean direction from the
                    # place's cells around the edge cell toward the edge cell (a corner or a block beside the edge no longer tilts it)
                    keep = []
                    for p, nrm in edges:
                        if all(float(np.hypot(*(p[:2] - q[:2]))) >= 1.0 for q, _ in keep):
                            near = P[tree.query_ball_point(p[:2], EDGE_NORMAL_U)]
                            out = p[:2] - near[:, :2].mean(axis=0)
                            L = float(np.linalg.norm(out))
                            keep.append((p, out / L if L > 1e-6 else nrm))
                    res = dict(z=z0, pts=P, edges=keep, tree=tree)
        self._places[key] = res
        if len(self._places) > 64:
            self._places.pop(next(iter(self._places)))
        return res

    def on_place(self, place, p):
        """True when position p (marble centre) is on the place (within half a cell of one of its cells, at its height)."""
        pts = place['pts']
        d, k = place['tree'].query(np.asarray(p[:2], dtype=float))
        return d <= max(0.5, PLACE_CELL_U) and float(p[2]) >= float(pts[k][2]) + RADIUS - 0.35

    def airborne(self, p):
        """True when no walk level lies within LEVEL_TOL_U of the marble's bottom (in the air, not standing on any floor)."""
        return self.floor_at(p[0], p[1], float(p[2]) - RADIUS) is None

    def up_there(self, climb, p):
        """True when position p is on the climb's place, or stands on a floor at the place's level (PLACE_BELOW_U) within
        PLACE_RADIUS_U of the target: the place's lattice can miss most of a deck (TreasureBox: 73-94 of its ~380 half-unit cells;
        sim 2084, a climb stayed staged for 5 s on the deck and the cluster plans were off meanwhile)."""
        place = climb.get('place')
        if place is not None and self.on_place(place, p):
            return True
        fz = self.floor_at(p[0], p[1], float(p[2]) - RADIUS)
        g = climb.get('goal') or climb.get('land_goal')
        return (fz is not None and g is not None and fz >= float(climb['z_place']) - PLACE_BELOW_U
                and math.hypot(float(p[0]) - g[0], float(p[1]) - g[1]) <= PLACE_RADIUS_U)

    def _lane(self, a, b, z, step=0.5):
        """Floor at height z (LEVEL_TOL_U) with HEADROOM_U clear above it every `step` from a to b."""
        a = np.asarray(a[:2], dtype=float); b = np.asarray(b[:2], dtype=float)
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step)))
        for s in range(n + 1):
            q = a + (b - a) * (s / n)
            fz = self.floor_at(q[0], q[1], z)
            if fz is None:
                return False
            if not self.clear((q[0], q[1], fz + 0.3), (q[0], q[1], fz + 0.3 + HEADROOM_U)):
                return False
        return True

    def options(self, target, zm):
        """Climb options from the floor level zm up to the target's place (cached per target and level)."""
        place = self.place(target)
        if place is None:
            return place, []
        key = (round(float(target[0]), 1), round(float(target[1]), 1), round(float(target[2]), 1), round(float(zm), 1))
        if key in self._options:
            return place, self._options[key]
        out = []
        zt = place['z']
        # open climbs: a spot by the place's edge, nothing overhead, a straight run-in behind it, a clear arc over the edge
        for e, n in place['edges']:
            for d in OPEN_SPOT_OUT_U:
                P = e[:2] + n * d
                fz = self.floor_at(P[0], P[1], zm)
                if fz is None or float(e[2]) - fz < CLIMB_MIN_RISE_U:
                    continue
                if not self.clear((P[0], P[1], fz + 0.3), (P[0], P[1], float(e[2]) + 2.0)):
                    continue
                S = P + n * OPEN_RUNIN_U
                if not self._lane(S, P, fz):
                    continue
                mid = P + (e[:2] - P) * 0.5
                inside = e[:2] - n * 1.0
                if not (self.clear((P[0], P[1], fz + 0.5), (mid[0], mid[1], float(e[2]) + 1.2))
                        and self.clear((mid[0], mid[1], float(e[2]) + 1.2), (inside[0], inside[1], float(e[2]) + 0.8))):
                    continue
                land = place['pts'][place['tree'].query(e[:2] - n * 2.0)[1]]
                out.append(dict(kind='open', launch=np.r_[e[:2], fz], normal=n, start=np.r_[S, fz], spot=np.r_[P, fz],
                                z_floor=fz, rise=max(float(e[2]), float(land[2])) - fz, land=np.asarray(land, dtype=float), run_u=d + OPEN_RUNIN_U))
                break
        # wall climbs: a face standing on this floor, its rebound (the face normal) heading at the place, a long run-in lane
        tri, nrm, zr, cen = self.faces
        if len(cen):
            near = (np.hypot(cen[:, 0] - float(target[0]), cen[:, 1] - float(target[1])) <= WALL_NEAR_U)
            near &= (zr[:, 0] <= zm + 0.6) & (zr[:, 1] >= zm + WALL_FACE_MIN_U)
            taken = []
            for f in np.flatnonzero(near):
                n = np.asarray(nrm[f][:2], dtype=float)
                L = float(np.linalg.norm(n))
                if L < 1e-6:
                    continue
                n = n / L
                q = np.asarray(cen[f][:2], dtype=float)
                if any(float(np.linalg.norm(q - t)) < 1.5 and float(n @ tn) > 0.8 for t, tn in taken):
                    continue
                for sgn in (1.0, -1.0):   # the run side: the floor at this level in front of the face
                    ns = n * sgn
                    fz = self.floor_at(q[0] + ns[0] * 0.8, q[1] + ns[1] * 0.8, zm)
                    if fz is not None:
                        break
                else:
                    continue
                rel = place['pts'][:, :2] - q
                along = rel @ ns
                lat = np.abs(rel[:, 0] * ns[1] - rel[:, 1] * ns[0])
                front = (along >= WALL_REACH_U[0]) & (along <= WALL_REACH_U[1]) & (lat <= WALL_LATERAL_U)
                if not front.any():
                    continue
                f0 = q + ns * 0.6
                if not self.clear((f0[0], f0[1], fz + 0.3), (f0[0], f0[1], zt + 2.0)):
                    continue   # something overhead in front of the face: the rebound would hit it
                run = 0.0
                for s in np.arange(1.0, WALL_RUN_MAX_U + 0.01, 0.5):
                    pt = q + ns * s
                    pz = self.floor_at(pt[0], pt[1], fz)
                    if pz is None or not self.clear((pt[0], pt[1], pz + 0.3), (pt[0], pt[1], pz + 0.3 + HEADROOM_U)):
                        break
                    run = float(s)
                if run < WALL_RUN_MIN_U:
                    continue
                cand = place['pts'][front]
                land = cand[int(np.argmin(np.abs((cand[:, :2] - q) @ ns - 4.0) + 0.3 * np.abs((cand[:, :2] - q)[:, 0] * ns[1] - (cand[:, :2] - q)[:, 1] * ns[0])))]
                taken.append((q, ns))
                out.append(dict(kind='wall', launch=np.r_[q, fz], normal=ns, start=np.r_[q + ns * run, fz], z_floor=fz,
                                rise=float(land[2]) - fz, land=np.asarray(land, dtype=float), run_u=run))
        self._options[key] = out
        if len(self._options) > 128:
            self._options.pop(next(iter(self._options)))
        return place, out

    # ------------------------------------------------------------------ choice
    def plan(self, position, velocity, target, meter, walk_dist, prefer=None):
        """The best climb up to the target's place as a crossing dict (climb2=True), or None (no climb, or walking is quicker).
        `prefer` = (kind, launch xy) of the climb chosen before: kept unless another is CLIMB_SWITCH_S quicker."""
        p = np.asarray(position, dtype=float)
        if float(np.hypot(p[0] - float(target[0]), p[1] - float(target[1]))) > CLIMB_RANGE_U:
            return None
        zm = self.floor_at(p[0], p[1], p[2] - RADIUS)
        if zm is None or float(target[2]) - zm < CLIMB_MIN_RISE_U:   # the gem floats ~0.5 above its floor: a cheap first test
            return None
        place, opts = self.options(target, zm)
        if place is None or place['z'] - zm < CLIMB_MIN_RISE_U or not opts:
            return None
        m = float(meter or 0.0)
        v = np.asarray(velocity[:2], dtype=float) if velocity is not None else np.zeros(2)
        best = None; kept = None
        for o in opts:
            to = o['start'][:2] - p[:2]
            dist = float(np.linalg.norm(to))
            reach = dist / CLIMB_RUN_SPEED_U_S
            sp = float(np.linalg.norm(v))
            if sp > 3.0 and dist > 1.0 and float(v @ to) / (sp * dist) < 0.0:
                reach += 0.6                      # turning back
            run = o['run_u'] / CLIMB_RUN_SPEED_U_S
            if o['kind'] == 'open':
                need = open_meter(o['rise'])
                if need is None:
                    continue
                t_up = reach + run + 0.6
            else:
                need = wall_meter(o['rise'])
                if need is None:
                    continue
                t_up = reach + run + 0.9
            wait = max(0.0, (need - m) * BLAST_CHARGE_S - reach - run) if need > m else 0.0
            total = t_up + wait + float(np.hypot(o['land'][0] - float(target[0]), o['land'][1] - float(target[1]))) / PLACE_ROLL_U_S
            if best is None or total < best[0]:
                best = (total, o, need, wait)
            if prefer is not None and o['kind'] == prefer[0] and float(np.hypot(*(o['launch'][:2] - np.asarray(prefer[1])))) < 0.3:
                kept = (total, o, need, wait)
        if best is None:
            return None
        if kept is not None and kept[0] <= best[0] + CLIMB_SWITCH_S:
            best = kept
        t_walk = float(walk_dist) / CLIMB_WALK_SPEED_U_S if walk_dist is not None and math.isfinite(walk_dist) else math.inf
        if not best[0] + CLIMB_SAVE_S <= t_walk:
            return None
        total, o, need, wait = best
        u = -o['normal']                          # the run heads at the edge / into the face
        return dict(climb2=True, kind=o['kind'], launch=o['launch'].tolist(), direction=u.tolist(), staging=o['start'].tolist(),
                    land=o['land'].tolist(), land_goal=o['land'].tolist(), rise_u=round(float(o['rise']), 2), z_place=float(place['z']),
                    z_floor=float(o['z_floor']), needed_meter=float(need), run_u=float(o['run_u']), wait_s=round(float(wait), 2),
                    t_s=round(float(total), 2), t_walk_s=round(t_walk, 2) if math.isfinite(t_walk) else None, place=place,
                    target_key=(round(float(target[0]), 1), round(float(target[1]), 1), round(float(target[2]), 1)),
                    source=None, target=None, climb=False, gap_route=False)

    def ready(self, climb, position, velocity, meter):
        """(in the run region, meter enough by the take-off) for a staged climb. The run region: in front of the wall / behind the
        launch spot, out to the run start plus RUN_REGION_EXTRA_U, widening with distance (the climb plans line up by themselves;
        only plans whose forecast lands up there are kept)."""
        p = np.asarray(position[:2], dtype=float)
        u = np.asarray(climb['direction'], dtype=float); L = np.asarray(climb['launch'][:2], dtype=float)
        rel = p - L
        along = float(rel @ u)                    # < 0 before the edge / face
        lateral = abs(float(rel[0] * u[1] - rel[1] * u[0]))
        in_lane = (-(climb['run_u'] + RUN_REGION_EXTRA_U) <= along <= (-0.8 if climb['kind'] == 'wall' else -0.3)
                   and lateral <= RUN_REGION_LATERAL_U + RUN_REGION_CONE * max(0.0, -along))
        t_run = max(0.0, -along) / CLIMB_RUN_SPEED_U_S
        m_at = float(meter or 0.0) + t_run / BLAST_CHARGE_S
        enough = climb['needed_meter'] <= 0.0 or m_at >= climb['needed_meter'] - READY_METER_SLACK
        return in_lane, enough


class ClimbProposal:
    """One climb as a plan: run along the climb line (pure pursuit onto it) at full throttle, press jump `jump_u` before the edge
    (open) or the wall face (wall), at a place (not a reply count, so re-plans keep the take-off), Blast `blast_after` replies later
    (-1: none), steer the flight to the landing point (after a wall's rebound), then follow `after` = (route, gem) up there or stop.
    Not a SequenceProposal subclass: the per-reply forecast runs it."""

    def __init__(self, climb, route, gem, steps, jump_u, blast_after=-1, after=None):
        import copy
        self.kind = climb['kind']
        self.launch = np.asarray(climb['launch'][:3], dtype=float)
        self.u = np.asarray(climb['direction'][:2], dtype=float)
        self.land = np.asarray(climb['land_goal'][:3], dtype=float)
        self.z_place = float(climb['z_place'])
        self.jump_u = float(jump_u); self.blast_after = int(blast_after)
        self.route = copy.copy(route); self.gem = np.asarray(gem[:3], dtype=float); self.steps = int(steps)
        self.after = after; self.jumped = None; self.turned = False

    def __call__(self, state, index):
        from nav.marble_dynamics.gem_demo import Proposal
        if index == 0:
            self.jumped = None; self.turned = False
        p = np.asarray(state['position'], dtype=float); v = np.asarray(state['velocity'], dtype=float)
        along = float((p[:2] - self.launch[:2]) @ self.u)
        if self.jumped is None:
            togo = -along - (RADIUS if self.kind == 'wall' else 0.0)
            press = togo - max(0.0, float(v[:2] @ self.u)) * LEAD_S <= self.jump_u
            ahead = min(along + 3.0, 3.0 if self.kind == 'wall' else 0.5)
            aim = np.r_[self.launch[:2] + self.u * ahead, self.launch[2]]
            action = Proposal(aim, 0., jump_at=0 if press else -1, speed_distance=1e3, min_speed=15., top=15.)(state, 0)
            if press:
                self.jumped = index
            action[8] = 0
            return action
        k = index - self.jumped
        up = p[2] >= self.z_place - PLACE_BELOW_U + RADIUS - 0.15 and abs(float(v[2])) < 2.5 and k >= 4   # at the place's level (up_there)
        if self.kind == 'wall' and not self.turned:
            self.turned = k >= 3 and float(v[:2] @ self.u) < 0.5   # the rebound has turned the marble back
        if up and self.after is not None:
            nroute, ngem = self.after
            goal = nroute.navigate(p, ngem)[0]
            remaining = float(nroute.distance(p))
            if math.isfinite(remaining):
                action = Proposal(goal, 0., speed_distance=remaining, min_speed=4.0, top=10.)(state, 0)
            else:
                action = Proposal(self.land, 0.)(state, 0)
        elif up:
            action = Proposal(self.land, 0.)(state, 0)
        elif self.kind == 'wall' and not self.turned:
            action = Proposal(np.r_[self.launch[:2] + self.u * 3.0, self.launch[2]], 0., speed_distance=1e3, min_speed=15., top=15.)(state, 0)
        else:
            action = Proposal(self.land, 0., speed_distance=1e3, min_speed=15., top=15.)(state, 0)   # the flight: toward the landing
        action[8] = int(self.blast_after >= 0 and k == self.blast_after)
        return action


def climb_specs(engine, route, dynamics, gem, climb):
    """The cross phase's plans for a staged climb: ClimbProposal at the OPEN_JUMP_U / WALL_TRIG_U take-off points with the Blast
    timings (the wall's no-Blast run too), each forecast long enough for the run, the flight and the landing."""
    from nav.marble_dynamics.blast import available as blast_available
    from nav.marble_dynamics.search import cached_route
    st = dynamics['native']['state']
    p = np.asarray(st['position'], dtype=float)
    togo = max(0.0, -float((p[:2] - np.asarray(climb['launch'][:2], dtype=float)) @ np.asarray(climb['direction'][:2], dtype=float)))
    steps = int(min(110, math.ceil(togo / 8.0 / .064) + 45))
    real = climb.get('goal')
    after = (cached_route(route.terrain, np.asarray(real[:3], dtype=float)), np.asarray(real[:3], dtype=float)) if real is not None else None
    blast = blast_available(dynamics.get('blast'))
    specs = []
    if climb['kind'] == 'open':
        if not blast:
            return []
        for ju in OPEN_JUMP_U:
            for b in OPEN_BLAST_AFTER:
                specs.append(('climb_open_blast_%g_%d' % (ju, b), ClimbProposal(climb, route, gem, steps, ju, b, after)))
    else:
        for tr in WALL_TRIG_U:
            specs.append(('climb_wall_jump_%g' % tr, ClimbProposal(climb, route, gem, steps, tr, -1, after)))
            if blast:
                for b in WALL_BLAST_AFTER:
                    specs.append(('climb_wall_blast_%g_%d' % (tr, b), ClimbProposal(climb, route, gem, steps, tr, b, after)))
    return specs


def landed_on_place(climb, rollout, skill):
    """First reply where the forecast stands up there (ClimbSkill.up_there; None if it never does)."""
    states = rollout['states']
    if climb.get('place') is None:
        return None
    for i in range(1, len(states)):
        s = states[i]
        if abs(float(s['velocity'][2])) < 2.5 and skill.up_there(climb, s['position']):
            return i
    return None
