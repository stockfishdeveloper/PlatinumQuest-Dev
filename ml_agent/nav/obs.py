"""Navigator observation: two height crops + a short vector. Map-independent, world frame.

Layout NAV_OBS_V1 (stored in every checkpoint; change => bump the version):
    crop  (6, 32, 32)  fine (0.5 u) then coarse (2 u): [rel height, present, 2nd level] each
    vec   (48,)
        0-3   waypoint: unit dx, dy; distance/50 clipped to 1; dz/10 clipped to +/-1
        4-9   self: vx/20, vy/20, vz/20, speed/20, on_floor (0/1), airborne decisions/8 clipped to 1
        10-47 edge rays (38): 16 headings [edge_dist, edge_dz] + 3 "gem" rays, of which ray 1 is the
              line to the waypoint (clear fraction, what lies beyond) and rays 2-3 are unused (0)
    later versions append blocks at the end: next gem (V2, 48-52), gap (V3-V5, 53-57), spin (V6, 58-60)
"""
import math
import numpy as np

from terrain_obs import EDGE_DIM
from nav.protocol import (RAW_POS, RAW_VEL, RAW_SPIN, RAW_POW_HELD, RAW_POW_BLAST, RAW_POW_SPECIAL, RAW_POW_MEGA,
                          RAW_POW_MEGA_LEFT, RAW_POW_BOUNCE_LEFT, RAW_POW_SHOCK_LEFT, RAW_POW_HELI_LEFT, RAW_POW_ITEMS,
                          POW_RESPAWN_S)
from nav.terrain import CROP_SHAPE, JUMP_DROP, JUMP_RISE
from nav.physics import crossable, speed_for_gap, MIN_JUMP_GAP, jump_verdict
from terrain_obs import RAY_RANGE
from nav import ss_two_gem

NAV_OBS_VERSION = 'NAV_OBS_V10'  # V10 (2026-10-04, log 40.26): kick result speed + the use state appended (POW_DIM 37)
                                 # V9 (2026-10-03, log 40.12): Super Speed redirect flag + aim appended (POW_DIM 30)
                                 # V8 (2026-10-03, log 40.11): Super Speed run-clear flag appended (POW_DIM 27)
                                 # V7 (2026-10-02, POWERUP_PLAN phase 1): the powerup block, appended (POW_DIM 26)
                                 # V6 (2026-09-26, HANDOFF 28.56): the marble's spin, appended (SPIN_DIM 3)
                                 # V5 (2026-09-25, HANDOFF 28.39): landing-predictor verdicts (GAP_DIM 4 -> 5)
                                 # V4 (2026-09-24, HANDOFF 28.36) adds the speed ratio for the gap ahead (GAP_DIM 3 -> 4)
                                 # V3 (2026-09-23, HANDOFF 28.27) appends GAP_DIM: the gap along the velocity
                                 # V2 (2026-09-19) appends the NEXT gem: see NEXT_DIM below
NEXT_DIM = 5                  # next gem: unit dx, unit dy, distance, dz, present flag
VEC_NEXT = 10 + EDGE_DIM      # 48: where the next-gem block starts (appended AFTER the edge rays,
                              # so nav/model.py's fixed gap-prior indices 2/7/8/42/43 still hold)
VEC_GAP = VEC_NEXT + NEXT_DIM # 53: where the gap block starts
GAP_DIM = 5                   # along the velocity heading (or the waypoint bearing when slow):
                              # [4] = lands even ballistic (V5); [2] = lands if forward is held (V5)
                              # lip distance / RAY_RANGE (1 = none), gap length / RAY_RANGE (1 = no
                              # landing), crossable at the CURRENT speed per nav/physics (0/1),
                              # speed ratio = current speed / speed the crossing needs, /2 clipped to 1
                              # (0.5 = exactly fast enough; below = accelerate; 0 = no gap ahead)
VEC_SPIN = VEC_GAP + GAP_DIM  # 58: where the spin block starts
SPIN_DIM = 3                  # V6: the spin as a ROLLING VELOCITY, /VEL_SCALE like vec[4:7]:
                              # [0] r*wy, [1] -r*wx = the velocity pure rolling on flat floor would give (rolling
                              # east spins about +y, |v|/|w| = 0.190 = r; measured live 2026-09-26), so a gap
                              # between vec[4:6] and these two IS the skid; [2] r*wz = spin about the vertical
VEC_POW = VEC_SPIN + SPIN_DIM # 61: where the powerup block starts (V7, 2026-10-02, POWERUP_PLAN phase 1)
POW_DIM_V9 = 30               # 40.15: [28-29] = the kick aim toward the CURRENT waypoint (v + 25 a on its line), [26] =
                              # that kick survivable (floor for the stop, or the next gem on the line); [27] = approach
                              # information: a redirect at the gem ahead toward the next gem would be survivable.
                              # [27] V9 (original meaning): Super Speed REDIRECT possible at the waypoint ahead: arriving at the current
                              # speed along the line to it, a kick aimed by ss_redirect_aim() turns the velocity onto
                              # the line to the NEXT waypoint and the floor runs to that gem; [28], [29] = that aim
                              # (world unit vector; the worker fires along it). Computed while a Super Speed is held and
                              # the next waypoint is known, at any distance, so the approach can be planned on it.
                              # [26] V8: Super Speed run clear (1 = a boost fired now along the line to the waypoint
                              # has floor for the whole overshoot, or the next waypoint lies on that line within the
                              # floor; only computed while a Super Speed is held, else 0); before it (V7):
                              # held type one-hot (7: SJ, SS, bounce, shock, heli, mega, blast pickup), blast meter,
                              # special blast armed, mega active, mega left /5, bounce left /5, shock left /5, heli
                              # left /5, then the NEAREST in-scope powerup item: type one-hot (7), unit dx, dy,
                              # distance /50, dz /10, seconds to respawn /7 (0 = there now); no item: all 0
POW_DIM = 37                  # V10 (40.26): [30] the kick's resulting speed / SS_VMAX (Super Speed held, else 0); the USE
                              # STATE, filled by the caller (fill_use_state): [31] use sent last decision, [32] the one
                              # before, [33] a kick pending (sent, not fired yet), [34-35] the latched aim (world unit
                              # vector), [36] time since the last Super Speed fire / USE_T_SCALE (1 = none for that long).
                              # The GRU never sees its own action and the kick fires two decisions after the use.
VEC_USE = VEC_POW + 31
USE_T_SCALE = 3.0             # s
VEC_DIM = VEC_POW + POW_DIM   # 98 (91 in V9, 88 in V8, 87 in V7)


def fill_use_state(vec, sent1, sent2, pending, aim, t_since_fire):
    """V10 use state (see POW_DIM): the worker, real_run and hybrid all call this after build()."""
    vec[VEC_USE + 0] = 1.0 if sent1 else 0.0
    vec[VEC_USE + 1] = 1.0 if sent2 else 0.0
    vec[VEC_USE + 2] = 1.0 if pending else 0.0
    vec[VEC_USE + 3] = aim[0] if (pending and aim is not None) else 0.0
    vec[VEC_USE + 4] = aim[1] if (pending and aim is not None) else 0.0
    vec[VEC_USE + 5] = min(max(t_since_fire, 0.0), USE_T_SCALE) / USE_T_SCALE
    return vec
SS_DV = 25.0                  # u/s a Super Speed adds along the camera yaw (powerup_physics.SUPER_SPEED_DV)
SS_VMAX = 32.0                # u/s the boosted marble reaches at most on the floor (seen ~30)
SS_DECEL = 14.0               # u/s^2 the floor takes off a boosted marble with no input (planner FAST_DECEL)
SS_RUN_MARGIN = 2.0           # u of floor past the stopping point / past the next waypoint
SS_LAM_MAX = 24.0             # u/s: a kick whose resolved speed (solver) is above this is not approved (40.22). The
                              # operator's demo kicks resolved to 12-25 u/s by the solver (the game reads ~1.7 less;
                              # median 17), all big turns, so this keeps every kick the operator makes and refuses the 25-33 u/s straight runs the policy
                              # also fired overshot and fell one time in five and taught it that kicks do not pay.
SS_NEXT_COS = 0.866           # the next waypoint counts as 'on the line' within 30 deg of the line to this one
SS_SCAN_U = 48.0              # how far along the line the floor is followed
SS_LEVEL_DZ = 0.75            # 40.34: braking room must be LEVEL floor: within this of the kick's floor height. The floor scan
                              # (terrain gap_along) follows a gradual bank step by step, so a rim the marble climbs counted
                              # as room: with 28897 39 % of the kicks fell within 3 s (60 % at 22-26 u/s), each one the same
                              # way (gem taken at 16-19 u/s, ~10 u overshoot up the outer rim, off it while turning to the
                              # next gem). With this rule (fire at every approval, 16 rounds): 0 falls after 12 kicks,
                              # 160.9 points a round vs 159.0 without Super Speed and 157.2 with the old rule
SS_LEVEL_STEP = 0.5           # u between the level-run samples
SS_REDIRECT_MIN_SPEED = 3.0  # u/s: below this a redirect is just a boost toward the next gem (still fine)
SS_TWO_GEM = True             # 2026-10-05 (log 40.53, operator: "start training on that"): the kick aim [28-29] and the
                              # approval [26] come from nav.ss_two_gem: the floor physics (spin, slip, the stick toward gem 1
                              # then gem 2) picks the direction that reaches gem 1 and then gem 2 (gem 1 alone when no next
                              # gem is known) soonest, and approves it only if that beats the same drive WITHOUT a kick by
                              # ss_two_gem.SAVE_MIN on followed floor below V2_MAX (17:45: no one-push two-gem requirement).
                              # The old aim pointed the instant velocity at gem 1
                              # only: 45 % of recorded kicks were fired where no aim could take both (5 % then did) and the
                              # stick cannot turn a sliding marble for ~0.4 s after a kick. False = the 40.26 rule below.
SS_TWO_GEM_MIN_U = 8.0        # u to gem 1: plan only where the use prior can fire (model.USE_RUN_U)


def ss_redirect_aim(vx, vy, nx, ny, dv=SS_DV):
    """Unit kick direction (ax, ay) so that (vx, vy) + dv (ax, ay) is parallel to the unit line (nx, ny) to the next
    waypoint, and the resulting speed. lam = n.v + sqrt((n.v)^2 + dv^2 - |v|^2) > 0 always for dv > |v|."""
    nv = nx * vx + ny * vy
    disc = nv * nv + dv * dv - (vx * vx + vy * vy)
    if disc < 0.0:
        # 40.21: the marble moves away faster than the kick can turn it (|v| > dv against the line): the best use is a
        # pure BRAKE, the kick straight against the velocity; the marble keeps rolling the old way at |v| - dv.
        # Returned as the aim plus the resultant along n, which is then NEGATIVE (still moving away).
        sv = math.hypot(vx, vy)
        ax, ay = -vx / sv, -vy / sv
        return (ax, ay), nv * (1.0 - dv / sv)
    lam = nv + math.sqrt(disc)
    ax, ay = (lam * nx - vx) / dv, (lam * ny - vy) / dv
    na = math.hypot(ax, ay)
    return ((ax / na, ay / na) if na > 1e-6 else (nx, ny)), lam


def ss_kick_result(vx, vy, nx, ny, dv=SS_DV):
    """40.26: the kick aim and the RESULTING VELOCITY VECTOR (not its projection on the line): on the line (lam n) when
    the kick can turn the velocity onto it, else the pure brake, which leaves the marble rolling along its OLD heading
    at |v| - dv. Returns (aim, (rvx, rvy), speed)."""
    aim, lam = ss_redirect_aim(vx, vy, nx, ny, dv)
    nv = nx * vx + ny * vy
    if nv * nv + dv * dv - (vx * vx + vy * vy) >= 0.0 and lam >= 0.0:      # the kick can put the velocity on the line
        rvx, rvy = lam * nx, lam * ny
    else:
        sv = math.hypot(vx, vy)
        k = max(0.0, sv - dv) / sv if sv > 1e-6 else 0.0
        rvx, rvy = vx * k, vy * k
    sp = min(SS_VMAX, math.hypot(rvx, rvy))
    return aim, (rvx, rvy), sp


def level_run(terrain, x, y, z, hx, hy, max_u):
    """40.34: distance along unit (hx, hy) from (x, y) over which a floor level stays within SS_LEVEL_DZ of the floor
    under the marble (inf if it does for max_u, or for a terrain without height levels)."""
    if not hasattr(terrain, 'heights_at'):
        return math.inf
    z0 = terrain.floor_height(x, y, z)
    ds = (np.arange(1, int(max_u / SS_LEVEL_STEP) + 1) * SS_LEVEL_STEP).astype(np.float32)
    h = terrain.heights_at(x + hx * ds, y + hy * ds)                      # (K, n) floor levels per sample
    near = np.where(np.isfinite(h), np.abs(h - z0), np.inf).min(axis=0) <= SS_LEVEL_DZ
    off = np.flatnonzero(~near)
    return float(ds[off[0]]) if off.size else math.inf


def ss_kick_ok(terrain, x, y, z, rvx, rvy, sp, d_target, ux, uy, nxt_rel=None):
    """40.26: ONE survivability rule for every Super Speed kick (fired now, or predicted at the gem ahead): the resulting
    speed within SS_LAM_MAX, and floor along the resulting heading for the stop at SS_DECEL (and past the target), or
    the following gem on that line with floor to it + SS_NEXT_MARGIN. (Before, the next-gem override skipped the cap and
    the approach flag used a different floor rule.) nxt_rel = (dx, dy) of the gem after the target, or None."""
    if sp > SS_LAM_MAX:
        return False
    if sp < 1e-3:
        return True
    hx, hy = rvx / max(math.hypot(rvx, rvy), 1e-6), rvy / max(math.hypot(rvx, rvy), 1e-6)
    lip, _, _ = terrain.gap_along(x, y, z, hx, hy, max_u=SS_SCAN_U)
    lip = min(lip, level_run(terrain, x, y, z, hx, hy, SS_SCAN_U))      # 40.34: a bank or rim is not braking room
    along = max(0.0, d_target * (hx * ux + hy * uy))        # how far the target lies along the resulting heading
    if lip > max(sp * sp / (2.0 * SS_DECEL), along) + SS_RUN_MARGIN:
        return True
    if nxt_rel is not None:
        ndx, ndy = nxt_rel
        nd = math.hypot(ndx, ndy)
        if nd > along and (ndx * hx + ndy * hy) / max(nd, 1e-6) >= SS_NEXT_COS and lip > nd + SS_NEXT_MARGIN:
            return True
    return False


def ss_aim(vec):
    """The world direction a Super Speed use is fired along, from the observation: the redirect aim when the redirect
    flag is on and the waypoint is within USE_REDIRECT_U, else the line to the waypoint. Shared by the worker and the
    evaluators so training and play fire the same way."""
    # 40.15: [28-29] = the aim toward the CURRENT waypoint (resultant on its line), set whenever a Super Speed is held
    if vec[VEC_POW + 1] > 0.5 and (vec[VEC_POW + 28] != 0.0 or vec[VEC_POW + 29] != 0.0):
        return float(vec[VEC_POW + 28]), float(vec[VEC_POW + 29])
    return float(vec[0]), float(vec[1])


SS_REDIRECT_U = 3.0           # u from the waypoint: the outer limit of the redirect window
SS_KEY_LAG_S = 0.128          # s between the use sent and the kick (2 decisions)
SS_FIRE_U = 0.8               # u: the kick must land within this of the gem (or past it), else the swing misses the pickup
SS_NEXT_MARGIN = 6.0          # u of floor past the NEXT gem for the braking after a redirect (23 u/s arrives there)


def ss_redirect_window(vec):
    """True while a redirect use sent NOW lands the kick at the pickup: the waypoint is within the distance covered
    during the key lag + SS_FIRE_U (log 40.14: at 3 u the kick came 1-2 u early and swung the marble off the gem)."""
    dist = vec[2] * WAYPOINT_DIST_SCALE
    speed = math.hypot(vec[4], vec[5]) * VEL_SCALE
    return dist <= min(SS_REDIRECT_U, SS_KEY_LAG_S * speed + SS_FIRE_U)
MARBLE_RADIUS = 0.19          # u (the probe's v/|w| on the floor, 0.1898)
SPIN_CLIP = 3.0               # rolling-speed features clipped to +-3 (60 u/s); rolling tops out near 1
GAP_MIN_SPEED = 1.0           # u/s: below this the gap features use the waypoint bearing
PREDICT_LIP_MAX = 6.0         # u: the landing predictor runs only when the lip is this close (a decision is 0.5 u)
WAYPOINT_DIST_SCALE = 50.0
VEL_SCALE = 20.0
ON_FLOOR_VZ = 0.3
ON_FLOOR_DZ = 0.6


class ObsBuilder:
    """Per-environment state (airborne counter) + the build() function."""

    def __init__(self, terrain):
        self.terrain = terrain
        self.airborne = 0

    def reset(self):
        self.airborne = 0

    def build(self, raw, goal, next_goal=None):
        x, y, z = (float(v) for v in raw[RAW_POS])
        vx, vy, vz = (float(v) for v in raw[RAW_VEL])
        gx, gy, gz = goal
        crop = self.terrain.crop(x, y, z)
        dx, dy = gx - x, gy - y
        d = math.hypot(dx, dy)
        ux, uy = (dx / d, dy / d) if d > 1e-6 else (0.0, 0.0)
        floor = self.terrain.floor_z(x, y, z)
        on_floor = abs(vz) < ON_FLOOR_VZ and abs(z - floor) < ON_FLOOR_DZ
        self.airborne = 0 if on_floor else self.airborne + 1
        vec = np.zeros(VEC_DIM, dtype=np.float32)
        vec[0] = ux; vec[1] = uy
        vec[2] = min(d / WAYPOINT_DIST_SCALE, 1.0)
        vec[3] = max(-1.0, min(1.0, (gz - z) / 10.0))
        vec[4] = vx / VEL_SCALE; vec[5] = vy / VEL_SCALE; vec[6] = vz / VEL_SCALE
        vec[7] = math.sqrt(vx * vx + vy * vy + vz * vz) / VEL_SCALE
        vec[8] = 1.0 if on_floor else 0.0
        vec[9] = min(self.airborne / 8.0, 1.0)
        vec[10:VEC_NEXT] = self.terrain.edge_rays(x, y, z, [(dx, dy, gz - z, True)])
        # the NEXT gem of the group, so the policy can shape its approach to this one; all zeros
        # (flag 0) while chasing the last gem, which is unambiguous because a real direction is a
        # unit vector and can never be (0, 0)
        if next_goal is not None:
            nx, ny, nz = next_goal
            ndx, ndy = nx - x, ny - y
            nd = math.hypot(ndx, ndy)
            vec[VEC_NEXT + 0] = ndx / nd if nd > 1e-6 else 0.0
            vec[VEC_NEXT + 1] = ndy / nd if nd > 1e-6 else 0.0
            vec[VEC_NEXT + 2] = min(nd / WAYPOINT_DIST_SCALE, 1.0)
            vec[VEC_NEXT + 3] = max(-1.0, min(1.0, (nz - z) / 10.0))
            vec[VEC_NEXT + 4] = 1.0
        # GAP block (V3): what lies along the marble's own heading
        sp = math.hypot(vx, vy)
        hx, hy = (vx / sp, vy / sp) if sp > GAP_MIN_SPEED else (ux, uy)
        if hx != 0.0 or hy != 0.0:
            lip, gap, ldz = self.terrain.gap_along(x, y, z, hx, hy)
            vec[VEC_GAP] = min(lip / RAY_RANGE, 1.0) if math.isfinite(lip) else 1.0
            vec[VEC_GAP + 1] = min(gap / RAY_RANGE, 1.0) if math.isfinite(gap) else 1.0
            # 2026-09-24 04:40 (HANDOFF 28.30): the flight must cover the run-up to the lip AS WELL as the gap;
            # a jump pressed 2.5 u before a 7 u hole needs 9.5 u of range. Testing the gap alone told the
            # policy 'crossable' too early and half of its deliberate hole jumps fell short.
            if math.isfinite(gap) and gap >= MIN_JUMP_GAP and (-JUMP_DROP <= ldz <= JUMP_RISE):
                need = speed_for_gap(lip + gap)
                vec[VEC_GAP + 3] = min(1.0, (sp / need) / 2.0) if need > 0 else 1.0   # 28.36: how much faster to go
            # V5 (2026-09-25, HANDOFF 28.39): the approval is the LANDING PREDICTOR, not a gap-vs-range test.
            # Integrate the jump arc from the current state over the height map (nav/physics.predict_landing),
            # with a 1 u lateral tolerance and floor required under the takeoff point:
            #   vec[VEC_GAP + 2]  lands on floor across a void if forward is held through the flight (achievable:
            #                     approves 15/17 of the human demo's jumps, 18/22 of the model's landings)
            #   vec[VEC_GAP + 4]  lands on floor whatever the marble does in the air (ballistic: 0 approved falls)
            # MIN_JUMP_GAP still gates both (KOTM hack, see nav/physics.py).
            ok_hold = ok_ball = False
            # only when a lip is within a flight's reach (a jump covers at most ~13 u): keeps the obs at ~1 ms
            if sp > GAP_MIN_SPEED and math.isfinite(gap) and gap >= MIN_JUMP_GAP and on_floor and lip <= PREDICT_LIP_MAX:
                v_hold, land = jump_verdict(self.terrain, x, y, z, vx, vy, hold=True)
                ok_hold = v_hold == 'floor' and bool(land and land['crossed_void'])
                if ok_hold:
                    v_ball, _ = jump_verdict(self.terrain, x, y, z, vx, vy, hold=False)
                    ok_ball = v_ball == 'floor'
            vec[VEC_GAP + 2] = 1.0 if ok_hold else 0.0
            vec[VEC_GAP + 4] = 1.0 if ok_ball else 0.0
        # SPIN block (V6): spin decides what a landing, an edge hit or a pre-spun start does next, and none of
        # it can be read from the velocity (a teleported or freshly landed marble can move with any spin)
        wx, wy, wz = (float(v) for v in raw[RAW_SPIN])
        k = MARBLE_RADIUS / VEL_SCALE
        vec[VEC_SPIN + 0] = max(-SPIN_CLIP, min(SPIN_CLIP, wy * k))
        vec[VEC_SPIN + 1] = max(-SPIN_CLIP, min(SPIN_CLIP, -wx * k))
        vec[VEC_SPIN + 2] = max(-SPIN_CLIP, min(SPIN_CLIP, wz * k))
        # POWERUP block (V7): what the marble holds and can use, and the nearest powerup item (raw 38-60,
        # observer.cs collectPowerups). The planner reads the raw block itself; the policy gets this summary.
        if len(raw) > RAW_POW_HELD:
            held = int(raw[RAW_POW_HELD])
            if 1 <= held <= 7:
                vec[VEC_POW + held - 1] = 1.0
            vec[VEC_POW + 7] = max(0.0, min(1.0, float(raw[RAW_POW_BLAST])))
            vec[VEC_POW + 8] = 1.0 if raw[RAW_POW_SPECIAL] > 0.5 else 0.0
            vec[VEC_POW + 9] = 1.0 if raw[RAW_POW_MEGA] > 0.5 else 0.0
            vec[VEC_POW + 10] = min(1.0, max(0.0, float(raw[RAW_POW_MEGA_LEFT])) / 5.0)
            vec[VEC_POW + 11] = min(1.0, max(0.0, float(raw[RAW_POW_BOUNCE_LEFT])) / 5.0)
            vec[VEC_POW + 12] = min(1.0, max(0.0, float(raw[RAW_POW_SHOCK_LEFT])) / 5.0)
            vec[VEC_POW + 13] = min(1.0, max(0.0, float(raw[RAW_POW_HELI_LEFT])) / 5.0)
            it = raw[RAW_POW_ITEMS]
            t0 = int(it[0])
            if 1 <= t0 <= 7:
                vec[VEC_POW + 14 + t0 - 1] = 1.0
                idx, idy, idz, left = float(it[1]), float(it[2]), float(it[3]), float(it[4])
                dd = math.hypot(idx, idy)
                vec[VEC_POW + 21] = idx / dd if dd > 1e-6 else 0.0
                vec[VEC_POW + 22] = idy / dd if dd > 1e-6 else 0.0
                vec[VEC_POW + 23] = min(dd / WAYPOINT_DIST_SCALE, 1.0)
                vec[VEC_POW + 24] = max(-1.0, min(1.0, idz / 10.0))
                vec[VEC_POW + 25] = min(1.0, max(0.0, left) / POW_RESPAWN_S)
            if held == 2 and on_floor and d > 1e-6:
                # V8: can a Super Speed fired NOW along the line to the waypoint be survived? The boosted marble
                # (its speed along the line + SS_DV, at most SS_VMAX) rolls v^2 / (2 SS_DECEL) u before it stops
                # (the floor's own braking; log 40.5: the overnight falls were overshoots past the gem at ~30 u/s).
                # Clear if the floor along the line continues that far (plus a margin) and past the waypoint, or if
                # the next waypoint lies on the line and the floor continues to it.
                # 40.15: the kick is AIMED so that v + 25 a points at the waypoint (ss_redirect_aim): a turn right after a
                # pickup and a straight run are the same rule; the aim goes out in [28-29] and the worker fires along it.
                # 40.26 (V10): ONE rule for every kick (ss_kick_result / ss_kick_ok): aim in [28-29], the resulting speed
                # in [30], survivable in [26]; the resulting VELOCITY VECTOR decides the heading the floor is followed
                # along (the brake case keeps the old heading), and the 24 u/s cap binds the next-gem-on-the-line case
                # too (before, that override skipped it: 8 u/s along the line + 25 = 33 u/s was approved).
                if SS_TWO_GEM and next_goal is not None and d >= SS_TWO_GEM_MIN_U:   # 18:20: gem 1 alone stays on the
                    p1, v1, w1 = ss_two_gem.advance((x, y), (vx, vy), (wx, wy, wz), (gx, gy))   # 40.26 rule (80 % of
                    ok, aim2, isp = ss_two_gem.plan_fast(self.terrain, p1, z, v1, w1, (gx, gy), next_goal[:2])   # those fell)
                    if aim2 is None:                    # no two-gem path: keep the old aim as information, not approved
                        aim2, _, isp = ss_kick_result(vx, vy, ux, uy)
                    vec[VEC_POW + 28] = aim2[0]; vec[VEC_POW + 29] = aim2[1]
                    vec[VEC_POW + 30] = min(isp, SS_VMAX) / SS_VMAX
                    vec[VEC_POW + 26] = 1.0 if ok else 0.0
                else:
                    aim0, (rvx, rvy), sp0 = ss_kick_result(vx, vy, ux, uy)
                    vec[VEC_POW + 28] = aim0[0]; vec[VEC_POW + 29] = aim0[1]
                    vec[VEC_POW + 30] = sp0 / SS_VMAX
                    nrel = (next_goal[0] - x, next_goal[1] - y) if next_goal is not None else None
                    vec[VEC_POW + 26] = 1.0 if ss_kick_ok(self.terrain, x, y, z, rvx, rvy, sp0, d, ux, uy, nrel) else 0.0
            if held == 2 and next_goal is not None and d > 1e-6:
                # approach information [27]: the SAME rule applied at the gem ahead with the predicted pickup velocity
                # (the current speed along the line to it, at least SS_REDIRECT_MIN_SPEED), kicking toward the next gem
                sp_in = max(SS_REDIRECT_MIN_SPEED, math.hypot(vx, vy))
                n2x, n2y = next_goal[0] - gx, next_goal[1] - gy
                n2 = math.hypot(n2x, n2y)
                if n2 > 1e-6:
                    n2x, n2y = n2x / n2, n2y / n2
                    _, (qvx, qvy), sq = ss_kick_result(sp_in * ux, sp_in * uy, n2x, n2y)
                    gzf = self.terrain.floor_z(gx, gy, gz)
                    if ss_kick_ok(self.terrain, gx, gy, gzf, qvx, qvy, sq, n2, n2x, n2y, None):
                        vec[VEC_POW + 27] = 1.0        # approach information only: the use fires after the pickup
        return crop.astype(np.float32), vec, on_floor


assert CROP_SHAPE == (6, 32, 32)
