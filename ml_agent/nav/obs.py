"""Navigator observation: two height crops + a short vector. Map-independent, world frame.

Layout NAV_OBS_V1 (stored in every checkpoint; change => bump the version):
    crop  (6, 32, 32)  fine (0.5 u) then coarse (2 u): [rel height, present, 2nd level] each
    vec   (48,)
        0-3   waypoint: unit dx, dy; distance/50 clipped to 1; dz/10 clipped to +/-1
        4-9   self: vx/20, vy/20, vz/20, speed/20, on_floor (0/1), airborne decisions/8 clipped to 1
        10-47 edge rays (38): 16 headings [edge_dist, edge_dz] + 3 "gem" rays, of which ray 1 is the
              line to the waypoint (clear fraction, what lies beyond) and rays 2-3 are unused (0)
"""
import math
import numpy as np

from terrain_obs import EDGE_DIM
from nav.protocol import RAW_POS, RAW_VEL
from nav.terrain import CROP_SHAPE, JUMP_DROP, JUMP_RISE
from nav.physics import crossable, speed_for_gap, MIN_JUMP_GAP, jump_verdict
from terrain_obs import RAY_RANGE

NAV_OBS_VERSION = 'NAV_OBS_V5'   # V5 (2026-09-25, HANDOFF 28.39): landing-predictor verdicts (GAP_DIM 4 -> 5)
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
VEC_DIM = VEC_GAP + GAP_DIM   # 58
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
        return crop.astype(np.float32), vec, on_floor


assert CROP_SHAPE == (6, 32, 32)
