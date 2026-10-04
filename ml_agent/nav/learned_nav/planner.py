"""Stage 4 planner (design sections 6-7, milestone M3): model-predictive control toward one floating gem.

Every decision, from the OBSERVED state (position, velocity, spin) and the two replies that shape the next physics
(the reply sent last decision acts first: bridge delay; the one before it is the step model's previous input):
1. propose control programs on the 64 ms grid: broad samples (a heading, maybe a turn or a brake, maybe a jump with
   an air direction), programs aimed at gem-side seeds (seeds.py: launch states predicted to take the gem), and the
   previous choice shifted by one decision with jittered copies (warm start);
2. roll every program forward with the stage 3 step ensemble, exactly as it will be executed: the pending reply
   first, then the program; once the prediction has landed the program's after-landing input replaces it;
3. score: P(pickup) from the closest predicted pass to the gem (logistic curve fitted in the P0 cross-check on the
   step rollouts: r0 0.85 u, slope 4) times a safe continuation (landed after any flight, on a floor for the last
   decisions, never below the start floor by more than FALL_DZ). Just before a jump from the floor the flight head
   re-judges the landing (the P0 cross-check's best combination: rollout pickup x flight-head safe), from the
   observed state: the observed-state launch check;
4. with no program reaching the gem, the one whose path comes closest to a seed state (position and velocity);
   after the pickup, the safest continuation;
5. send the chosen program's first decision; replan at the next observation.
Nothing here names a map: geometry comes from the exported terrain, the gem from the caller.
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav import dynamics3 as D3                                      # noqa: E402
from nav.learned_nav import eval3 as E3                                          # noqa: E402
from nav.learned_nav import powerup_physics as PW                                # noqa: E402
from nav.joystick import action_to_joystick                                      # noqa: E402

R = 0.19
SQ2 = math.sqrt(2.0)
H = 48                       # program decisions (3.07 s) after the pending reply (40 until stage 5: a late jump
                             # landed too close to the end of the horizon to show it had settled, and its plan vanished)
N_BROAD = 320
N_SEED = 96
N_WARM = 40
N_AIR = 256
N_LINE = 96                  # straight-line jump proposals on a floor-gem jump leg (approach_line)
PICK_R0, PICK_A = 0.85, 4.0  # P(pickup) = sigmoid(PICK_A (PICK_R0 - closest pass)), P0 cross-check fit (rollouts)
P_GO = 0.25                  # below this best P(success) the planner approaches instead (0.30 until dev v3)
P_JUMP = 0.25                # a jump from the floor is sent only above this (after the flight-head check). = P_GO since log
                             # 36.5 (0.35 before): a plan accepted at P_GO was followed to the lip and then forbidden to jump
                             # there, and the marble ran off (ring drill v6, 180 deg 9 u/s). Was, dev v3:
                             # all 70 floor jumps sent at >= 0.45 took the gem, so 0.45 was conservative
LAST_JUMP = 22               # latest jump decision in a program: time left in the horizon to land and settle
CONT_WIN = 24                # decisions after a landing over which the continuation is judged (1.5 s)
GROUND_WIN = 16              # ... and after the closest pass to the gem for a program that never flies (1 s; the plan is
                             # replanned there; the brake tail still has to be able to stop it inside the window)
A_STOP = 13.0                # u/s^2 of braking assumed after a landing (measured 14 on flat floor, stage 3 brake trials)
STOP_MARGIN = 1.15           # ... times the stopping distance, plus STOP_PAD u, must be floor straight ahead of a landing
STOP_PAD = 0.4               # (stage 5 gate: flights landing at 15 u/s on the 3 u strip between two holes rolled into the next)
FS_APPROACH = 0.0            # flight-head P(safe) a jump without a pickup plan needs at launch (0 = no check, as v8);
                             # tried 0.8 in v9 together with the stopping room: untested alone
# feature switches (stage 5b ablations; a variant module sets them before use)
STOP_CHECK = False           # stopping_room() after landings. OFF: straight-ahead stopping room ignores braking while
                             # turning and forbade landings v8 survived (devb ablation v9s: 7/13 vs v8 13/13)
AIR_BRAKE = False            # air-brake variants in the proposals (and the seeds' brake timing); untested alone
APPROACH_JUMPS = 'robust'    # 'robust': a jump without a pickup plan if safe from every perturbed start (v8);
                             # 'escape': only when every program without a jump falls
SETTLE_N = 4                 # decisions on a floor at the end of the horizon for a safe continuation
TAIL = 10                    # every ground program brakes for its last TAIL decisions: the plan must still be able to
                             # stop (terminal check; without it a program rolling toward a hole just past the horizon
                             # looked safe, first dev drills 2026-09-28)
CLEAR_OK = 1.0               # on-floor clearance from void/drop edges after a landing (or while rolling) counted full
ROLL_CLEAR_OK = 0.5          # ... for a program that never flies (stage 6b, log 34.12: the rolling error is a few hundredths
                             # of a unit a second, and the 1 u rule sank every sweeping turn on KOTM's floor)
CLEAR_MIN = 0.2              # ... and none at all below this (landing errors ~1 u: stage 3 flight eval, first drills)
SKIP_CLEAR = 3               # the first decisions of a rolling program are not charged (the marble may start near a lip)
FALL_DZ = 0.5                # below the start floor level by this much: fallen (2.0 until the first drill: a pocket
                             # under a hole's lip 1 u down counted as safe, and the marble was stuck there)
P_KEEP = 0.15                # a kept pickup plan (commitment) is dropped below this
SWITCH = 0.10                # ... or when another plan beats it by this much
K_ROBUST = 12                # best candidates re-checked from perturbed starts (first dev drills: landings 0.1-0.3 u
                             # past a hole's far lip clipped it in the game and bounced back in)
VARIANTS = (('s', 0.96), ('s', 1.04), ('h', math.radians(2.0)), ('h', -math.radians(2.0)))
N_VAR = len(VARIANTS)
APPROACH_JUMP = 3.0          # approach cost of a jump that takes no gem (u of seed distance)
CONT_SPEED_COST = 0.3        # continuation cost per u/s left at the end of the horizon (stage 5b: slow before handback)
CONT_JUMP = 3.0              # continuation toward a next gem: cost of a program that jumps (u of walking distance)
CONT_JUMP_NOW = 50.0         # continuation: cost of jumping NOW (was 5; hops on open floor after pickups, log 35.1)
TIME_W = 0.004               # P(success) worth one decision of earlier pickup, among pickup plans (a caller raises it
                             # for floor-gem shortcuts: the stage 5b jumps to KOTM corner gems picked the slow, careful
                             # plans and arrived at 2-6 u/s, 2.0-2.6 s a leg)
FLOOR_JUMP_CHARGE = 0.15     # ranking charge (in P units) for a jump in a pickup plan on a plain floor leg
TURN_S = 0.9                 # s per (1 - cos turn) at speed (turn drill: 90 deg 0.9 s), scaled by speed / TURN_REF_SPEED
TURN_REF_SPEED = 6.0
EXIT_W = 0.05                # (unused since the turn charge; kept for the attribute)  two-leg plans (exit_dir): ranking bonus per u/s of velocity along the exit direction at the
                             # pickup (up to EXIT_VMAX): a plan leaving the gem aligned and fast toward the jump gem beats one
                             # arriving from the wrong side (the human's crossings start aligned at 11 u/s, log 34.8)
EXIT_VMAX = 12.0
N_EXIT = 96                  # proposals that approach the gem from behind the exit line (a run-up onto it)
P_SAT = 1.0                  # ranking saturation among pickup plans: P(success) above this counts as this, so plans the
                             # model already trusts are ranked by time alone. 1.0 = off. Stage 6b crossing legs set 0.6:
                             # jumps sent with P >= 0.35 took the gem 180/186 (stage 5) and 187/188 (stage 4), but the
                             # planner still braked from 10 to 6 u/s before a crossing for a 0.9 plan over a 0.6 one
WALK_CLEAR = 0.4             # the walking-distance field keeps this far from any edge of the floor
USE_GUIDE = True             # time the seed-aimed run-ups with the time-to-gem guide (guide.py); proposals only.
                             # The M3 gate ran with it OFF (v5, 93.1 %). Dev A/B after the gate (log 31.8): 98/105
                             # with it against 93/105 without, pickups ~0.26 s sooner (not significant: 10 vs 5)
AIR_DRAG_RESID = 0.0         # per-step horizontal speed correction of airborne steps, x v^2 (u/s); 0 = off (log 34.15).
                             # OFF since AIR_EXACT (log 36.1): the game has NO air drag; the residual compensated the
                             # model's weak air control and made aligned flights land 2 u short
AIR_EXACT = True             # flight and flat-floor landings from the ENGINE's physics (marble.cc, log 36.1) instead of the
                             # step model: in the air a = gravity + AIR_ACCEL x the input vector (no drag, no cap); a
                             # landing with input and a shallow approach SLIDES (the speed is kept, redirected along the
                             # floor), one without input (or steep) BOUNCES with restitution 0.5 and the bounce-friction
                             # impulse. Steps near walls, lips or sloped floor stay with the step model. Measured on the
                             # centre drill (centredrill.py): the model flew 2-4 u long without input, 2 u short with it
AIR_ACCEL = 5.0              # MarbleData airAcceleration (u/s^2 per unit of input; the adapter sends up to sqrt 2)
FAST_GROUND = 18.0           # u/s: above this on the floor the step model is out of its data (log 39); analytic rolling
FAST_DECEL = 14.0            # u/s^2 measured after a Super Speed boost with no input (24.3 -> 16.6 in 11 decisions)
AIR_GAP_MIN = 0.05           # bottom this far above the floor = airborne for the exact step
BOUNCE_E = 0.5               # bounceRestitution; BOUNCE_FRICTION = bounceKineticFriction; MAX_DOT_SLIDE, MIN_BOUNCE_VEL as the datablock
BOUNCE_FRICTION = 0.2
MAX_DOT_SLIDE = 0.5
MIN_BOUNCE_VEL = 0.1
LAND_VZ_SETTLE = 1.5         # a bounce slower than this up (hop < 6 cm) counts as landed and settled
JUMP_PRIOR = True            # the jump impulse from the measured constant where it certainly fires (jump_fires): the
                             # step model underestimates it (apex -0.06 u median on its own data, -0.25 u for 12-20 u/s
                             # takeoffs within 1.5 u of a lip; it predicted no jump at all on some KOTM diagonal
                             # crossings the game flew: stage 5b, 2026-09-29)
JUMP_LIP_U = 0.6             # jump_fires: the floor must reach this far ahead along the velocity
JUMP_LIP_K = 0.0             # ... plus this times the speed. 0.064 until log 36.2 (one decision of travel; crossing drill v3: 5 falls in 86, every one
                             # a press at 11-13 u/s that acted past the lip; the engine fires 99 % at any speed with floor ahead)
JUMP_DZ = (0.4268, 0.3448)   # a jump from flat floor, the decision it fires and the next: height gained and vertical
JUMP_VZ = (6.108, 4.828)     # speed after each (every recorded one identical to 1e-4: 1,915 on KOTM and Sprawl); then
                             # gravity (20 u/s^2), which the model predicts well once airborne
MODE_DIR, MODE_BRAKE, MODE_NONE = 0, 1, 2
SEED_VERSION = 'v8'          # the seed set this planner's judgement built (datasets/learned_nav/seeds/<version>/)


def reply_vector(js):
    """World force vector and jump key of a joystick reply (fwd, back, left, right, jump, cam_yaw): what the step
    model was trained on (dynamics3: right - left, fwd - back)."""
    return np.array([js[3] - js[2], js[0] - js[1]], dtype=np.float64), float(js[4])


def p_pick(dmin):
    return 1.0 / (1.0 + np.exp(-PICK_A * (PICK_R0 - dmin)))


class Programs:
    """n programs of H decisions: mode, direction (world rad), throttle, jump key; after-landing mode/direction, held
    after_n decisions from the landing and then a brake (after_n H + 1: held to the end)."""
    KEYS = ('mode', 'ang', 'thr', 'jump', 'after', 'after_ang', 'after_n', 'tag', 'use', 'use_yaw')

    def __init__(self, n):
        self.mode = np.full((n, H), MODE_DIR, np.int8); self.ang = np.zeros((n, H)); self.thr = np.ones((n, H))
        self.jump = np.zeros((n, H), np.int8); self.after = np.full(n, MODE_BRAKE, np.int8); self.after_ang = np.zeros(n)
        self.after_n = np.full(n, H + 1, np.int16)
        self.tag = np.zeros(n, np.int8)                      # 0 broad, 1 seed, 2 warm, 3 air
        self.use = np.zeros((n, H), np.int8)                 # powerups (log 39): 1 = fire the held one, 2 = the blast meter
        self.use_yaw = np.zeros((n, H))                      # camera yaw the use fires along (Super Speed)

    @staticmethod
    def cat(parts):
        out = Programs(0)
        for k in Programs.KEYS:
            setattr(out, k, np.concatenate([getattr(p, k) for p in parts]))
        return out

    def take(self, idx):
        out = Programs(0)
        for k in Programs.KEYS:
            setattr(out, k, getattr(self, k)[idx].copy())
        return out

    def shifted(self):
        """The same programs one decision later (the first decision was executed). The brake tail (the last TAIL
        decisions of every ground program) stays at the END: the decision before it is extended by one, so a plan
        followed for many decisions does not brake to a stop in the open (stage 6b, log 34.10: the planner-alone
        rounds stalled to 0.1-0.9 u/s mid-leg after ~2 s on the same plan, 2.4 s legs against the navigator's 1.36)."""
        out = self.take(slice(None))
        for k in ('mode', 'ang', 'thr', 'jump', 'use', 'use_yaw'):
            a = getattr(out, k)
            a[:, :-1] = a[:, 1:].copy()
            if k in ('jump', 'use'):
                a[:, -1] = 0
        j = H - TAIL - 1                                          # the first decision of the (shifted) tail
        tailed = out.mode[:, H - 1] == MODE_BRAKE                 # programs that carry a brake tail at all
        for k in ('mode', 'ang', 'thr'):
            a = getattr(out, k)
            a[tailed, j] = a[tailed, j - 1]
        return out

    def __len__(self):
        return len(self.mode)


def world_input(mode, ang, thr, vel_send):
    """The world force vector the joystick adapter produces (action_to_joystick): sqrt(2) x throttle along ang; brake
    = the unit vector against the velocity known when the reply is built (none below 0.1 u/s); none = zero."""
    n = len(mode)
    u = np.zeros((n, 2))
    d = mode == MODE_DIR
    u[d] = (SQ2 * thr[d])[:, None] * np.c_[np.cos(ang[d]), np.sin(ang[d])]
    b = mode == MODE_BRAKE
    if b.any():
        v = vel_send[b, :2]; sp = np.hypot(v[:, 0], v[:, 1])
        ub = np.zeros((int(b.sum()), 2)); k = sp > 0.1
        ub[k] = -v[k] / sp[k, None]
        u[b] = ub
    return u


class FastEnsemble:
    """The step ensemble's planning outputs: eval3.predict's 'mu' (contact mode picked by each member's own collision
    probability, members averaged) and 'p' (sigmoid of the averaged contact logits), computed on the device with one
    transfer each way per step (eval3.predict moved every member's every output separately: about half the planner's
    time, profile 2026-09-28). Same arithmetic in float32."""

    BUCKETS = (64, 256, 512)     # batch sizes captured as CUDA graphs (rows padded up); larger batches run eagerly

    def __init__(self, models, dev):
        import torch
        self.models = models; self.dev = dev
        self.tstd = [torch.as_tensor(m.tstd, device=dev) for m in models]
        self.tmean = [torch.as_tensor(m.tmean, device=dev) for m in models]
        self.graphs = {}
        if str(dev).startswith('cuda'):
            # one graph replay per step instead of ~60 kernel launches: with four planners on one GPU the launches
            # queued behind each other (1.3 s a decision against 0.57 s alone)
            for b in self.BUCKETS:
                x = torch.zeros((b, D3.N_FEAT), device=dev)
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s), torch.no_grad():
                    for _ in range(3):
                        self._forward(x)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g), torch.no_grad():
                    out = self._forward(x)
                self.graphs[b] = (g, x, out)

    def _forward(self, x):
        import torch
        mus, logits = [], []
        for m, s, mu0 in zip(self.models, self.tstd, self.tmean):
            o = m(x)
            if getattr(m, 'has_modes', False):
                md = m.last_modes * s + mu0
                pc = torch.sigmoid(o[2][:, 1])
                mus.append(torch.where((pc >= 0.5)[:, None], md[:, 1], md[:, 0]))
            else:
                mus.append(o[0] * s + mu0)
            logits.append(o[2])
        return torch.cat([torch.stack(mus).mean(0), torch.sigmoid(torch.stack(logits).mean(0))], 1)

    def __call__(self, F):
        import torch
        n = len(F)
        b = next((b for b in self.BUCKETS if b >= n), None) if self.graphs else None
        if b is None:
            with torch.inference_mode():
                out = self._forward(torch.as_tensor(F, device=self.dev)).cpu().numpy()
        else:
            g, x, o = self.graphs[b]
            x[:n].copy_(torch.from_numpy(np.ascontiguousarray(F, dtype=np.float32)))
            g.replay()
            out = o[:n].cpu().numpy()
        return out[:, :D3.N_CONT], out[:, D3.N_CONT:]


def air_exact(g, P, V, W, U, J, gmult=None, amult=None, rest=None, force=None):
    """Engine physics for airborne steps (AIR_EXACT). Returns (mask, P1, V1, W1, sup, coll, last): mask marks the rows
    whose step is computed here: the marble starts in the air (bottom > AIR_GAP_MIN above the level below) and the
    step's path stays clear of walls and lips (nothing within 2.4 u above the marble at the end, the level below the
    midpoint and the end at or under the start's or the landing floor). Rows that would meet a flat floor during the
    step land here as well (velocityCancel: slide, cancel or bounce). The rest is the step model's."""
    n = len(P); dt = D3.DT; G = D3.G
    gm = np.ones(n) if gmult is None else gmult                 # helicopter: gravity x0.25, air control x2 (log 38)
    am = np.ones(n) if amult is None else amult
    ee = np.full(n, BOUNCE_E) if rest is None else rest         # Super Bounce 0.9 / Shock Absorber 0.01
    a = np.c_[AIR_ACCEL * am * U[:, 0], AIR_ACCEL * am * U[:, 1], -G * gm]
    P1 = P + V * dt + 0.5 * a * dt * dt; V1 = V + a * dt; W1 = W.copy()
    ztop = P[:, 2] + 0.3
    lv0, ab0 = D3.level_below(g, P[:, 0], P[:, 1], ztop)
    Pm = 0.5 * (P + P1)
    lvm, abm = D3.level_below(g, Pm[:, 0], Pm[:, 1], ztop)
    lv1, ab1 = D3.level_below(g, P1[:, 0], P1[:, 1], ztop)
    gap0 = np.where(np.isfinite(lv0), P[:, 2] - R - lv0, 9.0)
    gapm = np.where(np.isfinite(lvm), Pm[:, 2] - R - lvm, 9.0)
    gap1 = np.where(np.isfinite(lv1), P1[:, 2] - R - lv1, 9.0)
    airborne = gap0 > AIR_GAP_MIN
    if force is not None:
        airborne |= force                                        # an impulse this step (Super Jump, blast): it flies
    clear = ~ab1 & ~abm & (gapm > 0.0)
    fly = airborne & clear & (gap1 > 0.0)
    # landing: the end point is at or under a floor that the midpoint is still above, the floor level the same at both
    flat = np.isfinite(lv1) & np.isfinite(lvm) & (np.abs(lv1 - lvm) < 0.05)
    land = airborne & clear & (gap1 <= 0.0) & flat
    sup = np.zeros(n); coll = np.zeros(n); last = np.zeros(n)
    if land.any():
        i = np.nonzero(land)[0]
        h = P[i, 2] - R - lv1[i]                                 # height above the landing floor at the start
        vz = V[i, 2]
        ok = h > 0.0
        i = i[ok]; h = h[ok]; vz = vz[ok]
        land[:] = False; land[i] = True
        gi = G * gm[i]
        tau = (vz + np.sqrt(vz * vz + 2.0 * gi * h)) / gi        # time to contact (vz < 0 or h small)
        tau = np.clip(tau, 0.0, dt)
        Pc = P[i] + V[i] * tau[:, None] + 0.5 * a[i] * tau[:, None] ** 2
        Vc = V[i] + a[i] * tau[:, None]
        speed = np.linalg.norm(Vc, axis=1); vh = np.hypot(Vc[:, 0], Vc[:, 1])
        centred = np.hypot(U[i, 0], U[i, 1]) < 0.01
        slide = ~centred & (Vc[:, 2] > -MAX_DOT_SLIDE * speed) & (vh > 1e-6)
        cancel = ~slide & (Vc[:, 2] > -MIN_BOUNCE_VEL)
        bounce = ~slide & ~cancel
        Vn = Vc.copy(); Wn = W[i].copy()
        # slide: the velocity loses its normal part and is rescaled to its old length
        Vn[slide, 0] = Vc[slide, 0] / vh[slide] * speed[slide]; Vn[slide, 1] = Vc[slide, 1] / vh[slide] * speed[slide]; Vn[slide, 2] = 0.0
        Vn[cancel, 2] = 0.0
        if bounce.any():
            b = bounce
            normal_vel = -Vc[b, 2]
            vatc = np.c_[Vc[b, 0] - R * Wn[b, 1], Vc[b, 1] + R * Wn[b, 0]]          # horizontal velocity of the contact point
            mag = np.hypot(vatc[:, 0], vatc[:, 1])
            ang_v = np.minimum(BOUNCE_FRICTION * 5.0 * normal_vel / (2.0 * R), mag / R)
            k = mag > 1e-6
            d = np.zeros_like(vatc); d[k] = vatc[k] / mag[k, None]
            # delta omega = (n x dir) ang_v; the velocity loses R ang_v along dir
            Wn[b, 0] += -d[:, 1] * ang_v; Wn[b, 1] += d[:, 0] * ang_v
            Vn[b, 0] -= R * ang_v * d[:, 0]; Vn[b, 1] -= R * ang_v * d[:, 1]
            Vn[b, 2] = -ee[i][b] * Vc[b, 2]
            settle = Vn[b, 2] < LAND_VZ_SETTLE
            Vn[np.nonzero(b)[0][settle], 2] = 0.0
        rem = (dt - tau)[:, None]
        Pn = Pc + Vn * rem + 0.5 * a[i] * rem ** 2
        Vn2 = Vn + a[i] * rem
        on_floor = Vn[:, 2] <= 0.0                                # slid, cancelled or settled: stays on the floor
        Pn[on_floor, 2] = lv1[i][on_floor] + R; Vn2[on_floor, 2] = 0.0
        below = Pn[:, 2] - R < lv1[i]                             # a small hop that ends within the step
        Pn[below, 2] = lv1[i][below] + R; Vn2[below, 2] = 0.0
        P1[i] = Pn; V1[i] = Vn2; W1[i] = Wn
        coll[i] = 1.0
        landed_now = on_floor | below
        sup[i[landed_now]] = 1.0; last[i[landed_now]] = 1.0
    mask = fly | land
    return mask, P1, V1, W1, sup, coll, last


def simulate(steps, g, eidx, dev, P, V, W, Uc, Jc, Up, Jp, progs, airborne0, floor_ref, pow=None):
    """Roll each program forward from its start (P, V, W: (n, 3)); Uc, Jc: the pending reply (acts first); Up, Jp:
    the reply before it. State index k = k decisions after the observation; program decision d is built at state d
    (its brake direction uses that state's velocity) and acts from state d + 1 to d + 2."""
    n = len(P)
    P = P.copy(); V = V.copy(); W = W.copy()
    airborne = np.array(airborne0, bool) if np.ndim(airborne0) else np.full(n, bool(airborne0))
    started_air = airborne.copy()
    landed = np.full(n, -1)
    path = np.empty((n, H + 2, 3)); path[:, 0] = P
    vel = np.empty((n, H + 2, 3)); vel[:, 0] = V
    sup_t = np.zeros((n, H + 1)); gap_t = np.zeros((n, H + 1)); coll_t = np.zeros((n, H + 1))
    Ul = np.zeros((n, H + 1, 2)); Jl = np.zeros((n, H + 1))
    jumped_at = np.full(n, -1)
    unplanned = np.zeros(n, bool); reair = np.zeros(n, bool); in_air_t = np.zeros((n, H + 1), bool)
    after_fire = np.zeros(n, bool)
    # powerups (log 39): pow = {'held': type 0-7, 'meter': blast meter, 'special': bool, 'use_p': use sent two decisions
    # ago (fires at t 0), 'use_c': use sent last decision (fires at t 1), 'use_yaw_p', 'use_yaw_c', 'heli_left',
    # 'bounce_left', 'shock_left' (seconds)}. A use fires two decisions after it is sent (powdrill, log 38.1).
    pw = pow or {}
    held = np.full(n, int(pw.get('held', 0))); meter = np.full(n, float(pw.get('meter', 0.0)))
    special = np.full(n, bool(pw.get('special', False)))
    gmult = np.ones(n); amult = np.ones(n); rest = np.full(n, BOUNCE_E)
    heli_end = np.full(n, float(pw.get('heli_left', 0.0)) / D3.DT)
    bounce_end = np.full(n, float(pw.get('bounce_left', 0.0)) / D3.DT); shock_end = np.full(n, float(pw.get('shock_left', 0.0)) / D3.DT)
    used_at = np.full(n, -1); impulse = np.zeros(n, bool)
    for t in range(H + 1):
        if t == 0:
            U = Uc.copy(); J = Jc.copy()
        else:
            d = t - 1
            post = (landed >= 0) & (landed <= d)
            md = np.where(post, np.where(d - landed >= progs.after_n, MODE_BRAKE, progs.after), progs.mode[:, d])
            an = np.where(post, progs.after_ang, progs.ang[:, d])
            th = np.where(post, 1.0, progs.thr[:, d])
            U = world_input(md, an, th, vel[:, d])
            J = np.where(post, 0.0, progs.jump[:, d].astype(float))
        # the powerup use that fires in this transition, and the effects active during it
        if t == 0:
            uf = np.full(n, int(pw.get('use_p', 0))); uy = np.full(n, float(pw.get('use_yaw_p', 0.0)))
        elif t == 1:
            uf = np.full(n, int(pw.get('use_c', 0))); uy = np.full(n, float(pw.get('use_yaw_c', 0.0)))
        else:
            uf = progs.use[:, t - 2].astype(int); uy = progs.use_yaw[:, t - 2]
        impulse[:] = False
        if uf.any():
            fire_h = (uf == 1) & (held > 0)
            if fire_h.any():
                sj = fire_h & (held == 1); ss = fire_h & (held == 2)
                V[sj, 2] += PW.SUPER_JUMP_VZ; impulse |= sj
                V[ss, 0] += PW.SUPER_SPEED_DV * np.sin(uy[ss]); V[ss, 1] += PW.SUPER_SPEED_DV * np.cos(uy[ss])
                heli_end[fire_h & (held == 5)] = t + PW.EFFECT_S[5] / D3.DT
                bounce_end[fire_h & (held == 3)] = t + PW.EFFECT_S[3] / D3.DT
                shock_end[fire_h & (held == 4)] = t + PW.EFFECT_S[4] / D3.DT
                used_at[fire_h & (used_at < 0)] = t
                held[fire_h] = 0
            fire_b = (uf == 2) & ((meter >= PW.BLAST_REQUIRED) | special)
            if fire_b.any():
                vb = np.where(special[fire_b], PW.BLAST_POWER * PW.BLAST_SPECIAL_POWER, PW.BLAST_POWER * np.sqrt(np.maximum(meter[fire_b], 0.0)))
                V[fire_b, 2] += vb; impulse |= fire_b
                meter[fire_b] = PW.BLAST_AFTER; special[fire_b] = False
                used_at[fire_b & (used_at < 0)] = t
        meter = np.minimum(1.0, meter + D3.DT / PW.BLAST_CHARGE_S)
        heli_on = t < heli_end
        gmult[:] = np.where(heli_on, PW.HELI_GRAVITY_MULT, 1.0); amult[:] = np.where(heli_on, PW.HELI_AIR_ACCEL_MULT, 1.0)
        rest[:] = np.where(t < shock_end, PW.SHOCK_RESTITUTION, np.where(t < bounce_end, PW.SUPER_BOUNCE_RESTITUTION, BOUNCE_E))
        F, yaw = D3.features(g, eidx, P, V, W, U, J, Up, Jp)
        if isinstance(steps, FastEnsemble):
            mu, pc = steps(F)
        else:
            pr = E3.predict(steps, F, dev); mu, pc = pr['mu'], pr['p']
        # the jump key acts one decision LATER than the direction (turn drill and centre drill, log 36.2: steering sent
        # at decision d moves state d + 1 -> d + 2, the jump sent at d fires in d + 2 -> d + 3): the key that fires in
        # this transition is the one sent before the one that steers it (Jp)
        fires = jump_fires(g, P, V, Jp, t) if JUMP_PRIOR else None
        z_before = P[:, 2].copy(); W_before = W.copy()
        air_before = airborne.copy()
        if AIR_EXACT:
            ex, Px, Vx, Wx, sup_x, coll_x, last_x = air_exact(g, P, V, W, U, J, gmult, amult, rest, impulse)
            if fires is not None:
                ex &= ~fires
        P_before = P.copy(); V_before = V.copy()
        P, V, W = D3.apply_step(P, V, W, mu, yaw)
        if AIR_EXACT and ex.any():
            P[ex] = Px[ex]; V[ex] = Vx[ex]; W[ex] = Wx[ex]
            pc = pc.copy(); pc[ex, 0] = sup_x[ex]; pc[ex, 1] = coll_x[ex]; pc[ex, 2] = last_x[ex]
        # FAST GROUND (log 39): the step model never saw a marble rolling above ~18 u/s (a Super Speed boost gives 25-31)
        # and brakes it 11 u/s in one step; the game loses ~0.9 u/s a decision with no input (14 u/s^2, powdrill).
        # Above FAST_GROUND on the floor: the measured deceleration along the velocity instead of the model's.
        sp_b = np.hypot(V_before[:, 0], V_before[:, 1])
        fast = (~ex) & (sp_b > FAST_GROUND) & (~air_before)
        if fast.any():
            f = fast; k = np.maximum(0.0, 1.0 - FAST_DECEL * D3.DT / sp_b[f])
            V[f, 0] = V_before[f, 0] * k; V[f, 1] = V_before[f, 1] * k
            P[f, 0] = P_before[f, 0] + 0.5 * (V_before[f, 0] + V[f, 0]) * D3.DT
            P[f, 1] = P_before[f, 1] + 0.5 * (V_before[f, 1] + V[f, 1]) * D3.DT
        if AIR_DRAG_RESID > 0:
            # measured residual of the step model in the air (log 34.15): the game's horizontal speed change in flight
            # falls with speed like drag (~ -0.0011 v^2 per step with no input) and the model keeps ~ +0.1-0.25 u/s a
            # step too much above 9 u/s on every map checked; the same for the forward input. Correct the horizontal
            # velocity of airborne steps by -AIR_DRAG_RESID x v^2
            sp_h = np.hypot(V[:, 0], V[:, 1])
            fac = np.where(air_before & (sp_h > 1e-6), np.maximum(0.0, 1.0 - AIR_DRAG_RESID * sp_h), 1.0)
            V[:, 0] *= fac; V[:, 1] *= fac
        if JUMP_PRIOR:
            # the jump's first two decisions from the measured kinematics (vertical only; horizontal from the model)
            if after_fire.any():
                P[after_fire, 2] = z_before[after_fire] + JUMP_DZ[1]; V[after_fire, 2] = JUMP_VZ[1]
            if fires.any():
                P[fires, 2] = z_before[fires] + JUMP_DZ[0]; V[fires, 2] = JUMP_VZ[0]
                if AIR_EXACT:
                    # the impulse is along the normal (marble.cc): horizontally the fire step is an air step under the
                    # input that was still acting (the game gains +0.4 u/s at 13 u/s, log 36.2); the step model
                    # predicted a 6.5 u/s loss for a fire at a lip (centre drill)
                    f = fires
                    ah = AIR_ACCEL * Up[f]
                    P[f, 0:2] = path[f, t, 0:2] + vel[f, t, 0:2] * D3.DT + 0.5 * ah * D3.DT ** 2
                    V[f, 0:2] = vel[f, t, 0:2] + ah * D3.DT
                    W[f] = W_before[f]
            after_fire = fires
        sup = pc[:, 0]; coll = pc[:, 1]; last = pc[:, 2]
        lv, _ = D3.level_below(g, P[:, 0], P[:, 1], P[:, 2] + 0.3)
        gap = np.where(np.isfinite(lv), P[:, 2] - R - lv, 9.0)
        # support needs floor: the step model sometimes carries a marble over an edge still 'supported' (phantom floor;
        # hybrid gate g6r, log 36.5: three planner rolls into holes at 6-7 u/s with no fall predicted). No level within
        # 0.5 u under the bottom = airborne, whatever the model says; the judge then counts an unplanned flight
        sup = np.where(gap > 0.5, 0.0, sup)
        jumped_at = np.where((J > 0) & (jumped_at < 0), t, jumped_at)
        in_air = (sup < 0.5) & (gap > 0.5)
        # leaving the floor without a jump (a lip or corner throws the marble) and bouncing back up after a landing
        # are unplanned flights: the model's edge bounces are not reliable enough to plan on (dev v3 falls)
        unplanned |= in_air & (jumped_at < 0) & ~started_air
        reair |= in_air & (landed >= 0) & (t + 1 > landed + 1)
        in_air_t[:, t] = in_air
        airborne |= in_air
        land_now = airborne & (landed < 0) & (sup >= 0.5) & (last >= 0.5)
        landed = np.where(land_now, t + 1, landed)
        path[:, t + 1] = P; vel[:, t + 1] = V
        sup_t[:, t] = sup; gap_t[:, t] = gap; coll_t[:, t] = coll
        Ul[:, t] = U; Jl[:, t] = J
        Up, Jp = U, J
    # clearance: distance from each supported state to the nearest void / drop edge of its floor (EDGE_REACH at most)
    Q = path[:, 1:].reshape(-1, 3)
    E = eidx.nearest(Q[:, 0], Q[:, 1], Q[:, 2] - R, np.ones(len(Q)), np.zeros(len(Q))).reshape(len(Q), D3.N_EDGE, 7)
    risky = (E[..., 3] + E[..., 4]) > 0
    clear = np.where(risky, E[..., 0], D3.EDGE_REACH).min(1).reshape(n, H + 1)
    clear = np.where(sup_t >= 0.5, clear, np.inf)
    stop_ok = stopping_room(g, path, vel, landed) if STOP_CHECK else np.ones(n, bool)
    return {'path': path, 'vel': vel, 'landed': landed, 'airborne': airborne, 'started_air': started_air, 'stop_ok': stop_ok, 'used_at': used_at,
            'sup': sup_t, 'gap': gap_t, 'coll': coll_t, 'U': Ul, 'J': Jl, 'jumped_at': jumped_at, 'clear_t': clear,
            'unplanned': unplanned, 'reair': reair, 'in_air_t': in_air_t,
            'floor_ref': np.broadcast_to(np.asarray(floor_ref, float), (n,))}


def gem_disc(gem):
    """Source points of a walk to a floor gem: the gem and rings at 0.5 and 1 u (the pickup radius is ~1 u)."""
    a = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    return gem[0] + np.r_[0.0, 0.5 * np.cos(a), np.cos(a)], gem[1] + np.r_[0.0, 0.5 * np.sin(a), np.sin(a)]


def walk_distance_field(g, sx, sy, floor_ref):
    """Walking distance over the floor to the nearest source point (multi-source Dijkstra on the terrain raster): floor
    cells from floor_ref - 0.5 up to floor_ref + 4, kept WALK_CLEAR from anything else; inf elsewhere."""
    from scipy.ndimage import binary_erosion
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    Hh = g.heights
    ok = np.isfinite(Hh) & (Hh >= floor_ref - 0.5) & (Hh <= floor_ref + 4.0)
    trav = binary_erosion(ok.any(0), iterations=max(1, int(round(WALK_CLEAR / g.res))))
    ny, nx = trav.shape
    idx = -np.ones((ny, nx), np.int64); cells = np.nonzero(trav.ravel())[0]; idx.ravel()[cells] = np.arange(len(cells))
    def pair(A, dj, di):                                  # A[j, i] and A[j + dj, i + di] over the valid cells
        if di >= 0:
            return A[:ny - dj, :nx - di], A[dj:, di:]
        return A[:ny - dj, -di:], A[dj:, :nx + di]
    rows, cols, wts = [], [], []
    for dj, di, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, SQ2), (1, -1, SQ2)):
        a, b = pair(trav, dj, di); ia, ib = pair(idx, dj, di)
        m = a & b
        rows.append(ia[m]); cols.append(ib[m]); wts.append(np.full(int(m.sum()), w * g.res))
    r = np.concatenate(rows); c = np.concatenate(cols); w = np.concatenate(wts)
    G = coo_matrix((np.r_[w, w], (np.r_[r, c], np.r_[c, r])), shape=(len(cells), len(cells))).tocsr()
    si = np.clip(np.rint((np.asarray(sy) - g.ys[0]) / g.res).astype(int), 0, ny - 1)
    sj = np.clip(np.rint((np.asarray(sx) - g.xs[0]) / g.res).astype(int), 0, nx - 1)
    src = np.unique(idx[si, sj][idx[si, sj] >= 0])
    D = np.full(ny * nx, np.inf)
    if len(src):
        D[cells] = dijkstra(G, directed=False, indices=src, min_only=True)
    return D.reshape(ny, nx)


def field_at(g, D, x, y):
    """A raster field's value at world (x, y) (nearest cell, clipped to the raster)."""
    i = np.clip(np.rint((np.asarray(y) - g.ys[0]) / g.res).astype(int), 0, D.shape[0] - 1)
    j = np.clip(np.rint((np.asarray(x) - g.xs[0]) / g.res).astype(int), 0, D.shape[1] - 1)
    return D[i, j]


def jump_fires(g, P, V, J, t=3):
    """Rows whose acting jump key fires for certain: the marble resting on flat floor (bottom within 0.05 u of the
    floor, |vz| < 0.3, the floor level within 0.02 u at 0.3 u around). Measured in the stage 3 recordings: in that
    state the jump fired 3,857 times in 3,883 (KOTM, Sprawl, Tilo) with the vertical speed after it exactly JUMP_VZ."""
    m = J > 0
    if not m.any():
        return m
    lv0, _ = D3.level_below(g, P[:, 0], P[:, 1], P[:, 2] + 0.3)
    m &= np.isfinite(lv0) & (P[:, 2] - R - lv0 < 0.05) & (np.abs(V[:, 2]) < 0.3)
    sp = np.hypot(V[:, 0], V[:, 1])
    ux = np.where(sp > 0.1, V[:, 0] / np.maximum(sp, 1e-6), 0.0); uy = np.where(sp > 0.1, V[:, 1] / np.maximum(sp, 1e-6), 0.0)
    # plus up to one decision of travel for presses that act later in the rollout (34.16: at 12 u/s a press timed from
    # the predicted path acted over the void); a press acting on the next step needs no such margin (the state is
    # known; the operator's own crossing pressed with ~1 u of floor left at 13 u/s and the game fired it, log 35.3)
    ahead = JUMP_LIP_U + JUMP_LIP_K * sp * min(1.0, max(0.0, (t - 1) / 2.0))
    # flat around, and the same floor JUMP_LIP_U ahead along the velocity. Stage 3 data (10,971 presses on floor): the
    # jump fires 97-100 % with 0.1 u or more of floor ahead at any speed, 65-93 % at the lip itself; the rest of the
    # margin covers the model's position error (the first version of this prior had none: devb 13/20 vs v8 20/20,
    # every fall a jump pressed at the lip that the game never took)
    for dx, dy in ((0.3, 0.0), (-0.3, 0.0), (0.0, 0.3), (0.0, -0.3), (ux * ahead, uy * ahead), (0.5 * ux * ahead, 0.5 * uy * ahead)):
        if not m.any():
            break
        lv, _ = D3.level_below(g, P[:, 0] + dx, P[:, 1] + dy, P[:, 2] + 0.3)
        m &= np.isfinite(lv) & (np.abs(lv - lv0) < 0.02)
    return m


def closest_pass(path, gem):
    """Closest approach to the gem, the path interpolated between decisions: (dmin, state index after it, per-interval)."""
    a, b = path[:, :-1], path[:, 1:]
    ts = np.linspace(0.0, 1.0, 6)[None, None, :, None]
    q = a[:, :, None, :] + (b - a)[:, :, None, :] * ts
    d = np.linalg.norm(q - gem, axis=3).min(2)
    return d.min(1), d.argmin(1) + 1, d


def stopping_room(g, path, vel, landed, step=0.25):
    """After a landing: is there floor straight ahead (along the landing velocity, at the landing floor's level) for the
    marble to brake to a stop? Programs that never land pass. Geometry only, independent of the model's rollout."""
    n = len(path)
    ok = np.ones(n, bool)
    idx = np.nonzero(landed >= 0)[0]
    if not len(idx):
        return ok
    L = np.minimum(landed[idx], path.shape[1] - 1)
    P = path[idx, L]; V = vel[idx, L]
    spd = np.hypot(V[:, 0], V[:, 1])
    need = STOP_MARGIN * spd * spd / (2.0 * A_STOP) + STOP_PAD
    kmax = int(np.ceil(min(float(need.max()), 16.0) / step)) + 1
    ux = np.where(spd > 0.1, V[:, 0] / np.maximum(spd, 1e-6), 0.0); uy = np.where(spd > 0.1, V[:, 1] / np.maximum(spd, 1e-6), 0.0)
    d = np.arange(1, kmax + 1) * step
    qx = (P[:, 0:1] + ux[:, None] * d[None, :]).ravel(); qy = (P[:, 1:2] + uy[:, None] * d[None, :]).ravel()
    z0 = np.repeat(P[:, 2] - R, kmax)
    lv, _ = D3.level_below(g, qx, qy, z0 + 0.3)
    floor = (np.isfinite(lv) & (lv > z0 - 0.5)).reshape(len(idx), kmax)
    # free distance: up to the first sample without floor at the landing level
    first_gap = np.where(floor.all(1), kmax, np.argmin(floor, axis=1))
    free = first_gap * step
    ok[idx] = (free >= need) | (spd <= 0.1)
    return ok


def judge(sim, gem):
    """Pickup chance and a safe continuation. After a flight the continuation is judged over CONT_WIN decisions from
    the landing (falls, bounces, settling, clearance); what the fixed after-landing input would do later is left to the
    next plans (the planner keeps steering after a landing). A program that never flies is judged to the horizon."""
    path = sim['path']
    n = len(path)
    dmin, t_close, _ = closest_pass(path, gem)
    lnd = sim['landed']
    # last state judged: CONT_WIN after a landing; GROUND_WIN after the closest pass for a program that never flies
    # (stage 6b, log 34.11: judged to the horizon, 95 % of the proposals from a KOTM floor state at 9-16 u/s 'fell',
    # only the braking and hopping ones survived, and the planner alone stopped to turn: 2.4 s legs)
    end = np.where(lnd >= 0, np.minimum(H + 1, lnd + CONT_WIN), np.minimum(H + 1, t_close + GROUND_WIN))
    st = np.arange(H + 2)[None, :]                                              # state index
    fell = ((path[:, :, 2] < sim['floor_ref'][:, None] + R - FALL_DZ) & (st <= end[:, None])).any(1)
    tt = np.arange(H + 1)[None, :]                                              # transition t -> state t + 1
    win = (tt + 1 > end[:, None] - SETTLE_N) & (tt + 1 <= end[:, None])
    ok_t = (sim['sup'] >= 0.5) & (sim['gap'] < 0.5)
    settled = np.where(win, ok_t, True).all(1)
    reair = (sim['in_air_t'] & (lnd[:, None] >= 0) & (tt + 1 > lnd[:, None] + 1) & (tt + 1 <= end[:, None])).any(1)
    safe = ~fell & settled & (~sim['airborne'] | (lnd >= 0)) & ~sim['unplanned'] & ~reair & sim['stop_ok']
    # clearance after the landing (or from SKIP_CLEAR on for a program that never left the floor), to the end of the
    # judged window; clear_t index t is the state after transition t, i.e. state t + 1
    idx = tt + 1
    window = np.where(lnd[:, None] >= 0, (idx >= lnd[:, None]) & (idx <= end[:, None]),
                      (~sim['airborne'])[:, None] & (idx > SKIP_CLEAR) & (idx <= end[:, None]))
    clear = np.where(window, sim['clear_t'], np.inf).min(1)
    c_ok = np.where(lnd >= 0, CLEAR_OK, ROLL_CLEAR_OK)
    q = np.clip((clear - CLEAR_MIN) / (c_ok - CLEAR_MIN), 0.0, 1.0)
    pp = p_pick(dmin)
    return {'dmin': dmin, 't_close': t_close, 'fell': fell, 'settled': settled, 'safe': safe, 'p_pick': pp,
            'clear': clear, 'margin': q, 'p_succ': pp * safe * (0.4 + 0.6 * q)}


class Planner:
    def __init__(self, g, gem, dev=None, seeds=None, use_flight=True, rng=None):
        import torch
        self.dev = dev or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.g = g; self.eidx = D3.EdgeIndex(g)
        self.gem = np.asarray(gem, dtype=np.float64)
        self.steps = FastEnsemble(D3.load_ensemble(self.dev), self.dev)
        self.fmodel = None
        if use_flight:
            from nav.learned_nav import flight3 as F3
            self.F3 = F3
            self.fmodel = F3.load(self.dev, 'flight3.pth')
        self.seeds = seeds
        self.seed_tree = None
        if seeds is not None and len(seeds['x']):
            from scipy.spatial import cKDTree
            self.seed_tree = cKDTree(self._seed_key(seeds['x'], seeds['y'], seeds['vx'], seeds['vy']))
        self.best = None; self.last_kind = None
        self.walk = None; self._walk_cache = {}
        self.cont_next = False       # the target is the NEXT gem, driven toward after a pickup (continue_cost)
        self.time_w = TIME_W         # P(success) given up per decision of a later pickup, choosing among pickup plans
        self.approach_line = False   # approach a floor gem by the straight line (a jump leg) instead of the walk
        self.land_steer = False      # after-landing steering toward the gem in the proposals (floor-gem jump legs)
        self.n_broad = N_BROAD       # broad proposals per decision (a caller may lower it for a floor gem)
        self.p_sat = P_SAT           # P(success) above this counts as this when ranking pickup plans (time decides)
        self.exit_dir = None         # stage 6b two-leg plans: unit xy direction the marble should leave the gem in (toward
                                     # the next, jump gem); pickup plans are rewarded for their velocity along it at the pickup
        self.exit_w = EXIT_W
        self.flight_check = 1.0      # 0: no flight-head launch check (crossing legs, stage 6b: the head is unreliable near
                                     # the far lip, log 33.5, and it kept fast jumps below P_JUMP; robustness still applies)
        self.rng = rng or np.random.default_rng(0)
        self.guide = None
        if USE_GUIDE:
            from nav.learned_nav import guide as GD
            if os.path.exists(GD.MODEL):
                self.guide = GD.Guide(self.dev)

    @staticmethod
    def _seed_key(x, y, vx, vy):
        return np.c_[x, y, 0.25 * np.asarray(vx), 0.25 * np.asarray(vy)]

    def reset(self):
        self.best = None; self.last_kind = None

    def set_target(self, gem, seeds=None, cont_next=False):
        """A new gem to take (whole groups, stage 5): its seeds (None for a floor gem) and a fresh walking field.
        cont_next: the gem is the next one after a pickup, only driven toward (continue_cost), not taken."""
        self.gem = np.asarray(gem, dtype=np.float64)
        self.seeds = seeds
        self.cont_next = bool(cont_next)
        self.time_w = TIME_W; self.approach_line = False; self.land_steer = False; self.p_sat = P_SAT; self.flight_check = 1.0
        self.exit_dir = None; self.exit_w = EXIT_W
        self.seed_tree = None
        if seeds is not None and len(seeds['x']):
            from scipy.spatial import cKDTree
            self.seed_tree = cKDTree(self._seed_key(seeds['x'], seeds['y'], seeds['vx'], seeds['vy']))
        self.walk = None
        self.reset()

    # ------------------------------------------------------------------ proposals
    def _broad(self, n, p, v):
        rng = self.rng
        pr = Programs(n)
        tg = math.atan2(self.gem[1] - p[1], self.gem[0] - p[0])
        hv = math.atan2(v[1], v[0]) if math.hypot(v[0], v[1]) > 0.5 else tg
        base = np.where(rng.random(n) < 0.6, tg, np.where(rng.random(n) < 0.5, hv, rng.uniform(-math.pi, math.pi, n)))
        pr.ang[:] = (base + rng.normal(0, math.radians(30), n))[:, None]
        r = rng.random(n)
        for i in np.nonzero(r < 0.35)[0]:                                # one turn after k decisions
            k = int(rng.integers(1, 24)); pr.ang[i, k:] = tg + rng.normal(0, math.radians(40))
        for i in np.nonzero((r >= 0.35) & (r < 0.5))[0]:                 # brake first, then a heading
            k = int(rng.integers(1, 14)); pr.mode[i, :k] = MODE_BRAKE
        pr.thr[rng.random(n) < 0.15] = 0.6
        for i in np.nonzero(rng.random(n) < 0.75)[0]:
            j = int(rng.integers(0, LAST_JUMP + 1)); pr.jump[i, j] = 1
            q = rng.random()
            if q < 0.12:
                pr.mode[i, j:] = MODE_NONE                               # jump with no air input
            elif q >= 0.55:
                pr.ang[i, j:] = tg + rng.normal(0, math.radians(35))     # air input toward the gem
                if AIR_BRAKE and rng.random() < 0.5:                     # then brake in the air before landing, so
                    m = int(rng.integers(4, 12))                         # the landing has room to stop (stopping_room)
                    pr.mode[i, min(H, j + m):] = MODE_BRAKE
        pr.after[:] = np.where(rng.random(n) < 0.7, MODE_BRAKE, MODE_NONE)
        self._land_steer(pr, tg)
        pr.mode[:, H - TAIL:] = MODE_BRAKE
        return pr

    def _land_steer(self, pr, tg, frac=0.5):
        """Floor-gem jump legs (land_steer): after the landing, steer toward the gem for 2-13 decisions, then brake.
        With only a brake or no input after the landing, 15 of 16 programs through a KOTM corner gem went off the map
        (a fast landing short of the gem cannot brake straight; the human lands and turns): stage 5b, 2026-09-29."""
        if not self.land_steer:
            return
        rng = self.rng; n = len(pr)
        m = rng.random(n) < frac
        k = int(m.sum())
        pr.after[m] = MODE_DIR; pr.after_ang[m] = tg + rng.normal(0, math.radians(25), k)
        pr.after_n[m] = rng.integers(2, 14, k)
        # a brake-turn after the landing: steer well past the gem's direction (60-150 deg, either side), which turns
        # the velocity fastest at speed (turn drill, log 34.8) while shedding it; a fast landing beside the gem has no
        # other way to it (log 34.15: from the operator's 13 u/s crossing every after-landing input passed 3 u wide)
        m2 = m & (rng.random(n) < 0.5)
        k2 = int(m2.sum())
        pr.after_ang[m2] = tg + rng.choice([-1.0, 1.0], k2) * rng.uniform(math.radians(60), math.radians(150), k2)
        pr.after_n[m2] = rng.integers(2, 10, k2)

    def use_oracle(self, p, v, w, tel, Uc, Jc, Up, Jp, floor_ref, pow, airborne=False):
        """The planner as a POWERUP oracle (log 39): fixed programs that fire the held powerup (or the blast) NOW with
        a few air inputs / yaws, judged like any plan, against the same programs without the use. Returns
        (gain, best) with best = {'use': 1|2, 'yaw', 'p_succ', 't_close', 'p_pick'} of the best use and gain = its
        ranked score minus the best no-use program's (positive = the use helps), or (None, None) if nothing applies."""
        held = int(pow.get('held', 0)); meter = float(pow.get('meter', 0.0)); special = bool(pow.get('special', False))
        can_blast = special or meter >= PW.BLAST_REQUIRED
        if held not in (1, 2, 5) and not can_blast:
            return None, None
        tg = math.atan2(self.gem[1] - p[1], self.gem[0] - p[0])
        tgy = PW.SUPER_SPEED_YAW(self.gem[0] - p[0], self.gem[1] - p[1])
        progs = []
        airs = (None, 0.0, math.radians(60), -math.radians(60))
        def base(nn, tag):
            pr = Programs(nn); pr.tag[:] = tag; pr.ang[:] = tg
            for i in range(nn):
                a = airs[i % len(airs)]
                if a is None:
                    pr.mode[i, 1:] = MODE_NONE
                else:
                    pr.ang[i, 1:] = tg + a
                pr.after[i] = MODE_DIR; pr.after_ang[i] = tg; pr.after_n[i] = 8
            pr.mode[:, H - TAIL:] = MODE_BRAKE
            return pr
        # the no-use reference (walking on as the proposals would)
        ref = base(len(airs), 7); progs.append(ref)
        if held == 2:
            yaws = (tgy, tgy + math.radians(30), tgy - math.radians(30), tgy + math.radians(60), tgy - math.radians(60))
            pr = base(len(yaws) * 2, 8); pr.use[:, 0] = 1
            for i in range(len(pr)):
                pr.use_yaw[i, 0] = yaws[i % len(yaws)]
                if i >= len(yaws):
                    pr.mode[i, 1:H - TAIL] = MODE_BRAKE                # boost then brake (a short hop to the gem)
            progs.append(pr)
        if held in (1, 5):
            pr = base(len(airs), 8); pr.use[:, 0] = 1; progs.append(pr)
        if can_blast:
            pr = base(len(airs), 9); pr.use[:, 0] = 2; progs.append(pr)
        pr = Programs.cat(progs); n = len(pr)
        sim = simulate(self.steps, self.g, self.eidx, self.dev, np.tile(p, (n, 1)), np.tile(v, (n, 1)), np.tile(w, (n, 1)),
                       np.tile(Uc, (n, 1)), np.full(n, float(Jc)), np.tile(Up, (n, 1)), np.full(n, float(Jp)), pr, airborne, floor_ref, pow=pow)
        jd = judge(sim, self.gem)
        score = np.minimum(jd['p_succ'], self.p_sat) - self.time_w * jd['t_close']
        no_use = pr.use[:, 0] == 0
        ref_best = float(np.max(np.where(no_use, score, -np.inf)))
        use_rows = np.nonzero(~no_use)[0]
        k = int(use_rows[np.argmax(score[use_rows])])
        best = {'use': int(pr.use[k, 0]), 'yaw': float(pr.use_yaw[k, 0]), 'p_succ': float(jd['p_succ'][k]),
                't_close': int(jd['t_close'][k]), 'p_pick': float(jd['p_pick'][k]), 'tag': int(pr.tag[k]),
                'ref_p': float(np.max(np.where(no_use, jd['p_succ'], 0.0))), 'ref_t': int(jd['t_close'][int(np.argmax(np.where(no_use, score, -np.inf)))])}
        return float(score[k] - ref_best), best

    def jump_oracle(self, p, v, w, tel, Uc, Jc, Up, Jp, floor_ref):
        """The planner as a jump ORACLE for the navigator (log 35.4): no takeover. A few fixed programs that press the
        jump NOW (acting next decision) and fly with one of several air inputs (none, at the gem, 60 deg past it either
        side), then steer to the gem after landing; judged like any plan. Returns (p_succ of the best, its t_close,
        its p_pick) so the caller can set the jump bit on the navigator's own action."""
        tg = math.atan2(self.gem[1] - p[1], self.gem[0] - p[0])
        n = 10
        pr = Programs(n); pr.tag[:] = 6
        pr.ang[:] = tg
        pr.jump[:, 0] = 1
        airs = (None, 0.0, math.radians(60), -math.radians(60), 0.0, math.radians(100), -math.radians(100), None,
                math.pi, math.radians(150))                       # the last two: air brake (log 36.3)
        for i, a in enumerate(airs):
            if a is None:
                pr.mode[i, 1:] = MODE_NONE
            else:
                pr.ang[i, 1:] = tg + a
            pr.after[i] = MODE_DIR; pr.after_ang[i] = tg; pr.after_n[i] = 10 if i < 4 else 4
        pr.after[7] = MODE_BRAKE; pr.after[8] = MODE_BRAKE; pr.after[9] = MODE_BRAKE
        pr.mode[:, H - TAIL:] = MODE_BRAKE
        sim = simulate(self.steps, self.g, self.eidx, self.dev, np.tile(p, (n, 1)), np.tile(v, (n, 1)), np.tile(w, (n, 1)),
                       np.tile(Uc, (n, 1)), np.full(n, float(Jc)), np.tile(Up, (n, 1)), np.full(n, float(Jp)), pr, False, floor_ref)
        jd = judge(sim, self.gem)
        ps = jd['p_succ'] * (sim['jumped_at'] >= 0)
        k = int(np.argmax(ps))
        return float(ps[k]), int(jd['t_close'][k]), float(jd['p_pick'][k])

    def _line(self, n, p, v, floor_ref):
        """Floor-gem jump legs (approach_line): roll along the straight line to the gem and jump shortly before the
        first void on it, air input toward the gem (the broad samples found 2 such plans in 2000 from a KOTM centre
        gem to a corner gem). None when the line crosses no void."""
        rng = self.rng
        d = math.hypot(self.gem[0] - p[0], self.gem[1] - p[1])
        if d < 1.0:
            return None
        m = max(2, int(d / 0.25) + 1)
        t = np.linspace(0.0, 1.0, m)
        lv, _ = D3.level_below(self.g, p[0] + t * (self.gem[0] - p[0]), p[1] + t * (self.gem[1] - p[1]), np.full(m, floor_ref + 1.0))
        void = ~(np.isfinite(lv) & (lv >= floor_ref - 0.6))
        if not void.any():
            return None
        lip = float(np.argmax(void)) * d / (m - 1)
        tg = math.atan2(self.gem[1] - p[1], self.gem[0] - p[0])
        sp = max(4.0, float(np.dot(v[:2], [math.cos(tg), math.sin(tg)])))
        j_lip = lip / sp / 0.064
        pr = Programs(n); pr.tag[:] = 4
        pr.ang[:] = (tg + rng.normal(0, math.radians(4), n))[:, None]
        for i in range(n):
            # anywhere on the run-up: program decision d acts from state d + 1 (after the pending reply), so a press
            # timed at the lip fires over the void (the first version never took off)
            j = int(np.clip(math.floor(j_lip * rng.uniform(0.0, 1.0)) - 1, 0, LAST_JUMP)); pr.jump[i, j] = 1
            q = rng.random()
            if q < 0.1:
                pr.mode[i, j:] = MODE_NONE
            elif q < 0.45:
                pr.ang[i, j:] = tg + rng.normal(0, math.radians(8))
            elif q < 0.75:
                # air input well past the gem's direction (30-110 deg, either side): sheds speed and bends the flight
                # toward the gem; the human's recorded crossing held ~100 deg off the velocity in the air (log 34.15)
                pr.ang[i, j:] = tg + rng.choice([-1.0, 1.0]) * rng.uniform(math.radians(30), math.radians(110))
            else:
                # AIR BRAKE (log 36.3): the input against the flight direction decelerates 7 u/s^2 in the air (engine
                # airAcceleration 5 x sqrt 2), a 0.9 s flight lands 6 u/s slower; a slide landing on a 7 u block at
                # 18 u/s cannot stop (ring drill v5), at 9-12 it can
                pr.ang[i, j:] = tg + rng.choice([-1.0, 1.0]) * rng.uniform(math.radians(130), math.radians(180))
        pr.after[:] = np.where(rng.random(n) < 0.7, MODE_BRAKE, MODE_NONE)
        self._land_steer(pr, tg, frac=0.7)
        pr.mode[:, H - TAIL:] = MODE_BRAKE
        return pr

    def _seeded(self, n, p, v):
        """Aimed at seed launch states: drive to a point behind the seed, turn onto its heading, fly its jump."""
        s = self.seeds; rng = self.rng
        pr = Programs(n); pr.tag[:] = 1
        d = np.hypot(s['x'] - p[0], s['y'] - p[1])
        wgt = s['p'] / (0.5 + d) ** 2
        pick = rng.choice(len(d), size=n, p=wgt / wgt.sum())
        sp = max(3.0, math.hypot(v[0], v[1]))
        hs_all = np.arctan2(s['vy'][pick], s['vx'][pick]); ssp_all = np.maximum(3.0, np.hypot(s['vx'][pick], s['vy'][pick]))
        # run-up long enough to reach the seed's speed (from rest ~0.7 u/s per decision, ~10 u/s^2: dev drills; a
        # 0.5-3 u run-up left a marble resting beside its seeds unable to connect: dev v4)
        back_all = np.clip(ssp_all ** 2 / 20.0 * rng.uniform(0.6, 1.4, n) + 0.5, 0.5, 8.0)
        ax_all = s['x'][pick] - back_all * np.cos(hs_all); ay_all = s['y'][pick] - back_all * np.sin(hs_all)
        if self.guide is not None:
            # time to the run-up point from PPO's play (the time-to-gem guide): proposal timing only, never a cost
            t_run = self.guide(np.full(n, p[0]), np.full(n, p[1]), np.full(n, v[0]), np.full(n, v[1]), np.full(n, v[2]),
                               np.ones(n), ax_all, ay_all)
        else:
            t_run = np.hypot(ax_all - p[0], ay_all - p[1]) / max(4.0, sp)
        for r, k in enumerate(pick):
            sx, sy = s['x'][k], s['y'][k]
            hs = float(hs_all[r]); ssp = float(ssp_all[r]); back = float(back_all[r])
            ax, ay = float(ax_all[r]), float(ay_all[r])
            k0 = 0
            if rng.random() < 0.3:                                       # brake first
                k0 = int(min(12, math.hypot(v[0], v[1]) / 0.9)); pr.mode[r, :k0] = MODE_BRAKE
            k1 = int(np.clip(k0 + round(max(0.0, float(t_run[r])) / 0.064 * rng.uniform(0.7, 1.3)), k0, H - TAIL))
            k2 = int(np.clip(k1 + round(back / max(3.0, 0.6 * ssp) / 0.064 * rng.uniform(0.7, 1.3)), k1, H - TAIL))
            pr.ang[r, k0:k1] = math.atan2(ay - p[1], ax - p[0]) + rng.normal(0, 0.08)
            pr.ang[r, k1:] = hs + rng.normal(0, 0.06)
            j = k2 + int(s['j'][k])
            if j <= LAST_JUMP:
                pr.jump[r, j] = 1
                if np.isnan(s['air'][k]):
                    pr.mode[r, j:] = MODE_NONE
                else:
                    pr.ang[r, j:] = hs + s['air'][k]
                bk = s['brake'][k] if ('brake' in s and AIR_BRAKE) else np.nan
                if np.isfinite(bk):                                      # the seed's air brake after the gem
                    pr.mode[r, min(H, j + int(bk)):] = MODE_BRAKE
        pr.after[:] = MODE_BRAKE
        pr.mode[:, H - TAIL:] = MODE_BRAKE
        return pr

    def _air(self, n, p, v):
        rng = self.rng
        pr = Programs(n); pr.tag[:] = 3
        tg = math.atan2(self.gem[1] - p[1], self.gem[0] - p[0])
        hv = math.atan2(v[1], v[0]) if math.hypot(v[0], v[1]) > 0.5 else tg
        a1 = np.where(rng.random(n) < 0.5, tg, hv) + rng.uniform(-math.pi, math.pi, n) * rng.random(n)
        pr.ang[:] = a1[:, None]
        for i in range(n):
            if rng.random() < 0.4:
                k = int(rng.integers(1, 16)); pr.ang[i, k:] = rng.uniform(-math.pi, math.pi)
        pr.mode[rng.random(n) < 0.1] = MODE_NONE
        for i in np.nonzero(rng.random(n) < (0.3 if AIR_BRAKE else 0.0))[0]:   # air brake from some decision on
            pr.mode[i, int(rng.integers(0, 12)):] = MODE_BRAKE
        pr.after[:] = np.where(rng.random(n) < 0.7, MODE_BRAKE, MODE_NONE)
        pr.after_ang[:] = pr.ang[:, -1]
        self._land_steer(pr, tg)
        return pr

    def _warm(self, n):
        if self.best is None or n <= 0:
            return None
        rng = self.rng
        k = Programs.cat([self.best.shifted()] * (n + 1))
        k.tag[:] = 2
        k.ang[1:] += rng.normal(0, math.radians(6), (n, 1)) + rng.normal(0, math.radians(3), (n, H))
        jj = np.nonzero(k.jump[0])[0]
        if len(jj):
            j = int(jj[0])
            for i in range(1, n + 1):
                if rng.random() < 0.5:
                    j2 = int(np.clip(j + rng.integers(-2, 3), 0, H - 1)); k.jump[i] = 0; k.jump[i, j2] = 1
        return k

    def propose(self, p, v, airborne, floor_ref=None):
        parts = []
        w = self._warm(N_WARM)
        if w is not None:
            parts.append(w)
        if airborne:
            parts.append(self._air(N_AIR, p, v))
        else:
            parts.append(self._broad(self.n_broad, p, v))
            if self.seed_tree is not None:
                parts.append(self._seeded(N_SEED, p, v))
            if self.approach_line and floor_ref is not None:
                ln = self._line(N_LINE, p, v, floor_ref)
                if ln is not None:
                    parts.append(ln)
            if self.exit_dir is not None:
                parts.append(self._exitline(N_EXIT, p, v))
        return Programs.cat(parts)

    def jump_charge(self, sim):
        """Ranking charge for a plan that jumps on a plain floor leg (no seeds, no line crossing): a jump on the floor
        costs control and speed for nothing (planner-alone rounds: 143 jumps a round, log 34.10)."""
        if self.seeds is not None or self.approach_line:
            return 0.0
        return FLOOR_JUMP_CHARGE * (sim['jumped_at'] >= 0)

    def exit_bonus(self, sim, jd):
        """Two-leg plans (exit_dir set): the ranking is charged the TURN the next leg will cost from the state at the
        closest pass, from the turn drill (log 34.8: TURN_S x (1 - cos) at speed, scaled down at low speed), in the
        same units as the time weight (P per decision). A plan that leaves the gem aligned costs nothing; one that
        flies through at 11 u/s with the next gem behind is charged the ~1.5 s turn, so arriving slower can win.
        The earlier bonus (velocity along the exit line) was blind to gems behind the marble (log 34.12)."""
        n = len(jd['t_close'])
        if self.exit_dir is None:
            return np.zeros(n)
        t = np.clip(jd['t_close'], 0, sim['vel'].shape[1] - 1)
        v = sim['vel'][np.arange(n), t, :2]
        sp = np.linalg.norm(v, axis=1)
        cos = np.where(sp > 0.5, (v[:, 0] * self.exit_dir[0] + v[:, 1] * self.exit_dir[1]) / np.maximum(sp, 1e-6), 1.0)
        turn_s = TURN_S * (1.0 - cos) * np.minimum(1.0, sp / TURN_REF_SPEED)
        return -self.time_w * turn_s / 0.064

    def _exitline(self, n, p, v):
        """Two-leg plans: drive to a point BACK u behind the gem on the exit line, turn onto the line, roll through the
        gem at full throttle (a run-up onto the crossing); no jump. The broad samples rarely contain this shape."""
        rng = self.rng
        pr = Programs(n); pr.tag[:] = 5
        u = np.asarray(self.exit_dir, float)
        sp = max(3.0, math.hypot(v[0], v[1]))
        for i in range(n):
            back = rng.uniform(1.5, 6.0)
            wx, wy = self.gem[0] - back * u[0], self.gem[1] - back * u[1]
            d = math.hypot(wx - p[0], wy - p[1])
            k1 = int(np.clip(round(d / sp / 0.064 * rng.uniform(0.6, 1.1)), 0, H - TAIL))
            a1 = math.atan2(wy - p[1], wx - p[0]) + rng.normal(0, 0.1)
            a2 = math.atan2(u[1], u[0]) + rng.normal(0, 0.08)
            pr.ang[i, :k1] = a1; pr.ang[i, k1:] = a2
            if rng.random() < 0.3:                                       # over-steer into the turn (turn drill: fastest)
                k0 = max(0, k1 - int(rng.integers(2, 6))); pr.ang[i, k0:k1] = a2 + np.sign((a2 - a1 + math.pi) % (2 * math.pi) - math.pi) * rng.uniform(0.4, 0.9)
        pr.after[:] = MODE_DIR; pr.after_ang[:] = math.atan2(u[1], u[0]); pr.after_n[:] = H + 1
        return pr

    # ------------------------------------------------------------------ flight-head launch check
    def flight_safe(self, p, v, w, tel, sim, idx):
        """Flight-head P(safe) for programs idx, from the observed state with the pending reply first."""
        import torch
        F3 = self.F3
        feats = []
        for i in idx:
            U = sim['U'][i, :F3.H + 1]; J = sim['J'][i, :F3.H + 1]
            if math.hypot(v[0], v[1]) > 0.5:
                y = math.atan2(v[1], v[0])
            else:
                u = U[1]; y = math.atan2(u[1], u[0]) if math.hypot(u[0], u[1]) > 0.1 else 0.0
            feats.append(F3.sample_features(self.g, p, v, w, tel, y, U, J))
        with torch.no_grad():
            o = self.fmodel(torch.as_tensor(np.asarray(feats, np.float32), device=self.dev))
        return torch.sigmoid(o['safe']).cpu().numpy()

    # ------------------------------------------------------------------ one decision
    def plan(self, p, v, w, tel, Uc, Jc, Up, Jp, airborne, picked, floor_ref):
        """The next reply. p, v, w: observed state; tel: its contact telemetry (or None); Uc, Jc: the reply sent
        last decision (acts next); Up, Jp: the one before it; picked: the gem has been taken."""
        t0 = time.perf_counter()
        p = np.asarray(p, float); v = np.asarray(v, float); w = np.asarray(w, float)
        progs = self.propose(p, v, airborne, floor_ref)
        n = len(progs)
        sim = simulate(self.steps, self.g, self.eidx, self.dev, np.tile(p, (n, 1)), np.tile(v, (n, 1)), np.tile(w, (n, 1)),
                       np.tile(Uc, (n, 1)), np.full(n, float(Jc)), np.tile(Up, (n, 1)), np.full(n, float(Jp)),
                       progs, airborne, floor_ref)
        jd = judge(sim, self.gem)
        info = {'n': n, 'airborne': bool(airborne)}
        jump_now = progs.jump[:, 0] > 0
        rob = lambda idx: self.robust(idx, p, v, w, Uc, Jc, Up, Jp, progs, airborne, floor_ref)
        if picked:
            cost = self.continue_cost(sim, jd, jump_now)
            top = np.argsort(cost)[:K_ROBUST]
            sim2, jd2 = rob(top)
            worst = self.continue_cost(sim2, jd2, np.repeat(jump_now[top], N_VAR)).reshape(len(top), N_VAR).max(1)
            k = int(top[int(np.argmin(np.maximum(cost[top], worst)))]); kind = 'continue'
        else:
            ps = jd['p_succ'].copy()
            if not airborne and jump_now.any():
                # a jump from the floor now: re-judge its landing with the flight head from the observed state
                cand = np.nonzero(jump_now & (ps >= 0.5 * P_GO))[0]
                if self.fmodel is not None and self.flight_check > 0 and len(cand):
                    cand = cand[np.argsort(-ps[cand])[:16]]
                    fs = self.flight_safe(p, v, w, tel, sim, cand)
                    ps[cand] = jd['p_pick'][cand] * fs * jd['safe'][cand] * (0.4 + 0.6 * jd['margin'][cand])
                    info['flight_checked'] = int(len(cand)); info['flight_best'] = float(fs.max())
                    unchecked = jump_now.copy(); unchecked[cand] = False
                    ps[unchecked] = 0.0
                ps[jump_now & (ps < P_JUMP)] = 0.0
            kind = None
            info['p_nominal'] = float(ps.max())
            # commitment: the last pickup plan, shifted one decision (program 0 when there is one), is always
            # re-checked and kept while it stays above P_KEEP, unless another beats it by SWITCH. Without it, plans
            # appeared and vanished between decisions and 7 of the 16 M3 test failures were stalls
            inc = 0 if (self.best is not None and self.last_kind == 'pickup') else None
            keep = inc is not None and ps[inc] >= P_KEEP
            if ps.max() >= P_GO or keep:
                # robustness: the best candidates again from perturbed starts; P(success) is the mean pickup chance
                # times the mean safety x margin over the variants
                # ranked among the programs that can succeed (with a larger time_w, programs that never reach the gem
                # but end early filled the top K and left none: stage 5b route run, 2026-09-29)
                psr = np.minimum(ps, self.p_sat) + self.exit_bonus(sim, jd) - self.jump_charge(sim)
                top = np.argsort(-np.where(ps > 0, psr - self.time_w * jd['t_close'], -np.inf))[:K_ROBUST]
                top = top[ps[top] > 0]
                if keep and inc not in top:
                    top = np.r_[top, inc]
                sim2, jd2 = rob(top)
                pp2 = jd2['p_pick'].reshape(len(top), N_VAR).mean(1)
                # the MEAN over the variants (the worst case left almost nothing to fly: dev drills v2, 2026-09-28)
                sf2 = (jd2['safe'] * (0.4 + 0.6 * jd2['margin'])).reshape(len(top), N_VAR).mean(1)
                ps_top = np.minimum(ps[top], 0.5 * (jd['p_pick'][top] + pp2) * sf2)
                ps_top[jump_now[top] & (ps_top < P_JUMP)] = 0.0
                ps = np.zeros(n); ps[top] = ps_top
                psr = np.minimum(ps, self.p_sat) + self.exit_bonus(sim, jd) - self.jump_charge(sim)
                k = int(np.argmax(np.where(ps > 0, psr - self.time_w * jd['t_close'], -1.0)))
                if keep and ps[inc] >= P_KEEP and psr[k] - self.time_w * jd['t_close'][k] < psr[inc] - self.time_w * jd['t_close'][inc] + SWITCH:
                    k = inc
                if ps[k] >= P_GO or (k == inc and keep and ps[inc] >= P_KEEP):
                    kind = 'pickup'
                info['kept'] = bool(k == inc)
            if kind is None:
                cost = self.seed_cost(sim, jd) + APPROACH_JUMP * (sim['jumped_at'] >= 0)
                top = np.argsort(cost)[:K_ROBUST]
                sim2, jd2 = rob(top)
                fell2 = jd2['fell'].reshape(len(top), N_VAR).any(1)
                # a jump without a pickup plan must be safe from every perturbed start (M3 test: such jumps took the
                # gem about half the time and caused falls)
                unsafe_jump = (sim['jumped_at'][top] >= 0) & ~jd2['safe'].reshape(len(top), N_VAR).all(1)
                veto = fell2 | unsafe_jump
                k = int(top[int(np.argmin(cost[top] + 100.0 * veto))]); kind = 'approach'
                if jump_now[k] and veto[list(top).index(k)]:
                    # a jump that failed the check is replaced by the best program that does not jump now (dev v6: three
                    # such jumps fell), unless that one falls too: then the jump is the only way out (dev v7: marbles
                    # committed to a run-up rolled into the hole when the jump was taken away)
                    alt = int(np.argmin(np.where(jump_now, np.inf, cost)))
                    if not jd['fell'][alt]:
                        k = alt
                    info['escape_jump'] = bool(jd['fell'][alt])
                if APPROACH_JUMPS == 'escape' and jump_now[k] and not airborne:
                    alt = int(np.argmin(np.where(jump_now, np.inf, cost)))
                    if not jd['fell'][alt]:
                        k = alt                                          # a jump only when every alternative falls
                if jump_now[k] and not airborne and self.fmodel is not None and FS_APPROACH > 0:
                    # every jump from the floor passes the observed-state launch check, not only pickup jumps (stage 5
                    # gate: jumps without a pickup plan, sent from the lip at run-up speed, fell in with the gem)
                    fs = float(self.flight_safe(p, v, w, tel, sim, [k])[0])
                    info['approach_flight_safe'] = fs
                    if fs < FS_APPROACH:
                        alt = int(np.argmin(np.where(jump_now, np.inf, cost)))
                        if not jd['fell'][alt]:
                            k = alt
            info['p_succ'] = float(ps[k]); info['p_best'] = float(ps.max()); info['n_go'] = int((ps >= P_GO).sum())
        self.best = progs.take(np.array([k])); self.last_kind = kind
        info.update({'kind': kind, 'dmin': float(jd['dmin'][k]), 't_close': int(jd['t_close'][k]),
                     'safe': bool(jd['safe'][k]), 'fell': bool(jd['fell'][k]), 'clear': float(min(jd['clear'][k], 9.0)),
                     'landed': int(sim['landed'][k]), 'jump_at': int(sim['jumped_at'][k]),
                     'tag': int(progs.tag[k]), 'ms': 1000.0 * (time.perf_counter() - t0),
                     'pred_path': sim['path'][k].round(3)})
        return self.reply(progs, k, v), info

    def reply(self, progs, k, v):
        m, a, th, jmp = int(progs.mode[k, 0]), float(progs.ang[k, 0]), float(progs.thr[k, 0]), int(progs.jump[k, 0])
        if m == MODE_DIR:
            js = action_to_joystick(math.cos(a), math.sin(a), th, jmp, 0, v[0], v[1])
        elif m == MODE_BRAKE:
            js = action_to_joystick(0.0, 0.0, 1.0, jmp, 1, v[0], v[1])
        else:
            js = action_to_joystick(0.0, 0.0, 0.0, jmp, 0, 0.0, 0.0)
        return tuple(js)

    def robust(self, idx, p, v, w, Uc, Jc, Up, Jp, progs, airborne, floor_ref):
        """Programs idx rolled out again from N_VAR perturbed starts (speed x0.96 / x1.04, heading -2 / +2 deg, spin
        turned with the velocity): stands in for model and state error. Rows: program-major, N_VAR each."""
        K = len(idx)
        Vs, Ws = [], []
        for kind, a in VARIANTS:
            if kind == 's':
                Vs.append(v * np.array([a, a, 1.0])); Ws.append(w * a)
            else:
                c, s = math.cos(a), math.sin(a)
                Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                Vs.append(Rz @ v); Ws.append(Rz @ w)
        m = K * N_VAR
        sub = progs.take(np.repeat(idx, N_VAR))
        sim2 = simulate(self.steps, self.g, self.eidx, self.dev, np.tile(p, (m, 1)), np.tile(np.asarray(Vs), (K, 1)),
                        np.tile(np.asarray(Ws), (K, 1)), np.tile(Uc, (m, 1)), np.full(m, float(Jc)), np.tile(Up, (m, 1)),
                        np.full(m, float(Jp)), sub, airborne, floor_ref)
        return sim2, judge(sim2, self.gem)

    def continue_cost(self, sim, jd, jump_now):
        """After the pickup: stay on the floor, away from edges; no new jumps. Without a next gem, slow down (a hybrid
        hands back to the navigator). With one (cont_next): keep rolling toward it, costed as the approach to a floor
        gem (the least walking distance left over the path, sooner preferred): the stage 5b shortcuts braked 6-25
        decisions after every pickup before handing back and lost what the jump had saved."""
        base = (100.0 * jd['fell'] + 20.0 * (~jd['settled']) + 10.0 * np.maximum(0.0, 1.5 - jd['clear']) + CONT_JUMP_NOW * jump_now)
        if not self.cont_next:
            speed_end = np.linalg.norm(sim['vel'][:, -1, :2], axis=1)
            return base + CONT_SPEED_COST * speed_end
        path = sim['path']; n = len(path)
        walk = self.walk_at(path[:, 1:, 0].ravel(), path[:, 1:, 1].ravel(), float(sim['floor_ref'][0])).reshape(n, -1)
        straight = np.linalg.norm(path[:, 1:, :2] - self.gem[:2], axis=2)
        prog = (np.where(np.isfinite(walk), walk, straight + 5.0) + 0.03 * np.arange(path.shape[1] - 1)[None, :]).min(1)
        return base + prog + CONT_JUMP * (sim['jumped_at'] >= 0)

    def walk_field(self, floor_ref):
        """Walking distance over the floor to the nearest seed (multi-source Dijkstra on the terrain raster): floor
        cells from floor_ref - 0.5 up, kept WALK_CLEAR from anything else. The approach cost uses it so a marble on
        the far side of a hole goes around it (straight-line seed distance pulled it into the hole's edge: dev v3)."""
        if self.walk is not None:
            return self.walk
        ck = (tuple(np.round(self.gem, 2)), round(float(floor_ref), 2), self.seeds is None)
        if ck in self._walk_cache:
            self.walk = self._walk_cache[ck]
            return self.walk
        if self.seeds is not None and len(self.seeds['x']):
            sx, sy = self.seeds['x'], self.seeds['y']
        else:                                                    # a floor gem: walk to the gem itself (a 1 u disc)
            sx, sy = gem_disc(self.gem)
        self.walk = walk_distance_field(self.g, sx, sy, floor_ref)
        self._walk_cache[ck] = self.walk
        return self.walk

    def walk_at(self, x, y, floor_ref):
        return field_at(self.g, self.walk_field(floor_ref), x, y)

    def seed_cost(self, sim, jd):
        """Approach cost: how close the predicted path comes to a seed launch state (sooner is better): the larger of
        the walking distance to the nearest seed and the position-velocity distance to the nearest seed state."""
        path, vel = sim['path'], sim['vel']
        n = len(path)
        if self.seed_tree is None:
            # no seeds (a floor gem): walking distance to it (straight-line where off the walkable floor)
            walk = self.walk_at(path[:, 1:, 0].ravel(), path[:, 1:, 1].ravel(), float(sim['floor_ref'][0])).reshape(n, -1)
            straight = np.linalg.norm(path[:, 1:, :2] - self.gem[:2], axis=2)
            # a jump leg (approach_line) closes the straight line on the floor: toward the lip, not round the hole
            on_floor = walk if not self.approach_line else np.where(np.isfinite(walk), straight, np.inf)
            c = (np.where(np.isfinite(on_floor), on_floor, straight + 5.0) + 0.03 * np.arange(path.shape[1] - 1)[None, :]).min(1)
        else:
            q = self._seed_key(path[:, 1:, 0].ravel(), path[:, 1:, 1].ravel(), vel[:, 1:, 0].ravel(), vel[:, 1:, 1].ravel())
            dist, _ = self.seed_tree.query(q)
            walk = self.walk_at(path[:, 1:, 0].ravel(), path[:, 1:, 1].ravel(), float(sim['floor_ref'][0]))
            dist = np.maximum(dist, walk)                    # inf where off the walkable floor (airborne over a hole)
            c = (dist.reshape(n, -1) + 0.03 * np.arange(path.shape[1] - 1)[None, :]).min(1)
            c = np.where(np.isfinite(c), c, 50.0)
        c_ok = np.where(sim['landed'] >= 0, CLEAR_OK, ROLL_CLEAR_OK)
        return c + 100.0 * jd['fell'] + 20.0 * (~jd['safe']) + 5.0 * np.maximum(0.0, c_ok - jd['clear'])
