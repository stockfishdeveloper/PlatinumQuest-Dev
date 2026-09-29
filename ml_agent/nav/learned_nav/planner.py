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
from nav.joystick import action_to_joystick                                      # noqa: E402

R = 0.19
SQ2 = math.sqrt(2.0)
H = 40                       # program decisions (2.56 s) after the pending reply
N_BROAD = 320
N_SEED = 96
N_WARM = 40
N_AIR = 256
PICK_R0, PICK_A = 0.85, 4.0  # P(pickup) = sigmoid(PICK_A (PICK_R0 - closest pass)), P0 cross-check fit (rollouts)
P_GO = 0.25                  # below this best P(success) the planner approaches instead (0.30 until dev v3)
P_JUMP = 0.35                # a jump from the floor is sent only above this (after the flight-head check); dev v3:
                             # all 70 floor jumps sent at >= 0.45 took the gem, so 0.45 was conservative
LAST_JUMP = 22               # latest jump decision in a program: time left in the horizon to land and settle
SETTLE_N = 4                 # decisions on a floor at the end of the horizon for a safe continuation
TAIL = 10                    # every ground program brakes for its last TAIL decisions: the plan must still be able to
                             # stop (terminal check; without it a program rolling toward a hole just past the horizon
                             # looked safe, first dev drills 2026-09-28)
CLEAR_OK = 1.0               # on-floor clearance from void/drop edges after a landing (or while rolling) counted full
CLEAR_MIN = 0.2              # ... and none at all below this (landing errors ~1 u: stage 3 flight eval, first drills)
SKIP_CLEAR = 3               # the first decisions of a rolling program are not charged (the marble may start near a lip)
FALL_DZ = 0.5                # below the start floor level by this much: fallen (2.0 until the first drill: a pocket
                             # under a hole's lip 1 u down counted as safe, and the marble was stuck there)
K_ROBUST = 12                # best candidates re-checked from perturbed starts (first dev drills: landings 0.1-0.3 u
                             # past a hole's far lip clipped it in the game and bounced back in)
VARIANTS = (('s', 0.96), ('s', 1.04), ('h', math.radians(2.0)), ('h', -math.radians(2.0)))
N_VAR = len(VARIANTS)
APPROACH_JUMP = 3.0          # approach cost of a jump that takes no gem (u of seed distance)
WALK_CLEAR = 0.4             # the walking-distance field keeps this far from any edge of the floor
USE_GUIDE = True             # time the seed-aimed run-ups with the time-to-gem guide (guide.py); proposals only.
                             # The M3 gate ran with it OFF (v5, 93.1 %). Dev A/B after the gate (log 31.8): 98/105
                             # with it against 93/105 without, pickups ~0.26 s sooner (not significant: 10 vs 5)
MODE_DIR, MODE_BRAKE, MODE_NONE = 0, 1, 2


def reply_vector(js):
    """World force vector and jump key of a joystick reply (fwd, back, left, right, jump, cam_yaw): what the step
    model was trained on (dynamics3: right - left, fwd - back)."""
    return np.array([js[3] - js[2], js[0] - js[1]], dtype=np.float64), float(js[4])


def p_pick(dmin):
    return 1.0 / (1.0 + np.exp(-PICK_A * (PICK_R0 - dmin)))


class Programs:
    """n programs of H decisions: mode, direction (world rad), throttle, jump key; after-landing mode/direction."""
    KEYS = ('mode', 'ang', 'thr', 'jump', 'after', 'after_ang', 'tag')

    def __init__(self, n):
        self.mode = np.full((n, H), MODE_DIR, np.int8); self.ang = np.zeros((n, H)); self.thr = np.ones((n, H))
        self.jump = np.zeros((n, H), np.int8); self.after = np.full(n, MODE_BRAKE, np.int8); self.after_ang = np.zeros(n)
        self.tag = np.zeros(n, np.int8)                      # 0 broad, 1 seed, 2 warm, 3 air

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
        """The same programs one decision later (the first decision was executed)."""
        out = self.take(slice(None))
        for k in ('mode', 'ang', 'thr', 'jump'):
            a = getattr(out, k)
            a[:, :-1] = a[:, 1:].copy()
            if k == 'jump':
                a[:, -1] = 0
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


def simulate(steps, g, eidx, dev, P, V, W, Uc, Jc, Up, Jp, progs, airborne0, floor_ref):
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
    unplanned = np.zeros(n, bool); reair = np.zeros(n, bool)
    for t in range(H + 1):
        if t == 0:
            U = Uc.copy(); J = Jc.copy()
        else:
            d = t - 1
            post = (landed >= 0) & (landed <= d)
            md = np.where(post, progs.after, progs.mode[:, d]); an = np.where(post, progs.after_ang, progs.ang[:, d])
            th = np.where(post, 1.0, progs.thr[:, d])
            U = world_input(md, an, th, vel[:, d])
            J = np.where(post, 0.0, progs.jump[:, d].astype(float))
        F, yaw = D3.features(g, eidx, P, V, W, U, J, Up, Jp)
        pr = E3.predict(steps, F, dev)
        P, V, W = D3.apply_step(P, V, W, pr['mu'], yaw)
        sup = pr['p'][:, 0]; coll = pr['p'][:, 1]; last = pr['p'][:, 2]
        lv, _ = D3.level_below(g, P[:, 0], P[:, 1], P[:, 2] + 0.3)
        gap = np.where(np.isfinite(lv), P[:, 2] - R - lv, 9.0)
        jumped_at = np.where((J > 0) & (jumped_at < 0), t, jumped_at)
        in_air = (sup < 0.5) & (gap > 0.5)
        # leaving the floor without a jump (a lip or corner throws the marble) and bouncing back up after a landing
        # are unplanned flights: the model's edge bounces are not reliable enough to plan on (dev v3 falls)
        unplanned |= in_air & (jumped_at < 0) & ~started_air
        reair |= in_air & (landed >= 0) & (t + 1 > landed + 1)
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
    return {'path': path, 'vel': vel, 'landed': landed, 'airborne': airborne, 'started_air': started_air,
            'sup': sup_t, 'gap': gap_t, 'coll': coll_t, 'U': Ul, 'J': Jl, 'jumped_at': jumped_at, 'clear_t': clear,
            'unplanned': unplanned, 'reair': reair,
            'floor_ref': np.broadcast_to(np.asarray(floor_ref, float), (n,))}


def closest_pass(path, gem):
    """Closest approach to the gem, the path interpolated between decisions: (dmin, state index after it, per-interval)."""
    a, b = path[:, :-1], path[:, 1:]
    ts = np.linspace(0.0, 1.0, 6)[None, None, :, None]
    q = a[:, :, None, :] + (b - a)[:, :, None, :] * ts
    d = np.linalg.norm(q - gem, axis=3).min(2)
    return d.min(1), d.argmin(1) + 1, d


def judge(sim, gem):
    path = sim['path']
    dmin, t_close, _ = closest_pass(path, gem)
    fell = (path[:, :, 2] < sim['floor_ref'][:, None] + R - FALL_DZ).any(1)
    settled = (sim['sup'][:, -SETTLE_N:] >= 0.5).all(1) & (sim['gap'][:, -SETTLE_N:] < 0.5).all(1)
    safe = ~fell & settled & (~sim['airborne'] | (sim['landed'] >= 0)) & ~sim['unplanned'] & ~sim['reair']
    # clearance after the landing (or from SKIP_CLEAR on for a program that never left the floor); clear_t index
    # t is the state after transition t, i.e. state t + 1
    idx = np.arange(H + 1)[None, :] + 1
    lnd = sim['landed'][:, None]
    window = np.where(lnd >= 0, idx >= lnd, (~sim['airborne'])[:, None] & (idx > SKIP_CLEAR))
    clear = np.where(window, sim['clear_t'], np.inf).min(1)
    q = np.clip((clear - CLEAR_MIN) / (CLEAR_OK - CLEAR_MIN), 0.0, 1.0)
    pp = p_pick(dmin)
    return {'dmin': dmin, 't_close': t_close, 'fell': fell, 'settled': settled, 'safe': safe, 'p_pick': pp,
            'clear': clear, 'margin': q, 'p_succ': pp * safe * (0.4 + 0.6 * q)}


class Planner:
    def __init__(self, g, gem, dev=None, seeds=None, use_flight=True, rng=None):
        import torch
        self.dev = dev or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.g = g; self.eidx = D3.EdgeIndex(g)
        self.gem = np.asarray(gem, dtype=np.float64)
        self.steps = D3.load_ensemble(self.dev)
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
        self.best = None
        self.walk = None
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
        self.best = None

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
        pr.after[:] = np.where(rng.random(n) < 0.7, MODE_BRAKE, MODE_NONE)
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
        pr.after[:] = np.where(rng.random(n) < 0.7, MODE_BRAKE, MODE_NONE)
        pr.after_ang[:] = pr.ang[:, -1]
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

    def propose(self, p, v, airborne):
        parts = []
        w = self._warm(N_WARM)
        if w is not None:
            parts.append(w)
        if airborne:
            parts.append(self._air(N_AIR, p, v))
        else:
            parts.append(self._broad(N_BROAD, p, v))
            if self.seed_tree is not None:
                parts.append(self._seeded(N_SEED, p, v))
        return Programs.cat(parts)

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
        progs = self.propose(p, v, airborne)
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
                if self.fmodel is not None and len(cand):
                    cand = cand[np.argsort(-ps[cand])[:16]]
                    fs = self.flight_safe(p, v, w, tel, sim, cand)
                    ps[cand] = jd['p_pick'][cand] * fs * jd['safe'][cand] * (0.4 + 0.6 * jd['margin'][cand])
                    info['flight_checked'] = int(len(cand)); info['flight_best'] = float(fs.max())
                    unchecked = jump_now.copy(); unchecked[cand] = False
                    ps[unchecked] = 0.0
                ps[jump_now & (ps < P_JUMP)] = 0.0
            kind = None
            info['p_nominal'] = float(ps.max())
            if ps.max() >= P_GO:
                # robustness: the best candidates again from perturbed starts; P(success) is the mean pickup chance
                # times the WORST safety x margin (a landing just past a lip fails under a 4 % slower start)
                top = np.argsort(-(ps - 0.004 * jd['t_close']))[:K_ROBUST]
                top = top[ps[top] > 0]
                sim2, jd2 = rob(top)
                pp2 = jd2['p_pick'].reshape(len(top), N_VAR).mean(1)
                # the MEAN over the variants (the worst case left almost nothing to fly: dev drills v2, 2026-09-28)
                sf2 = (jd2['safe'] * (0.4 + 0.6 * jd2['margin'])).reshape(len(top), N_VAR).mean(1)
                ps_top = np.minimum(ps[top], 0.5 * (jd['p_pick'][top] + pp2) * sf2)
                ps_top[jump_now[top] & (ps_top < P_JUMP)] = 0.0
                ps = np.zeros(n); ps[top] = ps_top
                if ps.max() >= P_GO:
                    k = int(np.argmax(np.where(ps > 0, ps - 0.004 * jd['t_close'], -1.0))); kind = 'pickup'
            if kind is None:
                cost = self.seed_cost(sim, jd) + APPROACH_JUMP * (sim['jumped_at'] >= 0)
                top = np.argsort(cost)[:K_ROBUST]
                sim2, jd2 = rob(top)
                fell2 = jd2['fell'].reshape(len(top), N_VAR).any(1)
                k = int(top[int(np.argmin(cost[top] + 100.0 * fell2))]); kind = 'approach'
            info['p_succ'] = float(ps[k]); info['p_best'] = float(ps.max()); info['n_go'] = int((ps >= P_GO).sum())
        self.best = progs.take(np.array([k]))
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

    @staticmethod
    def continue_cost(sim, jd, jump_now):
        """After the pickup: stay on the floor, away from edges; no new jumps."""
        return 100.0 * jd['fell'] + 20.0 * (~jd['settled']) + 10.0 * np.maximum(0.0, 1.5 - jd['clear']) + 5.0 * jump_now

    def walk_field(self, floor_ref):
        """Walking distance over the floor to the nearest seed (multi-source Dijkstra on the terrain raster): floor
        cells from floor_ref - 0.5 up, kept WALK_CLEAR from anything else. The approach cost uses it so a marble on
        the far side of a hole goes around it (straight-line seed distance pulled it into the hole's edge: dev v3)."""
        if self.walk is not None:
            return self.walk
        from scipy.ndimage import binary_erosion
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        g = self.g
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
        si = np.clip(np.rint((self.seeds['y'] - g.ys[0]) / g.res).astype(int), 0, ny - 1)
        sj = np.clip(np.rint((self.seeds['x'] - g.xs[0]) / g.res).astype(int), 0, nx - 1)
        src = np.unique(idx[si, sj][idx[si, sj] >= 0])
        D = np.full(ny * nx, np.inf)
        if len(src):
            D[cells] = dijkstra(G, directed=False, indices=src, min_only=True)
        self.walk = D.reshape(ny, nx)
        return self.walk

    def walk_at(self, x, y, floor_ref):
        D = self.walk_field(floor_ref); g = self.g
        i = np.clip(np.rint((y - g.ys[0]) / g.res).astype(int), 0, D.shape[0] - 1)
        j = np.clip(np.rint((x - g.xs[0]) / g.res).astype(int), 0, D.shape[1] - 1)
        return D[i, j]

    def seed_cost(self, sim, jd):
        """Approach cost: how close the predicted path comes to a seed launch state (sooner is better): the larger of
        the walking distance to the nearest seed and the position-velocity distance to the nearest seed state."""
        path, vel = sim['path'], sim['vel']
        n = len(path)
        if self.seed_tree is None:
            c = np.linalg.norm(path[:, 1:, :2] - self.gem[:2], axis=2).min(1)
        else:
            q = self._seed_key(path[:, 1:, 0].ravel(), path[:, 1:, 1].ravel(), vel[:, 1:, 0].ravel(), vel[:, 1:, 1].ravel())
            dist, _ = self.seed_tree.query(q)
            walk = self.walk_at(path[:, 1:, 0].ravel(), path[:, 1:, 1].ravel(), float(sim['floor_ref'][0]))
            dist = np.maximum(dist, walk)                    # inf where off the walkable floor (airborne over a hole)
            c = (dist.reshape(n, -1) + 0.03 * np.arange(path.shape[1] - 1)[None, :]).min(1)
            c = np.where(np.isfinite(c), c, 50.0)
        return c + 100.0 * jd['fell'] + 20.0 * (~jd['safe']) + 5.0 * np.maximum(0.0, CLEAR_OK - jd['clear'])
