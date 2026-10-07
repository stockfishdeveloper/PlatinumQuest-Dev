"""Two-gem Super Speed aim and safety from the floor physics (2026-10-05; opt-in, not used by training yet).

Why: right after a kick the marble slides (the kick adds velocity, not spin) and the joystick cannot turn it until the
spin catches up: measured 10.4 u/s^2 of turn per unit of sideways stick in normal driving, about 0 for 0.4 s after a kick
and about 3 until 0.8 s. The aim in obs.ss_aim points the INSTANT velocity at the current gem only, so a kick toward
gem 1 often leaves gem 2 out of reach, and the 24 u/s cap on that instant speed is not the speed the marble keeps (slip
friction takes back 2/7 of the kick within about half a second).

What: for every candidate kick direction, roll the post-kick path forward with nav.ss_floor_model (the engine's contact
equations; 0.06 u off at 0.5 s on recorded kicks) under a fixed driving model: the stick held toward gem 1 until it is
taken, then toward gem 2, on the bridge's diagonal camera, with the kick yaw holding the camera for the first
HOLD_DEC decisions (mlAgent.cs PowYawHoldTicks). Pick the direction that takes both gems soonest while every sampled
point stays on level floor (terrain heights within LEVEL_DZ of the kick floor), then check the stop after gem 2.
Generic: physics, terrain heights and the two gems the tour already chose; nothing map specific.
"""
import math
import numpy as np
from nav.ss_floor_model import step_batch

KICK = 25.0                 # u/s added along the kick yaw (engine SuperSpeedVelocity)
DT = 0.064                  # one decision
SUB = 0.016                 # path sampling step (the model integrates in 8 ms substeps inside)
R_PICK = 0.85               # u: planner PICK_R0, horizontal
HOLD_DEC = 3                # decisions the kick yaw overrides the camera (12 ticks)
HORIZON_S = 3.0             # s of path checked after the kick
LEVEL_DZ = 0.75             # u: obs.SS_LEVEL_DZ (a bank or rim is not floor)
SPAN_DEG = 40.0             # candidates within this of the line to gem 1
STEP_DEG = 1.0
SQ2 = math.sqrt(2.0)


def _on_level(terrain, xy, z0):
    h = terrain.heights_at(xy[:, 0], xy[:, 1])                          # (K, N)
    near = np.where(np.isfinite(h), np.abs(h - z0), np.inf).min(axis=0) <= LEVEL_DZ
    return near


def rollout(terrain, p, z, v, w, aims, g1, g2, horizon_s=HORIZON_S):
    """Batch rollout of kicks along unit `aims` (N, 2) from state p (2,), z, v (2,), spin w (3,).
    Returns dict of arrays: t1 (time gem 1 taken, inf), t2 (gem 2 after gem 1, inf), off (first time off level floor,
    inf), v_t2 (speed when gem 2 is taken), path (N, steps, 2)."""
    n = len(aims)
    P = np.repeat(np.asarray(p, float)[None], n, 0)
    V = np.repeat(np.asarray(v, float)[None], n, 0) + KICK * aims
    W = np.repeat(np.asarray(w, float)[None], n, 0)
    kyaw = np.arctan2(aims[:, 0], aims[:, 1])                          # bridge yaw: forward = (sin, cos)
    g1 = np.asarray(g1, float); g2 = None if g2 is None else np.asarray(g2, float)
    got1 = np.full(n, np.inf); got2 = np.full(n, np.inf); off = np.full(n, np.inf); vt2 = np.full(n, np.nan)
    steps = int(round(horizon_s / SUB)); per_dec = int(round(DT / SUB))
    path = np.empty((n, steps, 2))
    u = np.zeros((n, 2)); yaw = kyaw.copy()
    for s in range(steps):
        if s % per_dec == 0:                                              # a new decision: the stick and the camera
            k = s // per_dec
            goal = np.where(np.isfinite(got1)[:, None], g2[None] if g2 is not None else P, g1[None])
            d = goal - P
            dn = np.linalg.norm(d, axis=1, keepdims=True)
            u = np.where(dn > 1e-6, SQ2 * d / np.maximum(dn, 1e-6), 0.0)
            yaw = kyaw if k < HOLD_DEC else (np.pi / 4 - np.arctan2(u[:, 1], u[:, 0]))
        P, V, W = step_batch(P, V, W, u, yaw, dt=SUB)
        t = (s + 1) * SUB
        path[:, s] = P
        d1 = np.linalg.norm(P - g1, axis=1)
        got1 = np.where((d1 <= R_PICK) & ~np.isfinite(got1), t, got1)
        if g2 is not None:
            d2 = np.linalg.norm(P - g2, axis=1)
            new2 = (d2 <= R_PICK) & np.isfinite(got1) & ~np.isfinite(got2)
            vt2 = np.where(new2, np.linalg.norm(V, axis=1), vt2)
            got2 = np.where(new2, t, got2)
    on = _on_level(terrain, path.reshape(-1, 2), z).reshape(n, steps)
    first_off = np.where(~on.all(axis=1), np.argmax(~on, axis=1), -1)
    off = np.where(first_off >= 0, (first_off + 1) * SUB, np.inf)
    return {'t1': got1, 't2': got2, 'off': off, 'v_t2': vt2, 'path': path}


FAST_SUB = 0.032            # plan_fast path step; passes are tested on the segment, so a gem is not stepped over
INST_SKIP = 27.5            # u/s: skip the rollouts when every candidate kick leaves the marble faster than this (on 196
                            # approved recorded states the slowest candidate was at most 25.75); keeps the observation
                            # cheap while the marble already rolls at the gem (15:58 training ran 33 s/update without it)
FAST_HORIZON_S = 2.5        # s: gem 1 then gem 2 within this, i.e. one push (the rule trained 15:57-18:00 and after 18:20)
COMPARE_NO_KICK = False     # 17:45 kick-vs-no-kick rule. OFF: on 18:00 64 rounds (main 30050, forced) it fired 3.6 kicks and
                            # had 3.8 falls a round, 154.3 vs navigator alone 159.9. It counted a no-kick drive that left
                            # the floor (a hole on the straight line, which the navigator drives around) as slow and so
                            # approved kicks across holes (121/156 fell within 3.2 s); gem 1 alone (no next gem) 39/49 fell;
                            # next gem known with the no-kick drive on floor 2/29. When on: the no-kick drive must ARRIVE on
                            # floor or nothing is approved, the horizon is COMPARE_HORIZON_S, and obs.py still gives no
                            # single-gem kick through this module. Validate in the game before training on it.
COMPARE_HORIZON_S = 4.0
SAVE_MIN = 0.15             # s: (COMPARE_NO_KICK) the kick must reach the gems this much sooner than not kicking
TRACK_DZ = 1.2              # u: floor level followed from sample to sample (slopes); none within it = off the floor
V2_MAX = 20.0               # u/s predicted at gem 2 (full stick, no braking; the game is ~10 u/s slower, correlation 0.93).
                            # Falls within 3.2 s of a kick by predicted speed at gem 2: recorded kicks < 20 -> 0/155; the
                            # 20:00 block (64 forced/learned rounds at V2_MAX 22): 18-20 -> 0/37, 20-22 -> 7/37, >= 22 -> 10/18.
                            # 22 (16:03-20:00) added risky kicks: forced 160.56 = navigator alone 160.56; back to 20 at 20:15.


def _seg_dist(a, b, g):
    """Distance from point g (2,) to segments a->b (N, 2)."""
    ab = b - a
    t = np.clip(((g[None] - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    return np.linalg.norm(a + ab * t[:, None] - g[None], axis=1)


def _fast_batch(terrain, p, z, v, w, aims, g1, g2, horizon_s=FAST_HORIZON_S, kicks=None):
    """kicks (N,) = 1 for a kick along aims[i], 0 for the same drive without one (no kick, no camera hold). g2 None =
    gem 1 only: t2 then repeats t1 so callers can read t2 as 'the last gem considered'."""
    n = len(aims)
    kicks = np.ones(n) if kicks is None else np.asarray(kicks, float)
    single = g2 is None
    g2 = g1 if single else g2
    P = np.repeat(p[None], n, 0); V = np.repeat(v[None], n, 0) + KICK * aims * kicks[:, None]; W = np.repeat(w[None], n, 0)
    kyaw = np.arctan2(aims[:, 0], aims[:, 1])
    zt = np.full(n, float(z))
    t1 = np.full(n, np.inf); t2 = np.full(n, np.inf); off = np.full(n, np.inf)
    v2 = np.full((n, 2), np.nan); p2 = np.full((n, 2), np.nan); z2 = np.full(n, np.nan)
    steps = int(round(horizon_s / FAST_SUB)); per_dec = int(round(DT / FAST_SUB))
    for s in range(steps):
        if s % per_dec == 0:
            k = s // per_dec
            goal = np.where(np.isfinite(t1)[:, None], g2[None], g1[None])
            d = goal - P
            dn = np.linalg.norm(d, axis=1, keepdims=True)
            u = np.where(dn > 1e-6, SQ2 * d / np.maximum(dn, 1e-6), 0.0)
            steer = np.pi / 4 - np.arctan2(u[:, 1], u[:, 0])
            yaw = np.where(kicks > 0, kyaw, steer) if k < HOLD_DEC else steer
        P0 = P
        P, V, W = step_batch(P, V, W, u, yaw, dt=FAST_SUB)
        t = (s + 1) * FAST_SUB
        live = ~np.isfinite(off) & ~np.isfinite(t2)
        h = terrain.heights_at(P[:, 0], P[:, 1])                          # (K, n)
        dz = np.where(np.isfinite(h), np.abs(h - zt[None]), np.inf)
        j = dz.argmin(0)
        okf = dz[j, np.arange(n)] <= TRACK_DZ
        zt = np.where(okf, h[j, np.arange(n)], zt)
        off = np.where(live & ~okf, t, off)
        hit1 = live & okf & ~np.isfinite(t1) & (_seg_dist(P0, P, g1) <= R_PICK)
        t1 = np.where(hit1, t, t1)
        hit2 = (hit1 if single else live & okf & np.isfinite(t1) & ~hit1 & (_seg_dist(P0, P, g2) <= R_PICK))
        t2 = np.where(hit2, t, t2); v2[hit2] = V[hit2]; p2[hit2] = P[hit2]; z2[hit2] = zt[hit2]
        if not (~np.isfinite(off) & ~np.isfinite(t2)).any():
            break
    return t1, t2, off, v2, p2, z2


def advance(p, v, w, goal, dt=DT):
    """The state one decision later with the stick toward `goal` on the diagonal camera: the kick goes off a decision
    after the last aim is sent (HANDOFF_SUPERSPEED_2026-10-05 section 7), and the policy steers at the gem meanwhile."""
    p = np.asarray(p, float)[None]; v = np.asarray(v, float)[None]; w = np.asarray(w, float)[None]
    d = np.asarray(goal, float)[:2] - p[0]
    dn = float(np.linalg.norm(d))
    u = (SQ2 * d / dn)[None] if dn > 1e-6 else np.zeros((1, 2))
    yaw = math.pi / 4 - math.atan2(u[0, 1], u[0, 0])
    p1, v1, w1 = step_batch(p, v, w, u, yaw, dt=dt)
    return p1[0], v1[0], w1[0]


def plan_fast(terrain, p, z, v, w, g1, g2, span_deg=SPAN_DEG):
    """Kick decision for the observation (one push by default; COMPARE_NO_KICK = the 17:45 comparison, off since 18:20).
    Returns (ok, aim, instant_speed). The best kick direction is the one that takes gem 1 and then gem 2 (or gem 1 alone
    when g2 is None) soonest, steering normally after gem 1, within FAST_HORIZON_S; ok = that kick stays on followed
    floor, arrives at the last gem below V2_MAX, and gets there at least SAVE_MIN sooner than the same drive without a
    kick (a drive that leaves the floor or misses counts as not arriving within the horizon)."""
    p = np.asarray(p, float); v = np.asarray(v, float); w = np.asarray(w, float)
    g1 = np.asarray(g1, float)[:2]; g2 = None if g2 is None else np.asarray(g2, float)[:2]
    base = math.atan2(g1[1] - p[1], g1[0] - p[0])
    coarse = np.radians(np.arange(-span_deg, span_deg + 1e-9, 4.0))
    inst = np.linalg.norm(v[None] + KICK * np.stack([np.cos(base + coarse), np.sin(base + coarse)], 1), axis=1)
    if inst.min() > INST_SKIP:
        return False, None, float(inst.min())
    horizon = COMPARE_HORIZON_S if COMPARE_NO_KICK else FAST_HORIZON_S
    t_nokick = None if COMPARE_NO_KICK else math.inf
    best = None
    for offs in (coarse, None):
        if offs is None:
            if best is None:
                break
            offs = best[0] + np.radians(np.arange(-3.0, 3.0 + 1e-9, 1.0))
        ang = base + offs
        aims = np.stack([np.cos(ang), np.sin(ang)], 1)
        kicks = np.ones(len(aims))
        if t_nokick is None:                                              # the same drive without a kick, once
            aims = np.vstack([aims, aims[:1]]); kicks = np.append(kicks, 0.0)
        t1, t2, off, v2, p2, z2 = _fast_batch(terrain, p, z, v, w, aims, g1, g2, horizon_s=horizon, kicks=kicks)
        if t_nokick is None:
            t_nokick = float(t2[-1])                                    # inf if the no-kick drive leaves the floor or misses
            if not np.isfinite(t_nokick):
                return False, None, 0.0
            t1, t2, v2, aims = t1[:-1], t2[:-1], v2[:-1], aims[:-1]
        ok = np.isfinite(t2) & (np.linalg.norm(np.nan_to_num(v2, nan=1e9), axis=1) < V2_MAX)
        if ok.any():
            i = int(np.argmin(np.where(ok, t2, np.inf)))
            if best is None or t2[i] < best[1]:
                best = (float(offs[i]), float(t2[i]), aims[i].copy())
    if best is None or (COMPARE_NO_KICK and best[1] > t_nokick - SAVE_MIN):
        return False, None, 0.0
    aim = best[2]
    return True, (float(aim[0]), float(aim[1])), float(np.linalg.norm(v + KICK * aim))


def plan(terrain, p, z, v, w, g1, g2, span_deg=SPAN_DEG, step_deg=STEP_DEG):
    """Best kick direction for taking gem 1 then gem 2, and whether it is safe. Returns
    {'aim': (ax, ay), 'both': bool, 't1', 't2', 'off', 'delta_deg'} for the chosen direction; delta_deg is relative to
    the line to gem 1. 'both' means both gems are taken within the horizon and the path stays on level floor until
    then (the stop after gem 2 is the caller's existing braking-room rule)."""
    p = np.asarray(p, float)
    base = math.atan2(g1[1] - p[1], g1[0] - p[0])
    offs = np.radians(np.arange(-span_deg, span_deg + 1e-9, step_deg))
    ang = base + offs
    aims = np.stack([np.cos(ang), np.sin(ang)], 1)
    r = rollout(terrain, p, z, v, w, aims, g1, g2)
    ok = np.isfinite(r['t2']) & (r['off'] > r['t2'])
    if ok.any():
        key = np.where(ok, r['t2'], np.inf)
    else:                                                                 # no two-gem path: best single gem
        ok1 = np.isfinite(r['t1']) & (r['off'] > r['t1'])
        key = np.where(ok1, 100.0 + r['t1'], 1000.0 - np.minimum(r['off'], 99.0))
    i = int(np.argmin(key))
    return {'aim': (float(aims[i, 0]), float(aims[i, 1])), 'both': bool(ok[i]), 't1': float(r['t1'][i]),
            't2': float(r['t2'][i]), 'off': float(r['off'][i]), 'delta_deg': float(math.degrees(offs[i])),
            'v_t2': float(r['v_t2'][i])}
