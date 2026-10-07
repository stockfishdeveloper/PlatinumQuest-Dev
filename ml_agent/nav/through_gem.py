"""Carry speed through a gem (2026-10-05 21:00; evaluation opt-in NAV_THROUGH_GEM in nav/real_run.py, training unchanged).

Measured on 21,301 navigator-alone pickups: in the last 0.5 s the stick is fully pressed but mostly sideways (33 % of it
along the motion with the next gem straight ahead, 10 % at 30-60 deg), so the marble loses ~3 u/s (12.2 -> 9.1 u/s) even
when the next gem lies ahead; the operator's demo takes those gems at ~11 u/s.

This helper acts only in the last ACT_U before the target gem when the next gem is within MAX_TURN_DEG of the marble's
motion. Each decision it rolls a few stick directions forward with nav.ss_floor_model (accurate to ~0.06 u over 0.5 s):
the policy's own, straight at the gem, aimed through the gem toward the next one (a few depths), and along the motion;
each is held until the gem is taken, then the stick goes at the next gem. It keeps the one that takes the gem with margin
(PICK_MARGIN_U), stays on followed floor, and reaches the next gem soonest; it replaces the policy only if that beats the
policy's own direction by GAIN_S. Physics, terrain heights and the two gems the tour chose; nothing map specific.
"""
import math
import numpy as np
from nav.ss_floor_model import step_batch

ACT_U = 6.0                 # u to the target gem: the last ~0.5 s where the speed is lost
MAX_TURN_DEG = 60.0         # the next gem must lie within this of the motion (sharper turns: slowing is right)
PICK_MARGIN_U = 0.6         # u: predicted pass must be this close to the gem centre (pickup radius ~0.85)
GAIN_S = 0.03               # s: override only if the next gem is reached at least this much sooner
HORIZON_S = 1.5
DT = 0.064
SUB = 0.016
TRACK_DZ = 1.2
SQ2 = math.sqrt(2.0)
DEPTHS = (0.75, 1.5, 3.0)   # u past the gem along the line to the next gem


def _seg_dist(a, b, g):
    ab = b - a
    t = np.clip(((g[None] - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    return np.linalg.norm(a + ab * t[:, None] - g[None], axis=1)


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def _rollout(terrain, p, z, v, w, dirs, scale, g1, g2):
    """Hold dirs (N, 2) at stick length scale (N,) until gem 1 is taken, then steer at gem 2. Returns t1, t2, off, d2end."""
    n = len(dirs)
    P = np.repeat(p[None], n, 0); V = np.repeat(v[None], n, 0); W = np.repeat(w[None], n, 0)
    zt = np.full(n, float(z))
    t1 = np.full(n, np.inf); t2 = np.full(n, np.inf); off = np.full(n, np.inf)
    steps = int(round(HORIZON_S / SUB))
    for s in range(steps):
        took = np.isfinite(t1)
        d2 = g2[None] - P
        d2n = np.linalg.norm(d2, axis=1, keepdims=True)
        steer2 = np.where(d2n > 1e-6, d2 / np.maximum(d2n, 1e-6), 0.0)
        u = np.where(took[:, None], SQ2 * steer2, SQ2 * scale[:, None] * dirs)
        yaw = np.pi / 4 - np.arctan2(u[:, 1], u[:, 0])
        P0 = P
        P, V, W = step_batch(P, V, W, u, yaw, dt=SUB)
        t = (s + 1) * SUB
        live = ~np.isfinite(off) & ~np.isfinite(t2)
        h = terrain.heights_at(P[:, 0], P[:, 1])
        dz = np.where(np.isfinite(h), np.abs(h - zt[None]), np.inf)
        j = dz.argmin(0)
        okf = dz[j, np.arange(n)] <= TRACK_DZ
        zt = np.where(okf, h[j, np.arange(n)], zt)
        off = np.where(live & ~okf, t, off)
        hit1 = live & okf & ~took & (_seg_dist(P0, P, g1) <= PICK_MARGIN_U)
        t1 = np.where(hit1, t, t1)
        hit2 = live & okf & took & (_seg_dist(P0, P, g2) <= 0.85)
        t2 = np.where(hit2, t, t2)
        if not (~np.isfinite(off) & ~np.isfinite(t2)).any():
            break
    return t1, t2, off, np.linalg.norm(P - g2[None], axis=1)


def through_gem_dir(terrain, raw, prev_world_u, prev_yaw, target, nxt, policy_dir, policy_throttle):
    """A world unit stick direction (full throttle) for this decision, or None to keep the policy's action.
    raw: observation (pos 0-2, vel 3-5, spin via RAW_SPIN by the caller as raw_spin); prev_world_u / prev_yaw: the command
    still pending (it drives the next step: the bridge's one-decision lag)."""
    from nav.protocol import RAW_SPIN
    if target is None or nxt is None:
        return None
    p = np.array([float(raw[0]), float(raw[1])]); z = float(raw[2])
    v = np.array([float(raw[3]), float(raw[4])])
    w = np.array([float(x) for x in raw[RAW_SPIN]])
    g1 = np.array([float(target[0]), float(target[1])]); g2 = np.array([float(nxt[0]), float(nxt[1])])
    d1 = float(np.linalg.norm(g1 - p))
    sp = float(np.linalg.norm(v))
    if d1 > ACT_U or sp < 3.0 or float(np.linalg.norm(g2 - g1)) < 2.0:
        return None
    cosang = float(v @ (g2 - g1)) / (sp * float(np.linalg.norm(g2 - g1)))
    if math.degrees(math.acos(max(-1.0, min(1.0, cosang)))) > MAX_TURN_DEG:
        return None
    # the pending command drives the next step: start the candidates from there
    pu = np.asarray(prev_world_u, float)[None]
    p1, v1, w1 = step_batch(p[None], v[None], w[None], pu, float(prev_yaw), dt=DT)
    p1, v1, w1 = p1[0], v1[0], w1[0]
    cands, scales = [], []
    pd = _unit(np.asarray(policy_dir, float))
    if pd is not None:
        cands.append(pd); scales.append(max(0.0, min(1.0, float(policy_throttle))))
    for c in [_unit(g1 - p1), _unit(v1)] + [_unit(g1 + dd * (g2 - g1) / np.linalg.norm(g2 - g1) - p1) for dd in DEPTHS]:
        if c is not None:
            cands.append(c); scales.append(1.0)
    dirs = np.array(cands); scale = np.array(scales)
    t1, t2, off, d2end = _rollout(terrain, p1, z, v1, w1, dirs, scale, g1, g2)
    valid = np.isfinite(t1) & ((off > t2) | (~np.isfinite(off)))
    if not valid.any():
        return None
    score = np.where(np.isfinite(t2), t2, HORIZON_S + d2end / 10.0)
    score = np.where(valid, score, np.inf)
    i = int(np.argmin(score))
    if pd is not None and valid[0] and score[i] > score[0] - GAIN_S:
        return None                                   # the policy is (nearly) as good: leave it
    if pd is not None and i == 0:
        return None
    return (float(dirs[i, 0]), float(dirs[i, 1]))
