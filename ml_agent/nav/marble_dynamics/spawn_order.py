"""Whole-spawn gem order (2026-10-24, operator: "the marble as it is is disgustingly slow at picking up clusters"; spawn
viewer + demos demo_20261021_151105 / demo_20261024_224609).

The tour (reliability_eval.tour -> nav.gems.plan_tour) orders only the 5 nearest gems of the observation, measures legs on
the walking grid (which routes AROUND the pads that the marble hops: 10-17 u between pad gems 2 u apart) and scales its
turn charge down with speed (half or less inside a spawn). The human sweeps whole spawns in straight serpentine runs.
best_order() orders EVERY gem of the spawn with a time model fitted to the demos: straight-line legs at SO_SPEED (the human's
in-spawn speed), SO_TURN_S_PER_90 per 90 degrees of heading change at a gem (the turn at a gem depends on the gem before
and after, so the dynamic programme carries the previous gem), a climb charge onto a higher pad, and the approach from the
marble's position along its current velocity. Pool spawns (65 planner visits): orders 1.35 s a spawn faster by this
model; 1 s of model time went with 0.75 s of real clear time within a cluster.
"""
import math
import numpy as np

SO_SPEED = 4.1            # u/s between gems (human median inside spawns: 4.1-4.2 u/s)
SO_TURN_S_PER_90 = .30    # s per 90 degrees of heading change at a gem
SO_CLIMB_S = .15          # s onto a pad at least SO_STEP_U higher (a timed jump instead of a roll)
SO_STEP_U = .25
SO_BIG_CLIMB_U = 1.5      # a rise this high needs more than a plain jump (Blast, edge hit): SO_BIG_CLIMB_S
SO_BIG_CLIMB_S = 1.0
SO_APPROACH_MAX = 10.     # u/s: the approach leg runs at the marble's speed (at least SO_SPEED, at most this)
SO_TURN_MIN_SPEED = 2.    # u/s: below this the marble has no heading to keep (no first-turn charge)
SO_MAX_GEMS = 10          # the dynamic programme orders at most this many (nearest first); the rest follow greedily
SO_PAD_DZ = 1.           # v2 (sim_so1: straight legs lost 12 percent on KOTM / Vortex, walls and gaps between gems): legs are
                         # walking-grid distances, except a detour around a hoppable pad (walk > 2 x straight + 1 u with at most
                         # this rise) counts the straight line; off the grid: 1.5 x straight + 2 u


def _leg(walk, straight, dz):
    if walk is None:return straight
    if not math.isfinite(walk):return 1.5 * straight + 2.
    if walk > 2. * straight + 1. and abs(dz) <= SO_PAD_DZ:return straight
    return walk


def _angle(a, b):
    na, nb = math.hypot(a[0], a[1]), math.hypot(b[0], b[1])
    if na < 1e-9 or nb < 1e-9: return 0.
    return math.degrees(math.acos(max(-1., min(1., (a[0] * b[0] + a[1] * b[1]) / (na * nb)))))


def _climb(z0, z1):
    dz = float(z1) - float(z0)
    if dz >= SO_BIG_CLIMB_U: return SO_CLIMB_S + SO_BIG_CLIMB_S
    return SO_CLIMB_S if dz >= SO_STEP_U else 0.


def order_time(pos, vel, gems, dist=None):
    """Model seconds to collect `gems` (tuples x, y, z, ...) in this order from the marble at pos with velocity vel;
    dist(point, gem) = walking-grid distance (None: straight lines)."""
    if not gems: return 0.
    sp = math.hypot(float(vel[0]), float(vel[1])); t = 0.
    px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
    hd = (float(vel[0]), float(vel[1])) if sp >= SO_TURN_MIN_SPEED else None
    for k, g in enumerate(gems):
        d = (g[0] - px, g[1] - py); L = math.hypot(*d)
        Lw = _leg(dist((px, py, pz), g) if dist is not None else None, L, g[2] - pz)
        t += Lw / (max(SO_SPEED, min(SO_APPROACH_MAX, sp)) if k == 0 else SO_SPEED)
        if hd is not None: t += SO_TURN_S_PER_90 * _angle(hd, d) / 90.
        t += _climb(pz, g[2])
        hd = d if L > 1e-6 else hd; px, py, pz = g[0], g[1], g[2]
    return t


def best_order(pos, vel, gems, dist=None):
    """The order of `gems` (list of tuples x, y, z, ...) with the least model time from the marble (pos, vel);
    dist(point, gem) = walking-grid distance (None: straight lines)."""
    gems = list(gems); n_all = len(gems)
    if n_all <= 1: return gems
    p0 = np.asarray(pos[:3], dtype=float)
    gems.sort(key=lambda g: math.hypot(g[0] - p0[0], g[1] - p0[1]))
    core, rest = gems[:SO_MAX_GEMS], gems[SO_MAX_GEMS:]
    n = len(core); P = np.array([g[:2] for g in core], dtype=float); Z = np.array([g[2] for g in core], dtype=float)
    sp = math.hypot(float(vel[0]), float(vel[1]))
    # legs between gems and the turn table T[prev, at, next] (prev = n is the marble's own heading)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    if dist is not None:
        for a in range(n):
            for b in range(n):
                if a != b: D[a, b] = _leg(dist(tuple(core[a][:3]), core[b]), float(D[a, b]), float(Z[b] - Z[a]))
    leg = D / SO_SPEED + np.array([[_climb(Z[a], Z[b]) for b in range(n)] for a in range(n)])
    head = np.zeros((n + 1, n, 2))
    for a in range(n):
        for b in range(n): head[a, b] = P[b] - P[a]
    for b in range(n): head[n, b] = P[b] - p0[:2]
    T = np.zeros((n + 1, n, n))
    for p in range(n + 1):
        for a in range(n):
            if p == a: continue
            for c in range(n):
                if c != a: T[p, a, c] = SO_TURN_S_PER_90 * _angle(head[p, a], P[c] - P[a]) / 90.
    first = np.array([_leg(dist(tuple(p0[:3]), core[j]) if dist is not None else None, math.hypot(*(P[j] - p0[:2])), float(Z[j] - p0[2]))
                      / max(SO_SPEED, min(SO_APPROACH_MAX, sp)) + _climb(p0[2], Z[j])
                      + (SO_TURN_S_PER_90 * _angle(vel[:2], P[j] - p0[:2]) / 90. if sp >= SO_TURN_MIN_SPEED else 0.)
                      for j in range(n)])
    INF = 1e18; full = (1 << n) - 1
    dp = np.full((1 << n, n, n + 1), INF); bp = np.full((1 << n, n, n + 1), -1, dtype=np.int16)
    for j in range(n): dp[1 << j, j, n] = first[j]
    for mask in range(1, full + 1):
        for last in range(n):
            if not mask >> last & 1: continue
            row = dp[mask, last]
            if row.min() >= INF: continue
            tot = row[:, None] + T[:, last, :]              # (prev, next)
            pmin = tot.argmin(axis=0); cand = tot[pmin, np.arange(n)] + leg[last]
            for nx in range(n):
                if mask >> nx & 1: continue
                m2 = mask | 1 << nx
                if cand[nx] < dp[m2, nx, last]: dp[m2, nx, last] = cand[nx]; bp[m2, nx, last] = pmin[nx]
    flat = dp[full].reshape(-1); k = int(flat.argmin()); last, prev = divmod(k, n + 1)
    order = []; mask = full
    while True:
        order.append(last)
        if prev == n: break
        pp = int(bp[mask, last, prev]); mask ^= 1 << last; last, prev = prev, pp
    out = [core[i] for i in order[::-1]]
    while rest:   # beyond SO_MAX_GEMS: nearest next
        q = out[-1]; j = min(range(len(rest)), key=lambda i: math.hypot(rest[i][0] - q[0], rest[i][1] - q[1])); out.append(rest.pop(j))
    return out
