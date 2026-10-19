"""Racing lines through a cluster (operator 10-17: clusters as fast as possible; the human's KOTM lines are wide fast
arcs, the planner's are tight slow hooks at every gem).

A line is a C1 cubic Hermite spline from the marble (tangent along its velocity) through waypoints near the next gems
(each within PICK_R of the gem centre; tangent direction and length free). Its drive time comes from a friction-circle
speed profile (|a| <= A_MAX, forward acceleration <= A_ACCEL, speed <= V_CAP; measured: human and planner both run at
~14.2 u/s^2), with a penalty for line points off the walkable floor or within EDGE_MIN of a drop (1 u walk grid at the
cluster's height). A cross-entropy search over the waypoint parameters minimises the time to the last gem.
LineProposal follows a line: pursuit of the point LOOKAHEAD ahead, at the profile speed there.
"""
import math
import numpy as np

from nav.marble_dynamics.gem_demo import Proposal

A_MAX = 14.
A_ACCEL = 10.
V_CAP = 13.
EDGE_MIN = 1.
PICK_R = .45
NSEG = 18            # samples per spline segment
POP = 160; ITERS = 6; ELITE = 16
EDGE_PEN_S = .4      # seconds per line sample on a bad cell
START_PEN_S = .25    # seconds per u/s the line's start speed is below the marble's (it would have to brake at once)


class Allowed:
    """Walkable, >= EDGE_MIN from a drop, at the cluster's height: a boolean raster in walk-grid cells."""
    def __init__(self, terrain, z):
        zs = np.where(np.isfinite(terrain.walk_z), terrain.walk_z, 1e9)
        k = np.argmin(np.abs(zs - float(z)), axis=0)
        jj, ii = np.indices(k.shape)
        zk = zs[k, jj, ii]; ok = (np.abs(zk - float(z)) < 1.) & terrain.walkable[k, jj, ii].astype(bool)
        ok &= terrain.edge_dist[k, jj, ii] >= EDGE_MIN
        self.ok = ok; self.x0 = float(terrain.wxs[0]); self.y0 = float(terrain.wys[0]); self.res = float(terrain.walk_res)

    def bad(self, pts):
        """pts (..., 2) -> bool (...) True where the line leaves the allowed floor."""
        i = np.rint((pts[..., 0] - self.x0) / self.res).astype(int); j = np.rint((pts[..., 1] - self.y0) / self.res).astype(int)
        inside = (i >= 0) & (j >= 0) & (i < self.ok.shape[1]) & (j < self.ok.shape[0])
        out = np.ones(pts.shape[:-1], dtype=bool)
        out[inside] = ~self.ok[j[inside], i[inside]]
        return out


def hermite(P0, T0, P1, T1, n):
    """(N,2) endpoints/tangents -> (N, n, 2) points."""
    t = np.linspace(0., 1., n)[None, :, None]
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1; h10 = t ** 3 - 2 * t ** 2 + t; h01 = -2 * t ** 3 + 3 * t ** 2; h11 = t ** 3 - t ** 2
    return h00 * P0[:, None] + h10 * T0[:, None] + h01 * P1[:, None] + h11 * T1[:, None]


def build(p0, v0, gems, X):
    """X (N, 4M+1): per gem (dx, dy, tangent angle, tangent scale) + start tangent scale -> points (N, L, 2), gem indices."""
    N = X.shape[0]; M = len(gems)
    sp = float(np.linalg.norm(v0))
    W = [np.repeat(p0[None], N, 0)]
    for m, g in enumerate(gems):
        W.append(g[None, :] + X[:, 4 * m:4 * m + 2])
    T = []
    first = W[1] - W[0]; L0 = np.linalg.norm(first, axis=1, keepdims=True)
    d0 = (v0 / sp)[None, :] if sp > 1. else first / np.maximum(L0, 1e-9)
    T.append(d0 * L0 * X[:, 4 * M:4 * M + 1])
    for m in range(M):
        ang = X[:, 4 * m + 2]; chord = np.linalg.norm(W[m + 1] - W[m], axis=1, keepdims=True)
        T.append(np.stack([np.cos(ang), np.sin(ang)], 1) * chord * X[:, 4 * m + 3:4 * m + 4])
    segs = []; idx = []
    for m in range(M):
        seg = hermite(W[m], T[m], W[m + 1], T[m + 1], NSEG)
        segs.append(seg if m == 0 else seg[:, 1:])
        idx.append(sum(s.shape[1] for s in segs) - 1)
    return np.concatenate(segs, 1), idx


def profile(pts, v_start):
    """Friction-circle speed profile along (N, L, 2) lines; returns (time to each point (N, L), speeds (N, L), start deficit)."""
    d = np.diff(pts, axis=1); ds = np.linalg.norm(d, axis=2) + 1e-9                     # (N, L-1)
    # curvature from the turn angle between consecutive segments over the mean segment length
    a1 = np.arctan2(d[:, :-1, 1], d[:, :-1, 0]); a2 = np.arctan2(d[:, 1:, 1], d[:, 1:, 0])
    dth = np.abs((a2 - a1 + np.pi) % (2 * np.pi) - np.pi)
    kap = np.zeros(pts.shape[:2]); kap[:, 1:-1] = dth / (.5 * (ds[:, :-1] + ds[:, 1:]))
    vmax = np.minimum(V_CAP, np.sqrt(A_MAX / np.maximum(kap, 1e-6)))
    N, L = kap.shape
    v = np.empty((N, L)); v[:, 0] = np.minimum(vmax[:, 0], v_start)
    for k in range(L - 1):   # forward: accelerate within the friction circle
        lat = v[:, k] ** 2 * kap[:, k]; along = np.minimum(A_ACCEL, np.sqrt(np.maximum(0., A_MAX ** 2 - lat ** 2)))
        v[:, k + 1] = np.minimum(vmax[:, k + 1], np.sqrt(v[:, k] ** 2 + 2 * along * ds[:, k]))
    for k in range(L - 2, -1, -1):   # backward: braking within the friction circle
        lat = v[:, k + 1] ** 2 * kap[:, k + 1]; along = np.sqrt(np.maximum(0., A_MAX ** 2 - lat ** 2))
        v[:, k] = np.minimum(v[:, k], np.sqrt(v[:, k + 1] ** 2 + 2 * along * ds[:, k]))
    deficit = np.maximum(0., v_start - v[:, 0])
    dt = ds / np.maximum(.5 * (v[:, :-1] + v[:, 1:]), .3)
    t = np.concatenate([np.zeros((N, 1)), np.cumsum(dt, 1)], 1)
    return t, v, deficit


def optimise(p0, v0, gems, allowed, rng, pop=POP, iters=ITERS):
    """Best line through `gems` (list of 2-D points) from (p0, v0): (points (L,2), speeds (L,), gem indices, time)."""
    M = len(gems); p0 = np.asarray(p0, float)[:2]; v0 = np.asarray(v0, float)[:2]; gems = [np.asarray(g, float)[:2] for g in gems]
    sp = float(np.linalg.norm(v0))
    mu = np.zeros(4 * M + 1); sd = np.zeros(4 * M + 1)
    pts_all = [p0] + gems
    for m in range(M):
        a = pts_all[m]; b = pts_all[m + 1]; c = pts_all[m + 2] if m + 2 < len(pts_all) else None
        u1 = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
        if c is not None:
            u2 = (c - b) / max(float(np.linalg.norm(c - b)), 1e-9); u = u1 + u2
            u = u / max(float(np.linalg.norm(u)), 1e-9) if np.linalg.norm(u) > 1e-6 else u1
        else:
            u = u1
        mu[4 * m:4 * m + 4] = [0., 0., math.atan2(u[1], u[0]), 1.]; sd[4 * m:4 * m + 4] = [.2, .2, .7, .45]
    mu[4 * M] = 1.; sd[4 * M] = .45
    best = (np.inf, None)
    for it in range(iters):
        X = mu + rng.normal(0, 1, (pop, 4 * M + 1)) * sd
        for m in range(M):   # offsets inside the pickup disk; positive tangent scales
            off = X[:, 4 * m:4 * m + 2]; r = np.linalg.norm(off, axis=1, keepdims=True)
            X[:, 4 * m:4 * m + 2] = np.where(r > PICK_R, off * PICK_R / np.maximum(r, 1e-9), off)
            X[:, 4 * m + 3] = np.clip(X[:, 4 * m + 3], .2, 3.)
        X[:, 4 * M] = np.clip(X[:, 4 * M], .1, 3.)
        pts, gi = build(p0, v0, gems, X)
        t, v, deficit = profile(pts, sp)
        cost = t[:, gi[-1]] + EDGE_PEN_S * allowed.bad(pts).sum(1) + START_PEN_S * deficit
        o = np.argsort(cost)
        if cost[o[0]] < best[0]:
            best = (float(cost[o[0]]), X[o[0]].copy())
        el = X[o[:ELITE]]; mu = el.mean(0); sd = np.maximum(el.std(0), .02)
    X = best[1][None]
    pts, gi = build(p0, v0, gems, X); t, v, _ = profile(pts, sp)
    return pts[0], v[0], gi, best[0], float(t[0, gi[-1]])


class LineProposal:
    """Follow a racing line (points, speeds) to its end, then keep its final heading."""
    LOOK = .8; LOOK_V = .22

    def __init__(self, pts, speeds, z, steps):
        last = pts[-1]; d = pts[-1] - pts[-2]; d = d / max(float(np.linalg.norm(d)), 1e-9)
        ext = last[None] + np.outer(np.linspace(.5, 8., 16), d)
        self.pts = np.vstack([pts, ext]); self.v = np.concatenate([speeds, np.full(16, speeds[-1])])
        self.s = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(self.pts, axis=0), axis=1))])
        self.z = float(z); self.steps = steps; self.j = 0

    def __call__(self, state, index):
        if index == 0:
            self.j = 0
        p = np.asarray(state['position'], dtype=float); v = np.asarray(state['velocity'], dtype=float)[:2]
        lo = self.j; hi = min(len(self.pts), self.j + 30)
        self.j = lo + int(np.argmin(np.linalg.norm(self.pts[lo:hi] - p[:2], axis=1)))
        look = self.LOOK + self.LOOK_V * float(np.linalg.norm(v))
        k = min(int(np.searchsorted(self.s, self.s[self.j] + look)), len(self.pts) - 1)
        vt = float(self.v[min(k, len(self.v) - 1)])
        goal = np.array([self.pts[k][0], self.pts[k][1], self.z])
        action = Proposal(goal, 0., min_speed=vt, top=vt)(state, 0)
        action[8] = 0
        return action


A_STICK = 14.2   # u/s^2 the marble gets from a full stick (measured)


class TrackLine:
    """Follow a racing line IN TIME: at reply i the reference is the line's state at t = (i + LEAD) * 64 ms;
    a_cmd = a_ref + KP * (p_ref - p) + KD * (v_ref - v), stick = a_cmd direction, throttle = |a_cmd| / A_STICK."""
    KP = 8.; KD = 5.; LEAD = .5

    def __init__(self, pts, speeds, z, steps):
        from nav.joystick import action_to_joystick
        from nav.marble_dynamics.bridge import reply
        self._js = action_to_joystick; self._reply = reply
        d = np.diff(pts, axis=0); ds = np.linalg.norm(d, axis=1) + 1e-9
        vm = .5 * (speeds[:-1] + speeds[1:]); dt = ds / np.maximum(vm, .3)
        self.t = np.concatenate([[0.], np.cumsum(dt)])
        tang = np.vstack([d / ds[:, None], (d[-1] / ds[-1])[None]])
        self.p = pts; self.v = tang * speeds[:, None]
        self.a = np.gradient(self.v, self.t, axis=0) if len(self.t) > 2 else np.zeros_like(self.v)
        self.z = float(z); self.steps = steps

    STOP_DECEL = 9.   # u/s^2: after the line's last gem the reference brakes to a stop (a supported, stable end)

    def ref(self, tau):
        if tau >= self.t[-1]:
            dt = tau - self.t[-1]; v_end = self.v[-1]; sp = float(np.linalg.norm(v_end))
            if sp < 1e-6:
                return self.p[-1].copy(), np.zeros(2), np.zeros(2)
            u = v_end / sp; t_stop = sp / self.STOP_DECEL
            if dt >= t_stop:
                return self.p[-1] + u * (sp * t_stop - .5 * self.STOP_DECEL * t_stop ** 2), np.zeros(2), np.zeros(2)
            return self.p[-1] + u * (sp * dt - .5 * self.STOP_DECEL * dt ** 2), u * (sp - self.STOP_DECEL * dt), -u * self.STOP_DECEL
        k = int(np.searchsorted(self.t, tau, side='right') - 1); k = max(0, min(k, len(self.t) - 2))
        w = (tau - self.t[k]) / max(self.t[k + 1] - self.t[k], 1e-9)
        return self.p[k] * (1 - w) + self.p[k + 1] * w, self.v[k] * (1 - w) + self.v[k + 1] * w, self.a[k] * (1 - w) + self.a[k + 1] * w

    def __call__(self, state, index):
        p = np.asarray(state['position'], dtype=float)[:2]; v = np.asarray(state['velocity'], dtype=float)
        pr, vr, ar = self.ref((index + self.LEAD) * .064)
        a = ar + self.KP * (pr - p) + self.KD * (vr - v[:2])
        mag = float(np.linalg.norm(a)); thr = min(1., mag / A_STICK) if mag > .05 else 0.
        js = self._js(float(a[0]), float(a[1]), thr, 0, 0, float(v[0]), float(v[1]))
        return self._reply(js, 0, math.atan2(float(a[0]), float(a[1])))


def order_lines(p0, v0, gems, allowed, rng, L=3, screen=(64, 4), final=(160, 6), top=4, tail_speed=7.):
    """Fastest visiting order of up to L of `gems` (2-D points) by racing-line time: every ordered L-subset is timed with
    a cheap CEM, the best `top` are re-optimised; score = time to the L-th gem + the straight run from it to the nearest
    gem left at tail_speed. Returns (order as indices, score, time) or None."""
    import itertools
    n = len(gems)
    if n < 1:
        return None
    L = min(L, n); res = []
    for perm in itertools.permutations(range(n), L):
        seq = [gems[i] for i in perm]
        _, _, _, cost, tend = optimise(p0, v0, seq, allowed, rng, pop=screen[0], iters=screen[1])
        left = [gems[i] for i in range(n) if i not in perm]
        tail = min(float(np.linalg.norm(np.asarray(g) - np.asarray(seq[-1]))) for g in left) / tail_speed if left else 0.
        res.append((cost + tail, perm, tail))
    res.sort(key=lambda r: r[0])
    best = None
    for score, perm, tail in res[:top]:
        _, _, _, cost, tend = optimise(p0, v0, [gems[i] for i in perm], allowed, rng, pop=final[0], iters=final[1])
        if best is None or cost + tail < best[0]:
            best = (cost + tail, perm, tend)
    return best[1], best[0], best[2]


def build_multi(p0, v0, G, X):
    """Like build() with per-row gems: G (N, M, 2), X (N, 4M+1)."""
    N, M, _ = G.shape
    sp = float(np.linalg.norm(v0))
    W = [np.repeat(p0[None], N, 0)] + [G[:, m] + X[:, 4 * m:4 * m + 2] for m in range(M)]
    first = W[1] - W[0]; L0 = np.linalg.norm(first, axis=1, keepdims=True)
    d0 = np.repeat((v0 / sp)[None, :], N, 0) if sp > 1. else first / np.maximum(L0, 1e-9)
    T = [d0 * L0 * X[:, 4 * M:4 * M + 1]]
    for m in range(M):
        ang = X[:, 4 * m + 2]; chord = np.linalg.norm(W[m + 1] - W[m], axis=1, keepdims=True)
        T.append(np.stack([np.cos(ang), np.sin(ang)], 1) * chord * X[:, 4 * m + 3:4 * m + 4])
    segs = []; idx = []
    for m in range(M):
        seg = hermite(W[m], T[m], W[m + 1], T[m + 1], NSEG)
        segs.append(seg if m == 0 else seg[:, 1:])
        idx.append(sum(s.shape[1] for s in segs) - 1)
    return np.concatenate(segs, 1), idx


def order_lines_fast(p0, v0, gems, allowed, rng, L=3, pop=64, iters=5, tail_speed=7., return_all=False):
    """All ordered L-subsets of `gems` optimised together (one batch per CEM iteration). Returns (perm, score, line time,
    the winning parameters) or None."""
    import itertools
    p0 = np.asarray(p0, float)[:2]; v0 = np.asarray(v0, float)[:2]; gems = [np.asarray(g, float)[:2] for g in gems]
    n = len(gems)
    if n < 1:
        return None
    L = min(L, n); perms = list(itertools.permutations(range(n), L)); P = len(perms); D = 4 * L + 1
    Gp = np.array([[gems[i] for i in perm] for perm in perms])                     # (P, L, 2)
    tails = np.array([min((float(np.linalg.norm(gems[j] - gems[perm[-1]])) for j in range(n) if j not in perm), default=0.) / tail_speed
                      for perm in perms])
    sp = float(np.linalg.norm(v0))
    mu = np.zeros((P, D)); sd = np.zeros((P, D))
    for q, perm in enumerate(perms):
        pts_all = [p0] + [gems[i] for i in perm]
        for m in range(L):
            a = pts_all[m]; b = pts_all[m + 1]; c = pts_all[m + 2] if m + 2 < len(pts_all) else None
            u1 = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
            u = u1
            if c is not None:
                u2 = (c - b) / max(float(np.linalg.norm(c - b)), 1e-9); w = u1 + u2
                if np.linalg.norm(w) > 1e-6:
                    u = w / float(np.linalg.norm(w))
            mu[q, 4 * m:4 * m + 4] = [0., 0., math.atan2(u[1], u[0]), 1.]; sd[q, 4 * m:4 * m + 4] = [.2, .2, .7, .45]
        mu[q, 4 * L] = 1.; sd[q, 4 * L] = .45
    best_cost = np.full(P, np.inf); best_x = np.zeros((P, D)); best_t = np.zeros(P)
    Grow = np.repeat(Gp, pop, 0)                                                     # (P*pop, L, 2)
    for it in range(iters):
        X = (mu[:, None, :] + rng.normal(0, 1, (P, pop, D)) * sd[:, None, :]).reshape(P * pop, D)
        for m in range(L):
            off = X[:, 4 * m:4 * m + 2]; r = np.linalg.norm(off, axis=1, keepdims=True)
            X[:, 4 * m:4 * m + 2] = np.where(r > PICK_R, off * PICK_R / np.maximum(r, 1e-9), off)
            X[:, 4 * m + 3] = np.clip(X[:, 4 * m + 3], .2, 3.)
        X[:, 4 * L] = np.clip(X[:, 4 * L], .1, 3.)
        pts, gi = build_multi(p0, v0, Grow, X)
        t, v, deficit = profile(pts, sp)
        tl = t[:, gi[-1]]
        cost = (tl + EDGE_PEN_S * allowed.bad(pts).sum(1) + START_PEN_S * deficit).reshape(P, pop)
        Xr = X.reshape(P, pop, D); tlr = tl.reshape(P, pop)
        o = np.argsort(cost, axis=1); k = max(4, pop // 10)
        bi = o[:, 0]; bc = cost[np.arange(P), bi]
        better = bc < best_cost
        best_cost[better] = bc[better]; best_x[better] = Xr[np.arange(P), bi][better]; best_t[better] = tlr[np.arange(P), bi][better]
        el = Xr[np.arange(P)[:, None], o[:, :k]]                                     # (P, k, D)
        mu = el.mean(1); sd = np.maximum(el.std(1), .02)
    score = best_cost + tails; q = int(np.argmin(score))
    if return_all:
        return perms, score, best_t
    return perms[q], float(score[q]), float(best_t[q]), best_x[q]
