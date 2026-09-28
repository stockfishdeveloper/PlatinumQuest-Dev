"""Stage 3 step model (design section 5, M2 pilot): one 64 ms decision of marble physics, learned from stage 3 data.

    python -m nav.learned_nav.dynamics3 build            # feature caches from the recorded shards
    python -m nav.learned_nav.dynamics3 train            # ensemble of ENSEMBLE models -> models/learned_nav/step3_<k>.pth

Frame: yaw of the horizontal velocity (above 0.5 u/s), else of the input, else world x; z stays up (no gravity maps).
Inputs, all in that frame, never map identity:
* velocity, spin, speed;
* the applied input: the world force vector the reply encodes (right - left, fwd - back; up to sqrt 2), jump key;
* the surface under the marble (gap between the marble's bottom and the floor level below it, exact face normal);
* a 9 x 9 crop, 0.5 u spacing: floor relative to the marble's bottom (the highest level up to 0.6 u above it),
  floor present, and blocked (a surface 0.6-3 u above: wall or step);
* the two nearest exact edge segments of the marble's floor (distance, outward normal, void / drop / wall, drop).
No contact telemetry as input: the natural-state replay check showed position, velocity and spin carry the state, so
the model can roll forward on its own predictions. Targets (next decision): change of position, velocity and spin,
with a predicted spread each (heteroscedastic Gaussian), and the next interval's contact: any supporting contact, a
collision, contact at the last sub-step, and the supporting normal.
Splits: validation = 8 u blocks of the training maps, one in eight by hash; the two held-out maps are evaluated
separately and never trained on.
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav.geometry import Geometry, VOID, DROP, WALL                  # noqa: E402

R = 0.19
STAGE3 = os.path.join(HERE, 'datasets', 'learned_nav', 'stage3')
FEAT_DIR = os.path.join(HERE, 'datasets', 'learned_nav', 'stage3_features')
MODEL_DIR = os.path.join(HERE, 'models', 'learned_nav')
TRAIN_MAPS = ['KingOfTheMarble_Hunt_phys', 'VortexEffect_Hunt_phys', 'GemsInTheRoad_Hunt_phys', 'Tilo_Hunt_phys',
              'BasinHill_Hunt_phys', 'ParkourPeaks_Hunt_phys', 'Duplex_Hunt_phys', 'Cragmire_Hunt_phys',
              'Sprawl_Hunt_phys', 'KingOfTheRing_Hunt_phys', 'MaximoCenter_Hunt_phys']
DEV_MAPS = ['GemsAhoy_Hunt_phys', 'Acropolis2_Hunt_phys']
CROP = np.arange(-4, 5) * 0.5                                  # 9 points, +-2 u
FINE = np.arange(-2, 3) * 0.2                                  # 5 points, +-0.4 u: the surface right around the marble
HIDDEN = int(os.environ.get('STEP3_HIDDEN', '512'))
BLOCKS = int(os.environ.get('STEP3_BLOCKS', '3'))
N_EDGE = 2
EDGE_REACH = 3.0
ENSEMBLE = 3
VAL_BLOCK = 8.0
VAL_MOD = 8
# step table columns (record3.STEP_FIELDS)
C_TRIAL, C_I, C_P, C_V, C_W = 0, 1, slice(2, 5), slice(5, 8), slice(8, 11)
C_SUB, C_CONTACT, C_SUPPORT, C_COLL = 11, 12, 13, 14
C_N = slice(17, 20)
C_LAST = 23
C_FWD, C_BACK, C_LEFT, C_RIGHT, C_JUMP = 24, 25, 26, 27, 28
C_OOB = 36


# ------------------------------------------------------------------ geometry features
class EdgeIndex:
    """The exact edge segments near each 1 u cell, for vectorized nearest-edge queries."""
    K = 32

    def __init__(self, g):
        from scipy.spatial import cKDTree
        e = g.edges
        self.e = e
        self.x0, self.y0 = float(g.xs[0]), float(g.ys[0])
        W = int(math.ceil((g.xs[-1] - g.xs[0]))) + 1; H = int(math.ceil((g.ys[-1] - g.ys[0]))) + 1
        self.W, self.H = W, H
        # sample every segment every 0.5 u and index the samples
        pts, owner = [], []
        for k, s in enumerate(e):
            L = math.hypot(s[3] - s[0], s[4] - s[1]); n = max(2, int(L / 0.5) + 1)
            t = np.linspace(0, 1, n)
            pts.append(np.c_[s[0] + t * (s[3] - s[0]), s[1] + t * (s[4] - s[1])]); owner.append(np.full(n, k))
        table = np.full((H, W, self.K), -1, dtype=np.int32)
        if pts:
            pts = np.concatenate(pts); owner = np.concatenate(owner)
            tree = cKDTree(pts)
            cx, cy = np.meshgrid(self.x0 + np.arange(W) + 0.5, self.y0 + np.arange(H) + 0.5)
            dist, idx = tree.query(np.c_[cx.ravel(), cy.ravel()], k=min(4 * self.K, len(pts)), distance_upper_bound=EDGE_REACH + 1.0)
            idx = np.atleast_2d(idx); dist = np.atleast_2d(dist)
            for c in range(len(idx)):
                ok = np.isfinite(dist[c])
                segs = np.unique(owner[idx[c][ok]])[:self.K]
                table.reshape(-1, self.K)[c, :len(segs)] = segs
        self.table = table

    def nearest(self, px, py, pz_floor, cos_y, sin_y):
        """(N, N_EDGE * 7): distance, outward normal in frame (2), one-hot void/drop/wall (3), drop (clipped)."""
        N = len(px)
        out = np.zeros((N, N_EDGE, 7), dtype=np.float32); out[:, :, 0] = EDGE_REACH
        if not len(self.e):
            return out.reshape(N, -1)
        ci = np.clip(np.floor(px - self.x0).astype(np.int64), 0, self.W - 1)
        cj = np.clip(np.floor(py - self.y0).astype(np.int64), 0, self.H - 1)
        cand = self.table[cj, ci]                                  # (N, K)
        valid = cand >= 0
        s = self.e[np.where(valid, cand, 0)]                        # (N, K, 11)
        ax, ay = s[..., 0], s[..., 1]; bx, by = s[..., 3], s[..., 4]
        dx, dy = bx - ax, by - ay
        L2 = np.maximum(dx * dx + dy * dy, 1e-12)
        t = np.clip(((px[:, None] - ax) * dx + (py[:, None] - ay) * dy) / L2, 0, 1)
        qx, qy = ax + t * dx, ay + t * dy
        d = np.hypot(px[:, None] - qx, py[:, None] - qy)
        zs = s[..., 2] + t * (s[..., 5] - s[..., 2])
        ok = valid & (d <= EDGE_REACH) & (zs >= pz_floor[:, None] - 1.0) & (zs <= pz_floor[:, None] + 0.6)
        d = np.where(ok, d, np.inf)
        order = np.argsort(d, axis=1)[:, :N_EDGE]
        rows = np.arange(N)[:, None]
        dsel = d[rows, order]; ssel = s[rows, order]
        has = np.isfinite(dsel)
        nx, ny = ssel[..., 6], ssel[..., 7]
        out[..., 0] = np.where(has, dsel, EDGE_REACH)
        out[..., 1] = np.where(has, cos_y[:, None] * nx + sin_y[:, None] * ny, 0)
        out[..., 2] = np.where(has, -sin_y[:, None] * nx + cos_y[:, None] * ny, 0)
        typ = ssel[..., 8]
        for k, tv in enumerate((VOID, DROP, WALL)):
            out[..., 3 + k] = np.where(has & (typ == tv), 1.0, 0.0)
        drop = np.where(np.isfinite(ssel[..., 9]), ssel[..., 9], 5.0)
        out[..., 6] = np.where(has, np.clip(drop, -3, 5) / 5.0, 0)
        return out.reshape(N, -1)


def frame_yaw(V, U):
    sp = np.hypot(V[:, 0], V[:, 1]); su = np.hypot(U[:, 0], U[:, 1])
    yaw = np.where(sp > 0.5, np.arctan2(V[:, 1], V[:, 0]), np.where(su > 0.1, np.arctan2(U[:, 1], U[:, 0]), 0.0))
    return yaw


def rot(vx, vy, c, s):
    return c * vx + s * vy, -s * vx + c * vy


def level_below(g, qx, qy, ztop):
    """For each query, the highest level at or below ztop (NaN if none), and any level in (ztop, ztop + 2.4]."""
    H = g.heights_at(qx, qy)                                        # (K, N)
    below = np.where(H <= ztop[None, :], H, -np.inf)
    lv = below.max(axis=0)
    lv = np.where(np.isfinite(lv), lv, np.nan)
    above = np.any((H > ztop[None, :]) & (H <= ztop[None, :] + 2.4), axis=0)
    return lv, above


def features(g, eidx, P, V, W, U, J, Up=None, Jp=None):
    """Vectorized input features for states P, V, W (N, 3) with applied input U (N, 2 world) and jump key J (N,).
    Up, Jp: the reply before it. Measured (2026-09-28): at an input switch the step's velocity change still follows
    the previous input by ~0.25 u/s, so the previous reply is part of the state (the controller always knows it)."""
    N = len(P)
    if Up is None:
        Up = U; Jp = np.zeros(N)
    yaw = frame_yaw(V, U); c, s = np.cos(yaw), np.sin(yaw)
    vf, vl = rot(V[:, 0], V[:, 1], c, s)
    wf, wl = rot(W[:, 0], W[:, 1], c, s)
    uf, ul = rot(U[:, 0], U[:, 1], c, s)
    bottom = P[:, 2] - R
    # under the marble
    lv, _ = level_below(g, P[:, 0], P[:, 1], P[:, 2] + 0.3)
    gap = np.where(np.isfinite(lv), np.clip(bottom - lv, -1, 3), 3.0)
    nrm = np.zeros((N, 3)); nrm[:, 2] = 1.0
    ok = np.isfinite(lv)
    if ok.any():
        i = np.rint((P[ok, 0] - g.xs[0]) / g.res).astype(np.int64).clip(0, len(g.xs) - 1)
        j = np.rint((P[ok, 1] - g.ys[0]) / g.res).astype(np.int64).clip(0, len(g.ys) - 1)
        Hc = g.heights[:, j, i]; k = np.argmin(np.abs(np.where(np.isfinite(Hc), Hc, 1e9) - lv[ok][None, :]), axis=0)
        tri = g.ntri[k, j, i]
        n = g.norms[np.maximum(tri, 0)].astype(np.float64)
        n[tri < 0] = (0, 0, 1)
        nrm[ok] = n
    nf, nl = rot(nrm[:, 0], nrm[:, 1], c, s)
    # crop in the frame
    ff, ll = np.meshgrid(CROP, CROP, indexing='ij')
    ff = ff.ravel(); ll = ll.ravel()
    qx = P[:, 0:1] + c[:, None] * ff[None, :] - s[:, None] * ll[None, :]
    qy = P[:, 1:2] + s[:, None] * ff[None, :] + c[:, None] * ll[None, :]
    zt = np.repeat(bottom + 0.6, len(ff))
    clv, cab = level_below(g, qx.ravel(), qy.ravel(), zt)
    clv = clv.reshape(N, -1); cab = cab.reshape(N, -1)
    pres = np.isfinite(clv)
    rel = np.where(pres, np.clip(clv - bottom[:, None], -3, 1), -3.0)
    # nearest exact edges
    zfl = np.where(np.isfinite(lv), lv, bottom)
    edges = eidx.nearest(P[:, 0], P[:, 1], zfl, c, s)
    # fine crop: floor relative to the marble's bottom at 0.2 u around it (appended last, earlier indices unchanged)
    f2, l2 = np.meshgrid(FINE, FINE, indexing='ij'); f2 = f2.ravel(); l2 = l2.ravel()
    fx = P[:, 0:1] + c[:, None] * f2[None, :] - s[:, None] * l2[None, :]
    fy = P[:, 1:2] + s[:, None] * f2[None, :] + c[:, None] * l2[None, :]
    flv, _ = level_below(g, fx.ravel(), fy.ravel(), np.repeat(bottom + 0.6, len(f2)))
    flv = flv.reshape(N, -1)
    frel = np.where(np.isfinite(flv), np.clip(flv - bottom[:, None], -1, 0.6), -1.0)
    F = np.concatenate([
        np.c_[vf / 10, vl / 10, V[:, 2] / 10, wf / 40, wl / 40, W[:, 2] / 40, np.hypot(V[:, 0], V[:, 1]) / 10,
              uf, ul, np.hypot(U[:, 0], U[:, 1]), J, gap / 3, nf, nl, nrm[:, 2]],
        rel / 3, pres.astype(np.float32), cab.astype(np.float32), edges, frel,
        np.c_[np.stack(rot(Up[:, 0], Up[:, 1], c, s), 1), np.hypot(Up[:, 0], Up[:, 1]), Jp]], axis=1).astype(np.float32)
    return F, yaw


N_FEAT = 15 + 3 * len(CROP) ** 2 + N_EDGE * 7 + len(FINE) ** 2 + 4


DT = 0.064
G = 20.0


def ballistic(V):
    """The known part of a step: (dp, dv) of a free marble under gravity, world frame."""
    dp = V * DT + np.array([0.0, 0.0, -0.5 * G * DT * DT]); dv = np.tile([0.0, 0.0, -G * DT], (len(V), 1))
    return dp, dv


def targets(P0, V0, W0, P1, V1, W1, tel1, yaw):
    """Continuous targets are the step's deviation from the ballistic step (ballistic()): the network learns
    contact and control, not kinematics it would only approximate."""
    c, s = np.cos(yaw), np.sin(yaw)
    bp, bv = ballistic(V0)
    dp = P1 - P0 - bp; dv = V1 - V0 - bv; dw = W1 - W0
    T = np.c_[np.stack(rot(dp[:, 0], dp[:, 1], c, s), 1), dp[:, 2],
              np.stack(rot(dv[:, 0], dv[:, 1], c, s), 1) / 5, dv[:, 2] / 5,
              np.stack(rot(dw[:, 0], dw[:, 1], c, s), 1) / 20, dw[:, 2] / 20]
    sup = (tel1[:, C_SUPPORT - C_SUB] > 0).astype(np.float32)
    coll = (tel1[:, C_COLL - C_SUB] > 0).astype(np.float32)
    last = (tel1[:, C_LAST - C_SUB] > 0).astype(np.float32)
    n = tel1[:, 6:9]
    nf, nl = rot(n[:, 0], n[:, 1], c, s)
    return np.c_[T, sup, coll, last, nf, nl, n[:, 2]].astype(np.float32)


N_CONT = 9          # continuous targets: dp - ballistic (3), (dv - ballistic) / 5 (3), dw / 20 (3)


def apply_step(P, V, W, mu, yaw):
    """World-frame next state from a prediction mu (denormalized, frame of yaw) and the ballistic part."""
    c, s = np.cos(yaw), np.sin(yaw)
    bp, bv = ballistic(V)
    dp = np.c_[c * mu[:, 0] - s * mu[:, 1], s * mu[:, 0] + c * mu[:, 1], mu[:, 2]] + bp
    dv = np.c_[c * mu[:, 3] - s * mu[:, 4], s * mu[:, 3] + c * mu[:, 4], mu[:, 5]] * 5 + bv
    dw = np.c_[c * mu[:, 6] - s * mu[:, 7], s * mu[:, 6] + c * mu[:, 7], mu[:, 8]] * 20
    return P + dp, V + dv, W + dw


def val_block(x, y):
    k = f'{math.floor(x / VAL_BLOCK)},{math.floor(y / VAL_BLOCK)}'.encode()
    return int(hashlib.md5(k).hexdigest(), 16) % VAL_MOD == 0


def load_shards(m):
    d = os.path.join(STAGE3, m)
    out = []
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if fn.startswith('shard_') and fn.endswith('.npz'):
            z = np.load(os.path.join(d, fn))
            out.append((fn, z['steps'], json.loads(str(z['meta']))))
    return out


def build_map(m, max_trans, rng):
    """Features and targets for up to max_trans transitions of map m (all its recorded shards)."""
    g = Geometry(m); eidx = EdgeIndex(g)
    Fs, Ts, IDs, VAL = [], [], [], []
    shards = load_shards(m)
    per = max(1, max_trans // max(1, len(shards)))
    for sidx, (fn, S, meta) in enumerate(shards):
        tr = S[:, C_TRIAL].astype(np.int64)
        nxt = np.r_[tr[1:] == tr[:-1], False]                       # row i has a successor in its trial
        ok = nxt.copy()
        ok[:-1] &= S[1:, C_OOB] == 0                                 # successor is not out of bounds
        ok &= np.isfinite(S[:, C_SUB]) & np.r_[np.isfinite(S[1:, C_SUB]), False]
        idx = np.nonzero(ok)[0]
        if len(idx) > per:
            idx = np.sort(rng.choice(idx, per, replace=False))
        P0 = S[idx, C_P].astype(np.float64); V0 = S[idx, C_V].astype(np.float64); W0 = S[idx, C_W].astype(np.float64)
        U = np.c_[S[idx, C_RIGHT] - S[idx, C_LEFT], S[idx, C_FWD] - S[idx, C_BACK]].astype(np.float64)
        J = S[idx, C_JUMP].astype(np.float64)
        first = np.r_[True, tr[1:] != tr[:-1]][idx]                # first row of its trial: the held input was reply 0 w/o jump
        pidx = np.where(first, idx, idx - 1)
        Up = np.c_[S[pidx, C_RIGHT] - S[pidx, C_LEFT], S[pidx, C_FWD] - S[pidx, C_BACK]].astype(np.float64)
        Jp = np.where(first, 0.0, S[pidx, C_JUMP]).astype(np.float64)
        F, yaw = features(g, eidx, P0, V0, W0, U, J, Up, Jp)
        T = targets(P0, V0, W0, S[idx + 1, C_P].astype(np.float64), S[idx + 1, C_V].astype(np.float64),
                    S[idx + 1, C_W].astype(np.float64), S[idx + 1, 11:24].astype(np.float64), yaw)
        Fs.append(F); Ts.append(T)
        IDs.append(np.c_[np.full(len(idx), sidx), tr[idx], S[idx, C_I]].astype(np.int32))
        VAL.append(np.array([val_block(p[0], p[1]) for p in P0], dtype=bool))
    if not Fs:
        return None
    return {'F': np.concatenate(Fs).astype(np.float16), 'T': np.concatenate(Ts), 'ids': np.concatenate(IDs),
            'val': np.concatenate(VAL), 'shards': [s[0] for s in shards]}


def build(max_per_map=None):
    os.makedirs(FEAT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    for m in TRAIN_MAPS + DEV_MAPS:
        t0 = time.time()
        cap = max_per_map or (1_200_000 if m == 'KingOfTheMarble_Hunt_phys' else 500_000 if m in TRAIN_MAPS else 200_000)
        d = build_map(m, cap, rng)
        if d is None:
            print(f'{m}: no data yet', flush=True)
            continue
        np.savez(os.path.join(FEAT_DIR, f'{m}.npz'), **{k: v for k, v in d.items() if k != 'shards'}, shards=np.array(d['shards']))
        print(f'{m}: {len(d["F"])} transitions ({d["val"].mean():.2f} validation), {time.time() - t0:.0f} s', flush=True)


# ------------------------------------------------------------------ model
def make_model(hidden=None, blocks=None):
    hidden = hidden or HIDDEN; blocks = blocks or BLOCKS
    import torch.nn as nn

    class Step(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(N_FEAT, hidden)
            self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
                                                       nn.Linear(hidden, hidden)) for _ in range(blocks)])
            self.norm = nn.LayerNorm(hidden)
            self.mu = nn.Linear(hidden, N_CONT)
            self.logstd = nn.Linear(hidden, N_CONT)
            self.cls = nn.Linear(hidden, 3)                         # support, collision, last contact
            self.normal = nn.Linear(hidden, 3)
            # contact modes (design section 5): the step's outcome given that it collides / does not collide.
            # The collision probability (cls[:, 1]) chooses or weights them; a single mean would blend the two.
            self.mu_mode = nn.Linear(hidden, 2 * N_CONT)

        def forward(self, x):
            h = self.inp(x)
            for b in self.blocks:
                h = h + b(h)
            h = self.norm(h)
            self.last_modes = self.mu_mode(h).view(-1, 2, N_CONT)      # [:, 0] = no collision, [:, 1] = collision
            return self.mu(h), self.logstd(h).clamp(-9, 3), self.cls(h), self.normal(h)

    return Step()


def load_features(maps, split):
    Fs, Ts = [], []
    for m in maps:
        p = os.path.join(FEAT_DIR, f'{m}.npz')
        if not os.path.exists(p):
            continue
        d = np.load(p)
        sel = d['val'] if split == 'val' else (~d['val'] if split == 'train' else np.ones(len(d['val']), bool))
        Fs.append(d['F'][sel]); Ts.append(d['T'][sel])
    return np.concatenate(Fs), np.concatenate(Ts)


def loss_fn(out, T, tmean, tstd, modes=None):
    """Continuous targets normalized per dimension. The mean is fitted by squared error and the spread by a
    Gaussian likelihood around the detached mean: fitting both through one likelihood let the spread run to its
    bound on low-variance dimensions and stall the mean there (first test run, 2026-09-27)."""
    import torch
    import torch.nn.functional as Fn
    mu, ls, cls, nrm = out
    y = (T[:, :N_CONT] - tmean) / tstd
    mse = ((y - mu) ** 2).mean()
    nll = mse + (0.5 * ((y - mu.detach()) / ls.exp()) ** 2 + ls).mean()
    bce = Fn.binary_cross_entropy_with_logits(cls, T[:, N_CONT:N_CONT + 3])
    sup = T[:, N_CONT]
    ln = ((nrm - T[:, N_CONT + 3:N_CONT + 6]) ** 2).sum(1)
    lnorm = (ln * sup).sum() / sup.sum().clamp(min=1)
    lmode = 0.0
    if modes is not None:
        coll = T[:, N_CONT + 1]
        e = ((y.unsqueeze(1) - modes) ** 2).mean(2)                 # (B, 2)
        lmode = (e[:, 0] * (1 - coll)).sum() / (1 - coll).sum().clamp(min=1) + (e[:, 1] * coll).sum() / coll.sum().clamp(min=1)
    return nll + bce + lnorm + lmode, (nll, bce, lnorm)


def train(epochs=6, batch=4096, lr=1e-3, log=print, prefix='step3'):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Ftr, Ttr = load_features(TRAIN_MAPS, 'train')
    Fva, Tva = load_features(TRAIN_MAPS, 'val')
    log(f'train {len(Ftr):,} transitions, validation {len(Fva):,}')
    Fva_t = torch.as_tensor(Fva.astype(np.float32), device=dev); Tva_t = torch.as_tensor(Tva, device=dev)
    tm = Ttr[:, :N_CONT].mean(0); ts = Ttr[:, :N_CONT].std(0) + 1e-6
    tmean = torch.as_tensor(tm, device=dev); tstd = torch.as_tensor(ts, device=dev)
    os.makedirs(MODEL_DIR, exist_ok=True)
    hist = []
    for k in range(ENSEMBLE):
        torch.manual_seed(100 + k)
        rng = np.random.default_rng(100 + k)
        boot = rng.integers(0, len(Ftr), len(Ftr))                  # bootstrap resample per member
        model = make_model().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        steps = epochs * (len(boot) // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
        n = 0
        for ep in range(epochs):
            model.train()
            perm = rng.permutation(boot)
            tot = 0.0; nb = 0
            for b in range(0, len(perm) - batch + 1, batch):
                ib = np.sort(perm[b:b + batch])
                x = torch.as_tensor(Ftr[ib].astype(np.float32), device=dev); t = torch.as_tensor(Ttr[ib], device=dev)
                o = model(x)
                loss, _ = loss_fn(o, t, tmean, tstd, model.last_modes)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += float(loss); nb += 1; n += 1
            model.eval()
            with torch.no_grad():
                vl = []
                for b in range(0, len(Fva_t), 65536):
                    vl.append(float(loss_fn(model(Fva_t[b:b + 65536]), Tva_t[b:b + 65536], tmean, tstd)[0]) * len(Fva_t[b:b + 65536]))
                vloss = sum(vl) / len(Fva_t)
            log(f'member {k} epoch {ep + 1}: train {tot / nb:.4f}  validation {vloss:.4f}')
            hist.append({'member': k, 'epoch': ep + 1, 'train': tot / nb, 'val': vloss})
        torch.save({'state': model.state_dict(), 'n_feat': N_FEAT, 'member': k, 'tmean': tm, 'tstd': ts,
                    'hidden': HIDDEN, 'blocks': BLOCKS},
                   os.path.join(MODEL_DIR, f'{prefix}_{k}.pth'))
    json.dump(hist, open(os.path.join(MODEL_DIR, f'{prefix}_history.json'), 'w'), indent=1)


def load_ensemble(dev=None, prefix='step3'):
    import torch
    dev = dev or ('cuda' if torch.cuda.is_available() else 'cpu')
    ms = []
    for k in range(ENSEMBLE):
        ck = torch.load(os.path.join(MODEL_DIR, f'{prefix}_{k}.pth'), map_location=dev, weights_only=False)
        m = make_model(ck.get('hidden', 512), ck.get('blocks', 3)).to(dev)
        missing = m.load_state_dict(ck['state'], strict=False)       # models from before the mode heads load too
        m.has_modes = not any('mu_mode' in k_ for k_ in missing.missing_keys)
        m.eval()
        m.tmean = np.asarray(ck['tmean'], dtype=np.float32); m.tstd = np.asarray(ck['tstd'], dtype=np.float32)
        ms.append(m)
    return ms


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'build':
        build()
    elif cmd == 'train':
        train()
    else:
        print(__doc__)
