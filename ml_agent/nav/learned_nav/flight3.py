"""Stage 3 direct flight head (design section 5, flight prediction contract), learned from stage 3 jump trials.

    python -m nav.learned_nav.flight3 build       # samples from the recorded jump trials (parallel, exact edge rays)
    python -m nav.learned_nav.flight3 train       # -> models/learned_nav/flight3.pth

A sample starts at the decision where the jump command is chosen: the marble's measured state then (row j-1, or the
exact teleported start when j = 0), the reply already pending (it acts first), then the exact replies of the next
H decisions as world force vectors turned into the start's heading frame, with their jump keys. Geometry: a crop in
the heading frame, 2 u behind to 16 u ahead and 6 u to each side at 0.5 u (floor relative to the marble's bottom,
floor present, blocked) and the exact edge rays of the start floor in 15 directions (-70 to +70 deg).
Outputs: the path (position relative to the start, heading frame) for H decisions, masked where not recorded; the
takeoff decision; landed; the landing point; safe (the trial did not end out of bounds and a flight that started
landed); exit speed. Split: 8 u blocks, one in eight validation, as the step model; the two held-out maps apart.
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav import dynamics3 as D3                                     # noqa: E402
from nav.learned_nav.geometry import Geometry                                   # noqa: E402

R = 0.19
H = 24                       # decisions predicted (1.5 s)
CF = np.arange(-2.0, 16.01, 0.5)             # 37
CL = np.arange(-6.0, 6.01, 0.5)              # 25
RAY_DEG = np.arange(-70, 71, 10)             # 15
FEAT_DIR = D3.FEAT_DIR
N_STATE = 12
N_SEQ = H * 3
N_GEO = 3 * len(CF) * len(CL) + 3 * len(RAY_DEG)
N_FEAT = N_STATE + N_SEQ + N_GEO


def _rot(x, y, c, s):
    return c * x + s * y, -s * x + c * y


def sample_features(g, p0, v0, w0, tel, yaw, useq, jseq):
    """One flight sample's features. useq (H+1, 2) world input vectors (pending first), jseq (H+1,) jump keys."""
    c, s = math.cos(yaw), math.sin(yaw)
    vf, vl = _rot(v0[0], v0[1], c, s); wf, wl = _rot(w0[0], w0[1], c, s)
    bottom = p0[2] - R
    zf = g.floor_below(p0[0], p0[1], p0[2])
    gap = min(3.0, bottom - zf) if zf is not None else 3.0
    n = g.normal_below(p0[0], p0[1], p0[2]) if zf is not None else None
    n = np.array([0, 0, 1.0]) if n is None else n
    nf, nl = _rot(n[0], n[1], c, s)
    sup = 0.0 if tel is None else float(tel[2] > 0)
    state = [vf / 10, vl / 10, v0[2] / 10, wf / 40, wl / 40, w0[2] / 40, gap / 3, nf, nl, n[2], sup, math.hypot(v0[0], v0[1]) / 10]
    uf, ul = _rot(useq[:, 0], useq[:, 1], c, s)
    seq = np.c_[uf, ul, jseq][:H].ravel()
    ff, ll = np.meshgrid(CF, CL, indexing='ij'); ff = ff.ravel(); ll = ll.ravel()
    qx = p0[0] + c * ff - s * ll; qy = p0[1] + s * ff + c * ll
    lv, ab = D3.level_below(g, qx, qy, np.full(len(qx), bottom + 0.6))
    pres = np.isfinite(lv); rel = np.where(pres, np.clip(lv - bottom, -3, 1), -3.0)
    zref = zf if zf is not None else bottom
    rays = g.edge_rays(p0[0], p0[1], zref, yaw + np.radians(RAY_DEG), max_u=16.0)
    rd = np.where(np.isfinite(rays[:, 0]), rays[:, 0], 16.0) / 16.0
    rt = rays[:, 1] / 2.0; rdrop = np.clip(rays[:, 2], -3, 5) / 5.0
    return np.concatenate([state, seq, rel / 3, pres, ab, rd, rt, rdrop]).astype(np.float32)


def flight_sample(g, rows, mt, d):
    """Sample starting at decision d of a jump trial (d <= jump decision): features, targets."""
    j = int(mt['prog']['j'])
    if d == 0:
        p0 = np.asarray(mt['start']['pos']); v0 = np.asarray(mt['start']['vel']); w0 = np.asarray(mt['start']['spin']); tel = None
        pend = np.array([rows[0][D3.C_RIGHT] - rows[0][D3.C_LEFT], rows[0][D3.C_FWD] - rows[0][D3.C_BACK], 0.0])  # held, no jump key
    else:
        r = rows[d - 1]
        p0 = r[D3.C_P].astype(np.float64); v0 = r[D3.C_V].astype(np.float64); w0 = r[D3.C_W].astype(np.float64); tel = r[11:24]
        pend = np.array([r[D3.C_RIGHT] - r[D3.C_LEFT], r[D3.C_FWD] - r[D3.C_BACK], r[D3.C_JUMP]])
    seq = [pend] + [np.array([r[D3.C_RIGHT] - r[D3.C_LEFT], r[D3.C_FWD] - r[D3.C_BACK], r[D3.C_JUMP]]) for r in rows[d:d + H]]
    seq = np.asarray(seq, dtype=np.float64)
    if len(seq) < H + 1:
        seq = np.r_[seq, np.zeros((H + 1 - len(seq), 3))]
    useq, jseq = seq[:, :2], seq[:, 2]
    if math.hypot(v0[0], v0[1]) > 0.5:
        yaw = math.atan2(v0[1], v0[0])
    else:
        u = useq[1]; yaw = math.atan2(u[1], u[0]) if math.hypot(*u) > 0.1 else 0.0
    F = sample_features(g, p0, v0, w0, tel, yaw, useq, jseq)
    c, s = math.cos(yaw), math.sin(yaw)
    P = rows[d:d + H, D3.C_P].astype(np.float64) - p0
    path = np.zeros((H, 3), np.float32); mask = np.zeros(H, np.float32)
    pf, pl = _rot(P[:, 0], P[:, 1], c, s)
    path[:len(P)] = np.c_[pf, pl, P[:, 2]]; mask[:len(P)] = 1
    took = mt['took_off']; landed = mt['landed']
    take_rel = (took - d) if (took is not None and took >= d) else -1
    safe = (not mt['oob']) and (took is None or landed is not None) and not mt['censored']
    lp = np.zeros(3, np.float32)
    if landed is not None:
        q = rows[landed, D3.C_P].astype(np.float64) - p0
        lf, ll_ = _rot(q[0], q[1], c, s); lp = np.array([lf, ll_, q[2]], np.float32)
    T = np.r_[path.ravel(), mask, take_rel, (landed - d) if landed is not None else -1, float(landed is not None), lp, float(safe)]
    return F, T.astype(np.float32), p0


def _shard_samples(args):
    m, path, max_samples, seed = args
    g = Geometry(m)
    rng = np.random.default_rng(seed)
    z = np.load(path); S = z['steps']; meta = json.loads(str(z['meta']))
    Fs, Ts, Ms, VAL = [], [], [], []
    tr = S[:, D3.C_TRIAL].astype(np.int64)
    starts_idx = np.r_[0, np.nonzero(np.diff(tr))[0] + 1]
    for i0 in starts_idx:
        mt = meta[int(tr[i0])]
        if not mt['prog']['family'].startswith('jump'):
            continue
        rows = S[i0:i0 + mt['n']]
        j = int(mt['prog']['j'])
        if j >= len(rows) - 2:
            continue
        ds = [j] + ([int(rng.integers(max(0, j - 6), j))] if j >= 1 else [])
        for d in ds:
            F, T, p0 = flight_sample(g, rows, mt, d)
            Fs.append(F); Ts.append(T); Ms.append((os.path.basename(path), int(tr[i0]), d)); VAL.append(D3.val_block(p0[0], p0[1]))
        if len(Fs) >= max_samples:
            break
    return m, (np.asarray(Fs, np.float16).reshape(-1, N_FEAT), np.asarray(Ts, np.float32), Ms, np.asarray(VAL, bool))


def build(per_map=None, workers=10):
    os.makedirs(FEAT_DIR, exist_ok=True)
    jobs = []
    for k, m in enumerate(D3.TRAIN_MAPS + D3.DEV_MAPS):
        d = os.path.join(D3.STAGE3, m)
        files = sorted(f for f in os.listdir(d) if f.startswith('shard_') and f.endswith('.npz')) if os.path.isdir(d) else []
        cap = per_map or (300_000 if m == 'KingOfTheMarble_Hunt_phys' else 120_000 if m in D3.TRAIN_MAPS else 30_000)
        for q, f in enumerate(files):
            jobs.append((m, os.path.join(d, f), max(1, cap // max(1, len(files))), 1000 * k + q))
    t0 = time.time()
    acc = {}
    with Pool(workers) as pool:
        for m, part in pool.imap_unordered(_shard_samples, jobs):
            acc.setdefault(m, []).append(part)
    for m, parts in acc.items():
        F = np.concatenate([p[0] for p in parts]); T = np.concatenate([p[1] for p in parts])
        V = np.concatenate([p[3] for p in parts]); M = sum((p[2] for p in parts), [])
        if not len(F):
            continue
        np.savez(os.path.join(FEAT_DIR, f'flight_{m}.npz'), F=F, T=T, val=V, meta=np.array(json.dumps(M)))
        print(f'{m}: {len(F)} flight samples ({V.mean():.2f} validation)', flush=True)
    print(f'flight build {time.time() - t0:.0f} s', flush=True)


# ------------------------------------------------------------------ model
OFF_MASK = H * 3
OFF_TAKE = OFF_MASK + H
OFF_LAND = OFF_TAKE + 1
OFF_LANDED = OFF_LAND + 1
OFF_LP = OFF_LANDED + 1
OFF_SAFE = OFF_LP + 3


def make_model(hidden=768):
    import torch
    import torch.nn as nn

    class Flight(nn.Module):
        def __init__(self):
            super().__init__()
            n_g = 3 * len(CF) * len(CL)
            self.geo = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
                                     nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d((5, 4)), nn.Flatten())
            n_in = N_STATE + N_SEQ + 32 * 20 + 3 * len(RAY_DEG)
            self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU())
            self.path = nn.Linear(hidden, H * 3)
            self.take = nn.Linear(hidden, H + 1)                  # takeoff decision 0..H-1, or none (last)
            self.landt = nn.Linear(hidden, H + 1)                 # landing decision 0..H-1, or none / later (last)
            self.heads = nn.Linear(hidden, 1 + 3 + 1)             # landed, landing point, safe

        def forward(self, x):
            a = N_STATE + N_SEQ
            grid = x[:, a:a + 3 * len(CF) * len(CL)].view(-1, 3, len(CF), len(CL))
            rays = x[:, a + 3 * len(CF) * len(CL):]
            z = self.mlp(torch.cat([x[:, :a], self.geo(grid), rays], dim=1))
            h = self.heads(z)
            # the path is a correction to "keep the start velocity" (heading frame): the network learns the jump,
            # the air control and the contacts, not the straight-line part of every flight
            v = x[:, 0:3] * 10.0
            tt = torch.arange(1, H + 1, device=x.device, dtype=x.dtype).view(1, H, 1) * 0.064
            prior = v.view(-1, 1, 3) * tt
            return {'path': prior + self.path(z).view(-1, H, 3) * 5.0, 'take': self.take(z), 'landt': self.landt(z), 'landed': h[:, 0],
                    'land': h[:, 1:4] * 5.0, 'safe': h[:, 4]}

    return Flight()


def load_flight(maps, split):
    Fs, Ts = [], []
    for m in maps:
        p = os.path.join(FEAT_DIR, f'flight_{m}.npz')
        if not os.path.exists(p):
            continue
        d = np.load(p)
        sel = d['val'] if split == 'val' else (~d['val'] if split == 'train' else np.ones(len(d['val']), bool))
        Fs.append(d['F'][sel]); Ts.append(d['T'][sel])
    return np.concatenate(Fs), np.concatenate(Ts)


def flight_loss(o, T):
    import torch
    import torch.nn.functional as Fn
    path = T[:, :H * 3].view(-1, H, 3); mask = T[:, OFF_MASK:OFF_MASK + H]
    lp = (Fn.smooth_l1_loss(o['path'], path, reduction='none').sum(2) * mask).sum() / mask.sum().clamp(min=1)
    take = T[:, OFF_TAKE].long(); take = torch.where(take < 0, torch.full_like(take, H), take.clamp(max=H - 1))
    lt = Fn.cross_entropy(o['take'], take)
    lnd = T[:, OFF_LAND].long(); lnd = torch.where((lnd < 0) | (lnd >= H), torch.full_like(lnd, H), lnd)
    lt = lt + Fn.cross_entropy(o['landt'], lnd)
    landed = T[:, OFF_LANDED]
    ll = Fn.binary_cross_entropy_with_logits(o['landed'], landed)
    lland = (Fn.smooth_l1_loss(o['land'], T[:, OFF_LP:OFF_LP + 3], reduction='none').sum(1) * landed).sum() / landed.sum().clamp(min=1)
    ls = Fn.binary_cross_entropy_with_logits(o['safe'], T[:, OFF_SAFE])
    return lp + lt + ll + lland + ls, {'path': float(lp), 'take': float(lt), 'safe': float(ls), 'land': float(lland)}


def train(epochs=10, batch=1024, lr=1e-3, log=print, maps=None, out_name='flight3.pth'):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    maps = maps or D3.TRAIN_MAPS
    Ftr, Ttr = load_flight(maps, 'train'); Fva, Tva = load_flight(maps, 'val')
    log(f'flight samples: train {len(Ftr):,}, validation {len(Fva):,}')
    torch.manual_seed(0); rng = np.random.default_rng(0)
    model = make_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(Ftr) // batch))
    Fva_t = torch.as_tensor(Fva.astype(np.float32), device=dev); Tva_t = torch.as_tensor(Tva, device=dev)
    hist = []
    for ep in range(epochs):
        model.train(); perm = rng.permutation(len(Ftr)); tot = 0; nb = 0
        for b in range(0, len(perm) - batch + 1, batch):
            ib = np.sort(perm[b:b + batch])
            loss, _ = flight_loss(model(torch.as_tensor(Ftr[ib].astype(np.float32), device=dev)), torch.as_tensor(Ttr[ib], device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss); nb += 1
        model.eval()
        with torch.no_grad():
            parts = []
            for b in range(0, len(Fva_t), 8192):
                l, d = flight_loss(model(Fva_t[b:b + 8192]), Tva_t[b:b + 8192]); parts.append((float(l), d, len(Fva_t[b:b + 8192])))
            vl = sum(p[0] * p[2] for p in parts) / len(Fva_t)
        log(f'flight epoch {ep + 1}: train {tot / nb:.4f} validation {vl:.4f} ' + str({k: round(np.mean([p[1][k] for p in parts]), 4) for k in parts[0][1]}))
        hist.append({'epoch': ep + 1, 'train': tot / nb, 'val': vl})
    torch.save({'state': model.state_dict(), 'hist': hist, 'maps': maps}, os.path.join(D3.MODEL_DIR, out_name))
    return model


def load(dev=None, name='flight3.pth'):
    import torch
    dev = dev or ('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(os.path.join(D3.MODEL_DIR, name), map_location=dev, weights_only=False)
    m = make_model().to(dev); m.load_state_dict(ck['state']); m.eval()
    return m


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'build':
        build()
    elif cmd == 'train':
        train()
    else:
        print(__doc__)
