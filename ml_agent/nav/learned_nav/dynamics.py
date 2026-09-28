"""P0b flight predictor (KOTMJUMP_START_HERE.md): a small network trained on the TRAINING starts only.

    python -m nav.learned_nav.dynamics train          # validation-based selection, then the final fit

Input (all in the frame of the start: f = along the start heading, l = to its left, u = up):
* the actual start state: velocity (f, l, u) and spin (f, l, u);
* the height crop: floor relative to the marble's floor level on a 0.5 u grid, 1 u behind to 14.5 u ahead and
  6 u to either side, from the 0.1 u raster (two channels: relative height, floor present), plus the lip and
  gap distance along 15 rays from -70 to +70 deg (the same raster, 0.05 u steps);
* the candidate: jump decision (none, 0..4) and air input (none, 8 directions), one-hot.
Output: the timed path (position relative to the start, every 64 ms decision, 45 decisions), whether the flight
ends safely (no out of bounds; a flight that started lands and survives 0.5 s), whether it lands, the landing
point, and the pickup. Pickup comes from a small head that also sees the gem's position; the physical part
(trunk, path, landing) never sees the gem. The alternative pickup estimate, the predicted path's closest pass
to the gem through a fitted logistic curve, is compared on validation starts and the better one is used.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav.record import (DATA_DIR, CANDS, N_AIR, MAX_STEPS, TARGET, R, fine)   # noqa: E402

MODEL_DIR = os.path.join(HERE, 'models', 'learned_nav')
CROP_F = np.arange(-1.0, 14.51, 0.5)          # 32
CROP_L = np.arange(-6.0, 6.01, 0.5)           # 25
RAY_DEG = np.arange(-70, 71, 10)              # 15
RAY_MAX = 16.0
PATH_SCALE = 10.0
N_CAND = len(CANDS)


# ------------------------------------------------------------------ data
def load_trials(pattern='trials_'):
    """All ok trials from datasets/learned_nav/p0/<pattern>*.jsonl, grouped {start_id: {cand: record}}."""
    by = {}
    for fn in sorted(os.listdir(DATA_DIR)):
        if not (fn.startswith(pattern) and fn.endswith('.jsonl')):
            continue
        for ln in open(os.path.join(DATA_DIR, fn)):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get('schema') != 'p0_trial_v2' or r.get('status') != 'ok':
                continue
            by.setdefault(r['start_id'], {})[r['cand']] = r
    return by


def frame(h0):
    c, s = math.cos(h0), math.sin(h0)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])     # rows: f, l, u


def start_features(p0, v0, w0, h0):
    """Physical state + geometry for one start (no gem)."""
    F = frame(h0)
    v = F @ np.asarray(v0, dtype=np.float64)
    w = F @ np.asarray(w0, dtype=np.float64)
    zf = float(p0[2]) - R
    fz = fine()
    ff, ll = np.meshgrid(CROP_F, CROP_L, indexing='ij')
    wx = p0[0] + F[0, 0] * ff + F[1, 0] * ll
    wy = p0[1] + F[0, 1] * ff + F[1, 1] * ll
    h = fz.heights_at(wx.ravel(), wy.ravel())                       # (K, N)
    h = np.where(np.isfinite(h) & (h <= zf + 1.5), h, np.nan)
    top = np.nanmax(np.where(np.isfinite(h), h, -np.inf), axis=0)
    present = np.isfinite(top)
    rel = np.where(present, np.clip(top - zf, -3.0, 3.0), 0.0)
    rays = []
    ds = np.arange(1, int(RAY_MAX / 0.05) + 1) * 0.05
    for deg in RAY_DEG:
        a = h0 + math.radians(float(deg))
        hh = fz.heights_at(p0[0] + math.cos(a) * ds, p0[1] + math.sin(a) * ds)
        on = np.any(np.isfinite(hh) & (np.abs(hh - zf) < 0.3), axis=0)
        off = np.nonzero(~on)[0]
        if len(off) == 0:
            rays += [RAY_MAX, 0.0]
            continue
        lip = float(ds[off[0]])
        back = np.nonzero(on[off[0]:])[0]
        gap = float(ds[off[0] + back[0]] - lip) if len(back) else RAY_MAX
        rays += [lip, gap]
    kin = np.concatenate([v / 10.0, w / 40.0])
    return np.concatenate([kin, rel.astype(np.float64), present.astype(np.float64), np.asarray(rays) / RAY_MAX]).astype(np.float32)


def gem_rel(p0, h0):
    return (frame(h0) @ (np.asarray(TARGET) - np.asarray(p0, dtype=np.float64)) / 10.0).astype(np.float32)


def cand_features():
    """(N_CAND, 15): jump one-hot (none, 0..4) + air one-hot (9; all zero for no jump)."""
    X = np.zeros((N_CAND, 6 + N_AIR), dtype=np.float32)
    for ci, c in enumerate(CANDS):
        if c is None:
            X[ci, 0] = 1
        else:
            X[ci, 1 + c[0]] = 1
            X[ci, 6 + c[1]] = 1
    return X


CAND_X = cand_features()


def path_target(r):
    """(MAX_STEPS, 3) positions relative to the start in its frame, and the mask of recorded decisions."""
    F = frame(r['h0'])
    p0 = np.asarray(r['p0'], dtype=np.float64)
    P = np.zeros((MAX_STEPS, 3), dtype=np.float32); M = np.zeros(MAX_STEPS, dtype=np.float32)
    for st in r['steps']:
        i = st['i']
        if i < MAX_STEPS:
            P[i] = F @ (np.asarray(st['p']) - p0); M[i] = 1
    return P, M


def build(by, start_ids):
    """Arrays over (start, candidate) pairs that were recorded."""
    rows = {'si': [], 'g': [], 'c': [], 'path': [], 'mask': [], 'safe': [], 'pick': [], 'succ': [], 'landed': [], 'land': [],
            'sid': [], 'ci': []}
    feats = {}; table = []
    for sid in start_ids:
        d = by[sid]
        any_r = next(iter(d.values()))
        feats[sid] = len(table)
        table.append(start_features(any_r['p0'], any_r['v0'], any_r['w0'], any_r['h0']))
        g = gem_rel(any_r['p0'], any_r['h0'])
        for ci, r in d.items():
            P, M = path_target(r)
            rows['si'].append(feats[sid]); rows['g'].append(g); rows['c'].append(CAND_X[ci])
            rows['path'].append(P); rows['mask'].append(M)
            rows['safe'].append(float(r['safe'])); rows['pick'].append(float(r['pickup'])); rows['succ'].append(float(r['success']))
            rows['landed'].append(float(r['landed'] is not None))
            lp = np.zeros(3, dtype=np.float32)
            if r['land_p'] is not None:
                lp = (frame(r['h0']) @ (np.asarray(r['land_p']) - np.asarray(r['p0']))).astype(np.float32)
            rows['land'].append(lp)
            rows['sid'].append(sid); rows['ci'].append(ci)
    out = {k: np.asarray(v) for k, v in rows.items() if k not in ('sid', 'ci')}
    out['s_tab'] = np.asarray(table, dtype=np.float32)          # one row of start features per start;
    out['si'] = np.asarray(rows['si'], dtype=np.int64)           # each trial row points at its start
    out['sid'] = rows['sid']; out['ci'] = np.asarray(rows['ci'])
    return out


N_CROP = len(CROP_F) * len(CROP_L)
AIR_MIRROR = [0] + [1 + ((9 - a) % 8) for a in range(1, N_AIR)]      # heading + t deg -> heading - t deg


def mirror(data):
    """The same trials reflected left-right (l -> -l): a physically valid flight in the mirrored world. Velocity and
    path flip l; spin is an axial vector, w -> (-w_f, w_l, -w_u); the crop flips its lateral axis; the rays reverse;
    air directions mirror about the heading."""
    out = dict(data)
    S = data['s_tab'].copy()
    S[:, 1] *= -1                                                   # v_l
    S[:, 3] *= -1; S[:, 5] *= -1                                    # w_f, w_u
    shp = (len(S), len(CROP_F), len(CROP_L))
    for off in (6, 6 + N_CROP):
        S[:, off:off + N_CROP] = S[:, off:off + N_CROP].reshape(shp)[:, :, ::-1].reshape(len(S), -1)
    r0 = 6 + 2 * N_CROP
    rays = S[:, r0:].reshape(len(S), len(RAY_DEG), 2)[:, ::-1, :]
    S[:, r0:] = rays.reshape(len(S), -1)
    out['s_tab'] = S
    g = data['g'].copy(); g[:, 1] *= -1; out['g'] = g
    C = data['c'].copy(); air = C[:, 6:].copy(); C[:, 6:] = air[:, AIR_MIRROR]; out['c'] = C
    P = data['path'].copy(); P[:, :, 1] *= -1; out['path'] = P
    L = data['land'].copy(); L[:, 1] *= -1; out['land'] = L
    return out


def with_mirror(data):
    m = mirror(data)
    out = {k: np.concatenate([data[k], m[k]]) for k in data if k not in ('sid', 'ci', 'si')}
    out['si'] = np.concatenate([data['si'], m['si'] + len(data['s_tab'])])
    out['sid'] = list(data['sid']) * 2; out['ci'] = np.concatenate([data['ci'], data['ci']])
    return out


# ------------------------------------------------------------------ model
def make_model(n_s, hidden=384, dropout=0.1):
    import torch
    import torch.nn as nn

    class Flight(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(n_s + CAND_X.shape[1], hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU())
            self.path = nn.Linear(hidden, MAX_STEPS * 3)
            self.out = nn.Linear(hidden, 2 + 3)            # safe, landed logits; landing point
            self.pick = nn.Sequential(nn.Linear(hidden + 3, 128), nn.ReLU(), nn.Linear(128, 2))   # pickup, success

        def forward(self, s, c, g):
            z = self.trunk(torch.cat([s, c], dim=1))
            path = self.path(z).view(-1, MAX_STEPS, 3) * PATH_SCALE
            o = self.out(z)
            k = self.pick(torch.cat([z, g], dim=1))
            return {'path': path, 'safe': o[:, 0], 'landed': o[:, 1], 'land': o[:, 2:5] * PATH_SCALE, 'pick': k[:, 0],
                    'succ': k[:, 1]}

    return Flight()


def closest_pass(path, g):
    """Closest distance (u) from the start (origin) and the predicted path, interpolated 8x between
    decisions, to the gem g (frame units, same scale as path). path (N, T, 3), g (N, 3)."""
    import torch
    z = torch.zeros_like(path[:, :1])
    p = torch.cat([z, path], dim=1)
    a, b = p[:, :-1], p[:, 1:]
    ts = torch.linspace(0, 1, 8, device=path.device).view(1, 1, 8, 1)
    pts = a.unsqueeze(2) + (b - a).unsqueeze(2) * ts                 # (N, T, 8, 3)
    d = (pts - g.view(-1, 1, 1, 3)).norm(dim=3)
    return d.flatten(1).min(dim=1).values


def train(data, epochs=60, seed=0, device=None, log=print, eval_fn=None, eval_every=5):
    import torch
    import torch.nn.functional as Fn
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    T = {k: torch.as_tensor(v, device=device) for k, v in data.items() if k not in ('sid', 'ci')}
    n = len(T['si'])
    model = make_model(T['s_tab'].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for k in range(0, n, 512):
            idx = perm[k:k + 512]
            o = model(T['s_tab'][T['si'][idx]], T['c'][idx], T['g'][idx])
            m = T['mask'][idx]
            lp = (Fn.smooth_l1_loss(o['path'], T['path'][idx], reduction='none').sum(2) * m).sum() / m.sum().clamp(min=1)
            ls = Fn.binary_cross_entropy_with_logits(o['safe'], T['safe'][idx])
            ll = Fn.binary_cross_entropy_with_logits(o['landed'], T['landed'][idx])
            lm = T['landed'][idx]
            lland = (Fn.smooth_l1_loss(o['land'], T['land'][idx], reduction='none').sum(1) * lm).sum() / lm.sum().clamp(min=1)
            lk = Fn.binary_cross_entropy_with_logits(o['pick'], T['pick'][idx])
            lsu = Fn.binary_cross_entropy_with_logits(o['succ'], T['succ'][idx])
            loss = lp * 0.2 + ls + ll + lland * 0.2 + lk + lsu
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        rec = {'epoch': ep + 1, 'loss': tot / n}
        if eval_fn is not None and ((ep + 1) % eval_every == 0 or ep + 1 == epochs):
            rec.update(eval_fn(model))
            log(f'epoch {ep + 1}: ' + ', '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k} {v}' for k, v in rec.items()))
        hist.append(rec)
    return model, hist


def predict(model, S, G, device=None):
    """S (n_starts, n_s), G (n_starts, 3) -> dict of (n_starts, N_CAND[, ...]) numpy arrays for all candidates."""
    import torch
    device = device or next(model.parameters()).device
    model.eval()
    n = len(S)
    s = torch.as_tensor(np.repeat(S, N_CAND, axis=0), device=device)
    g = torch.as_tensor(np.repeat(G, N_CAND, axis=0), device=device)
    c = torch.as_tensor(np.tile(CAND_X, (n, 1)), device=device)
    with torch.no_grad():
        o = model(s, c, g)
        dmin = closest_pass(o['path'], g * 10.0)
    out = {'p_safe': torch.sigmoid(o['safe']), 'p_landed': torch.sigmoid(o['landed']), 'p_pick': torch.sigmoid(o['pick']),
           'p_succ': torch.sigmoid(o['succ']),
           'dmin': dmin, 'path': o['path'], 'land': o['land']}
    return {k: v.cpu().numpy().reshape((n, N_CAND) + tuple(v.shape[1:])) for k, v in out.items()}


def fit_pass_curve(dmin, pick):
    """Logistic fit P(pickup) = sigmoid(a (r0 - dmin)) on training predictions (grid search, log loss)."""
    best = None
    for r0 in np.arange(0.2, 2.01, 0.05):
        for a in (2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0):
            p = 1 / (1 + np.exp(-a * (r0 - dmin)))
            p = np.clip(p, 1e-4, 1 - 1e-4)
            ll = -np.mean(pick * np.log(p) + (1 - pick) * np.log(1 - p))
            if best is None or ll < best[0]:
                best = (ll, float(r0), float(a))
    return {'r0': best[1], 'a': best[2], 'logloss': best[0]}


def pass_prob(dmin, curve):
    return 1 / (1 + np.exp(-curve['a'] * (curve['r0'] - dmin)))


VARIANTS = ('direct', 'pass', 'joint')


def score(pr, variant, curve=None):
    """(P(pickup), ranking score) for all candidates. direct: P(pickup) x P(safe) from the two heads; pass: the
    predicted path's closest pass to the gem through the fitted curve, x P(safe); joint: the success head."""
    if variant == 'direct':
        return pr['p_pick'], pr['p_pick'] * pr['p_safe']
    if variant == 'pass':
        pp = pass_prob(pr['dmin'], curve)
        return pp, pp * pr['p_safe']
    return pr['p_pick'], pr['p_succ']


def save(model, meta, path):
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'state': model.state_dict(), 'meta': meta}, path)


def load(path, device=None):
    import torch
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(path, map_location=device, weights_only=False)     # our own file
    model = make_model(ck['meta']['n_s']).to(device)
    model.load_state_dict(ck['state']); model.eval()
    return model, ck['meta']


def main():
    from nav.learned_nav import evaluate as E
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'train'
    if cmd == 'train':
        E.train_and_select()


if __name__ == '__main__':
    main()
