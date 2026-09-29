"""Time-to-gem guide (design section 6, from M3): seconds from a rolling state to a gem, learned from PPO's recorded
KOTM play. A search heuristic for ordinary rolling only: PPO rarely jumps, so it overestimates jump routes; it
guides expansion and is never used to prune or reject a route, and it is kept apart from any reward.

    python -m nav.learned_nav.guide          # -> models/learned_nav/guide_ttg.pth, logs/learned_nav/guide_eval.json

Data: the navigator training traces logs/nav/trace_*.csv (one row per 64 ms decision, 8 game instances interleaved).
Instances 0-6 play King of the Marble (7 is the Islands map of the 7:1 rotation; rows below z 15 are dropped too).
A goal segment is the run of an instance's rows with the same goal (gx, gy); it ends in a pickup when its last row
carries the pickup reward (>= 10) and the next row has a new goal. Label of each row of a completed segment: the
seconds until that pickup. Segments ending otherwise (fall, round end, a goal change without a pickup) are dropped.
Features, in the frame of the velocity (below 0.5 u/s: of the goal direction): goal ahead / left, distance, speed,
vertical speed, on floor. Validation: the most recent trace file, never trained on. Baseline: distance over the
best constant speed (fitted on training rows).
"""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

TRACE_GLOB = os.path.join(HERE, 'logs', 'nav', 'trace_*.csv')
MODEL = os.path.join(HERE, 'models', 'learned_nav', 'guide_ttg.pth')
OUT = os.path.join(HERE, 'logs', 'learned_nav', 'guide_eval.json')
DT = 0.064
PICKUP_REWARD = 10.0
MAX_ROWS_PER_FILE = 6_000_000
T_MAX = 20.0                 # longer legs (stalls) are dropped


def features(x, y, vx, vy, vz, on_floor, gx, gy):
    dx, dy = gx - x, gy - y
    sp = np.hypot(vx, vy)
    yaw = np.where(sp > 0.5, np.arctan2(vy, vx), np.arctan2(dy, dx))
    c, s = np.cos(yaw), np.sin(yaw)
    f, l = c * dx + s * dy, -s * dx + c * dy
    d = np.hypot(dx, dy)
    return np.c_[f / 20, l / 20, d / 20, sp / 10, vz / 10, on_floor].astype(np.float32)


def load_file(path):
    import pandas as pd
    cols = ['inst', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'on_floor', 'reward', 'gx', 'gy']
    df = pd.read_csv(path, usecols=cols, nrows=MAX_ROWS_PER_FILE)
    Fs, Ts = [], []
    for inst, d in df.groupby('inst', sort=False):
        if inst == 7:
            continue
        a = {k: d[k].to_numpy() for k in cols}
        n = len(d)
        if n < 10:
            continue
        g = np.c_[a['gx'], a['gy']]
        change = np.r_[np.any(g[1:] != g[:-1], axis=1), True]           # row i is the last of its segment
        seg = np.r_[0, np.cumsum(change[:-1])]
        picked_end = change & (a['reward'] >= PICKUP_REWARD)
        picked_end[-1] = False                                           # the file may end mid-segment
        last_idx = np.full(seg.max() + 1, -1)
        ends = np.nonzero(change)[0]
        last_idx[seg[ends]] = ends
        ok_seg = np.zeros(seg.max() + 1, bool); ok_seg[seg[ends[picked_end[ends]]]] = True
        t = (last_idx[seg] - np.arange(n)) * DT
        keep = ok_seg[seg] & (t <= T_MAX) & (a['z'] > 15.0)
        if keep.sum() == 0:
            continue
        Fs.append(features(a['x'][keep], a['y'][keep], a['vx'][keep], a['vy'][keep], a['vz'][keep],
                           a['on_floor'][keep].astype(float), a['gx'][keep], a['gy'][keep]))
        Ts.append(t[keep].astype(np.float32))
    return np.concatenate(Fs), np.concatenate(Ts)


def make_model():
    import torch.nn as nn
    return nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                         nn.Linear(128, 1))


def main(log=print):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    files = sorted(glob.glob(TRACE_GLOB), key=os.path.getmtime)
    files = [f for f in files if os.path.getsize(f) > 50_000_000]
    val_f, train_f = files[-1], files[:-1]
    t0 = time.time()
    Ftr, Ttr = zip(*[load_file(f) for f in train_f]); Ftr = np.concatenate(Ftr); Ttr = np.concatenate(Ttr)
    Fva, Tva = load_file(val_f)
    log(f'rows: train {len(Ttr)} from {len(train_f)} files, validation {len(Tva)} ({os.path.basename(val_f)}), {time.time() - t0:.0f} s')
    # baseline: distance / constant speed (least squares through the origin)
    dtr = Ftr[:, 2] * 20
    k = float((dtr * Ttr).sum() / (dtr * dtr).sum())
    model = make_model().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = torch.as_tensor(Ftr, device=dev); Y = torch.as_tensor(Ttr, device=dev)
    Xv = torch.as_tensor(Fva, device=dev)
    n = len(X)
    for ep in range(8):
        perm = torch.randperm(n, device=dev)
        for b in range(0, n, 8192):
            i = perm[b:b + 8192]
            loss = torch.nn.functional.smooth_l1_loss(model(X[i]).squeeze(1), Y[i])
            opt.zero_grad(); loss.backward(); opt.step()
        for g_ in opt.param_groups:
            g_['lr'] *= 0.7
        with torch.no_grad():
            pv = torch.cat([model(Xv[b:b + 65536]).squeeze(1) for b in range(0, len(Xv), 65536)]).cpu().numpy()
        log(f'epoch {ep}: validation MAE {np.abs(pv - Tva).mean():.3f} s')
    base = k * Fva[:, 2] * 20
    err, berr = np.abs(pv - Tva), np.abs(base - Tva)
    bins = [0, 1, 2, 4, 8, T_MAX + 1]
    by = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (Tva >= lo) & (Tva < hi)
        by.append({'t_s': [lo, hi], 'rows': int(m.sum()), 'mae_guide': float(err[m].mean()) if m.any() else None,
                   'mae_distance': float(berr[m].mean()) if m.any() else None})
    rep = {'train_files': [os.path.basename(f) for f in train_f], 'val_file': os.path.basename(val_f),
           'rows_train': int(len(Ttr)), 'rows_val': int(len(Tva)), 'baseline_s_per_u': k,
           'mae_guide_s': float(err.mean()), 'mae_distance_baseline_s': float(berr.mean()),
           'median_err_guide_s': float(np.median(err)), 'median_err_baseline_s': float(np.median(berr)),
           'by_time_to_go': by, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    torch.save({'state': model.state_dict(), 'schema': 'guide_ttg_v1', 'features': 'f,l,d /20, speed/10, vz/10, on_floor'}, MODEL)
    json.dump(rep, open(OUT, 'w'), indent=1)
    log(json.dumps(rep, indent=1))


class Guide:
    """Seconds from states (x, y, vx, vy, vz, on_floor arrays) to goal points (gx, gy)."""

    def __init__(self, dev=None):
        import torch
        self.dev = dev or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = make_model().to(self.dev)
        self.model.load_state_dict(torch.load(MODEL, map_location=self.dev)['state'])
        self.model.eval()

    def __call__(self, x, y, vx, vy, vz, on_floor, gx, gy):
        import torch
        F = features(x, y, vx, vy, vz, on_floor, gx, gy)
        with torch.no_grad():
            return self.model(torch.as_tensor(F, device=self.dev)).squeeze(1).cpu().numpy()


if __name__ == '__main__':
    main(log=lambda s: print(s, flush=True))
