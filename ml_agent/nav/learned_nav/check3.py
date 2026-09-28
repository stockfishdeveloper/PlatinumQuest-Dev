"""Stage 3 M1 gate checks on one map: repeatability and natural-state replay.

    python -m nav.learned_nav.check3 --port 9011 --map Sprawl_Hunt_phys

(a) Repeats: REPEAT_TRIALS sampled (start, controls) trials, each run REPEATS times: every recorded position within
    REPEAT_TOL of the first run, the same takeoff / landing / out-of-bounds decisions (the out-of-bounds flag may be
    a decision late or early, as in P0).
(b) Natural-state replay: NATURAL natural trajectories (a sampled start, then REPLAY_LEN decisions of recorded
    replies). At a cut decision t (no jump key at t or t+1), the marble is teleported into the recorded state of
    decision t with reply t held, and replies t+1, t+2, ... are replayed. The replayed states must match the natural
    ones: declared (2026-09-27, before the first run) pass rule: 95 % of replays within NATURAL_TOL over the next
    HORIZON decisions. A mismatch means hidden engine state (contact history, ground time, pending input) that a
    teleported start does not carry.
Results: datasets/learned_nav/stage3/<map>/check3.json
"""
import argparse
import json
import os
import time

import numpy as np

from nav.learned_nav.record3 import Recorder3, StartSampler, sample_controls, DATA_DIR
from nav.learned_nav.session import RoundOver
from nav.protocol import RAW_POS, RAW_VEL, RAW_SPIN

REPEAT_TRIALS = 10
REPEATS = 20
REPEAT_TOL = 0.01
NATURAL = 40
REPLAY_LEN = 40
HORIZON = 15
NATURAL_TOL = 0.02


def run_natural(rec, start, prog):
    """A trial whose replies are recorded as sent: returns rows of (pos, vel, spin, js)."""
    meta, rows = rec.trial(start, dict(prog, len=REPLAY_LEN))
    return meta, rows


def replay(rec, rows, t, nudge=None):
    """Teleport into row t's state with row t's reply held, replay rows t+1.. replies; returns the positions.
    nudge: an rng; when given, position and velocity are offset by up to half the reporting precision."""
    s = rec.s
    st = rows[t]
    pos, vel, spin = st[2:5].astype(np.float64), st[5:8].astype(np.float64), st[8:11].astype(np.float64)
    if nudge is not None:
        pos = pos + nudge.uniform(-5e-5, 5e-5, 3); vel = vel + nudge.uniform(-5e-5, 5e-5, 3)
    js_col = slice(24, 30)
    hold = tuple(float(v) for v in st[js_col])
    if s.place(pos.tolist(), vel.tolist(), spin.tolist(), hold=hold) is None:
        return None
    out = []
    for i in range(t + 1, min(len(rows), t + 1 + HORIZON)):
        js = tuple(float(v) for v in rows[i][js_col])
        msg, info = s.step(js)
        out.append(np.asarray(msg.obs[RAW_POS], dtype=np.float64))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--map', required=True)
    a = ap.parse_args()
    rec = Recorder3(a.port, a.map)
    rng = np.random.default_rng(12345)
    sampler = StartSampler(rec.g, rng)
    t0 = time.time()
    # (a) repeats
    reps = []
    for k in range(REPEAT_TRIALS):
        start = sampler.sample(); prog = sample_controls(rng)
        runs = []
        while len(runs) < REPEATS:
            try:
                r = rec.trial(start, prog)
            except RoundOver:
                continue
            if r is not None:
                runs.append(r)
        n = min(len(x[1]) for x in runs)
        P = np.stack([x[1][:n, 2:5] for x in runs])
        dp = float(np.abs(P - P[0]).max())
        ev = set((x[0]['took_off'], x[0]['landed']) for x in runs)
        oobs = sorted(set(x[0]['n'] for x in runs if x[0]['oob']))
        ok = dp <= REPEAT_TOL and len(ev) == 1
        reps.append({'family': prog['family'], 'kind': start['kind'], 'max_dp': dp, 'events_agree': len(ev) == 1,
                     'oob_decision_spread': (oobs[-1] - oobs[0]) if len(oobs) > 1 else 0, 'ok': bool(ok)})
        print('repeat', k, reps[-1], flush=True)
    # (b) natural replay
    nat = []
    tries = 0
    while len(nat) < NATURAL and tries < NATURAL * 5:
        tries += 1
        start = sampler.sample(); prog = sample_controls(rng)
        if prog['family'] in ('none', 'brake') or start['kind'] == 'air':
            continue
        try:
            r = rec.trial(start, dict(prog, len=REPLAY_LEN))
        except RoundOver:
            continue
        if r is None or len(r[1]) < 20:
            continue
        meta, rows = r
        jk = rows[:, 28]                                   # jump key column
        cands = [t for t in range(5, len(rows) - HORIZON) if jk[t] == 0 and jk[t + 1] == 0 and (t == 0 or jk[t - 1] == 0)]
        if not cands:
            continue
        t = int(rng.choice(cands))
        try:
            P = replay(rec, rows, t)
        except RoundOver:
            continue
        if P is None or not len(P):
            continue
        Q = rows[t + 1:t + 1 + len(P), 2:5].astype(np.float64)
        err = np.linalg.norm(P - Q, axis=1)
        try:
            P2 = replay(rec, rows, t, nudge=rng)
        except RoundOver:
            P2 = None
        m2 = min(len(P), len(P2)) if P2 is not None else 0
        err_nudge = float(np.linalg.norm(P[:m2] - P2[:m2], axis=1).max()) if m2 else None
        air = bool(meta['took_off'] is not None and (meta['landed'] is None or meta['landed'] > t) and meta['took_off'] <= t)
        nat.append({'family': prog['family'], 't': t, 'airborne_at_cut': air, 'max_err': float(err.max()),
                    'nudged_replay_max_diff': err_nudge,
                    'err_at_5': float(err[min(4, len(err) - 1)]), 'ok': bool(err.max() <= NATURAL_TOL)})
        print('natural', len(nat), nat[-1], flush=True)
    rep_ok = sum(r['ok'] for r in reps); nat_ok = sum(r['ok'] for r in nat)
    out = {'map': a.map, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'seconds': round(time.time() - t0),
           'repeats': {'n': len(reps), 'ok': rep_ok, 'pass': rep_ok == len(reps), 'tests': reps},
           'natural': {'n': len(nat), 'ok': nat_ok, 'rate': nat_ok / max(1, len(nat)), 'pass': nat_ok >= 0.95 * len(nat),
                       'max_err_p50': float(np.median([r['max_err'] for r in nat])) if nat else None,
                       'max_err_p95': float(np.percentile([r['max_err'] for r in nat], 95)) if nat else None,
                       'nudged_p95': float(np.percentile([r['nudged_replay_max_diff'] for r in nat if r['nudged_replay_max_diff'] is not None], 95)) if nat else None,
                       'nudged_over_tol': sum(1 for r in nat if (r['nudged_replay_max_diff'] or 0) > NATURAL_TOL),
                       'tests': nat},
           'rules': {'repeat_tol': REPEAT_TOL, 'natural_tol': NATURAL_TOL, 'horizon': HORIZON}}
    os.makedirs(os.path.join(DATA_DIR, a.map), exist_ok=True)
    json.dump(out, open(os.path.join(DATA_DIR, a.map, 'check3.json'), 'w'), indent=1)
    print(f'{a.map}: repeats {rep_ok}/{len(reps)}; natural replay {nat_ok}/{len(nat)} within {NATURAL_TOL} u '
          f'(max error median {out["natural"]["max_err_p50"]}, p95 {out["natural"]["max_err_p95"]}); nudged replays over the '
          f'tolerance {out["natural"]["nudged_over_tol"]}, nudged p95 {out["natural"]["nudged_p95"]}', flush=True)


if __name__ == '__main__':
    main()
