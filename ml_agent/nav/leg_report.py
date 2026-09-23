"""Leg-by-leg comparison of a real-round trace against the human demo (HANDOFF section 28).

Prints the readouts that measure hesitation directly, for the agent trace and the human demo:
  * pickup speed at centre (edge-cell) gems vs outer gems
  * after each pickup: min speed in the next 1 s, share of pickups followed by a dip below 2 u/s,
    time to regain 8 u/s, turn demanded at the pickup (velocity 0.5 s before vs bearing to the next gem)
  * leg durations by type (ring->ring, ring->out, out->ring, out->out) and path/straight ratio
  * thrust-opposes-velocity share (the human's "derived brake" measured the same way on both)
  * mean speed and time share inside the centre block, falls per 100 u

    python -m nav.leg_report                                   # logs/nav/real_trace.csv vs the KOTM demo
    python -m nav.leg_report logs/nav/real_trace_x.csv         # another trace
    NAV_LEG_CENTRE="-25.2,15.0" NAV_LEG_RING_R=5 ...           # centre point / ring radius (KOTM defaults)

The "ring" is everything within NAV_LEG_RING_R of NAV_LEG_CENTRE; the "block" is the square of half-side
3.5 u around it. Both are KOTM geometry; on another map set the centre to its densest gem cluster or
ignore the ring rows and read the per-pickup and thrust rows, which are map-independent.
"""
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(HERE, 'demos', 'demo_20260914_214854.npz')
CX, CY = (float(v) for v in os.environ.get('NAV_LEG_CENTRE', '-25.2,15.0').split(','))
RING_R = float(os.environ.get('NAV_LEG_RING_R', '5'))


def load_agent(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda k: np.array([float(r[k]) for r in rows])
    cmd = np.c_[f('right') - f('left'), f('fwd') - f('back')]
    return dict(P=np.c_[f('x'), f('y')], S=f('speed'), gem=f('gem') > 0, fell=f('fell') > 0,
                floor=f('on_floor') > 0, V=np.c_[f('vx'), f('vy')], dt=0.064, rnd=f('round'), cmd=cmd)


def load_human(path=DEMO):
    d = np.load(path, allow_pickle=True); raw = d['obs_raw']; act = d['action']
    return dict(P=raw[:, :2].astype(float), S=np.hypot(raw[:, 3], raw[:, 4]), gem=d['gem_delta'] > 0,
                fell=d['oob'] > 0, floor=np.abs(raw[:, 5]) < 1.0, V=raw[:, 3:5].astype(float), dt=0.016,
                rnd=d['game'], cmd=np.where(act[:, 2:3] > 0.5, act[:, 0:2], 0.0))


def in_ring(p):
    return math.hypot(p[0] - CX, p[1] - CY) < RING_R


def legs(D):
    idx = np.where(D['gem'])[0]; out = {k: [] for k in ('ring->ring', 'ring->out', 'out->ring', 'out->out')}
    for a, b in zip(idx[:-1], idx[1:]):
        if D['rnd'][a] != D['rnd'][b]:
            continue
        k = ('ring' if in_ring(D['P'][a]) else 'out') + '->' + ('ring' if in_ring(D['P'][b]) else 'out')
        out[k].append((a, b))
    return out


def report(name, D):
    P, S, V, dt = D['P'], D['S'], D['V'], D['dt']
    n1, n05 = int(round(1.0 / dt)), int(round(0.5 / dt))
    ring = np.array([in_ring(p) for p in P]); gem = D['gem']
    blk = (np.abs(P[:, 0] - CX) < 3.5) & (np.abs(P[:, 1] - CY) < 3.5)
    travelled = np.sum(np.minimum(np.linalg.norm(np.diff(P, axis=0), axis=1), 1.5))
    idx = np.where(gem)[0]
    gaps = np.array([(b - a) * dt for a, b in zip(idx[:-1], idx[1:]) if D['rnd'][a] == D['rnd'][b]])
    print(f'== {name}: {int(gem.sum())} pickups, {int(D["fell"].sum())} falls ({100 * D["fell"].sum() / travelled:.2f} per 100 u), mean speed {S.mean():.2f}')
    print(f'   SECONDS BETWEEN PICKUPS: mean {gaps.mean():.2f} s, median {np.median(gaps):.2f} s (n={len(gaps)})')
    print(f'   pickup speed: ring {S[gem & ring].mean():.2f} (n={int((gem & ring).sum())})  outer {S[gem & ~ring].mean():.2f} (n={int((gem & ~ring).sum())})')
    print(f'   block: {100 * blk.mean():.1f}% of time, mean speed {S[blk].mean():.2f}, below 2 u/s {100 * (S[blk] < 2).mean():.1f}%, above 10 u/s {100 * (S[blk] > 10).mean():.1f}%')
    cmd = D['cmd']; mag = np.hypot(cmd[:, 0], cmd[:, 1])
    m = (S > 3) & D['floor'] & (mag > 0.1)
    cos_a = (cmd[m, 0] * V[m, 0] + cmd[m, 1] * V[m, 1]) / (mag[m] * S[m])
    print(f'   thrust vs velocity while moving >3 u/s: opposes (cos<0) {100 * (cos_a < 0).mean():.1f}%, strongly (cos<-0.5) {100 * (cos_a < -0.5).mean():.1f}%, median angle {math.degrees(math.acos(float(np.clip(np.median(cos_a), -1, 1)))):.0f} deg')
    L = legs(D)
    for k, lst in L.items():
        if not lst:
            continue
        dur = np.array([(b - a) * dt for a, b in lst])
        straight = np.array([np.linalg.norm(P[b] - P[a]) for a, b in lst])
        path = np.array([np.sum(np.linalg.norm(np.diff(P[a:b + 1], axis=0), axis=1)) for a, b in lst])
        dip = np.array([S[a:min(b + 1, a + n1)].min() for a, b in lst])
        t8 = np.array([(np.where(S[a:b + 1] >= 8.0)[0][0] * dt) if (S[a:b + 1] >= 8.0).any() else np.nan for a, b in lst])
        turn = []
        for a, b in lst:
            v = V[max(0, a - n05)]; g = P[b] - P[a]
            if np.linalg.norm(v) > 0.5 and np.linalg.norm(g) > 0.5:
                turn.append(math.degrees(math.acos(float(np.clip(v @ g / (np.linalg.norm(v) * np.linalg.norm(g)), -1, 1)))))
        print(f'   {k:10s} n={len(lst):3d} dur mean {dur.mean():.2f} s | path/straight {np.median(path / np.maximum(straight, 0.1)):.2f} | '
              f'min speed in 1 s after pickup med {np.median(dip):.1f}, below 2 u/s {100 * (dip < 2).mean():.0f}% | '
              f'time to 8 u/s med {np.nanmedian(t8) if np.isfinite(t8).any() else float("nan"):.2f} s | turn at pickup med {np.median(turn) if turn else float("nan"):.0f} deg')


def main():
    trace = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'logs', 'nav', 'real_trace.csv')
    report('human ' + os.path.basename(DEMO), load_human())
    report('agent ' + os.path.basename(trace), load_agent(trace))


if __name__ == '__main__':
    main()
