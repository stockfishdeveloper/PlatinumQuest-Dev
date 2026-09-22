"""Speed as a function of distance-to-gem, human vs agent, over gem-to-gem legs.

The question this answers: WHERE in a gem approach does the agent lose time? Mean speed hides it.
Plotting speed against remaining distance to the gem separates "slow everywhere" from "fast on the
straight and then brakes far too early", which need completely different fixes.

A LEG is one gem pickup to the next. For every decision inside a leg we know how far the marble
still has to travel to the gem that ends it, so each leg contributes a curve, and we take the
median across legs in each distance bin (median, not mean, because a few stalled legs would drag a
mean down and misrepresent the typical approach).

Agent data: logs/nav/real_trace.csv, which carries `tdist` (distance to the current target) and
`gem` (pickup marker) per decision at 64 ms.
Human data: demos/demo_<stamp>.npz, 16 ms ticks, obs_raw = [pos(3), vel(3), gems(25), ...] and
gem_delta marking pickups. Distance to the gem is computed as the remaining PATH length along the
human's own trajectory to the pickup point, which is the like-for-like quantity: it is what the
agent's `tdist` measures once the line is clear, and it avoids assuming which gem slot they chose.

    python plot_approach_profile.py [--demo demos/demo_xxx.npz] [--out logs/nav/approach.png]
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BINS = np.arange(0.0, 24.1, 1.0)          # distance-to-gem bins, units


def agent_legs(path):
    """(distance_to_gem, speed) pairs for every decision inside a pickup-to-pickup leg."""
    rows = list(csv.DictReader(open(path)))
    picks = [k for k, r in enumerate(rows) if float(r['gem']) > 0]
    d, v = [], []
    legs = 0
    prev = -1
    for b_ in picks:
        # Walk BACKWARDS from the pickup for as long as a gem was actually on the map and the
        # target did not change. On a sparse map most legs begin with a wait for the next group to
        # spawn, during which `tdist` points at a spawn centroid rather than a gem, so taking the
        # whole leg would mix "approaching a gem" with "loitering". The approach is the part that
        # matters and it is the part that is comparable with the human.
        tx, ty = rows[b_]['tx'], rows[b_]['ty']
        k = b_
        seg = []
        while k > prev + 1:
            r = rows[k]
            if int(r['nvis']) == 0 or r['tx'] != tx or r['ty'] != ty:
                break
            seg.append((float(r['tdist']), float(r['speed'])))
            k -= 1
        prev = b_
        if len(seg) < 4:
            continue
        legs += 1
        for dd, vv in seg:
            d.append(dd); v.append(vv)
    return np.array(d), np.array(v), legs


def human_legs(path):
    """Same, from a recorded demo. Distance is remaining path length to the pickup."""
    z = np.load(path)
    raw = z['obs_raw']
    gd = z['gem_delta']
    pos = raw[:, 0:2]
    spd = np.hypot(raw[:, 3], raw[:, 4])
    picks = np.where(gd > 0)[0]
    d, v = [], []
    legs = 0
    for a, b in zip(picks[:-1], picks[1:]):
        if b - a < 6:
            continue
        step = np.hypot(np.diff(pos[a:b + 1, 0]), np.diff(pos[a:b + 1, 1]))
        step = np.clip(step, 0, 1.5)                      # drop teleport/respawn jumps
        remaining = np.concatenate([step[::-1].cumsum()[::-1], [0.0]])
        legs += 1
        d.extend(remaining.tolist())
        v.extend(spd[a:b + 1].tolist())
    return np.array(d), np.array(v), legs


def profile(d, v, bins=BINS):
    """Median speed per distance bin, plus the interquartile band."""
    mid, med, lo, hi, n = [], [], [], [], []
    for i in range(len(bins) - 1):
        m = (d >= bins[i]) & (d < bins[i + 1])
        if m.sum() < 20:
            continue
        s = v[m]
        mid.append(0.5 * (bins[i] + bins[i + 1]))
        med.append(np.median(s)); lo.append(np.percentile(s, 25)); hi.append(np.percentile(s, 75))
        n.append(int(m.sum()))
    return np.array(mid), np.array(med), np.array(lo), np.array(hi), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', default=None)
    ap.add_argument('--trace', default=os.path.join(HERE, 'logs', 'nav', 'real_trace.csv'))
    ap.add_argument('--out', default=os.path.join(HERE, 'logs', 'nav', 'approach_profile.png'))
    a = ap.parse_args()

    demo = a.demo
    if demo is None:
        import glob
        cand = sorted(glob.glob(os.path.join(HERE, 'demos', 'demo_*.npz')))
        if not cand:
            sys.exit('no demo found')
        demo = cand[-1]

    hd, hv, hlegs = human_legs(demo)
    ad, av, alegs = agent_legs(a.trace)
    print('human %s: %d legs, %d ticks' % (os.path.basename(demo), hlegs, len(hd)))
    print('agent %s: %d legs, %d decisions' % (os.path.basename(a.trace), alegs, len(ad)))

    hm, hmed, hlo, hhi, hn = profile(hd, hv)
    am, amed, alo, ahi, an = profile(ad, av)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax.fill_between(hm, hlo, hhi, color='#2a7fbf', alpha=0.15)
    ax.plot(hm, hmed, color='#2a7fbf', lw=2.5, marker='o', ms=4, label='human (you)')
    ax.fill_between(am, alo, ahi, color='#d1495b', alpha=0.15)
    ax.plot(am, amed, color='#d1495b', lw=2.5, marker='s', ms=4, label='agent')
    # Travel should read left to right, so far-from-gem on the left and the gem at the right edge.
    # Invert ONCE: ax2 is created with sharex=True, so inverting it as well flips the axis back and
    # silently contradicts the subtitle below (that bug shipped in the 2026-09-21 graphs, which
    # rendered with the gem at x=0 on the LEFT).
    ax.invert_xaxis()
    ax.set_ylabel('speed (u/s)')
    ax.set_title('Speed through a gem approach, FlatGemTraining\n'
                 'each point = median across legs; band = interquartile range; x runs from far (left) to the gem (right)')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left')
    ax.axvline(0, color='k', lw=1, ls=':')

    # the gap, so the crossover point is unmistakable
    common = sorted(set(np.round(hm, 2)) & set(np.round(am, 2)))
    hmap = dict(zip(np.round(hm, 2), hmed)); amap = dict(zip(np.round(am, 2), amed))
    gx = np.array(common); gy = np.array([hmap[c] - amap[c] for c in common])
    ax2.bar(gx, gy, width=0.8, color=['#d1495b' if g > 0 else '#3a9c5c' for g in gy])
    ax2.axhline(0, color='k', lw=1)
    # NO invert here: sharex=True means ax's inversion already applies (see the note above).
    ax2.set_xlabel('distance still to travel to the gem (u)')
    ax2.set_ylabel('human minus agent (u/s)')
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print('wrote', a.out)

    print()
    print('  %-10s %10s %10s %10s' % ('dist to gem', 'human', 'agent', 'gap'))
    for c in sorted(common, reverse=True):
        print('  %7.1f u %10.2f %10.2f %+10.2f' % (c, hmap[c], amap[c], hmap[c] - amap[c]))


if __name__ == '__main__':
    main()
