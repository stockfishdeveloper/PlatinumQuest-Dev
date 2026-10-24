"""Points ledger (docs/POINTS_PLAN_2026-10-22.md step 1): where does the round's time go, planner against human?

Every moment of a round lands in exactly one bucket:
  approach   from the last pickup of a spawn to the first pickup of the next (and before the first pickup)
  in-spawn   between the first and the last pickup of a spawn
  tail       after the last pickup of the round
each split into moving / stalled (stalled = the marble moved less than STALL_U in the last STALL_S seconds), plus
  respawn    out of bounds or respawning (journal: native mode != 0 or oob; demo: the oob flag and the ticks until the
             marble moves again)
A spawn = a run of pickups with no gap longer than SPAWN_GAP_S (travel_split.py's 47-decision rule), applied the same
way to the planner journals and the human demos, so both sides are cut with the same knife.

Inputs: planner pool journals  dynamics_reports/<pool>/<arm>/<map>/<seed>/physics_plan/<n>/decisions.jsonl.gz
        human demos            demos/demo_<stamp>.npz (16 ms ticks: obs_raw[:, 0:3] position, 3:6 velocity, gem_delta, oob)
Usage (ml_agent):
  python tools/points_ledger.py --arms dynamics_reports/handoff_safety_20261020/wallclimb1_* --demos demos/demo_20261021_151105.npz
"""
import argparse
import glob
import gzip
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SPAWN_GAP_S = 47 * .064
STALL_S = 1.5
STALL_U = 1.0
ENDGAME_S = 10.0
ML = Path(__file__).resolve().parents[1]


def timeline_from_journal(path):
    """(dt, positions, speeds, pickup, respawn, labels, blasts, round_points, name, seed) from a pool journal."""
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8') as h:
        for line in h:
            rows.append(json.loads(line))
    pos = np.array([r['before']['native']['state']['position'] for r in rows], dtype=float)
    vel = np.array([r['before']['native']['state']['velocity'] for r in rows], dtype=float)
    pickup = np.array([float((r.get('outcome') or {}).get('gem_delta') or 0) for r in rows])
    respawn = np.array([bool(r['before']['native']['state'].get('oob')) or int(r['before']['native']['state'].get('mode', 0)) != 0 for r in rows])
    labels = [str((r.get('choice') or {}).get('label') or '') for r in rows]
    acts = [r.get('action') or [] for r in rows]
    blasts = sum(1 for a, b in zip(acts, acts[1:]) if len(b) > 8 and int(b[8]) and not (len(a) > 8 and int(a[8])))
    parts = Path(path).resolve().parts
    return dict(dt=.064, pos=pos, speed=np.hypot(vel[:, 0], vel[:, 1]), pickup=pickup, respawn=respawn, labels=labels,
                blasts=blasts, points=float(pickup.sum()), name=parts[-5], seed=parts[-4], arm=parts[-6])


def timeline_from_demo(path):
    d = np.load(path, allow_pickle=True)
    o = d['obs_raw']; pos = o[:, 0:3].astype(float); vel = o[:, 3:6].astype(float)
    pickup = d['gem_delta'].astype(float); oob = d['oob'].astype(bool)
    # respawn: the oob tick and every tick until the marble moves again at least 0.3 u/s
    respawn = np.zeros(len(pos), dtype=bool); on = False
    sp = np.hypot(vel[:, 0], vel[:, 1])
    for i in range(len(pos)):
        if oob[i]:
            on = True
        if on and sp[i] > .3 and not oob[i]:
            on = False
        respawn[i] = on
    name = os.path.basename(str(d['terrain_map'])).replace('terrain_', '').replace('.npz', '')
    # rounds: the clock is not in the demo; a jump of more than 25 u in one tick that is not an oob respawn starts a
    # new round (the marble is placed on the spawn point)
    # rounds: obs_raw[:, 31] is the clock (ms left, counting down); a new round is where it jumps back up
    clock = o[:, 31].astype(float) if o.shape[1] > 31 else None
    if clock is not None and np.ptp(clock) > 1000:
        cuts = [0] + [i + 1 for i in np.nonzero(np.diff(clock) > 5000.)[0]] + [len(pos)]
    else:
        jumps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        cuts = [0] + [i + 1 for i in np.nonzero(jumps > 25.)[0] if not oob[max(0, i - 30):i + 2].any()] + [len(pos)]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a < 60 * 60:
            continue                                   # under a minute: a fragment
        out.append(dict(dt=.016, pos=pos[a:b], speed=sp[a:b], pickup=pickup[a:b], respawn=respawn[a:b], labels=None,
                        blasts=None, points=float(pickup[a:b].sum()), name=name, seed='human', arm=os.path.basename(path)))
    return out


def ledger(tl):
    dt = tl['dt']; n = len(tl['pos']); pk = np.nonzero(tl['pickup'] > 0)[0]
    gap = int(round(SPAWN_GAP_S / dt)); win = int(round(STALL_S / dt))
    spawns = []
    for i in pk:
        if spawns and i - spawns[-1][-1] <= gap:
            spawns[-1].append(i)
        else:
            spawns.append([i])
    # moving / stalled per step
    disp = np.full(n, np.inf)
    if n > win:
        disp[win:] = np.linalg.norm(tl['pos'][win:] - tl['pos'][:-win], axis=1)
    stalled = disp < STALL_U
    seg = np.zeros(n, dtype=int)                       # 0 approach, 1 in-spawn, 2 tail
    for s in spawns:
        seg[s[0]:s[-1] + 1] = 1
    if spawns:
        seg[spawns[-1][-1] + 1:] = 2
    resp = tl['respawn']
    b = defaultdict(float)
    for i in range(n):
        if resp[i]:
            b['respawn'] += dt
        else:
            key = ('approach', 'in-spawn', 'tail')[seg[i]] + ('/stalled' if stalled[i] else '/moving')
            b[key] += dt
    b['total'] = n * dt
    # stall sites: runs of stalled steps outside respawns, with where they were
    sites = []
    i = 0
    while i < n:
        if stalled[i] and not resp[i]:
            j = i
            while j < n and stalled[j] and not resp[j]:
                j += 1
            sites.append((i, j, (j - i) * dt))
            i = j
        else:
            i += 1
    pickup_speed = float(np.mean(tl['speed'][pk])) if len(pk) else float('nan')
    exit_speed = float(np.mean([tl['speed'][s[-1]] for s in spawns])) if spawns else float('nan')
    last = int(round(ENDGAME_S / dt))
    return dict(buckets=dict(b), spawns=len(spawns), pickups=len(pk), points=tl['points'], pickup_speed=pickup_speed,
                exit_speed=exit_speed, stall_sites=sites, endgame_points=float(tl['pickup'][-last:].sum()),
                approach_per_spawn=(b['approach/moving'] + b['approach/stalled']) / max(len(spawns), 1),
                inspawn_per_gem=(b['in-spawn/moving'] + b['in-spawn/stalled']) / max(len(pk) - len(spawns), 1),
                blasts=tl['blasts'])


KEYS = ('approach/moving', 'approach/stalled', 'in-spawn/moving', 'in-spawn/stalled', 'respawn', 'tail/moving', 'tail/stalled')


def fmt_row(label, L, n=1):
    b = L['buckets']
    return '%-34s %5.0f pts %3d spawns %3d gems | appr mov %5.1f stl %5.1f | spawn mov %5.1f stl %5.1f | resp %4.1f | tail %4.1f | %4.2f s/approach %4.2f s/gem | pick %4.1f exit %4.1f u/s | blasts %s | last10s %3.0f' % (
        label, L['points'] / n, L['spawns'] / n, L['pickups'] / n, b.get('approach/moving', 0) / n, b.get('approach/stalled', 0) / n,
        b.get('in-spawn/moving', 0) / n, b.get('in-spawn/stalled', 0) / n, b.get('respawn', 0) / n,
        (b.get('tail/moving', 0) + b.get('tail/stalled', 0)) / n, L['approach_per_spawn'], L['inspawn_per_gem'],
        L['pickup_speed'], L['exit_speed'], 'n/a' if L['blasts'] is None else '%.1f' % (L['blasts'] / n), L['endgame_points'] / n)


def pool_mean(Ls):
    n = len(Ls); out = dict(buckets=defaultdict(float))
    for L in Ls:
        for k, v in L['buckets'].items():
            out['buckets'][k] += v
    for k in ('spawns', 'pickups', 'points', 'endgame_points'):
        out[k] = sum(L[k] for L in Ls)
    out['blasts'] = None if any(L['blasts'] is None for L in Ls) else sum(L['blasts'] for L in Ls)
    for k in ('pickup_speed', 'exit_speed', 'approach_per_spawn', 'inspawn_per_gem'):
        out[k] = float(np.nanmean([L[k] for L in Ls]))
    return out, n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arms', nargs='+', required=True, help='pool arm directories (globs ok)')
    p.add_argument('--demos', nargs='*', default=[])
    p.add_argument('--rounds', action='store_true', help='also print every round')
    p.add_argument('--stalls', type=int, default=0, help='print the N longest stall sites per map')
    a = p.parse_args()
    by_map = defaultdict(list); rounds = []
    for pat in a.arms:
        for arm in sorted(glob.glob(pat)):
            for f in sorted(glob.glob(os.path.join(arm, '*', '*', 'physics_plan', '*', 'decisions.jsonl.gz'))):
                tl = timeline_from_journal(f); L = ledger(tl); L['tl'] = tl
                by_map[tl['name']].append(L); rounds.append((tl, L))
                if a.rounds:
                    print(fmt_row('%s %s %s' % (tl['arm'][:14], tl['name'][:12], tl['seed']), L))
    humans = defaultdict(list)
    for f in a.demos:
        for tl in timeline_from_demo(f):
            L = ledger(tl); L['tl'] = tl; humans[tl['name']].append(L)
            print(fmt_row('HUMAN %s %s' % (tl['name'][:14], tl['arm'][5:20]), L))
    print()
    print('per map, mean per round (planner) and the human demo rounds; seconds per bucket')
    allL = []
    for name in sorted(by_map):
        Ls = by_map[name]; allL += Ls; M, n = pool_mean(Ls)
        print(fmt_row('%-16s planner x%d' % (name[:16], n), M, n))
        if name in humans:
            H, hn = pool_mean(humans[name]); print(fmt_row('%-16s HUMAN   x%d' % (name[:16], hn), H, hn))
            print('   gap %+.0f pts: approach %+.1f s, in-spawn %+.1f s, stalled %+.1f s, respawn %+.1f s per round; s/approach %+.2f, s/gem %+.2f' % (
                H['points'] / hn - M['points'] / n,
                (M['buckets']['approach/moving'] + M['buckets']['approach/stalled']) / n - (H['buckets']['approach/moving'] + H['buckets']['approach/stalled']) / hn,
                (M['buckets']['in-spawn/moving'] + M['buckets']['in-spawn/stalled']) / n - (H['buckets']['in-spawn/moving'] + H['buckets']['in-spawn/stalled']) / hn,
                (M['buckets']['approach/stalled'] + M['buckets']['in-spawn/stalled']) / n - (H['buckets']['approach/stalled'] + H['buckets']['in-spawn/stalled']) / hn,
                M['buckets']['respawn'] / n - H['buckets']['respawn'] / hn,
                M['approach_per_spawn'] - H['approach_per_spawn'], M['inspawn_per_gem'] - H['inspawn_per_gem']))
        if a.stalls:
            sites = sorted(((s[2], s[0], L) for L in Ls for s in L['stall_sites']), key=lambda x: -x[0])[:a.stalls]
            for dur, i, L in sites:
                tl = L['tl']; q = tl['pos'][i]
                print('   stall %4.1f s at decision %4d pos %s label %s (%s %s)' % (dur, i, np.round(q, 1).tolist(), tl['labels'][i] if tl['labels'] else '', tl['arm'][:14], tl['seed']))
    M, n = pool_mean(allL)
    print(fmt_row('ALL MAPS planner x%d' % n, M, n))


if __name__ == '__main__':
    main()
