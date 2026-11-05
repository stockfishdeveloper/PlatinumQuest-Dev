"""Spawn clear times (2026-11-04, operator: "it takes probably twice as long on average to collect spawns as it should").

One definition for every source: pickups grouped by position gaps (a pickup farther than GAP_U from the previous one starts a new
spawn group); travel = the previous group's last pickup -> this group's first; clear = first -> last pickup of the group;
"s a spawn" = travel + clear; "gems/s inside" = (taken - 1) / clear.
Sources: planner decision journals (dynamics_reports/.../decisions.jsonl.gz, 64 ms decisions), sim rounds (tools/sim_bench/full_round.py
--out, picks with positions) and human demos (demos/*.npz, 16 ms ticks).
Reference 11-04: human TreasureBox demo_20261021_151105: travel 5.3 s, clear 4.9 s, 9.2 s a spawn, 1.46 gems/s inside;
production planner (edge_pops1 baseline, 20 native rounds) TreasureBox: travel 7.3 s, clear 7.9 s, 15.0 s a spawn, 1.11 gems/s.
usage (repo root): python ml_agent/tools/spawn_times.py journals "<glob of decisions.jsonl.gz>"
                   python ml_agent/tools/spawn_times.py sim "<glob of rounds jsonl>" [label]
                   python ml_agent/tools/spawn_times.py demo ml_agent/demos/<demo>.npz [tick_s]
"""
import gzip, json, glob, os, sys, collections
import numpy as np

GAP_U = 8.


def group_picks(picks):
    """picks: list of (t_s, xyz). Groups by position gaps; returns dicts(taken, travel_s = previous group's last pickup -> first,
    clear_s = first -> last pickup)."""
    out = []; cur = []; prev_last = None
    for t, p in picks:
        if cur and float(np.linalg.norm(np.asarray(p) - np.asarray(cur[-1][1]))) > GAP_U:
            out.append(cur); cur = []
        cur.append((t, p))
    if cur: out.append(cur)
    res = []
    for g in out:
        res.append(dict(taken=len(g), travel_s=(g[0][0] - prev_last) if prev_last is not None else None, clear_s=g[-1][0] - g[0][0]))
        prev_last = g[-1][0]
    return res


def summarize(name, spawns):
    full = [s for s in spawns if s['taken'] >= 2]
    if not spawns:
        print('%-28s no spawns' % name); return
    clear = [s['clear_s'] for s in full]; trav = [s['travel_s'] for s in spawns if s['travel_s'] is not None]
    rate = [(s['taken'] - 1) / s['clear_s'] for s in full if s['clear_s'] > 0]
    per = [s['clear_s'] + s['travel_s'] for s in spawns if s['travel_s'] is not None]
    print('%-28s spawns %3d taken %.1f | travel %.1f s | clear %.1f s (p50 %.1f) | %.1f s a spawn | %.2f gems/s inside' % (
        name, len(spawns), np.mean([s['taken'] for s in spawns]), np.mean(trav) if trav else 0., np.mean(clear) if clear else 0.,
        np.median(clear) if clear else 0., np.mean(per) if per else 0., np.mean(rate) if rate else 0.))
    return spawns


def from_journal(path, dt=.064):
    decs = [json.loads(line) for line in gzip.open(path, 'rt')]
    prev = {}; picks = []
    for i, d in enumerate(decs):
        gems = {it['id']: it for it in d['before'].get('items', []) if it.get('gem')}
        for g in prev:
            if g not in gems: picks.append((i * dt, prev[g]['position']))
        prev = gems
    return group_picks(picks)


def from_demo(path, tick_s=None):
    z = np.load(path, allow_pickle=True); raw = z['obs_raw']; gd = z['gem_delta']
    meta = json.load(open(path.replace('.npz', '.json')))
    if tick_s is None: tick_s = float(meta['seconds_of_play']) / float(meta['ticks'])
    picks = [(i * tick_s, raw[i, :3].tolist()) for i in range(len(raw)) if gd[i] > 0]
    g = group_picks(picks)
    print('demo tick %.4f s, pickups %d, spawn sizes %s' % (tick_s, len(picks), [s['taken'] for s in g]))
    return g


def from_sim(path, label=None):
    """Sim rounds (tools/sim_bench/full_round.py --out): picks with t_ms and position."""
    out = []
    for line in open(path, encoding='utf-8'):
        try: r = json.loads(line)
        except Exception: continue
        if label and r.get('label') != label: continue
        picks = [(p['t_ms'] / 1000., p['position']) for p in r['picks'] if 'position' in p]
        out.extend(group_picks(picks))
    return out


if __name__ == '__main__':
    kind = sys.argv[1]
    if kind == 'sim':
        by_map = collections.defaultdict(list)
        for f in sorted(glob.glob(sys.argv[2])):
            by_map[os.path.basename(f)].extend(from_sim(f, sys.argv[3] if len(sys.argv) > 3 else None))
        for m, sp in sorted(by_map.items()): summarize(m, sp)
        summarize('ALL', [s for sp in by_map.values() for s in sp]); sys.exit()
    if kind == 'journals':
        by_map = collections.defaultdict(list)
        for f in sorted(glob.glob(sys.argv[2])):
            m = [p for p in os.path.normpath(f).split(os.sep) if p.endswith('_Hunt')]
            by_map[m[0] if m else f].extend(from_journal(f))
        for m, sp in sorted(by_map.items()): summarize(m, sp)
        allsp = [s for sp in by_map.values() for s in sp]; summarize('ALL', allsp)
    else:
        summarize(os.path.basename(sys.argv[2]), from_demo(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None))
