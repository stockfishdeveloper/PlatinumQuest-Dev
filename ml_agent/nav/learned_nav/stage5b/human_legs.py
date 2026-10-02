"""Human vs navigator per gem-to-gem leg on KOTM: time, and whether the human crossed a hole."""
import json, sys, math, glob
import numpy as np
sys.path.insert(0, 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent')
from nav.gems import visible_gems
from nav.learned_nav.geometry import Geometry
g = Geometry('KingOfTheMarble_Hunt')
ML = 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent/'

def key(q):
    return (round(q[0] * 2) / 2, round(q[1] * 2) / 2)

def over_void(path):
    n = 0
    for x, y, z in path:
        f = g.floor_below(x, y, z)
        if f is None or f < z - 0.5 - 3.0:
            n += 1
    return n

human = {}
for f in sorted(glob.glob(ML + 'demos/demo_*.npz')):
    z = np.load(f, allow_pickle=True)
    o = z['obs_raw']; gd = z['gem_delta']; game = z['game']; oob = z['oob']
    info = json.load(open(f.replace('.npz', '.json')))
    tots = info.get('game_totals_from_game', [])
    for gi in np.unique(game):
        idx = np.nonzero(game == gi)[0]
        tot = tots[int(gi)] if int(gi) < len(tots) else None
        picks = [t for t in idx if gd[t] > 0]
        prev = None
        for t in picks:
            vis = visible_gems(o[t - 1])
            p = o[t, 0:3]
            if not vis:
                prev = None; continue
            q = min(vis, key=lambda q: math.hypot(q[0] - p[0], q[1] - p[1]))
            cur = (key(q), t)
            if prev is not None and not oob[prev[1]:t].any():
                path = o[prev[1]:t:4, 0:3]
                human.setdefault((prev[0], cur[0]), []).append({'t': (t - prev[1]) * 0.016, 'void': over_void(path),
                                                               'total': tot, 'file': f[-19:-4]})
            prev = cur
print('human legs', sum(len(v) for v in human.values()), 'pairs', len(human))

nav = {}
for f in glob.glob(ML + 'logs/learned_nav/rounds/kotm_navlegs_KingOfTheMarble_Hunt.jsonl') + glob.glob(ML + 'logs/learned_nav/rounds/kotm_control_KingOfTheMarble_Hunt.jsonl'):
    for line in open(f):
        r = json.loads(line)
        L = r.get('legs', [])
        for a, b in zip(L, L[1:]):
            if b['i0'] - a['pick_i'] <= 1:
                nav.setdefault((key(a['gem']), key(b['gem'])), []).append({'t': (b['pick_i'] - a['pick_i']) * 0.064})
print('nav legs', sum(len(v) for v in nav.values()), 'pairs', len(nav))

# human by round quality
strong = lambda e: e['total'] is not None and e['total'] >= 160
rows = []
for k in set(human) & set(nav):
    hs = [e for e in human[k] if strong(e)]
    if not hs:
        continue
    ht = np.mean([e['t'] for e in hs]); nt = np.mean([e['t'] for e in nav[k]])
    vf = np.mean([e['void'] > 0 for e in hs])
    rows.append((nt - ht, k, len(hs), len(nav[k]), ht, nt, vf))
rows.sort(key=lambda r: -r[0] * r[3])
print('pairs in both (strong human rounds):', len(rows))
tot_gain = sum(r[0] * r[3] for r in rows)
print('sum over nav legs of (nav - human) time: %.1f s over %d nav legs' % (tot_gain, sum(r[3] for r in rows)))
vg = sum(r[0] * r[3] for r in rows if r[6] >= 0.5); print('  of which pairs where the human crossed void >= half the time: %.1f s' % vg)
print('%-30s %4s %4s %6s %6s %6s %5s' % ('pair', 'nh', 'nn', 'human', 'nav', 'diff', 'void'))
for r in rows[:30]:
    print('%-30s %4d %4d %6.2f %6.2f %6.2f %5.2f' % (str(r[1]), r[2], r[3], r[4], r[5], r[0], r[6]))

print()
H = [e | {'pair': k} for k, v in human.items() for e in v if strong(e)]
N = [e | {'pair': k} for k, v in nav.items() for e in v]
ht = np.array([e['t'] for e in H]); nt = np.array([e['t'] for e in N])
hv = np.array([e['void'] > 0 for e in H])
print('human strong legs %d mean %.3f s; void legs %d (%.0f%%) mean %.3f s; non-void mean %.3f' % (len(H), ht.mean(), hv.sum(), 100 * hv.mean(), ht[hv].mean(), ht[~hv].mean()))
print('nav legs %d mean %.3f s' % (len(N), nt.mean()))
# what would the nav's time be on the human's pairs (pairs the nav has): human pair mix at nav speed
hp = [e for e in H if e['pair'] in nav]
nav_on_h = np.mean([np.mean([x['t'] for x in nav[e['pair']]]) for e in hp])
print('human legs whose pair the nav also ran: %d of %d; human mean %.3f, nav on those pairs %.3f' % (len(hp), len(H), np.mean([e['t'] for e in hp]), nav_on_h))
hv_p = [e for e in hp if e['void'] > 0]
print('  of those void: %d, human %.3f, nav on those pairs %.3f' % (len(hv_p), np.mean([e['t'] for e in hv_p]), np.mean([np.mean([x['t'] for x in nav[e['pair']]]) for e in hv_p])))
# void legs of the human by pair, with the nav time when known
from collections import defaultdict
vp = defaultdict(list)
for e in H:
    if e['void'] > 0:
        vp[e['pair']].append(e['t'])
print('human void pairs:')
for k, v in sorted(vp.items(), key=lambda kv: -len(kv[1])):
    n = nav.get(k)
    print('  %-30s n %2d  human %.2f  nav %s (n %d)  human share of this pair void %.2f' % (str(k), len(v), np.mean(v), ('%.2f' % np.mean([x['t'] for x in n])) if n else '  - ', len(n) if n else 0,
          len(v) / len([e for e in H if e['pair'] == k])))
