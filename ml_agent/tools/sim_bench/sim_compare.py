"""Paired comparison of sim_bench variants. usage: python sim_compare.py <dir> [base variant] [other variants...]
Score per episode: time to the K-th pickup; an episode that fell or hit the cap counts cap + 5 s (fell) or cap (cap);
a sim failure/error drops the pair."""
import json, glob, sys, os, math
from collections import defaultdict
d = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else 'base'
rows = defaultdict(dict); caps = {}
for f in glob.glob(os.path.join(d, '*.jsonl')):
    for line in open(f, encoding='utf-8'):
        try: r = json.loads(line)
        except Exception: continue
        rows[(r['map'], r['pool'], r['decision'])][r['variant']] = r
variants = sys.argv[3:] or sorted({v for x in rows.values() for v in x if v != base})
def score(r, cap=25.):
    if r['why'].startswith('sim'): return None
    if r['why'] == 'fell': return cap + 5.
    if r['why'] == 'done': return r['t_s']
    return max(r['t_s'], cap) if r['why'] == 'cap' else r['t_s'] + 5.
for v in variants:
    per = defaultdict(lambda: [0, 0., 0., 0, 0, 0, 0])
    for key, x in rows.items():
        if base not in x or v not in x: continue
        a, b = score(x[base]), score(x[v])
        if a is None or b is None: continue
        p = per[key[0]]; p[0] += 1; p[1] += a; p[2] += b; p[3] += b < a - .3; p[4] += b > a + .3
        p[5] += x[base]['why'] == 'fell'; p[6] += x[v]['why'] == 'fell'
    print('== %s vs %s (time to the K-th pickup, s; fell = cap + 5)' % (v, base))
    T = [0, 0., 0., 0, 0, 0, 0]
    for m in sorted(per):
        p = per[m]; T = [t + q for t, q in zip(T, p)]
        print('  %-22s n %2d  %s %6.1f  %s %6.1f  diff %+5.1f (%+4.0f%%)  better %2d worse %2d  falls %d -> %d' % (m[:22], p[0], base, p[1] / p[0], v, p[2] / p[0], (p[2] - p[1]) / p[0], 100 * (p[2] - p[1]) / max(1e-9, p[1]), p[3], p[4], p[5], p[6]))
    if T[0]:
        print('  %-22s n %2d  %s %6.1f  %s %6.1f  diff %+5.1f (%+4.0f%%)  better %2d worse %2d  falls %d -> %d' % ('ALL', T[0], base, T[1] / T[0], v, T[2] / T[0], (T[2] - T[1]) / T[0], 100 * (T[2] - T[1]) / max(1e-9, T[1]), T[3], T[4], T[5], T[6]))
        # robust view: episodes that did not hit the cap / fall in either variant; per-episode ratio
        import statistics as st
        ds = []; rs = []
        for key, x in rows.items():
            if base in x and v in x and x[base]['why'] == 'done' and x[v]['why'] == 'done':
                ds.append(x[v]['t_s'] - x[base]['t_s']); rs.append(x[v]['t_s'] / max(.1, x[base]['t_s']))
        if ds:
            print('  both done: n %d  mean diff %+.2f s  median diff %+.2f s  geo-mean ratio %.3f' % (len(ds), st.mean(ds), st.median(ds), math.exp(st.mean(math.log(r) for r in rs))))
# fidelity: base sim vs recorded time
fs = [(x[base]['t_s'], x[base]['recorded_t']) for x in rows.values() if base in x and x[base]['why'] == 'done' and x[base].get('recorded_t')]
if fs:
    import statistics as st
    print('fidelity (%s, done episodes with a recorded time): n %d  sim mean %.2f  recorded mean %.2f  median |diff| %.2f' % (
        base, len(fs), st.mean(a for a, b in fs), st.mean(b for a, b in fs), st.median(abs(a - b) for a, b in fs)))
