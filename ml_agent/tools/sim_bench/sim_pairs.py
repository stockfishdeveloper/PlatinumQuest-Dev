"""Paired sim summary across batch dirs. usage: python sim_pairs.py <dir>[,<dir>...] <variant> [<variant> ...] [--base base] [--per-map]
Per variant vs base: pairs; episodes done in both: mean / median time difference, geometric-mean time ratio, better/worse
(> 0.3 s); falls and cap hits in each."""
import json, glob, sys, os, math, statistics as st
from collections import defaultdict
args = sys.argv[1:]; base = 'base'; per_map = False
if '--base' in args:
    i = args.index('--base'); base = args[i + 1]; del args[i:i + 2]
if '--per-map' in args:
    args.remove('--per-map'); per_map = True
dirs = args[0].split(','); variants = args[1:]
rows = defaultdict(dict)
for d in dirs:
    for f in glob.glob(os.path.join(d, '*.jsonl')):
        for l in open(f, encoding='utf-8'):
            try: r = json.loads(l)
            except Exception: continue
            rows[(r['map'], r['pool'], r['decision'])][r['variant']] = r
def summary(keys, v):
    ds = []; rs = []; b = w = 0; fb = fv = cb = cv = n = 0
    for k in keys:
        x = rows[k]
        if base not in x or v not in x: continue
        n += 1; a, c = x[base], x[v]
        fb += a['why'] == 'fell'; fv += c['why'] == 'fell'; cb += a['why'] == 'cap'; cv += c['why'] == 'cap'
        if a['why'] == 'done' and c['why'] == 'done':
            ds.append(c['t_s'] - a['t_s']); rs.append(c['t_s'] / max(.1, a['t_s']))
            b += c['t_s'] < a['t_s'] - .3; w += c['t_s'] > a['t_s'] + .3
    if not n: return None
    return '%3d pairs | both done %3d: mean %+5.2f s median %+5.2f s ratio %.3f | better %2d worse %2d | falls %d->%d cap %d->%d' % (
        n, len(ds), st.mean(ds) if ds else 0, st.median(ds) if ds else 0, math.exp(st.mean(math.log(r) for r in rs)) if rs else 1, b, w, fb, fv, cb, cv)
for v in variants:
    s = summary(list(rows), v)
    if s: print('%-8s ALL  %s' % (v, s))
    if per_map:
        for m in sorted({k[0] for k in rows}):
            s = summary([k for k in rows if k[0] == m], v)
            if s: print('   %-20s %s' % (m[:20], s))
