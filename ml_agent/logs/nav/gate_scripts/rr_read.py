# Read real_run gate results: per-round points, falls, use decisions by held type ('0' = blast with nothing held).
# python rr_read.py <tag> [<tag> ...]   (run from ml_agent; reads logs/learned_nav/rr_<tag>.txt round lines)
import ast, re, sys
for tag in sys.argv[1:]:
    rows = []
    for line in open(f'logs/learned_nav/rr_{tag}.txt', encoding='utf-8', errors='replace'):
        m = re.match(r'\s+round (\d+): (\{.*\})\s*$', line)
        if m:
            rows.append(ast.literal_eval(m.group(2)))
    if not rows:
        print(tag, 'no rounds'); continue
    p = [r['points'] for r in rows]
    n = len(p); mean = sum(p) / n
    sd = (sum((x - mean) ** 2 for x in p) / max(n - 1, 1)) ** 0.5
    uses = {}
    for r in rows:
        for k, v in r.get('pow_uses', {}).items():
            uses[k] = uses.get(k, 0) + v
    print('%-12s n=%d mean %.1f (sd %.1f) best %.0f | rounds %s | falls %s | speed %.2f | use decisions/round %s | stuck %s' % (
        tag, n, mean, sd, max(p), ' '.join('%.0f' % x for x in p), ' '.join(str(r['falls']) for r in rows),
        sum(r['speed'] for r in rows) / n, {k: round(v / n, 1) for k, v in sorted(uses.items())},
        ' '.join(str(r['stuck_breaks']) for r in rows)))
