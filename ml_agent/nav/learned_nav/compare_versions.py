"""Two planner versions on the same drill starts (paired): success, failure types, time per start, disagreements.

    python -m nav.learned_nav.compare_versions devb v8 v9
Reads logs/learned_nav/drills/<set>_<tag>_<map>.jsonl for both tags; only feasible (fixture-safe) starts both ran.
McNemar's exact test on the discordant pairs says whether the difference in success could be chance.
"""
import glob
import json
import math
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(HERE, 'logs', 'learned_nav', 'drills')


def load(set_name, tag):
    out = {}
    for f in glob.glob(os.path.join(D, f'{set_name}_{tag}_kotmjump_p?.jsonl')):
        for line in open(f):
            r = json.loads(line)
            if r['safe_start'] and 'success' in r:          # rows without a result: the game refused the teleport
                out[(r['map'], r['id'])] = r
    return out


def mcnemar_p(b, c):
    """Two-sided exact McNemar p for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main(set_name, a, b):
    A, B = load(set_name, a), load(set_name, b)
    keys = sorted(set(A) & set(B))
    rep = {'set': set_name, 'paired_starts': len(keys)}
    for tag, R in ((a, A), (b, B)):
        rows = [R[k] for k in keys]
        ct = [0.064 * (r['decisions'] if r['success'] else 188) for r in rows]
        per = {}
        for m in sorted({k[0] for k in keys}):
            rr = [R[k] for k in keys if k[0] == m]
            per[m[-2:]] = f'{sum(r["success"] for r in rr)}/{len(rr)}'
        rep[tag] = {'success': sum(r['success'] for r in rows), 'rate': round(sum(r['success'] for r in rows) / max(1, len(rows)), 4),
                    'per_gem': per, 'failures': dict(Counter(r['status'] for r in rows if not r['success'])),
                    'time_per_start_s': round(float(np.mean(ct)), 2) if ct else None}
    only_a = [k for k in keys if A[k]['success'] and not B[k]['success']]
    only_b = [k for k in keys if B[k]['success'] and not A[k]['success']]
    rep['only_' + a] = len(only_a); rep['only_' + b] = len(only_b)
    rep['mcnemar_p'] = round(mcnemar_p(len(only_a), len(only_b)), 4)
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == '__main__':
    main(*sys.argv[1:4])
