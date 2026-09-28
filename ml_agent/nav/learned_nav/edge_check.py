"""Stage 3 M1: where marbles actually lose the floor at exact edges, at the speeds and angles recorded.

    python -m nav.learned_nav.edge_check          -> logs/learned_nav/stage3_edge_check.json

From recorded trials without a jump key: the decision where a supported marble first has no supporting contact and
then falls. The release point is estimated along the last supported decision's path at the fraction of its physics
sub-steps that still had support; its signed distance from the nearest exact void/drop edge (outward positive) says
where the game lets go. Consistent values near the edge line on every map mean the exported edges are where the game
has them.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav import dynamics3 as D3                                     # noqa: E402
from nav.learned_nav.geometry import Geometry, VOID, DROP                        # noqa: E402

OUT = os.path.join(HERE, 'logs', 'learned_nav', 'stage3_edge_check.json')


def signed_edge(eidx, p, zfl):
    e = eidx.e
    ci = int(np.clip(math.floor(p[0] - eidx.x0), 0, eidx.W - 1)); cj = int(np.clip(math.floor(p[1] - eidx.y0), 0, eidx.H - 1))
    cand = eidx.table[cj, ci]; cand = cand[cand >= 0]
    best = None
    for k in cand:
        s = e[k]
        if s[8] not in (VOID, DROP):
            continue
        a = s[0:2]; b = s[3:5]; d = b - a
        t = np.clip(np.dot(p[:2] - a, d) / max(1e-12, np.dot(d, d)), 0, 1)
        q = a + t * d
        z = s[2] + t * (s[5] - s[2])
        if abs(z - zfl) > 0.4:
            continue
        dist = np.linalg.norm(p[:2] - q)
        if best is None or dist < best[0]:
            best = (dist, float(np.dot(p[:2] - q, s[6:8])), s[6:8])
    return best


def main(max_per_map=3000):
    res = {}
    for m in D3.TRAIN_MAPS:
        g = Geometry(m); eidx = D3.EdgeIndex(g)
        rows_out = []
        for fn, S, meta in D3.load_shards(m):
            tr = S[:, D3.C_TRIAL].astype(np.int64)
            starts = np.r_[0, np.nonzero(np.diff(tr))[0] + 1]
            for i0 in starts:
                mt = meta[int(tr[i0])]
                if mt['prog']['family'].startswith('jump') or mt['start']['kind'] == 'air':
                    continue
                R_ = S[i0:i0 + mt['n']]
                sup = R_[:, D3.C_SUPPORT] > 0
                for i in range(1, len(R_) - 3):
                    if i >= 2 and sup[i - 2] and sup[i - 1] and not sup[i] and not sup[i + 1] and R_[i + 3, 4] < R_[i, 4] - 0.3:
                        # release point: the supported fraction of the last supported decision's sub-steps along its path
                        f = float(R_[i - 1, D3.C_SUPPORT] / max(1.0, R_[i - 1, D3.C_SUB]))
                        p_prev = R_[i - 2, D3.C_P].astype(np.float64)
                        p_last = p_prev + f * (R_[i - 1, D3.C_P].astype(np.float64) - p_prev)
                        p_first = R_[i, D3.C_P].astype(np.float64)
                        zfl = p_last[2] - 0.19
                        a = signed_edge(eidx, p_last, zfl); b = signed_edge(eidx, p_first, zfl)
                        if a is None or b is None or a[0] > 1.5:
                            break
                        v = R_[i - 1, D3.C_V].astype(np.float64)
                        across = float(np.dot(v[:2], a[2]))
                        rows_out.append((a[1], b[1], across, math.hypot(v[0], v[1])))
                        break
                if len(rows_out) >= max_per_map:
                    break
            if len(rows_out) >= max_per_map:
                break
        if not rows_out:
            continue
        A = np.asarray(rows_out)
        step = A[:, 2] * 0.064
        res[m] = {'n': len(A), 'release_signed_u': {'p5': float(np.percentile(A[:, 0], 5)), 'median': float(np.median(A[:, 0])),
                                                           'p95': float(np.percentile(A[:, 0], 95))},
                  'first_unsupported_signed_u': {'p5': float(np.percentile(A[:, 1], 5)), 'median': float(np.median(A[:, 1])),
                                                 'p95': float(np.percentile(A[:, 1], 95))},
                  'release_within_0_to_0.25u': float(np.mean((A[:, 0] >= -0.05) & (A[:, 0] <= 0.25))),
                  'speed_across_median': float(np.median(A[:, 2]))}
        print(m, json.dumps(res[m]), flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
