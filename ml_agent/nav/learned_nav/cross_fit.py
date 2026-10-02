"""Stage 6b: fit the route chooser's crossing-leg time (route.py JUMP_K, JUMP_EXTRA_S) from crossing-drill results.

    python -m nav.learned_nav.cross_fit v1 [--set dev]

Model (route.Route.legs): t_jump = JUMP_K x straight + NAV_TURN_S x (1 - cos turn) x speed / PICK_SPEED + JUMP_EXTRA_S.
Fitted on the successful trials (least squares over JUMP_K and JUMP_EXTRA_S with the turn term fixed), then the
failure cost is folded in: EXTRA += fall rate x FALL_COST_S (respawn and the walk instead) + timeout rate x TIMEOUT_S.
Prints the constants and the mean residual, and the same leg's fitted walking time for comparison.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from nav.learned_nav import route as RT                                          # noqa: E402
from nav.learned_nav.crossdrill import OUT_DIR                                   # noqa: E402

FALL_COST_S = 3.5
TIMEOUT_S = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--set', default='dev')
    a = ap.parse_args()
    rows = []
    for tag in a.tags:
        f = os.path.join(OUT_DIR, f'{tag}_{a.set}.jsonl')
        rows += [json.loads(l) for l in open(f)]
    ok = [r for r in rows if r['ok']]
    st = np.array([r['straight'] for r in ok])
    turn = np.array([RT.NAV_TURN_S * (1.0 - math.cos(math.radians(r['angle']))) * r['speed'] / RT.PICK_SPEED for r in ok])
    t = np.array([r['t_s'] for r in ok])
    A = np.c_[st, np.ones(len(ok))]
    k, extra = np.linalg.lstsq(A, t - turn, rcond=None)[0]
    res = t - (k * st + turn + extra)
    fall = np.mean([r['fell'] for r in rows]); to = np.mean([r['timeout'] for r in rows])
    print(f'{len(rows)} trials ({len(ok)} ok, fall rate {fall:.3f}, timeout rate {to:.3f})')
    print(f'fit: JUMP_K = {k:.4f} s/u, JUMP_EXTRA_S = {extra:.3f} s (residual sd {res.std():.2f} s, mean |res| {np.abs(res).mean():.2f})')
    print(f'with failures: JUMP_EXTRA_S = {extra + fall * FALL_COST_S + to * TIMEOUT_S:.3f} s')
    print(f'current route.py: JUMP_K {RT.JUMP_K}, JUMP_EXTRA_S {RT.JUMP_EXTRA_S}')
    tn = np.array([r['t_nav'] for r in ok])
    print(f'mean leg time {t.mean():.2f} s vs the navigator fitted {tn.mean():.2f} s on the same legs; '
          f'crossing faster on {np.mean(t < tn):.0%} of them, by {np.mean(tn - t):+.2f} s on average')


if __name__ == '__main__':
    main()
