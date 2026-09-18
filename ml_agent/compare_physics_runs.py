"""Compare two physics_identity_probe.py result files (e.g. the shipped marbleblast.exe vs a
self-built engine): same key script, same start pose -> the rest positions and the tick-by-tick
trajectories should agree to within the async-bridge noise floor (the within-speed spread the
probe itself reports, ~0.2-0.8 u).

    python compare_physics_runs.py logs/physics_identity_shipped.json logs/physics_identity_mbx.json
"""
import sys
import json
import numpy as np


def main(a_path, b_path):
    a = json.load(open(a_path)); b = json.load(open(b_path))
    if a['script'] != b['script']:
        print('WARNING: the two runs used different key scripts; trajectories are not comparable')
    ta, tb = a['trials'], b['trials']
    n = min(len(ta), len(tb))
    print(f'{a_path}: {len(ta)} trials ({a["when"]}) | {b_path}: {len(tb)} trials ({b["when"]})')
    for i in range(n):
        x, y = ta[i], tb[i]
        ex, ey = np.array(x['end']), np.array(y['end'])
        tx, ty = np.array(x['traj']), np.array(y['traj']); m = min(len(tx), len(ty))
        dev = np.linalg.norm(tx[:m] - ty[:m], axis=1)
        print(f'trial {i + 1} ({x["speed"]}x vs {y["speed"]}x): rest {ex.round(2).tolist()} vs {ey.round(2).tolist()} '
              f'-> {np.linalg.norm(ex - ey):.3f} u apart; trajectory max dev {dev.max():.3f} u at tick {int(dev.argmax())}, '
              f'mean {dev.mean():.3f} u over {m} ticks{"; FELL" if x["fell"] or y["fell"] else ""}')
    # noise floor: within-run spread at the same speed
    for label, ts in (('A', ta), ('B', tb)):
        by = {}
        for t in ts:
            by.setdefault(t['speed'], []).append(np.array(t['end']))
        for spd, ends in sorted(by.items()):
            spread = max((np.linalg.norm(p - q) for p in ends for q in ends), default=0.0)
            print(f'  {label} within-run spread at {spd}x: {spread:.3f} u')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
