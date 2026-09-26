"""Score nav/physics.predict_landing against what actually happened (2026-09-25, HANDOFF 28.39).

  python -m nav.validate_landing [trace.csv]      default: logs/nav/real_trace_night142_1x.csv

For every floor takeoff in a real-run trace: verdict (floor / edge / wall / void) vs the outcome
(landed on present floor after crossing void / fell / hop), for hold-forward and ballistic modes, plus
the flat-strip sanity table and the human demo's landing error.
"""
import csv
import json
import math
import os
import sys

import numpy as np

from terrain_obs import TerrainMap
from nav.terrain import TerrainGrid
from nav.physics import predict_landing, jump_verdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def takeoffs_from_trace(t, path):
    present = np.isfinite(t.heights).any(0)

    def floor_here(x, y):
        i = int(round((x - t.x0) / t.res)); j = int(round((y - t.y0) / t.res))
        return 0 <= i < present.shape[1] and 0 <= j < present.shape[0] and present[j, i]
    rows = list(csv.DictReader(open(path)))
    out = []; prev = 0
    for k, r in enumerate(rows):
        j = int(float(r['jump'])); onf = float(r['on_floor']) > 0.5
        if j and not prev and onf and k > 0:
            pr = rows[k - 1]; x, y, z = float(pr['x']), float(pr['y']), float(pr['z']); vx, vy = float(pr['vx']), float(pr['vy'])
            if math.hypot(vx, vy) >= 1.0:
                res = 'hop'; saw = False
                for q in rows[k + 1:k + 40]:
                    qx, qy = float(q['x']), float(q['y'])
                    if int(float(q['fell'])):
                        res = 'fell'; break
                    if not floor_here(qx, qy):
                        saw = True
                    elif saw and float(q['on_floor']) > 0.5:
                        res = 'landed'; break
                out.append((x, y, z, vx, vy, res))
        prev = j
    return out


def score(t, takeoffs, label, **kw):
    c = {}
    for x, y, z, vx, vy, res in takeoffs:
        v = predict_landing(t, x, y, z, vx, vy, **kw)['verdict']; c[(v, res)] = c.get((v, res), 0) + 1
    fp = sum(n for (v, o), n in c.items() if v == 'floor' and o == 'fell')
    fn = sum(n for (v, o), n in c.items() if v != 'floor' and o == 'landed')
    ok = sum(n for (v, o), n in c.items() if (v == 'floor') == (o == 'landed') and o != 'hop')
    print(f'{label:14s}: approved-and-fell {fp}, refused-but-landed {fn}, right {ok}/{sum(1 for tk in takeoffs if tk[5] != "hop")}   {dict(sorted(c.items()))}')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'logs', 'nav', 'real_trace_night142_1x.csv')
    t = TerrainGrid(TerrainMap.resolve('KingOfTheMarble_Hunt'))
    d = json.load(open(os.path.join(HERE, 'physics', 'jump_envelope.json')))
    print('flat strip (measured -> predicted range, hold-forward):',
          ', '.join(f"{r['speed']:.1f}: {r['range']:.1f} -> {predict_landing(t, -12.0, 1.0, 20.9, 0.0, r['speed'])['dist']:.1f}" for r in d['hold'] if r['speed'] > 1))
    tk = takeoffs_from_trace(t, path)
    print(f'{os.path.basename(path)}: {len(tk)} takeoffs', {o: sum(1 for x in tk if x[5] == o) for o in ('landed', 'fell', 'hop')})
    score(t, tk, 'hold-forward'); score(t, tk, 'ballistic', hold=False)
    for hold in (True, False):
        c = {}
        for x, y, z, vx, vy, res in tk:
            v, _ = jump_verdict(t, x, y, z, vx, vy, hold=hold); c[(v, res)] = c.get((v, res), 0) + 1
        fp = sum(n for (v, o), n in c.items() if v == 'floor' and o == 'fell'); fn = sum(n for (v, o), n in c.items() if v != 'floor' and o == 'landed')
        print(f'jump_verdict hold={hold!s:5s}: approved-and-fell {fp}, refused-but-landed {fn}   {dict(sorted(c.items()))}')
    # the human demo: all of its gap jumps landed; how many does the robust test approve?
    demo = os.path.join(HERE, 'demos', 'demo_20260924_212808.npz')
    if os.path.exists(demo):
        d = np.load(demo, allow_pickle=False); raw = d['obs_raw']; inp = d['inputs_raw']; pos = raw[:, 0:3]; vel = raw[:, 3:6]
        fz = np.array([t.floor_z(float(x), float(y), float(z)) for x, y, z in pos])
        onf = (np.abs(vel[:, 2]) < 0.3) & (np.abs(pos[:, 2] - fz) < 0.6); jump = inp[:, 4] > 0.5
        onsets = np.nonzero(jump[1:] & ~jump[:-1] & onf[:-1])[0] + 1
        for hold in (True, False):
            hc = {}
            for k in onsets:
                v, _ = jump_verdict(t, *map(float, pos[k - 1]), float(vel[k - 1, 0]), float(vel[k - 1, 1]), hold=hold); hc[v] = hc.get(v, 0) + 1
            print(f'human demo ({len(onsets)} jumps, all landed): verdicts hold={hold!s:5s} {hc}')


if __name__ == '__main__':
    main()
