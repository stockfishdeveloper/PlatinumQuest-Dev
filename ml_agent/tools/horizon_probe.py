"""At recorded stuck states, does a longer planning horizon find a plan that collects the target gem?

usage (repo root, PYTHONPATH=ml_agent): python ml_agent/tools/horizon_probe.py clips.json [--horizons 24,48,96] [--samples 3]
clips.json: the stuck-clip list (tools/stuck_clips.py format). For a few decisions inside each stuck stretch the planner's
full search runs unbudgeted on the compiled forecast with each horizon; reported per horizon: candidates that collect the
target gem, the best such plan (label, pickup time) and the search time.
"""
import argparse
import copy
import gzip
import json
import time
from pathlib import Path

import numpy as np

import nav.marble_dynamics.search as S
from nav.routing import RoutePlanner, goal_key
from nav.marble_dynamics.reliability_eval import tour

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_search_check as N


def main():
    p = argparse.ArgumentParser()
    p.add_argument('clips', type=Path)
    p.add_argument('--horizons', default='24,48,96')
    p.add_argument('--samples', type=int, default=3)
    a = p.parse_args()
    horizons = [int(h) for h in a.horizons.split(',')]
    clips = json.loads(a.clips.read_text(encoding='utf-8'))
    setups = {}
    for n, clip in enumerate(clips, 1):
        folder = Path(clip['journal']); name = folder.parts[-4]
        if name not in setups:
            setups[name] = N.setup(name)
        terrain, walk, engine, nav, sim = setups[name]; engine.native_sim = sim
        R = {}
        with gzip.open(folder / 'decisions.jsonl.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line); R[d['decision']] = d
                except ValueError:
                    pass
        decs = [k for k in sorted(R) if clip['start'] + 47 <= k <= clip['end'] - 16]
        decs = decs[::max(1, len(decs) // a.samples)][:a.samples]
        print('== clip %d: %s' % (n, clip['title']), flush=True)
        for k in decs:
            d = R[k]; base = d['before']; st = base['native']['state']
            if st.get('mode', 0) != 0 or st.get('oob'):
                continue
            dest = (d['choice'].get('goal_context') or {}).get('destination')
            if not dest:
                continue
            raw = np.asarray(d['raw_observation'], np.float32)
            tgt, nxt, _ = tour(terrain, {}, raw, list(dest) + [0, 0])
            engine.next_for = {goal_key(tgt): nxt} if tgt is not None and nxt is not None and goal_key(tgt) == goal_key(dest) else {}
            pos = np.asarray(st['position']); g = np.asarray(dest[:3])
            line = '   dec %5d (t %.1f s) gem %.1f u away, %+.1f u up |' % (k, k * .064, np.linalg.norm((g - pos)[:2]), g[2] - pos[2])
            for h in horizons:
                route = RoutePlanner(walk); engine.planning_deadline = None
                t0 = time.perf_counter()
                try:
                    chosen, pairs = S.search(engine, route, copy.deepcopy(base), raw, g, nav, h, brake_replies=0)
                except Exception as exc:
                    line += ' H%d error %s |' % (h, str(exc)[:40]); continue
                dt = time.perf_counter() - t0
                coll = [(i['possible_pickup_interval'], i['label'], i['rejected']) for i, r in pairs if i.get('possible_pickup_interval') is not None]
                ok = [c for c in coll if c[2] is None]
                best = min(ok) if ok else None
                pick = chosen[0] if chosen else None
                line += ' H%d: %d collect (%d pass gates)%s, chosen %s %s, %.2f s |' % (
                    h, len(coll), len(ok), ', best %s at %.1f s' % (best[1], best[0] * .064) if best else '',
                    pick['label'] if pick else None, ('pk %.1f s' % (pick['possible_pickup_interval'] * .064)) if pick and pick.get('possible_pickup_interval') else '', dt)
            print(line, flush=True)


if __name__ == '__main__':
    main()
