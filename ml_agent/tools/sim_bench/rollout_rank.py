"""Rollout ranking test (stage 3 fallback, 2026-10-31): can a cheap in-planner rollout rank the root candidates where the
learned ranker could not?

For every labelled situation (branch_bench --label-mode rows), the deterministic prefix is replayed, the root search is
recorded, and each candidate's own replies are extended with the planner's cheap route-following controller (the candidate's
prefix, then SequenceProposal toward the target gem with the through context to the next gem) to `--replies` replies. The
rollout is scored by the game's tick pickup test on the root's gems: points within the rollout, then the time of the first
pickup. The score's ranking is compared with the executed 10 / 20 s outcomes stored in the labels (pairwise agreement; the
realized gain of its argmax on the labels' futures), against the planner's cost and the learned ranker's 0.52-0.54.

usage (repo root, PYTHONPATH=ml_agent):
  python ml_agent/tools/sim_bench/rollout_rank.py --labels labels/run1/KingOfTheMarble_Hunt_base_aa3_explore_baseline.jsonl [...]
         --replies 128 --out rollout_rank.jsonl
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_world as W
import hunt_world as H
import full_round as FR


class GreedyFollow:
    """The cheap rollout policy: drive the route toward the nearest gem not yet passed (passed = within 0.6 u), with the
    planner's own stopping law (Proposal). No search, no jumps, no powerups: a lower bound on what the continuation collects."""
    def __init__(self, route, gems, steps):
        import nav.marble_dynamics.search as S
        self.S = S; self.route = route; self.gems = [np.asarray(g[:3], dtype=float) for g in gems]; self.steps = steps
        self.passed = [False] * len(self.gems); self.previous = None

    def __call__(self, state, index):
        S = self.S; GD = S._gem_demo
        p = np.asarray(state['position'], dtype=float)
        if index == 0:
            self.passed = [False] * len(self.gems); self.previous = None
        for k, g in enumerate(self.gems):
            if self.passed[k]:
                continue
            if self.previous is not None:
                a = self.previous; ab = p - a; tt = float(np.clip(((g - a) @ ab) / max(float(ab @ ab), 1e-12), 0., 1.))
                if float(np.linalg.norm(a + tt * ab - g)) < S.THROUGH_SWITCH_U:
                    self.passed[k] = True
            elif float(np.linalg.norm(p - g)) < S.THROUGH_SWITCH_U:
                self.passed[k] = True
        self.previous = p
        left = [g for g, done in zip(self.gems, self.passed) if not done]
        if not left:
            return [0.] * 9
        d = [float(np.linalg.norm(g[:2] - p[:2])) for g in left]
        gem = left[int(np.argmin(d))]
        goal = self.route.navigate(p, gem)[0]
        remaining = float(self.route.distance(p))
        distance = remaining if math.isfinite(remaining) else float(np.linalg.norm(gem[:2] - p[:2]))
        top = S.leg_top(self.route.terrain, state, distance)
        action = GD.Proposal(goal, 0., speed_distance=distance, min_speed=S.PICKUP_MIN_SPEED, top=top)(state, 0)
        action[8] = 0
        return action


class Extended:
    """The candidate's own replies, then the cheap controller toward the gem (and through it to the next gem)."""
    def __init__(self, prefix, follow, steps):
        self.prefix = list(prefix); self.follow = follow; self.steps = steps

    def __call__(self, state, index):
        if index < len(self.prefix):
            return self.prefix[index]
        return self.follow(state, index)


def rollout_score(engine, rec, info, roll, replies, horizon_s):
    import nav.marble_dynamics.search as S
    import nav.marble_dynamics.gem_demo as GD
    from nav.marble_dynamics.bridge import predict_replies
    gem = np.asarray(rec['gem'], dtype=float); dyn = rec['dynamics']; route = rec['route']
    item = rec.get('target_item') or next((i for i in dyn.get('items', []) if i.get('gem') and np.linalg.norm(np.asarray(i['position']) - gem) < .1), None)
    gems = [it['position'] for it in dyn.get('items', []) if it.get('gem')]
    # the target gem first (as the candidate intended), then greedy nearest
    follow = GreedyFollow(route, [gem] + [g for g in gems if np.linalg.norm(np.asarray(g) - gem) > .1], replies)
    ext = Extended(roll['proposed'], follow, replies)
    engine.planning_deadline = None
    try:
        r2 = predict_replies(engine, dyn, ext)
    except ValueError as exc:
        return dict(ok=False, why=str(exc)[:60])
    states = r2.get('native_states'); n_rep = len(r2.get('proposed') or [])
    picks = []
    for it in dyn.get('items', []):
        if not it.get('gem'):
            continue
        k = GD.tick_pickup(states, it)
        if k is not None:
            picks.append((k * .064, float(it.get('value') or 1.)))
    picks.sort()
    pts = sum(v for t, v in picks if t <= horizon_s)
    first = picks[0][0] if picks else horizon_s
    fell = bool(engine.mission_bounds is not None and any(FR.fully_outside_inbounds(s, engine.mission_bounds) for s in r2['states'][::4])) if hasattr(FR, 'fully_outside_inbounds') else False
    last = picks[-1][0] if picks else horizon_s
    return dict(ok=True, points=pts, first_pick_s=first, last_pick_s=last, picks=len(picks), replies=n_rep, failure=r2.get('failure') is not None, fell=fell,
                score=pts - 0.05 * first - 0.02 * last, score_fall=pts - 0.05 * first - 0.02 * last - (3. if fell else 0.))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--replies', type=int, default=128); ap.add_argument('--horizon-s', type=float, default=8.)
    ap.add_argument('--lead', type=int, default=47); ap.add_argument('--limit', type=int, default=0)
    o = ap.parse_args()
    import ctypes
    _k = ctypes.windll.kernel32; _k.GetCurrentProcess.restype = ctypes.c_void_p
    _k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]; _k.SetPriorityClass(_k.GetCurrentProcess(), 0x40)
    from nav.marble_dynamics.fast_nav import fully_outside_inbounds
    FR.fully_outside_inbounds = fully_outside_inbounds
    W.install_virtual_clock()
    cache = {}
    class O:
        cap_s = 0.; trace = False
    with open(o.out, 'a', encoding='utf-8') as out:
        done = 0
        for path in o.labels:
            for line in open(path, encoding='utf-8'):
                if not line.strip():
                    continue
                r = json.loads(line); name = r['map']
                if name not in cache:
                    a = W.planner_args(name); terrain, engine = W.setup(a, name, native=True)
                    cache[name] = (a, terrain, engine, H.MissionHunt(name), {})
                a, terrain, engine, mission, journals = cache[name]
                key = (r['pool'], str(r['seed']))
                if key not in journals:
                    journals[key] = W.load_journal(W.journal_path(r['pool'], name, r['seed']))
                rows = journals[key]; idx = next(i for i, x in enumerate(rows) if x['decision'] == r['decision'])
                t0 = time.process_time()
                res = FR.play_round(a, terrain, engine, mission, rows, 500, O, label='rr', start=rows[idx - o.lead], max_decisions=o.lead + 1,
                                    root_at=o.lead, record_root=True, future_seed=2000)
                recorded = res['root_candidates']
                # the labels' branches by label name -> executed outcomes
                exec_by_label = {}
                for bname, b in r['arms']['branches'].items():
                    exec_by_label[b['label']] = dict(kind=b['kind'], runs=b['runs'])
                scores = {}
                for rec in recorded:
                    for info, roll in rec['pairs']:
                        if roll is None or not isinstance(info, dict) or not roll.get('proposed'):
                            continue
                        lab = info.get('label')
                        if lab in scores or (lab not in exec_by_label and lab != r.get('root_chosen')):
                            continue
                        scores[lab] = rollout_score(engine, rec, info, roll, o.replies, o.horizon_s)
                        scores[lab]['cost'] = float(info.get('cost', math.inf)) if info.get('cost') is not None else None
                        scores[lab]['rejected'] = info.get('rejected')
                row = dict(map=name, pool=r['pool'], seed=r['seed'], decision=r['decision'], bucket=r['bucket'], root_chosen=r.get('root_chosen'),
                           base_runs=r['arms']['base'], branches=exec_by_label, rollout=scores, cpu_s=time.process_time() - t0)
                out.write(json.dumps(row, default=float) + '\n'); out.flush(); done += 1
                print('%s dec %5d: %d candidates scored, %d matched labels, cpu %.0fs' % (name[:12], r['decision'], len(scores),
                      sum(1 for l in scores if l in exec_by_label), row['cpu_s']), flush=True)
                if o.limit and done >= o.limit:
                    return


if __name__ == '__main__':
    main()
