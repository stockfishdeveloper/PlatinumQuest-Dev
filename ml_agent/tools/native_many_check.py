"""predict_many / resume_at (nav/native/forecast.py, ABI 15) against the one-by-one compiled forecasts, on recorded planner states.

For each sampled decision of a recorded round: a set of candidates (sequence plans with offsets / jumps / direct / through, a brake
probe, a fixed action list, a gap run when a crossing is at hand) is forecast one by one with predict_replies and together with
predict_many; every field compared (actions, tick rows, reply states, ambiguity, failure, written-back plan state). resume_at(k)
is compared with viability.resume_forecast of a k-reply forecast of the same plan. The virtual-clock accounting is compared too:
with a planning deadline, both runs must leave the same clock and the same set of deadline refusals.

usage (repo root, PYTHONPATH=ml_agent): python ml_agent/tools/native_many_check.py --pool overnight_20261026/qr3_explore_baseline
       --seed 482017593 --map Sprawl_Hunt [--decisions 12]
"""
import argparse
import copy
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'sim_bench'))
import sim_world as W


def same_rollout(a, b):
    if isinstance(a, Exception) or isinstance(b, Exception):
        return type(a) is type(b), 'exception %r vs %r' % (a, b)
    if a['proposed'] != b['proposed']:
        return False, 'proposed'
    ra, rb = a['native_states'].rows, b['native_states'].rows
    if ra.shape != rb.shape or not np.array_equal(ra, rb):
        return False, 'rows %s %s' % (ra.shape, rb.shape)
    if len(a['states']) != len(b['states']):
        return False, 'states length'
    for i in range(len(a['states'])):
        sa, sb = a['states'][i], b['states'][i]
        for key in ('position', 'velocity', 'spin', 'held', 'clock_ms', 'radius', 'yaw'):
            if not np.array_equal(np.asarray(sa[key]), np.asarray(sb[key])):
                return False, 'state %d %s' % (i, key)
        if (sa.get('blast_context') or {}).get('value') != (sb.get('blast_context') or {}).get('value'):
            return False, 'state %d blast' % i
    if a['ambiguous_intervals'] != b['ambiguous_intervals'] or a['failure'] != b['failure'] or a['stopped_on_ambiguity'] != b['stopped_on_ambiguity']:
        return False, 'ambiguity / failure'
    if list(a['controls']) != list(b['controls']):
        return False, 'controls'
    if (a.get('bridge_resume') is None) != (b.get('bridge_resume') is None):
        return False, 'bridge_resume'
    if a.get('bridge_resume') is not None and a['bridge_resume'] != b['bridge_resume']:
        return False, 'bridge_resume values'
    return True, ''


def plan_state(p):
    out = {}
    for k in ('passed', 'previous', 'jumped', 'turned', 'hop_jumped'):
        if hasattr(p, k):
            v = getattr(p, k); out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    for name in ('route', 'next_route'):
        r = getattr(p, name, None)
        if r is not None:
            out[name] = r.waypoint
    return out


def candidates(S, GD, terrain, d, gems):
    gem = np.asarray(gems[0][:3], dtype=float); route = S.cached_route(terrain, gem)
    specs = [S.SequenceProposal(route, gem, 48), S.SequenceProposal(route, gem, 48, direct=True), S.SequenceProposal(route, gem, 48, offset=.5, turn_after=12),
             S.SequenceProposal(route, gem, 48, jump=3), S.SequenceProposal(route, gem, 40, use=2), S.SequenceProposal(route, gem, 48, scale=.7),
             GD.Proposal(gem, steps=20, brake=True), [[0., 0., 0., 0., 0, 0., 0, 0., 0.]] * 10]
    if len(gems) > 1:
        g2 = np.asarray(gems[1][:3], dtype=float)
        specs.append(S.SequenceProposal(route, gem, 48, through=(S.cached_route(terrain, g2), g2, float(np.linalg.norm(g2[:2] - gem[:2])))))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True); ap.add_argument('--seed', required=True); ap.add_argument('--map', required=True)
    ap.add_argument('--decisions', type=int, default=12)
    ap.add_argument('--only-decision', type=int, default=0); ap.add_argument('--verbose', action='store_true')
    o = ap.parse_args()
    W.install_virtual_clock()
    a = W.planner_args(o.map); terrain, engine = W.setup(a, o.map, native=True)
    import nav.marble_dynamics.search as S
    import nav.marble_dynamics.gem_demo as GD
    from nav.marble_dynamics.bridge import predict_replies
    from nav.native.forecast import predict_many, resume_at
    from nav.marble_dynamics.viability import resume_forecast
    from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded
    rows = W.load_journal(W.journal_path(o.pool, o.map, o.seed))
    rows = [r for r in rows if r['before']['native']['state'].get('mode', 0) == 0 and not r['before']['native']['state'].get('oob')
            and any(it.get('gem') for it in r['before'].get('items', []))]
    step = max(1, len(rows) // o.decisions); picked = rows[::step][:o.decisions]
    if o.only_decision:
        picked = [r for r in rows if r['decision'] == o.only_decision]

    def brief(x):
        if isinstance(x, Exception):
            return type(x).__name__
        return 'n%d fail=%s amb=%s stop=%s end=%s' % (len(x['proposed']), (x['failure'] or {}).get('reason'), x['ambiguous_intervals'][:3],
                                                    x['stopped_on_ambiguity'], np.round(x['states'][-1]['position'], 3).tolist())
    n_ok = n_bad = n_resume = n_resume_bad = 0; n_clock_bad = 0
    for r in picked:
        d = copy.deepcopy(r['before']); gems = [it['position'] for it in d['items'] if it.get('gem')]
        gems.sort(key=lambda g: float(np.linalg.norm(np.asarray(g[:2]) - np.asarray(d['native']['state']['position'][:2]))))
        engine.planning_deadline = None
        # 1. one by one against together, no deadline
        specs_a = candidates(S, GD, terrain, d, gems); specs_b = copy.deepcopy(specs_a)
        one = []
        for sp in specs_a:
            try:
                one.append(predict_replies(engine, d, sp, stop_on_ambiguity=True))
            except (ValueError, PlanningBudgetExceeded) as exc:
                one.append(exc)
        many = predict_many(engine, d, specs_b, stop_on_ambiguity=True)
        for i, (x, y) in enumerate(zip(one, many)):
            ok, why = same_rollout(x, y)
            if ok and plan_state(specs_a[i]) != plan_state(specs_b[i]):
                ok, why = False, 'plan state %r vs %r' % (plan_state(specs_a[i]), plan_state(specs_b[i]))
            n_ok += ok; n_bad += not ok
            if not ok:
                print('decision %d candidate %d (%s): %s' % (r['decision'], i, type(specs_a[i]).__name__, why))
            if o.verbose or not ok:
                print('   one: %s\n   many: %s' % (brief(x), brief(y)))
        # 2. resume_at(k) against resume_forecast of a k-reply forecast
        full = one[0]
        if not isinstance(full, Exception) and full['failure'] is None and not full['ambiguous_intervals'] and len(full['proposed']) >= 12:
            for k in (1, 5, 12):
                short = predict_replies(engine, d, S.SequenceProposal(S.cached_route(terrain, np.asarray(gems[0][:3], dtype=float)), np.asarray(gems[0][:3], dtype=float), k), stop_on_ambiguity=True)
                if short['failure'] or short['ambiguous_intervals'] or len(short['proposed']) != k:
                    continue
                ra = resume_at(d, full, k); rb = resume_forecast(d, short)
                same = (ra['native']['frame_clock_ms'] == rb['native']['frame_clock_ms'] and ra['native']['pending_move'] == rb['native']['pending_move']
                        and ra['bridge_action'] == rb['bridge_action'] and ra['pow_yaw_hold'] == rb['pow_yaw_hold'] and ra['pow_yaw'] == rb['pow_yaw']
                        and np.array_equal(ra['native']['state']['position'], rb['native']['state']['position'])
                        and np.array_equal(ra['native']['state']['velocity'], rb['native']['state']['velocity'])
                        and (ra.get('blast') or {}).get('value') == (rb.get('blast') or {}).get('value'))
                n_resume += 1; n_resume_bad += not same
                if not same:
                    print('decision %d resume_at(%d) differs: %r vs %r' % (r['decision'], k, {k2: ra[k2] for k2 in ('bridge_action', 'pow_yaw_hold', 'pow_yaw')},
                                                                        {k2: rb[k2] for k2 in ('bridge_action', 'pow_yaw_hold', 'pow_yaw')}))
        # 3. the virtual clock with a deadline that cuts the set
        for budget in (0.05, 0.2):
            specs_a = candidates(S, GD, terrain, d, gems); specs_b = copy.deepcopy(specs_a)
            vc0 = W.VC[0]; engine.planning_deadline = vc0 + budget
            one = []
            for sp in specs_a:
                try:
                    one.append(predict_replies(engine, d, sp, stop_on_ambiguity=True))
                except (ValueError, PlanningBudgetExceeded) as exc:
                    one.append(exc)
            vc_one = W.VC[0]; W.VC[0] = vc0
            try:
                many = predict_many(engine, d, specs_b, stop_on_ambiguity=True)
            except PlanningBudgetExceeded as exc:
                many = [exc] * len(specs_b)
            vc_many = W.VC[0]; W.VC[0] = vc0
            kinds_one = [type(x).__name__ if isinstance(x, Exception) else len(x['proposed']) for x in one]
            kinds_many = [type(x).__name__ if isinstance(x, Exception) else len(x['proposed']) for x in many]
            if vc_one != vc_many or kinds_one != kinds_many:
                n_clock_bad += 1
                print('decision %d budget %.2f: clock %.6f vs %.6f; %s vs %s' % (r['decision'], budget, vc_one, vc_many, kinds_one, kinds_many))
        engine.planning_deadline = None
    print('%s: candidates identical %d / %d; resume_at identical %d / %d; clock mismatches %d' % (o.map, n_ok, n_ok + n_bad, n_resume - n_resume_bad, n_resume, n_clock_bad))


if __name__ == '__main__':
    main()
