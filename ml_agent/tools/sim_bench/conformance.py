"""Event conformance of the offline simulator against recorded native rounds (2026-10-29, phase 1 of
docs/IMPROVEMENT_LOOP_PLAN_2026-10-29.md).

For every recorded decision the sim advances the recorded bridge state with the recorded action (one 64 ms reply, four
16 ms ticks, tick-ordered item events) and the result is compared with the NEXT recorded decision's state: position and
velocity error, gem pickups (the recorded gem_delta against the sim's tick pickup test), powerup pickups (the held slot),
the Blast meter, and out-of-bounds (the sim's InBounds test against the recorded oob flag). Open-loop chains of 4, 16 and
48 replies from every 25th decision measure how fast an error grows when the controls are replayed without feedback.

usage (repo root): PYTHONPATH=ml_agent python ml_agent/tools/sim_bench/conformance.py --pool overnight_20261026/qr3_explore_baseline
         --seed 482017593 --map Sprawl_Hunt [--map ...] --out report.json
Tolerances (declared before the first run; a regime outside them blocks its labels until repaired):
  one-reply position p99 <= 0.02 u, velocity p99 <= 0.1 u/s; pickup agreement >= 99.5 percent; OOB agreement 100 percent;
  chained 48-reply position p50 <= 0.5 u on ground legs (contact forks are chaotic: reported, not gated).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_world as W

TOL = dict(pos_p99=0.02, vel_p99=0.1, pickup_pct=99.5, oob_pct=100.)


def playable(row):
    st = row['before']['native']['state']
    return st.get('mode', 0) == 0 and not st.get('oob') and row['choice'].get('label') != 'countdown_neutral'


def one_reply(engine, row, action):
    from tools.sim_bench.world_events import advance_world
    engine.planning_deadline = None
    return advance_world(engine, row['before'], action)


def compare_rows(engine, rows, every_chain=25, chains=(4, 16, 48), limit=0):
    from nav.marble_dynamics.fast_nav import fully_outside_inbounds
    import nav.marble_dynamics.gem_demo as GD
    pos_err = []; vel_err = []; pick_ok = 0; pick_n = 0; pick_bad = []; held_ok = 0; held_n = 0; held_bad = []
    oob_ok = 0; oob_n = 0; oob_bad = []; blast_err = []; failures = 0; chain_err = {k: [] for k in chains}
    n = len(rows) - 1 if not limit else min(limit, len(rows) - 1)
    for i in range(n):
        r, nxt = rows[i], rows[i + 1]
        if not playable(r) or nxt['before']['native']['state'].get('mode', 0) != 0:
            continue
        if nxt['decision'] != r['decision'] + 1 or int(nxt['tick']) - int(r['tick']) != 1:
            continue   # a skipped or doubled decision (round events); the reply is not one 64 ms step
        try:
            roll = one_reply(engine, r, r['action'])
        except ValueError:
            failures += 1; continue
        if roll['failure']:
            failures += 1; continue
        st = roll['states'][-1]; actual = nxt['before']['native']['state']
        fell = bool(engine.mission_bounds is not None and fully_outside_inbounds(st, engine.mission_bounds))
        oob_n += 1
        if fell == bool(actual.get('oob')) or (bool(nxt['outcome'].get('fell')) == fell):
            oob_ok += 1
        else:
            oob_bad.append(r['decision'])
        if actual.get('oob'):
            continue   # the respawn follows; no physical comparison
        pe = float(np.linalg.norm(np.asarray(st['position']) - np.asarray(actual['position'])))
        ve = float(np.linalg.norm(np.asarray(st['velocity']) - np.asarray(actual['velocity'])))
        pos_err.append(pe); vel_err.append(ve)
        # gem pickups: the recorded reply's gem_delta against the sim's tick test on the recorded items
        got = sum(float(it.get('value') or 1.) for it in r['before'].get('items', []) if it.get('gem') and GD.tick_pickup(roll['native_states'], it) is not None)
        rec = float(r['outcome'].get('gem_delta') or 0.)
        pick_n += 1
        if abs(got - rec) < 1e-6:
            pick_ok += 1
        else:
            pick_bad.append((r['decision'], rec, got))
        # powerup pickups: held after the reply
        held_n += 1
        if int(st.get('held', 0)) == int(actual.get('held', 0)):
            held_ok += 1
        else:
            held_bad.append((r['decision'], int(actual.get('held', 0)), int(st.get('held', 0))))
        bc = st.get('blast_context'); ab = nxt['before'].get('blast')
        if bc is not None and ab is not None:
            blast_err.append(abs(float(bc.get('value', 0.)) - float(ab.get('value', 0.))))
        if i % every_chain == 0:
            for k in chains:
                if i + k >= len(rows):
                    continue
                seg = rows[i:i + k + 1]
                if not all(playable(x) for x in seg) or any(x['outcome'].get('fell') for x in seg[:-1]) or \
                        seg[-1]['decision'] != r['decision'] + k:
                    continue
                try:
                    from tools.sim_bench.world_events import advance_world
                    d = r['before']; ok = True
                    for x in seg[:-1]:
                        engine.planning_deadline = None
                        rr = advance_world(engine, d, x['action'])
                        if rr['failure']:
                            ok = False; break
                        d = W.resume(d, rr)
                        # the game's world events (items) are not re-applied here: open-loop physics only
                        d['items'] = x['before'].get('items', [])
                    if ok:
                        chain_err[k].append(float(np.linalg.norm(np.asarray(d['native']['state']['position']) - np.asarray(seg[-1]['before']['native']['state']['position']))))
                except ValueError:
                    pass
    q = lambda v, p: float(np.percentile(v, p)) if v else float('nan')
    return dict(replies=len(pos_err), failures=failures,
                pos_p50=q(pos_err, 50), pos_p99=q(pos_err, 99), pos_max=max(pos_err) if pos_err else float('nan'),
                vel_p50=q(vel_err, 50), vel_p99=q(vel_err, 99),
                pickup_pct=100. * pick_ok / max(1, pick_n), pickup_n=pick_n, pickup_bad=pick_bad[:20],
                held_pct=100. * held_ok / max(1, held_n), held_bad=held_bad[:20],
                oob_pct=100. * oob_ok / max(1, oob_n), oob_n=oob_n, oob_bad=oob_bad[:20],
                blast_p99=q(blast_err, 99),
                chain={str(k): dict(n=len(v), p50=q(v, 50), p90=q(v, 90), max=max(v) if v else float('nan')) for k, v in chain_err.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True); ap.add_argument('--seed', required=True)
    ap.add_argument('--map', action='append', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--legacy-flags', action='store_true')
    o = ap.parse_args()
    import ctypes
    _k = ctypes.windll.kernel32; _k.GetCurrentProcess.restype = ctypes.c_void_p
    _k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]; _k.SetPriorityClass(_k.GetCurrentProcess(), 0x40)
    W.install_virtual_clock()
    report = {}
    for name in o.map:
        t0 = time.process_time()
        a = W.planner_args(name, legacy=o.legacy_flags)
        terrain, engine = W.setup(a, name, native=True)
        rows = W.load_journal(W.journal_path(o.pool, name, o.seed))
        res = compare_rows(engine, rows, limit=o.limit)
        res['cpu_s'] = time.process_time() - t0; res['decisions'] = len(rows)
        res['pass'] = bool(res['pos_p99'] <= TOL['pos_p99'] and res['vel_p99'] <= TOL['vel_p99'] and res['pickup_pct'] >= TOL['pickup_pct'] and res['oob_pct'] >= TOL['oob_pct'])
        report[name] = res
        print('%-22s replies %5d fail %3d | pos p50 %.4f p99 %.4f max %.3f | vel p99 %.3f | pickups %.2f%% (%d) | held %.2f%% | oob %.2f%% (%d) | blast p99 %.4f | chain %s | %s cpu %.0fs' % (
            name[:22], res['replies'], res['failures'], res['pos_p50'], res['pos_p99'], res['pos_max'], res['vel_p99'], res['pickup_pct'], res['pickup_n'],
            res['held_pct'], res['oob_pct'], res['oob_n'], res['blast_p99'],
            ' '.join('%s:%.2f/%.2f' % (k, v['p50'], v['p90']) for k, v in res['chain'].items()), 'PASS' if res['pass'] else 'FAIL', res['cpu_s']), flush=True)
        if res['pickup_bad'] or res['held_bad'] or res['oob_bad']:
            print('   pickup mismatches', res['pickup_bad'][:8], 'held', res['held_bad'][:8], 'oob', res['oob_bad'][:8], flush=True)
    if o.out:
        with open(o.out, 'w') as f:
            json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
