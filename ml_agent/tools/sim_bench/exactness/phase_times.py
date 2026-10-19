"""Where a planner decision's wall time goes, without a profiler: perf_counter wrappers around the controller's action,
pick / order_choice, search, the parallel candidate evaluation, and search's post-loop phases (landing projections,
braking backups, two-level, the previous plan). Runs tools/sim_bench episodes on the real clock (pass --real-clock and
optionally --search-workers N). Writes <out>.json (per-decision records) and prints the distribution and the split.
usage (repo root, PYTHONPATH=ml_agent): python phase_times.py <out prefix> <sim_bench args>"""
import sys, os, time, json, importlib.util, collections
out = sys.argv[1]; sys.argv = [sys.argv[0]] + sys.argv[2:]
if os.environ.get('SPEEDUP_OFF') == '1':          # every 10-19 speed-up switched off (as run_baseline.py)
    import nav.native.forecast as _F; _F.LAZY_ROLLOUTS = False
    import nav.marble_dynamics.navigator as _NV; _NV.FAST_ACT = False; _NV.FAST_OBS = False
    _NV.native_navigate = lambda sim, route, pos, gem: route.navigate(pos, gem)
    import nav.native.route_native as _RN; _RN.register = lambda sim: None
    assert '--search-workers' not in sys.argv
spec = importlib.util.spec_from_file_location('sim_bench_pt', 'ml_agent/tools/sim_bench/sim_bench.py')
SB = importlib.util.module_from_spec(spec); spec.loader.exec_module(SB)
import numpy as np
RE = SB.RE
import nav.marble_dynamics.search as S
import nav.marble_dynamics.parallel_search as PS
REC = []; CUR = None; DEPTH = collections.Counter()


def wrap(mod, name, key, nested_ok=True):
    f = getattr(mod, name)
    def w(*a, **k):
        global CUR
        if CUR is None or (DEPTH[key] and not nested_ok):
            return f(*a, **k)
        DEPTH[key] += 1; t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            DEPTH[key] -= 1
            if DEPTH[key] == 0:
                CUR[key] = CUR.get(key, 0.) + time.perf_counter() - t; CUR['n_' + key] = CUR.get('n_' + key, 0) + 1
    setattr(mod, name, w)


for key in ('project_airborne', 'check_backups', 'apply_two_level', 'search'):
    wrap(S, key, key)
wrap(RE, 'search', 'search')            # reliability_eval imported search by name
wrap(RE, 'pick', 'pick'); wrap(RE, 'order_choice', 'order_choice')
wrap(PS.ParallelEvaluator, 'evaluate', 'parallel_evaluate')
_orig_action = RE.ContinuousController.action


def action(self, msg, remaining):
    global CUR
    CUR = {}
    t = time.perf_counter()
    try:
        return _orig_action(self, msg, remaining)
    finally:
        CUR['total'] = time.perf_counter() - t; REC.append(CUR); CUR = None


RE.ContinuousController.action = action
SB.main()
json.dump(REC, open(out + '.json', 'w'))
L = np.array([r['total'] for r in REC])
print('decisions %d  mean %.3f s  median %.3f  p90 %.3f  p99 %.3f  max %.2f  <=64 ms %.0f%%  <=128 ms %.0f%%  <=256 ms %.0f%%' % (
    len(L), L.mean(), np.median(L), np.percentile(L, 90), np.percentile(L, 99), L.max(), 100 * np.mean(L <= .064),
    100 * np.mean(L <= .128), 100 * np.mean(L <= .256)))
tot = L.sum()
for key in ('pick', 'order_choice', 'search', 'parallel_evaluate', 'project_airborne', 'check_backups', 'apply_two_level'):
    v = sum(r.get(key, 0.) for r in REC); n = sum(r.get('n_' + key, 0) for r in REC)
    print('  %-18s %8.2f s  %5.1f%% of decision time  %6d calls  %.1f ms a call' % (key, v, 100 * v / tot, n, 1000 * v / max(1, n)))
srch = np.array([r.get('n_search', 0) for r in REC])
print('  searches per decision: mean %.2f, decisions with a search %.0f%%, max %d' % (srch.mean(), 100 * np.mean(srch > 0), srch.max()))
