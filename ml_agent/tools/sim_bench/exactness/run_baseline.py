"""tools/sim_bench with every 10-19 speed-up switched OFF (the reference for exactness checks on other maps): eager
compiled rollouts, the learned model through model.act, NumPy observation queries, Python routing everywhere, and no
worker pool (do not pass --search-workers). usage (repo root, PYTHONPATH=ml_agent): python run_baseline.py <sim_bench args>"""
import sys, importlib.util
import nav.native.forecast as F
F.LAZY_ROLLOUTS = False
import nav.marble_dynamics.navigator as NV
NV.FAST_ACT = False; NV.FAST_OBS = False
NV.native_navigate = lambda sim, route, pos, gem: route.navigate(pos, gem)
import nav.native.route_native as RN
RN.register = lambda sim: None                 # attach() then installs no compiled distance / navigate
import nav.routing as RT
assert RT.NATIVE_DISTANCE is None and RT.NATIVE_NAVIGATE is None
assert '--search-workers' not in sys.argv
spec = importlib.util.spec_from_file_location('sim_bench_baseline', 'ml_agent/tools/sim_bench/sim_bench.py')
SB = importlib.util.module_from_spec(spec); spec.loader.exec_module(SB)
SB.main()
assert RT.NATIVE_DISTANCE is None and RT.NATIVE_NAVIGATE is None
