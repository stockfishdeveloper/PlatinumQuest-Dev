# Search speed-up exactness checks (2026-10-19)

The 10-19 search speed-up (parallel candidate workers, lazy compiled rollouts, the learned model's observation and
routing in C++) must leave every planner decision unchanged. These tools check that; run them from the repo root with
`PYTHONPATH=ml_agent` after any change to the planner, `nav/native` or the observation code.

* **Decision traces (the main check).** `tools/sim_bench/sim_bench.py --trace` prints one `   t ...` line per decision.
  Compare the lines of two runs with `cmp`.
  * Cragmire against the original v86 planner: `cragmire_v86_reference.trace` holds 1,875 decisions from 3 episodes
    (decisions 93, 513 and 630 of the v84 pools). Reproduce it with:
    `python ml_agent/tools/sim_bench/sim_bench.py --map Cragmire_Hunt --pool planner_v84_start0_20261017 --pool planner_v84_start1_20261017 --per-map 24 --whole-group --cap-s 40 --native --pflag=--horizon=48 --real-clock --budget 1000 --max-episodes 3 --trace --search-workers 8 --out NUL`
  * Any map: `run_baseline.py` with the same arguments, minus `--search-workers`, runs the planner with every
    speed-up switched off. Compare its trace with the normal run's.
* **Compiled ports.** `test_obs_terrain.py <n> <maps...>` checks the observation queries and
  `test_route_native.py <map> <goals> <positions>` checks the routing. Every count of differing answers must be 0.
* **Latency.** `phase_times.py <out prefix> <sim_bench args>` reports the per-decision latency distribution and where the
  time goes (search, parallel evaluation, landing projections, braking backups). Set `SPEEDUP_OFF=1` to time the
  original planner instead.

The checks use `--budget 1000` (no deadline) so that the serial and parallel searches evaluate the same candidates.
Under a real wall-clock budget, a faster search legitimately evaluates more candidates before the deadline.
