# Points replay training run, 2026-10-05

The operator authorized this agent to collect replay data and start training on 10-05, superseding
the earlier instruction to leave startup to another model. No commits or pushes are authorized.

## Run setup

* Starting policy: protected `models/nav/nav_v10_28897.pth`. Previous endpoint 29665 remains in
  `models/nav/nav_r2_stop_29665.pth`; pre-run checkpoint hashes and old logs are archived in
  `logs/nav/points_20261005/before_run/`.
* Data: eight inference-only collectors on ports 9970-9977, 72000 decisions each, seeds
  20261005 + 101 * worker index. Real source rounds, original approach targets, actual items,
  no synthetic fixture data. See `logs/nav/points_20261005/collect_*.out`.
* Training: eight KOTM games on 8888-8895. Four stage-4 replay windows, four ordinary real rounds,
  actual weighted game points throughout. Base navigator frozen. First 20 updates train only
  the migrated critic, then the use head and approach/post-kick residual also learn.
* Numbered outputs use `models/nav/nav_points_20261005_<update>.pth` so old numbered snapshots
  are preserved. `nav_latest.pth` is the resume point for this run.
* `supervise_points_training.ps1` owns the trainer and game loop. Games launch offline and hidden,
  only after all worker listeners are ready. It resumes the same latest checkpoint if the trainer
  crashes or produces no output for ten minutes. It never stops because of a score or loss metric.
* The existing dashboard remains at http://localhost:8990. Run health/PIDs are recorded in
  `logs/nav/points_20261005/pids.json` and `supervisor.log` after startup. Training stdout/stderr
  remain `logs/nav/stdout.txt` and `stderr.txt`; timestamped trainer logs remain in `logs/nav/`.

To stop this run **only when the operator requests it**, create an empty file
`logs/nav/points_20261005/STOP`. The supervisor then closes its training processes and leaves the
last saved checkpoint. Otherwise leave training running, report poor metrics, and let the operator
decide. To restart later, remove that explicit STOP file and launch the supervisor with PowerShell
`-NoProfile -ExecutionPolicy Bypass -File supervise_points_training.ps1` from `ml_agent`.

## Validation

Before startup, 11 CPU regression tests passed. The first 20 collected development fixtures restored
in both identical-policy evaluation branches (40/40 successful). Their paired score difference was
0.05 points, standard error 0.05; paired fall difference was zero. This is a replay repeatability
check, not evidence of a learned powerup gain. Details: `logs/nav/points_20261005/restore_audit.json`.

The design and remaining limitations are documented in `HANDOFF_POINTS_REPLAY_2026-10-04.md`.
Real score improvement still requires the matched four-arm window gate and full-round comparisons.

## Startup results

Training started at **10:03:36 local time**. Supervisor PID 13248, trainer PID 27244, game-loop PID
35344; all eight game workers were ready by 10:04:06. See `pids.json` for current IDs after any restart.
The checkpoint named `nav_v10_28897.pth` internally records update **28895**, so the first new update
is 28896. The source file and its hash are unchanged.

Collection completed all 576000 requested decisions. It produced **120 eligible starts**:
102 development starts from 77 source rounds, 18 evaluation starts from 16 separate source rounds.
No source round appears in both sets. These are an initial corpus, not a broad transfer benchmark.
The ordinary-round workers keep recording fresh starts, but the loaded training dataset stays fixed
until a deliberate dataset refresh/reload. Existing workers do not automatically consume new files.
Provenance and hashes are in `logs/nav/points_20261005/dataset.json`.

All **120/120 starts** passed real restoration and one timed decision, with zero failed attempts
(`all_restore_audit.json`). The 20-start, full-length identical-policy audit above is a separate check.
The complete held-out baseline gate finished all **72/72 attempts** across the same 18 evaluation
starts, with no failed restores (`logs/nav/points_20261005/baseline_gate.json`). Mean window points:
base 10.61, no-use 10.83, force 11.17, initially learned 10.78. Forced use beat no-use by 0.33 points
(SE 0.31), which is inconclusive. The identical base/no-use policies differed by -0.22 points
(SE 0.22), including one fall in the base arm: near-edge replay variation can amplify into a
different outcome. Use larger/repeated gates before interpreting small gains. This baseline used
the untrained source policy, not a newly learned checkpoint. Its isolated evaluation game is closed.

The first saved new checkpoint checked at update **28900** has `reward_version=game_points_v1`,
15 critic-only updates remaining, and **maximum actor-weight change 0.0** from the loaded starting
policy. Critic normalization had begun adapting (mean 0.896, standard deviation 0.553).
The first observed PPO updates were finite, roughly 19-20 seconds each, with zero frame flips.
These startup checks establish that training is operating; they do not establish a score improvement.

**Verified after warm-up at 10:11:** checkpoint update **28920** has zero critic warm-up updates
remaining. The use head and residual have both changed (maximum parameter changes 0.0338 and
0.0448), while every frozen actor parameter is still exactly unchanged. See `policy_start_check.json`.
Updates were then taking about 15 seconds, value loss was 0.093, KL 0.001, and no restart had been
needed. A check of the first 500 completed windows found every duration exactly 12032 ms.
All eight main games remain running; collection and baseline evaluation games are closed.

The summary process was restarted with the required Git/Python PATH and run-specific snapshot prefix
`nav_points_20261005_best`. Its PID is in `summary.pid`; stdout/stderr are `summary.out` / `summary.err`
in the run folder. It reports every 15 minutes and saves new 50-round highs. The supervisor restarts
the trainer/game loop on failures; the summary and dashboard are separate reporting processes.

Checkpoint preservation: the previous latest was additionally copied to
`models/nav/nav_before_points_20261005_29665.pth`; the protected base was copied to
`models/nav/nav_points_20261005_seed.pth` before initializing the new `nav_latest.pth`.
Old numbered checkpoints are protected by the new run-specific numbered prefix. Nothing committed.
