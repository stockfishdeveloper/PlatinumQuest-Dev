# Super Speed continuation handoff, 2026-10-05, 12:05 MDT

Read this first. Written at the operator's request so another agent can continue.
Training remains running. No commits or pushes were made. The score goal is NOT
complete, and no Super Speed change has yet demonstrated a reliable score gain.

Final process check at 12:09: main reached 29297, separate experiment 29315 of
29360. Both are progressing. The detailed numerical snapshot below is from 12:04.

**STATUS 22:40 (continuing agent):** RETRAINING on the corrected terrain map UNTIL PARITY (operator 22:20: "start
training on this new map until it's at parity with the old version or is better"). Supervisor restarted 22:34 (trainer
37300; FREEZE_BASE False, LR 1e-4, all 8 full rounds, two-gem kick rule). Parity = 64 deterministic rounds with learned
use on the corrected map at 162 or more (old version 161.98 on the old map). Check about every 2 h with main paused.
Operator 23:15, goals for tonight in order: (1) parity or better vs the old map, (2) then a single round above 175 WITH
Super Speeds used. Do NOT stop training at parity; keep going for (2). Any change to the training setup needs his OK.
10-06 07:02: (1) REACHED: 163.00 over 64 rounds (31435; checks 158.0, 161.05, 161.47 before). 09:10 final check
(31715): 164.94 (95% 163.8-166.1), falls 0.78 a round, 11 rounds >= 170, best 173 (3 kicks). Best of the night 175 (1
kick); none above 175, so (2) is not met. Resumed 09:26 on his OK; checks 165.48, 162.50, 166.55, 165.86. TRAINING
STOPPED 17:48 by the operator (STOP file left): waiting for his decision on what next. 18:00 rolled back on his
request: nav_latest = nav_points_20261005_parity_32565.pth (166.55, the best); checkpoints after it deleted. 576 test
rounds, best 175 twice, never above. Chart logs/nav/goal175/retrain_checks_20261006.png. Log 40.60. The corrected map (symmetric) passed an in-game teleport test on all 2,358 changed
spots (log 40.59). The 30295 endpoint (old map) is kept as nav_points_20261005_stop_30295.pth. Operator rules: max 8
game instances; a plain-language plan and his OK before any training start; never tail supervisor.log from a monitor.
Details: GOAL_175_2026-10-05.md, log 40.51-40.60; the process table below is historical.

## 1. Goal, permissions, and first actions

The operator wants BOTH:

1. A complete normal KingOfTheMarble_Hunt round above 175, with actual Super Speed
   consumption and a measured kick, preserving the checkpoint and trace.
2. Super Speed to contribute to higher scores. Establish this against the same
   checkpoint with uses disabled; a lucky high score alone does not prove it.

Earlier "do not start training" instructions were superseded: the operator
explicitly authorized collecting replay data and running training. Continue the
work autonomously. Standing restrictions:

* No map-specific policy hacks, coordinates, KOTM-specific route tables, or reward
  changes. Train from actual game points. Terrain/physics-based rules may be generic.
* Do not stop the MAIN training run for its metrics. Report regressions; only the
  operator ends that run. Recover the same run after crashes/hangs.
* Do not overwrite protected checkpoints. Keep experiments isolated. No commits.
* Do not train, calibrate, or tune on Horizon or Archipelago. Human demonstrations
  are measurements, not trajectories to copy into the policy.
* Leave client/lobby/live behavior alone. Focus on training and evaluation.

Next actions, in order:

1. Read the two runtime manifests below and inspect current updates. Do not launch
   another main trainer. There are deliberately TWO trainers right now.
2. Let the separate fast-state experiment finish its PREDECLARED 120 updates at
   update 29360. Its finite end does not end the main run. Expected around 12:19
   if its present 12-14 seconds/update persists; verify, do not assume completion.
3. When that trainer has exited normally, preserve its final checkpoint and free
   ONLY its two games on 9970-9971. Evaluate the candidate and its seed on the
   prescribed held-out windows, then ordinary full rounds (recipes below).
4. **Main run regression needs attention:** at 12:04, latest 50 sampled training
   rounds averaged 143.38 points with 4.3 falls/round, versus about 153 earlier.
   Evaluate an immutable main checkpoint when a probe slot is free. Do not
   promote main latest or assume that more updates mean a better model.
5. If the expanded data does not help, the most recent unfinished investigation
   is a short-horizon spin-aware floor predictor, section 7. It is a diagnostic,
   NOT an implemented controller or a proven improvement. Do not repeat the
   already failed controller changes in section 5 without a specific new reason.

## 2. Environment and current processes

Repository: C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev

Use PowerShell. Python commands below run with working directory ml_agent.

```powershell
$ErrorActionPreference = 'Stop'
$ml = 'C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent'
$py = 'C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe'
$gameDir = Join-Path $ml '..\Marble Blast Platinum'
$gameExe = Join-Path $gameDir 'marbleblast_mbx.exe'
Set-Location $ml
```

At 12:04:

| Process | PID | Role |
| --- | ---: | --- |
| Main supervisor | 13248 | Same-run crash/hang recovery |
| Main trainer | 27244 | Eight workers, ports 8888-8895 |
| Main game loop | 35344 | Owns the eight main games |
| Dashboard | 29932 | http://localhost:8990 |
| Summary bash | 35148 | Periodic main-run summaries |
| Separate CPU trainer | 30732 | Two replay workers, ports 9970-9971 |
| Separate worker/game on 9970 | 45948 / 46456 | Fast-state experiment |
| Separate worker/game on 9971 | 14492 / 35608 | Fast-state experiment |

Main game PIDs by port: 8888=4152, 8889=34104, 8890=28696, 8891=41756,
8892=44192, 8893=41524, 8894=19524, 8895=49060. Verify ownership before any cleanup:
PIDs can become stale. Never kill all marbleblast or python processes.

Runtime manifests, authoritative over this table:

* Main: logs/nav/points_20261005/pids.json
* Separate learner: logs/nav/ss_fast_20261005/pids.json
* Probe manifest: logs/nav/goal175/active_gates.json is currently EMPTY; the two
  spare ports now belong to the separate learner, not to evaluation.

GPU has 8 GB. Main eight games plus trainer and two spare games use about 7.27 GB.
Run at most TWO additional games, with CPU-only probe models. Do not add a third
probe while the separate learner owns both slots.

```powershell
Get-Content logs/nav/points_20261005/pids.json -Raw
Get-Content logs/nav/ss_fast_20261005/pids.json -Raw
rg 'NAV upd=' logs/nav/stdout.txt | Select-Object -Last 2
rg 'NAV upd=' logs/nav/ss_fast_20261005/stdout.txt | Select-Object -Last 2
Get-Content logs/nav/stderr.txt -Tail 5
Get-Content logs/nav/ss_fast_20261005/stderr.txt -Tail 5
```

Both stderr files currently contain only the known torch.load FutureWarning.
All ten game listeners were present at 12:04. Launch helpers/games hidden and
offline. Avoid blocking tool waits longer than 60 seconds.

## 3. What is being trained

### Main run

Started 10:03:36, all eight games ready 10:04:06. Last checked update 29280 at
12:03:48. Four real replay-window workers and four ordinary-round workers.
The original development dataset has 102 starts, loaded once at startup. Main
workers have NOT reloaded the expanded dataset described below.

Architecture: protected ordinary navigator is frozen, including its recurrent
core. Train only use_head, ss_res, and value_head. Direct observations give the
powerup heads inventory/use state. The steering residual can prepare a turn
within 18 units, act while a kick is pending, and for two seconds after firing.
Actual fire still requires the existing floor/speed/terrain approval, including
the predicted resolved-speed cap of 24 u/s. The use sampling mixture has a 0.2
exploration floor and is scored consistently by PPO. Deterministic evaluation
uses the learned preference without that exploration floor.

Reward is actual weighted gem points. Stage 4 restores real world snapshots and
runs exactly 12032 simulated milliseconds, including falls/recovery. It uses the
actual final state for bootstrapping, with elapsed-time discounting. No synthetic
six-gem cap or fake arrival reward. Replay starts warm recurrent state with up to
32 recorded observations. The migration reset the old shaped critic/Adam and
ran 20 critic-only updates; that warm-up is long finished.

Main outputs:

* models/nav/nav_latest.pth, saved every five updates.
* models/nav/nav_points_20261005_<six-digit-update>.pth, every 25 updates.
* logs/nav/stdout.txt, stderr.txt, train_nav_20261005_100337.log.
* logs/nav/points_20261005/ for supervisor and summary logs.
* Best-50 snapshot so far: nav_points_20261005_best_28955_154.pth. This is a
  training-score selection, not held-out evidence of a powerup gain.

At 12:04: 548 complete sampled training rounds, mean 152.27, best 168. Latest 50
mean 143.38, best 160, 215 falls. Earlier last-50 mean was about 153.8. Metrics
and protected hash: logs/nav/goal175/handoff_status_20261005_1204.json.
Do not compare sampled training means directly to deterministic evaluation means.

### Separate fast-state experiment

Started 11:52:04. Same PPO, CPU with two threads, two stage-4 replay workers,
no full-round workers, no hand-coded control overrides. Resume source is an
immutable copy of main update 29240, predating the latest main-run deterioration.
The source critic and optimizer are unchanged. Frozen navigator tensors were
checked against the protected base before launch.

Fixed duration: update 29240 -> 29360, exactly 120 additional updates. At
12:03:57 it reached 29291. Its last 100 replay windows averaged 10.29 points,
0.23 falls, 60% with a recorded fire. These are training diagnostics, not gains.

* Source: models/nav/ss_fast_20261005/seed.pth
* Resume/output: models/nav/ss_fast_20261005/nav_latest.pth
* Numbered output: models/nav/ss_fast_20261005/nav_ss_fast_<update>.pth
* Full config/source/data hashes: logs/nav/ss_fast_20261005/manifest.json
* Runtime PIDs: logs/nav/ss_fast_20261005/pids.json
* Logs: logs/nav/ss_fast_20261005/{stdout.txt,stderr.txt,train_nav_*.log}
* Source SHA256: 86019653081af32571cfe57776a9dbe79aafbf8955fe51bff80746da3ac05315

There is NO separate supervisor for this finite experiment. train_nav retries
exceptions from its own latest checkpoint, but a dead game may need targeted
relaunch/recovery. The two games may remain open after normal trainer completion.
Do not start the main supervisor for this experiment; it is hardcoded to the
main ports/output. If recovery is required, re-establish these environment values
and launch nav.train_nav, then games after the two worker listeners appear:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:NAV_CKPT = Join-Path $ml 'models/nav/ss_fast_20261005/nav_latest.pth'
$env:NAV_OUTPUT_DIR = Join-Path $ml 'models/nav/ss_fast_20261005'
$env:NAV_LOG_DIR = Join-Path $ml 'logs/nav/ss_fast_20261005'
$env:NAV_CHECKPOINT_PREFIX = 'nav_ss_fast'
$env:NAV_INSTANCES = '2'
$env:NAV_PORT0 = '9970'
$env:NAV_STOP_UPDATE = '29360'
$env:NAV_TORCH_THREADS = '2'
$env:NAV_DRILL_PLAN = '0:4,1:4'
$env:NAV_DRILL4_STARTS = Join-Path $ml 'datasets/ss_drill/fast_20261005/train_dev.json'
$env:NAV_DRILL_EVAL = '0'
$env:NAV_DRILL_NO_USE = '0'
$env:NAV_LIVE_RECORD = '0'
$env:NAV_TOUR = 'walk'
$env:NAV_REAL_GEMS = '1'
$env:NAV_NO_SAVE = '0'
$env:NAV_TRAIN_WATCH = '0'
$env:NAV_OBS_MS = '64'
$env:NAV_SPEED = '3'
$env:NAV_RENDER_EVERY = '100'
foreach ($flag in @('NAV_SS_ROUTE','NAV_SS_PREP','NAV_SS_FOLLOW','NAV_SS_COMMAND_AIM','NAV_SS_YAW_FIX')) {
    [Environment]::SetEnvironmentVariable($flag, '0', 'Process')
}
# After verifying the previous experiment processes have ended and preserving logs:
# Start-Process $py -ArgumentList @('-u','-m','nav.train_nav') ...
```

Never reset STOP_UPDATE to "current + 120" on a restart. At/above 29360 the trainer
exits before starting workers. The finite-run configuration defaults leave
ordinary training indefinite and in its original paths. Main is still running
its previously imported code; disk edits do not hot-reload existing processes.

## 4. Data, validation, and protected checkpoints

Original protected navigator file: models/nav/nav_v10_28897.pth. Despite the
filename, internal update is 28895. SHA256, rechecked 12:04:
539e2641319052460514c1838ebf1bc23956c5ea4cb314078b9a46bb9d738883.
The earlier endpoint remains nav_r2_stop_29665.pth, with another preserved copy
nav_before_points_20261005_29665.pth. Do not replace these or the seed files.

Original real replay data:

* datasets/ss_drill/starts_replay_dev.json: 102 starts, 77 source rounds.
* datasets/ss_drill/starts_replay_eval.json: 18 starts, 16 disjoint source rounds.
* Provenance: logs/nav/points_20261005/dataset.json.

Additional data was collected from two inference-only 72000-decision runs of the
protected navigator with deterministic preparation and forced EXISTING approvals.
The preparation controller is a coverage generator; its controls are NOT PPO
training labels. No success-based filtering. Recording requests both 24-decision
and four-decision lookbacks to retain fast entry states, with real velocity/spin.

Sources: logs/nav/live_starts_20261005_113252_10.jsonl (178 accepted) and
live_starts_20261005_113244_11.jsonl (234 accepted), 52 unsupported states rejected.
Provenance/hashes: logs/nav/goal175/fast_dataset_manifest.json.

In datasets/ss_drill/fast_20261005/:

| File | Starts | Source rounds | Use |
| --- | ---: | ---: | --- |
| starts_replay_dev.json | 241 | 30 | New training starts |
| starts_replay_eval.json | 171 | 18 | New held-out starts |
| train_dev.json | 343 | 107 | Original + new development only |
| benchmark_eval.json | 36 | 18 | Prespecified first candidate gate |
| new_all.json | 412 | 48 | Restoration audit input only |
| repeatability_dev.json | 20 | 20 | Late-state repeatability audit |

The 36-start benchmark was fixed BEFORE fitting: one long and one late start per
held-out source round. Do not reshuffle splits. The random group split produced
more evaluation records than usual; this is not a reason to move them into train.

All 412 new starts passed restore + 64 ms. Twenty late starts from distinct dev
rounds passed 40 full identical-policy windows: difference -0.10 points, SE 0.216,
equal falls, no failures. Reports fast_all_restore.json / fast_repeatability.json
in logs/nav/goal175. Script snapshots are not a complete engine savegame; small
near-edge variation remains. Report failures instead of substituting starts.

New ss_window_eval rows include split_group. replay.paired_summary now adds
round_grouped estimates (equal-weight source-round averages of paired differences)
when group labels are present. Use those for the 36-start benchmark: there are
18 independent source rounds, not 36 independent starts. Older reports retain
their old per-start summaries. A new regression test covers this distinction.

## 5. What has and has not worked today

Detailed experiments and file names: [GOAL_175_2026-10-05.md](GOAL_175_2026-10-05.md).
Raw reports/traces are in logs/nav/goal175/. Full-round arms use fresh independent
rounds; only restored-window arms are paired. These screens are not a final
confirmation set and have no correction for repeated checkpoint selection.

| Experiment | Enabled mean | Comparator mean | Rounds/arm | Conclusion |
| --- | ---: | ---: | ---: | --- |
| Learned update 28950 / same no-use | 158.56 | 161.44 | 16 | Worse |
| Learned update 28975 / protected forced-use | 159.88 | 160.94 | 16 | No gain |
| SS first-turn route / unchanged forced route | 159.00 | 160.13 | 8 | No gain |
| Preparation / ordinary forced approach | 158.63 | 158.50 | 8 | More fires, no gain |
| Preparation + velocity tracking / preparation | 152.50 | 158.13 | 8 | Worse, more falls |
| Learned update 29070 / same no-use | 161.19 | 160.06 | 16 | Inconclusive |
| Recent-acceleration command-aim probe | 155.75 | Prior preparation about 158 | 8 | No gain established |
| Optional yaw bridge fix / original bridge | 154.38 | 155.25 | 8 | No gain |

For update 29070, enabled minus no-use = +1.125, Welch 95% interval [-3.18, 5.43].
Seven actual fires across 16 enabled rounds, 16 falls in each arm. Best enabled
round 170, best no-use 169. Held-out original windows: learned minus no-use
+0.278 points, SE 0.300; forced minus no-use +0.167, SE 0.316. Not convincing.

Preparation raised uses from 4 to 32 across eight rounds without raising scores.
Do not optimize fire count alone. The velocity-tracking follow controller raised
falls from 10 to 28 and is rejected. The optional route/preparation/follow/aim/
yaw experiments all remain DEFAULT OFF and are absent from both training runs.

Immutable candidate from the clearest completed comparison:
models/nav/nav_goal175_20261005_1059_probe.pth, internal update 29070, SHA256
a2634ac8fed42b7acf21a9a7197a86b2b088bb7767c6c756e19190bf62f6999d.
Report: logs/nav/goal175/learned_29070_comparison.json.

## 6. Evaluation recipes after the separate learner finishes

First verify normal completion in its stdout and the checkpoint's internal update
29360. Preserve final under a unique immutable filename in models/nav/; this also
lets ss_goal_report resolve its basename for hashing. For example:

```powershell
# ONLY after confirming final save and trainer exit; refuse existing destination.
$candidate = 'models/nav/nav_goal175_fast_29360.pth'
if (Test-Path $candidate) { throw 'Candidate filename already exists' }
Copy-Item 'models/nav/ss_fast_20261005/nav_ss_fast_029360.pth' $candidate
```

Close only the recorded experiment games after checking their command lines
still contain the expected -aiport. Do not reuse a half-played experiment game
as a new evaluator: launch a fresh game after its Python listener is ready.

Window command (CPU-only environment, NAV_TOUR=walk; unused probe flags zero):

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:NAV_TOUR = 'walk'
$env:NAV_REAL_GEMS = '1'
& $py -u -m nav.ss_window_eval --ckpt models/nav/nav_goal175_fast_29360.pth --starts datasets/ss_drill/fast_20261005/benchmark_eval.json --branches base,no_use,force,learned --port 9970 --out logs/nav/goal175/fast_final_windows.json
```

That command waits for a game. Launch it as a hidden background Python process
with redirected stdout/stderr, record its PID, wait for LISTEN on 9970, then:

```powershell
$g = Start-Process $gameExe -WorkingDirectory $gameDir -ArgumentList @('-autotrain','KingOfTheMarble_Hunt','-aiport','9970','-offline') -WindowStyle Hidden -PassThru
```

On 9971, evaluate the experiment seed with --ckpt models/nav/ss_fast_20261005/seed.pth
and --branches no_use,learned using the SAME 36-start set and a different output.
This distinguishes improvement from the source from the value of firing within
the final policy. base means the protected navigator with uses disabled; no_use
means the evaluated checkpoint, including its learned residual, with uses disabled.
Keep all requested starts, failures, and group-level uncertainty visible.

For ordinary full rounds, use nav.real_run. No training is performed by it:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:NAV_CKPT = Join-Path $ml 'models/nav/nav_goal175_fast_29360.pth'
$env:NAV_PORT = '9970'
$env:NAV_ROUNDS = '16'
$env:NAV_TORCH_THREADS = '1'
$env:NAV_TOUR = 'walk'
$env:NAV_NO_USE = '0'
$env:NAV_FORCE_USE = '0'
$env:NAV_SAMPLE = '0'
$env:NAV_H_RESET = 'never'
$env:NAV_WATCH = '0'
$env:NAV_VIEW_SUBSTEPS = '1'
$env:NAV_TAG = 'fast_final_learned'
$env:NAV_TRACE = Join-Path $ml 'logs/nav/goal175/fast_final_learned.csv'
foreach ($flag in @('NAV_SS_ROUTE','NAV_SS_PREP','NAV_SS_FOLLOW','NAV_SS_COMMAND_AIM','NAV_SS_YAW_FIX')) {
    [Environment]::SetEnvironmentVariable($flag, '0', 'Process')
}
# Start hidden with unique redirected stdout/stderr; fresh game after LISTEN.
# & $py -u -m nav.real_run
```

Companion arm on 9971: same immutable checkpoint, NAV_NO_USE=1, different tag and
trace filename. A forced-use arm is diagnostic, not the learned policy. Compare:

```powershell
& $py -m nav.ss_goal_report logs/nav/goal175/fast_final_learned.csv.rounds.jsonl logs/nav/goal175/fast_final_no_use.csv.rounds.jsonl --out logs/nav/goal175/fast_final_comparison.json
```

real_run flushes one .csv.rounds.jsonl row per complete round. It records actual
engine_points, completed, fires, kick velocities, and experiment flags.
ss_goal_report requires >175 engine points plus actual use and velocity impulse
to mark a peak; it also reports the independent-round mean comparison. Preserve
fresh confirmation rounds for any selected winner, and compare with the protected
base so ordinary driving damage is not hidden by an internal no-use comparison.

## 7. Most recent unfinished investigation: spin-aware floor prediction

Files added immediately before this handoff:

* nav/ss_floor_model.py: scalar step and vectorized step_batch.
* nav/test_ss_floor_model.py: batch/scalar equivalence and sliding momentum test.
* logs/nav/goal175/floor_model_calibration.json.
* logs/nav/goal175/floor_model_postkick_rollout.json.

This module is NOT imported by real_run, train_nav, or vec_worker. No controller,
aim optimizer, or policy-learning integration has been implemented. No game was
launched using it. Both training runs are completely unaffected.

Physical motivation: Super Speed adds linear velocity without rotating the spin
to match. Friction acts along contact slip (velocity minus rolling velocity),
not necessarily along current velocity. An aim that aligns the instantaneous
result with the gem can still bend away during the following half-second. A
constant 14 u/s^2 brake along velocity misses this mechanism. The earlier follow
controller ignored spin and failed its score test.

Equations were read directly from the local engine, not fitted to a map:
C:/Users/doug/src/OpenPQ-TGEMIT-mbx/engine/game/marble/marble.cc,
Marble::computeMoveForces / applyContactForces / advancePhysics. Defaults come
from DefaultMarble in platinum/server/scripts/marble.cs. The predictor is LIMITED
to ordinary, stationary, horizontal uninterrupted floor: radius .18975, gravity
20, friction coefficients .7/1.1, angular acceleration 75, max roll 15, 8 ms
substeps. It has no collision/edge/jump/slope/active-powerup handling. Camera yaw
and per-axis input saturation matter. Do not apply it to unsupported surfaces.

Offline checks used two development probe traces (no fitting):
20261005_111202_prepare_command_aim.csv and 20261005_112158_prepare_yaw_old.csv.
Only consecutive horizontal-floor transitions without jumps/falls were included.

* 3318 transitions, mean speed 13.43: one-step velocity error mean .0482 u/s,
  p90 .0802; position error mean .00574 units. A one-row action alignment shift
  raised velocity error to .38-.43, so the pending-action timing is material.
* Recorded-command post-kick rollouts: .256 s, 62 eligible / 1 excluded, mean
  position error .0074; .512 s, 62 / 1, mean .0281; 1.024 s, 60 / 3, mean .376;
  1.536 s, 60 / 3, mean .875. Camera yaw was reconstructed from commands/hold,
  not read from engine telemetry. The long-horizon error is too large to assume
  exact gem intersections. Excluded ramps/falls are outside this model's domain.

If pursuing this, start with a DEVELOPMENT-only short forecast or closed-loop
probe, repeatedly correct from observed position/velocity/spin, and require
supported floor/contact evidence. Validate predicted trajectories in the engine
before changing fire approvals. Keep all old approvals/fallbacks initially.
Do not repeat the failed recent-acceleration predictor under a new name. The
existing learned step3 ensemble had .50 u/s mean pending velocity error versus
.91 for constant velocity on 33 measured fires; its high-speed long rollouts
were already known to be unreliable.

Timing details for reproducing analysis:

* Trace row dec=k contains state C_k AFTER action A_(k-1) was sent; its
  spin_before belongs to C_(k-1). Obtain spin at C_k from row k+1.
* To predict C_k -> C_(k+1), the pending joystick is recorded on row k, not row
  k+1. The correct action offset was established by the above transition check.
* In an example, first use was sent at decision 287 (row 288); the kick appeared
  on C_289 -> C_290. Its direction matched command aim on row 289. The last aim
  reported in the fire event (row 290's command) was already too late.
* Measured net kick magnitude in 32 development fires was 24.104 +/- .0001 u/s,
  consistent with 25 minus one 64 ms friction interval. Do not confuse that
  observed delta with the impulse applied at the fire tick.

## 8. Implementation and operational traps

The earlier replay/points implementation is described in
[HANDOFF_POINTS_REPLAY_2026-10-04.md](HANDOFF_POINTS_REPLAY_2026-10-04.md).
Main startup/recovery: [RUN_POINTS_2026-10-05.md](RUN_POINTS_2026-10-05.md).
Today's history: docs/HANDOFF_NAV_TRAINING_LOG.md sections 40.49-40.50 and
GOAL_175_2026-10-05.md. Older overnight docs contain historical recipes and rules;
do not restart their old synthetic/shaped-reward runs.

Recent implementation additions:

* trainingReplay.cs provides schema-2 REPLAYCAPTURE/RESTORE, item timers/RNG,
  real score/clock/inventory/pose/spin, and unsupported-state rejection. A pending
  powerup yaw hold is not a supported fixture. Old synthetic drill files are
  NOT interchangeable with real snapshots.
* env.py / real_run.py / ss_goal_report.py provide authoritative round proof.
* ss_live_smoke --prepare --force-use --deterministic --lookbacks 24,4 was used
  only for broader state collection. NAV_LIVE_LOOKBACKS elsewhere defaults to 24.
* train_nav supports NAV_PORT0, NAV_LOG_DIR, NAV_OUTPUT_DIR, NAV_STOP_UPDATE,
  NAV_TORCH_THREADS for the isolated finite experiment. Defaults preserve main.
* ss_routes, ss_prepare, ss_follow, ss_command_aim are opt-in diagnostic modules.
* mlAgent.cs SSYAWFIX is local-lockstep-only and excluded from live mode. It
  releases yaw hold on consumption and uniformly scales saturated camera axes.
  Its first eight-round gate showed no gain and predated explicit ACK telemetry.
  Subsequent evaluators require DEBUG|ssyawfix=1 and report release counts; this
  newer ACK path has NOT yet been game-tested. Keep it off unless testing that.

PowerShell traps:

* Set $ErrorActionPreference='Stop'. A failed manifest parse once accidentally
  launched an evaluator against an old game. That attempt was killed and marked
  logs/nav/goal175/20261005_110655_no_use.invalid.json. Valid replacement is
  20261005_110759_no_use. Reports reject marked-invalid attempts.
* Some old array manifests are wrapped as {value:[...],Count:2}. After
  ConvertFrom-Json, unwrap .value if present. Write newly constructed row objects,
  not a reserialized wrapped object. Do not infer a port is free from a stale PID.
* Fresh evaluator listener first, fresh offline game second. Record BOTH PIDs.
  Close a game only after its evaluator exits and its command line verifies the
  expected port. The main supervisor owns 8888-8895; leave these alone.
* Engine console.log is stale and not evidence that a current option was enabled.
  Use protocol acknowledgments and current output files.
* PowerShell execution policy is restricted. For repo helper scripts use
  powershell -NoProfile -ExecutionPolicy Bypass -File ...; do not change policy.
* Processes launched before code edits keep old imports/scripts. New files on
  disk do not establish that a running process used them.
* There are many pre-existing/uncommitted changes and continuously written logs.
  Do not reset/revert the working tree. The operator handles commits.

Final verification at 12:04:

```powershell
& $py -m unittest nav.test_replay nav.test_ss_routes nav.test_ss_floor_model -q
# Ran 22 tests: OK. No game connections or checkpoint writes.
```

The next report should distinguish: complete-round score evidence, matched
held-out window evidence, training diagnostics, and untested hypotheses. Keep
the operator informed of the recent main-run regression and the finite
experiment's outcome. The goal remains active and unachieved.
