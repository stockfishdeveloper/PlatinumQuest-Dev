# Real-map training preparation — 2026-10-09

**Repair follow-up, 2026-10-10:** reconnect fixes, explicit reward modes/metadata,
59 passing tests, and the powerup practice discussion are recorded in the
[follow-up handoff](HANDOFF_POWERUP_PRACTICE_PREPARATION_2026-10-10.md).
No new training was launched for that follow-up.

**Later state, 2026-10-10:** overnight training changed the use rules, starting
checkpoint, jump scaffolding, reward, and approval setting described below. This
document records the original preparation. See the independent
[overnight audit](OVERNIGHT_TRAINING_AUDIT_2026-10-10.md) for verified A2/B2 results,
remaining reconnect failures, shaping-accounting concerns, and the successful
14-start Sprawl replay probe. No new training was launched during that audit.

## Authorization and state

The operator requested implementation of the reviewed plan up to, but excluding,
training. Prepare and verify routing, escape-powerup exploration, legal delayed
rescue, uniform map rotation, and checkpoint baselines. Long-term training requires
the operator's review and approval. No game training or checkpoint optimization is
authorized. Existing synthetic PPO unit tests are allowed as validation.

Starting commit: `4ed9089b7` (Right before phase 5). Working tree was clean.
Starting checkpoints: `nav_latest.pth` (37575) and
`nav_points_20261005_parity_32565.pth` (166.55 KOTM base).
Existing cluster supervisor STOP file remains in place. No trainer was running.

Final state: no game instances, baseline harnesses, or trainers remain running.
Both source checkpoint SHA-256 hashes still match the baseline report. Preparation
selects cluster checkpoint 37575; `training_approved=false` and both STOP files remain
in place. No isolated training seed or trained checkpoint has been created. Only
the dashboard remains running (PID 33732, http://localhost:8990).

## Implementation record

### Routing and terrain

Added `nav/routing.py`, shared by `vec_worker.py` and `real_run.py`. Both now default
to routed observations (`NAV_DIRECT_GEM=0`). Fields include destination height;
distances, goal changes, floor tests, and next-goal fields retain source height.
Routes follow the directed multilevel graph, then string-pull with a destination
floor check. A clear XY line under an upper deck cannot skip its ramp. Waypoints
are held until within 2.5 u or no longer visible on the correct floor.

`obs.py` keeps the true gem and next gem separate from the navigation waypoint.
Super Speed pickup/redirect approval is disabled at a route corner; a corner is
not an invented gem. Observation/checkpoint dimensions have not changed.
`waypoints.py` carries height through every field call. Replay histories retain
both navigation and gem targets; restored stage-4 windows use shared routing.

The terrain recorder and multilevel graph work were already in the starting
commit. This task does not regenerate those ten maps. `terrain.py` only fixes an
unclosed mission-file handle. Geometry limitations below still apply.

Supporting edits: `nav/protocol.py` recognizes the new ACK messages;
`nav/env.py` waits for delayed ACKs and clears prior-connection metadata.
`nav/test_replay.py` updates its flat-terrain test stub to accept optional height.

### Powerups and practice recordings

`NAV_ESCAPE_POWERUPS=1` opens the existing use head to held Super Jump and charged
blast at a nearby rise or elevated route target while slow/on the floor. Existing
moving-gap branches remain available. Worker/evaluation dispatch now sends the
corresponding ordinary use key. Super Speed uses its existing aiming behavior.
There is no granted inventory, forced successful escape, Helicopter lesson, new
head, or weight migration. The existing differentiable use-exploration mixture is
scored by PPO; deterministic evaluation uses the learned part of that head.

Added `nav/escape_practice.py`. Assisted baselines record supported schema-2 starts
from normal inventory and game state, with up to 32 warm-up observations. Starts
are deduplicated and split by source round (paired checkpoint runs of the same
map/round seed share a split); only the ten training maps are allowed
into the practice set. Missing/unsupported states and round tails are rejected.
Stage-4 loading now filters fixtures by the connected mission, and a map without
fixtures falls back to ordinary rounds. Practice is prepared but **disabled** in
the initial training launch (`NAV_DRILL_PLAN=none`). A later reviewed practice run
must not take all full-round workers away or silently substitute another map.

### Recovery and telemetry

Added `nav/recovery.py`. A pickup or at least 1 u of cumulative improvement toward
the same gem resets the progress timer. A gem-chooser change alone does not.
Countdown time is excluded from the progress/rescue window in workers and evaluation.
After 8 seconds of play without progress, a marble with gems available can
request the normal human `serverCmdQuickRespawn` in noncompetitive mode. Falls/OOB
retain the existing separate OOB-click path. Rejected requests wait another
window; the bridge never bypasses competitive mode or the game's cooldown.

`mlAgent.cs` adds acknowledged `GAMERULES`/`QUICKRESPAWN` controls and retains the
chosen competitive preference across rounds. Offline Hunt omitted the initial
quick-respawn permission given by multiplayer's on-client-enter path: the
autostart hook now initializes that same first permission on offline rounds.
Subsequent cooldowns are still set by the ordinary server command.

Rescue time is included in the simulation-clock transition duration used for PPO
discounting. Rescue resets observation/GRU/use history, but does not finish or
restart the round. Rescue is an automatic environment controller, not a newly
learned actor action. No extra rescue reward penalty was added. Current real-round
reward is game points; legacy `FALL`/shaping constants are not an additional
penalty in this mode. Pickups during delayed control acknowledgements are retained.

Logs and baseline rows include rescues, attempts, stalled seconds, actual SJ/blast
uses, progress within 8 s of use, failed uses, and engine group completions.
`powerup_escapes` means **progress observed after use**, not proven causal escape.
Stalled seconds means time at least 1 s past the last meaningful progress.
`huntGems.cs`/`trainingReplay.cs` provide actual fully-cleared-group counts;
competitive timer respawns are not counted as clears. `dashboard_nav.py` parses
the new telemetry and adds the selected-map recovery chart. Old KOTM-specific
score filters were removed so longer real-map rounds remain visible. Checkpoint
discovery includes isolated run directories, so future saves appear on the dashboard.

### Rotation and guarded launch

`configs/real_maps_20261009.json` defines the ten maps, both checkpoints and profiles.
`run_game_loop.ps1` rotates an instance after four completed rounds, using atomic
worker state files. Advancing by eight through ten assignments gives equal round
coverage of all maps over a complete rotation cycle. Rotation restarts only the
owned game on its port, and stale files from exited workers are ignored. Reconnected
workers reload map, rules, and replay telemetry.
One to eight game instances are permitted; hidden windows remain the default.

`prepare_real_maps.ps1` defaults to preparation only: it writes `prepared.json`
and a STOP file, and launches no trainer/game. `-StartTraining` also requires
`training_approved=true` in the reviewed config. The future run snapshots its
source and saves/resumes in `models/nav/<RunTag>`; it cannot overwrite either
comparison checkpoint. The supervisor supports a finite 120-minute block and
leaves STOP in place afterward (polling may add up to about 30 s).

Future launch settings: eight full-round workers, routed observations, normal
SJ/blast permissions, noncompetitive rescue at 8 s, 64 ms decisions, 3x engine,
uniform map rotation, exact-likelihood 1% jump exploration, and **zero off-policy
forced-jump probability**. The old global 3% forced-stall default is retained for
legacy configurations, but explicitly disabled in this run and every baseline.
The model may lose scaffolded stall jumping; that must be measured, not hidden.

`nav/phase_r.py` is a finite read-only baseline harness: two checkpoints x ten
maps x two profiles x four rounds = 160 rounds, maximum eight games. Python
warms the model/terrain before launching each game. Settings pin deterministic
actions, zero forced jumps/exploration/stuck detours, routed goals, normal escape
powerups and 64 ms decisions. Each checkpoint gets the same per-map round seeds.
Every completed round must reconcile accumulated points with the final engine
score. SHA-256 verifies checkpoint immutability. It never invokes the trainer.
Large traces/streams and replay starts stay under `%TEMP%/pq_checks`.

`review_real_maps.ps1` provides one finite 1x round for operator review with four
16 ms display slices per 64 ms decision and the existing timing delay. It warms
Python first, cleans up owned processes, and uses `-ShowWindow` only when asked.
`nav/preflight_r.py` runs a finite diagnostic worker/game probe without a model or
optimizer. Its diagnostic teleports are explicit and are not part of baselines.

## Validation and artifacts

* 42 focused/existing unit tests passed:
  `python -m unittest nav.test_phase_r nav.test_replay nav.test_ss_routes nav.test_ss_floor_model -q`.
  The existing replay suite includes small synthetic PPO/critic tests, not a live
  training run. New checks cover Cragmire heights/routes, true-gem preservation,
  escape masks, progress timing, delayed ACK reward accounting, competitive gates,
  terminal recovery accounting, dashboard fields/high scores, paired practice splits
  exclusion of held-out maps, recording armed special blast with empty regular charge,
  and dashboard discovery of checkpoints in isolated run directories.
* All four PowerShell scripts parse; preparation and approval-rejection paths ran
  without launching training. Rotation dry-run: every map appears eight times in
  ten eight-instance assignment waves. Dashboard JavaScript passes `node --check`.
  Restarted the existing dashboard (initial PID 29328, final PID 33732); HTTP 200 at
  `http://localhost:8990/` serves the recovery chart. Streams and restart metadata
  are in `%TEMP%/pq_checks/phase_r_dashboard*`. No training process was restarted.
* Cragmire one-round smoke: completed engine/accumulated score 19 points, checkpoint
  unchanged. This was a plumbing check, not a selection baseline.
* Actual engine-speed-1 worker probe (hidden, unpaced diagnostic): low-deck distance 0 versus upper-deck 34.6569 u from
  `(1,-2,13.18975)`, reachable ramp waypoint `(7.25,-3.75,14)`. Stationary worker
  rescued after 8.192 game seconds; transition duration 3 x 64 ms, reward 0,
  `new_segment=true`, `done=false`. Cooldown, Python competitive gate, and a direct
  engine competitive challenge refused rescue. Normal map SJ pickup/use and
  naturally charged blast dispatch were exercised. Strengthened probe: real
  SuperJumpItem at `(-17,-12,12.1775)` was picked normally; isolated Super Jump
  vertical speed peaked at 18.608 u/s, rise 9.79968 u, no fall. Naturally charged
  blast fell from 0.524799 to 0 and lifted the marble 1.24030 u. Reconnecting Cragmire -> Sprawl reloaded
  the mission, noncompetitive rules, world telemetry, reset map-round count to 0,
  and reset the recurrent segment while discarding the disconnected transition.
  The latest probe forced on_floor=false during the stationary wait and still
  rescued after 8.192 play seconds; 28 countdown transitions left the progress
  clock unstarted. A captured Bowl blast start restored successfully, checking
  catalog, pose, inventory, items, RNG, round clock and score; 32 warm observations
  built successfully. Artifact copied to [REAL_MAP_PREFLIGHT_2026-10-09.json](REAL_MAP_PREFLIGHT_2026-10-09.json).
* Final full baseline sweep completed **40/40 jobs, 160/160 rounds** in 2399.6 s
  (about 40 minutes), every score reconciled, source checkpoints unchanged.
  Artifacts: `%TEMP%/pq_checks/phase_r_baseline_verified_20261009`.
  Persistent [comparison tables](REAL_MAP_BASELINES_2026-10-09.md) and
  [complete summary](REAL_MAP_BASELINES_2026-10-09.json) include metrics and hashes.
* Recorded 224 normally acquired escape starts: 165 development and 59 evaluation,
  grouped by paired source round. Files: `escape_practice/starts_dev.json` and
  `escape_practice/starts_eval.json` inside the verified baseline artifact directory.
  Preserve these TEMP artifacts before cleanup. No practice optimizer was run.

Failures found and corrected: floating-point equality assertions; missing delayed
rules ACK wait; uninitialized trace aim for SJ/blast; final-frame score/time omission;
offline first-respawn permission; a terrain-derived on-floor rescue gate blocking
motionless wedged marbles (Sprawl trace showed 1,000 consecutive stopped decisions
with on_floor=0; removed that gate and repeated affected assisted baselines);
replay fixture goal accidentally becoming a route
corner; reconnect retaining old map telemetry. An initial live probe also had a
missing diagnostic `next_goal` argument, fixed before the passing probe. The first
baseline sweep was stopped because first-round rescues were refused; a second
was stopped after discovering the on-floor gate. Their partial artifacts in
`phase_r_baseline_20261009` and `phase_r_baseline_final_20261009` are **superseded**.
The final harness reuses freed slots as jobs finish instead of waiting for all
long maps in a wave, while retaining the eight-game cap. Smoke
failure directories `_smoke_20261009` and `_b` are retained as evidence. Windows
execution policy required process-local Bypass for script validation; no persistent
policy setting was changed. Deleted stale compiled `.cs.dso` caches for the four
edited Torque scripts so the engine loaded their current sources.
The first diagnostic Super Jump loop also pressed use after the jump was spent,
which could add blast; the final probe separates the resources and reports the
isolated impulse above. Late collector-history clearing only changes future
recordings; it does not alter the baseline policy's actions or scores.

Preparation state: `training_approved=false`; STOP present in both the new prepared
run and the old cluster run. No live trainer/optimizer/checkpoint save was started.

## Commands for review and later use

From `ml_agent` with the existing Python environment:

```powershell
# Safe preparation; no game or training launch
powershell -NoProfile -ExecutionPolicy Bypass -File .\prepare_real_maps.ps1 -Checkpoint clusters
python -m nav.phase_r                       # print configuration only
python -m nav.phase_r --baseline --rounds 4 --instances 8

# One visible 1x round, no training (change checkpoint/profile/map for comparison)
powershell -NoProfile -ExecutionPolicy Bypass -File .\review_real_maps.ps1 -Checkpoint clusters -Mission Cragmire_Hunt -Profile assisted -ShowWindow

# AFTER explicit operator approval and reviewed training_approved=true:
powershell -NoProfile -ExecutionPolicy Bypass -File .\prepare_real_maps.ps1 -Checkpoint clusters -StartTraining
# Stop the future run by creating logs/nav/real_maps_20261009/STOP.
```

Use the explicit Python 3.9 path recorded in `prepare_real_maps.ps1` if `python`
does not select the project environment. Choose a fresh RunTag when changing the
starting checkpoint. Do not remove STOP or change approval during this task.

## Review decisions

**Recommendation: start the reviewed first block from cluster checkpoint 37575.**
It won 7/10 assisted maps (one tie) and 8/10 competitive maps. Equal-map raw-point
means are 30.73 versus 20.75 assisted, and 27.03 versus 18.83 competitive. These
are descriptive four-round samples, not par-normalized scores or proof of significance.
Fractured Islands is the clearest transfer evidence: competitive 69.0 versus 0.25;
Treasure Box is 15.0 versus 0.0. The learned cluster skill was not confined to its
original distribution, so discarding that checkpoint just because the map pool
changed would throw away useful behavior.

The tradeoff is real: competitive KOTM is 132.25 versus the base's 148.5, and Cube
Isle 7.75 versus 17.25. Preserve the base as a fallback and judge retention against
these **fresh routed** baselines. This task did not undo routing to recover the
historical direct-gem 166.55. The prepared run/default review scripts now select
clusters, while the operator can still choose base explicitly. Selection is a
recommendation; training approval remains false.

Confirm behavior with the wall-clock-paced 1x review script and approve actual
training separately. Score each map/profile against its fresh own
baseline; do not compare assisted training scores directly with competitive lobby
scores or the historical direct-gem KOTM 166.55. Watch falls, stalls, rescue reliance,
group clears and learned/forced jumps as well as points. If rescues persist, inspect
route quality and missing escape skills before increasing a penalty.

ExampleMission and BlockClustersHoldout are regression/development maps. They have
been repeatedly inspected; they are not untouched tests. Horizon stays excluded
from training. Tier-held-out selection remains an operator decision. Complex maps
may still require a separately reviewed Helicopter/other-powerup curriculum and
graph links for those abilities. Existing routing graphs cover ordinary walk/jump
links; newly permitting SJ/blast does not create boosted route links or teach use.

Terrain remains static height geometry: moving platforms, bumpers, gravity changes,
and some obstruction details are absent. A failed/unreachable route retains the
legacy direct fallback and will rely on learned behavior/rescue; it is not proof of
reachability. The policy's observation target changed from gem to waypoint, so even
a good old checkpoint can initially regress and needs matched evaluation.

The terrain-derived on-floor observation can miss a stationary marble wedged
against a slab: the sampled Sprawl point `(-4.11,-12.63,4.67)` had estimated floor
5.33. Rescue now works independently of that estimate. Jump/powerup permission
still uses the existing floor estimate; those missed contacts need review before
claiming that powerups alone solve every trap. The assisted profile is a temporary
learning aid; it does not establish competitive/lobby readiness.

Replay warm-up is a 32-observation approximation, not a stored actor hidden-state
snapshot. The collector now clears that history at falls/rescues, and the worker
clears approach history at rescue. Baseline-collected starts predate the collector
history-clear follow-up; review their burn-in before a fair replay-policy comparison.
Their physical game snapshots are separate from this limitation. Initial training
has replay practice disabled.
