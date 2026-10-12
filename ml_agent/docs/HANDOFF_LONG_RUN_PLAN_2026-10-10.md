# Handoff: the long real-map training run (plan, commands, decision rules) — 2026-10-10

For the agent who runs this. Read it whole before touching anything. Everything here was true at 19:00 on 2026-10-10.

## 1. What you are being asked to do

Run ONE long, continuous training block on the 10 real hunt maps from checkpoint A2, with the A2 settings, for 12 to
24 hours, evaluating every two hours with the same clean sweep, and report on two-hour trends. Do not run short
experiments. Do not change rewards, actions, exploration floors, routes or game rules during the run. The operator
decides when it ends.

Nothing starts until the operator says "go". The config flag `training_approved` is true from the overnight blocks;
that is not authorization for this run. Ask, in plain words, then start.

## 2. Operator rules that apply to you (standing, non-negotiable)

* Never edit code without the operator's approval; present options and wait. Bug fixes needed to keep the run alive
  are the exception, and every one goes into the dated log (section 8).
* No training start or restart without a brief non-technical plan and the operator's OK.
* Hard cap 8 game instances in total (training + evaluations). Training uses all 8; an evaluation needs training
  paused or finished. Never kill a game you did not start.
* Every time estimate in the operator's wall-clock time, measured, with a finish time of day. Code edits take you
  minutes; say minutes.
* Any game launched for the operator to watch gets `-aifreecam`, every time (review_real_maps.ps1 does this).
* Reward = the game's rules plus the ONE approved shaping term (section 4). No map-specific rules, no geometry
  gates, no per-situation powerup gates (the operator removed those on purpose).
* Do not touch OneDrive. The repo lives under OneDrive; write repo files via temp file + `os.replace`, and keep runtime
  state (rotation files, stdout/stderr, traces) under `%TEMP%\pq_checks`.
* Announce every result, including "it produced nothing". Report at milestones; do not ask small questions.
* Banned phrasings: the em dash; "drift"; "the one thing"; "this is X, not Y".

## 3. State you are starting from

* Best checkpoint: `ml_agent/models/nav/real_maps_20261010_A2/nav_latest.pth`, update 37760, SHA-256 begins
  `ae88f2e70a376fda`. Registered in `configs/real_maps_20261009.json` as `A2`. Protected: never overwrite it; a run
  writes into its own `models/nav/<RunTag>/` folder.
* Reference numbers for A2 (4 rounds x 10 maps, assisted profile, deterministic, exploration 0): ten-map mean 34.4,
  falls 22.3 a round, KOTM 122.0 (measured 18:09 by the autoloop; artifact
  `logs/nav/powerup_autoloop_20261010_r3/reference_A2_four_rounds.*`). Per map at 2 rounds: KOTM 124.5, Sprawl 25.5,
  Cube Isle 11, Bowl 3, Duplex 28.5, Fractured Islands 91, Cragmire 4.5, Vortex Effect 0, Treasure Box 34,
  Concentric 9. Four-round numbers carry standard errors of 5 to 15 points on the volatile maps; never decide on a
  single sweep.
* ExampleMission (never trained on) is the generalisation check: the morning-of-10-09 model scored 25.3 there with
  38 % of decisions stalled (log 40.73). No A2 number exists yet; take one before the run starts.
* Checkpoints are not tracked in git (`*.pth` ignored); which file is which lives in `ml_agent/models/CHECKPOINTS.md`.
  Add a row there when you save a milestone.
* Nothing is running. STOP files are present in every run folder under `logs/nav/real_maps_*`.

## 4. The settings (what A2 was trained with; keep them)

All set by `prepare_real_maps.ps1` from `configs/real_maps_20261009.json`:

* 10 maps: KOTM, Sprawl, Cube Isle, Bowl, Duplex, Fractured Islands, Cragmire, Vortex Effect, Treasure Box,
  Concentric. 8 game instances rotating through them every 4 completed rounds.
* Non-competitive Hunt; stuck rescue after 8 s without route progress via the game's own quick respawn
  (`NAV_RESCUE_S=8`), no penalty beyond the time lost. The game locks quick respawn for 25 s after any respawn,
  including the fall recovery's click; refused rescues retry every 8 s. That is the game's rule, leave it.
* Routed observations (`NAV_DIRECT_GEM=0`): the policy steers at a waypoint on the planner's height-aware route; the
  gem stays the target for pickup, next-gem and Super Speed logic.
* Every powerup usable at any time, blast whenever charged, exploration floor `USE_EPS` 2 % (nav/model.py).
* Forced stall jump samples 3 % (`NAV_JUMP_FORCE_P=0.03`) with their own 3 u drop clearance; jump exploration 1 %.
* Progress shaping: `NAV_POTENTIAL_K=0.05`, `NAV_SHAPING_MODE=legacy` (undiscounted difference of -route distance,
  no term on gem change, settled at round end, frozen over void). The other agent added an exact `discounted` mode;
  it is NOT the A2 setting. Do not switch modes mid-run.
* 64 ms decisions, 8 instances, `NAV_SEED` 1.

## 5. Commands

From `C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent`, Python 3.9 at
`C:\Users\doug\AppData\Local\Programs\Python\Python39\python.exe`.

Before the run (no games may be running; each sweep is ~9 minutes with 8 games):

```powershell
# ExampleMission reference for A2 (held-out map), 4 rounds, assisted profile
python -m nav.phase_r --baseline --checkpoints A2 --maps ExampleMission --profiles assisted --rounds 4 --instances 4
```

If the harness refuses a map outside the config list, add `"ExampleMission"` to a new config key and use
`--checkpoint-file` with the A2 path; do not add ExampleMission to `maps` (that would train on it).

Start the run (after the operator's go). RunTag must be new. 24 h = 1440 minutes; the operator may stop earlier:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\prepare_real_maps.ps1 -Checkpoint A2 -RunTag real_maps_20261010_long -StartTraining -ShapingMode legacy -MaxWallMinutes 1440
```

It copies A2 into `models/nav/real_maps_20261010_long/starting_checkpoint.pth`, writes `prepared.json` (check
POTENTIAL_K 0.05, FORCE_P 0.03, RESCUE_S 8, COMPETITIVE 0, DRILL none), removes the STOP file and runs
`supervise_points_training.ps1`, which starts the trainer and the game loop and restarts them on a crash or a hang
(stall rule 240 s without trainer output). The launcher blocks for the whole run: start it with `Start-Process`
(hidden) and redirect its stdout/stderr under `%TEMP%\pq_checks\real_maps_20261010_long\`.

Two-hour check (the supervisor keeps training; a sweep needs the 8 game slots, so pause first):

```powershell
New-Item -ItemType File logs\nav\real_maps_20261010_long\STOP -Force   # supervisor ends within ~30 s, closes its games
python -m nav.phase_r --baseline --checkpoint-file models\nav\real_maps_20261010_long\nav_latest.pth --profiles assisted --rounds 4 --instances 8
# then resume: same prepare command with the same RunTag; it resumes from the run's own nav_latest (the hash check
# compares the starting snapshot only). Confirm "resume point ... further along" or the update number in logs/nav/stdout.txt.
```

A pause plus sweep costs about 12 minutes of wall clock per check. Every third check, also run the ExampleMission
sweep (4 instances, ~4 minutes).

Stop for good: create the STOP file and wait for "Supervisor ended" in `logs/nav/<RunTag>/supervisor.log`.

Watch the operator a round at 1x (free camera) on any map:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\review_real_maps.ps1 -Checkpoint A2 -Mission Cragmire_Hunt -Profile assisted -ShowWindow
```

(`-Checkpoint` takes a config key; register the run's nav_latest under a key first, or copy the script's env block.)

## 6. What to watch while it runs

* `logs/nav/stdout.txt`: `NAV upd=` lines (expect one every 25 to 40 s), `GAME map=... points= ... rescues= ...
  shape=` per finished round, `RESCUE ... game_reply=` lines, `WARNING` lines (non-fatal reconnect waits).
* `logs/nav/<RunTag>/supervisor.log`: "Attempt N" lines are supervisor restarts; "Run stalled" is the hang rule.
  "trainer crashed; restarting" in stdout is the trainer's internal restart; it must resume from the run's own latest
  save (fixed 10-10 02:30; the line "resume point: ... is further along" confirms it).
* Updates per hour. The overnight blocks did 60 to 90. Below 40 for an hour means something is wrong: look for games
  stuck in the level select (`-autotrain` could not find the mission; screenshot the window, no console.log exists),
  workers looping on reconnects after a map rotation (count "waiting for the game"), or the trainer hung.
* Internal restart causes live in `logs/nav/stderr.txt` (overwritten per supervisor attempt; older copies under
  `%TEMP%\pq_checks\<RunTag>\*_stderr.txt`). The known ones (no RULES ack, no replay telemetry, no QUICKRESPAWN ack)
  are non-fatal now; anything new goes in the log with its traceback.
* The dashboard at http://localhost:8990 reads these files; its falls card double-counts on full rounds (halve it).

## 7. Decision rules for the two-hour checks

Report each check as a table: update, ten-map mean, KOTM, falls, rescues a round, ExampleMission when run, learned
powerup uses a round by type (from `pow_uses` in the sweep's `*.rounds.jsonl`), against the previous check and the
A2 reference (34.4 / 122.0).

* Trend, not level: three consecutive checks (6 h) flat or down on the ten-map mean while KOTM holds is a plateau;
  tell the operator, keep running unless told otherwise. Never stop a run on your own for a metric.
* KOTM below 100 on two consecutive checks with rising powerup uses means the use head is relearning "fire
  everything"; stop the run and report (that is the failure mode seen in the points-only block B2).
* Falls a round doubling against the reference with points flat is the same signal.
* ExampleMission is the generalisation number: report its points and its stalled share; a rise there matters more
  than a rise on a training map.
* Do not promote checkpoints by yourself. Name the best two-hour save in the report; the operator decides.

## 8. Logging

Append to `ml_agent/docs/HANDOFF_NAV_TRAINING_LOG.md` as numbered sections (last one is 40.85): every start, pause,
resume, check table, bug and fix, with times. The previous agent's own logs are
`docs/HANDOFF_POWERUP_LOOP_2026-10-10.md` (the afternoon's practice loop; its conclusion: practice found no useful
situations, short blocks measure noise, the 40.4 "best" was a two-round fluke) and
`docs/OVERNIGHT_TRAINING_AUDIT_2026-10-10.md` (independent audit of A2 vs B2).

## 9. Known traps

* OneDrive: in-place writes to repo files fail with Errno 22 or PermissionError; use temp + replace; keep state out
  of the repo. Never pause, quit or reconfigure OneDrive (operator).
* After editing any `.cs` game script, delete the matching `.dso` or the game loads the old code.
* The checkpoint's `update` counter is global (37760 at A2); the run's own saves are `nav_<RunTag>_<update>.pth`
  every 25 updates plus `nav_latest.pth` every 5.
* `phase_r` refuses to start while any game runs. The review script seeds nothing by default; only the baseline
  harness seeds rounds (paired comparisons).
* Falls do NOT lose a held powerup (the game respawns you with it). A held slot reading 0 during a respawn is a blip.
* The practice-window fixtures (`escape_starts_*.json`, `powerup_curriculum_*`) are parked: their windows showed
  activation scoring below saving almost everywhere; do not enable `-DrillPlan` for this run.

## 10. What comes after (not for you to start)

Powerup learning as its own step once the long run shows situations where a use pays; the exact discounted shaping as
an A/B against legacy at equal updates; throughput work (games run at ~2x real time, trainer at 60 to 90 updates an
hour); widening the map pool when ExampleMission and the held-out trend stall while training maps rise.
