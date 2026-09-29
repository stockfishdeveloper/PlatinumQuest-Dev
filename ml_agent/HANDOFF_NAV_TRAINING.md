# Navigator training: handoff

The CURRENT state of the King of the Marble (KOTM) navigator: what exists, what it scores, what is
wrong with it, how to run it, and what comes next. Keep this file current: when something changes, edit
it here. The dated history of every experiment (sections 1 to 28.x, 2026-09-17 to 2026-09-26) is in
`docs/HANDOFF_NAV_TRAINING_LOG.md`. Code comments that cite "HANDOFF 28.52" and the like point into that
log, and new dated entries go there too.

Last updated: 2026-09-29 06:10 (jump physics stage 5: whole rounds work, single-gem check not met).

## 1. Where things stand

* **Goal:** a 170-level KOTM round, the level of the human's best rounds. The earlier goal of 150 was
  reached on 2026-09-26.
* **Model:** `models/nav/nav_latest.pth` = `nav_v6_start_26110.pth`, which is 26113 (the best checkpoint,
  154.0 at 3x) converted to observation V6 (with the marble's spin). The operator made it the checkpoint to
  train from on 2026-09-26 22:50. The V6 run's endpoint is kept as `nav_v6_end_26595.pth` (150.0).
* **Verified KOTM scores** (3x, 8 deterministic rounds each, `nav/real_run.py`, greedy gem chooser):

  | checkpoint | points | falls/round | note |
  |---|---|---|---|
  | nav_night_26113_142 (V5) | 153.1 | 1.6 | the best checkpoint; converted to V6 as nav_v6_start_26110 |
  | nav_v6_start_26110 (V6 = 26113) | 154.0 | 1.6 | re-measured 2026-09-26 22:00 (section 2) |
  | nav_fall25_end_26830 (V5) | 151.4 | 1.9 | |
  | nav_planner_end_27280 (V5), walk planner | 151.4 (16 rounds) | 1.9 | planner training: no gain (log 28.52) |
  | nav_v6_end_26595 (V6) | 150.0 | 1.4 | 2.5 h of training with spin (section 2) |

* **Human reference** (`demos/`): the 2026-09-14 session has rounds of 163, 177, 172, 121, 119 and 109,
  and the 2026-09-24 session one of 143. Always compare per round: the 143.5 mean mixes strong and weak
  rounds. The `HUMAN` constants in `real_run.py` are that older mean.
* **Training:** stopped. The last run was V6 (spin), updates 26110 to 26595 on 2026-09-26 19:10-21:45:
  1,344 KOTM training rounds at 140.4 points and 2.13 falls a round, level with the runs before spin
  (137.6-140.0 points, 2.3-2.8 falls). Training rounds are sampled play and read about 12 points below
  the deterministic eval.
* **Maps:** training rotates 7 KOTM / 1 FlatIslands instances. FlatGemTraining was dropped on
  2026-09-22 (it had saturated at 93-95 % of the human). `kotmjump` (a KOTM copy with a gem floating
  over each big hole) is in the game as the jump test map and is NOT in the rotation.
* **KOTM is the base model** that other maps will fine-tune from (per-map checkpoints later).
* **Jump physics P0 (2026-09-27): PASSED.** A small learned flight predictor, trained on 184,000 recorded flights at
  one kotmjump hole, chose a jump that collected the floating gem and landed safely on 67.7 % of 251 feasible
  held-out starts, against 18.7 % for the best simple rule. Separate code (`nav/learned_nav/`); the PPO navigator
  and its checkpoints are untouched. M0 on kotmjump: the navigator scores 4.6 points a round and stalls after its
  first three gems. Details: log section 29 and `KOTMJUMP_START_HERE.md`.
* **Jump physics stage 3 (2026-09-27/28 night): built and run.** Engine contact telemetry (mbx, local commits), 13
  verified practice maps, 2.42 M recorded trials, a step model with collision branches (one-step error 0.003 u, the
  same on two never-trained maps; 1 s rollouts 0.4 u) and a flight model. P0's held-out test chosen by the general
  models together: 52 % (P0's own model 68 %, the fixed rule 19 %). M1 passed; M2 met with two notes (keep both
  collision branches in planning; heavy error tails). Log section 30; `nav/learned_nav/`. Next (operator): stage 4.
* **Jump physics stage 4 / M3 (2026-09-28): gate PASSED.** A model-predictive planner on the stage 3 models (about
  450 candidate programs rolled 2.6 s ahead every 64 ms, pickup x safe landing with edge margin, re-checked from
  perturbed starts, seed-aimed approach by walking distance) took the floating gem and landed on 217 of 233 feasible
  held-out starts (93.1 %, 95 % 89.1-95.7; per target 90-97 %; 7 abstentions; unsafe starts 15/23 apart). Jumps sent
  with a pickup plan took the gem 187 of 188 times; the failures are approach stalls, roll-ins and landings near the
  strips between holes. Time-to-gem guide from PPO's traces trained (0.115 s error vs 0.202 s). Log section 31;
  page https://claude.ai/artifact/G82yZ3WkUWYziUizq6eTK1; watch: `nav\learned_nav\watch4.ps1 -Map kotmjump_p2`.
  Training still stopped. Next (operator): M4.
* **Jump physics stage 5 / M4 (2026-09-28/29 night): whole rounds work, the single-gem check did not pass.** Whole
  3-minute kotmjump rounds with the navigator driving and the planner taking only the floating gems (hybrid.py): 72.75
  points a round (85, 85, 67, 54, memory kept current; 60.75 with it reset, not a settled difference) against 5.25 for
  the navigator alone (9, 4, 4, 4; 33 falls a round); 14 floating gems and 2.25 falls a round, 1 fall within 3 s of 111
  handbacks; planner alone 65 (1 round, no falls). Single floating gems on 228 new frozen starts: 90.8 % (86.3-93.9) against a bar of
  95 % per gem (p0 84.7 %, p1 90.2 %, p2 94.2 %, p3 94.6 %): tuning on the same 105 dev starts reached 95.2 % there but
  overfitted. Largest failure: landing on the strip between two holes at speed and rolling into the next (no
  stopping-distance check). Planning 570 -> 400 ms a decision with identical results. Log section 32; page
  https://claude.ai/artifact/P6coTg3eQAtKZveks1cXMp; results
  logs/learned_nav/drills/report_test5_gate.json and logs/learned_nav/rounds/. Training still stopped.

## 2. Spin A/B (2026-09-26 22:00-22:40, 3x, 8 rounds per arm)

| arm | mean (se) | falls/round | speed |
|---|---|---|---|
| nav_v6_start_26110 (= 26113; spin weights still zero) | 154.0 (1.4) | 1.6 | 8.44 |
| nav_v6_end_26595, real spin | 150.0 (1.2) | 1.4 | 8.35 |
| nav_v6_end_26595, spin zeroed | 150.0 (1.8) | 0.75 | 8.32 |

* **Spin neither helps nor hurts yet:** real against zeroed spin on the same weights is +0.0 +- 2.2 points.
  Influence, measured by re-running every deterministic decision of the real-spin arm (22,771) with spin
  zeroed (same inputs and hidden state): steering changes by a median 0.6 deg (p90 3.3, p99 11.7), jump
  decisions flip on 0.009 %. The spin weights are ~4 % of the velocity weights after 2.5 h of training.
* **The training run itself cost 4 points:** the endpoint scores 4.0 +- 1.8 below its own start, with
  spin real or zeroed alike (fewer falls, slightly slower). That matches the whole day: no checkpoint
  trained after 26113 beats it at the deterministic eval (step 2 end 146.8, FALL-25 end 152.0 / 151.4,
  FALL-25 best 152.8, planner end 148.5 greedy, V6 end 150.0), while the sampled training rounds stayed
  flat at 137-140. With the current reward, more training on KOTM does not raise the eval.
* Tool: the wrapper that zeroes spin and logs the influence was a scratch script around
  `nav/real_run.py` (monkeypatching `ObsBuilder.build` and `NavActorCritic.act`); nothing in the repo.

## 3. What separates the model from a 170 round

Measured on 2026-09-25/26 against the human's 177 and 172 rounds, per gem spawn (log 28.49-28.50):

* **Route length is the gap, not speed.** Human 45.1 u and 5.15 s per spawn, model 50.7 u and 5.90 s.
  Mean speed is equal (human 8.13 u/s, model 8.1-8.25).
* **About 2/3 of the route gap is path straightness:** the human crosses holes with 12-15 gap jumps a
  round, the model with about 1.6. This is what the jump physics work targets (section 4).
* **About 1/3 is gem order and pickup position.** Greedy takes the best order 38 % of the time. The
  whole-spawn planner fixed the order (54-61 %) but the policy got slower on its routes; parked (section 9).
* **The model brakes into every gem:** about 7.7 u/s at pickup whatever comes next, where the human
  carries 11.4 u/s when the next gem is straight ahead. Unfixed; the reward steps of 2026-09-26 moved it
  only from 7.8 to 8.2.

## 4. Next: jump physics

**The plan we follow** (operator, 2026-09-27): [KOTMJUMP_NAVIGATION_DESIGN.md](KOTMJUMP_NAVIGATION_DESIGN.md),
starting with [KOTMJUMP_START_HERE.md](KOTMJUMP_START_HERE.md). In short: a dynamics model learned offline from
engine trials on the game's own maps (a step model plus a direct flight head), forward search from the
marble's real state seeded from the gem side, and a feedback controller, with a second arm where PPO drives
and the controller takes over for planned manoeuvres. No engine rollouts during play. The first experiment
(P0) is a recorded flight experiment on one kotmjump hole, then learned action selection on held-out starts.
All six decisions of the earlier `PHYSICS_PLANNER_PLAN.md` are settled in the design; that file is history.
Nothing is implemented yet, and no code, checkpoint, training rotation or runtime setting has changed.
The operator's answers (2026-09-27): P0 runs on a new drill map `kotmjump_p0`; building P0 is approved;
PPO training stays stopped during P0; Horizon and Archipelago are the fixed reference maps, never
trained on. Implementation starts when the operator says.
The 3 u minimum jump gap in `nav/physics.py` (`MIN_JUMP_GAP`) stays a KOTM-only hack of the baseline; the new
planner must not depend on it.

Longer term (the 2026-09-16 roadmap, archived): the navigator on every map, a planner above it
(targets, powerups), then multiplayer and self-play.

## 5. The system

**Game side** (`Marble Blast Platinum/platinum/client/scripts/ai/`)
* `mlAgent.cs`: the update loop and the control words Python sends: SPEED, FIXEDSTEP, LOCKSTEP,
  RENDEREVERY, VIEWYAW, TELEPORT x y z [vx vy vz [wx wy wz]], OOBCLICK (the legal quick respawn),
  RESPAWN, MARK, INFO, DEBUG, STATS, DELAY. The policy plays the Ready/Set countdown, so spin builds up on
  the pad before GO (pre-spin). In `-autotrain` mode the view yaw is fixed from the first frame.
* `observer.cs`: the raw observation, 38 numbers: position 3, velocity 3, 5 gems x 5, time left,
  time elapsed, score, gems remaining, spin 3. Gems come from the live ServerConnection
  (`$AIObserver::GemSource = "server"`; the old item cache blinded the agent after every pickup).
* Engine: `marbleblast_mbx.exe`, OpenPQ-TGEMIT branch `ai-training-mode` (off `mbx`), worktree
  `C:/Users/doug/src/OpenPQ-TGEMIT-mbx`. Adds fixed step, lockstep, render-every and a render-only
  view yaw.

**Python side** (`ml_agent/nav/`)
* `train_nav.py` (trainer: 8 instances on ports 8888-8895, 1,024 decisions per instance per update),
  `vec_worker.py` (one torch-free worker per game), `ppo_recurrent.py` (sequences of 32, LR 3e-4,
  3 epochs, clip 0.2, an entropy controller holding about 0.30), `env.py`, `protocol.py`.
* `obs.py`, NAV_OBS_V6: two 6x32x32 height crops (0.5 u and 2 u cells) and a 61-number vector:
  0-3 waypoint, 4-9 self (velocity, speed, on floor, airborne), 10-47 edge rays, 48-52 next gem,
  53-57 gap block (lip, gap, landing-predictor verdicts, speed ratio), 58-60 spin as a rolling velocity.
* `model.py`: CNN + MLP -> GRU(256) -> heads (direction Gaussian, throttle with a 0.90 floor, jump,
  brake, PopArt value). `DIR_GOAL_GAIN` 30 on the goal bearing. Jump prior: +8.5 on the jump logit for
  a physics-approved gap, added after the clamp; `JUMP_DAMP` 1.
* `waypoints.py` (reward): PROGRESS 1.0, PROGRESS_NEXT 0.3, ARRIVE 10 per gem plus GEM_SPEED_BONUS 24
  falling to 0 at 60 decisions, TIME 0.05, FALL 25, FALL_AFTER_JUMP 5. AIR, RUNUP_K, JUMP_TAKEOFF,
  ALIGN_BONUS, BRAKE, CARRY, EDGE_K and TURN_COST are all 0.
* `gems.py`: the greedy chooser with a momentum turn cost (`MOMENTUM_K` 1.6). `NAV_TOUR=walk` selects
  the whole-spawn planner (parked).
* `physics.py`: the landing predictor (`predict_landing`, `jump_verdict`) behind the gap block and the
  jump prior. `terrain.py`: height grids, walk grid, Dijkstra fields.
* `real_run.py`: real rounds for evaluation and watching (steers straight at the gem, stuck-breaker
  after 3 s, legal quick respawn). `dashboard_nav.py`: the dashboard, port 8990.
* Training mode: real-gem continuous (`NAV_REAL_GEMS=1`: the game's own gems, not synthetic
  waypoints), 3x, 64 ms decisions under fixed step + lockstep.

## 6. How to run

* **Train:** from `ml_agent`, `Start-Process powershell -ArgumentList '-File','start_training.ps1'`
  (do not pipe it; that hangs the calling shell). It starts the trainer, waits for the 8 ports, then
  `run_game_loop.ps1`. Training resumes from `nav_latest.pth`. While training, run from `ml_agent` in
  bash: `logs/nav/game_watchdog.sh` (relaunches the games when throughput halves) and
  `logs/nav/game_summary.sh` (a KOTM summary every 15 min, plus a snapshot
  `models/nav/nav_night_<update>_<mean>.pth` at each new 50-round high; state in
  `logs/nav/night_best.json`, which should be archived and reset at the start of a run).
* **Stop training:** kill `nav.train_nav`, `run_game_loop` and every `marbleblast_mbx`, then copy
  `nav_latest.pth` to a named endpoint.
* **Evaluate** (training stopped; the 8 GB GPU cannot take a 9th game beside the trainer):
  `nav/real_run.py` with `NAV_ROUNDS=8`, `NAV_PORT`, `NAV_CKPT`, `NAV_TAG`, `NAV_TRACE`. Several arms
  can run side by side on ports 8921 and up: start Python, wait for its port (`Wait-NavPort` in
  `nav_ready.ps1`), then `marbleblast_mbx.exe -autotrain KingOfTheMarble_Hunt -aiport <port>`.
  `real_run.ps1` runs one arm; `eval_cycle.ps1` stops training, evaluates and restarts it.
* **Watch at 1x:** `NAV_SPEED=1`, `NAV_WATCH=1`, `NAV_VIEW_SUBSTEPS=4`. Sub-steps are for viewing
  only: they change the integration, so do not measure with them.
* **Change the observation:** bump `NAV_OBS_VERSION`, append the new inputs at the END of the vector,
  and warm-start with `python logs/nav/migrate_obs_width.py <src> <dst>`. It adds zero weight columns, so
  the converted model behaves identically until training moves them. Checkpoints of the old version then
  need the old code.

## 7. Traps that have cost hours

* **Game throughput halves after a while** (every game at rtf 2 instead of 3); relaunching the games
  fixes it. `game_watchdog.sh` does that.
* **Edited `.cs` / `.mcs` files do not take effect** until the matching `.dso` is deleted.
* **PowerShell `$env:X = ""` DELETES X,** so an "off" arm silently runs the code default. Set empty
  values inside Python, or use an explicit word (`NAV_TOUR=greedy`).
* **A PowerShell command that kills processes by command line matches ITSELF.** Exclude `$PID`.
* **Python cannot always write repo files under OneDrive** (Errno 22 on open for writing). Use cp or an
  editor.
* **The engine buffers its console output;** a force-kill loses the latest `echo` lines.
* **Python must be listening before the game reaches GO** (the game dials Python). Every launcher
  starts Python first and waits for the port.
* **Training rounds are not eval rounds.** Sampled training play reads about 12 points below the
  deterministic 3x eval, and training curves have misled repeatedly.
* **`game_summary.sh` drops KOTM rows over 130 gems:** those are two rounds merged into one line.

## 8. Standing rules from the operator

* Never stop a training run on your own for any metric; report, and let the operator end it.
  Restarting to apply an approved change is fine.
* No code changes without the operator's approval. Present options, then wait.
* Everything must transfer to other maps: no map-specific hacks, no geometry gates such as "open floor
  between marble and gem". The reward pays game rules only (gems, plus a small fall marker).
* Training waypoints may only appear where real gems can spawn, and a marker may only be drawn on a
  real gem.
* Judge changes by 8-round `real_run.py` evaluations, never by the training curve.
* An unspecific "turn off X" can mean an earlier feature: check what is live and ask before reverting.

## 9. Tried and parked, or dead (details in the log)

* Whole-spawn walk planner (28.50-28.54): shortened the route 2-3 u per spawn, but the policy slowed
  and overshot gems; no score gain. Parked; revisit if scores stall.
* Reward = game score, steps 1-2 (28.42-28.47): no measurable gain. The current reward is where that
  work ended.
* Direction smoothing at inference (2026-09-25/26): no gain at three settings.
* Time repricing (28.9): bought speed with falls; reverted.
* Jump on the prior at inference (28.33): 117.1; rejected.
* Behaviour cloning, as a start or as an extra loss (2026-09-14, old architecture): both regressed.
* Video recording mode (28.55) and the Super Speed aim test (28.57): built for demos, then removed.
  A Super Speed fires along the marble's camera yaw (`marble.cc` doPowerUp case 2), which the actions
  set every decision, so any direction works.
* Powerups in training: deferred. The human used none in the 143-177 rounds.

## 10. Documents

* `HANDOFF_NAV_TRAINING.md`: this file, the current state.
* `docs/HANDOFF_NAV_TRAINING_LOG.md`: the dated history, sections 1 to 28.x.
* `BIG_QUESTIONS.md`: the open questions for general-purpose navigation (physics, terrain, routes, plan to policy).
* `JUMP_PLAN_OVERVIEW.md`: the jump physics plan in plain words, with the happy-path roadmap.
* `KOTMJUMP_START_HERE.md`: the first experiment of the jump physics plan (P0). Start here.
* `KOTMJUMP_NAVIGATION_DESIGN.md`: the jump physics plan we follow, as the reference specification.
* `PHYSICS_PLANNER_PLAN.md`: the first jump-predictor draft (2026-09-26); superseded by the design, kept as history.
* `PHYSICS_SKILLS_DESIGN.md`: the 2026-09-23 design. Its plan was superseded by
  `PHYSICS_PLANNER_PLAN.md`; its engine facts (`marble.cc` line references) are still the reference.
* `README.md`: quick start, architecture and lessons.
* `docs/archive/`: superseded documents (the 2026-09-16 roadmap and implementation plan, the old
  dashboard's chat context, earlier designs).
