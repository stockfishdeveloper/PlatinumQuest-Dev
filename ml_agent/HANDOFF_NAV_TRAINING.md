# Navigator training: handoff

**Read first, 2026-10-05 12:05 MDT:**
[HANDOFF_SUPERSPEED_2026-10-05.md](HANDOFF_SUPERSPEED_2026-10-05.md) has the current
two trainers, process ownership, latest score regression, finite fast-state
experiment, evaluation commands, and unfinished spin-aware physics investigation.
Both runs remain running; no qualifying score or reliable Super Speed gain yet.

**Active operator goal, 2026-10-05:** achieve a complete KOTM round above 175 with actual Super Speed use,
and demonstrate that enabled use improves scores against a no-use control. Neither is established yet.
See [GOAL_175_2026-10-05.md](GOAL_175_2026-10-05.md) for the current gates, rejected experiments and data collection.

The CURRENT state of the King of the Marble (KOTM) navigator: what exists, what it scores, what is
wrong with it, how to run it, and what comes next. Keep this file current: when something changes, edit
it here. The dated history of every experiment (sections 1 to 28.x, 2026-09-17 to 2026-09-26) is in
`docs/HANDOFF_NAV_TRAINING_LOG.md`. Code comments that cite "HANDOFF 28.52" and the like point into that
log, and new dated entries go there too.

**Current run, 2026-10-05 10:03 (log 40.49):** training is running with four real replay workers and
four ordinary KOTM rounds, starting from the protected 28897-named policy (internal update 28895).
Fresh data: 102 development / 18 evaluation starts from disjoint source rounds; all 120 restores passed.
At checkpoint 28920, critic warm-up is complete, the use head/residual are learning, and the frozen
actor parameters remain exactly unchanged. Supervisor, checkpoints and operational details:
[`RUN_POINTS_2026-10-05.md`](RUN_POINTS_2026-10-05.md). Earlier stopped-run statuses below are historical.

**Latest training implementation, late 2026-10-04 (log 40.48):**
[`HANDOFF_POINTS_REPLAY_2026-10-04.md`](HANDOFF_POINTS_REPLAY_2026-10-04.md) is the next-run handoff.
Real replay now restores gems/items/spawn state and the original approach target, pays game points in both
rounds and windows, counts recovery inside the 12032 ms window, and migrates the critic to point-scale values.
Eleven unit checks and isolated game probes passed. Fresh schema-2 data must be collected before the default
four-replay/four-round training run; old `starts_live_*.json` are incompatible. No main training was started,
no trained checkpoints changed, and nothing committed. Training remains stopped at update 29665.

Earlier summary, 2026-10-04 17:30 (log 40.23-40.40: the Super Speed curriculum night (safe, chosen, no points: 160.6 vs
161.1 without uses), the second review and its implementation: the use head and critic read the powerup state, the
residual acts on the approach, points-only drills with final-state bootstrap, live starts with real spin and GRU
warm-up, 12 s window evals; nothing trained since 10:06. Next run: HANDOFF_SUPERSPEED_MANEUVER_2026-10-04.md.
Protected navigator: models/nav/nav_v10_28897.pth; its deterministic score 161.1-161.6 (32-64 rounds), best 173).

## 1. Where things stand

* **Live play against people (production, 2026-10-04, log 40.46):** `ml_agent/play_live.ps1` starts
  `python -m nav.live_play` and `marbleblast_mbx.exe -ailive -aifreecam`; log in, join or host a KOTM server, start
  the round. The production bridge is `platinum/client/scripts/ai/live/agentLive.cs`, loaded only with `-ailive` (one
  hook in `client/init.cs`); the training files are unchanged. Operator 2026-10-04: training is offline; the login
  question is closed, do not raise it. Only if `-autotrain` games ever stop at "Ready..." with the pause menu open:
  `-offline` launched them normally on 10-04 (log 40.46).
* **Watching at 1x with a free camera (log 40.47):** `ml_agent/watch_model.ps1 [map]` (default KOTM); the mouse and
  arrows turn the view only (`-aifreecam`, `platinum/client/scripts/ai/watch/freeCam.cs`), also in live play.
* **2026-10-04 evening (the manoeuvre run, HANDOFF_SUPERSPEED_MANEUVER_2026-10-04.md, log 40.41-40.45):** TRAINING
  STOPPED 21:13 by the operator at update 29665 (`nav_latest.pth` = `models/nav/nav_r2_stop_29665.pth`; run 1 collection
  17:17-18:20 ended at `nav_run1_end_29115.pth`). Fixed on the way: model.evaluate_seq restarts a warm-started live drill
  from its stored GRU state (it zeroed it; log 40.42). Live starts: 848 (dev 647 / eval 201, datasets/ss_drill/
  starts_live_*.json; run 1's sets kept as *_run1.json). Final matched window eval (12 s, 167 paired starts): fire at
  every approval +0.17 gems (se 0.09) over no use; the learned use head -0.02 (0.08), i.e. it skips kicks that pay; no
  full-round gate. Sampled rounds flat at ~147 (navigator frozen). Drills pay points while rounds pay the shaped reward:
  one critic for both (its round value fell ~145 -> ~120); the operator's call (log 40.43).
* **2026-10-04 (Super Speed curriculum night, log 40.26-40.39; report https://claude.ai/artifact/LPz1RsoFBgARwWbtTYME6a):**
  TRAINING STOPPED 10:06 by the operator at update 29970 (`nav_latest.pth` = `models/nav/nav_p4i_stop_29970.pth`). That
  run trained only the use head, the post-kick residual and the critic's head on top of 28897; these are now the code
  DEFAULTS for any next run (model.py FREEZE_BASE True, SS_LR_MULT 10, SS_RES_HELD False; drill ends bootstrapped in
  train_nav v_cont; vec_worker NAV_DRILL_PLAN '0:1,1:1,2:2,3:2,4:2' = 5 of 8 games run drills). Super Speed kicks need
  LEVEL braking room (obs.py SS_LEVEL_DZ 0.75: the old floor scan counted the platform's outer rim, and 39 % of kicks
  fell). Full rounds (deterministic, pooled): learned use over three snapshots (29069, 29231, 29377) **160.6** (112
  rounds, best 172) vs the same navigator without Super Speed **161.09** (64 rounds, best 173; last evening 161.62 over
  32): no measurable Super Speed gain. Kicks ~1 a round, 4 of 87 fell within 3 s. The trained post-kick residual adds
  falls (1.12 vs 0.69 a round with it zeroed, same points; log 40.38). The first part of the night trained the whole
  network on drills and wore the navigator down to 141.5 (nav_p4g_*): do not train the navigator on drills without an
  anchor. Diagnostics: real_run NAV_FORCE_USE, ss_drill_eval --force-use / --no-use, logs/nav/gate_scripts/ (ss_eval.sh,
  rr_gate_patch.sh + rr_patch.py, probe_eval.sh + use_probe.py, kick_falls*.py, fall_vs_kick.py).
* **Goal:** a 170-level KOTM round, the level of the human's best rounds. The earlier goal of 150 was
  reached on 2026-09-26.
* **Model (2026-10-02):** the best checkpoint is `models/nav/nav_best_20261002_tour_28897.pth` (a copy of
  `nav_night_28897_151.pth`, update 28897), trained 10-01/10-02 from 26110 on the WHOLE-SPAWN gem order
  (NAV_TOUR=walk, gems.plan_tour) with all 8 instances on KOTM. `nav_latest.pth` is the endpoint of that run,
  update 29794 (`nav_tour_end_29794.pth`), stopped by the operator at 15:30 on 10-02. Play it with the same
  chooser: the hybrid's ORDER_TOUR = 'walk' (hybrid.py), or NAV_TOUR=walk for real_run.py; on greedy routes
  this policy is slower. The earlier base, 26113 / `nav_v6_start_26110.pth` (154.0, greedy), is still on disk.
* **Verified KOTM scores of 28897** (8 rounds each, 10-02, log 37.3-37.6; `NAV_CKPT=<pth>` picks the
  checkpoint for hybrid.py, `NAV_PLANNER_OFF=1 --shortcuts 0 --rescue 0` is the navigator alone):

  | arm | rounds | mean | best | falls/round |
  |---|---|---|---|---|
  | navigator alone, whole-spawn order | 162 163 170 165 150 156 157 158 | 160.1 | 170 | 1.0 |
  | hybrid (oracle + consult at the pickup), same | 163 163 164 152 158 157 162 152 | 158.9 | 164 | 1.5 |
  | hybrid on 27493 (the earlier snapshot) | 159 158 158 162 162 152 150 162 | 157.9 | 162 | 1.6 |
  | hybrid on 26110, greedy order (10-01) | | 149.9 | 153 | 2.3 |

  The planner is worth about -1 point on KOTM at this level (inside noise): its 0.6 shortcuts and 1.5 jump
  hints a round save ~0.1 u per gem and no time, its 0.6 falls give that back. The operator's goal, a round
  above 167 WITH the planner, is not met; the navigator alone has one 170 round.
* **Earlier verified KOTM scores** (3x, 8 deterministic rounds each, `nav/real_run.py`, greedy gem chooser):

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
* **Stage 5b (2026-09-29): KOTM shortcuts, goal (beat 154.0) not met.** 8 rounds per arm at 3x: navigator alone
  151.0, navigator + shortcuts 148.4 (the planner acted 17 decisions a round: noise), + route jump legs 146.1 (slow
  mid-leg takeovers, more falls). Found and fixed: the step model underestimates jumps (apex -0.25 u median for fast
  takeoffs near lips); planner.py JUMP_PRIOR uses the measured takeoff where the jump certainly fires (drills 58/67 vs
  v8 55/67). With it the planner flies the human's centre-gem -> corner crossing in game 22 of 24 times at human speed
  (1.35-1.86 s). Fall guard and route legs tried and turned off: taking control mid-leg cost more than it saved.
  Log section 33.
* **Stage 6b (2026-09-29 night / 09-30): the crossing leg measured from the navigator's real pickup states; goal (one
  KOTM round above 167 with the planner) not met.** A crossing drill (crossdrill.py, 89 dev / 91 test starts = the
  navigator's states right after a pickup, aimed across a hole) shows the planner's crossing takes 2.06 s on average
  against the navigator's fitted 2.10 s walk: a real gain only for starts within 30 deg of the line (1.66-1.73 s, the
  human's 1.66), a loss at 30-60 deg (2.30 s). Fix kept: ranking saturation (planner.P_SAT 0.6 on crossing legs: the
  planner braked from 10 to 6 u/s before jumps because slower flights judge safer); dropping the flight-head check
  (v2) added nothing. Same-day gate: navigator alone 152.0 (6 rounds, best 162), hybrid with shortcuts 151.5 (4
  rounds, best 156) and it took no shortcut at all: one cold consultation a few decisions after a pickup finds no
  pickup plan where the drill's continuous planning does. Next: consult 2-3 decisions at the pickup with the planner
  state kept, crossings only near-aligned or with the previous leg's exit chosen for them. Also found: with the screen
  LOCKED every late bridge reply costs the game ~200 ms (cause open; run games in parallel, or leave the screen
  unlocked). Log section 34; scripts nav/learned_nav/stage6b/.
* **Stage 6b, daytime 09-30 (log 34.7-34.13): where the human's crossings come from, the turn is physics-limited,
  the pure controller measured and improved 85 -> 93.** Every human crossing is the leg into a freshly spawned group;
  our navigator gets ~8 aligned chances a round but a consult-by-driving at the pickup (hybrid CONSULT_DEC, route
  setup orders, planner exit objective) found no crossing that pays from its states: ceiling ~2 points. Turn drill
  (turndrill.py, engine only): a 90 deg turn costs 0.9 s at speed, 120 deg ~1.2 s; the step model reproduces it
  within 5 deg. Planner alone on KOTM (rounds.py): 85 -> 91-96 with seven physics-derived levers (brake tail kept
  at the end of shifted programs, floor-jump charge, GROUND_WIN, leg time weight and saturation, ROLL_CLEAR_OK,
  turn-costed order, next-leg turn charge); aligned legs 1.15-1.28 s (navigator 1.29), but a third of its legs start
  turned away and cost 2.2-3.5 s. Navigator 152. Nothing committed.
* **Stage 6b, night 09-30 (log 34.14-34.18): the operator's own jump found and fixed three general planner faults.**
  From the recorded launch state (13.2 u/s, 16 deg off the line) the planner had no plan: the step model gains speed in
  the air where the game loses it like drag (fixed: planner.AIR_DRAG_RESID 0.0009 x v^2 on airborne steps, validated
  on 16 human jumps: air speed error +1.2 -> +0.1 u/s, landing error 1.3 -> 0.5 u); its proposals lacked the air input
  past the gem's direction and the brake-turn after landing (added as families the judge ranks); presses at 12 u/s
  acted past the lip (JUMP_LIP_K: one decision of floor ahead). Crossing drill on the untouched test set: 90/91,
  1.73 s vs 2.07 walking (before: 35/36, 1.98 s). From the operator's state in the game 5/5, three in 1.15 s. KOTM
  gate, 8 rounds each: navigator 153.8 (best 161), hybrid with the consult at the pickup 149.4 (best 154; 1.9
  crossings a round at 1.4-2.4 s, fewer falls) and 145.8 with the planner also driving the leg before. The
  crossings pay ~0.8 s each; the handovers and the planner's slow floor legs cost more. The recorder
  (record_demos.py) had dropped every tick since the 38-number observation; fixed. Settings now in hybrid.py:
  SETUPS False, CONSULT_DEC 3, SHORTCUT_MAX_ANGLE 60.
* **Night of 09-30 / 10-01 (log 35): seven more hybrid batches, all at the navigator's level.** Built and measured, 8
  KOTM rounds each: the crossing plan charged for the next leg's turn, handback as soon as the continuation is clear,
  a cheap exit from a failed consult, no jump outside a committed plan, a step-scaled takeoff margin (the fixed one
  had killed the operator's own jump), the across-gem consult (harmful: 137.9, off), and the planner as a jump ORACLE
  with no takeover (8 rollouts of "jump now", 77-155 ms, sets the navigator's jump bit): hybrids 147-154 (best 164),
  oracle 152-154 (sd 2), navigator 153.8 and 155.4 (best 162). The crossing skill is real (test set 90/91) and worth
  ~2 points a round here; every integration costs about that. Latency: a consult decision 300-450 ms with a ~200 ms
  floor in CPU feature extraction; real-time needs features on the GPU or a non-lockstep mode where the navigator
  answers at once and the planner's result applies when ready. Training not started: the one idea the physics work
  enables (the oracle inside the training loop, so the policy learns routes around a jump it can rely on) is a build
  to do with the operator. Defaults now: ORACLE on (P 0.75, 8 u/s), consult at the pickup on, SETUPS off, ACROSS off.
* **10-01 daytime (log 36): the flight physics measured against the engine and put into the planner.** The operator
  saw the hybrid never jump from the ring into the centre and asked for the physics to be fixed. A fixed-program
  drill (nav/learned_nav/centredrill.py: 378 game trials from 42 ring starts, the same inputs replayed through the
  step model) and marble.cc gave the facts: in the air the engine has NO drag and 5 u/s^2 of air control per unit
  of input (the adapter sends up to sqrt 2); a landing with input held and a shallow approach SLIDES keeping the
  full speed (13 + 6.7 vertical -> 19 u/s), one without input BOUNCES at restitution 0.5; and the JUMP KEY fires one
  decision after the steering acts. The step model lost 12 % air speed without input, 30 % of the air control with
  it, never landed a no-input flight, and the jump prior fired a decision early; the AIR_DRAG_RESID of 09-30 was
  compensating the model's weak air control and made aligned flights land 2 u short. Now (planner.AIR_EXACT):
  airborne steps and flat-floor landings are the engine's equations (air_exact), the learned model keeps walls,
  lips and the ground; the jump prior fires on the previous reply's key (Jp) with an analytic fire-step horizontal;
  JUMP_LIP_K 0, AIR_DRAG_RESID 0, P_JUMP = P_GO 0.25, air-brake proposals (130-180 deg off), and a marble with no
  floor within 0.5 u under it is never 'supported'. On identical inputs the model now lands within 0.1 u at the
  same decision as the game. Drills: test set 90/91 at 1.87 s (unchanged), new ring set 34-35/42 at 1.1 s against
  1.2-2.6 s walking (falls at 112-157 deg: fast slide landings on the 7 u block, the groove round the block that
  launches a fast marble, one start with no plan that ran off the lip). hybrid.py: BIG_GAIN_U / CONSULT_DEC_BIG
  (legs with a 4 u+ detour are consulted at any aligned decision of the leg). KOTM gate g6r (physics + big consults
  from any heading, 8 rounds): ~141, 1-3 planner falls a round, the planner's takeovers from slow misaligned states
  slower (2.6-2.8 s) than the navigator's own centre legs with the oracle (1.4-2.0 s); g6s (aligned-only big
  consults + the support guard) 147.1; g6t (big consults off, the g6q configuration with the new physics) 149.2
  (sd 1.9, best 153): the physics are right now and the hybrid's score is unchanged (log 36.5-36.6). Goal not met.
* **10-01 evening to 10-02 (log 37): the gem ORDER, then training on it.** The operator saw the marble take one
  centre gem, carry on to the ring and come back for the second centre gem: the greedy chooser (nearest gem plus
  a momentum penalty, never looking past the next gem). The whole-spawn chooser of 09-26 (gems.plan_tour: every
  order of the visible gems scored over walk-only terrain fields with turn costs) scored 139 on the untrained
  policy (falls on unfamiliar routes, as in log 28.50), so the operator ordered training on it: 21:14-01:37 and
  02:01-06:16 (from 22:17 all 8 instances on KOTM, Islands dropped by the operator), then 06:37-11:17, 11:30-14:55,
  14:59-15:30. Training-side 50-round means 138 -> 150.7 (28897), falls 2.8 -> 1.5 a round, speed held at
  8.0-8.2 u/s (the 09-26 attempt had lost speed), s/gem 1.49 -> 1.42; plateau from ~02:30 with a trickle of new
  highs (~1 point per 500 updates); training rounds of 165-167 appeared on 10-02 (the summary and the dashboard
  had been dropping every round with more than 130 gems as "merged double rounds"; cut raised to 190). Gates in
  the table above. The hybrid's score follows the navigator's: the whole-spawn routes are 5 % shorter per gem.
  Not done: a jump oracle inside the training loop (the learned rollout is 80-400 ms a decision against the
  trainer's 440 decisions/s; an analytic ballistic check, nav/learned_nav/fastjump.py, was written and validated
  on the drills, 62/90 recall on the test crossings, 18 false goes in 203 pickup states, then the operator
  stopped that work; it is not wired in anywhere). Repo: *.npz are gitignored (kept on disk); the four unpushed
  commits were rewritten without the .npz and the intermediate .pth (2.9 GB -> 150 MB) and pushed; the old
  history is the local branch backup/navigator-architecture-before-rewrite-20261002.
* **Night of 10-02/03 (log 40.3-40.7): powerup training with the use prior; goal not met, still training.** From
  update 29295 at 23:36 to 30916 at 06:18 (5,218 KOTM rounds, no crash): training rounds 109 -> 126 a round (phase 4a
  ended at 138-140; 127-130 by 07:50), falls 3.7 -> 2.5; 8-round gates of seven checkpoints 132.1-138.3 (best round 154,
  the last 31275 at 138.1) against 160.1 for
  28897 without uses, and 128.1 for 28897 with the prior's rule alone. As coded, the head cannot cancel the prior
  (model.py adds USE_PRIOR after the -9 clamp), so every approved use fires in deterministic play; the Super Speed
  overshoot (about 30 u/s past the gem) and the blasts approved at that speed cost the points. Gates beside the trainer:
  CPU real_run (logs/nav/gate_scripts/rr_gate.sh); the hybrid gate overflows the 8 GB GPU.
* **10-03 13:10 (log 40.8): the use floor fix is in and training restarted.** model.py USE_PRIOR_FLOOR -4: where the prior
  approves, the head can now take the use logit down to -4 (no fire in deterministic play). The same checkpoint (32425)
  under the new rule, before any training with it: 144.5 (best 156), falls 2.5 a round, Super Speed uses 0.1 a round
  (old rule 138.1, falls 5.0). Training relaunched 13:19 from nav_latest (32445; the old run's endpoint is
  nav_p4b_end_32445.pth, its logs *_phase4b_20261003.txt). First high 139.0 at 32473: gate 147.9 (best 152) with almost
  no uses. **Training STOPPED 13:36 by the operator** at 32510 (nav_p4c_stop_32510.pth = nav_latest); resume with
  start_training.ps1.

* **10-03 night (log 40.23-40.25): training STOPPED 23:41 by the operator** at 30575 (nav_p4f_stop_30575.pth =
  nav_latest), after 45 minutes of the 24 u/s cap run (138.8 a round, Super Speed turns reaching the gem 56 %). The
  operator's goal: one round over 175 WITH the powerup model. Fixed for evaluation: real_run's stuck-breaker fired only
  falsely on KOTM (loops through gem clusters); NAV_STUCK_PICKUP (default on) lifted 28897 from 159.6 to 162 in 8-round
  batches. The approval admits 13 of the operator's 20 demo kicks; the gap is the Super Speed supply and the kicks'
  hit rate (proposals in 40.25).

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
* IMPORTANT (2026-10-06): before training on a new map, or after making or changing a terrain map, verify the map in
  the game: teleport the marble onto edges, corners and holes; it must fall or stay exactly as the map says. Train
  only on a map that passed. KOTM's map was ~0.5 u off for a month (log 40.58-40.59); the probe tool is
  logs/nav/gate_scripts/map_diff_probe.py (written for KOTM; a new map needs its own edge spots).

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
