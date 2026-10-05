# Handoff: training the complete Super Speed manoeuvre (2026-10-04, after the second review)

Written for the agent that starts and watches the next training run. Read this file, then `HANDOFF_NAV_TRAINING.md`
sections 6-7 (how to run, traps), then log sections 40.26-40.40 in `docs/HANDOFF_NAV_TRAINING_LOG.md` (the night's
curriculum, its results, the two reviews and what was built from them). The operator (doug) decides; the standing
rules below are his.

## 0. Rules

* **Never stop training on your own**, for any metric. Report; only the operator ends a run. Relaunching the SAME run
  after a crash or a hang is expected.
* **No map hacks** ("do NOT make a hack that will only work on this map"). Everything here is physics (the 25 u/s
  kick, the measured 14 u/s^2 brake) plus the terrain map along a line; no coordinates, no item positions, no KOTM
  geometry in the policy path. Reward = the game's points (the drills pay +1 per pickup and nothing else). Never
  train, calibrate or tune on Horizon or Archipelago. The operator's demo (log 40.19) gives targets, never movements
  to copy.
* Code edits: the Super Speed work in this file, fixes that keep the run and the evals going, docs and logs. Anything
  else is a proposal for the operator.
* The operator commits the repo himself. Do not commit, do not push.
* Writing for the operator: no em dashes, no "drift", no "the one thing", no "this is X, not Y". ASCII only in Python
  log output.

## 1. The goal and the question

The operator's Super Speed technique (demo, log 40.19): roll fast into a gem, fire 0.1-0.2 s after the pickup aimed so
the velocity turns 90-150 deg onto the line to the next gem, exit at 11-23 u/s (median 17), brake into the next gem,
20 times a round, no falls. One policy has to (a) choose the use, (b) control the marble after the kick, and (c) go in
faster BECAUSE it holds a Super Speed and the redirect will work. The night of 10-03/04 made (a) learnable and the
kick safe (log 40.26-40.39: the mask + differentiable mixture, the level-floor rule, the frozen navigator) but found no
points: the learned model 160.6 over 112 deterministic rounds vs 161.1 for the same navigator with Super Speed off,
firing 0.8 times a round. The second review found why the rest could not be learned, and that is what was built
today (log 40.40):

* the use head and the critic now READ the powerup block and the use state (before, only the frozen hidden state,
  whose powerup columns are zero: the head could not tell a held Super Speed from an empty slot);
* the residual (steering + brake) also acts on the APPROACH (held, approach flag on, gem within 10 u), so (c) can be
  learned; never while merely holding an item;
* drills pay the game's objective only (+1 per pickup) and their cut is bootstrapped with the critic's value of the
  drill's actual final state (x GAMMA, x GAMMA^47 after a fall), not a mean bonus;
* LIVE STARTS: round instances record every fire of their own play with the actual spin, 2 s of observation history
  (GRU warm-up) and the gems taken in the 12 s after; stage 4 drills replay those as 12 s continuation windows,
  falls included, with the policy deciding whether and when to fire; matched comparisons on the same start ids.

The question for the run: does the complete manoeuvre (approach, use, recovery, continuation) earn more points than
not using the Super Speed, on matched starts, and then in full rounds against both controls?

## 2. What exists (code and data)

* `nav/model.py`: FREEZE_BASE True (only use_head, ss_res, value_head train; SS_LR_MULT 10 on the first two), USE_EPS
  0.2 mixture inside the physics mask, USE_SS_ONLY True, SS_APPROACH_U 10, AUX_DIM 47, SS_RES_HELD False, BRAKE_EPS_POST
  0 / STD_POST_MULT 1 (off). The residual is zero at init.
* `nav/obs.py` (V10, VEC_DIM 98): the kick aim [28-29], its resulting speed [30], the approval [26] (SS_LAM_MAX 24,
  level floor for the stop at 14 u/s^2 or the next gem on the line + 6 u), the approach flag [27], the use state [31-36].
* `nav/vec_worker.py`: NAV_DRILL_PLAN 'idx:stage,...' (stage 0 rounds, 1 post-kick, 2 pre-kick, 3 approach, 4 live
  window); LIVE_RECORD on for round instances (files `logs/nav/live_starts_<stamp>_<idx>.jsonl`); GAME lines carry the
  powerup diagnostics (log 40.13-40.20: ap2, u2/r2, f2, rhit/rmiss, fss, ksA-C/kfA-C, pkon/pkoff/pknone, rwfire/rwall);
  DRILL / DRILL2 / DRILL4 lines per drill, DRILLSUM per 100.
* `nav/waypoints.py`: DRILL_REWARD 'points', DRILL_WINDOW_DEC 188, drill_window (falls continue), the continuous arrival.
* `nav/train_nav.py`: the final-state bootstrap, the GRU warm-up, the two Adam groups under FREEZE_BASE.
* `nav/ss_drill_starts.py`: `--live [files]` -> `datasets/ss_drill/starts_live_{dev,eval}.json` (split by id hash);
  the older trace-based starts (inferred spin) are `starts_{dev,eval}.json` (stage 1), `starts2_*` (2), `starts3_*` (3).
* `nav/ss_drill_eval.py --ckpt <pth> --stage 4 --port <p> --tag <t> [--no-use | --force-use] [--n N]`: the window
  eval (one game window; `logs/nav/gate_scripts/ss_eval.sh` wraps the game launch for stages 1-3; for stage 4 use
  `nav/learned_nav/run_one.ps1 -Module nav.ss_drill_eval -Map KingOfTheMarble_Hunt -Port <p> -Tag <t> -ArgStr "--ckpt ... --stage 4 --port <p> --tag <t>"`).
  Output `logs/nav/ss_drill_eval_<tag>.json` with per-start rows keyed by id, and a WINDOW EVAL line: pickups in
  12 s, falls, first pickup time, fired %.
* `nav/ss_live_smoke.py`: the end-to-end check on one game (round instance sampled -> live file -> converter).
* Checkpoints: `models/nav/nav_v10_28897.pth` = the protected navigator (V10, no use head trained, residual zero):
  START HERE. `nav_p4i_29231.pth` / the frozen-run snapshots of the night = the previous powerup comparison (their
  heads have the old width: they load with a fresh use head, so they are only a reference for the residual-off
  gate, 161.4). `nav_p4i_stop_29970.pth` = the night's endpoint.
* Full-round gates: `logs/nav/gate_scripts/rr_gate.sh <rounds> <port> <ckpt> <tag> [...]` + `rr_read.py <tags>`
  (CPU real_run, walk tour; NAV_NO_USE=1 for the no-use arm via `rr_gate_patch.sh`/env, NAV_FORCE_USE=1 for
  fire-at-every-approval); 32 rounds per arm, the differences are 1-2 points. References (log 40.37, 40.39): the
  navigator alone 161.1-161.6 (se 0.6), best 173; learned use + residual 160.6; residual off 161.4.

## 3. The run (the operator decides when; nothing is running now)

Step 0, before anything: make sure the smoke test passed (log 40.40 / the end of this file) and that
`datasets/ss_drill/starts_live_{dev,eval}.json` exist. If they do not, run 1 below creates them.

Run 1 (collection + stages 1-2, ~1 h): `start_training.ps1` with the worker's default plan but no stage 4 yet:
set NAV_DRILL_PLAN '0:1,1:2,2:2' in the environment of the trainer (or edit the default in vec_worker.py, no CLI
flags): 1 post-kick drill, 2 pre-kick drills, 5 round instances recording live starts. From `models/nav/nav_v10_28897.pth`
copied to `nav_latest.pth` (keep the night's nav_latest aside as `nav_p4i_stop_29970.pth`, it already is). Archive
`logs/nav/night_best.json` and `stdout.txt` first. Watch the GAME lines' f2 / r2 / fss and the DRILL2 lines
(fired vs not fired return). After ~1 h: `python -m nav.ss_drill_starts --live` (all live files), expect several
hundred starts.

Run 2 (the manoeuvre, overnight): restart with NAV_DRILL_PLAN '0:1,1:2,2:4,3:4': 1 post-kick, 1 pre-kick, 2 live
window drills, 4 rounds (still recording; re-run the converter every few hours and restart only if the start set
has grown a lot). Resumes from nav_latest.

Every hour: the diagnostics above by quarter (log 40.20's one-liner), DRILLSUM / DRILL4 summaries (picks, falls,
fired), the use preference where approved (`logs/nav/gate_scripts/probe_eval.sh` / `use_probe.py`), and the KL per
update (the heads must move: 0.002-0.005 with SS_LR_MULT 10).

Every 2-3 hours, on the newest snapshot (copy nav_latest first): the matched window eval, three branches on the same
ids: learned (default), `--no-use`, `--force-use`; report pickups in 12 s, falls, first-pickup time, paired
differences (per id) with a standard error. Then, if learned > no-use by more than its standard error, the full-round
gate (32 rounds) against 28897 no-use and 28897 force-use.

## 4. What good and bad look like

Good, in this order: the use preference where approved rises above 0.5 and stays (deterministic play fires); the
window eval's learned branch beats no-use on pickups with equal or fewer falls; the residual's approach output grows
and pkon (pickup speed with the approach flag on) rises above pkoff by more than today's ~1.4 u/s; full rounds with the
learned use beat the navigator alone by more than one standard error over 32+ rounds, with 5-15 kicks a round.

Bad: the use preference collapses to 0 again (then the kicks are losing in the drills: check falls and the window
eval); falls after fires above 0.2 a round (then the level rule or the residual is at fault: compare force-use on 28897,
which had 1 fall in 26 kicks); the navigator's own rounds slipping (FREEZE_BASE must stay True; if the base moved, the
run is wrong). Points under 140 in sampled rounds for two hours: write it down, do not change SS_LAM_MAX, the level
rule, USE_EPS or the reward yourself.

## 5. Report

Points by hour (sampled rounds), the diagnostics by hour, every window eval's three branches with paired differences,
every full-round gate with its 32 rounds and uses per round, the best snapshot, and a plain verdict on the question in
section 1. Suggestions, not actions, on: the approvals (the level rule approves ~1 kick a round where the operator
fires 20: what the window evals say about loosening it), and inventory-aware routing (the model never detours for a
Super Speed; the operator takes it at every respawn), which comes after a measured manoeuvre gain.

## 6. Smoke test (10-04 17:12-17:20, one game, training stopped): PASSED

`nav.ss_live_smoke` on nav_v10_28897 (3,000 sampled round decisions): the round instance recorded 1 fire with its
continuation (actual spin [-9.8, 41.6, -0.5] rad/s, 6 chained gems, 32 history observations) to
logs/nav/live_starts_20261004_171221_0.jsonl; `ss_drill_starts --live` made starts_live_dev/eval.json; `ss_drill_eval
--stage 4 --n 3` ran three 12 s window drills on it (GIVEPOW, warm-up, the window): 5-6 of the 6 gems taken, first
pickup 2.24 s, one fall in one of the three (respawned and continued), fired 0 % in deterministic play (the fresh head),
return = pickups (the points reward). The sample start set is only that one fire: run 1 collects the real set.
The worker's default plan is now the run 1 plan ('0:1,1:2,2:2'), so `start_training.ps1` needs no arguments.
