# Overnight loop, 2026-10-03/04: the navigator learns the Super Speed turn on KOTM

Written for the model that watches this run overnight. Read this file first, then `HANDOFF_NAV_TRAINING.md`
sections 6-7 (how to run, traps), then log sections 40.11-40.22 in `docs/HANDOFF_NAV_TRAINING_LOG.md` (today's
work, newest last). The operator (doug) is asleep. The previous overnight handoff, `OVERNIGHT_2026-10-02_POWERUPS.md`,
is superseded by this one; its sections 2-5 (the loop, gates, restarts, the morning report) still apply word for word
except where this file says otherwise.

Standing rules (from the operator, do not break them):

* **Never stop training on your own**, for any metric. Report; only the operator ends a run. Relaunching the SAME
  run after a crash or a hang is allowed and expected.
* **No code edits beyond the approved scope**: the Super Speed work described here (observation flags, the use
  prior, the worker's diagnostics), small fixes that keep the run and the gates going, docs and logs. Anything
  else: write it up as a proposal for the morning.
* **No map hacks, ever.** The operator was explicit tonight: "do NOT make a hack that will only work on this map."
  Everything in the Super Speed rule is physics (the kick, the measured brake) plus the terrain map along a line;
  no coordinates, no item positions, no KOTM geometry in the policy path. Keep it so. Reward = the game's points
  only. Never train, calibrate or tune on Horizon or Archipelago. Human demos are for measuring targets, never a
  template to replay.
* The operator commits the repo himself. Do not commit, do not push.
* Writing style for the operator: no em dashes, no "drift", no "the one thing", no "this is X, not Y". ASCII only in
  Python log output.

## 1. What we are trying to make the model do

The operator's demo (log 40.19, `demos/demo_20261003_211632.npz`, 153 points): a Super Speed is fired 0.1-0.2 s
AFTER a pickup, aimed so the velocity turns 90-150 deg onto the line to the next gem, resolving to 11-23 u/s
(median 17), 20 times a round, braking into the next gem, no falls. The operator wants ONE model that goes into a
gem fast because it knows it holds a Super Speed and can redirect at the pickup, instead of braking and curving
round the gem as if it had none.

What the model has for that (all map-independent):

* Observation V9 (`nav/obs.py`, VEC_DIM 91), while a Super Speed is held: `vec[VEC_POW + 28..29]` = the kick aim
  solved from physics (`ss_redirect_aim`: v + 25 a parallel to the line to the CURRENT waypoint; when the marble
  moves away faster than 25 u/s the aim is the pure brake, 40.21); `vec[VEC_POW + 26]` = that kick is approved:
  resolved speed <= 24 u/s (SS_LAM_MAX, 40.22), floor along the line for the stopping distance at the measured
  14 u/s^2 or the next gem on the line with 6 u of floor past it; `vec[VEC_POW + 27]` = approach information:
  "a redirect at the gem ahead toward the next gem would be survivable".
* `nav/model.py` use_prior: +11 on the use logit where [26] is on (held, on the floor, waypoint >= 8 u); the head's
  floor there is USE_PRIOR_FLOOR -1 (so approved kicks are sampled >= 27 %, never deterministic until learned);
  elsewhere -9. Super Jump / blast branches as before (a lip with an uncrossable gap the boost covers).
* The worker (`nav/vec_worker.py`) fires a Super Speed along `obs.ss_aim(vec)` (= [28-29]); hybrid.py and
  real_run.py do the same.

## 2. What is running

`nav.train_nav` restarted 22:56 on 10-03 from `nav_latest.pth` at update 30420 (lineage: 28897 V7 best, migrated
to V8 then V9 with zero columns, trained since 14:28 through the day's changes), 8 games on KOTM, launcher
`start_training.ps1` (defaults right, no arguments). Helpers: `logs/nav/game_watchdog.sh` (output
`logs/nav/watchdog_20261003k.out`), `logs/nav/game_summary.sh` (`logs/nav/summary_20261002b.out`; snapshots
`models/nav/nav_night_<update>_<mean>.pth` at new 50-round highs; `night_best.json` currently 144.8 at 30097),
dashboard http://localhost:8990 (two powerup charts). The deterministic reference without uses: 28897 at 160.1
(best round 170); phase 4 gates so far 132-156 (40.7, 40.17).

Every GAME line in `logs/nav/stdout.txt` ends with the diagnostics (all per round):
`pow=` with `ap2` approved Super Speed decisions, `u2` run uses, `r2` turn uses (velocity > 30 deg off the
waypoint line), `f2` Super Speed fires (held -> none without a fall; also counts a type switch), `rhit`/`rmiss`
(waypoint taken within 1.5 s of a turn's fire or not), `fss` falls within 3 s of a fire, `ksA/B/C` fires by the
resolved speed 3 decisions later (< 18 / 18-24 / > 24 u/s) and `kfA/B/C` the falls per bucket, `f1/u1` Super
Jump, `f6/u6` Mega, `blast`; then `pkon/pkoff/pknone` = mean pickup speed / count with a Super Speed held and the
approach flag on, held with it off, none held; `rwfire/rwall` = reward per decision within 30 decisions of a fire
vs all decisions.

## 3. The loop (every 45-60 minutes)

1. Summary: `tail -n 2 logs/nav/summary_20261002b.out`; trainer line `grep "NAV upd" logs/nav/stdout.txt | tail -n 1`.
2. The round diagnostics by quarters of the rounds since the last restart (the one-liner in log 40.20 / 40.18:
   grep the GAME lines with `rwall`, parse `pow=` into a dict, sum by quarter). Report per round: points, falls,
   f2, r2, rhit, rmiss, fss, ksA/B/C with kfA/B/C, pickup speed on/off/none, rwfire vs rwall.
3. Health: `tasklist | grep -ic marbleblast` = 8; NAV lines advancing every 15-20 s; stderr.txt free of Tracebacks
   (torch.load warnings are noise). Restart recipe in `OVERNIGHT_2026-10-02_POWERUPS.md` section 4 (kill
   `nav.train_nav`, `run_game_loop`, `start_training`, every `marbleblast_mbx` and the watchdog; copy stdout.txt
   aside; relaunch `start_training.ps1`; restart the watchdog with nohup). `logs/nav/gate_scripts/kill_proc.ps1 -Rx
   <regex>` kills by command line and excludes itself.
4. Gates: `bash logs/nav/gate_scripts/rr_gate.sh 4 9961 <ckpt> <tag_a> 9962 <ckpt> <tag_b>` (CPU real_run, two
   games of 4 rounds, ~3 min) then `python logs/nav/gate_scripts/rr_read.py <tag_a> <tag_b>`. Copy nav_latest to
   `models/nav/nav_p4f_<update>.pth` first (the trainer overwrites nav_latest). Run one at each new night snapshot
   and at least once around 03:00 and 06:00. Deterministic play fires a use only where the head has learned it,
   so the gate's `use decisions/round` says whether anything has been learned; compare points with 160.1 / 170.
5. Log every check as a "40.x" subsection with the time in `docs/HANDOFF_NAV_TRAINING_LOG.md` (append; use the Edit
   tool or write-new-then-os.replace: OneDrive refuses python writes to existing files). Update
   `HANDOFF_NAV_TRAINING.md` section 1 only at the end of the night.

## 4. What good and bad look like

Good (in this order, hours apart): turn hit rate rising from ~70 % toward the demo's 100 % and `fss` falling toward
0 (the follow-up control is being learned); then `rwfire` reaching `rwall`; then `pkon` rising above `pkoff`
by more than today's ~1.4 u/s (the fast approach forming); then training points above 142 and a gate with
Super Speed use decisions > 0 in deterministic play. Bad: `fss` staying at 0.7+ a round or points under 130 for
two hours (then write it down; do not change USE_PRIOR_FLOOR, SS_LAM_MAX or the approvals yourself). `ksC`
should be rare now (the cap): if it is not, note the counts for the morning.

## 5. What was tried today (so you do not repeat it; details in the log)

* Random uses, no prior (40.1): the head learned "do not" in 20 minutes. Prior with no cancel (40.2-40.7): forced
  uses all night, gates 132-138. Floor fix (40.8-40.10): the head cancelled everything; 147.9. Fresh start from
  28897 with a run-clear check (40.11), the pre-pickup redirect (40.12-40.14: missed the gem 40 %), the
  post-pickup rule (40.15-40.16: neutral), gate 155.9 with no learned use (40.17), braking measured (40.18:
  7.7 / 14.2 u/s^2), the demo (40.19), floor -1 (40.20: more fires, a fifth fell), the reversal brake (40.21),
  the resolved-speed cap (40.22, running now).
* Not built: powerup items as route-chooser waypoints with a time credit (the model never detours for a Super
  Speed; the operator takes it at every respawn). A proposal for the morning if the kicks start paying.

## 6. Morning report

Points by hour (50-round means), falls, the diagnostics above by hour, every gate's 8 rounds with its use
decisions, the best snapshot, and a plain verdict: did the turn hit rate and the reward after fires rise, and did
the pickup speed with the approach flag on rise with them? Suggestions for the operator, not actions.
