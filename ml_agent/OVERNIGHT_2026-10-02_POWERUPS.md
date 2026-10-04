# Overnight loop, 2026-10-02/03: the navigator learns to use powerups on KOTM

Written for the model that watches this run overnight. Read this file, then `HANDOFF_NAV_TRAINING.md` sections 6-7
(how to run, traps), then log sections 38-40 in `docs/HANDOFF_NAV_TRAINING_LOG.md`. The operator (doug) is asleep.
Standing rules (from the operator, do not break them):

* **Never stop training on your own**, for any metric. Report; only the operator ends a run. Relaunching the SAME
  run after a crash or a hang is allowed and expected (section 4).
* **No code edits beyond this plan without the operator's approval.** The approved scope tonight: the powerup work
  in `POWERUP_PLAN.md` (phase 4 = this run), small fixes to keep the run and the gates going, docs and logs.
* No map hacks, reward = the game's points only, nothing that only works on KOTM. Never train, calibrate or tune on
  Horizon or Archipelago. Never hardcode or replay human movements.
* The operator commits the repo himself. Do not commit. Do not push anything.
* Writing style for the operator: no em dashes, no "drift", no "the one thing", no "this is X, not Y". ASCII only in
  Python log output (cp1252 crashes on arrows).

## 1. What is running

* `nav.train_nav` (PPO, 8 game instances on ports 8888-8895, all KingOfTheMarble_Hunt, 3x game speed), started
  23:36 on 10-02 from `models/nav/nav_latest.pth` at update 29295. Launched by `start_training.ps1` (defaults are
  right; no arguments). The trainer writes `logs/nav/stdout.txt` / `stderr.txt`, saves `nav_latest.pth` every
  update block and numbered `nav_<update>.pth` checkpoints.
* `logs/nav/game_watchdog.sh` (bash, background; output `logs/nav/watchdog_20261002c.out`): relaunches the games
  when their throughput halves (a known trap).
* `logs/nav/game_summary.sh` (bash, background; output `logs/nav/summary_20261002b.out`): every 15 min a KOTM
  summary line and, at each new 50-round high, a snapshot `models/nav/nav_night_<update>_<mean>.pth` (state in
  `logs/nav/night_best.json`). The dashboard is `python -u -m nav.dashboard_nav` on http://localhost:8990 (it reads
  the logs; restart it only if it dies).
* What is new in this run (log 40): the policy has a sixth action element `use` (nav/model.py, ACTION_DIM 6).
  The worker turns it into the bridge words: the held powerup fires (a Super Speed along the commanded direction,
  a Mega via the server command), with nothing held a usable blast meter fires. Since 23:36 a fixed, physics-based
  USE_PRIOR (+10 on the use logit, floor -9) marks the situations where a use can pay: Super Jump or blast at a lip
  with a gap the plain jump cannot cross, Super Speed on an aligned 8 u+ run with the edge ray clear. The learned
  head cancels the prior where it hurts (as the jump prior works). Every GAME line in stdout.txt ends with
  `pow=`: `u<type>` use decisions by held type (1 Super Jump, 2 Super Speed, 6 Mega), `f<type>` the held powerup
  went off (also counts a type switch by pickup), `blast` meter blasts.

## 2. The loop (every 30-60 minutes)

1. `tail -n 3 logs/nav/summary_20261002b.out` (50-round mean, max, falls) and the last GAME lines:
   `grep -h "GAME map" logs/nav/stdout.txt | tail -n 20`. Fires per round by type:
   `grep -h "GAME map" logs/nav/stdout.txt | grep pow= | tail -n 300` and sum the `pow=` fields by thirds (see the
   one-liner at the end of log section 40.2). Also `grep "NAV upd" logs/nav/stdout.txt | tail -n 1`
   (entd = discrete entropy, falls100, speed, sgem).
2. Health: `tasklist | grep -ic marbleblast` must be 8 and the trainer alive (`NAV upd` lines advancing, about one
   per 15-20 s). stderr.txt: ignore torch.load / weights_only warnings; a Traceback means the trainer died
   (section 4).
3. Reference numbers (8-round gates on 10-02, log 37): before powerups the best checkpoint 28897 scored
   160.1 navigator alone (best round 170) and 158.9 as the hybrid; the training-time 50-round mean of that run was
   150.7 (sampled policy, so about 10 below the deterministic gate). Phase 4a (random uses, no prior, 21:52-23:30)
   settled at 50-round means of 136-140 with 2.7 falls a round and about 1 fire per type per round, which was the
   old exploration floor. The run now (prior on) started at about 110 with 6-8 Super Speeds and 8-11 blasts a round.
4. What good looks like: the 50-round mean climbing back through 140 toward 150+ while fires stay well above the
   old floor (the head keeps the uses that pay), falls back toward 1.5 a round, and on the dashboard's "powerup use
   learning" chart the green line (points of rounds with a fire) at or above the grey one.
5. What bad looks like: the mean stuck below 130 after 2 hours, or fires per round falling back to about 1 per type
   (the head cancelling the prior everywhere). Keep training either way (rule 1) and write it down; the operator
   decides in the morning. Do not change USE_PRIOR, USE_BIAS or the approval rules yourself.
6. At every new night snapshot above the previous high (and at least once, at about 3-4 h in), run a gate
   (section 3) and log it.
7. Append findings to log section 40 (`docs/HANDOFF_NAV_TRAINING_LOG.md`, "40.x" subsections with the time), not to
   this file. Update `HANDOFF_NAV_TRAINING.md` section 1 only at the end of the night.

## 3. Gates (8 deterministic rounds; the GPU takes at most 4 game processes beside the trainer, so run 4 games x 2 rounds)

The scripts live in the scratchpad of the previous session; copies are in `logs/nav/gate_scripts/` (gate_snap.ps1 =
the hybrid, gate_navonly.ps1 = navigator alone, kill_tag.ps1). Both take `-Ckpt <pth> -Tag <tag> -Port0 <port>`
and write `logs/learned_nav/<tag><k>_KingOfTheMarble_Hunt.txt` plus `logs/learned_nav/rounds/<tag><k>_*.jsonl`
(one JSON per round: points, falls, pow_uses, policy_uses, policy_use_log, oracle stats). Use ports 9901+ for
one gate and 9911+ for the next; tags like `g9a_nav`, `g9a_hyb`. Example from bash:

    powershell -ExecutionPolicy Bypass -File logs/nav/gate_scripts/gate_navonly.ps1 -Ckpt models/nav/nav_night_<u>_<m>.pth -Tag g9a_nav -Port0 9901

Wait until each of the 4 logs has "round 2: points"; then
`python - <<EOF` with json over `logs/learned_nav/rounds/g9a_nav*.jsonl` to print per-round points, falls,
policy_uses (the loop used in log 40). Compare with 160.1 / 170 (navigator alone) and 158.9 / 164 (hybrid).
A gate that beats 160.1 on the mean or 170 on the best round is the news the operator wants; copy that checkpoint to
`models/nav/nav_v7pow_best_<update>.pth`. The games of a gate must be killed when done
(`kill_tag.ps1 -Tag g9a`), and the trainer's 8 games must stay up (check the count).

## 4. If something breaks

* Trainer dead (Traceback in stderr.txt, no NAV lines): kill `run_game_loop`, `start_training` and every
  `marbleblast_mbx` (PowerShell, exclude your own process), copy stdout.txt to `stdout_<time>.txt`, then from
  `ml_agent` in PowerShell: `Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','start_training.ps1' -WorkingDirectory <ml_agent> -WindowStyle Hidden`.
  It resumes from nav_latest.pth. Restart the watchdog afterwards (`nohup bash logs/nav/game_watchdog.sh > logs/nav/watchdog_<time>.out 2>&1 &`).
* Games at 4 instead of 8, or rtf under 2 on all of them: the watchdog should relaunch them; if it does not within
  10 minutes, do the trainer restart above.
* Everything frozen with the games up: the screen may be locked (every bridge reply late by ~200 ms); note it, the
  operator must unlock. Do not fight it.
* OneDrive refuses python writes to existing repo files (Errno 22): write a new file and os.replace it, or use the
  editor tool.
* Checkpoints saved tonight need the current nav/model.py (use head); older checkpoints still load.

## 5. Morning report (write to log 40.x and tell the operator)

Points by hour (50-round means), falls a round, fires per type per round over the night, every gate's 8 rounds,
the best snapshot and whether any round beat 170 or any 8-round mean beat 160.1. Say plainly whether the policy
learned to use powerups (fires kept above the floor AND points up) or learned to cancel the prior (fires back to
about 1 per type) or neither. Suggestions for the operator's decision, not actions.
