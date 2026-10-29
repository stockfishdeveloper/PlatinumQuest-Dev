# Ten-map points: takeover of the overnight queue (2026-10-27)

Continues `docs/OVERNIGHT_POINTS_2026-10-26.md` (the other agent's log; legacy-encoded, edit only via shell). Same method:
native matched campaigns (two games, one per arm, 8 forecast workers each, fixed .172 s allowance, order gate .35, all ten
maps, exploratory seed then a fresh confirmation seed, layout / initial motion / actual GO / source / workload audits).

## Where the time goes (40 baseline native rounds: launch_* and bump_window_* baselines, read-only)

| | per round |
|---|---|
| approaches (previous group's last pickup to the next group's first) | 124 s of ~247 s |
| inside groups | ~83 pickups, 1.39 s per pickup |
| marble still (< 0.75 u) beyond 1.5 s, countdown and round end excluded | ~2 s (Cragmire ~10, Fractured ~4, Vortex ~3) |
| no 1 u of route progress for > 1 s (`recovery.stalled_s`) | 24 s (Cragmire 55, Vortex 35, TreasureBox 33, Concentric 31) |
| stall rule firings (5 s) / 8 s rescues | 1.02 / 0.45 |

Approach speed (straight line / time): Bowl 14.6 u/s, Cube 10.8, Sprawl 9.2, Duplex 9.3, TreasureBox 8.0, Fractured 6.0,
Vortex 6.1, Concentric 5.8, Cragmire 3.6.

Inside groups the marble holds ~7 u/s between gems straight ahead (it neither brakes nor accelerates after the pickup);
sharp turns (90-180 deg) cost ~2.15 s per pickup against ~0.96 s straight ahead. Speed caps were tested before
(13.5 u/s tied in game, gem_demo.TARGET_SPEED_CAP notes); not repeated.

## Stalls the planner knows about

Forecasts match the game exactly in every stall examined (predicted state = actual state), so these are choices:

1. A kept plan that waits at a gap run-up point (Vortex launch_explore 856201943 t=41: 6.2 s still at the staging point,
   cost -99.8 "arrived"; Fractured launch_confirm 317594681 t=46: 4.2 s pinned at a wall, the gap run controller waiting for
   a run-up speed it cannot reach). The v47 rule keeps a running plan unless a new one collects a gem.
2. A hold in a wall pocket under a raised gem (Cragmire bump_window_explore 598172643 t=121.5 and launch_confirm 317594681
   t=230.7: 8.0 / 7.4 s of safe_hold at (24.2, -10.7, 6.9) until the 8 s rescue respawn).
3. An opportunistic detour to a powerup 17 u below, re-searched EVERY decision with its jump 9 replies ahead, so the jump
   never fires (Cragmire 842619357 t=5.0 and t=126, 598172643 t=86.6: 7-8.6 s each). Cause: the plan lock is dropped every
   decision because the lock check in ContinuousController._search knows only gem and staged goals (`held` lookup), never a
   detour item; the detour accepts any plan with route progress > .3 u.

## Prepared changes (scratchpad patch scripts, default OFF, applied between campaigns)

- `patch_staging_giveup_v2.py`: GAP_STAGING_GIVE_UP v2 (24 decisions at the run-up point, the count restarts away from it;
  after the give-up the gem is staged again without the gap instead of chased). v1 pair: +0.2 a round (Vortex +27,
  Fractured -18 from a workable gap dropped while arriving). Flag `--gap-staging-give-up`.
- `patch_stall_still.py`: STALL_STILL_S (flag `--stall-still` = 1.5 s): the existing stall response (fetch a Super Jump /
  Helicopter for a raised gem, else skip the gem and its platform for 20 s) runs when the marble has made no route progress
  AND stayed within .75 u (2 u vertically) for 1.5 s; a detour powerup it was heading for is banned from detours for 20 s.
- `patch_detour_flag.py`: `--detour-by-time` for the existing DETOUR_BY_TIME.
- `patch_sj_use_gain.py`: search.SJ_USE_GAIN names the other agent's additive Super Jump completion-time guard
  (apply_use_gain) with `--sj-use-gain` / `--no-sj-use-gain`; its default follows the sj_native result.

## Queue operation

- 10:27 run_sj_native.py (guard off = "historical" vs production = "fixed"), both seeds regardless.
- 11:05 the waiting run_powerup_order_native.py (no scoring yet) was stopped; its plan/status are in
  `dynamics_reports/overnight_20261026/powerup_order_deferred_20261027/`; relaunch the same driver later.
- Two campaigns may run side by side (CPU ~25 % with one): the other agent's drivers use ports 9440/9444,
  `run_flag_native.py --port-base 9460` uses 9460/9464. Each campaign's arms still run together.
- `run_flag_native.py` (dynamics_reports/overnight_20261026): generic matched native flag campaign driver.

## sj_native closed (12:00): the Super Jump time guard stays ON

20 matched native pairs, guard off ("historical") vs production ("fixed", guard on): seed 471392681 127.4 -> 132.2,
seed 803156429 (ports swapped) +1.4 a round; combined +3.1 a round for the guard. Per-map swings were +-30 to +-54 in single
rounds (TreasureBox 124 -> 178, Cragmire 139 -> 105 and 125 -> 154, Vortex 98 -> 66), the usual divergence noise.
search.SJ_USE_GAIN now names the guard (default True; --no-sj-use-gain turns it off).
New best: Cragmire 154 (1 fall; sj_native_1_fixed seed 803156429) -> docs/BEST_REAL_SCORES_2026-10-27.csv, sheet updated
and read back (HISTORY_SEED extended to 10-26 in tools/update_best_scores_sheet.py).
12:00 the four patches were applied to the repo (all default off / unchanged; 62 tests pass; pre-patch copies in the session
scratchpad backup_pre_takeover_patches_1200).

## Falls come from the A2 actor after search budget overruns (12:40)

40 baseline native rounds: 34 of 37 falls end an A2 actor streak (median 1.5 s) that a search budget overrun started
(choice reason planning_time_budget at the .172 s allowance; the other 3 began in crossing brakes). Overruns a round: Vortex 56
(3.75 falls), Cube 36 (2.5), Duplex 23 (1.5), Cragmire 17 (1.0), the rest 2-9 (about 0 falls). Each overrun hands the marble
to A2 for 3 decisions (the overrun plus the 2-decision search cooldown); 230 of the 575 overruns came right after a planned
decision. On Vortex / Cube / Duplex half of the streak-start overruns happen in the air (searches there take 0.23 s median,
0.33 s p90).
Prepared (scratchpad patch_budget_keep.py, default off, `--budget-keep`): on an overrun, the remainder of the plan running at
the start of the decision is forecast again with the same gates (the existing keep-plan code) and kept instead of A2, with no
cooldown.

## Stall screen in the sim (12:11)

58 recorded still stalls (time to the first pickup, cap 20 s, a fall = 25 s): give-up v2 alone -0.53 s a stall (8 better,
0 worse); STALL_STILL v1 alone -0.22 s (9 better, 8 worse); both -0.55 s (12 / 7). The gem skip at holds near gems lost 16 s
over 7 cases (TreasureBox: the platform skipped while the marble got up there; Vortex: A2 drove off after the skip); the
detour ban won 22 s over the 2 Cragmire detour pins. Native giveup2 (give-up v2 alone) launched 12:12. STALL_STILL v2 (the
detour ban only, scratchpad patch_stall_still_v2.py) is prepared.

Two native campaigns cannot run together: one is ~38 GB of commit (16 forecast workers at ~2 GB), the limit is 63.6 GB;
the 12:12 powerup_order relaunch beside giveup2 failed at its first round ("paging file is too small"), no round scored,
archived in dynamics_reports/overnight_20261026/powerup_order_failed_commit_limit_20261027/.

## giveup2 closed (13:37): GAP_STAGING_GIVE_UP v2 ON by default

Native, give-up v2 alone (`--gap-staging-give-up`), 20 matched pairs: seed 688646045 134.9 -> 141.3 (+6.4), seed 662636522
(ports swapped) -2.8; combined +1.8 a round. The map deltas are dominated by divergence (Fractured explore split at 9.9 s in a
cluster, Cragmire confirm at the first decision; neither involved a crossing). Mechanism check: the baseline arm had 3 stretches
of >= 24 decisions parked at a gap staging point (Vortex 2, TreasureBox 1), the candidate none. With the sim screen (58
recorded stalls, -0.53 s each, 8 better / 0 worse) the give-up is promoted: crossings.GAP_STAGING_GIVE_UP = True (13:38),
control flag `--no-gap-staging-give-up`. This departs from the other agent's rule (fresh seed must be positive): the give-up
acts in ~3 of 20 rounds, so whole-round deltas cannot resolve it; the mechanism evidence is clean.
13:38 also applied (default off): STALL_STILL v2 (detour ban only) and BUDGET_KEEP (`--budget-keep`).

## budgetkeep1 (14:28): the mechanism works, the points are noise so far

`--budget-keep`, seed 444609552, 10 pairs: 135.9 -> 134.7 (-1.2). Mechanism: falls 6 -> 0, A2 decisions 358 -> 36, stalled
25.6 -> 22.9 s a round, rescues 4 -> 5; kept plans 191 (4 rejected). Vortex 116 -> 122 (falls 3 -> 0, A2 116 -> 9), Cragmire
104 -> 136 (A2 48 -> 0), Fractured 110 -> 114 (falls 1 -> 0). The four losing maps (Duplex 153 -> 118, Sprawl -7, Concentric
-6, TreasureBox -5) diverged at 1.7-13.6 s on ordinary decisions where no overrun or kept plan was involved: divergence noise.
The driver stops after a negative pilot; budgetkeep2 (seeds 618695228 / 314717812) launched 14:29 for more pairs.
Noise floor: both arms of a native pair diverge within seconds from wall-clock search timing alone, so a treatment that acts in
a few windows per round cannot be resolved by whole-round points from 20 pairs; mechanism counts (falls, A2 decisions,
stalls, activations) are the sharper evidence.

## budget-keep closed (15:11): stays OFF

budgetkeep2 (seed 618695228): 10 pairs -3.2 a round; with budgetkeep1 (-1.2) 20 pairs -2.2 a round. Mechanism over the 20 pairs:
falls 13 -> 4, A2 decisions roughly halved or better on every overrun-heavy map, stalled time down on most maps. The points do not
follow: a fall here costs about as much as an ordinary approach (respawn to next pickup ~8 s median, 40 baseline rounds), and
the losing maps diverged on ordinary decisions. BUDGET_KEEP stays off (flag kept for later bundles).
Side finding (budgetkeep2 Fractured candidate, 74-82 s): give-up v2 (now ON) fired at the familiar run-up point (2, 9.4) after 2 s
and re-staged a long way around; the 8 s no-progress rescue then respawned the marble mid-trip because the progress route still
measures toward the gem through the dropped gap. Candidate fix: treat a give-up re-stage as progress for the rescue monitor.
New best: Duplex 166 (0 falls; budgetkeep2_explore_candidate seed 618695228; baseline arm 160) -> CSV 10-27 + sheet (read back).
15:12 launch3 (`--launch-completion`, seeds 564785272 / 834393685): third native seed set for LAUNCH_COMPLETE (earlier +2.7 / +3.7).

## launch3 closed (16:38): LAUNCH_COMPLETE ON by default

`--launch-completion`, seeds 564785272 (+6.9 a round; Vortex 103 -> 147, Cragmire 105 -> 136) and 834393685 (ports swapped,
+3.3); combined +5.1 a round. With the other agent's two native seeds (+2.7, +3.7) that is 4 of 4 seeds positive, about +4 a round
over 40 pairs; its one live 1x seed was -2.9 (10 pairs). Promoted 16:38: search.LAUNCH_COMPLETE = True, control
`--no-launch-completion`; the main() / realtime settings lines used to set the attribute from the flag alone, they now keep the
module default unless the control flag is given. New best Vortex 147 (0 falls, launch3_explore_candidate seed 564785272) -> CSV
10-27 + sheet. Note for the other agent's run_matched_impulse_native.py: its baseline check requires LAUNCH_COMPLETE off; with the new
default its baseline must pass --no-launch-completion, or (the relevant question now) compare against production.
16:39 powerup_order relaunched (the other agent's driver, unchanged, seeds 487913621 / 109736581).
Production defaults now: launch completion ON, give-up v2 ON, Super Jump time guard ON; budget-keep, stall-still v2,
detour-by-time OFF.

## powerup_order closed (17:22): rejected

The other agent's driver, unchanged (seed 487913621, its production now includes launch completion and give-up v2): 10 pairs
-5.3 a round (Cragmire -25, Sprawl -13, Cube -12, TreasureBox -18; Vortex +15). Exploratory mean negative, so no confirmation
seed by its own rule. The candidate (nearest-first powerup slots in navigator observations) was never on in production.
17:22 still2 (`--stall-still`, STALL_STILL v2 = detour ban only; seeds 775639187 / 647506797).

17:48 still2 failed at its 6th map (Cragmire) in both arms with MemoryError although it ran alone: the commit limit is 63.6 GB
(RAM 31.8 GB + a system-managed 32.5 GB pagefile, 26 GB free on C:), other programs hold ~22 GB and one campaign ~38 GB plus the
worker restart at each map. Maps 1-5 were valid (the detour ban fired 0 times there). Archived in
dynamics_reports/overnight_20261026/still2_failed_memory_20261027/; retried as still2b (same seeds) at 17:49.

## still2b pilot (18:31): a near-null treatment scored +6.3

still2b seed 775639187: 10 pairs +6.3 a round, but STALL_STILL v2 fired ONCE in the 10 candidate rounds (Cragmire; the detour
chases this seed were short). The +6.3 is therefore almost pure divergence: an A/A-like measurement of the native noise. Per-map
pair differences have an SD of ~18, so a 10-pair seed mean has an SD of ~5.7 and a 20-pair campaign ~4. For comparison, launch
completion (promoted at 16:38) acts in every launch evaluation and scored +2.7 / +3.7 / +6.9 / +3.3 on four seeds (mean +4.2,
sign test 4/4, p ~ 0.06 one-sided): positive but not certain; a live check is the next useful evidence.
The confirmation seed (647506797) runs; a promotion of STALL_STILL v2 would rest on the mechanism (sim: the detour pins end) and
on no harm, not on this score.

## Proposal (not built): a deterministic native mode for low-noise A/B tests

Why: both arms of a native pair diverge within seconds from wall-clock search timing alone (the .172 s budget lets a different
set of candidates finish in each arm), so a change that acts a few times a round is invisible under ~+-4 a round of noise from
20 pairs (still2b: +6.3 with ONE activation in 10 rounds). The game already runs in lockstep; only the planner is
nondeterministic.
What: an opt-in budget mode (default off, production unchanged) where the parallel evaluator waits for every candidate, each
worker returns its physics-step count, and the parent charges a virtual clock in spec order (the sim_bench model: NATIVE_STEP_US
per step), dropping the candidates past the virtual deadline; the parent-side stages read the same virtual clock. Then two arms
are identical until the change first acts, rounds where it never acts score exactly the same, and the paired variance comes
only from rounds where it acts.
Cost: parallel_search.py + the bench (one flag), an A/A check (two identical arms must match to the decision), then the
calibration of the step cost so the virtual budget matches the live .172 s throughput. Caveat: timing-driven effects (budget
overruns, budget-keep) still need the wall-clock mode.

## still2b closed (19:14): STALL_STILL v2 ON by default; the noise lesson

still2b: seed 775639187 +6.3, seed 647506797 +6.0, combined +6.15 a round over 20 pairs, yet the detour ban fired once in the 20
candidate rounds (stall_still_detour 1, stall_detour_banned 0); the baseline also showed more stall rule firings (25 vs 11) and
rescues (14 vs 5) by divergence alone. Two seeds at +6 from a treatment that almost never acts: the native noise is larger than
the ~5.7 per seed estimated above (or this pair was a ~1-in-40 draw). Treat single native campaigns as weak evidence.
Promoted 19:15 as a low-risk fix of a known failure (the Cragmire Super Jump chase at (0, 7.5, -4.4); sim: the pins end, 84 starts
-0.46 s each, 6 better / 3 worse): reliability_eval.STALL_STILL_S = 1.5, control `--no-stall-still`. The stillness trail now
skips states without a position (a test fixture had none; real states always do).
Production defaults now: launch completion, give-up v2, the detour ban and the Super Jump time guard ON; budget-keep and
detour-by-time OFF.

## Overnight queue (20:00)

roll1 (`--roll-launches`, seeds 551458782 / 769251468): pilot +4.3 a round, but roll launches were chosen only 4 times in the 10
candidate rounds (67 ordinary take-off launches): another low-activation result. Confirmation running.
dynamics_reports/overnight_20261026/run_queue_takeover.py (detached, log run_queue_takeover.log) then runs, one campaign at a time
with 60 s between: the other agent's blocked_order, landshort1 (`--land-short`, seeds 153748520 / 939487706), final_backup,
follow_backup, query_route, powerup_timers, sj_backup_reference (their drivers unchanged). A HOLD file in that folder pauses the
queue between campaigns (ignored when older than 30 min); killing the runner stops the queue after the current campaign.
matched_impulse is left out: its baseline check requires LAUNCH_COMPLETE off, which is no longer production.

## roll1 closed (20:39): roll launches stay OFF

`--roll-launches` on top of launch completion: seed 551458782 +4.3 (4 roll launches chosen in 10 rounds), seed 769251468 -4.6
(Vortex 117 -> 87, Duplex 131 -> 117); combined -0.15 a round. POW_LAUNCH_ROLL stays off. The queue runner starts blocked_order next.

## blocked_order closed (21:23): rejected

The other agent's driver, unchanged (seed 681297413; production now with launch completion, give-up v2 and the detour ban):
10 pairs -3.3 a round (Duplex 162 -> 138, Sprawl -8, KOTM -9, Concentric -13; Cube +26). Negative exploratory mean, so no
confirmation by its own rule; the alternative-only target selection was never on in production. The queue moves to landshort1.

## Operator stop (21:3x)

Operator: "after this current experiment, report results then stop all training". The queue runner was stopped immediately;
landshort1 finishes; nothing further starts. Not run: final_backup, follow_backup, query_route, powerup_timers, sj_backup_reference
(their drivers are ready and unchanged), matched_impulse (needs re-aiming for LAUNCH_COMPLETE on).

## landshort1 closed (22:08); all work stopped at the operator's request

`--land-short` (seed 153748520): 10 pairs -3.7 a round (Duplex 144 -> 117, Cube 149 -> 131; Sprawl +13), no confirmation by the
driver's rule; LAND_SHORT stays off. 22:10: no campaign, game or watcher process running; nothing queued.

Day summary (native, .172 s, two games x 8 workers; combined over the seeds run):
| test | result a round | decision |
|---|---|---|
| sj_native (Super Jump time guard, guard on vs off) | +3.1 (20 pairs) | stays ON (SJ_USE_GAIN) |
| giveup2 (gap give-up v2) | +1.8 (+6.4 / -2.8) | ON (mechanism: 3 parked stretches -> 0) |
| budgetkeep1 + 2 | -2.2 (20 pairs; falls 13 -> 4) | off |
| launch3 (launch completion) | +5.1 (+6.9 / +3.3; with the other agent's +2.7 / +3.7) | ON |
| powerup_order (other agent) | -5.3 (pilot) | off |
| still2b (detour ban v2) | +6.15 (1 activation: noise) | ON (low-risk fix of the Cragmire chase) |
| roll1 (roll launches) | -0.15 (+4.3 / -4.6) | off |
| blocked_order (other agent) | -3.3 (pilot) | off |
| landshort1 (land short) | -3.7 (pilot) | off |
Production (uncommitted): search.LAUNCH_COMPLETE, crossings.GAP_STAGING_GIVE_UP (v2), reliability_eval.STALL_STILL_S 1.5,
search.SJ_USE_GAIN ON; control flags --no-launch-completion / --no-gap-staging-give-up / --no-stall-still / --no-sj-use-gain.
New bests today: Duplex 166, Cragmire 154, Vortex 147 (sheet + docs/BEST_REAL_SCORES_2026-10-27.csv).
Not run: final_backup, follow_backup, query_route, powerup_timers, sj_backup_reference (drivers ready), matched_impulse (needs
re-aiming). Open items: the commit-limit headroom (Firefox / pagefile), the deterministic native mode proposal, a live check of
launch completion, the give-up + 8 s rescue interaction (2 of 5 give-ups followed by a respawn in 200 rounds).
