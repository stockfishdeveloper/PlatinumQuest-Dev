# Night 2026-10-28/29: new ideas for the ten-map average

Operator 10-28 23:1x: "ok iterate all night on some new ideas for increasing average points". Rule added at 23:2x: only ideas
that work in a real lobby game (no next-spawn prediction; memory feedback-lobby-valid-ideas-only).

## Summary (kept current through the night)

- **Nothing from tonight is in production.** At 07:25 the defaults are the 10-28 production again; every flag below is off by
  default and kept for later tests.
- **QUERY_ROUTE_COPY: promoted at 02:58, rolled back to OFF at 07:25.** The other agent's query_route prototype as code
  (powerup-detour queries on a private copy of the route) scored +3.0 a round over 20 native pairs with HOLD_RESEARCH on
  (query_route6), but its confirmation on fresh seeds with HOLD_RESEARCH off (qr3, production vs `--no-query-route-copy`)
  scored -4.35 over 20 pairs (seeds -3.8 / -4.9, se 2.4). Over the 40 pairs about -0.7: no gain shown.
  `--query-route-copy` keeps it available.
- **Main lesson: a 20-pair campaign cannot resolve the effects tested tonight** (see "How far a campaign can be trusted"
  below): ten campaigns since 10-26 whose first seed came out positive (mean +4.4) averaged -0.25 on the fresh seed.
- **HOLD_RESEARCH: promoted at 01:30, rolled back to OFF at 05:52.** When a search runs out of time the marble falls back to
  a "safe hold" that ran its whole 2.6 s plan unsearched; re-searching it every decision cut hold time 4.5 -> 0.9 s a round
  and scored +3.05 over 20 pairs (hold1b), but the confirmation on fresh seeds with QUERY_ROUTE_COPY in both arms (hold3)
  scored -5.7 over 20 pairs (se 1.9). Locally the candidate needs 10.1 s from a hold to the next pickup in both campaigns,
  while the baseline needed 12.6 s in hold1b and 8.3 s in hold3: with the route fix the plain hold no longer costs what it
  did. `--hold-research` keeps it available.
- New bests Fractured 134 (hold1b candidate) and Duplex 176 (query_route6 baseline arm); sheet and CSV 10-29 updated.
- Tried in the simulator and set aside (all lobby-valid, all default off): Super Speed turn kicks with a brake, re-searching
  kept pickup plans, the kick lead aim (DIRECT_LEAD), a cost-based switch for kept plans, a "boxed in -> respawn" rule.
- Found, not yet fixed: the stall rule counts hops as progress (Cragmire pit); a Cragmire wall pocket where every search
  overruns; native campaigns at 8 workers page the machine (tonight: 6).
- Not adopted: the other agent's final_backup prototype (+0.5 over 20 pairs).
- 08:13 order1 (the same route fix for the cluster order choice) ended on its exploratory seed: the Concentric pair failed
  the matched-start audit (layout same, initial motion and GO observation different), so the campaign stopped before the
  confirmation seed; the nine valid pairs: +2.3 a round (noise). No new map best.
- **Operator 08:1x: "after the current experiment stop training and wait for my direction".** Cancelled: the PENALIZED_SWITCH_U
  sim screen, the det_aa A/A check, penal1, follow_backup6. 08:13 the queue runner was stopped and the HOLD file removed:
  nothing runs (no python, no game).

Method as on 10-27 (docs/TAKEOVER_POINTS_2026-10-27.md): native matched campaigns, two games (8 forecast workers each on
10-27, 6 tonight), fixed
.172 s allowance, all ten maps; tonight every campaign runs BOTH seeds (run_flag_native.py --always-confirm, 20 pairs), because
one seed has an SD of ~6 a round. Mechanism counts are read beside the points (scratchpad mech_audit.py).

## Where production loses time (60 production native rounds of 10-27, read-only)

- Legs with more than 4 s of excess time (time beyond straight distance / 9 u/s) lose ~27 s a round over the ten maps
  (Cragmire 85, Vortex 56, Concentric 41, Duplex 24, TreasureBox 24, Fractured 20); they are also most of the noise.
- Safe holds: every one starts at a search overrun (planning_time_budget) and runs its whole forecast UNSEARCHED (41 replies,
  2.62 s; the retained-plan branch exempts label 'safe_hold' from the v47 re-search). Cragmire 16 s a round, Vortex 10,
  Concentric 3.4, Duplex 2.2; the marble brakes to ~1 u/s, and on Vortex the first search after a hold found a plan in most cases.
- Human TreasureBox demo (217 points, 10-24) against six production TreasureBox rounds (148-199): legs >= 20 u at 11.7 u/s
  against 7.7-8.7 (5.9 s against 6.6-8.6 s a leg); legs < 20 u 0.84 s against 0.92-1.53 s; on the long legs the human spends
  38 % of the time above 15 u/s (the planner 14 %) with 21 powerup uses in 17 legs.
- Super Speed is held for a median 16 s on KOTM, 11 s on Cragmire, 6 s on Cube and Concentric (1.3-3.5 s elsewhere); the
  human fires within 0.2 s of a pickup (memory superspeed-human-technique).
- The Blast meter sits full 83 % of a KOTM round (1.5 blasts a round), 38 % on Cube; not pursued (operator 10-18: "Blast only
  when necessary").
- Not worth a rule: powerup detours (failed detour goal time 3.7 s a round; detours past 2-3 s still succeed 81-91 %), a leg
  timeout (in 22 of 30 long single-gem chases no other gem was visible).

## Experiments

| label | idea | flag | result | decision |
|---|---|---|---|---|
| hold1 | safe holds searched again every decision; any useful plan replaces the hold; an overrun keeps it | `--hold-research` (HOLD_RESEARCH) | 8 workers: failed at map 6 (Cragmire): both games disconnected at 23:58 under heavy paging (Windows dropped C: shadow copies for IO load); maps 1-5 valid, +0.4 a round | rerun as hold1b |
| hold1b | same | same, 6 workers per game | **20 pairs 134.75 -> 137.8, +3.05 a round** (seed 317205948 +3.7, seed 846193027 +2.4; paired se 2.3, ci -1.5..7.6) | promoted 01:30 (HOLD_RESEARCH = True); **rolled back 05:52 after hold3** |
| query_route6 | the other agent's query_route prototype: the opportunistic powerup detour's distance queries and speculative searches run on a private copy of the route (before, they left the shared route's held waypoint computed from the powerup's position and its field on the last item; ~390 such waypoints a round) | their wrapper `--isolate-powerup-query-routes` | **20 pairs 133.85 -> 136.85, +3.0 a round** (seed 935714621 +4.0, seed 412896357 +2.0; se 2.8) | **PROMOTED 02:58 as code**: reliability_eval.QUERY_ROUTE_COPY = True, control `--no-query-route-copy` |
| final_backup6 | the other agent's final_backup prototype: braking-backup checks re-run after the final two-level / resource ranking | their wrapper `--final-backup` | 20 pairs 134.75 -> 135.25, +0.5 a round (seed 390817463 -0.9, seed 721934087 +1.9; se 2.3); ~125 extra checks a round, 0.7 failures | not adopted (prototype stays outside production) |
| hold3 | HOLD_RESEARCH confirmation on fresh seeds, QUERY_ROUTE_COPY in both arms | candidate production, baseline `--no-hold-research` | **20 pairs 137.3 -> 131.6, -5.7 a round** (seed 604917382 -6.9, seed 278350961 -4.5; se 1.9); holds 2.7 -> 0.9 s a round, falls 0.70 -> 0.40 | **HOLD_RESEARCH ROLLED BACK to off 05:52** (`--hold-research` keeps it) |
| qr3 | QUERY_ROUTE_COPY confirmation on fresh seeds, HOLD_RESEARCH off in both arms | candidate production, baseline `--no-query-route-copy` | **20 pairs 136.2 -> 131.85, -4.35 a round** (seed 482017593 -3.8, seed 916305284 -4.9; se 2.4); no new map best | **QUERY_ROUTE_COPY ROLLED BACK to off 07:25** (`--query-route-copy` keeps it; 105 tests pass) |
| order1 | the shared-route fix in the cluster order choice (alternative first gems searched on a private route copy) | `--order-route-copy` (ORDER_ROUTE_COPY) | exploratory seed 739104826 only: 9 valid pairs +2.3 a round (KOTM -3, Sprawl +5, Bowl -4, Cube -11, Duplex -9, Cragmire +2, Fractured +7, Vortex +19, TreasureBox +15); Concentric pair (119 / 109) failed the start audit, campaign stopped | not adopted (stays off; stopped by the operator's stop) |

hold3 per map (seed 1 / seed 2): KOTM -9 / 0, Sprawl -7 / -4, Bowl -2 / -2, Cube +3 / +5, Duplex -15 / +2, Cragmire +3 / -9,
Fractured -12 / 0, Vortex +1 / -7, TreasureBox -28 / -21, Concentric -3 / -9. TreasureBox seed 1 diverged at decision 66
(4 s in, no hold before it): timing noise on that map, but the round total is negative on both seeds. Local effect
(scratchpad hold_local.py, time from a hold start to the next pickup, cap 20 s): hold1b baseline 12.6 s (63 holds) ->
candidate 10.1 s (55); hold3 baseline 8.3 s (52) -> candidate 10.1 s (60). The candidate is unchanged; the baseline holds
cost less once QUERY_ROUTE_COPY keeps the route state, so the re-search no longer pays.

qr3 per map (seed 1 / seed 2): KOTM -2 / +1, Sprawl -8 / -26, Bowl -10 / -8, Cube -1 / +7, Duplex +4 / 0, Cragmire +13 / -9,
Fractured -15 / -16, Vortex 0 / -18, TreasureBox -1 / +11, Concentric -18 / +9. Sprawl swung from +40 / +8 in query_route6 to
-8 / -26 here.

## How far a campaign can be trusted

- The four campaigns of tonight do not add up. Writing A = neither change, B = HOLD_RESEARCH only, C = both, D =
  QUERY_ROUTE_COPY only: hold1b measured B - A = +3.05, query_route6 C - B = +3.0, hold3 C - D = -5.7, qr3 D - A = -4.35. Around
  the loop the four must sum to 0; they sum to +16.1 against a combined se of about 4.7. No set of fixed effects (with or
  without an interaction) fits them, so most of what they measured is noise.
- Regression to the mean across all native campaigns since 10-26: ten campaigns whose first seed came out positive (mean +4.4 a
  round) scored -0.25 on average on their fresh second seed. The two that also passed the second seed tonight (hold1b,
  query_route6) both reversed in a third test.
- still2b (10-27) scored +6.15 over 20 pairs (se 2.2) with ONE activation of its rule in 20 rounds: a 20-pair campaign can
  show +6 from search timing alone.
- No sign of an arm or port bias: the arms swap ports on the second seed, and tonight's seed-1 minus seed-2 deltas average
  -0.2.
- Why pairing does not help: in hold1b and hold3 (40 pairs) the two arms' first different action or position came before
  either arm's first safe hold in 37 pairs, so search timing split them, not the change (HOLD_RESEARCH acts only in a hold).
  Median split 6.3 s into the round (quartiles 3.3 / 9.7 s); those 37 pairs differ by 7.4 points on average (scratchpad
  pair_split.py).
- The spawns stay paired after a split: in three split hold3 pairs both arms saw the same gem groups in the same order, only
  at different times (Fractured 12 of 12, Duplex 15 of 15; TreasureBox 5, then different). So the +-7 points a pair are the
  planner's own swings in time per group, not spawn luck: in the TreasureBox pair (-28) the candidate arm went 36.1 s
  without a pickup from 72.7 s, while the baseline's longest gap was 13.9 s (scratchpad spawn_seq.py, round_diverge.py).
  Long no-pickup gaps are both most of the noise and a real loss of points.
- The timing that splits them is rare: in a qr3 Sprawl round the median decision takes 0.016 s (a kept plan), p90 0.147 s,
  p99 0.193 s, and 7 of 1,500 decisions record a search stopped at the .172 s deadline. With a budget that almost never
  binds the planner should be deterministic: `run_flag_native.py --fixed-lead 16` (~1 s budget; option added 07:5x with
  `--maps`, defaults unchanged). The A/A check det_aa (identical arms, three maps) was cancelled by the operator's stop:
  the mode is untested.
- What a campaign can do: reject a large loss, and show mechanism counts (holds, rescues, falls) beside the points. What it
  cannot do: separate a +2 change from 0. Every idea tested tonight changed rounds by a few points at most.

query_route6 per map (seed 1 / seed 2): KOTM -2 / +5, Sprawl +40 / +8, Bowl +7 / -16, Cube +5 / -10, Duplex -14 / +10,
Cragmire 0 / -1, Fractured +1 / 0, Vortex 0 / -5, TreasureBox 0 / +26, Concentric +3 / +3. The code version (patch
scratchpad patch_query_route_copy.py) replaces the same 12 route references in pick()'s detour block as the prototype's AST
transform, with the selected detour still committing the shared route.

Queue (run_queue_night.py, since 10-29 in ml_agent/tools/ with run_flag_native.py; one campaign at a time; the list is night_queue_20261029c.json
since 01:49): after query_route6, final_backup6 (the other agent's prototype through run_flag_native.py `--bench <their
wrapper>` at 6 workers, both seeds), hold3 (HOLD_RESEARCH confirmation on two fresh seeds: candidate production, baseline
`--no-hold-research`), hold2 (`--hold-wide`), follow_backup6 (the other agent's prototype). A HOLD file paused it from 02:55 for
a simulator screen.

Dose-response check (the gain should follow the room the change has to act): pairs whose baseline arm held >= 3 s: 6 pairs,
mean +8.0 (Cragmire +24 / +8, Vortex +21 / 0, TreasureBox -1 / -4); pairs under 3 s: 14 pairs, mean +0.9; correlation of the
baseline arm's hold seconds with the pair delta r = 0.54 over the 20 pairs.

hold1b per map (seed 1 / seed 2): KOTM -9 / 0, Sprawl -9 / +5, Bowl 0 / -2, Cube -13 / -7, Duplex 0 / +15, Cragmire +24 / +8,
Fractured +12 / +14, Vortex +21 / 0, TreasureBox -1 / -4, Concentric +12 / -5. Over the 20 pairs: safe holds 4.5 -> 0.9 s a round,
A2 3.0 -> 2.6 s, falls 0.85 -> 0.60, 8 s rescues 0.40 -> 0.35, 1.6 holds a round replaced by a useful plan. New best Fractured
134 (1 fall, confirm candidate; CSV docs/BEST_REAL_SCORES_2026-10-29.csv, sheet updated and read back).

Mechanism (hold1 partial Cragmire rounds before the disconnect, and hold1b): Cragmire safe holds 19.3 s in 3,494 decisions
(longest 5.6 s) -> 2.9 s in 2,466 (longest 0.8 s); hold1b Cragmire holds 25.0 -> 4.2 s, 12 holds replaced by a useful plan,
no-progress time 66.6 -> 47.0 s, 8 s rescues 2 -> 1, A2 7.0 -> 4.8 s. Maps without holds diverge by +-9 from search timing alone.

hold1b runs with 6 forecast workers per game (both arms): 16 workers held ~40 GB private on the 31.8 GB machine and paged.
Tonight's later campaigns use 6 as well; the A/B comparison stays fair, the absolute level is a little lower than at 8.

## What still loses time in the new production (hold1b candidate arms, 20 rounds)

Legs with > 4 s of excess: Cragmire 89 s a round (raised gems 38), Vortex 75 (raised 26), Concentric 28, TreasureBox 28, Duplex 21.
Respawns (rescues and falls) in 100 recent rounds come 10-30 s after the last pickup; Cragmire repeats the same traps:
- the central pit (0, 0, -5.5) under the z 13-14 cluster: the stall timer's progress (route distance, or 3D distance when there
  is no route) improves by > 1 u at every hop apex, so the 5 s rule never fires; with progress counted only on the ground
  (scratchpad variants/re_stallground.py) it fires in the sim but skips all 7 cluster gems, leaving nothing else to chase; the
  8 s rescue (nav/recovery.py ProgressMonitor) has the same weakness and fired after 15 s;
- the wall pocket (24.2, -10.7, 6.2) (3 respawns in these rounds, also the 10-27 holds): an edge-hit plan runs into it at
  12.6 u/s toward a gem 35 u away, then nearly every search there overruns the budget (A2 and holds) until the 8 s rescue.
A "boxed in -> respawn" rule (no pickup 8 s, ground positions within 5 u for 5 s) would fire 0.66 times a round: 27 true dead
ends (respawn 4.9 s sooner) against 18 firings with a pickup within 5 s and 21 escapes without a respawn: a net loss, not built.

## Long no-pickup gaps and plan churn (census of tonight's 200 rounds, 07:5x)

Scratchpad gap_census.py over hold1b, query_route6, final_backup6, hold3 and qr3 (both arms, both seeds):
- Legs of 8 s or more without a pickup: 1,348 legs, 16,976 s, 35 % of all round time (~85 s a round). Per map a round: Vortex
  157 s, Bowl 149, Cragmire 122, Duplex 108, Sprawl 106, Fractured 81, TreasureBox 63, Concentric 37, Cube 26, KOTM 0.
- By class, measured to GEM goals only (goal_context kind 'gem'; scratchpad gap_census_gem.py, 08:2x): far (gem >= 60 u away
  at the start) 706 gaps, 8,365 s, **~42 s a round** (Duplex, Vortex, Sprawl, Bowl, Fractured); raised (gem >= 2 u above for
  half the gap) 3,133 s, ~16 s a round (Bowl 107 gaps); other 2,824 s, ~14 s; with a fall 1,534 s, ~8 s; **near miss (gem
  <= 20 u away, the marble came within 1.5 u, still >= 8 s) 65 gaps, 911 s, ~4.6 s a round** (Concentric 19, Cragmire 15,
  Vortex 14); boxed in 210 s. CORRECTION: the first pass (07:5x) measured the distance to any goal item, so powerup detours
  counted as near misses (320 gaps, ~20 s a round): that figure was wrong.
- Plan drops (a pickup plan replaced while its pickup was > 2 replies ahead; any goal kind) run at about the same rate in
  ordinary legs (23 % of decisions) and in long legs (25 %).
- Phantom gem pickups (a plan for a gem predicts the pickup at reply 0-1, no gem within 3 decisions; scratchpad
  phantom_census.py --gem-only): 1.7 episodes a round, in 1.1 legs that run on 8.5 s after the first phantom (an upper bound on
  the loss). Duplex 7.1 episodes / 36.6 s, Fractured 3.1 / 14.5 s, Vortex 2.0 / 12.0 s, Concentric 2.0 / 5.7 s, the rest
  <= 1.1 / 7 s. Plans: throttle 34 %, accelerate 24 %, gap runs 15 %. (Counting every goal kind gave 13.8 episodes / 50 s:
  powerup pickups and staging-point arrivals, not phantoms.)
- Anatomy (hold3 explore candidate, TreasureBox 72.7-108.8 s, raised gem 7 u up): 431 of 565 decisions chose a fresh plan;
  167 fresh plans already carried a side-impact penalty when chosen; a new jump plan predicting the pickup 1.3-3 s out
  replaced the last one every 4-6 decisions (the window never counted down). The round's counters: 420 penalized_plan_researches,
  354 research_found_pickup. Every map but KOTM shows 100-475 penalized re-searches and 80-360 replacements a round.
- The mechanism: the 10-23 rule searches a kept plan again on EVERY decision while its re-forecast carries a side-impact or
  ledge-drop penalty, and then any new plan that predicts a pickup replaces it. The search's winner is the cheapest of many
  forecasts, so a fresh plan a hair cheaper almost always wins; near a raised ledge it smacks too, gets penalized, and the cycle
  repeats. The stall rule does not catch it: its progress (route distance improving by 1 u) is re-based on every switch to a
  neighbouring gem, and jitter keeps resetting it.
- The sim reproduces these legs (tonight's nine stuck legs: sim 12.2 / 9.5 / 8.7 / 16.2 s against recorded 12.4 / 9.5 / 8.3 /
  20.0 s), so a fix can be screened there.

Expected size after the correction: the loops this targets are a few seconds a round, so a gain of more than ~2 a round is
unlikely and a 20-pair campaign cannot resolve it; most long-leg time is far travel (~42 s a round).

New idea, lobby-valid (the marble's own forecasts only): **PENALIZED_SWITCH_U** (scratchpad patch_penalized_switch.py, flag
`--penalized-switch U`, default 0 = off; not yet in the repo): a kept plan searched again for its penalty gives way only to a
plan for the same gem that costs U less. A plan that avoids a 50 u smack still replaces it, so the 10-23 smack penalty keeps
its effect; only the near-equal switching stops. Sim screen after order1 (scratchpad after_order1.sh): stuck legs of all 20
arm folders with U = 15 and 30, normal legs with U = 15.

## The shared-route bug in the order choice (order1, queued)

order_choice() (the cluster first-target choice, ORDER_CANDIDATES 3) navigates the shared route to each alternative first gem
before its light search, and when it keeps the current gem (or finds nothing) it returns without restoring the route: the
current gem's held waypoint is lost on most full searches in a cluster, the pattern QUERY_ROUTE_COPY fixed for powerup detours.
Patch scratchpad patch_order_route_copy.py (ORDER_ROUTE_COPY, `--order-route-copy`, default off): alternatives are searched on
a private copy; a switch still commits the shared route. Open question: the CPU side. A goal field (scipy Dijkstra, no cache) costs
6-8 ms on the plain TreasureBox / Sprawl grids (0.5 ms on KOTM); a sim TreasureBox episode counted only 4 builds on the plain
TerrainGrid class in 98 decisions, but the route planner may build its fields on the walking terrain, which that count may have
missed. Not settled.

## Infrastructure notes for the operator

- Cleanup 10-29 (operator-approved): the 109 one-off experiment scripts of 10-26 / 10-27 (prototypes, wrappers, drivers,
  tests, audits of single ideas) were zipped to C:\Users\doug\pq_archive\experiment_scripts_2026-10-29.zip (MANIFEST.txt lists
  each group and its result; restore = unzip at the repo root) and removed. The campaign runners run_flag_native.py and
  run_queue_night.py moved to ml_agent/tools/ (results still go to dynamics_reports/overnight_20261026/). The three result
  folders (overnight_20261026, traversal_recovery_20261026, planner_review_20261026) are in .gitignore and stay on disk.

- Memory: a campaign at 8 workers per game holds ~40 GB private on the 31.8 GB machine (VS Code ~6 GB private besides); the
  per-map planner caches are pickled objects (terrain ~150 MB a map plus query and walking caches) unpickled into every worker,
  and Cragmire's workers are ~30 % larger than the others. Options: a fixed larger pagefile, fewer workers (tonight 6), or caches
  in a memory-mapped format shared by the workers (a code change). 10-29 07:4x during order1 (6 workers, nothing else of
  mine running): free commit 0.6-2.4 GB of the 63.6 GB limit (planner pythons 38.7 GB, VS Code 5.9, games 1.9, browsers
  2.4); physical RAM 5.7 GB free and the pagefile is system-managed (33.6 GB, 1.4 GB used, C: 42 GB free), so Windows can
  grow it; the hold1 failure was real paging at 8 workers.
- Noise: maps where the treatment never acted swung -9 / -9 / -13 from search timing alone. A deterministic test mode (the
  10-27 proposal in docs/TAKEOVER_POINTS_2026-10-27.md: worker step counts charged to a virtual clock) would make such maps
  score identically and leave only the maps where the change acts; still not built (needs the operator's OK). After tonight
  (see "How far a campaign can be trusted") this is the most useful next step: without it a +2 a round idea needs about
  110 pairs (per-pair SD ~10.7 points, so se 1; 5-6 campaigns, ~8 hours) to show.

## Simulator screens (no game)

- Fixed tools/sim_bench/sim_bench.py: setup() set search.LAUNCH_COMPLETE from the flag alone, so every sim since the 10-27
  promotion ran WITHOUT launch completion (production has it on); it now keeps the module default. Also: planner flags passed
  with --pflag reach the parser but not the main() settings block (the reliability_eval globals); use --set module.ATTR=value.
- Super Speed turn kicks (scratchpad variants/search_usebrake.py, repo patch scratchpad patch_ss_kick.py, `--ss-kick`): at a KOTM
  post-pickup turn (decision 499 of landshort1_explore_baseline, gem 13.6 u away behind the marble) the chosen plan reached the
  gem in 2.88 s; kicks aimed with lead_aim collected at reply 19-20 but left the platform (rejected no_supported_continuation);
  kick + full brake 4 or 8 replies later collected at reply 21 and stayed on (1.34 s), but the use margin (8 u of cost, while
  1.5 s of pickup time is worth 6 u at PICKUP_TIME_WEIGHT 4) rejected it until a seconds-based margin was added
  (SS_TIME_MARGIN: a Super Speed use 1 s sooner than the best non-use pickup passes). Screen of 86 legs that start holding
  Super Speed (two pools, all maps, LAUNCH_COMPLETE off in both arms then): 4 changed (1 better, 3 worse), mean +0.06 s. Rarely
  applies; no game slot for now.
- Stall progress only on the ground (STALL_GROUND_MS, `--stall-ground`, applied default off; screen in the 03:00 queue gap with
  `--set reliability_eval.STALL_GROUND_MS=50`): only 9 recorded stuck legs qualified (two pools, six maps), 0 better / 1 worse
  (Cragmire +1.3 s). No evidence for it; stays off.
- Pickup re-search (PICKUP_RESEARCH_U 4, `--pickup-research 4`, applied default off): 139 paired normal legs, mean +0.06 s,
  29 better / 26 worse, Concentric 5 of 7 worse (+2.4 s), 4x the searches. No game slot.
- DIRECT_LEAD (the other agent's 10-26 lead aim for the direct_use Super Speed kicks, never game-tested; flag `--direct-lead` added
  tonight, default off): 99 paired legs that start holding Super Speed, mean 0.00 s, 2 better / 3 worse. No game slot.
- Cost switch for kept plans without a pickup (scratchpad variants/re_switch.py, not in the repo: a searched plan 10 u of cost
  cheaper replaces the kept one; the v47 rule keeps it unless the new plan collects): 70 normal legs all identical (in clusters
  the kept plans collect); 46 long trips mean -0.01 s, geometric ratio 0.990, 12 better / 7 worse (Duplex +0.5 s, Fractured
  -0.3 s). Weak; no game slot yet.
