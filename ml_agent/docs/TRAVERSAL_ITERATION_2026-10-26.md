# Traversal iteration results, 2026-10-26

## Current production state

Powerup-through is now ON by default. A fetched powerup can be treated as part of
the route toward the real gem, preserving momentum rather than stopping at it.
Use --no-powerup-through for the matched old control. This is general goal/physics
logic; no map-specific rules or controller selection were added. The new default
arguments exactly match the configuration tested in the live candidate.

Adoption is a practical choice supported by positive native pilot scores and a
matched live Sprawl gain. It is NOT a statistically established improvement of the
expected score on every map. Cragmire and Vortex regressed in the native pilot.

Land-short, no-jump launches, launch completion, and general fast-supported tails
remain OFF by default. Staging, acquisition-lookahead and learned-prefix handoff
experiments were not promoted. Learned weights were not trained or changed.

## Confirmed real-game results

Quiet sequential 1x Sprawl pair, CLI seed81910417 (observed engine seed81910420):

| Metric | Control | Powerup-through |
| --- | ---: | ---: |
| Full-round points | 154 | 175 |
| Falls | 0 | 0 |
| Rescues | 0 | 0 |
| Total game holds | 11.02 s | 10.28 s |
| Maximum hold | 0.37 s | 0.17 s |
| Wall duration, including countdown/holds | 317.6 s | 316.4 s |

Gain:21 points,13.64%. Both rounds finished with unchanged source hashes. Their
opening, RNG, physical motion, inventory, Blast and pending controls match.
Comparison normalizes absolute clock epochs, unused disabled-effect timer memory
and process-local authority/client object IDs; other fields remain compared.
Artifacts: dynamics_reports/planner_review_20261026/live1_comparison.json,
live1_start_audit.json, and live1_baseline/live1_candidate folders.

New native full-round records, all zero falls and zero rescues:

| Map | Best at task start | New best | Evidence |
| --- | ---: | ---: | --- |
| Sprawl | 192 | 213 | trav3_candidate seed105831917, powerup-through alone |
| Cube Isle | 163 | 175 | trav3_candidate seed105831917, powerup-through alone |
| Bowl | 146 | 150 | trav2_candidate seed61918379, optional short+through bundle |

Bowl's record used an experimental bundle and is not evidence for the adopted
single feature. The score CSV and canonical Google Sheet were updated and read
back. The live175 did not exceed the native Sprawl record213.

## Ten-map native evidence and limits

Three complete pairs,120 scored native rounds in total, four workers per arm,
production horizon48. No heavy simulations ran alongside scored games. Starting
motion and queued-control comparisons pass. Protected learned checkpoints stayed
unchanged; no scored grants/restores/teleports were used.

trav1: short+through, seeds7347019/12091843,2732->2774,+2.1 points/round.
trav2: same bundle, fresh seeds16455731/61918379,2638->2658,+1.0/round.
Combined raw mean134.25->135.8,+1.55; approximate paired95%CI[-2.045,+5.145].
A repeated Bowl opening in trav2 failed its opening audit. All round scores remain
valid, but the duplicated opening is not independent confirmation. Keeping the
lowest seed per map/layout, independent of scores, leaves39 distinct openings:
134.075->135.2583,+1.1833, approximate stratified95%CI[-2.191,+4.558]. Independent
source rehashes pass; the original failed audit is retained. No bundle promotion.

trav3: powerup-through ALONE, predeclared fresh seeds39247103/105831917. All source,
opening and input audits pass;20 distinct matched openings,40 valid native rounds.
2753->2783, mean137.65->139.15,+1.5/round. Approximate95%CI[-3.572,+6.572].

| Map | Mean point change |
| --- | ---: |
| Bowl | +1.0 |
| Concentric | -4.5 |
| Cragmire | -9.0 |
| Cube Isle | +6.5 |
| Duplex | +12.5 |
| Fractured Islands | +4.5 |
| KOTM | -0.5 |
| Sprawl | +12.0 |
| TreasureBox | +1.0 |
| Vortex | -8.5 |

Sprawl improved BOTH seeds174->190 and205->213; Cube146->147 and163->175;
Duplex119->143 and155->156. Cragmire's worst loss108->83 had fewer falls but more
stalls/rescues:0falls/3rescues against1fall/1rescue. Survival alone is not the goal.

These scores do not establish the human-opening5.9s target. On the new randomized
Sprawl layouts, first pickups were6.080->6.464s and1.792->1.792s; full-round gains
came later. Do not compare those different layouts directly to the human opening.

## Implementation and verification

- Shared continuation rules and extension time pricing between serial search and
  worker evaluation. Previously4/8 recorded candidate sets and1/8 choices differed;
  repaired paths agree. Additional actual multiprocessing checks also pass.
- Removed no-jump launch grid duplication:64 schedules became8 unique schedules.
- Compiled short-landing controls, ABI14/Q_SIZE64, with0.4/1.0 setbacks retaining
  original aimed options. Optional launch completion preserves feedback controller
  and absolute event timing. These performance options stay off.
- Corrected offline world events: different powerups replace held inventory;
  Mega pickups are represented independently of proposal switches. Reproduce the
  former benchmark with --legacy-item-world. This does not change native scores.
- Added score-pair monitoring, comparisons, worker audits and bounded screen runner.
  Final audit failures are surfaced; source integrity is written before opening
  audit exceptions. Comparisons enforce starting-motion/input equivalence.
- Added an observed-gem-set ledger instead of guessing groups from pickup gaps.
  Its point totals AND completed-group counts match all40 trav3 native reports.

After default promotion:99 search/native/runtime/full-round regressions passed;
8 compiled/Python pickup/replacement/Mega/duplicate checks passed. Default config
is exactly equal to explicit --powerup-through; the negative control works and
other experimental flags remain off. git diff --check passed.

Earlier randomized compiled controller checks passed on Sprawl and TreasureBox
(--n500 each). Launch prefix16+feedback32 had identical controls and zero position
error between compiled and per-reply forecasts. Actual2worker powerup-target audit
passed4 recorded Sprawl/Cube approaches with identical candidates/costs/choices.

## Experiments not promoted

880 new simulated episodes across5 screens, with recorded starts, native physics,
whole group plus next group's first pickup,40s cap, horizon explicitly48. Reused
sealed baselines are SHA-manifested and not counted as new trials. Screens2/3
baseline behavioral outputs match exactly on all40 starts. Simulations stop at a
fall and do not establish native full-round points; known future spawns are only
used in the benchmark world, never given to planner proposals.

- Screen1:240 episodes,6 variants. Through was simplest promising; bigger bundles
  did not outperform it. This screen used the older incomplete item-world model.
- Screen2:280 episodes, corrected world. Base37/40 completed; through40/40 with
  successful time ratio0.97657. Short37/40,0.99701; combined39/40,0.97384;
  acquisition39/40,0.97624; adaptive37/40,0.99301; incidental-points38/40,1.01199.
- Screen3:160 episodes. Learned-prefix pickup handoff37/40,1.00033; through+handoff
  40/40,0.98209, worse than plain through. Not promoted.
- Screen4:120 new episodes. Completion37/40,1.0100; through+completion39/40,0.98651;
  through+short+completion39/40,0.99466. Plain through remained better.
- Screen5:80 new episodes. Fast-supported37/40,0.99585; through+fast38/40,0.9550 on
  36 mutually completed starts, versus37 for the through comparison. Faster
  survivor trips came with2failures versus0 for plain through. Not promoted.

An adaptive-runup diagnostic on Fractured seed9083321 decision246 changed a9.4s
fall/zero pickups into a19.2s completed group/seven points. Wider evidence was weak;
no native staging improvement claimed. The earlier decision2508/seed7347019 probe
did not reach the problematic crossing and is not evidence for adaptation.

Prototypes/configurations are archived under each screen's prototypes folder.
The proposed larger maneuver beam, fully goal-switching neural forecasts and
additional pickup-state lookahead remain research directions, not shipped wins.

## Operational state

No benchmark games or heavy jobs remain running. Changes are uncommitted.
Current gameplay default is powerup-through ON with --no-powerup-through control;
no map-specific exceptions, learned training, or other performance-option promotion.
All claimed score evidence remains under dynamics_reports/planner_review_20261026
and handoff_safety_20261020/trav1_20261026_*, trav2_20261026_*, trav3_20261026_*.
