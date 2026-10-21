# Plan transitions and landing safety — 2026-10-20

## Scope

User authorized continuing the recommended investigation after the speed fix: safer transitions after pickups/target changes while preserving the real-time Bowl gain. Corrected fast controllers remain enabled. No training is needed. Dashboard33732 must remain untouched; maximum8 game processes. Preserve prior working-tree changes and evidence.

## Initial findings

Both Cube Isle falls in `dynamics_reports/speed_planner_20261020` follow a similar sequence: a ramp-gem approach, a retained plan with an unsupported braking endpoint, then a run of A2 fallback actions and a fall. Seed7347019: fallback begins at1345 just before pickup, fall1365. Seed12091843: pickup2251, fallback2252, fall2269. The latter approach uses ordinary `route`, not fast control.

The braking backup runs20 replies (1.28s). If it ends airborne, a downward ray finding floor within15u passes, without predicting contact with that floor. The saved backups have already dropped13.38/18.53u. This is a hypothesis for false reassurance, not yet proof of the cause or a scoring fix.

Historical experiments already rejected unconditional keep-plan and edge-ahead braking: they slowed open-map navigation and did not reliably solve the eventual fallback. Do not simply re-enable them.

## Investigation in progress

Evidence root: `ml_agent/dynamics_reports/handoff_safety_20261020`.

Using maintained `tools/sim_bench/sim_bench.py`, start24 decisions before the recorded seed7347019 pickup, horizon48, fast fix enabled. First one-pickup smoke run stopped at the pickup and cannot assess landing; retained as `cube_baseline.jsonl`. Corrected follow-through run requests two pickups and reproduces a fall after the first pickup (`cube_baseline_followthrough.jsonl`). This simulator starts a fresh controller and uses a virtual planning clock; it is a diagnostic, not an exact replay or native scoring result.

Next comparison extends only braking backup horizon to48 replies, without changing production defaults. Investigate physical landing validation if this supports the hypothesis; run native matched scores before adopting a behavior change.

Longer backup alone reproduces the same fall (2.4s, one pickup); it cannot rescue an already rejected search. Existing `--checked-recovery` instead completes both pickups in6.0s with no fall or unchecked fallback in the same simulator start. Starting60 decisions earlier also reproduces the baseline fall. Eight ordinary Bowl two-pickup starts are being compared with recovery plus the existing actor-task and cooldown fixes.

Declared native screen: current fast-enabled planner versus the same planner with `--checked-recovery --actor-task-during-recovery --recovery-preserve-search-cooldown`; Cube Isle and Bowl, seeds7347019/12091843, full original-duration rounds, two workers per arm (four games maximum), ports9090–9091 and9094–9095. All other settings match the prior speed experiment (horizon48, fallback cooldown2,2s search budget,40reply/.2s recovery). Exact source archives, actual opening/motion checks, points/falls/rescues and timing required. This revisits recovery with the current planner, not a claim that the historically losing recovery recipe now wins. No source changes during the screen.

## More precise root cause from the original trace

The initial floor-ray suspicion is not the explanation for these two cases: both saved braking checks already carry `no_braking_backup=True`. Their endpoints are below the Cube bounds, and the planner did flag the backup failure. Do not report these as falsely passing backups.

Seed7347019 decision1299 selects `direct_fast_20` with a failed backup and cost63.29 (including150 risk penalty). Decision1300 revalidates the same plan but drops the backup metadata and penalty, reporting cost-86.97. The controller automatically retains pickup plans, so it follows that known-risk approach until too late. `PlanMemory.accept` stores no backup verdict and `retained_candidate` reconstructs only the nominal-horizon assessment. This is the concrete next candidate: retain/recheck known backup risk and permit new alternatives to replace a risky pickup plan before the committed approach. Preserve current fallback behavior when no better plan exists.

Native recovery screen Cube results complete: seed7347019 baseline126points/1fall versus recovery113/2; seed12091843 baseline133/1 versus recovery137/1. Recovery is not a demonstrated safety improvement. Bowl rounds remain running; source freeze remains in force until they finish. `native_comparison.json` retains completed matched pairs, with exact starting motion/layout checks. Eight sampled ordinary Bowl episodes were identical, but that alone cannot justify adoption.

## Recovery rejected; targeted risk-recheck candidate

The full8-round recovery screen finished. Bowl is exactly unchanged in points/speed:129 and120 points, zero falls/rescues; recovery never activated there. Cube mean129.5 to125 points, falls2 to3. Recovery remains off. All4 paired opening/motion checks and source audits pass (`native_recovery_audit.json`); no sources changed during scoring.

Implemented opt-in `--recheck-risky-plans` in maintained search/controller/runtime/simulator/benchmark files. PlanMemory now retains whether the selected plan failed or lacked a checked braking backup. Under the flag, retained risky plans recheck that backup; failed or unknown results retain the150 cost penalty. Risky pickup plans no longer automatically bypass fresh search. A backed alternative can replace them even if it only makes progress, and the first risky target does not prevent evaluating another gem. Existing discrete-action commitment is preserved. A deadline while searching for alternatives retains the already revalidated nominal prefix rather than handing control to unchecked A2.

No blanket speed reduction, no map-specific maneuver, no new scripts, no native binary change. Defaults remain unchanged pending evidence. Tests cover warning/penalty retention, unknown forecasts, clearing risk after a successful check, ordinary-plan non-interference, safe-progress replacement and deadline behavior.

Offline Cube start1286: baseline falls at5.0s after one pickup; risk-recheck completes both pickups at6.6s, zero unchecked fallback. Saved as `cube_risk_recheck.jsonl`. Initial244-test run passed; a245-test final run includes the added deadline regression. Eight ordinary Bowl starts are being checked with the final flag.

Declared next native screen: same two maps/two seeds, current source with flag off versus on, recovery off in both, fast controllers on in both. Four workers per arm, ports9100–9103/9104–9107 (eight games maximum). Full original-duration rounds; preserve all failures and require paired actual starts and unchanged source. Do not adopt based solely on the simulator result.

Final245-test suite passed (`risk_final_tests.log`); all8 ordinary Bowl simulator trajectories/end states/speeds/pickup times match exactly (`risk_offline_comparison.json`). Native `risk_baseline`/`risk_candidate` pools are active with eight total games and final sources frozen. `risk_initial_audit.json` verifies all4 pairs' actual starting raw observations, motion, inventory/effects and gem layouts. Early Cube telemetry confirms the new branch executes (`risky_plan_researches` and `risky_plan_replaced`); no conclusion from partial scores.

Both Cube risk-screen pairs now complete: seed7347019 baseline126/1fall versus candidate131/0; seed12091843 baseline137/1 versus candidate146/1. No rescues. Mean131.5 to138.5 (+5.32%); total falls2 to1. The fresh control differs slightly from the earlier screen due wall-clock search, so use these contemporaneous matched controls, not a cherry-picked historical score.

Declared transfer extension on unchanged sources: Duplex and Cragmire, both same seeds, two workers per arm on ports9110–9111/9114–9115 while the four Bowl games finish (maximum8 native games). `risk_transfer_baseline`/`risk_transfer_candidate`. After those finish, compare the remaining six maps on both seeds with four workers per arm. This creates a complete ten-map comparison before considering promotion; KOTM is one equal-weight map. Recovery remains off, no training, no source edits during pools.

## 2026-10-20 11:50-13:20 (second agent): ten-map verdicts, real-time Cube, Mega

- `--recheck-risky-plans` v1 (any risky retained plan re-searched, replaceable by safe progress): 20 matched pairs,
  equal-map 129.50 -> 128.05, falls 7 -> 8 (`risk_tenmap_comparison.json`). Journal audit: the Duplex losses and Cube
  12091843 diverge from wall-clock search timing before the flag fires; where the flag acts outside Cube it loses
  (Cragmire 7347019 fall at decision 376 after dropping a slow steering plan at 308).
- v2 (risky = pickup plan only; only a backed pickup plan replaces it; `reliability_eval._search`, tests updated):
  `risk2_*` pools, 129.35 -> 128.70, falls 8 -> 6 (`risk2_tenmap_comparison.json`). Cube gains in all four Cube pairs
  (+5, +9, +5, +13), Fractured/Vortex/Duplex lose more. Default stays off (points-only rule); the risk-metadata
  retention in PlanMemory is kept under the flag.
- Real-time Cube with current defaults (`live_cube_start0/1`, same layouts as the pool starts): 111 (4 falls) and
  138 (1 fall) vs pool 126/133. In range; round 1's two early falls (decisions 296, 341) are new. realtime.py now
  journals the chosen plan label per decision (`label` field) so live falls can be diagnosed; a labelled rerun of
  seed 7347019 is in `live_cube_start0_labels`.
- Mega: see the 13:15 log entry and TODO.md item 2. Built behind `--mega-proposals`, forecastable, never selected
  offline (slower physics + post-hop bouncing fails the stability gate).
- Trap: `reliability_eval --workers N` children run from the repo root and need PYTHONPATH=ml_agent (first launch
  from ml_agent failed with "No module named nav" and preserved 12 invalid jobs).

## 2026-10-20 14:00-17:35 (second agent): the A2 fall mechanism and --pickup-stop

See the log entries 14:20 through 17:35. Short form: every fall follows a 1-2 s unchecked-actor run that starts when a
fast approach to an edge gem leaves no candidate alive (they all collect the gem and are over the edge at the horizon).
`--pickup-stop` adds last-resort stop candidates (full brake / stop 3 u short of the gem) when nothing passes and the
free roll leaves the floor (30 u ray, 1.5 s), with a parked-streak cap and release of held stop plans. Five ten-map
screens: points a wash (within +-2), falls roughly halved (8 -> 3..5) with Cube/Duplex fixed and Cragmire hurt.
Default off; evidence `pstop*_tenmap_comparison.json`, offline `pickup_stop_offline/`, reproduction command in the
14:20 log entry.

## 2026-10-20 23:50 - 2026-10-21 05:35 (second agent): edge hits made to pay, ON by default

Operator directive: make edge hits work and raise points. Eight ten-map native pairs (log entries 23:45 through 05:30):
v2 +1.10, v4 +2.55, v5 +0.20, v6 +2.95, v7 +1.85, v7 on fresh seeds +1.35, v8 +0.60 (reverted), v9 +2.15 equal-map
points. The working recipe (`nav/marble_dynamics/edge_hits.py`, `search.py` edge block, `reliability_eval` retry guard):
real lips from the mesh (wall-or-nothing other side AND a drop beyond, probed), lips 0.4-2.6 u above the marble toward
a target that needs a climb, jump timed to clip the lip (or meet the wall at the apex above 1.4 u, where the spin-friction
kick carries the marble over), lip-aimed plans on their own route with a through box plus timed jumps on the route and
direct lines, landing extension, a climb gate (the forecast must reach the lip), evaluation after the learned navigator,
a 3 s retry guard per lip. Things that did not work: lips below the marble / far targets (v1), unguarded wall hits (v5:
valid-looking jumps that never reached the lip won on tiny margins), charging the landing extension's extra time (v8:
removed the genuine climbs too). `--no-edge-hits` is the matched control. Evidence: `edge*_tenmap_comparison.json`.
