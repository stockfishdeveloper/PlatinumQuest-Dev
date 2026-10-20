# Planner movement speed — 2026-10-20

**Final status: complete.** Corrected fast controllers remain enabled. Forty native comparison rounds and four quiet real-time Bowl rounds finished; speed and points improved. No training or evaluation process is running. Dashboard33732 remains untouched. Final results and remaining limitations are at the end of this document; earlier progress sections are chronological.

## Authorization and scope

The user observes unnecessarily slow movement, especially on Bowl, and authorized autonomous iteration until speed and points improve or no useful progress remains. Preserve real physics and ordinary game rules. Keep dashboard33732 untouched, at most8 native games total, log each substantial change and retain negative evidence. No learned-model training is needed. Existing prior-turn changes are uncommitted and must remain intact.

## Diagnosis

Historical notes already tested terminal speed limits, velocity cost, longer post-pickup scoring and modest speed-cap changes. Avoid repeating those blindly. Ordinary proposals target11u/s, ramping to13.5 on long legs. Fast16/20u/s families exist but reportedly never win.

Fresh baseline Bowl journal: mean horizontal speed14.25u/s while the target is over30u away,10.09 nearby. Opposing input at speed>8 occurs659/2727 travel decisions. Braking may be appropriate when turning; these counts alone do not prove every instance is wasteful.

**Reproduced a compiled-controller bug:** `SequenceProposal.fast` and `FAST_MIN_SPEED` are used by Python but absent from native serialization/C++ control. Thus compiled fast candidates are duplicates of normal candidates. On baseline Bowl decision80, normal and fast20 compiled controls/states are identical (mean13.316u/s); the Python fast20 controller averages17.394u/s and ends farther toward the gem. This is an implementation mismatch, not merely a search preference. Evidence: `dynamics_reports/speed_planner_20261020/native_fast_before.json`. Exact pre-change source/native binaries archived in `before_fix/source_snapshot.zip`.

Next: port the existing fast semantics faithfully, compare compiled/per-reply forecasts and protect defaults. Then score native Bowl rounds against the unchanged control on multiple actual gem layouts; compare movement speed, points, falls, rescues and selected fast plans. Broaden to other maps before adopting globally. If the repair alone is insufficient, diagnose its actual selections/rejections before another targeted change.

## Native repair and validation

Ported the existing Python fast-speed controller to C++ and serialized both the candidate's speed and FAST_MIN_SPEED. Native sequence size49→51 and ABI9→10; an old DLL now fails explicitly instead of silently omitting the behavior. No reward, fall/landing guard, stopping-distance rule or ordinary speed cap changed. Added a serialization regression and fast checks to the maintained native_controller_check tool.

Rebuilt the native libraries. `native_controller_check --n240 --fast-pool .../baseline_start0 --through-pool .../baseline_start0 Bowl_Hunt Sprawl_Hunt CubeIsle_Hunt` passes: randomized sequence controls120/120 per map, full48-reply fast forecasts32/32 per map (96/96 total), through forecasts16/16 per map, plus all scalar/grid/translated checks. Each map also verifies fast controls actually differ from normal. Logs: `%TEMP%/pq_checks/speed_native_build_20261020.log` and `speed_native_check_20261020.log`.

Added diagnostic `--legacy-fast-proposals` to the evaluation and real-time harnesses: fast-labelled families become ordinary proposals, faithfully reproducing the previous compiled duplicate behavior while retaining candidate count/order. Default uses the corrected fast controllers. This allows matched comparisons using one binary and frozen source. The flag is recorded in prepared metadata. Real game scoring is still pending at this point; forecast parity alone is not a scoring result.

All241 dynamics tests passed (`unittest discover -s ml_agent/nav/marble_dynamics -p test_*.py`, log `%TEMP%/pq_checks/speed_tests_20261020.log`).

## Matched native screen in progress

`legacy_screen` and `fast_screen`: Bowl, Sprawl, Cube Isle; random seeds7347019 and12091843, three workers per arm (six games total), ports9060–9062/9064–9066. Full original-duration rounds (Bowl/Sprawl300s, Cube Isle210s), horizon48/commit4/reuse, geometry route, resource acquisition, recharge/landing search,20 proposals, static caches, global goals,16ms physics, fallback cooldown2,2s search budget, pickup landing40, component crossing, ballistic landing, use horizon48/use margin1s. Each output archives the exact source and DLLs. Legacy flag is the sole controller difference. Runtime logs `%TEMP%/pq_checks/speed_{legacy,fast}_screen_20261020.log`.

Early starts checked for both Bowl seeds and Sprawl seed7347019: raw observation, position, velocity, spin, mode and neutral-count metadata match exactly. Fixed fast plans are selected in native play; legacy selects none. Early scores/speeds are incomplete and are not promotion evidence. Source frozen until both pools finish; dashboard33732 untouched.

Added two Concentric starts in `legacy_concentric`/`fast_concentric`, ports9068/9069, one worker each. Exactly8 games total during overlap; identical controller protocol. Completed first matched results: Bowl seed12091843 improves102→120 points, mean horizontal speed11.093→12.362u/s, speed when goal is>30u horizontally away12.870→14.665u/s; both0falls/0rescues. Sprawl seed7347019 improves161→168 points, mean speed8.169→8.118, long-approach speed10.888→11.229; both0falls/0rescues. Actual opening layout dictionaries and starting position/velocity/spin/mode match. Other results still pending; do not mistake these for the final pooled result.

## Four-map screen complete; broader check running

Two-start means: Bowl90.5→124.5, Sprawl170→175.5, Cube Isle123→129.5, Concentric103→108 points. Bowl mean movement11.840→12.728u/s and long-approach speed13.557→15.041u/s. All actual opening layout dictionaries and starting motion match; per-pool randomized opening audits pass. Overall four-map falls0→2 (both Cube), accepted rescues6→1. Both Bowl rounds have0falls/0rescues with the fix. Concentric is mixed across starts (97→118,109→98), so don't imply every round wins.

Cube fall windows retained as `cube_fall_case.json` and `cube_fall_start1_case.json`. First follows a fast ramp-gem approach then a target change/A2 fallback; second follows an ordinary route plan then A2 fallback. Both have retained forecasts with unsupported braking backups. This flags an existing plan-handoff/recovery risk; it is not proof that the new arithmetic is wrong or that a fast-only gate would fix both. No safety gate changed during this experiment.

Continuing identical protocol on the remaining6 maps (KOTM, Duplex, Fractured, Cragmire, Vortex, Treasure), two seeds each, three workers per arm, ports9070–9072/9074–9076: `legacy_remaining`/`fast_remaining`. KOTM finishes144.5→143.5 mean points,0falls/0rescues either side; counted as one equal-weight map, not a veto.

`round_metrics.json` caches completed journal metrics; `native_comparison.json` is updated as matched pairs finish and explicitly marks whether all20 pairs are present. Speed averages exclude countdown/OOB decisions. Long-approach speed uses horizontal target distance>30u; no qualifying samples is null (never0). Aggregate long-approach comparisons use only maps with samples on both sides. Full per-map/seed data and opening/motion matching retained.


## Full native comparison and audit complete

All40 original-duration rounds completed (10maps x2starts x2controllers). Equal-map points124.7 to129.6 (+3.93%); equal-map horizontal speed7.992 to8.120u/s (+1.60%). Bowl points90.5 to124.5 (+37.57%), mean speed11.840 to12.728 (+7.50%), long-approach speed13.557 to15.041 (+10.94%). Falls5 to7; accepted rescues15 to10. A commentary initially said17 baseline rescues; corrected immediately after summing audited reports.

| Map | Before points | Fixed points | Before/fixed falls (2 rounds) | Before/fixed rescues |
|---|---:|---:|---:|---:|
| Bowl_Hunt | 90.5 | 124.5 | 0/0 | 2/0 |
| Concentric_Hunt | 103.0 | 108.0 | 0/0 | 3/1 |
| Cragmire_Hunt | 119.0 | 117.5 | 1/2 | 4/3 |
| CubeIsle_Hunt | 123.0 | 129.5 | 0/2 | 1/0 |
| Duplex_Hunt | 124.0 | 118.5 | 2/1 | 0/2 |
| FracturedIslands_Hunt | 89.0 | 95.5 | 0/0 | 3/2 |
| KingOfTheMarble_Hunt | 144.5 | 143.5 | 0/0 | 0/0 |
| Sprawl_Hunt | 170.0 | 175.5 | 0/0 | 0/0 |
| TreasureBox_Hunt | 175.0 | 178.0 | 0/0 | 0/2 |
| VortexEffect_Hunt | 109.0 | 105.5 | 2/2 | 2/0 |

`native_audit.json` passes:231 unique live inputs unchanged; all6 root source archives verified; every native score/input/checkpoint valid; actual opening layouts and starting position/velocity/spin/mode agree in20/20 pairs; two distinct gem layouts per map; native RNG seeds match the requested seeds. Paired prepared metadata differs only in legacy_fast_proposals. Exact per-round metrics are in native_comparison.json. Treat mixed maps and extra falls explicitly; this is a scoring gain, not universal dominance.

Full Bowl seed12091843 fast-step replay has4/2112 position errors above0.001u,3above0.01u,max0.116882u at a contact transition. Saved in bowl_contact_error_case.json. Earlier early-run report of<0.001u was limited to the first1100 samples. Native-vs-Python fast-controller forecasts remain96/96 exact; actual native contact geometry can still disagree.

## Quiet real-time confirmation in progress

Native pools stopped cleanly; dashboard33732 is the only pre-existing Python process. Running tools/realtime_bench.py with Bowl, seed7347019,8search workers, --trace, --legacy-fast-proposals in live_legacy_start0, port9380. Next: fixed controller on the same seed, then a second distinct start if the first comparison is valid. Run these sequentially on a quiet machine; audit actual initial and GO states as well as gems, because earlier phases found a bootstrap mismatch on Treasure. No controller/source edits planned.

First quiet real-time pair complete: baseline119→fixed133points (+11.76%); mean horizontal speed11.1768→12.6412u/s (+13.10%); fraction of playable samples at least14u/s18.31%→35.20%;0falls/0rescues both. Measured holds9.82→11.53s, total round wall316.3→316.7s. Exact initial layout/raw and GO position/velocity/spin/mode/held/effects/ground_ms/raw/pending bridge action match. Source unchanged in both. `live_comparison.json` retains the metrics and physical starts. Second opening seed12091843 now running in reverse arm order (fixed first, then legacy), same benchmark flags and quiet machine.

## Final real-time result and decision

Both quiet real-time pairs completed. Second start: legacy110 to fixed120 points, mean horizontal speed10.9258 to11.9323u/s, zero falls or rescues in either arm. Combined two-start mean points114.5 to126.5 (+10.48%); horizontal speed11.0513 to12.2868u/s (+11.18%). Total round wall time315.85 to316.10s is essentially unchanged. Holds increased9.22 to10.755s; this cost is retained rather than omitted. Median planning latency remains13ms.

| Bowl start | Before points | Fixed points | Before speed (u/s) | Fixed speed (u/s) | Falls/rescues |
|---|---:|---:|---:|---:|---|
| 7347019 | 119 | 133 | 11.1768 | 12.6412 | Zero in both |
| 12091843 | 110 | 120 | 10.9258 | 11.9323 | Zero in both |

`live_audit.json` passes: all four source archives verified and identical,283 unique live source inputs unchanged, both pairs have matching initial observations/layouts and GO motion/raw observations/pending actions, and the two seeds produce genuinely different gem openings. Prepared arguments differ only in the legacy controller switch and output location. Four rounds finish normally with no skipped decisions. The real-time comparison is separate from the ten-map native comparison; do not present the native Bowl37.6% gain as the real-time result.

The requested speed-and-points goal is met. Keep corrected native fast controllers enabled (`FAST_PROPOSALS=True`); retain `--legacy-fast-proposals` for reproducible comparison. No learned-model training was performed. All241 dynamics tests and96 complete native/Python fast forecast parity checks passed. This phase used44 full game rounds in total.

Remaining limitations: native results are mixed on tighter maps, with total falls5 to7 despite fewer accepted rescues15 to10. Cube Isle retains a plan-handoff/A2 fallback landing risk, including a fall after an ordinary route plan. Rare actual contact transitions still diverge from the forecast, as documented above. Two starts demonstrate an improvement in this experiment, not universal superiority. The next useful investigation is safe continuation after pickups and target changes on tighter terrain, preserving the demonstrated open-map speed gain.

## Evidence, cleanup, and stopped state

All artifacts are under `ml_agent/dynamics_reports/speed_planner_20261020`. `completion.json` records the final decision; `native_comparison.json` and `live_comparison.json` retain per-round metrics and physical starts; the corresponding audit files record validation. Thirteen execution/test logs are preserved in verified `execution_logs.zip`. Exact source/binary snapshots and failure traces remain available.

Removed80 temporary uncompressed runtime journal copies totaling3,714,404,775 bytes after verifying their retained compressed archives byte-for-byte. `runtime_cleanup.json` records the exact files, hashes, archives and successful removals. No new throwaway Python scripts were created. No checkpoint or active source was removed.

Final process inventory: dashboard Python PID33732 only; no marbleblast process, evaluation worker or training job remains. All pre-existing unrelated working-tree changes are preserved.
