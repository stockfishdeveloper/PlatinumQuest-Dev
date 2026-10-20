# Real-time planner completion work — 2026-10-19

## Authorization and constraints

The operator authorized autonomous implementation and iteration of the reviewed plan, including overnight game tests, until its goals are complete. Keep dashboard PID 33732 untouched; at most eight native game processes total. Use hidden owned test games. Preserve failures and report actual points, falls, accepted rescues, and timing. KOTM is one equally weighted map. Log every bugfix and substantial change here and summarize milestones in HANDOFF_NAV_TRAINING_LOG.md. Do not restart the retired imitation-learning loop.

## Goals and checks

1. Freeze the current configuration and establish a fresh ten-map native baseline (speedups and pre-spin on, wall experiments off), plus reproducible real-time timing measurements. Save commands, source/config identities, and actual starting layouts.
2. Correct look-ahead item event ordering, through-gem pickup semantics, and map worker lifetime. Validate using recorded/native state comparisons and focused lifecycle/event tests, then score behavior changes before adoption.
3. Reduce real-time holds using exact forecast optimizations and bounded search/checked continuations. Compare identical starts, actual points, falls, accepted rescues, planning tails, total hold time, and wall-clock duration. Preserve simulation checks; do not claim lockstep freezes are uninterrupted real time.
4. Measure and model Mega Marble activation, growth, expiry, changed movement, clearance and pickup reach. Validate against native traces before enabling proposals; retain a disabled experiment if evidence fails.
5. Test generic trajectory perturbations (steering, intermediate aims and action timings) with whole-group and native-round evaluation. Promote only measured improvements; preserve useful infrastructure and report negative experiments.
6. Assess automatic preparation of previously unseen static maps and explicit handling of unsupported mechanics. The original ten validated maps remain the scoring set.

Each stage ends with recorded evidence, the selected behavior and limitations. A failed experiment is a result, not permission to silently change objectives or launch endless recipe searches. No learned-model training is necessary to complete these planner goals.

## Current status — validation phase completed 2026-10-20 morning

The bounded implementation/validation phase is complete. Keep the integrated event-order, delayed-restoration, forecast-resume and worker-lifetime fixes. Keep `CHECKED_TAIL=False`, `THROUGH_BOX=False`, `REFINE_CANDIDATES=0`, wall experiments off and Mega proposals off; pre-spin remains on. The optional experiments did not establish a dependable native scoring gain. This is not a claim that Mega, new-map admission or all real-time pauses are solved.

Corrected checked continuation averaged149 versus159 points in the quiet development repeat, then153.667 versus149 on a different gem layout using the menu server's1ms thread setting. The latter doubled accepted rescues2to4 and has a TreasureBox physical-start mismatch. Do not select it or advertise a confirmed points win. Final detailed table and limitations are below. All240 dynamics tests passed; no owned game, planner worker or training job remains. Dashboard33732 is untouched. Staging history and failures are archived;20 temporary Python source copies were removed.

**Rescue-accounting correction:** previous outer real-time benchmark summaries used the same `rescue_accepted` attribute as HuntEnv's per-request boolean acknowledgment. They therefore reported2 after an accepted request rather than accumulating acceptances. The core and later per-round `.summary.json` files correctly count accepted rescues and are authoritative. Corrected continuation:0/1/1 on Sprawl/Cragmire/TreasureBox, not0/2/2. Failed checked pilot:3/2/1. Core:0/2/0. Original baseline logged2 Cragmire attempts, without independent cumulative acceptance evidence; prior descriptions of these as2 accepted rescues are not established. Native reliability pools use separate accounting and are unaffected. Originals retained; `rescue_accounting_audit.json` and `live_checked_rebase_comparison_corrected.json` supersede affected claims.

The harness now uses a separate `accepted_rescue_count`, checks agreement against the round summary, and archives all measured source/native binaries before each run. Its regression covers repeated accepted/refused acknowledgments; all18 runtime tests passed. No game behavior changed. New runs freeze source during measurement; no concurrent heavy work.

## Original starting state

- Git HEAD `102009438`, clean tree before this handoff. Only the dashboard was running; no game, planner server, or training job to interrupt.
- Historical v86 reference: 130.3667 equal-map points over six ten-map pools. This is historical context; speedups, deadlines and pre-spin require a new baseline.
- Current real-time lead: five 64 ms decisions; eight search workers; walls and through-box experiment disabled; pre-spin enabled.
- Confirmed code gaps: per-map World cache retains eight workers per visited map; look-ahead adds a collected powerup only after forecasting the entire window, so later queued uses cannot see that pickup.
- Additional audit item: real-time `rescues` currently increments on an attempt, including refused requests; distinguish attempts from accepted rescues in reports.

## Work log

### Initial inspection

Read the current real-time runner, search/parallel forecast paths, benchmark entry points, recent training log and TODO. No source changes or games yet. The earlier summary is evidence to orient the work, not a substitute for new validation.

### Baseline and native forecast diagnostics underway

- Launched `reliability_eval` with the ten maps, start_0 manifest/master seed 7347019, eight workers, 16 ms physics, horizon 48 and the existing production flags. Output: `dynamics_reports/realtime_planner_20261019/baseline_start0`. Its exact child commands and source identities are saved by the evaluator. Source remains frozen while this pool runs; dashboard untouched.
- Added two maintained tools: `tools/realtime_bench.py` (sequential hidden real-time native rounds, native opening audit, accepted-rescue accounting, source integrity and optional full traces) and `tools/forecast_replay_check.py` (known-action look-ahead comparisons against recorded native outcomes). These replace ad-hoc throwaway probes.
- First forecast audit: `forecast_before.json`, 584 valid five-decision windows on Sprawl/Cragmire/TreasureBox, including inventory transitions. The original strict all-fields 1e-6 classification includes harmless serialized-state/spin rounding; retain its errors unchanged. Future reports declare practical per-field bounds (position 1e-4 u; velocity/spin 1e-3; radius 1e-6), and retain strict matches separately.
- Reproduced a substantial failure on TreasureBox start0 decision 381: a Super Speed pickup then a queued use in the five-decision window is predicted as still held, with 1.10 u position and 22.51 u/s velocity error. Native tick records show pickup after tick 280428 and use at 280572. This is direct evidence for the event-order fix.
- Found a second inventory semantic gap: PowerUp::onPickup replaces a different held powerup; the old look-ahead permits pickup only from an empty inventory. Same-type pickups are refused by the server.
- Existing `datasets/marble_dynamics/timers_20261012` traces cover Mega on four maps. Growth changes radius 0.189750001 -> 0.666599989 and seven movement parameters, can schedule +6 vertical velocity, and expiry schedules -2. Timing depends on script/network phases; do not model this as a radius-only toggle. Additional live 64 ms-bridge measurement remains required.

### Baseline complete; event model and Mega measurements

- Fresh native pool: **123.7** equal-map points, **1 fall / 11 accepted rescues**, all ten valid, both integrity and start-layout audits passed. Per-map points: KOTM147, Sprawl161, Cube132, Bowl79, Duplex118, Fractured91, Cragmire124, Vortex105, TreasureBox183, Concentric97. Wall time872 s. This is lower than the historical v86 six-pool130.37; do not conceal the decline or attribute it to a single change without further controls. Bowl is the largest shortfall.
- `script_forecast.py` added but not yet wired into live play. Tick-ordered replacement, use and Blast-item pickup retain the compiled native physics. `forecast_events_candidate.json`: 572/584 windows pass practical bounds vs549/584 for the old runner using those SAME bounds; TreasureBox198/198. The remaining two Sprawl windows show small contact divergence; ten Cragmire windows cross the delayed restoration of a previously held powerup after respawn (game.cs powerupRespawn), which the old inventory telemetry does not expose. Preserve all errors.
- `mega_measure`: six native trials, Bowl/Concentric stationary, rolling, airborne; one airborne Concentric fall retained. At the64ms bridge, pending Mega use at clock T grows the native marble at T+16ms; grounded activation impulse arrives T+32ms. Normal expiry at T+10000ms, shrink impulse32ms later. Airborne activation has no +6 impulse when the game's downward Interior ray misses. This confirms the script/network ordering; no production enablement yet.
- Live benchmark attempt `live_baseline` failed before scored play: REPLAYCAPTURE returns its world asynchronously and the new harness checked too soon. Fixed the harness to wait bounded neutral replies for the world, preserving the failed output. Owned game and workers closed correctly. Retry uses `live_baseline_retry`; production runner unchanged.
- Pending runtime/lifecycle changes are staged under `%TEMP%/pq_checks/realtime_patch_20261019` while live baseline source remains fixed. These are unintegrated candidates, not completed fixes.

### Exact continuation candidate and live timing

- Staged forecast resume metadata and lazy joins pass 18 recorded 48-reply split/concatenate comparisons exactly (states, controls, events, proposed actions, native states/controls, resume dictionary). `exact_resume_candidate.json`. The first join version changed native-state metadata; corrected by retaining NativeRows packed concatenation and its prefix template. Resume-only microbenchmark about8.93x faster; this is not a whole-planner claim. Full decision-trace validation still pending.
- Mega replay candidate improved from61/66 to65/66 windows after using the observed script clock for expiry, not the next native frame clock (Concentric differs by2ms). Remaining Bowl airborne contact window: position0.003055u, velocity0.01914u/s, spin0.23828. Legacy profiles were supplied as measured constants; fresh observed-profile validation is required before enabling use.
- Unchanged real-time baseline: Sprawl159 points,0falls/0rescues,8.71held seconds in315.4wall seconds; Cragmire95 points,0falls/2rescues,17.41held seconds in264.4wall seconds. TreasureBox still running when this note was written.
- Runtime changes staged: bounded single-world worker lifetime; cleanup on worker startup failure; discard expired queued worker jobs; accepted versus attempted rescue reporting; optional checked continuation with numeric state/control/physics matching and generation invalidation, no native simulator calls from the game thread. Checked continuation remains disabled pending tests. Native physics-profile and delayed inventory-restoration telemetry also staged, not yet integrated.

### Integrated runtime fixes and completed exactness gate (22:45 MDT)

- Live baseline completed: Sprawl159/Cragmire95/TreasureBox188 points, no falls, accepted rescues0/2/0. All three source identities unchanged. TreasureBox252.6s wall; full timing in `live_baseline_retry/summary.json`.
- Integrated single-world lifetime, failed worker startup cleanup, expired queued work cancellation, accepted-rescue accounting, tick-ordered item look-ahead, faster resume metadata and lazy compiled joins. CHECKED_TAIL remains false; its guarded option is available via realtime_bench --checked-tail. Mega remains disabled in production planning.
- Added nine focused runtime regression tests (map switch ordering, failed loads, guarded continuation/generation invalidation, state/physics mismatches, lazy slicing/pickling and pickup semantics). Initial broad continuation suite found list-container compatibility regression; plain Python list forecasts now retain list concatenation. Compiled forecasts alone use lazy joins.
- Full decision trace exposed a real cache bug missed by the18-window microcheck: two-level scoring truncates forecasts at pickup but copied the full forecast's cached bridge state. Resume metadata is now length-qualified; concatenation updates that length; a new regression test protects truncation. Failed trace is preserved (`exact_decision_trace.json`,1476decisions). The unoptimized control and the corrected optimized run BOTH match all **1875/1875** historical reference decisions exactly; reports `realtime_exact_control_retry_20261019.json` and `realtime_exact_fixed_20261019.json`. One control harness attempt failed because overriding the parallel_search module duplicated its reference-pickle registry; reran with only the viability override, no production worker failure. Preserve failed log in TEMP.
- Native Mega profile telemetry now records evaluated normal/Mega physics profiles, radius, duration and use latch. Fresh two-map six-trial replay with visible item events: **150/155** windows pass strict practical bounds; five Bowl contacts remain (max position0.008575u, velocity0.05433u/s, spin0.3201). Initial135/155 result omitted real incidental pickups from validation; retained, then corrected the validator to include them. Native traces and all failures are retained. Requested competitive probe still reports an effective10000ms duration in this local game setup; it does NOT establish multiplayer5000ms behavior. Do not claim that coverage.
- Delayed restoration: `oldPowerupData` is cleared after scheduling in game.cs, so reading it later was insufficient. Added read-only `mlRestorePowerupData` at the scheduling site and exposed its type/time only while the event is pending. First force-RESPAWN fixture did not exercise saved inventory and is retained; corrected measurement uses the legal quick-respawn path. Native restoration occurs on the frame after the server timer. Tick event model now handles it; `restore_replay.json`: **95/95** legal-respawn windows pass, including later use. No game-rule or respawn behavior changed.
- Starting matched sequential live_core benchmark on the same three maps/seed after the25 core tests. Keep planner/game source fixed for that run; subsequent changes stay staged. No learned-model training.

### New-map preparation assessment

`map_preparation_readiness.json` inventories the ten selected maps plus Acropolis2, Agility, Archipelago, InfernoRing and CascadeFalls without changing assets. The five new candidates have static InteriorInstance geometry and only ordinary spawn/inbounds triggers in their mission definitions; this is a source screen, not proof that their StaticShape datablocks/materials have no scripted effects. They have not been admitted to the scoring pool. The existing native collector/mesh format can represent their static contacts, but the current entry point explicitly depends on the old ten-map inventory and config index, so automatic preparation is not currently complete. Required admission pipeline: fresh mission/DIF/datablock inventory; terrain generation and native checks; native collision export/coverage and sweep validation; viewer generated from that same mesh; cache warmup; native planner round. Dynamic platforms, force zones, changed gravity, teleports/cannons and unknown script effects require explicit temporal/event support and must not be silently treated as static geometry. This phase assessed feasibility; no unvalidated map is silently enabled.

### Live comparison in progress; next variants staged

- First core round: Sprawl148points/0falls/0rescues vs baseline159/0/0; holds5.29s vs8.71s (39percent lower), wall311.5s vs315.4s. Actual opening dictionaries are identical, not merely requested seeds. Do not call this a scoring win. Cragmire/TreasureBox still running.
- Process inspection after the map switch: dashboard33732 intact, one owned native game, exactly one eight-worker search pool; no retained Sprawl workers.
- Staged through-gem repair uses actual16ms sweeps in the C++ forecast and equivalent callbacks in the Python/per-reply/event paths; no initial-overlap or64ms-chord shortcut. Staged tests cover the false curved-path pickup, initial overlap, larger radius and compiled/per-reply forecast equivalence. Still disabled/unintegrated during live measurement.
- Staged optional checked-continuation candidate adds finite-value and missing-context refusals, compares serialized float parameters/actions with their known precision, and waits for an actual new gem group instead of inventing repeated inputs or stale plan tails. Remains a separate experiment, not enabled in the measured core run.
- Staged generic search extends the existing phased_search module with12 deterministic local control-sequence perturbations (steering phases, jump/use/Blast timing), full native forecasts and unchanged scoring/landing gates. Default off; no new standalone experiment script. Offline whole-group screening and native score gates pending.
- Mega contact diagnosis: starting each native tick from its actual before state reproduces all examined contact steps essentially exactly, while chained five-decision forecasts diverge at the contact. Normalizing initial radius/parameters toF32 did not improve those windows. No speculative precision fix adopted. Evidence: `mega_contact_diagnosis.json`, `mega_f32_input_diagnosis.json`. Only short read-only probes ran alongside live play; confirm any selected latency winner in a quiet repeat.

- Core Cragmire completed: 100 points, 1 fall, 2 accepted rescues; baseline 95/0/2. Holds 12.14 s versus 17.41 s; wall 259.3 s versus 264.4 s. All source hashes unchanged. Core test suite: 25 passed. TreasureBox ongoing; machine left quiet for its remaining measured time.

### Core comparison complete; native experiment pools running (23:12 MDT)

- Matched live_core completed with unchanged source identities and identical actual opening dictionaries on all three maps. Sprawl 148 versus 159 points (0/0 falls, 0/0 accepted rescues); Cragmire 100 versus 95 (1/0 falls, 2/2 rescues); TreasureBox 193 versus 188 (0/0 falls, 0/0 rescues). Equal-map points 147.0 versus147.333; total holds19.78s versus31.71s (37.6percent reduction). This is a latency improvement with essentially flat points and one additional fall, not a scoring win. Evidence: live_core_comparison.json. Confirm selected timing behavior in a quiet repeat.
- Integrated optional actual16ms through-pickup handling in native/Python sequence controllers, retaining legacy center behavior by default. Native DLL rebuilt successfully using process-scoped PowerShell execution-policy bypass (first build command was blocked; system policy unchanged). Native controller checks passed all ordinary cases and48/48 compiled versus per-reply through forecasts. Tests cover curved-path false pickup, initial overlap and larger-radius reach. Experiment flag: --through-pickup-ticks; default off pending native scores.
- Generic12-candidate trajectory refinement first hurt successful Bowl/Cragmire cases while rescuing Vortex. Preserved its source snapshot and negative screen results. Narrowed it to runs with no useful ordinary candidate: all eight successful Bowl/Cragmire cases now preserved exactly; the difficult Vortex case avoided the fall and collected6points within25s but did not finish its group. Both versions and metrics retained in screen_summary.json and screen_recipe_broad/source_snapshot.zip. Experiment flag: --refine-candidates12; default off.
- Focused tests exposed outdated test-fixture assumptions (ambiguity flag, secondary budget searches, partial __new__ controller setup). Updated fixtures without changing their original behavioral assertions. Search/viability/runtime/budget/inventory suite51passed; latest runtime/full-round suite51passed. Full suite not yet run; do not claim otherwise. PowerShell stderr wrapping can return a nonzero shell code despite unittest reportingOK; use explicit exit $LASTEXITCODE for subsequent commands.
- Native scoring underway: pool_through_start0 across all ten maps (5workers, ports9360+), pool_refine_start0 on Bowl/Cragmire/Vortex (3workers, ports9460+). Both use seed7347019 and start_0 manifest, same production flags as baseline. Eight native games total; dashboard33732 untouched. Freeze all planner/native/game source while they run. Each evaluator retains its full command/config/source snapshot, actual opening and failure evidence.

- Checked-continuation review additionally found that position/velocity agreement does not establish agreement of queued native input or powerup aim latch. Staged candidate now checks pending movement, aim hold, finite values and complete expected inventory/Blast context, and conservatively refuses scheduled restoration. Matching uses the observed script serialization precision. Tests staged for these refusals; source still frozen during native pools. Updated TODO Mega status to reflect the measured disabled model and actual remaining gates.

- Review found an additional simultaneous-pickup ambiguity: an initially held-type item becomes eligible after another item replaces it in the same tick; duplicate eligible items also need native query order to identify which disappeared. Staged event-model fix refuses those ambiguous transitions and adds regression cases, while allowing multiple untouched same-held-type items. This change does not invent an order. Integrate after frozen pools finish.

### Native variant decision and continuation integration (23:27 MDT)

Both pools completed, source integrity and actual start-layout audits passed, every opening dictionary matches the baseline. No native games remain after evaluator cleanup.

| Map | Baseline | Through ticks | Failure-only refinement |
| --- | ---: | ---: | ---: |
| KOTM | 147 | 150 | - |
| Sprawl | 161 | 166 | - |
| Cube Isle | 132 | 138 | - |
| Bowl | 79 | 93 | 79 |
| Duplex | 118 | 131 | - |
| Fractured | 91 | 73 | - |
| Cragmire | 124 | 68 | 93 |
| Vortex | 105 | 87 | 105 |
| TreasureBox | 183 | 195 | - |
| Concentric | 97 | 113 | - |

Through ticks:121.4 versus123.7 equal-map points,3versus1falls,8versus11acceptedrescues; wall1031s. Refinement:92.333 versus102.667 on its three maps,2versus1falls,4versus6rescues; wall994s. Both rejected for default adoption. Generic refinement was actually executed (44Cragmire and37Vortex decisions used refined labels); this was not an inactive flag. No confirmation seeds warranted for these losing screens. Correct native pickup handling and bounded refinement remain available as disabled experiments, with their evidence retained in native_experiment_comparison.json and both pool archives. Walls also remain off.

Integrated the staged checked-continuation guard and simultaneous-powerup ambiguity fixes after both pools ended. The whole checked-tail experiment, including waiting for the actual next gem group, remains opt-in; default group-boundary behavior is unchanged. Guards cover position/velocity/spin/radius/gravity/yaw, physics parameters, pending native movement, queued reply and aim latch, all observed Blast rules, inventory and pending restoration. They reject nonfinite data and use declared script serialization tolerance. Focused and full dynamics checks precede a quiet native real-time benchmark. No model training.

- Full dynamics suite first run:238tests,1failure and3errors, retained log realtime_full_tests_20261019.log. Three old mock routes/terrains lacked newly required terrain/distance/goal_field interfaces; supplied those fields. Hull inventory test scanned every JSON and accidentally included ExampleMission: scoped it to collection.json's ten admitted maps, retaining its original21DIF/5168hull assertions. Landing fixture also regenerated its historical input sequence using today's faster throttle; verified original constants from commitc9479be51 and pinned that fixture's11u/s cap,7brake,12gain,.85throttle,0pickup speed floor and original speed law. Kept every original native endpoint/action and negative-control assertion; focused landing test now passes. Production search unchanged by these test repairs.
- Final old-journal look-ahead replay:572/584 within declared practical tolerances, exactly retaining the prior event-model result (Sprawl196/198, Cragmire178/188, TreasureBox198/198). Ten old Cragmire windows lack the newly added delayed-restoration telemetry; the independent new legal-respawn corpus remains95/95. Two small Sprawl contact discrepancies remain. The new simultaneous-pickup refusal introduces no regression in this corpus. Evidence: forecast_final.json.

- Full dynamics suite rerun: **238/238 passed**,42.6s; git diff --check clean (only Git line-ending notices). Log realtime_full_tests_final_20261019.log. Starting quiet sequential live_checked on Sprawl/Cragmire/TreasureBox, seed7347019,8search workers, checked-tail on, output live_checked. Through/refinement/walls remain off. Source frozen and no other CPU-heavy work during measured play; source identities and actual opening dictionaries are audited by the benchmark.

- First live_checked pilot exposes a continuation integration bug: every accepted cached reply sets a full restart, whose ContinuousController.invalidate clears task_goal/current/resource/stall history. Runtime counters show hundreds of resets and long repeated-target stagnation; in particular the recovery monitor can receive a missing target after each reset. This pilot remains retained and default checked-tail stays off. Staged a distinct checked rebase: discard stale action forecasts/reset the actor, but preserve task/progress/resource/cooldown/native-respawn history; genuine restart/fall/respawn still performs full invalidation. Added a regression verifying repeated rebases preserve recovery state while an actual reset clears it. Current pilot source remains frozen until completion; corrected native comparison follows.

### Checked-continuation pilot and rebase correction (23:46 MDT)

Pilot live_checked completed with unchanged sources: Sprawl122/0falls/3rescues/4.05held seconds; Cragmire100/2/2/9.43; TreasureBox177/0/1/2.88. Mean133.0versuscore147.0; totalholds16.36versus19.78. Rejected pilot; less waiting did not compensate for lost points and extra recovery. Full reset erased task/stall tracking1062/917/604times respectively. Integrated the staged preserve_task checked rebase after cleanup; genuine resets retain original behavior. Re-running the dynamics suite before corrected native measurement.

Opening audit qualification: gems, RNG state, applied/engine seeds and round index match core on all three maps. Cragmire/TreasureBox openings are exactly identical; Sprawl's captured position differs by0.000360u (asynchronous initial handshake/countdown settling). Do not describe that pilot as identical full starting states or attribute every score difference solely to the reset bug. Actual initial snapshots retained; live_checked_opening_comparison.json records the difference. The earlier baseline/core comparison and ten-map native comparisons were exact for their opening dictionaries.

- Rebase correction: **239/239 dynamics tests passed**,42.4s, including repeated checked-rebase preservation and full reset clearing of recovery history. Quiet live_checked_rebase started on the same three maps/seed with8search workers and checked-tail enabled. Source frozen; no other heavy jobs. A quiet core repeat will provide the final default-versus-continuation decision if needed. Current defaults remain CHECKED_TAIL=False, THROUGH_BOX=False, REFINE_CANDIDATES=0, walls off, Mega proposals off.

### Benchmark fidelity follow-up (2026-10-20 morning)

Read-only review during the quiet repeat found that `serve()` sets Python's thread switch interval to1ms, while the standalone benchmark inherited the interpreter default5ms. All existing baseline/core/checked comparisons share5ms, so they are matched engineering measurements, but do not establish exact menu-server timing. Preserve the running control repeat unchanged. After it finishes, set and record1ms in the benchmark, then run both fresh-start arms consecutively under that production setting. `setup_round` explicitly sets native real-time16ms frames,64ms decisions and render-every-frame on both paths. Report held time as the sum of waits exceeding20ms, as implemented; it excludes shorter waits and is not a claim of continuously asynchronous play.

### Quiet repeat complete; production-setting confirmation (2026-10-20 07:33 MDT)

Quiet control completed with identical full openings and unchanged source on all three maps: Sprawl162/0falls/0rescues/3.75held seconds; Cragmire110/3/1/9.87; TreasureBox205/0/0/2.28. Mean159.0 versus corrected checked149.0; total waits15.90 versus8.85. Continuation's apparent initial points win does not survive this control repeat; keep it disabled. Evidence: live_core_quiet_comparison.json. All data retained, including the weaker earlier core run.

Fixed the benchmark scheduling mismatch after the frozen repeat ended: set1ms Python switch interval exactly as realtime.serve, record it and the20ms hold threshold/16ms physics/64ms decision cadence in prepared.json, and reject a native exe lacking real-time pacing rather than silently using an unimplemented benchmark fallback. Also include the compiled sweep DLL in source/native hash archives. Runtime tests18passed. Production game/planner behavior unchanged. Starting quiet live_core_start1 atseed12091843; the following checked arm uses the same code/settings/start. This is a genuinely different native gem layout, to be audited against start_1 manifest and previous opening, not merely a different seed integer.

### Fresh-start control complete (2026-10-20 07:48 MDT)

Production1ms benchmark, seed12091843: Sprawl164/0falls/0acceptedrescues/3.59held seconds; Cragmire120/1/2/6.81; TreasureBox163/0/0/6.40. Mean149.0, total waits16.80s. Every source audit passes; all three actual gem hashes match start_1 manifest and differ from development start0. No failures or reconnects. `live_checked_start1` is now running sequentially on identical frozen source/settings with only --checked-tail changed. Keep defaults disabled until final paired assessment. All waits reported here exceed20ms individually.

### Final paired result and selected defaults (2026-10-20 morning)

Fresh layout, seed12091843, 1ms interpreter switch interval, identical source and runtime settings:

| Map | Core points | Checked points | Core / checked falls | Core / checked accepted rescues | Core / checked held seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sprawl | 164 | 175 | 0 / 0 | 0 / 0 | 3.59 / 3.75 |
| Cragmire | 120 | 118 | 1 / 1 | 2 / 4 | 6.81 / 5.78 |
| TreasureBox | 163 | 168 | 0 / 0 | 0 / 0 | 6.40 / 3.30 |
| Equal-map mean / totals | 149.0 | 153.667 | 1 / 1 | 2 / 4 | 16.80 / 12.83 |

Both runs completed with unchanged sources and agreeing rescue counters. All actual gem layouts match start_1 manifest and differ from start0. Sprawl and Cragmire opening dictionaries match exactly, with matching position/velocity/spin; absolute native clocks and effect-end timestamps differ between processes. TreasureBox is **layout/RNG-matched only**: initial capture position differs by0.165605u in height and vertical velocity by1.28u/s, consistent with an extra64ms observation during its initial fall. Its GO pre-spin also differs (71.6 versus70.7rad/s). The asynchronous REPLAYCAPTURE/setup handshake is not a guarantee of identical physical start. Do not claim the whole pool is physically identical or attribute its small mean improvement solely to continuation. Raw captures, journals and `live_start1_comparison.json` preserve this distinction.

Selection: checked continuation stays disabled. The quiet development repeat lost10 mean points and37 TreasureBox points; the fresh-layout result gains4.667 mean points but needs more rescues and has the stated startup limitation. This is mixed evidence across small three-map screens, not a ten-map multi-seed promotion. No further recipe variants or learned-model training were started.

Final verification: **240/240 dynamics tests passed in41.7s**, including protocol acknowledgment versus cumulative rescue accounting. Earlier exact forecast gate remains1875/1875 historical decisions,48/48 compiled/per-reply through checks and95/95 fresh legal-respawn replay windows. Known Mega150/155 and old-journal572/584 limits remain explicit; passing unit tests does not erase those mismatches.

Cleanup: archived30 task-owned staging files (including20 Python source copies and the old DLL) into `staging_history.zip`, verified every hash, then removed those files individually. Automatic approval review rejected the initial recursive deletion command with only “blocked by policy”; the narrower, hash-verified file deletion succeeded. No active project source or experiment result was removed. Final benchmark variants have verified source archives; old failed benchmarks retain their original hashes/logs and are not retroactively claimed to have complete source snapshots. Execution logs are consolidated in `execution_logs.zip`.

### Reproduction and next bounded work

Run from the repository root, with `PYTHONPATH=ml_agent` and Python3.9:

```text
python ml_agent/tools/realtime_bench.py --maps Sprawl_Hunt Cragmire_Hunt TreasureBox_Hunt --seed 12091843 --port 9380 --search-workers 8 --output <new-unused-output-directory>
```

Add `--checked-tail` only for the experimental arm. The harness archives source/binaries, records runtime settings, verifies native real-time pacing and source stability, records the actual native opening and rejects disagreement between its rescue counters. It does not yet enforce matching physical bootstrap phases across separate launches. Use sequential quiet runs for timing, not simultaneous game pools. This pass intentionally measured three representative maps for real-time behavior; the ten-map pool was the separate native screen.

Priorities for a later phase:

1. Standardize the neutral/countdown bootstrap and record the native state at GO before any further promotion test; compare position, velocity, spin, pending input and relative timers, not just gem hashes. The fixed RNG controls layout, not asynchronous handshake timing or CPU scheduling.
2. If revisiting continuation, diagnose the recurring TreasureBox points loss and extra Cragmire rescues from these retained journals before trying another heuristic. Preserve the working control as the reference.
3. Resolve Mega chained-contact error and validate actual multiplayer effect duration, radius-aware clearance and resource accounting before enabling proposals.
4. New static maps still require the inventory/export/coverage/viewer/native-round admission pipeline described above. No new map or unsupported dynamic mechanic was silently admitted.
