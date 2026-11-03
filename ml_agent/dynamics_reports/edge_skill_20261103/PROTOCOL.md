# Standalone edge-hit campaign, 2026-11-03

Scope: lip contacts without Blast or powerups; no production-planner integration.

Primary task: nontrivial rising/descending lip impacts followed by a supported landing, requested exit velocity held for 0.512 s, and a supported 0.384 s braking fallback. Low-lip rolling is diagnostic only. Contact audit rejects wall assists.

Frozen geometry: 128 cells across eight maps. Four training maps, two development maps, two untouched final-test maps. Landing width, length, lateral offset, angle, rise, gap and world rotation vary independently. Start distance, speed, angle and spin vary. Generated arbitrary starts measure coverage, not guaranteed reachability. Execution from known feasible states and uncertainty perturbations are reported separately.

Native game runs are hidden, owned, isolated and unpowered. Initial planning uses the observed post-reset state. Forecast and native execution use the same feedback controller. Log actual per-substep contacts and compare native replay with the C++ simulator; never replace native future states with forecasts.

v3a baseline: first 64 development starts yielded 24 accepted solutions (22 high-lip ground, 2 rolling), with zero solutions among 18 arbitrary airborne starts. These are development results, not held-out accuracy. Six familiar previous jump examples passed the new contract. Initial native checks are in native_development_v3/.

Frozen test maps/groups 6 and 7 must remain unopened for optimization until controller and proposal selection are frozen. Evaluate broad unselected native starts as well as accepted plans. Report admission coverage separately from execution reliability. Do not imply success from lockstep establishes a real-time latency guarantee.

Prior implementation is archived in previous_implementation.zip. Iteration source checkpoints are stored under checkpoints/. Consolidate maintained modules before completion; preserve reviewable results and representative 1x examples.

## Progress through approximately 08:05 UTC (01:05 Mountain)

Training v3: 1,152 unselected starts, 591 solved (537/720 ground, 4/360 arbitrary air, 50/72 low lip). Derived 1,734 airborne anchors, 1,731 exact continuations. Forced descending-contact curriculum: 151/240 solved; 755/755 derived airborne continuations. These are training/development results only.

C++ standalone controller is in edge_execution.cpp and compiled.py. It includes the unchanged production physics source in a private, hashed DLL under TEMP. Production source/DLL and planner imports are untouched. Python/C++ conformance tests include queued-state guidance and native bridge resumption. Successful v4 trajectories: all 34 exact, including 14 queued-state-guided plans.

Fast proposer baseline on all 576 development starts at budget64: analytic198, retrieval235, combined239. These used v3a trained on initial + airborne-anchor data. Timing under concurrent work was about70ms median /108ms p95, not a real-time certificate. v3b also includes descending training.

Native: v3 development24 accepted/24 success (32 total); descending31/31; v4 development28/31 accepted succeeded (64 total). The three failures exposed a numerical feedback/input quantization instability: about1e-8 state differences changed rounded yaw/world input by1e-6, causing a camera-axis quantization jump of1/16. Fixed-action replays remained accurate. New stable_inputs encoding moves requested input amplitudes inside their intended quantization bins (ordinary engine-clamped axes, no stronger physics). Unit tests cover7,007 angle/throttle combinations. The next stable native batch passed33/33 accepted,31declined. More native confirmation required.

Queue-aware controller family2 forecasts the already committed64ms before selecting its correction. Families0/1 remain in the search as alternatives. Do not confuse queued prediction with substituting future native observations.

Parameter tolerance and fresh-state planning are different metrics. Original fixed-parameter perturbation success was204/576 at size1 (position +/-0.03, planar speed +/-0.25, spin +/-1). Fresh96-candidate replanning solved292/372 perturbed grounded starts and392/488 perturbed airborne starts. Deeper2048-candidate repair solved96 of the remaining176; total780/860. Late airborne states have little control authority before contact. Robust parameter selection alone only produced5 roots passing32/32 neighborhood tests; do not claim broad fixed-parameter robustness.

Real geometry frozen: six locations on each of ten maps, alternating real_dev / sealed real_test locations;374 cases total,194 development. Ground resets and synthetic airborne encounters are explicit. Current scope remains horizontal lips/landings; world yaw, landing-relative yaw and offsets vary. No real-route reachability or points claim. Old real survey demanded13m straight landings and found only6 edges; the new shorter supported exits cover all10maps. Real development teacher v4:98/194 solved. Final real_test180 cases remain unopened.

Current active jobs: four training_v5_00..03 shards (288 each,budget1024,stable_inputs=True,warm v3); two training_descend_v5_00..01 shards (120each,budget1024,stable,warm v3 descending). New training labels must retain the stable_inputs flag. Older labels are for the legacy encoding and must not be silently mixed with stable labels.

Optional retained execution was added to experiment.native for an ablation: send the checked action table while errors stay small; refresh before contact on meaningful drift. Default remains policy execution. It has not yet been validated and is not the production recommendation.

Cleanup: twelve obsolete Python entry points removed from the working tree after confirming previous_implementation.zip. Current maintained files are common, maps, native_lab, execution, compiled, audit, learn, experiment, review, test_skill, __init__, and edge_execution.cpp. Old data/videos remain preserved. README needs replacement and final report/viewer still needed.

## Progress through approximately 08:43 UTC (01:43 Mountain)

Stable flat training: 609/1152; forced descending 162/240. Exact airborne training continuations: 1808 ordinary +810 descending, all passing. Sloped laboratory: 64 additional geometry cells (32train/16dev/16sealedtest), tilted lip lines and independently pitched/rolled landing planes. Native geometry corner agreement <0.00005; private controller Python/C++ parity on96 flat/sloped trajectories. Sloped training282/576;1107 exact air continuations. Forced descending sloped training108/128;540 exact continuations.

Native stable real development:26 accepted/26success,14declined,all10maps represented. Native sloped development25/25accepted,7declined. These are still development data. New full native runs use the fast learned proposer independently (no teacher parameter warm start):194real starts,288sloped starts, ports9580/9581.

Proposer v6a trained on stable flat+sloped data before the forced descending slope and neighborhood additions. Development at48candidate budget:240/576flat,129/288slope,101/194real. Lowered ordinary inference verification from3alternatives to1 without changing the selected first verified candidate. New compiled contact replay matches Python audit on290trajectories (including failed plans and real maps), physical states exact;1.728sPython vs0.144sC++. It independently replays initial state+command table; no future-state copying. Six tests pass, including32compiled-vs-reference audits and resource rejection.

Private DLL version7 hash4a6e8a757f44c17b. Production code untouched. A subsequent inference-only early-accept option skips refinement when a warm proposal already passes the full audit; deeper teacher collection remains exhaustive. This needs its own coverage/latency/native confirmation before freezing.

Training augmentation in progress:366successful roots on92training geometries,12perturbations each, size8 (position +/-0.24m,planar velocity +/-2,spin +/-8),384candidate fresh solve warmed by parent parameters. Held-out locations remain sealed. Current source entry points consolidated; README updated; review viewer still needs wider dynamic framing.

## Freeze, 2026-11-03 08:10 Mountain

Overnight work stopped at 01:49 with nothing running. Unit tests 6/6 pass on the maintained tree. Proposer v7a (neighborhood-augmented) re-evaluated quietly: dev 238/576, real_dev 98/194, sloped_dev 134/288, versus v6a 240 / 101 / 129. A tie at 470/1058, so v6a is frozen (it already carries the native development confirmation: 88/88 accepted real_dev, 127/127 accepted sloped_dev, zero sim/native mismatches). Hashes of the checkpoint, every controller source file and the three sealed suites are in freeze.json. The sealed suites are opened after this freeze and run once: simulation evaluation at budget 48, then native execution with the same proposer, policy execution. No retuning after opening.

## Sealed suites opened and run, 2026-11-03 08:08 to 08:27 Mountain

Simulation admission at budget 48: flat test 239/576, sloped test 129/288, real test 95/180. Native, proposer planning from the observed post-reset state: flat 222 accepted / 222 succeeded, sloped 122/122, real 72/72; 1,044 starts, 416/416 accepted successes, zero sim/native mismatches among accepted trials, zero setup errors, 0/628 exploratory successes on declined starts. Falls after the success window on accepted trials 12/416; later departure below landing height 16/416. Report in REPORT.md, numbers in report.json, viewers review_sealed_showcase.html and review_real_test.html. Campaign closed before planner integration.
