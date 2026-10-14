# General marble dynamics: autonomous preparation handoff

Started 2026-10-11 (America/Denver); final review completed 2026-10-12 at approximately 01:37.
Operator authorized autonomous implementation of the staged
map-aware dynamics plan, game probes, replay/data collection, bounded predictor fits, viewer overlays,
and an encoder prototype if the prediction foundation validates. Existing policy/PPO training remains
paused. Record every bugfix and medium/major change here as work proceeds.

## Required scope

The target is general marble mechanics: rolling/slipping/braking, linear and angular momentum,
floor/wall/ceiling/ramp/edge/corner contacts, rebounds, jump timing and multi-contact sequences.
Powerups are modifiers of the same system. Edge hits and spin-dependent launches must arise from
contact dynamics, with varied incoming state and geometry; named tricks are not the architecture.
Predict position, velocity, spin, contacts/effect state and continuation through landing.

Use the verified collision maps from `collision_maps/` as geometry authority. Preserve all contact
orientations/materials, including undersides and glass; exact normal/tarmac seam ties are accepted
material-boundary uncertainty. Static geometry does not replace live dynamic/effect state.

## Milestones and gates

1. Versioned state/action contract; confirm tick timing, delayed controls, reset/replay reliability.
2. Diverse, provenance-tagged trajectories with whole-trajectory/map splits; retain failures honestly.
3. General dynamics predictor and predicted/actual 3D overlays. Measure one-step and rollout errors,
   contact timing, rebound height, landing position/velocity, and uncertainty against clearance.
4. Whole-map surface hierarchy plus detailed queries; benchmark encoding and preserve distant detail.
5. Bounded planning pilot only after prediction gates; evaluate actual gems and safe continuation.
6. Policy integration/comparison follows review; this work does not resume long-term policy training.

## Current state / next action

- Native physical-state/applied-control recording, six archived datasets, an independent general
  mechanics predictor, bounded candidate replay, actual/predicted 3D overlays, exact whole-map
  hierarchy queries and a randomly initialized neural encoder prototype are implemented.
- Ordinary mechanics are in scope throughout; no list of named tricks or map-specific maneuvers
  controls the predictor. Powerups change the same contact/velocity/spin system.
- 632 accepted principal fixtures / 128,733 native transitions across the ten-map pilot and four-map
  supplements, plus 96 retained repeats. Repeats are not additional independent fixtures. Detailed
  development replays are also retained separately. Final archive audit and report refresh completed.
- Fifteen regressions pass. Most one-step motion is near numerical roundoff, but longer contact
  sequences retain multi-unit outliers. The prediction/landing gate has not passed. The nominal
  predictor and selected microscopic perturbations are not calibrated uncertainty estimates.
- The current policy still uses its existing observations/planner. No policy/reward/checkpoint
  changes, PPO run, dynamics optimizer, encoder training or checkpoint promotion occurred.
  The independent mechanics port is the predictor; the neural map prototype has untrained weights.
- All game probes are owned hidden instances on 9053 and have completed/closed. The operator's
  installed game/dashboard were never stopped. Final report/viewer checks are read-only.
- Next substantive work: calibrate multi-contact outcome uncertainty on new geometry/state starts,
  implement the external-event/control-delay layer, and establish landing-clearance/continuation
  gates before attempting a bounded live planner. A learned batch predictor can use these native
  trajectories and the exact map query API; a generic residual fit is not proof it fixes branch errors.

## Reproduction / artifacts

Run from the repository root in PowerShell. Python used here is 3.9 at the path below. No new package
was installed into the user's interpreter; Playwright verification dependencies live under TEMP.

```powershell
$taskPython = 'C:/Users/doug/AppData/Local/Programs/Python/Python39/python.exe'
$env:PYTHONPATH = 'ml_agent'
& $taskPython -m unittest nav.marble_dynamics.test_native nav.collision.test_queries
& $taskPython -m nav.marble_dynamics.review
```

- Primary machine-readable review: `dynamics_reports/review_20261012.json`. Human summary/table:
  [REVIEW_MARBLE_DYNAMICS_2026-10-12.md](REVIEW_MARBLE_DYNAMICS_2026-10-12.md).
- Review the exact geometry and motion: [viewer/index.html](../collision_maps/viewer/index.html).
  Choose a Recording, Focus trajectory, scrub Time, and use Largest error. Cyan is native motion;
  coral is predicted. Material seam A1–A4 controls remain. Cube Isle includes 24 rebound hypotheses.
- Predictor/input implementation: `nav/marble_dynamics/{physics,contact_order,rollout}.py`.
  `rollout.predict(engine, initial_state, controls, hierarchy=...)` accepts controls containing a
  native `move` and optional absolute camera-command `events`. It advances the known clock;
  no future position/spin/effect/contact/camera observation is supplied. A plain Move assumes
  no camera setter. This API does not translate delayed TCP ACTION commands or model item events.
- Map access: `representation.SurfaceHierarchy` and saved `collision_maps/hierarchies/*.npz/.json`.
  `detail(lo,hi,origin)`, `candidate_corridor(positions,radius,margin)` and `model_inputs(origin)`
  expose distant surfaces as well as local contacts. Exact detail includes normals and materials.
  All 47,932 triangles are retained; coarse neural tokens are explicitly not a lossless substitute.
- `datasets/marble_dynamics/`: pilot_20261011 (all ten), effects_20261012, timers_20261012,
  interventions_20261012, controls_20261012 and wall_jumps_20261012 (four-map supplements).
  Per-map manifests retain raw hashes, actual applied input/state, requested interventions,
  fixture/reset failures, external changes and executable/instrumentation identity. Old tapes
  are not silently rewritten to invent missing input/state channels.
- Partial legacy schemas: pilot lacks previous best-contact normal/exact timers; early effects
  lacks exact clocks; tapes before controls_20261012 lack absolute camera command events.
  Requested held item can change through a real pickup. Native used item IDs are authoritative.
- `contact_diagnostics_20261012`: detailed same-fixture native force/contact phases for debugging;
  these are ground truth, never prediction inputs. `smoke*` are earlier development recordings.
- Reports: `dynamics_reports/<dataset>/*.report.json` / `.trajectories.json` / `summary.json`.
  Current-version checks prevent mixing old baselines. Source-order correspondence checks are in
  `dynamics_reports/source_order_checks.json`. Each viewer `.viewer.json` fingerprints its source
  trajectories; `browser_checks.json` verifies exact rendered geometry/paths and controls.
- Generated reports/hierarchies and the separate EXE are ignored by current Git rules; datasets
  and code are uncommitted. They are present in this workspace. No commit or remote upload made.
  Preserve these artifacts or regenerate them before moving this work to another checkout.
- Runtime logs/raw files: `%TEMP%/pq_checks/marble_dynamics_20261011`. Archived gz evidence in the
  repository is the durable authority; TEMP source filenames may be reused by later collections.

Read-only reevaluation / regeneration:

```powershell
foreach ($taskSet in @('pilot_20261011','timers_20261012','interventions_20261012','controls_20261012','wall_jumps_20261012')) {
    & $taskPython -m nav.marble_dynamics.evaluate --data "ml_agent/datasets/marble_dynamics/$taskSet" --output "ml_agent/dynamics_reports/$taskSet"
}
& $taskPython -m nav.marble_dynamics.uncertainty
& $taskPython -m nav.marble_dynamics.review
& $taskPython -m nav.marble_dynamics.representation --encoder-smoke
& $taskPython -m nav.collision.viewer --trajectories ml_agent/dynamics_reports/pilot_20261011 ml_agent/dynamics_reports/timers_20261012 ml_agent/dynamics_reports/candidate_replay_20261012 ml_agent/dynamics_reports/contact_sensitivity_20261012
$env:PYTHONPATH = 'ml_agent;' + (Join-Path $env:TEMP 'pq_checks/collision_maps_20261011/browser_deps')
& $taskPython -m nav.collision.browser_check
```

Probe rebuilding / new collection (these launch only owned hidden games; port 9053 must be free):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ml_agent/nav/marble_dynamics/build_probe.ps1
& $taskPython -m nav.marble_dynamics.pairs --output ml_agent/datasets/marble_dynamics/controls_NEW
& $taskPython -m nav.marble_dynamics.pairs --jump-contact --groups 3 --output ml_agent/datasets/marble_dynamics/wall_jumps_NEW
# Pilot was: all ten config map names, --trials 36 --replays 8.
# Timers were: Concentric, Sprawl, CubeIsle, Bowl, --families powerup effect_expiry --trials 12 --replays 2.
```

External engine checkout: `C:/Users/doug/src/OpenPQ-TGEMIT-mbx`, baseline
`cb44e8a69f7ae79ce349c8519bd519647f429bb6`. Canonical repository includes are
`nav/collision/engine_geometry.inc` and `nav/marble_dynamics/engine_dynamics.inc`; builders copy them
to the external engine, insert read-only hooks and compile `marbleblast_geometry_probe.exe`.
The shipped `marbleblast_mbx.exe` is unchanged. VS build has preexisting vswhere/audio/cast warnings.
Camera setter hooks record actual commands; no native forces/collision/timer/control decisions change.

## Evaluation contract and remaining limits

- Score actual applied controls, not requested joystick timing. Native decisions are usually 16 ms
  with <=8 ms physics substeps. Reset/bridge acknowledgements consume simulated time; first recorded
  native state is the predictor start. Jump/use can arrive several native calls after being sent.
- Open-loop p/v/spin remain predicted throughout. Applied camera input is an exogenous action tape;
  complete-control candidate checks replay setter commands instead of copying future camera state.
- Stop continuous rollouts at mode/OOB, respawn/warp, item, radius/datablock/parameter or other
  external velocity/spin changes. Keep excluded samples and zero-length prefixes explicit. This
  tests continuous mechanics, not full-round pickups/respawns/Mega/dynamic actor prediction.
- Report fixed-horizon errors, endpoint tails, peak height, contact interval matching, landing
  p/v/spin and one-second continuation. First support is not proof of a stable or useful landing.
  Clearance-specific reliability needs both state-error and landing/continuation checks.
- Native original polygon membership, F32 coefficients/radius/material parameters and local DIF
  hull/bin order matter. Rotated OBB culling is conservative; cross-object order and repeated native
  contact multiplicity may differ. Tangency sqrt/NaN and grazing slide/bounce branches are diagnosed,
  not declared resolved. The accepted tarmac seam ties remain ordinary boundary uncertainty.
- General downward-gravity, static-map mechanics and five native powerup modifiers are implemented.
  Mega/scripted size changes, Blast, arbitrary gravity matrices, live item events, moving platforms/
  other marbles and full bridge command-delay prediction still need explicit state/event handling.
- Encoder smoke proves access/shape/finite output/latency only. No learned semantics, skill, policy
  improvement or trick discovery has been demonstrated. Full-map information is available to the
  new predictor/access API; the existing trained policy does not yet consume it.

## Change log

- Initial handoff created before implementation, explicitly including ordinary contact/spin physics
  and the operator's edge-hit requirement. No policy or reward changes.
- Added recording-only begin/end hooks around native `advancePhysics` in the separate probe. Samples
  include the effective Move, exact before/after linear/angular state, gravity, radius, native effect
  timers, datablock parameters, per-call telemetry deltas and all final-substep contact records.
  Added guarded `DYNAMICSRECORD` / `DYNAMICSSTATE` bridge commands; normal executables ignore them.
- Build correction: MarbleData's physical parameters are private; read the registered F32 field
  offsets, preserving native float values without changing access or physics. The helper normalizes
  line endings before inserting its hooks. Use process-scoped PowerShell `-ExecutionPolicy Bypass`
  for builds on this machine; no global execution-policy setting was changed.
- Initial Concentric smoke produced continuous 16 ms transitions, with two native 8 ms substeps,
  and bit-identical repeat trajectories for roll, switch and braking (202 steps each; position,
  velocity and spin max error zero). Data is in `datasets/marble_dynamics/smoke_20261011`.
- Recorder correction: edge-timing floor lookup now uses the collision API's `cull_backfaces`
  argument (the first smoke stopped cleanly at this keyword mismatch, preserving earlier trials).
  Invalid repeats are retained as failed replay checks rather than assuming a repeat file exists.
- Second Concentric smoke: 18/18 valid trials, 2,780 actual native transitions. Eight ordinary/
  impact replay cases have zero position/velocity/spin error on shared prefixes. One OOB trial
  differed in termination length despite identical shared motion; do not call episode timing
  deterministic. Native records identify the actual Super Jump and Super Speed firing Move.
- Extended state contract with smooth-camera state, next yaw/pitch, pitch and exact registered
  datablock mass. Powerup recording supports complete five-second effects / ten-second Mega
  duration plus continuation, and configurable trial families. These remain controlled fixtures,
  not a claim of complete naturally encountered state coverage.
- Completed the first all-ten-map collection: 360/360 accepted fixture trajectories and 57,678
  native transitions, plus 80 retained repeat runs. All repeats have zero shared-prefix physical
  error; 17 episode lengths differ. Ten trials contain native/server respawn discontinuities.
  Raw evidence stays intact; evaluation truncates at external position changes/invalid modes,
  rather than training a predictor to teleport. Roles were fixed before sampling: Concentric /
  Cube Isle dev, Bowl / Vortex test, remaining six maps train. These are held out for this new
  dynamics experiment, not unseen by historical policies.
- Added an independent general contact-mechanics baseline over the verified collision mesh:
  <=8 ms integration, rolling/slipping/braking, angular/linear impact transfer, jump impulses,
  surface sliding, bounce/force materials, simultaneous contacts and native powerup modifiers.
  Uses all relevant candidate geometry, grouping native polygon fans before contact selection.
  This is a separately implemented port, not calling native advancePhysics as its prediction.
- Added per-step and open-loop evaluation with native applied inputs and camera settings.
  Predicted p/v/spin are never replaced with observations during a rollout. Raw command-delay
  translation remains a separate integration task; applied-input replay is not policy execution.
  Initial sparse checks are excellent, but full contact/activation and rollout gates are pending;
  no planner or learned encoder is approved by those sparse checks.
- 2026-10-12: corrected near-zero repeat collision handling: the predictor previously discarded
  remaining substep time. It now retries a tiny boundary resolution and explicitly refuses an
  unresolved iteration limit. Native comparison sweeps optionally reproduce F32 discriminants
  and the engine's 0.1 ms exit-root tolerance; default geometric queries/NPZs remain unchanged.
- Recover exact native polygon/fan IDs from hash-checked archived snapshots. Source key/plane
  alone is not a unique polygon identity. Native contact selection now uses original vertices
  and stored planes, including their rounding mismatch. Corrected Torque's opposite three-point
  plane convention while implementing this; the conventional cross-product sign failed loudly
  in the development regression and was fixed before any planning or model fitting.
- Extended recording with the last best-contact normal (needed by Super Speed even while airborne)
  and actual per-datablock effect durations. Rebuilt only the separate probe successfully. Collecting
  an additional four-map effects dataset including Helicopter and script-driven Mega, with long
  continuations and preserved replay/OOB/external-state evidence. Mega remains unsupported by
  the continuous predictor until its script-driven datablock/radius changes are modeled.
- Evaluation summary now merges existing per-map reports during partial runs, rather than
  overwriting previously evaluated maps. Full held-out-map checks and viewer overlays pending.
- First full all-map evaluation completed. Medians are near numerical roundoff, but rare longer
  rollouts diverge by tens of units; this explicitly fails the blanket landing/planning gate.
  Do not headline only median error. All raw trials remain available, including failures.
- Added actual/predicted motion selector, trajectory focus, time scrubber, largest-error focus,
  and velocity/spin/time hover and readouts to the existing hash-checked ten-map viewer. Browser
  checked all ten, rendered geometry hashes unchanged, plotted coordinates exactly match saved
  trajectory arrays, controls and seam focus work, and no JavaScript errors. Preview PNGs saved.
- Additional effects collection completed: 48/48 valid fixtures, 12,977 transitions, eight repeats.
  Found two state-contract issues during its evaluation: native timers expire after integration
  using the simulation clock (not subtracting dt immediately at activation); and Mega changes
  radius/parameters in script callbacks outside native physics. Added exact clock/deadline state
  recording and timing handling; one-step legacy Super Speed samples now refuse missing best-
  contact normals instead of assuming zero. Open-loop checks stop at external parameter/radius
  changes, not secretly supplying future script state. Mega activation/expiry and Blast require
  a separate discrete-event layer; their full mechanics have not been validated.
- Added neutral airborne-to-floor effect-expiry trials for clock/expiry coverage, complementing
  forward-moving fixtures that often fall before expiry. New collection with the complete clock
  schema is in `datasets/marble_dynamics/timers_20261012`; data collection remains owned/bounded.
- Added native interval-aligned collision matching, support disagreement, rebound/peak-height,
  first landing time/position/velocity/spin and one-second continuation checks to every trajectory.
  Landing means first solver support after two airborne intervals, not certified stable recovery.
- Whole-map access prototype implemented: all 47,932 triangles partitioned exactly once, 64–512
  leaf tokens per map, detailed AABB/candidate-corridor queries anywhere in the map, no near-only
  or nearest-K cap. 2,000 hierarchy queries match the reference exactly after serialization/reload.
  Untrained 64-channel encoder consumes every triangle and runs in ~0.75–2.1 ms/map on CPU.
  Pooling is lossy; exact retrieval remains mandatory. Python tree queries are slower than the
  current vectorized reference at these map sizes; this is an access/representation prototype,
  not a claimed acceleration or learned map understanding. Artifacts: collision_maps/hierarchies/.
- Important ordinary-physics fix: overlapping support constraints must recompute their normal
  force after each preceding constraint changes acceleration. The first port reused the original
  force on every surface, double-canceling gravity on floor joins and multi-contact ramp contacts.
  Corrected the sequential native rule and native F32 friction/jump rounding. Development-map
  contact/rollout regression runs now check this before any learned residual or planning fit.
- Added opt-in read-only native per-substep contact-entry state, contacts, post-contact/pre-
  integration state and linear/angular acceleration traces. `DYNAMICSDETAIL 1` enables these only
  on the separate probe. The reproducible saved-fixture diagnostic tool reruns Concentric's
  remaining worst contact cases, to localize geometry/order/force differences without claiming
  aggregate accuracy proves safety. Ordinary recording remains compact by default.
- Native force diagnostics exposed an integer-inference bug: JSON serializes max-roll speed 15
  as an integer, so a NumPy target array truncated preserved rolling overspeed components. Force
  target arrays now explicitly use float. This removes the ~0.02–0.08 rad/s spin error band; retain
  a regression case with overspeed input. Major corner differences remain tied to native contact
  ordering, which the whole-world polygon export does not preserve for every local hull query.
- While reconstructing native hull query order, found/fixed a vendored hxDif reader/writer bug:
  convex hull bounds are stored minX,maxX,minY,maxY,minZ,maxZ in InteriorIO, but the parser used
  minX,minY,minZ,maxX,maxY,maxZ. Verified all 5,168 hulls in all 21 unique source DIFs against their
  actual hull vertices: no reversed bound and zero vertices outside bounds. Existing verified
  collision NPZs came from the engine, so they do not need rebuilding. No map file was rewritten.
- Reconstructed native DIF coordinate-bin/hull/surface traversal for contact response, using
  source-file hashes and polygon-to-source vertex matching. All 23,348 referenced source surfaces
  across the ten maps match reviewed exported polygons (see source_order_checks.json for counts).
  The local native ordering removed Concentric's major corner error: final rollout p95 ~0.0083u,
  max ~0.057u on its 36 pilot fixtures, versus earlier multi-unit drift. Rotated-box hull culling
  is conservative and cross-object traversal/multiplicity are still possible limitations. Further
  development checks focus on Cube Isle's remaining outlier; no map-specific maneuver rule added.
- Found an inventory event mislabeled as physics drift in the old Cube Isle Super Bounce fixture:
  it picked up Helicopter before firing, replacing held ID 3 with 5. Open-loop recordings now stop
  at external inventory/velocity/spin changes as well as size/parameter/position changes, retaining
  the raw evidence. Requested fixture powerup is not proof of the actual fired powerup; actual
  native use events are recorded separately. Future full-round prediction needs live item events.
- Restored exact F32 plane/vertex values from the nine-digit export before doing F64 contacts,
  and reproduced native arithmetic ordering in contact projection and velocity cancellation.
  Tiny grazing-dot sign differences at a ramp/floor join can select a slide or a bounce, causing
  substantial later drift. This is a general multi-contact uncertainty issue, not a named trick.
- Added explicit grazing-contact branch ambiguity flags and a 24-member development sensitivity
  experiment. On Cube Isle's ramp/floor fixture, 1e-8u pose / 1e-6 velocity-spin perturbations
  produce ~0.526u error and different rebound trajectories after one second, while one member
  matches the native trace within 1.7e-7u. This proves a single nominal rollout can hide outcome
  branches. It is a selected-case sensitivity check, not calibrated probabilistic coverage.
  Saved all hypotheses for viewer review in dynamics_reports/contact_sensitivity_20261012/.
- Added matched physics interventions across four maps: reverse/zero/forward spin from the same
  drop, four jump timings at the same wall/floor fixture, and none versus all five native powerup
  modifiers from the same airborne state. These collect mechanics labels independently of gem
  reward. Initial p/v equality is checked in native recordings; falls and pickups remain visible.
  Dataset: datasets/marble_dynamics/interventions_20261012/ (collection currently in progress).
- Reports now fingerprint predictor/evaluation/geometry-loader/parser code and dataset manifests.
  Summary files separate reports from older code versions so partial reruns cannot silently mix
  baselines. Requested powerup and actual native firing events are separate in the viewer readout.
- Added six native-outcome regressions: fractional overspeed/spin, actual local wall/floor contact
  order with measured error bounds, exact effect expiry, missing Super Speed state refusal,
  inventory-event segmentation, and all 5,168 source hull bounds. A too-tight initial corner
  assertion failed after F32 restoration; retained exact order checking and documented measured
  tolerances (0.005u, 0.5u/s, 2rad/s) that reject the original large ordering error, without claiming
  numerical identity or safe landing. Existing seven analytic collision tests also run.
- Fixed hxDif.Dif.Load/Save file-handle leaks exposed by repeated source checks (close in finally);
  no map files were saved or changed. Dataset collection completed before changing the parser.
- Restored recorded F32 radius/datablock parameters to their exact F32 values before F64 mechanics;
  nine printed significant digits round-trip F32, but reading them directly as F64 changes contacts.
- Resolved the failing corner regression explicitly: native accepts one extra floor-edge contact
  through a last-bit sphere/plane sqrt/NaN comparison branch. The nominal predictor still omits it;
  it now reports the valid alternate contact as an ambiguity. The regression requires exact order
  of the shared contacts, the omitted native normal to appear as an alternate, and measured p/v/spin
  error bounds. This is a disclosed mismatch, not an exact five-contact match or a landing guarantee.
  All six native regressions and seven analytic geometry tests pass (13 total).
- Matched intervention collection completed: 52/52 valid fixtures across four maps. All twelve
  matched groups have exactly equal native initial positions/velocities; spin/timing/effect differ
  by design. Final totals and outcome comparisons will be added after consistent-version evaluation.
- Final evaluation fingerprint now also includes outcome scoring code. Rerunning pilot, complete-
  clock timers and interventions with the same final predictor version before rebuilding the viewer.
- Added a bounded candidate rollout API (`rollout.predict`): it takes an initial full physical state
  and proposed native Moves, advances its own simulation clock, and receives no future camera or
  physical observations. Optional surface-hierarchy queries return every triangle along the predicted
  swept corridor. Exact native expiry regression passes without future recorded state (14 total tests).
  This is the prediction interface, not an approved planner or bridge command-delay model.
- Viewer now labels matched intervention values explicitly, handles sensitivity cases without landing
  scores, and awaits exact trajectory coordinates during browser checks (avoids asynchronous same-
  length restyle races). Native use events exclude use presses when the inventory is empty.
- Major additional control-contract finding: replaying only native Moves from the initial state
  produces up to ~57u error on matched cases, despite the applied-camera evaluation predicting them
  well. The bridge calls setCameraYaw directly (steering and powerup aim) while Move.yaw stays zero.
  Added recording-only native yaw/pitch setter hooks and a queued `control_events` input channel;
  candidate prediction applies these actual/proposed commands, not future camera observations.
  Old tapes remain intact and are labeled incomplete for autonomous candidate replay. Collecting
  a separate complete-control dataset after rebuilding the owned probe. Delayed ACTION translation
  remains a separate gate. Also retained sent powerup/camera aim explicitly in collector manifests.
- Recorder close now emits valid JSON marked `aborted` if interrupted during an open transition;
  it still fails sample/structure acceptance rather than silently counting an incomplete transition.
- Complete-control native collection succeeded (52/52 fixtures in `controls_20261012`, with new
  executable/instrumentation provenance). An eight-step real corner fixture now guards absolute
  camera-command replay: complete commands match, while Move-only controls demonstrably fail.
  Fifteen tests pass. Native camera setters retain original behavior; hooks only append inputs.
- Corpus audit now recomputes all repeated p/v/spin prefixes directly from the archived files,
  rather than trusting replay summaries. Principal raw hashes and recording completion/counts are
  checked. All zero-length continuous prefixes are explicitly listed in evaluation reports and
  endpoint tables use accepted native fixture denominators, preventing excluded initial modes
  from looking like evaluated successes. Sensitivity reports also fingerprint predictor sources.
- Coverage finding: most early matched wall-jump commands occurred after an incidental rebound
  and had no support at the firing interval (Bowl's four were effective). Do not count varied
  requested timing as proof of an actual jump/edge-hit experiment. Added supported, rolling-spin
  approaches with 2.5u wall clearance, a no-jump control and five timings. Native commands/contacts
  reveal which jumps really actuate. Collecting three independent geometry starts per four maps
  in `wall_jumps_20261012`; no named trick or map-specific navigation rule is introduced.
- Supported wall-approach collection completed: 72/72 accepted fixtures / 14,323 native transitions.
  46/60 jump-command intervals have native support, versus 4/16 originally. This is a support proxy
  for actuation, not an exact impulse counter. Twelve no-jump controls remain. All matched groups
  preserve equal initial p/v. Contact outliers persist (Bowl endpoint max ~0.535u); no planner launch.
- Complete-control candidate replay succeeds without future camera/state: the initial 52-case set
  has endpoint p95 ~5.56e-6u and max ~0.000424u. The newer wall-contact set is harder; both are scored
  separately and together in the review. Saved direct candidate replay paths in
  `dynamics_reports/candidate_replay_20261012` for the final viewer, rather than showing only the
  older applied-camera-state evaluation. Broad ten-map pilot still contains multi-unit tails.
- Final audit: 632 principal raw hashes match; all 728 principal/repeat archives have complete
  structure/counts. Recomputed 96 shared repeat prefixes have zero p/v/spin difference; 19 termination
  lengths differ. All 124 complete-control candidate rollouts finish: combined endpoint p95
  0.000394u, max 0.5021u; wall-only p95 0.001717u. These selected checks do not erase broad-map tails.
  Consistent evaluation version: 44aba1cfc83f8e131b11ef8982c9a71b7088eb6541b28e894136fe77ed50ad63.
- Broad pilot scores 350 continuous prefixes from 360 accepted native fixtures; ten start in an
  excluded mode/OOB state and are listed explicitly. Native/local source-order unranked contacts
  are zero on the final pilot. First-landing comparison detects both in 214 cases; worst landing
  position difference is 11.55u and one-second continuation error 6.50u. Peak-height difference can
  reach 3.81u. This is direct evidence for withholding the general landing/planner gate.
- Final viewer generation contains 546 review recordings/hypotheses across all ten maps, including
  direct complete-control candidate replays. All ten browser checks pass: 47,932 rendered triangles
  match saved model coordinates/hashes, native/predicted traces are exact saved arrays, selection/
  scrub/focus work, A1–A4 are preserved, and no JS errors. Visually inspected the Concentric screenshot;
  matching paths correctly overlap. Constrained the longer recording selector to the sidebar width
  and added hover titles; regenerated/rechecked after this presentation fix.
- Final owned-process check: only the existing dashboard (33732) remains. No trainer, evaluator,
  probe game or helper browser remains. Python compilation and git diff --check pass; native probe
  build succeeds; 15 geometry/native regressions pass. No policy training, learned fit or promotion.

## Final probe identity / continuation priorities

Current separate executable SHA-256:
`b603b18060fc745d0f559004c0a875f894a59f8002f180d78fd03883b8442a5e`.
Current recording include SHA-256:
`14fb6e105b0dcb1f118d2b75cf487c9151936694278a4b0ce768e22a7b019b2d`.
Earlier datasets intentionally retain their earlier native binary/schema hashes.

1. Review Cube Isle sensitivity and the KOTM/Sprawl pilot outliers in the viewer. Debug using detailed
   native contact/force phases; retain independent geometry/state starts for subsequent uncertainty
   measurement. Avoid reducing a rare collision branch to an averaged guaranteed landing.
2. Measure complete input/event timing in the intended 64 ms decision regime. At 16 ms collection,
   steering camera commands can apply two native calls later, with jump/use another call later;
   powerup yaw hold is its own channel. Do not assume this directly matches four-tick policy actions.
   Add item/respawn/Mega/Blast/dynamic actor events before full-round prediction. Native spin and
   prior contact/effect state are available only in the instrumented probe, not current policy obs.
3. Train or accelerate the general world model on these physical labels, with contact/outcome losses
   and map/trajectory splits, if it improves speed/fidelity. The engine-based prior already covers
   rolling, impacts and native powerup modifiers; fitting sparse residuals cannot by itself certify
   omitted contacts or live event handling. The untrained whole-map encoder needs supervision and
   detail-preservation evaluations before supplying an actor.
4. Establish horizon/clearance and safe-continuation criteria on new native trials, then run a bounded
   live MPC/planning comparison and only afterward integrate/distill into the navigation policy.
   Broad long-horizon reliability remains the gate. Current policy/PPO remains paused.
