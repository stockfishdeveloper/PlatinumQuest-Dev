# First bounded complete-map motion fit

The authorized first supervised diagnostic completed all 1,000 updates. The student failed the useful one-step prediction gate and stays outside gameplay. A2/native executable hashes are unchanged; no PPO or promotion occurred.

Each evaluation uses the measured native BEFORE state. These are 16ms one-step errors, not whole-maneuver forecasts. Test maps were evaluated once after development selection.

| Map | Student position mean (u) | Constant velocity (u) | Student velocity mean (u/s) | Constant velocity (u/s) | Student spin mean (rad/s) | Constant velocity (rad/s) |
|---|---:|---:|---:|---:|---:|---:|
| Bowl_Hunt | 0.047837 | 0.004440 | 1.892321 | 0.430135 | 1.260375 | 0.853075 |
| VortexEffect_Hunt | 0.041133 | 0.004518 | 2.158858 | 0.417589 | 1.079828 | 0.747146 |

Selected update 1,000 by the predeclared map/stratum-balanced development MSE 0.01566797. Fitting/validation after setup took 4.81 seconds; the 300 second limit was a ceiling, not a requirement to train that long. Isolated checkpoint SHA256: `754763d8307f0ad0945f6c5e5ea29dbd6914d5f3797a2a9f20ce01358b5ce187`.

Power activation is still poorly learned: test-map Super Speed mean velocity error 18.82u/s (Bowl) and 18.98u/s (Vortex), with only 3 independent activation rows per type/map. Overall sample count 111,083 does not make those rare interventions abundant.

The raw-delta student must learn ordinary kinematics as well as impacts/impulses. A sensible next student version would predict corrections to known kinematic/analytical motion and improve contact geometry features. That is a hypothesis, not a demonstrated fix. Use development maps for this redesign; these inspected test maps are not a fresh untouched holdout for later tuning.

Continuous analytical planning remains the primary route to actual scores. Do not extend neural fitting or feed this predictor into action search merely because its training loss decreased. Whole 0.5/1/3/6 second mechanics, contact forks, scripted state changes, and native score gains remain separate gates.

[Machine-readable result](../dynamics_reports/native_motion_fit_20261012/result.json), [training/development history](../dynamics_reports/native_motion_fit_20261012/history.json). The original source versions are preserved in the run's verified `captured_sources.zip`; active fit tooling now creates a direct `source_snapshot.zip`.


### Residual development fit completed; no gameplay integration

Ran5000bounded updates in36.19s; map/stratum-balanced development selection
chose update1200 (MSE0.01031593), not the last update. Isolated checkpoint SHA256
8f6883077229bf68b70bba9ca14bbbcf3533267afc4e7ccdf2e235c4fb325a50.
The two development maps are Concentric AND Cube Isle; do not mistake this for
a one-map selection. Bowl/Vortex test maps remain closed for this variant.

| Development map | Student position u | Student velocity u/s | Student spin rad/s | Velocity-baseline position | Baseline velocity | Baseline spin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Concentric_Hunt | 0.011092 | 0.907371 | 1.517822 | 0.005476 | 0.548818 | 1.187087 |
| CubeIsle_Hunt | 0.009565 | 0.743168 | 1.207837 | 0.004338 | 0.412322 | 0.834287 |

Some activation transitions improve substantially, especially Super Jump;
ordinary-motion averages still exceed the constant-velocity baseline. Thus this
is a useful diagnostic of the learning tradeoff, NOT a useful learned rollout
model yet. No long-trajectory test, student gameplay integration or policy
training is claimed. All protected artifact bytes verified unchanged by fit.
Source is captured directly in source_snapshot.zip, no loose .source.py copies.


### Physics diagnosis: commands learned without enough contact conditioning

contact_development_diagnostic.json uses native within-step telemetry only AFTER
prediction to bucket errors; no future contact data enters the model or selects
weights. Inspected test maps remain closed. The residual student hallucinates
jump response in air: Concentric/Cube ordinary jump-command frames with no
native contact have velocity error5.027/5.078u/s versus constant velocity0.338/
0.339. Ground-contact jump frames improve2.161/2.929 versus7.068/7.068. Thus it
has learned some impulse mapping but insufficiently distinguishes when jumping
actually applies. Ordinary sweep-collision frames barely improve (actually
slightly worse in aggregate), while simple airborne/contact continuation also
regresses. This is not just sparse gem reward; it is a physical-model data/
conditioning problem, and no gameplay reward bonus solves this benchmark.

Next isolated model hypothesis after the frozen native batch: encode surface
vertices/plane distances relative to the current marble in physical units, retain
all faces/materials/boundary flags, and inspect separate airborne/contact/jump/
activation errors. Multi-contact information must survive surface pooling. Test
translation consistency and exact complete-surface retention before any fit.
Reject scripted future-contact inputs or one hand-written recipe per trick.
Keep constant-velocity and analytical baselines; use development-only bounded
experiments and ordinary-motion checks so activation improvement cannot hide
hallucinated airborne jumps. This is a candidate design, not a proven remedy.


### Contact-relative residual variant prepared and verified

Added RelativeGeometryResidualStudent inside the maintained fit tool, not a new
script. Every face is re-encoded relative to marble position in radius units;
operative/source vertices, center, extent, area and signed native-plane distance
use continuous atan coordinates. Normals, material physics and polygon boundary
flags remain present; every triangle participates, no hard top-k/range cut.
Absolute origin is removed only from the state encoder, while it still computes
relative geometry/drift. No measured future contact, map ID or reward enters.
This is the one-step physical kernel, not the navigator's far-destination input.

Eight focused contracts pass: all47932prepared faces/metadata retained across
all10maps, input purity, nonzero-output origin consistency to1e-6, complete-face
forward shape, exact zero-head velocity drift and original-cache refusal of the
new forward. CUDA feasibility on the largest training map Duplex10383faces,
batch32, finite forward/backward,0optimizer updates, peak0.899GB on8GB RTX3070;
first cold pass0.171s. This cold timing is not a realtime claim.

Predeclared next isolated fit:5000updates, batch32,300s loop ceiling,
--relative-geometry --dev-only, fresh native_motion_fit_relative_geometry_20261012.
Same native arrays and train6/dev2/test2 roles, same60/20/20 sampler and loss,
initial drift baseline eligible for selection. Batch differs from earlier64,
so this is a development experiment, not a clean single-factor causal comparison.
Test maps remain closed. Check ordinary/contact/no-contact jumps/activations
separately after fitting; no integration or policy checkpoint promotion.

Native delayed OOB click test PASSED nine checks on Cube Isle:320ms delay exceeds
the old128ms timer, no early key or spawn, no pending callback, unchanged key
permission, exactly one clicked respawn, then legal explicit rescue accepted and
one additional spawn. New source pool retry uses the fully disabled connected-
controller auto-key path. Older input-invalid54score stays diagnostic only.


### Third isolated physics fit completed; selected result still not deployable

Contact-relative residual fit completed5000updates in136.67s; predeclared
map/stratum-balanced selection chose update1600, development MSE0.00802690
(~22% below prior residual0.01031593). Checkpoint SHA256
905cd26a91945808f9fa27db67c41f94d77f0023f8112fa2fa4120d4adc5e23a.
Selected mean errors [position u,velocity u/s,spin rad/s]: Concentric
[0.012913,0.814161,1.420524], Cube[0.012082,0.899083,1.055315], still above
velocity baseline on both. The last5000update dev summary has better ordinary
averages but was NOT the selected checkpoint; do not report that as this model.
Test maps remain closed, no new policy training or learned gameplay integration.

This exposes a selection/sampling tradeoff: equal stratum selection and heavy
oversampling can prefer rare activation improvement while ordinary motion
regresses. Future bounded fitting should predeclare ordinary/contact/jump/power
criteria and sampling changes, not cherry-pick a different checkpoint after
looking at results. Separate origin-consistent geometry and multi-contact context
hypotheses remain useful development work. During the first136s the isolated
GPU fit ran alongside Cube analytical evaluation; machine load can affect the
1s search ordering/budget and is disclosed rather than treated as a clean A/B.

Fresh full-pool r3 source is frozen, Cube first, then Bowl/Duplex/Fractured/Crag/
Vortex/Treasure/Concentric/Sprawl/KOTM, seed681012, previous limits/rules. This
includes the connected-bridge no-auto-key fix and strict live input audit.
Dashboard and A2/native executable bytes remain protected; Python count476.


### Selected contact-relative model: airborne jump failure and geometry audit

Development-only post-hoc contact diagnostics are complete. On airborne jump
commands the selected relative model has mean velocity errors 5.01 / 6.26 u/s
(Concentric / Cube) against drift baselines 0.338 / 0.339. Grounded jump errors
improve to 3.23 / 2.92 against 7.07, but this is not enough for a useful predictor.
Ordinary collision velocity errors are 5.91 / 5.84 versus 5.82 / 5.88; position
and spin remain worse. No test maps reopened, model integration or promotion.

An additional BEFORE-only geometry audit uses all exact operative triangles
and native starting poses. The current attention prior uses distance to each
triangle's bounding box: a conservative lower bound, not surface contact. Of
21 Concentric jump inputs, 9 appear within 1.05 marble radii by bounding boxes
while every triangle is farther away. Training Sprawl has 5 / 21 such cases;
Treasure has 1 / 11. No AABB mismatch on Cube's 21 jump inputs, so this finding
alone does not explain its airborne error. ground_ms and best_contact_normal
also retain recent/stale contact information; neither is a current-contact flag.
The raw rows and source/frame links are in before_geometry_jump_diagnostic.json;
contact_development_diagnostic.json preserves all native post-hoc buckets.
This motivates exact before-state proximity and multiple contact contexts,
plus predeclared ordinary-motion selection checks. No runtime source changed
while the full-pool batch is frozen. No one-off Python file was created.



### Next isolated physics experiment: predeclared direction, not yet run

The next development variant should add continuous exact closest-point/plane
features computed from BEFORE pose and the complete reviewed mesh, and expose
several learned contact contexts rather than one averaged context. Keep native
source/operative geometry, normal/material/boundary information and every face;
no named maneuver rules, future contact labels, score, map ID or reward inputs.
Exact proximity is an input representation, not a rule that a jump is allowed.
The geometry query must be validated on faces, edges, corners, degenerate fans
and origin translations before fitting; compare with the existing F64 reader.

Predeclare a bounded development-only comparison using the same six training /
two development maps and a fresh output/source archive. Reduce rare-command
oversampling (proposed 90% ordinary / 5% jump / 5% use) and select with an ordinary-
motion non-regression gate on each development map, then compare physically
scaled errors on the rare commands. Retain update-zero drift as the explicit
reference; if no candidate passes, report failure rather than silently promote
it. Final exact selection thresholds, model shape and limits must be written
in the executable protocol before starting; this note is not that sealed
protocol. No reopening the already inspected test maps. No longer fit until
input and selection contracts pass, and no GPU fit during a wall-budgeted native
comparison. Native full-pool r3 sources remain frozen until its sweep finishes.


### Complete geometry feature precision audited

Read-only reconstruction audit covered all 47,932 faces across the ten meshes,
including retained degenerate fans. Maximum Float32 neural coordinate error is
8.68e-6 u (Bowl); maximum native-plane offset reconstruction error is 6.43e-6 u.
Both are below 0.00005 of the normal marble radius. Geometry was not simplified
or altered, and this opens no learned test-map prediction. This small numerical
approximation is far below the selected motion student's roughly 0.01 u mean
position errors, so coordinate rounding alone does not explain that failure.
Degenerate fans must remain supported by any new proximity feature implementation
because native edges/corners can still matter. Full values are archived in
native_motion_fit_relative_geometry_20261012/geometry_feature_precision_audit.json.
