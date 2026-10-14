# Native physics learning: preparation for review

2026-10-12. Engineering/data preparation only. No learned-model fitting, PPO,
policy integration, deployment or checkpoint promotion occurred. Frozen A2 and
the native test executable retain their protected hashes.

The aim is a general predictor of the marble's physical response to controls and
the map, including spin, friction, rolling, jumping, collisions, rebounds and
power effects. Gem choice and game points belong to the planner/policy layer.
This first benchmark tests the motion-prediction building block. It is deliberately
not yet a complete multi-step world model or a proposed replacement for A2.

## Concrete artifacts

- [Prepared data manifest](../dynamics_reports/native_motion_student_preparation_v2_20261012/manifest.json):
  53,026 playable16ms native transitions from238 matched trajectories;57 terminal/
  OOB transitions excluded and counted. All10 maps and47,932 collision triangles.
  Six training maps contain28,738 transitions; Concentric/Cube are development,
  Bowl/Vortex test. Map roles stay together; neighboring frames/pair groups are
  never randomly split across roles. All maps were inspected during analytical
  engineering, so these are held out from student fitting, not previously unseen.
- [Unfitted GPU forward evidence](../dynamics_reports/native_motion_student_preparation_v2_20261012/unfitted_forward_feasibility.json):
  exact preparation parity for every surface,240 raw input/label reconstructions,
  64-state batches on RTX3070. Median forward0.50–1.22ms across maps; maximum
  allocated tensor memory about40MB.83,081 random parameters remain unchanged.
  This is feasibility only, not accuracy, training-memory or whole-planner speed.
- [Interactive controlled-mechanics review](../collision_maps/viewer/matched_mechanics_all_maps_20261012/index.html):
  full geometry and native/predicted trajectories. These analytical forecasts use
  recorded external commands from one initial state, with no future pose supplied.
- Code: `nav/marble_dynamics/student_data.py`, `physics_student.py`,
  `student_dryrun.py`, `test_student_data.py`. Source snapshots and hashes are
  archived with preparation. The earlier78-feature draft remains in its separate
  v1 directory with archived source; v2 has84 state and38 surface features.

## Inputs and targets

Expanded continuation after the first preparation:280 additional valid native
trials/58,386 frames,99 actual activations and173 retained falls. Use/no-use starts
match position/velocity/spin exactly. [Expanded bundle](../dynamics_reports/native_motion_student_expanded_20261012/manifest.json)
has111,083 playable transitions from518 source trajectories, including90 training
activation rows. The earlier bundles/evidence remain unchanged. New native tests
broaden controls and supported/falling power starts; the average does not certify
coverage of every contact or material combination.

The analytical continuation audit also exposes substantially larger sensitivity
than the earlier matched pool: Sprawl reaches64.14u error after a flagged contact;
Bowl reaches15.38u. [Targeted step diagnosis](../dynamics_reports/rich_mechanics_error_diagnosis_20261012/summary.json)
separates one-step errors from accumulated divergence. The Sprawl fork is flagged
at182, immediately before the first state above0.001u error at183. Six selected
unflagged/native-BEFORE single-step cases are near-exact, but Treasure reaches
0.061u unflagged over6s and Cragmire0.002u. A good one-step model must therefore
earn its long-rollout claims separately; contact flags are not calibrated bounds.

All collision triangles are available, including walls, undersides, original
vertices, native planes, polygon boundary flags and friction/restitution/force.
The surface encoder attends over every triangle; no ray-range, nearest-K crop,
decimation or walkability filter discards geometry. A continuous distance bias
helps one-step queries focus on nearby contact; it is not a certified collision
bound. Float32 neural features approximate the reviewed geometry, whose original
archives stay untouched. Object/surface/material IDs remain traceable in the
prepared geometry; IDs are not learned input shortcuts.

Before-state inputs include position, velocity, spin, radius, gravity direction,
native contact/camera state, physical parameters, held inventory, active effects
and timing, plus actually applied Move and ordered native camera callbacks.
Signed effect deadlines distinguish exact expiry from already-past expiry, while
inactive uninitialized timer values are masked. No future observation, reward,
gem coordinate, map ID or score enters the physical input.

The target is the native16ms delta in position, velocity and spin. Labels are
formed in Float64 before conversion/normalization. No reward or power-use bonus
is used. This measures what a control does, not whether it collects a gem.

## Data limits that must shape the experiment

There are52,846 ordinary-control rows,130 jump-command rows and only50 held-power
activation rows across all maps; just30 activations are in the training split.
Long post-activation effects are present, but six training activations per power
are insufficient evidence of general use from varied starts. An average over all
frames would heavily favor ordinary rolling. Existing360-trial pilot and older
effect archives lack complete camera-command audit; do not silently add them or
pretend their observed future camera state is a proposed input.

Those counts describe the first bundle. Expanded data increases actual training
activations30->90, still only18 per power across six training maps and limited
initial situations. Count distinct trials, contexts and contacts as well as frames.

Terminal/OOB rows are excluded from this narrow continuous-motion task. They stay
in raw archives and must be evaluated separately before any safety use. No
claim of safe fall prediction follows from this prepared set. Matched setups use
explicit teleports/respawns/grants before recording; scored navigation uses none.

The student outputs motion only. Camera smoothing, effect/inventory evolution,
contact metadata, script callbacks, Blast actuation and resource timing still
need a complete continuation contract for autoregressive prediction. Mega growth/
shrink, moving actors, Fireball/cannons and later pickups are not solved by a
successful16ms benchmark. Fixed-body Mega reproduction does not verify activation.

## Proposed milestones, with fitting still paused

1. Broaden audited native transitions before fitting: varied speeds/spins,
   friction/materials, supported and airborne power activation, jump/brake/turn
   controls and collision setups. Use the same general mechanics pipeline and
   immutable map roles. Increase diverse trials rather than counting adjacent
   frames as independent experiences. Keep unsupported script/body events labeled.
2. First bounded fit, after review: supervised native motion loss only, balanced
   across maps/control strata and reported separately for jumps, each power, impacts and
   ordinary motion. Compare against constant-velocity and the existing analytical
   baseline. Do not select on the two test maps or promote based on pooled error.
3. Require complete auxiliary-state continuation before long rollout claims.
   Evaluate held-out full trajectories from one initial state, without replacing
   future pose/spin/contact/camera with measurements. Report0.5/1/3/6s errors,
   landing position/timing/spin, material changes, collision tails and failures.
4. If the student earns a useful ranking role, first use it to cheaply shortlist
   general control sequences while the analytical/native checks retain final
   validation. Repeat paired gem-collection attempts, real-time compute checks
   and endpoint/landing negatives. Independent native play remains the authority.
5. Only later consider policy observations or PPO. A better physical forecast
   should help the planner discover useful maneuvers without named-trick rules,
   but the motion benchmark alone does not demonstrate that outcome.

The next autonomous engineering work can improve audited data coverage and
continuation metadata. Actual fitting/policy changes remain the reserved review
decision; nothing here silently starts training.

## Prepared fitting protocol and score evaluation

[Refined bounded motion-fit protocol](../dynamics_reports/native_motion_fit_review_r2_20261012/protocol.json)
is now concrete:1000 updates, batch64, AdamW, isolated random motion student,
300s training-loop bound (bounded validation may extend total wall time). Six
training maps are sampled uniformly; supervised control strata are60% ordinary,
20% jump and20% held-power activation. These are regression sampling proportions,
not probabilities of forced gameplay actions. Dev map/stratum-balanced MSE
selects the checkpoint; per-power errors are reported separately. Test maps are
evaluated once at the selected checkpoint. No fitting has been started.

Rich-gallery geometry/browser/readout checks pass (837 time samples,115 local
links,0 JavaScript errors). Initial-only perturbation diagnosis confirms that
some long collision futures branch drastically; this is not calibrated
uncertainty. Single-step regression remains a deliberately narrow first task.

[Path to actual points](ROADMAP_FULL_MAP_LEARNING_2026-10-12.md) separates this
motion experiment from continuous controller integration and policy training.
A2 complete-round references have been evaluated independently; they cannot be
replaced by first-fall planner scores or student accuracy. Three-seed preparation
exists, but the completed ten-map sweep is only one round per map. Historical-seed
Fractured83/88 versus prior91 highlights the need for comparable replicated starts. A supplied game
seed is not proof of identical observed worlds/countdown history.
