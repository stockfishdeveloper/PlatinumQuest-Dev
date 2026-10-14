# Geometry routing and physics search component evaluation

Declared before implementation/run. Continue under the operator's instruction
to iterate until stopped. This is engineering evaluation; no PPO, checkpoint
promotion or long policy training is authorized by this protocol.

## Question and controllers

The ongoing r4 controller combines collision-filtered routing, analytical
forecast/search, global gem alternatives and resource continuations. Its score
cannot identify which component helps. Compare three explicitly named branches:

- `frozen_navigator`: unchanged A2 weights, legacy terrain routing.
- `geometry_navigator`: identical frozen A2 weights and observation builder,
  collision-filtered routing, no physical search or predicted action selection.
- `physics_plan`: current combined controller, including the same geometry
  routing and A2 fallback; navigator proposals checked first.

The geometry branch retains the same nearest-observed-gem tour selection as the
hybrid's initial selection. It cannot perform the hybrid's searched global-goal
alternatives, resource acquisitions or predicted landing selection. Therefore
this isolates routing's contribution, but the remaining hybrid difference
includes goal selection and resource handling alongside physical forecasting.
Do not attribute that residual entirely to physics prediction.

## Declared pool and settings

All ten maps, one original-duration natural round per branch, seed **681013**:
CubeIsle, Bowl, Duplex, FracturedIslands, Cragmire, VortexEffect, TreasureBox,
Concentric, Sprawl, KingOfTheMarble (all `_Hunt` identifiers).

Branch order is a cyclic rotation of physics_plan, geometry_navigator,
frozen_navigator, indexed by (seed + zero-based map index) modulo three. Every
branch occurs once per map; no result-dependent retry/reordering/substitution.
Separate native launches; supplied seeds do not certify exactly matched starting
worlds. Archive starting contexts and disclose any difference. No scored-play
restore, teleport, powerup grant or artificial gem placement.

Use experimental 16ms physics, 64ms decisions; competitive off, explicit legal
OOB click, eight-second no-progress rescue charged by elapsed game time only.
Hybrid: horizon24, reuse/revalidation, strict ABI3 sweep DLL and bounded exact
query/clearance caches, one-second search deadline, fallback cooldown64,
preserve/acquire resources, recharge/landing continuations, twenty proposals,
global observed gem alternatives. Round wall bound900s for all branches.
Search deadline does not cap all work in a decision; measure full latency.

All new branches log the actual policy destination, route waypoint, next goal,
destination kind/item identity when observed, and actor action. These are
before-action fields, never reconstructed from the outcome. Keep the source
version frozen, frozen checkpoint hashes, explicit input/OOB audit and native
score contracts. No concurrent GPU fits or unrelated native evaluations.

## Reporting and checks

Require all thirty score-valid, declared-input-valid completed rounds before a
complete-pool comparison. A failure keeps partial evidence and invalid status;
use a distinct run/version if a fix changes source. Do not invent missing scores.

Report each map, equal-map means, median per-map delta, wins/ties/losses, falls,
accepted/attempted rescues, native-confirmed activations versus commands, checked
and fallback fractions, full decision p95/p99/max and round wall cost. KOTM is
one map of ten: modest loss there is no rejection gate. A nine-other-map
diagnostic may accompany the full table, never replace it. One seed is a
development comparison, not statistical proof of generalization or promotion.

Before running: meaningful tests that the geometry branch actually receives
geometry routing, never invokes search, leaves frozen weights unchanged, logs
actual before-action targets, and preserves round/start/reset/input contracts.
Replay export must show actual controls/unknown intervals and leave respawn or
unobserved control gaps disconnected while preserving every measured point.

Interpret results to choose the cheapest useful teacher and the next data
collection/learning step. Record unfavorable maps and execution cost. Stop only
for an operator stop, a material blocker or a critical artifact/approval decision.
