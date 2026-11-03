# Edge-hit skill: proposed implementation plan

Date: 2026-11-02. Reference checkout: eaad0ebe7.
Status (updated 2026-11-03): campaign 1 (11-02) failed the transfer gate; campaign 2 (overnight 11-02/03) rebuilt the controller and passed the sealed evaluation: 416/416 accepted native successes on 1,044 sealed starts, real-map admission 72/180. See [the campaign 2 report](../dynamics_reports/edge_skill_20261103/REPORT.md). Stopped before milestone 5 (planner integration) as authorized. Campaign 1 results remain in [edge_skill_20261102](../dynamics_reports/edge_skill_20261102/REPORT.md).

The user's subsequent authorization was to implement autonomously through the skill evaluation and pause before planner integration. The earlier intermediate joint-review suggestions below did not require renewed approval during this implementation.

Confirmed scope: lip contacts first, including rolling, rising, and descending approaches. Wall-face rebounds and wall-assisted climbs are a separate follow-up.

## Objective and skill contract

Given the current physical/control state, nearby edge geometry, and a desired landing region and exit velocity, produce an ordinary-control trajectory that uses a lip contact to reach that goal quickly and reliably. Measure improvement against the existing edge-hit proposals and the full current planner. Full-round points remain the final promotion metric.

Initial conditions: normal marble, ordinary gravity, no active powerup effects. Jumping is available; Blast and powerup activation are excluded from the action space. Verify actual controls and events, since removing powerup items alone does not prevent regular Blast. Setup/reset operations occur only before a trial.

Use edge-relative coordinates, full-resolution collision geometry, linear/angular velocity, support/flight state, radius, surface properties, desired landing region/exit velocity, and the pending input state. Map identity and absolute location are not policy features. A landing is a region with an allowed velocity range, not an exact floating-point point.

The skill proposes a small number of short control plans; compiled forecasts check them. It observes the actual state and updates its plans at the existing 64 ms action cadence. Physics remains at the existing 16 ms step, including the real input delay. Keep the previous remaining plan as an option during replanning. Skill phase and already-triggered actions must survive replanning.

The first version does not need a neural network evaluated at every simulated reply. A learned proposer can emit a few action sequences or their parameters, which the existing native machinery evaluates. Benchmark this design before deciding on an inference port or a different controller architecture.

## Reuse and new work

Reuse:

- `nav/marble_dynamics/edge_hits.py`: mesh lip extraction and successful existing proposals as seeds and a comparison. Its current height/target filters and approximate timings are not the definition of what the new skill can attempt.
- `nav/native/forecast.py`: `predict_many`, complete control-state continuation where supported, and native trajectory evaluation. The forecast resume helper is not a full item-world snapshot.
- Existing ordinary physics, gem pickup checks, collision queries, contact telemetry, and recorded game states.
- Drill map/session/reset examples in `nav/learned_nav`. These are reference implementations: older crossing drills approximate pending controls and are not an exact-state evaluator to reuse unchanged.
- Native and full-round benchmark infrastructure for later integration checks.

New work: parameterized drill geometry, a short-trial runner with correct initial state, lip-contact outcome classification, a targeted trajectory solver, training/evaluation datasets, a compact skill proposer, and planner integration.

The existing map generator edits a fixed KOTM mission; it is not already a general procedural geometry generator. Build drill missions from compatible game geometry and export the actual collision mesh. Do not train against a separately invented mesh that only resembles the playable map.

Suggested new home: `ml_agent/tools/edge_skill/` for drill construction, measurement, solving, and evaluation; choose the production skill module only at integration. Preserve data, manifests, and summaries in a persistent project artifact directory rather than only temporary scratch folders.

## Milestone 1: a trustworthy edge-contact laboratory

1. Freeze the source/configuration reference and define the trial record and success criteria.
2. Build two parameterized map templates: a run-up and lip with a landing platform; and an elevated approach permitting a jump or descent onto a lip followed by a landing region. Templates produce varied heights, offsets, edge lengths/orientations, and platform dimensions.
3. Implement reset, warm-in, and recording. Record actual native state after setup, not just requested teleport coordinates. Preserve or reconstruct pending controls and contact/support state. For transfer comparisons where exact controller history is unavailable, use the same explicit reset/warm-in protocol for both arms.
4. Add classification of intended lip contact, wall-face contact, ordinary floor landing, miss, gem pickup, final landing, and later fall. Use existing native/contact diagnostics; add narrowly scoped telemetry only if it cannot distinguish the mechanisms.
5. Compare a small canonical set of native trials with simulator predictions. Include successes, near misses, and wall contacts across the three approach modes. Inspect actual contact timing and post-contact motion, not just a final position.

Initial scope: approximately 24 canonical situations. This is a measurement check, not a reliability claim. Fix or exclude any mismatched physical regime before producing labels in it.

Deliverable: playable drill maps, reproducible trial records, side-by-side predicted/native traces, and short 1x examples of all three approaches. First joint review: confirm that the measured contact is the mechanic the user intends.

## Milestone 2: determine what can be executed and solve it offline

1. Sweep approach angle, normal/tangential speed, vertical speed, spin, contact height, edge position, and target landing region. Define angle explicitly: 90 degrees to the edge tangent means a perpendicular approach. Do not assume a universal 45-degree boundary.
2. Use staged search rather than a Cartesian product of all parameters: propose contact conditions that could produce the desired departure; find an approach from the actual start; verify the whole approach/contact/landing in the simulator.
3. Seed with measured successes and existing proposals. Refine contact position, steering/throttle segments, and discrete jump timing in small native batches. Include the three approach modes explicitly. Do not average incompatible solutions into one trajectory.
4. Test promising plans with small variations of starting position, velocity, spin, and action phase. Seek a region of successful executions, not a single perfect starting point.
5. Retain multiple solutions with useful time/exit-state tradeoffs. Judge landing and continued control; maximum height alone is not success.

Start with roughly 30 development situations, ten per approach mode. Measure throughput on the first ten, then fix a per-case forecast limit and total run cap before expanding. One bounded solver revision is reasonable if a specific diagnosis supports it; do not launch an open-ended search.

Report solved/attempted counts for every mode and difficulty bucket. A solver failure is unresolved, not proof of physical impossibility. Keep misses and their outcomes, but do not label all of them as infeasible starts.

Deliverable: a measured operating envelope and successful sequences that improve coverage, time, or exit state over current proposals, with representative native verification. If this fails, address the solver/representation before starting a large training run.

## Milestone 3: build a transferable dataset and train a fast proposer

1. Generate diverse geometry/start/goal combinations from the two templates, using the solver as a teacher. Begin with easy successful contacts, then widen angles, speeds, spin, offsets, and landing requirements. Include all approach modes rather than allowing the easy mode to dominate.
2. Split by geometry configurations and source rounds/edge locations, not neighboring frames or slightly perturbed copies. Hold out geometry combinations and real-map edge locations before tuning.
3. Train a compact goal-conditioned proposer to output a few short plans in the same representation used by the solver. Train separately conditioned approach modes if needed so rising/descending solutions are not averaged together. Compare against the teacher and the existing generator.
4. Execute the learner with feedback, collect the states it actually reaches, solve those states with the teacher, and perform a bounded correction round. Open-loop imitation alone does not test recovery from its own errors.
5. Keep success, coverage, time, landing error, exit velocity, and compute cost as separate reported metrics. A confidence-based decline must not inflate reliability by silently excluding most cases.

Size the dataset from measured diversity and throughput. Begin with a small end-to-end training smoke test; authorize a larger collection only after it can learn and execute a varied development subset. No fixed overnight labeling campaign is assumed in advance.

Deliverable: a small model whose checked proposals approach the teacher's useful coverage and quality at much lower search cost, plus a closed-loop held-out report.

## Milestone 4: demonstrate transfer and reliability

1. Evaluate unseen generated geometries and untouched real-map situations. Include new edge locations, both approach directions, different spin states, obstacles near the landing, and supported surface properties. Use a fixed suite selected independently of which cases the model accepts.
2. Compare existing proposals, the offline teacher, and the learned proposer plus its bounded local refinement. Keep starts, targets, input timing, and resource restrictions identical.
3. Report both execution reliability among accepted cases and coverage of the teacher-solved cases, as well as total attempted-case outcomes. Retain results for unresolved cases.
4. Validate in the native game. An initial small stratified screen exposes mismatches; it cannot certify 99% reliability. Choose the larger confirmation count and confidence rule before opening that test, grouping related trials by geometry/source.

Proposed objective for discussion: approximately 99% success on accepted cases within the declared operating envelope, while preserving substantial useful coverage of teacher-solved cases. Report each approach mode separately. A claim about all angles, all maps, or all physically possible situations is not justified by this test.

Deliverable: a transfer report, failure breakdown, accepted operating range, and representative successes and failures at 1x. If synthetic performance does not transfer, revise the representation or data distribution before expanding to more tricks.

## Milestone 5: integrate and measure additional points

1. Detect relevant local edges and useful landing goals. Invoke the skill only where appropriate, within a shared planner budget, initially returning roughly two to four proposals. Preserve the ordinary alternatives and current fallback.
2. Expose predicted elapsed time, landing position/velocity, resource usage, and measured reliability to the planner. The skill's own state/phase persists when its plan is selected.
3. Audit proposal generation, physics admission, ranking, selection, and execution separately. Count cases where a better trajectory is absent, rejected, loses on ranking, or fails when executed. Do not turn off broad safeguards or add map-specific exceptions to force usage.
4. Run paired local continuations and full-round simulations with Blast and normal gameplay resources restored for the overall planner. The edge skill itself remains unpowered in this version.
5. If simulation shows a meaningful aggregate improvement, freeze the candidate and criteria, then run fresh native rounds across all ten maps at actual 1x queue/deadline conditions. Report per-map points, falls, resource use, skill usage, and latency.

The old arbitrary-snapshot branch harness has known identity/timer/prefix issues. The unpowered short-trial lab does not need a general Hunt-world rewrite first. Repair any affected branch-comparison path before using it for integration conclusions; complete rounds from valid GO states and actual native runs remain separate checks.

No automatic 10% full-round requirement is imposed on one skill. Measure the attainable opportunity and campaign resolution before declaring the points floor; a useful reusable skill can contribute a smaller gain. A drill pass alone is not a promotion criterion.

Deliverable: either a confirmed scoring improvement ready for promotion, or an explicit diagnosis of why the learned capability did not translate into points.

## Milestone 6: extend only after the core is established

Next candidates are lip contacts with active Blast/powerup combinations and the separately scoped wall-face skills. Add physical state/resource conditioning and new validation regimes while keeping the unpowered skill's evaluation intact. The same laboratory/teacher/learner/integration workflow should be reusable.

## Checkpoints and scheduling

Joint review points: (1) contact mechanics and canonical replays, (2) solver coverage and improvement, (3) learned transfer/reliability, (4) actual full-round points. Between them, implementation should proceed without repeated routine confirmation.

The first concrete work package is milestone 1 and the ten-case solver timing probe. It produces measured cost and a reviewable demonstration before a large training commitment. Re-estimate subsequent runtime from those measurements; previous project timing estimates confused virtual and real time, so unmeasured calendar promises are inappropriate here.
