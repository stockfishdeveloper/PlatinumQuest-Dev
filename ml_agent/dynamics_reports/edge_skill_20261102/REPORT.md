# Edge-hit skill review — 2026-11-02

**Status: stopped before planner integration, as requested.** The standalone laboratory, teacher, model, feedback executor, and transfer evaluation are implemented. The transfer and reliability gates are **not passed**. No production planner, AI script, or C++ source was changed.

Open [the 1x native trace viewer](review.html). It includes all 108 attempts, with successful rolling/rising/descending examples first. It shows recorded game motion and collision geometry, not rendered game video. Raw controls, per-tick game states, contacts, and forecasts are retained in `native_final_v2/`.

## What the results establish

- Accepted native executions: **19/19 (100.0%)**. Admission covered only **19/108 (17.6%)** of the fixed suite.
- Total native successes, including exploratory execution of declined starts: **42/108 (38.9%)**. Declined recoveries are not evidence of safe admission.
- These are short local landing successes. **33 successful trials, including 17 accepted ones, later departed below the landing height** in the extended rollout. A safe handoff/continuation has not been demonstrated.
- Real-map native screen: **0/12 accepted; 1/12 exploratory success** across six edge locations on Duplex, Fractured Islands, and Sprawl. This is the clearest missing gate.
- The final proposer checked 48 initial candidates against the teacher’s cap of 384. Median root cost was **33.1 ms**, versus **258.5 ms** for the teacher; proposer p95 was **78.6 ms**, above the 64 ms action interval.
- These are landing-skill trials, not scoring rounds. There is no demonstrated points gain and no production promotion.

## Native results

| Approach | Fixed starts | Accepted | Successful accepted | All exploratory successes |
|---|---:|---:|---:|---:|
| roll | 32 | 15 | 15 | 20 |
| rise | 38 | 3 | 3 | 15 |
| descend | 38 | 1 | 1 | 7 |

The descriptive Wilson 95% interval for 19/19 accepted executions is 83.2–100.0%. Trials share geometry, so even that binomial interval does not establish broad independent reliability. **No 99% reliability claim is justified.**

The game and replay forecast agreed on all 108 final success classifications. Maximum position difference across the screen was 0.001643 game units. The 24 earlier calibration trials stayed below 0.00005 units. No Blast or power activation occurred.

There were 20 game-reported falls during the full 4.096-second recording horizon, including 2 after trials that met the local landing criterion. Separately, 33 successful local landings later dropped below the landing height, including departures that had not yet triggered the game’s fall event. Success requires a supported target entry followed by at least 0.384 seconds above the landing-height threshold; it does not certify the rest of a route.

## Frozen simulation comparison

| Suite | Starts | Existing direct branch | Teacher | Neural heads only | Neural + retrieval + refinement | With feedback |
|---|---:|---:|---:|---:|---:|---:|
| dev | 48 | 3 | 1 | 0 | 2 | 8 |
| test | 192 | 1 | 48 | 14 | 42 | 87 |
| real_test | 12 | 0 | 5 | 0 | 1 | 4 |

On the 192 untouched generated cases, the checked proposer covered 31/48 (64.6%) of teacher-solved cases. The teacher is bounded; its failures are unresolved starts, not proof of impossibility.

At the same 48-candidate budget, retrieval plus refinement without the network solved **40/192**, versus **42/192** with the network. Before the correction round, the combined proposer solved **45/192**. The neural contribution is small, and the correction round did not improve aggregate held-out success. The strongest observed improvement came from feedback execution, whose repeated searches spend more forecasts over the entire trajectory; the initial-search speedup is not a total-compute speedup.

The baseline reuses current lip eligibility, jump timings, and the direct `SequenceProposal` controller. It excludes route/lip-through variants and the full planner candidate bank. These numbers must not be advertised as beating the complete planner.

Synthetic and native starts are separate populations. Native trials solve from the actual post-reset observation, roughly 32 ms after the requested teleport. In the real-map suite, the teacher solved 5/12 nominal synthetic starts but 0/12 actual native roots. That sensitivity itself is a failure of useful coverage, not a simulator success to substitute for native performance.

## Implementation and measured scope

- Two generated maps contain 32 isolated geometry cells. Dimensions are baked into compatible DIFs; runtime scaling is uniform. Earlier rotated/nonuniformly scaled fixtures failed collision conformance and are retained only as retired diagnostics.
- Exact native reset snapshots preserve pending inputs, angular velocity, support state, and 16 ms physics / 64 ms reply timing. Independent contact audits distinguish a lower-half lip impact from a wall face or ordinary jump.
- The teacher searches jump timing, pre-contact throttle/heading, and post-contact throttle/heading. It retains multiple successful candidates and has a fixed 384-forecast cap. This five-parameter, two-segment representation is deliberately small and currently limits coverage.
- Collection: 96 smoke cases; 1,440 diverse training starts (429 solved); 189 learner-visited correction states (73 solved). The final 14,612-parameter network trained on 534 successful training labels. Geometry splits exclude development/test cells from training.
- Four neural heads preserve alternative solutions. Eight nearest teacher examples and bounded local refinement supplement them. The feedback executor retains its previous plan, remembers issued jumps, and checks observed contact progression before switching to landing control.
- Generated starts vary approach angle (approximately 53–90 degrees to the edge tangent), speed (8–17 units/s), rolling spin by ±25%, extra spin, lateral alignment, and landing target. No universal 45-degree edge-hit boundary was established.
- Development stress test: 44 perturbations of canonical roots, up to 0.02 longitudinal / 0.03 lateral game-unit displacement, 0.2 units/s velocity changes, and 1 rad/s spin changes. Fixed initial plans succeeded 21 times; observed-state feedback succeeded 38 times. This is a development stress test, not the held-out reliability denominator.
- A trial passes only with the intended lip contact and approach mode, no prior wall impact, entry into a supported horizontal target radius of 0.85 units, at least 0.384 seconds above the landing-height threshold, forward exit speed >2 units/s and lateral speed <6 units/s. Desired speed 10 units/s is a soft preference, not a hard guarantee of momentum preservation.
- Landing goals are virtual and vary independently of decorative practice-map gems. Actual `gem_delta` is preserved separately in `native_contact_details.json`; landing success must not be equated with points.
- The native runner uses lockstep to measure execution, not a measured real-time planner deadline. The model and direct-branch comparison use ordinary controls only.

## What remains unproven before step 5

The current representation has not produced broad, robust real-map admission. The teacher and proposer need to cover actual observed approaches and post-contact steering more effectively. The feedback results suggest that a contact-conditioned landing controller or richer short-plan representation is a better next experiment than simply collecting many more labels for the current two segments. That is a proposed next experiment, not a hidden change made after seeing this test.

The real-map selector deliberately required a broad, approximately flat run-up and landing. Only three of ten maps contributed cases. Obstructed landings, varied surface materials/slopes, active powers, wall tricks, angles near 45 degrees, and live gameplay approaches remain outside demonstrated coverage. Exit preference is fixed at 10 units/s forward; arbitrary desired exit directions/speeds were not trained. Native test roots start on the ground, with jumps producing rising/descending contacts; learner-visited airborne states appear in correction data but are not a separate certified start regime. Forward momentum retention is recorded contact by contact, but high retention has not been certified.

Do not integrate this version merely because all 19 accepted starts succeeded. Review coverage, the real-map failures, the weak neural ablation, and the latency tail first. The production planner, its ranking/gates, and full-round scoring tests remain untouched.

## Artifacts and reproduction

- `review.html`: self-contained 1x / 0.25x native trajectory viewer, all 108 trials.
- `report.json`: machine-readable aggregate results, confidence calculation, timing, and source/checkpoint hashes.
- `frozen_suite.json`, `native_suite.jsonl`: test selection frozen before final evaluation.
- `teacher_train.jsonl`, `corrections.jsonl`, `proposer_final.pt`: successful and failed labels and trained model.
- `evaluation_final_v2.jsonl`, `native_final_v2/`: complete simulation and native results.
- `calibration_v2/`, `stress.json`, `measurement.json`: conformance, perturbation, and ablation evidence.
- Earlier `*_v0`, `*_v1`, and incomplete `evaluation_final` / `native_final` runs are diagnostics, excluded from the final statistics. The first final-run attempt stopped on a baseline-adapter field error; only the adapter was fixed, with no model or acceptance-threshold retuning.

See [`ml_agent/tools/edge_skill/README.md`](../../tools/edge_skill/README.md) for commands and limits. Four focused measurement/control tests pass.
