# Improvement-loop results review, 2026-11-01

Reviewed checkout: `fc581efb2` (Optimize planner). This is an assessment and a proposed next experiment, not authorization to launch training. No gameplay changes or new simulation/native rounds were made during this review.

## Assessment

The measurement infrastructure is a substantial advance. There is still no demonstrated points improvement from the new learning work. The inexpensive ranker branch failed, and the existing planner switches tested are unpromising. The broader multi-event trajectory search in the agreed plan has not been implemented and evaluated. That is the next bounded investment I recommend, after correcting the branch comparison contract.

I supported trying ranking early. These results reduce my confidence in getting a large improvement by learning another score over the existing candidate summaries. They do not establish that richer value learning is impossible or that the current score is the limit of the physics planner.

## What the experiments establish

Sources: HANDOFF_NAV_TRAINING_LOG.md, the implementation in tools/sim_bench, and raw JSONL artifacts under:

`C:\Users\doug\AppData\Local\Temp\claude\c--Users-doug-OneDrive-Documents-GitHub-PlatinumQuest-Dev\4fb9efcd-2a02-4d95-9256-6dcdf55d0d01\scratchpad`

The foundation includes repeatable full-round simulation, legal sampled solo spawns, recovery, branch outcomes, and strong short-horizon native physics agreement. The log reports 37,900 one-reply comparisons, near-exact position/velocity agreement, and 99.97-100% pickup agreement. That is valuable infrastructure for systematic improvement.

| Experiment | Result | Interpretation |
| --- | --- | --- |
| Expensive rollout selection, 400 situations | +4.64% mean per-map relative points over 20-second windows | Some existing alternatives help when evaluated with costly continuations. This is not a live or full-round gain. |
| More existing search work | Reported +2.3% at 3x, +3.1% at 10x | Modest measured benefit; the prefix issue below limits the matched-state interpretation. |
| Learned ranking, 1,400 generated training situations | Final held-out argmax ratios 0.998 ridge / 0.986 small GBM; 0.995 for all-data ridge | The tested rankers failed. More of the same labels/features is not justified. |
| Cheap eight-second continuation ranking | Argmax ratios 0.998 / 1.003 | The tested substitute did not recover expensive rollout selection's benefit. |
| Coverage batches | 16 variants, now all 80 paired rounds each; two variants inert | No convincing positive configuration. Existing switch tuning is exhausted enough to stop. |

I independently recomputed coverage deltas from the completed JSONL files. Baseline: 134.775 points. Uncapped speed: +0.887 points, SE 1.775; falls increased in the logged comparison. Other effective variants had negative means. Final batch 3: no A2 -3.350, no climbs -2.450, no lines -2.763, no gap routes -1.025, no edge hits -1.525 points. The last number supersedes the incomplete-batch log's -1.84.

The log's `+-` numbers are standard errors, not 95% intervals. Several negative means remain uncertain. CORNER_PROPOSALS and proposal-samples 40 were inert through the harness, not successful tests showing no headroom.

There is also a reporting correction: `clairvoyant_alt` excludes retaining the baseline, so its supposed best-choice improvement is negative in 67/400 situations. Including the baseline changes the mean per-map retrospective gain from 9.14% to 11.73%. With rejected candidates included it remains 14.18%. These are optimistic selections on the same evaluation samples, not attainable expected gains or evidence of a live ceiling. The corrected incremental retrospective advantage of adding rejected candidates is about 2.45 percentage points.

## Why the negative ranker result has a limited scope

The feature vector contains 38 planner/state summaries and eight coarse path summaries. It lacks the terrain layout and detailed arrangement of remaining targets needed to distinguish many routes. Powerup locations are reduced to a count. This tests a compact feature-based ranker, not a general learner with the full spatial state.

Most alternatives are executed for only 16 replies (1.024 seconds), then handed to the existing planner for the remainder of 20 seconds. Their saved features usually describe a 48-reply forecast, sometimes substantially longer. Thus the label concerns a brief intervention followed by the old controller, while some input features describe a hypothetical endpoint that was never executed.

I found additional comparison defects in source:

1. **Baseline identity:** branch_bench.py takes the chosen candidate from the last recorded internal `search()` call. The controller can choose an earlier target search or rerank/replace its result. That assumption can associate baseline outcomes with another candidate's features. The frequency and effect have not been measured.
2. **Unequal continuation protocols:** the baseline immediately runs closed loop; alternatives run their forced prefix and invalidate the controller's plan at handoff. Compare identical commitment/handoff protocols when measuring candidate value, and retain the untouched production baseline as a separate reference.
3. **Different starts for the compute arms:** the multiplied budget is set before replaying the 47-decision lead-in. Those arms can arrive at different root states. Restore the same baseline root/controller state before changing the budget.
4. **Missing powerup respawns at snapshot starts:** HuntWorld initializes its powerup list from visible `before.items` and starts with an empty respawn schedule. The game export explicitly excludes hidden items. A cooling powerup at initialization can therefore remain absent forever. Verified example: Sprawl, hold3 seed 278350961, decision 3443, a sampled pilot lead-in start, has a helicopter due back in 6.824 seconds in the raw observation but absent from `before.items`. The constructor does not restore it. Initialize all mission powerups and their legitimate observable timers, or exclude snapshots whose full timer state cannot be reconstructed.

These issues warrant a small repair and verification pass before generating new labels. They do not demonstrate that repairing the ranker experiment will produce a winning model. Do not regenerate the full label set automatically.

The physics conformance test also does not validate all HuntWorld behavior: it tests advance_world, restores recorded item lists during chains, and skips respawn physics comparisons. Full-world timer/recovery behavior needs its own narrow checks. Whole-round aggregate sim/native agreement (134.0 versus 134.2) masks map discrepancies and materially different fallback rates under the virtual versus native parallel budget. Treat the sim as a screening environment until native confirmation.

## Recommended next experiment

The missing pilot arm is explicit in CONVO.md: broader multi-event search/refinement under the same executed-points objective. branch_bench.py still describes arm 4 as a later option. Increasing order count, removing speed caps, or adding six refinements scored by the old cost does not implement that arm.

Build one offline trajectory-improvement prototype that can change successive decisions together:

- Start with baseline trajectories. Keep several physically different intermediate outcomes, including different velocity, direction, and inventory at the same target.
- Search across successive pickups, junctions and resource decisions. Refine steering, throttle, timing and target order across those events. Avoid reducing every branch to the old candidate cost before its consequences can be evaluated.
- Use actual points over equal 10-20-second windows. Sample unknown future groups legally; choose on planning draws and evaluate on independent draws. Include ordinary continuation beyond the optimized segment so exit state matters.
- Execute short prefixes with feedback. Include small start perturbations and native replay checks to expose brittle simulated solutions.
- Always retain the baseline option and report total work, including continuations. This first prototype is an offline teacher; live feasibility is a separate gate.

Proposed scope: a 10-situation throughput/sanity pass; roughly 40 development situations distributed across all ten maps; then one fresh evaluation of roughly 100 situations split by source round. Set an explicit compute cap from the timing pass before the evaluation. Use a small ladder such as 3x/10x/30x, with 100x only on a limited subset if the curve is still improving. Do not spend more time on a flat curve.

The investment threshold should be material, approximately 10% held-out window-points improvement as a proposed target, with uncertainty and per-map results reported. This is not a prediction and does not translate automatically into a round-score gain. One prototype and one reserved evaluation; no repeated tuning on that evaluation set. Small or inconclusive effects at the cap do not justify another large labeling campaign.

If it succeeds, learn reusable proposals/control segments from the improved behavior using spatial observations, then iterate on states the new policy actually visits. Value learning can support that loop. Require full-round sim improvement and actual live native confirmation before promotion.

If it fails, stop this search design. My next candidate would be a learned motion/proposal policy with a richer observation and broader action space, using usable expert demonstrations where available and then simulation-based policy improvement. That is a separate proposal requiring its own feasibility check, not an automatic training launch. Another fit of the same summary-feature ranker would not answer the failure.

The search/learning separation is motivated by [Expert Iteration](https://arxiv.org/abs/1705.08439); physics-based trajectory refinement has precedent in [Predictive Sampling](https://arxiv.org/abs/2212.00541). Neither paper establishes effectiveness for Marble Blast. The bounded project-specific experiment must do that.

## Role of runtime work

Keep the measured terrain lookup optimization and consider other small equivalent speedups. A wholesale C++ port is not yet supported as the main route to a large score increase. Slow-family removals show their contribution under the fixed virtual work budget; that budget does not reproduce their differing Python overhead or the live queue. The ablations neither prove every family must be ported nor measure the score gained from doing so.

Before committing to that large port, require a live timing comparison that establishes substantial quality gains from a realistically achievable runtime improvement. Speed remains useful for reliability and teacher throughput, but faster execution of the present search is a different investment from discovering better trajectories.

Preserve compact experiment summaries and artifact manifests outside the temporary scratch directory. The main deliverable sought next is a reproducible improvement in behavior under the points objective, with a clear path to reuse what was learned.
