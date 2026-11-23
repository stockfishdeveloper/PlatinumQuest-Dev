# Connected pickup planning — 2026-11-07

**Complete: no verified points improvement, no promotion, and no training campaign.** The best development variant failed both its reserved-window and full-round tests. Isolated retention and the final retention-with-replanning follow-up also failed their separate reserved tests. All experimental settings remain disabled by default.

| Comparison | Pairs | Baseline points | Candidate points | Change | Forecast work |
|---|---:|---:|---:|---:|---:|
| Retain existing second-leg controls, development windows | 20 | 7.90 | 8.80 | +11.4% | 0.975x |
| Retention + consistent candidate scoring, development | 20 | 7.90 | 9.45 | +19.6% | 0.943x |
| Above + bounded local refinement, development | 20 | 7.90 | 9.25 | +17.1% | 0.956x |
| Selected scoring variant, reserved windows | 20 | 9.45 | 9.05 | **-4.2%** | 0.879x |
| Consistent outer first-target ranking, development | 20 | 7.90 | 9.15 | +15.8% | 0.947x |
| Isolated retention, separate reserved windows | 20 | 9.95 | 8.80 | **-11.6%** | 0.881x |
| Retention with ordinary replanning, development | 20 | 7.90 | 8.85 | +12.0% | 1.055x |
| Scoring with ordinary replanning, development | 20 | 7.90 | 9.45 | +19.6% | 0.964x |
| Retention with replanning, fresh reserved windows | 20 | 10.65 | 10.10 | **-5.2%** | 0.944x |

Points are equal-map mean points per approximately 16-second played window, not predicted pickup counts. Every comparison covers all ten maps. Each arm replays identical baseline history from native GO to its root, preserving queued inputs, hidden powerup timers, active plan and recurrent state; root hashes matched. Forecast work includes continuation construction and validation. Total experiment work, including the repeated prefixes, is recorded separately in each summary.

The development map-cluster bootstrap interval for the scoring variant was +0.4 to +2.7 points/window. The reserved interval was -1.15 to +0.25. Isolated retention's independent reserved interval was -2.85 to +0.15; it produced 34 pickup handoffs and two fewer falls, but fewer points. These are small sets with substantial variation between situations. The development gain is not a verified improvement. Fewer forecast steps also do not imply better play: the reserved candidates searched less while scoring less.

An additional descriptive check tracks completion of exactly the gem group alive at each root, using mission gem indices. In the selected scoring variant's reserved windows, both arms cleared 18/20 starting groups. Candidate completion was 0.474 seconds slower on average when incomplete groups are capped at the window end. This is a restricted-time measure, not an uncensored completion-time estimate. It provides no evidence that improved local execution was merely hidden by later spawn differences (`root_group_clear.json`).

Full-round confirmation completed 20 pairs across all ten maps: **140.45 -> 136.50 points/round (-2.81%)**, mean difference -3.95 points, map-cluster interval [-10.1,+1.4]. Forecast work was 0.942x baseline, CPU 0.938x, with 283 pickup handoffs and two more falls. Map deltas were Bowl +1, Concentric -5.5, Cragmire +3, Cube Isle +4.5, Duplex 0, Fractured +2, King of the Marble -10, Sprawl -20, Treasure Box +6 and Vortex -20.5. This candidate does not advance to native confirmation. All natural rounds finished and complete-root hashes/source guards passed.

The last replanning confirmation's interval was [-1.45,+0.15 points/window], with 41 observed handoffs and no fall-count difference. It also fails the improvement gate. These intervals do not establish that every variant has a negative population effect; they do establish that the development gains are insufficient grounds for promotion. Losses in the full-round scoring test also occurred without falls: both seeds on King of the Marble and Sprawl lost points with zero falls in either arm.

The current proposed experiment ends here. Keeping a checked second leg, making its modeled score comparable, and refining that score did not produce a reliable points gain. The replanning follow-up shows that bypassing the ordinary research policy was not the sole explanation. Keep the baseline and retain the implementation only as an opt-in experimental capability.

The next research investment I recommend is a bounded **offline trajectory teacher selected by actual executed points**, retaining several physically distinct routes and exit states across successive pickups. Use the same complete roots and handoff rules, include the baseline, count all continuation work, and separate planning futures from evaluation futures. Establish a material fresh-case teacher gain before training a proposal policy. The current local refinement used a modeled spawn-clear-time objective; it did not implement that broader executed-points search. The earlier compact ranker and cheap rollout scorer had already failed ([November 1 review](../../docs/IMPROVEMENT_RESULTS_REVIEW_2026-11-01.md)); repeating that training is not supported by this result. No such new teacher or training campaign was launched here.

## Implementation

- Retain the actual follow-up controls already computed by two-level search. Validate their complete root trajectory and braking continuation before execution. Attach the continuation at the controller's **final** selection, since the intermediate search winner may be discarded by later selection.
- Carry the plan across a confirmed pickup, reforecast it against the next observation, and cancel on divergence, context change, missing target or exhausted schedule. Guard against item IDs reused in a new spawn.
- Give ordinary and connected candidates the same modeled spawn-clear-time objective. Remove the bonus for merely predicting more gems over a longer horizon, and prevent the two-level reranker from replacing the whole-spawn score. Keep the physical admission checks.
- Test bounded changes to subsequent gem order, extension to up to four pickups, steering/throttle, braking before a cluster and jump/power timing. Every accepted update starts from a feasible sequence and is reforecast from the actual root. This refinement did not beat the simpler scoring variant in development.
- A separate `order` experiment carries the candidate objective into the outer first-target selector. It gained less than `score` in development, so its separately declared reserved set was not run.
- Bounded refinement made 409 local trials, accepted 19 cost improvements and selected 11 refined plans in development. Its lower point score than `score` means improvement in this modeled objective did not translate into a better campaign result. This does not establish that a broader search or a better value model would fail.
- The final follow-up adds `connected.RESEARCH_HANDOFF` (default false; native flag `--connected-research-handoff`). A checked continuation can use the existing pickup-plan research cadence and replacement test. Its actual-state forecast is reused once, its absolute schedule remains intact when it wins, and a failed/timed-out search keeps that checked schedule. No cost weights or map rules changed. This follow-up's regression suite passes 144 tests.

Replanning added only one point across all 20 retention development windows versus unchanged retention, while increasing forecast work. Its interval versus baseline was [-0.35,+2.65 points/window], and Duplex again supplied most of the gain. It narrowly satisfied the declared parent-improvement rule and received one fresh confirmation, which failed. Scoring with replanning tied the original scoring arm's points and per-map deltas while using more work; it did not advance.

The default is `connected.MODE='off'`. Experimental native runs can choose `--connected-mode retain|score|refine|order`. No neural-network weights were changed.

## Verification and evidence

Archived initial search/controller versus disabled mode produced identical points, pickup events/times, labels, forecast steps, search counts, falls and fallback counts in both 45-second probes (King of the Marble and Sprawl). The frozen scoring revision also reproduced its earlier probes exactly. Source archives, root hashes, matrix declarations and per-case logs are beside each summary.

- `baseline_source.zip`, `baseline.json`, `disabled_conformance.json`: reference and compatibility checks.
- `retention_dev/summary.json`: isolated retention experiment.
- `scoring_dev/summary.json`: scoring/refinement comparison.
- `heldout/summary.json`: reserved result for the selected scoring variant.
- `order_dev/summary.json`: outer-ranking development experiment, did not advance.
- `retention_reserved/summary.json`: independent component confirmation, failed.
- `full_score/summary.json`: complete-round confirmation with fresh future seeds, failed.
- `replan_dev/summary.json`: separately declared handoff-research comparison.
- `replan_conformance.json`: all 20 baseline development cases reproduced exactly after the handoff-research change (points, pickups/timings, labels, steps, falls, searches, fallbacks and full-root hashes).
- `replan_reserved/summary.json`: final retention-with-replanning confirmation, failed.
- `campaign_summary.json`: all eight declared comparisons, source identities and compute totals.

The declared matrices completed **400 simulation runs**, using **11.28 CPU hours and 2.539 billion forecast steps**, including replayed prefixes, continuation construction and validation. Development cases are reused across variants, so this is not 400 independent situations. Preflight probes and tests are additional. Each matrix archives its exact source and checks it remained unchanged through completion. No evaluation processes remain running.

The native live harness currently waits for late plans. An opt-in deadline benchmark was added and unit-tested to expose that behavior: use a matching checked continuation, otherwise send an ordinary braking fallback and discard stale in-flight work. It is **not native-validated or enabled by default**, and braking is not claimed to be a checked recovery on arbitrary terrain. Native point/timing confirmation was conditional on a promising candidate; no native games were launched because the simulator confirmation gates failed. There is no live timing or native points improvement claim.

Automatic approval review blocked cleanup of the temporary archived-source copies under `%TEMP%/pq_connected_baseline_20261107`, returning only “blocked by policy.” They remain outside the Git working tree. No commits were made.
