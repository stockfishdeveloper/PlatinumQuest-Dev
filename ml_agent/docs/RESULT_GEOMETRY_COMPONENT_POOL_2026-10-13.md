# Geometry routing and physical search: complete component comparison

Completed2026-10-13. Seed681013, all30original-duration natural native rounds. All score/input audits pass; no implicit OOB keys. Frozen A2 weights and170archived source files verified unchanged at completion. This is engineering evaluation, with no policy training or promotion.

| Map | Combined points | Geometry route points | Legacy points | Falls combined/geometry/legacy |
|---|---:|---:|---:|---|
| CubeIsle_Hunt | 58 | 54 | 59 | 18 / 3 / 5 |
| Bowl_Hunt | 66 | 17 | 19 | 0 / 0 / 0 |
| Duplex_Hunt | 22 | 34 | 13 | 49 / 35 / 53 |
| FracturedIslands_Hunt | 89 | 12 | 12 | 14 / 40 / 51 |
| Cragmire_Hunt | 43 | 8 | 1 | 6 / 1 / 0 |
| VortexEffect_Hunt | 73 | 0 | 0 | 32 / 109 / 138 |
| TreasureBox_Hunt | 95 | 80 | 42 | 0 / 0 / 0 |
| Concentric_Hunt | 27 | 13 | 0 | 9 / 10 / 59 |
| Sprawl_Hunt | 106 | 1 | 0 | 0 / 0 / 0 |
| KingOfTheMarble_Hunt | 133 | 129 | 124 | 0 / 6 / 7 |

Equal-map means **71.2 /34.8 /27.0**. Combined versus legacy: median map gain44.5,9wins/0ties/1loss. Geometry versus legacy: median gain3,6wins/2ties/2losses. Combined versus geometry: median gain25,9wins/0ties/1loss. KOTM is one equal-weight map; it has no special rejection gate. Nine-other-map diagnostic64.33/24.33/16.22 remains supplementary.

Falls **128/204/313**, accepted rescues15/39/29, rescue attempts19/67/52. Whole-round wall3333.39/685.15/617.43seconds (55.56/11.42/10.29minutes), including measured controller work. Search has a one-second deadline, which is not a whole-decision or realtime guarantee.

The combined approach shows broad benefit, but Cube has18falls against routing-only3, and Duplex scores22against routing-only34 with49versus35falls. Duplex fell repeatedly under actor fallback; earlier checked actions and queued commands still need causal inspection. Routing-only helps some maps substantially, especially Treasure, but is insufficient on Vortex and Sprawl. Preserve these unfavorable outcomes.

All branches use the unchanged local-observation A2 policy. Geometry-only changes routing, without teaching A2 to read the full mesh. The combined branch also changes global gem selection and resource handling, so its residual gain cannot be credited solely to physics search. Separate native launches and one development seed do not establish statistically matched worlds or generalization. Compare the earlier r4 batch separately; do not pool unlike versions/settings.

Next: queue-aware native activation audit, complete trajectory review, generic fall/recovery and game-rule-volume investigation. Collision metadata already contains bounds triggers, but current physical acceptance checks do not enforce them; verify actual trajectories and native semantics before attributing a failure or deploying a guard. Long PPO and checkpoint promotion remain paused.

[Authoritative component results](../dynamics_reports/geometry_component_pool_20261013/pool_result.json), [all reports](../dynamics_reports/geometry_component_pool_20261013/summary.json), [declared protocol](PROTOCOL_GEOMETRY_COMPONENT_EVALUATION_2026-10-13.md).
