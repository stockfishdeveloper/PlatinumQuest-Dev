# Full-round A2 reference and comparability

2026-10-12. Read-only native evaluation; no student fitting, PPO or model changes.

[Sealed ten-map reference](../dynamics_reports/full_round_a2_pool_reference_20261012/summary.json)
completed10/10 original-duration rounds in340.73s wall. Frozen A2 update37760,
SHA256 ae88f2e70a376fda06be6aed59e8bd17fd6b7b0a876bdde9c40838eadddbfb04.
Current64ms physics/64ms decisions, deterministic learned actions, routed walk
tour, Competitive off, legal OOB click and eight-second stuck rescue. Rescue
costs native round time. Native end score equals accumulated pickup points in
every round; start score0 and full remaining clock verified. Every source/weight
hash stayed fixed for the sweep. First-play world and recurrent state are archived.

One seed481012, one round per map; equal-map mean **17.2 points**. This is a
preliminary CURRENT-runtime reference, not candidate improvement or a reliable
estimate of A2's best/representative score. All unfavorable maps remain included.

| Map | Official round seconds | Native points | Falls | Rescues |
| --- | ---: | ---: | ---: | ---: |
| KingOfTheMarble | 180 | 110 | 16 | 0 |
| Sprawl | 300 | 9 | 0 | 9 |
| CubeIsle | 210 | 4 | 60 | 0 |
| Bowl | 300 | 4 | 0 | 4 |
| Duplex | 240 | 13 | 40 | 3 |
| FracturedIslands | 180 | 19 | 43 | 0 |
| Cragmire | 240 | 7 | 31 | 3 |
| VortexEffect | 300 | 1 | 114 | 0 |
| TreasureBox | 240 | 5 | 0 | 7 |
| Concentric | 240 | 0 | 15 | 8 |

## Why this does not replace the earlier32-33-point reference

The same protected checkpoint's Oct10 two-round mean was33.1 points, KOTM124.5;
its earlier four-round shaping sweep was32.5, KOTM128.5. Those are separately
preserved historical runtime/start conditions. Example: historical Treasure
scored8 then60 in its two rounds, making a single-map seed a poor estimate.

[Historical-seed Fractured check](../dynamics_reports/full_round_a2_historical_seed_fractured_20261012/summary.json)
repeats original config-order seed10095 with two successive complete rounds under
current inference/runtime. It scores83 and88, mean85.5, versus historical91;
new seed481012 had19. This supports substantial starting-condition variation
contributing to the difference. It does not establish that every runtime change
is harmless or that the seed alone is the complete cause. Do not claim model
regression from17.2, or promote a candidate merely because it clears17.2.

KOTM smoke launches also varied111/124/110 at the same requested481012 seed.
An RNG seed is not a certified full replay. Starting/first-play records help
identify differences but do not by themselves certify equality of all native
physical, control-queue, script and recurrent-history state. Use exact matched
fixtures only where those complete states are genuinely checked; otherwise
report independently replicated score distributions.

## Next comparison

[Three-seed preparation](../dynamics_reports/full_round_reference_preparation_r4_20261012/protocol.json)
declares30 rounds, not yet run. Replicate the candidate and A2 under the SAME
full-round cadence/rules/map pool before promotion, with native game points as
primary and individual-map/KOTM regressions visible. The experimental16ms
physics planner needs a matched16ms A2 full-round reference; it cannot be compared
as though the current64ms reference or first-fall snippets were identical tests.

The existing real_run replay getSimTime() duration also includes roughly4.3s
outside the official countdown/end boundary in these runs. Its legacy rate
telemetry is separately archived; the primary reference is native points over
the original Hunt duration, not equality of the two clocks. The new score audit
retains both clocks and rejects partial starts, reconnects and missing/disagreeing
native scores. Five meaningful score-contract regressions pass.

The new launcher initially failed on a relative output path; its first retry
completed a round but hit the incorrect clock-equality audit. Both failed audits
and original native outcomes remain preserved. Subsequent smoke/pool checks pass.
No test artifact overwrote any earlier experiment or trained checkpoint.

[Path to higher scores](ROADMAP_FULL_MAP_LEARNING_2026-10-12.md).

## Completed three-seed replication

All30 requested native rounds completed and passed full-start, native end-score/pickup agreement and source/checkpoint checks. Original64ms physics/64ms decisions, original Hunt durations, competitive off, legal OOB click and8s time-cost rescue. Equal-map mean24.9333; wall1,008.64seconds (two owned instances). No policy fitting or checkpoint change.

| Map | Native points by seed481012/481013/481014 | Mean |
|---|---|---:|
| KingOfTheMarble_Hunt | 118, 125, 129 | 124.00 |
| Sprawl_Hunt | 9, 29, 8 | 15.33 |
| CubeIsle_Hunt | 9, 4, 36 | 16.33 |
| Bowl_Hunt | 4, 9, 9 | 7.33 |
| Duplex_Hunt | 13, 56, 22 | 30.33 |
| FracturedIslands_Hunt | 19, 21, 74 | 38.00 |
| Cragmire_Hunt | 17, 0, 7 | 8.00 |
| VortexEffect_Hunt | 2, 0, 0 | 0.67 |
| TreasureBox_Hunt | 5, 0, 15 | 6.67 |
| Concentric_Hunt | 2, 3, 3 | 2.67 |

This24.9mean replaces neither the previous17.2 single-seed diagnostic nor historical32.5/33.1 as a universal baseline. KOTM124 is close to the earlier same-weight124.5; hard-map/start variability remains large. Current/historical profiles and seed sets must remain distinct. No new controller is compared to the64ms result unless it uses those same settings; the16ms continuous hybrid uses its own16ms A2 branch. Three starts/map are an initial screen, not a promotion claim.

[Complete audit/provenance](../dynamics_reports/full_round_a2_replicated_20261012/summary.json). Captured source bytes are in `source_snapshot.zip`, member hashes included. Every failure from earlier smoke attempts remains archived.
