# General marble dynamics review (2026-10-12)

Native recording, complete-map access, an independent mechanics baseline and review overlays are implemented. The policy has not been connected or trained. Long-horizon contact outliers and external events keep planning gated.

## Evidence

| Dataset | Maps | Accepted fixtures | Native transitions | Repeats |
|---|---:|---:|---:|---:|
| pilot_20261011 | 10 | 360 | 57678 | 80 |
| effects_20261012 | 4 | 48 | 12977 | 8 |
| timers_20261012 | 4 | 48 | 16092 | 8 |
| interventions_20261012 | 4 | 52 | 13832 | 0 |
| controls_20261012 | 4 | 52 | 13831 | 0 |
| wall_jumps_20261012 | 4 | 72 | 14323 | 0 |

All 728 archives passed structure checks; 632 principal recordings match their saved raw hashes. Repeats independently match on shared physical prefixes; 19 differ in episode termination length. Repeat data is not counted as independent fixtures.
Map/trajectory roles were assigned before collection; historical policies have seen these maps. These fixtures do not prove exhaustive mechanics coverage.

## Broad ten-map validation

Errors below are game units. Endpoints use variable-length continuous prefixes (external item/respawn/size changes stop a rollout). Failures are retained separately; endpoint statistics exclude failed rollouts. Each legacy map has one refused Super Speed one-step sample because the old schema lacks its prior contact normal.

| Map | 16 ms p95 | 1 s p95 | 2 s p95 | Endpoint p95 | Endpoint max | Complete / accepted fixtures |
|---|---:|---:|---:|---:|---:|---:|
| Bowl | 2.6e-10 | 0.009895 | 0.04923 | 0.1388 | 1.829 | 36 / 36 |
| Concentric | 2.64e-10 | 6.148e-07 | 5.014e-05 | 0.01431 | 0.3054 | 36 / 36 |
| Cragmire | 2.14e-10 | 4.778e-07 | 1.773e-06 | 2.487e-05 | 8.681e-05 | 34 / 36 |
| CubeIsle | 3e-10 | 1.323e-05 | 0.00119 | 0.02332 | 4.604 | 36 / 36 |
| Duplex | 2.31e-10 | 0.01007 | 7.764e-06 | 0.04674 | 0.1634 | 36 / 36 |
| FracturedIslands | 2.36e-10 | 0.01573 | 0.002198 | 0.04711 | 0.2964 | 32 / 36 |
| KingOfTheMarble | 2.31e-10 | 0.04309 | 4.063 | 0.02006 | 12.98 | 33 / 36 |
| Sprawl | 4.44e-10 | 0.007734 | 1.31 | 0.7991 | 7.637 | 36 / 36 |
| TreasureBox | 4.26e-10 | 2.385e-06 | 0.1707 | 0.2396 | 3.445 | 36 / 36 |
| VortexEffect | 2.36e-10 | 4.749e-05 | 0.02305 | 0.01344 | 0.0497 | 35 / 36 |

## Matched mechanics and map access

The 124 spin/jump/powerup candidates also run from their initial state and applied Moves/absolute camera commands with no future camera/state observations. Endpoint p95 = 0.000394193u; max = 0.502085u on these selected fixtures. Physical outcomes, initial spin, actual used item IDs and event truncations are in review_20261012.json. Matched landing differences illustrate mechanics, not learned policy skill.
The additional wall-approach set has 46/60 actual jump-command intervals with native support, versus 4/16 in the original wall-timing set. Support is a necessary actuation proxy, not an exact native impulse count. No-jump controls and longer approaches expose timing-dependent contact outcomes.
An additional control-contract check found that Move-only replay of old tapes can miss by ~57u: the bridge sets absolute camera yaw outside Move. The new controls dataset records those actual command events explicitly. Proposed camera commands now drive the candidate interface; future observed camera state is not copied.
All 47,932 static contact triangles are retained in the hierarchy; 2,000 reloaded queries match the exact reference. 64–512 whole-map leaf tokens are accessible. The encoder is randomly initialized; exact distant detail remains queryable. The Python hierarchy is slower than the vectorized reference on these map sizes.

## Remaining gates

- Rare multi-contact/contact-set errors can lead to multi-unit rollout drift. Native local hull order and F32 handling reduce errors but do not eliminate them. Tangent sqrt/NaN and slide/bounce branches are explicitly diagnosed.
- Microscopic perturbations produce alternative rebound outcomes. The selected 24-member demonstration is not calibrated coverage; uncertainty must be measured on independent contact trajectories before relying on landing clearance.
- Mega/scripted size changes, Blast, pickups, respawns, arbitrary gravity rotation and moving actors need a discrete-event or dynamic-object layer. Applied Move replay does not validate delayed bridge action execution.
- No learned dynamics model or learned map encoder has been fit, and no online planner or policy integration is approved.

Open collision_maps/viewer/index.html, choose a recording, then focus/scrub the actual and predicted paths. The primary change log and exact commands are in docs/HANDOFF_MARBLE_DYNAMICS_2026-10-11.md.

Predictor/evaluation version: `44aba1cfc83f8e131b11ef8982c9a71b7088eb6541b28e894136fe77ed50ad63`.
