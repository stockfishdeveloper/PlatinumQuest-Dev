# Scaled iteration 1 collection profile

All time buckets are exclusive. These are summed worker times, not parallel elapsed wall time; cold map setup is recorded separately.

| Work | Worker seconds | Share |
|---|---:|---:|
| Route building/search excluding sweep | 2215.07 | 24.10% |
| Exact collision sweeps | 2162.66 | 23.53% |
| Dynamics forecast excluding route/sweep | 4297.90 | 46.75% |
| Other collector work | 516.85 | 5.62% |

Overall collection elapsed: 41.03 minutes, including 4.57 minutes of additional native student rounds. This includes the completed initial collection carried across the maintenance resume.
Checked corrections: 4,292 new; 4,589 cumulative. No raw proposal is counted as an accepted causal label.

| Batch/map | Checked corrections | Query median ms | Query p95 ms | Cold setup seconds |
|---|---:|---:|---:|---:|
| iteration_1_collection/Cragmire_Hunt | 126 | 242.7 | 641.0 | 0.974 |
| iteration_1_collection/FracturedIslands_Hunt | 256 | 260.0 | 517.6 | 0.817 |
| iteration_1_collection/VortexEffect_Hunt | 200 | 364.9 | 835.1 | 0.787 |
| iteration_1_collection/Sprawl_Hunt | 352 | 368.6 | 683.6 | 0.827 |
| iteration_1_collection/Duplex_Hunt | 338 | 403.6 | 736.5 | 0.830 |
| iteration_1_collection/KingOfTheMarble_Hunt | 280 | 419.9 | 740.1 | 0.774 |
| iteration_1_collection/CubeIsle_Hunt | 295 | 417.9 | 834.0 | 0.806 |
| iteration_1_collection/Bowl_Hunt | 214 | 570.5 | 1019.2 | 1.094 |
| iteration_1_collection/Concentric_Hunt | 339 | 405.6 | 840.4 | 0.328 |
| iteration_1_collection/TreasureBox_Hunt | 383 | 422.6 | 954.2 | 0.314 |
| iteration_1_extra_states_1_corrections/Cragmire_Hunt | 127 | 215.1 | 559.5 | 1.041 |
| iteration_1_extra_states_1_corrections/FracturedIslands_Hunt | 256 | 231.9 | 470.8 | 0.808 |
| iteration_1_extra_states_1_corrections/VortexEffect_Hunt | 199 | 321.7 | 690.5 | 0.822 |
| iteration_1_extra_states_1_corrections/KingOfTheMarble_Hunt | 280 | 358.4 | 601.6 | 0.783 |
| iteration_1_extra_states_1_corrections/CubeIsle_Hunt | 297 | 361.5 | 715.8 | 0.805 |
| iteration_1_extra_states_1_corrections/Bowl_Hunt | 223 | 478.1 | 1017.5 | 1.153 |
| iteration_1_extra_states_2_corrections/Cragmire_Hunt | 127 | 188.1 | 496.5 | 0.700 |

Bowl exceeded 500 ms in the initial batch and uses four candidates in iteration 2. The fixed prediction horizons and causal acceptance filter are unchanged.

Source: `iterations.json`, iteration 1. Collection integrity passed; independent full reply replay is pending unless this candidate is selected.
