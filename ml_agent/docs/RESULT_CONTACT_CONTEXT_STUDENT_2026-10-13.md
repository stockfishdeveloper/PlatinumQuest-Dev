# Contact-context motion result

Bounded development-only run completed5000updates in200.29seconds, seed121014. Dev-selected update4600 passes the predeclared ordinary-motion gate on both maps. Balanced normalized development MSE falls from0.02462934 to0.01446070 (41.3%reduction). Update0 and4600 were the only eligible scheduled checkpoints. This is useful one-step development progress, with no test reopening, policy training or deployment.

| Development map | Ordinary mean position/velocity/spin errors | Velocity baseline |
|---|---|---|
| Concentric_Hunt | 0.005089 / 0.525343 / 1.219785 | 0.005231 / 0.533972 / 1.187615 |
| CubeIsle_Hunt | 0.003936 / 0.381000 / 0.828977 | 0.004052 / 0.394384 / 0.827360 |

Units are worldu, u/s andrad/s. Position and velocity ordinary means improve slightly; spin may worsen within the declared5%tolerance. The selected model has109,001parameters and keeps all47,932operative/source surfaces across the dataset. Four attention summaries, exact closest face/real-edge features and90/5/5sampling changed together: this is not an isolated architecture attribution.

The model remains unsuitable for gameplay forecasting. Airborne jump velocity errors are5.19/7.15u/s versus the baseline0.338/0.339; grounded jump errors improve to2.53/2.08against7.07but are still large. Held-use mean errors improve in aggregate, but a single Super Speed frame still misses position by0.438/0.428u and velocity by27.5/30.2u/s. Power counts are only3per type per development map. Ordinary contact tails also remain large. Native within-step support labels are post-hoc audit proxies, never inputs.

No learned world rollout, effect/actuation evolution or calibrated uncertainty is established. Existing inference cache intentionally refuses this alternate forward. A2/native binaries and immutable datasets/readers are preserved; source ZIP byte verification passes. No newPython files.

[Result](../dynamics_reports/native_motion_contact_context_fit_20261013/result.json), [all scheduled evaluations](../dynamics_reports/native_motion_contact_context_fit_20261013/history.json), [jump contact diagnostic](../dynamics_reports/native_motion_contact_context_fit_20261013/jump_contact_diagnostic.json), [declared protocol](PROTOCOL_CONTACT_CONTEXT_STUDENT_2026-10-13.md).
