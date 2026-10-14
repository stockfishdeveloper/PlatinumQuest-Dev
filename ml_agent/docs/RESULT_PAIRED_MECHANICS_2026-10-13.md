# Direct supervision of native action effects

The verified native physical fixtures now support90 one-step action-effect examples across the six training maps. Each example compares two actual controls at the first changed native Move, when the recorded physical BEFORE state is still identical. It teaches what the control changes physically, using measured position, velocity and spin deltas. It has no reward, gem score, route, maneuver name or future contact label as an input.

The earlier data-only contact-context fit completed5000updates in199.831seconds and failed ordinary-motion preservation at every learned checkpoint. Its selected update0 is unchanged velocity drift. Initial development metrics and all development arrays matched the preceding preparation exactly. Keep that negative result: more diverse mechanics data alone did not fix this learner. [Failed fit](../dynamics_reports/native_motion_interventions_run_20261013/result.json), [comparison audit](../dynamics_reports/native_motion_interventions_run_20261013/comparison_audit.json).

## Paired-label contract

The collection supplies66 verified ordinary-mechanics pairs, and the transfer preflight supplies24 verified response pairs including held Super Speed. Six refused or world-event cases are excluded from paired supervision and retained in the original collection. We do not pair later frames after the two actions have produced different states. Native archive hashes, copied physical fields, first changed Move, earlier prefix, clock phase, playable bodies and unchanged within-step body parameters are all checked again.

| Training map | Verified action-effect examples |
| --- | ---: |
| Sprawl |16|
| King of the Marble |13|
| Duplex |16|
| Cragmire |15|
| Treasure Box |16|
| Fractured Islands |14|

Inputs retain the existing84-feature native BEFORE/control schema and complete-map geometry. Only `move_y`, `move_jump` or `move_use` differs between the paired input vectors. The targets are native AFTER motion differences. Within-step support counts are retained for diagnostic grouping only. Original recordings, metadata, readers, tensor bundles and frozen A2 are unchanged. [Pair preparation](../datasets/marble_dynamics/native_action_difference_20261013/manifest.json), [per-example audit and exclusions](../datasets/marble_dynamics/native_action_difference_20261013/audit.json), [input-difference audit](../datasets/marble_dynamics/native_action_difference_20261013/preflight_audit.json).

## Bounded experiment

The optional fitter loss adds the MSE between predicted and measured action-effect differences to the existing marginal native-motion MSE, weight1.0. Four pairs are sampled on the chosen training map per update. Architecture, optimizer, seed121014,5000updates/batch32/300second training-loop wall limit,90/5/5 marginal sampling and development-only selection remain unchanged. A separate paired sampler seed121015 leaves the original marginal random sequence intact. No new output head or rule assigns a meaning to jump or use.

Twenty preparation/loss/geometry contracts pass, including unchanged default loss, rejection of mismatched starts/delayed Moves, and future motion affecting labels without changing inputs. The run completed5000updates in237.559seconds. No learned checkpoint meets ordinary preservation; selected update0 remains velocity drift, and initial Dev metrics match the data-only fit exactly. The auxiliary intervention loss alone does not fix this architecture. All records/checkpoints remain isolated; no deployment. Each development map must still meet the existing ordinary p/v/spin preservation threshold. Grounded versus airborne jumps and activation tails need separate interpretation. Test labels stay closed. No policy training, native forecast integration or checkpoint promotion occurs. KOTM remains one equally weighted gameplay map, with no special score veto.

[Prepared protocol](../dynamics_reports/native_motion_paired_effect_protocol_20261013/protocol.json), [execution protocol](../dynamics_reports/native_motion_paired_effect_fit_20261013/protocol.json), [general-mechanics plan](PLAN_GENERAL_MECHANICS_INTERVENTIONS_2026-10-13.md).

[Completed result](../dynamics_reports/native_motion_paired_effect_fit_20261013/result.json), [all scheduled evaluations](../dynamics_reports/native_motion_paired_effect_fit_20261013/history.json), [comparison audit](../dynamics_reports/native_motion_paired_effect_fit_20261013/comparison_audit.json).


### Physically grounded baseline,13:39UTC

The existing general analytical simulator was compared with velocity drift on the same90verified train-only action pairs/180branches, using complete geometry, physical BEFORE and known actually applied native Move/camera commands. First-changed16ms forecasts complete8.540s; maximum position/velocity/spin errors are1.168e-8u/7.298e-7u/s/2.078e-8rad/s. [Immediate audit](../dynamics_reports/native_paired_analytical_baseline_20261013/independent_audit.json). This is a physical predictor, not a policy training or score result.

The complete-prefix retry then evaluates all2496forecast steps, starting once and never replacing predicted physical state with future observations. Source includes132sixteen-frame branches/48eight-frame branches, hence256/128ms horizons. All90pairs/180branches complete13.285s; source/input/geometry/protected/ABI3DLL hashes and105source archive members verify. The first incomplete attempt incorrectly required16frames for older8-frame records and remains preserved.

Most prefixes closely match native motion. Maximum prefix position/velocity/spin errors are0.079711u/0.505490u/s/1.491779rad/s, at Fractured Islands flight_down8. Error first appears at frame6 contact: the predictor already flags a near-tangent edge candidate at triangles2922/2933 as ambiguous; both paired branches receive the same error. Close paired action-effect agreement therefore does not imply perfect absolute physics. [Full prefix audit](../dynamics_reports/native_paired_open_loop_r2_20261013/independent_audit.json). These short known-control training-map diagnostics support investigating corrections around the existing physical baseline; they do not validate long flights, proposed bridge-reply rollouts, multiplayer or unseen-map performance. No learned fit/PPO/promotion/Dev/test reopening occurred.


### Native contact radius correction,13:52UTC

The native contact formula multiplies float32 mRadius before subtracting double plane-distance squared. The analytical generator instead used double radius multiplication and zero clamping. Correcting the native operation order removes the Fractured error without a location-specific rule. All90pairs/180branches/2496steps complete13.163s; maximum p/v/spin errors3.378e-7u/1.349e-6u/s/2.794e-6rad/s. Both Fractured flight_down8 maximum position errors become8.882e-16u, and no audited sample worsens above1e-8. [Corrected audit](../dynamics_reports/native_paired_open_loop_r3_20261013/independent_audit.json).

Existing Concentric corner now reproduces all native ordered contacts and its measured one-step result. Existing Treasure edge continuation also matches its original24native64ms outcomes over1.536s (maxp/v/spin1.539e-9u/8.733e-9u/s/4.603e-8rad/s). Its known controls and canonical native archive are verified. Original fixtures/outcomes are preserved with additional expected targets; old assertions requiring a now-resolved ambiguity are replaced with direct native comparisons, while genuine remaining ambiguity/refusal tests remain. These outcomes strengthen the physics baseline proposal; they do not certify arbitrary longer maneuvers or a gameplay score increase.
