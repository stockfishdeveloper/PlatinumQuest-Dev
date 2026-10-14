# Native physical snapshot roundtrip

The separate `marbleblast_fixture_probe.exe` passed all nine atomic copy/perturb/restore checks on Cube Isle, seed881013, in7.58wall seconds and22post-startup native decisions. Three repeated checks at each of three contexts: supported at rest, supported rolling, and airborne with spin[0,60,30]rad/s and vertical velocity-2.64u/s. Support counts were1/1/0; ground-history values250/250/242ms. This tests general physical state, including spin and cached contact history, without a named maneuver.

Every measured native state and copied-field fingerprint matched before/after; each atomic check left its clock unchanged, and the bridge resumed normally afterward. Copied fields include full best/last/current native contact structures, ground history, position/velocity/spin, body/datablock, movement camera, native Move state, inventory/effects and telemetry. Contact structures may contain unused/uninitialized bytes: the prototype preserves their representation but does not treat padding or unused fields as semantic model features.

The hook runs only in the separately built fixture executable and refuses unsupported/nonplayable/OOB bodies, active native events/effects and a local simulation that is not frozen waiting for its reply. Existing geometry/production executables, bridge scripts and A2 are unchanged. Repository/native source snapshots and artifact hashes are retained in [the completed probe](../dynamics_reports/native_fixture_roundtrip_r2_20261013/result.json).

The first attempt is retained as incomplete. Its first native check passed, but the Python caller used `HuntEnv.control`, which consumes the debug acknowledgement while waiting for a new observation. A frozen fixture command deliberately produces no next observation. The owned attempt was aborted; the replacement sends the command directly and reads only its acknowledgement with a bounded timeout, refusing unsolicited observations/disconnects. Regression coverage verifies the pending observation stays unchanged and no reply advances physics.

The initial atomic test did not establish action-prefix reproduction or whole-world restoration; the subsequent prefix checks below establish short physical reproduction in their measured contexts. No clock or world is rewound. Active effects and world/script state remain explicit limitations. Dataset collection and the bounded isolated physical fit are described below; PPO and checkpoint promotion remain paused.

## Subsequent native prefix checks

Seed881014 reproduced supported and rolling neutral128ms prefixes exactly; spinning-flight setup was refused because native server updates remained pending after teleport. The refusal is retained. Native guard checks were preserved. Seed881015 instead reached flight with an actual ordinary jump and natural rolling spin. All three contexts passed:48native frames across six branches, exact initial physical/control state and tick phase, zero applied-Move/camera differences and zero position/velocity/spin difference throughout. These checks restore both owned physical bodies, not the game world. [Natural-flight result](../dynamics_reports/native_fixture_prefix_natural_20261013/result.json).

## Changed-action physical responses

Seed881016 completed four neutral/changed-reply pairs,8native16msframes per branch,7.52wall seconds and104 counted manual replies (105 total observations including the inventory setup reply). Copied client/server fields and first actual native BEFORE matched in every pair. The first changed native Move is frame1; the state before that Move and earlier measured motion agree. No fall, pickup, reconnect or round-end event occurred. Source/protected artifact hashes remain unchanged.

| Physical context and changed reply | Maximum measured position difference | Velocity difference | Spin difference |
| --- | ---: | ---: | ---: |
| Supported ordinary jump | 0.693056 u | 7.068000 u/s | 0 rad/s |
| Rolling reverse input | 0.008640 u | 0.080000 u/s | 1.200000 rad/s |
| Airborne ordinary jump | 0 u | 0 u/s | 0 rad/s |
| Held Super Speed use | 2.799455 u | 25.000001 u/s | 0.699205 rad/s |

These are response differences between native actions, not prediction errors. Super Speed has a recorded playable native inventory consumption; a requested use flag alone is not its success label. The airborne jump pair confirms no impulse in this observed unsupported context, despite a genuine jump Move. The result supports developing a general action-conditioned mechanics dataset, including ordinary contacts/spin/control and powers, rather than named maneuver rules. Four short contexts on one map do not establish broader state coverage, active-effect restoration, long-rollout prediction or skillful gameplay. [Full response records](../dynamics_reports/native_fixture_responses_20261013/result.json).


## Atomic placement and six-map response transfer

The guarded setup-only native placement setter passed all three repeated-prefix contexts on Cube, seed881017:48 native frames with exact initial state, tick phase, controls, position, velocity and spin. Requested pose/velocity/spin remain separate from actual action-time observations after neutral physics; for example requested vertical velocity-2u/s became-3.28u/s before the recorded branch. Contact caches are initialized empty and established by actual physics. Pending updates are refused, never cleared to force a pass. [Placement result](../dynamics_reports/native_fixture_pose_20261013/result.json).

The same four response pairs then passed on each of the six training maps, seeds981013 through981018:24/24 verified pairs, zero refusals. Native Super Speed responses vary with contact state; they are measured, not hardcoded targets. An independent audit checks declared map/seed identities, native controls/initial state, source archives and protected hashes. [Transfer results](../dynamics_reports/native_fixture_transfer_pool_20261013/result.json), [independent audit](../dynamics_reports/native_fixture_transfer_pool_20261013/independent_contract_audit.json).

## First varied ordinary-mechanics collection

All72 requested fixtures were attempted across the six training maps, with12 strata/map covering rolling, braking, ordinary jump, rising/falling flight, wall/ceiling/floor interactions and low/high spin. Each accepted pair compares neutral input with one changed legal reply, followed by neutral replies, over16 native16msframes per branch. This batch uses no powers; the prior held-Speed response proof remains separate.

There are66 verified physical pairs and136 retained branches,2155 measured transitions of which2153 are playable. Five unsupported-body refusals and one KOTM world-event pair remain explicit. Individual playable one-step records can be valid even when the pair cannot support a clean causal comparison. KOTM follows the same contract as every other map; a physics-contract failure is not a score veto.

| Training map | Verified pairs / requests | Playable transitions | Native collision steps |
| --- | ---: | ---: | ---: |
| Sprawl |12/12|384|20|
| King of the Marble |9/12|325|9|
| Duplex |12/12|384|14|
| Cragmire |11/12|352|16|
| Treasure Box |12/12|384|25|
| Fractured Islands |10/12|324|13|

Independent audit finds no input reconstruction failure, native clock gap or physical discontinuity. Actual55 jump Moves include10 with supporting steps and45 without; these are audit labels, not future contact information supplied to the model. Observed jump-action vertical velocities extend from-21.6 to+14.4u/s, and measured spin reaches369.81rad/s. All110 repository and5 native source archive members verify, along with geometry and protected artifact hashes. [Collection manifest](../datasets/marble_dynamics/native_inprocess_interventions_20261013/manifest.json), [independent audit](../datasets/marble_dynamics/native_inprocess_interventions_20261013/independent_audit.json).

A metadata-only reader index supplies the existing reader's required family field and points to canonical compressed recordings. Original manifests and sealed readers are untouched. Combining with previous recordings produces90,539 examples and38,439 full source faces across six training/two development maps. Every development input, target and geometry array exactly matches the previous preparation; Bowl/Vortex labels stay closed. Seven generated reader source copies were compressed, SHA verified and removed.

A bounded supervised physical fit completed5000updates in199.831s with the previous contact-context architecture, seed121014,5000updates/batch32/300s and unchanged development selection. Only data changed. It is not PPO, a deployment or a policy promotion; airborne jump prediction and subsequent open-loop/native validation remain required. The review-only prepared protocol is retained separately because the fitter deliberately refuses output-directory reuse. [Development preservation audit](../dynamics_reports/native_motion_student_interventions_20261013/development_preservation_audit.json), [running fit protocol](../dynamics_reports/native_motion_interventions_run_20261013/protocol.json).

The data-only fit fails the ordinary-motion preservation gate at every learned checkpoint, selecting update0 unchanged velocity drift. Initial development metrics match exactly. New causal recordings are useful infrastructure, but this fit does not establish improved physical prediction. The final learned parameters were not promoted or substituted for the selected baseline. [Completed fit](../dynamics_reports/native_motion_interventions_run_20261013/result.json), [scheduled diagnostics](../dynamics_reports/native_motion_interventions_run_20261013/history.json), [comparison audit](../dynamics_reports/native_motion_interventions_run_20261013/comparison_audit.json).
