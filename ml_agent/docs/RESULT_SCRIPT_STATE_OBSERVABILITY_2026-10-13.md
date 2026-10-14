# Native and script mechanics observability

All six isolated Cube Isle cases pass their recording contracts:384 native16msframes,46.872total wall seconds,73post-startup observations/type, no falls, disconnects or round ends. Each case starts a fresh owned process, grants one item through the existing game setup path, establishes a descending spinning fixture, then requests one use after an initial neutral recorded frame. It is an engineering observability test, not paired causality, a skill evaluation or training data from the development map.

The expanded read-only export reports both target and local-authority script inventory, native inventory, script power IDs, lock/custom fields, script active flags and effect/Mega schedule IDs, pending status and remaining milliseconds. Both protected executables, bridge scripts, frozen A2 and sealed readers are unchanged. The old experimental executable is retained in a verified compressed archive; only the separate fixture executable was rebuilt.

| Requested item | Native held ID before use | Script held type | Actual native use / consumption frame | Observed activation |
| --- | ---: | ---: | --- | --- |
| Super Jump |1|1|2 / 2|Native effect1 active|
| Super Speed |2|2|2 / 2|Native effect2 active|
| Super Bounce |3|3|2 / 2|Native effect3 active|
| Shock Absorber |4|4|2 / 2|Native effect4 active|
| Helicopter |5|5|2 / 2|Native effect5 active|
| Mega |0|6|2 / no native inventory consumption|Script inventory clears, Mega timer scheduled, body grows|

The old84-feature motion inputs contain native held ID0..5 only. They can alias holding Mega with empty inventory. The game deliberately keeps custom power IDs>5 in script fields, setting native held ID to zero. This is a real missing-state issue for general mechanics prediction; it is not a demonstrated explanation of ordinary jump errors.

Mega's radius changes from0.189750001 to0.666599989 between native frame1AFTER and frame2BEFORE. The frozen-body16ms motion task correctly excludes or segments these script boundaries, but cannot by itself learn to predict them. A general world model needs a separate event/actuation transition task or a combined observed-state transition that includes known queued inputs, inventory, timers and body parameters. It must predict the change from its previous state, rather than receive the already enlarged body as unexplained context.

Every live observation's position, velocity, spin and radius matches its corresponding recorded native BEFORE exactly. Observations precede physics and action execution; they are not that row's AFTER. Native/observation clocks differ by0,1or2ms depending on the case, retained explicitly. The live observation already exposes the pending bridge reply and native pending Move; the newly sent reply and that queued reply are distinct control context. Target alignment must preserve this timing, rather than label a requested use as an immediate native impulse.

Independent audit verifies native16ms clocks, finite motion, native effect deadline arithmetic, script schedule invariants, archive byte/SHA checks and110repository/5native source snapshot members. All source/native/protected hashes were unchanged throughout the batch. No effect-expiry, whole-world restoration, complete hidden-state, long-rollout prediction or learned policy claim follows from these six short cases.

[Completed cases](../dynamics_reports/native_script_state_preflight_20261013/result.json), [independent audit and event boundaries](../dynamics_reports/native_script_state_preflight_20261013/independent_audit.json), [finite protocol and source fingerprints](../dynamics_reports/native_script_state_preflight_20261013/prepared.json), [general-mechanics plan](PLAN_GENERAL_MECHANICS_INTERVENTIONS_2026-10-13.md).


The observed-next-state prototype preserves378adjacent current/new-reply/next-state examples, including Mega's body boundary at source_step1. Every prototype input/target/reply/frame/source reconstructs from the original data. It is explicitly dev_engineering, raw semantic data rather than finalized neural features, and incomplete hidden state. The current implementation retains future falls as targets; the original six-case batch contains none. [Prototype](../dynamics_reports/observed_world_transition_prototype_20261013/manifest.json), [independent reconstruction](../dynamics_reports/observed_world_transition_prototype_20261013/independent_audit.json), [source-navigation inventory](../dynamics_reports/observed_world_transition_prototype_20261013/native_state_usage_inventory.json).

The interactive native-only replay passed full geometry/source/trajectory and browser checks:1950Cube triangles unchanged,6recordings, exact native coordinates, optional sphere world-radius bounds/readout/stage errors all0, no JavaScript errors.65samples/type and66forMega retain its between-frame body update at duplicate elapsed time. No forecast is invented. [Open replay](../collision_maps/viewer/native_script_responses_20261013/index.html), [browser checks](../collision_maps/viewer/native_script_responses_20261013/browser_checks.json), [all-sample source audit](../collision_maps/viewer/native_script_responses_20261013/trajectory_source_audit.json).


### Ten-map passive scene inventory, completed 13:16 UTC

All ten selected maps were loaded independently with seeds1281013..1281022. The guarded fixture command exported every native ForceObjectType and PlayerObjectType container member while the bridge waited for a reply. All ten cases pass; total wall time73.994s. Each client/server startup sample has zero native force objects and one uniquely identified owned Marble. A neutral reply resumes each case without a fall, disconnect or round end. Both native clocks and selected copied physical fields stay unchanged during export.

The independent audit verifies all source/native/geometry/protected hashes, the experimental executable, all ten scene/result hashes and identities, finite body motion/radius, and108repository/5native archived source members. [Audit](../dynamics_reports/native_scene_inventory_20261013/independent_audit.json). Bowl/Vortex were passive inventory checks only; no test motion labels were recorded or opened.

Scope is one offline startup snapshot per map. Zero native force objects at these samples does not establish absence of later scripted forces, moving geometry or other players. Player records explicitly declare partial body state. This is an observation/export contract, not a force forecast or complete world-state restoration.


### Extended bridge/native context, completed13:23UTC

Six fresh Cube cases1381013..1381018 all pass:384native frames/47.958s/73post-startup observations per case. Known use requests occur on steps1,2,3,5; actual native use occurs on frames2,3,4,6. Independent110repo/5native source archive, source/protected/executable/gzip byte hashes, all finite extended fields and all live physical BEFORE values pass.

Live bridge state has the previous reply's use latch; corresponding native BEFORE/AFTER state has the current reply's latch. Each case has four verified latch differences between these phases, despite identical live/native physical BEFORE position/velocity/spin/radius. The exporter captures yaw-hold configuration and remaining ticks, six pending reliable-update flags with active-only values, global/previous/default gravity matrices and active-only interpolation, current contacts and guarded cached best contact. Historical uninitialized last-contact fields are never decoded; runtime mass remains explicitly unavailable. Registered maxForceRadius is50 normally and75 for Mega, rather than the original84-feature omission.

Measured contact-containing phases by power: Jump0, Speed4, Bounce0, Shock101, Helicopter2, Mega0. All pending-update masks are zero and gravity interpolation stays inactive; the exporter/schema exists but those active transitions are not validated by this batch. Per-case native/observation clock offsets are0 or-1ms, so no universal timestamp-offset assumption is permitted. [Independent audit](../dynamics_reports/native_state_coverage_preflight_20261013/independent_audit.json). Source freeze has ended.


### Semantic observed-context preparation, completed13:30UTC

[Observable-context artifact](../dynamics_reports/observable_context_tokens_20261013/manifest.json) contains378adjacent current-live-state/known-new-reply inputs and next-state targets, role dev_engineering, no fit/training. Structured features distinguish script-held Mega from empty native inventory and preserve spin, body radius, material contact tokens, live use latch, queued controls, gravity frames and pending-update masks. Raw object/event/datablock handles, absolute clock values and inactive uninitialized effect deadlines are excluded. Missing runtime mass/historical contacts/external scene are explicitly unavailable.

All378examples reconstruct exactly from original current/future observations and the declared reply schedule, including the Mega boundary. Complete geometry has1950faces, with every array exactly equal to the existing full-map feature encoder; float32 operative vertex reconstruction error is at most0.000001890u.52current contact tokens are retained. Eight new semantic contracts plus existing preparation/transport checks pass42tests: future/fall targets cannot affect inputs; ID/inactive-garbage invariance; Mega/latch distinction; relative active expiry; active/inactive update masks; contact/spin retention; missing/nonfinite/unknown state refusal. [Audit](../dynamics_reports/observable_context_tokens_20261013/independent_audit.json), [feature ordering/units](../dynamics_reports/observable_context_tokens_20261013/schema_notes.json). This is a reviewed feature interface, not a complete hidden-state/world model or deployed neural architecture. Original physical84-feature readers and all prior failed fits remain unchanged.
