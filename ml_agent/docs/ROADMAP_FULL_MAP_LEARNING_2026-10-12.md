# From complete geometry to higher Hunt scores

The end goal is more actual game points per complete round across the selected
ten maps, with every map reported individually. A modest King of the Marble
dip is not a veto on pool-wide progress; its prior specialized training makes
it an unusually strong baseline on that one map. Accurate
motion predictions and successful short maneuvers are intermediate evidence.
The new system has not yet demonstrated full-round superiority or trained a
replacement policy. A2 remains frozen.

## How the pieces connect

Actual marble state + complete contact geometry + live gem/resource locations
feed a controller. Its global route/gem layer chooses destinations; general
control-sequence search predicts rolling, jumps, spins, collisions and power
effects, chooses a useful continuation, executes a short prefix and observes
the game again. Native pickups, material/body changes and respawns invalidate
stale plans. Replanning repeats until the round ends. The game's score decides
whether that whole system is better.

The existing analytical physics prototype already demonstrates this loop over
bounded stretches. The separate experimental motion student is a possible way to
make searching many futures cheaper. Completing continuous analytical planning
and fitting the physical student are parallel paths: we can test controller
score gains before PPO, and a student is optional if exact physics becomes fast
enough by other means. A trained actor can later learn from the
map/planning information and useful trajectories, reducing search overhead.
The navigator must receive the chosen maneuver/landing destination and relevant
map or forecast context beyond its old local crop. Demonstrations alone cannot
teach it to distinguish locally identical ramps with different distant landings
if its inputs still hide that difference. There is no requirement to hand-author
a rule for each named trick. Nor should
a slow or unreliable student replace a working analytical calculation merely
because it is learned.

## Concrete milestones and pass conditions

1. **Establish the full-round reference.** Run frozen A2 from predeclared fresh
   map/seed starts under one explicit rules/timing profile. Use native end-message
   scores, original round durations, normal falls and the existing assisted
   eight-second rescue, charged in game time. Record every completed round,
   failure, fall, rescue, use and wall-time. Missing or interrupted runs invalidate
   the declared sweep; never substitute first-fall/19-second prefixes for full
   scores. Prepare three seeds per map for initial screening, then a larger
   independently declared replication before promotion. Three rounds are not
   strong statistical proof. The legacy64ms reference and an experimental16ms
   physics comparison require separate matched A2 runs at the same decision rate.
2. **Earn a useful physical predictor.** The prepared111,083 native16ms
   transitions and all47,932 collision triangles have supported three bounded
   supervised experiments. None yet passes useful ordinary-motion checks;
   activation improvements must not hide incorrect airborne jumps. Continue
   development-only experiments with predeclared sampling/selection and bounded
   limits. Test maps never select a checkpoint. Report jumps, each
   power and ordinary motion separately; compare to constant-velocity and the
   analytical predictor. No game reward or use bonus enters this task.
3. **Make predictions work over whole maneuvers.** Finish camera/contact/effect/
   inventory continuation and independently forecast0.5/1/3/6-second trajectories
   from one initial state. Measure landings, rebounds, spin, impacts and fall
   cases. Never feed later observed pose or contact into a supposed open-loop
   test. Today's64u/15u sensitive contact failures stay in the evaluation. A
   good16ms error average alone does not pass this milestone.
4. **Make continuous planning practical.** Use validated cheap predictions to
   rank general sequences, with analytical checks where useful. Reuse geometry
   encodings/plans and measure complete decision latency, including worst tails.
   Handle native OOB/respawn, eight-second rescue, stale resources and subsequent
   gem groups; always start the next forecast from newly observed state. If A2
   is used when search times out or has no checked candidate, label and count
   that fallback as part of a hybrid controller. Test actual full rounds under
   matched settings before claiming a gain. Lockstep gives extra wall time in
   development; that is not proof of real-time play or economical PPO throughput.
5. **Train and compare the gem-collecting policy.** After integration review,
   useful native planner trajectories can supply demonstrations, followed by
   bounded PPO blocks on all ten maps. The planner can also remain part of the
   deployed controller if its speed permits. Evaluate checkpoints using native
   game points, not shaped training reward or predictor MSE. Any progress shaping
   must have explicitly recorded discount/target-change/terminal semantics.
   Preserve A2, compare geometry/physics additions by ablation, and promote only
   after repeatable equal-map score gains with every individual-map result
   visible. Investigate regressions across the pool without giving KOTM a
   special acceptance gate.

## Current preparation

- Expanded native dataset, immutable physical inputs, full-map features and GPU
  forward/provenance checks are complete. Rich failure-gallery browser/readout
  checks pass. Three separate bounded motion fits are archived; none is deployed.
- [Bounded fit protocol](../dynamics_reports/native_motion_fit_review_r2_20261012/protocol.json)
  is reviewable and has not been executed. Physics fitting and PPO are different
  approvals/experiments; the oldOct9 training config is not a new authorization.
- `nav.marble_dynamics.full_round_benchmark` prepares a sealed legacy-cadence A2
  reference using the existing full-round evaluator. Its default only writes a
  protocol. `--run-a2` runs read-only evaluations and validates complete clocks,
  native scores and identical rules. It does not run either trainer.
- `real_run.py` now archives start score/time, expected round duration,
  evaluation settings and actual starting/first-play world plus recurrent state.
  This adds provenance; it changes no controller actions. Same explicit seed
  produced different KOTM native outcomes111/124 in separate smoke launches;
  neither is discarded or called an exactly matched world.
- Complete-map frozen inference cache matches uncached outputs bitwise for all
  ten maps at batches1/16/64 and rejects stale weights/changed bounds. It retains
  all surfaces. Random-network timing/equality does not establish useful learned
  physics, complete-rollout correctness or actual planner speed.
- Continuous full-round handling now works with counted A2 fallback and audited
  native inputs. A ten-map comparison is active; practical search speed remains
  unfinished. Existing first-fall galleries remain diagnostic and unchanged.

Full-round score-contract smoke and the ten-map reference pass. The preliminary
single-seed average is17.2 versus historical same-weight averages32.5/33.1 under
older evaluation conditions. Those values must remain separate; do not lower a
promotion target based on the new low seed. Historical Treasure already varied
8->60 between two rounds. Historical-seed Fractured recheck is complete:83/88, near earlier91 and far
above new-seed19. This supports substantial start variability contributing to
the difference; it does not prove every runtime change is harmless. Immediate engineering priorities: resolve
reference comparability and finish the new controller lifecycle/timing adapter.
A separate three-seed30-round proposal is sealed but unstarted. The first learned-motion fit is a small diagnostic learning experiment,
not a long PPO run or a promise of higher gem scores.

See [daily progress](PROGRESS_2026-10-12.md),
[physics review](REVIEW_PHYSICS_STUDENT_2026-10-12.md) and
[incremental handoff](HANDOFF_PLANNING_RELIABILITY_2026-10-12.md) for actual evidence.


## Latest authorized execution and cleanup

Operator approved proceeding after the recommendation. The first isolated
1,000-update physical fit has now run, with its predeclared development selection
and test evaluation. It **does not pass** the useful-predictor gate: both test-map
mean position/velocity/spin errors exceed the constant-velocity baseline.
`native_motion_fit_20261012/result.json` preserves that outcome. There is no
student integration, learned autoregressive claim, PPO or promotion.

A30-round original64ms A2 replication and separate natural-full-round
experimental16ms continuous hybrid/A2 smoke are in progress. The full-round
adapter cancels old plans/input/resource history on native transitions and
counts every search timeout and A2 fallback. A1-second search allowance may
leave few complete searches; that is measured, not called realtime planning.

Source-copy cleanup packed4,523 generated Python copies before removing loose
files; total Python count4,994->476. Per-run archives include member hashes and
original paths. Active/tracked code, historical benchmark imports, all native
records/arrays, checkpoint bytes and unfavorable results remain available.
Temporary helpers stay outside the repo; completed experiments use the shared
`nav.source_archive` writer/packer, with new main experiment writers already
switched to direct archives. Details/results are in the incremental handoff.


### Operator correction: pool-wide progress, no KOTM veto

Operator clarified that KOTM's unusually specific earlier training should not
make it a pedestal. A modest KOTM score dip is explicitly NOT grounds to reject
progress. Evaluate equal-map pool results, consistency, newly useful mechanics,
falls/rescues and cost; keep per-map outcomes visible without a KOTM-flat gate.
The current124vs133 KOTM observation does not invalidate the131vs32 Sprawl gain.
Neither two-map observation alone demonstrates all-pool improvement.

Continuous search ordering variant prepared: check the existing frozen learned
proposal first when explicitly requested, so the1s budget does not starve it
behind every hand-built control. Forecasts/physical acceptance checks remain
identical; the frozen actor only proposes, not bypasses verification. This is a
general search-budget efficiency experiment, not a KOTM-specific remedy.
Earlier completed-candidate trials/ordering stay reproducible in source ZIPs.


### Pool assessment before the remaining results arrive

The operator's KOTM correction applies to interpretation, not just one gate:
an absolute-point mean can still be dominated by that specifically trained
map. Report the ten-map native point mean together with every per-map delta,
win/tie/loss counts and the median per-map delta. Also show a predeclared nine-
other-map diagnostic alongside KOTM's individual result, never as a replacement
for the complete ten-map table or by dropping other unfavorable maps. Consider
reliability, useful mechanics, rescues/falls and measured cost. A modest KOTM dip
alone does not reject engineering progress, demonstrations or a candidate.
Repeated scores are still needed before a superiority/promotion claim. No
new metric will hide zero rounds or invalid input/score audits.


### Predeclared follow-up: separate route guidance from physical search

Cragmire currently uses few checked plans and mostly frozen A2 fallback. The
hybrid fallback follows the geometry route, while standalone A2 uses its legacy
terrain route. Therefore score gains cannot all be attributed to predicting
physical maneuvers; global destination selection and improved routing also
contribute. This is the intended composite controller, not a clean physics-only
ablation. After the sealed sweep, add a clearly labelled geometry-routed frozen
actor with physical search disabled and otherwise matched rules/goals/timing.
Compare complete rounds and actual power/Blast use alongside cost. Use a
predeclared map/seed pool, keep every outcome and do not privilege KOTM. This
can reveal cheaper useful guidance before committing to policy training. No
third controller or runtime source was changed during current r4 play.


### 2026-10-13 continuation: full pool and component comparison

The r4 natural full-round pool is now complete:77.2assisted versus30.4frozenA2 points, all20native score/input auditsPASS, zero implicit OOB keys;10/0/0wins/ties/losses and median map gain44. One seed/independent launches, experimental16/64ms; not a newly trained policy or generalization proof. Costs/weaknesses preserved:66/259falls,20/36acceptedrescues,3289.23/587.94wall seconds, Sprawl6accepted/10attempted rescues. Geometry review is complete for all20recordings and allten maps, with native respawn gaps disconnected and missing fallback goals unknown.

Component comparison is implemented,20unit checks and native no-search preflightPASS, now running all30rounds/all10maps/seed681013 under source freeze. Cube geometry54/legacy59/combined58, Bowl17/19/66. Do not aggregate a partial pool or reject over a single map. Geometry_navigator changes route guidance only: the existing A2 observation builder still uses legacy terrain/local observations. It does not suddenly give the old actor every collision triangle. The full-map analytical teacher queries the reviewed complete mesh; the separate physical student receives every face but its three fits have failed useful prediction checks. Neither student nor a new map-aware actor is deployed.

Next physical experiment is declared in PROTOCOL_CONTACT_CONTEXT_STUDENT_2026-10-13.md: exact BEFORE proximity, multiple contexts, ordinary-motion-preserving development selection, same data roles/testclosed and bounded fit. Implement after the current source freeze, not during wall-budgeted native comparisons. General mechanics remain the goal. Long policy training and checkpoint promotion remain paused. [Results](RESULT_CONTINUOUS_POOL_2026-10-13.md), [component protocol](PROTOCOL_GEOMETRY_COMPONENT_EVALUATION_2026-10-13.md), [ten-map replay review](../collision_maps/viewer/continuous_full_pool_r4_20261013/index.html).


### Current status, 2026-10-13
The component comparison finished all30 natural rounds: combined71.2, geometry-only34.8, frozenA227.0 equal-map points, allinput/score audits PASS. The geometry-only actor still receives its original local observation; whole-map knowledge is currently used by the route/planning teacher and the isolated motion student. Combined gains include destination/resource handling as well as physics. All30 native recordings are available in the component gallery.
The contact-context fit finished5000updates, dev-selected4600 passing the ordinary5%preservation gate and41.3%balanced-loss improvement. It still hallucinates airborne jump impulses (5.19/7.15u/s devvelocityerror) and has large activation tails; no neural rollout or actor deployment. It needs causal varied-state mechanics data and useful prediction gates, rather than another fit selected solely by aggregate loss.
Checked stopping/landing backups now have a two-map native engineering preflight (61/85points,0/0falls) and corrected real-task progress measurement; revised preflight is active. Next declared fresh20round alltenseed681014 comparison isolates these backups against the same combined controller. Full-pool native points, reliability and wall cost remain the engineering measures, with KOTM one equal-weight map. Long PPO/checkpoint promotion remain paused.

### Teaching-data requirement from checked recovery

The current Cube comparison shows why physical validity and task usefulness must be evaluated separately: checked stopping can match native motion and reduce falls while collecting fewer gems. Future actor teaching data must retain real goals, observed actions, actual returns/pickups, time spent in backups and failures. Do not label every checked control as a desirable imitation target. Keep motion supervision (what the action physically does) distinct from action-value/policy supervision (when it helps collect gems); neither needs a hand-coded catalog of tricks. The full-map geometry is shared context, and a useful physical model still needs multi-step state evolution and refusal checks before it can select plans. Long policy training/promotion remain paused for review.


### Current continuation, 2026-10-13 13:16 UTC

The revised recovery preflight is complete, Cube47/Bowl9 with0falls, but stopping consumes substantial time without pickups; the larger pool remains incomplete and provides no improvement claim. The offline actor8 extension qualifies no new candidate and stays unintegrated. Varied native fixtures now provide66verified paired starts/2153playable transitions; all development tensors are unchanged. Two new bounded physical fits, data-only and paired-loss, fail ordinary preservation and retain only update0. Further fits should wait for an improved state/geometry interaction contract rather than repeat the same architecture blindly.

Read-only native/script instrumentation verifies all six powers, including script-only Mega inventory and growth between native frames. The378transition current-observation/new-reply-to-next-observation prototype retains those changes as targets and explicitly remains unnormalized engineering data. A six-case interactive native response gallery displays every Cube collision face and exact observed body radius. Ten-map passive force/player startup inventory passes without modifying clocks or physical fields. Next preparation is to close known state omissions, define semantic bounded features and verify queue/phase and dynamic-state transitions before another useful-prediction experiment. This includes ordinary rebound, spin and edge contacts; powers are one part of general mechanics. Long PPO/promotion stay paused. KOTM remains one equal-weight map and modest dips there never veto progress.
