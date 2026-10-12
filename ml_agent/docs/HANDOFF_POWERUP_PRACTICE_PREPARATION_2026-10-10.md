# Reconnect repairs and powerup practice discussion, 2026-10-10

**Later authorization and active loop:** the operator subsequently authorized driving
the practice/training loop. See [HANDOFF_POWERUP_LOOP_2026-10-10.md](HANDOFF_POWERUP_LOOP_2026-10-10.md)
for implementation, finite launches, evaluations and current status. The statements
below describe the earlier preparation-only turn.

The operator authorized proceeding with the audit's plan while discussing practice
methods and goals. Completed the independent infrastructure/reward work below.
No live training or checkpoint optimization was started. Practice curriculum and
its first powerup remain a discussion proposal; no practice workers were enabled.

## Completed changes

* `nav/vec_worker.py`: handle round/reconnect boundaries immediately after the game
  step, before interpreting inventory changes or recording against an old map.
  Approach recording requires current world/catalog telemetry, rejects reconnect
  state and backwards clocks, and continues recording valid starts. Duration
  accounting falls back safely when replay telemetry disappears. Rescue controls
  that cross a boundary also finish before old-map logic. Terminal points/shaping
  are counted once, with the actual transition duration.
* `nav/env.py`: `round_started()` treats missing round metadata as unknown, not as
  a finished countdown. This removes the reconnect `round_ms` lookup failure.
* `supervise_points_training.ps1`: archive the current attempt's stdout/stderr
  after stopping it, including startup-failure/final exits. Do not copy the prior
  run's shared streams into a new run directory at launch. Preserve existing
  stdout paths used by the dashboard. Archive only attempts whose trainer started.
* `nav/recovery.py`: separate the overnight `legacy_progress_term` from the exact
  duration-discounted `potential_term`. The exact term includes goal changes and
  settles at true terminals. Add explicit validated shaping configuration.
* `nav/vec_worker.py`: `NAV_SHAPING_MODE=discounted` uses PPO's actual gamma,
  preserves the potential ledger through target changes, and refreshes the route
  field before measuring a new target. Existing `legacy` mode reproduces the
  overnight target-skipping heuristic. Default remains legacy for comparison;
  the newly prepared review configuration explicitly selects discounted.
* `nav/train_nav.py`: save the complete reward configuration with checkpoints and
  log it on startup. Changing/unknown shaping metadata resets the critic and Adam
  and starts critic-only warm-up; actor weights are preserved. Old points-only
  checkpoints can still resume points-only without a metadata-induced reset.
  This task ran no live warm-up or optimization. Existing synthetic unit tests
  exercised critic migration in temporary files only.
* `prepare_real_maps.ps1`: expose `-ShapingMode legacy|discounted` and A2/B2 source
  selection. Record the requested drill plan/path in preparation metadata and the
  actual TEMP rotation directory. No actual training was invoked.
* `nav/preflight_r.py`: optional separate output path and finite repeated reconnect
  checks. A rescue transition's shaped reward may be negative or positive; verify
  finite reward rather than presuming it must be nonnegative.
* Tests: replace incorrect undiscounted invariance assertions with complete
  discounted trajectory checks; add missing-state, early-boundary, target-switch,
  terminal accounting, duration fallback, and reward-migration regressions. Update
  the old preparation test to the operator's open-use rule.

The exact shaping tests demonstrate cancellation of delayed approach/reversal,
variable-duration flight/respawn, idle trajectories, and target switches when
terminal accounting is included. They also verify a truncated trajectory retains
its final potential. This establishes the tested ledger's accounting identity,
not finite PPO performance or the quality of its navigation heuristic. Floor
history remains frozen far from floor; flight-appropriate progress is still a
separate curriculum question.

## Verification and final state

* 59 tests passed in 3.751 s:
  `python -m unittest nav.test_potential nav.test_phase_r nav.test_replay nav.test_ss_routes nav.test_ss_floor_model -q`.
* Both edited PowerShell scripts parsed successfully. Preparation-only command
  executed with A2, a fresh review tag, and discounted shaping. It created STOP
  and metadata, with `training_started=false`; no training seed was copied.
* Actual hidden engine probe with discounted shaping: 22 countdown checks,
  under-deck distance 0 versus upper-deck 34.6569, rescue after 8.192 play seconds,
  cooldown and competitive refusals, normal SJ/blast operation, then three map
  reconnects (Sprawl, Cragmire, Sprawl) with fresh telemetry/rules and skipped
  boundary transitions. A recorded Bowl start restored with strict pose,
  inventory/items, clock, RNG, and score checks. It is an unpaced diagnostic at
  engine speed 1, not the operator's wall-clock-paced gameplay review.
* Persistent probe artifact:
  [REAL_MAP_REPAIRS_PREFLIGHT_2026-10-10.json](REAL_MAP_REPAIRS_PREFLIGHT_2026-10-10.json).
* This complements the prior audit's successful restoration of all 14 Sprawl
  development starts. It does not establish stable multi-hour operation or replay
  restoration across every mission/effect.

Prepared review directory: `logs/nav/real_maps_20261010_review`, source A2/37760,
discounted shaping k=.05, drills none, 120-minute limit for a future authorized
launch. Both overnight STOP files remain. The existing config's approval boolean
is true from overnight authorization, but this task did not use `-StartTraining`.
Final checks reconfirmed the source A2/B2 hashes match the independent audit.
No trainer, worker, or game remains running. The new review STOP file is present;
its output directory contains no `starting_checkpoint.pth` or trained weights.

Safe preparation command from `ml_agent`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\prepare_real_maps.ps1 -Checkpoint A2 -RunTag real_maps_20261010_review -ShapingMode discounted
```

## Proposed practice methods and goals — discussion draft

Teach three distinct abilities: control after activation, judgment about whether
and when to activate, and profitable use inside ordinary rounds. Do not reward an
activation by itself or treat a forced activation as a learned decision.

**Stage 1: handling.** Restore a real recorded pre-use state with normally
collected inventory. During environment reset, activate that existing powerup via
the ordinary game action; then start the learning window and let the policy steer
the complete maneuver. The reset activation is excluded from PPO's actor action
records. This creates real active physics without fabricating an unsupported
active-flight snapshot. It teaches steering/landing, not the use decision, and
must be reported that way. Begin with easy nearby gem/landing opportunities, then
increase distance, approach variation, and landing difficulty after success.

**Stage 2: spending judgment.** Restore similar states with the powerup held and
let the policy decide activation as well as movement. Include useful-use and
save-it situations, plus failed attempts; do not collect only successes. Restore
the same named states for paired evaluation of normal use permission, use withheld,
and a diagnostic activation-now reference. References are evaluation treatments,
not off-policy PPO samples. Evaluate actual points, time to useful outcomes,
falls, and activation timing rather than assuming any progress proves causation.

**Stage 3: transfer.** Use initially one of eight rollout workers for practice,
keeping seven on normal rounds; expand to two only if necessary. With equal
per-worker rollout sizes, this makes practice 12.5%, then at most 25%, of samples.
Keep the same shared policy and existing action head. Fade practice as learned
skills transfer to ordinary rounds. Normal rounds teach pickup/search, long-term
resource value, target selection, and when not to take the shortcut.

| Resource | First useful skill | What earns a good result |
|---|---|---|
| Blast / Super Jump | Leave a low ledge or trap and control the ascent/descent | Reach the useful surface and collect sooner than the no-use reference |
| Super Speed | Align, activate, then steer/brake through the maneuver | Faster pickups or a clean sequence without extra falls |
| Helicopter | Take off, steer toward a landing, and finish the descent | Reach/land on the intended surface, then collect; sustained flight alone is insufficient |

**Feedback.** Use nearby achievable gem pickups initially so successful handling
produces frequent game reward. Stable landing, landing error, escape completion,
and crashes are diagnostic measurements, not automatically new reward terms.
Correct discounted shaping can be evaluated separately. The current ground-route
potential is unsuitable as continuous long-flight guidance; freezing it does not
solve that problem. Before long Helicopter flights, choose a flight-appropriate
bounded potential and verify its behavior, or use a curriculum of short successful
flights that grows in length. Do not present progress-only flights ending in falls
as successful powerup use.

**Data.** Use normally reachable pre-use states and multiple approaches, speeds,
maps, landing heights, and useful/no-use cases. The current 165 development starts
are 162 blast and 3 SJ, so they can validate mechanics but cannot constitute the
Helicopter or Super Speed curriculum. New pre-use collections are needed for
those resources. Keep source rounds together in dev/evaluation splits. Preserve
fixture identity, strict restoration checks, and enough round time for the full
window. The 32-observation recurrent warm-up remains approximate and must be
checked before policy comparisons. Do not silently replace failed fixtures.

**Evaluation.** Report handling success separately from learned use. Measure
successful landings, pickup/time advantage, falls, resource wasted/lost, and actual
activation timing/probabilities. Evaluate both deterministic actions and learned
stochastic actions without the exploration floor so a sub-0.5 learned preference
is not mistaken for no learning. That reporting/control option is proposed for
the pilot; it was not implemented here. Compare matched update counts/map exposure,
full-round scores and KOTM retention, then competitive/no-rescue behavior. Keep
forced stall jumps separate from learned behavior in reports.

The operator was asked which first skill to emphasize (blast/SJ, Helicopter, or
Super Speed). With no response yet, blast/SJ is a starting proposal to validate
mechanics, not a committed curriculum or authorization to launch practice training.
Curriculum duration, flight feedback, progression criteria, and mixture are still
discussion choices.
