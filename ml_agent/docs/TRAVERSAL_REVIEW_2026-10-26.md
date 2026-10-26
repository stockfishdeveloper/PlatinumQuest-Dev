# Traversal review, 2026-10-26

The three proposed fixes address real limitations. The larger opportunity is to
plan the approach, resource pickup, launch, landing and gem collection together,
preserving velocity between them. The current mechanics are much stronger than
the decisions that connect those mechanics.

This review adds a diagnostic tool, not a production behavior change. No new
full-round points improvement was established during this review.

## Evidence checked

- Recent commits through 95766d0a8 and the existing uncommitted launch work.
- HANDOFF_NAV_TRAINING_LOG.md, including failed alternatives and all three launch
  pairs: 7969 -> 8149 points across 60 rounds per arm, +3.0 points per round.
- Current search, resource acquisition, crossings, native controller, parallel
  evaluation, real-time scheduling and offline benchmark implementations.
- Fresh points ledger over launch1/2/3 candidate arms, 60 rounds on ten maps,
  compared with demo_20261024_224609.npz on TreasureBox.
- Fresh serial/worker evaluator audit on eight recorded native states and an
  additional held-Super-Speed state with the no-jump prototype enabled.
- native_controller_check.py --n 120 Sprawl_Hunt TreasureBox_Hunt: every reported
  comparison passed, including 120/120 sequence actions on each map. This checks
  low-level controller agreement; it does not establish whole-search agreement.

## New concrete findings

### Serial search and live worker evaluation have drifted apart

search.search has newer candidate-extension rules that
parallel_search.evaluate_candidate does not share:

- Extend airborne edge-hit candidates even before they predict a pickup.
- Extend certain wall-climb candidates whose terminal support check passes while
  their vertical motion still indicates flight.
- Exclude edge-hit plans from ordinary ballistic projection, reserving their
  intended landing continuation instead.
- Charge extended wall/time-priced plans for additional forecast duration.

The new tools/search_path_audit.py runs the actual worker evaluator inline, with
identical recorded inputs and no deadline. This removes CPU scheduling and IPC
as explanations. It omits the navigator proposal in both paths to isolate
ordinary candidate evaluation.

Four of eight sampled states had different candidate sets. On TreasureBox,
launch1_candidate seed 7347019 decision 3712, serial search chose
edge_hit_1_route_jump_15_landing; worker evaluation lacked that continuation and
chose steering_0.5. At that state ten serial candidates were missing from the
worker set and four additional ballistic candidates appeared in the worker set.
Seven of eight selected labels agreed. These were deliberately selected
edge-hit/held-powerup states, not a random sample of play.

Artifacts: dynamics_reports/planner_review_20261026/search_path_audit.json and
roll_family_audit.json. The audit records source hashes. The tool tests evaluator
logic, not multiprocessing serialization, real-time scheduling or round scores.

Recommendation: put the candidate forecast/extension/assessment rules in one
function used by both paths. Choose the intended rules explicitly, then compare
candidate labels, rejection reasons, costs, pickups and selected actions on
recorded states. Follow with real-game score pairs; restoring agreement alone
does not prove either current behavior scores better.

### The no-jump prototype repeats its entire timing grid eight times

The POW_LAUNCH_ROLL block is inside the outer POW_LAUNCH_AT loop and contains a
second complete POW_LAUNCH_AT loop. With the current eight timings it produces
64 no-jump candidates for eight unique schedules. The held-Super-Speed audit
at Sprawl decision 380 confirms each use_launch_roll_* label occurs eight times.

Deduplicate before testing the feature. This is currently an opt-in prototype,
so it does not explain the defaults' existing live deficit. It does contaminate
the cost of prototype experiments, especially ones using Python landing control
and the offline virtual planning clock.

### High speed and useful preparation conflict with current acceptance rules

search.terminal_support requires all four tail states to have total speed below
18 u/s, including motion supported by a flat floor. Supported high-speed motion
and unsettled flight therefore share the rejection no_supported_continuation.
Extending forecasts until this test passes can require a successful launch to
slow down before it becomes acceptable.

search.useful_candidate also requires either a pickup or more than 0.3 u of
route-distance progress at the endpoint. A necessary run-up, a moving turn, or
a preparation maneuver can be useful without satisfying that endpoint test.
Simply allowing all non-progress plans would invite stalls. Instead, assess a
completed maneuver and its continuation to the real task, with a bounded time
to the next event and a verified escape/continuation.

## Assessment of the three proposed changes

1. **Keep speed through powerups:** good, but arrival speed should depend on the
   next maneuver. A fast pickup facing away from a narrow ramp can be worse than
   a slower aligned pickup. Compare pickup position, velocity, heading and spin,
   then simulate the launch from that actual state. Today RESOURCE_BY_TIME uses
   hypothetical inventory at the CURRENT pose to estimate an item's usefulness;
   it does not prove the same maneuver works after fetching that particular item.

2. **Land short and roll in:** good as an additional proposal family. Keep options
   to land on the gem, short of it, and on other reachable support patches. The
   current formula at 29 u/s, arrival speed 11 and deceleration 7 requests about
   51 u of setback, versus the observed human landing about 22 u short. This is
   a search seed, not a general physical optimum. Sample a small set of offsets
   and arrival speeds; require actual support and simulate through collection,
   ideally also the next gem. Reuse landing.py's support-surface sampling where
   practical. Different slopes, contact losses and exit turns need different
   arrival states.

3. **Follow launches through landing:** necessary, but extend the SAME feedback
   controller, including launch-specific air steering, then evaluate its roll-in.
   Current LandingContinuation replaces the original controller with ordinary
   SequenceProposal routing; ballistic.project_landing uses no-input controls.
   Those are useful alternatives, but neither is the original aimed launch's
   feedback policy. Use bounded extensions in chunks through the landing event
   and onward to collection or a checked continuation. Reuse forecast prefixes
   rather than replaying each prefix for every extension.

The claim that #3 closes most of the remaining human gap is unproven. Existing
scratchpad results already show that simply raising POW_LAUNCH_FLIGHT to 60 does
not solve the opening: 8.192 s remains 8.192 s without the other changes. The
short-landing variant gets 7.296 s; adding the no-jump prototype at that horizon
gets 11.648 s. These are diagnostics using reconstructed demo starts, not exact
native replay or full-round evidence. Candidate duplication, changed plan
selection and controller/horizon handling need separating before drawing a
conclusion from that regression.

## Where the current time goes

Fresh ledger results, six launch-enabled rounds per map:

| Map | Mean points | Approach seconds per round | Seconds per in-group gem |
| --- | ---: | ---: | ---: |
| Bowl | 134 | 216.9 | 2.14 |
| Concentric | 112 | 140.8 | 1.68 |
| Cragmire | 119 | 152.1 | 1.12 |
| CubeIsle | 141 | 99.6 | 1.10 |
| Duplex | 130 | 151.0 | 1.03 |
| FracturedIslands | 120 | 109.2 | 0.80 |
| KingOfTheMarble | 149 | 6.8 | 1.49 |
| Sprawl | 178 | 183.2 | 1.03 |
| TreasureBox | 174 | 136.4 | 0.92 |
| VortexEffect | 100 | 200.3 | 1.39 |

Mean approach time is about 139.6 s per round; mean total recorded duration is
about 247 s including countdown. The ledger's groups are inferred from gaps of
more than 47 decisions between pickups. A slow pickup inside a real group can
therefore be classified as an approach; these are prioritization estimates,
not exact spawn identities. Its roughly 4.2 s labelled respawn per round also
includes countdown/non-playable states. Do not interpret that as fall time.

TreasureBox human demo: 217 points, 5.28 s per inferred approach and 0.73 s per
in-group gem, against 174, 7.84 and 0.92 in the six planner rounds. Human pickup
speed is LOWER, 5.0 versus 5.6 u/s. The comparison is not a matched-layout pair,
but it reinforces the need for quick approaches and precise group collection,
rather than higher speed everywhere.

Vortex and Cragmire show roughly 14 s per round classified as approach stalls.
Some long unproductive excursions remain moving and do not count as stalls.
Identify groups by actual item-set changes in the next ledger revision, and
report time without a pickup and failed maneuver attempts separately from speed.

## Recommended implementation order

### 1. Repair evaluation agreement and prototype duplication

This makes the next experiments interpretable. Audit serial native, per-reply
native and real-time worker paths separately; identical physics does not imply
identical search. Preserve each actual state's pending input pipeline, timers,
inventory and spin when comparing live and offline behavior.

### 2. Search pickup -> launch -> landing -> gem as one maneuver

Use the current candidate families as seeds. For the best few resource-approach
plans, branch from the actual predicted pickup state, apply the item's observed
pickup event at the correct time, and search launch timing/yaw plus landing and
roll-in. Compare against driving directly to the gem.

Keep several pickup arrival states rather than just the quickest item pickup:
a small Pareto set over time, speed, heading and contact state. The forecast
currently has no general automatic mid-trajectory item-acquisition step; this
needs an explicit event model preserving the bridge's delayed inputs. Never
assume hypothetical possession before the marble actually crosses the item.

Start narrow: only nearby on-route Super Speeds and long trips. This addresses
the Sprawl failure without introducing a full global route solver at once.

### 3. Event-based continuation and speed-aware viability

Separate airborne, supported-but-fast and settled terminal states. For a fast
supported state, test the ability to keep following the route or brake on the
available surface; speed alone should not decide validity. Keep physical checks
for missing support, collisions, bounds and required future controls.

Commit a selected ramp/gap maneuver through its imminent timed events while
revalidating from observations. Do not restart its launch countdown every frame.
When repeated forecasts fail to reach the next maneuver event, change staging
speed, lane or route. Previous give-up-to-direct-pursuit and roll-up changes did
not establish score gains, so do not just re-enable them.

### 4. A small beam over physical maneuvers

The larger architecture change worth pursuing is bounded search over several
events, retaining 4-8 diverse states per event as an initial experiment. Branches
use existing rolling, jump, Blast, edge-hit and wall-hit controllers; each child
is simulated, not selected by a named map rule. Preserve different headings,
speeds, floors and remaining resources even when positions are similar.

Search at group changes or meaningful route choices; keep feedback control
between events. Refine timing and yaw around successful trajectories instead
of adding the full Cartesian product of all timings, directions and landings.
Batch the expensive forecasts in native code if profiling warrants it.

This differs from the already failed open-loop chain variants and bounded local
refinement: it carries verified intermediate states across pickup/launch/landing
boundaries and re-plans their execution. No assumption that beam search will win
without a score test.

### 5. Improve the objective with measured continuations

The existing -100 pickup credit, distance terms and cost overrides are rough
proxies for round throughput. The two-level cluster evaluator is a useful start,
but assigns a fixed horizon-based substitute when its follow-up cannot collect
the next gem. Prefer a few actual rollouts through the next one or two pickups,
and ultimately group clear, before introducing another global time-cost formula.

Both TIME_COST and time-price experiments have already failed score or sim tests.
Do not repeat them under a new name. If a terminal value model becomes necessary,
fit remaining collection time from measured continuations using geometry,
velocity, spin, contact mode and resource state. First audit known fast human
trajectories against the current proposal set and gates: missing proposals and
bad scoring are different problems and need different fixes.

## Experiment contract

Use exact saved native starts for mechanical diagnosis. Add a human-trajectory
inclusion audit: can the planner express the maneuver, does it pass the checks,
and does its score beat the slower alternative? Check first-gem time, whole-group
clear and the next approach; a fast first pickup can leave a costly exit state.

Screen all ten maps, including ordinary, held-powerup, raised, gap, edge and
previously stalled states. Prefer recoverable episodes or fixed-time points
tests when possible: terminating every episode at a fall can reverse the ranking
of a strategy that recovers and scores well in a complete round.

Promotion is based on actual full-round points, using simultaneous matched-load
4+4 pools and multiple independent seed pairs, then real-time 1x confirmation.
Report the equal-map mean and per-map differences. Falls, safe holds, frame
delays and rejection counts are explanations, not independent success criteria.
For a putative gain near the historical two-point noise level, require additional
seeds rather than selecting the best round.

No production planner behavior was changed as part of this review. The audit
tool and this proposal are the reviewable outputs.
