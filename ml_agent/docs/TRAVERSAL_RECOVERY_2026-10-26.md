# Traversal recovery, 2026-10-26

The operator rejected the observed live Sprawl play and requested renewed work.
The demonstration was stopped at 77 points / about 148 seconds of scored play;
it is a partial run, not a completed-round score. There were no observed falls.
Its first pickup took 17.472 seconds; several subsequent pickup gaps exceeded 8s.
The journal lacks full states, so it cannot support an exact physics replay.

## Diagnosis

In the prior trav3 candidate's Cragmire native journals, optional powerup goals
account for 71.872s (seed 39247103) and 45.120s (seed 105831917). Of those, 50.240s
and 22.912s respectively pursue an item more than 5u below the marble with no
predicted pickup. The principal item is Super Jump at (0, 7.5, -4.3825), while
the marble repeatedly drives around the upper floor near z14. Walking progress
alone was accepted as justification for abandoning the gem for this detour.
The 5u cutoff is analysis only; it is not a controller threshold or map rule.

## Experimental correction

--checked-powerup-detours requires an accepted forecast that collects an
opportunistic item. The existing physical gates still apply. It filters the
candidate set before selecting, and leaves the ordinary gem search available
when a detour has no pickup plan. Necessary recovery acquisition is unchanged.
The flag is OFF by default in all runtime paths.

100 search, real-time, native and full-round benchmark tests passed, including
a regression where horizontal progress approaches an item on another floor.
The new test also checks that a predicted reachable pickup remains eligible.

## Offline screen

80 new episodes: 40 identical starts per arm, all ten maps, horizon48, native
physics, complete current gem group plus next group's first pickup, 40s cap.
Control40/40 finishes; experimental39/40. On mutual finishes, time ratio0.990134,
mean -0.22646s, trip ratio0.986869; identical current-group clearing times.
Total points282 control /281 experimental. One Cragmire trip that previously
finished instead falls. This screen does not justify promotion.

A separate two-episode Cragmire diagnostic from decision501 was identical in both
arms (15.808s, seven pickups). It did not reproduce the chase and is not a fix win.

## Native evaluation declaration

Three maps: Sprawl, Cragmire and Vortex. Seeds39247103 (known diagnosis seed) and
91876331 (fresh), two independent game workers per arm, current production flags
including powerup-through in BOTH arms. The only candidate difference is
--checked-powerup-detours. No heavy sims alongside scored games; policy sources
remain frozen. Native opening, motion, queued-control and source audits must
pass before claiming matched score changes. This is a targeted experiment, not
proof of ten-map improvement. No default change is authorized by partial totals.

Artifacts: dynamics_reports/traversal_recovery_20261026/.

## First completed native diagnostic

Sprawl seed39247103: control190 / strict checked-detour185. This is a loss,
not a promoted improvement. The new control reproduces the previous trav3
candidate's190 exactly. First action divergence is decision57 (298.528s left):
control approaches Super Jump at(-3,17,1.92) from(-8.383,-5.008,.212), while the
strict gate returns to the gem. The item is a legitimate longer approach;
rejecting all progress-only detours is too broad. An isolated prototype checks
whether the snapped goal floor intersects the native item pickup bounds instead.
Only mismatched/unknown floors then require a predicted pickup. It is not part
of the current native experiment. Four mathematical geometry checks pass;
no gameplay improvement is claimed for this narrower prototype yet.

## Actual graph diagnosis and refined prototype

Static cache inspection (no rollout during native scoring) confirms that the
Cragmire goal DOES have a lower floor, snapped z=-3.16, but its goal field is
infinite at the actual upper-floor origin. RoutePlanner.distance nevertheless
returns17.1234 via straight-line fallback. Sprawl's legitimate pickup approach
has a connected graph distance32.8148. Floor matching alone does not distinguish
the failure, so the floor-only prototype is superseded before gameplay testing.
The corrected isolated route-proof prototype accepts ordinary approaches when
the source node reaches the item's pickup floor in the actual graph. Otherwise
it requires a checked pickup. It bypasses the fallback estimate for this test.
Exact observed bounds, positions and graph values are in recorded_route_proof.json.

## Timing sensitivity of historical native controls

The fresh Cragmire seed39247103 control scores125 versus historical83 with
identical start layout, motion and checkpoint. These are NOT a policy gain.
First action divergence is decision671: old latency2.017s yields crossing_brake,
fresh control latency1.594s yields jump_5. The wall-clock2s search cap makes native
results sensitive to work scheduling (old batch4 games/arm, new2). Use current
matched controls, and quiet actual 1x production-worker comparisons before any
promotion; historical max scores or changed-load controls are insufficient.
Evidence: deadline_first_divergence.json.

## Strict variant rejected and batch stopped

Completed pairs: Sprawl39247103 190->185; Sprawl91876331 189->190;
Cragmire39247103 125->93; Cragmire91876331 100->91. Total604->559.
All eight completed rounds have zero native falls; that does not imply fast play.
The unsuccessful variant was stopped before completing Vortex. STOP files and
strict_stop.json preserve that decision. Both arms' declared source hashes still
match. The original12-round declaration is INCOMPLETE; Vortex partial runs are
invalid and excluded from scores. The nonzero runner exits reflect this deliberate
stop, not a completed/valid pool. No defaults were changed.

Next screen: route-connectivity proof,40 new episodes against the40 sealed
controls. Baseline SHA manifest is route_baseline_reuse.json; controls are not
new trials. Gameplay code remains frozen while the isolated prototype is tested.

## Bounded alternate-item screen and implementation

Trying up to3 ranked eligible optional items (rather than only the cheapest)
restores40/40 simulated completions. Points282->282, time ratio0.982416,
mean -0.4192s, trip ratio0.979155, identical current-group clearing times.
This is40 new episodes against sealed controls, not full-round score evidence.
The implementation's pick and detour_route_proven ASTs match the screened
prototype exactly. The flag remainsOFF by default.102 regression tests pass,
including disconnected graph versus straight-line fallback, floor mismatch,
actual pickup eligibility, and trying another item after the cheapest fails.

Quiet live tests are declared in live_plan.json: Cragmire and Sprawl, seed81910417,
8 search workers, lead5, native1x pacing, traceON in both arms, each map's control
then candidate strictly sequential. No competing games or simulations. Source
and full opening/motion/control audits must pass. No promotion from partials.

An unconfirmed suspicion about delayed powerup-jump retiming did not produce
any matching repeated-label examples in the recorded controls and was not
changed. It is not an established cause.

A future corrected-world ramp/short ablation is prepared in ramp_variants.json;
it has not been run, and the isolated settings remainOFF by default.

## First actual 1x result: no promotion

Cragmire CLIseed81910417: baseline113 (2falls/3rescues) / route-options109 (0falls/0rescues).
Opening/RNG, normalized motion/queued controls and source hashes match. The
complete pair is preserved in live_comparisons.json. The feature remainsOFF.

Earlier virtual-clock screens emulate serial candidate work (100us/physics tick)
and do not reproduce the production eight-worker work budget. Combined with
the observed native2s-cap divergence, this warrants a production-worker screen:
real clock, eight candidate workers, one episode process at a time, normal
priority, horizon48. Prepared production_worker_variants.json independently
separates route-options, land-short, roll+completion, their landing combination,
and the isolated observed-progress timeout. None of these future tests has run
alongside the scored live games. No new default or training change.

## Sprawl audit failure, preserved

Raw scores178 control /184 route-options are NOT a matched gain. The final initial
motion audit fails: Blast .07104 versus .06848. At actual GO the meter converges,
but spin differs (-71.954,-15.642,0) versus(-73.577,-15.995,0), due to one extra
64ms countdown/prespin reply. Layout/RNG and source hashes match, but physical
starts do not. Original failed audits remain unchanged in live_comparisons.json,
sprawl_motion_audit_failure.json and sprawl_go_audit.json. The pending flight
probe was stopped before it launched, rather than reusing this unmatched control.
A fixed countdown phase using only neutral controls is needed before re-testing.

## Countdown synchronization repair and new live plan

realtime_bench now samples16ms observations while idle in Start mode until the
naturally charging meter reaches1984ms of elapsed charge, then restores64ms
observations (physics remains16ms throughout). No state restore, grant, forced
respawn or post-GO manipulation. Starts that miss the phase fail the test.
Two actual native starts with different incoming charge phases1712/1776ms both
reach2000ms after cadence restoration, execute23 prespin replies, and have EXACT
matching normalized initial AND GO states and opening/RNG. Evidence:
countdown_smoke/audit.json and both states.json files.104 tests now pass.

The previous queued flight probe was stopped without launching. A fresh control
and flight candidate use this synchronized setup; old178 is not their paired
control. phase_flight_plan.json predeclares seed81910417 and fresh confirmation
seed59386729 if the exploratory candidate scores higher. Both initial and GO
motion audits are mandatory. No competing simulations or other games.

Future eight-worker screens must also reflect the shorter runtime budget:
AheadPlanner uses max(.05, lead*.064-.02), not the serial2s native budget, and
suppresses extra order/two-level allowances. A timed170ms/eight-worker screen
approximates fully-ahead planning; only actual1x tests include queue reaction
latency and changing lead. It must not be described as exact deployment behavior.

## First valid synchronized live gain

Sprawl seed81910417, native1x, current production versus roll-launches +
launch-completion + land-short + fast-supported (route-options remainsOFF):
156->173 (+17,10.897%). Both0falls/0rescues. Holds9.28->7.76s. Opening/RNG,
initial motion, actual GO motion and source hashes all match exactly after
normalizing only previously declared process identities/absolute epochs/unused
disabled timers. phase_flight_explore_comparison.json preserves the checks and
reports. This is an exploratory matched win, not a new historical record or a
ten-map improvement claim. Predeclared fresh seed59386729 confirmation follows.

## Fresh-seed flight confirmation

Seed59386729: synchronized control157->candidate155 (-2), both0falls0rescues.
Holds10.14->16.99s. All initial/GO motion, opening/RNG and source audits pass.
Both declared pairs together313->328 (+15), but n2 is not reliable evidence of
higher expected score and the fresh case is a loss. No flight defaults enabled.

Next timed six-variant screen uses all ten maps,2 starts/map,8 candidate workers,
one episode process, real clock, .172s budget, zero extra order/two-level budget,
.05s fallback-jump allowance. It ablates fast alone, full flight bundle, no-roll
flight bundle, earliest-pickup preservation in two-level ranking, and fast plus
that preservation. The search currently selects top8 pickup plans by proxy
before actual follow-up ranking; the isolated prototype includes the fastest
accepted pickup first (at most one extra), retaining the ordinary physical gates.
Both arms are rerun; no reuse of serial virtual-clock controls. This remains a
screen, not native full-round points or an exact real-time queue simulation.

## Screen objective correction

Some next-group first pickups have different colors/point values across variants
(e.g. Sprawl case230 control8 total points versus fast9). A pure trip-time ratio
then compares different rewards. traversal_results now retains all point/failure
totals, reports point-mismatch counts, and computes trip/episode time ratios only
on equal-point successful pairs. Current-group clear ratios use all shared
successes because their initial group is the same. Historical reports remain
preserved; no historical native score is changed. Full-round points remain primary.

Prepared tools/native_worker_bench.py adds candidate pools to the existing full
native-round controller via an explicitly archived wrapper. It does not change
planning methods or use the live queue. Fixed lead3 means .172s budget/extras0;
worker count, full wrapper source, CLI, effective runtime globals and final
layout/source audits are recorded. It is NOT running while the timed screen is
active. Ownership tests verify pool binding/cleanup on success and failure.

## Live order-search budget mismatch and next experiment

With realtime lead capped at5 and ORDER_EXTRA_S=0, the maximum nominal search
budget is5*.064-.02=.300s. ORDER_MIN_LEFT_S=.35 therefore excludes every
alternative first-target search even at a full queue. It still searches the
current target. This is a deterministic gate mismatch, not yet evidence that
lowering the threshold improves scores. Need test budget-aware order selection
with actual same-start live rounds and retain physical safety checks.

Corrected experimental roll-launch generation to preserve ordinary roll launches
when LAND_SHORT is on and add independent .4/1 braking-share options. Added a
realtime benchmark minimum-budget knob (default.05 unchanged), and native round
pre-score setup callback for canonical countdown.107 targeted tests passed.

New budget_flight_plan.json: exploratory81910417, fresh confirmation81727349;
corrected four-flight-flag candidate plus minimum budget.12 versus default.05.
Fresh controls and candidates; canonical initial AND GO audits, sequential8-worker
1x rounds. Original flight exploratory+17/fresh confirmation-2 remain preserved;
no defaults promoted and no claim of ten-map or human-level performance.

## Minimum-budget launch test rejected

Fresh synchronized1x seed81910417: default183 vs corrected four-flight-flag bundle
with minimum budget.12 at159 (-24). Both0falls0rescues. Initial, actualGO,
layout/RNG and source audits PASS. Holds9.38->53.01s; wall316.3->355.8s.
No confirmation run because exploratory candidate lost. No defaults changed.
The previous same-policy control scored156; initial/GO/layout all match this183
control. Preserve baseline_repeat_timing_variation.json; timing makes one win
insufficient evidence of better expected policy performance.

Order gate diagnostic: replay of exact real-state decision303 (original18.816s
pickup gap), fixed.172s/eight-worker real-clock simulation: baseline23.936s to
1point, threshold.015 candidate8.704s to1point, bothcomplete. Candidate0side
impacts vs3baseline and0fallbacks vs6. Different same-value gems; diagnostic only,
not matched native full-round points or a generalization claim. Ten-map20start/
variant timed screen now compares current.35 gate, .015 gate, and order passOFF.
All use identical recorded starts and existing physical acceptance rules.
Prepared tools/realtime_order_bench.py archives its own source, hash and exact
threshold around the ordinary realtime benchmark; no default mutation.

## Complete ten-map order-gate screen

All30 jobs /60 new episodes completed; source hashes unchanged. Lower .015 gate:
20 pairs, candidate20/20complete versus baseline18/20; replay points152->156.
Equal-point shared-completion time ratio.951684 (17pairs, one point mismatch
excluded); current-group ratio.955063 on18 shared completions. This is a timed
simulation screen, not native round points. Recorded cases include nearby or
identical physical states across pools; do not treat n20 as independent worlds.
Disabling order pass: points152->150, failures2->3, group ratio1.00256; rejected.
Cube185 and Vortex182 falls in current control were completed by .015 candidate.
Cragmire409 next-group trip worsened12.864->17.024 while current-group clear time
stayed4.864; retain this loss. Physical acceptance rules/deadlines unchanged.

Queue gate passed. Fresh synchronized Sprawl1x current.35 vs .015 threshold now
running, predeclared exploratory81910417 then fresh confirmation81727349 if the
exploratory native score improves. Same source/own-wrapper archives,8workers,
lead5 and initial+GO+opening/RNG audits. All flight flags remainOFF/default.05
minimum budget. Sources remain frozen during measurements.

## Live order result and native point pilot

Quiet synchronized Sprawl seed81910417: baseline173 -> smaller gate171 (-2),
zero falls/rescues both, holds8.79->8.70s. Candidate exercised26order switches.
All initial, actualGO, opening/RNG, common sources and wrapper-source auditsPASS.
No live confirmation because exploratory score lost. No performance defaults
changed. The ten-map replay gain remains a screen, not a live improvement.

Native fixed-lead/eight-worker pilot predeclared fresh163904771 and conditional
confirmation429531107 on all ten maps, two concurrent games (one/arm),8forecast
workers/game, .172s fixed allowance, no competing sims. It uses actual complete
native rounds and declared legal rescues, direct observations without the live
queue. Both initial AND GO audits required on every map, with source fingerprints.

First native invocation failed before any game: wrapper prematurely created the
output directory, conflicting with RE.main(exist_ok=False). Original failed
order_native_explore_* artifacts retained; no scores. Fixed wrapper to leave
creation to RE, archive exact wrapper bytes/protocol in finally, refuse existing
results unchanged. Two new full-main ownership/non-overwrite tests:109testsPASS.
Retry uses separate order_native_v2_* output names with same still-unplayed fresh
seeds; both arms now actually scoring KOTM. No source edits during measurement.

## Native pilot prefix and noncharging-map preparation

v2 stopped BEFORE scoring Cragmire: its Blast is enabled but charging=false, so
the charge-phase synchronizer correctly refused it. Five completed pairs retained:
KOTM146->144, Sprawl181->181, Bowl123->127, Cube137->129, Duplex128->140.
Total715->721 (+6, mean+1.2 on these FIVE only). All five initial+GO/layout audits
PASS, source integrity PASS. Original ten-map summaries remain incomplete;
never fill missing rounds with zero or relabel that original batch complete.

Native wrapper now has a separate noncharging preparation: neutral16ms replies
until naturalGO, restore64ms observations, start unspun immediately afterGO
(with at most.128s scored time consumed). Charging maps retain the old1984ms
charge preparation. No state setting, grants, respawns or physics-frame changes.
Both arms must match actual initial and GO snapshots. This differs from live
prespin on noncharging maps and is explicitly a native point pilot, not exact
live deployment. Wrapper now also archives/fingerprints its preparation helper.
Seven focused wrapper/start tests passed, including64ms setup jitter on the
noncharging path and late-start rejection; old109 +2new unique checks covered.

run_order_native_resume.py finishes ALL remaining maps at original fresh seed
163904771 in new order_native_v3_remaining_* outputs. Then analyze the full10
pairs as a SPLIT exploratory pilot (old prefix incomplete flag remains) and, if
mean positive, run fresh429531107 on ALL10 with one new wrapper version.
No source edits during scored rounds. Current status in order_native_resume_status.
No performance defaults enabled. An isolated optional-item stagnation v2 prototype
is prepared only; it observes3D pickup-box progress and times out only optional
items without a proven connected pickup-floor route. Not tested or promoted.

## All ten exploratory native pairs scored; fresh confirmation running

Split pilot result order_native_split_comparison.json:130.6->134.5 equal-map mean
(+3.9/round,+2.99%, total1306->1345). All10 originally planned pairs present,
no unmatched/invalid scored jobs, all initial+GO/layout auditsPASS. Original v2
prefix summary remains complete=false after its setup exception; combined report
also preserves that flag and separately states all_planned_rounds_scored=true.
Do not relabel the original prefix batch complete or hide the split protocol.
Approximate pairedCI95[-1.6549,+9.4549]: expected improvement not established yet.
Per-map deltas: KOTM-2, Sprawl0, Bowl+4, Cube-8, Duplex+12, Crag+12,
Fractured+21, Vortex-5, TreasureBox-1, Concentric+6. No defaults enabled.
Crag111->123: falls0->1, rescues3->2, stalled67.712->45.504s.
Fractured90->111: falls1->1, rescues1->0, stalled35.264->16.832s.
Primary success criterion remains points; neither falls nor one map's win proves
better expected score across all ten.

Fresh predeclared429531107 confirmation runs all10 with the same repaired wrapper
and archived preparation helper. Session75973 owns runner; ports9440/9444, two
native games total,8candidate workers/game, fixed.172s allowance, no live queue.
Current state order_native_resume_status.json; confirmation outputs
order_native_v3_confirm_*. No canonical NAV/source edits during measurements.

Conditional quiet live holdout queue session58685 waits for this whole native
confirmation tofinish with positive mean, then runs fresh81727349 Sprawl1x control
versus.015 gate, second fresh276413983 only if that first pair improves. Its plan
is order_holdout_queue_plan.json; outputs order_holdout_*. No other heavy jobs
may overlap. Earlier81910417 live gate173->171 loss is retained.

Prepared ONLY, not run: optional-item watch v2 and six isolated safety-contract
unit tests (test_optional_watch.py); reserved-order time-sharing prototype; and
escape-only order prototype (retain any checked current-target pickup; only use
alternative targets after that pickup forecast fails). No outcome claims.
IMPORTANT: watch v2.action sees actual observations in direct-native/sim tests,
but realtime AheadPlanner.action receives predicted states. A production watch
must update from the actual-observation hook, NOT treat speculative states as
observed progress. Do not port unchanged to live. Canonical performance defaults
remain powerup-throughON; new gate/flight/timeout changesOFF.
