# Continuous ten-map point improvement

User authorization: work autonomously all night, then clarified **do not stop
until explicitly told to stop**. There is no morning cutoff. Active goal was
created for continuous average-score improvement. Do not mark it achieved just
because one experiment finishes. On an explicit stop, halt owned work and pause
the goal as appropriate.

Starting production policy includes the final additive Super Jump time guard.
That specific recorded failure was fixed and tested, but has no post-fix live
score confirmation. Preserve dirty working-tree changes and protected model
checkpoints; no training or commits are currently planned.

## Active scoring campaign

`dynamics_reports/overnight_20261026/run_escape_native.py`, process exec session
26049. Two simultaneous native games, one per arm, eight forecast workers each,
fixed .172-second search allowance, 16 ms physics. No competing games or heavy
simulations. Native direct observations are not an exact live queue benchmark.

Baseline: current production, alternative-target gate .35. Candidate: identical
source with the reviewed one-line escape-only order rule and gate .015. Both
run all ten maps at seed 187241923. If equal-map mean improves, the driver runs
all ten again at predeclared fresh seed 451809337. Keep every map loss and all
failed audits. Require identical layout/RNG, starting motion and actual GO
states, matching source hashes and workload. The first completed KOTM pair was
148 -> 152 with all three physical audits passing; not an average-score claim.

Outputs: `escape_native_plan.json`, `escape_native_status.json`,
`escape_explore_*`, `escape_confirm_*` if warranted,
`escape_combined_comparison.json`. Runner log `escape_native_runner.log`.
Wrapper `tools/native_escape_bench.py` archives its own code and prototype,
checks the prototype AST is exactly the one-line change, and leaves defaults
unchanged. Avoid editing any canonical NAV, native-worker/preparation helper,
wrapper or escape prototype while the campaign is active: sources are frozen.

## Prepared next experiment

`dynamics_reports/overnight_20261026/optional_watch.py` is isolated from the active
campaign. `tools/native_watch_bench.py` can opt into it for native point tests.
No score results yet; do not globally enable it from earlier replay gains.

Only unproven opportunistic powerup pursuits are eligible. Actual 3D pickup-box
progress must improve by 1 unit within the existing five-second stall window;
otherwise skip that item for the existing 20-second interval. Proven connected
pickup-floor routes and explicit resource fetches remain available. No actor
reset, state change, grants or teleports. The item is excluded only from the
optional selection list; it remains in physical forecasts and observations.

Native controllers update from actual action inputs. The real-time installer
disables updates from speculative `.action` inputs and attaches observation to
`AheadPlanner._measure_progress`, using actual simulation clock and duplicate
observation protection. Route proof also uses actual pose. Ten pure-Python
contract tests pass in `overnight_20261026/test_optional_watch.py`, including
predicted-clock exclusion, duplicate actual observations, vertical progress,
respawn gaps, TTL and no actor reset. They take 1 ms and do not run forecasts.

## Another hypothesis to investigate

The 50-unit side-impact charge currently spans the entire candidate forecast,
including continuation after a pickup invalidates the plan. A fast ordinary
jump can therefore lose to a slow approach because of predicted contacts in a
tail that is never executed. Keep physical support/fall/braking gates over the
full forecast. Investigate bounding only this soft contact charge to the next
real pickup plus the live queue/bridge reaction window. First measure impact
tick locations on the recorded Fractured incident; do not assume the contacts
are post-pickup. Prior global time-price/low-penalty variants failed, so do not
blindly repeat those sweeps. No prototype or outcome claim for this idea yet.

Prepared since the initial note: isolated `overnight_20261026/search_bump_window.py`
changes only the soft contact-count window; seven replies after a real pickup,
with full costs for virtual crossing markers and plans without a pickup. Five
pure tests pass, preserving the complete input forecast. This remains a
hypothesis until actual Fractured impact tick times are measured; no forecasts
or score tests of it have run. Do not overlap those with native campaigns.

## Additional queued native campaigns

Optional watch is queued in exec session 60450, script `run_watch_native.py`.
It waits for the entire escape exploration/conditional confirmation campaign
to close, then uses fresh seeds 318697241 / 739156831, all ten maps. Both arms
retain gate .35, no escape ordering. It only schedules confirmation when mean
points improve AND actual `OPTIONAL_ITEM_STALL_SKIP` activations occurred.
Outputs `watch_native_status.json`, `watch_explore_*`, `watch_confirm_*`.

Primary-search reuse is queued after the watch campaign in exec session 73394,
script `run_reuse_native.py`, seeds 125903417 / 921773123. Wrapper
`tools/native_reuse_bench.py`, isolated helper `overnight_20261026/primary_reuse.py`.
This is an architecture experiment, not a new score weight: only immediately
consecutive identical completed searches within a single `pick` can reuse a
result. There is no across-observation cache. Different goals, warm controls,
actor state, next-goal maps, reservation/crossing settings or physical inputs
prevent reuse. An intervening different search invalidates the cache. Reuse
after a shrinking deadline returns already completed checked work, not new
computation outside the budget. Eleven pure tests pass, including scope cleanup
on failure. Native pool integration/hit rates still need testing. Confirmation
requires a positive ten-map mean AND actual `primary_search_reused` counts.

Neither waiting queue starts a game while its predecessor is active. There are
never more than two scoring games / sixteen forecast workers across these
campaigns. Freeze canonical code and all wrappers/prototypes used by each
active campaign. All defaults remain unchanged by these experiments. No scores
from the new watch/reuse wrappers have been measured yet.

Early escape pilot results (all initial/layout/GO audits pass): KOTM148->152,
Sprawl166->156, Bowl136->136, Cube135->134, Duplex163->163, Crag122->123,
Fractured110->106. These are seven completed pairs, not a ten-map average.
Keep the Sprawl and Fractured losses.

## Subsequent measured results and queued isolation tests

Escape exploration completed all ten matched native pairs: equal-map mean
138.2 -> 138.5 (+0.3), approximate interval -9.118 to +9.718. This is weak
evidence, not a promotion. Vortex gained 37, Treasure lost 24 and Sprawl lost
10. Fresh-seed confirmation (451809337) is still running. Its first six
completed pairs: KOTM155->143, Sprawl174->178, Bowl143->143, Cube143->143,
Duplex130->133, Crag111->87. Keep every loss; no partial ten-map mean claim.

Primary-search reuse now has 13 passing isolated tests. Two added cases
cover array/tuple goal equivalence and inference tensors without a version
counter. Native integration and actual cache hit rates remain unmeasured.

The corrected soft-contact timing probe is queued in exec session 81856,
`probe_bump_window.py`, after the reuse campaign. The earlier waiting probe
was stopped before forecasts to correct JSON serialization; preserve its log.
The corrected script must finish and write `bump_probe.json` before any later
live test starts. Its output is a forecast diagnostic, never native points.

`group_wait.py` isolates only the playable/no-visible-gems ahead boundary:
wait for actual observations instead of queuing the previous task's tail.
All other checked-tail features remain off. Four tiny boundary tests pass.
`tools/realtime_group_bench.py` archives helper/wrapper/checkpoint identity.
Queued driver `run_group_live.py`, exec session 97250, waits for the probe,
then runs two sequential 1x Sprawl pairs at seeds 287451913 and 618903217.
Both seeds run regardless of the first outcome; baseline first on seed one,
candidate first on seed two. Initial/layout/actual-GO/source audits must pass.
These are one-map tests, not evidence of a ten-map average gain.

Unused isolated `gem_demo_late_through.py` enables next-goal credit for late
powerup pickups (after reply24), evaluated at pickup+8 replies capped to the
forecast. Six isolated cost checks pass. Precise current behavior: when a
pickup is later than the common evaluation time, the existing through branch
returns ZERO post-progress credit, plus end-speed and pickup-time terms. It
does NOT charge distance back to the collected item in this branch. The new
prototype may bias comparisons toward longer forecasts and is not yet a
recommended fix. No physics runs or point tests have evaluated it. Early
pickups, gems, missing pickups and missing next goals are unchanged by tests.

Read-only `audit_pickup_gaps.py` audited the completed Crag confirmation
journals (111 control, 87 escape candidate). Both have 20-29 s gaps without
pickups and many retained safe_hold decisions. The candidate's first sampled
safe hold follows a planning_time_budget failure. These are observed gaps,
including legitimate travel/falls/rescues; do not label all of them idle time.
Three additional cheap tests execute the REAL pick/order_choice function bodies
with stub forecasts: default searches the primary progress target twice;
reuse removes the duplicate; a changed through target keeps two searches.
This raises reuse verification to 16 tests, without native integration yet.

`tools/native_sj_guard_bench.py` and queued `run_sj_native.py` (exec24452) isolate
the final additive SJ fix itself, after the group-wait live campaign. Historical
control disables ONLY apply_use_gain, retaining apply_use_margin and every
physical acceptance check. Current production is the fixed arm. Both fresh
seeds471392681/803156429 run all ten native maps regardless of first result;
reverse arm port assignments on seed two. Same two-game/eight-worker/.172s
workload and physical/layout/actual-GO/source audits. No other experimental
policy enabled. This test is necessary to measure points from the final fix,
rather than attributing earlier precursor replay gains to a different source.
It remains queued; no SJ-guard native average has been established yet.

## Closed escape campaign: reject default adoption

All20 matched full native pairs complete, zero invalid jobs/unmatched pairs,
all actual layout/initial/GO audits pass. Exploration mean138.2->138.5 (+0.3),
fresh confirmation134.3->132.8 (-1.5), combined136.25->135.65 (-0.6), approximate
interval[-6.308,5.108]. Combined map deltas: Bowl0, Concentric+8.5, Crag-11.5,
Cube-0.5, Duplex+1.5, Fractured-6, KOTM-4, Sprawl-3, Treasure-13, Vortex+22.
Escape-only order switching is REJECTED for default use. Keep all archives;
Vortex's consistent gain alone does not establish a general or map-specific
promotion. No production changes made for this experiment.

Optional-item observed-progress watch campaign has automatically STARTED at
seed318697241 (exec60450). Freeze its helper/wrapper and canonical inputs now.
Reuse remains queued, then bump diagnostic81856, group live97250 (now7 isolated
tests including installation), final SJ native24452. No new forecast workloads
while either native arm is active. No ten-map gain from any of these later
experiments has yet been measured.

## Prepared follow-up from the rejected ordering rule

The rejected rule preserved only forecasted current-target pickups. It could
still switch away from a useful checked non-pickup approach, especially on long
legs. New isolated `blocked_order.py` preserves BOTH pickup and useful checked
progress; only a primary with NO useful accepted future may search alternatives.
It adds two guarded breaks to the real order_choice AST, leaves the existing
restoration/acceptance logic intact and still obeys .015 remaining-time gate.
Five tests execute that real body with cheap forecasts: primary approach kept,
primary pickup kept, blocked target escapes, no alternative leaves ordinary
fallback, expired budget prevents alternatives. No physics/score results yet.

Wrapper `tools/native_blocked_order_bench.py` archives itself/helper and counts
actual alternative searches. Driver `run_blocked_native.py`, exec49770, waits
for the final SJ campaign to close, then seed681297413 on all ten maps; fresh
194835721 confirmation only if mean improves AND alternatives actually run.
Same fixed .172s/two-game/eight-worker workload and complete physical/source
pair audits. It is a distinct, narrower follow-up to a rejected experiment;
do not relabel the original -0.6 result or call this a confirmed fix.

Endgame idea reviewed, not queued: removing the higher-single-gem-value filter
might allow better total-points routes among equal-value gems, but the Oct17-18
handoff already records rough walking-time estimate losses on Crag/Fractured/
Cube. Keep it diagnostic only until observed real ending cases justify it;
do not blindly repeat the broad total-points switch. The current robust rule's
old offline effect was only +5 across159 endings, below native pool resolution.
The historical native reference reset/load caveats in that entry still apply.

## Predicted powerup-slot ordering mismatch (new, not promoted)

observer.cs collectPowerups sorts its three retained slots by straight 3D
distance. ObsBuilder reads only slot0 as the policy's nearest-item summary.
Both NavigatorProposal.project_observation and realtime.predict translate the
item coordinates to virtual positions but retain the old slot order. Actual
observations remain correctly sorted; this concerns projected observations.

Read-only audit uses RECORDED movement, not future knowledge in the policy:
`sprawl_powerup_projection_5_v2.json`, confirmation baseline seed451809337.
Among4682 playable exact320ms windows, retained nearest index changes810 times,
nearest TYPE changes757. Old first-type mismatch against actual future slot0
771, re-sorted retained first-type mismatch37 (738 corrections,4 regressions).
Remaining limitations include original-three truncation/previously unrepresented
items and changed worlds. This is an ordering diagnostic, NOT forecast/score
accuracy or points gained. Original v1 report retained separately.

Isolated `powerup_order.py`: stable 3D sort of known positive-type records at
virtual positions; all fields/timers/metadata preserved, actual navigator
index0 unmodified. Seven tests use the REAL project_observation AST, without
Torch/dynamics, plus realtime-wrapper metadata checks. Importable worker setup
patches function globals in the canonical navigator module, preserving class
identity/pickling. Both arms use the same setup; only candidate enables it.
`tools/powerup_order_bench.py` supports native and realtime; native covers only
NavigatorProposal; realtime also corrects AheadPlanner.predict. No other knobs
or resource policies change.

Queued `run_powerup_order_native.py`, exec13909, waits for blocked campaign49770.
Seeds487913621 exploratory /109736581 conditional confirmation (mean must rise),
all ten native maps, same .172s/two games/eight workers each, gate .35 botharms,
complete layout/initial/actualGO/source audits. No physics integration or score
evidence yet. Canonical sources/checkpoint still untouched by these prototypes.

## Closed optional-watch pilot: inactive, no default promotion

All10 native pairs seed318697241 completed; all initial/layout/actual-GO/source
audits passed. Mean128.6->133.4 (+4.8), approximate interval[-2.320,11.920].
Watch activations ZERO. Preserve the positive scores as measured, but do not
attribute them to skipping stalled items or call this a confirmed policy gain.
The predeclared gate requires actual activations for confirmation, so no watch
confirmation was run. Current defaults remain unchanged. This near-null result
also reinforces how deadline-sensitive native play can produce score variation
when additional bookkeeping/queries exist; exact causation is not established.
Reuse campaign73394 automatically started seed125903417. All other queues wait.

## Reuse v1 route-state limitation found during review

Active v1 native campaign sources remain frozen. Its key includes the route's
terrain identity but omits its held waypoint/key/field identity. A new focused
regression reproduces v1 reusing across a waypoint change. The cached controls
remain physically checked from the same dynamics, but this is NOT strict
identical-proposal-input equivalence; no v1 default promotion on that premise.
Preserve its measured scores as v1 results, not strict-v2 results.

Unused `primary_reuse_strict.py` copies v1 and adds route.key, route.waypoint,
and route.field identity to the key. `test_reuse_route_profile.py` passes17
checks: original13 contracts on strict code,3 new route-state exclusions,1
characterization of v1's omission. It has not been physics/native tested yet.
V1 has actually fired twice in Sprawl so far (unlike watch's zero activations).
First completed v1 native pairs KOTM151->149, Sprawl171->181, physical audits
pass; these are NOT a ten-map average or strict-version score evidence.

## Latest operational checkpoint (around03:05 MDT, continuous goal still active)

Reuse v1 exploration COMPLETE10pairs, mean130.0->135.1 (+5.1), actual reuse hits
418, approxCI[-2.414,12.614]. Map deltas Bowl+10, Concentric-2, Crag+22, Cube+7,
Duplex+6, Fractured-9, KOTM-2, Sprawl+10, Treasure-14, Vortex+23. Fresh921773123
confirmation ACTIVE in exec73394; KOTM148->149 completed. Version1 route-state
omission still means no default promotion on a strict-equivalence premise.

Strict follow-up wrapper `tools/native_reuse_strict_bench.py`, driver
`run_reuse_strict_native.py`, exec40135, queued after powerup-order13909. It runs
all10 at637190843 only if v1's COMPLETE combined/pilot mean is positive with
real reuse hits. Confirmation260817359 requires its own positive mean/hits.
Both arms use strict-helper sources; no v1/v2 score mixing.17 pure tests pass.

Final-ranking backup audit: `audit_final_backup.py` and saved
sprawl/fractured/treasure_final_backup_audit.json (journal hashes recorded).
Two-level chosen decisions lacking a current recorded backup result:219/303,
189/233,176/240 respectively. Counts are decisions, NOT unique plans or causal
fall evidence. Original check_backups can check one cheap plan then return;
apply_two_level later changes ranking/costs. Selected other futures can lack a
recorded result. Re-ranking can also overwrite an earlier150u backup penalty
while retaining failure/unchecked flags. This is the isolated ordering issue.

`final_backup.py` prototype checks newly untested useful finalists after the
original search's two-level/resource filtering, max4 additional brake forecasts;
restores dropped existing soft penalties, reuses existing results, keeps
unproven plans available with existing penalties, no new hard gates or stop
families. Existing unknown/ambiguous backup behavior stays the same. Added
braking calls use the original deadline exemption, so .172s denotes the MAIN
search allowance, not a new bound on total decision time. Nine pure tests pass,
including real check_backups ordering repro, no double penalty, no resurrection
of rejected SJ futures and preserved signature for cache composition.

Wrapper `tools/native_final_backup_bench.py`; queued `run_final_backup_native.py`,
exec60479, waits for strict40135. Seeds390817463 exploratory/721934087 conditional
confirmation, all10 native maps, usual matching states/source/workers/budget.
Confirm only when mean rises AND a new failed backup/restored lost penalty
actually changes scoring; successful checks alone cannot certify a policy gain.
No physics/native score tests yet. This prototype is OFF by default.

Current remaining execution order: reuse73394 -> bump diagnostic81856 ->
group1x97250 -> SJnative24452 -> blocked49770 -> powerup-order13909 ->
conditional strict40135 -> finalbackup60479. Closed26049/60450 exec sessions
have been polled and released. No competing heavy simulations. Freeze active
canonical/wrapper/helper inputs. Native SJ aggregate guard-count additions to
its driver were edited after waiting process launch; per-map report counters
from its wrapper remain the authoritative activity source if aggregates omit
those optional fields. Test commands/policies/seeds were not changed.

Additional ideas REVIEWED, not implemented: original current-code SS pickups
in three sampled Sprawl baseline seeds enter at roughly8.6-12.8u/s total (not
the old4.9 example and NOT the same human5.9s opening). Acquire-and-use macros
need actual held-state/next-goal handling: current use families require initial
held>0, and SequenceProposal's passed-through branch returns before scheduled
use. Do not just add fixed use timing to a through plan. Existing-pair forecast
reuse also needs care: ordinary plans invalidate/replan at pickup; named
line/bounce continuation assumptions differ. Advisory detour distance queries
mutate the main route's waypoint; a future copied-query-route experiment may
explain/improve strict cache hit rates, but no such source change/test exists.
Prior broad endgame/time-price/refinement/FAST_SUPPORTED/keep-plan failures
remain relevant. Do not blindly rerun them or mistake raw forecast diagnostics
for native points. There is NO morning cutoff and no user stop authorization.

## PRIORITY QUEUE REVISION1 - supersedes the old waiting-driver chain

Current source/score owners retained: reuse73394 (fresh confirmation),
bump81856, group1x97250. Five verified WHOLLY UNSTARTED idle managers were
retired by owned PID: SJ25076, blocked28704, powerup2444, strict20256,
finalbackup32696. Original prospective plans were MOVED, byte-hash verified,
into priority_queue_1/original_*_plan.json; driver source copies and original
statuses/PIDs/cmdlines/hashes are in retirement_preflight.json and
retired_waiters.json. Their exec sessions24452/49770/13909/40135/60479 exited1
because the idle managers were deliberately replaced, NOT because any scored
round failed. No scored games or forecast workers were interrupted.

Single new owner `run_priority_queue.py`, exec98254, status
`priority_queue_1/status.json`, log `priority_queue_1/runner.log`. It waits for
original group live completion, then enters an AGENT-ONLY closed-batch review
gate. There is no user permission/input dependency. After inspecting evidence
and finishing any needed isolated checks/source preparation, agent writes
priority_queue_1/advance_group_live.json with {"advance":true,"reason":"..."}.
Then each job runs sequentially, and the same gate appears after it, using
advance_<jobname>.json. Coor captures each driver source and checks it did not
change during that batch. All original wrappers still enforce real scoring
input/source/layout/initial/GO audits. Per-job child PID and logs are recorded.
Do NOT leave an agent review gate pending on the mistaken belief it needs the
user's approval. Agent should review/advance or conduct justified follow-up
verification while all scoring jobs are closed. Goal remains active throughout.

New order AFTER group1x: launch_native -> reuse_strict_native -> sj_native ->
powerup_order_native -> final_backup_native -> blocked_native. Original seeds,
source variants and confirmation gates retained for those old experiments.
Strict waits for launch now, powerup waits for SJ; old versions/plans preserved
in the receipt directory. Other dependencies are already closed by their turn.
Code/source changes for scored phases happen only at closed-batch review gates;
unstarted preparations are archived before being revised. No source promotion
has occurred yet; canonical NAV/model inputs remain frozen for active scoring.

New `tools/launch_completion_bench.py`, `run_launch_native.py`: tests ONLY the
existing --launch-completion flag, separating it from the earlier losing
roll/short/FAST_SUPPORTED bundle. Prior timed variants did NOT test completion
alone. Fresh856201943 exploratory /317594681 conditional confirmation, all10
native maps, same2games/8workers/.172s main allowance, gate .35 both arms.
Explicit runtime-global audit per map: LAUNCH_COMPLETE False/True, with
POW_LAUNCH_ROLL/LAND_SHORT/FAST_SUPPORTED/SS_RIDE all False. Old cost margin and
SJ time guard retained. Wrapper also supports --realtime for quiet follow-up
pairs (adds --planner-flag=--launch-completion). Source/checkpoint archiving is
included in that live mode. No scored run of the standalone flag has begun.

Reuse v1 fresh confirmation currently has KOTM148->149, Sprawl174->168,
Bowl128->120, Cube146->143, Duplex144->131, Crag126->92, Fractured123->109.
Do not call the earlier pilot+5.1 a reproduced average gain. All losses remain.
Complete the remaining maps/audits before the combined decision. V1's known
route-state cache omission also prevents promotion on strict equivalence.
There is STILL no user stop request and no morning cutoff.

### 2026-10-27 04:12 continuation: workload incident, quiet retry, bump-window priority

User explicitly requires continuous autonomous work until told to stop. No morning cutoff.
Reuse v1 CLOSED: combined20 native135.9->132.95 (-2.95), rejected; pilot positive reversed. Optional watch CLOSED128.6->133.4 but ZERO activations, no causal gain or promotion. Strict reuse remains gated and should skip because v1 combined negative.

OWNED WORKLOAD INCIDENT: extra bump probe v3 ran during original group_live_0_baseline. Original group attempt is INVALID as quiet evidence. Driver/game stopped by verified owned PID; partials retained. group_live_workload_incident_1 contains process receipt, cleanup records, exact v3 report/source. Probe v3 overwrote v2 bump_probe.json; original v2 full JSON is LOST, runner log remains. Do not claim preserved v2 JSON. Probe now refuses existing completed output. Never proceed with dependent launches after a failed preflight.

Fresh quiet retry: run_group_live_retry1.py (exec22592), original declared seeds287451913/618903217, fresh labels group_live_retry1_*, port9464, sequential1x8workers. Same canonical group_live_status.json. Pair0 passed all audits:159->157 (-2), holds16.43s->12.73s; no benefit proven. Seed1 underway. Only this retry may support quiet evidence. No other forecasts while active.

Bump probe v3 unbudgeted recorded Fractured SJ incident: full-window costs select Nav47replies/3.008s; pickup+7reply soft window selects ordinary jump18/1.152s. Both and fastest jump independently pass braking backup (supported, drop0, nonambiguous). New jump retains one early charged impact50u, ignores three late impacts. Diagnostic only, NOT points. Full physics/support/landing/ledge/resource/braking gates retained. Ordinary pickup normally replans, but committed modes can retain future controls; no universal unexecuted-tail claim.

Prepared tools/bump_window_bench.py + bump_window_install.py. Installer injects ONLY side_impacts and cluster_penalty into canonical module; AST validates exact delta, no stale proposal-class/source rollback. Worker setup forwards candidate flag. Three installer tests +five window tests pass. New run_bump_window_native.py:598172643 exploration,842619357 fresh confirmation only positive mean AND window action counter; all10 maps/2games8workers/.172s/gate.35, full source/layout/initial/actualGO audits. No scoring started yet.

Retired idle coordinator1 PID26240 after verifying waiting_for_group_live and zero children. Source/plan/status/PID/hash receipt preserved in priority_queue_2; original priority_queue_1 retirement receipts remain authoritative. New run_priority_queue_v2.py exec9072 uses priority_queue_2, waits current group status complete then agent review. Order bump_window_native -> launch_native -> reuse_strict_native -> sj_native -> powerup_order_native -> final_backup_native -> blocked_native. Agent-only advance_<job>.json gates require review of closed results; no user approval. No original campaign seeds or confirmation gates changed. Active scoring driver remains untouched. No source/default promotion yet.

2026-10-27 04:17: quiet group retry CLOSED, audits passed;159->157 and174->168, mean-4. REJECT/no promotion. Runtime bump preflight passed:8tests, only two canonical functions change, class identities preserved, globals canonical, worker setup importable, duplicate install rejected. Coordinator2 advanced to bump-window native all10.

2026-10-27 04:34 active bump window native campaign (driver2616, coordinator2 exec9072).
Completed valid pairs: KOTM149->143 (-6), Sprawl182->174 (-8), Bowl133->133. Cube running; do NOT stop remaining maps or mutate scored sources. No winner yet. Sprawl selected-window count about1000, so candidate is active. Source preflight8tests plus runtime identity check pass.
Read-only trace diagnostics: bump_window_kotm_first_divergence.json (first different action at29, same raw; no selected-window flags acrossKOTM). First LABEL difference at24: jump_5_landing vs direct_jump_9, identical raw and two_level_replies60/cost-134.64; unselected changed costs/deadlines NOT isolated, do not assert pure noise or causality. bump_window_sprawl_pickup_gaps.json: baseline182points/1fall/133pickup events; candidate174/0falls/126events. Both have >18s pickup gaps; candidate also16.448s gap ending withheldSJ1 and Nav landing/ballistic. Group sequences can diverge after actual falls/pickups; these are diagnostic executed gaps, NOT matched per-trip causal evidence. No new forecasts were run during scoring.
Current next phase after bump gate: launch completion alone, then strictreuse(expectedskip v1negative), SJaudit, poweruporder, finalbackup, blockedorder. Agent must review closed evidence and write priority_queue_2/advance_bump_window_native.json before launch may start. User has NOT requested stop; goal stays ACTIVE.

2026-10-27 ~04:42: prepared separate powerup TIMER candidate, not launched or in coordinator2 yet. Static review found Navigator.project_observation advances raw31/32 clock but leaves raw powerup countdowns unchanged; RT.predict already decrements correctly. Read-only executed Sprawl baseline audit sprawl_projected_powerup_timer_audit.json: at5/24/48replies598/256/147 matched positive timer records,6/23/27 availability crossings missed by static timers. Subtracting observed elapsed time matches future recorded countdowns to float rounding; resets/discontinuities excluded, no point/forecast claim.
New powerup_timers.py wraps Navigator.project_observation only: future index>0 positive known timers clamp(timer-index*.064,0), actual raw unchanged, no item creation/pickups, no RT double decrement. Four isolated tests use real AST projector and confirm only timer fields, index0/padding/input immutability, sorting composition both orders, duplicate install guard. tools/powerup_timers_bench.py and run_powerup_timers_native.py prepared with fresh704391827/263857149, all10/.172s/8workers/gate.35, waits blocked campaign. Not launched; coordinator2 does not include it. Consider adding after current queue closes, or at safely archived idle review. Do not silently fold it into the separate sorting experiment.
Bump campaign partial fifth map: Duplex146->156 (+10), total5maps -11 points. Cragmire running. All scored sources unchanged.

2026-10-27 ~04:57 bump native ongoing: Cragmire95->135 (+40), Fractured102->117 (+15), Vortex124->116 (-8); first8maps total+36. Do not call confirmed gain yet. Treasure baseline closing, candidate Concentric underway. Current native driver automatically runs fresh842619357 if complete exploratory mean positive AND selected-window counters >0. Driver still2616 /coor2exec9072; original sources unchanged.
Prepared CONDITIONAL quiet all-map live follow-up (NOT LAUNCHED): run_bump_window_live.py, seeds529781263 and conditional917462803, port9468,8workers,lead5, sequential games, alternating first arm by map/reversed nextseed. Requires closed positive20-pair native mean AND fresh native positive with window decisions. Runs only while coordinator2 idle at bump_window_native review and advance_bump_window_native.json ABSENT. Keep that gate closed until any live campaign closes. Do not launch it alongside native confirmation or another phase. Initial root preflight must verify no forecasts/games remain before manual live driver launch and record ownedPID/session. If source/native evidence fails, skip live and continue native queue.
New separate tools/bump_window_live_bench.py copies exact window install; active tools/bump_window_bench.py was NOT edited. Separate live wrapper supports noncharging maps by using existing native_worker_bench.prepare_countdown neutral natural GO, no prespin, while charging starts keep original alignment. The existing realtime_bench.align_countdown requires charging; raw all-map live would fail otherwise. Preparer source is included in live source archive/hashes. Benchmark requires --realtime. Need run existing tools.test_native_worker_bench at CLOSED gate before live launch; do not run a mechanics forecast while native active.
Live comparator: exact initial/layout/RNG/actualGO/source/flag/runtime/workload checks. Sources frozen across all maps and both seeds. Driver source copied and hashed; review_gate rejects changed driver or a next-native advance file. Four pure evidence tests pass (actual compare/review AST): validpair, botharms silent source switch blocked, actualGO mismatch despite matchinginitial blocked, nextphaseadvance prevents live. No game/forecast in these tests. Live fresh all10 confirmation runs only if first all10mean positive AND window decisions active; no default promotion from one map.
New audit_bump_runtime.py reads completed baseline/candidate wrapper_runtime.json, requires10maps, seven-reply candidate window absent inbaseline, all other saved control globals equal after excluding only TL_STATS diagnostic counters. Run after full explore/confirm snapshots exist; no output written yet. Partial read-only check all9startedmap snapshots showed ONLY BUMP_PICKUP_TAIL_REPLIES and TL_STATS differ. Do NOT mutate scored driver to add audit midcampaign.

2026-10-27 05:03: first complete bump-window native pilot CLOSED,10validpairs/0invalid:138.3->142.2 (+3.9), CI[-5.318,13.118],5120 selected-window decisions. NOT confirmed. Concentric109->110 (+1); allpermap losses retained. Fresh842619357 running automatically under originaldriver2616. audit_bump_runtime.py bump_window_explore PASSED all10: only intended7replywindow difference, TL_STATS diagnostic counters excluded explicitly. Active canonical/wrapper/helper sources untouched.

New potential SJ correctness issue: apply_use_gain plain reference set accepts info.no_braking_backup True (because backup failure is a soft cost, not info.rejected). Actual AST characterization proves a known failed ordinary pickup can veto a resource-use future. No new safety gates needed: exclude only these known failed ordinary futures from the comparison, keeping them in actual planner candidates; unknown backups remain references, old margins/all existing rejections retained. Prepared sj_backup_reference.py helper with cheap old-shadow comparison/no added forecasts, counters exclusions/restoreduses. Six isolated actual-guard ASTtests pass: currentbug, correction, safer slower reference, unknown remainseligible, originalFractured waste stillblocked, otherrejectionsnotrevived. NOT enabled inproduction/current campaigns.
Readonly existing Fractured pilot trace exposure: baseline275 heldSJ decisions/148ordinarypredictedpickups/0chosenknownfailed; candidate717/554/1. Candidate decision1987 jump_0 pickup13 has known braking supportFalse/drop15.4443u/replies20/nonambiguous. sj_backup_reference_exposure_fractured.json and exact row sj_backup_reference_state.json saved. These selected-decision data DO NOT prove suppressed use candidates existed or caused that choice. Need CLOSED-gate replay/search audit to establish real effect; do NOT run forecasts during confirmation.
Prepared tools/sj_backup_reference_bench.py and run_sj_backup_reference_native.py fresh326791453/954812367, all10/.172s/8workers/gate.35; waitsSJ audit complete. Confirm only positive mean AND restored_uses>0. Unlaunched, NOT part of coordinator2. Root may schedule after safely closed single-owner review; do not accidentally launch a waiting concurrent queue or overlap live/scoring. Source/test/preparation only so far.

2026-10-27 05:13 confirmation progress: KOTM150->144 (-6, repeated pilot-6), Sprawl192->193 (+1; pilot-8); Bowl finalizing. Original Bump20-native campaign remains sole active scorer. No new variant promoted or queued-manager launched.
Future diagnosis notes ONLY, not implemented: TL shortlist interaction may matter: KOTM pilot first different plan at24 had equal pair ETA60/cost-134.64 but jump_5_landing vsdirect_jump_9. apply_two_level CLUSTER_LEG_U12 compares TARGET->NEXT gem leg from through_context, NOT marble->target distance. Do not implement an incorrect start-distance predicate. Broad window may affect TL shortlisted alternatives before final cost replacement. Also line/bounce/launch/landing commitments can execute controls beyond first pickup+7; a more principled follow-up would bound soft cost at actual next replanning opportunity, rather than assume all tails are unused. Must inspect commitment contracts and test fresh seeds before any such change. No map-specific KOTM exception. Current prototype deliberately retains all physical/backup/ledge gates but is still unconfirmed on points/live.
Advisory-route query issue remains unimplemented: pick() powerup distance queries route.navigate(q,current), route.navigate(position,q) mutate main held waypoint even for samegoal (different origin can clear it). RoutePlanner.navigate resets waypoint on goal change or directvisibility. Successful speculative searches explicitly navigate selected goal before return, but failed advisory branches can reset persistence before current-gem search. A correct isolation would clone query routes AND optional-detour speculative search routes, preserving actual selected-route commitment. Merely copying the entire pick route does not protect the same-pick ordinary search from speculative mutations. This is an investigation idea, not diagnosed score cause/new queued experiment.

2026-10-27 05:18 confirmation: KOTM150->144 (-6), Sprawl192->193 (+1), Bowl130->130 (0), Cube156->139 (-17). First4fresh total-22; repeated KOTM/Cube regressions. Duplex running. Keep full fresh10 and all audits, no default promotion/claim of broad win.
Prepared (NOT RUN) probe_sj_backup_reference.py. Hard gate before imports/forecast requires coordinator2 awaiting_agent_review at bump_window_native, no advance_bump_window_native file, and any bump_window_live status complete. AST-only guard check correctly refused current active confirmation. At closed gate, first root must verify no other game/forecast process before launching probe. Probe loads savedFractured1987 state; runs unbudgeted root search under original full-window and window7 separately, captures candidates immediately before additive guard, then compares original/filtered guards on exactly the same frozen physical candidate set with no extra forecast per comparison. Original navigator/controller history NOT reconstructed; no exact executed-choice/time/points claim. Preserves outputs, sourcehashes, flags. Existing guard/helperclass identities canonical; no stale module replacement. Probe is useful before deciding whether to schedule SJreference native experiment; source/data counters might prove no actual restored use in this state, retain such null result.
No live campaign has started. Native confirmation remains the sole scorer, driver2616/coor2exec9072. Standalone timer/SJreference scripts remain UNLAUNCHED and outside fixed coordinator2 queue. User has NOT said stop, goal ACTIVE, no morning cutoff.

2026-10-27 05:29 IMPORTANT fresh bump confirmation is NOT reproducing pilot: KOTM-6, Sprawl+1, Bowl0, Cube-17, Duplex-10, Cragmire136->113 (-23, versus pilot+40). First6fresh sum-55. Finish allremaining Fractured/Vortex/Treasure/Concentric regardless; retain everyloss. If fresh nonpositive, NO conditional live campaign (run_bump_window_live prerequisites reject it) and no promotion even if combined happens positive. Need run audit_bump_runtime.py bump_window_confirm after all10, review final source/layout/actualGO/budget/control audits. Likely next declared phase is isolated launch completion via coordinator2 after explicit agent review receipt. Do not silently change parameter midconfirmation or call pilot+3.9 confirmedhigher.
Only current scorer: driver2616 under coordinator2 exec9072. No futureSG/timer driver started. At CLOSED bump gate, can execute source-protected SJbackup-reference probe FIRST with no other game/forecast running; leave advance_bump_window_native absent throughout diagnostic. Then if no qualifying live (freshpositive requirement), review/reject bump and advance native queue. Existing native_worker_bench tests for live neutralGO matter only if qualifyinglive is actually scheduled. No morning cutoff; continuous goal activeuntiluserstop.

2026-10-27 05:49 ACTIVE OWNER CHANGED: coordinator2 exec9072/PID25616 now running launch_native driverPID5776. launch_native_status exploratory, seeds856201943/317594681. Source currentproduction unchanged; onlyLAUNCH_COMPLETE flag candidate. No other forecast/game campaign running. Bump driver2616 closed successfully. Goal stillACTIVE, user has notsaidstop.

Bump20 FINAL: exploratory138.3->142.2 (+3.9); fresh135.3->135.3 (0.0); combined136.8->138.75 (+1.95), CI[-4.815,8.715],20valid/0invalid. FreshConcentric96->123 (+27), so latepositiveoffsets heldoutlosses toTIE, notnegativebatch. Combined permap: Bowl0,Concentric+14,Crag+8.5,Cube-12,Duplex0,Fract+4.5,KOTM-6,Sprawl-3.5,Treasure+14,Vortex0. Both explore/confirm runtime audits PASS10maps, allothercontrols equal exceptTL_STATS diagnostic counters. Decision leave_disabled_no_live in bump_window_decision.json: fresh gain did not repeat; repeatedKOTM/Cube losses. No default/map-specific exception. Conditional all-map live driver NOT RUN; fresh-positive prerequisite not met. Agent review receipt advance_bump_window_native.json written only after allquiet workclosed; coor movedto launch.

SJREF DIAGNOSTIC correction +null: original selectedFractured1987 witness has virtualstaginggoal[4,6,4], item_id null, NOTrealgem. First probe failed StopIteration ininputlookup BEFOREWorld/forecasts. Exactsource/log/preflight/failure preserved sj_backup_reference_probe_attempt1. Do NOT claimFracturedknownbadrealgemcausedresourceveto. Read-only40round trace audit thenfound4selectedREALgem cases, allKOTM (3explorebaseline,1confirmcandidate), plus1virtualFractured. Counts11461heldSJ /8217ordinarypredictedpickups. Originalaudit mislabelled map path(-6 instead of-5); retainedv1 files, corrected metadata/sourcejournalpaths in sj_backup_reference_exposure_all_maps_v2.json. USEV2. Original sj_backup_reference_real_gem_state.json has wrongmaplabel; correctedstate_v2.json containsKOTMdecision1005 actualtarget30860. Probe source updated tocorrectactualstatefile, secondpreflight sourcehash/procsreceipt passed whilecoor idle/no games/workers.
Second probe completed ~3s, source_unchangedTrue,56physicalcandidates underfull/window7, margin1, bothold/fixedchoose line2 pickup11/cost-142.32. NOknownbadordinaryinnewunbudgetedrootset, NOchangedrejections. Null result, does not reproduce suppressed use or actualolddecision (coldnavigator/nofullcontrollerhistory). Exactsource/helper/input/log/report archived sj_backup_reference_probe_attempt2. PS command exit1 fromtorchFutureWarning stderr/NativeCommandError but no traceback; sourceverified finalJSON andcompleteTrue arepresent, no rerun/overwrite. No nativepointsclaim. Latentguardfunctioncounterexample6tests remainsvalid; prototypeNOTenabled/unlaunched.

New future investigations NOTIMPLEMENTED: follow_time() physicalsecondgemestimate currently checks rollfailure andpossible_pickup, no explicit brakingbackup; maycreditunsafeoptimisticfollowup. Extra backup checks should preserve unknown outcomes/softfallback, notdeletecurrentplans. resume_forecast uses rollout.states[-1] +controlprefixcorrectly; leaving native_states unsliced is NOTa provenstart-statebug. Mainlanding_goal onlysetfor terminal_stop check; ordinarylandingprojection doesnotautomaticallyforcecommittedpostpickup controls. Do notassumeeveryjump islockedthroughlanding.
RE.tour usesrawvisible5 thoughglobal_goals fallbackseesallcurrentobserveditems7; nav.gems.plan_tour isfactorialmax5 exactwalk/climb/momentum/wallturn objective. ExistingSPAWN_ORDER fullpooltime-model alreadytestedglobalbad(Crag/Vortex), donotblindlyrepeat. Possiblelargerarchitecture DIFFERENT fromexistingSO: exactsamecurrenttourcost onallobservedgems usingprevious-two-node HeldKarp DP insteadfactorial, preservingterrainwalk/gap/climb/wallturn/stickycurrent costs. Needsproofsame5outputs/identities/ties and runtime, thenfreshfullscoretest; no codewritten. Priority remainsdeclaredlaunchtestuntilitcloses.

2026-10-27 06:03 active launch-only native pilot: KOTM137->140 (+3), Sprawl185->176 (-9), Bowl running. Driver5776/coor2exec9072 remainsoleowner; all other future scriptsunlaunched. No dawn cutoff, userhasnotstoppedgoal.
Prepared NEW independent follow_backup.py (NOTinstalled in current scorer): AST-transforms actual canonical follow_time only to record its existing finite next-gem arrivals, then reorders those time estimates after at mostexistingBACKUP_CHECKS braking checks resumed at the nextpickup. Same5+optional controllers/horizons retained. Known failed arrivals excluded from timing; unknown/ambiguous/skipped remainacceptable underoldbackupsemantics; iflimit reached, next UNTESTED arrival stillcounts. Ifallfinitearrivalscheckedandknownbad, returnNone =>existingtwo-level softfallbacktime, neverdelete currentplan candidates. Usesexistingbraking_backup deadlineexemption, soADDSboundedforecastcompute; .172mainallowancenotchanged. Mustmeasureliveholdsifwins. No newhardcurrent-plan gates or inventorypolicychanges.
Seven actual-follow-function isolated tests pass in0.2s: fasterknownbadlosestoslowersupported, allbad->no proventimewithoutdeletingfutures, unknown preserved, checklimitkeepsuntested, disabledbackupnoextra, ASTsourcedriftdetected, existingunrelatedwrapperNOTsilentlyreplaced. Prefixstate/control/proposed lengths atpickup verified, originalrolloutlistsunchanged. Installeronlyreplaces follow_time in canonicalglobals, retainsclassidentities; checks originalco_filename/globals torequireexplicitfuturewrappercomposition.
Prepared tools/follow_backup_bench.py and run_follow_backup_native.py withfresh219468731/786253419, complete10/.172s/two8worker games/gate.35. ConfirmpilotpositiveAND follow_backup_timing_changes>0. Waits final_backup_native_status complete. NOTin fixedcoordinator2 queue; don'tstartwaitingmanager duringother scoring. Counterscomparisons/checks/knownfailures/unknown/untested/timingchanges. No physicalforecastorpointtestyet, onlymockoutcomeswithactualcontrol-familyfunction. Thisaddressesunvalidated SECOND-leg optimizer times, distinct fromqueued final_backup whichchecks FIRSTplan afterTL re-ranking. No changes to activecanonical/source/default/checkpoint.

2026-10-27 ~06:44 sole active scorer still launch_native driver5776 undercoor2exec9072. Pilot10complete129.5->132.2 (+2.7), CI[-8.533,13.933], fullstart/GO/source/isolatedflag audits pass. Perpilot KOTM+3,Sprawl-9,Bowl-4,Cube+16,Duplex+4,Crag-31,Fract-8,Vortex+29,Treasure+28,Concentric-1. Confirmation317594681 underway: KOTM140->150(+10), Sprawl175->185(+10), Bowl running. NOdefaultpromotionyet.

CompletedCragreadonly selectedlaunch audit launch_cragmire_selected_extension_audit.json: baseline87launch-labelled decisions/1extendedroot; candidate184/6extendedroot,5pickups afteroriginalhorizon. Examplescandidate:heldSJ virtualdestination84replypickup vs56original/96evaluated; heldSS realgem69vs50/90; SJ realgem58vs56/96; SS66vs48/88. Countsareselecteddecisions NOTuniquetrajectories; differencesdonotprovewhy31points lost. Key structuralrisk: longconsumingfutures can getpickupbonus beyondordinary48replyhorizon while noacceptedplainpickup exists, grantingoldresourceguardreachabilityexemption. Existing continuation_time_price pricesextraonlywhenNOpickup; globaltime-price variantsalreadyfailed, donotblindlyrepeat.

Prepared matched_impulse.py (unscored, NOTinstalledactive): forheldSJ1/SS2 only, ifacceptedconsumingpickup>basesearchhorizon ANDnoacceptednonconsumingpickup: continueatmost2checkedresource-free ordinaryprefixes toconsumingforecastend usingcanonicalLandingContinuation+SequenceProposalthroughknownnextgem. Addonlyactualpickup withoriginalphysicalacceptance/noresourcesconsumed/passingsameexistingbrakingbackup. No inventedevents/newspawns, no newhardgate/blanketallinventorytimingrule. Extensionbudgetusesmin(originalremainingdeadline,now+.04), expiredoriginalnotextended; backupsretainexistingdeadlineexemption(addboundedcompute). Reapplyoldmargins(no revival) andexistingactualtimecomparison onlyafterthisnewmatchedordinaryevidence; appliesalsoSSinthisnarrownew-evidencecase. Persistent-effecttypes3/4/5unchanged. Existingplainpickupcaseunchanged. OriginalSGwaste18vs34alreadyguardedbeforehelper. ThisisNOTtheearlierfailedbroadall-inventoryguard.
Eightisolatedactualguardtests pass: oldunmatcheduseaccepted thenverifiedfasterordinaryveto, unchangedexistingplainpickup, inventory5unchanged, missing/failedbackupkeepsreachability, pendingconsumptionnotplain, originaldeadlineexpiredstays, exceptionrestoresdeadline, limit2/oldrejectionspreserved. No realphysicalforecast/pointsforhelperyet. Matchedthroughcontext infersrealgemitemwhenroot target_itemNone. HelperwrapsS.search+RE.searchaliaseswithsignaturepreserved, no staleclass/module replacement.
Prepared tools/matched_impulse_bench.py (ownflagsmatch-impulse-horizons/complete-launches) +run_matched_impulse_native.py fresh664193287/145728963. BaselineproductionflightflagsOFF; candidateLAUNCH_COMPLETEplusmatchedcounterfactuals; roll/short/FAST_SUPPORTED/SS_RIDE remainOFF. ThuscoherentJOINTfeature, nothelper-alonecausaltest. Mainallowance.172/2games8workers/.35gate,full10/source/initial/GO/runtime/protocol audits. ConfirmpilotpositiveANDnewusevetoes ORselectedmatchedordinary decisions. Waitslaunch_nativecomplete; UNLAUNCHED, NOTincoor2queue. FirstgeneratorfailedassertBEFOREwriting(driverprotocolname); correctedagainstactuallaunch_protocolschema, ownargfields/protocolsourcesvalidated. Do notstartwaitingmanagerduringcurrentconfirmation.

Prepared query_route.py +tools/query_route_bench.py +run_query_route_native.py fresh935714621/412896357,waitsblockedcomplete,UNLAUNCHED/notincoor2. Exactly12routeLOADreferences inoptionalpowerup query/speculativeblock useprivatecopy; selecteddetouroriginalroutecommitprotected. ReverseASTdelta reproduceswholeoriginalpick. PrivatequerycopycounterandpreexistingnonnullprimarywaypointdifferencesafterFAILEDquerycounter(doesnotclaimwaypointactuallyusedlater). No newforecasts/gates/costchanges; ordinaryprimaryroutehistorypreserved. Fiveactualdetour-block tests passwithfaithfulheld-corner route: oldfailedqueryloseshyst, privatequerykeeps, selecteddetourstillcommitsoriginal, emptyitemsunchanged, contractdriftreject. WrappergenerationownflagrenamemismatchcaughtBEFORElaunchandfixed; staticownfieldlintPASSquery/follow/timer/SJref/livebump/matchedwrappers. Needclosed-gateactualmodule/class/worker/preflightbeforeanynewscorephase. No source/default/checkpoint changes toactiveproduction.

2026-10-27 06:57 launch confirmation active: KOTM140->150(+10), Sprawl175->185(+10), Bowl137->139(+2), Cube167->157(-10), Duplex138->138(0), first5fresh+12. Cragmirecandidatefinished,baseline~35sleft. Onlydriver5776/coor2exec9072active; no native/live extra jobs.
Prepared CONDITIONAL launch actual1x follow-up (UNLAUNCHED): tools/launch_completion_live_bench.py and run_launch_live.py, all10maps seeds731948263/286175943, port9472,8workers,lead5, sequential only. Requirespositivecomplete20native mean ANDpositivefreshnative mean. HoldsCOOR2atlaunch_native review, advance_launch_native.json MUSTremainABSENT until anylivebatchcloses. Sameordinarychargingphase; existingneutralnaturalGOpreparer fornonchargingmaps, nomidroundresets/prespinnoncharging. Sourcepreparer+wrapper+checkpoint includedinfrozenliveidentity; wrapperrecordsACTUALSflightflagsafterWorldloads/cleansitsworkersifwrong, LAUNCH_COMPLETEFalse/True, otherfourflightflagsFalse.
Livecomparator checksinitial/layout/RNG/actualGO, samecanonicalinputsintersectnativepreparedsourcehashes(normalizedpathkeys), sourceidentityfrozenacrossallmaps/seeds, expectedplannerflagdifferenceexactlyoneappended--launch-completion, actualruntimeflightflags, samebudget/physics/workers. Driverhash/copiedsource/advanceguard. PositivefirstallmaplivemeanANDlaunchswitchactivity triggersall10freshconfirmation; nofirstmapcherrypick. FIVEpureevidencetestsPASS includingactualruntimeflagfalse despiteprotocoltrue. No physical1xpreparerfixture/game launchedyet. AtCLOSEDqualifiedlaunchgate, runexistingtools.test_native_worker_bench testsandquietprocesspreflightbeforelive; preserveallfailedpreparedattemptsifany. Neverrunmatched/SGprobesduringquietlive. NOdefaultpromotionyet.
ALL new prototypes(query_route/follow_backup/matched_impulse/timers/SJref)remainunlaunched/outsidefixedcoor2. Coor2nextstrictreuse(expectedskip), SJoriginalaudit, poweruporder, finalbackup, blockedorder. Anyreprioritizationmusthappenatclosedreviewwithsingleowner/source/seedreceipts, not editrunningdriverorlaunchwaitingmanager. Goalcontinuousuntiluserstop, nomorningcutoff.

2026-10-27 07:31 CRITICAL ACTIVE OWNER: launch_live actual1x campaign is RUNNING, driverPID32924 execsession92914, port9472,8forecastworkers,lead5, SEQUENTIAL. launch_live_status.json authoritative. launch_live_takeover/active_owner.json recordscommand/session/PID. COORDINATOR2 PID25616 was VERIFIEDIDLEATlaunchreview,0children/0games/0workers, source/plan/status/jobclose archived underlaunch_live_takeover thenSTOPPED. Its status.json STILLsaysawaiting_agent_review but PROCESSISDEAD. DoNOTwriteadvance_launch_native orstartANYcoordinator/native/probe/sim/forecastuntilALLliveworkcloses. No other queue manager active. Nextcoordinator(notbuilt)musthard-gateonlivecomplete/explicitclosed-failure review. ContinuousgoalACTIVE; userhasnotsaidstop, no morning cutoff.

Launch20nativeFINAL: pilot129.5->132.2(+2.7); fresh131.7->135.4(+3.7); combined130.6->133.8(+3.2),20valid/0invalid,CI[-3.736,10.136]. PromisingOBSERVEDreplication, NOTstatisticalcertainty/liveproof. PerMap Bowl-1,Concentric-11,Crag-8,Cube+3,Duplex+2,Fract+8,KOTM+6.5,Sprawl+.5,Treasure+22.5,Vortex+9.5. Native_decision.json leavedefaultunchangedandqualifiesquietall-map1x. Allmotion/source/actualGO/worker/isolatedactualflagsdriverauditspass; closedreceipt exit0driverunchanged.

Startpreflight: existingtools.test_native_worker_bench5tests +test_launch_live_audit5testsPASSatclosednativegate. Addedcheck_launch_live_starts.py; actual4UNSCOREDnative1xgames(twoindependentCragmire,twoTreasure selectedfromnative neutralGOmode) no modeldecisions/forecasts/grants/restores, seed812736419 port9476. Bothmapslayout/initial/actualGOmatch, source_unchangedTrue, statuscompletepassed, session97635exit0, allgamesclosed. Evidence launch_live_start_preflight. Native1xscoringpreflight thenCIMverifiedNOotherowner/game/worker, storedlaunch_live_takeover/scoring_preflight.json, launchedlivedriver ONLY.

Liveplanall10freshseed731948263, alternatewhicharmfirstbyMAPindex. IfwholebatchmeanpositiveANDlaunchswitchactivity, all10fresh286175943 reversedfirstarm. Onebatch~80minutes, secondanother~80; no inferredmorningdeadline. FirstbaselineKOTMROUND148points/0falls/holds0.3s/p99plan.167s, closed07:29:47; candidateKOTMstartedGO07:30:09. Pairnotyetcomplete/audited, doNOTclaimlivegain. ActualWorldflightflagsrecordedlaunch_live_runtime.json(BASECOMPFalse,otherfourFalse;CANDTruecheckedbeforegame). Sourceidentityfrozenallmaps/seeds+sharednativehashes+checkpoint, actualGO/layout/source/flags/workload checksperpair. Sourcefile/drivers frozenwhilecampaignrunning.

Livewrappertraceenhancementbeforestart: with--trace, BOTHarmsgziplogplanner_choices.jsonl.gz(fullactualplanner input includingpredictedmsg,action,selectedinfo) andimpulse_comparisons.jsonl.gz(alreadycomputedheldSJ/SS rootsearchcandidateINFOS+input+target), NOnewforecasts. CanonicalS.search+RE.searchaliastransparentwrappedandController.actionwrapped; globals/classesnotreplaced. gzipcompresslevel1, measuredI/Ooverheaddeclaredsamebotharms; unknownnonJSONtypesexplicitlytagged, notsilentloss. Runtimeflagsandloggingprotocolcheckedbycomparator. Filesmayappear0byteswhilegzipbuffered; DO NOTpoke/flush viaadditionalprocess. Fullcandidateinfoimprovesforensicswithoutcoldunbudgetedreplay. Nevereditlaunch_completion_live_bench.py/run_launch_live.py/native_worker_bench.py/nav/core orcheckpointduringlive. Futureprototypesunderreports/toolsUNLAUNCHED; avoidevenlightnewtestswhilequietscoringunlessrequiredread-onlyaudits.

### 2026-10-27 07:46 MDT — live sweep in progress; counter caveat
Only active scorer remains launch-live PID 32924, session 92914, port 9472. Coordinator 2 is retired; do not advance its launch gate or start future drivers while live scoring runs. Completed audited live pairs: KingOfTheMarble 148 -> 139 (-9); Sprawl 176 -> 166 (-10), both Sprawl arms zero falls. Bowl baseline now running. Production defaults unchanged.
Read-only inspection found realtime.play_round emits counts=dict(controller.counts.most_common(12)). Therefore absent launch_switch in a summary does NOT mean zero switches, and launch_switch itself tracks replacement of a held launch plan, not general launch-completion activation. Existing frozen live driver's positive-mean confirmation gate uses this censored switch count. Do not edit running sources. At closed campaign review, audit actual planner_choices/impulse_comparisons before interpreting activation; preserve original gate outcome and any prospective supplemental decision separately. No extra forecasts or tests were run during these rounds.

### 2026-10-27 08:09 MDT — historical launch ablation audit (read-only)
The earlier phase_flight_plan.json explicitly tested roll-launches + launch-completion + land-short + fast-supported together on Sprawl. Earlier production_worker_variants.json declares ramp (roll + completion) and ramp_short (roll + completion + short), but the handoff says that config was prepared and NOT run (line 14257), and checked_screen contains no ramp/short result filenames. Thus the failed four-flag flight bundle does not establish that roll+completion without FAST_SUPPORTED fails. Treat that narrower combination as an unscored candidate requiring its own prospective all-map native/full-round experiment, not as a demonstrated improvement or proof that it has never run anywhere. Current live completion-only sweep remains frozen and sole active scorer; no new experiment was launched from this observation.

### 2026-10-27 08:29 MDT — six live pairs closed, continuous goal active
User asked for overnight recap; answered candidly in commentary and continued work (no stop requested). No new overnight default promotion. Completion-only native20 +3.2 remains experimental. Current live deltas: KingOfTheMarble -9 (148->139), Sprawl -10 (176->166), Bowl +5 (138->143), CubeIsle +7 (125->132), Duplex 0 (135->135), Cragmire +11 (97->108). Six passed pair audits; equal-map mean +0.6667, unfinished. Candidate falls: KOTM1/base0, Sprawl0/0, Bowl0/0, Cube4/3, Duplex0/1, Crag3/1. Fractured baseline closed, candidate running. Only owner remains PID32924/session92914/port9472; no competing games, tests, probes or forecasts started.
Prepared audit_launch_live_selections.py under reports, UNRUN: refuses large gzip parsing until launch_live_status says complete and complete campaign comparison exists; refuses overwrite; scans selected launch labels, extension extent, pickup beyond original horizon, and separately root held-impulse searches. Logs are planner inputs (can be ahead), not proof all selected actions executed; baseline can have extensions too, no causal score claims. Validate this pure reader at closed review before use. Existing launch driver, wrappers and canonical sources untouched. No new coordinator yet; retired coordinator2 must remain idle/dead, no advance_launch_native marker.
