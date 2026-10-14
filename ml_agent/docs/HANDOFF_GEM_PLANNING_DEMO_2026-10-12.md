# Map-aware gem planning demonstration

Started 2026-10-12 (America/Denver). The operator authorized continuing the general
mechanics/map-aware planning plan with a bounded live gem-collection demonstration.
Policy/PPO training remains paused. Record every bugfix and medium/major change here.

## Objective and scope

Connect exact collision geometry, full live physical state, proposed controls and
actual game gem pickups. The demonstration must plan from current observations,
execute ordinary legal controls, and measure native outcomes. It must not use future
physical observations as candidate inputs, teleport to collect gems, or claim a
general landing guarantee from selected easy cases. Controlled setup teleports, if
used, must be labeled and separated from spawn-to-gem results.

The analytical general mechanics model remains the prediction foundation. Mechanics
include rolling, spin, impacts, jumps and powerup modifiers. Ground routing may
propose candidates; it must not forbid physics-based alternatives. Full-round policy
integration, learned fits and long-term training are outside this demonstration.

## Starting evidence and gates

- Previous handoff: [HANDOFF_MARBLE_DYNAMICS_2026-10-11.md](HANDOFF_MARBLE_DYNAMICS_2026-10-11.md).
- Fifteen geometry/native regressions previously passed. Complete-control fixtures
  agree closely, but broad-map long-contact trajectories retain multi-unit tails.
- Native camera setters and delayed bridge actions are separate inputs. The intended
  policy interval is 64 ms; the earlier collector used 16 ms. Live state snapshots
  themselves consume a bridge reply unless a synchronized observation is added.
- Item pickups, respawns and script-driven changes need explicit observation/event
  handling; they are not continuous mechanics.
- Workspace already contains prior uncommitted changes and staged snapshot deletions.
  Preserve them; no reset, commit, or artifact cleanup is authorized by this work.

## Work log

- Inspected the existing bridge, general predictor and collection helpers. The first
  implementation gate is synchronizing full native state and pending input with
  each observation, then verifying the delayed-control contract in 64 ms lockstep.
- No training or optimizer has started.
- Added opt-in `DYNAMICSLIVE` telemetry in the separate native probe and bridge
  observation. It preserves full precision state, frame clock, the already
  collected native Move, pending bridge reply and powerup aim hold. Normal
  policy observations are unchanged unless this local probe mode is enabled.
- Fixed the live snapshot clock contract: script observations can occur 1-2 ms
  before the frame's physics call. The snapshot now exposes the native frame
  clock instead of assuming observation time is the physics time.
- Implemented `bridge.py` translation of the existing delayed controls and
  native F32/per-axis 1/16 movement quantization. New reply camera applies one
  frame later; movement/jump/use applies two frames later. This changes no
  native input behavior. Initial live timing archives are retained, including
  the first archive without the improved frame-clock/pending-Move channels.
- `live_timing_v2_20261012/Concentric_Hunt`: 44 independently initialized
  five-step replay windows, 220 applied control intervals, zero control
  mismatches or prediction failures. Maximum p/v/spin errors are 7.97e-9 u,
  3.56e-8 u/s and 4.62e-7 rad/s. This is a normal steering/jump development
  fixture; it does not establish powerup/event timing or general landing safety.
- Added a bounded gem planner prototype (`gem_demo.py`). It proposes rolling,
  steering variations, neutral input, braking and varied jump timing using the
  shared geometry/physics predictor. Direct-gem candidates accompany routed
  candidates. Game score changes, not predicted proximity, count as collections.
- First natural-spawn Concentric pilot stopped honestly after five decisions:
  no candidate met the terminal-support/contact-ambiguity checks at a 1.536 s
  horizon. Live one-step position error stayed below 4.78e-9 u. Preserve this
  failed attempt in `dynamics_reports/gem_demo_20261012`; investigate longer
  landing horizons and add explicit active-braking candidates rather than
  weakening ambiguity rejection. No gems collected in that first attempt.
- The longer-horizon KOTM pilot collected two actual additional game points in
  two confirmed pickup events from a natural spawn. No teleport, fixture restore,
  item grant or gem replacement was used in these gem pilots. Initial pad/warmup
  pickups are excluded from the reported increment. Other map attempts remain
  retained even when their short decision budget collects nothing.
- Diagnosed the major Concentric timer outlier: a nearly tangent ramp contact
  immediately preceding a floor impact flips native slide/rebound behavior.
  Existing model ambiguity detection flags this sequence. Native detailed force
  traces are archived in `contact_diagnostics_demo_20261012`.
- Ran ten **native** repeats/perturbations of that development fixture. Initial
  p/v vary by at most 2.10e-5 u / 2.00e-5 u/s, yet native peak heights span
  14.0800-18.6018 u and endpoints differ by 5.0716 u after 1.44 s. Baseline repeat
  is identical. This confirms a physically sensitive branch in the game itself;
  an averaged nominal trajectory is not a safe landing forecast. Keep this
  selected experiment separate from population uncertainty calibration.
- Added three native-derived control/parser regressions; all eighteen geometry,
  mechanics and bridge regressions pass.
- Added live active-item IDs, positions, values and native world bounds to opt-in
  telemetry. Candidate pickup scoring uses the game's swept item-query AABB
  (marble radius + 0.2u), with actual game eligibility/score still authoritative.
  Also prices predicted pickup delay and adds continuous end distance to reduce
  coarse route-field plateaus. No policy/reward/checkpoint was changed.
- 64 ms command-timing fixtures for Super Speed, Super Jump and Helicopter each
  passed 44 five-step windows with zero native movement/camera mismatches. Maximum
  position errors: 7.27e-7 u, 1.95e-8 u and 8.22e-9 u respectively. These controlled
  physics fixtures use explicitly labeled item grants; natural gem demos do not.
- Audit found a bug in the prototype external-event classifier: native JSON's
  short F32 decimal spelling differs from Python's widened F32 value, so exact
  dict/radius comparisons falsely labeled ordinary intervals as parameter events.
  Consequently the first pilots' live error gate was bypassed. Their actual errors
  are retained and audited, not claimed to have passed that gate. Added
  `events.external_changes` with native F32 equality, a regression demonstrating
  both round-trip equality and genuine parameter/inventory changes, and corrected
  the independent audit. Apply this classifier to the next live pilot.
- The corrected v2 audit scores 251 five-step (320 ms) replay windows over three
  maps from one initial state each, without future physical/camera observations.
  Zero movement mismatches; maximum position errors below 2.0e-6 u. No external
  windows were excluded. Two KOTM pickups/points match the native score increase;
  Concentric and Sprawl collected zero within their 100-decision budgets.
- The first candidate generator sometimes prefers staying still on Concentric.
  Accurate physics alone has not solved longer-range action search. Preserve that
  result; add slower approach proposals and test committing several selected
  replies before replanning. Planner performance is roughly 0.6-1.6 seconds per
  search in these pilots, so this Python prototype is not yet real-time control.
- Added matched-world short-plan execution (`commit_demo.py`): restore a captured
  settled natural spawn/world only between experimental branches, validate pose,
  inventory/group state and score, then execute four selected replies (256 ms)
  before replanning. Actual live errors are now checked with the corrected F32
  event classifier. The comparator is a simple routed proportional controller,
  **not the trained navigation policy**. Each branch has a fixed decision budget;
  STOP is cooperative, and all game processes are owned hidden instances.
- First matched KOTM fixture: both branches collected one actual point. Controller
  reached it at decision 28; physics planner at decision 26. Controller stopped
  at decision 47 after a genuine native/model position mismatch of 0.05467u;
  the planned branch stayed within 0.000214u over four-reply commits but stalled
  on the next gem through decision 160. This is not evidence of overall superiority.
  Native recorder confirms that controller error is inside physics, not a bridge
  snapshot warp. Preserve this additional KOTM contact failure for diagnosis.
- Matched Concentric fixture: proportional controller fell after 33 decisions;
  physics planner remained stationary through 160 decisions. Both scored zero.
  The conservative checks prevent one bad candidate, but candidate search is still
  weak. Neither failure is counted as a successful navigation demonstration.
- Fixed two prototype issues exposed by committed execution: braking at almost
  zero speed passed an unnormalized desired acceleration into the legacy joystick
  helper and could become an unintended launch; braking now returns neutral input
  below that helper's 0.1u/s threshold. Slower proposals previously used a different
  cost expression from other candidates; all now share `candidate_cost`. Added a
  stationary-braking regression and retained prior artifacts. Running one final
  bounded KOTM matched fixture after these fixes.
- Corrected fixture after braking/cost fixes still exposed an architectural gap:
  the prototype picked the nearest gem before asking whether any candidate made
  useful progress toward it. It could remain still indefinitely although other
  gems existed. Added general planner feedback to gem selection: if the current
  target has no supported collection/progress candidate, evaluate alternatives
  using the same dynamics, prefer predicted pickups/value/time and feasible route
  progress, and stop explicitly if none qualifies. No gem is permanently declared
  unreachable and no map/trick-specific route is added. Testing a bounded matched
  KOTM fixture with this correction; earlier stalled attempts remain preserved.

## Current status

The bounded live demonstration is complete. Final matched KOTM fixture collected
two gems/points in 59 decisions with the physics planner, versus one gem/point and
a fall at decision 58 with the simple routed controller. This is selected-case
evidence, not a trained-policy benchmark or a general landing guarantee. Earlier
Concentric/Sprawl failures and stalled KOTM fixtures remain preserved. Policy/PPO,
encoder training and dynamics fitting remain paused/not started. All owned games
have closed; final viewer and regression verification passed.

## Final result and practical limits

- Final attempt: `dynamics_reports/commit_demo_v3_20261012/KingOfTheMarble_Hunt`.
  Both branches start from the same replay world captured at a settled natural
  spawn. Pose mismatch is 5.86e-7u, velocity mismatch 2.01e-10u/s; score and item/
  group states match. Restore occurs only between branches, never as a rescue.
- Planned branch: two confirmed one-point gem pickups, 59 decisions / 3.776s
  simulated play, no fall. Native recording contains two jump-trigger intervals,
  both with support; these are support proxies, not instrumented impulse counts.
  No powerup activation occurs in this successful fixture.
- Simple routed-controller branch: one confirmed point, then an OOB/fall event at
  decision 58. Both scores agree with native UI score differences. This controller
  is an intentionally simple comparator; the current trained navigator has not
  been benchmarked against the prototype or replaced.
- Corrected native input/event pipeline is used in the final committed fixture.
  Maximum committed-position error: 2.94e-7u. Independent five-step replay audit:
  55 planned windows, zero movement mismatches/excluded external windows, maximum
  position error 7.57e-7u. Controller audit: 53 continuous windows; one crossing its
  fall is disclosed/excluded. Five-step audits use recorded actual replies as
  exogenous commands and one initial physical state, never future state/camera.
- Planning horizon is 40 replies / 2.56s. Four replies / 256ms are executed before
  replanning, interrupted by a real pickup, external event or error/fall gate.
  This does not certify the unexecuted tail of every candidate's landing.
- Gem selection now asks the general physics planner whether the current target
  has a supported collection/progress candidate, then considers other live gems
  if it does not. It uses value/time/progress to rank alternatives. Ground routing
  proposes aims; direct trajectories, different steering/throttle/jump timings
  and held-item use alternatives are evaluated by the same full-mesh mechanics.
- Final median search time is 1.18s (some alternative-gem searches ~3.1s) per
  256ms of simulated execution. Lockstep makes this inspectable; it is not yet
  suitable for normal real-time multiplayer control.
- All 14 attempted episode/branch results, including zeros, are listed with source
  fingerprints in `dynamics_reports/gem_demo_session_20261012.json`. Earlier failed
  external-event gating is disclosed above and independently audited, not erased.
- Separate review viewer: [viewer/gem_demo/index.html](../collision_maps/viewer/gem_demo/index.html).
  It uses exact reviewed NPZ triangles and labels the five-step replay basis.
  The original ten-map geometry/mechanics viewer remains intact.
- Native probe EXE remains `marbleblast_geometry_probe.exe`; shipped executable,
  policy parameters/rewards/checkpoints and training processes are unchanged.

## Files and reproduction

New modules: `bridge.py` (native queued control translation), `events.py` (F32-safe
external changes), `live_timing.py` (controlled 64ms evidence), `gem_demo.py`
(natural-spawn prototype), `commit_demo.py` (matched short-plan execution and gem
feedback), `native_sensitivity.py`, `audit_demo.py`, `demo_viewer.py`, `test_bridge.py`
and the native-derived `fixtures/live_bridge.json`. Native include adds read-only
live state/pending-Move serialization. `mlAgent.cs` adds guarded opt-in telemetry
and active-item bounds; `nav/protocol.py` parses its optional field. `record.game`
accepts an explicit observation interval. Browser checker accepts a separate
geometry root for the standalone viewer. Default policy behavior is unchanged.

```powershell
$taskPython = 'C:/Users/doug/AppData/Local/Programs/Python/Python39/python.exe'
$env:PYTHONPATH = 'ml_agent'
& $taskPython -m unittest nav.marble_dynamics.test_bridge nav.marble_dynamics.test_native nav.collision.test_queries
& $taskPython -m nav.marble_dynamics.audit_demo ml_agent/dynamics_reports/commit_demo_v3_20261012/KingOfTheMarble_Hunt --output ml_agent/dynamics_reports/commit_demo_v3_replay_20261012
& $taskPython -m nav.marble_dynamics.demo_viewer --trajectories ml_agent/dynamics_reports/gem_demo_v2_replay_20261012 ml_agent/dynamics_reports/gem_demo_v3_replay_20261012 ml_agent/dynamics_reports/commit_demo_replay_20261012 ml_agent/dynamics_reports/commit_demo_concentric_replay_20261012 ml_agent/dynamics_reports/commit_demo_v3_replay_20261012
$env:PYTHONPATH = 'ml_agent;' + (Join-Path $env:TEMP 'pq_checks/collision_maps_20261011/browser_deps')
& $taskPython -m nav.collision.browser_check --directory ml_agent/collision_maps/viewer/gem_demo --geometry-directory ml_agent/collision_maps
```

For another bounded live matched test, use a **new output** (port 9054 must be free):

```powershell
$env:PYTHONPATH = 'ml_agent'
& $taskPython -m nav.marble_dynamics.commit_demo --maps KingOfTheMarble_Hunt --decisions 120 --gems 2 --output ml_agent/dynamics_reports/commit_demo_NEW
```

No policy is loaded or trained by this command. An output `STOP` file cooperatively
ends a branch. It always closes its owned hidden game. Native evidence is archived
under each report directory; raw TEMP files are not the durable authority. Generated
reports are ignored by Git and must be copied/regenerated with the workspace.

## Next substantive work

1. Validate the final gem-feedback selector on new reserved starts on other maps;
   preserve failure cases. Compare against the **frozen trained navigator** under
   matched rules/worlds, including falls, pickups, simulated time and compute time.
2. Improve general action proposals/search (including frozen navigator proposals),
   and prevent repeated replanning from postponing a future jump/use indefinitely.
   Do not build named trick rules or declare a gem unreachable from one failed search.
3. Resolve/flag the newly archived KOTM 0.05467u contact mismatch. Extend native
   perturbation checks beyond the selected sensitive ramp/floor development fixture;
   use contact/landing clearance and continuation, not a nominal averaged forecast.
4. Complete dynamic event coverage: item availability/eligibility/timing, OOB volumes,
   Blast/Mega/scripted radius/gravity changes, respawns and moving actors. Live known
   state is reanchored on observed item changes; candidate futures do not yet model
   all of those events. Restrict the API to the verified local 64ms, DELAY 0,
   SSYAWFIX-off lockstep contract until other regimes are tested.
5. Measure batch/compiled/learned acceleration against this native-labeled foundation
   before real-time deployment. Map encoder and learned dynamics remain untrained.
   Policy integration/long-term PPO requires a later review; it has not started here.

## Final verification follow-up

- The first separate-viewer browser run passed all three map pages and their controls,
  then exposed a checker assumption: the original gallery had ten table rows, but
  the pilot gallery has three list entries. Updated the checker to verify the exact
  expected map links and either table rows or list entries against the summary.
  This preserves the original ten-map check while supporting the separate gallery.
- Visual inspection also found that mesh/metadata download links assumed the
  original viewer directory depth. The shared generator now computes their actual
  relative paths; the gallery label is neutral. Added checks that every map-page
  header link resolves to an existing local artifact. Exact geometry was unaffected.
- Final verification: 20 geometry/native/bridge regressions passed; Python modules
  compile and `git diff --check` passes (existing line-ending warnings only). All
  three browser map checks passed with zero JavaScript errors, exact geometry
  coordinate hashes, matching native/predicted trace arrays, scrub/focus controls,
  seam marker controls, visibility toggles and working local links. The gallery
  has the three exact expected links. No owned probe, trainer/evaluator or browser
  helper remained after process inspection.
- Explicitly opened the final successful KOTM physics-plan recording (`source`
  `commit_demo_v3_replay_20261012/physics_plan`, trial 54); both plotted arrays
  match, six samples cover 320ms, max position error 2.73e-7u, no JavaScript errors.
  Screenshot: `collision_maps/viewer/gem_demo/KingOfTheMarble_Hunt.final_planner.preview.png`.
  In the map page choose that source under **Recording**, then **Focus trajectory**
  and scrub **Time**. These are independently replayed short windows around
  collections, not a single open-loop prediction of the full episode.
- Separate native EXE SHA256:
  `084ebc5cce1444f032788282c66a35f95a728cd99c59ef4f75daa40175807926`.
  Session report now includes final viewer checker/generator/fixture fingerprints
  and browser verification. Per-attempt archives and their original fingerprints
  remain intact. Central training log entry 40.101 links this handoff.


## Continuation

The authorized reliability/frozen-navigator follow-up is recorded in
[HANDOFF_PLANNING_RELIABILITY_2026-10-12.md](HANDOFF_PLANNING_RELIABILITY_2026-10-12.md).
The earlier KOTM mismatch is fixed, future jump/use commitment and broader search
are implemented, and fully corrected fresh-seed comparisons tie A2 on KOTM (5/5)
without evidence of overall improvement. A remaining distinct-ramp force tie and
real-time search latency remain explicit. New replay gallery:
[reliability/index.html](../collision_maps/viewer/reliability/index.html).
