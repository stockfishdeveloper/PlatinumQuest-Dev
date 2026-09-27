# The big questions: general-purpose navigation from a map's terrain

Started 2026-09-26 with the operator; decisions reconciled 2026-09-27 after the architecture reviews.
This document records the goal, evidence, current decisions and remaining questions. The
[kotmjump design and implementation plan](KOTMJUMP_NAVIGATION_DESIGN.md) is authoritative for the agreed
experimental architecture and implementation gates. [PHYSICS_PLANNER_PLAN.md](PHYSICS_PLANNER_PLAN.md)
is the earlier physics proposal; [HANDOFF_NAV_TRAINING.md](HANDOFF_NAV_TRAINING.md) remains authoritative
about the deployed navigator. Historical alternatives below are labelled as such; they are not pending
instructions or requests for renewed approval. These documentation decisions do not implement runtime
changes, start training or change the preserved baseline.

## The goal

Given a map's verified collision geometry, the marble's physical state and the game's rules, an agent
that plays Hunt on any map the way a strong human does: it understands what the marble can physically do, finds
the fastest way to each gem (rolling, ramps, gap jumps, drops, and later powerups), and executes it. Nothing is
written for a particular map. A supported new map should need only its verified geometry/material artifact
and live objectives, without handwritten routes; frozen-weight transfer must be demonstrated.

## Why the current approach will not get there

* **The score gap is route, not speed.** On King of the Marble (KOTM) the model scores about 154 at 3x; the
  human's best rounds are 172 and 177. Per gem spawn the human covers 45.1 u in 5.15 s and the model 50.7 u in
  5.90 s, at the same average speed. About two thirds of the extra distance is path straightness: the human
  crosses holes with 12-15 gap jumps a round, the model with about 1.6.
* **More training on the current setup does not raise the evaluation.** No checkpoint trained after 26113 has
  beaten it at the 3x eval, while the training-round scores stay flat.
* **The jump knowledge is hand-built and KOTM-shaped:** a flat-ground landing predictor, a 3 u minimum gap
  (a KOTM hack), a "gem on the other side of the gap" gate, a fixed jump bonus, and a route graph with a fixed
  list of jump edges that assumes cruise speed and any approach direction. The policy never sees the route; the
  reward only pays for it after the fact.
* **The terrain processing has map-specific failure modes.** KOTM's walk grid had phantom holes (solid floor
  marked missing). On Sprawl, 2,133 of the 2,150 cells flagged "too steep" are really the cells beside ledges
  and walls, because slope is measured across two neighbouring cells.

## Principles (standing operator rules)

* Everything must transfer to other maps: no map-specific hacks, no geometry gates, no hand-coded oracles.
* The game engine is the ground truth. Physics is measured from the engine, not modelled by hand.
* The reward pays game rules only: gem points, plus a small fall marker.
* Judge every change by real rounds (8 rounds at 3x, deterministic), never by the training curve.
* Verify a new map's terrain in the game before training on it.

---

## Q1. How does the agent learn the physics?

*What will the marble do from this state if it does this?*

**Direction agreed:** learn it offline from the engine, then use the learned model cheaply during play. No
engine simulations during a round (that plan, v0, was rejected).

1. Start with a measured pilot dataset from verified game maps. Record actual physical/contact state and
   legal control sequences, including rolls, jump commands, air-input switches and post-landing controls.
   Validate teleport/spin restoration against naturally reached states before scaling collection.
2. Train a separate step ensemble and direct flight head. The latter starts **before the jump command**,
   predicts delayed/failed takeoff and a timed trajectory sufficient to query floating-gem pickup, then
   landing and an action-conditioned continuation. A landing point alone cannot prove pickup.
3. Use frozen learned predictions for search and feedback execution. Initially, airborne corrections use
   bounded step-model rollouts from actual state; a jump-command-only head cannot accept arbitrary airborne
   starts. Measure full planning latency from the start and gate it at M5; the old 1 ms batch estimate is
   not an end-to-end measurement.
4. Compare predictors against engine trials by situation and horizon. Disagreement schedules offline data;
   calibrated reliability decides acceptance. A drifting long step rollout need not veto a validated flight
   head, and agreement alone is not evidence of correctness.

**What we know**
* Spin can be read and set in a live game, and the marble radius is 0.19 u. Spin is in the navigator's input
  since NAV_OBS_V6, though the current policy barely uses it yet (real against zeroed spin: +0.0 +- 2.2 points).
* In the air, the held direction adds a constant acceleration: 5 u/s^2, or about 6.8 with our diagonal-keys
  steering. A flat-ground jump lasts 0.74 s, so air control moves the landing up to about 1.9 u in any
  direction; over a 1.5 s drop it is about 7.6 u. The landing is a region, not a point.
* The human's air input during real jumps is mostly forward or forward-diagonal early in the flight (about
  70 %), sideways about a quarter of the time, and often braking in the last third. Across all flights the
  input changes mid-air about 70 % of the time.
* With no key held on the floor, the engine brakes the marble, so data must hold the intended input throughout.

**Decided for the experiment**
* D1: condition on control sequences, including mid-air switches; constant holds are initial examples.
* D2: record and model continuation from the start under explicit controls, including the exit after pickup.
* D3: a separate ensemble, frozen during evaluation; no joint PPO/dynamics training initially.
* Include rolling as well as jumping. Track joint geometry, motion and control coverage, including speed,
  heading, spin, contact, edge offset and switch timing. Many slow trials do not cover a fast approach.

**Still empirical:** sufficient data/model size, trajectory/event resolution, calibrated error limits by
situation, and when unusual edge hits can be executed reliably. No claim of coverage follows from map count
or aggregate prediction accuracy alone (design sections 4-5).

## Q2. How do we represent the terrain so this generalizes?

**Now:** a height stack at 0.5 u resolution with up to 4 floor levels per point, generated from the map's
geometry. The navigator sees two 32 x 32 crops around the marble (16 u and 64 u across) and 16 edge rays.
Ramps are stored correctly: Sprawl's ramps are exact 18.3 degree planes, and the rays follow any surface
rising less than 1 u per 0.5 u step, so a ramp reads as floor, not as stairs.

**Known problems**
* Slope computed by differencing neighbouring cells turns every ledge into a fake steep strip.
* A 2.5-D height stack cannot represent overhangs, tunnels or ceilings.
* A 0.5 u grid aliases fine geometry; a smooth ramp and small stairs of the same average slope look alike.

**Decided:** export actual collision surfaces/materials and verify each contributing map in the engine.
Start the predictor with coarse/fine height rasters plus normals/materials from the same surfaces/layers;
preserve the complete mesh for later local triangle queries. Normals fix slope differencing but cannot
recover missing walls, bevels or sub-cell lips. Test edge contacts at varied offsets, speeds and angles.
The flight head needs context covering the flight and continuation, not only a fine crop at launch.

**Still empirical:** whether raster resolution is sufficient for the pilot. Add the triangle-level encoder
when lip/wall errors show that representation is the limit, or before supporting overhang/tunnel geometry.
A universal voxel representation is not a prerequisite (design section 3).

## Q3. How does the agent find routes, including jumps nobody listed?

This is the central question. A route can involve jumps from anywhere: off the side of a ramp using its
height, from mid-slope, over a corner, onto a ledge.

**Rejected: enumerating jumps.** Precomputing a list of jump edges between surfaces does not scale, and
anything missing from the list is invisible, so routes are suboptimal by construction.

**Historical candidate A (superseded as the complete navigator): evaluate every decision forward.**
At each decision, ask the physics model about a small fixed menu of moves from exactly where the marble is:
jump now while holding each of 8 air directions or none. Also
ask "what if" versions from nearby hypothetical states (faster, pointed differently). Score each outcome by
time to the gem from where it lands, starting with walking time and later the learned value. In the ramp
example, one option ("jump now, hold left") shows up as 1.2 s faster than walking around, and nobody listed it.
Cheap and reactive, but it only sees one or two moves ahead, so long setups ("climb the ramp first, then
jump") need the learned value to catch them.

**Historical candidate B (superseded): search backwards from the gem.** Start at the gem (pickup range
about 1 u). Sample candidate states around it, run the physics model forward on all of them in one batch,
and keep those that land on the
gem. Repeat outward from that frontier, keeping the fastest time for each state: a shortest-path search whose
connections the physics model generates on demand. Stop when the search reaches the marble's state and read
off the chain. It finds multi-step setups, and coverage depends on sampling density, which we control.

The former preference for B plus A is superseded. Neither local one-step lookahead nor an assumed ability
to invert the dynamics defines the agreed route finder.

**Agreed:** sparse forward search from the actual physical state, guided by gem-side seeds. Sample
jump-command states and control sequences whose predicted path collects the target and has a viable exit;
use them to guide the approach search. Backward cost propagation is allowed over connections already found
and validated. A floating gem is an in-flight pickup volume, never a floor-snapped landing target.

Seeds preserve full motion/contact state and controls. Feasible speed may have both bounds, and speed,
heading and spin are coupled. Once an approach reaches a seed neighborhood, re-evaluate the flight from
that approach's exact predicted endpoint and uncertainty. Approximate position matching cannot authorize
the ideal seed's jump. Check again from observed state before launching.

First solve one floating-gem task from varied approaches, then compare executable routes and complete-group
orders while preserving arrival velocity/spin. Keep broad proposals independent of seeds and PPO. Replan
against actual state every decision while retaining a stable route unless invalidated or meaningfully
improved. Search-budget exhaustion means no route found within budget, not physical impossibility.

**Still empirical:** pruning/bin widths, proposal distributions, useful horizons and computational budgets.
Do not merge states solely by XYZ or assume a minimum-speed threshold describes launch feasibility (design
section 6).

## Q4. How is the plan executed, and how does PPO participate?

**Agreed:** internal plans are trajectories and motion regions, including velocity, spin, controls,
pickup/landing events and continuation. Compare two arms: a predictive controller owns every action, or
preserved PPO drives between controller-owned manoeuvres. PPO alone remains the baseline.

The hybrid uses explicit phases: PPO driving, controlled approach, flight, landing/exit, then PPO driving.
Takeover requires a validated route and either a necessary objective/progress condition or a supported
expected time saving over a declared alternative. It must happen early enough to reach the launch state.
Handback requires an exit region validated with actual PPO continuation trials, including the preceding
recurrent history; touching the floor or waiting 0.5 seconds is insufficient. Feed real observations through
PPO during takeover, but discard its actions. This keeps its GRU current without guaranteeing recovery from
unfamiliar states. Log post-handover falls, braking, progress and repeated ownership switches.

**History:** passing waypoint states as PPO's target was the earlier proposal and is superseded for this
experiment. The operator-approved design allows internal motion targets and controller action selection;
it does not require renewed approval to implement those choices. The deployed gems-only waypoint behavior
remains the baseline, and visible in-game markers still identify real gems. Earlier shifting waypoints
caused jitter: preserve route identity and use explicit invalidation/improvement rules. Any later policy
training must match deployment inputs and controls.

**Still empirical:** takeover thresholds, progress timeouts, validated exit regions and post-handover
acceptance criteria. Freeze them before scored evaluations (design section 7).

## Q5. What does the policy still have to learn, and how do we train it?

**Agreed:** the first learning task is supervised dynamics learning, followed by search and feedback control.
Preserve PPO unchanged for the baseline and hybrid driving. The first implementation target is a complete
floating-gem manoeuvre from non-ideal approaches on kotmjump; then compare useful routes, test handovers and
run full rounds. All four floating objectives remain in the benchmark.

Policy distillation, a plan-conditioned policy and further PPO changes are deferred until the controller
demonstrates useful behavior. Any new RL path pays actual game points and the agreed small fall marker,
not jump/waypoint/route-progress bonuses. The existing shaped reward is not already that objective. Do not
put controller-selected actions into PPO rollouts with the old policy's action log-probabilities.

**Later questions:** whether a learned travel-time value helps long setups, whether distillation retains
route competence, and whether new policy inputs improve real-round evaluation (design section 8).

## Q6. How do we know it generalizes?

* Report physics errors by joint geometry/motion/control slice, including command/takeoff timing, floating
  pickup false positives/negatives and continuation. The older 98% outcome, 0.3 u landing-position and
  0.5 u/s landing-speed targets are diagnostics, not sufficient deployment gates.
* First demonstrate pickup plus continuation from varied starts (M3); then reliable route choice and tested
  hybrid handovers (M4). Include misses, falls, abstentions and stalls when reporting completion time.
* Evaluate eight deterministic 3x rounds per main arm on kotmjump and KOTM (M5), with uncertainty estimates
  and more independent rounds if comparisons are inconclusive. Use the design's declared KOTM noninferiority
  margins and latency gate. Compare fixed-start route performance and real-round score, not training curves.
* D4: use the game's own verified maps with joint situation coverage. Keep a fixed reference set outside
  training/calibration for comparable reports, and separate rotating unseen-map tests for fresh transfer
  evidence. Declare family splits and fixture versions first; KOTM/kotmjump count as one geometry family.
  Rotating families may enter training after testing but then cease to be unseen. Repeated reference results
  informed by prior reviews are regression evidence, not fresh generalization claims (design sections 4, 10).
* A general lab map was considered and dropped; a small targeted piece remains a fallback for missing
  required coverage. Existing terrain files include KOTM, kotmjump, FlatIslands, FlatWithJump, JumpOnly,
  FlatGemTraining and Sprawl, but each export must be verified before contributing data.

## Q7. Later: powerups and other players

* Powerups fit the same slot as jumps: a Super Speed is a 25 u/s push along the marble's camera direction
  (which the actions already set every decision), a Super Jump a push upward. They become more moves on the menu
  of Q3. Deferred: the human used none in the 143-177 rounds.
* Opponents and self-play come after single-player navigation works across maps.

## Current decisions for the experimental implementation

These decisions are agreed, including the follow-up review refinements; the design is the plan we follow (operator, 2026-09-27). They supersede the earlier
proposals above where they differ. Remaining empirical choices use the design's measurement/evaluation
process; this table is not an approval queue.

| # | Question | Agreed decision |
|---|---|---|
| D1 | Air input | Control sequences, including switches, on the actual supported timing grid. |
| D2 | After landing | Model explicit continuation controls and useful exits from the start. |
| D3 | Dynamics model | Separate step ensemble and direct flight head; frozen evaluation and engine-calibrated arbitration. |
| D4 | Data and hold-out | Verified game maps; joint geometry/motion/control coverage; fixed reference (Horizon and Archipelago, whole families) plus rotating unseen-map tests. |
| D5 | Spin/contact | Use spin and actual timestamped contact telemetry; preserve baseline observation compatibility. |
| D6 | Old jump code | Preserve baseline; new decisions do not depend on the 3 u gap rule, fixed cruise-speed jump edges or binding prior. |
| | Flight contract | Start before the jump command; predict timed pickup-relevant paths, takeoff/landing events and explicit continuation. Step model handles initial airborne replanning. |
| | Route finding | Forward reachable-state search with gem-side seeds; recheck flight from the connected approach endpoint. Backward costs only over discovered directed transitions. Approach candidates aimed at seed launch regions; from M3 a time-to-gem guide learned from PPO play (guidance only). |
| | Execution | Pure controller and PPO-plus-planner arms; required-objective or time-saving takeover, tested handback, one action owner. No waypoint-to-PPO prerequisite. Handback memory (kept current vs reset) chosen by measurement. |
| | Geometry | Height rasters plus real normals/materials first, tested at edges; triangle queries when evidence requires them. Stage 1 adds exact edge features from the mesh (lip distance, height, gap width). |
| | Policy learning | Deferred until route/controller evidence; any new RL reward follows game score plus the small fall marker. |
| | Implementation | Prototype first ([KOTMJUMP_START_HERE.md](KOTMJUMP_START_HERE.md)): a recorded flight experiment on one kotmjump hole, then learned action selection on held-out starts. Then the general data pipeline, reachable approaches and route comparisons, hybrid handovers and full rounds. Five initial modules. |
