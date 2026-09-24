# Physics-aware planning for the marble: measured physics, engine-as-oracle, and learned dynamics

Written 2026-09-23 for whoever picks this up next (human or model). It assumes the navigator stack in
`ml_agent/nav/` (read `HANDOFF_NAV_TRAINING.md`, CURRENT STATE block, first) and the engine worktree
`C:/Users/doug/src/OpenPQ-TGEMIT-mbx` (branch `ai-training-mode`, off `mbx`). Every file:line below was
verified on 2026-09-23; re-verify before relying on it.

## 0. The ask, in one example

Two cubes side by side, one two to three times taller than the other. The marble is on the floor and must
reach the top of the tall cube. A plain jump reaches the small cube but not the tall one, even from on top
of the small cube. The trick a human uses is an **edge hit**: roll at the small cube with some speed,
jump so that the underside of the marble strikes the top EDGE of the small cube, and the bounce off that
edge throws the marble far higher than a jump can, high enough to land on the tall cube.

The goal is a system that reads any map's geometry, knows the marble's current state, and uses tricks
like the edge hit as a general skill to reach places fast on maps it has never seen.

## 1. Why this is feasible (what the engine actually does)

The whole marble simulation is one file, `engine/game/marble/marble.cc` (4381 lines). Three facts make the
project tractable:

**1a. The edge hit is not folklore, it is three lines of code.**

* Resting contacts against an edge or corner use the **centre-to-contact-point direction as the normal** instead of the face normal (`Marble::findContacts`, marble.cc:1966-1969). Against a top edge that normal is
  tilted away from the cube.
* A jump adds `jumpImpulse` (datablock default 1.0) **along that contact normal**, minus whatever
  velocity already leaves the surface (`applyContactForces`, marble.cc:2199-2224).
* A bounce reflects the normal component with `restitution = contact.restitution * bounceRestitution`
  (default 0.9), and a spinning marble converts spin into linear velocity through the
  `mVelocity -= mCross(-deltaOmega, -normal * radius)` term (`velocityCancel`, marble.cc:2450-2486).

So an edge hit is: arrive with horizontal speed v, meet the edge so the tilted normal has a large vertical
component, and the bounce (plus a jump pressed during that contact) turns horizontal speed into vertical
speed at 0.9 restitution. The output is a deterministic function of (position, velocity, angular velocity,
inputs) in the last few milliseconds before contact.

**1b. The simulation is deterministic given a fixed step.** Integration is F64 over an integer-millisecond
frame delta, sub-divided into steps of at most 8 ms and further cut at the first collision time
(`advancePhysics`, marble.cc:2646-2724). The only randomness is the bounce sound choice. Our training
build already pins the delta with `AI::FixedStepMs` (`engine/mbx/FrameRateUnlock/FrameRateUnlock.cpp:154-194`).
The same state and inputs give the same trajectory every time.

**1c. A silent re-simulation mode already exists.** Client-side prediction sets `runningPredictions = true`
and calls `advancePhysics` again (marble.cc:1239-1255); sounds, particles and script callbacks are
suppressed under that flag. The marble's complete dynamic state is six fields
(`writePacketData`, marble.cc:1486-1515: position, velocity, omega, camera yaw, pitch, mouseZ) plus
`mGroundTime` and power-up timers. Nothing saves and restores them today, but the pieces are there.

**What is missing today** (from the nav-side survey):

* The observation carries **no angular velocity, no contact normal, no on-floor flag** from the game
  (`observer.cs:511-535` serialises 6 self values, 25 gem values, time and score). `on_floor` is inferred
  in Python (`nav/obs.py:25-26`). The engine has all three (`getAngularVelocity` marble.cc:4335; the
  `onCollide` callback marble.cc:2006-2015 delivers contact point, normal, friction, material).
* Jump edges in the walk graph are pure geometry with no velocity gate (`nav/terrain.py:260-279`:
  `JUMP_GAP 4.0`, `JUMP_RISE 1.5`, `JUMP_DROP 6.0`, cost `1.5 * gap`). Nothing in the planner knows that
  a 5 u gap is fine at 12 u/s and impossible at 4 u/s, or that an edge hit can gain 4 u of height.
* The policy acts every 64 ms; an edge hit is timed to within one 8 ms physics step.
* No script-side polygon or edge enumerator exists. Collision polygons are available in C++
  (`buildPolyList`, marble.cc:1563, 1900) and offline from the `.dif` files (`hxDif.py` already parses
  them for `generate_terrain_map.py`).

## 2. Three ways to build it

### Option A. Engine as the oracle (sampling-based planner on the real simulator). RECOMMENDED FIRST.

Idea: skip modelling the physics. Add a save/restore of the marble state and a "step N ticks
silently" call to the engine, then plan by rolling out candidate input sequences on the actual simulator
and keeping the best. Edge hits are not hand-coded; the optimiser finds them because they score well.

Engine work (all on the `ai-training-mode` branch, next to the existing FixedStep block):

1. `Marble.saveState()` / `Marble.restoreState(id)`: the six packet fields plus `mGroundTime`,
   `mContacts` cleared, power-up timers, and `mOOB`. ~60 lines.
2. `Marble.simulate(moveList, stepMs)`: run `advancePhysics` under `runningPredictions = true` for a list
   of moves (each move = x, y, jump flag, held for `stepMs`), returning the trajectory as a string of
   `t px py pz vx vy vz contactFlag` rows. ~120 lines. The move struct is `Move` (x, y in [-1,1],
   `trigger[2]` = jump).
3. A console word in `mlAgent.cs` (`ROLLOUT <n_seq> <steps> <stepMs> <moves...>`) so Python can submit a
   batch and read back trajectories in one socket round trip.

Cost per rollout: physics is microseconds per 8 ms step. 500 sequences x 64 steps (~1 s of game time)
is ~32k steps, well under one frame. The plan can be recomputed every decision.

Planner (Python, `nav/planner_mpc.py`): MPPI / cross-entropy over input sequences at 16 ms resolution
(4 sub-moves per 64 ms decision), horizon 0.75-1.5 s, cost = time-to-goal estimate from the walk graph at
the rollout's end state + fall penalty + goal reached bonus. Output = the first 64 ms of the best sequence
(direction, throttle, jump timing at 16 ms). Warm-start from the previous plan.

Where the navigator fits: the planner acts as a **teacher for the policy**. Use it (i) as the action source
in real rounds for hard legs (jumps, edge hits, gap crossings), with the PPO policy for ordinary rolling,
and (ii) as a data source: log (state, terrain crop, planner action) pairs and distil them into the policy
(behaviour cloning as an auxiliary loss on the same network, or a dedicated "skill head"). Distillation is
what makes it transfer: the network learns the takeoff-state-to-outcome map from thousands of planner
solutions across maps.

Caveats:
* Client and server marbles are separate objects (`mlAgent.cs:357-379` teleports both). Rollouts must run
  on the object that owns the physics in our single-player training setup (the client marble drives the
  camera; verify which one `advancePhysics` is called on with `isServerObject()`).
* Moving platforms (pathed interiors) advance with time; rollouts from a saved state see them frozen unless
  their state is saved too. Acceptable for a first version (KOTM, FlatGem and Islands are static).
* The 64 ms decision loop still applies to the policy. Rollout-planned jump timings must be executed at
  16 ms, which needs the bridge to accept a sub-tick action list per decision (see section 4, item 3).

### Option B. Measured physics, analytic skill library, graph planner

Idea: treat the marble as a black box and measure its envelopes with the game as a batch oracle, then
store skills as tables and plan on a richer graph.

Measurements (each a script in `ml_agent/tools/`, each runs against one game instance in fixed-step mode
using `TELEPORT x y z vx vy vz` to set the initial state exactly, `nav/env.py:249`):

* `measure_jump.py`: from flat ground at speed s in {0, 2, ..., 14}, press jump, record apex height, range
  to landing at the same height, time to apex. Gives the ballistic envelope and the true `jumpImpulse`
  effect (expected: 20 u/s^2 gravity, apex ~ v_up^2 / 40).
* `measure_edge_hit.py`: a step of height h (use a real map feature or a purpose-built test mission; see
  section 5). Sweep approach speed s, lateral offset, and the jump press time relative to contact in 8 ms
  steps. Record peak height, horizontal carry, and the timing window width. This is the single most
  important measurement: **if the window is one 8 ms step wide, the executor must be engine-side** (a
  timed jump primitive), because the 64 ms policy cannot hit it.
* `measure_slope_launch.py`: ramps/lips at angle a, speed s, with and without jump: range and height.
* `measure_bounce.py`: fall height h onto flat, record rebound height (restitution check, 0.9 expected)
  and spin-to-speed conversion when landing with omega.

Products: lookup tables `jump_envelope.npz`, `edge_hit.npz` (inputs: h, s, offset, timing -> outputs:
dz gained, dx carried, success probability under +-8 ms timing jitter).

Geometry: extend `generate_terrain_map.py` to also emit **convex top edges** of every interior polygon
(from the `.dif` polygons already parsed by `hxDif.py`: an edge shared by an upward face and a
steep/vertical face, with its direction and the drop on the far side). Then in `nav/terrain.py`, beside
the existing jump edges, add **skill edges**: for each convex edge and each walkable landing cell within
the edge-hit table's reach (height gain up to the table max, carry distance within range), add a graph
edge with cost = approach run-up time + flight time, tagged with the required takeoff state (approach
direction, speed band, contact point). Same for plain jump edges: replace the flat `1.5 * gap` cost and
the 4 u limit with the measured envelope, conditioned on the speed the walk graph predicts at that node.

Planner: Dijkstra/A* on the augmented graph gives a route with tagged skill legs. Executor: the existing
`path_waypoint` string-puller (`nav/real_run.py:176`) for rolling legs; for a skill leg, a scripted
controller that steers to the takeoff point at the required speed and fires the timed jump via an
engine-side primitive (`JUMPAT <ms>` or "jump on next edge contact").

Pros: every piece is inspectable, and the tables transfer to any map because they describe the marble.
Cons: tables cover only what you thought to measure (a 4-D edge-hit table is fine; combinations such as
"edge hit off a slope while spinning" are not), and the executor is hand-written control.

### Option C. Learned dynamics model plus short-horizon planning (world model)

Idea: learn f(s_t, a_t) -> s_{t+1} from transitions (s = position, velocity, omega, contact normal, local
terrain crop; a = direction, throttle, jump) at 16 ms, then plan with CEM/MPPI in the learned model, or
train the policy inside it (Dreamer-style).

Honest assessment: contact-rich dynamics with a 0.9-restitution discontinuity are the worst case for a
neural dynamics model. The edge hit lives entirely in the discontinuity: a 0.3 u change in position turns
"roll over the lip" into "launch 6 u up", and a learned model smooths that. Achieving 8 ms accuracy on
that boundary needs a large dataset concentrated exactly on edge contacts and a model with explicit
contact structure. With the true simulator available at microseconds per step (Option A), a learned
model buys nothing except portability to hardware without the engine. Recommend Option C only as a
**distilled feasibility model**: train a small network on Option A/B outcomes to answer "from this state,
can I reach that platform, and how long does it take" in one forward pass, for use as a fast edge cost
inside the graph planner and as an input feature to the navigator.

### Verdict

Build A first (it is the smallest amount of new physics reasoning and cannot be wrong about the physics),
use B's measurements as the acceptance tests for A and as the source of the geometry edges, and keep C to
the distilled feasibility model. Every option needs section 4's engine and observer changes first.

## 3. Measurement protocol (do this first, one evening)

Run with one instance at `FIXEDSTEP 8` (the physics quantum; note the decision-denominated constants in
`nav/env.py:35-38` do not apply to these tools) and `LOCKSTEP 1`. Each trial: `TELEPORT x y z vx vy vz`
(zeroes omega, `mlAgent.cs:366`), settle 2 ticks, then send the scripted move list one tick at a time and
read position/velocity each tick.

1. **Reproduce the two-cube example.** Find a step-and-platform pair on an existing map (KOTM's centre
   block has 6.5 u lips; the ring has 4 u steps) or build the test mission in section 5. Confirm by sweep
   that an edge hit reaches a height a jump cannot. Record the sweep as a heat map (speed x timing ->
   peak height). This single plot decides whether the skill is worth a planner.
2. **Timing window.** Width in ms of the jump-press window that produces > 80 % of the peak gain. If it is
   <= 16 ms, item 3 in section 4 is mandatory.
3. **Jump envelope** vs speed (apex, range, hang time). Compare with the datablock: gravity 20,
   `jumpImpulse` 1.0, `maxRollVelocity` 25, `airAcceleration` 5 (marble.cc:3106-3128).
4. **Restitution and spin transfer**: drop tests and rolling-into-wall tests.
5. **Determinism check**: repeat any trial 20 times, assert identical trajectories. If they differ, the
   frame delta is not pinned (check `AI::FixedStepMs` and that `SPEED` is 1 during measurement).

Save all raw trials to `ml_agent/logs/physics/<test>_<ts>.csv` and the fitted tables to
`ml_agent/physics/`. Every later component (graph edges, planner cost, acceptance tests) reads the tables.

## 4. Integration points in the current stack

1. **Observer** (`platinum/client/scripts/ai/observer.cs:88-123, 511-535`): serialise angular velocity
   (3), the last contact normal (3) and a contact flag (1) from an `onCollide` handler
   (marble.cc:2006-2015 fires `onCollide(contactVert, normal, surfaceVelocity, contactDistance, objId,
   friction, force, materialId)`). Bump `RAW_DIM` (35 -> 42) in `nav/protocol.py`, add the slots to
   `nav/obs.py` (`VEC_DIM` 53 -> 60) and bump `NAV_OBS_VERSION`. This alone lets the policy see spin and
   contact, which it needs for any timing skill. It is also what makes Option C's dataset possible.
2. **Engine** (`ai-training-mode` branch): `Marble.saveState/restoreState`, `Marble.simulate(...)`
   (Option A) and a `stepSim(n)` next to `TimeManagerProcessOverride` in `FrameRateUnlock.cpp:154-194`.
3. **Bridge sub-tick actions** (`mlAgent.cs:290-410`, `nav/protocol.py:74-82`): extend the action line
   so one 64 ms decision can carry four 16 ms sub-moves (`fwd,back,left,right,jump` x 4) or a
   `JUMPAT <ms>` field. Without this, no planner can execute an edge hit through the existing loop.
4. **Geometry** (`generate_terrain_map.py`, `hxDif.py`): emit convex edges per map into the `.npz`
   (`edges (E, 8)`: x0 y0 z0 x1 y1 z1 drop_far height_above_floor). `nav/terrain.py` loads them and adds
   skill edges to the graph (section 2B).
5. **Planner seam** (`nav/real_run.py` ~430-510 and `nav/vec_worker.py` real-gem branch): both already
   reduce to "emit goal (x, y, z) and next goal per decision". A planner replaces `choose` +
   `path_waypoint` there and may additionally emit a sub-tick action override for skill legs. The worker
   path matters because the user's rule is training parity with real rounds (HANDOFF 28.25): whatever the
   planner does in real rounds must be what training sees.
6. **Distillation into the policy** (`nav/train_nav.py`, `nav/ppo_recurrent.py`): add a BC auxiliary loss
   on planner-labelled decisions (weight ~0.1, the same seam the old `bc_pretrain.py` used; note
   `bc-cold-start-failure` in memory: cloning from a cold start regressed, so apply it as an auxiliary
   term on the trained navigator, on skill legs only).
7. **Curriculum**: with real-gem training in place, skill practice comes from goals the walk graph says
   need a jump or edge hit (`JUMP_GOAL_P` already exists at `nav/terrain.py:78`; add `SKILL_GOAL_P`).

## 5. Test mission for the two-cube case

Torque missions are object dumps (`InteriorInstance { interiorFile; position; rotation; scale; }`,
e.g. `game/marble/data/missions/intermediate/marbletris.mis:73`). The quickest way to get the two cubes is
to reuse any existing cube `.dif` (search `data/interiors/` for a unit block, several PQ levels ship one)
with two `InteriorInstance` entries at `scale = "4 4 2"` and `scale = "4 4 6"` on a flat floor interior,
plus a `StartPoint`, `MissionInfo` with `gameMode = "Hunt"` off, and no gems. Put it under
`platinum/data/multiplayer/hunt/custom/EdgeHitTest.mcs` next to `FlatGemTraining_Hunt.mcs`; delete the
`.dso` after edits. Then `generate_terrain_map.py EdgeHitTest` and `verify_walk_grid.py` as for any map.

## 6. Risks and unknowns, ranked

1. **Timing window width** (unknown until measured). If it is one 8 ms step, everything must go through
   the engine-side primitive; 64 ms policy actions can never do it alone.
2. **Which marble object simulates** in our setup (client vs server). Rollouts on the wrong one plan
   against a ghost.
3. **F32 truncation of the collision discriminant** (marble.cc, `mSqrtD((F32)discriminant)`): edge-contact
   times have limited precision; identical-looking states can resolve differently. Determinism still
   holds; smoothness does not. Plan with timing jitter (+-8 ms) in the cost.
4. **Moving geometry** is invisible to saved-state rollouts (not an issue on the three training maps).
5. **Distillation regressions**: cloning has hurt before. Gate every change on 8 real rounds, as always.

## 7. Effort estimate

| piece | size | depends on |
|---|---|---|
| observer + protocol + obs slots (omega, contact) | 1 day | nothing |
| measurement scripts + tables + two-cube reproduction | 1-2 days | test mission |
| engine save/restore + simulate + stepSim | 2-3 days | C++ build of the mbx worktree |
| sub-tick action line + JUMPAT | 1 day | bridge |
| convex-edge extraction + skill edges in terrain graph | 2 days | measurements |
| MPPI planner on the oracle, real_run integration | 3-4 days | engine pieces |
| distillation into the navigator + curriculum | 3-5 days | planner data |

Roughly three weeks of focused work to a first map-general edge-hit capability, with a usable
intermediate result (measured jump envelopes replacing the flat jump-edge cost) after the first week.
