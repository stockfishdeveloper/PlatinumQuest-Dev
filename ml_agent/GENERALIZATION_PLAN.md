# Generalization plan: play any map, climb fast, then beat an opponent (2026-10-07)

Status: DIRECTION APPROVED 2026-10-07 (world frame kept + rotation augmentation, weights kept, no restart;
per-decision cost allowed; held-out gate = a new generated map the operator approves). Nothing built, no
training started.
Each phase's build still needs the operator's OK before code is written.
Companion docs: POWERUP_PLAN.md (powerup bridge and physics), KOTMJUMP_NAVIGATION_DESIGN.md (gap jumps),
docs/HANDOFF_NAV_TRAINING_LOG.md 40.60-40.62 (where the KOTM model stands: 166.55 at update 32565).

## Why

The KOTM model (nav_points_20261005_parity_32565.pth) is clueless on other maps: at a ledge with gems on
top it drives into the wall. Three reasons, none of them "it never trained there":

1. The route comes from a walk grid. A ledge is a separate floor level with no link to the floor below,
   so the route cannot say "get on top"; the waypoint points at the gem through the wall.
2. The only Super Jump rule the use prior knows is "at a lip, a hole too wide for a plain jump".
   A wall with a gem above matches nothing, so a use cannot even be sampled (nav/model.py use_prior).
3. The landing predictor only answers "do I cross this void", never "do I reach that height".

The corrected-map retrain (9 h to parity, 17.5 h to the best) shows the policy also memorizes KOTM's exact
local patterns. The inputs are already relative (no map position), but on one map every local pattern is
unique, so pattern recognition and place memorization are the same thing. Two more causes:

4. Inputs are in world axes (gem direction, velocity, edge rays). The same ledge rotated 90 degrees is a
   different situation to the network; each orientation must be learned separately.
5. KOTM never presents an upward climb, a slope or a bumper, so those skills have no training signal.

## Principles (fixed)

* Reward = the game's rules only: gems inside a fixed time budget. No arrival or "reached the ledge"
  bonuses. Speed pays by itself because an earlier climb leaves time for more gems.
* The policy never judges geometry itself. It reads physics verdicts computed over the local height map
  from measured constants (nav/learned_nav/powerup_physics.py, nav/physics.py), the way the gap jump
  works today. A verdict is true on every map by construction.
* No absolute position, no map identity, no per-map rules anywhere in the inputs or priors.
* Generalization is measured only on a held-out map the model never trained on.
* Human play is the clock, never the template.
* Hard cap 8 game instances; the reference maps (Horizon, Archipelago families) are never trained on.

## Phase 0: one observation migration (V11), designed once; the weights are KEPT

Everything below changes the observation. Design the full V11 layout now, APPENDED to V10 the way V7-V10
were, so the current weights (166.55 at 32565) carry over and no retrain from scratch is needed:

* Option simulator block (phase 2): per-option landing verdicts and the takeoff window.
* Opponent block (phase 5): relative position, velocity, distance, held powerup, mega state, score
  difference, time left. Zero-filled until phase 5.
* Gem value of the current and next gem (needed for sniping in phase 5).

Frame decision (operator 10-07, after the sanity check): the WORLD frame stays. The camera is fixed, so
the world frame and the camera frame are the same thing, and the model's inputs and stick output already
share it. A marble-heading frame was proposed and REJECTED: this marble brakes constantly and its heading
jitters, so a heading frame would spin and look "backward" while travelling forward.

Direction independence (so one cluster approached from any side is one lesson) comes instead from
ROTATION AUGMENTATION in the trainer: for each training sample, also train on copies where the whole
picture is turned (gem and next-gem direction, velocity, spin, edge rays and the stick answer, all by the
same angle). Turns are multiples of the edge-ray spacing (22.5 deg, 16 headings) so ray vectors shift
slots without resampling. At play time nothing is rotated. Valid because with a fixed camera the game is
symmetric under turning (the stick maps straight to a map direction, physics are isotropic, Super Speed
aim is set per kick by the bridge). If clusters still learn slowly per direction, that is the evidence to
revisit; the 2-day restart is OFF the table unless that happens.

Gate: V11 with augmentation holds KOTM at 160+ (64 deterministic rounds, learned use) before phase 2
training starts (the augmentation may dip the score briefly while the network adjusts).
Build: obs.py layout, model.py priors (fixed indices move with the layout), trainer augmentation, a
migration script like V7, a rotation test (the same situation turned four ways gives turned actions).

## Phase 1: climb links in the map (route level, no training)

Build time, per map, from the existing height stack (generate_terrain_map.py, nav/terrain.py):

* For every pair of adjacent floor levels, test the measured physics from a grid of takeoff points:
  plain jump, jump plus blast at the apex, Super Jump (+20 u/s), Helicopter glide. Where one lands on
  floor with level run-out, add a climb link: takeoff spot, method, required speed and heading.
* The Dijkstra goal field uses the links, so the route leads to the takeoff spot on any map, and the
  waypoint sequence includes the ledge as a step. Walkable slopes and bumpers get their own links later.
* Check: render the links for 3 maps and teleport-verify a sample of them (the verify-terrain rule).
  A link that does not land in the real game is a bug.

Gate: on 3 varied Hunt maps every ledge gem has a route; no phantom links in the teleport sample.

## The skill list (operator decision 2026-10-07: this is the way)

The model will not discover tricks on its own at our scale (8 instances, a use key pressed at one moment
with a payoff seconds later). Its history proves it: gap jumps and Super Speed kicks each came from a
physics verdict, a prior that lets the action be sampled, and drills. Hand-writing one rule per trick
per map does not scale either. So: a SKILL LIST, each skill generalized to any map, added one at a time.

A skill is defined by five things, and it is not "done" until all five exist:

1. Situation family: the geometry it applies to, described relatively (a ledge of height h within reach,
   a gap of width w, a slope of grade g), never a place on a map.
2. Option(s): which macro move(s) solve it, from the option menu below.
3. Verdict: what the option simulator reports for it (lands where, how fast, floor and run-out there,
   earliest feasible fire point). Pure measured physics over the local height map.
4. Drill: a randomized generator for the family (height, width, orientation, approach distance, speed,
   angle all random; real gems; fixed time budget; rolling starts), plus 1-2 real maps that contain it.
5. Gate: forced-option success (no learning) first; then learned use on a HELD-OUT map, with the human
   clock for the family.

The option menu (one line each in the simulator; a new powerup is a new line, not a new project):
plain jump, jump + blast at the apex, Super Jump, Super Speed kick, Helicopter glide, Super Bounce,
Shock Absorber landing, Mega Marble (activation, air boost, spin launch), blast (floor and air).

Skills, in the order to add them (status 10-07):

| # | Skill | Family | Options | Status |
|---|-------|--------|---------|--------|
| 1 | Edge avoidance | any drop within the edge rays | none (steer) | DONE (KOTM) |
| 2 | Gap jump | void of width w ahead, same level | plain jump, SJ, blast | DONE for KOTM gaps (MIN_JUMP_GAP is a KOTM hack to remove) |
| 3 | Super Speed kick | straight run to 1-2 gems with level floor | SS | DONE (two-gem rule), ~0.8 a round |
| 4 | Block cluster collection | 5-8 gems in a small area on stacked blocks at 2-3 heights, each top reachable by a plain jump (operator 10-07: the most common spawn layout in the game) | plain jump up, controlled drop down, brake on a top, cluster ORDER from block-scale jump links in the tour | NEXT: THE FIRST NEW SKILL |
| 5 | Ledge climb | next level above a plain jump's reach, gem on it | jump+blast, SJ, heli | after 4 |
| 6 | Controlled drop | gem on a lower level, drop without OOB (the deep version of 4's drops) | steer, shock absorber | with 5 |
| 7 | Slope climb and descent | graded floor up/down to a gem | steer, SS up a slope | after 5 |
| 8 | Ramp launch | a ramp that throws the marble to a level or across a void | steer, SS, SJ | with 7 |
| 9 | Wide gap or height by Helicopter | void or climb beyond a jump's range | heli | with 5 |
| 10 | Bumper use | bumpers as launchers to a level or a gem | steer into the bumper | after 7 |
| 11 | Mega Marble spin launch | high-friction floor, speed from spin | mega | later |
| 12 | Corridor and corner speed | narrow floor between walls or edges | steer, brake | later |
| 13 | Opponent: denial routing, blast push, mega knock | another marble nearby | blast, mega, routing | phase 5 |

Skill 4 in detail (why it goes first): it needs only the plain jump, so the whole pipeline (block-scale
jump links in phase 1, the option simulator in phase 2, the family generator and held-out gate in phase 3,
the rotation augmentation) is exercised on the simplest option with no powerup timing to debug at the same
time. It bundles three sub-skills KOTM never taught: jumping onto a RISE while rolling (nav/physics
already integrates rises, JUMP_RISE, nothing uses it), stopping on a small top, and dropping to a lower
block on purpose (on KOTM every drop is death, so the current model is edge-shy and will fight this until
a drop verdict says which drops are safe). The speed is in the ORDER: up one side, across the tops, down
the far side; the tour decides the order from the jump links, the policy learns the timings.
Generator: random block count, heights (all under the plain-jump apex, about 1.3 u measured), spacing,
orientation, gems on real spawn points, 5-8 gems per cluster, rolling start from a random side.
Held-out test (operator 10-07): real maps with such clusters are full of other hazards, which would hide
whether the clusters themselves transferred. So the test is a NEW GENERATED MAP the model never sees in
training: clusters like the training family but laid out differently enough to be a real transfer test.
The operator plays it and approves it before it counts as the gate. Real maps with clusters are still
copied into the playground's spawn pool for realism, but are not the gate.

Adding a skill is always the same recipe (the five things above), so each one is a day or two of build
plus drills, not a redesign. The list grows when a new map shows a situation that fails; the growth is
expected to stop around a dozen families for Hunt.

Discovery still happens, inside the menu: in drills the prior fires options at random across their
feasible windows (the 10-04 kick-exploration widening), so two-option chains a decision apart (kick then
jump, jump then blast) are found by the policy. Chains of three or more precise inputs are not, and those
are added to the list when humans are seen using them.

Human replays are the detector, never the teacher: per family, the model's time against a good player's.
A large gap means a missing skill or an unknown trick, and names the next list entry.

## Phase 2: the option simulator and the takeoff window (observation and priors)

* One simulator, not per-trick verdicts: every option in the menu is rolled forward from the current
  state over the local height map with the measured constants (powerup_physics.py, nav/physics.py AIR_EXACT),
  hold and ballistic variants like today's landing predictor. Per option the observation reports: lands
  on floor (yes/no), landing level relative to now, landing speed, floor run-out after landing, and the
  EARLIEST distance from which firing lands on the target level (the takeoff window).
* The window is cut by the kick rules already in use: level floor past the landing (level_run), arrival
  speed under a cap, a line from the landing to the next gem (the two-gem idea applied to a climb).
* Use prior: an option may be sampled whenever its verdict is on and the route's next step needs it. In
  drills the prior samples across the WHOLE window (early ones included) so the policy can discover that
  early is faster.
* Timings the policy must genuinely learn (no verdict can fire them): the blast at the apex, the mega air
  boost (two-decision sequences like the kick, with the 2-decision key lag).

Gate (skill 4 first): with the plain jump forced at the simulator's earliest point (eval only, no learning),
the marble lands on the block in 90 % of drill starts and takes the cluster without falling in 80 %. If the
forced version cannot do it, the verdict is wrong and training would only learn around the bug.

## Phase 3: family drills with momentum, then real maps

* A situation-family generator (the FlatGemTraining_Hunt precedent): per family, random geometry
  (height, width, orientation, approach distance), real gems spawned where the family says (the
  waypoints-from-gem-spawns rule), regenerated every episode so it cannot be memorized. Skill 4 (block
  clusters) first; each later skill adds a family to the generator.
* The playground (operator 10-07): a generated mission of block clusters is the training ground for skill
  4, as a GENERATOR with many variants, never one fixed map (one map = KOTM-style memorization again).
  Each variant: a flat floor with 20-40 clusters a little apart (travel between them is part of the skill);
  per cluster random block count, heights (under the plain-jump apex), spacing, stacking and orientation;
  real gems in the mission file as spawn groups. Mix in spawn groups COPIED from the training maps
  (geometry and gem positions together) so the distribution matches the real game; never from the
  held-out map. Each of the 8 instances loads a different variant; variants are regenerated on restart.
  Density is the win: on a real map a cluster is ~20 s of 180; here nearly every second is the skill.
  Mechanics: FlatGemTraining_Hunt is the precedent; generate_terrain_map.py runs per variant; the custom
  mission info function naming rule applies (MP_PQ_<name>_GetMissionInfo) or the game sits in the level
  select. The playground score proves nothing by itself; only the held-out map does.
* BUILT 10-07 (operator approved the first map in-game): nav/maps/cluster_extract.py pulls tight multi-height gem
  clusters and the boxes under them (upward polygons above the local base, rectangle-fitted, rebuilt as scaled
  interiors_mbp/Cube.dif) from Sprawl, Duplex, ExampleMission, Incidious, SprawlEvolved, TreasureBox, Acropolis2
  (pool nav/maps/cluster_pool.json, 57 usable clusters). nav/maps/build_cluster_variants.py writes the FIXED files
  BlockClusters_v0..v7_Hunt.mcs (14 clusters each, random angle 0-360 deg and mirror per cluster, operator: not just
  quarter turns) and BlockClustersHoldout_Hunt.mcs (Acropolis2 only, never trained on), overwriting them and their
  .dso on every run, and builds the terrain maps. nav/maps/teleport_probe.py is the in-game check (box centres,
  inset corners, off-edge points vs the terrain map's prediction): v0 470/470 agree after the interior rotation
  fix (log 40.63). Maps are READY for training; the holdout awaits the operator's play-through.
* Every segment starts ROLLING toward the situation from a random distance, speed and angle, never
  parked at it. Fixed budget per segment (about 8 s); the score is gems taken inside it.
* Mix: drills on 3 instances, full rounds on 5 (KOTM plus 2-3 real Hunt maps that contain the families
  in training). The 10-04 lesson applies: drills alone wear the navigator down; full rounds keep it honest.
* Human clock per family: time from the gem before the situation to the gem after it, measured once
  from a good player on the drill map and on a real map. Every check reports the model's time against it.
* A small per-decision cost (standard RL impatience) is ALLOWED (operator 10-07). Try the fixed budget
  alone first; add the cost if cluster runs stay slow or late-jumping.

Gate per skill: on a HELD-OUT map never trained on, the family's gems are taken, falls after the move
under 10 % of attempts, and the time within 1.5x the human clock. KOTM stays at 160+.

## Phase 4: per-map fine-tunes from the broadened base (measurement only)

* Freeze the phase 3 model as the base. Fine-tune one or two maps from it, each with its own checkpoint;
  the game loads the checkpoint by map name at connect.
* The first fine-tune is the measurement: if a new map reaches a good score in 1-2 h instead of 10+, the
  per-map plan is affordable for all maps. If not, go back to phase 3 with more training maps.
* Base changes after this point (phase 5) are re-rolled into the branches, so phase 5 goes BEFORE the
  bulk of the branches are made.

## Phase 5: opponent play by self-play (on the base)

* Two-marble training matches: a host instance plus a joined AI instance in one round, both bridged, both
  on the fixed step (the lockstep refusal in agentLive.cs protects other people's games; both players are
  ours). Fix the joined-client powerup read on the way (today a joined client cannot use Super Speed).
  4 matches instead of 8 solo games; two learners per match, so per-marble throughput is unchanged.
* Measure first (powdrill): how far a blast throws ANOTHER marble by distance, what a mega collision does.
* Reward: own points plus a margin term (mine minus theirs). Own points alone never rewards denial or a
  knock-off; the margin term is what teaches sniping and blasting.
* Opponent pool: frozen past checkpoints (starting with the base), refreshed as the learner improves.
* Expected: route-level denial (the gem the opponent cannot reach first, high-value gems near them)
  should work, it is routing with one more input. Blast and mega knock-offs are a timing skill with weak
  credit (payoff only if they fall): a maybe, and a smaller gain.

Gate: positive margin against the frozen base over 64 matches with solo score within 5 of the base;
then score margin in real lobbies against people (live_play already measures that).

## Phase 6: branches for all maps

Per-map fine-tunes from the phase 5 base for every Hunt map, checkpoint loaded by map name.

## Phase R (2026-10-09, operator decision): train on the game's own maps, not a curated pool

Why: the block-cluster pool (an endless flat floor) taught a shortcut. The model learned "never jump when a drop is
within 5-10 u" because no training gem was ever within 17 u of a void, and KOTM only ever showed that a jump near a
rim ends in a fall. On ExampleMission it parks at 0.5 u ledges beside drops for 57 s a round (log 40.75). Curating
maps to balance that out does not scale to all maps ("if we're already trying to balance out the map pool we are
completely fucked"). The target distribution is the game's hunt maps; train on them, measure on held-out ones.

Map list: taken from the PQ app (AppData/Roaming/PlatinumQuest), the mbx repo's lists are outdated. Tiers used:
Beginner, Intermediate, Advanced, Expert only (62 maps; Horizon is a reference map and never trained on).
First stage (operator, 10-09): 10 maps the model can start on, all gems reachable without powerups:
KOTM, Sprawl, Cube Isle, Bowl, Duplex, Fractured Islands, Cragmire, Vortex Effect, Treasure Box, Concentric.
Held out: ExampleMission (Custom), Horizon, plus held-outs from the tiers to be named. The pool widens when the
held-out scores stall while training scores rise. Fractured Islands, Treasure Box and Prophetic mission files come
from the PQ app (Treasure Box: 240 s round and PQ par scores; Prophetic: PQ's tight bounds trigger).

Rules that follow from Hunt itself (operator, 10-09):
* A spawn group only clears when every gem in it is collected; a gem that needs a powerup pins the round.
  Competitive Mode ($MPPref::Server::CompetitiveMode, 20 s respawn after the FIRST pickup of a group, leftovers kept)
  is allowed in training; it does not help when a whole group is unreachable.
* Most of these maps need a route (a road up a tower, a ramp), not a straight line at the gem. Routing is the
  planner's job: the navigator is steered at a string-pulled waypoint on the Dijkstra path (real_run.path_waypoint,
  12 u lookahead). The walk graph is single-level (top floor per cell), so gems under overhangs or on lower decks
  and ramps that pass under something are invisible to it (Cragmire 12 % of gems, Concentric 11 %, Treasure Box 5 %).

Order, each part shown to the operator at 1x where it is behaviour, and each part asked before starting:
1. DONE 10-09: mission files in place, terrain maps built for the 10 (generate_terrain_map.py).
2. DONE 10-09 13:35: teleport verification of all 10 terrains (footprint rule fixed for tilted slabs; log 40.76).
3. DONE 10-09 14:30 (log 40.77): floor = physics per surface (texture friction from init.cs, tan a <= 1.1 mu; the
   push cap was measured wrong on Bowl's 51 deg wall and dropped); multi-level walk graph (nodes per level, directed
   edges, slide rule for steep cells beside a lip); every gem on all 10 maps reachable; KOTM at parity (A/B 144.1 vs
   144.8 with the committed code).
4. DONE 10-09 14:45 (log 40.78): forced-stall clearance 3 u (JUMP_FORCE_CLEAR_U); MIN_JUMP_GAP 0.5 u; KOTM 143.1.
5. NEXT: training setup: 8 games across the 10 maps (uniform at first; progress-weighted by par/platinum later),
   Competitive Mode on in the training games, baseline sweep of the current model first (4 clean rounds a map);
   decide whether respawn becomes an action (a human presses it when stuck; the only price is its time).
6. Watch the held-out trend and KOTM retention, not single-map scores. Expect slow numbers on complex maps; the
   model has never rolled on a slope.

Known limits this does not fix: the terrain pipeline is static heights (moving platforms, bumpers, gravity changes
are invisible); the policy sees ~20 u and has no map memory, so the planner's graph quality is on the critical path.

## Order and why

0 (design) -> 1 -> 2 -> 3 (skill 4 block clusters first, then 5, 6, 9, then 7, 8, 10) -> 4 (one or two
branches as the measurement) -> 5 (skill 13) -> 6. Skills 11 and 12 slot in wherever a map shows they are
missing.

Navigation generalization goes before the opponent work because an opponent model is only useful where
the marble can navigate, opponent play in lobbies happens on many maps, and the frame change forces a
retrain anyway; doing it before the opponent complexity keeps the retrain debuggable.

## What it does not cover

Bumper chains, moving platforms, gravity changes, and anything outside the physics envelope of a jump,
a blast-assisted jump, a Super Jump or a Helicopter glide. Those need their own links later.

## Risks

* Rotation augmentation could dip KOTM while the network adjusts: the phase 0 gate (160+) catches it; the
  fallback is to train without augmentation and let the playground's random orientations do the work.
* Verdict bugs train in as superstition: every verdict is checked forced (no learning) before it is
  trained on (phase 2 gate), the Super Speed lesson from 10-05.
* Drill wear-down: full rounds stay in the mix at all times (10-04 lesson).
* The physics step: all of this trains on the 64 ms observation interval with DELAY 4 in live play, as
  established on 10-06. Any change to that is a separate decision.
