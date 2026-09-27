# Jump physics: start here

Status: **the plan we follow** (operator, 2026-09-27); final, implementation not started. This page is
the concrete first experiment. The full
reference specification is [KOTMJUMP_NAVIGATION_DESIGN.md](KOTMJUMP_NAVIGATION_DESIGN.md); where the two
differ on what to do first, this page wins. The deployed PPO navigator, its checkpoints and training
settings are untouched by this work ([HANDOFF_NAV_TRAINING.md](HANDOFF_NAV_TRAINING.md)).

## Build order

1. **P0a. A small recorded flight experiment** on one kotmjump hole (below).
2. **P0b. Learned action selection on held-out starts:** does the learned predictor pick better jumps than
   simple alternatives, from starting states it has not seen?
3. **The trustworthy general data pipeline** (design M1, M2): telemetry, geometry export and verification,
   step model and flight head, calibration, more maps.
4. **Reachable approaches and route comparisons** (design M3, M4): search from far away, seeding,
   feedback control, all four floating gems.
5. **Hybrid handovers and full rounds** (design M4, M5), then transfer to other maps (M6).

M0 (baseline manifest, plus 8 rounds of the preserved PPO baseline on kotmjump at 3x) is cheap and can run
alongside P0. It answers how many floating gems the current navigator takes, which it has never attempted.

## P0: the prototype

**The question.** Can a small learned flight predictor choose jump and air-control sequences that collect a
floating gem and land safely, from starting states held out of its training, more often than simple
alternatives choosing from the same candidates?

**Scope.** One kotmjump hole and its floating gem. Starts near the launch (no approach planning). Existing
controls: 64 ms decisions under fixed step and lockstep, the existing joystick mapping. Existing terrain data,
with the local area checked in the game. No PPO, no route search, no step-model ensemble.

**Before any data**
* The target gem must be present for every trial: P0 runs on the drill map `kotmjump_p0` (decision 1 below).
* Check the local geometry in the game: teleport tests on the lips around the hole, and a few jumps whose
  landings are compared with the terrain map.
* Give the Python teleport the spin words the game already supports (`TELEPORT x y z vx vy vz wx wy wz`), so
  starting states roll without skidding.
* Repeat check: 20 repeats of a few trials must give the same trajectory within a declared tolerance.

**Recording (kept after P0; the controller is expendable)**
* The actual state after any settle ticks: position, velocity, spin, and contact as far as the current
  telemetry allows. Requested teleport values are metadata only.
* Every applied control per 64 ms tick, as actually serialized, with the camera yaw and the jump press time.
* The trajectory each tick until landing plus 0.5 s, or out of bounds.
* Outcomes from the game: the pickup (score change, unambiguous while only the target gem is present),
  out of bounds, landing.
* Provenance: engine build, script versions, mission, settings, trial ID.

**Starting states.** On the floor within about 6 u of the hole's lips, varied heading and speed (0-15 u/s),
rolling spin. Split into training and held-out starts **by region and heading bin**, never by random trial,
so neighbouring starts cannot leak between the two.

**Candidates.** Jump at decision tick 0-4 or not at all; hold one of 8 air directions relative to the heading,
or none; optionally one switch (for example to braking) at a later tick. Inputs before the jump are part of
the candidate. The candidate set is the same for every arm.

**The predictor.** A small network trained on the training starts only. Input: the start state relative to the
marble, the height crops, the candidate's controls. Output: the timed path through landing plus 0.5 s, the
probability of pickup, and the landing outcome.

**Evaluation on held-out starts.** Each start is run in the engine once per arm:
* **Learned:** the candidate with the highest predicted chance of pickup plus a safe landing.
* **Hand-written:** the same candidates ranked by the existing landing predictor in `nav/physics.py`
  (where it applies: hold forward or no input).
* **Fixed manoeuvre:** one simple rule, declared before the run (for example: jump at the first tick the lip
  is within a set distance, hold forward).
* **Random:** a random candidate from the same set.
* **Oracle (offline, not an arm):** run every candidate from each held-out start. It shows which starts have
  any successful candidate (the ceiling) and keeps impossible starts from counting as the predictor's failures.

Metrics: pickup rate, pickup plus safe landing rate, out-of-bounds rate, predicted against actual landing and
path error, pickup false positives and negatives, and how well predicted probabilities match outcomes.

**Pass.** Learned beats every baseline on pickup plus safe landing over the held-out starts the oracle shows
are feasible: at least 100 such starts, with the learned arm's 95 % interval clear of the best baseline's
(decision 4).

**Stop and report** when the measurement checks fail (fix measurement before anything else), when pass or fail
is reached, or at the end of the time budget: one or two sessions, a budget, not a promise of success.

**If P0 fails:** diagnose before changing the architecture. Is the measurement sound? Does the oracle find
solutions the candidate set contains? Is the predictor's error at the lip the problem? Is there enough data?

**What P0 does not establish:** approach planning from far away, route discovery, whole gem groups, PPO
handovers, or anything about other maps.

**Deferred until after P0:** general contact telemetry (an engine accessor only if the existing collision
information is insufficient), full geometry export and the triangle encoder, the step-model ensemble,
calibration and predictor arbitration, search and seeding, handovers, multi-map data.

## Decisions (operator, 2026-09-27)

1. **The target gem:** build a new drill map for P0, `kotmjump_p0`, a copy of kotmjump that contains only the
   target floating gem, with a long round time. Like kotmjump it stays out of PPO training. The operator
   allowed replacing and deleting the old kotmjump; it is kept for now because M0, the four-target drills of
   M3 and the full rounds of M5 use it.
2. **Approval:** the operator approved building the P0 prototype: new files under `nav/learned_nav/`, the
   spin words for the Python teleport, small `mlAgent.cs` additions the recorder needs, and the drill map.
   Engine (C++) changes, and work beyond P0, are asked for separately.
3. **Resources:** PPO training stays stopped during P0.
4. **Pass margin:** as proposed: at least 100 held-out starts the oracle shows are feasible, and the learned
   arm's 95 % interval clear of the best baseline's.
5. **Fixed reference maps (from M1 on):** **Horizon** (`hunt/advanced/Horizon_Hunt.mcs`, `Horizon.dif`) and
   **Archipelago** (`hunt/bonus/Archipelago_Hunt.mis`, `archipelagio.dif` in `interiors/custom/` and
   `interiors/custom-final/`). Never used for training, calibration or tuning. The whole geometry family is
   excluded, including Horizon's other modes (`snowball/advanced/Horizon_xmas.mcs` with `HorizonXmas.dif`,
   and `training/advanced/Horizon_Training.mis`). Archipelago replaced Fun in the Sun the same day.

Implementation has not started; the operator will say when.
