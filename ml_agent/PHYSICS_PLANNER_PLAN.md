# Learned jump physics: working plan (DRAFT v1, 2026-09-26)

Status: draft for iteration with the operator. D5 is done (spin is in the observation since NAV_OBS_V6); the
rest is not built yet, and decisions D1-D4 and D6 are open. Related: `PHYSICS_SKILLS_DESIGN.md` (2026-09-23; this
plan is its Option C, a learned physics model, promoted to the main approach by the operator).
Current state of the navigator: `HANDOFF_NAV_TRAINING.md`.

## 1. The idea (operator, 2026-09-26)

Learn the physics BEFORE play, not during it:
1. **Measure exhaustively, offline.** Use the game engine as ground truth to generate a very large number of
   jumps from many states on many maps, and record exactly what happened.
2. **Train a jump-outcome model** that takes the marble's state and the surrounding geometry, and predicts what
   happens if it jumps at this instant: where it lands, and how fast and in which direction it is moving when it
   lands. Landing speed lets the navigator judge whether the marble can stay on the floor after landing.
3. **At play time the prediction is one network forward pass** (well under 1 ms). No simulation during a round.

The navigator internalises the physics through this model, trained in advance. Nothing is written for a specific
map: the data comes from real engine outcomes, so every map, slope and edge the data covers is learned the same
way.

## 2. What the jump-outcome model predicts

For "jump at this instant", with a fixed assumption about what is pressed in the air (see decision D1):

| output | why |
|---|---|
| outcome class: lands on floor / hits a wall / falls (no floor within 2.5 s) | the go / no-go decision |
| landing point (x, y, z relative to the marble) | is that where I want to be |
| velocity at landing (speed, heading, vertical speed) | too fast to stay on after landing? |
| flight time | timing |
| K arc points (e.g. every 0.1 s) | does the flight pass a gem (pickup radius ~1.0 u, measured)? |
| confidence | near-miss lips and edges are genuinely knife-edge cases |

## 3. Inputs

* **Marble state:** velocity (3), spin (3; in the navigator's observation since V6, see D5), on-floor / contact
  normal.
* **Geometry around the marble:** height crops like the navigator's existing ones (fine 0.5 u and coarse 2 u,
  two levels), rotated to the heading so the model does not have to learn every direction separately.
* **Air input assumption** (D1), if the model is trained for more than one.

## 4. How the navigator uses it

1. **"Jump now" features in the observation.** Outcome, landing point relative to the gem, landing speed. These
   replace today's hand-built gap block (lip distance, gap length, "lands if forward held", speed ratio).
2. **"What if" features for the run-up.** This is the gap in the current system. Because the model is cheap, each
   decision it can be queried for a small grid of hypothetical states from the current position, for example 16
   headings x 8 speeds = 128 queries in one GPU batch, about 1 ms. From that grid the navigator gets: the best
   landing toward the gem, the heading and minimum speed it needs, and how far the marble is from that. The
   policy then learns to build that speed on that line. It gets a target instead of having to discover one.
3. **Later, optionally: an auxiliary head.** The same predictions as a supervised second task on the policy's own
   network, so the policy's internal features encode physics. See D3.

The policy still learns WHEN to jump through RL, but from accurate, general inputs instead of a guessed flag.

## 5. Data generation (offline)

Engine as ground truth, using the existing lockstep machinery. No engine C++ changes are needed to start.

* **Set the state exactly:** `TELEPORT x y z vx vy vz wx wy wz`. Spin support was added and verified live on
  2026-09-26 (section 10). Use rolling spin = (-vy, vx, 0) / 0.19 so a teleported marble does not skid.
* **Resolution:** run at `FIXEDSTEP 16` (or 8) so landing points are exact to about 0.1 u. At 64 ms a 10 u/s
  marble moves 0.64 u per tick.
* **One trial:** teleport, settle 1-2 ticks, press jump, apply the air input, record every tick until landing plus
  about 0.5 s (to see whether it stays on), or 2.5 s with no floor.
* **Sampling (per map):**
  * positions: floor points, weighted toward the 0-8 u next to a lip or edge;
  * headings: any;
  * speed: 0-20 u/s;
  * air input: per D1.
  * Oversample near-miss states around lips, where the outcome flips.
* **Throughput estimate:** at ~60x real time per instance and ~2 s of game time per trial, 8 instances give on the
  order of 200 trials per second, about 1M trials in 1.5 h per map. Measure the real rate in the first batch; the
  finer timestep will slow it.
* **Maps:** KOTM, kotmjump, FlatIslands, FlatWithJump, JumpOnly first, then the complex maps (see D4). Hold out
  whole maps to test generalisation.

## 6. Training and validating the predictor

* **Network:** CNN over the rotated crops + MLP over the state -> heads for class, landing point, landing velocity,
  flight time, arc points.
* **Losses:** cross-entropy for the class, regression for the rest (only on "lands" samples for the landing
  fields).
* **Pass criteria, on held-out states from training maps:** class accuracy >= 98 %; median landing error < 0.3 u;
  landing speed error < 0.5 u/s.
* **Pass criteria, on a held-out map:** report the same numbers. This is the transfer test.
* **Compare against the current hand-written predictor** (`nav/physics.predict_landing`) on the same samples.

## 7. Milestones

* **M0. Data generator:** spin on teleport, fine timestep, trial recorder. Pass: 20 repeats of one trial give
  identical trajectories; spot-check against `nav/measure_arc.py`'s captures.
* **M1. KOTM + kotmjump dataset** (~1-2M trials).
* **M2. Predictor trained and validated** (section 6 criteria).
* **M3. Multi-map dataset and held-out-map test.**
* **M4. Wire into the navigator:** "jump now" and "what if" features replace the gap block (vector 53-57) and the
  jump prior. New observation version; the checkpoint is migrated.
* **M5. Train, then test kotmjump at 1x.** Targets: every gem group completed; >= 80 % of floating-gem jumps take
  the gem and land; KOTM not worse (8 rounds at 3x against the current best in `HANDOFF_NAV_TRAINING.md`).

## 8. Open decisions for the operator

* **D1. Air input during the predicted flight.** (a) Hold forward along the heading (what the current predictor
  assumes; the human does this). (b) No input. (c) Both, as two predictions. Proposal: (c), which is cheap in
  data and lets the policy see what holding forward buys.
* **D2. Post-landing outcome.** Predict only the landing state (operator's ask), or also "stays on the floor
  vs rolls off" for about 0.5 s after landing, given a default input such as braking? Proposal: landing state
  first; add the post-landing class if landing speed alone proves hard for the policy to use.
* **D3. Separate model vs auxiliary head.** A separate, frozen predictor is testable on its own and reusable on
  every map. An auxiliary head truly internalises the physics but mixes supervised and RL training (behaviour
  cloning has hurt before). Proposal: separate model first.
* **D4. Which maps feed the data,** and which complex maps to hold out as the transfer test.
* **D5. Spin in the observation. DONE 2026-09-26 (NAV_OBS_V6).** The observer sends the angular velocity; the
  navigator gets it at vector 58-60 as a rolling velocity (r * (wy, -wx, wz) / 20, r = 0.19). A 2.5 h training
  run showed no regression; the A/B of real against zeroed spin is in `HANDOFF_NAV_TRAINING.md` section 2.
* **D6. The current jump stack** (terrain-map gap scripts, "crossable" flag, jump prior): retire it at M4, or keep
  the prior until the new features are proven? Proposal: replace the features at M4, keep the prior only if
  kotmjump fails without it.

## 9. Risks

1. **Geometry input.** A 2.5-D height crop cannot represent overhangs, tunnels or ceilings. Fine for KOTM-like
   maps; complex maps may need an occupancy (voxel) crop.
2. **Knife-edge cases.** Nearly identical states can land or clip a lip. Needs dense data there and the
   confidence output; the policy should treat low confidence as risk.
3. **Teleport fidelity.** The data is only as good as how exactly a teleport reproduces a real rolling state (spin,
   contact). M0 checks it against real captured jumps.
4. **The policy still has to learn to act.** Better inputs are not a guarantee: the RL still has to learn to build
   the run-up. The "what if" target features are meant to make that easy; kotmjump is the test.
5. **Moving platforms** are not in the data. They do not exist on the first maps.

## 10. Verified facts (live game, 2026-09-26)

* **Spin can be READ in a live game.** `Marble.getAngularVelocity()` works on both the client marble and the
  server player object, and they report identical values. Rolling east at full input, the reported spin was
  exactly the rolling axis (cosine 1.000 to the expected axis at every sample) with speed / spin = 0.190 at every
  speed from 1 to 12.4 u/s. So the **marble radius is 0.19 u**; `nav/physics.py` assumes a centre height of 0.25,
  a small known error. Probe: DEBUG word, fields `clientOmega`, `serverOmega`, `clientVel`, `serverVel`
  (mlAgent.cs).
* **Spin can be SET.** `TELEPORT x y z vx vy vz wx wy wz` (mlAgent.cs, optional words 7-9; without them spin is
  zeroed as before) sets it on both objects; it reads back exactly. The physics uses it:

  | teleport | speed at 0.08 / 0.16 / 0.32 / 0.64 s (no input held) |
  |---|---|
  | 10 u/s, spin 0 | 8.92 / 7.80 / 6.46 / 4.73 (skids) |
  | 10 u/s, rolling spin (52.7 rad/s) | 9.60 / 9.15 / 8.24 / 6.42 |
  | 0 u/s, spin only (52.7 rad/s) | 1.08 / 2.20 / 2.59 / 1.88 (spin turns into motion) |

* **"No input" is not coasting.** With no key held the engine brakes the marble (desired spin 0), which is why
  even the matched-spin trial slows. The data generator must hold the intended input throughout, including in the
  air.

## 11. Change log

* v0 2026-09-26: planner that runs engine simulations during play (Option A). Superseded the same day. The
  operator wants the physics learned offline and used at negligible cost during play.
* v1 2026-09-26: learned jump-outcome model trained offline on engine data (this document).
* v1.1 2026-09-26: section 10 added. Spin can be read and set in a live game (probe results); marble radius
  0.19 u; "no input" brakes the marble.
* v1.2 2026-09-26: D5 done (spin in the observation, NAV_OBS_V6). References point at the new current-state
  handoff.
