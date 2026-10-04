# Powerups: plan (2026-10-02, draft for the operator)

Goal: King of the Marble around 180 with powerups, and powerup use that transfers to every Hunt map: the model
"knows" what each powerup does in any situation on any map and uses it to get round the terrain.

## 0. What is in the game (checked in the 60 Hunt mission files)

| item | items in 60 Hunt maps | on KOTM | positions on KOTM |
|---|---|---|---|
| Super Jump | 253 | 3 | (-19.2, 3), (-13.2, 21), (-37.2, 21) |
| Super Speed | 234 | 3 | (-31.2, 3), (-19.2, 27), (-13.2, 9.2) |
| Blast (the pickup) | 222 | 2 | (-31.2, 27), (-25.2, 15) |
| Mega Marble | 186 | 1 | centre area |
| Helicopter | 127 | 0 | |
| Shock Absorber / Super Bounce | 9 / 8 | 0 | |

Operator (10-02 17:10): IN SCOPE are exactly seven: Super Speed, Super Jump, Helicopter, Super Bounce, Shock Absorber,
Mega Marble and Blast; nothing else (no time travel, anti-gravity, teleport, fireball, bubble, anvil, party items).
The observation encodes those seven types; other items are ignored. Plus the regular blast meter every marble has in
Hunt mode. Items respawn 7 s after pickup (item.cs / powerups.cs
default). The operator's rules of the game (10-02): the meter fills linearly with time to a full mark; a blast is
usable from a threshold up, and its height is proportional to how full the meter is; the pickup Blast is a few
percent stronger in height (and pushes opponents, out of scope); Mega Marble makes the marble larger and slightly
slower for about 5 s (opponent knock-back out of scope), a small hole does not stop it, jump-then-mega gives an
extra height boost, and mega activated in the air while spinning fast launches the marble forward on a high-friction
landing; picking up a different type replaces the held powerup, the same type is not picked up; Super Speed is the
only directional powerup, fired along the camera yaw, any direction regardless of the velocity.

Engine facts already known (marble.cc, log 28.57 and 36): Super Speed = 25 u/s impulse along the camera yaw
projected onto the contact plane; Super Jump = an impulse along gravity; Helicopter = gravity x a multiplier and air
control x a multiplier; the bridge already carries a use-powerup action word (always 0 today) and can set the camera
yaw per decision; the held powerup is readable on the listen server. The observation carries NOTHING about powerups.
The human's 177 / 172 / 163 rounds used no powerups, so 180 on KOTM is a hypothesis to test, not a measurement.

## 1. Who knows what, who decides, who presses

**Told explicitly (observation / planner inputs), because the model cannot infer them in time:**
* the held powerup type (none / Super Speed / Super Jump / Helicopter / Blast / Mega), and the meter: the regular
  blast meter value (0 to 1), mega active and seconds left;
* the powerup items near the marble, like the gems: type, offset, and "spawned now or seconds to respawn" (the
  bridge can time each pickup; 7 s is the rule);
* the rules that are constants: respawn time, mega duration, the switch / same-type rules (these go into the route
  chooser and the planner as numbers, and the policy sees their effects through the observation fields above);
* the Super Speed direction is part of the ACTION (a yaw word), never inferred.

**Learned by the navigator (from the game's points, nothing else, per the operator's rule):** when a use pays,
routing through powerup items, handling the mega marble's size and speed, the spin-launch trick. The planner gets
the measured physics of every powerup as equations and uses them in its rollouts, so "what does X do here" is
computed on any map; the navigator learns to exploit it.

**Who presses, in two steps (the same scheme that worked for the jump hint):**
1. The planner decides: with the measured physics it evaluates "use the held powerup now (aimed at d)" like a jump
   plan: landing, gem pass, stopping room, falls. When a use wins it sets the use bit (and the yaw) on the
   navigator's action. The navigator keeps driving. This is safe from day one and needs no new training.
2. The navigator owns it: trained with the use action and the yaw in its action space, the planner's decision as the
   teacher in the first run (the bit is set for it, it sees the consequences), then on its own with the held
   powerup and the meter in its observation. Gates decide when the hand-over is earned.

## 2. Phases

### Phase 1: instrument (bridge + observation, no behaviour change), ~half a day
* observer.cs / mlAgent.cs: append to the observation the held powerup, the blast meter, mega state and time left,
  and the nearest powerup items (type, offset, spawned / seconds to respawn), read from the live ServerConnection
  like the gems (never the cached item list, log trap). Observation version V7; the existing zero-column migration
  keeps the current checkpoint's behaviour until training moves the new weights.
* Action: the use bit (exists) plus a powerup yaw word for Super Speed; the joystick adapter's steering yaw must not
  be overwritten on the fire tick (log 28.57 gotcha).
* Control word to read the engine's powerup constants and a MEASURE helper for phase 2.
* Verify both ends of the contract with a drill (teleport, pick up, read back).
* Needs the operator's approval for mlAgent.cs / observer.cs edits (no engine change).

### Phase 2: measure every powerup against the engine, ~half a day of drills
Fixed programs from teleported states on the drill map and KOTM, recorded per decision, the same method as the
flight physics (log 36):
* Regular blast: meter fill rate (seconds to full), the usable threshold (sweep the meter in 1 % steps: the first
  value where the use takes), and the height / vertical speed gained as a function of the meter value over the whole
  usable range (a table the planner interpolates); whether a blast in the air works; what it does to spin and speed.
* Pickup Blast: the same curve (the operator: 1-2 % higher); confirm whether it uses the meter.
* Super Jump: the vertical impulse, from floor and in the air, at 0-15 u/s.
* Super Speed: the impulse magnitude (25 u/s expected), that it follows the yaw regardless of the velocity, from
  rest and at speed, on floor and in the air, the speed cap if any.
* Helicopter: the gravity and air-control multipliers and the duration (checked on a helicopter map).
* Mega Marble: radius, duration, speed and acceleration on the floor, braking, jump height, which holes it rolls
  over (the KOTM 1.3 u groove and the 3 u strips), the jump-then-mega height boost, and the spin-launch: spin up,
  jump, activate in the air, land on a high-friction surface, record the launch speed as a function of spin.
* Rules: respawn time per item type, the switch rule, the same-type block, mega ending while airborne.
Output: a constants file the planner and the route chooser read, and a log section with every curve.

### Phase 3: physics into the planner and the route chooser, 1-2 days
* planner.simulate gets powerup effects as analytic steps: Super Speed impulse (yaw), Super Jump / blast vertical
  impulse by meter value, Helicopter gravity and air-control multipliers, Mega radius / speed / duration (the learned
  ground model is still the ground model; the mega's size changes the geometry queries: a larger R).
* Proposal families: "use now toward d" and "pick up item P on the way"; judged like jumps (landing floor, gem pass,
  stopping room, perturbed-start robustness).
* Route chooser (the whole-spawn order the navigator is now trained on): powerup items as optional waypoints with
  the measured time effect on the next legs (a Super Speed leg time, a Super Jump crossing where the plain jump
  cannot), so the order itself can choose "the Super Speed, then the two far gems".
* The hybrid's hint path: the planner's use decisions set the use bit (and yaw) on the navigator's action.
* Gates: KOTM 8 rounds navigator-alone and hybrid; the two held-out practice maps (GemsAhoy, Acropolis2) for
  transfer; never Horizon / Archipelago.

### Phase 4: train the navigator with powerups, 1-2 days of runs
* Action space + use bit + yaw; observation V7; whole-spawn chooser with powerup waypoints; all 8 instances on
  KOTM first, then the map rotation for transfer.
* Run 1: the planner's hint presses for the policy (safe uses only); the policy learns the routes and the pickups.
* Run 2: the policy presses itself; the hint stays as a safety check if run 1 shows falls from uses.
* Judged by snapshot gates (NAV_CKPT), by time per gem and falls, not by the training bars (log 28.52 lesson).

### Phase 5: other maps
Helicopter maps and Super Jump maps in the rotation (KOTM stays the base, 7 KOTM / 1 other as before), the
transfer gates on the held-out maps, then the planner's crossing drill on those maps with powerups allowed.

## 3. Questions for the operator
1. Opponents stay out of scope (Blast push, Mega knock-back): the model only needs the navigation effects. Yes?
2. Approval to edit mlAgent.cs and observer.cs for phase 1 (no engine change expected).
3. For the measurements I will build a flat measurement map (like kotmjump_p0) with every powerup type placed; the
   mega spin-launch needs a high-friction surface: which maps have one, or is the default floor enough?
4. Is "180" the navigator-alone number, the hybrid, or either? (Today: navigator alone 160 / 170 best.)
5. The whole-spawn gem chooser stays the default for everything from here on. Yes?


## 4. Status (2026-10-02 evening)
* Phase 1 DONE (observer V7, action words 8-9, bridge yaw latch, mega server command): log 38.1.
* Phase 2 DONE for all seven (constants nav/learned_nav/powerup_physics.py): log 38.2-38.4.
* Phase 3 PARTLY DONE and closed: planner rollout effects + use_oracle + hybrid hook (log 39). Three 8-round gates
  of the hybrid with the oracle (g7e 158.9, g8a 161.5, g8b 160.0) sit on the navigator's own 160.1: the rule
  fires about once in 8 rounds. Route-chooser powerup waypoints and planner proposal families with uses were
  NOT built (nothing to build on when the rule never fires; log 40).
* Phase 4 RUNNING since 2026-10-02 21:52: a learned `use` action (model.py ACTION_DIM 6, USE_BIAS -3), the
  worker turns it into the bridge words (Super Speed along the commanded direction, blast when nothing is held),
  from checkpoint 28897, KOTM x8, reward = points only. GAME lines carry `pow=` use / fire counts. Gate: 8 rounds
  navigator alone and hybrid against 160.1 / 170 best (log 40).
* 2026-10-03 23:05: phase 4 continues as the Super Speed turn (log 40.11-40.22; overnight handoff
  OVERNIGHT_2026-10-03_SUPERSPEED.md). Obs V9 (aim + approval + approach flag), use prior sampled >= 27 % where
  approved, resolved speed capped at 24 u/s. Not built: powerup items as route waypoints with a time credit.
