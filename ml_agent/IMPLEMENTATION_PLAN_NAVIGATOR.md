# Implementation plan: navigator + planner agent

Written 2026-09-16 (late). Companion to `ROADMAP_NAVIGATOR_PLANNER.md` (the
"why" and the architecture) and `CHAT_CONTEXT.md` (the current stack). This
document is the "how": work packages, file layout, data contracts, numbers,
tests and acceptance criteria, in the order they should be built. Status:
**status 2026-09-17: WP1-WP4 implemented as `nav/` (see CHAT_CONTEXT.md), stage-0 training started; WP5+ pending.** The old per-map trainer
(`train_ppo.py`) is left untouched as the King-of-the-Marble baseline; all new
code goes in a new package so the two never interfere.

---

## 0. Decisions carried in from the throughput work (2026-09-16)

1. **Control granularity.** The script bridge delivers one effective action
   and one fresh observation per wall-clock frame (engine frame loop batches
   script timers between physics batches; see roadmap 3.1). At 3x that is one
   action per 48 ms of sim, at 10x one per 160 ms. Per-tick control at speed
   needs the engine change. Consequences for this plan:
   - The navigator's **decision period is fixed at 64 ms of sim time (2
     physics ticks of 32 ms)**, the same as today's action repeat 4. Under
     the engine lockstep this is exact; under the script bridge at 3x it is
     "about 1.3 frames", good enough to develop and validate the code on.
   - **Do not train at 10x on the script bridge.** Speed comes from the engine
     work (fixed-step + lockstep + N instances), not from the time scale.
2. **Engine path runs in parallel** with everything that does not depend on
   throughput. Work packages below are tagged `[now]` (can start today on one
   3x instance) or `[engine]` (needs lockstep / multi-instance).
3. **Everything new is world-frame, map-independent, and per-instance.** No
   coordinates in observations, no per-map reward terms, all trainer state
   keyed by instance id.

---

## 1. Package layout

```
ml_agent/
  nav/                      new package, nothing imports train_ppo.py
    __init__.py
    protocol.py             one place for the socket line format (obs fields, reply fields, control words)
    env.py                  HuntEnv: one game connection as a step()/reset() environment; VecHuntEnv over N
    terrain.py              TerrainGrid: crops, walkability, A*, jump gaps (extends terrain_obs.TerrainMap)
    waypoints.py            goal sampling, segment bookkeeping, waypoint reward
    obs.py                  navigator observation builder (crops + vector), fixed layout + version tag
    model.py                NavActorCritic: CNN trunk + GRU + existing heads
    ppo_recurrent.py        PPO with sequence minibatches and stored GRU states
    train_nav.py            the navigator trainer (curriculum, checkpoints, logs, dashboard feed)
    eval_nav.py             deterministic held-out evaluation (fixed start/goal seeds per map)
    demo_relabel.py         human recordings -> waypoint-labelled navigator samples
    bc_nav.py               navigator behaviour-cloning bootstrap
    planner/
      heuristic.py          v0 planner: gem/group value, A* travel time, waypoint hand-off
      state.py              PlannerState: gems, opponents, spawn-rule state, powerups (from the observer)
    play_hunt.py            run navigator + planner in a live round (the "agent")
  terrain_maps/             one height stack per map (generate_terrain_map.py, batch script below)
  logs/nav/                 navigator training logs (analyze_log.py learns the new lines)
  models/nav/               navigator checkpoints (gitignored like the others)
```

Game side stays `Marble Blast Platinum/platinum/client/scripts/ai/`:
`mlAgent.cs` (bridge, control words), `observer.cs` (state), `socketBridge.cs`.
Additions are listed per work package.

---

## 2. Work packages

Each package lists: goal, files, contract, tests, acceptance. Estimates are
working days for one person with the game available; `[now]`/`[engine]` as
in section 0.

### WP1 `[now]` Protocol and environment wrapper (2 days)

Goal: a clean, testable Python environment over the existing bridge, so the
navigator code never touches sockets or string parsing directly.

- `nav/protocol.py`: parse `obs_json|gemDelta|oob|done[|inputs]|tick` into a
  dataclass `GameMessage(obs: np.ndarray, gem_delta, oob, done, tick,
  human_inputs)`; format replies `fwd,back,left,right,jump,camYaw,usePow[,t<tick>]`;
  control words `SPEED n`, `TELEPORT x y z vx vy vz`, `STATS`, `DELAY n`,
  `RECORD`. One function per direction, unit-tested against captured lines.
- `nav/env.py`:
  - `HuntEnv(port)`: owns one accepted connection. `step(action) ->
    GameMessage` (sends the reply, waits for the next message), `teleport(pos,
    vel)`, `set_speed(n)`, `reconnect()` (rounds end; the game reconnects
    every 2 s). Blocking, one thread per env.
  - `VecHuntEnv(ports)`: N envs stepped in lock step from the trainer's
    thread pool; returns stacked messages. Per-env counters (ticks, falls,
    rtf) and the three existing guards (rtf, offmap, cmd_accel_cos) live here.
  - Decision period: the env consumes 4 messages (64 ms) per `step()` and
    repeats the action, exactly like today's action repeat 4. When the
    engine lockstep exists this becomes 2 physics ticks.
- Tests: `tests/test_protocol.py` (round-trip of every message form),
  `tests/test_env_fake.py` (a fake game speaking the protocol; already used
  once for the replay tool: reuse `replay_demo.py`'s pattern).
- Acceptance: `python -m nav.env --smoke` connects to a running game, steps
  500 decisions with no-op actions, prints rtf and tick counts, exits clean.

### WP2 `[now]` Terrain layer (3 days)

Goal: every observation and every planner query comes from one map-agnostic
height-stack API.

- Batch height stacks: `generate_terrain_map.py --all` over
  `data/multiplayer/hunt/{beginner,intermediate,advanced,custom}`; writes
  `terrain_maps/terrain_<file>.npz` + PNG; a manifest
  `terrain_maps/manifest.json` (map, footprint, z-span, levels, moving
  platforms yes/no from PathedInterior count, generation date). Maps whose
  PNG looks wrong go on an exclusion list with a reason.
- `nav/terrain.py` `TerrainGrid(TerrainMap)`:
  - `crop(x, y, z, size=32, res=0.5) -> (C, 32, 32)` channels: nearest floor
    height relative to z clipped +/-10 and /10; floor-present flag; second
    level height (same scaling) or zero. World-axis aligned (frame is fixed).
    Vectorized with `np.take` on precomputed index grids; target < 0.2 ms.
  - `walkable(cell) -> bool`, `walk_grid(res=1.0)`: floor present, slope
    below the measured max (from the ramp maps), not within 0.4 u of an edge
    drop > 2 u.
  - `astar(start, goal, jump_gap=4.0, jump_drop=6.0) -> path, cost`: 8-connected,
    edge cost = distance x slope factor; gap edges allowed if the far cell
    is within `jump_gap` horizontally and not more than `jump_drop` below,
    cost x 3. Numbers are placeholders until WP5 measures the navigator.
  - `sample_goal(pos, dmin=8, dmax=40, rng)`: random walkable cell in the
    annulus with a finite A* path (so goals are always reachable).
- Tests: crops at known positions on King of the Marble (hole cells read
  "no floor"), A* on Lupus between two spawn points returns a path whose
  cells are all walkable, `sample_goal` never returns a hole.
- Acceptance: `python -m nav.terrain --check Sprawl_Hunt` renders crop +
  walk grid + one A* path over the PNG for visual sign-off.

### WP3 `[now]` Navigator observation + model (3 days)

- `nav/obs.py` `NavObs.build(msg, terrain, waypoint, hidden_state)`. Layout
  (version tag `NAV_OBS_V1` stored in checkpoints; any change bumps it and
  forces a critic warmup like today):

  | block | shape | source |
  |---|---|---|
  | fine crop | 3 x 32 x 32 (0.5 u) | `TerrainGrid.crop` |
  | coarse crop | 3 x 32 x 32 (2.0 u) | `TerrainGrid.crop` |
  | waypoint | 4: unit dx, dy; dist/50 clipped 1; dz/10 | waypoints.py |
  | self | 6: vx, vy, vz (/20); speed/20; on-floor flag; airborne ticks/32 | msg + history |
  | edge rays | 38 | `TerrainMap.edge_rays` (kept) |
  | powerup | 4 (v2): held one-hot(3) + active flag | observer, later |

  On-floor comes from vz history (|vz| < 0.05 for 3 ticks and floor height
  under the marble within 0.3 u), computed in Python; no engine calls.
- `nav/model.py` `NavActorCritic`:
  - crop trunk: conv 3->16 (5x5, stride 2), 16->32 (3x3, stride 2),
    32->32 (3x3), flatten -> 128, one trunk per crop (weights not shared).
  - vector MLP 48 -> 64. concat 320 -> 256 -> GRU(256).
  - heads reused from `train_ppo.Actor`: 2D direction Gaussian (unit-circle
    mean, clamped log std), throttle Gaussian, jump Bernoulli, brake
    Bernoulli; later use-powerup Bernoulli. Critic: linear on the GRU state,
    PopArt as today (copy the class, do not import train_ppo).
  - ~600k parameters; batch of 256 sequences x 32 steps forward in < 20 ms
    on the GPU.
- Tests: shapes; a synthetic "flat plane with a hole" crop makes the model
  run; `action_to_joystick` copied with its tests (direction -> keys in the
  world frame, throttle floor).

### WP4 `[now]` Waypoint task, reward, recurrent PPO (4 days)

- `nav/waypoints.py`:
  - `Segment`: start pose, goal, t0, ticks, path length (A*), outcome
    (arrived / fell / timeout), distance travelled.
  - `SegmentManager(env_id)`: on `reset` teleports to a random walkable cell
    (or leaves the marble where it is 50 % of the time, to keep natural
    states), samples a goal, ends the segment on arrival (< 1.5 u
    horizontal, < 2 u vertical), fall (oob flag), or 20 s.
  - Reward per decision (64 ms), all map-free:
    `r = 1.0 * (d_prev - d_now)            progress (A* path distance, not straight line)
       + 10.0 * arrived
       - 20.0 * fell
       - 0.02                              time
       - 0.1 * airborne_flag` (the existing free-fall cost, per decision)
    Progress uses the A* distance-to-goal field (one Dijkstra from the goal
    per segment, cached), so going around a hole is rewarded, not punished.
- `nav/ppo_recurrent.py`: rollout buffer of `(T=32) x (B)` sequences with
  the GRU state at each sequence start; PPO-clip, GAE (gamma 0.99, lambda
  0.95: segments are short), 4 epochs, 8 minibatches, clip 0.2, entropy
  0.005, value clip, grad norm 0.5, actor/critic LR 3e-4 with linear decay
  (fresh training, not the 1.5e-5 of the mature per-map policy). KL early
  stop 0.3 as today. Burn-in: the first 8 steps of each sequence recompute
  the hidden state without loss.
- `nav/train_nav.py`: loop over `VecHuntEnv`, per-env `SegmentManager`,
  curriculum controller (WP5), checkpoint every 50 updates to
  `models/nav/nav_<update>.pth` with obs version + curriculum stage + map
  set + eval numbers, log lines (`NAV upd=… arrive=… falls100=… speed=…
  stage=… maps=…`), dashboard feed (reuse the SSE server, new tiles).
- Tests: buffer indexing (sequence boundaries, hidden-state alignment),
  reward on a scripted trajectory, a 2-map fake env end-to-end for 3 updates.
- Acceptance (stage 0, one 3x instance, `[now]`): on the flat custom map with
  holes, arrival > 90 % and falls/100 u < 2 within 2 hours of wall time.

### WP5 `[now]` Curriculum, evaluation, metrics (2 days)

- Stages as in the roadmap (0 flat customs; 1 KOTM, Hunting Around, Marble
  Agility Course, Triple Decker, Basic Agility; 2 Lupus, Par Pit, Pyramid,
  All Angles, Core; 3 Sprawl, Marble City, Ziggurat, Cube Isle). Held out
  forever: Gems Ahoy, Playground, Horizon, Zenith, plus one per stage.
- With one instance the map changes by restarting the game on a new
  `-autotrain <Map>` (WP1's env does this through `run_game_loop.ps1`); with
  N instances each instance runs a different map of the current stage.
- Promotion rule: last 300 segments on the current stage have arrival > 90 %
  and falls/100 u < stage threshold (2.0, 1.0, 0.7, 0.5) -> next stage; mix
  25 % of earlier-stage maps to avoid forgetting.
- `nav/eval_nav.py`: for a map, 100 fixed (seeded) start/goal pairs, greedy
  policy, reports arrival %, falls/100 u, mean speed, time per 10 u, and
  writes `logs/nav/eval_<map>_<ckpt>.json`. Run on the held-out maps at every
  promotion and every 500 updates. **Milestone numbers live here**, never
  in training curves.
- `analyze_log.py`: parse `NAV` lines and eval files; dashboard tiles:
  arrival %, falls/100 u, speed, stage, per-map table.
- Measure the navigator's real jump gap / drop tolerance from eval traces
  and write them back into `TerrainGrid.astar` defaults (WP2 placeholders).

### WP6 `[now]` Demo bootstrap (2 days, optional but cheap)

- `nav/demo_relabel.py`: for every recording in `demos/`, split into
  segments at gem pickups (gem count increments; the picked gem is the
  nearest gem slot that disappears), label every tick of a segment with
  waypoint = that gem's position, build `NavObs` with the terrain grid of
  the recorded map, keep the copycat guard (history zeroing is moot with a
  GRU: instead train with random hidden resets) and the throttle mask.
- `nav/bc_nav.py`: clone the actor heads on the relabelled set, then hand
  the checkpoint to `train_nav.py` as the initial policy; compare stage-1
  time-to-promotion with and without. Keep only if it helps; the BC aux
  loss during PPO is NOT repeated (it regressed the per-map policy).

### WP7 `[engine]` Engine: fixed step, lockstep, multi-instance, headless (5-10 days, parallel track)

Gate for everything above 1 instance at 3x. Either RandomityGuy provides the
private tree / a build, or the changes are made on the public
OpenPQ-TGEMIT build and carried over later. All behind flags so the normal
game is untouched:

| flag | change | verify with |
|---|---|---|
| `-fixedstep N` | `TimeManager::process` posts TimeEvents of exactly N ms sim, back to back, no wall clock | `throughput_probe.py`: timer steps all N, rtf limited by CPU only |
| `-lockstep` | in `processElapsedTime`, after `serverProcess` of one tick: call a script hook `MLAgent::onTick`, then block on `Net::process()` until `$AI::replyReady`, then continue; one obs and one action per physics tick | `action_granularity_probe.py`: alternating F/B drifts < 0.05 u at any speed; `physics_identity_probe.py` 21 s script identical across speeds to 1e-3 |
| `-aiport N` (exists in script) + skip `excludeOtherInstances` | several exes on one machine; separate `prefs`/`console.log` paths via `-prefs <dir>` | 4 instances at once, each on its own port |
| `-headless` | no swap chain / render; `TextureManager::smDGLRender=false` and skip `syncRender` | 4-6 headless instances, CPU per instance, aggregate ticks/s |

With lockstep the `HuntEnv.step()` becomes exact (2 ticks per decision) and
`VecHuntEnv` runs N processes; the trainer code does not change. Target:
>= 150 decisions/s aggregate (roadmap milestone 1).

### WP8 `[now]` Heuristic planner v0 (4 days)

- Observer extension (game side, `observer.cs`): send ALL spawned gems
  (id, x, y, z, value) as a variable-length block, opponents (id, x, y, z,
  vx, vy, vz, score, held powerup id), leader id and score gap, alarm flag.
  Serialize as a second JSON field in the message (`|planner_json`) so the
  navigator's fixed-width obs is unaffected. No engine calls that print
  (the 2026-09-13 crash rule); powerup-held detection via the script-level
  `$MP::MyMarble.getPowerUp()` only if it is silent, else skip in v0.
- `nav/planner/state.py`: dataclass mirror of that block.
- `nav/planner/heuristic.py`:
  1. Group detection: gems within `radiusFromGem` of each other (mission
     info) form a group; group value = sum of points.
  2. Travel time per group: A* path length / measured mean speed (WP5) +
     3 s per gem in the group.
  3. Choose the group maximizing value / (travel + collection time), with
     a hysteresis of 20 % to avoid flip-flopping; re-plan every 0.5 s or on
     gem-set change.
  4. Waypoint = the A* path point 10-15 u ahead (shorter near the goal);
     hand the gem position itself when within 15 u.
  5. Between groups (no gems visible for > 1 s): head toward the centroid
     of walkable area outside the leader's spawn-block radius (from the
     score gap), or the map centre in single player.
- `nav/play_hunt.py`: navigator checkpoint + planner in a live round;
  logs score, falls, gems/min. Also the tool for the held-out score
  evaluation ladder.
- Acceptance: King of the Marble single-player score over 20 rounds vs the
  per-map baseline (83 avg / 105 best) and the human 166; then Lupus and
  Sprawl scores as the first multi-map numbers.

### WP9 `[engine]` Powerups in the navigator (3 days)

- Observation block "powerup" (WP3 v2) + use-powerup head; train the
  waypoint task on maps with super speed / super jump / helicopter with the
  item spawned near the start 30 % of the time (TELEPORT + the server's
  powerup give command, to be found in `powerups.cs`).
- Reward unchanged (faster arrival pays by itself).

### WP10 `[engine]` Learned planner v1 and self-play (open-ended)

- Set encoder over gems and opponents, PPO at 1 Hz over the frozen
  navigator, reward = score delta + fall cost, action = target group
  (pointer) + use-powerup + blast.
- Self-play: N instances joined to one lobby (needs the multi-instance
  engine and the lobby scripts driven by `-autotrain` host/join); frozen
  snapshots as opponents; league sampling.
- Out of scope until WP8 beats the baseline on three maps.

---

## 3. Order and what can start immediately

```
week 1   WP1 protocol/env        WP2 terrain batch + crops/A*        WP7 engine (parallel, ask RandomityGuy)
week 2   WP3 obs/model           WP4 reward + recurrent PPO          WP7 cont.
week 3   WP4 stage-0 run on one 3x instance (acceptance)   WP5 eval/curriculum   WP6 demo relabel
week 4+  with the engine: N instances, stages 1-3 (WP5)    WP8 planner v0 (can be written offline against WP2 earlier)
```

Nothing in weeks 1-3 needs the engine. Stage-0/1 training on one instance
at 3x is slow (about 47 decisions/s) but is enough to prove the code path
and the reward; the numbers that matter come after WP7.

---

## 4. Data contracts to freeze early

- **Message**: `obs_json|gemDelta|oob|done[|inputs][|planner_json]|tick`
  (tick is always last; the planner block is optional and only sent when
  the Python side has replied `PLANNER 1` once).
- **Reply**: `fwd,back,left,right,jump,camYaw,usePow[,t<tick>]` plus the
  control words. The delay tag stays optional (off by default).
- **Checkpoint** (`models/nav/*.pth`): `obs_version`, `model_cfg`, `stage`,
  `maps_trained`, `eval` dict, optimizer state; loading with a different
  `obs_version` refuses unless widening rules are written for it.
- **Terrain manifest**: `terrain_maps/manifest.json` as in WP2; the trainer
  refuses a map without an entry.
- **Log line** `NAV upd=N steps=… arrive=… falls100=… speed=… stage=… maps=…
  rtf=… late=…` and the eval JSON schema (map, ckpt, n, arrival, falls100,
  speed, t10u, seeds).

---

## 5. Risks and how each is checked early

| risk | check | fallback |
|---|---|---|
| height stacks wrong on multi-interior / moving-platform maps | WP2 PNG sign-off per map, exclusion list | static-only maps first; platforms later as observed objects |
| crop too small for high-speed travel (16 u at 8 u/s = 2 s) | stage-2 falls concentrated at crop edge (trace) | coarse crop already covers 64 u; raise fine crop to 48 cells |
| GRU training instability | stage-0 acceptance in 2 h; compare against a no-GRU frame-history variant kept in `model.py` | frame history (known to work) |
| progress reward exploited (oscillation near goal) | arrival rate vs progress reward ratio in logs | potential-based shaping only, arrival radius 1.5 u, timeout |
| script bridge jitter at 3x hides bugs | run stage 0 also at 1x once; compare | engine lockstep |
| observer extension prints and crashes the game | 1-hour soak with the planner block on | drop the offending call |
| single instance too slow to reach stage 1 before the engine lands | measure decisions/s in WP1 | prioritize WP7; surrogate physics is plan B (roadmap 6) |

---

## 6. Definition of done per milestone (numbers on held-out maps)

| milestone | metric | target |
|---|---|---|
| M1 env + terrain | smoke test, all stage 0-3 maps in the manifest | 100 % of listed maps, PNGs signed off |
| M2 navigator stage 0 | arrival, falls/100 u on the flat custom map | > 90 %, < 2 |
| M3 navigator stage 1 | same on a held-out small map | > 90 %, < 1, >= 6 u/s |
| M4 engine | aggregate decisions/s, determinism probes | >= 150/s, granularity drift < 0.05 u |
| M5 navigator stage 3 | held-out maps | > 90 %, < 0.5, >= 7 u/s |
| M6 planner v0 | KOTM score, 20 rounds | > 105 avg; then Lupus/Sprawl reported |
| M7 powerups | same maps with powerups on | speed up, no fall increase |
| M8 planner v1 / self-play | 1v1 vs frozen snapshots, then human recordings | wins > 70 %, then > 166 |

---

## 7. What training and production look like

### Training, phase by phase

1. **Navigator training** (WP4-5). N game instances run rounds on stage
   maps with the gems ignored. `train_nav.py` owns every marble: each
   instance gets a random start (teleport) and a random reachable goal,
   the network drives it there, the segment ends on arrival, fall or
   timeout, and a new goal is drawn. Millions of such segments across many
   maps teach one skill: "read the local terrain and reach a point fast
   without falling". No map is memorized because the goal, start and map
   change constantly and the observation never contains coordinates. The
   output is one checkpoint that works on maps it never saw (measured on the
   held-out set).
2. **Planner v0** (WP8) is not trained: it is A* over the height stack plus
   a value rule for gem groups. It is measured, not trained: score per
   round with the frozen navigator underneath.
3. **Planner v1** (WP10) is trained with the navigator frozen: one decision
   per second (which group, use powerup, blast), reward = score. Because
   the navigator already handles locomotion, this policy learns strategy
   only, and it learns it against copies of itself in a shared lobby
   (self-play), so opponents are always as good as the current agent.

### Why this plays better than the current agent

- The current agent is one network that must learn locomotion AND gem
  choice AND the map at the same time from sparse gem rewards, on one map,
  from one game instance. Its falls (~6-9 per game) come from memorizing
  coordinates; its route choice is "nearest gem".
- The navigator gets a dense, fully informative signal (distance to a known
  goal) on every decision, across many maps and N instances, so locomotion
  is learned orders of magnitude faster and transfers.
- Route choice is solved exactly by search (A* over the real geometry)
  instead of being learned; group value and the spawn rule are computed
  from the game's own rules. The current agent cannot represent either.
- Opponents, powerups and blast are added as inputs and actions to the
  planner without retraining locomotion.

### Production: playing a real person

Three processes, one machine, the normal game at 1x with the window visible:

| process | what runs | what it does |
|---|---|---|
| PlatinumQuest client (`marbleblast.exe`) | the unchanged game plus the `ai/` scripts (`mlAgent.cs`, `observer.cs`, `socketBridge.cs`) | hosts or joins the multiplayer lobby like any player; every 16 ms sends the observation line (marble state, all gems, opponents, time, powerups) to port 8888 and applies the reply as key presses in the marble camera frame |
| agent (`nav/play_hunt.py`) | navigator checkpoint (`models/nav/nav_<n>.pth`) + planner (heuristic v0 or learned v1 checkpoint) + the map's height stack | **every 64 ms**: build the crops and vector obs, run the navigator, reply with direction/throttle/jump/brake/use-powerup. **every 0.5-1 s**: planner picks the target group and hands a waypoint 10-15 u ahead along the A* path (v1: also powerup/blast decisions) |
| dashboard (optional) | `dashboard.py` | live score, falls, target group, planner choice, for watching |

Nothing else is needed: no trainer, no multiple instances, no engine
modifications (the engine work only speeds up training). The agent process
is CPU-light (one small network forward per 64 ms, one A* per re-plan) and
runs on the same PC as the game. The human plays from another machine
joined to the same lobby, exactly as in the recorded sessions.
