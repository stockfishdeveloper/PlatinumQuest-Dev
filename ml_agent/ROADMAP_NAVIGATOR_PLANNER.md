# Roadmap: from a King-of-the-Marble policy to a multiplayer hunt agent

Written 2026-09-16. Status: **planning document, nothing here is implemented yet.**
Read `CHAT_CONTEXT.md` first for the current stack; this document assumes it.

Goal restated: play Hunt mode in multiplayer better than any human, on the
whole map catalog, including powerups and offensive play (blast, mega marble).

---

## 1. Where we are and why the current formulation caps out

Current agent (2026-09-16): one PPO policy trained on King of the Marble for
~2 weeks. Best 100-game average ~83 pts, best game 105, ~6 falls/game.
A strong human: ~166 pts/round, 0 falls, 8.6 u/s (our recordings).

What was proven along the way (keep):
- Game <-> Python bridge (`mlAgent.cs`, `socketBridge.cs`, `train_ppo.py`), 62.5 Hz
  tick protocol, action repeat 4, world-fixed observation frame (2026-09-14 fix),
  game-speed guard (real-time factor), critic warmup on observation changes.
- Terrain tooling: `generate_terrain_map.py` (height stack from .dif),
  `terrain_obs.py` (point samples + edge rays). Verified against map geometry.
- Human demo pipeline: `record_demos.py` (auto handshake, atomic saves),
  `demo_data.py` (copycat guard, throttle mask), `bc_pretrain.py`.
- Diagnostics: per-decision trace, `probe_frame.py`, `replay_demo.py`,
  `analyze_log.py`, dashboard with ray-uptake / falls charts.

What caps out (retire, in stages):
1. **One policy per map, from scratch, one game instance.** ~50 decisions/s at
   3x; 1e8 decisions = 3 weeks. 61 hunt maps.
2. **Falls learned by memorizing coordinates.** Absolute position + nearest-gem
   offsets is a lookup table of one map; nothing transfers.
3. **Nearest-5 gems, 16 s horizon, no memory.** Half of a human's "detours"
   are routing to the 2nd/3rd gem of a group; the spawn rule (section 2) is
   invisible to the policy.
4. **No opponents, no powerups.** Each is a new head on a per-map MLP and a
   new multi-week campaign.

---

## 2. What the target contains (survey of `data/multiplayer/hunt`, 61 maps)

| | King of the Marble | typical hunt map | hard end |
|---|---|---|---|
| gem spawn points | 12 | 100-300 | 570 (Wrap Zone), 659 (Citadel) |
| footprint (u) | 24 x 24 | 100 x 120 | 505 x 508 (Citadel) |
| vertical span (u) | 0 | 10-30 | 104 (Citadel), 186 (Gravity Tower) |
| moving platforms | 0 | 0-6 | 13 (Wonky Waters) |
| powerups on map | 9 | 20-60 | 265 (Citadel) |
| round length | 3 min | 5 min | |
| special items | none | helicopter, anti-gravity | teleport, fireball, bubble, cannon |

Lupus: 38x31, z-span 20, 79 spawns, 1 moving platform, 12 path nodes.
Sprawl: 122x128, 178 spawns, 40 powerups, 13 interiors.

Game rules that decide strategy (server/scripts/huntGems.cs, powerups.cs):
- Gems spawn as a **group of up to `maxGemsPerSpawn` (4-7) around a random
  center gem within `radiusFromGem` (15-20 u)**. Next group spawns when the
  current one is collected.
- In multiplayer the next group is **blocked from spawning near the leader**;
  the block radius scales with the score gap (0-3x). Winning play = be where
  the next group will appear (away from the leader, near the trailing player).
- **Blast** pushes nearby marbles; **Mega marble** makes you bigger/heavier
  (5 s timeout in competitive hunt). Both are used to displace opponents from
  a group. Super speed / super jump / helicopter are traversal powerups.
- Scoring: red 1, yellow 2, blue 5 (platinum on some maps).

---

## 3. Target architecture

Three parts, matching how the game is structured. Build in this order; each is
useless without the previous one.

```
                +---------------------------+
   full map --> |  PLANNER (global, ~1 Hz)  |  which gem/group, which powerup,
   all gems --> |  v0 heuristic, v1 learned |  blast/mega, where to wait
   opponents -> +-------------+-------------+
                              | waypoint (x, y[, z]), flags (jump ok, use powerup)
                +-------------v-------------+
   local -----> |  NAVIGATOR (local, 16 Hz) |  reach the waypoint fast without falling
   height crop  |  one network, all maps    |
   velocity --> +-------------+-------------+
                              | fwd/back/left/right, jump, brake, use-powerup
                        game (N instances)
```

### 3.1 Throughput first: N game instances

Prerequisite for everything else. Machine: 24 logical cores.

- Run **4-6 game instances**, each with its own `mlAgent.cs` bridge, all
  connected to one trainer (or a vectorized trainer). Each instance is an
  independent environment; rollouts are concatenated per update.
- Engine support to use: `-mission`/`-autotrain` startup hook (mlAgent.cs),
  dedicated-server mode exists (`$Server::_Dedicated`, main.cs), prefs
  `backgroundSleepTime = 0` (already set) so unfocused windows keep full speed.
- Needed changes:
  - `socketBridge.cs`: port from a command-line arg (`-aiport N`) instead of
    the hard-coded 8888; `run_game_loop.ps1` launches N instances with
    distinct ports (and distinct prefs/console.log locations if the engine
    locks them).
  - `train_ppo.py`: accept N connections, tag every transition with an
    instance id, keep per-instance frame history / pending decision / game
    counters (today these are single-instance attributes of `PPOServer`).
    One shared rollout buffer; update when the total reaches `rollout_size`.
    Guards (rtf, offmap, cmd_accel_cos) per instance.
  - Measure CPU/GPU per instance at 3x before deciding N. If the engine cannot
    run >2 instances usefully, fall back to section 6 (surrogate physics).
- Later: instances joining one lobby = self-play opponents for free (3.4).

### 3.2 The navigator (map-independent locomotion skill)

**Task.** Given a waypoint, reach it fast without falling. One network for all
maps. Trained across many maps at once with random start/goal pairs.

**Observation (all local, all computable from the existing height stack):**
- *Fine crop*: 32 x 32 cells at 0.5 u (16 u each way), world-axis aligned
  (frame is world-fixed since 2026-09-14). Channels: (a) nearest floor height
  relative to the marble, clipped to +/-10 u and /10; (b) floor present (0/1);
  (c) optionally second-level height (for multi-level maps; the stack has 4
  levels). Built by a new `TerrainMap.crop(x, y, z)` next to `sample()`.
- *Coarse crop*: 32 x 32 cells at 2 u (64 u span), same channels.
- *Waypoint*: direction unit vector, distance /50 clipped, dz /10.
- *Self*: velocity (world, /20), vz, on-floor flag (from vz history), speed.
- *Short memory*: GRU hidden state (64-128) instead of the current 4-frame
  history. Rationale: copycat problems were with cloned history; a recurrent
  state trained by RL is fine and handles airborne phases.
- *Powerup state* (later): held powerup one-hot, active-effect flags.
- Keep the existing edge rays too (cheap, complementary).

**Network.** Small CNN per crop (3 conv layers, 16-32 channels) -> concat
with the vector inputs -> GRU -> the existing heads (2D direction Gaussian,
throttle, jump Bernoulli, brake Bernoulli). Reuse `Actor` heads and
`action_to_joystick`; only the trunk changes. Critic shares the trunk or
mirrors it.

**Reward (map-free, dense):**
- progress: `k_p * (d_prev - d_now)` toward the waypoint (potential-based)
- arrival: `+R_arrive` when within 1.5 u (waypoint radius), episode segment ends
- fall: `-R_fall` on OOB (respawn ends the segment), plus the existing free-fall
  per-tick cost (already map-independent)
- time: small per-decision cost so speed matters
- no gem reward at all in this phase (gems can be hidden or ignored)

**Episodes.** The game runs its normal round; Python picks goals: a random
walkable cell 8-40 u from the marble (walkability from the height stack), and
uses the existing `TELEPORT` reply to randomize the start after each segment
(or every k segments). Segment ends on arrival, fall, or a 20 s timeout.

**Curriculum / map set.**
- Stage 0: the three custom flat maps (holes, one platform) - sanity.
- Stage 1: King of the Marble, Hunting Around, Marble Agility Course, Triple
  Decker, Basic Agility Course (small, static, little vertical).
- Stage 2: Lupus, Par Pit, Pyramid, All Angles, Core (ramps, vertical, one
  moving platform to ignore at first).
- Stage 3: Sprawl, Marble City, Ziggurat, Cube Isle (large, flat-ish).
- Hold out 4-5 maps at every stage (e.g. Gems Ahoy, Playground, Horizon,
  Zenith) and never train on them: the transfer metric is measured there.
- Generate every map's height stack with `generate_terrain_map.py` first
  (check the PNGs; maps with PathedInterior need the static parts only).

**Bootstrap from human demos.** Every gem pickup in the recordings is a
reached waypoint: relabel each tick with "waypoint = position of the next
pickup" and clone the navigator on that (`demo_data.py` needs the relabel;
copycat guard and throttle mask stay). This is a far better use of the demos
than whole-round imitation (see `CHAT_CONTEXT.md`, BC findings).

**Metrics.** On held-out maps: arrival rate, falls per 100 u travelled, mean
speed, time per 10 u. Milestone: falls per 100 u below 0.5 on a map it never
trained on, at >= 7 u/s.

### 3.3 The planner (global, ~1 Hz)

**Inputs (global, from the observer + map):**
- All currently spawned gems with value (observer must send the full set; the
  current 5-gem slots go away or become the planner's own top-k).
- Full walkable grid + pathfinding costs from the height stack.
- Opponents: position, velocity, score, held powerup (observer had these
  fields once; restore them).
- Spawn rule state: who is the leader, score gap (=> where the next group can
  spawn), time left, alarm state.
- Powerups on the map (positions, type, respawn timers if available) and
  what the agent holds.

**v0: heuristic planner (no learning).**
1. Walkable grid from the height stack: cell walkable if floor exists; edges
   between cells cost by distance, slope, and a learned/measured table of the
   navigator's speed on flat/ramp; gaps up to the navigator's measured jump
   distance are traversable at a cost.
2. Travel time to every visible gem via A*; pick the gem/group maximizing
   points per second of travel + collection (group = all gems within
   `radiusFromGem`). Re-plan every 0.5 s or when the gem set changes.
3. Waypoint = the path point 10-15 u ahead; hand to the navigator.
4. Between groups: move toward the region where the next group can spawn
   (the spawn block around the leader is known from the score gap).
This alone should beat the current agent's nearest-gem chasing and is the
first end-to-end result to measure on King of the Marble against 166.

**v1: learned planner.** PPO at ~1 Hz over the frozen navigator: action =
target gem (attention/pointer over the gem set) + powerup use + blast; reward
= game score + the existing OOB cost; horizon 5 min. Inputs encoded as a set
(gems, opponents) with a small transformer/pooling encoder; the map enters
only through path costs (features per candidate gem: travel time, points,
group size, distance from the leader). Nothing map-specific is memorized.

### 3.4 Powerups, opponents, offense

- Powerup **use action**: the 7th action field already exists in
  `mlAgent.cs::executeAction`. Observation of held/active powerups needs the
  `isMegaMarble()`/`getPowerUp()` calls restored in `observer.cs` WITHOUT
  the engine warning spam that crashed 8-hour runs (2026-09-13 fix); test a
  call path that does not print.
- **Traversal powerups** (super speed/jump, helicopter) belong to the
  navigator as extra inputs and the use action; train them in the waypoint
  task on maps that have them.
- **Blast / mega** belong to the planner: the decision is "displace that
  opponent from this group now". Requires opponents in the observation and
  self-play.
- **Self-play**: run 2-4 game instances joined to one lobby, each driven by
  its own bridge; the trainer owns all marbles. The existing
  `server/scripts/aiBot.cs::serverCmdSetAIMove` is impulse-based (not the
  real input path) and is NOT the way to drive opponents; use full instances.
- Opponent curriculum: frozen earlier navigator+planner snapshots as
  opponents; then league-style sampling.

---

## 4. Order of work and milestones

1. **Instancing** (3.1). Milestone: 4 instances, one trainer, per-instance
   guards, aggregate >= 150 decisions/s. Nothing else starts before this.
2. **Height stacks for the stage-1/2 maps** + `TerrainMap.crop()`.
3. **Navigator v0**: waypoint task on stage 0-1 maps, GRU trunk, dense reward.
   Milestone: held-out small map, falls/100 u < 1, arrival > 90 %.
4. **Demo bootstrap** of the navigator (relabelled waypoints). Compare with
   and without; keep if it shortens stage 1.
5. **Navigator v1**: stages 2-3, jump/ramp/gap curriculum. Milestone: falls per
   100 u < 0.5 on held-out maps at >= 7 u/s.
6. **Heuristic planner** over the frozen navigator. Milestone: King of the
   Marble score vs the 166 human benchmark; then Lupus, Sprawl.
7. **Observer extension**: all gems, opponents, powerups (with the no-print
   constraint). Powerup use in the navigator.
8. **Learned planner** (v1) with self-play instances; blast/mega.
9. Evaluation ladder: single-player score on 10 maps -> 1v1 vs frozen
   snapshots -> 1v1 vs the human recordings' scores -> live human matches.

Each milestone is a number on a held-out map; do not proceed on "looks better".

---

## 5. Engineering notes and reuse map

| existing | reuse as |
|---|---|
| `mlAgent.cs` update loop, RECORD/SPEED/TELEPORT replies, `-autotrain` | per-instance bridge; TELEPORT = start randomization; add `-aiport` |
| `observer.cs` (world frame, gems, game state) | extend: all gems, opponents, powerups; keep the no-print rule |
| `terrain_obs.py` `TerrainMap` (height stack, sample, edge rays) | add `crop()`, walkable grid, A* costs; keep rays |
| `generate_terrain_map.py` | run per map; verify PNGs; handle multi-interior maps |
| `train_ppo.py` PPOTrainer, Actor heads, PopArt critic, guards, warmup rule | multi-instance server; new trunk (CNN+GRU); waypoint reward module |
| `record_demos.py`, `demo_data.py`, `bc_pretrain.py` | waypoint relabelling; navigator bootstrap |
| `probe_frame.py`, `replay_demo.py`, per-decision trace | per-map sanity checks when a new map is added |
| dashboard | per-instance tiles; navigator metrics (falls/100 u, arrival %) |

Known constraints to carry forward:
- Every observation must be in the world frame (`$AIObserver::ForceYaw = 0`)
  and every command rotated into the camera frame (`executeAction`). Verify on
  each new map with `offmap:`/`cmd_accel_cos:` on the GAME END line.
- Any observation-width change => critic warmup (automatic on widening).
- Game window speed guard stays (rtf); prefs `backgroundSleepTime = 0`.
- Reward must stay map-independent (no coordinates, no per-map shaping).
- `analyze_log.py` must learn every new log field.
- No CLI flags for normal operation; defaults in code.

---

## 6. Alternatives considered

- **Python surrogate physics** (marble on the height map, thousands of envs,
  1000x throughput). Only if instancing cannot reach ~150 decisions/s. Risk:
  sim-to-game gap in friction, bounce, ramps, jumps; would still need real-game
  fine-tuning. Plan B, not plan A.
- **Whole-map image as network input.** Rejected: constant on a single map
  (a bias term); across maps it asks the network to learn pathfinding from
  pixels, which A* does exactly. The map enters via the planner's search.
- **Imitation from the game's replay files** (`client/scripts/replay.cs`
  writes `.rrec` with positions/movement per frame). No corpus exists locally;
  if community hunt replays can be obtained, they are a large navigator
  bootstrap set (positions only; inputs would have to be inferred).
- **Keep tuning the per-map MLP.** Rejected as the main line: ends at one map.
  Keep the current run only as the King-of-the-Marble baseline.

---

## 7. Open questions to settle early

1. CPU/GPU cost of one game instance at 3x; max useful N on this machine.
2. Can several instances share one install (prefs/console.log locks)?
3. Do dedicated-server rounds run the hunt spawn logic identically (gem
   spawn code checks `$Server::Hosting && !$Server::_Dedicated`)? If not,
   instances must be listen servers, or that check needs a training flag.
4. Observer cost for "all gems" (Wrap Zone: 570 spawn points; only spawned
   gems are sent, typically <= 7).
5. Powerup state calls in `observer.cs` without engine prints.
6. Moving platforms: static-only height stacks first; platform observation
   later (position + velocity of the nearest PathedInterior).
