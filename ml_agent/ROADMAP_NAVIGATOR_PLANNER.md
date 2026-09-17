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

**Measured 2026-09-16 (throughput_probe.py, King of the Marble, 24-core PC):**

| requested speed | measured | ticks/s | duplicated physics states | trajectory drift vs 1x after a 3 s maneuver |
|---|---|---|---|---|
| 3x | 3.0x | 187 | 0 | 0.26 u |
| 6x | 6.0x | 375 | 0 | 0.19 u |
| 10x | 9.9x | 620 | 0 | 0.60 u |
| 15x | 14.9x | 934 | 1 of 355 | 0.75 u |
| 20x | 19.8x | 1238 | 22 % | 0.72 u |
| 25x+ | capped ~20x | | 40 %+ | broken |

- The engine advances at most about one physics tick per rendered frame, and it
  renders ~1300 fps at 1280x720 on one saturated core, so ~20x is the hard
  ceiling per instance and the observation schedule starts firing twice per
  physics tick above ~15x. **Clean per-instance setting: 10x** (0 duplicates,
  sub-unit drift that comes from reply-latency jitter, not physics). Rendering
  is CPU-bound at that fps; window size and `setMaxFPS` did not lower CPU.
- **Lockstep does not work** (freezing the sim via time scale 0 / 0.001 after
  each observation): scale 0 makes the engine catch up wall time on resume
  (2x speed), 0.001 gives no gain and the engine echoes every scale change to
  the console (2 lines per tick). `$MLAgent::Lockstep` stays false.
- **Background throttle**: an instance drops to ~1.5x (1 tick per ~10 ms frame)
  when its window is deactivated AND overlapped by the active window, or
  minimized, or off-screen. Deactivated but unoverlapped = full speed; fully
  covered by a non-active topmost window = full speed. `backgroundSleepTime`
  has no effect. A second monitor (or a PC nobody uses) with the instances
  tiled and never overlapped is the only script-side arrangement that keeps N
  windows fast. Off-screen windows still run at ~3x on 10 % CPU, but in bursts
  of ~6 ticks per frame with one action per frame, which changes the control
  timing; not acceptable for training.
- **Single-instance guard**: a second marbleblast.exe on the same machine
  exits with "Another instance of this application is already running"
  (engine-level, a named mutex; not a script or file lock). So N instances on
  one PC is impossible in this build without an engine change or one Windows
  session per instance.
- **No headless path in this build**: `-dedicated` exists but the engine has no
  AIConnection / script-driven moves, so a marble needs a rendering client.
  The engine is the "OpenPQ" rebuild (WebGPU); its source is not in this
  repository. Engine changes (headless renderer, multi-tick-per-frame, no
  throttle, script-driven moves) would remove every limit above and are the
  first thing to pursue if the source is obtainable (README points at
  MBExtender-Dev for engine modifications).
- Game-side additions made for this: `-aiport N`, `-aispeed N`, `-autotrain`
  now drives the single-player Play menu (no login), the agent auto-starts in
  single-player hunt rounds, keeps retrying the Python connection every 2 s
  while a round runs, and rounds auto-restart even without a trainer.
  Probe-only control replies: `SLEEPTIME n`, `MAXFPS n` (`setMaxFPS(0)` puts
  the engine in a different timing mode where trajectories were not
  reproducible across speeds: do not use for training).
- **Aggregate realistic today**: ONE instance at 10x = 620 ticks/s (155
  decisions/s at action repeat 4), 3.3x what training used until now, and only
  while its window is not overlapped by whatever the user is working in. More
  than that needs engine changes (single-instance guard, headless renderer,
  ticks decoupled from frames, no activation throttle, script-driven moves)
  or more machines.

**Engine source (2026-09-16 evening).** The engine is
https://github.com/The-New-Platinum-Team/OpenPQ-TGEMIT (TGE 1.4.2 MIT port,
WebGPU/dawn renderer; cloned to `C:/Users/doug/src/OpenPQ-TGEMIT`, outside
OneDrive). It builds with CMake + Ninja + MSVC exactly as `.circleci/config.yml`
does (dawn DLL is in `lib/webgpu-dawn`); the built exe replaces
`marbleblast.exe`. Every limit above maps to a few lines:

| limit | where in the engine | change |
|---|---|---|
| single instance | `engine/gui/core/guiCanvas.cc:202` `Platform::excludeOtherInstances("TorqueTest")` | skip when `-aiport`/`-headless` is given |
| background sleep | `engine/platformWin32/winWindow.cc` ~1240 (`backgrounded` = foreground window is another process) and `TimeManager::process()` ~1899 | ignore `backgrounded` in AI mode |
| occluded-window stall (1.5x) | WebGPU present in `engine/dgl/pipeline/webgpu` / `platformWin32/webgpu` | headless mode: no swap chain, skip `renderFrame`/present |
| wall-clock-bound sim (~62 TimeEvents/s, `if (event.elapsedTime > 15)`) | `TimeManager::process()` | `-fastsim`: post TimeEvents with a fixed virtual elapsed (16 ms) back to back, no wall clock; sim runs as fast as the CPU allows, deterministic |
| stale actions during PPO updates | `TimeManager::process()` + socket | `-lockstep`: block until the AI reply arrives before posting the next TimeEvent (true lockstep, exact tick<->action mapping) |
| physics tick 32 ms (`TickMs`, `engine/game/gameBase.h:359`), marble sub-stepping in `Marble::advancePhysics` (`engine/game/marble/marble.cc:1933`) | | leave; expose a per-tick observation hook (C++ callback into script after each marble physics step) so observations are tick-exact without a 16 ms script timer |
| no script-driven marbles | `engine/game/moveManager.h` (Move), `gameConnection` | later: server-side AI connection producing Moves from script, for headless self-play without one client per marble |
| search-based control (RandomityGuy's suggestion) | `Marble::advancePhysics` | later: expose save/restore marble state + one physics step to script |

**Where `marbleblast.exe` actually comes from (checked 2026-09-16 night).**
The shipped exe is a private build: it is the OpenPQ-TGEMIT engine (same
`webgpu_dawn.dll`/`dxcompiler.dll` renderer) with the MBExtender plugin code
compiled in natively (the exe carries `RTAAS`, `Discord`, `MBPlatinum`, `add64`
strings and links `cryptopp-shared.dll` and `libcurl.dll`; the public build has
none of these). Neither public repo produces it: `PlatinumQuest-Dev` contains
zero C/C++ files (scripts + data only), and `MBExtender`
(https://github.com/RandomityGuy/MBExtender, last commit 2022-09-05, cloned to
`C:/Users/doug/src/MBExtender`) is the *2007-engine* plugin set, injected as
DLLs that hook the old exe by memory address (`MBX_ADDRESS` macros) - it cannot
be loaded by the modern exe. The org's `PQBinaries` repo ships only compiled
binaries (Windows/Mac/Linux folders).

What MBExtender does give us is the source of the missing layer. Of the 232
console functions the built engine lacks, 182 have an implementation in
MBExtender:

| plugin | missing fns | needed for headless training? |
|---|---|---|
| MBPlatinum | 57 | yes: powerup tuning (`set/getSuperJumpVelocity`, shock absorber, helicopter), `clientContainerRayCast`, sync objects, path nodes, radar, string utils, `traceGuard`, `setPrintTime`. Also engine hooks (host trigger override, particle fix, save-fields fix) that are *not* console functions |
| MathExtension | 50 | yes, mechanical (vector/box/quaternion helpers, `min`/`max`/`mClamp`) |
| GraphicsExtension | 32 | no: stubs (shaders, postFX, blur, reflective marble) |
| FileExtension | 13 | yes, mechanical (`mkdir`, `deleteFile`, `copyFile`, base64, SHA256, zip) |
| Disco / JoystickSupport / MBCrypt | 7 / 5 / 6 | stubs (Discord, joystick; MBCrypt = `.mbpak` loading, only `marbleland.cs` references it) |
| FrameRateUnlock | 5 | yes: `setTimeScale`/`getTimeScale`/`onNextFrame` + the `Timeskip` hook on `TimeManager::process` |
| JSONSupport / others | 4 / 3 | yes, mechanical (`Array`, `jsonParse`, `regexMatch`, `textLen`, `getTransform`, `setVisibleDistance`) |

The remaining 50 are not in MBExtender at all (written after 2022 for the
modern engine): 14 `RTAAS_*` speedrun-timer setters (stub), 20 `*64`
64-bit-int math functions (trivial), date/time functions (trivial),
`anticheatDetect`/`cheat_joj` (stub), `setMaxFPS`, `cancelScheduleIgnorePause`,
`setSimulatingPathedInteriors` (gameplay: moving platforms), `replaceDiffuseTexture`,
`onMarbleDataPreSend/PostSend`, `Clone`, `HcJ`.

The real risk is not the console functions but the engine-behaviour hooks the
private layer also carries: MBExtender's `MarbleGhostingFix` hooks
`cMarbleSetPosition`/`cSetGravityDir`/`cSetTransform`, `MovingPlatformsFix`,
`Timeskip` hooks `TimeManager::process`, `MBPlatinum` hooks host triggers and
particle emitters - none of these names appear anywhere in the public engine
source. A ported engine could therefore run the scripts and still differ from
the real game in physics-adjacent behaviour (moving platforms, gravity changes,
trigger ordering), which would break the "train here, play there" contract.

Decision: (1) ask RandomityGuy for the private engine tree (or a build with
`-headless`/`-fastsim`/`-lockstep` flags); the shipped exe proves it is
OpenPQ-TGEMIT plus a PQ layer, so this is a zero-port path. (2) Meanwhile
develop and verify the engine mods on the public build using its bundled
OpenMBG game, where every change is testable with `throughput_probe.py`; the
mods live in engine files the private layer shares, so they carry over as a
patch. (3) Porting the 232 functions ourselves (MBExtender sources as the
reference, ~2-4 days for a running game, stubs for graphics/Discord/RTAAS) is
the fallback only if (1) fails, and it must be validated against the real exe
tick-for-tick (teleport + scripted maneuver comparison, as in the probe) before
any training on it counts.

**Physics identity 3x vs 10x (Sprawl, 2026-09-16, `physics_identity_probe.py`).**
Six alternating trials, same teleport pose, same 150-tick key script
(forward/back/left/right), rest position recorded:

| speed | rest positions (x, y) | spread within speed |
|---|---|---|
| 3x | (-26.045, -51.752), (-26.089, -51.744), (-26.044, -51.710) | 0.056 u |
| 10x | (-26.088, -51.749), (-26.088, -51.749), (-26.088, -51.749) | 0.0005 u |

Distance between the 3x and 10x means: 0.032 u, inside the 3x spread; one 3x
trial landed on the 10x point to 1 mm. Peak trajectory deviation between any
two trials 0.10 u. Per-tick physics are the same at both speeds.

A 21 s script (1342 ticks, direction pairs, diagonals, jumps; same probe)
diverges completely: 3x trials ended ~24 u north of the start (one fell off),
10x trials ~12 u west; 3x-vs-3x differed by 6 u and 10x-vs-10x by 4 u. The
maneuver is chaotic (jumps, edges, wall bounces), so it amplifies the bridge's
timing jitter and cannot separate physics from control timing. The cause is
measured by `action_latency_probe.py` (ticks from sending FORWARD to the
marble moving, 12 reps per speed, random frame phase):

| speed | latency (ticks) |
|---|---|
| 1x | 2, always |
| 3x | 2-3 (mean 2.5) |
| 10x | 3-6 (mean 4.3) |

The game reads socket replies only when the engine processes network events,
so at higher time scales several script updates run on a stale action. Physics
is identical; the *control loop* is not: at 10x the policy's command lands
1-4 ticks later than at 1x and the delay varies tick to tick. Options:
(a) engine lockstep (needs the engine source; exact tick<->action mapping at any
speed); (b) script-only fixed delay: stamp every observation with its tick,
have the game apply the reply for tick t exactly at tick t+K (K=6 covers the
10x maximum), so latency is a constant 96 ms at every speed and the policy
learns one deterministic delay; (c) accept it and train at the speed you play
at (3x).

**Fixed delay implemented and tested (2026-09-16 night).** Every observation
now ends in `|<tick>`; a reply ending in `,t<tick>` is applied at tick +
`$MLAgent::ActionDelay` (queue in `socketBridge.cs`/`mlAgent.cs`; control
words SPEED/TELEPORT/STATS/DELAY have their own slot). With K=6 the
message-level latency became exactly 7 ticks at 1x, 3x and 10x (0 late
replies), and a 2.4 s script landed 3x and 10x within 0.0008 u. But the 21 s
script still diverged at 10x, so two more probes were run
(`action_granularity_probe.py`):

| test | 3x | 10x |
|---|---|---|
| 6 single-tick FORWARD pulses, displacement | 0.42, 0.43 u | 0.54 u, then **0.00 u** (all pulses lost) |
| FORWARD/BACK alternating every tick, 200 ticks | drifts 0.07-0.16 u | drifts **0.26-0.99 u**, peak speed 1.27 |

Root cause, confirmed in the engine source (`engine/game/main.cc`,
`DemoGame::processElapsedTime`): per wall-clock frame the engine runs
`serverProcess(elapsed)`, then `Sim::advanceTime(elapsed)` (ALL script timers
due in the frame, back to back), then `clientProcess(elapsed)`. At 10x a
16 ms wall frame is 160 ms of sim: the 10 `MLAgent::update` calls run in one
batch between physics batches, so 9 of the 10 observations are ghost
interpolations of the same physics state and only the last of the 10 actions
is in force for the 5 physics ticks that follow. Which update is "last"
depends on wall-clock frame boundaries, hence the nondeterminism. At 3x it is
3 updates per 48 ms chunk (1.5 physics ticks); at 1x, 1 update per 16 ms
chunk (0.5 tick), i.e. per-tick control only exists at <= 2x. **No script-side
scheme can give per-tick control above ~2x**; the fixed delay is therefore
OFF by default (`$MLAgent::ActionDelay = 0`, `DELAY n` control for
experiments). The engine change (fixed-step TimeEvents + lockstep: advance
one physics tick, run the script, read the socket, repeat) is the only real
fix and moves to the top of the engine list. Interim for the existing 3x
trainer: its 64 ms decisions span ~1.3 frames, so most decisions do reach the
physics, with 1-frame jitter; do not train at 10x with the script bridge.

Toolchain on this PC (checked 2026-09-16): no Visual Studio, no CMake, no
Ninja; winget is available. Install: `winget install Microsoft.VisualStudio.2022.BuildTools`
(with the C++ workload), `winget install Kitware.CMake`, `winget install Ninja-build.Ninja`,
then the CI recipe: `cmake -S buildFiles -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
-DCMAKE_POLICY_VERSION_MINIMUM=3.5` inside a VS dev shell, `cmake --build build`.
First milestone: an unmodified build that runs King of the Marble identically
(throughput_probe.py at 1x/3x/10x must match the tables above). Then the
changes in the order of the table, each verified with the probe.

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
