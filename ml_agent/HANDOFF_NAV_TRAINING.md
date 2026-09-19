# Navigator training: handoff (2026-09-18, midday)

Read this first if you are picking up the navigator work. It says what exists, how to run it,
what the numbers are, what is known to be wrong, and what comes next. Design background is in
`ROADMAP_NAVIGATOR_PLANNER.md` (why a navigator + planner) and `IMPLEMENTATION_PLAN_NAVIGATOR.md`
(work packages); the running notes of everything tried are in `CHAT_CONTEXT.md`.

## 1. What is being trained

A map-independent *navigator*: a recurrent PPO policy that drives the marble from wherever it
is to a waypoint 8-40 u away, on any hunt map, using only a local terrain crop and a short
feature vector. Later a planner picks the waypoints (gems); the navigator never sees gems.

- Task: segments start with a verified teleport to a random walkable spot, a random reachable
  goal (Dijkstra over the walk grid; half the goals across a gap), and end on arrival (within
  the gem hitbox radius 0.65 u), a fall, a 20 s timeout, or the round end. `nav/waypoints.py`.
- Observation `NAV_OBS_V1`: 6x32x32 crops (0.5 u and 2 u cells: relative height, presence,
  second level) + 48-vector (waypoint direction/distance, own velocity, on-floor, airborne
  count, 38 edge-ray features). `nav/obs.py`, `nav/terrain.py`, `terrain_obs.py`.
- Model: CNN + GRU(256) actor-critic with PopArt, continuous direction/throttle + jump + brake
  (brake disabled), a hard-coded "gap prior" that pushes the jump logit near a jumpable edge.
  `nav/model.py`.
- Reward per 64 ms decision: progress along the path field 1.0/u, arrive +10, fall -10, time
  -0.05, airborne -0.1 after 1 s, brake -0.1. Arrival radius ratchets 1.5 -> 0.65 u as the
  policy gets reliable (it is at 0.65 now).

## 2. The stack that runs it (all new since 2026-09-18 morning)

The game engine is built from source with a training mode; eight game instances run at once.

| piece | where | what |
|---|---|---|
| engine | `C:\Users\doug\src\OpenPQ-TGEMIT-mbx` (git worktree, branch `ai-training-mode` off `mbx`; the team repo is remote `upstream`, there is NO `origin` on purpose) | 3 files changed, uncommitted: `engine/mbx/FrameRateUnlock/FrameRateUnlock.cpp` (fixed step + lockstep wait), `engine/game/main.cc` (console vars, render skip), `engine/gui/core/guiCanvas.cc` (mutex skip). `build-engine.ps1` builds in ~2 min incremental / 12 min clean -> `game\TGEMit_Demo.exe`. |
| game exe | `Marble Blast Platinum\marbleblast_mbx.exe` (gitignored copy of the build) | physics identical to the shipped exe (0.000 u over 1342 ticks on 3 of 4 probe trials) |
| engine flags | console vars, all default off | `$AI::FixedStepMs` (64 = one decision per frame, 2 physics ticks), `$AI::Lockstep` + `$AI::WaitReply` (sim stands still until our reply; 5 s safety timeout), `$AI::RenderEvery` (100), `$AI::MultiInstance` (set by `platinum/core/client/canvas.cs` when `-aiport` is on the command line) |
| game scripts | `platinum/client/scripts/ai/mlAgent.cs`, `socketBridge.cs`, `platinum/server/scripts/gems.cs`, `platinum/core/client/canvas.cs` | control words FIXEDSTEP / LOCKSTEP / RENDEREVERY / RESPAWN / MARK / DEBUG / INFO; control words are a queue; WaitReply handshake; the black waypoint gem (`AIMarkerGem`) |
| game loop | `run_game_loop.ps1` | launches N instances with `-autotrain <map> -aiport 8888+i`, relaunches any that exit; `-Mission a,b -Split 6,2` assigns maps per instance |
| trainer | `nav/train_nav.py` | one process: CPU copy of the policy for acting (batched over instances), GPU copy for PPO, pooled rollouts, logging, checkpoints |
| workers | `nav/vec_worker.py` | one torch-free subprocess per instance: socket, map, observation, segment logic, frame check, round handling; talks to the trainer over `multiprocessing.connection` |
| env | `nav/env.py` | `HuntEnv`: protocol, control words, `step_async`/`step_wait`, `drain`; `TRAINING_MODE=True` sends the engine flags on every (re)connect |
| PPO | `nav/ppo_recurrent.py` | sequences of 32, 3 epochs x 2 minibatches over the pooled rollouts (8 x 1024) |
| dashboard | `python -m nav.dashboard_nav` -> http://localhost:8990 | tails `logs/nav/train_nav_*.log` |
| probes | `fixedstep_probe.py`, `physics_identity_probe.py`, `compare_physics_runs.py`, `frame_probe.py` | speed / determinism / physics-identity measurements |

### How to run
```
# from ml_agent/  (order: trainer first is fine, the games reconnect on their own)
python -m nav.train_nav                       # resumes models/nav/nav_latest.pth
.\run_game_loop.ps1 -Mission FlatIslands_Hunt   # 8 instances of marbleblast_mbx.exe
python -m nav.dashboard_nav                   # http://localhost:8990
```
Both the trainer and the loop are normally started hidden with stdout to `logs/nav/stdout.txt`
and `logs/nav/game_loop.txt`. `N_INSTANCES` in `train_nav.py` and `-Instances` must match.
After editing any `.cs` delete its `.dso` next to it. After rebuilding the engine: stop the
instances, copy `TGEMit_Demo.exe` over `marbleblast_mbx.exe` (retry until the file is free),
let the loop relaunch them; the trainer's workers reconnect.

## 3. Where training stands (2026-09-18 21:35, training PAUSED for a RAM install)

- Checkpoint `models/nav/nav_latest.pth` = update ~3305, 18.1 M decisions. Numbered checkpoint
  `nav_003300.pth`. Revert points: `nav_pre_jumpcost.pth`, `nav_pre_jumpdamp.pth` (before the two
  changes of 2026-09-18), `nav_stage0_flat_upd75.pth` (flat map).
- Map mix across instances, not rotation in time: `-Split "4,4"`. Mixing maps *within* each update
  is what stopped the forgetting; the overnight hourly rotation knocked the islands to 76-82 %
  every time KOTM ran. Read the per-map `MAPS ...` line, not the pooled `arrive=` (which just
  tracks the instance split).

| map | instances | arrivals | falls/100 u | speed | airborne |
|---|---|---|---|---|---|
| FlatIslands_Hunt | 4 | 93-94 % | 0.33-0.43 | 5.7 u/s | 17 % |
| KingOfTheMarble_Hunt | 4 | 83-86 % | 0.70-0.96 | 5.2 u/s | 32 % |

  KOTM over 2026-09-18: 53 % -> 86 %, falls 4.3 -> 0.70, speed 2.9 -> 5.2 u/s. It sat flat at
  50-59 % for five hours overnight before the two changes in 3b. The arrival radius is at its
  final 0.65 u (gem hitbox) throughout.
- Throughput ~450-580 decisions/s, 10-12x the single shipped-exe instance at 3x.
- **Open item: entropy drift.** Policy entropy rose monotonically 0.33 -> 0.80 over the day
  (direction spread 0.23 -> 0.26) while reward stayed flat-to-up. This is the usual PPO pattern
  of a near-converged policy where the fixed entropy bonus starts to dominate a shrinking
  advantage signal. It has not cost performance yet, but it will if left for a long unattended
  run. Fix: `ENTROPY_COEF` 0.005 -> ~0.002 in `nav/ppo_recurrent.py`. Proposed, NOT applied.

## 3b. King of the Marble diagnosis (2026-09-18, `logs/nav/trace_20260918_132716.csv`)

**KOTM fails because stray jumps throw the marble off high platforms, not because it cannot
navigate.** Measured over 1152 KOTM and 3945 island segments in one mixed run (6 islands +
2 KOTM instances).

Use **takeoffs** (a jump commanded while ON the floor), never the raw jump rate: a jump pressed
in mid-air does nothing in the engine, and the raw rate is dominated by those inert presses
(KOTM 32 % raw vs 2.9 % takeoffs per on-floor decision). An earlier version of this section
reported the raw rate and overstated the case; the corrected numbers are below.

| | KOTM | islands |
|---|---|---|
| takeoffs per on-floor decision | 2.9 % | 1.7 % |
| fall risk per takeoff | **40.7 %** | 12.0 % |
| fall risk per involuntary roll-off | 12.9 % | 5.6 % |
| takeoffs with the gap prior active (a real gap) | 17 % | 46 % |
| median airborne stretch after a takeoff | 13 decisions | 10 |
| timeouts | 0 | 0 |

Jumping is not needed on KOTM and actively hurts:

| KOTM segments | arrivals | net height change |
|---|---|---|
| no takeoff | **87 %** (n=511) | 0.0 u |
| 1-2 takeoffs | 29 % (n=243) | -6.2 u |
| 3-5 takeoffs | 29 % (n=382) | -5.9 u |

70 % of arriving KOTM segments contain no takeoff at all and arrive at the height they started,
so goals never require climbing; a jump simply drops the marble ~6 u off the platform. On the
islands jumping IS needed (half the goals are across gaps): 46 % of island takeoffs are gap-prior
driven and segments with 3-5 takeoffs still arrive 81 %.

**Why the plateau sits where it does.** 0.029 takeoffs/decision x ~24 on-floor decisions per
segment x 40.7 % fall risk = 0.28 falls per segment from stray takeoffs alone, which caps
arrivals near 55-60 %. Observed: 53-56 %.

**Why a reward penalty is the wrong lever.** A takeoff already carries an implicit 0.407 x FALL =
~4.07 reward cost, and the policy has already pushed its takeoff rate BELOW its own -3 bias prior
(4.7 % base -> 2.9 % measured). The residual is exploration noise from the Bernoulli jump head,
held up by the entropy bonus. `JUMP_TAKEOFF = 0.4` was added on 2026-09-18 (user's choice) and
moved raw jumping 32.2 -> 28.5 % and arrivals 53 -> 55 % over ~25 updates: inside the noise, as
the arithmetic predicts. It is harmless and was kept.

**The lever that worked:** `JUMP_DAMP = 2.0` in `nav/model.py`, a constant subtracted from the
jump logit (the mirror of `BRAKE_SUPPRESS`). Applied 14:19 on 2026-09-18. Measured over the next
25 min (3133 KOTM / 11728 island segments):

| | before | after |
|---|---|---|
| KOTM arrivals | 57 % | **64 %** |
| KOTM takeoffs per on-floor decision | 5.9 % | 4.3 % |
| KOTM takeoffs at a real gap (prior active) | 17 % | 28 % |
| KOTM airborne | 47 % | 40 % |
| KOTM falls per 100 u | 3.6 | 2.7-3.0 |
| islands arrivals | 91 % | 91-92 % |
| islands takeoffs at a real gap | 46 % | 54 % |

It did not merely suppress jumping: on both maps the share of takeoffs that happen at a real gap
rose, i.e. the jumps that remain are the useful ones. No cost to island gap crossing. KOTM went
from the 50-59 % overnight plateau to 64 %. Checkpoints `nav_pre_jumpcost.pth` and
`nav_pre_jumpdamp.pth` are the revert points for the two changes.

## 4. Known limits and gotchas

- This PC: 16 GB RAM (8 game instances ~400 MB each, trainer 1.6 GB, <2 GB free), GPU ~40 %
  busy with the 8 windows. 16 instances do not fit; +16 GB RAM would give ~1.5-1.8x.
- The game's per-decision cost (~2.4 ms: TorqueScript observer + JSON + socket) and the
  Python observation build (~1.1 ms) are the throughput floor now; a C++ observer is the next
  big lever. Rendering is not a cost.
- Do not run acting on the GPU while the games are on it (25 ms per batch-8 forward vs 2.7 on CPU).
- A burst of control words needs the queue in `socketBridge.cs`; the lockstep wait must sleep
  (it spins 4 ms, then 1 ms sleeps); an instance that has just started can miss INFO (retried).
- Port 8890 is instance 2's socket; the dashboard is on 8990.
- **Disk**: `TRACE = True` writes one CSV row per decision per instance: ~200 MB/hour at 8
  instances, ~1 GB per 5-hour run. The repo lives under OneDrive, so those files are also synced
  to the cloud. On 2026-09-18 19:37 this filled the drive and the trainer died with
  `OSError: [Errno 28] No space left on device` (it auto-restarted from the checkpoint and the
  workers reconnected, losing ~1 update). Old traces compress ~5x (`gzip logs/nav/trace_*.csv`).
  Before any long unattended run: check free space, and either set `TRACE = False` or point the
  trace at a directory outside OneDrive.
- Fixed step 128 ms (4 ticks per action) changes jump outcomes; stay at 64.
- The shipped exe still works with the same trainer (set `TRAINING_MODE=False` in `nav/env.py`,
  `N_INSTANCES=1`, `-Exe marbleblast.exe -Instances 1`).
- Uncommitted: everything above in both repos. Do not push the engine anywhere public (the
  owner's request); set up a private `origin` first.

## 5. Plan (agreed 2026-09-18)

1. Islands only until arrivals are back at ~90 % at 0.65 u (should be an hour or two now).
2. Map mix across instances instead of rotation in time: start
   `.\run_game_loop.ps1 -Mission "FlatIslands_Hunt,KingOfTheMarble_Hunt" -Split "6,2"`, move
   toward 4,4 as KOTM improves. The trainer logs a `MAPS ...` line with per-map numbers.
3. Add maps: `FlatWithJump_Hunt`, `JumpOnly_Hunt` have terrain maps already; generate more with
   `generate_terrain_map.py <Mission>`; keep one map out of training as the transfer test.
4. Diagnose the KOTM plateau from the trace CSV (`logs/nav/trace_*.csv`, per decision, `inst`
   column): falls at ledges vs stacked floors in the crop vs timeouts.
5. Then the time-optimisation reward (reach the waypoint faster; explicitly deferred by the
   user until the policy is reliable), then the planner (WP8) and gems.
6. Engineering, when throughput matters again: C++ observer in the engine (WP7 remainder),
   per-instance console.log/prefs paths, headless mode, more RAM.
