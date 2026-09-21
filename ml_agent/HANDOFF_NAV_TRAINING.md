# Navigator training: handoff

> ## CURRENT STATE (2026-09-21 10:05) -- READ THIS BEFORE ANYTHING ELSE
>
> Everything below section 2 is a dated running log. Where it disagrees with this block, this
> block wins. Sections are appended newest-last; 14-18 are today's.
>
> **Verified score: 93.5 points on KOTM**, 8 deterministic rounds (92,101,98,87,90,84,95,101) on
> `models/nav/nav_eval_edge1v2_0955.pth`, update 13,037. Human 143.5. Previous: 90.8.
>
> | metric | agent | human | note |
> |---|---|---|---|
> | points / round | 93.5 | 143.5 | 3.0 min rounds |
> | points / min | 30.9 | 47.3 | |
> | gems / min | 24.5 | 37.5 | **the live gap** |
> | speed | 6.31 u/s | 8.05 | |
> | speed at pickup | 4.90 u/s | 6.3+ | **the live mechanism** |
> | distance per gem | 11.3 u | 12.9 u | already better than human |
> | falls / 100 u | 0.37 | 0.046 | no longer binding |
>
> **FALLS ARE NOT THE BOTTLENECK ANY MORE.** They were 0.585 on 2026-09-20 and section 3c is
> written against that. The fall-credit bug (section 17) was the cause; fixing it took KOTM
> training falls 0.73 -> 0.40, where the curve flattened, and the 8-round eval flattened with it
> (90.8 -> 93.5, t = 0.95, not significant). Chasing falls further is not where the points are.
>
> **THE LIVE PROBLEM IS PACE.** The marble travels less distance per gem than a human and still
> takes longer, because it decelerates to 4.9 u/s at every pickup. See section 16's time budget
> and section 18's mechanism.
>
> **Current config** (do not re-derive from older sections): `EDGE_K = 1.0`, `FALL = 25.0`,
> `TIME = 0.05`, `BRAKE = 0.05`, `BRAKE_SUPPRESS = 1.0`, `JUMP_DAMP = 3.0`, `BRAKE_ENABLED = True`,
> `THROTTLE_FLOOR = 0.90`, `CARRY = 0.0`, `TURN_COST = 0.0`, adaptive entropy in band 0.2-0.8
> (`ENTROPY_COEF` is a starting value only, the controller owns it), `DIRECT_GEM = 1`
> (real runs steer at the exact gem; no routing waypoints, no markers off a gem).
>
> **Standing rules set by the user**
> * Waypoints may ONLY ever appear inside a real gem. No routing corner points, no marker on
>   empty ground (2026-09-20 23:50).
> * READY AT GO: every run that rolls the marble must be able to move the instant the round says
>   GO. Python binds its socket before any model/CUDA/terrain work, and every launcher starts
>   Python and waits for the port before launching the game (`nav_ready.ps1`, `start_training.ps1`).
> * Judge only by 8-round `nav/real_run.py` evaluations. Training curves have misled repeatedly.
> * The engine rule was lifted for the render-only view yaw, which is built and deployed.
>
> **Start training with `.\start_training.ps1`** (trainer first, ports bound, then the games).
> **Evaluate with `.\eval_cycle.ps1 -Tag <name> -Rounds 8`** (stops training, evals, restarts).


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
- Throughput ~490-510 decisions/s (collection 615-642/s, PPO update 3.3-3.7 s per 8192 samples).
  The trainer loop is PIPELINED (`PIPELINE_GROUPS = 2` in `nav/train_nav.py`): the workers are
  split into interleaved groups so one group's policy forward overlaps the other group's game
  stepping. In lockstep a game holds until its reply arrives, so without this the games idled
  through every forward and the trainer idled through every game step. Collection went 450 -> 630/s.
- **Instance count is capped by VRAM, not system RAM (measured 2026-09-18, 8 GB GPU).** Each game
  instance holds ~450 MiB of VRAM and the PPO update needs ~1.9 GB. More instances speed up
  collection but starve the update, and the update loses more than collection gains:

  | instances | free VRAM | update | collection | overall |
  |---|---|---|---|---|
  | 8 (serial, before) | 3.0 GB | 3.0-4.5 s | ~450/s | 414-449/s |
  | **8 (pipelined, current)** | 3.0 GB | 3.3-3.7 s | 615-642/s | **484-512/s** |
  | 10 (pipelined) | 2.0 GB | 9.1-9.3 s | 690-728/s | 383-397/s |
  | 12 (pipelined) | 1.1 GB | 15.3 s | 754-781/s | 312/s |
  | 16 (pipelined) | 0.3 GB | >8 min, spilled to shared memory | 736/s | stalled |

  8 is the optimum on this machine. Adding system RAM (16 -> 32 GB on 2026-09-18) does NOT raise
  it; only a bigger GPU, or a game build loading fewer textures per instance, would. If you change
  `N_INSTANCES`, re-measure `upd_s`: anything above ~4 s means the update is being squeezed out of
  VRAM and the extra instances cost throughput rather than buying it.
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

## 3c. Human baseline (the yardstick) -- `demos/demo_20260914_214854.npz`

> **STALE AS A TARGET (2026-09-21).** The human columns below are still correct; the AGENT
> columns are from 2026-09-20 and the fall gap they describe has largely been closed (0.585 ->
> 0.37 per 100 u). Do not open a "close the fall gap" work plan from this section. See the
> CURRENT STATE block at the top: the live gap is pace, not falls.


The user played King of the Marble for 18.2 min / 6 rounds using only direction and jump
(recorded 2026-09-14 with `record_demos.py`; 68,208 ticks at 16 ms, 682 pickups, 4 OOBs).
This is the reference for "better than any human", replacing the arbitrary 90 % arrival gate.

| metric (same map) | human | agent 2026-09-18 23:00 | gap |
|---|---|---|---|
| ground covered per second of play | 8.06 u/s | 7.18 u/s | 0.9x |
| horizontal speed while grounded | 7.90 u/s (p90 12.2) | 7.49 u/s (p90 11.1) | close |
| **falls per 100 u** | **0.046** | **0.585** | **12.7x worse** |
| airborne share of the time | 17.0 % | 27.5 % | +10.5 pts |
| jumps started per second | 0.11 | 0.20 | 1.8x more |
| brake held | 14.3 % of ticks | DISABLED in the policy | - |

**Read this carefully: speed is nearly solved, falling is not.** The agent moves within ~10 % of
the user's pace; it falls thirteen times more often per unit covered. So a naive time-optimisation
reward is the WRONG next move -- told to go faster, a policy that already falls 13x more will buy
speed with falls. Close the fall gap first, then add time pressure with the fall penalty scaled so
that trade can never pay.

Two leads from the demo:
1. The user jumps 0.11 times per second; the agent 0.20, and ~75 % of the agent's takeoffs have no
   gap prior active (nothing to cross). `JUMP_DAMP` is the proven lever (see 3b).
2. The user brakes on 14.3 % of ticks. We have braking DISABLED (`BRAKE_ENABLED = False` in
   nav/model.py) because an early policy learned to sit still and brake for free. It now costs
   `BRAKE = 0.1` per decision and is suppressed at gap edges, so the degenerate strategy is no
   longer free. Re-enabling is worth a clean ablation: braking is plausibly how a skilled player
   controls entry speed into turns and stops precisely on gems, and part of why they rarely fall.

Caveat: the demo is continuous gem-chasing Hunt play; the agent runs short waypoint segments from
teleported starts. Per-unit-of-ground rates travel reasonably between the two; do not read them to
two decimals. A second demo on FlatIslands or a real catalog map would show whether the fall gap is
KOTM-specific or general (20 min of the user's time, high value).

## 3d. Held-out evaluation, 2026-09-19 01:58 (checkpoint `nav_both90_20260919_0158.pth`)

Gate: both training maps held >=90 % arrivals for 22 consecutive readings (islands 94 % / 0.35
falls per 100 u, KOTM 90 % / 0.46). Training stopped, `eval_heldout.ps1` run on the three maps the
policy has NEVER trained on (60 seeded greedy segments each):

| held-out map | arrivals | falls per 100 u | speed | verdict |
|---|---|---|---|---|
| FlatWithJump_Hunt | 100 % | 0.0 | 8.76 u/s | solved, NOT added |
| JumpOnly_Hunt | 100 % | 0.0 | 8.93 u/s | solved, NOT added |
| FlatGemTraining_Hunt | 100 % | 0.0 | 9.03 u/s | solved, NOT added |

Per the rule agreed with the user (>=90 % = solved, don't add; <85 % = a real gap, add it), none
joined the rotation and training resumed on the unchanged 4,4 split. The navigator transfers to
unseen maps -- but note all three are flat or simple, so this validates transfer to EASY terrain,
not to hard terrain. The honest next test is a real catalog map (user's call).

## 3e. Overnight 2026-09-18/19: two null results, and what they teach

Goal was to close the fall gap (section 3c). Progress DID happen -- KOTM 84-85 % / 0.8 falls per
100 u at 22:45 to 90 % / 0.38 at 03:49, islands steady at 94-95 % / 0.22-0.26 -- but it came from
ordinary continued training. **Both deliberate interventions were null.** Do not re-run them
expecting a different answer.

**Experiment 1, JUMP_DAMP 2.0 -> 3.0 (02:00-02:48): no measurable effect.**

| map | arrivals | falls/100 u | takeoffs/s | airborne |
|---|---|---|---|---|
| islands | 94 -> 95 % | 0.26 -> 0.22 | 0.519 -> 0.505 | 17.1 -> 16.4 % |
| KOTM | 89 -> 89 % | 0.39 -> 0.41 | 0.172 -> 0.167 | 24.9 -> 25.6 % |

Lesson: **a constant offset on a logit gets learned around.** PPO shifts the head's output to
compensate within ~190 updates. This also undermines the earlier attribution in 3b -- the climb
from 57 % to 86 % on 2026-09-18 was credited to JUMP_DAMP = 2.0 after a 25-minute window, but the
improvement continued for hours and was probably mostly ordinary learning. Treat 3b's causal claim
as unproven. (A METHOD WARNING: an intermediate verdict that night was computed by comparing two
POST-change traces, one holding ~1 minute of data, and briefly showed a large fake win. Always
compare matched windows either side of the change, and check the trace file timestamps.)

**Experiment 2, BRAKE_ENABLED False -> True (02:52), then BRAKE 0.1 -> 0.05 (03:05): no effect,
because the policy will not use the action.** Brake usage 0.6 % of decisions at cost 0.1, falling
to 0.4 % at cost 0.05, against the human's 14.3 %. Arrivals and falls unchanged within noise.

Why: while `BRAKE_ENABLED = False` the head's output was overwritten by a constant, so almost no
gradient reached it and it stayed at its initial bias (-3, ~4.7 %). On re-enabling, braking costs
reward immediately while its benefit (not falling) is delayed and rare, so PPO pushes usage DOWN
from 4.7 % rather than discovering the payoff. Lowering the price does not fix this; the obstacle
is exploration, not cost.

**What the data says to try next (NOT implemented, needs the user).** The failure mix has shifted:
falls after a deliberate jump fell from 76 % to 57-59 %, while ROLL-OFFS (driving off an edge with
no jump) rose from 23 % to 39-43 %, happening at 7.5-8.0 u/s, i.e. ordinary cruising speed.
Braking is the right control for that, so the goal is to make the policy explore it where it pays:
add a **brake prior**, the exact mirror of the existing `JUMP_PRIOR`/`BRAKE_SUPPRESS` machinery --
raise the brake logit when an edge ray ends close ahead along the direction of travel AND the goal
is NOT across that gap AND speed is high. Map-independent, same style as the accepted gap prior,
and it forces exploration precisely in the roll-off situation. Alternatives if that fails:
re-initialise the brake head on re-enabling (it is stale), or shape the reward for approaching an
edge fast rather than only for the fall itself.

**Also fixed overnight (real bug):** the trainer died at 02:48 with
`PermissionError: [WinError 5]` renaming `nav_latest.pth.tmp`. OneDrive holds the checkpoint open
while uploading it, and a checkpoint is written every 5 updates (~100 s), so the collision is not
rare. `save_ckpt` now retries the replace for ~6 s (`_save_atomic`). Better long-term fix: move
`models/` outside OneDrive.

Current config left running: JUMP_DAMP 3.0, BRAKE_ENABLED True, BRAKE 0.05, ENTROPY_COEF 0.002,
8 instances, 4/4 map split. The two experiment changes are neutral, not harmful, and braking being
enabled is a prerequisite for the brake-prior idea above. Revert points:
`nav_both90_20260919_0158.pth` (gate, pre-experiments) and `nav_before_brake_0305.pth`.

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

### 3f. Gem groups: the stale-goal bug (2026-09-19)

Deployed 09:25. Symptoms within minutes: `arrive=0%` on both maps, `rew=-13`, outcomes only
`fell` and `timeout`, never `arrived`.

The decisive measurement was comparing the two distances in the trace:

| closest approach per segment | median | within 0.65 u |
|---|---|---|
| path distance (`goal_d`, what the trace logged) | 1.00 u | 3 % |
| straight-line (what the arrival test actually uses) | 0.05 u | 91 % |

Those can only disagree if the goal moved and the trace kept logging the old one. It had.

**The bug.** `_advance_goal()` retargets `SegmentManager.seg.goal` onto the next gem, but
`vec_worker.InstanceWorker.goal` was assigned only in `start_segment()`. So after the first
pickup, reward / arrival / `prev_d` chased gem 2 while the *observation* still pointed at gem 1 —
which the marble was standing on. The policy saw a waypoint at its own feet and orbited it until
the timeout. `eval_nav.py` had the identical bug.

**Why the dry test missed it.** The offline chaining test teleported the marble exactly onto each
goal, so the arrival test passed regardless of what the observation said. A dry test that drives
the goal must also assert on the observation the policy receives, not just on the manager's state.

Fixed by refreshing the cached goal wherever the mark is taken. Verified offline (the cached goal
now tracks `seg.goal` across three chained pickups) and in training:

    before fix: arrive=0%   gems=0/grp   rew=-13
    after fix:  arrive=29%  gems=3.3/grp rew=+67

The 40 updates (328k steps) trained against the broken observation were discarded; the checkpoint
was re-converted from `nav_latest.v1backup_20260919_0926.pth` (update 5560). The broken one is
kept as `nav_gemgroups_broken_*.pth`.

**Open, needs watching:** falls100 jumped to ~0.9 (KOTM) / ~1.8 (islands) against ~0.09 before.
Expected in part — chaining means far longer continuous runs with no teleport reset, and the
greedy nearest-gem order can route a chain straight across KOTM's holes — but it has not been
shown to come back down yet. `pickup=8-10 u/s` is the control metric: the marble is still
arriving at full speed, which is exactly what the change exists to fix, so it should fall.

### 3g. The right way to measure braking (2026-09-19)

The `pickup=` field alone is misleading. It fell 9.4 -> 7.0 u/s after gem groups went in, but the
agent's overall speed fell too, and the obvious ratio test (`pickup` / `speed`) is NOT like-for-like:
the `speed` field is `travelled / (decisions * 0.064)`, which excludes airborne travel and applies
TRAVEL_CLIP per step, so it is a different quantity from an instantaneous horizontal speed.

**Use the approach profile instead.** Speed at the pickup decision versus speed 12 decisions
(0.77 s) earlier. Both are instantaneous horizontal speeds, so the comparison is sound, and
pickups are detectable in the trace because `gx, gy` change when the group retargets.

Human (`demos/demo_20260914_214854.npz`, 682 pickups), speed before pickup:

    0.80 s: 9.49   0.38 s: 8.57   0.19 s: 8.14   0.10 s: 7.94   at pickup: 7.75 u/s
    -> sheds 18 % over the final 0.80 s; mean speed overall 8.05 u/s

Agent across the first ~110 updates on gem groups (14,497 pickups), by fifth of the run:

    -13 %  ->  -10 %  ->  -6 %  ->  -5 %  ->  -4 %      (negative = still ACCELERATING into the gem)

So gem groups are moving the right quantity in the right direction -- the agent has gone from
hard acceleration into a gem to nearly flat -- but it has **not** begun to decelerate, and the gap
to the human is still ~22 points. Do not claim braking from the `pickup=` field; re-run the
profile.

**Watch for a cheap shortcut.** Absolute speed is drifting down across the run (7.27 -> 6.53 u/s
at the 0.77 s mark) against a human 8.05. Slowing down everywhere is an easy way to cut falls
without learning control. This is the argument for bringing in the deferred time-optimisation
reward once falls are near human, rather than after.

### 3h. Why is FlatIslands 3x worse than KOTM? (2026-09-19, unresolved)

Islands sits at falls100 ~0.90 while KOTM keeps improving (0.27). Two hypotheses tested, both
dead -- recorded so nobody spends the time again.

**1. "The greedy chain routes legs across the water."** Cut goal legs by detour ratio
(path distance / straight-line distance at the leg start) and compared fall rates. The result was
non-monotonic nonsense (direct 0.0 %, slight 8.4 %, detour 0.1 %, big detour 0.0 %). **The cut is
confounded**: a fall ends the segment, so only a segment's *final* leg can ever be marked fallen,
and the bins do not have comparable exposure. Do not reuse this method without fixing attribution.

**2. "Islands falls are detected late, so the signal is corrupted."** This asymmetry is REAL:

    FlatIslands: 2,141 falls caught by our own off-map counter, 0 by the game's OOB flag
    KOTM       :    23 caught by the counter,                 596 by the game's OOB flag

So on islands the fall is only noticed OFFMAP_FALL_DECISIONS (4 decisions, ~0.26 s) after the
marble leaves the map, while KOTM gets an immediate flag. But it does **not** matter: the reward
accrued during those 4 decisions is -0.18 per fall on islands (+0.18 on KOTM) against a -10
terminal penalty -- under 2 %. Not worth changing OFFMAP_FALL_DECISIONS for.

**Still unexplained.** Islands records more falls than arrivals (2,131 vs 1,351; KOTM is 619 vs
1,576) and collects 4.0 of a group against KOTM's 5.1. Note the maps differ in shape in a way that
matters: on KOTM 95.6 % of decisions are within 4 u of an edge (median 2.2 u), on islands only
32 % (median 5.0 u). Islands has open ground with lethal boundaries; KOTM is narrow everywhere.
So the islands failure is probably about what happens at a boundary the marble reaches at speed,
not about how often it is near one.

### 3i. What actually causes the post-gem-group falls (2026-09-19)

Falls are **not** overshoot at the gem. Distance to the target gem at the last grounded decision
before a fall:

                      on top (<2 u)   closing (2-5)   approaching (5-12)   travelling (>12)
    FlatIslands            3.2 %          10.5 %            40.6 %              45.6 %
    KOTM                   1.4 %          19.9 %            48.2 %              30.5 %

So approach braking is not the lever for falls, whatever else it is worth.

They cluster instead on the **redirect after a pickup** -- the thing gem groups introduced:

    decisions since the last pickup      % of falls    % of time    ratio
    FlatIslands  0-8  (first 0.5 s)         59.9 %        23.7 %     2.53x
    KOTM         0-16 (first 1.0 s)         70.9 %        41.3 %     1.72x

And on islands the required turn predicts it monotonically. `_advance_goal` picks the nearest
remaining gem by distance alone, ignoring which way the marble is already moving:

    turn needed at pickup   share of pickups   fall within 0.5 s
    straight on (<30 deg)        15.3 %              3.0 %
    gentle      (30-60)          16.4 %              3.9 %
    sharp       (60-90)          18.4 %              7.3 %
    very sharp  (90-135)         26.7 %             11.7 %
    reverse     (>135)           23.2 %             13.2 %

Half of island pickups demand a turn of more than 90 degrees at full speed, and those fall over
4x as often as straight-on ones. **KOTM shows no such relationship** (0.7 / 2.1 / 2.9 / 1.7 /
1.2 %) -- it is narrow enough that falls come from elsewhere -- so this is an islands fix.

Proposed (NOT implemented, awaiting approval): make the greedy order momentum-aware -- cost the
next gem by distance *and* the turn away from current heading, instead of distance alone. It is
also the more realistic stand-in for the planner, since chaining along the direction of travel is
what a human does.

### 3j. Turn-aware ordering: proposed, then WITHDRAWN (2026-09-19)

Three further measurements turned the 3i proposal around. Recorded in full because the
correlation there is real and will tempt the next reader the same way.

**1. The achievable gain is small.** Replaying group selection offline, a turn-cost order
(`d * (1 + k*(1 - cos theta))`) moves islands >90 deg links from 42.7 % to 35.7 % at k=1.0 and
saturates by k=2.0. Median turn 81.7 -> 68.8 deg. Groups are sampled as spatial chains, so
nearest-gem already roughly follows them.

**2. Edge proximity dominates the turn, though the turn is real.** Islands fall rate within
0.5 s of a pickup:

                      edge <3 u   edge 3-6 u   edge >6 u
    turn <60 deg        18.1 %       4.5 %       0.2 %
    turn 60-120         33.4 %      10.1 %       0.1 %
    turn >120           40.1 %      11.6 %       0.1 %

The turn effect survives the control (2.2x, 2.6x within the two near bands), but a pickup beyond
6 u of an edge is free at 0.1 % whatever the turn. 53 % of pickups are within 6 u, and those are
essentially all of the falls. Do NOT "fix" this by sampling gems away from edges -- real hunt gems
spawn near edges, and that trades transfer for the metric (see the standing rule in MEMORY.md).

**3. The decisive one: the policy is already learning the skill, from the V2 next-gem input.**
Arrival speed at a pickup, split by the turn the NEXT gem demands:

                          next straight on   next behind (>120 deg)
    FlatIslands               8.00 u/s             7.14 u/s
    KOTM                      7.83 u/s             6.10 u/s

    gap over the run:  islands +0.72 -> +0.93 | KOTM +1.54 -> +1.81 (first -> last third)

It anticipates the turn and arrives slower, and the effect is STRENGTHENING. Turn-aware ordering
would cut exposure to exactly the case it is learning from. Withdrawn; `patch_turnaware.py` is
staged in the session scratchpad if the judgement ever changes.

**Conclusion: the right action is to keep training.** Both curves are still climbing.

### 3k. Reading the arrive metric, and the reward economics (2026-09-19)

**Always convert group-arrive to per-gem before reacting.** `arrive` now means "collected the whole
group", so it is roughly per-gem^6. That amplifies small changes: KOTM falling from 67.9 % to
62.2 % group-arrive looks alarming but is 93.5 % -> 92.4 % per gem, 1.1 points. Parity with the
pre-gem-group model (95 % at a single waypoint) is **~74 % group arrive**, NOT 95 %.

    per gem   85 %    88 %    90 %    92 %    95 %    99 %
    group     37 %    46 %    53 %    61 %    74 %    94 %

**The reward is not mispriced.** Reward per decision by outcome:

                    complete   timeout    fall
    FlatIslands       0.555     0.421     0.374
    KOTM              0.536     0.413     0.369

Completing > stalling > falling on both maps, which is the correct ordering. A fall costs
FALL=10, the same as one gem (ARRIVE=10), and ends a group worth 4-8 of them; the measured gap is
that falling earns ~68 % of the completing rate.

Raising FALL is the obvious lever and is KNOWN BAD: the comment on FALL records that 20.0 (twice
the arrival bonus) made the policy freeze and brake 73 % of the time. Do not retry it blind.

**Also beware `falls100` and per-segment falls disagreeing.** `falls100` normalises per 100 u
travelled. Segments got much longer under gem groups, so through the KOTM dip `falls100` FELL
(0.52 -> 0.49) while falls per segment ROSE (15.7 % -> 22.6 %). `arrive` depends on the
per-segment rate. Judge outcome quality per segment, throughput per distance.

### 3l. KOTM's ceiling is precision at speed, not falls (2026-09-19)

KOTM sat flat at ~92.5 % per gem (62 % group) for ~200 updates. Its failure split is 22.6 % falls
but also **14.7 % timeouts**, against islands' 5.3 %, and the timeouts are the whole gap:
converting them to arrivals gives 77 % group = 95.8 % per gem, which IS the parity target.

What a KOTM timeout looks like, over its final 60 decisions:

    got within 2 u of the gem and missed   80.4 %
    moving but never closing               17.6 %
    wedged / barely moving                  0.0 %
    median closest approach 0.8 u | median speed 5.8 u/s | median area covered 15.9 u

So the marble passes within a fraction of a unit, repeatedly, at speed, and cannot tighten onto
the gem. Not a fall problem, not a navigation problem -- precision at speed, i.e. the same braking
deficit section 3g measures, surfacing as timeouts because KOTM is too narrow to loop around.

Ruled out: **vertical tolerance**. Only 1.4 % of timeouts got inside the 0.65 u horizontal radius
without collecting, so ARRIVE_DZ is not rejecting valid pickups.

**Two traps in this measurement.** (1) The closest-approach histogram piles 46.9 % into
0.65-0.80 u. That is CENSORING, not an orbit radius: a segment that dips under 0.65 collects the
gem and stops being a timeout, so timeouts are selected for min > 0.65 by construction. (2) The
trace's `seg` id does NOT increment when a segment is abandoned (round end, frame flip), so a few
trace rows merge several logical segments -- visible as "timeouts" crediting 10-14 gems for groups
capped at 8. Treat per-segment gem counts as slightly inflated in the tail.

## 4. Prep for a real KOTM run (2026-09-19)

Everything the navigator has trained on is a proxy: terrain-sampled waypoints, reached from a
teleport, with arrival judged by our own radius check. It has **never chased a real gem**. Before
spending more hours optimising the proxy, verify the skill transfers.

**Built (additive; no training code touched):**
* `nav/real_run.py` -- plays real Hunt rounds with the game's own gems as goals.
* `real_run.ps1` -- launches one instance on a spare port and runs it. Refuses to start while the
  trainer is up (pass -Force to override) so it cannot silently OOM a training run.

**What already existed and needs no work:**
* Real gem positions. `RAW_GEMS` carries the nearest 5 as (dx, dy, dz, value, dist). They are
  documented as camera-relative, but mlAgent.cs pins `$AIObserver::ForceYaw = 0`, so the
  observation frame IS the world frame: world = marble + delta. No game-side change needed.
* Arrival. In a real run the game scores the pickup (gemDelta), so ARRIVE_R tuning is irrelevant.
* Round, timer, respawn and reconnect handling, all via the existing HuntEnv.

**Traps found while building, do not re-learn them:**
* `gemDelta` is POINTS, not a gem count. The human demo averages 1.26 points per gem, so the two
  differ by ~26 %. Human reference, measured (not estimated) from
  `demos/demo_20260914_214854.npz`: **37.5 gems/min, 47.3 points/min, 8.05 u/s, 0.046 falls/100 u**
  over 6 rounds of 3 minutes.
* Target flapping. Re-picking the nearest gem every decision oscillates between two near-equal
  candidates and hands the navigator a new heading every tick. `choose()` is sticky: it holds the
  current gem until collected/gone unless another is much closer (SWITCH_GAIN).
* The GPU cannot hold a 9th instance beside a PPO update, so **the transfer test requires stopping
  training** -- it cannot run in parallel. Same constraint as `eval_heldout.ps1`.

**Known limit to measure, not to fix yet:** only the 5 nearest gems are sent, roughly one group.
There is no long-range view, so the agent cannot plan past its current cluster; when a group is
cleared and the next spawns far away it goes blind. `real_run.py` reports `blind_pct` for exactly
this. If it is large, the fix is a planner with a wider gem view, which is the roadmap's next
stage anyway.

**Remaining gaps after the transfer test, in rough priority:**
1. Gem selection is greedy-nearest. `NAV_VALUE_WEIGHT` switches to value-weighted (yellow first);
   neither is a real planner.
2. Continuous 5-minute play and real respawn recovery are untested end to end.
3. Opponents: the observation carries none. Single-player first.

### 4a. The run peaked at update ~6050 -- use that checkpoint, not nav_latest

Combined per-gem (mean of the two maps) peaked at **93.87 % around update 6049** and drifted to
93.07 % over the following ~80 updates. Both maps rolled over together (islands from 93.67 % at
~6017, KOTM from 94.41 % at ~6058), so this is the policy degrading, not one map interfering with
the other.

`models/nav/nav_006050.pth` sits almost exactly on the peak and is preserved as
`nav_peak_20260919_1215.pth`. **Use it for the transfer test and any evaluation**; nav_latest is
~0.8 points worse and still sliding. Numbered checkpoints land every 25 updates
(CHECKPOINT_EVERY), so the peak of any run is recoverable -- check before assuming nav_latest is
the best model.

Progress this session, for reference: the stale-goal fix at update 5560 left the policy at
79 %/90 % per gem (islands/KOTM); it reached 93.7 %/94.4 % by ~6050. Parity is 95 %.

**Update (12:5x):** the 6050 "peak" was a dip-and-recovery, not a ceiling -- the run went on to
94.08 % combined at update ~6310, which was also the newest data, i.e. still climbing. Best
checkpoint is now `nav_006300.pth`, preserved as `nav_peak_current.pth`. The run has now dipped
and recovered three times today; **do not call a plateau from a single downward stretch of
~50-80 updates.** Recompute the smoothed peak before picking a checkpoint rather than trusting
any name written here.

### 4b. TRANSFER TEST RESULT: it fails, and why (2026-09-19)

Ran `nav/real_run.py` on real KOTM rounds with `nav_006775.pth` (94.6 % per gem on synthetic
waypoints, islands at 95.0 %). Result, against a human 37.5 gems/min:

    round 1: 1.65 gems/min   round 2: 7.95   round 3: 0.33   round 4 (traced): 0.00
    speed 1.8-3.3 u/s   (human 8.05; the SAME policy does 5.0-5.6 in training)
    blind_pct ~0 -- the 5-gem window is NOT the bottleneck

**Root cause: training goals are never on edge cells; real gems often are.**
`TerrainGrid.sample_goal` draws from `_interior_cells()` -- its docstring says "a walkable,
**non-edge** cell". `interior = walkable & ~edge`. So:

    KingOfTheMarble_Hunt: 443 of 1082 walkable cells (41 %) can NEVER be a training goal
    FlatIslands_Hunt    : 876 of 5775 (15 %)

Real gems spawn anywhere walkable. In the traced round the agent locked onto a gem at
(-23.2, 17.0) -- walkable but NOT interior; the nearest synthetic goal ever sampled to it is
**8.43 u away**. It never closed within 4.43 u in 2550 decisions, while ranging 37 u across the
map, with `fwd` and `back` each active ~50 % of decisions.

That oscillation is the signature: the policy has learned hard edge-avoidance (edge rays + the
-10 fall penalty), so a gem sitting ON an edge puts the goal term and the edge term in direct
opposition. It approaches, the edge rays push back, it reverses -- forever. A gem it was never
trained to want is a gem it has been trained to avoid.

**Proposed fix (NOT applied, needs approval):** sample training goals from `walkable` rather than
`interior`, so edge-adjacent goals appear in training at their real frequency. Probably keep a
majority interior and mix in edge goals, rather than flipping wholesale, and watch falls -- the
fall rate on those goals will start high by construction.

**Two corrections to earlier reasoning in this session, recorded so they are not repeated:**
* I first reported a "22 u mismatch" between the game's gem distance and mine and concluded the
  world reconstruction was broken. That was MY error: the game's `%dist` is 3D, I compared a 2D
  horizontal distance. Checked in 3D the agreement is 0.0001 u -- the reconstruction
  (world = marble + delta, valid because ForceYaw pins the frame) is exact.
* `collectGems` breaks out of its scan once MaxGems slots are full, so it yields the first 5 gems
  in ItemArray order and *then* sorts -- "5 nearest" is a misnomer. In practice only the active
  spawned group is visible (3 seen, `gemsRemaining` 11 counts hidden/unspawned ones), so with
  groups of <= 5 it does not bite. It WILL bite if a group ever exceeds 5 gems.

### 4c. Edge goals deployed (2026-09-19 16:28)

`sample_goal` now draws from ALL walkable cells (`EDGE_GOAL_P = 1.0` in nav/terrain.py), not
interior-only, so goals appear on edge cells at each map's own rate: 43 % on KOTM, 16 % on
islands, matching where real gems spawn. Observation layout unchanged, so training resumed from
`nav_latest.pth` with no conversion. Snapshot: `docs/snapshots/pre_edgegoals_20260919_1628` and
`nav_before_edgegoals_20260919_1628.pth`.

Immediate cost, within ~10 updates:

                      before        after
    islands per gem   95.0 %   ->   90.5 %     (group 74 % -> 55 %)
    KOTM per gem      94.2 %   ->   86.5 %     (group 70 % -> 42 %)
    islands falls100   0.24    ->    0.62
    KOTM falls100      0.31    ->    0.70
    pickup speed       6.6     ->    6.0 u/s   (it brakes more near edges)

KOTM loses roughly twice what islands does, matching its 41 % vs 15 % edge share. **This is not a
regression -- it is the mismatch being paid for.** The old 94-95 % was measured on a task that
excluded 41 % of KOTM's walkable ground. Every per-gem number before 16:28 is optimistic and is
NOT comparable to anything after it; the 95 % parity target now means 95 % on the real,
edge-inclusive task, which is a strictly harder bar than the one the pre-gem-group model met.

### 4d. Scoring model: gem VALUE is irrelevant, time-per-group is everything (2026-09-19)

From the user, who plays this game: **every gem in a spawned group must be collected before the
next group spawns.** Therefore:

    score = (groups cleared in the round) x (points per group)

The points in a group are fixed the moment it spawns, so collecting a yellow (2 pts) before a red
(1 pt) changes NOTHING about the final score. **Do not weight gem selection by value.**
`NAV_VALUE_WEIGHT` in nav/real_run.py exists but should stay 0; the value-weighting idea is dead.

What DOES matter is the wall-clock time to clear each group: order the 4-6 gems in the group so
the route through them is fastest, and move quickly between them. That makes the "planner" a
shortest-route problem over the current group, not a selection problem -- and it makes raw SPEED
the dominant term in the score, which matches the measured gap (human 8.05 u/s, agent 5.0-5.4 in
training and 1.8-3.3 in real rounds).

Human reference on KOTM (the demo was recorded on this map), per 3-minute round:
163, 177, 172, 121, 119, 109 -> mean 143.5 points, best 177. Agent's best real round so far: 29.

### 4e. Why real-round scores are ~6-11 points against a human's 143 (2026-09-19 evening)

Three transfer runs on KOTM with the edge-goal + fall-continue model (update 7080). Rounds are
3 minutes; the human demo on this same map scored 163/177/172/121/119/109, mean 143.5.

    baseline          7, 7, 19      mean 11.0 points
    + goal snapping   1, 16, 6      mean  7.7
    + path waypoints  6, 11, 0      mean  5.7

**Method warning: none of those differences are real.** Individual rounds range 0-19, so the
round-to-round variance is larger than every effect measured. Three rounds cannot distinguish
these configurations; use 8-10 rounds per arm before believing any harness comparison.

**The actual failure mode, which IS consistent across every round:** one target consumes 71-99 %
of the round and is never collected.

    round 1: target (-23.2,17.0)  2520 decisions (99 %), stalls at 9.4 u
    round 2: target (-27.2,13.0)  1963 decisions (71 %), stalls at 8.2 u
    round 3: target (-27.2,17.0)  2834 decisions (100 %), stalls at 13-15 u

Because a group must be cleared before the next spawns, ONE unreachable gem ends the round --
the agent clears a couple of groups, stalls, and scores single digits. When it is not stalled it
is fine: 9 of 10 walkable targets collected, median 73 decisions (4.7 s) each.

**Four hypotheses tested and REFUTED** (do not re-run these):
* *Recurrent state drift over a 2800-decision round.* H reset never/pickup/every-60 gave speeds
  2.85/2.89/3.02 -- no effect. (`NAV_H_RESET` in real_run.py.)
* *Gems on non-walkable cells.* Real: 1 of 10 targets, and it did eat 42 % of one round, so
  `snap_to_walkable()` is worth keeping -- but most stalls are on walkable cells.
* *Gems needing a jump.* KOTM is ONE connected component without jump edges (1082 cells), so
  every gem is rollable-to.
* *Path waypoint flip-flopping between routes.* Waypoints are stable: median heading change
  2.8 deg, never above 90.

**Leading explanation: training has an escape hatch that a real round does not.** A stuck
segment times out after ~312 decisions per remaining gem and the marble is TELEPORTED to a fresh
start. The policy has therefore never had to self-recover from a bad state -- it can simply wait
one out. In a real round nothing rescues it, so the same state is terminal. This is a capability
gap, not a harness bug, and no amount of harness tuning will close it.

`nav/real_run.py` keeps the snapping and path-waypoint machinery (both are correct and cheap);
`LOOKAHEAD_U` aims the navigator 12 u along the Dijkstra path so it is never pointed across a hole.

## 5. THE PHANTOM-HOLE BUG (2026-09-19 evening) -- the one that mattered

The user watched a real round and said the marble jitters "on flat ground with no danger of
falling". The terrain grid disagreed, so we checked against the game itself by teleporting the
marble into cells the grid called non-walkable and seeing whether it stood:

    (-18.7, 17.5)  grid: not walkable  ->  STOOD on solid ground
    (-19.7, 17.5)  grid: not walkable  ->  STOOD
    (-15.7, 17.5)  grid: not walkable  ->  STOOD
    (-18.7, 24.5)  grid: not walkable  ->  STOOD
    (-17.7, 17.5)  grid: not walkable  ->  fell   (a real hole)
    (-18.7, 20.5)  grid: not walkable  ->  fell   (a real hole)

**4 of 6 "holes" were solid floor.** The navigator's edge rays and goal sampling are built from
that grid, so it had been refusing ground that is actually there -- in training as well as at
inference.

**Two bugs in `TerrainGrid._build_walk_grid`, both fixed:**

1. **The slope killed gap neighbours.** `np.gradient(np.nan_to_num(filled, nan=0.0))` replaced a
   missing cell with z = 0 while these floors sit at z ~ 20.65, so every gap read as a 20-unit
   cliff and the gradient at its 8 neighbours blew past `MAX_SLOPE = 0.6`. One missing cell made
   all of its neighbours non-walkable, so small rasterisation gaps bloomed into large phantom
   holes. Now the slope is computed against each gap's nearest finite height (no artificial
   gradient); `present` still masks genuinely missing cells and the edge test is untouched
   because it reads `filled`, which keeps its NaNs.
2. **Subsampling threw away floor.** `heights[0][::step, ::step]` took ONE 0.5 u sub-cell to stand
   for each 1 u walk cell; 174 KOTM cells were marked empty although their block did contain
   floor. Now each walk cell is `nanmax` over its whole block (and the array is padded, not
   truncated, so the last row/column survives).

    KOTM walkable cells   1082 -> 1664  (+54 %)
    FlatIslands           5775 -> 6006  (+4 %)
    grid vs ground truth   2/6 -> 4/6

**Effect on real rounds, same checkpoint (update 7080), 6 rounds:**

    before:  0-19 points, mean ~8,    speed 2.0-2.5 u/s, 1-3 gems/min
    after:   42, 53, 65, 34, 6, 45    mean 40.8,  speed 4.03 u/s, 11.6 gems/min

**A 5x score improvement from fixing the map, not the policy.** Human reference on KOTM is 143.5
points per round, so this moves us from ~6 % to ~28 % of human.

**Still open:** 2 of 6 ground-truth cells remain wrong because the HEIGHT MAP itself has no
surface there -- `generate_terrain_map.py` reads only `InteriorInstance` geometry, and the marble
settled at z = 21.35 and 20.91 on those cells, ABOVE the 20.65 floor, so it is standing on
something raised (the mission also has 12 StaticShapes). Fixing that needs the generator and a
map regeneration. Round 5 of the six above (6 points, 88 % pathed) is very likely one of these.

**Method note:** this was found only because the user watched the game. Four plausible hypotheses
had already been tested and refuted from traces alone (recurrent drift, non-walkable gems, jump
requirements, waypoint flip-flop), and a fifth -- "training's timeout-teleport escape hatch" --
was wrong too. When a measurement and a human observation disagree, check the measurement against
the game.

### 5a. A/B on the walk-grid aggregation: the SLOPE fix is the win, the block aggregation HURT

After the slope fix I also changed the walk grid to aggregate each 1 u cell over its 2x2 block of
0.5 u height cells instead of sampling one of them, on the theory that 174 KOTM cells were being
discarded. Measured against real-round score, 8 rounds per arm, same checkpoint (update 7080):

    NAV_WALK_AGG=sub       1490 walkable   29,31,19,10,1,36,51,25   mean 25.3 points
    NAV_WALK_AGG=majority  1643 walkable   1,5,26,0,10,12,10,1      mean  8.1
    NAV_WALK_AGG=any       1664 walkable   14,6,0,32,1              mean 10.6

**Aggregation costs ~3x the score, so `sub` is the default.** A 1 u walk cell that is only half
floored is genuinely unsafe for a rolling marble; those 174 cells were correctly excluded. The
env var is kept so this can be re-tested, but do not change the default without an 8-round A/B.

The earlier "40.8 points" figure was the same `sub` config over only 6 rounds -- a lucky sample.
Treat 25.3 as the real post-slope-fix baseline, against ~8 before it and a human 143.5.

**Rounds are extremely noisy** (1 to 51 points within a single arm). Never compare terrain or
harness changes on fewer than 8 rounds; three rounds is worthless, as an earlier sequence of
11 -> 7.7 -> 5.7 "regressions" showed, all of which were noise.

### 5b. The speed gap is STEERING JITTER, not braking (2026-09-19 night)

Score is groups cleared per round, so speed is the dominant term, and the agent travels ~4.0 u/s
against the human's 8.05. Measuring the commanded direction against the marble's own velocity,
both at 64 ms decisions (the human demo subsampled 4:1 to match):

                                   human        agent
    heading change per decision     0.4 deg     39.5 deg
    heading flips > 90 deg          3.8 %       25.8 %
    thrust vs velocity, mean cos   +0.26        -0.07
    driving forward (cos > 0.5)    51.5 %       31.7 %
    consecutive driving runs        10 dec       1 dec
                                   (0.64 s)     (0.06 s)

**The agent cannot accelerate because it never holds a direction.** It commands full throttle
(magnitude 0.99) but re-aims a median 39.5 degrees every 64 ms, so thrust opposes its own motion
more often than it helps. A marble needs sustained thrust to reach 8 u/s; 0.06 s is nothing.

**This is NOT the brake action.** Exact velocity reversals -- the signature of the brake, which
forces `-v/|v|` -- are only 12.4 % of moving decisions against the human's 6.6 %, so braking is
roughly human-like. The other 27 % of backwards thrust is the DIRECTION HEAD itself. Do not
attack this by changing BRAKE or FALL.

It is also not exploration noise: real_run evaluates with `deterministic=True`, so this is the
policy's MEAN direction oscillating. It looks like a learned bang-bang limit cycle (thrust back,
speed drops, thrust forward, repeat) that the reward never penalised, because total progress
reward over a segment is fixed by path length regardless of how jaggedly it is covered.

Being A/B tested now via `NAV_SMOOTH` in nav/real_run.py, an EMA low-pass on the commanded
direction at inference (1.0 = off), 8 rounds per arm. If smoothing helps, the permanent fix is
training-side: either a smoothness penalty on the change in commanded direction, or the same
low-pass applied inside the action pipeline so the policy trains in a smoothed action space.

### 5c. Direction smoothing is a speed/precision dial (2026-09-19 night)

`NAV_SMOOTH` in nav/real_run.py low-passes the commanded direction at inference (EMA weight on
the new heading; 1.0 = raw). 8 rounds per arm, checkpoint 7080, KOTM:

    SMOOTH   points   speed
    1.0       31.6    3.51 u/s      (raw policy)
    0.5       18.5    3.65
    0.3       18.8    6.14
    0.15       4.1    7.75          (human is 8.05)

**Smoothing buys speed almost exactly as predicted and costs score**, because the policy has
never learned to arrive at a gem while moving that fast -- at 0.15 it matches human speed and
collects almost nothing. This confirms 5b: steering jitter, not braking or throttle, is what caps
speed at 4 u/s.

**So the filter belongs in TRAINING**, where the policy can adapt its approach and braking to the
smoothed dynamics. Deployed 19:00 in nav/vec_worker.py as `ACTION_SMOOTH` (env
`NAV_ACTION_SMOOTH`, default 0.3), applied to the direction before `action_to_joystick`, reset on
segment start because a teleport makes the previous heading meaningless.

Implementation note: the EMA accumulates UNNORMALISED and normalises only the output. Keeping a
normalised state makes an exact 180-degree reversal a fixed point (opposite unit vectors cancel),
so the heading could never flip -- caught in a dry run before deploying.

First updates after deploying (expect a dip while it adapts):

    islands  speed 5.0 -> 5.5, gems/grp 5.1, falls 0.93   (fine)
    KOTM     speed 3.9, arrive 37 % -> ~20 %, falls 1.75 -> ~2.8   (needs to adapt)

If KOTM does not recover overnight, try ACTION_SMOOTH 0.5 as a gentler setting, or revert from
`docs/snapshots/pre_actionsmooth_20260919_1858`. Judge it by real-round score (8 rounds), never by
the training curve -- the raw-policy baseline to beat is **31.6 points**.

### 5d. Why 3 days of KOTM training never revealed the phantom holes

The maps are byte-identical: same .mcs, same .dif, same terrain_*.npz. The phantom holes were in
the grid during training too. Nothing about the geometry changed when we ran real rounds.

**What changed is who chooses the goals, and that closed a self-validating loop.**

    KOTM walkable cells:  training's grid 1082  |  reality 1490
    ground the grid denied:  408 cells = 27 % of the real walkable area

`sample_goal` draws goals from that same broken walkable grid, and the Dijkstra reward field
routes around those same phantom holes. So training never placed a goal in the denied 27 %, never
routed a path through it, and actively REWARDED the policy for treating phantom holes as walls --
which is optimal behaviour given the map it was handed. 94 % per gem was an honest score on a
task whose map and whose goals shared one error.

Real rounds break the loop at exactly one point: **gems come from the game, not from our grid.**
The game spawns them over 100 % of the real floor. Of 12 real gem positions observed in traces,
1 sat on ground training believed was void, and routes to others cross it. The goal term then says
"go there" while the terrain perception says "cliff", and the marble oscillates on flat ground.

Note the policy's PERCEPTION (crop + edge rays, built from the same height map) was equally wrong
in training. The difference is that nothing in training ever required it to disbelieve its eyes.

**The lesson: a proxy that generates its own targets validates itself against its own bugs.**
Every self-consistency metric -- arrive %, falls/100 u, per-gem reliability -- is computed inside
that loop and is structurally blind to this class of error. Only a goal source from OUTSIDE the
loop can catch it. Practical consequences:
* Treat `nav/real_run.py` as the gate for anything terrain- or perception-related, not the
  training curve. Run it after ANY change to terrain.py or generate_terrain_map.py.
* Ground-truth every map against the game before trusting training on it (probe_ground.py:
  teleport into cells the grid calls non-walkable and see whether the marble stands).
* When a human watching the game contradicts a measurement, the measurement is the suspect.

### 5e. Action smoothing TRAINED IN: 31.6 -> 58.1 points (2026-09-19 night)

Deployed `ACTION_SMOOTH = 0.3` in nav/vec_worker.py (env `NAV_ACTION_SMOOTH`), low-passing the
commanded direction before it reaches the game so the policy trains against the smoothed
dynamics. After ~90 updates, 8 real KOTM rounds per arm:

    trained WITH smoothing, inference filter OFF   65,62,64,53,40,64,57,60   mean 58.1   5.07 u/s
    trained WITH smoothing, inference filter ON    26,28,29,19,38,23,25,35   mean 27.9   6.10 u/s
    raw policy, no smoothing anywhere                                        mean 31.6   3.51 u/s

**58.1 points, up from 31.6, and the catastrophic rounds are gone** -- worst round 40, where the
raw policy regularly scored 0-10. The policy internalised steadier steering, so it no longer
needs the filter at inference; applying it twice over-damps and costs half the score. **Run
real_run.py WITHOUT NAV_SMOOTH against a smoothing-trained checkpoint.**

Note the training curve said the opposite: KOTM arrive fell 37.7 -> 22 %, falls rose 1.77 -> 2.7,
and training speed did NOT rise (3.90 vs 3.80) because training samples actions stochastically so
the noise survives the filter. Another case where the training metric is not the objective --
this change looked like a clear regression right up until it was measured on real rounds.

Session total on real KOTM rounds: **8 -> 58.1 points** against a human 143.5 (40 %).

### 5f. Gem promptness bonus, filter removed: 62.8 -> 72.2 points (2026-09-20)

The steering filter of 5e was a crutch and the user rejected it as one ("that's a fucking hack.
you're letting the model fudge its way through. i'd rather it learn properly"), with a specific
redirection: "stop trying to affect the marble's direction and instead put some pressure on it to
get the next gem quickly or at least get the current one faster."

Two changes shipped together, because the second replaces the first:

1. `ACTION_SMOOTH = 1.0` in nav/vec_worker.py -- the filter is OFF.
2. `GEM_SPEED_BONUS = 8.0`, `GEM_SPEED_REF = 60` in nav/waypoints.py, paid **on collection**:

        r += ARRIVE + GEM_SPEED_BONUS * max(0.0, 1.0 - s.gem_decisions / float(GEM_SPEED_REF))
        s.gem_decisions = 0                   # the next gem is timed from here

`gem_decisions` counts decisions since THIS gem became the target, reset on each collection.
Nothing about direction, throttle or technique -- only "arrive sooner". Calibration against
measured behaviour, a gem being worth ARRIVE = 10: human pace ~24 decisions -> +4.8, agent median
36 -> +3.0, 60 or slower -> +0.0. So closing to human pace is worth ~+1.8 a gem, ~11 over a
six-gem group against ~60 for the group.

**Why not just raise TIME.** TIME is a flat charge on every decision including useful ones;
raising it 0.05 -> 0.12 made speed WORSE and was reverted. The promptness bonus pays only on the
outcome, so there is no way to earn it except by getting there earlier, and it cannot be farmed by
abandoning a gem because it is only paid on collection.

Result, 8 real KOTM rounds on `models/nav/nav_gemspeed_20260920_0453.pth` (update 8765):

    78,77,67,77,69,74,69,67   mean 72.2 points   5.69 u/s   19.3 gems/min   0.57 falls/100u
    previous best (5e + spawn goals + bevel fix) 62.8 points   5.26 u/s
    human                                       143.5 points   8.05 u/s   37.5 gems/min   0.046

**The learned version is faster than the imposed one ever was** (5.69 vs 5.26), and the worst
round is 67 where the raw pre-smoothing policy regularly scored 0-10. Speed came from the reward,
not from a filter, so it should carry to other maps.

**Remaining gap to human, decomposed.** 143.5 / 72.2 = 1.99x, and it factors almost exactly:

    route inefficiency   19.0 u travelled per gem vs human 12.9   = 1.47x
    raw speed             5.69 u/s vs human 8.05                  = 1.41x  (1.60x at the peak-era 5.0)

Falls are no longer the bottleneck (0.57 vs 0.046 per 100u costs little at 180 s). **I first read
the route factor as a planning problem and that was WRONG -- see 5h, which measures it properly.**

**Continuation run from that checkpoint (started 04:58, 511 KOTM samples).** It dipped for ~250
samples and then recovered; do not read the dip as a plateau:

                   arrive  falls  speed  gems/grp
    run start        78.0   1.19   5.11    5.59
    the dip          78.1   1.31   4.88    5.63
    latest           81.8   1.05   4.94    5.70

Arrive, falls and gems/group are all at run highs; only speed is still ~3 % below the opening.
The trade looks favourable on paper -- more gems per group is worth more than 3 % speed -- but
**this has NOT been measured on real rounds**, so 72.2 remains the last verified number and
`nav_gemspeed_20260920_0453.pth` the last verified checkpoint. Entropy 0.13 and falling.

I called the dip a sustained decline while it was happening and was wrong, which is exactly what
4a already warns about: **do not call a trend from a stretch of 50-250 updates on this setup.**
The navigator dips and recovers repeatedly; only real rounds settle it.

**Methodology, restated because it has now been right five times.** Judge only by 8 real rounds
through nav/real_run.py. The training curve disagreed with the real score on the terrain fix, the
walk-grid aggregation, and smoothing 0.2->0.3; it also failed to show the phantom holes at all
(5d). It is a training diagnostic, not the objective.

### 5g. The other four maps were never affected -- and could not have been (2026-09-20)

Static check of all five terrain maps, building the walk grid twice (new: gaps filled from the
nearest finite neighbour; old: `nan_to_num(nan=0.0)`) and counting cells that differ. No game
instance needed, ~20 lines, runs in seconds:

    map                          floor  walk NEW  walk OLD  phantom  % of floor
    FlatGemTraining_Hunt         15750     15750     15750        0       0.0%
    FlatIslands_Hunt              5775      5775      5775        0       0.0%
    FlatWithJump_Hunt            15750     15750     15750        0       0.0%
    JumpOnly_Hunt                15750     15750     15750        0       0.0%
    KingOfTheMarble_Hunt          1488      1488      1078      410      27.6%

**The four training maps have no missing walk cells at all** (walk NEW == walk OLD == floor), so
the old code had nothing to set to z = 0 and produced an identical grid. The bug can only appear
on a map with interior holes or a ragged boundary, and KOTM was the only one -- where **27.6 % of
the arena floor was marked unwalkable**.

This closes out 5d. The curriculum was not merely slow to reveal the phantom holes; it was
*structurally incapable* of revealing them, and would have stayed green for any number of updates.
Two consequences:

* **The flat curriculum validates nothing about hole handling.** Any future catalogue map with
  interior holes is exposed to this bug class, and the training maps will not warn about it.
* **Run the comparison above on every new map before training on it.** It is static, cheap, and
  catches the entire class. Combined with the spawn-pool rule (waypoints only where real gems can
  spawn), it removes the self-validating loop described in 5d.

### 5h. The gap is ONE defect, not two: the marble does not hold a line (2026-09-20)

Measured gem-to-gem legs on both sides -- agent from `logs/nav/real_trace.csv` (8 real rounds on
nav_gemspeed_20260920_0453), human from `demos/demo_20260914_214854.npz` (6 KOTM rounds). For each
leg: distance actually travelled, against the straight-line distance between the two pickup
points. The ratio is scale- and rotation-invariant, and the human figure is identical at 26, 53
and 79 ms sampling, so it is not an artefact of decision rate.

                        straight-line   travelled    wander
                        between gems     per gem
    human (682 legs)        11.0 u        12.8 u      1.17x
    agent (465 legs)        11.1 u        17.8 u      1.60x

**The agent's gem SELECTION is already at human parity.** 11.1 u between consecutive pickups
against the human's 11.0 means it is choosing gems that are just as close together. There is no
ordering deficit and **no planner problem** -- the "1.47x route inefficiency" quoted in 5f is
entirely excess travel between gems it already picked correctly.

That completes the arithmetic of the gap:

    wander   1.60 / 1.17 = 1.37x
    speed    8.05 / 5.69 = 1.41x
    product              = 1.93x   vs the observed 143.5 / 72.2 = 1.99x

Two factors cover essentially the whole 2x, and they are **the same underlying defect**: a marble
that wobbles pays twice, once in extra distance and once in the speed it cannot sustain through
the wobble. Per-leg wander is p25 1.13, median 1.25, p75 1.61, p90 2.41 -- so a heavy tail of bad
legs rather than a uniform slackness.

**Consequences for what to build next.**

* The planner (section 5 of the plan) is NOT the next lever. It would be optimising something
  already at parity.
* Both remaining factors respond to the same thing: holding a commanded heading. GEM_SPEED_BONUS
  attacked it indirectly through the outcome and moved speed 5.26 -> 5.69; the wander number says
  there is another ~1.37x sitting in the same place.
* ACTION_SMOOTH was the wrong instrument (imposed, not learned, and rejected on those grounds) but
  it was aimed at the right target. Whatever replaces it should make straightness pay rather than
  enforce it.
* 8.1 % of decisions have NO gem visible and account for 6.1 % of distance travelled. Real, but an
  order of magnitude smaller than the wander term; not where to start.

## 6. The entropy bonus bought braking, not steering (2026-09-20)

Raising `ENTROPY_COEF` 0.002 -> 0.01 to break the points plateau had an effect nobody asked for.
Over 90 updates the brake rate went from 0.34 % of decisions to **17.25 %**, speed fell 4.9 -> 4.7
and arrivals 86 % -> 83 %.

Cause: `model.evaluate_seq` summed the entropy of all four action distributions and the loss paid
one coefficient on that sum:

    ent = d_dir.entropy().sum(-1) + d_thr.entropy() + d_jump.entropy() + d_brake.entropy()

Jump and brake are Bernoulli. Bernoulli entropy rises very steeply as p leaves 0, so when the
coefficient goes up, gradient descent buys entropy from those heads first because they are far
cheaper than widening a Gaussian. The decomposition over those 90 updates:

                        upd 9416    upd 9502    change
    direction            -0.592      +0.066     +0.658
    throttle (pinned)    +0.522      +0.522       0
    jump + brake         +0.080      +0.403     +0.323
    total                 0.010       0.990

A third of what we paid for went into randomising braking, which is the opposite of the goal,
since the whole point of raising entropy was to find a FASTER way to drive.

**Fix.** `evaluate_seq` now returns continuous and discrete entropy separately.
`ppo_recurrent.ppo_update` prices them apart: the adaptive coefficient applies only to the
continuous heads, and the discrete heads get `DISCRETE_ENT_COEF = 0.002`, fixed, which the
controller cannot reach. That keeps jump exploration alive for the island maps (it needs to be,
see the JUMP_PRIOR history) while making it impossible to buy entropy by braking at random.
NAV lines now log `entd=` alongside `ent=` so the split is visible.

The damaged policy is kept as `models/nav/nav_brakebug_20260920_0842.pth` if anyone wants to
confirm the behaviour. Training was restarted from `nav_wander_20260920_0805.pth` (74.1 points,
brake 0.34 %) rather than from the damaged weights, because a 17 % brake habit is in the weights
and not only in the sampling.

**General lesson.** An entropy bonus on a mixed action space is not a single dial. It flows to
whichever head sells entropy cheapest, which is rarely the head you want explored. Price
continuous and discrete heads separately, and watch a behavioural rate (brake %, jump %) rather
than the scalar entropy, because the scalar looked perfectly healthy the whole time: it went from
0.01 to 0.99 exactly as intended while the policy was being quietly wrecked.

## 7. Correction to 5h: the human does not arrive aimed at the next gem (2026-09-20)

Section 5f/5h and the speed analysis that followed quoted a "human carry of ~7.4 u/s". That number
is wrong and the mistake is worth recording, because it is easy to repeat.

7.37 u/s is the human's SPEED when 1-2 u from the gem they are collecting. It is not the component
of that velocity pointing at the NEXT gem. Measuring the angle directly on the demo, 682 pickups:

                        arrival angle to next gem    carry (v toward next)
    human                      72 deg mean                 2.28 u/s
    agent                      90 deg mean                 ~0.0 u/s
    human within 45 deg        19 % of pickups

**The human arrives mostly sideways too.** Their advantage over the agent is about two thirds raw
speed at arrival (7.4 against 5.4) and one third better aiming (72 deg against 90). The quantity
`carry = speed * cos(angle)` captures both and reproduces the measurement exactly: human
7.4*cos(72) = 2.29, agent 5.4*cos(90) = 0.

Consequence for the CARRY term (section 6 of overnight_notes): it was first shipped with
`CARRY_REF = 8.0`, i.e. the bonus saturating at 8 u/s of aligned speed, a level no player reaches.
The entire achievable range sat in the bottom 29 % of the curve, the gradient per u/s was a third
of what it should have been, and 96 updates produced no movement at all. Recalibrated to
`CARRY_REF = 2.5`, just above the measured human 2.28.

**The general lesson.** A speed profile and an aligned-speed profile are different measurements
and they differ by cos(angle), which here is a factor of three. Before calibrating a reward term
against "what the human does", measure the exact quantity the term pays for, not a related one.

## 8. The speed limit is the MEAN policy output oscillating (2026-09-20)

Three experiments on 2026-09-20 all returned null, and measuring why produced the most useful
number of the day. Commanded heading change per 64 ms decision, measured from the trace:

    first fifth of the run    41.4 deg   (direction std 0.26)
    last fifth of the run     39.7 deg   (direction std 0.22)
    human demo                 0.4 deg

    sampling noise alone explains:  std 0.26 -> ~20.6 deg | std 0.22 -> ~17.5 deg

Sampling accounts for less than half of it. Subtract it and the policy's **mean** commanded
direction still swings about 36 deg every decision. **Driving entropy to zero would leave 36 deg
against a human 0.4.** Exploration noise is not the speed constraint and never was.

This is the common cause behind all three nulls of the day:

* **CARRY (arrival alignment)** asked the policy to arrive at each gem already aimed at the next.
  A marble whose commanded heading rotates 40 deg per decision cannot hold any approach geometry,
  so the behaviour being paid for is not one its control can produce. Two calibrations, 160
  updates, no movement.
* **BRAKE_SUPPRESS** was never binding: the brake head's bias is -2.97 but its effective logit in
  play is about -6.5, so the policy had learned ~3.5 logits of suppression of its own and the
  guard made no difference (brake share 0.22 % -> 0.15 %).
* **Entropy setpoint** cut std 0.26 -> 0.22 and moved speed not at all (4.8 -> 4.7), exactly as
  the decomposition above predicts.

**What this implies.** Every remaining lever that goes through "drive better" is blocked behind
the mean-output jitter. Outcome pressure worked while there was slack elsewhere (GEM_SPEED_BONUS
took real rounds 62.8 -> 72.2) and has now plateaued, because you cannot incentivise a controller
into a behaviour its output cannot express.

The untried mechanism aimed at it is a per-decision cost on changing the commanded direction,
`TURN_COST * (1 - cos theta)`. Jitter pays it ~230 times a segment while a genuine turn pays a
handful of times, so it falls on twitching rather than on turning. At the measured 40 deg,
`1 - cos(40) = 0.234`, so a segment currently carries about 54 * TURN_COST of cost.

It is worth recording WHY this was not tried earlier: `ACTION_SMOOTH`, a low-pass filter on the
commanded direction, did the same job mechanically and was rejected by the operator as a crutch
("you're letting the model fudge its way through, i'd rather it learn properly"), with an
instruction to use outcome pressure instead. That instruction was right at the time and produced
the largest single gain of the project. `TURN_COST` is the learned version of the same target: a
cost the policy has to satisfy itself rather than a filter applied to its output.

## 9. The heading jitter is generated by the RECURRENT STATE, not the input (2026-09-20)

Section 8 established that the policy swings its commanded heading ~40 deg per 64 ms decision
against a human 0.4, and that this blocks every speed lever. This section localises the cause.

**It is not sampling.** `real_run.py` acts with `deterministic=True`, and its trace shows the mean
policy swinging **37.5 deg mean / 20.7 median / 103 p90** per decision with no sampling at all.
Cutting direction std 0.26 -> 0.22 in training moved total jitter only 41.4 -> 39.7 deg.

**It is not the observation.** Feeding the network two observations that differ by one decision's
worth of movement, with the hidden state held fixed:

    perturbation                                median    p90
    0.01 u of position                           0.1       2.6
    0.05 u of position                           0.1       2.5
    0.32 u (one decision at 5 u/s)               0.2       3.5
    0.50 u of position                           0.2       3.5
    velocity by 0.2 u/s                          0.0       0.3

A full decision of movement moves the output 0.2 deg. The network is not input-sensitive.

**It is the recurrent state.** Same network, same smooth straight-line trajectory toward a fixed
goal, 30 steps, the only difference being whether the GRU hidden state carries forward:

                                              mean    median    p90
    hidden state reset every step              3.7      0.3     10.2
    hidden state evolving (as in play)        14.1      7.3     36.8
    observed in play (deterministic)          37.5     20.7    103.2

**Letting the hidden state evolve multiplies the median output change by 24** on an input that is
changing perfectly smoothly. Play is worse again (37.5 vs 14.1), most likely because the synthetic
marble travels in a straight line regardless of what the policy commands, so it cannot exhibit
closed-loop feedback instability; in the game the command moves the marble, which changes the
observation, which changes the command.

**Candidate mechanism, NOT yet verified.** During acting the hidden state evolves continuously
across thousands of steps (`w.h = o['h_next']` in train_nav.py). During the PPO update, rollouts
are cut into SEQ_LEN=32 chunks and the GRU is re-run from the STORED hidden state at each chunk
boundary (`h0 = ...view(nseq, SEQ_LEN, HIDDEN)[:, 0, :]` in ppo_recurrent.py). If the recurrent
dynamics are even mildly unstable, the hidden trajectory PPO reconstructs diverges from the one
that actually produced the actions, so importance ratios are computed against a history that never
happened. That is a known failure mode of truncated BPTT with stored states and it would both
allow and reinforce an unstable recurrent state.

**Why this matters for everything else.** It explains the three nulls of 2026-09-20 at a stroke:
CARRY asked for an approach geometry the controller cannot hold, BRAKE_SUPPRESS was never binding,
and cutting exploration noise could not help because the noise was not the source. It also means
TURN_COST is treating a symptom: it does reduce the jitter (40.4 -> 34.9 deg over 180 updates
across two settings) but it is fighting the recurrent dynamics rather than fixing them.

### 9a. The mechanism: dir_head reads ONLY a churning hidden state (2026-09-20)

Replaying the RECORDED observation sequence from the 74.1-point run back through the checkpoint
that produced it, 1200 decisions, changing only whether the GRU state carries forward:

                                    mean    median    p90
    hidden state RESET each step    13.0      3.4     32.8
    hidden state EVOLVES            32.0     16.8     83.2
    actually recorded that run      37.5     20.7    103.2

The evolving-state replay reproduces the recorded behaviour, so the replay is faithful. Making the
same weights memoryless **cuts median jitter by 5x**. For scale the human is 2.7 mean / 0.0 median,
so a memoryless version of this network is already near human smoothness.

The hidden state is not exploding and barely saturating, but it CHURNS:

    |h| over 1500 decisions      5.6 -> 11.1, mean 8.0     bounded
    saturated units (|h| > 0.95) 15 of 256 (6 %)           mild
    |h_t - h_{t-1}|              2.08, i.e. 26 % of |h|    EVERY decision

A quarter of the state turns over per decision on an input moving a few percent, so its effective
memory is ~4 steps. It is not storing anything, it is re-randomising.

**Why that reaches the steering.** In model.py:

    mean_xy = self.dir_head(h)        # reads ONLY the hidden state

The unit vector to the waypoint is already an input feature (`vec[0:2]`), exact and perfectly
smooth, and the network must reconstruct it through a 256-unit GRU that rewrites a quarter of
itself every frame. Zeroing `h` removes the churn and the output becomes a clean function of the
current input, which is exactly the 5x result above.

**Ruled out, with evidence, so nobody re-treads them:**

* Sampling noise. `real_run` is `deterministic=True` and still jitters 37.5 deg. `|mean_xy|` is
  0.87-1.05 (healthy), giving angular noise of only 11-12 deg at the current action std.
* Input sensitivity. With `h` fixed, one decision of movement (0.32 u) changes the output 0.2 deg
  median, 3.5 deg p90. Velocity by 0.2 u/s changes it 0.0 deg.
* Frame or conversion error. Commanded direction vs actual acceleration has a circular mean offset
  of +0.7 deg (islands) and -0.1 deg (KOTM): the marble goes exactly where it is told.
* Hidden state plumbing. Carried per worker (`w.h = o['h_next']`), reset to `initial_state` on a
  new segment with `reset_flag` aligned to `resets[t]` in `evaluate_seq`, and `act_model` is
  synced from `model` immediately after every update with the pipeline settled first. Correct.
* Contradictory reward flipping each frame. Lag-1 autocorrelation of the signed heading change is
  -0.04, i.e. a random walk, not an alternation.

**Proposed fix (not yet applied):** give `dir_head` a direct path to the observation, i.e.
`dir_head([h, vec])` instead of `dir_head(h)`, so the direction is anchored to the goal bearing and
the recurrent state contributes corrections rather than carrying the whole signal. Warm-start with
the new input weights zeroed so the converted policy is bit-identical and nothing is lost (same
technique as `convert_ckpt_v2.py` for the V2 observation change). Optionally add LayerNorm on the
GRU output to damp the churn itself.

### 9b. The fix, and an honest weakening of the story (2026-09-20)

APPLIED: `dir_head` now reads `[h, vec]` instead of `h` alone (model.py), warm-started by
`convert_ckpt_dirskip.py` which appends VEC_DIM zero columns to `dir_head.0.weight`
(64 x 256 -> 64 x 309). Verified against reference outputs captured before the change: value and
h_next bit-identical, action differs by 1.9e-07 (float32 rounding in a wider matmul). Backup at
`models/nav/nav_latest_predirskip_1349.pth`.

A converter bug was made and caught here, worth recording: the first version reset Adam moments by
MATCHING TENSOR SHAPE, and throttle_head.0.weight, jump_head.0.weight, brake_head.0.weight and
value_head.0.weight are all also 64 x 256, so it wiped four heads' optimiser state as well. The
fixed version locates the parameter by its index in `model.parameters()` order and asserts the
moment shape matches before touching it. Any future resize converter must do the same.

**Honest weakening.** A linear probe from the hidden state to the goal bearing gives R^2 0.981 and
0.971, residual 0.086 on a unit vector (about 5 deg median, 10 deg at p90). So the bearing IS well
encoded, and "the network is forced to reconstruct it through a churning GRU" overstates the case:
5 deg of reconstruction wobble does not by itself explain 16.8 deg of output jitter.

The defensible statement: `dir_head` is a nonlinear function of all 256 units, not only the stable
subspace carrying the bearing. The rest churn 26 % per decision and drag the output with them.
A direct, exact bearing input gives the head a clean anchor it can weight heavily instead of
integrating over a mostly noisy state. **Well motivated, not guaranteed.**

**Synergy with TURN_COST.** The skip weights start at zero, so the head only learns to use them if
that reduces loss. TURN_COST (0.3, live) is what makes a steadier heading pay. The turn cost is the
incentive, the skip connection is the means; neither obviously works alone, which is a reason to
run them together rather than to insist on one-change-at-a-time here.

**Judge by `turn=` on the NAV lines first** (mean commanded heading change in degrees, human 0.4),
then speed, then 8 real rounds against 74.1. Pre-fix baseline: turn 38.2 falling to 34.9 under
TURN_COST alone, speed 4.7-4.8, arrive 83-88 %, update 10142.

### 9c. TURN_COST was charging the policy for its own exploration noise (2026-09-20)

Found by an adversarial multi-agent audit of the jitter (41 agents, 36 findings, 34 refuted).

`vec_worker` passed `cmd_dir=(a[0], a[1])` where `a` is `action_game`, and training runs
`model.act` with `deterministic=False`, so those elements are `d_dir.sample()`. TURN_COST was
therefore billed on the SAMPLED direction.

At |mean_xy| ~ 1.0 and action std ~0.21, two consecutive samples differ by ~17 deg from sampling
alone even if the mean never moves. Confirmed exactly by the metric after the fix:

    sampled-direction jitter   32.3 deg
    mean-direction jitter      27.5 deg
    sqrt(32.3^2 - 17^2)      = 27.5 deg

The real harm was not the wasted charge, which is roughly constant. It is that the policy could
only reduce that component by shrinking `log_std`, and the entropy controller added the same day
exists precisely to hold `log_std` up. **Two mechanisms shipped hours apart were fighting over the
same parameter.**

FIX: `act()` returns `mean_dir`; the trainer appends it to the action as elements 5-6; the worker
charges TURN_COST on that, falling back to the sample when a bare 5-vector arrives (eval paths).
`action_to_joystick` indexes `a[0..4]` explicitly, so the extra elements are inert there.

**`turn=` changed meaning with this fix.** Pre-fix readings (40.4 falling to 32.3) include ~17 deg
of sampling noise. Compare against the post-fix baseline of 27.5, or against the deterministic
37.5 deg measured from `real_run`, never across the change.

**What else the audit settled.** 34 claims were refuted with evidence, which is worth as much as
the survivors because each is a mechanism nobody needs to chase again: the Dijkstra field being a
1 u staircase that rotates the progress gradient, `dist_at` switching metrics mid-fallback,
EDGE_COST making progress reverse sign, crop and ray quantisation to the 0.5 u grid, edge-ray
channels snapping full scale, `on_floor` chattering, the gap prior flipping, banker's rounding in
the crop offsets, two-decision actuation dead time, and a frame mismatch in the conversion. Two
survived: the trace columns being post-conversion (measured effect 0.24 deg, negligible) and a
clean negative result confirming the h0 contract between acting and training is correct.

One audit measurement contradicts section 9a's framing and should be taken seriously: the GRU is
CONTRACTIVE, roughly 0.44x per step, so a perturbation is forgotten in ~3 steps. The hidden state
is not unstable. "Churn" in 9a describes the state tracking a changing input, not an instability.
The dir_head skip connection is still well motivated (the head integrates over all 256 units, most
of which move for reasons unrelated to the bearing) but the mechanism is amplification of input
variation, NOT recurrent instability.

## 10. How much is the jitter actually worth? Less than section 8 assumed (2026-09-20)

Sections 8-9c treat heading jitter as the thing blocking human speed. That was never measured
against speed directly. Doing so, at segment level across 1637 segments:

    Islands   smoothest quarter  28.5 deg -> 6.50 u/s     KOTM  29.0 deg -> 5.73 u/s
              jitteriest quarter 37.8 deg -> 6.25 u/s           37.4 deg -> 5.54 u/s
    correlation(jitter, speed)        -0.21                           -0.22

**A 9 deg difference in jitter is worth about 0.25 u/s.** Extrapolated to the full 24 deg gap to
the human's 2.67 deg, perfect steering buys roughly **0.5-0.7 u/s, about 10-12 %**. The speed gap
is 1.41x. So jitter is a real but modest contributor, not the blocker.

Caveat, stated honestly: this is within-policy variation across segments, and segments needing more
turning may be both jitterier and slower for legitimate geometric reasons, so the causal effect of
removing GRATUITOUS jitter is not pinned down by this. It is the best evidence available and it
points one way.

**A second, independent lever, found in the same analysis.** Correlation of segment speed with:

                        Islands    KOTM      mean value (Islands / KOTM)
    heading jitter       -0.20     -0.22
    input magnitude      +0.03     +0.28      0.954 / 0.881
    airborne share       +0.03     +0.12
    jump rate            +0.26     +0.29      (likely confounded: fast marbles leave the floor)
    brake rate           +0.17     -0.03

On KOTM the marble runs at **88 % throttle**, and throttle is the strongest positive predictor of
speed there. Islands is already at 95 % with no headroom, which is why its correlation vanishes.
Consistent with two known facts: `throttle_log_std` sits pinned at its clamp ceiling (-0.897
against a -0.9 maximum) so throttle carries maximum sampling noise, and the throttle head's bias is
initialised at 2.0, giving sigmoid(2.0) = 0.88. Cheap things to try: lower THROTTLE_LOG_STD_MAX so
the noise cannot sit at maximum, or raise the throttle head bias.

**Where this leaves the speed gap.** Jitter ~10-12 %, throttle maybe another ~10 % on KOTM. Neither
alone nor together reaches 1.41x. **There is currently no measured mechanism that accounts for the
speed gap**, and the honest position is to say so rather than keep tuning TURN_COST against a target
it cannot reach. The human's mid-leg peak is 10.6 u/s against the agent's 7.3 on the same physics,
and what produces that difference is not yet identified.

## 11. The speed gap is THROTTLE, and the jitter thesis is dead (2026-09-20)

### The jitter thesis, closed
Sections 8-9c pursued heading jitter as the blocker. It is not, and both halves of the gap were
tested directly:

    correlation(jitter, segment speed)          -0.21 Islands, -0.22 KOTM
      a 9 deg difference buys 0.25 u/s -> the full 24 deg gap is worth ~10-12 %
    correlation(jitter, per-leg path ratio)     +0.007 Islands, -0.106 KOTM
      the JITTERIEST quarter has the STRAIGHTEST paths (1.12x vs 1.33x)

Jitter drives neither speed nor distance. TURN_COST was reverted to 0.0 and that was correct.
(An earlier version of this analysis measured path ratio over whole multi-gem SEGMENTS and got
ratios of 3-6x; that is meaningless, since a group visiting six gems naturally travels far more
than the straight line between its endpoints. Measure per LEG.)

### What TURN_COST actually did
It worked as designed, and the design was wrong. Reaching a gem requires turning toward it, so
charging for turning made the policy turn less, and the cheapest way to turn less while still
hitting a 0.65 u target is TO GO SLOWER: a slower marble needs less heading change per decision to
track the same curve. Pickup speed fell monotonically with every raise, 5.19 -> 4.98 -> 4.82 ->
4.70 -> 4.64, and pickup speed is the floor each leg accelerates from.

### The measured chain that DOES explain the gap
Acceleration while pushing along the direction of travel, on the floor:

    speed      agent    human            input magnitude    agent accel at 4-7 u/s
     4-5       +6.40    +7.34                0.5                  +1.66
     6-7       +4.68    +6.86                0.7                  +3.07
     8-9       +3.64    +6.00                0.9                  +5.52
    10-11      +2.37    +5.00                1.0                  +6.36

Two facts together settle it. **Acceleration scales almost linearly with input magnitude**, and
**at full throttle the agent accelerates nearly as well as the human** (+6.36 against +6.86-7.34).
The agent's problem is that it rarely applies full throttle: only 26.7 % of forward-pushing
decisions are at 0.99+, median 0.955, p10 0.51. By map, throttle is 0.94 on Islands and 0.85 on
KOTM (dipping to 0.78 on approach). In Marble Blast the marble accelerates toward a target velocity
set by the input, so a lower input gives both lower acceleration at every speed AND a lower
ceiling, which is exactly the observed shape: the agent's curve dies at ~11-12 u/s, the human's is
still pulling at 14.

`action_to_joystick` normalises the direction then scales by throttle, so the measured input
magnitude IS the throttle exactly.

**A keyboard player cannot throttle down at all** and still arrives at gems at 7.37 u/s against the
agent's 5.4, so the caution is not buying precision that the task requires.

### The experiment
`THROTTLE_FLOOR` 0.15 -> 0.90 (model.py), i.e. throttle forced into [0.90, 1.0]. The floor MUST sit
above the 0.85 the policy currently chooses, or it is nullified: with any lower floor the policy
reaches its preferred throttle by lowering its sigmoid output. Modelled effect on KOTM: throttle
0.85 -> 0.93+, acceleration 4.93 -> ~6.0, about +20 %.
Revert if arrivals drop below 78 % or falls exceed 1.75: that would mean low throttle IS
load-bearing for this policy, which the human baseline says it should not be.

## 12. THE ACTION MAPPING THREW AWAY 41 % OF THE FORCE (2026-09-20)

The largest bug in the project, and the answer to the speed gap chased through sections 8-11.

The game takes PER-AXIS inputs, so the reachable input set is a SQUARE: holding forward and right
together sends a vector of length sqrt(2) = 1.414. `nav/joystick.py` normalised the policy's
direction by the 2-NORM, confining it to the INSCRIBED CIRCLE of length 1.0.

    human raw key-vector magnitude (68,208 demo decisions): median 1.414, p90 1.414, max 1.414
    76.6 % of active human frames hold TWO keys -- a perfect diagonal
    agent: capped at exactly 1.000, always

**The speed gap measured all day is 1.41x. The input magnitude ratio is 1.414.**

It explains everything that did not add up:

* The acceleration curve. Marble Blast accelerates toward a target velocity set by the input, so
  less force gives BOTH lower acceleration at every speed AND a lower ceiling -- exactly the
  measured shape (agent dies at ~11-12 u/s, human still pulling at 14).
* Why raising THROTTLE_FLOOR 0.15 -> 0.90 bought only ~4 %: throttle was never the binding
  constraint. It could only scale a vector already clipped to the unit circle.
* Why the jitter thesis failed on both halves (sections 10, 11). The marble was underpowered, not
  badly steered.
* Why TURN_COST, CARRY and BRAKE_SUPPRESS were all null: every one of them asked the policy to
  drive better when the constraint was how hard it could push.

FIX: scale by the INFINITY norm (largest absolute component). This reproduces the keyboard's
reachable set exactly -- 1.414 on a diagonal, 1.000 on an axis, never above 1.0 on either axis.
The commanded DIRECTION is identical under both scalings, so this changes only how hard the marble
is pushed, never where.

IMMEDIATE EFFECT, no training at all:

                     before      after
    applied magnitude  1.000      1.114 median (66 % above 1.05, max 1.414)
    speed              4.8        5.0-5.1
    pickup speed       4.9        5.1
    arrive             87 %       88-90 %
    falls100           1.23       1.08

**Headroom remains.** The agent sits at 1.114 rather than the human's 1.414 because it commands
continuous directions and most are not perfect diagonals. Diagonals are now worth 41 % more force
than axis-aligned moves, and until this fix that was not true, so there was nothing to learn. A
policy that biases toward diagonals can claim the rest.

**Lesson.** Every reward term tried on 2026-09-20 (CARRY, BRAKE_SUPPRESS, TURN_COST at four
settings, the entropy setpoint) was a null, and each null was correct: they were all attempts to
make the policy drive better when it was physically underpowered. The bug was in the four lines
that convert an action to a game input, a file nobody had looked at in weeks. **When many
well-reasoned interventions all return null, stop tuning and audit the interface between the agent
and the world.**

### 12a. The other half: rotate the force square with the camera (2026-09-20)

Section 12 let the agent reach the CORNERS of the input square (infinity norm, +8 points,
74.1 -> 82.1). But the square was still pinned to WORLD axes, because `format_action` defaults
`cam_yaw = 0.0` (nav/protocol.py:74) and nothing ever overrode it, so mlAgent.cs called
`setMarbleCamYaw(0)` on every decision. The agent got 1.414 only near world diagonals and 1.0 on
the axes: mean applied magnitude 1.141.

**A human rotates the square with the mouse.** Measured on the demo: yaw sweeps the full circle,
changes on 19.9 % of ticks, and the applied force averages 1.23-1.36 in EVERY 15 deg world bucket,
with frames spread evenly across directions (13.5-20.1 % per bucket). Their 76.6 % "diagonal
preference" was never a preference. They turn the camera so that wherever they want to go IS a
diagonal.

FIX. mlAgent.cs:568-573 rotates world -> camera as camera_angle = world_angle + yaw, so setting

    cam_yaw = pi/4 - phi        (phi = commanded world angle)

puts any direction on the camera diagonal, and sending world magnitude sqrt(2)*throttle makes each
camera axis exactly `throttle`, at the cap and never over. The protocol already carried the field;
only joystick.py changed. NOTE: the infinity norm from section 12 had to become a 2-NORM here --
inf-norming first and THEN scaling by sqrt(2) overshoots the cap (camera axes of 1.414). That was
caught by an assertion in the patch's own self-test, not in training.

MEASURED, immediately, no training:

                        before      after
    applied magnitude    1.141       1.409 (Islands) / 1.404 (KOTM)   +23 %
    frame offset        +0.7/-0.1   +0.9/+0.8 deg    frame INTACT
    command->accel R     0.65-0.70   0.953/0.955     control authority
    speed                5.10        5.35
    arrive               89.4 %      ~94 % (early-window, needs confirming)
    falls100             1.32        ~1.05

The concentration jump (R 0.65 -> 0.95) is the bigger story. It measures how reliably the marble
accelerates in the direction commanded. At full force the commanded push dominates gravity, slope
and existing momentum, so the marble goes where it is told rather than being deflected. Same
mechanism that halved falls in section 12.

ALWAYS RE-RUN THE FRAME CHECK after touching the action path: commanded direction vs actual
acceleration, circular mean offset must stay near 0. This project has had two frame bugs before.

## 13. RESULT: 91.8 points. The whole day was two bugs in four lines (2026-09-20)

Verified 8-round real KOTM means:

                  points   speed   gems/min   falls/100u   u per gem
    morning         62.8    5.26       -          -            -
    74.1 ckpt       74.1    5.47      19.7       0.34         16.7
    inf-norm        82.1    5.49      22.0       0.16         15.0    t = 3.80
    camera yaw      91.8    5.80      24.4       0.18         14.2    t = 3.62
    human          143.5    8.05      37.5       0.046        12.8

    camera-yaw rounds: 81, 91, 95, 97, 93, 101, 90, 86

Gap closed from 1.99x to 1.56x. Every component moved: speed, gems per minute, distance per gem.
Falls went from 7x human to under 4x.

BOTH gains were the same bug in `nav/joystick.py`, in the four lines that turn a policy action into
a game input:

1. The direction was normalised by the 2-NORM, confining the agent to the inscribed circle of an
   input space that is a SQUARE. Up to 41 % of available force unreachable. (Section 12)
2. The square was pinned to WORLD axes by `cam_yaw = 0.0`, while a human rotates it with the mouse
   and gets sqrt(2) in any direction. (Section 12a)

**Why it took all day.** Six reward and hyperparameter experiments were run first and ALL returned
null: CARRY at two calibrations, BRAKE_SUPPRESS, TURN_COST at four settings, the entropy setpoint,
THROTTLE_FLOOR. Every null was CORRECT. The agent was underpowered by 41 % and being deflected off
its commanded heading (command->acceleration concentration 0.65, now 0.95). No incentive fixes
that.

**THE RULE, which is the generalisable output of the day:** when several well-reasoned
interventions all return null, stop tuning and audit the interface between the agent and the world.
The reward, the architecture and the hyperparameters were all fine. The action conversion was not,
and nobody had read it in weeks.

Corollary habits worth keeping:
* Measure the exact quantity a term pays for, not a related one (the "human carry 7.4 u/s" error in
  section 7 was their SPEED read as their ALIGNED speed, a factor of cos(72 deg)).
* Judge by 8 real rounds, never the training curve, which disagreed four separate times.
* Check the MECHANISM metric, not just the outcome: brake share answered in 20 minutes what falls100
  could not answer in 90.
* Re-run the frame check after ANY change to the action path.

REMAINING. 1.56x, mostly speed (5.80 vs 8.05). There is currently NO measured mechanism for it: at
equal force the agent now OUT-accelerates the human below 6 u/s. Open leads: `THROTTLE_FLOOR = 0.90`
(raised today, pre-fix justification now superseded) denies the zero-input coasting a human uses on
9.9 % of ticks; and the 64 ms decision rate against a human's 16 ms, which the interface audit
bounded at ~5 % of impulse fidelity but did not settle for gem-approach precision.

## 14. The camera must rotate, so the VIEW was split off it in the engine (2026-09-20)

THE COMPLAINT. Section 12a's camera yaw is what earned +9.7 points, and it swings the camera about
15 times a second, because it tracks a commanded direction that moves ~38 deg per decision. Nobody
can watch that. The camera had been pinned to one direction for the whole project.

WHAT WAS TRIED BEFORE THE ENGINE CHANGE, and what each measured.

1. Rate-limit the yaw so it pans like a mouse (`patch_yaw_rate.py`, NAV_YAW_RATE deg/decision).
   Simulated against the measured jitter: at 3 deg/decision the mean applied force is 1.125, which
   is LOWER than the 1.141 a fully locked world-aligned square already gives. Any rate slow enough
   to be watchable lags so far behind the command that the square is worse than not rotating it.
   Removed at the user's request.
2. Give the renderer its own camera object and leave the marble's camera to the physics. The client
   control object was swapped; the marble FROZE. Local marble physics runs on the CLIENT control
   object, so anything that takes control away from it stops the simulation. `getCameraObject()`
   returns mControlObject with the camera-object branch commented out, and
   `ShapeBase::setControlObject` is an empty body, so there is no script-side seam here. Reverted.
3. Send a desired velocity vector instead of keys ("the game supports joysticks"). The move is
   packed per axis and each axis is clamped to +/-1 and quantised to 1/16 in
   `clampRangeClamp` (gameConnectionMoves.cc:98-104). The reachable set IS the square; there is no
   magnitude channel to send. A joystick reaches exactly what two keys reach and no more.

So the force genuinely requires the square to rotate, and the square is the camera. What does NOT
follow is that the VIEW has to rotate: the same matrix is used for movement and for rendering only
because nothing ever needed them apart.

THE ENGINE CHANGE (worktree C:/Users/doug/src/OpenPQ-TGEMIT-mbx, branch `ai-training-mode`).
`Marble` gains `F32 mViewYaw; bool mUseViewYaw;` (marble.h, after mNextCameraPitch). In
`Marble::getCameraTransform`, when mUseViewYaw is set, the camera offset is rebuilt from
gravity * yaw(mViewYaw) * pitch(mCameraPitch) and used for the view basis, the eye direction and
the collision probe, while `mCameraOffset` (and therefore `getMarbleAxis`, and therefore every
force the marble applies) is untouched. Console methods `Marble::setViewYaw(F32)` and
`Marble::clearViewYaw()`. Script side: mlAgent.cs gains a `VIEWYAW <rad|off>` control command that
applies it once per marble; nav/real_run.py sends it in WATCH mode from NAV_VIEWYAW (default 0).
This is render-only by construction, and the numbers below confirm it.

VERIFIED PHYSICS-NEUTRAL. Real round, fixed view at yaw 0, one 64 ms sim step per decision, same
checkpoint (nav_stopped_20260920_2145, update 11615), 2548 decisions:

    speed bucket   this run   training (camyaw)   human
      4-5 u/s       +13.82         +14.04         +7.34
      5-6           +13.13         +13.52         +7.34
      6-7           +12.16         +12.61         +6.86
      7-8           +11.31         +11.56         +6.80
      8-9           +10.42         +10.58         +6.00
      9-10           +9.71          +9.65         +5.39

    frame offset +0.13 deg, concentration 0.968 (training 0.95)
    applied magnitude median 1.390, p90 1.414 (the square corner)
    round: 81.0 points in 2.72 min = 29.8 points/min (human 47.3), 64 gems, 3 falls

## 14a. TRAP: watch-mode sub-steps cost 38 % of the acceleration

NAV_VIEW_SUBSTEPS splits each 64 ms decision into N sim slices so the viewer runs at ~62 fps
instead of 15.6. It is NOT physics-neutral, and the first fixed-view run used N=4, which is why
that run looked slow to the eye and measured slow:

                         SUBSTEPS=4     SUBSTEPS=1     training
    accel 4-7 u/s          +8.11          +13.09      +12.6..14.0
    frame concentration     0.548           0.968          0.95

The concentration collapse says the force is landing in the wrong DIRECTION, not that the marble
is weaker: the camera yaw is set once per decision, so with 4 ticks per decision only one of them
gets a fresh value and the other three run against whatever the ghost update leaves behind.

RULES. Never measure anything from a WATCH run with SUBSTEPS > 1. Viewing at 1x means 15.6 fps and
choppy, which is the correct trade. An untested question this raises: the human demo was recorded
at the game's own 16 ms tick, and the agent's ~1.9x acceleration advantage is measured at 64 ms, so
part of that advantage may be a large-step integration artifact rather than better play. Testing it
needs the camera yaw re-applied per slice so alignment is not the confound.

## 15. Between groups the marble drove back to the spot it had just cleared (2026-09-20)

WHAT THE USER SAW at 1x: "the marble went right through the waypoint, didn't pick it up, and had
to go back for it". The trace (logs/nav/real_trace.csv of that run) says the pickup DID register:
row 188 has gem=2, the last gem of its group, and nvis drops to 0 on row 189. Rows 189-208 show
the marble reversing, driving back to (-37.2, 3.0), the gem it had just collected, and hovering
there until the next group spawned on row 209. Nothing was missed. The target was not stale
either: choose() returned None the instant the gem vanished. The fault was the FALLBACK for
"no gem on the map", which steered at the NEAREST real spawn point. The nearest spawn point to a
marble that has just cleared a gem is that gem's own spawn point.

SIZE. 5316 decisions, 2.x rounds: 33 gaps, all 33 immediately after a pickup, length median 18
decisions (1.15 s), mean 16.5, max 31; 10.3 % of every decision in the run (34.9 s). The marble's
speed at the end of a gap was 3.49 u/s against a run mean of 5.96, and the new group landed a
median 14.7 u away (min 3.4, p10 10.3, max 20.0). This is the `blind_pct` field that every REAL
RUN line has been printing at 9-10 % all along.

WHY THE NEXT GROUP IS FAR. huntGems.cs getCenterGems (single-player branch): block a radius
spawnBlock = 2 * radiusFromGem round $Game::LastGemSpawner, the last group's centre. KOTM sets
radiusFromGem = 15 and maxGemsPerSpawn = 4, so the block is 30 u on a map 24 u across; nothing
qualifies and the code takes the FURTHEST of its 10 sampled gems as the next centre. The next
group is on the far side of the map by design. KOTM has 12 spawn points: a 4-point cluster round
the centre (-27.2/-23.2, 13/17) and 8 on the rim; the centre cell itself (-25.2, 15) is a 2x2
unwalkable spot.

WHICH HEADING TO TAKE, tested on the 33 gaps against where the group actually appeared:
    marble's own velocity at the pickup (keep rolling)      mean error 92.9 deg   (useless)
    centroid of the spawn points beyond R of the marble     13.3-13.6 deg for R = 0..30
    5 u toward that centroid gains 4.4 u on the new target (of a 14.7 u median)

THE FIX (nav/real_run.py, gap branch). On a gap: centroid of the spawn pool excluding points within
GAP_EXCLUDE_U = 6 of the marble (the cleared group), snapped to the nearest walkable cell with
snap_to_walkable, falling back to the spawn point nearest the centroid only if no floor is within
3 cells. HELD until gems reappear (one Dijkstra field per gap). Self-test: from all 12 KOTM spawn
points the gap goal is walkable; from the rim it is 13-18.5 u toward the centre, from the centre
cluster it is the floor beside the centre 1.6-4.3 u away, which is right because from there every
rim point is equidistant. Transfers: needs only the map's spawn list, which every trained map has
(terrain.gem_spawns).

NOT a training change. The policy sees an ordinary goal; the training env has no gaps (a new
waypoint appears on arrival) and needs none.

RESULT: 97.5 POINTS. 8 rounds on nav_camyaw_20260920_2015, the 91.8 checkpoint, so nothing but
the gap goal changed: 91,102,94,101,98,99,99,96 against 81,91,95,97,93,101,90,86. +5.8, Welch
t = 2.23, significant. Mechanism on the eval trace, 148 gaps: speed when the next group appears
5.59 u/s (was 3.49), distance to it 11.9 u (was 14.9), ended closer than it started in 124 of 148.
Round speed 6.02 (was 5.80), falls 0.17/100u, blind_pct unchanged at 9.7 % because the gap length
is the game's, not ours. Gap to human 143.5: 1.47x.

## 15a. Routing waypoints sat 0.71 u beside gems (2026-09-20)

USER: "at least 3 waypoints were less than one unit from a gem but not in it." Correct. The
training pool is exact (12 pool points, 12 game-reported gem positions, 0.00 u apart, every pool
point seen live), so this was the REAL-RUN routing. path_waypoint returns walk-cell centres, and
KOTM's gems sit on cell corners: (-27.2, 13.0) against centres at x.7 / x.5. When the straight
line to the gem clips a non-walkable cell (the 2x2 hole at the map centre is 2.8 u from the four
centre gems) the string-pull handed over the last path cell, 0.71 u from the gem. The watched run
steered at (-27.7, 12.5) and (-26.7, 12.5), both 0.71 u from (-27.2, 13.0). The marker made it
look worse: it was only re-placed when the goal moved > 1.0 u, so when the goal became the gem
the marker stayed beside it.

FIX (nav/real_run.py): a string-pulled pick within GEM_SNAP_U = 1.5 u of the gem returns the gem
itself; marker re-placed at 0.25 u. Replay of the watched run, 741 decisions with a gem on the
map: at the gem 535 -> 594, beside it 59 -> 0, genuine routing corners farther out 147.

## 16. Gems-only steering, and the fall mechanism it exposed (2026-09-20 night)

DECISION (user, 23:50): the waypoint handed to the policy is the gem's exact position, always.
nav/real_run.py DIRECT_GEM is now default ON: no routing corner points, no snap_to_walkable. The
marker is hidden while no gem is on the map ("MARK off", mlAgent.cs); the between-groups steering
toward the spawn centroid (section 15) is kept but never drawn. Rationale: routing was a planner
bolted onto eval on 2026-09-19 that training never had, so every real-round score since then was
policy + planner. The honest policy-only number is below.

COST, 8 rounds on nav_camyaw_20260920_2015 (the 97.5 routed checkpoint): 82,76,92,88,77,95,92,83
= 85.6, -11.9, t = -4.16. Speed UP 6.02 -> 6.69 u/s; falls 0.17 -> 0.65 per 100 u (15 -> 63 in 8
rounds); stalling only 2.5 % of decisions. 49 of the 63 falls came within 0.5 s of the straight
line to the target crossing a hole (base rate 30 %). Training's own KOTM falls100 with direct goals
had been 0.64-0.78 all along; routing was hiding it.

THE MECHANISM, 10,225 KOTM training falls (trace_20260920_154049.csv):
    line to the goal crossing a hole 0.5 s before     89 %   (base rate 36 %)
    a takeoff within 1 s before                       13 %   (87 % simply rolled off)
    speed 0.5 s before                                median 6.35 (run mean 6.22): not a speed spike
    command vs velocity angle 0.5 s before            median 81 deg: turning, and too late
At 6.3 u/s with ~13 u/s^2 the marble needs ~3 u to turn 90 deg. FALL = 10 is one sparse hit per
~312 decisions, and falls100 was RISING (0.64 -> 0.73) under it between 21:17 and 21:47.

EXPERIMENT 1, 23:43: edge-time shaping (nav/waypoints.py edge_time_cost). Every floor decision,
ray-march the walk grid along the velocity for the first non-walkable cell within EDGE_LOOK = 6 u;
t = distance / speed; charge EDGE_K * (1 - t / EDGE_T_SAFE) with EDGE_K = 0.5, EDGE_T_SAFE = 0.6 s,
nothing below EDGE_V_MIN = 1.5 u/s. A run-up at a JUMPABLE gap (walkable landing within JUMP_GAP
past the edge, height within the walk graph's jump limits) is exempt, so Islands jumping is not
taxed; KOTM's ~7 u quadrant holes are charged, its 2 u centre hole is exempt. Snapshot
nav_before_edge_2341.pth (update 11615). Measured on the first 20 minutes of trace: charged on
6.1 % of KOTM floor decisions, mean 0.014 per decision (~4.5 per fall interval against the flat
10), recall 76 % (a charge in the second before three quarters of falls), precision 13 % (it taxes
risky approaches whether or not they end in a fall, which is the intent).
Eval tooling: eval_cycle.ps1 stops loop/trainer/workers/games, snapshots nav_latest, runs 8
gems-only rounds, restarts training, and appends the result to overnight_notes.txt.

## 17. Training paid the marble to fall (2026-09-21, 01:40)

Three reward changes aimed at falls (edge-time shaping v1 at K 0.5 and 1.5, v2 with an earlier
start and a pickup exemption, then FALL 10 -> 25) left KOTM falls100 at 0.69-0.79 across ~400
updates. The reason is in SegmentManager.recover(): after a mid-group fall the game respawns the
marble at its own rim pads (a median 16.9 u from the fall point, measured on the real rounds),
recover() then set prev_d to the path distance AT THE RESPAWN POINT, and every decision of the trip
back to the gem earned PROGRESS as new ground, ~+17 per fall. The respawn wait runs outside the
rollout, so no TIME was charged either.

    training reward for a fall, FALL = 10:   -10 + ~17 - ~2   = ~ +5    (a fall PAID)
    FALL = 25:                               -25 + 17 - 2     = ~ -10
    real round (section 16 decomposition):   ~7.5 s lost      = ~55-60 units of forgone gems

FIX. s.fall_mark = the path distance at the last on-map decision before the fall. After the
respawn, progress is 0 while d >= fall_mark and resumes (two-sided, as before) once the marble is
back inside it; the mark clears on retarget. A fall now nets about -(FALL + TIME of the trip
back) = ~-30 at FALL 25, still under the real ~55 but no longer a source of reward.
Snapshot before: nav_before_fallmark_0133.pth (update ~12050). Edge v2 and FALL 25 kept.

Real-round fall anatomy for reference (63 falls): fall flag to rolling again median 3.3 s, then a
median 16.9 u back to the group; 27 % of all round time was inside legs containing a fall.

## 18. "It drives straight into the centre hole": the reward pays it to (2026-09-21)

USER, after watching at 1x: "the marble simply doesn't know the centre hole exists, it drove
straight through and fell without trying to route around it."

IT CAN SEE IT. The centre hole is a 2x2 u void at (-25.2, 15.0), NaN in the height stack and
non-walkable in the walk grid. Built the actual fine crop the policy reads at (-29, 15) and
(-27, 15): the hole is a 4x4 block of empty cells dead ahead, 446 and 550 void cells of 1024.
Perception is not the problem.

THE ROUTE IS RIGHT TOO. The Dijkstra field goes around, north via y~16.5, and prices the crossing
at 8.90 u against 7.00 u for the detour.

THE SIGNAL ARRIVES TOO LATE. PROGRESS is paid on geodesic distance, and a shortest-path field is
built for a point mass that turns instantly. Driving due east at the hole:

    x = -30.2  d = 18.14  PROGRESS +1.00
    x = -29.2  d = 17.14  PROGRESS +1.00
    x = -28.2  d = 16.14  PROGRESS +1.00
    x = -27.2  d = 14.90  PROGRESS +1.24      <- the lip is ~0.5 u further on

Full reward up to the lip, then an instantaneous 90 degree turn is required. At 6-7 u/s with
~13 u/s^2 the marble needs ~3 u to turn 90 degrees. Driving in is locally optimal, so training
alone converges to it. The centre hole is 20 % of KOTM falls in trace_20260921_074134 (9 of 44);
the quadrant holes are the other 80 % and fail identically.

TRIED AND REJECTED: costmap inflation. Give every walkable cell a routing multiplier of
EDGE_INFLATE_K at a lip decaying to 1.0 over EDGE_INFLATE_U = 3 u, so the field bends away early.
DEGENERATE ON THIS GEOMETRY: KOTM's walkways are ~3 u wide, so distance-to-nearest-lip is 0.00 for
essentially every walkable cell; the multiplier became a constant, every edge scaled by the same
factor and no gradient DIRECTION changed (measured: PROGRESS went 1.00 -> 2.33 per unit step with
the identical route). Reverted. The same objection applies to EDGE_COST = 2.0, which is the 1-cell
version of the same idea and is near-constant here for the same reason. Any future attempt must be
checked against `terrain.edge_dist` (kept, as a measurement) before it is trusted.

WHAT IS IN PLACE INSTEAD: EDGE_K = 1.0, the time-to-void charge (section 14a/v2), which fires from
EDGE_T_SAFE = 1.2 s out (~8 u at cruising speed) and is exempt when the goal itself lies before the
drop. It was zeroed at 07:50 as an ablation that never ran, and restored at 09:05.

ALSO ANSWERED: "does it just need more training?" Before 01:45 no, because falling was profitable
(section 17). After the fix, six hours of ordinary training took KOTM falls100 0.73 -> 0.42 and it
was still falling when training stopped. More training is now productive; the timing defect above
is what caps it.
