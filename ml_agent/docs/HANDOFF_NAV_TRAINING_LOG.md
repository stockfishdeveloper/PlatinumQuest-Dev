# Navigator training: the dated log (2026-09-17 to 2026-09-26)

HISTORY, newest last. Every experiment, measurement and decision, with its date. Anything here can be
superseded by a later section; the CURRENT state is in `../HANDOFF_NAV_TRAINING.md`. Code comments that
cite "HANDOFF 28.52" and the like refer to the section numbers in this file (unchanged by the move).
Until 2026-09-26 this was HANDOFF_NAV_TRAINING.md itself; its old "CURRENT STATE" block (2026-09-21) was
removed when it was split (git history has it).

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

## 19. The policy was not failing to optimise: the reward did not pay for speed (2026-09-21)

After adding FlatGemTraining to the rotation (4 FlatGem / 3 KOTM / 1 Islands, 11:07) speed there
went 7.30 -> 8.90 u/s in 50 minutes and then PLATEAUED for five consecutive checks, short of the
0.95-human target of 9.42. An 8-round eval on that map gave 80.75 gems, speed between pickups
8.54 u/s against the human's 9.92 recorded on the same map that morning
(`demos/demo_20260921_105710.npz`, 102 pickups, frame check 0.885).

WHAT THE APPROACH PROFILE SHOWED (`plot_approach_profile.py`, 331 agent legs vs 101 human legs,
median speed binned by distance-still-to-travel):

    dist to gem   human   agent     gap
      20 u         9.72   10.57   agent faster
      15 u         8.48    9.08   agent faster
      12.5 u      12.13   10.23   human +1.91
       8.5 u      14.41   10.29   human +4.12
       6.5 u      14.23   10.12   human +4.11
       4.5 u      13.39    9.72   human +3.67
       2.5 u      11.36    9.35   human +2.01
       0.5 u       8.72    9.09   agent faster

The agent's profile is FLAT (10.9 far out, 10.3 mid, 9.1 at the gem). The human's is an arc: they
sprint to 14.4 in the middle of a leg and brake down to 8.7 at contact, braking on 23 % of ticks
against the agent's 0.1 %. The loss is entirely mid-leg, 4-13 u out.

RULED OUT, both measured rather than assumed:
* "the two-key diagonal is broken": applied key-vector magnitude median 1.409, p90 1.415, at the
  1.414 corner on 77 % of active decisions, identical to the human's 77 % two-key share. Agent max
  speed 16.55 u/s. The human/agent p99 ratio is 1.275, not the 1.414 a lost axis would give.
* "heading jitter is costing speed": command swing CORRELATES POSITIVELY with speed (6.92 u/s at
  0-5 deg of swing, 10.58 at 60-120), because the bearing to a gem changes faster when moving
  faster. Not a cause.

THE ACTUAL CAUSE, arithmetic on the reward for the median 21.4 u approach:

    98 % of a leg's reward is SPEED-INDEPENDENT.
    PROGRESS pays 1.0 per unit closed, so it totals 21.4 however long the leg takes; ARRIVE is a
    flat 10. Only ~0.7 of 32.1 responds to speed at all.
    Going 19 % faster (8.43 -> 10.0) is worth +1.14 = +3.5 %.
    One overshoot (3 u past the gem and back) costs ~2.15.
    => one overshoot every 1.9 legs erases the entire benefit of going faster.

The cautious cruise is the higher-expected-value policy under this reward, and PPO converged to it
correctly. The human's sprint-and-brake is better in GEMS PER MINUTE, which is what we actually
want, but roughly break-even under what we were optimising.

THE FIX (12:47), a rebalance rather than a new term, and explicitly NOT copying the human's curve
(that curve is a property of this map's geometry; pricing time correctly lets the policy find
whichever curve is optimal on any map):

    PROGRESS  1.0  -> 0.3      keeps the scaffold that makes the task learnable, removes its 66 %
                               dominance of the signal
    TIME      0.05 -> 0.20     the real opportunity cost of a decision (~16 gems/min at ~13 reward
                               a gem is ~0.2 per decision)

Effect on the same leg: +3.5 % becomes +18.5 % for the same 19 % speed gain, and 14 u/s is worth
+47 % against the plateau instead of +9 %. A slow 6 u/s leg stays net positive, so the policy is
never paid to abandon one.

NOTE ON THE 2026-09-19 FAILURE: TIME was raised to 0.12 then and made KOTM speed WORSE, concluding
it was "a weak lever on pace". That was correct at the time and does not apply now: with PROGRESS
at 1.0 a uniform per-decision cost shifted the value baseline without changing the ordering of fast
and slow legs. The lever only bites once the speed-independent bulk is cut, which is why the two
constants move together as one change.

Snapshot before: `nav_before_rewardrate_1247.pth`.

RESULT: REVERTED at 13:26 after ~100 updates. FlatGem speed moved 8.90 -> 9.00 (+0.1), while
FlatIslands broke: arrivals 97 -> 84 %, falls100 1.08 -> 1.42, a monotone slide over 13 minutes
while KOTM held flat at 96 %. Checkpoint restored from the snapshot, constants back to
PROGRESS 1.0 / TIME 0.05.

WHY IT BROKE ISLANDS, and the constraint any retry must respect: TIME is charged per DECISION
including airborne ones, and crossing a gap is unavoidably airborne. At 0.20 a jump costs 4x what
it did, on top of JUMP_TAKEOFF 0.4 and AIR 0.1, so the policy stopped jumping and started missing
gaps. Islands is the only map with meaningful jump edges (1708 against KOTM's 228), so it took the
damage alone. A correct version of this change must exempt or discount airborne decisions from the
time cost, otherwise pricing time taxes gap-crossing out of existence.

WHAT SURVIVES: the diagnosis in this section is unaffected and still needs an answer. 98 % of a
leg's reward is speed-independent, and going 19 % faster is worth +3.5 % against an overshoot cost
of 2x that. The measurement was right; this particular fix was not.

ALSO RULED OUT while investigating the speed gap (both measured, not assumed):
* powerups: the human used one in 4.8 min, 0.7 % of ticks; their speed excluding it is 9.98
  against 10.00 overall.
* command persistence as the mechanism: the human holds a direction within 30 deg for a median of
  4 decisions against the agent's 1, but at MATCHED persistence the agent is still ~2.7 u/s slower
  in every bucket (1-2 held: 12.98 vs 10.04; 9+ held: 12.72 vs 10.28), and likewise at matched
  force-vs-travel alignment. Neither explains the gap; it remains open.

## 20. FlatGemTraining: the agent uses proportional control, the human uses bang-bang (2026-09-21)

The flat map strips out holes so the pace problem can be studied alone. Agent 15.7 gems/min against
a human 21.5 recorded on the same map the same day (demos/demo_20260921_105710.npz, 102 pickups,
frame check 0.885 against a mirrored 0.008).

WHERE THE TIME GOES. Splitting a round by whether a gem is even on the map:

                            agent     human
    waiting for a spawn     1.02 s    1.01 s   per gem  <- IDENTICAL, not the problem
    chasing a visible gem   2.81 s    1.79 s   per gem  <- the entire gap
    total                   3.83 s    2.80 s

Match the human on chasing alone and the agent scores 21.4 gems/min, i.e. human parity. The spawn
delay is the game's and both pay it equally.

FACTORISING THE 1.44x CHASE GAP (236 agent chases, 97 human):

    speed                          9.78 vs 11.74 u/s   1.20x
    path ratio (travelled/straight) 1.08 vs 1.03       1.04x   <- NOT route curvature
    straight-line distance to the
    gem chosen                     24.8 vs 21.2 u      1.17x   <- worse position when it spawns

Note 225 of 236 agent chases (and 96 of 97 human) are the FIRST gem after a blind gap: on this map
gems are collected one at a time, so there is no multi-gem ordering to get wrong and no planner to
build. The distance factor is about where the marble is standing when the gem appears.

THE MECHANISM, and it is the cleanest result of the day. Alignment of the commanded direction with
the direction of travel, during a chase:

    cos bucket                      agent   human
    < -0.5   hard braking            21 %    33 %
    -0.5..0  mostly against          10 %     3 %
    0..0.5   sideways                12 %     3 %
    0.5..0.8 partly pushing          13 %     5 %
    0.8..0.95 mostly pushing         16 %    13 %
    > 0.95   PURE ACCELERATION       27 %    43 %
    mushy middle (0..0.8)            26 %     9 %
    decisive (<0 or >0.95)           59 %    78 %

The human is either flooring it or hard on the brakes 78 % of the time. The agent spends a quarter
of every chase applying partial thrust at an intermediate angle, which neither accelerates nor
turns efficiently. For minimum-time arrival with bounded acceleration the optimal control IS
bang-bang; the human is doing textbook time-optimal control and the agent is doing proportional
control, which is smooth and slower.

This ties together what survived the day:
* the approach profile (section 19): the human sprints to 14.4 u/s mid-leg then brakes to 8.7,
  which is bang-bang. The agent's flat ~10 u/s cruise is proportional control.
* why smoothing failed: NAV_SMOOTH 0.3 cut command swing 38 deg -> 10.6 but the ALIGNED share FELL
  27 % -> 21 % and speed did not move. Smoothing removes command changes, not the mushy middle.
* CORRECTION (2026-09-21, user challenged this and was right): the claim "human brakes 23 %,
  agent 0.1 %" was a CATEGORY ERROR and is withdrawn. The 23 % comes from the demo recorder, whose
  own docstring says "brake is DERIVED: 1 when the intent opposes the velocity" -- it is a physics
  measure, not a key. The 0.1 % is the agent's DISCRETE BRAKE FLAG, a different mechanism. Measured
  like for like, the agent brakes by commanding against its own momentum at close to the human rate:

      force against momentum      agent   human
      within 25 deg of opposite   13.9 %  17.0 %
      within 45 deg of opposite   22.2 %  26.2 %
      within 60 deg of opposite   26.6 %  30.9 %
      any component against       36.4 %  37.4 %

  So the agent CAN and DOES decelerate, and "it cannot do the second half of bang-bang" is false.
  The candidate fix that followed from it (give the discrete brake head more exploration) is
  withdrawn with it.

WHAT IS RULED OUT for the speed gap, all by measurement:
* input magnitude: agent median 1.409 of a 1.414 maximum, at the square's corner on 77 % of active
  decisions, the same share as the human's two-key holds.
* impulse vs hold: a 64 ms decision reaches 99.6 % of what four 16 ms ticks reach
  (step_size_probe.py); single-axis target 15.00 u/s at every step size.
* effective target velocity: human 21.84 u/s, matching the probe's 21.8 exactly; agent 20.43
  (94 %), with throttle pinned at 0.99. Only 6 % is lost at the input.
* powerups: the human used one in 4.8 min, 0.7 % of ticks; their speed without it is 9.98 vs 10.00.
* heading jitter, command persistence, frame error, route curvature: all measured and refuted
  (sections 19 and above).

WHAT SURVIVES THE CORRECTION. Braking is NOT the difference. The difference is at the other end
of the distribution and in the middle:

      during a chase          agent   human
      PURE ACCELERATION       27 %    43 %    <- the agent accelerates decisively far less often
      mushy middle (0..0.8)   26 %     9 %    <- and dithers at partial thrust far more
      braking (cos < -0.5)    21 %    33 %    <- real but smaller, and the whole-trace figures
                                                 (26.6 vs 30.9) are closer still

So the agent's deficit is that it spends a quarter of each chase applying partial thrust at an
intermediate angle instead of committing to full acceleration. It decelerates about as often as
the human; it ACCELERATES decisively much less often.

NO CANDIDATE FIX IS PROPOSED HERE. Three reward changes failed today (time pricing, PBRS,
smoothing) and two mechanism theories were withdrawn after measurement (dead brake action, pure
pursuit). Whatever comes next should start from why a policy at 13,000 updates sits at partial
thrust when full thrust in the same direction is available and better, which is not yet answered.

## 21. THE FLAT SPEED CURVE IS AIM SCATTER (2026-09-21, FlatGemTraining)

The question: the human's speed-vs-distance profile is a smooth arc peaking at 14.9 u/s around
7-9 u from the gem; the agent's is flat at 8.4-11.0 across the whole approach. That difference is
the entire 85 vs 104 gem gap on this map.

Second human recording taken deliberately WITHOUT JUMPING (demo_20260921_155130.npz, 18,744 ticks,
104 gems, 0 % jump, frame check 0.897): jumping is NOT the lever on this map, and the arc shape
reproduced exactly, so it is not a stylistic artefact of the first run.

THE MEASUREMENT. Decompose the commanded direction against the direction to the gem, signed, so
systematic aiming error can be told apart from scatter:

    dist     AGENT bias  conc.   HUMAN bias  conc.
    21 u        +8.4     0.736      +0.9    0.997
    17 u        +7.1     0.615      +1.0    0.993
    13 u        -3.0     0.604      +2.5    0.939
    11 u        +1.6     0.487      -2.0    0.941
    10 u        +7.4     0.412      -1.0    0.891
     8 u        +3.8     0.263     -10.0    0.499
     4 u        +0.3     0.138    -163.5    0.552   <- human braking, a CLEAN reversal

BIAS IS NEAR ZERO FOR BOTH: the agent does aim at the gem on average. The difference is entirely
CONCENTRATION. The human holds within a few degrees from 22 u down to ~8 u (0.93-0.99), then flips
deliberately to -160..-170 deg to brake. The agent's command oscillates around the correct
direction with concentration 0.74 far out, falling to 0.41 at 10 u and 0.14 at 4 u. Concentration
0.5 is an angular spread of about 68 deg.

WHY THAT FLATTENS THE CURVE. Force at angle theta contributes cos(theta) toward the gem, so the
USEFUL fraction of thrust IS the concentration:

    useful thrust fraction, 8-22 u:  agent 0.57   human 0.91   = 1.60x
    observed mid-leg speed (7-12 u): 10.4 vs 14.5              = 1.39x
    observed peak speed:             11.05 vs 14.94            = 1.35x

A 1.60x force advantage yielding 1.35-1.39x speed is the right relationship for a saturating
system. The agent has full magnitude (1.40 of 1.414) and correct average aim, and still only half
the useful thrust, because the other half cancels itself out.

THIS IS THE POLICY'S MEAN OUTPUT, not exploration: real_run acts with deterministic=True.

IT EXPLAINS THE FAILED EXPERIMENTS:
* forced throttle 100 % (section 20 follow-up): +5 % gems, and the alignment distribution did not
  move at all. Full magnitude scattered over +-68 deg still averages to half.
* NAV_SMOOTH: raised concentration only 15 % (0.502 -> 0.579 over 10-20 u) against the 81 % needed
  to reach the human's 0.91, while costing 14 % of gems to lag. It barely touched the quantity that
  matters, so it is NOT evidence against this explanation.
* time pricing, PBRS: neither addresses aim steadiness at all.

CAUTION BEFORE ACTING. Section 9 traced heading jitter to the RECURRENT STATE rather than the
input, and section 11 then declared the jitter thesis dead on the grounds that throttle was the
real constraint. That verdict now looks wrong, but sections 8-11 contain measurements that any new
attempt must be reconciled with rather than ignored. Read them first.

### 21a. WHY the aim scatters: the dir_head skip connection was switched off by training

Measured on nav_eval_flatgem_1208.pth.

MECHANISM.
    dir_head first layer, weight norm on the HIDDEN state (256) : 9.810  (0.613 per unit)
    dir_head first layer, weight norm on the OBS VECTOR (53)    : 1.320  (0.181 per unit)
    weight norm on vec[0:2], the exact smooth GOAL UNIT VECTOR  : 0.333

    output sensitivity, median over 300 random states:
      hidden state perturbed by one step's worth  ->  8.6 deg of commanded-direction change
      goal direction rotated by 5 deg             ->  0.1 deg
      => the hidden state moves the output 65x more than the goal bearing does.

    versus a FRESH initialisation (no weight decay in the optimiser):
      hidden columns    0.0285 -> 0.0598   GREW 2.10x
      vec columns       0.0288 -> 0.0157   SHRANK to 0.55x
      vec[0:2] goal     0.0279 -> 0.0231   SHRANK to 0.83x, BELOW random init

    synthetic constant-input approach (gem dead ahead, 10 u/s, 60 decisions; the observation is
    off-distribution so absolute angles are meaningless, but the contrast is not):
      hidden evolving : the command sweeps +41 -> -129 deg on a CONSTANT goal bearing
      hidden zeroed   : flat at -85/-86 deg for all 60 decisions
    All of the output variation is generated inside the network, none by the input.

WHY TRAINING DID THAT. The skip was added 2026-09-20 (section 9a) to a policy with ~11,000 updates
that had ALREADY learned to compute direction from the hidden state, and whose MEAN direction was
already correct (measured bias today: +1 to +8 deg, the same as the human's). So:

  1. The skip was redundant for the MEAN. Adding its contribution on top of an already-correct mean
     overshoots the target, which is a positive gradient reason to shrink those weights. The
     observed 0.83x is consistent with exactly that.
  2. The only thing the skip could improve is VARIANCE, and variance is nearly free: aim scatter
     costs ~1.6x useful thrust -> ~1.35x speed, but 98 % of a leg's reward is speed-independent
     (section 19), so a perfectly steady aim is worth a few percent of return.
  3. A few percent of return cannot rewire a head that already works.

So the policy did not learn jitter as a strategy. It learned a direction function that is right on
average and noisy, and nothing in the objective pays enough to clean it up. The fix that was
supposed to address this was added in a form the gradient had every reason to switch off.

RECONCILING WITH SECTIONS 10 AND 11. Section 10 estimated jitter at 10-12 % of speed by correlating
jitter against speed ACROSS SEGMENTS WITHIN one policy (9 deg of difference -> 0.25 u/s) and
extrapolating linearly to the human's 2.67 deg; its own caveat says the causal effect "is not
pinned down by this". The geometric measurement in section 21 is not a correlation: useful thrust
IS the cosine of the aim error, 0.57 vs 0.91, and that 1.60x predicts the observed 1.35-1.39x speed
ratio. Section 11 then concluded throttle was the real constraint on the basis that KOTM ran at
88 % throttle; that is now falsified directly (throttle measures 0.99, and pegging it to 100 %
moved the alignment distribution not at all).

## 21b. RESULT of DIR_GOAL_GAIN = 30: real, and smaller than the training curve implied (2026-09-21)

Section 21a's fix (amplify the goal bearing 30x on the way into `dir_head`, `NAV_DIR_GOAL_GAIN`
in `nav/model.py`) trained from 16:36 to 19:34, about 500 updates. It is the first change of the
day to survive an 8-round evaluation, and it is also a lesson in how far training metrics can
diverge from real score.

| | training arrive | training speed | 8-round real |
|---|---|---|---|
| FlatGem before | 99 % | 8.50 | 80.75 gems |
| FlatGem after | 100 % | 9.80 (+15 %) | 84.9 gems (+5 %) |
| KOTM before | 96 % | 5.40 | 93.5 pts |
| KOTM after | 96 % | 6.00 (+11 %) | 95.9 pts (+2.6 %) |
| Islands before | 92 % | 4.80 | never evaluated |
| Islands after | 81 % | 5.00 | never evaluated |

Aim concentration went 0.46 -> 0.70-0.74 (human 0.91) and turning share above 10 u/s went 54 %
-> 39-41 % (human 42 %), so the mechanism did exactly what section 21a predicted. Islands paid
for it, losing 11 points of training arrivals, and that cost is still unmeasured on real rounds.

**The interesting part is the conversion.** Training speed rose 11-15 % and real score rose
2.6-5 %. I called this "the bottleneck has moved" and guessed cornering. That guess was wrong;
section 22 is what was actually eating it. The general lesson is that training uses synthetic
teleport-to-waypoint segments while real rounds use the game's own gems, so a defect that lives
in the GEM PIPELINE is invisible to every training metric by construction.

Also settled during this run: the adaptive entropy controller did a full cycle unaided (0.42 ->
0.15, coefficient driven to its -0.03 floor, then reversed and climbed back through zero). The
fall was read as a runaway and that was wrong; `ENT_BAND_LO = 0.2` and `ENT_CUT_AT = 0.30` in
`nav/ppo_recurrent.py` define a dead zone the controller holds, and the undershoot to 0.15 is
the documented lag. Do not intervene in it.

## 22. THE OBSERVATION TOLD THE AGENT THE MAP WAS EMPTY FOR UP TO 1.9 s AFTER EVERY PICKUP (2026-09-21)

Found from an operator observation while watching a 1x round: "after picking up a gem the marble
sometimes moves in completely the wrong direction to start out, corrects itself after like 2
seconds, then makes a straight beeline". Their read was that the fault was in what we TELL the
model, and they were right.

**Measured on one FlatGem round (2,087 decisions, 36 pickups):**

* 92 % of pickups were followed by a window with NO gem in the observation
* window length median 21 decisions = **1.34 s**, max 30 = **1.92 s**
* **31 % of all decisions** in the round were blind
* the fallback goal was a median **35 deg** off where the gem turned out to be, more than 45 deg
  on 21 % of windows and more than 90 deg on 3 % (the "completely wrong direction" observed)
* **151 u of the round's 1,324 u of travel closed on nothing**, 11.4 %

**The map is not at fault.** `FlatGemTraining_Hunt.mcs` carries `maxGemsPerSpawn = 1` and
`minGemsPerSpawn = 1`, so exactly one gem exists at a time and `nvis` never exceeding 1 is
correct. Nor is the server: `unspawnGem` decrements `$Hunt::CurrentGemCount` and calls
`spawnHuntGemGroup` then `spawnGem` then `hide(false)` synchronously inside the same call, with
no `schedule` anywhere on that path. A replacement gem exists essentially instantly.

**The observer was reading a stale cache.** `AIObserver::collectGems` iterated `ItemArray`, a
client-side snapshot built by `buildItemList()` (`client/scripts/mp/items.cs`) which skips hidden
items at build time. In single player nothing refreshes it: its only callers there are
`updateClientItems()`, which returns unless `$Server::ServerType $= "MultiPlayer"`, and
`updateItemCollision()`, which returns on `"SinglePlayer"`. The observer's live fallback to
`ServerConnection` only fires when `ItemArray` is EMPTY, and it was not empty; it held one entry,
which was the hidden gem. The game's own log said so all along: `AIObserver: 1 objects from
ItemArray`.

**The fix** (`$AIObserver::GemSource = "server"`) prefers `ServerConnection`, which holds the live
ghosted objects. The per-object `!isHidden()` and `classname $= "Gem"` filters are unchanged,
which matters because those filters are what keeps powerups and BackupGems out. A
`$TypeMasks::ItemObjectType` bitmask pre-filter keeps the wider scan cheap (the same idiom
`buildItemList` uses). `$AIObserver::GemDiag` logs disagreements with the old path for proof.

**Proof, verbatim, 40 consecutive lines:** `AIObserver GEMDIAG: server 1 itemarray 0 size 1`.

**Result, same checkpoint, no retraining:**

| | before | after |
|---|---|---|
| FlatGem, 1 round | 89 gems, 26.2 % blind, 9.55 u/s | 98 gems, **0.0 % blind**, 10.02 u/s |
| FlatGem, 8 rounds | 84.9 gems | **96.8 gems** |
| KOTM, 8 rounds | 95.9 pts | **98.4 pts** |
| decisions / round | 4,706 | 4,706 (no throughput cost) |

`blind_pct` is 0.0 on all sixteen rounds across both maps. KOTM gained far less than FlatGem
because it spawns groups of 5-6, so most pickups leave other gems on the map and the stale cache
still had something to report; the bug's cost scaled with how often the map went empty.

**Consequences for everything above this section.** Every real-round number recorded before
20:00 on 2026-09-21 was measured through this bug, including the "gems/min is the live gap" and
"decelerates to 4.9 u/s at every pickup" claims in the morning state block, and section 15's
whole gap-goal analysis (which remains correct about what the fallback DID, and is now nearly
dead code because the fallback no longer fires). Training numbers were never affected: training
uses synthetic waypoints, which is exactly why this hid for so long.

## 23. WHY THE AGENT STILL DOES NOT ACCELERATE LIKE THE HUMAN (2026-09-21, research only)

With section 22 fixed, the approach-profile graph shows the agent tracking the human from 23 u in
to about 12 u and then plateauing at 12.8 u/s where the human arches to 14.94. Measured on one
100-gem FlatGem round against the no-jump human demo:

**Ruled out: command magnitude.** Agent mean 1.398 with 73 % on the 1.414 corner of the input
square; human 1.356 with 86 % on the corner. The agent commands slightly HARDER. This also
corrects an assumption worth recording: the human is NOT riding a single axis (which would cap
them at the measured 15.00 u/s single-axis terminal and make 14.94 a physics wall). 86 % of
their moving ticks are diagonal, so both share the 21.8 u/s diagonal ceiling and neither is near
it. The human's peak is a choice.

**Ruled out: heading sustain, and this REVERSES a claim still quoted in the code.** `real_run.py`
and `vec_worker.py` both carried comments saying the policy swings 39.5 deg per decision and
sustains a direction for 1 decision (0.06 s). Measured now: the agent holds its heading inside
20 deg for a median of **14 decisions = 0.90 s**, the human for 46 ticks = **0.74 s**. The agent
holds a line LONGER than the human. Those comments predate the gain work and are annotated.

**The cause: thrust direction relative to the marble's own velocity.**

| distance to gem | agent cmd-vs-velocity | human | agent aim at goal | human |
|---|---|---|---|---|
| 18-26 u | 48 deg | 38 deg | 0.93 | 0.92 |
| 12-18 u | 63 deg | **31 deg** | 0.71 | 0.85 |
| 9-12 u | 57 deg | **25 deg** | 0.51 | 0.86 |
| 6-9 u | 67 deg | 76 deg | 0.34 | 0.26 |
| 3-6 u | 76 deg | 144 deg | 0.29 | 0.58 |
| 0-3 u | 81 deg | 156 deg | 0.24 | 0.82 |

Far out the two are indistinguishable, which is why the graph's far side now overlaps. In the
9-18 u acceleration band the human puts thrust within 25-31 deg of its velocity, so cos = 0.91
of it adds speed; the agent is 57-63 deg off, so cos = 0.54. A **1.7x difference in the fraction
of thrust that accelerates**, landing exactly where the human's arch takes off. The human's
144-156 deg inside 6 u is deliberate braking; the agent never brakes and carries 10.55 u/s
through pickups against the human's 8.09.

Read plainly: the agent aims at the gem and then spends the approach on sideways line
corrections, while the human commits to a line early and spends everything on acceleration.

**Fix candidates, and what is already dead.** Charging for heading change (`TURN_COST`) was tried
at 0.15, 0.3 and 0.6 and reverted 2026-09-20: it worked as designed and the design was wrong,
because reaching a gem requires turning and the cheapest way to turn less is to go slower
(pickup speed fell monotonically 5.19 -> 4.64). A lump-sum arrival-momentum bonus (`CARRY`) was
a null twice, with the recorded lesson to make any retry DENSE. Surviving candidates, in the
order worth trying:

1. **Distance-dependent `DIR_GOAL_GAIN`.** The flat gain of 30 restored far-out aim to human
   level and left the near field alone (0.93 at 18-26 u decaying to 0.51 by 9-12 u where the
   human holds 0.86). Testable at inference only, one round, no training, the way
   `dirskip_scale.py` tested the flat gain.
2. **A dense per-decision payment for thrust along motion**, roughly
   `k * max(0, cos(angle(command, velocity)))` gated above a speed floor. It prices the measured
   0.54 against 0.91 difference, is dense rather than a lump sum, never charges for turning (so
   it avoids `TURN_COST`'s failure), and cannot be farmed by slowing down because a slow marble
   earns less of it. No flat-ground assumption, so it transfers.
3. **Tighten `GEM_SPEED_REF`** rather than raise `GEM_SPEED_BONUS`. The agent now reaches gems
   well inside the 60-decision reference on FlatGem, so the term is near saturation and its
   gradient is weak in the regime of interest. Likeliest to trade arrivals for speed; hold it.

**Caveat on the size of the prize.** The agent already averages 10.08 u/s against the human's
9.75 and still scores 100 to 104, because it covers 30.4 u per gem against 28.3. Peak speed is
real but ROUTE LENGTH is the larger remaining term, and none of the three candidates touches it.

## 24. TRAP: the approach-profile graph shipped with a flipped x-axis (2026-09-21)

`plot_approach_profile.py` called `invert_xaxis()` on both subplots while creating them with
`sharex=True`. The second call undid the first, so both graphs produced on 2026-09-21 rendered
with the GEM AT x=0 ON THE LEFT while the subtitle claimed "x runs from far (left) to the gem
(right)". The per-distance tables in this document are unaffected because they carry explicit
distances. Fixed by inverting once, with a comment.

## 25. PICK UP HERE (2026-09-21 21:00) -- state, blockers, and what to do first

Written for whoever takes this over cold. Everything in sections 3 through 24 is a dated log;
this section and the CURRENT STATE block at the top of the file are the only two places that
describe the present. If they disagree with each other, the top block wins.

### 25.1 Where the project is

**The goal set by the operator: 150 points on KingOfTheMarble.** Human baseline 143.5, so the
target is 4.5 % ABOVE a strong human, not merely parity.

Verified 2026-09-21 20:15 on `models/nav/nav_eval_dirgain_1934.pth` (update 13,895), 8 real
rounds per map via `eval_both.ps1`:

| | agent | human | share | round length |
|---|---|---|---|---|
| KingOfTheMarble | **98.4 points** | 143.5 | 69 % | 3.02 min |
| FlatGemTraining | **96.8 gems** | 104 | 93 % | 5.02 min |

**The single most important framing: FlatGem is nearly solved and KOTM is not.** 93 % against
69 %. The remaining work is KOTM-specific, and general "make the marble faster" work has largely
run its course. Evidence: on FlatGem the agent now moves at 10.08 u/s against the human's 9.75,
i.e. **103 % of human speed**, while on KOTM it moves at 6.68 against 8.05, i.e. **83 %**. The
agent can already drive faster than a human on open flat ground. What it cannot do is carry that
onto terrain with holes, slopes and edges.

### 25.2 The KOTM gap, decomposed (this is the map that matters)

Full 8-round means: 78.5 gems, 98.4 points, **1.253 points per gem** (human 1.26), 6.68 u/s,
**15.4 u travelled per gem**, 0.33 falls per 100 u, 25.98 gems/min, blind_pct 0.00.

**RECOMPUTED 2026-09-21 23:10 after section 27.** Current 8-round means: 81.25 gems, 102.9 points,
**1.266 points per gem** (human 1.26, still parity), 6.38 u/s, **14.23 u per gem**, 2.50 falls,
26.90 gems/min.

The gems-per-minute gap is 37.5 / 26.90 = **1.394x**, and it still factors cleanly, though the two
halves are no longer equal:

    speed              8.05 / 6.38   = 1.262x     <- now the DOMINANT half
    distance per gem   14.23 / 12.8  = 1.112x
    product                           = 1.403x   vs 1.394x observed

Section 27 cut route length (15.4 -> 14.23 u per gem) without touching speed, so **speed is now
clearly the bigger term**, and section 26 identifies what it is: the policy cannot hold a heading
on KOTM. The pre-section-27 figures, kept for the record, were 25.98 gems/min factoring as
1.205x speed and 1.203x distance.

Two conclusions follow, and both are load-bearing:

* **Speed is now the larger factor** (1.262x against 1.112x), after section 27 cut route length.
  Fixing speed alone reaches roughly 130 points; fixing route alone reaches about 114. **Reaching
  150 still requires progress on both**, but speed is where to start, and section 26 says the
  speed problem on KOTM is sustain.
* **Gem selection is NOT a lever and should be left alone.** Points per gem is 1.253 against the
  human's 1.26, i.e. parity. The agent already picks gems of human-equivalent value. Do not spend
  time on `NAV_VALUE_WEIGHT` or yellow-gem preference; section 3 of the roadmap's planner idea is
  also not the bottleneck (straight-line distance between consecutive pickups was already at
  parity, 11.1 u agent vs 11.0 u human).

**What 150 points requires, quantified (updated 23:10).** At 1.266 points per gem over a 3.02 min
round, 150 points needs 118.4 gems, i.e. **39.2 gems/min**, which is **1.46x** the current 26.90
and 105 % of the human's 37.5. Reaching it purely on speed would need about 8.9 u/s at today's
route; splitting it across both factors, roughly 8.0 u/s and 12.3 u per gem.

### 25.3 Blockers, ranked

**B1. Nothing has been trained against the corrected observation.** Highest priority and by far
the cheapest. Until 2026-09-21 20:00 the observation reported an EMPTY MAP for up to 1.9 s after
every pickup (section 22). Every policy ever trained, and every reward-shaping conclusion in this
document, was formed under that defect. The current checkpoint scores 98.4 only because
inference-time behaviour improved; the weights have never seen a truthful gem stream. Training on
the fix is unexplored and requires no new ideas.

**B2. KOTM speed, 6.68 u/s against 8.05. NOW MEASURED, see section 26.** The cause is that the
policy cannot hold a heading on KOTM: median run 0.19 s and 42 % of decisions change heading by
more than 20 deg, against the human's 0.29 s and 14 %. On FlatGem the same policy holds 0.90 s
and is fine, so the twitch is terrain-reactive rather than a general defect. When the agent does
sustain a heading on KOTM it reaches 10.48 u/s mean and 16.27 u/s at p90, above the human's
average, but it reaches that state on only 3 % of decisions. **This is the primary KOTM lever.**

**B3. KOTM route length, now 14.23 u per gem against 12.8** (was 15.4 before section 27).
Improved from 17.8 earlier in the project but still 11 % long, and now the SMALLER of the two
factors behind B2. Unexplored: whether the excess is edge-avoidance detours, overshoot and
re-approach, or wide arcs out of pickups (the agent carries 10.55 u/s through gems on FlatGem
where the human brakes to 8.09, which widens the exit arc; whether that also happens on KOTM is
unmeasured).

**B4. FlatIslands is unmeasured and regressed.** It fell from 92 % to 81 % training arrivals
during the `DIR_GOAL_GAIN` run and has NEVER had a real-round evaluation, so its true state is
unknown. It is 1 of 8 training instances, so a broken Islands is quietly polluting the gradient.

**B5. Mid-leg peak speed on FlatGem, 12.8 u/s against 14.94.** Real and measured (section 23) but
the LOWEST priority of these, because FlatGem is already at 93 % and the agent is already faster
than the human there on average. **It will NOT transfer to B2**: section 26 measured the same
quantities on KOTM and the FlatGem mechanism is absent there (thrust-versus-velocity angles match
the human within a few degrees, and the human's own KOTM profile has no arch at all). Treat B2
and B5 as separate problems.

**B6. `Sprawl` walk grid unverified.** 2,150 slope-excluded cells, 19 % of walkable area, status
unknown. `verify_walk_grid.py` exists but is unreliable: its drift criterion is invalid on ramps
and 25-41 u "slides" turned out to be respawns. It needs the game's own OOB flag instead. Sprawl
is NOT in the rotation, so this blocks only map expansion.

### 25.4 Do these first, in this order

1. **Restart training on the corrected observation and let it run.** `.\start_training.ps1`
   (defaults are already the current 4/3/1 rotation). Keep `DIR_GOAL_GAIN = 30`. Evaluate with
   `.\eval_both.ps1 -Tag postfix -Rounds 8` after a few hundred updates. This is B1 and it needs
   no design work. **Expect the training metrics to shift** even with no config change, because
   the observation itself changed; do not read that shift as a regression.
2. **DONE 2026-09-21 22:05, see section 26.** (Was: measure KOTM the way FlatGem was measured.) Run the section 23 analysis on a KOTM trace:
   thrust-versus-velocity angle and aim concentration per distance band, plus speed versus
   distance-to-gem. **The human KOTM demo already exists: `demos/demo_20260914_214854.npz`**,
   6 games, 18.2 min, 682 pickups, 861 points, which is exactly the 143.5 points/round and
   37.5 gems/min baseline quoted throughout this document. Pass it with
   `--demo demos/demo_20260914_214854.npz`. Use `demos/demo_20260921_155130.npz` for FlatGem
   (1 game, 98 pickups, 0 % jumps). Both carry the map they were recorded on in their
   `terrain_map` field, so check that rather than guessing. Without this analysis, B2 and B3 are
   guesswork.
   **Those two are the ONLY demos that exist.** Sections 19 and 20 cite
   `demos/demo_20260921_105710.npz`; the operator deleted that recording and re-made it without
   jumping as `demo_20260921_155130.npz`, so its numbers stand as history but the path is gone.
3. **Give FlatIslands a real-round evaluation** so B4 stops being unknown. One line:
   `.\real_run.ps1 -Map FlatIslands_Hunt -Rounds 8`.
4. **Then, and only then, pick a shaping change** from the surviving candidates in section 23.
   The distance-dependent `DIR_GOAL_GAIN` is testable at inference in a single round with no
   training, so it costs almost nothing to rule in or out.

### 25.5 Already tried and DEAD. Do not redo these without new information.

| tried | result |
|---|---|
| `TURN_COST` at 0.15 / 0.3 / 0.6 | Reverted. Worked as designed; design was wrong. Charging for heading change makes the policy go SLOWER, because turning less is achieved by going slower. Pickup speed fell monotonically 5.19 -> 4.98 -> 4.82 -> 4.70 -> 4.64. |
| `CARRY` (arrival momentum), two calibrations | Null both times. Fires once at pickup for a property created 10-20 decisions earlier. Any retry must be DENSE per-decision. |
| Potential-based reward shaping (PBRS) | Indicator stepped once then flat; KOTM lost 8 arrivals. Reverted. |
| PROGRESS / TIME rebalance | Broke Islands via the airborne tax on TIME. Reverted. |
| Forcing throttle to 100 % | Moved the alignment distribution not at all. Throttle already measures 0.99 mean. |
| Action smoothing (`NAV_SMOOTH`, `ACTION_SMOOTH`) | A crutch, since replaced by `GEM_SPEED_BONUS`. Also: the policy internalised the old smoothing, so applying a filter at inference now over-damps and halves the score. Run real rounds WITHOUT it. |
| Behaviour cloning / BC aux term | Both regress the policy (ablation proved it). |
| Jumping as the FlatGem lever | Operator re-recorded their demo with 0 % jumps and still scored 104, which disproves it. |
| Gem selection / yellow preference | Points per gem is already at human parity (1.253 vs 1.26). |
| Planner as the next lever | Straight-line distance between consecutive pickups already at parity (11.1 vs 11.0 u). |

Also ruled out by measurement, so do not re-investigate: input magnitude (1.398 mean, 73 % on the
1.414 corner, HARDER than the human's 1.356), impulse-versus-hold (a 64 ms decision delivers
99.6 % of four 16 ms ticks), heading sustain (agent holds 0.90 s, human 0.74 s, agent holds
LONGER), command persistence, frame error, route curvature (1.08 vs 1.03), powerups.

### 25.6 Operating traps that will cost you hours

* **Python is the SERVER.** It binds and listens; the game dials it 100 ms after "GO!". Start
  Python first and wait for the port. Every launcher does this via `nav_ready.ps1`.
* **The trainer AUTO-RESTARTS on crash**, so a broken run looks like a running one. The giveaway
  is the log line count going DOWN between checks. Always check line count, not just process
  count.
* **A real-round evaluation requires stopping training.** The 8 GB GPU cannot hold a 9th game
  instance beside a PPO update.
* **Delete the matching `.dso`** after editing any `.cs` or `.mcs`, or the engine keeps running
  the old compiled version.
* **`nav_latest.pth` is often NOT the best checkpoint.** Recompute the smoothed peak from the
  MAPS lines and use the nearest numbered checkpoint (saved every 25 updates). Do not call a
  plateau from one downward stretch of 50-80 updates; that was called twice and was wrong twice.
* **Do not intervene in the entropy controller.** It legitimately drives its coefficient negative
  and undershoots below the band before reversing. `ENT_BAND_LO = 0.2` and `ENT_CUT_AT = 0.30`
  define a dead zone it holds. Reading the fall as a runaway was a mistake made on 2026-09-21.
* **`arrive=` in training means the WHOLE GROUP was collected**, roughly per-gem^6. 62 % group
  arrive is 92.5 % per gem. Convert before reacting to a swing.
* **`gemDelta` is POINTS, not a gem count.** Both are tracked separately in `real_run.py` output.
* **Training metrics can be flatly disconnected from real score.** The most recent example: a
  15 % training-speed gain converted to 5 % of real score because the defect lived in the gem
  pipeline, which training never touches. When training and real rounds disagree, suspect the
  real-round pipeline FIRST.

### 25.7 Standing rules set by the operator. These are not negotiable.

1. **Waypoints may ONLY ever appear inside a real gem.** No routing corner points, no marker on
   empty ground.
2. **READY AT GO.** Every run that rolls the marble must be able to move the instant the round
   says GO, training and evaluation alike.
3. **Judge only by 8-round `nav/real_run.py` evaluations.** Never by the training curve.
4. **Never stop the trainer without explicit permission.** Only the operator ends a run. If an
   experiment is clearly failing, report it with numbers and KEEP TRAINING. This rule exists
   because it was broken once.
5. **No code changes without the operator's approval.** Present analysis and options, then wait.
6. **Solutions must transfer across maps.** No oracles or observation dimensions that bake in
   flat-ground physics.

### 25.8 Current config, for reference

`EDGE_K = 1.0`, `FALL = 25.0`, `TIME = 0.05`, `PROGRESS = 1.0`, `ARRIVE = 10.0`,
`GEM_SPEED_BONUS = 8.0`, `GEM_SPEED_REF = 60`, `BRAKE = 0.05`, `BRAKE_SUPPRESS = 1.0`,
`JUMP_DAMP = 3.0`, `BRAKE_ENABLED = True`, `THROTTLE_FLOOR = 0.90`, `CARRY = 0.0`,
`TURN_COST = 0.0`, `DIR_GOAL_GAIN = 30.0`, `NAV_SMOOTH = 1.0` (off), `DIRECT_GEM = 1`,
`NAV_OOB_CLICK = 1` (the legal quick respawn, section 27; training sends it too, from
`waypoints.py::recover()`), `$AIObserver::GemSource = "server"`, adaptive entropy in band 0.2-0.8, rotation
4 FlatGemTraining / 3 KingOfTheMarble / 1 FlatIslands across 8 instances.

**Training is DOWN as of 2026-09-21 20:16.** Last run reached update 13,895.

## 26. KOTM AND FLATGEM LOSE SPEED FOR DIFFERENT REASONS (2026-09-21 22:05)

Section 25 step 2 asked for the section 23 analysis to be repeated on KOTM, and guessed under B5
that whatever fixes FlatGem's peak speed "may transfer" to KOTM. **It does not.** Run on one
85-gem / 103-point KOTM round (`nav_eval_dirgain_1934.pth`) against the human KOTM demo
`demos/demo_20260914_214854.npz` (682 pickups over 6 rounds).

### 26.1 The FlatGem mechanism is absent on KOTM

On FlatGem the defect was thrust pointing sideways relative to the marble's own motion: agent
57-63 deg off velocity where the human held 25-31 deg. On KOTM the two are nearly identical:

| distance to gem | agent cmd-vs-velocity | human | agent speed | human speed |
|---|---|---|---|---|
| 18-26 u | 77 deg | 57 deg | **8.95** | 6.93 |
| 12-18 u | 77 deg | 69 deg | 6.55 | 7.01 |
| 9-12 u | 62 deg | 52 deg | 5.65 | 7.32 |
| 6-9 u | 53 deg | 46 deg | 7.76 | 8.91 |
| 3-6 u | 82 deg | 82 deg | 6.87 | 9.48 |
| 0-3 u | 108 deg | 101 deg | 5.98 | 7.65 |

The human ALSO turns hard on KOTM, because the map demands it, so "commit to a line and push
along it" is not available to either player and is not the gap. Note also that the human's own
KOTM profile is flat at 6.9-9.5 u/s with no 14.94 arch anywhere: **the FlatGem approach-profile
shape is a property of flat open ground, not of human play.** Do not carry FlatGem's arch over as
a target for KOTM.

Ignore the 0-3 u aim-concentration figures on any map: both players read ~0.03-0.06 there because
the goal bearing changes faster than the command can track it. The measure is only meaningful
beyond ~6 u.

### 26.2 What the KOTM defect actually is: the policy cannot hold a line at all

| heading held inside 20 deg | agent | human |
|---|---|---|
| median run length | **3 decisions = 0.19 s** | 18 ticks = 0.29 s |
| decisions changing heading >20 deg | **42 %** | 14 % |
| share of time in runs of 16+ | **3 %** | 46 % |

Compare the same measurement on FlatGem, where the agent is FINE: median 14 decisions = 0.90 s
against the human's 0.74 s, and 32 % heading changes. **The same policy twitches on KOTM and does
not twitch on FlatGem.** That is the finding.

And the twitch is what costs the speed, because when the agent does hold a line it is fast:

| agent run length | mean speed | p90 speed | n |
|---|---|---|---|
| 0-1 dec | 7.04 | 10.40 | 1180 |
| 3-6 dec | 5.56 | 9.80 | 401 |
| 6-10 dec | 5.22 | 7.64 | 264 |
| **16+ dec** | **10.48** | **16.27** | 86 |

A sustained heading reaches 10.5 u/s mean and 16.3 u/s at p90, well above the human's 8.05
average. The agent reaches that state on only 3 % of decisions. The human's speed by contrast is
nearly FLAT across run length (7.2 to 9.1), i.e. the human is not sustain-limited at all; they
simply hold lines as a matter of course.

### 26.3 Why the same policy behaves differently on the two maps

Best available explanation, consistent with every number to hand: the twitch is
**terrain-reactive**. FlatGem is open and flat, so the terrain crop and the edge ray-marches are
nearly constant and nothing perturbs the direction head. KOTM has holes, edges and slopes, so
those inputs change every decision and their contribution to the hidden state moves the commanded
direction with them. `DIR_GOAL_GAIN = 30` strengthened the goal bearing relative to the hidden
state, which is exactly why it was worth +11.9 gems on FlatGem and only +2.5 points on KOTM:
on KOTM the hidden state still carries enough terrain reaction to dominate.

This is an explanation, not a proof. It predicts two things that are cheap to check and have not
been checked: (1) the agent's heading changes on KOTM should correlate with changes in the edge
features or crop rather than with changes in the goal bearing; (2) a higher `DIR_GOAL_GAIN`, or
any change that damps the terrain pathway into `dir_head` specifically, should lengthen KOTM run
lengths without hurting FlatGem.

### 26.4 Revised priority for KOTM

Route length is 14.1 u per gem on this round against the human's 12.8 (8-round mean 15.4), so it
remains a real but secondary term. The primary KOTM lever is **sustain**, and it is a different
lever from FlatGem's. Recommended order:

1. Verify the terrain-reaction hypothesis by correlating heading change against edge/crop change
   versus goal-bearing change on the existing KOTM trace. Pure analysis, no training, no new runs.
2. If it holds, test a raised or terrain-damped `DIR_GOAL_GAIN` at INFERENCE on KOTM first
   (`dirskip_scale.py` already does inference-time scaling of the goal columns, and
   `NAV_DIRSKIP` needs no training run to tell you whether run length moves).
3. Only then consider reward shaping. Note that `TURN_COST`, which is the obvious way to buy
   sustain, is already dead for a well-understood reason (section 23): charging for heading change
   makes the marble go slower, because turning less is achieved by going slower.

**Supersedes:** section 25 B5's guess that a FlatGem peak-speed fix "may transfer" to B2. It will
not; the mechanisms are different. Section 25's decomposition of the KOTM gap into equal speed and
route factors still stands, and this section identifies what the speed half actually is.

## 27. THE LEGAL QUICK RESPAWN: +4.5 points on KOTM (2026-09-21 23:10)

Operator observation while watching a 1x KOTM round: "after the marble falls OOB, it falls all the
way down and waits for the game to respawn it. however there's a mechanism in the game that all
human players use to respawn quicker, as soon as the text out of bounds appears, clicking the left
mouse will respawn immediately". With the constraint attached: **do not respawn before the text
appears, because a real competitive Hunt game will not allow it.**

### 27.1 The mechanism

A human's left click runs

    input_mouseFire -> commandToServer('MouseFire') -> serverCmdMouseFire (mp/commands.cs:80)
      -> MPOutofBounds() -> if (%client.isOOB) %client.respawnFromOOB()   (mp/server.cs:144)

and in `GameConnection::outOfBounds` (server/scripts/game.cs, around line 971) three things happen
in ONE synchronous function, in this order:

    %this.isOOB = true;                            // source comment: "used for OOB Click in Multiplayer"
    %this.setMessage("outOfBounds", 2000);         // the on-screen text
    %this.respawnSchedule = %this.schedule(2500, respawnFromOOB);   // the automatic respawn

So `isOOB` turns true on the SAME TICK the text appears, six lines earlier in the same call.
**Gating on `isOOB` is exactly the gate a human click passes, not an approximation**, and it
cannot fire earlier than a human could legally click because the flag is false until the function
that prints the text sets it. The operator's constraint is satisfied by construction rather than
by a timer we chose.

`respawnFromOOB` is also the identical function the automatic path calls 2.5 s later, so calling
it early changes the timing and nothing else.

**It is the competitively legal path, and the game itself says so.** The other quick respawn,
`serverCmdQuickRespawn` (the respawn KEY), is explicitly blocked in competitive Hunt:
`(!$MPPref::Server::CompetitiveMode || !$Game::isMode["hunt"])`. The mouse OOB path carries no
such check.

### 27.2 What we were doing instead, on both sides, and both were wrong

* `nav/real_run.py` never respawned at all. It reset the hidden state on a fall and waited out the
  full automatic 2.5 s.
* `nav/waypoints.py::recover()` DID respawn instantly, but through the `RESPAWN` control, which
  force-clears `isOOB` and calls `respawnPlayer()` unconditionally. That is a teleport, not an OOB
  click, and it is precisely the move a real competitive round refuses. **Training was therefore
  being handed a recovery the agent could never have in a scored round**, which is the exact
  failure the operator's constraint was aimed at.

The two halves also disagreed with each other: training recovered instantly, real rounds took
3.26 s. Training was learning that falls are cheaper than they actually are when scored.

### 27.3 The fix

New control `OOBCLICK` in `mlAgent.cs`, whose entire body is

    %cl = ClientGroup.getObject(0);
    if (isObject(%cl) && %cl.isOOB)
        %cl.respawnFromOOB();

Sent by `real_run.py` on the decision a fall is reported (`NAV_OOB_CLICK`, default on), and by
`waypoints.py::recover()` as the FIRST action on a mid-group fall. `RESPAWN` is kept in `recover()`
as an escalation after `FORCE_RESPAWN_DECISIONS`, because a marble that is off the map but never
flagged OOB (seen 2026-09-17) gets no respawn from the game at all and `OOBCLICK` is a no-op for
it, so the segment would hang. That fallback is not legal play and is labelled as such in the code.

### 27.4 Measurement

Per-fall dead time on KOTM, measured by altitude from the fall flag until the marble is back on the
floor, is **bimodal**, and the gap between the clusters is the 2500 ms schedule almost exactly:

    WITHOUT:  0.83 0.83 0.83 3.26 3.26 3.26 3.26 3.26 3.26     mean 2.45 s
    WITH:     0.77 0.77 0.77 1.09 1.09 1.09                     mean 0.93 s

The 3.26 s cluster disappears entirely. The ~0.8 s floor is the respawn drop-in, which happens
either way and is not recoverable. Dead time per round went **11.0 s -> 2.8 s**.

**8-round evaluation, `nav_eval_dirgain_1934.pth`, the only change being this control:**

| map | before | after | |
|---|---|---|---|
| KingOfTheMarble | 98.4 pts | **102.9 pts** | 106,106,97,96,102,104,107,105 |
| FlatGemTraining | 96.8 gems | 96.5 gems | unchanged, see below |

**+4.5 points on KOTM**, against a prediction of +3.9 from the dead-time measurement. The spread
also tightened, 96-107 against the previous 85-107, because the worst rounds were the fall-heavy
ones.

**FlatGem is an unplanned control and a useful one.** That map runs at zero falls, so `OOBCLICK`
never fires, and the score is unchanged at 96.5 against 96.8. The change is a confirmed no-op where
there is nothing to recover from, which makes the KOTM movement attributable to the fix rather than
to checkpoint variance.

**An unexplained side effect, flagged rather than claimed.** KOTM falls per round fell from 4.00 to
2.50. The OOB click does not prevent falls, so this is not the fix acting directly. It may be that
a marble back in play in 0.9 s rather than 3.3 s meets a differently-timed board and gets fewer
chances to repeat a bad approach, or it may be variance (falls ranged 0-7 per round in the earlier
sample). **Not verified. Do not build on it without measuring it.**

### 27.5 Where this leaves KOTM

98.4 -> **102.9**, i.e. **72 % of the human 143.5**. Progress across 2026-09-21:
93.5 (fall-credit fix) -> 95.9 (`DIR_GOAL_GAIN = 30`) -> 98.4 (observer gem source) -> 102.9
(legal quick respawn).

This recovers wasted clock and does nothing for the two structural gaps. Section 25's decomposition
still holds and should be recomputed against the new numbers: distance per gem is now 14.2 u
against the human's 12.8, and speed 6.38 u/s against 8.05. Section 26's sustain finding (the policy
holds a heading for 0.19 s on KOTM against the human's 0.29 s, and only 3 % of its decisions sit in
runs of 16+) remains the primary lever and is untouched by this section.

**Method note worth carrying forward.** Three of the four gains today came from an operator
WATCHING a 1x round and describing something that looked wrong: the wrong-direction start after a
pickup (section 22), and this one. Neither was visible in any metric being tracked. Watching real
play at 1x is a first-class diagnostic tool, not a demo.

## 28. THE CENTRE IS SLOW BECAUSE EVERY PICKUP IS A STOP-AND-GO (2026-09-22 00:10, analysis only, nothing changed)

Operator observation: "it plays too cautious and slow, especially in the centre around the holes."
Measured on a fresh 8-round KOTM evaluation of `nav_eval_dirgain_1934.pth` (670 pickups, 106 points
equivalent at 27.7 gems/min, consistent with section 27's 102.9) against the human demo (682 pickups).
Scripts are in the session scratchpad; every number below comes from `logs/nav/real_trace.csv` (8 rounds,
22,645 decisions) and `demos/demo_20260914_214854.npz`.

### 28.1 Where the time goes: legs into and out of the centre block

The centre is a 6.5 u square block around the 2x2 hole at (-25.2, 15.0), reached by 2.5-3 u bridges.
Its four gem spawns sit on walk-grid EDGE cells (edge_dist 0) and supply 36 % of the human's gems.

| leg type (consecutive pickups) | agent n / mean s | human n / mean s | agent excess |
|---|---|---|---|
| ring -> ring (3.9 u) | 76 / 0.96 | 112 / 0.85 | 8 s |
| ring -> out | 164 / 1.94 | 135 / 1.60 | 55 s |
| **out -> ring** | 161 / **3.12** | 133 / 1.84 | **205 s** |
| out -> out | 261 / 1.95 | 296 / 1.71 | 64 s |

Legs touching the ring are 73 % of the 332 s gap. Leg MIX explains only ~15 % of it: at the human's
per-type durations the agent's own leg mix gives 1070 s against its actual 1402 s, i.e. **x1.31
gems/min = 139 points**, and matching the human's fall rate as well (2.1 falls/round vs 0.2, ~7 s of
dead time and return trip) gives ~144. Gem VALUE is at parity (1.266 vs 1.26 points per gem), as
section 25 already said.

Region speeds (share of time, mean speed): ring, agent 27 % at 5.12 u/s vs human 23 % at 7.43 (63 % of
human); walkways 7.52 vs 8.76 (86 %); intersections 6.45 vs 7.27 (89 %). Inside the block itself the
agent averages 4.6 u/s with 13 % of its time below 2 u/s and 0.6 % above 10; the human averages 6.9 with
22 % above 10 u/s.

### 28.2 The mechanism: braking into the gem, near-stop after it, one second to recover

Speed by distance still to go on out -> ring legs: agent plateaus at 7.9 u/s from 8 u out and brakes to
4.7 at the gem; the human accelerates 5.6 -> 10.9 and takes the gem at 9.8. Pickup speed at the four
ring gems: **agent 4.50, human 8.21**; one second earlier both are near 7 (6.6 vs 7.5). On outer gems the
agent is 6.62 vs 7.49, so the braking is specific to the edge-cell gems.

After the pickup, every leg type shows the same dip, at matched turn angles:

| after pickup | agent min speed in the next 1 s / share below 2 u/s | human | time to be back at 8 u/s, agent / human |
|---|---|---|---|
| ring -> out | 2.0 u/s / **51 %** | 6.8 / 14 % | 1.02 s / 0.00 |
| ring -> ring | 2.8 / 29 % | 4.4 / 10 % | (short legs) |
| out -> ring | 3.6 / 14 % | 6.0 / 6 % | 0.90 / 0.51 |
| out -> out | 3.7 / 12 % | 6.4 / 7 % | 0.83 / 0.43 |

Decision-by-decision replays (e.g. decisions 762-790) show it plainly: 5.7 u/s at the pickup, 0.3 u/s
seven decisions later, then 1.5 s of acceleration to 10.6. The human turns 86 deg at pickup on
out -> out legs (same as the agent) and keeps 6.4 u/s through it. Section 26's "sustain" finding is this
same phase seen differently: the heading-change spikes are 13-17 % ONLY at 8-17 u to go on
ring -> out legs, i.e. the first second after a ring pickup; elsewhere both players are at 0-3 %.

Two smaller items in the same data:
* Ordering at ring exits: with the velocity 0.5 s before the pickup, the human's turn to the next gem is
  a median 31 deg on ring -> out (leaves the block along the line it swept) against the agent's 79 deg.
  Other leg types match (83-91 deg both). Inference-only lever (`real_run.py::choose`), secondary.
* 40 % of the agent's ring entries start at a corner gem (human 11 %), 3.74 s each. Also ordering.
* Hanging on the hole's lip: 5 stalls of 2+ s in 8 rounds (0.5 % of time), the marble driving west along
  y ~14.4 straight onto the south rim, sitting at ~0 u/s half over the void, then dropping in. Minor.

### 28.3 Why the policy does this: the reward, in numbers

* **Braking into ring gems buys no safety.** Training trace `trace_20260921_163647.csv`, KOTM instances,
  16,952 ring-goal arrivals: a fall within 1.5 s of arrival is 0.11 %, and FLAT in arrival speed
  (0.20 % at 0-3 u/s, 0.14 % at 7-9, 0.00 % at 9-12; outer goals 0.13 %). Training arrives at ring goals
  at 5.05 u/s against 5.91 at outer goals, so training teaches the brake and real rounds amplify it
  (4.50). The fear is priced in, the danger is absent at these speeds.
* **Time is under-priced ~4x.** What a decision is worth in the reward: TIME 0.05 plus the
  GEM_SPEED_BONUS slope 8/60 = 0.13, total **0.18**. What a decision is worth in a scored round: per
  gem ~27 reward (ARRIVE 10 + progress ~14 + bonus ~3.5) x 27.7 gems/min / 937 decisions/min =
  **~0.8**. Section 19 measured the same thing as "98 % of a leg's reward is speed-independent".
* **PROGRESS 1.0/u makes momentum read as loss.** Carrying speed past a gem in an arc costs ~1-1.5 u of
  geodesic progress = 1.0-1.5 reward; stopping and turning in place costs ~8 decisions x 0.18 = 1.4.
  Break-even under the current prices, so the policy stops. At the true price (0.8/decision) the stop
  costs ~6 and the arc ~2. The human's curve is what a correctly priced clock produces.
* **The edge charge taxes human-speed play in the block.** Ignoring the goal exemption it fires on
  61-69 % of in-block decisions at 7+ u/s (mean 0.41-0.47 per decision); with the exemption it still
  charges the human's own trajectory 35 % more per ring decision than the agent's (0.081 vs 0.060).
  Secondary to the pricing above, and not proposed for change in the first run.
* Training never sees consecutive goals closer than GROUP_LINK_DMIN = 6 u; the ring gems are 3.9 u
  apart and ring -> ring legs are 11 % of real legs (agent exit speed there 4.55 vs human 6.30).

### 28.4 Proposal (pending operator approval; no code changed)

1. **Price time through the collection-paid bonus, not per-decision TIME.** `GEM_SPEED_BONUS 8 -> 40`,
   `GEM_SPEED_REF 60 -> 80` in `nav/waypoints.py`: slope 0.5 per decision (was 0.13), saturating at
   5.1 s (95 % of real legs are under 61 decisions, 99 % under 80). Paid only on collection, so it cannot
   be farmed by not collecting, never goes negative, and does NOT tax airborne decisions as such, which
   is what broke FlatIslands when TIME went to 0.20 (section 19). PROGRESS stays 1.0 as the scaffold;
   FALL 25 stays (a fall now also forfeits the bonus through the clock, the correct incentive).
2. **`GROUP_LINK_DMIN 6.0 -> 3.0`** so 4 u gem pairs exist in training.
3. Restart training on the current rotation, judge ONLY by 8-round real runs on KOTM and FlatGem
   (FlatGem is the control: it has no edges, so any gain there is pure pacing). Watch the MAPS lines for
   FlatIslands (baseline 86-87 % arrive, falls100 1.32) and KOTM falls100 (baseline 0.48) as canaries;
   report, do not stop (operator rule).
4. Secondary, inference-only, can run any time training is down: momentum-aware ordering in
   `real_run.py::choose` (prefer the gem that continues the current heading when costs are close),
   aimed at the 79 vs 31 deg ring-exit turn and the 40 % corner starts.

Expected size of the prize: the pickup dip alone is worth up to the 139 points computed in 28.1 if
matched to the human; 150 additionally needs the fall rate (2.1/round) brought toward the human's 0.2.
Note on the target: 150 POINTS is 118 gems at 1.266 points per gem; 150 GEMS would be ~190 points.

Handoff's B1 ("nothing trained on the corrected observation") is moot for the policy: training goals
come from `terrain.gem_spawns` via `vec_worker`, never from the observer, so the section 22 bug only ever
touched real rounds. Training is DOWN (it was down when this session began and was not restarted).

### 28.5 APPLIED (2026-09-22 00:05): the section 28.4 change is live

Operator approved the plan with the goal confirmed as **150 POINTS**. Applied in `nav/waypoints.py`:
`GEM_SPEED_BONUS 8 -> 40`, `GEM_SPEED_REF 60 -> 80` (slope 0.5 per decision), `GROUP_LINK_DMIN 6 -> 3`.
Pre-change weights snapshot: `models/nav/nav_before_timeprice_20260922.pth` (update 13,895).
Training restarted 23:55 from `nav_latest.pth` via `start_training.ps1`; first update 13,899, per-update
`rew=` now ~270 (was ~60-80) because the bonus is larger, so do not read the reward level against old runs.

**Pool changed 00:00 at the operator's request: 2 FlatGem / 5 KOTM / 1 Islands** (was 4/3/1), now the
default `-Split` in `start_training.ps1`. Games were relaunched with the trainer left running; workers
reloaded their maps on reconnect and the MAPS line shows n=2/5/1. First readings on the new split:
FlatGem arrive 99 %, speed 10.0; Islands arrive 94 %, falls100 1.12-1.21; KOTM arrive 98 %, falls100
0.52-0.55, speed 5.7. Those are the canary baselines for this run.

Operator note for this night only: stopping training is allowed if warranted (e.g. for the 8-round eval).
Evaluate with `.\eval_both.ps1 -Tag timeprice -Rounds 8` after a few hundred updates; judge by KOTM points
with FlatGem as the no-edges control. The pickup-dip metrics of 28.2 (min speed in the 1 s after a
pickup, time to regain 8 u/s, ring pickup speed) are the direct readout of whether the change worked.

### 28.6 First eval of the timeprice run: NO CHANGE after 360 updates (2026-09-22 02:10)

`eval_cycle.ps1 -Tag timeprice` on `nav_eval_timeprice_0142.pth` (update 14,255, ~360 updates on the new
reward): **105.0 points** (106,102,105,108,105,106,101,107), 27.6 gems/min, 16 falls. A second copy of
the cycle fired one minute later on the same weights (`_0143.pth`) and scored 103.8 with 27 falls, but it
ran alongside a full training stack (see the incident below) so it is confounded. Baseline was 106 / 102.9.
The pickup-dip readout is unchanged too: 53 % of ring exits below 2 u/s, 0.9-1.0 s to regain 8 u/s.
Training-side KOTM pickup speed 5.5 -> 5.2 and falls100 0.52 -> 0.56-0.73 over the same window, i.e. no
sign yet of faster pickups. Verdict withheld: 360 updates may be too few for a new motor pattern (arc
turns at speed) as opposed to re-weighting an existing one. Next automatic eval at update 14,900 (tag
`timeprice2`). If that is also flat, the pricing hypothesis needs a different lever (see 28.3: the edge
charge, or a dense per-decision term), not a bigger constant.

**Incident, for the record.** Two waiter scripts were alive (the first was launched through the tool's
background runner and did not die when "stopped"), so `eval_cycle.ps1` ran twice, one minute apart. Each
restarted a full trainer + loop + 8 games afterwards, giving two stacks on ports 8888-8895 and 16 game
instances; the second trainer never bound and its games dialled the first stack's workers. Cleaned up
02:05 by killing the 02:01 stack; the 01:45 stack (resumed at update 14,255) is the live one. Lesson:
launch long waiters ONLY as detached processes (`Start-Process` on Git Bash), never via the tool runner,
and check `Get-Process marbleblast_mbx` count = 8 after any eval cycle.

### 28.7 MOMENTUM-AWARE GEM ORDERING: +8 POINTS AT INFERENCE, NO TRAINING (2026-09-22 02:35)

The secondary lever from 28.4 item 4, tested as a pure A/B on FIXED weights
(`models/nav/nav_eval_timeprice_0142.pth`, update 14,255) with `eval_cycle.ps1` (8 rounds each, ~3.5 min
of training downtime per run). `real_run.py::choose` now adds to each gem's distance the turn-around
cost `MOMENTUM_K * speed * (1 - cos(angle between the marble's velocity and the bearing to the gem))`;
`NAV_MOMENTUM_K` env, **default now 1.6** (0 restores greedy-nearest).

| K (s) | points, 8 rounds | mean |
|---|---|---|
| 0 (greedy nearest) | 106,102,105,108,105,106,101,107 | 105.0 |
| 0.4 | 103,111,110,112,109,112,118,110 | 110.6 |
| 0.8 | 112,115,110,113,103,112,116,123 | 113.0 |
| 1.2 | 114,114,115,101,106,104,110,105 | 108.6 |
| 1.6 | 115,118,120,117,118,120,117,121 | **118.3** |
| 1.6 (replicate) | 116,115,111,94,115,105,108,118 | 110.3 |
| 2.2 | 124,87,115,116,111,123,122,116 | 114.3 |
| 3.2 | 112,117,109,122,112,103,113,108 | 112.0 |

Pooled K >= 0.8 (40 rounds): **113.6**, i.e. **+8.6 over greedy-nearest on the same weights**. The
8-round mean has roughly +-3 of noise (single fall-heavy rounds of 87 and 94 appear), so K is not tuned
finer than "1.6 to 2.2". Mechanism it targets: the agent's ring->out turn at the exit gem was 79 deg vs
the human's 31 (28.2); with the term the marble leaves the centre along its momentum instead of
reversing into a stop-and-go. Per-leg readout at K=0.4: ring->out mean 2.02 -> 1.88 s, path/straight
1.10 -> 1.04, and 697 legs per 8 rounds against 652.

**Attribution note for later evals.** Every real run from 02:35 on uses K=1.6 by default, so the
`timeprice2` eval at update 14,900 must be read against ~113.6 (momentum baseline on the 14,255 weights),
NOT against 105. To isolate training progress from the ordering term, run with `NAV_MOMENTUM_K=0`.

**Still open on this lever:** distance in `choose` is Euclidean; on KOTM's lattice a geodesic (goal_field)
distance would stop it picking a 14 u corner-to-ring gem that is 20 u by path. Untested.

Geodesic variant tested 02:36 (`NAV_GEO_ORDER=1`, K=1.6, same weights): 124,120,110,115,115,103,110,111
-> **113.5**, indistinguishable from the momentum-only pool (113.6). Left OFF by default; the code stays
(`real_run.py`, one cached Dijkstra per walk cell, 0.3 ms on KOTM) for maps where the straight line lies.

Cost of the sweep: `eval_cycle.ps1` kills the trainer without saving, so each run lost the updates since
the last checkpoint; the trainer sat at update ~14,300 from 02:04 to 02:38 (seven evals). If sweeps like
this become routine, make the cycle send a save request first, or use a copied checkpoint and leave
training up (needs a 9th game's worth of VRAM, which the 8 GB card does not have).

### 28.8 TRAP: game instances silently halve training throughput after a while; relaunching them fixes it (2026-09-22 04:15)

At 03:36 every instance's rtf dropped from 4-5 to 2.0-2.1 in one step; per update `wall_s` 16 -> 35,
`dps` 520 -> 235, profile `w_game` 21 -> 136 ms. CPU 13 %, GPU 53 %, no throttle, no new process, all
eight games equally slow, uniform 32 ms extra per decision. Ruled out: display sleep (the 23:55 stack ran
107 min through the display timeout at a steady 17 s), process age (same evidence), CPU/GPU contention.
Measured: a plain sleep on this machine is 15.5 ms unless the process holds a 1 ms timer request, so the
extra 32 ms per decision is exactly two coarse sleeps, i.e. the games lost their 1 ms timer resolution
(Windows 11 revokes it per process under conditions it does not document). Waking the display did not
help; **killing the eight games (the loop relaunches them, workers reconnect) restored 16 s immediately**.
`logs/nav/game_watchdog.sh` now runs detached and does that automatically after three slow readings.
If a run's update rate halves with nothing else wrong, check `wall_s`/`w_game` before anything else.

### 28.9 VERDICT ON THE TIME REPRICING: REVERTED. It bought speed with falls (2026-09-22 05:55)

Second eval at update 14,913 (~1,000 updates on GEM_SPEED_BONUS 40 / REF 80), 8 rounds each:

| weights | ordering | points | falls / 8 rounds | gems/min |
|---|---|---|---|---|
| 14,255 (start of run) | K=0 | 105.0 | 16 | 27.6 |
| 14,255 | K>=0.8 pooled | 113.6 | n/a | n/a |
| **14,913** | K=1.6 | 116.3 (119,106,115,125,112,116,117,120) | **45** | 30.7 |
| **14,913** | K=0 | 102.6 (108,99,104,105,105,100,96,104) | **51** | 27.1 |

Same ordering, the trained weights score 105.0 -> 102.6 while real-round falls triple (0.18 -> 0.54 per
100 u; ring-exit legs carry a fall 19 % of the time). The pickup dip DID move (ring exits below 2 u/s
53 % -> 42 % at K=0, time to regain 8 u/s 0.96 -> 1.09 s, i.e. mixed), gems/min at K=1.6 reached 30.7,
but every gained second was spent falling. Training-side KOTM falls100 stayed 0.45-0.55 throughout, so
the training curve did not show it; only the real rounds did (rule 3 again). This is the trade the
2026-09-18 note predicted: "told to go faster, a policy that already falls more will buy speed with falls".

Action taken under the one-night stop permission: GEM_SPEED_BONUS/REF back to 8/60, weights restored to
`nav_eval_timeprice_0142.pth` (update 14,255, the best-verified checkpoint: 105.0 at K=0, 118.3/110.3 at
K=1.6), endpoint of the 40/80 run kept as `nav_timeprice_end_14913.pth`. `GROUP_LINK_DMIN = 3` and the
2/5/1 pool are kept (not implicated; ring->ring legs carry a fall 2 %). Training restarted 05:57.

**What the night established.** The centre gap is real and its mechanism (stop-and-go at pickups, 28.2)
is measured, but pricing time harder is the wrong lever for it while the policy cannot turn at speed
without leaving the block. The lever that worked is the ordering term (+8 to +14 points at inference).
Candidates that survive for the pickup dip itself: FALL raised together with the time price (so speed
cannot be bought with falls), or a dense per-decision term for thrust along motion (section 23 item 2);
either needs the operator's decision. **Current best real-round configuration: 14,255 weights +
K=1.6 = 113-118 points.**

### 28.10 FEAR REMOVAL, STEP 1 APPLIED: stopping-distance edge charge + post-pickup grace (2026-09-22 08:25)

Operator, after watching a 1x round: "slow and hesitant, acts like it's really scared to fall off,
especially around holes; a human is much more confident". Before proposing, two claims were checked:
* `THROTTLE_FLOOR = 0.90` is real (`nav/model.py`) but barely binds: mean throttle in real rounds 0.98,
  above 0.95 on 82 % of KOTM decisions. Not a lever.
* The explicit brake action is 0.09 % of training decisions and 0 % of real-round decisions (current, not
  stale). But the human's 14.3 % is a DERIVED number (intent opposing velocity). Measured the same way
  (command vs velocity while moving above 3 u/s): **agent opposes its own motion on 42 % of decisions,
  human 33 %; strongly (beyond 120 deg) 28 % vs 20 %; median angle 72 vs 57 deg.** The agent does not
  lack braking; it hesitates by thrusting backwards and sideways. The brake-cost and throttle-floor
  ideas were dropped on that evidence.

Applied in `nav/waypoints.py` (from the 14,255 weights, `nav_eval_timeprice_0142.pth`; the reverted-run
endpoint 14,600 is kept as `nav_revert_end_0721.pth`):
1. **`edge_time_cost` v3: stopping-distance test.** Charge only inside `v^2 / (2 * EDGE_DECEL) +
   EDGE_MARGIN_U` of a lip along the velocity (EDGE_DECEL 13, measured 13.9 u/s^2 under full reverse for
   both agent and human; margin 1 u). At 8 u/s the charge now starts 2.5 u out instead of 9.6 u. Fires on
   6.3 % of the human's decisions and 2.7 % of the agent's (was 13.4 % / 10.4 %); the goal and jump
   exemptions are unchanged. `EDGE_T_SAFE` is no longer used.
2. **`PICKUP_GRACE = 16`**: for ~1 s after a pickup, negative progress is forgiven (TIME and the speed
   bonus still charge). Removes the break-even that made stopping at the gem optimal (28.3).
Not changed: FALL 25, GEM_SPEED_BONUS 8/60, GROUP_LINK_DMIN 3, pool 2/5/1, ordering K=1.6 at inference.

Readout tool committed: `python -m nav.leg_report [trace.csv]` prints pickup speed at ring gems, the
post-pickup dip, time to regain 8 u/s, turn at pickup, thrust-opposes-velocity share, block speed, falls,
for the trace and the human demo side by side. Baselines on the 14,255 weights (K=0 trace): ring pickup
4.50, ring->out dip below 2 u/s 53 %, thrust opposes 42 %, block 4.6 u/s, 16 falls / 8 rounds.
Judge by `eval_cycle.ps1` (ordering on, compare against 113-118 points and 16 falls) at ~+300 and
~+1,000 updates. Step 3 (dense thrust-along-motion bonus, k ~0.1) waits for that result.

### 28.11 STEP 3 APPLIED: dense commitment bonus (2026-09-22 14:10). Plus the results that led here.

**Edge-fix run results (28.10, weights 14,255 -> 15,285, ~1,030 updates), 8-round KOTM evals with
ordering on:** 14,565: 114.1 points, 1.97 s/pickup, 22 falls. 15,057 (noon): rounds 121,105,122,124,
125,121 then 71 and 43, because the marble FROZE at (-35, 11) for 54 s and 86 s (on the floor, full
command magnitude, heading swinging randomly, next gem 12 u east across the SW hole, route 4 u north).
Earlier checkpoints never had a gap over 9 s in a round. 15,285 (13:41): **120.9 points**
(121,115,120,120,121,125,125,120), **1.86 s/pickup** (median 1.73), 15 falls, longest gap 11 s. Best
8-round result to date; the training-side KOTM s/pickup sat at 1.73-1.77 the whole run. FlatGem control
at 15,057: 95.5 gems, unchanged (96.5 before), no freezes. The hesitation readouts did NOT move: centre
pickup speed 4.4 u/s (human 8.2), thrust-opposes-velocity 41 % (human 33 %), block speed 4.6 (human 6.9).
The gain came from the pickup terms, not from confident driving.

**Operator directive:** seconds between pickups is the headline metric (human 1.57; 150 points needs
~1.53). It is logged as `sgem=` on the MAPS/NAV lines (pickup-to-pickup inside a group), on the dashboard
(first chart, per map, with a map selector and time window), and printed first by `nav/leg_report.py`.
The dashboard also has a per-map speed heat map (green fast, red slow) from `logs/nav/real_trace_<map>.csv`,
which `real_run.py` now writes; the KOTM map shows red at the centre block, every junction, and the
outer gem spots, green on straight walkways.

**Applied now:** `ALIGN_BONUS = 0.10`, `ALIGN_V_REF = 8`, `ALIGN_V_MIN = 2` in `nav/waypoints.py`: per
decision on the floor, `0.10 * min(1, v/8) * max(0, cos(thrust, velocity))`. Restarted 14:10 from
`nav_before_align_1405.pth` (= 15,275). Evals scheduled at 15,575 (`align1`) and 16,175 (`align2`).
Readouts to move: thrust-opposes share 41 % -> 33 %, centre pickup speed, s/pickup; falls must not rise.

### 28.12 PENDING: entropy-controller fix, to apply after the align2 eval (operator, 2026-09-22 14:30)

Measured on the 14,255 -> 15,285 lineage: entropy cycles 0.13 <-> 0.43 with a ~165-update period; the
bang-bang controller's coefficient swings -0.03 <-> +0.07 and is NEGATIVE on 45 % of updates (an entropy
penalty). KL (0.019) and clip (0.16) are healthy, so it is not what blocks improvement, but it pushes a
hesitant policy toward determinism half the time and puts every A/B eval on a different cycle phase.
Fix (operator-approved, to run right after the `align2` eval at update 16,175 restarts training):
`python logs/nav/apply_entropy_fix.py` then restart the trainer. It replaces `_adapt_entropy` with a
proportional rule toward ENT_TARGET 0.30 (gain 0.002/update, clamp [0, 0.02], never negative) and starts
the coefficient at 0.005. Backup of the file before: `logs/nav/ppo_recurrent_backup_pre_entropyfix.py`.

### 28.13 JUMP RE-ENABLE + CURRICULUM + 7/1 POOL (2026-09-22 20:50). Also: the align run, and where it left us

**Align run result (28.11):** at update 15,590 the 8-round KOTM eval gave **123.1** (128,121,120,122,122,
120,129,123), 1.82 s/pickup, 10 falls; a second eval of the same weights 121.9 / 1.83 s / 13 falls. Best
checkpoint to date, **`nav_eval_align1_1536.pth`**, now `nav_latest`. The run was stopped at 15,870
(`nav_align_end_1655.pth`) after the operator watched it at 1x: still slow and hesitant, never jumping
gaps or cutting corners. The commitment bonus did not move the thrust-opposes share (41 % throughout).
Checked and ruled out: the section 21 weight imbalance (goal path 10.97 vs hidden path 10.38 into the
direction head, ratio 0.95; aim scatter unchanged and within a few degrees of the human inside 12 u).
Entropy fix (28.12) APPLIED: proportional controller, target 0.30, gain 0.002, coefficient in [0, 0.02].

**Diagnosis behind this step.** Jumping had been trained out of the policy on purpose (JUMP_DAMP 3.0 on
2026-09-19 to cut falls): 0.1 % of real-round decisions, 0 % in the centre block, 0.22 takeoffs/min vs the
human's 0.87. A hop over the centre hole is 5.7 u against 12.5 u round it. Worse, the walk graph could
never route through a jump on KOTM: jump edges were DOUBLE-COUNTED (each gap found from both edge cells,
coo_matrix summed the duplicates) and JUMP_COST was 3.0, so a 3 u hop cost 18. With jumps never on the
shortest path, PROGRESS never paid for one and no goal ever required one.

**Applied, from the 15,590 weights:**
* `terrain.py`: jump edges deduplicated (KOTM 228 -> 114 real edges), `JUMP_COST 3.0 -> 1.5`,
  `goal_field(..., jumps=False)` for the walk-only distance, and a jump curriculum in
  `_sample_spawn_goal`: with `JUMP_GOAL_P = 0.33` prefer a spawn whose jump route saves >= 1 u
  (NW -> SE centre gem now saves 2.6 u; the sampler picks jump-saving goals on ~1/3 of draws).
* `model.py` `JUMP_DAMP 3.0 -> 1.0`; `waypoints.py` `JUMP_TAKEOFF 0.4 -> 0.1`; `ppo_recurrent.py`
  `DISCRETE_ENT_COEF 0.002 -> 0.01`. FALL stays 25. Falls WILL rise in training while it practises.
* Pool **7 KOTM / 1 Islands** (FlatGem dropped: 93-95 % of human, unchanged by every change today;
  Islands kept for its 854 jump edges). Defaults in `start_training.ps1` and `eval_cycle.ps1`.
* Dashboard: jump heat map (takeoffs as % of floor decisions per cell, green high, red none) next to the
  speed heat map, both following the map selector. `leg_report` prints takeoffs/min and the share of
  centre pickups preceded by airborne (human 16 %).
Evals scheduled at 15,890 (`jump1`) and 16,490 (`jump2`). Readouts: takeoffs/min (0.22 -> toward 0.87),
s/pickup (1.83), falls per 8 rounds (10-15), and whether centre-to-centre legs start using the hole.

### 28.14 OVERNIGHT 2026-09-22/23: stay on the jump track, goal 150 on KOTM (operator, 22:30)

jump1 eval (update 15,900, ~310 updates on 28.13): 119.3 points (121,122,114,121,114,117,128,117),
1.88 s/pickup, 17 falls, **0.95 takeoffs/min (was 0.22; human 0.87)**, 9 of 23 takeoffs in the centre
block, 108 airborne decisions over the centre hole (was ~0). The move is back; the timing is not yet paid
for (score/pace within noise of the 15,590 best, falls slightly up). Entropy controller: rose to 0.46
with the coefficient at its 0.02 cap, then the coefficient went to 0 and entropy is easing back (0.40).
Operator: keep this track all night; if it has not paid by morning, consider model-based alternatives
(measured jump envelope fed to the gap prior and the routing graph; or a learned dynamics model with
short-horizon jump planning). `logs/nav/wait_and_eval_loop.sh` runs an 8-round eval every 600 updates
(jump2 at 16,490, then jump3, ...), training restarts itself after each; per-eval traces saved as
`real_trace_jump<N>.csv`. Plan step 4 (restore time pressure gradually) is to be applied once an eval
shows takeoffs holding with falls back at <= 13 per 8 rounds and s/pickup not worse.

### 28.15 jump2 eval (update 16,500): the standoff freeze again, and jumps fading (2026-09-23 01:05)

jump2: 103.6 mean (107,110,122,125,**25**,106,119,115), 1.92 s/pickup, 18 falls, **takeoffs 0.46/min**
(jump1 had 0.95). Round 5 = the noon standoff again: from 38 s to the end (143 s) the marble sat at
(-34.3, 21.7), inner edge of the west walkway, on the floor at ~1 u/s of jitter, last gem of the group
12 u east across the NW hole, walkable route a few units north. Without that round: 114.9. The
pickup-gap statistic missed it (no pickup followed), so `leg_report` now prints the longest stall per
round. Takeoffs halving between jump1 and jump2 says the reward is currently pricing the practised jumps
as net-negative (they still end in falls or cost time); if jump3 shows the same, the curriculum share or
the gap prior needs raising rather than waiting. Stuck-breaker for real runs: see 28.16.

### 28.16 STUCK-BREAKER in real runs: +16 points on the same weights (2026-09-23 01:05)

`nav/real_run.py`: if the marble moved < `STUCK_U` (1 u) in the last `STUCK_S` (3 s) while a target
exists, reset the recurrent state, steer at the string-pulled path waypoint instead of the straight line
to the gem, and SAMPLE actions instead of taking the mean, for `STUCK_DETOUR` (24) decisions.
`NAV_STUCK_S=0` disables. Validation on the jump2 weights (16,500): 103.6 -> **120.1** (121,117,123,121,
121,112,124,122); the 147 s and 22 s standoffs of the unassisted run did not recur. 13 triggers in 8
rounds, several on ordinary short hesitations (harmless: 1.5 s of sampled steering). `leg_report`
prints the longest stall per round by displacement (moved < 1 u in 3 s), which catches a freeze even
when no pickup follows it. Default ON for every real run from here, including the eval cycle.

### 28.17 PRACTICE DISCOUNT on jump falls (2026-09-23 01:15)

Takeoffs/min across the jump run's evals: 0.95 (15,900) -> 0.46 (16,500) -> 0.33 (16,500 with breaker).
The reward was extinguishing the move: a hop saves ~2.6 u of progress, a failed hop cost FALL 25, so the
expected value is negative unless 9 in 10 land, and the policy never got enough practice to reach that.
Applied: `FALL_AFTER_JUMP = 8.0` for a fall within `JUMP_FALL_WINDOW = 24` decisions (~1.5 s) of a real
takeoff (any takeoff; the gap prior is trainer-side and not available in the worker). Takeoff and airborne
costs unchanged. Trainer restarted 01:16 from nav_latest (~16,560). REVERT to FALL for jump falls once
takeoffs/min holds >= 0.8 with falls <= 13 per 8 rounds; the eval loop (jump3 at 17,090, ...) continues.

### 28.18 jump3 + step 4 applied (2026-09-23 03:45)

jump3 (17,095, ~590 updates on the practice discount): 117.8 (121,118,117,118,118,116,115,119), 1.89
s/pickup, **9 falls** (lowest ever), takeoffs 0.50/min (recovering from 0.33), no stall over 4 s. The
discount stopped the extinction; the policy is now over-safe (speed 6.3, centre pickups 4.8 u/s).
Applied plan step 4 at half strength: `GEM_SPEED_BONUS 8 -> 16` (slope 0.27/decision; 0.5 tripled
falls on 2026-09-22 with the fear terms still present). Restarted from 17,095. Judge at jump4 (17,690):
revert to 8 if falls > 13 per 8 rounds without a pace gain.

### 28.19 jump4 (2026-09-23 06:15): jumping at the human's rate, not yet paying

jump4 (17,695, ~600 updates with GEM_SPEED_BONUS 16): 120.1 (121,126,121,118,118,123,120,114), 1.86
s/pickup (median 1.73), 16 falls, **takeoffs 1.03/min** (human 0.87), jump held 1.24 % of decisions
(human 1.8 %), no stall over 3 s. Falls back at the pre-jump level (not beyond), small pace gain over
jump3, so the bonus stays at 16. Overnight sequence of 8-round evals on this lineage: 119.3 (15,900) ->
103.6/120.1 with breaker (16,500) -> 117.8 (17,095) -> 120.1 (17,695). The move is learned; it is not
yet shortening legs. Morning decision for the operator: keep practising on this track, or start the
model-based alternative (measured jump envelope -> gap prior + routing graph; HANDOFF 28.14).

### 28.20 jump5 and STOP (2026-09-23 08:52). Operator gate: "not serious progress -> stop"

jump5 (18,295): 120.3 (119,122,118,120,125,115,125,118), 1.86 s/pickup (median 1.73), 19 falls,
takeoffs 1.66/min, no stalls. Below the 125 gate -> training stopped 08:51 (trainer, loop, games,
watchdog, eval loop all down). The jump lineage's five evals: 119.3, 120.1 (with breaker), 117.8,
120.1, 120.3: flat at 118-120, 1.86-1.89 s/pickup, while takeoffs went 0.22 -> 1.66/min and the
hesitation readouts never moved (centre pickup 4.8-4.9 u/s, thrust-opposes 42 %).

**Best verified checkpoint remains `nav_eval_align1_1536.pth` (update 15,590): 121.9-123.1 points,
1.83 s/pickup, 10-13 falls**, to be run with the ordering term (K=1.6) and the stuck-breaker, both
defaults in `nav/real_run.py`. `nav_latest.pth` is the 18,295 jump-lineage endpoint. The lineages differ
in reward constants (see 28.13, 28.17, 28.18); to resume the best checkpoint's own settings, see 28.11.

**Where the remaining ~25 points are, unchanged by two days of reward work:** the policy decelerates
into and out of pickups and near edges (centre pickup speed 4.8 vs human 8.2; thrust against velocity
42 % vs 33 %), and its jumps, now frequent, do not yet save time. Next candidate (operator's morning
choice, HANDOFF 28.14): the model-based route, starting with a measured jump envelope fed to the gap
prior and the routing graph, then a learned short-horizon dynamics model for jump timing.

### 28.21 TRAP FIXED: watch mode ran 4x from round 2 on (2026-09-23 09:15)

Operator: "the game is running at 3x speed and is falling a lot". Measured: Python paced 15 decisions
per wall second (correct), but rounds 2+ lasted 0.75 min with 705 decisions (round 1: 3.01 min, 2,826).
Cause: `env.set_speed()` re-sends the training setup (`FIXEDSTEP 64`) at the end of `wait_new_round()`
(and on connect), undoing watch mode's `FIXEDSTEP 16`; `real_run` kept sending `VIEW_SUBSTEPS` = 4
slices per decision, so each decision covered 256 ms of game time from round 2 on. The game looked 4x
fast and the policy, acting at a quarter of its trained rate, fell 11 times a round. Every multi-round
1x watch run before this had the same defect after its first round (single-round watch runs were fine;
all 8-round EVALS are unaffected, they do not use sub-steps). Fix: `apply_watch()` in `real_run.py`,
called at start and after every `wait_new_round()`.

### 28.22 NEXT-GEM PROGRESS TERM (2026-09-23 09:40). Operator's insight, confirmed by ablation

Operator, watching 1x: the marble takes gem 1, nearly stops, then accelerates to gem 2, even when both
lie on a straight walkway; a player who saw both would blast through gem 1 at full speed.
**Checked:** the observation DOES carry the next gem (NEXT_DIM block since 2026-09-19; real runs pass
the chooser's second pick, training the nearest remaining of the group), and hiding it costs 8 points
(align1 checkpoint: 121.9-123.1 -> **114.0**, 1.83 -> 1.96 s/pickup with `NAV_NO_NEXT=1`). So it is
used, but weakly: with the next gem within 20 deg the human takes gem 1 at 11.3 u/s and holds 7.1; the
agent takes it at 5.8 and dips to 4.6 (21 % of those legs below 3 u/s). Round a corner: 3.1 vs 5.6.
**Why:** no reward ever paid for carrying speed through gem 1 toward gem 2 (PROGRESS is to the current
gem only; ARRIVE is exit-blind; CARRY was a null lump sum).
**Applied:** `PROGRESS_NEXT = 0.3` in `nav/waypoints.py`: every decision also pays 0.3 x the change in
walking distance to the NEXT gem (its own Dijkstra field, `_set_next_field`, rebuilt on every goal
change; forgiven-negative during PICKUP_GRACE; off while a fall mark is active). On a straight line both
terms pay 1.3/u; on a corner the approach that already curves toward gem 2 is paid during the approach.
Started 09:37 from the jump5 endpoint (18,295; snapshot `nav_before_next_0935.pth`), all other settings
as in 28.13/28.17/28.18. Evals `next1` at 18,895 then every 600 (`wait_and_eval_next.sh`). Readouts:
straight-line dip (min speed in the 1 s after a pickup with the next gem within 20 deg: 4.6 -> toward
7.1), s/pickup (1.86), points, falls.

### 28.23 next1 (2026-09-23 12:45): PROGRESS_NEXT WORKS. New best: 126.6

next1 (update 18,900, ~610 updates on PROGRESS_NEXT 0.3): **126.6** (125,126,125,128,127,128,123,131),
**1.79 s/pickup** (median 1.73), 11 falls, mean speed 7.06, takeoffs 1.03/min, no stalls. Straight-line
pickups (next gem within 20 deg): speed at gem 1 5.8 -> **7.0**, min speed after 4.6 -> **5.8**,
near-stops 21 % -> **7 %** (human 11.3 / 7.1 / 2 %). Outer-gem pickup speed 5.57 -> 6.93; centre gems
still 5.1. Best 8-round result of the project and the tightest spread. Checkpoint copied to
`nav_best_next1_18900.pth`. The operator's diagnosis (28.22) was the right one: the input was there,
the reward for using it through the pickup was not. Training continues; next2 at 19,495.

### 28.24 next2 (2026-09-23 15:36): 121.8, falls doubled

next2 (19,505): 121.8 (117,125,121,122,125,122,127,115), 1.83 s/pickup (median 1.73), **21 falls**
(next1: 11), takeoffs 1.45/min, straight-line near-stops 2 % (human 2 %), centre pickup speed 5.5. The
pickup mechanics held; the extra ~10 falls (~35 s per 8 rounds) are the whole difference from next1.
Likeliest source: the rising takeoff rate under the practice discount (FALL_AFTER_JUMP 8). Plan: leave
settings for next3 (20,095); if falls stay > 15 there, restore FALL_AFTER_JUMP to 25 (the jump rate is
already above the human's). Best remains `nav_best_next1_18900.pth` (126.6).

### 28.25 TRAINING PARITY: real-gem continuous training (2026-09-23 16:41). Operator's call

**Diagnosis (16:00).** Routing is at parity with the human (travelled/optimal 1.05 both); the gap is pace,
and the training world explained the pace: the last gem of every synthetic group was a TERMINAL event
followed by a teleport to rest (TELEPORT_P 1.0, "after arrival always teleport"), so every group began
from rest and every last gem was approached as an episode end; EDGE_K charged the centre block for being
fast rather than for falling; FALL_AFTER_JUMP 8 doubled falls at next2. The operator: "bring training to
parity with the 1x speed version I've been watching". Approved and applied:

1. `nav/waypoints.py`: `EDGE_K 0.0`, `FALL_AFTER_JUMP 25.0`, `CONTINUOUS = True` (synthetic mode only:
   a finished group rolls into a new one 30-60 u away via `_chain_group`, no episode end, no teleport),
   `_record()` factored out, and a **real_mode** API: `begin(env, real_goal=, real_next=)` (no teleport,
   one-gem segment), `retarget(x, y, goal, next_goal)`, `set_next(x, y, next_goal)`, `next_goal()` returns
   the chooser's next gem, `step(..., picked=gem_delta)` where the GAME's pickup is the arrival (+ARRIVE
   +GEM_SPEED_BONUS, PICKUP_GRACE, no done), timeout = 312 decisions on ONE gem. Backup of the
   pre-change file: `logs/nav/waypoints_backup_pre_continuous.py`.
2. `nav/gems.py` (new, torch-free): `visible_gems(raw)` and the sticky momentum-aware `choose()` used by
   both `real_run.py` and the workers, so training chases exactly what a scored round chases.
3. `nav/vec_worker.py`: `NAV_REAL_GEMS` (default 1, on when the terrain has `gem_spawns`). Each decision
   after the game step the worker re-runs the chooser on the observer's gem list, retargets the segment
   (and the on-screen mark) when the target changed by > STICKY_TOL, updates the next-gem block
   otherwise, keeps the old goal while blind. `gem_delta` and `fell` are summed per round and every
   round end logs `[i] GAME map=<map> points=<p> gems=<g> falls=<f> rtf=<r>`: a training round IS a real
   KOTM round (sampled policy, no stuck-breaker, 3x speed), so no separate 8-round evals are needed.
4. `nav/dashboard_nav.py`: gauges Reward/segment, KL/clip, Env warts removed. New chart after
   seconds-between-pickups: POINTS PER TRAINING GAME, one bar per GAME line of the selected map, window
   by update, 10-game mean line, human 143.5 dashed. Refreshes with the 3 s push.

Patch scripts (all asserted on the pre-change text, run once): `logs/nav/apply_continuous.py`,
`apply_realmode_waypoints.py`, `apply_realmode_worker.py`, `apply_dashboard_games.py`.

**Run:** 16:41 from `nav_best_next1_18900.pth` (126.6) copied over `nav_latest.pth`; the 19,505 endpoint
saved as `nav_pre_realgems_19505.pth`. 7 KOTM / 1 Islands, watchdog only, NO eval loop (operator rule:
the GAME lines are the score). First lines: Islands 45 pts (partial round), KOTM inst 2 108 pts / 87
gems / 4 falls. Sampled-policy training rounds score below a deterministic eval of the same weights, so
compare GAME bars against each other, and confirm any candidate with `real_run.py` before calling it a
best. Readouts to judge by: the GAME bars' 10-game mean on KOTM, falls per round, sgem.

**28.25 follow-up (17:00).** Two dashboard complaints: gaps between bars (bars were placed by update, and
rounds end in bursts every ~100 s wall = 5 updates) and a 222-point bar (177 gems = two rounds merged:
the RoundOver paths inside `start_segment()` and the recover branch restarted without logging or resetting
the counters). Fixed: `InstanceWorker._round_over()` is the single log-and-reset point for all three
paths (live from the next trainer restart; the workers now running still have the old code), the chart
plots bars in game order with the window still filtering by update, and the dashboard drops entries with
> 130 gems as merged rounds. Hover shows the points only; no bar labels (operator request).

### 28.26 parity1 (2026-09-23 20:52): NEW BEST 131.5 after 4 h of real-gem training

Eval `nav_eval_parity1b_2048.pth` (update ~19,640, 740 updates of real-gem continuous training from
18,900): **131.5** (137,116,131,138,127,129,137,137), 1.71 s/pickup (median 1.60), 18 falls, mean speed
7.42. Copied to `nav_best_parity1_19640.pth`. Sampled training rounds over the same period: 100-round
means 116 -> 125 in two hours, flat at ~125 since 18:46 (sampled policy scores ~6 below deterministic).
Falls are now the variance source (the 116 round had 6). First eval attempt (parity1, 20:46) died 7 s
after launch before its first print with nothing in stderr; rerun worked. Likeliest a native crash while
the GPU was being released; `real_run.ps1` now prints the navigator's exit code.

### 28.27 MEASURED JUMP ENVELOPE in the graph, the observation and the prior (2026-09-23 21:33)

**Operator's three observations after watching 1x (131.5 checkpoint):** (1) training scores below 1x
scores, (2) the marble still rolls into the centre holes, (3) it rolls around corners instead of jumping
across. Findings: (1) training bars are a SAMPLED policy (~6 points under deterministic); 1x vs 3x is two
rounds vs eight, unproven. (2) the .dif has vertical walls on the four 7 x 7 holes (a 45 deg bevel only on
the 2 u centre hole); the crop shows them correctly; EDGE_K 0 (28.25) removed the only speed-near-lip
charge and falls went 11 -> 18. (3) STRUCTURAL: `nav/gap_map.py` (new; `python -m nav.gap_map <map>`,
PNG in logs/nav/) showed the walk graph's jump edges were 8 compass directions, <= 4 u: on every 7 u hole,
bay and notch only diagonal corner cuts existed, never straight across, so PROGRESS paid to walk around.
Nothing anywhere computed a gap from a position along a heading.

**Measured (`python -m nav.measure_jump`, game as oracle, 16 ms step, flat east walkway x=-12):**
apex 1.33 u and flight 0.75-0.77 s at every speed; range = 0.75 v + 2.0 u (5.05 -> 5.95, 9.52 -> 8.85,
13.29 -> 11.71, 17.16 -> 14.67). Identical repeat trials: the sim is deterministic. Table in
`physics/jump_envelope.json`, raw rows in `logs/physics/`. (First two attempts failed: jump pressed while
still airborne after the teleport, then a walkway that turned out to be a ramp; both fixed in the script.)

**Applied (patch `logs/nav/apply_measured_jumps.py` + follow-ups):**
* `nav/physics.py` (new): `jump_range(v)`, `crossable(gap, v)`, `speed_for_gap`, `MAX_JUMP_GAP` = range at
  CRUISE_SPEED 8 minus LANDING_MARGIN 0.5 = 7.0 u. Describes the marble, so it holds on any map.
* `nav/terrain.py`: jump edges now from `_measured_jump_edges()`: from every edge cell, 16 headings on the
  FINE 0.5 u map, lip within 1.5 u, first floor beyond within MAX_JUMP_GAP with dz in [-JUMP_DROP,
  +JUMP_RISE]; cost gap x JUMP_COST 1.2 (was 1.5: a 10 u hypotenuse must beat 14 u of legs). KOTM: 228 ->
  1593 edges; west-to-east across a hole 10.4 u (walk-only 20.7). `jump_edge_cells` kept for gap_map.
* `terrain_obs.py`: `gap_along(x, y, z, ux, uy)` -> (lip, gap, landing dz) along one heading.
* `nav/obs.py`: NAV_OBS_V3, VEC_DIM 53 -> 56: along the marble's VELOCITY heading (waypoint bearing when
  < 1 u/s): lip distance /20, gap length /20 (1 = none), crossable at the CURRENT speed per physics.
* `nav/model.py`: the jump prior's landing test is now `old short-hop crop test OR vec[VEC_GAP+2]`, so it
  fires for a 7 u hole at 8 u/s and not at 5 u/s.
* Checkpoint: `logs/nav/migrate_obs_v3.py` expanded `nav_best_parity1_19640.pth` with zero columns (model
  + Adam moments) -> `nav_v3_start_19640.pth` = `nav_latest.pth`; behaviour identical at step 0. The V2
  best files can only be evaluated with V2 code (git `709c01ac3` + 28.25 patches).

**Run:** 21:33, real-gem mode, 7 KOTM / 1 Islands, watchdog (via the session's background shell: the
detached Start-Process bash launch silently fails now), no eval loop. Judge by the GAME bars, falls per
round and takeoffs/min (`nav.leg_report`). Expected first effect: more takeoffs at holes/bays; falls may
rise before the speed gating is learned. Not yet done: EDGE_K restore (awaiting the operator), 1x vs 3x
8-round test.

### 28.28 OVERNIGHT 2026-09-23/24: run to 150, operator rules

Operator 22:40: "keep on this track all night, try to get this to 150 overnight; you are allowed to tweak
the jump physics implementation if there's a bug or obvious improvement." Rules in force: training never
stops except for a deliberate real eval of a candidate; judge by the GAME bars (sampled policy, ~6 under
deterministic); no periodic evals; jump-physics changes only for a bug or an obvious improvement, logged
here. `logs/nav/game_summary.sh` prints a 15-minute summary and snapshots `nav_latest.pth` to
`models/nav/nav_night_<upd>_<mean>.pth` on every new 50-round high (`logs/nav/night_best.json`).
Baseline at the start: measured-jump run from 19,640, first hour 126.6 mean / 1.9 falls per round
(previous plateau 125 / 2.3). Best verified: `nav_best_parity1_19640.pth` 131.5 (V2 obs).

### 28.29 JUMP REWARD BUG: airborne hold + landing clip (2026-09-23 22:46)

Audit of the reward along a hole crossing (simulated on the NW hole, west to east, gem 2 u past the far
lip): over the void `dist_at` has no field value under the marble and falls back to a neighbour or
straight-line + 5, so the distance sat at ~13.4 for the whole flight and dropped 6.7 in ONE decision at
the far lip, where PROGRESS_CLIP 3 cut it. A 7 u crossing earned 7.5 of its 11.2 u of progress; walking
earns 1 per u. Jumping was taxed 33 % against walking, on top of the fall risk. Fix in `nav/waypoints.py`:
`Segment.air_hold`; while `airborne and not terrain.walkable_at(x, y)` prev_d (and prev_dn) are HELD; the
first decision back over walkable floor pays the full gain with `JUMP_PROGRESS_CLIP = 12` (the normal
clip still applies everywhere else; respawns go through fall_mark, teleports do not happen in real-gem
mode). `TerrainGrid.walkable_at` added. Unit test: crossing now earns 11.0 (13.6 -> 2.4 u), flat walking
of 12.6 u earns 13.2. Trainer restarted 22:46 from `nav_latest.pth` (update 19,860); the night snapshot
`nav_night_19852_127.pth` predates it.

### 28.30 crossable flag counts the run-up (2026-09-24 04:34)

Six hours after 28.29 the sampled 50-round mean sat at 128-131 (highs 127.0 -> 131.6, snapshots
`nav_night_*.pth`). Trace audit of the 7 KOTM instances (last 900k rows): deliberate jumps over voids
(takeoff within 0.5 s before entering the void) succeeded 43 % at 3-4 u, 32 % at 4-5 u, 22 % at 5-6 u,
53 % at 6-7 u, with takeoff speeds 6.5-8.5 u/s; most void "crossings" are height drops without a jump.
Cause found in `nav/obs.py`: the crossable flag tested `crossable(gap, speed)` and ignored the distance
to the lip, while the prior fires up to JUMP_PRIOR_DIST 2.5 u before it: a jump from 2.5 u back over a
7 u hole needs 9.5 u of range, which 8 u/s (7.5 u) does not have. Now `crossable(lip + gap, speed)`: the
flag turns on only when the remaining run-up plus the gap fits the range at the current speed, which
also times the takeoff. Unit test on the NW hole: at 8 u/s the flag is off 1-3.5 u before the lip (it
needs ~8.7 u/s at 1 u); corner cuts (4-5 u) are on at 7 u/s within 1.5 u. Trainer restarted 04:34 from
`nav_latest.pth` (update ~20,900).

### 28.31 Overnight result as of 07:05 on 2026-09-24 (run still going)

Sampled 50-round means on KOTM: 126-131 from 22:46 to 04:34 (28.29 fix), then **131-134** since the
04:34 run-up flag fix (28.30), falls 1.5-2.4 per round, best single training round 149. Snapshots of
every new 50-round high are in `models/nav/nav_night_<upd>_<mean>.pth` (`logs/nav/night_best.json`
names the current one). The morning job: 8 deterministic real rounds on the latest night snapshot (and
on `nav_latest.pth`) with `eval_cycle.ps1`; sampled 134 has corresponded to ~140 deterministic, so a
new verified best above 131.5 is expected. Unchanged tonight: FALL 25, EDGE_K 0, entropy controller.
Not done: EDGE_K restore (operator's call), 1x vs 3x test.

### 28.32 MORNING EVALS 2026-09-24: 144.9 at 1x, 142.5 at 3x. NEW BEST, above the human mean

`nav_best_night_20945.pth` (= `nav_night_20949_135.pth`, update 20,945, snapshotted at a sampled
50-round mean of 134.8):
* 1x watch mode, 8 rounds (08:05): **144.9** (144,150,151... exact: 144,150,140,151,141,145,142,146),
  1.53 s/pickup (median 1.47), 9 falls, speed 7.66, takeoffs 0.37/min.
* 3x, 8 rounds (08:55): **142.5** (142,142,145,144,145,140,139,143), 9 falls, speed 7.84.
Both above the previous best 131.5 (19,640) and level with the human 143.5. 1x vs 3x: 2.4 points apart
on 8 rounds each, inside round-to-round spread; no evidence of a speed-setting effect. Endpoint of the
night's run is `nav_night_end_morning.pth` (update ~21,4xx). Training is STOPPED pending the operator.

### 28.33 Jump-on-prior inference test (2026-09-24 13:13): 117.1, REJECTED

Why: the deterministic policy never jumps (jump head ~-7, prior +6, damping -1: logit stays < 0; 893
prior firings -> 1 jump in the 3x eval). Operator asked what forcing the jump when the prior fires is
worth. `NavActorCritic.JUMP_ON_PRIOR` (env NAV_JUMP_ON_PRIOR, default 0) makes the deterministic action
jump whenever `gap_prior` is 1. Same checkpoint 20945, 3x, 8 rounds: **117.1** (112,131,114,125,123,120,
103,109), **56 falls** (baseline 142.5 / 9 falls), 1.90 s/pickup, takeoffs 10.5/min. Of 253 forced
takeoffs 99 landed, 52 fell (66 %), 102 never left the floor (pressed at the lip while not in contact or
already committed). Verdict: a 66-71 % success rate is worth about -25 points; the prior is far too
loose to act on directly. Left off. Next lever if jumping is pursued: make approved jumps reliable
first (LANDING_MARGIN 0.5 -> 1.0, tighter lip window), then price approved-jump falls at true cost.

### 28.34 SUSTAINABLE JUMPING: reliability, price, decisiveness (2026-09-24 13:19). Operator's call

Operator: "jumps are necessary to get us to human level play; go ahead with your plan." Applied, in
the order reliability -> price -> decisiveness, all in TRAINING so that what is watched is what is learned:
1. `nav/physics.py` LANDING_MARGIN 0.5 -> 1.0 (MAX_JUMP_GAP at cruise 7.0 -> 6.5 u: KOTM's 7 u holes are
   admitted straight across only above ~9.5 u/s; corner cuts and bays stay). `nav/obs.py`: the crossable
   flag also requires the marble to be over a walkable cell (40 % of the forced jumps in 28.33 were pressed
   already over the void and did nothing).
2. `nav/waypoints.py` FALL_AFTER_JUMP 25 -> 6, now GATED on `Segment.takeoff_approved` (the obs crossable
   flag was on at the takeoff; the worker passes `approved=`). Every other fall, including an unapproved
   jump, stays at FALL 25. 6 ~ the true time cost of a fall in a scored round. Expected value of an
   approved jump: at 80 % success ~ +4 (GEM_SPEED_BONUS pays ~5 for 1.3 s saved, TIME ~1), so the head
   is no longer pushed negative by every attempt.
3. `nav/model.py` JUMP_PRIOR 6 -> 8.5: an approved gap now lifts the logit to ~+0.5 (62 % sampled in
   training, and the deterministic action jumps). JUMP_ON_PRIOR (28.33) stays off.
Started 13:19 from `nav_latest.pth` (= night endpoint, update 21,4xx). Night snapshot state moved to
`night_best_phase1.json`; the summary now snapshots this phase's own highs. Judge by: GAME bars, falls
per round, and (from the trace) approved-takeoff success rate; an 8-round eval when the 50-round mean
holds above 134. Watch for: falls spiking above ~3 per round for more than an hour (then raise
LANDING_MARGIN further or lower JUMP_PRIOR to 7.5).

### 28.35 The prior made binding (2026-09-24 16:10). Operator: "Do it"

Three hours after 28.34 the score was up (sampled 50-round 135-137, falls 1.4-2.4) but the jump stats
were DOWN: approved takeoffs 0.56-0.99/min (night) -> 0.18-0.33/min, crossings with a jump 0.43-0.69 ->
0.20-0.29/min. Trace measurement: the prior was on for 41-50 decisions per instance-minute (5 % of all
decisions) and P(jump | prior on) was 0.5-1.7 %. Two causes: (1) `gap_prior` still ORed the old crop
short-hop test, which fires at any lip with floor within 4.5 u (block-to-ring drop-offs), so the prior
was noise; (2) the head had gone to ~-12, cancelling the +8.5 added before the clamp. Fix in
`nav/model.py`: the prior uses ONLY the physics-approved flag, and it is added AFTER the clamp
(`clamp(head - damp, -7, 3) + gp * JUMP_PRIOR`), so an approved gap is >= +1.5 whatever the head does.
On the current weights: no gap -5.1, approved +4.3 (99 % sampled), short-hop-only -7.0. Restarted 16:12
from `nav_latest.pth` (update ~21,900; copy `nav_pre_prior_fix_21900.pth`). Judge by the dashboard's
APPROVED TAKEOFFS chart: takeoffs/min should rise to the approved-decision rate and the success % is
the number that decides whether the plan holds (below ~60 % for an hour: widen LANDING_MARGIN to 1.5).
Deterministic evals will now jump at approved gaps; compare against 142.5 (3x) / 144.9 (1x).

### 28.36 HUMAN DEMO vs MODEL: the graph never valued the cuts (2026-09-24 21:28-21:34)

Operator watched `nav_night_22680_138` at 1x (rounds 152, 144, 150: first real 150s) and saw no corner
cuts, only pointless hops at the centre. A recorded demo round (`demos/demo_20260924_212808.npz`, 160 pts
in 2.7 min, analysed by the new `python -m nav.demo_jumps <npz>`): **17 gap jumps, all landed, 6.3/min,
takeoff 8.4-14.8 u/s (mean 10.7), takeoff-to-landing 9-15 u (mean 10.8), walking detour saved 7.3 u each
(~125 u, ~15 s, ~10 points per round). The physics flag would approve 14/17 at the human's speed; the
training graph had an edge for 2/17** (MAX_JUMP_GAP 6.5 at CRUISE 8). The model at the same spots: 6.8 u/s
(range 7 u), 24 takeoffs in 2,367 visits. Closed loop: slow because the cut is not rewarded, not rewarded
because the graph assumed slow. The centre hops: the 2 u centre hole was an approved in-graph crossing.
**Applied:** `nav/physics.py` CRUISE_SPEED 8 -> 11 (MAX_JUMP_GAP 6.5 -> 9.0; KOTM edges 938 with gaps up
to 8 u, the 7 u holes straight across included), MIN_JUMP_GAP 3.0 (graph and flag); `nav/obs.py`
NAV_OBS_V4, VEC_DIM 57: gap block gains the speed ratio current / speed_for_gap(lip + gap) (/2, 0.5 = just
enough) so the policy can learn to accelerate for a cut ahead. Checkpoint migrated with
`logs/nav/migrate_obs_width.py` (V3 endpoint kept as `nav_v3_end_22700.pth`). The flag is unchanged
(speed-conditioned), the prior binding (28.35): the field now pulls toward the cut, the policy is paid on
landing only if it arrives fast enough, and it can see how much faster it needs to be.
Prior best real rounds before this: 152/144/150 at 1x on 22680 (3 rounds, watch mode).
Started 21:34 from `nav_v4_start_22700.pth` (= migrated V3 endpoint, update 22,715) as `nav_latest.pth`;
phase-2 snapshot state moved to `night_best_phase2.json`. Judge by: APPROVED TAKEOFFS chart (expect longer
crossings and higher takeoff speed over hours), speed at takeoff from `nav.demo_jumps`-style analysis of
the trace, GAME bars. Human reference for the jump metrics: 6.3 gap jumps/min, 10.7 u/s, 10.8 u.

### 28.37 OVERNIGHT 2026-09-24/25: speed nudges under operator autonomy. Target: a 170 single round

Operator 23:25: "you are given autonomy overnight to apply the speed nudges you mentioned as needed" and
"I want a 170 single score by morning" (best single training round so far 155, best real 152).
After 2 h on the carrying-speed graph (28.36): score 136-137 sampled, approved jumps 0.35/min at 83-85 %
success, crossings 8.7 u, but takeoff speed flat at 7.4 u/s (human 10.7). The flag opens only when the
marble is fast; nothing paid for getting fast before the lip.
**Nudge 1 applied 23:34 (this restart): RUNUP_K 0.3** in `nav/waypoints.py`: per decision, while a jumpable
gap lies ahead along the heading (obs speed ratio > 0), the marble is not yet fast enough (ratio < 0.5)
and the lip is within RUNUP_RANGE 10 u, pay 0.3 x speed gained (clipped to 1 u/s per decision; losses
pay nothing). An 8-decision 6 -> 10 u/s build-up earns ~1.2. Worker passes `gap_ratio` and `lip_u` from
the obs gap block. Pre-change checkpoint `nav_pre_runup_23060.pth`.
**Protocol:** every 30 min read takeoff speed / approved rate / success / falls / score (trace analysis +
GAME bars). ~01:30: if median takeoff speed at approved jumps is still < 8.0, nudge 2 = GEM_SPEED_BONUS
16 -> 24 (general pace pressure). Revert any change whose 50-round mean sits below 130 for an hour.
Snapshots of every 50-round high continue (`nav_night_*.pth`, `night_best.json`).
**Nudge 1 result (23:34-01:25):** score rose to a new sampled high, 50-round 140.4 at 23,368 (falls 1.3;
snapshot `nav_night_23368_140.pth`), approved rate 0.3-0.5/min at 76-89 % success, crossings 8.5-8.7 u,
but takeoff speed at approved jumps stayed 7.1-7.6 u/s in every 16-min bucket: the credit did not change
the approach. **Nudge 2 applied 01:23: GEM_SPEED_BONUS 16 -> 24** (pre-change `nav_pre_gsb24_23420.pth`).
Revert trigger unchanged (50-round mean < 130 for an hour).

### 28.38 Overnight result as of 05:50 on 2026-09-25 (run still going, nudges 1+2 in)

Sampled 50-round means on KOTM since nudge 2 (01:23): 137.5, 135.7, 136.9, 138.2, 139.2, **142.2**
(falls 0.8), 140.1, 138.3, 140.5, 140.2, 139.5, 139.7, 139.3, 140.3, 137.8, 139.6, 139.8. Run average
139.3, falls 1.6/round, best single training round 154 (155 earlier in the evening). Best snapshot
`nav_night_23705_142.pth` (142.3 sampled, 0.76 falls). KOTM pace between pickups 1.49 s (human 1.57).
The 170 single-round target was NOT reached. Takeoff speed at approved jumps stayed 7.0-7.7 u/s through
both nudges (human 10.7); approved jumps 0.25-0.5/min at 76-89 % success, crossings 8.1-9.1 u. Neither
the run-up credit nor the bigger speed bonus changed how the marble arrives at a lip; the score gains
came from pace and fewer falls. Morning job: 8 deterministic rounds on `nav_night_23705_142.pth` and on
`nav_latest.pth` (expect ~145-150 real). Next lever for the cuts, not yet tried: a landing bonus scaled
by crossing length, or a curriculum that starts rounds facing a cut (real-gem mode makes the latter hard).

### 28.39 LANDING PREDICTOR replaces the gap-vs-range flag (2026-09-25 morning). Operator's direction

Morning 1x rounds on `nav_night_23705_142.pth` (= `nav_best_1x_150_23705.pth`): 30 rounds, mean ~150,
best 159, first 8: 157,151,148,148,140,148,147,152 (148.9). Operator: the approval must come from
physics ("if the marble is going at x speed on y heading, where will it land, on floor or gap?"), measured
on a flat map, general to any map. Also: two of my earlier explanations were wrong and withdrawn
(in-air steering as the cause of falls: air control is at most 5-7 u/s^2, ~1.4 u over a flight; and a
"committed flight" override, rejected: no control taken from the policy).
**Built (`nav/physics.py`):** `predict_landing(terrain, x, y, z, vx, vy, hold, jump)`: integrate the arc
(VZ0 7.31, G 20, A_AIR 7.0 along the heading when forward is held, R_MARBLE 0.25) over the height map in
16 ms steps, one heights_at call per arc; verdict floor / edge / wall (met a face) / void; landing point,
flight time, drop. `jump_verdict(terrain, ..., hold)`: floor only if the arc from a point WITH floor under
it lands, and so do the arcs shifted +-1 u laterally. `python -m nav.validate_landing` scores it:
* flat strip: predicted range within 0.4 u of the measured at 5-17 u/s (conservative side).
* human demo (17 jumps, all landed): hold=True approves 15/17; hold=False 8/17.
* model 1x rounds (32 takeoffs): hold=True approved-and-fell 3 / refused-but-landed 4 (the 3 falls were
  flights where the policy BRAKED mid-air: receipts in the session log); hold=False 0 / 12.
Root causes of the earlier "approved but fell": the ray flag assumed forward held (3 cases), pressed
jump past the lip with no contact (2), a bounce misread as a jump (1). None was air steering.
**Wired (NAV_OBS_V5, VEC_DIM 58):** gap block = [lip, gap, LANDS-IF-FORWARD-HELD, speed ratio, LANDS-
BALLISTIC]; the prior fires on the hold verdict (index unchanged); MIN_JUMP_GAP still gates (KOTM hack).
Obs build 1.3 ms. Checkpoint migrated: `nav_v4_end_24570.pth` -> `nav_v5_start_24570.pth` = `nav_latest`.
Not yet done: retrain on it (needs the 1x loop stopped), replace the graph's MAX_JUMP_GAP rule with the
predictor, drop MIN_JUMP_GAP once the predictor's field costs make it unnecessary.

### 28.40 The predictor made engine-exact (2026-09-25, ~10:00-12:00)

Operator: "the goal is to be literally as close to deterministic about where the marble will land".
Done: (1) `nav/fine_terrain.py`: a 0.1 u height raster of the .dif floors per map
(`terrain_maps/terrain_fine_<map>.npz`, built on first use; KOTM 476x475 cells); the predictor reads it,
the crop / walk grid stay at 0.5 / 1.0. (2) `predict_landing` integrates the way marble.cc does: 8 ms
sub-steps, velocity then position; VZ0 7.40 (jumpImpulse 7.5 minus the ~0.1 outgoing velocity of a
rolling marble), G 20, A_AIR 6.8 along the heading when forward is held, LAUNCH_DELAY 0.032 s.
(3) `nav/measure_arc.py` captured per-tick trajectories (trainer stopped, one game): 6/8/10/12 u/s x
{forward held, hands off, stick left} + a drop, in `logs/physics/arcs_KingOfTheMarble_Hunt.csv`.
Receipts: apex 1.336-1.346 measured vs 1.340 simulated on all 12 runs; same-height flight 0.736 s vs
0.732; hold-forward ranges 8.55/9.91/11.30 vs 8.56/9.92/11.32 at 8/10/12 u/s; hands-off at 12 u/s 9.39 vs
9.40; lateral acceleration with the stick held 6.8 u/s^2; gravity 20.0 tick to tick. Also learned: the
impulse fires 2 ticks (32 ms) after the key; a landing with NO input bounces (~0.45 restitution), with
forward held it does not (relevant near lips). Verdict scores on the 32 model takeoffs unchanged by the
exactness pass (the errors were inputs and contact, not physics): hold-forward 18/22 landings approved,
3 approved falls (all mid-air braking), ballistic 0 approved falls; human demo 15/17 approved.
The 1x loop on `nav_night_23705_142` ran 78 rounds in the meantime: mean 150.6 (best 159, 129 gems max).
**Training restarted 12:41 on V5** from `nav_v5_start_24570.pth` (= nav_latest, update 24,590): the prior
fires on the hold-forward landing verdict, the ballistic verdict is a second input, run-up credit and
GEM_SPEED_BONUS 24 kept, watchdog + 15-min summary on (phase-3 snapshot state fresh). Judge by: approved
takeoffs/min and their success on the dashboard (expect success to climb toward the verdict's own 85-90 %
as the policy learns to hold forward), takeoff speed, GAME bars. Reference real-round score for the
weights before this change: 150.6 over 78 rounds at 1x.

### 28.41 Afternoon 2026-09-25 on V5: plateau diagnosed as LINES, training stopped 18:05 by the operator

V5 run 12:44-18:05 (updates 24,592-25,520): sampled 50-round means 138-142 the whole time (best 143.4 at
25,237, falls 1.34: `nav_night_25237_143.pth`), jump side: approved takeoffs 0.7-0.85/min (twice the old
flag) at 82-90 % success, crossings 8.3-8.8 u, takeoff speed 6.9-7.2 u/s unchanged. Endpoint
`nav_v5_end_25520.pth`. Operator: "why is the score not going up". Time budget, 78 model rounds at 1x vs
the human demo: time below 5 u/s identical (11.3 vs 11.2 %), falls now ~0.5 s/round, but time at >= 12 u/s
2.7 % vs 12.6 % (mean speed 8.09 vs 8.65). The gap is sustained top-speed LINES through several gems; the
reward pays progress to the CURRENT gem (PROGRESS_NEXT only 0.3) and DIR_GOAL_GAIN 30 drags the heading
onto it every decision, so the policy zigzags at 9-11 u/s. Proposed, awaiting the operator: (1) progress
against a route through the next 2-3 gems with the next-gem weight raised toward parity; (2) ease
DIR_GOAL_GAIN 30 -> ~15; (3) in reserve, a speed premium above 10 u/s. Training STOPPED; nothing running.
Rollback: commit "Baseline before jump physics" + `nav_best_1x_150_23705.pth` (150.6 over 78 real rounds).

### 28.42 Night of 2026-09-25/26: measurements, pre-spin, kotmjump, and REWARD = GAME SCORE step 1

**Measurements (analysis only):**
* Gap crossings >= 3 u of void, per minute: human 6.3 (4.5 with a jump), training 0.12-0.20, 1x inference
  0.00-0.05. Training crosses 2.5-4x more often than inference (sampled jumps; heading noise sweeps the landing
  check), but both are 30-50x below the human. Jump approval fires 1.1-1.3/min in training, 0.25/min at 1x.
* "The model is slow before jumps" was a SELECTION ARTIFACT: the human's 10.4 u/s is the speed at which the
  human chose to jump. Mean speed is equal (human 8.13, model 8.09-8.25). What differs is the top end (>= 12 u/s
  11.3 % vs 2.7-3.4 %) and BRAKING INTO GEMS. Speed at the gem by where the next gem is (ahead / side /
  behind): human 11.4 / 7.1 / 5.4, model 7.7 / 7.6 / 5.8. The model uses one arrival speed. It brakes by
  steering backward (brake key 0 %) from ~5 u out, in training as well. Of ~29 straight-ahead pickups per
  round, ~20 are the 4 centre gems (human 11.1, model 7.4). The game's pickup radius is ~1.0 u (max over 9,272
  pickups: 0.99).
* Heading at 1x (deterministic, nothing sampled): the command's error against the straight line to the gem is
  the same as the human's when the gem is >= 6 u away (median 22 deg vs 21). The command changes 8-12 deg
  PER DECISION (median), where the human holds (0.4-0.9 deg).
* Inference-only ring override (no backward thrust on straight approaches along the outer ring) raised
  ring-row arrival 11.0 -> 12.8 u/s. Score was not measurable in 2-3 rounds. REVERTED at the operator's
  request.

**Smoothing sweep (NAV_SMOOTH, 3x, 8 rounds per arm, nav_night_25237_143):** off 151.2 (1.5 falls/round),
0.75 152.6 (+1.4 +- 2.2, 1.0 falls), 0.5 134.9 (4.4 falls), 0.3 50.2 (28 falls). Acceleration at 8-12 u/s did
not improve with smoothing, so jitter is not what limits acceleration. Not adopted. This 151.2 is the current
3x baseline (pre-spin on).

**PRE-SPIN (operator request, verified by the operator at 1x): mlAgent.cs.** The Ready/Set countdown is now
PLAYED. The marble is in Start mode (marble.cc, mMode == 2: friction and horizontal velocity zeroed, control
torque applied), so input spins it in place and GO turns the spin into speed. Changes:
* onGameStart connects at round load; autoRestart 1000 -> 100 ms; start() skips the 500 ms wait when the socket
  is already up.
* update() allows the countdown via MLAgent::inCountdown(): $Game::State in start/ready/set, the marble not at
  the origin, 6 settle ticks.
* checkDone's time test only fires after the clock starts.
* -autotrain games apply VIEWYAW 0 from the first frame (MLAgent::viewYawWatch), so the camera no longer
  swings after GO.
* Recording mode is unchanged.

real_run.py keeps countdown decisions out of the stuck-breaker, the trace and minutes/speed
('countdown_decisions' in the round row). It also gained NAV_TRACE / NAV_TAG, so A/B arms can run side by side.

**kotmjump (operator request): `data/multiplayer/hunt/custom/kotmjump.mcs`.** A KOTM copy with gemGroups = 1:
4 groups, one per big hole. Each group is corner + side + centre gem + a gem floating over the hole centre
(x = -30.25/-20.25, y = 10.05/20.05, z = 21.7, ~2 marble diameters above the floor). Terrain maps
terrain_kotmjump / terrain_fine_kotmjump were generated; the walk grid is identical to KOTM's.
nav/terrain's spawn list keeps only 14 of the 16 gems: the west hole gems are > 2 cells from floor. Real
rounds are unaffected.
* 1x result on nav_night_25237_143: 3 gems / 4 points / 45-46 falls per round. It never took a floating gem,
  so the round stalls on the first group. It arrives at the lip at 5-6 u/s and freezes; most "jumps" are
  stuck-breaker samples.
* Physics model (predict_landing arc, west approach): a real jump 0.5-1.3 u before the lip at >= 6 u/s already
  passes within 0.2 u of the gem. Landing on the centre walkway needs 8-11 u/s; >= 12 overshoots into the next
  hole.
* Operator: keep the map, DO NOT train on it for now. Generalisation plan discussed: varied jump maps plus one
  held-out test map.

**REWARD = GAME SCORE, step 1 (operator-approved "reward = game rules only, small OOB marker"): nav/waypoints.py**
* FALL 25 -> 5 and FALL_AFTER_JUMP 6 -> 5. Hypothesis: 25 on top of the implicit cost (the lost ground is
  unpaid progress; later gems are discounted) made caution cheap, hence braking into every gem.
* ARRIVE is paid PER POINT in real-gem mode (yellow = 20).
* Everything else unchanged. Note that a flat TIME cost cannot create urgency in fixed-length rounds.

Started 01:04 on 2026-09-26 from `nav_night_25237_143.pth` (update 25,235) as nav_latest; the V5 endpoint is
kept as nav_v5_end_25520.pth. Rotation 7 KOTM / 1 Islands. Snapshot state reset (V5 state archived as
night_best_v5.json). Helpers deduplicated: 43 stale copies of game_summary.sh / game_watchdog.sh were killed
and one of each restarted.

**Judge step 1 by:** the sampled 50-round KOTM mean (V5 lineage 138-142; best50 143.4), falls per round (~1.3-2
before; a rise is expected), and speed at the gem with the next gem ahead (7.7). Then 8 rounds at 3x against
151.2.

**Step 2** (approved, my call after step 1 reads out): ALIGN_BONUS 0.10, RUNUP_K 0.3, JUMP_TAKEOFF 0.1, AIR 0.1
and BRAKE 0.05 all -> 0. PROGRESS 1.0, GEM_SPEED_BONUS 24/60, PROGRESS_NEXT 0.3 and TIME 0.05 are kept.

### 28.43 Step 1 read-out and STEP 2 applied (2026-09-26 03:45)

**Step 1, 01:04-03:40** (updates 25,235-25,715, 1,323 KOTM training rounds, sampled policy):
* Score held: mean 140.0 over the run, 50-round means 138.5-143.0 (best 143.0 at 25,485, snapshot
  nav_night_25485_143.pth), the same as the pre-change level of 138-142.
* Falls rose from ~1.8 to 2.3 per round on average (3.1 in the last 50).
* Speed at the gem when the next gem is ahead, training trace: 7.79 (V5) -> 7.97 -> 8.12 -> 8.20. Side 7.9 and
  behind 7.0 stayed put, so arrival speed is starting to depend on the next gem.
* Seconds per pickup: 1.42 -> 1.39.
* Endpoint saved as nav_step1_end_25715.pth; snapshot state archived as night_best_step1.json.

Verdict: holds the score and moves arrival speed the intended way, so step 2 was applied (operator-approved on
this condition).

**Step 2, nav/waypoints.py:** ALIGN_BONUS 0.10, RUNUP_K 0.3, JUMP_TAKEOFF 0.1, AIR 0.1 and BRAKE 0.05 all -> 0.
The reward is now:
* per point picked up: ARRIVE 10
* per gem, for time between pickups: GEM_SPEED_BONUS 24/60
* distance guide: PROGRESS 1.0 and PROGRESS_NEXT 0.3
* per decision: TIME 0.05 (neutral)
* per fall: FALL 5

Trainer restarted at ~03:46 from nav_latest (25,715); the games stayed up and reconnected. Watch: the brake-key
share (free now), takeoffs and falls (jumping is free now), 50-round mean vs 140, and speed at the gem when the
next gem is ahead (8.2).

### 28.44 Overnight read-out, 07:40 on 2026-09-26 (step 2 still running)

Sampled KOTM training rounds (not deterministic evals):

| phase | updates | rounds | mean | 50-round range | falls/round | ahead / side / behind speed at gem | s/pickup |
|---|---|---|---|---|---|---|---|
| V5 before (baseline) | - | - | 138-142 | best 143.4 | ~1.3-2 | 7.79 / 7.91 / 6.97 | 1.42 |
| step 1 (FALL 5, ARRIVE/point) | 25,235-25,715 | 1,323 | 140.0 | 138.5-143.0 | 2.3 | 8.20 / 7.83 / 7.04 | 1.39 |
| step 2 (behaviour terms 0) | 25,720-26,408 | 1,905 | 139.9 | 135.1-142.5 | 2.4 | 8.36 / 8.02 / 7.07 | 1.38 |

* Score has not moved. Arrival speed when the next gem is ahead rose steadily (+0.57 u/s) and seconds per pickup
  fell 1.42 -> 1.38; the gems-behind speed did not rise. That is the intended shape, but small.
* Falls: 2.3-2.4/round on average, up from ~1.8. They peaked at 3.3 around 04:40 and have been 1.6-2.9 since.
* Freezes: an occasional frozen round (65, 68, 35 points) shows up about once an hour. Training has no
  stuck-breaker.
* Nothing crashed; the watchdog never had to relaunch the games.

Snapshots:
* nav_night_25485_143.pth (step 1 best 50)
* nav_step1_end_25715.pth
* nav_night_25759_141.pth
* nav_night_26113_142.pth (step 2 best 50, falls 1.6)

Not done: a deterministic 8-round eval at 3x against 151.2. It needs a 9th game instance, which does not fit on
the GPU next to the trainer (real_run.ps1 refuses without -Force), and training must not be stopped without
the operator.

### 28.45 Morning eval 2026-09-26 (3x, 8 deterministic rounds, training paused 08:27-08:35)

| checkpoint | rounds | mean (se) | falls/round | vs pre-change |
|---|---|---|---|---|
| pre-change nav_night_25237_143 (28.42 sweep, off arm) | 150 146 155 149 153 156 155 146 | 151.2 (1.4) | 1.5 | - |
| step 2 best 50 window, nav_night_26113_142 | 156 149 144 159 154 150 152 161 | 153.1 (2.0) | 1.6 | +1.9 +- 2.4 |
| step 2 endpoint, nav_step2_pause_26535 | 134 158 141 143 138 148 161 151 | 146.8 (3.4) | 3.4 | -4.5 +- 3.7 |

Verdict: no measurable gain from steps 1+2 on the deterministic score.
* The best-window snapshot matches the pre-change checkpoint. It was chosen as the peak of noisy sampled means,
  so its +1.9 is the upper edge of what the run offers.
* The endpoint is lower, and the whole difference is falls (3.4 vs 1.5 per round).
* Arrival speed rose (training trace: ahead-gem speed 7.8 -> 8.4) but did not convert into points; the cheaper
  fall (5) was spent on falling.

Training resumed 08:35 on step 2 from nav_latest (26,535); the operator decides the next change.

### 28.46 FALL back to 25 (2026-09-26 08:47, operator)

Following 28.45 (FALL 5 spent the cheaper falls on falling), FALL is 25 again. FALL_AFTER_JUMP stays 5 (was 6
before 28.42), ARRIVE stays per point, and the step 2 zeros stay. Trainer restarted from nav_latest (26,570);
the endpoint was saved as nav_step2_end_26570.pth and the step 2 snapshot state archived as
night_best_step2.json. The games stayed up. Judge against: 3x eval 151.2 / 1.5 falls (pre-change), 153.1 / 1.6
(nav_night_26113_142); training 50-round mean ~140 and ahead-gem speed 8.4. The hope is that falls return to
~1.5-2 while the small arrival-speed gain stays.

### 28.47 FALL-25 read-out and eval (2026-09-26 10:15-10:30); training STOPPED by the operator

The FALL-25 run (28.46) went from update 26,570 to 26,830. Training 50-round means were 135-141, falls 2.0-3.3.
Training was stopped by the operator at ~10:14. Endpoint saved as nav_fall25_end_26830.pth.

3x eval, 8 deterministic rounds each:

| checkpoint | rounds | mean (se) | falls/round | vs pre-change 151.2 |
|---|---|---|---|---|
| nav_fall25_end_26830 | 159 144 158 144 153 157 152 149 | 152.0 (2.1) | 1.9 | +0.8 +- 2.5 |
| nav_night_26778_141 | 158 151 146 159 153 151 148 156 | 152.8 (1.6) | 2.0 | +1.5 +- 2.2 |

Summary of the whole reward change (28.42-28.47):
* Every evaluated checkpoint is within noise of the pre-change 151.2: step 2 best +1.9, FALL-25 end +0.8, FALL-25
  best +1.5.
* The only outlier is the step-2 endpoint: -4.5, with 3.4 falls/round under FALL 5.
* Restoring FALL 25 brought eval falls back to ~2 per round.

Net result: the reward is now closer to the game rules (points per pickup, the five behaviour terms at 0) at no
cost, but there is no measurable gain. The speed-into-gems habit moved only slightly (training trace 7.8 ->
8.4 u/s).

Current reward: ARRIVE 10 per point, GEM_SPEED_BONUS 24/60, PROGRESS 1.0, PROGRESS_NEXT 0.3, TIME 0.05, FALL 25,
FALL_AFTER_JUMP 5, ALIGN/RUNUP/JUMP_TAKEOFF/AIR/BRAKE 0.

Next candidates, waiting on the operator:
* spin in the observation (V6; verified readable and settable, see PHYSICS_PLANNER_PLAN.md section 10);
* the learned jump-outcome model (PHYSICS_PLANNER_PLAN.md v1).

### 28.48 ARRIVE back to per gem (2026-09-26 ~10:45, operator)

ARRIVE pays 10 per gem again, whatever its colour. Paying per point could pull the marble toward yellow gems
instead of along the most efficient path through the spawn. The gem ORDER itself is set by nav/gems.choose, not
the policy.

The reward is now the pre-28.42 reward minus the five behaviour terms, with FALL_AFTER_JUMP 5 instead of 6.
Training remains stopped.

### 28.49 Distance per gem spawn: the 170-level human wins on ROUTE LENGTH (2026-09-26 ~11:00)

KOTM spawns are always 4 gems. Each spawn was measured from the pickup that cleared the previous group to the
pickup that clears this one (respawn teleports excluded; the first spawn of each round skipped).

| | distance / spawn | time / spawn | speed over spawn |
|---|---|---|---|
| human, best two rounds (177, 172; demo 09-14 rounds 2-3) | 45.1 u | 5.15 s | 8.77 u/s |
| human, all 7 recorded rounds | 50.4 u | 6.18 s | 8.16 u/s |
| model nav_fall25_end_26830 + nav_night_26778_141, 16 rounds 3x | 50.7 u | 5.90 s | 8.59 u/s |

* NOTE: "human = 143.5" (the recorded-demo average used since 2026-09-14) mixes 163 / 177 / 172 with 121 / 119 /
  109. The 170-level rounds ARE recorded (09-14 rounds 2-3). Compare per round.
* Against the best human rounds the model takes 15 % longer per spawn: 12 % from a longer route, 2 % from speed.
* The 09-24 demo's 17 gap jumps saved ~7.3 u each (28.36), ~4.6 u per spawn, close to the whole 5.6 u
  difference. So hole-crossing jumps are the main lever on KOTM itself, not only on kotmjump. To confirm: count
  gap jumps in 09-14 rounds 2-3.

**28.49 addendum: jump count and route split (same morning).**
* Gap jumps per round (flight over real void, then landing):
  * human 09-14: 18 / 12 / 15 / 9 / 18 / 27 in rounds of 163 / 177 / 172 / 121 / 119 / 109;
  * the 177 and 172 rounds saved 3.0-3.3 u of walking per spawn;
  * model: 1.6 per round, 0.3 u per spawn.
  * The weakest human round jumped the most and still had the longest route, so jumps are not the whole story.
* Route split, spawns without falls:
  * human best two: straight lines between the 4 pickups 41.6 u, travelled 45.1 u (ratio 1.084);
  * model: 43.2 u, travelled 50.1 u (ratio 1.160).
* Of the 5.0 u gap:
  * ~3.3 u is path straightness. It matches the human's jump savings: flying straight across holes vs rolling
    around them.
  * ~1.7 u is where the pickups happen (gem order, and/or grabbing gems at the edge of the ~1 u pickup range
    instead of rolling through the centre). Not yet separated.
* Rough time split against the 170-level rounds (model 15 % slower per spawn): about 8 from jumps, about 4 from
  pickup order and position, about 2 from speed.

### 28.50 Whole-spawn gem order (2026-09-26 ~11:10): fixes the order, no points at inference; PARKED

Setup:
* Analysis first (per spawn, vs the human's 177 / 172 rounds): of the model's ~1.6 u route gap from pickup
  positions, ~0.7 u is gem ORDER (set by the fixed chooser, not learnable today), ~0.5 u is pickup clipping
  (human 0.67 u off-centre vs model 0.53 u), and ~0.3 u is spawn luck.
* Built: nav/gems.py plan_tour() scores every order of the visible gems (leg distances + the momentum cost of the
  first turn + TOUR_TURN_U 6 x sin(turn/2) at each gem, sticky with TOUR_SWITCH 0.9). Used only by real_run.py
  when NAV_TOUR is 'euc' (straight lines) or 'walk' (walk-only Dijkstra). Default '' = unchanged; training is
  unchanged.

3x, 8 rounds each, nav_fall25_end_26830:

| chooser | mean | falls/rd | distance/spawn | time/spawn | order loss (walk) | best order taken |
|---|---|---|---|---|---|---|
| greedy (current) | 151.4 | 1.9 | 51.6 u | 5.97 s | 3.28 u | 38 % |
| tour, walk | 149.8 (-1.6 +- 2.2) | 3.1 | 50.5 u | 6.02 s | 1.00 u | 55 % (human 51 %) |
| tour, straight line | 147.9 (-3.5 +- 2.1) | 3.0 | 51.9 u | 6.10 s | 1.82 u | 43 % |

* The walk tour fixes the order and shortens the route, but the policy moves slower and falls more on those routes.
  25 falls vs 15; the extra ones are mostly in the centre between the holes (17 vs 10), not after pickups.
* The policy was trained on the greedy chooser's targets. To cash the order in, it would have to be trained with
  plan_tour as its chooser (vec_worker too). The gain is at most a few points and depends on centre edge control.
* Parked by choice; the jump work comes first. Code stays behind NAV_TOUR (off).

### 28.51 Training WITH the whole-spawn planner (2026-09-26 11:03, operator)

The operator expects the planner (28.50) to give new highs once the policy trains on its routes. Changes:
* nav/vec_worker.py: TOUR = NAV_TOUR, default 'walk'. The real-gem worker picks targets with plan_tour (walk-only
  Dijkstra fields, cached per gem position per map; about 0.3 ms per decision). '' = the old greedy chooser.
* nav/real_run.py: default NAV_TOUR is now 'walk' too (training parity). For a greedy eval, NAV_TOUR must be
  EMPTY inside Python. PowerShell `$env:NAV_TOUR = ""` DELETES the variable, so the run gets 'walk' (see 28.52).

Training started from nav_latest = nav_fall25_end_26830 (update 26,830), with the reward of 28.48 and the
7 KOTM / 1 Islands rotation. The FALL-25 snapshot state was archived as night_best_fall25.json.

Judge by:
* the GAME bars;
* falls in the centre (the inference test fell 17 vs 10 there);
* a 3x eval against greedy 151.4 and tour-at-inference 149.8.

### 28.52 Planner training result (2026-09-26 13:41 stop, 3x evals 13:42-14:20): NO GAIN

The operator stopped training at 13:41 after one more hour; training stays STOPPED until the operator says otherwise.
During the run (11:03-13:41, updates 26,830 -> 27,280), KOTM averaged 139.7 over 1,245 rounds with 2.8 falls. The last
50 rounds averaged 141.9 with 2.0 falls. Best snapshot: nav_night_27189_142.pth (best50 141.74). Endpoint:
nav_planner_end_27280.pth.

3x, 8 rounds per arm (16 for the endpoint with walk):

| arm | weights | chooser | points | falls/rd | speed | distance/spawn | time/spawn | best order |
|---|---|---|---|---|---|---|---|---|
| ev_greedy (28.50) | 26830 | greedy | 151.4 (se 1.5) | 1.9 | 8.57 | 51.6 u | 5.97 s | 38 % |
| ev_tour_walk (28.50) | 26830 | walk | 149.8 | 3.1 | 8.32 | 50.5 u | 6.02 s | 55 % |
| ev_pl_end + ev_pl_end_b | 27280 | walk | 151.4 (se 1.2), +0.0 +- 1.9 | 1.9 | 8.10 / 8.22 | 48.4 / 48.8 u | 5.91 / 5.90 s | 54 / 56 % |
| ev_pl_best | 27189 | walk | 154.1 (se 1.9), +2.8 +- 2.5 | 2.6 | 8.38 | 49.5 u | 5.84 s | 61 % |
| ev_pl_end_greedy | 27280 | greedy | 148.5 (se 1.7), -2.9 +- 2.3 | 2.4 | 8.27 | 50.6 u | 6.06 s | 43 % |

* The planner shortens the route by 2-3 u per spawn (51.6 -> 48.4-49.5), but the marble got slower
  (8.57 -> 8.10-8.38 u/s). Time per spawn improved only 1-2 %, so points are unchanged.
* The policy itself got slower. On the same chooser (greedy), 27280 runs at 8.27 u/s vs 8.48-8.57 for every
  pre-planner checkpoint (s2best, f25end, f25best, 26830), and scores 148.5 vs 151.4.
  Walk vs greedy on the same 27280 weights: +2.9 +- 2.1. The chooser's gain was paid for by a slower policy.
* The best snapshot's +2.8 is inside noise, and it was picked by the training bars (selection bias).
* EVAL BUG, caught and fixed: the first ev_pl_end_greedy arm was launched with PowerShell `$env:NAV_TOUR = ""`,
  which deletes the variable. It therefore ran 'walk'. Its route stats (56 % best order) exposed it; its rounds were
  renamed ev_pl_end_b. The greedy arm was re-run with `python -c "import os, runpy; os.environ['NAV_TOUR']=''; ..."`
  (43 % best order, as greedy should be). Earlier greedy evals (ev_greedy etc.) ran while the default was '' and are valid.
* Code state: NAV_TOUR default is still 'walk' in vec_worker and real_run. Nothing reverted; the operator decides.

### 28.53 Rollback to nav_night_26113_142 (2026-09-26 ~14:55, operator)

The operator watched the planner endpoint (27280, walk) at 1x: 7 rounds averaging 149.7, 7.8-8.2 u/s. They judged it
bad: it rolls past gems and has to come back for them. On request, nav_latest.pth is now nav_night_26113_142.pth
(step-2 best: 153.1 at 3x, 1.6 falls). The planner model is kept as nav_planner_end_27280.pth.
At 1x with greedy, 26113 scored 153 and 153 (1 fall each, 8.63 / 8.19 u/s) before the operator ended the run.
The NAV_TOUR default in vec_worker and real_run is still 'walk' (not reverted; the operator decides).

### 28.54 Greedy chooser is the default again (2026-09-26 ~15:10, operator); FUTURE NOTE on the planner

NAV_TOUR default is '' (greedy choose()) again in nav/vec_worker.py and nav/real_run.py. NAV_TOUR=greedy also
means greedy, because PowerShell `$env:NAV_TOUR = ""` deletes the variable. NAV_TOUR=walk / euc still select plan_tour.

**FUTURE NOTE (operator):** the whole-spawn planner (nav/gems.plan_tour, NAV_TOUR=walk) is parked, not dropped.
If KOTM gets stuck and scores stop rising, revisit it: the gem ORDER is still about 1/3 of the route gap to the
human's 170-level rounds (28.49), greedy takes the best order only 38 % of the time (planner 54-61 %), and the planner
cut 2-3 u per spawn. The 28.52 attempt failed because the policy got slower and overshot gems on the planner's routes.
The route gain itself was real.

### 28.55 Video mode for screen recording (2026-09-26 15:22-16:29, operator)

The operator recorded a 1x KOTM round over 160. New, off by default:
* mlAgent.cs: control word `RESTART`. It cancels a pending OOB respawn, sends the 'end' message, then takes the
  normal autoRestart path (restartLevel, loop restarted 100 ms later).
* nav/real_run.py: `NAV_VIDEO_TARGET=160`. A fall restarts the round at once. So does the game's own score
  predictor (hunt.cs updatePredictor: floor(score * round length / elapsed), from obs 31-33) at or under the
  target at 2:00 or 1:00 left.
Launcher: scratchpad video_1x.ps1 (NAV_SPEED 1, NAV_WATCH 1, NAV_VIEW_SUBSTEPS 4, nav_latest = 26113, greedy).

Result over 49 rounds: 2 clean rounds over 160 (round 22: 167, ~15:47-15:50; round 35: 163, ~16:07-16:10),
plus one clean 149 (round 29).
Most others fell or were under target at 2:00. Six were cut at 1:00 with the predictor at exactly 160 (107 points).
REMOVED the same day on the operator's request: the RESTART control word (mlAgent.cs) and NAV_VIDEO_TARGET
(real_run.py) are gone, and fall handling is back to the plain OOBCLICK. To record again, re-add both as described above.

### 28.56 NAV_OBS_V6: marble spin in the observation (2026-09-26 ~18:30, operator)

The operator asked for spin as a model input (needed on every map; see PHYSICS_PLANNER_PLAN.md D5).
* observer.cs: collectSelfState reads `$MP::MyMarble.getAngularVelocity()` (rad/s), rotated like the velocity;
  serializeToJSON appends it LAST, so raw indices 0-34 are unchanged. Raw obs 35 -> 38 numbers.
* nav/protocol.py: RAW_DIM 38, RAW_SPIN = slice(35, 38). A 35-number observation (a stale observer.cs.dso)
  now raises a clear error instead of hanging.
* nav/obs.py: NAV_OBS_V6, VEC_DIM 58 -> 61, spin block at 58-60 as a ROLLING VELOCITY /20:
  [r*wy, -r*wx, r*wz], r = 0.19. Rolling east spins about +y with |v|/|w| = 0.190 (probe, 09-26). So on the
  floor vec[58:60] equals vec[4:6] unless the marble skids.
* Checkpoint: `logs/nav/migrate_obs_width.py` converted nav_night_26113_142 (V5 best, 153.1 at 3x) to
  `nav_v6_start_26110.pth` = nav_latest. The spin columns start at zero. Checked offline: actions within 6e-7,
  value and hidden state identical for random spin inputs; Adam moments correctly shaped.
  V5 checkpoints can only be run with V5 code now (the V5 original is kept).
* Live smoke round (3x, 1 round, not training): 159 points, 0 falls, 8.52 u/s. Spin on every decision.
  On the floor, moving: corr(v, rolling velocity) 1.000, skid median 0.01 u/s. So on KOTM spin adds information
  mainly in the countdown (pre-spin up to 92 rad/s at zero velocity), in the air, and after landings or edge hits.
  Expect a small effect on KOTM; the payoff is for maps with slopes, bumpers and landings.

Training NOT started: waiting for the operator's approval. Plan: a couple of hours from nav_latest, same reward
and 7 KOTM / 1 Islands rotation, greedy chooser. Judge "no regression" by the GAME bars against the FALL-25 run
(last 50 ~141-142, ~2 falls), then a 3x 8-round eval against 153.1. Archive night_best.json before the launch.

### 28.57 Super Speed can be aimed in any direction (2026-09-26 ~19:05, operator confirmed at 1x)

The engine boosts along the marble's CAMERA yaw (marble.cc doPowerUp case 2: camera forward projected onto the
contact plane, 25 u/s). The actions already set that yaw every decision, and the view is pinned separately by
VIEWYAW. So a boost can go anywhere: set camYaw = atan2(dx, dy) and press use-powerup (7th action word).
Test mode, off by default: `NAV_SUPERSPEED_TEST=1` makes real_run send `SSTEST 1`, and mlAgent.cs
MLAgent::superSpeedTest fires any held Super Speed straight back along the velocity. It aims 2 ticks before the
press and holds 2 after, so the server marble has the yaw. The operator watched it reverse the marble at 1x.
The console echoes were lost to the force-kill (the engine buffers console output), so there are no logged velocities.
A policy that uses powerups would need the held powerup in the observation (observer.cs collects none today).
REMOVED the same day on the operator's request (SSTEST, MLAgent::superSpeedTest, NAV_SUPERSPEED_TEST). How to rebuild it
is in the operator's session memory (superspeed-aim-recipe); everything needed is also above.

### 28.58 V6 (spin) training run: no regression (2026-09-26 19:10-21:45, operator stopped it)

The run went from nav_v6_start_26110 to update 26595, same reward, 7 KOTM / 1 Islands, greedy chooser. KOTM training
rounds: 1,344, mean 140.4, 2.13 falls/rd, last 50 140.7 / 1.80. Earlier runs today: 137.6-140.0 with 2.3-2.8 falls.
The 50-round windows went 142.7, 143.5, 140.2, 139.2, 139.7, 138.3, 141.0, 139.3, 141.4, 140.7, never below 138.
Best snapshot nav_night_26206_144.pth (best50 143.5, 1.64 falls). Endpoint nav_v6_end_26595.pth = nav_latest.
The spin weights grew slowly from zero: vec-layer spin columns 0.14 -> 0.20 -> 0.26 against 6.1 for velocity (~4 %).
So training with spin did not hurt. But spin's influence is still small, so this does not show that spin helps.
Proper test (proposed, not done): a 3x A/B on one checkpoint, real spin vs spin zeroed (needs an eval-only switch),
plus the 153.1 baseline. Operator watching the endpoint at 1x afterwards (logs/nav/watch_1x_v6.txt).

### 28.59 Docs split, and the spin A/B (2026-09-26 22:00-22:45, operator)

* The handoff was split: `../HANDOFF_NAV_TRAINING.md` is now a short current-state document, and this file is
  the dated log (its stale 2026-09-21 "CURRENT STATE" block removed). CHAT_CONTEXT.md (the old dashboard's
  prompt), ROADMAP_NAVIGATOR_PLANNER.md and IMPLEMENTATION_PLAN_NAVIGATOR.md moved to docs/archive/ with
  headers; README.md and PHYSICS_PLANNER_PLAN.md (D5 done) updated; PHYSICS_SKILLS_DESIGN.md marked background.
* Spin A/B, 3x, 8 rounds per arm: nav_v6_start_26110 154.0 (1.6 falls); nav_v6_end_26595 real spin 150.0
  (1.4), spin zeroed 150.0 (0.75). Real vs zeroed +0.0 +- 2.2: spin has no measurable effect yet (steering
  influence median 0.6 deg, p99 11.7 deg; jump flips 0.009 %). The 2.5 h run cost 4.0 +- 1.8 points against its
  start, spin or not. No checkpoint trained after 26113 has beaten it at the deterministic eval.
* 22:50, operator: nav_latest.pth = nav_v6_start_26110 (26113 in V6) as the checkpoint to train from.

### 28.60 Sprawl's "too steep" cells are ledge borders, not slopes (2026-09-26 ~23:30)

Found while building a 3D view of Sprawl's terrain data for the operator. Sprawl's ramps are smooth 18.3 deg planes,
stored exactly (14,315 half-unit cells, 92 % walkable). Of the 2,150 walk cells flagged too steep (log 25, B6), 2,133
(99 %) sit next to a ledge or wall: `nav/terrain.py` `_build_walk_grid` takes the slope as a central difference over
1 u cells, so a vertical step reads as a steep slope on the cells beside it. That is a strip of non-walkable cells
along every ledge top and wall bottom, the same family as KOTM's phantom holes. Not fixed (code change needs the
operator): measure slope only between neighbours on the same surface, or take normals from the map geometry.
The model's own features treat the ramps as floor: rays and the gap block follow up to 1 u of rise per 0.5 u step.

## 29. Jump physics P0 (2026-09-27, autonomous session; operator offline)

The plan: `../KOTMJUMP_START_HERE.md` (P0) with `../KOTMJUMP_NAVIGATION_DESIGN.md` as the reference. Scope approved by the
operator: new files under `nav/learned_nav/`, the spin words for the Python teleport, small `mlAgent.cs` additions and
the drill map. Training stayed stopped.

### 29.1 M0: the preserved navigator on kotmjump (16:48-17:13, 3x, 8 rounds)

nav_v6_start_26110 (26113), greedy chooser: 4.6 points a round (9, then 4 in each of the other seven), 37 falls a
round, 23 stuck-breaker firings a round. One floating gem in 8 rounds (round 1, the north-west hole). In rounds 2-8 the
last pickup came 6-15 s into the round: after the first three floor gems the navigator never finished a group with a
floating gem and spent the other ~165 s stuck. `logs/nav/kotmjump_m0_baseline.txt`,
`logs/nav/real_trace_kotmjump_m0_baseline.csv`, `logs/nav/real_run_kotmjump_nav_v6_start_26110_kotmjump_m0_baseline.json`.

### 29.2 Stage 0: the drill map and the checks

* `nav/learned_nav/make_drill_map.py` writes `hunt/custom/kotmjump_p0.mcs`: kotmjump with only the gem at
  (-30.25, 10.05, 21.7), no powerups, a one-hour round. First attempt did not load: the game looks the info function
  up as `MP_PQ_<alphanumerics of the file name>_GetMissionInfo` (`shared/mission.cs` getMissionInfo), so the
  underscore in `kotmjump_p0` must be dropped (`MP_PQ_kotmjumpp0_`). The autotrain sat in the level select; this
  engine build writes no console.log, so it was found with a screenshot.
* `mlAgent.cs`: control word `GEMRESET` (respawn the gem group when no gem is up; the game respawns a group
  excluding the gem just taken, so a one-gem map stays empty after a pickup). `.dso` deleted.
* `nav/learned_nav/probe_drill.py` (in the game): mission and one-hour round OK; gem present; spin teleport OK
  (with the rolling spin words the marble keeps 8 u/s and accelerates; without them it skids to 6.8 u/s and spins
  up over 4 decisions); teleport onto the gem gives a pickup, the fall, recovery and GEMRESET brings it back;
  11 of 12 lip points match the 0.5 u terrain grid. The miss: the south lip. The 0.1 u raster
  (`nav/fine_terrain`, built for kotmjump_p0) has the hole at x -33.75..-26.78, y 6.58..11.58 with the bay on the
  west half to 13.58, and matches the game at all 12 points; the 0.5 u grid puts the south lip 0.3 u too far into
  the hole. The recorder uses the 0.1 u raster.
* Teleport timing, measured: the observation after TELEPORT plus 2 decisions is the teleported state itself
  (position, velocity and spin exactly as requested); the first physics step follows it. The recorder's start
  state is that observation, checked against the request to 0.05 u (this also catches a teleport the game ignored).
* Repeat check (`nav/learned_nav/check_repeat.py`, declared tolerance 0.01 u): 4 starts x 3 candidates x 20
  repeats: every recorded position identical (max difference 0.0), identical replies and events. The
  out-of-bounds flag arrives one decision later in some repeats (a trigger event, not the physics); allowed +-1.
* Edge hits are real and large: a marble clipping the far edge of a hole left it at vz +11 to +12.7 u/s (a jump
  gives 7.4). A flight can therefore go on well past its first contact, so trials run until the marble has landed
  and rolled 0.5 s (cap 75 decisions); an undecided flight at the cap is censored and counts as unsafe.

### 29.3 P0a: the recorder and the data

`nav/learned_nav/record.py`. Start: hold the run-up input one decision, TELEPORT onto the floor with rolling spin,
2 more run-up decisions. Candidates: no jump, or jump at decision 0-4 holding one of 9 air inputs (none, or heading
+ 0, 45, ..., 315 deg): 46 per start, the same for every arm. After landing, no input for 8 decisions. Recorded per
decision: the reply as serialized (with camera yaw), position, velocity, spin, the floor height under the marble,
gem delta, out-of-bounds flag, the game tick. Outcomes: pickup, out of bounds, takeoff and landing decisions,
censored, ends over floor; safe = no out of bounds, not censored, a started flight landed, ends over floor;
success = pickup and safe. Provenance line first in every file (engine/script/mission hashes, settings).
Starts: on the floor around the hole, 0-15 u/s, heading within 60 deg of the gem, the lip 0.3-6 u ahead along the
heading. Split by region and heading: 3 u blocks x 20 deg bins of heading relative to the gem, one cell in four held
out (md5 of the cell), starts within 0.25 u / 2 deg of a cell edge skipped. 6 games at ~21 trials/s each.
Two batches: 2,400 starts at 0-15 u/s (448 held out) and, because only 26 % of those were feasible, 1,600 more at
6-15 u/s (308 held out). 4,000 starts x 46 candidates = 184,000 flights, 0 teleport retries, 0 failed trials,
about 34 min of wall time (datasets/learned_nav/p0/, 0.9 GB, not git-ignored: keep it out of commits). Rounds end every ~100 s of wall time (the
one-hour game clock runs ~35x real time); the restart costs one tick. Of the pickups in the first 8,000 flights,
65 % fell into the hole before landing, 16 % ended over void (the narrow east strip, or a wall slope in the hole)
and 19 % succeeded, every success landing at 8.7 u/s or more.

### 29.4 P0b: the predictor, the arms, the freeze

`nav/learned_nav/dynamics.py`: an MLP (3 x 384) on the start state in its own frame (velocity, spin), a 0.5 u height
crop from the 0.1 u raster (1 u behind to 14.5 u ahead, 6 u each side; relative height and floor present) and
lip/gap distances along 15 rays, plus the candidate one-hot. Heads: the timed path (45 decisions), safe, landed,
landing point; a small head that also sees the gem gives pickup and success (the physical part never sees the gem).
Chosen on VALIDATION cells inside the training starts (437 starts, 100 feasible), never on held-out: scoring
variant (direct P(pickup) x P(safe) / path closest pass / joint success head), mirror augmentation (left-right
reflection, spin as an axial vector) and epochs. Selected: direct, mirror on, 40 epochs (validation 0.83; without
mirror the best was 0.66). Final fit on all 3,244 training starts: `models/learned_nav/p0_flight.pth`.
`nav/learned_nav/evaluate.py freeze` also fitted each baseline's single parameter on the training starts (the
hand-written arm's pickup radius made no difference, 14.1 %; the fixed rule's lip distance D = 2.0 u, 23.8 %),
computed every arm's pick for the 756 held-out starts from their start states only, and wrote
`configs/learned_nav/p0_frozen.json` (18:10) before any held-out outcome was read. Each arm then ran once per
held-out start in the game (3,024 trials, `datasets/learned_nav/p0/arms_*.jsonl`).

### 29.5 Result: PASS (18:12, `logs/learned_nav/p0_results.json`)

756 held-out starts, 251 feasible (the oracle, all 46 candidates recorded, found a success; 3.3 working candidates
on average). Success = pickup and safe landing, over the feasible starts, 95 % Wilson intervals:

| arm | success | 95 % | pickup | out of bounds |
|---|---|---|---|---|
| learned | 170/251 = 67.7 % | 61.7-73.2 | 93.2 % | 11.6 % |
| fixed rule (D 2.0) | 47/251 = 18.7 % | 14.4-24.0 | 53.0 % | 12.4 % |
| hand-written (physics.py) | 30/251 = 12.0 % | 8.5-16.5 | 35.5 % | 33.5 % |
| random | 15/251 = 6.0 % | 3.7-9.6 | 21.5 % | 62.5 % |

The learned arm's lower bound (61.7 %) is far above every baseline's upper bound (24.0 %), with 251 >= 100
feasible starts: the declared pass rule holds. The engine re-runs agreed with the oracle's recorded outcome in
3,024 of 3,024 trials.
* Calibration of its own choices: predicted 67.7 % success on average, actual 67.7 %; predicted > 0.8: 86 % actual
  (103 starts), 0.5-0.8: 62 % (94), < 0.5: 43 % (54). On infeasible starts it predicts 4 % on average: it knows when
  no candidate works (the basis for abstaining later).
* Hardest starts (only 1 of 46 candidates works, 74 starts): learned 53 %, fixed 12 %.
* Failures (81): 40 picked the gem up but ended over void after landing plus 0.5 s of no input, 22 picked it up and
  fell before landing, 2 fell after landing, 17 missed the gem. The no-input continuation is harsh on the narrow
  east strip and the wall slopes; a braking or steering continuation is the obvious next candidate family.
* Predictor, all 34,776 held-out flights: pickup AUC 0.985 (568 false positives, 632 false negatives at 50 % of
  3,685 pickups), safe AUC 0.953, success AUC 0.973 and calibration error 0.007; median path error 0.45 u per
  flight (under 0.3 u for the first 0.6 s, growing on long falls), median landing point error 0.39 u.
* Validation (0.83) read higher than held-out (0.68): the validation number was the best of 72 settings on 100
  starts, so it is optimistic. The held-out number is the one to quote.

### 29.6 Files, and what P0 does not show

New: `nav/learned_nav/{make_drill_map, probe_drill, record, check_repeat, dynamics, evaluate, watch}.py`,
`nav/learned_nav/watch_p0.ps1`, `hunt/custom/kotmjump_p0.mcs`, `terrain_maps/terrain{,_fine}_kotmjump_p0.npz`,
`configs/learned_nav/p0_frozen.json`, `models/learned_nav/p0_flight.pth`, `datasets/learned_nav/p0/`,
`logs/learned_nav/`. Changed: `mlAgent.cs` (GEMRESET), `nav/protocol.py` + `nav/env.py` (teleport spin words).
The PPO navigator, its checkpoints and training are untouched.
P0 shows flight prediction and action choice from near-launch states at one hole. It does not show approach
planning, route finding, whole gem groups, PPO handovers or other maps (KOTMJUMP_START_HERE.md). Watch it:
`powershell -ExecutionPolicy Bypass -File nav\learned_nav\watch_p0.ps1` (held-out feasible starts at 1x;
`-Fresh` for new random starts). Next (needs the operator): stage 3, the general data pipeline (design M1, M2).

## 30. Jump physics stage 3: the general data pipeline and the pilot models (2026-09-27 night, autonomous)

Operator approvals (2026-09-27 evening): stage 3; a small C++ change in the mbx engine (commit locally, never push);
minimal script additions; replacing marbleblast_mbx.exe; training stays stopped; 15 maps (below); data under a 20 GB
cap. The operator went to bed at ~23:20 and asked for as much progress as possible, stopping only for major issues.

### 30.1 Maps (operator-approved, normal surfaces only)

Only beginner/intermediate/advanced/expert maps; no special-friction surfaces, moving platforms, gravity changes,
pushers or many stacked floors; Horizon and Archipelago excluded; Pyramid and Skate Park Square removed by the
operator. Training (10): Vortex Effect, Gems in the Road, Tilo, Basin Hill, Parkour Peaks, Duplex, Cragmire, Sprawl,
King of the Ring, Maximo Center; plus KOTM (the base task). Held-out development (2, never trained on): Gems Ahoy,
Acropolis 2. Reserves (unused): Treasure Box, Marble Agility Course, Skatium. A survey of all 113 Hunt missions
(107 geometry families) and an edge survey (lips, drops, gaps, narrow paths) guided the choice.

### 30.2 Engine: contact telemetry (mbx repo, branch ai-training-mode, local commits only)

* 3cb4c53f: the existing, previously uncommitted AI training-mode changes (fixed step, lockstep, render skip,
  multi-instance, view yaw) and build-engine.ps1.
* d0124cd2: `Marble::getContactTelemetry()`: a summary of the physics sub-steps since the last read (sub-steps,
  with contact, with a supporting contact, with a collision, most contacts, fastest approach into a surface, the
  supporting normal and its material friction / restitution / force, contact at the last sub-step). Bookkeeping in
  advancePhysics outside prediction replays; testMove's return value (previously discarded) counts collisions.
* The rebuilt exe replaced marbleblast_mbx.exe (old one kept as marbleblast_mbx_pretelemetry.exe). 340 recorded P0
  flights replayed: every outcome and reply identical, 26 of 43,776 values off by the 0.0001 u recording step (no
  growth): not strictly bit-identical (a rebuild's floating-point noise), well inside the 0.01 u tolerance.
* Scripts (minimal): mlAgent.cs control word `CONTACT 1|0`; observer.cs appends the 13 numbers after the 38 only
  when on. nav/protocol.py: `GameMessage.extra`, `CONTACT_FIELDS`. The 38-number observation is unchanged.
* In-game check (probe_contact.py): resting 8/8 sub-steps supported, normal (0, 0, 1), friction 1; a landing from
  a drop shows a collision at 7.69 u/s against a measured 7.68.

### 30.3 Practice copies, geometry export and in-game verification

* practice_maps.py: `<Mission>_phys` copies in hunt/custom: all geometry, gems, spawn and bounds kept; removed
  powerups (every non-gem item), Duplex's 42 glass panes, Gems Ahoy's physics-modifier zone and graffiti, Marble
  Agility Course's signs; one-hour round. All 16 load in the game (check_load.py).
* geometry.py (`geometry_v1`): collision surfaces (single detail level in all 27 DIFs), materials (none special on
  any of the 16), a 0.2 u layered raster with exact face normals, and exact edge segments from the mesh classified
  void / drop / wall. Two traps found and fixed: seams between floor faces whose vertices do not match (T-junctions)
  looked like edges (now dropped when the floor continues beyond), and edge queries caught surfaces under the floor
  (rays now follow the marble's floor on the raster). The exact KOTM lips (y 6.5, x -33.7, y 11.5, x -26.7) differ
  from P0's 0.1 u raster by 0.05-0.08 u (grid quantization in P0).
* verify_map.py (declared rules: floor 95 %, lips 95 %, ramps 90 %): all 16 maps PASS. Floors 97-100 % (contact
  normal error 0.0 deg at p95 on every map, height error p95 0.0002-0.035 u), ramps 97-100 % (downhill direction
  within 10 deg), lips 96-100 % (0.3 u inside supported, 0.35 u outside falls). The few floor misses are within a
  marble radius of a crease between faces, where either face can be the contact.

### 30.4 Recorder, repeatability and natural-state replay

* record3.py: starts half near an exact edge (0.3-6 u inside, heading mostly toward it) and half anywhere on the
  floor incl. slopes; 0-16 u/s along the surface; rolling spin from the surface normal (15 % deliberately slipping);
  10 % airborne. Control families: roll, switch, brake, none, jump, jump with an air switch. Per decision: state,
  telemetry, the exact reply, the intent, gem delta, out of bounds. ~24 decisions and ~1.2 KB per trial.
* Trap found: after an out of bounds the game schedules a respawn that landed in the next trial (55 of 300 KOTM
  test trials jumped to z 24 at a spawn point). session.py now triggers the quick respawn and waits until the
  marble has visibly moved; any trial with a > 2.5 u jump between decisions is discarded as a guard.
* check3.py on KOTM, Sprawl, King of the Ring: repeats 30/30 (positions identical; only the out-of-bounds flag's
  timing varies, up to 2 decisions). Natural-state replay (teleport into a naturally reached state, replay the same
  replies): 115/120 within 0.02 u over 15 decisions (declared rule 95 %): KOTM 40/40, Sprawl 39/40, King of the
  Ring 36/40; median error 0.00012-0.00015 u. Of the 5 misses, 2 are fully explained by the reporting precision
  (a 0.00005 u nudge alone gives the same divergence; chaotic contacts, several on King of the Ring's panel seams),
  1 (an airborne cut, 0.04 u) is not: possibly a small hidden state for later; 2 could not be classified.
* Edge contacts at speed and angle (edge_check.py, recorded roll-offs without a jump): the release point (the
  supported fraction of the last supported decision's sub-steps along its path) lies a median +0.03 u beyond the
  exact edge on every map (p95 +0.09 to +0.10 u; King of the Ring +0.06, its rim is a 22 deg fold), 92-97 % within
  0-0.25 u; ~20,000 roll-offs on 11 maps at a median 7 u/s across the edge. The exported edges are where the game
  lets go.
* Collection (collect3.py): 121 jobs of 20,000 trials over 13 maps, 8 games, ~17-20 trials/s each, resumable,
  stall/crash restarts, disk guard. Cragmire: its map files include scenery out to +-200 u; a marble that leaves the
  bounds box sideways and falls was sometimes not respawned; recovery was made soft (logged, the jump guard discards
  any trial a late respawn lands in): 7 cases in 20,000 trials, the job completed.
* check3 on four more maps after collection (04:20): repeats 40/40; natural-state replay Tilo 40/40, Basin Hill
  40/40, Parkour Peaks 39/40, Duplex 35/40. All seven maps: 269/280 = 96.1 % (declared rule 95 %). The game reports
  state with ~6 significant digits: on maps whose coordinates exceed 100 (Duplex z 133-141, Tilo x to 210) the
  replayed start is only good to 0.001 u, and the median replay error there is 0.001 u (0.0001 elsewhere); the Duplex
  misses are that coarser rounding amplified by contacts. Remedy if wanted: more digits in observer.cs's state
  (a one-line script change, not done).

### 30.5 Data

2,420,000 trials (04:21), 2.65 GB: KOTM 500,000; the ten training maps 180,000 each; Gems Ahoy and Acropolis 2 60,000
each (evaluation only). 121 jobs, none abandoned. Feature caches (datasets/learned_nav/stage3_features, derived, can
be deleted and rebuilt): 6.4 M step transitions, 1.5 M flight samples.

### 30.6 The step model (dynamics3.py, 3 bootstrap members, 768 x 4 residual blocks, 6 epochs)

Chosen on validation blocks of the training maps (8 u blocks, one in eight), never on the held-out maps. Three
measured improvements on the way: (1) train the mean by squared error and the spread separately (fitting both through
one likelihood stalled two dimensions at the spread bound); (2) the previous reply as an input: at an input switch the
step's velocity change still follows the previous input by ~0.25 u/s (rolling velocity error 0.358 -> 0.211 u/s);
(3) targets as the deviation from the ballistic step (one-step position error 0.033 -> 0.006 u). Contact telemetry is
a target, never an input (the state is Markov, section 30.4), so the model rolls forward on its own predictions.
Results (logs/learned_nav/stage3_eval.json), one step (64 ms), validation blocks, 916,816 transitions:
position 0.0043 u median / 0.045 p95 (air 0.0034, rolling 0.0056, collisions 0.028 / 0.30); velocity 0.116 u/s;
support AUC 0.999, collision AUC 0.970, contact normal 0.98 deg; held-out maps: Gems Ahoy 0.0042 u, Acropolis 2
0.0045 u (the same). Rollouts on its own predictions (1.02 s): 0.67-0.86 u median (ballistic 8.6-11 u); held-out maps
0.67-0.76 u; from airborne starts 0.045-0.052 u at 4 decisions (bounded airborne corrections). Systematic error: none
overall (position bias 0.0002 u), small while rolling (speed -0.07 u/s), large in collisions (+0.76 u/s along,
-0.50 u/s vertical): a unimodal prediction averages bounce and roll, as the design warned; collisions need modes.
Uncertainty is conservative: 85-94 % of errors within one predicted std (ideal 68 %); needs a calibration scale.
Ensemble disagreement correlates with the error at 0.53.

### 30.7 The flight head (flight3.py) and arbitration

CNN on a heading-frame crop (2 u behind to 16 u ahead, +-6 u), exact edge rays, the state, and the exact reply
sequence of 24 decisions (the jump anywhere in it, so a candidate is scored from the current state); path as a
correction to the start velocity, takeoff and landing decisions, landing point, safe. 1.26 M training samples,
10 epochs. Validation: path 0.43 u median (p90 2.7), takeoff decision exact 90 %, landing point 1.23 u, safe AUC
0.971; held-out maps: path 0.46 / 0.54 u, takeoff 89 %, safe AUC 0.968 / 0.956. Arbitration on the same validation
jumps: the step ensemble rolled forward is better at 4 decisions (0.075 u vs 0.11-0.13), level at 8 (0.25 vs
0.24-0.34), mixed at 16.

### 30.8 Cross-check on P0's held-out test, and the verdict

The general models never saw a P0 recording (KOTM is in the stage 3 data, the same geometry as the P0 hole, so this
tests the objective and the controls, not unseen geometry). Each P0 candidate written as the reply sequence P0 flew;
pickup from the predicted path's closest pass to the gem through a curve fitted on P0 training starts; the landing
predicted first and the input released after it, as P0 did (p0cross.py). Flight head: 118/251 feasible held-out starts
= 47.0 % (95 % 40.9-53.2), pickup query AUC 0.958 (776 false positives, 1,050 false negatives of 3,685 pickups).
Step-model rollouts flying the same candidates closed-loop: 80/251 = 31.9 % (26.4-37.9), pickup AUC 0.935. For
reference: P0's own model 67.7 %, the fixed rule 18.7 %, physics.py 12.0 %, random 6.0 %.

Verdict against the design's gates. M1: passed (20-repeat agreement, natural-state replay 96.1 % against the declared
95 %, rolling / takeoff / flight / landing all recorded, edge contacts verified at speed and angle, materials: none
special on these maps). M2: mostly met. Takeoff timing (90 % exact), pickup errors, landing / exit errors, coverage and
disagreement by slice, and bounded airborne corrections are reported. Two items are open: collisions carry a
systematic contact error (a unimodal prediction averages bounce and roll; the design asks for modes), and the
predicted uncertainty is too wide (needs a calibration scale). Landing points from the flight head are 1.2 u off
(median). Results page: https://claude.ai/artifact/N9DBr7e7ASBDuc6vbx8Wxr. Files: nav/learned_nav/{practice_maps,
geometry, verify_map, check_load, probe_contact, session, record3, check3, collect3, edge_check, dynamics3, flight3,
eval3, eval_flight3, p0cross, final3}.py; datasets/learned_nav/{geometry, stage3, stage3_features (derived, 14 GB)};
models/learned_nav/{step3_0-2, flight3}.pth; logs/learned_nav/. Changed shared files: nav/protocol.py (extra,
CONTACT_FIELDS), mlAgent.cs (CONTACT word), observer.cs (the 13 numbers when on). Next (operator): finish the two M2
items (collision modes, uncertainty calibration; within stage 3), then stage 4.

### 30.9 The two M2 items, fixed (05:10-06:00)

* Uncertainty calibration (calibrate3.py): one scale per output (0.47-0.65, i.e. the spread was about twice too wide),
  fitted on the training maps' validation blocks, checked on the held-out maps: within one predicted std 68-71 % on
  Gems Ahoy, 62-65 % on Acropolis 2 (ideal 68.3 %); within two 86-92 % (ideal 95.4 %): the tails are heavier than a
  Gaussian (a heavy-tailed or mixture spread would fix that). models/learned_nav/step3_calibration.json.
* Collision modes (design section 5): the step model now also predicts the step's outcome given a collision and given
  none (heads trained on their own outcomes; the collision probability picks or weights them). With the true branch
  the collision bias falls from +0.75 / -0.47 u/s (along / up) to +0.06 / -0.01; picking the branch by the predicted
  probability cuts every error by ~40 %. Final mode ensemble (step3_0-2.pth; the single-mean ensemble is kept as
  step3u_0-2.pth), validation: one-step position 0.0026 u median / 0.027 p95, velocity 0.065 u/s; held-out maps
  0.0026 / 0.0029 u; 1.02 s rollouts 0.37-0.42 u (held-out 0.43-0.44 u; ballistic 8.6-11 u); from airborne starts
  0.025-0.030 u at 4 decisions. Remaining note: whether a collision happens is ranked well (AUC 0.97) but rarely above
  50 %, so the collision slice keeps a bias under automatic branch choice; a planner should keep both branches.
* Arbitration with the mode ensemble: the step rollout is now more accurate than the flight head at every horizon on
  the same validation jumps (4 / 8 / 16 decisions: 0.05 / 0.17 / 0.46 u against 0.11-0.13 / 0.24-0.34 / 0.57-0.83).
  P0 cross-check: step rollouts alone 89/251 = 35.5 %; the pickup from the rollout's path times the safe landing from
  the flight head: 131/251 = 52.2 % (95 % 46.0-58.3), the best general result (flight head alone 47.0 %; P0's own model
  67.7 %; fixed rule 18.7 %). Verdict: M2 met for the pilot, with the two notes above. Next (operator): stage 4.

### 30.10 Full-precision marble state (operator-approved, 2026-09-28 morning)

The observer pins the frame to the world (ForceYaw 0), so its rotation was an identity that only cost digits:
TorqueScript re-prints arithmetic results with ~6 significant digits. observer.cs now passes the engine's own strings
through when the yaw is 0 (position to 6 decimals, velocity and spin to 7 significant digits); format_teleport sends 6
decimals (was 4); protocol.py skips a non-list first message (a partial line seen once at connect). Natural-state
replay after the change: Duplex 40/40 (was 35/40, median error 0.0000086 u, was 0.001), Tilo 40/40 (0.000008 u),
KOTM 40/40 (0.0000038 u, was 0.00014); repeats 30/30. So the earlier misses were reporting precision, not hidden
engine state. The stage 3 data was recorded before the change (states good to 1e-4 to 1e-3 u); the models need no
retraining for it. The navigator's observation carries the same values, now exact.

## 31. Jump physics stage 4 (design M3): jumps from anywhere (2026-09-28, autonomous)

Operator, 2026-09-28 morning: "go ahead and start working on phase 4". Training stayed stopped; no engine change; no
commit in this repo. Code in nav/learned_nav/: planner.py, seeds.py, drills.py, drill_report.py, guide.py, watch4.py
(+ watch4.ps1); make_drill_map.py now builds four drill maps.

### 31.1 Drill maps and geometry

kotmjump_p0-p3: kotmjump with one floating gem each (p0 -30.25 10.05, p1 -20.25 10.05, p2 -20.25 20.05, p3 -30.25
20.05, all z 21.7), no powerups, one-hour round; GEMRESET puts the gem back. Geometry exported per drill map (the KOTM
interior: 1752 triangles, 210 void / 52 wall edges); the in-game verification is KOTM's (same interior; noted in each
verify.json). The four holes are 7 u (x) by 5 u (y) with bays toward each other; each gem floats 0.86 u above the
rolling marble's centre, 1.5 u inside one lip and 3.5 u inside the opposite one.

### 31.2 The planner (planner.py): model-predictive control on the stage 3 models

Every 64 ms decision, from the observed state and the two replies that shape the next physics (the reply sent last
decision acts first; the one before it is the step model's previous input):
* proposals: 320 broad programs (a heading toward the gem, the current velocity or anywhere; maybe a turn or a brake
  first; 75 % with a jump at decision 0-22 and an air direction), 96 aimed at gem-side seeds (drive to a point behind
  the seed, turn onto its heading, fly its jump), 40 warm starts (last choice shifted, jittered); 256 air-control
  programs while airborne. Every ground program brakes for its last 10 decisions (terminal check: it must still be
  able to stop);
* prediction: all programs rolled 41 decisions (2.6 s) with the step ensemble; after a predicted landing the
  program's after-landing input takes over;
* judgement: P(pickup) from the closest pass to the gem (the logistic curve fitted on P0 in stage 3: r0 0.85 u, slope
  4) x safe (landed after any flight; on a floor for the last 4 decisions; never 0.5 u below the start level; no
  flight without a jump; no bounce back into the air after landing) x an edge-clearance factor (1 u from void / drop
  edges after the landing counts fully, 0.2 u not at all). A jump from the floor is re-judged by the flight head from
  the observed state (the observed-state launch check);
* robustness: the 12 best are rolled again from four perturbed starts (speed x0.96 / x1.04, heading -2 / +2 deg);
  P(success) = mean pickup x mean safety-and-margin over the variants; approach and continuation plans are vetoed if
  any variant falls;
* choice: best P(success) - 0.004 x decisions to the pickup if above P_GO (0.25); a floor jump only above P_JUMP
  (0.35). Otherwise approach: the program whose path comes closest to a seed state (position and velocity, and at
  least the walking distance over the floor to the nearest seed, so a marble on the far side of a hole goes around
  it). After the pickup: stay on the floor, away from edges, no new jumps.
Planning takes ~0.3 s a decision with one game, ~1.3 s with four (lockstep: the game waits).

### 31.3 Gem-side seeds (seeds.py)

6000 floor states within 8 u of the gem, heading toward it within 60 deg, 2-14 u/s, rolling spin, run-up pending; 28
jump candidates each (jump at decision 0-3; no air input or 0 / +-45 / +-90 / 180 deg), judged exactly as the planner
judges (robustness included). Seeds: states whose best candidate reaches P_JUMP. Counts for the gated version (built
for v4): p0 214, p1 168, p2 187, p3 185 of 6000 each (datasets/learned_nav/seeds/, older sets in v1-v3/). Robust launch
states are rare because a flight through a gem 3.5 u into a 7 u hole tends to land right at the far lip.

### 31.4 The drill runner (drills.py) and the fixture

Starts per map and set (dev 30, test 64), drawn once from a hash of the map and set and frozen in
logs/learned_nav/drills/starts_<set>_<map>.json: on the gem's floor level, 4-18 u from the gem, at least 1 u from any
void or drop edge; 30 % at rest, otherwise 2-12 u/s with rolling spin, heading toward the gem (within 40 deg) for 35 %
and anywhere for the rest. Fixture (engine only, before any planner trial): brake / steer 90 deg left / right then
brake for 3 s; a start none of them keeps on the map is unsafe, run anyway and reported apart. Feasibility of the task
from a safe start: one connected floor level, and a pickup plus continuation demonstrated in the engine for every
target (dev runs). A trial runs until success, out of bounds or 188 decisions (12 s); a pickup late in the window gets
up to 40 more decisions for its continuation. Success: the engine's pickup, then a landing ON a floor (a bounce back
into the air restarts the count) and 8 decisions (0.5 s) on the map, ending over a floor at the start level.

### 31.5 Dev iterations (tuning on the dev starts only)

| Version | Change | Safe-start success |
|---|---|---|
| v0 (8 trials) | first run | 3/8; stuck in a pocket under a lip (fall rule 2 u), roll-ins |
| v1 | fall rule 0.5 u, brake tail, clearance, diagnostics | 40/51 (78 %); landings just past a far lip clipped it |
| v2 | + robustness, worst case over variants | 6/11: almost nothing left to fly; stalls |
| v3 | mean over variants, seeds rebuilt the same way | 90/105 (85.7 %, 77.8-91.2); pickup jumps 70/70; all failures in the approach |
| v4 | + no flight without a jump / no bounce after landing, walk-distance field, P_JUMP 0.35 / P_GO 0.25, seeds rebuilt | 47/54 (87 %, stopped halfway); stalls beside the seeds |
| v5 | seed-aimed run-ups sized to the seed's speed (v^2 / 20 + 0.5 u; measured ~10 u/s^2 from rest) | 93/105 (88.6 %, 81.1-93.3); frozen for the gate |

Paired on the 55 safe starts all of v3, v4 and v5 ran: 49, 48, 53. v5's failures on the dev set: 4 jumped-then-stalled,
4 stalls, 2 took the gem then fell, 2 approach jumps that fell. Launch check (presses within 6 decisions = one launch):
jumps sent in pickup mode at P >= 0.35 took the gem 87 of 89 times (every band 94-100 %); approach-mode jumps (no pickup
plan) 8 of 28. The runner's jump log counts every press while the marble is still on the floor, so a launch can show
two or three presses; drill_report groups them.

### 31.6 The time-to-gem guide (guide.py)

Trained on the PPO navigator's traces (logs/nav/trace_*.csv, KOTM instances only, 22.9 M decisions from 6 files;
validation: the most recent file, 3.2 M decisions): seconds from a rolling state to the goal gem, from the goal's
position in the velocity frame, speed, vertical speed and floor contact. Validation MAE 0.115 s against 0.202 s for
distance over the best constant speed (median 0.047 against 0.121 s). models/learned_nav/guide_ttg.pth,
logs/learned_nav/guide_eval.json. It has no geometry input, so it cannot know a hole is in the way. As the design
allows, it only times proposals (planner.USE_GUIDE: the seed-aimed run-up leg), never scores or rejects a route. It was
OFF in the gated configuration; its A/B on the dev starts is 31.8.

### 31.7 The gate: PASSED (13:46, logs/learned_nav/drills/report_test_gate.json)

Test starts: 64 per floating gem, 256 in all, drawn from a hash of map and set (starts_test_<map>.json, sha1 p0
ee53cf5dd12f, p1 b7abd6ea0028, p2 2e6c9758f2e4, p3 4844034a5888), written and fixture-checked before any planner
trial. Gated configuration: planner.py sha1 a649c1d0e805 (v5; USE_GUIDE off, P_GO 0.25, P_JUMP 0.35), the v4 seeds.

| | Result |
|---|---|
| Feasible (fixture-safe) starts | 233 of 256 (p0 58, p1 56, p2 59, p3 60) |
| Pickup plus continuation | **217 / 233 = 93.1 % (95 % 89.1-95.7)**; gate 80 % on >= 200: PASS |
| By target | p0 56/58 (96.6 %), p1 52/56 (92.9 %), p2 55/59 (93.2 %), p3 54/60 (90.0 %) |
| By start | at rest 62/68, rolling toward 74/81, across 55/58, away 26/26 |
| Abstentions (no jump in 12 s) | 7 |
| Unsafe starts (reported apart) | 15 of 23 solved |
| Median time to the pickup | 2.3 s |
| Planning | 1.28 s a decision on average with four games (p95 at most 2.05 s); lockstep |

Failures on feasible starts (16): 7 stalls (pickup plans appear for a few decisions, then fall below the jump
threshold before the launch), 3 rolled into a hole during the approach, 3 took the gem and fell after the landing, 2
fell after a jump that missed (1 approach jump, 1 pickup-mode jump predicted 0.49), 1 took the gem at decision 187 and
was still bouncing at the end of the window. Observed-state launch check (presses within 6 decisions = one launch):
jumps sent with a pickup plan (P >= 0.35) took the gem 187 of 188 times; approach jumps (no pickup plan) 34 of 65.

### 31.8 Guide A/B on the dev starts (after the gate)

The gated planner (v5) against the same planner with USE_GUIDE on (planner.py sha1 8540e929dbd4 during the run), same
seeds, same 105 feasible dev starts: 98/105 (93.3 %) with the guide against 93/105 (88.6 %) without; discordant
starts 10 for the guide, 5 against (McNemar p ~0.3: not significant). Where both succeeded, the pickup came after 32.5
decisions instead of 36.5 (median, ~0.26 s sooner). The guide run had no falls (5 stalls, 2 jumped-then-stalled). The
default is now USE_GUIDE = True for the next work; the gate result stands for v5 with it off.
Results: logs/learned_nav/drills/dev_kotmjump_p*.jsonl, report_dev_guide.json.

### 31.9 Files and what M3 does not show

nav/learned_nav/: planner.py (MPC), seeds.py, drills.py (starts, fixture, trials), drill_report.py, guide.py,
watch4.py + watch4.ps1 (1x viewing: `powershell -ExecutionPolicy Bypass -File nav\learned_nav\watch4.ps1 -Map
kotmjump_p2`), make_drill_map.py (four drill maps). Data: datasets/learned_nav/seeds/ (+ v1-v3 older sets), results
logs/learned_nav/drills/ (dev_v0..v5, test_*, reports). Not shown by M3: route choice between approaches, gem groups,
other maps (all four drills share the KOTM interior), handing control to and from PPO (M4/M5). Next (operator): M4.

## 32. Jump physics stage 5 (design M4): reliability, whole gem groups, handing control back and forth (2026-09-28/29 night, autonomous)

Operator, 2026-09-28 evening: "alright proceed with the next stage! i'm going to bed". Training stayed stopped; an
mlAgent.cs change was approved if needed (none was); no engine change; nothing committed in this repo. Stage numbering:
the overview's stage 5 "Reliable and smart" is the design's M4.

Summary: whole rounds of kotmjump went from 5.25 points (navigator alone, 4 rounds, 33 falls a round) to 72.75
(navigator plus planner for the floating gems, memory kept current: 85, 85, 67, 54), 14 floating gems and 2.25 falls a
round, no fall within 3 s of any of 60 handbacks. The single-gem reliability check was NOT met: 90.8 %
(207/228) on a new frozen test set against a bar of 95 % per gem (p0 84.7 %, p1 90.2 %, p2 94.2 %, p3 94.6 %); the
planner tuned to 95.2 % on the dev starts but that did not carry over.

### 32.1 Planning speed (same results, faster)

Profile of one decision (570 ms, one game): half in eval3.predict (each ensemble member moved each output to the CPU
separately: dozens of small transfers and ~60 kernel launches per step), a third in the terrain features.
* planner.FastEnsemble: the three members on the GPU with one transfer each way, captured as CUDA graphs for padded
  batch sizes 64 / 256 / 512 (one replay per step). Same outputs as eval3.predict to 1e-6.
* Geometry.heights_at: the raster levels laid out per cell (one row gather per query). Identical values.
* EdgeIndex.nearest: edge columns laid out per cell. Identical features (checked on four maps, 20000 random states).
One game: 570 -> 400 ms a decision. Several games are GPU-bound at ~4.4 decisions/s in total (processes or threads
alike). On this card (GeForce RTX 3070) TF32 is no faster than fp32, and fp16 only 23 % faster with ~1e-3 output noise:
both left off. Rule learned the hard way: no GPU benchmarks while drills run (they halved the drill rate).

### 32.2 Single-gem reliability: dev iterations (the 30 dev starts per gem only)

| Version | Change | Feasible dev starts | Mean time per start (failure = 12 s) |
|---|---|---|---|
| v5 (M3 gate) | | 93/105 | 5.35 s |
| guide | v5 + guide-timed run-ups | 98/105 | 5.38 s |
| v6 | horizon 40 -> 48; continuation judged over 1.5 s after the landing; commitment to the last pickup plan (kept above 0.15 unless another is 0.1 better); approach jumps must be safe from every perturbed start | 99/105 | 5.52 s |
| v7 (stopped at 38) | an unverified approach jump is never sent | roll-ins: marbles committed to a run-up had the jump taken away | |
| v8 | ... unless every program without a jump falls (then the jump is the way out) | 100/105 (95.2 %) | 4.93 s |

Why: in the M3 test 7 of 16 failures were stalls. Pickup plans appeared for a few decisions and vanished: their jumps
were late in the 2.6 s horizon, and the landing had to show 4 settled decisions before it ended, so a run-up one
decision slower made the plan vanish. The longer horizon then made easy jumps look unsafe (the fixed after-landing
brake could not stop the marble before the next hole 2 s later, a continuation the planner never flies since it
replans after landing), so the continuation is judged over 1.5 s past the landing.

### 32.3 The gate: 64 new starts per gem (test5), v8 frozen first: NOT MET

planner.py sha1 7b6d0882d06a (USE_GUIDE on), seeds v5 (datasets/learned_nav/seeds/, 162-212 per gem). Fixture: 228
feasible of 256. Result: 207/228 = 90.8 % (95 % 86.3-93.9); per gem p0 50/59 = 84.7 %, p1 55/61 = 90.2 %, p2 49/52 =
94.2 %, p3 53/56 = 94.6 %; bar 95 % each. Failures: 7 took the gem then fell, 6 jumped without the gem then stalled, 4
stalls, 2 fell after an approach jump, 2 rolled in. Mean time per start (failure = 12 s) 5.33 s; the M3 gate's planner
on its own (different) test set: 93.1 %, 5.08 s. So on held-out starts v8 is not better than v5: six versions tuned on
the same 105 dev starts overfitted them. (A paired run of both planners on test5 would settle it; v5's code was not
kept separately.) Launch check: jumps sent with a pickup plan (P >= 0.35) took the gem 180 of 186 times.
The falls after a pickup share one cause: the flight lands on the 3 u strip between two holes at speed and rolls into
the next hole. The planner's clearance measures the distance to the nearest edge in any direction, not whether the
marble can stop before the edge it is heading for. First fix next: a stopping-distance check along the landing
velocity.

### 32.4 Whole rounds of kotmjump (rounds.py, hybrid.py, round_report.py)

Real 3-minute rounds: 4 groups of 4 gems (1 yellow and 2 red on the floor, 1 red floating over the group's hole); a new
group spawns only when all 4 are taken. Arms:
* navigator alone: nav/real_run.py, nav_latest (update 26110), 4 rounds;
* hybrid (hybrid.py): the navigator drives as in real_run (its chooser, the gem as goal, stuck-breaker); when its target
  floats, the planner owns the marble until the pickup and a landing held 4 decisions, then hands back. Memory at
  handback: 'current' (the navigator runs on every observation while the planner drives, its actions discarded) or
  'reset';
* planner alone (rounds.py): gem order by the guide over all permutations (+1.5 s for a floating gem), floor gems with
  a smaller budget (160 broad programs) and walking distance to the gem.

| Arm | Rounds | Points | Floating gems a round | Falls a round | Falls within 3 s of a handback |
|---|---|---|---|---|---|
| navigator alone | 4 | 5.25 (9, 4, 4, 4) | 0 | 32.8 | |
| hybrid, memory current | 4 | 72.75 (85, 85, 67, 54; sd 15.1) | 14.0 | 2.25 (1.25 navigator, 1.0 planner) | 0 of 60 |
| hybrid, memory reset | 4 | 60.75 (60, 87, 9, 87; sd 36.8) | 12.25 | 2.25 | 1 of 51 |
| planner alone | 1 | 65 | 14 | 0 (longest gap between pickups 8.3 s) | |

(A first hybrid round with the pre-v6 planner scored 74.) The memory comparison is not settled by 4 rounds each: the
difference in means (12 points) is well inside the spread, and the reset arm's 9-point round was the navigator
stalling on a floor gem, not a handover problem. The stage 5 check "handovers that don't cause falls": 1 fall within
3 s of 111 handbacks.

The navigator alone takes its first gems, then keeps trying to roll to the floating gem and falls (33 falls a round).
In one reset round the navigator itself stalled on a floor gem for 140 s (100 stuck-breaker resets) after its second
handback; the planner only takes floating gems, so nothing took over. Next: let the planner take over a floor gem too
when the navigator stalls. Planner-alone rounds are slow to run (0.4-2 s a decision, ~2800 decisions a round).

### 32.5 Files

nav/learned_nav/: planner.py (FastEnsemble, commitment, horizon 48, set_target, floor-gem walking field), rounds.py,
hybrid.py, round_report.py, drills.py (test5 set, replies recorded, provenance), drill_report.py (completion time),
seeds.py (for_gem: seeds reused on identical geometry), geometry.py and dynamics3.py (faster, identical results),
watch4.py (1x viewing via a hidden planning game and a replay). Data: datasets/learned_nav/geometry/kotmjump.*,
seeds v5; results logs/learned_nav/drills/ (dev_v6..v8, test5_*, report_test5_gate.json) and logs/learned_nav/rounds/.

## 33. Stage 5b: KOTM shortcuts (2026-09-29, autonomous)

Operator, 2026-09-29: proceed with the recommended plan; the long-term goal is an architecture that is good on (nearly)
all maps; the short-term measure is the real KOTM map: take shortcuts and beat the best score (154.0 = 8-round mean at
3x, nav_v6_start_26110; best single round 167; human 177). Training stayed stopped; no engine change; nothing committed.

### 33.1 Evaluation hygiene and the v9 regression

* New start sets: devb (60 per gem, iterate here), val (30 per gem, untouched), test6 (64 per gem, only for a gate).
  planner_v8.py is the frozen gated planner; seeds are versioned per planner (datasets/learned_nav/seeds/<version>/).
  compare_versions.py pairs two tags on the same starts (exact McNemar).
* v9 (a straight-ahead stopping-room check after landings, air brake, escape-only approach jumps) regressed: the
  stopping check ignores braking while turning. Ablation v9s (stopping check alone) 7/13 vs v8 13/13 on the same
  starts. Reverted: STOP_CHECK, AIR_BRAKE off, APPROACH_JUMPS 'robust' (v8 behaviour).

### 33.2 KOTM rounds today (3x, nav_latest = 26110)

| arm | rounds | points |
|---|---|---|
| navigator alone (hybrid with shortcuts and rescue off, or consult-only) | 7 | 152.1 (143, 152, 159, 160, 156, 153, 142) |
| shortcuts (probe the planner when the walk detours round a hole), several versions | 8 | 150.8 (143, 150, 143, 149, 160, 156, 152, 153) |
| route jump legs (route.py, below) | 2 | 134, 127 (3 planner falls) |

First shortcut round: 115 points, 7 falls, all within 3 s of a handback (the planner handed back fast marbles next to
holes). Handing back only below 4 u/s fixed the falls but the braking cost 6-25 decisions per shortcut; now the planner
rolls on toward the navigator's next gem and hands back once the marble moves down its walking field with an edge
clearance of 1 u (hybrid.py HANDBACK_*; planner continue_cost with cont_next). Shortcuts stayed rare (1-3 a round).

### 33.3 Where the human gains (demos, the 163 / 177 / 172 rounds)

* Leg times on the same gem pairs: human 1.27 s, navigator 1.36 s. About 40 % of that difference is the hole-crossing
  legs: the human crosses a hole on ~16 legs a round, 0.67 s faster each (1.54 vs 2.20 s on those pairs), mostly the
  diagonals from a centre gem to a corner (e.g. -27.2 17 -> -37.2 27, straight 14.1 u, walk 18.1 u, void 7.1 u).
* 29 of the human's 41 hole-crossing legs go to the second-nearest gem: the order is chosen around the jump. The
  navigator's greedy chooser takes those legs about once a round.
* The human enters the void at 10-14 u/s (median 11.2), ~2.4 u after the previous pickup (picked up at ~7.8 u/s).
* The navigator's leg time, fitted on 689 of its KOTM legs (fine walking field): 0.211 + 0.090 x walk + 0.384 if the walk
  detours 3 u or more + 0.253 x (1 - cos turn) s, residual sd 0.29 s (route.py). The coarse terrain walk field
  overestimated its time on detour legs by ~0.4-1 s.

### 33.4 Route chooser (route.py): tried, off by default

Keeps the navigator's greedy target (a walking tour chooser at inference cost points, 28.50-28.54) and only overrides
it with a jump-first order that beats greedy's best order by ROUTE_GAIN_S; the planner then drives that leg. On the
human's recorded states it matches few of the human's crossings (7-9 of 41). In game: 134 and 127 points; the jump legs
took 2.2-2.7 s and arrived at 3-4 u/s. Off (hybrid --route 0). Known bug: the greedy target's walking time is inf when
the marble is within 0.4 u of an edge (WALK_CLEAR); take the nearest finite cell.

### 33.5 The physics model underestimates jumps (the main finding)

Offline, from a centre-gem pickup state the planner found no pickup plan to the corner at any heading, speed or time
weight. In game (teleport on kotmjump_p0, same geometry; logs/learned_nav/diagjump*.jsonl; scripts in nav/learned_nav/stage5b/, see its README), the same straight-line jumps at 9-12 u/s
cross the hole and take the gem; with no air input after the takeoff most land safely (with forward air input they
overshoot off the map; air brake falls short). The step model's replay disagreed:
* the jump itself: in the game a jump from flat floor is deterministic (1,915 recorded: +0.4268 u and vz 6.108 in the
  decision it fires, +0.3448 u and vz 4.828 in the next, then 20 u/s^2), and it fires in 3,857 of 3,883 cases on flat
  floor (KOTM, Sprawl, Tilo). The model's apex is low on its own training data: median -0.06 u, 27 % below -0.15 u,
  and -0.25 u median (57 % below -0.15 u) for 12-20 u/s takeoffs within 1.5 u of a lip; on some takeoffs it predicted no
  jump at all;
* a false collision at 11 u/s beside the corner of the 2 x 2 centre hole (collision p 0.94, a launch of +3.6 u/s the
  game does not have).
Fix for the first (planner.py JUMP_PRIOR, jump_fires): where the jump certainly fires (resting on flat floor, the key
acting), the two decisions of the takeoff take the measured vertical kinematics; horizontal motion and everything after
stay the model's. Replay of the 32 recorded trials: apex 1.31-1.35 (game 1.33); median path error at decision 16
1.96 -> 0.58 u, at 20 4.59 -> 0.81 u. The judge now agrees with the game on 19 of 24 programs (gem taken and no fall);
the remaining 5 are landings near the far lip the model sees bouncing back into the hole. The flight head (launch
check) is also unreliable there (1.00 for jumps that overshoot, 0.12-0.26 for safe ones).
Also added for floor-gem jump legs (off elsewhere): straight-line jump proposals (_line; the first version pressed jump
too late: a program decision acts one decision after it is sent), steering toward the gem for 2-13 decisions after the
landing before braking (Programs.after_n), and a larger time weight among pickup plans (planner.time_w).

The first version of the prior HURT the floating-gem drills: devb v10a 13/20 vs v8 20/20 on the same starts (McNemar
p 0.016); every fall was a jump pressed at the lip that the game never took (the old model's low apex there was partly a
correct hedge). Stage 3 data, 10,971 presses on floor: the jump fires 97-100 % with 0.1 u or more of floor ahead along
the velocity at any speed, 65-93 % at the lip itself. jump_fires now also requires the same floor JUMP_LIP_U = 0.6 u
ahead (margin for the model's position error). v10b vs v8 on devb (paired, first 45 feasible starts): 39 vs 38,
p = 1.0: no regression. Replay accuracy kept (median error 0.57 u at decision 16).

Closed loop in game (nav/learned_nav/stage5b/cornerdrill.py, logs/learned_nav/cornerdrill.*): from pickup-like states
at the centre gem (6-10 u/s, heading 100-170 deg) the planner with the shortcut settings takes the corner gem across the
hole in 15 of 16 trials, 21-29 decisions (1.35-1.86 s; human 1.27-1.69 s, navigator 2.2-2.4 s on that pair), no fall
after the pickup; the one failure an approach jump at 9.4 u/s that fell.

### 33.6 Fall guard (guard.py): tried, harmful, off

A safety filter over the navigator: near a void, keep its action only if one of 11 recovery programs (brake; steer
toward the target, along the velocity or away from the nearest void edge) stays up with 0.4 u clearance after it,
else send the best recovery. One round: 122 points, 6 falls (navigator alone ~1.2 a round), 209 overrides in 2,353
checks; every fall came ~20 decisions after a burst of overrides. Interrupting the navigator puts it in states it
handles badly (as the fast handbacks did). Off (hybrid --guard 0). A check costs ~0.3-0.7 s.

### 33.7 Stage 6 measurement on KOTM (8 rounds per arm, 3x, run in parallel, 15:08-16:10)

| arm | points (mean, sd) | rounds | falls a round | planner |
|---|---|---|---|---|
| navigator alone | 151.0 (6.0) | 153 158 147 145 155 156 153 141 | 1.00 | none |
| shortcuts + rescue | 148.4 (7.4) | 147 153 141 158 137 156 151 144 | 1.75 | 0.6 shortcuts, 17 decisions a round |
| shortcuts + rescue + route legs | 146.1 (3.8) | 148 152 148 143 148 147 141 142 | 2.38 | 6.0 route legs, 276 decisions a round |

* The shortcut arm is effectively the navigator (the planner acted 17 decisions a round): its -2.6 is round noise
  (sd 6-7, se ~2.3 per arm). Today's navigator-only rounds over all runs: 22 rounds, mean ~152.
* Route legs cost ~5 points: in real rounds they took 2.30 s median from takeover to pickup (the teleport test's
  1.35-1.86 s came from ideal pickup states), 3 planner falls and 4 navigator falls within 3 s of a handback.
* Short-term goal (beat 154.0) NOT met. The planner can fly the human's key crossing (33.5), but the navigator's
  route rarely sets it up, and taking control mid-leg (route legs, guard, fast handbacks) costs more than it saves.

Code state (nothing committed): planner.py = v8 + JUMP_PRIOR (with the lip margin) + time_w / approach_line /
land_steer / _line / after_n (off unless a caller sets them) + cont_next continuation + the ps > 0 ranking fix;
route.py, guard.py new (both off by default in hybrid.py); hybrid.py shortcuts on by default, route/guard off,
probe/leg/fall logs per round. Frozen v8: planner_v8.py. Drills with the prior: devb v10b 58/67 vs v8 55/67.

### 33.8 Orphan marbles in every -autotrain game (evening, operator question)

The operator saw a second, motionless marble on an arm-end slope during the 1x KOTM rounds. A new read-only bridge
word MARBLES (mlAgent.cs; the reply is a DEBUG line: server marbles id:pos:client, client ghosts, $MP::MyMarble, the
player) showed three server marbles for our one client from the countdown on: the player and two orphans sitting
exactly on spawn points (e.g. -45.25 27 and -13 35.25; which points varies per launch, sometimes both on one). The
-autotrain startup (level preview, then Play) spawns the player more than once and GameConnection::createPlayer
creates a new marble even when one exists ("Attempting to create an angus ghost!"), leaving the old one behind. Not
new: the 2026-09-20 engine build (marbleblast_mbx_pretelemetry.exe) gives the same two orphans (same object ids), and
kotmjump has them too; they were out of view in the stage 4/5 viewing (centre holes). On the server they hang at spawn
height (~1.8 u above the slope); a respawn on those two spawn points lands on one.

A removal (delete every server marble that is not its client's player, at start() and GO) worked, but an
interleaved A/B (4 games at once, KOTM navigator only, 10 rounds each) read 150.8 (1.2 falls/rd) with the orphans
removed vs 153.2 (1.0) kept: -2.4 +- 1.8, not significant but not shown harmless, and every baseline includes them.
REMOVED on the operator's request the same evening: the orphans stay; the operator will fix it later. Only the
read-only MARBLES word remains in mlAgent.cs.

## 34. Stage 6b: the crossing leg from real pickup states (2026-09-29 night / 09-30, autonomous)

Operator, 2026-09-29 evening: edits in nav/learned_nav/ approved, parallel work allowed if the machine is not
overloaded, iterate overnight; the goal is ONE real KOTM round above 167 (the navigator-only best) with the planner
involved. Stated odds 15-20 %. Training stayed stopped; no navigator change.

### 34.1 Why the route legs of stage 6 were slow: what the data says

Stage 6's route jump legs (log 33.7) took 2.34 s median from takeover (32 legs; the navigator walks the same pairs in
2.20 s) and started from ordinary pickups (mean speed 7.0 u/s). Of 88 shortcut consultations, 80 came back as
'approach' (no pickup plan from the pickup state): the planner rolls toward a launch seed first. Two of the three
planner falls were jumps sent at 4-8 u/s that fell short; the third rolled in at 3-5 u/s without a jump.
The human (strong rounds, voidentry.py): crossings start from ordinary pickups too: speed median 7.8 u/s, heading
median 30 deg off the line (49 % within 30 deg), void entry at 11.2 u/s 2.6 u after the pickup, leg 1.66 s median.
Same pairs, human vs navigator (human_legs.py): all legs 1.27 vs 1.36 s; hole-crossing pairs 1.54 vs 2.20 s; the
human crosses on 12 % of legs (~16 a round), the navigator ~1.6.

### 34.2 The locked-session stall (2 h lost; the cause is still open)

The first runs crawled: a navigator-only KOTM round took 12 min instead of 40 s. The screen was locked (LockApp.exe).
Probe (scratchpad step_probe*.py): bare decisions 0.7 ms; any reply slower than ~3 ms made the game's next
observation arrive 200-250 ms late in ~20 % of decisions, and every decision once the reply took 20 ms or more. Per
game, not global. Tested and ruled out: the engine's 1 ms sleep in the lockstep wait (a busy-wait changed nothing;
the game is not spinning, 13 % CPU during a wait), TCP_NODELAY, process priority, Windows power throttling
(SetProcessInformation exemption, nav/learned_nav/unthrottle.py), the audio update (skipped in the engine), the SDL
joystick poll (disabled). Engine stage timing (new $AI::SlowLogMs / SLOWLOG word, ai_mainloop_slow.txt) puts every
stall in TimeManager::process or Game->processEvents; net, platform and input stages never stall. Left in: engine
$AI::SpinWaitMs and $AI::SlowLogMs (mbx local commits b5a18bdd, cb44e8a6; both off by default), mlAgent.cs words
SPINWAIT and SLOWLOG, env.py TCP_NODELAY. Consequence for the night: the planner is GPU-bound at ~4.4 decisions/s over
all games, so 6-8 parallel games hide the stall; navigator-only rounds run 8 in parallel.

### 34.3 Crossing drill (crossdrill.py, cross_starts.py)

Starts: the navigator's real states right after a pickup (hybrid.py now logs them: 'pickups'), one KOTM round per
set (s6b_st0 -> dev, s6b_st1 -> test), every crossing pair (a straight line crossing one void run of 3 u or more
that a jump can clear, at most 17.5 u) within 60 deg of the heading: dev 89 starts, test 91. Teleport to the state
(velocity, spin, a push along the heading as the pending reply), planner with the shortcut settings aimed at gem B,
success = within 0.9 u of B and 16 decisions on the map afterwards; the navigator's fitted walking time for the same
leg is stored for comparison. Per-decision records (position, speed, plan kind, program tag, jump, P(success),
airborne, distance to the gem) let the leg be split into run-up, flight and landing-to-pickup.

v0 = the stage 5b planner with the shortcut settings (time weight 0.015, straight-line proposals, after-landing
steering, jump prior). First 13 trials (dev 6/7 ok, one fall on a 5.5 u/s jump; test 6/6): median leg time about
2.1 s on dev and 1.9 s on test against fitted navigator walking times of 2.0-2.4 s: the crossing as flown saves
nothing yet (human 1.66 s median). The records show where the time goes: (1) the planner BRAKES before the jump (one
trial 10 -> 6 u/s over 12 decisions, jumped at 6.6 u/s) because a slower flight judges safer and 0.015 P per decision
does not outweigh a 0.9 vs 0.6 P; (2) approach jumps fire far from the lip (decision 4-10, 1-2 u into the run-up) and
land 6-10 u short of the gem, then roll; (3) 8-17 'approach' decisions before the first pickup plan on some starts.
v1 = ranking saturation (planner.P_SAT, 0.6 on crossing legs): among pickup plans P above 0.6 counts as 0.6, so
trusted plans rank by time (stage 4/5: jumps sent with P >= 0.35 took the gem 187/188 and 180/186). The one v0 fall
(dev 12) is the pathology in full: a 0.9 plan from decision 2, then braking from 9.4 to 5.5 u/s over 8 decisions, a
jump pressed at the lip that never fired, and a roll into the hole.
v2 = v1 without the flight-head launch check (planner.flight_check 0): the head is unreliable near the far lip (33.5)
and multiplies fast jumps below P_JUMP, which looked like the reason the jumps fire at 6-9 u/s.
Paired on the same dev starts (08:10, drills still running): v0 vs v1 on 23 starts: ok 22 vs 23, v1 0.08 s faster on
the both-ok legs, expected time (a failure = the walk plus 3.5 s) 2.31 vs 2.06 s; v1 vs v2 on 10 starts: 10 vs 10,
+0.01 s: the flight check was not what held the speed back (v2 stopped). Jump speed median stays 9.1 u/s (human void
entry 11.2). The leg as flown averages 2.06 s against the navigator's fitted 2.10 s on the same legs: faster on about
two thirds of them, but the slow ones (a late jump at 5-7 u/s after braking, or a landing 6-10 u short) eat the gain.

### 34.4 What the crossing costs, by start angle (v1, 26 dev starts)

| start heading off the line | n | leg time median | navigator fitted | gain |
|---|---|---|---|---|
| 0-15 deg | 6 | 1.66 s | 1.99 s | +0.19 s |
| 15-30 deg | 5 | 1.73 s | 2.01 s | +0.18 s |
| 30-45 deg | 6 | 2.30 s | 1.96 s | -0.25 s |
| 45-60 deg | 9 | 2.30 s | 2.35 s | +0.09 s |

Fit over the 26 legs: t = 1.12 + 0.039 x straight + 1.47 x (1 - cos turn), residual sd 0.34 s. The turn term is ~5x the
walking turn cost route.py used for jump legs; route.py's crossing model now carries these constants (JUMP_K 0.039,
JUMP_EXTRA_S 1.12, JUMP_TURN_S 1.47; the earlier guess was 0.09 x straight + 0.3). Aligned starts reach the human's
1.66 s. The human's crossings are aligned because the order is chosen around the jump (29 of 41 to the second-nearest
gem, log 33.3); ours start from whatever heading the greedy leg leaves.

### 34.5 KOTM gate (08:11-08:50, 3x, run in parallel; GPU contention capped it at 4 games per arm)

| arm | rounds | points | mean (sd) | best | falls a round |
|---|---|---|---|---|---|
| navigator alone | 6 | 153 146 162 150 152 149 | 152.0 (5.5) | 162 | 2.0 |
| hybrid, shortcuts + rescue, v1 settings | 4 | 156 155 141 154 | 151.5 (7.0) | 156 | 1.75 |

The hybrid arm took NO shortcut: 9 consultations a round, 36 of 37 answered 'approach' (no pickup plan from that
state), the one pickup plan (1.66 s vs the navigator's fitted 1.85 s) missed the 0.3 s margin. So the arm is the
navigator plus 1 planner decision a round, and its 151.5 is round noise. The goal (a round above 167 with the planner)
was NOT met; the operator's stated odds were 15-20 %.
Why the consultation fails where the drill succeeds: the drill plans from the pickup state continuously (commitment
and warm proposals build up; the first pickup plan appears at decision 1-2), while the hybrid asks one cold question
(set_target resets the planner's state, one decision, 480 programs) a few decisions after the pickup, and a single
cold decision finds a pickup plan on roughly a third of the drill starts too. Next step, in order: (1) consult over
2-3 consecutive decisions right at the pickup, with the planner state kept; (2) take crossings only within ~30 deg of
the line, or choose the previous leg's exit so the crossing is aligned (the human's order); (3) then the speed at the
lip (jumps still fire at 6-9 u/s; the human enters the void at 11).

### 34.6 Files

nav/learned_nav/: crossdrill.py (the drill, --report / --analyze), cross_starts.py, cross_fit.py, run_one.ps1 and
run_many.ps1 (launchers), unthrottle.py, stage6b/ (probes, README); planner.py P_SAT (p_sat) and flight_check;
hybrid.py pickups log, SHORT_P_SAT, SHORT_FLIGHT_CHECK; route.py calibrated crossing constants; session.py SPINWAIT;
env.py TCP_NODELAY; mlAgent.cs SPINWAIT / SLOWLOG words. Data: datasets/learned_nav/cross/starts_{dev,test}.json;
results logs/learned_nav/cross/{v0,v1,v2}_dev.jsonl, v0_test.jsonl, rounds/g6b_nav*, g6c_hyb*. Engine (mbx worktree,
local commits only): b5a18bdd, cb44e8a6; exe backups marbleblast_mbx_prespin.exe, marbleblast_mbx_spin0.exe.
Nothing committed in this repo.

### 34.7 Daytime 09-30: consult by driving, setup orders, and the real gap is the TURN

Screen unlocked (the operator set the display timeout to never): the bridge answers in 0.3 ms at any reply delay; the
stall of 34.2 is gone.
* hybrid.py consult-by-driving: at a pickup (leg started within CONSULT_AT_DEC = 2 decisions) the planner owns the
  marble for CONSULT_DEC = 3 decisions with its state kept; a pickup plan that beats the navigator's fitted time by
  SHORTCUT_MARGIN_S commits the shortcut, else hand back. Only crossings within SHORTCUT_MAX_ANGLE = 35 deg.
  Smoke round: 0 consults, 146 points: the greedy target right after a pickup is a near floor gem, never a crossing.
* route.py: the greedy baseline now walks every leg (it let later legs jump, which the navigator never does); a
  jump-first order and a SETUP order (walk to the greedy gem A, jump to B second; the navigator gets B as its next-gem
  hint on the way to A, the consult runs on B at A's pickup) are searched with the drill-calibrated crossing time.
  hybrid.py carries the setup (st['setups'], probe candidate B). Offline on 60 real pickup states: 1 jump-first order,
  0 setups. The visible group lies BEHIND the marble at a pickup (the crossing candidates need a 100-146 deg turn), so
  with the planner's measured turn cost no crossing pays. Smoke round 2 (before the setup code): 0 consults, 153.
* The human's crossings, same fit (42 legs, strong rounds): t = 0.10 + 0.101 x straight + 0.33 x (1 - cos turn);
  by angle: 0-30 deg 1.41 s (n 24, arriving at 11.4 u/s), 30-60 deg 1.79 s, 90-180 deg 2.11 s (n 8, from 5.4 u/s).
  Planner v1: 1.66 s aligned, 2.30 s at 30-60 deg, turn cost 1.47 s per (1 - cos): the planner turns 4-5x slower
  than the human, and that, not the flight, is what keeps crossings from paying from the navigator's real states.
  Next lever (general physics, transferable): the run-up turn. A turn-and-cross drill from 30-150 deg, the human's
  recorded inputs during their turns (demos carry the keys) as the reference technique, then proposals that turn
  the way the human does (brake-turn or a tight full-throttle arc) judged by the same model.

### 34.8 Turn drill: the turn is physics-limited, the planner is near the limit (turndrill.py, Skatium, 09-30)

Operator rule (2026-09-30): the human is the benchmark and at most a guide; never hardcode or replay human movements
into the planner. So the turn was measured from the engine over the input space: the marble at 8 u/s on a 12 u open
disc of the same floor material as KOTM (friction 1, restitution 1 for both), 16 fixed input programs, 1.5 s each,
deterministic (repeats identical). Decisions to turn the velocity by 60 / 90 / 120 deg, and the speed then:

| program | 60 deg | 90 deg | 120 deg |
|---|---|---|---|
| steer at +90 (held) | 11 dec, 9.9 u/s | not within 24 | |
| steer at +120 | 9 dec, 6.2 | 14 dec, 8.3 | not within 24 |
| steer at +150 | 10 dec, 3.3 | 11 dec, 3.5 | 14 dec, 5.0 |
| brake 2-6 then +90 | 12-14 dec, 5.9-8.7 | 28-30 dec | |
| over-steer +150 for 2-6 then +90 | 10-11 dec, 5.7-8.9 | 25-29 dec | |
| jump then +90 in the air | 14 dec, 10.7 | | |

A 60 deg turn costs 0.6-0.7 s, a 90 deg turn 0.9 s at speed (steering well past the target direction, 120-150 deg,
is the fastest way; braking first or jumping never helps). The crossing drill's fitted planner turn cost, 1.47 s per
(1 - cos), is 0.74 s at 60 deg: the planner already turns about as fast as the physics allows. The human's smaller
fitted turn cost (0.33) comes from where the human's turns start: the wide-angle crossings begin at 5.4 u/s (slow =
easy to turn) and the aligned ones at 11.4 u/s. So the human's edge on crossings is arriving at the launch gem
ALIGNED AND FAST, which is set up by the leg before; it is not a turning technique the planner lacks.
Consequence: the lever is the leg before the crossing (drive to A aimed at B, carrying speed), i.e. a two-leg plan
owned by the planner, judged against the navigator's two walking legs; the human's numbers stay the benchmark.

### 34.9 Two-leg plans in the hybrid, and where the human's crossings actually come from (09-30 afternoon)

Built (hybrid.py, planner.py, route.py): a setup order (route.choose 'setup': walk to the greedy gem A, jump to B
second, the jump costed as aligned plus SETUP_EXTRA_S 0.4); the planner drives the A leg with an exit objective
(planner.exit_dir: pickup plans get EXIT_W x the velocity along the A -> B line at the pickup; N_EXIT proposals that
run up onto that line, with the turn drill's over-steer), then the consult on B. Consult skip counters.
* Demos (42 human crossings <= 18 u): EVERY one goes to a gem of the group that has just spawned: the crossing is the
  leg between groups. At the launch gem the human's velocity is 25 deg off the line (median), the previous leg within
  45 deg of it 69 %, speed 8.4 u/s.
* Our navigator (126 pickups, 2 rounds): 63 group-clearing pickups, 41 with a crossing candidate in the new group
  (<= 18 u, void >= 3 u); best candidate within 30 deg of the heading 41 %, within 45 deg 56 %: about 8 aligned chances
  a round exist for us too.
* Route model verdict from the navigator's real pickup states (96): jump-first orders 1-2, setup orders that pay 3;
  even with the HUMAN's crossing pace as the what-if 3 (9 without the setup cost). Median setup margin +1.1 s.
* Rounds (1 each): 155 (2 setups, both picked A fast: 10.4 and 9.3 u/s, no consult on B), 148 (0 consults of 269
  evaluations: 207 skipped for angle > 35 deg, 47 no walking gain, 15 far), 149 with the gate at 60 deg (4 consults:
  2 pickup plans of 27-40 decisions = 1.7-2.6 s, not 0.3 s under the fitted walk of 1.96-2.0; 2 no plan).
Ceiling of this mechanism as the numbers stand: ~8 aligned chances a round x ~0.3 s = about 2 points. Not the 15.

### 34.10 The pure controller on KOTM (design M5 arm, first run) and its two stalls (09-30, 13:00-14:00)

rounds.py (planner alone, order by the guide, floor gems by walking distance), 3 rounds at 3x: 85, 86, 84 points; 0-2
falls; groups take 8-16 s (navigator 5.5); leg time mean 2.58 s / median 2.43 (navigator 1.36) although the pickup
speed is HIGHER (median 10.3 u/s vs 7.7): the planner does not brake into gems, it stalls between them. Per-decision
trace (rounds.py 'trace', 'legs'): 143 jumps a round on floor legs (three quarters of the broad proposals carry a jump
and nothing charged for it), and full stops in the open: (1) after ~2 s on the same plan the marble brakes to
0.1-0.9 u/s: the warm start shifts the previous best program one decision forward each step and its TAIL brake slid to
the front (Programs.shifted now keeps the tail at the end, extending the decision before it); (2) after a pickup the
next target is often behind the marble, which reverses from 12-16 u/s (the guide's order; momentum). Fixes 1 and the
jump charge (planner FLOOR_JUMP_CHARGE 0.15 in P units on plain floor legs) are in; measured next.

### 34.11 Why the pure controller stops to turn (09-30, 14:00-14:40)

Turn drill replayed through the step model (turndrill.py --model): the model reproduces every program within ~5 deg
and 0.7 u/s at 8, 16 and 24 decisions. The physics is right; the choice is wrong.
Leg time by the angle between the marble's heading at the leg start and the direction to the gem (pl_fix rounds,
137 legs): 0-45 deg 1.22 s (the navigator's pace), 45-90 deg 2.30 s (min speed 2.1 u/s), 90-135 deg 3.46 s (0.7 u/s),
135-180 deg 2.94 s. The planner brakes to a near stop to turn while the turn drill turns 90 deg in 0.9 s at speed.
Offline from three of those 90-130 deg starts (KOTM, 9-16 u/s): 142-154 of the 160 proposals were judged to FALL,
because a rolling program was judged safe to the end of the 3 s horizon and almost any full-throttle 3 s path on this
floor reaches a hole; the survivors brake, or hop a hole (hence 140+ jumps a round). Tail fix (Programs.shifted) and a
floor-jump ranking charge: 84 / 90 (before: 85, 86, 84), jumps unchanged. Judged window for rolling programs cut to
GROUND_WIN = 16 decisions past the closest pass (clearance likewise): 80 / 83, legs unchanged (45-90 deg 2.50 s).
Most decisions are 'pickup' plans, ranked by P - 0.004 x t_close (TIME_W): a braking turn at P 0.95 beats a fast turn
at 0.8 that is a second quicker. rounds.py now sets the hybrid's shortcut weights on every leg (time_w 0.015, P_SAT
0.6); measured next.

### 34.12 Pure controller, one lever at a time (09-30, 14:00-15:40; 2 KOTM rounds per step, 3x)

| step | change | points | legs starting 0-45 / 45-90 / 90-135 deg off the gem (median s) |
|---|---|---|---|
| baseline | rounds.py as in stage 5 | 85, 86, 84 | 1.22 / 2.30 / 3.46 |
| tail + jump charge | Programs.shifted keeps the brake tail at the end; FLOOR_JUMP_CHARGE | 84, 90 | (1.22 / 2.30 / 3.46 measured here) |
| ground window | rolling programs judged GROUND_WIN 16 past the closest pass, clearance too | 80, 83 | 1.31 / 2.50 / 3.04 |
| leg weights | time_w 0.015, P_SAT 0.6 on every leg | 89, 90 | 1.28 / 2.30 / 2.94 |
| rolling clearance | ROLL_CLEAR_OK 0.5 for programs that never fly | 83, 91 | 1.15 / 2.46 / 2.75 |
| turn-costed order | TURN_S 0.9 x (1 - cos) per leg in choose_order | 81, 82 | 1.25 / 2.69 / 3.01 |

Aligned legs are now faster than the navigator's (1.15-1.25 s for ~12 u; navigator 1.29). The whole deficit is the
legs that start turned away, a third of them more than 90 deg (navigator: 12 %), and no order rule changed that share:
the planner passes through each gem at 10-11 u/s and keeps going, so the group's next gem is behind it; the turn
beyond 90 deg then costs ~1 s by physics (34.8). Next: the exit objective (planner.exit_dir, built for the crossing
setup) on every leg, aimed at the order's next gem.

### 34.13 Exit objective and the next-leg turn charge (09-30, 15:40-16:20)

* exit_dir bonus (velocity along the line to the order's next gem at the pickup) on every leg: 88 (one round; the
  other game stalled after connecting and was stopped). Turned-away starts unchanged (30 %): the bonus is blind to a
  gem behind the marble (the velocity along the exit line is clipped at zero), so it never asks for a slower arrival.
* Replaced by a two-leg TIME charge (planner.exit_bonus, now a charge): the next leg's turn cost from the state at the
  closest pass, TURN_S 0.9 x (1 - cos) x min(1, speed / 6) from the turn drill, in P per decision like time_w.
  Rounds: 91, 96 (the day's best; baseline 85, 86, 84). Pickup speed 8.6-9.0 (it arrives slower when the next gem is
  behind), 45-90 deg legs 2.24 s (from 2.3-2.7), 90-135 deg 3.07, aligned 1.28; 25 % of legs still start more than
  90 deg off.
Day's arc for the pure controller: 85 -> 93 with seven levers, all physics-derived, none map-specific; the navigator
is at 152. The remaining deficit is the third of the legs that start turned away; by the turn drill those cost ~1 s
each by physics, so the answer is not a faster turn but not arriving in that state: a real multi-leg plan over the
group (speed at each pickup chosen for the leg after), which the two-leg charge only approximates.

### 34.14 The human's own jump, replayed: the model is wrong at 12-14 u/s (09-30 evening)

The operator recorded the jump the hybrid keeps declining (record_demos.py, which had silently dropped every tick since
the observation grew to 38 numbers on 09-26; fixed): from the centre gem (-23.2, 13) at 13.2 u/s, 16 deg off the
line, straight over the hole to the corner gem (-13.2, 3) in 1.31 s. From that exact state: the hybrid's gate would
have consulted (detour 3.4 u, angle 16), but the planner finds NO pickup plan (P 0.00 over 416 programs: 0 of 342
jumping programs pass within 0.85 u of the gem; 321 'fall'). In the game from the same state (crossdrill, 5 trials)
the planner jumps anyway (an escape jump) and makes it in 1.79-2.30 s.
The human's inputs run through the step model from that state (validation only): the game LOSES speed in the air
(13.2 -> 12.3 u/s) and sheds a third of it on landing (12 -> 8); the model GAINS speed in the air (-> 14.1) and keeps
11.5 after landing, overshooting the gem by 3.5 u. Over 16 recorded human jumps, by takeoff speed: model air speed
minus real +0.6 (9-11 u/s) to +1.5 u/s (13+); speed 2 decisions after landing +1.1 to +3.0 u/s too high. The planner
rejects the very jumps the human makes because its physics is optimistic about speed at 12+ u/s, in the air and at the
landing. General, map-independent, and fixable with data: collect flights and landings at 11-15 u/s with air inputs
(record3) and refit the step model; the human recordings stay the benchmark.

### 34.15 From the operator's jump to a planner that finds it (09-30, 22:00-23:00)

Three model-side causes, each measured, each general:
1. Air speed. On the step model's own held-out data (KOTM, Sprawl, Tilo), the game's horizontal speed change in flight
   falls with speed like drag (no input: -0.01 u/s a step at 4.5 u/s, -0.13 at 10, -0.28 at 14.5, -0.44 at 20, about
   -0.0011 v^2) and the forward input adds +0.33 a step at 3-6 u/s but nothing by 13-16; the model keeps +0.18-0.25 at
   11-16 u/s with the input and under-shoots the drag without it, on every map. Over a flight that is +1.4 u/s.
   Fix: planner.AIR_DRAG_RESID 0.0009: airborne steps lose AIR_DRAG_RESID x v^2 of horizontal speed in simulate().
   Validation on 16 recorded human jumps (inputs replayed through the model, benchmark only): air speed error
   +0.73/+1.17 u/s (takeoff 6-11 / 11-20) -> +0.05/+0.14; landing position error 1.01/1.31 u -> 0.69/0.54; speed two
   decisions after landing +1.20/+2.07 -> -0.16/-0.22.
2. After the landing the model predicts a short bounce where the game rolls and brakes (support 0 for 5 decisions);
   left for the model refit: it costs ~1 u of lateral error at the end of a fast crossing.
3. Proposals. With the model corrected, the human's inputs reach 1.17 u from the gem; the planner's own line programs
   only 1.75: they steer at the gem in the air (+-8 deg) or not at all, and after landing either steer at it at full
   throttle (too wide at 10 u/s) or brake straight. The human held ~100 deg off the velocity in the air and brake-turned
   after landing. Added, as proposal families the judge ranks like any other: air input 30-110 deg past the gem's
   direction (either side) in _line, and a brake-turn after landing (60-150 deg past the gem's direction) in
   _land_steer. From the operator's state: p_succ 0.79-0.95, pickup predicted at decision 17 (1.09 s; the human 1.31).

### 34.16 v3 on the dev set, and the falls at speed (09-30, 23:00-23:40)

In the game from the operator's state (5 trials, planner with 34.15's changes): 5/5, three in 1.15 s (the operator
1.31), two late pickups after a miss (2.8, 3.1 s), no fall. Dev set, v3 (air residual, wide air inputs, brake-turn
after landing; 86 starts): 81 ok, 5 falls, median leg 1.73 s (v1 2.05, walking 2.10); aligned starts 1.54 s vs 2.00
walking, 30-60 deg 1.89 vs 2.19. Paired v1 vs v3 on 26 starts: 26 vs 25, v3 0.16 s faster on both-ok legs, expected
time equal because of the falls. The five falls: a press at 11-13 u/s that never fired, the marble rolling off the
lip (in one case pressed with floor under it, 2 u before the edge). Engine data (12 KOTM shards, 3,386 presses on
flat floor with 0.8 u ahead): the jump fires 99.2-99.8 % at 10-14 u/s, 98.5 % at 14-16. So the presses acted late:
at 12 u/s the marble covers 0.8 u a decision and a press timed from the predicted path arrives over the void. Fix:
jump_fires needs JUMP_LIP_U + JUMP_LIP_K x speed of floor ahead (one decision of travel, JUMP_LIP_K 0.064), so the
plan presses one decision earlier at speed. v4 = v3 + this; measured on dev next, v3 on the untouched test set.

### 34.17 Crossing drill, final numbers (09-30, 22:45)

| planner | set | starts | ok | falls | leg median | walking fitted | expected time (fall = walk + 3.5 s) |
|---|---|---|---|---|---|---|---|
| v0 (stage 5b) | dev | 35 | 34 | 1 | 1.98 s | 2.11 s | 2.32 s |
| v1 (+ saturation) | dev | 26 | 26 | 0 | 2.05 s | 2.11 s | 2.06 s |
| v3 (+ air residual, wide air inputs, brake-turn) | dev | 89 | 84 | 5 | 1.73 s | 2.10 s | 2.15 s |
| v4 (+ speed-scaled lip margin) | dev | 89 | 85 | 4 | 1.79 s | 2.10 s | 2.09 s |
| v0 | TEST | 36 | 35 | 1 | 1.98 s | 2.06 s | 2.22 s |
| v3 | TEST | 91 | 90 | 1 | 1.73 s | 2.07 s | 1.95 s |

Paired on the test set, v0 vs v3 (36 shared starts): 35 vs 36 ok, v3 0.30 s faster on the legs both flew, expected
time 2.22 vs 1.84 s. By start angle (v4, dev): within 30 deg 1.54 s vs 2.00 walking; 30-60 deg 1.98 vs 2.19. v4's
four dev falls are all roll-ins during the approach (no press at all), not failed takeoffs. v4 goes to the gate.

### 34.18 KOTM gate with the v4 planner (09-30, 22:45-23:15, 3x, 4 games x 2 rounds per arm, same session)

| arm | rounds | points | mean (sd) | best | falls a round | planner |
|---|---|---|---|---|---|---|
| navigator alone (hybrid, shortcuts 0) | 8 | 156 159 161 151 146 151 160 146 | 153.8 (6.1) | 161 | 1.62 | none |
| hybrid, consult at the pickup + setups (g6d, instant handback) | 8 | 144 145 152 140 148 150 150 151 | 147.5 | 152 | 2.5 / 1.75 | 1.5 shortcuts, 2 setups |
| hybrid, same, failed consult -> safe continuation (g6e) | 8 | 146 154 153 142 147 142 140 142 | 145.8 (5.3) | 154 | 1.88 | 1.6 shortcuts, 2.4 setups |
| hybrid, setups OFF (g6f) | 8 | 148 151 151 147 148 147 149 154 | 149.4 (2.4) | 154 | 1.12 | 1.9 shortcuts, 0.9 failed consults |

Accounting (g6e legs): the crossings pay (shortcut legs 1.4-2.4 s for walks of 19-25 u that the navigator's fit puts
at 2.3-2.9 s; 15 of 15 committed consults succeeded in g6f) but the setup legs lose (the planner's floor leg to the
first gem 2.0-2.9 s where the navigator walks it in 1.3 s, 2.4 a round) and the handovers cost: g6d had a fall within
3 s of a handback every round (instant handback at ~10 u/s beside the hole after a failed consult; now the safe
continuation), and each failed consult (0.9 a round) ends in braking to 4 u/s before the handback. Net with setups
off: 4 points below the navigator (-4.4 +- 2.3), fewer falls, lower variance, best round 154. The operator's goal
(a round above 167 with the planner) is not met; the navigator's best tonight was 161.
What would close the gap, in order: a cheaper exit from a failed consult (hand back at once when clear of edges), a
faster handback after a crossing pickup, and the consult on more legs than the greedy target (the second-nearest gem
across the hole) once the planner's time estimate at the consult is trusted. The remaining structural ceiling of the
hybrid on this map is a few points; beyond it lies the navigator's own driving (pickup speed, order), which is
training, off-limits under the standing rule without the operator.

## 35. Night of 09-30 / 10-01: the hybrid's handovers, real-time planning, needless jumps (autonomous)

Operator, 23:30, after watching the hybrid at 1x: the crossings are real time savers, but (1) after a crossing pickup
the marble carries its momentum far past the gem, stops and comes back where the human barely passes the gem and
turns at once; (2) the frame rate drops to ~0.5 fps while the planner thinks, unacceptable for a real game; (3) a
couple of needless jumps on open floor. Goal: a round above 167 with the hybrid. Approved: the "what is left" items
(cheap failed-consult exit, faster handback, consulting the across-gem when not nearest) and training if the rest is
optimized first.

### 35.1 Momentum after the pickup and the handovers (hybrid.py)

* EXIT_TO_NEXT: on a consult leg the planner's exit_dir is set toward the navigator's next gem, so the crossing plan is
  charged the next leg's turn time (planner.exit_bonus, 34.13): a plan that lands slower or aligned for the next gem
  outranks one that flies through at 13 u/s.
* HANDBACK_ANY_DIR: after a planner pickup the marble is handed back as soon as the continuation is safe, clear of edges
  (>= 1 u) and on the floor, whatever the direction (the old rule waited for the marble to roll DOWN the walking field
  toward the next gem, 10-25 decisions).
* A failed consult hands back at once when the state is clear and on the floor (consult_back), else the continuation.
* Jump counters by owner (jumps_planner / jumps_nav) for the needless-jump question.

### 35.2 Gates through the night (8 KOTM rounds each, 3x, 4 games x 2 rounds)

| batch | change | mean (sd) | best | falls nav / after handback | shortcuts | planner jumps |
|---|---|---|---|---|---|---|
| g6e_nav | navigator alone | 153.8 (6.1) | 161 | 1.62 / - | - | - |
| g6f | consult at the pickup, setups off | 149.4 (2.4) | 154 | 1.12 / 0.12 | 1.9 | |
| g6g | + exit charge toward the next gem, handback when clear, cheap failed-consult exit | 151.0 (7.9) | 164 | 1.75 / 0.50 | 2.0 | 5.5 |
| g6h | + no jump outside a committed pickup plan, keep braking instead of the capped handback | 154.2 (3.7) | 161 | 0.88 / 0.25 | 1.5 | 3.5 |
| g6i | + consult on the aligned across-gem even when not the nearest | 137.9 | 146 | 3.0 / 1.12 | 3.1 | 6.6 |

g6i: 11 retargets a round, 70 % of the consults failed (31 approach, 12 pickup plans not 0.3 s under the walk), each
failure retargeting the navigator and handing a 10-13 u/s marble back beside a hole. Off (ACROSS_CONSULT False), and
the instant handback after a failed consult now also needs the marble at or under HANDBACK_FAST 9 u/s. g6h is the
configuration that matches the navigator with half its falls; g6j = g6h + the speed condition, measured next.
Needless jumps: the planner's, in approach and continuation plans during consults (5.5 a round in g6g for 2
crossings); now suppressed outside a committed pickup plan (CONT_JUMP_NOW 50 in the continuation as well).

### 35.3 The takeoff margin had killed the operator's jump; latency (10-01, 00:00-00:20)

From the operator's recorded state the planner again found NO plan (p 0.00): the speed-scaled takeoff margin of 34.16
(JUMP_LIP_U + 0.064 x speed = 1.44 u of floor ahead at 13 u/s) forbade a press with ~1 u of floor left, which the game
fires 99 % of the time (34.16) and the operator made. The margin exists for presses timed from the PREDICTED path
several decisions out, where the model's position error has built up; a press acting on the next step needs none.
jump_fires now takes the rollout step: margin x min(1, (t - 1) / 2). The operator's state: p 0.70-0.90 again, with or
without the exit charge. g6j/g6h ran with the strict margin (fewer crossings than possible).
Latency of one consult decision from that state (GPU free, the model's CUDA graphs): 457 programs 411-457 ms; 257
programs 311-324 ms; horizon 32 instead of 48: 209 ms; 65 programs: 204 ms. The floor is ~200 ms of fixed cost
(terrain features on the CPU per step, the robust re-check); a real-time budget at 1x is 64 ms, at 3x 21 ms. A
2x cut is available by budget; below that the feature extraction has to move to the GPU or the rollout be shortened.
Not done tonight: the score comes first.

### 35.4 The planner as a jump oracle, no takeover (10-01, 00:30)

Six hybrid batches tonight (35.2, 35.3) all sit at or under the navigator (147-154 vs 153.8 / 155.4): the crossings
gain ~2 points a round and the takeover machinery (consult decisions driven by the planner, continuations, extra
jumps, stuck-breaks after handbacks) costs more. The variant without any handover: the navigator keeps driving; when it
is on the floor at >= 6 u/s, its floor target lies across a void on the line within 18 u, the heading within 40 deg of
the line and the void begins within 3 u along the velocity, the planner's model rolls out 8 fixed programs (jump NOW,
air input none / at the gem / 60 and 100 deg past it either side, land-steer to the gem after, one plain brake) and
judges them as any plan; if the best P(success) >= 0.5 the jump bit is set on the navigator's own action (planner.
jump_oracle, hybrid ORACLE_*). 77 ms a query on a free GPU (8 programs, 48 steps); the operator's state: 0.91.

### 35.5 Oracle results and the night's conclusion (10-01, 00:30-00:45)

| batch | arm | mean (sd) | best | falls | planner |
|---|---|---|---|---|---|
| g6e_nav, g6l_nav | navigator alone (two batches) | 153.8 (6.1), 155.4 (4.3) | 161, 162 | 1.62, 0.75 | none |
| g6m | + oracle (P 0.5, >= 6 u/s), consults off | 152.0 (2.1) | 156 | 1.00 | 9.6 oracle jumps a round, 155 ms a query |
| g6n | + oracle (P 0.75, >= 8 u/s) | 153.8 (1.9) | 156 | 1.12 | 3.0 oracle jumps at 9.6 u/s |
| g6o | oracle + the pickup consult | 152.1 (4.0) | 161 | 1.0 (0 after handbacks) | 2.3 oracle jumps, 1.5 shortcuts |

Every hybrid variant tonight lands at the navigator's level, within round noise: the takeover design (consult at the
pickup) 147-154 with best rounds of 164, the oracle 152-154 with the lowest spread of any arm. The crossing skill is
real (test set 90/91 at 1.73 s; the operator's own jump reproduced at 1.15 s) but on this map it is worth ~2 points a
round and the integration costs eat it. No round above 167; the navigator's best tonight 162, the hybrid's 164.
Not started: training. The one training idea the physics work makes possible, the oracle inside the training loop so
the policy learns routes around a jump it can rely on, needs the planner's model in the trainer's decision loop
(8 instances, one CUDA context) and is a build plus an unattended multi-hour run with the history of post-26113 runs
losing points; it is the next thing to do WITH the operator, not alone at 1 am.

### 35.6 Crossings into the centre (10-01 morning, operator's observation)

The operator saw the hybrid never jump from the outer ring into the centre. Numbers: of 104 committed crossings in
last night's rounds, 2 went to a centre gem (consults toward the centre 13, 11 failed); in the drills the direction
was asked only 5 times (the navigator's corner pickups rarely point at the centre) and made 5/5, but with the first
pickup plan at decision 5-7 (0 for centre -> corner). Offline from a corner start aimed at the centre (9-13 u/s): the
line programs reach the gem (p_pick > 0.5 on 11-30 of 96) but the judge marks nearly all 'fell' / 'not settled':
after the predicted landing on the centre blocks the model has the marble CLIMB 3 u (a bounce off the block edges or
the 1.3 u slots) and fall. In the game (crossdrill, 6 starts, 9-13 u/s): 6/6, no fall, 0.90-1.28 s against 2.2 s
walking, the planner finding its plan after 3-7 decisions as the state nears the lip. So the model's known corner /
lip bounce (stage 4) also hits centre landings; the hybrid's 3-decision consult then expires first. CONSULT_DEC 3 -> 8;
the bounce itself is a step-model error to refit (centre landings), listed with the post-landing bounce of 34.15.

### 35.7 Across-gem consult, second try (10-01, 10:40-10:55)

CONSULT_DEC 8, ACROSS_CONSULT on at 20 deg: 138.6 (sd 4.0), 6.25 crossings a round (centre 9 of 42 legs, side 28,
corner 5), consults 50 of 65 committed, legs 1.3-2.9 s; falls 3.25 a round (planner 1.25, navigator 2.0). With
CONSULT_DEC 8 and ACROSS off (g6p): 148.5, 2.4 crossings, 1.25 falls, none into the centre. More crossings cost more
falls than they save on this map as long as the model misjudges the centre landing (35.6) and the navigator gets the
marble back fast beside holes. ACROSS off again; CONSULT_DEC stays 8. The next general step is the step model: refit
or correct its landing-and-bounce behaviour on the centre blocks and after fast landings (34.15 item 2, 35.6), with
the crossing drill (now 91 test starts plus the 6 corner -> centre starts) as the check.

## 36. The physics of a crossing, measured against the engine (2026-10-01, 11:00-)

The operator (10-01 morning): the hybrid never jumps from the ring into the centre, "fix the physics". Measured
with fixed programs instead of the planner (nav/learned_nav/centredrill.py): from 42 ring starts (16 directions round
the centre gem, RUNUP 4 u beyond the far lip, 9/11/13 u/s, aimed at the gem) the marble rolls straight at the gem and
presses the jump when the void begins within 1/2/3 u ahead, flies with no input / the line to the gem / 60 deg past
it, and brakes after the landing: 378 trials in the game, then the identical input sequence through the step
ensemble (planner.simulate) from the recorded start.

### 36.1 What the game does in the air (and what the model did)

| | game | step model (before) |
|---|---|---|
| horizontal speed, no input | constant to 0.01 u/s over the flight | -12 % over a 0.9 s flight (plus the AIR_DRAG_RESID correction of 34.15 on top) |
| horizontal speed, input toward the gem | +0.44 u/s a decision | +0.30 u/s, then the residual took it to +0.10 |
| landing decision | 16 in both cases | 20-22 without input, 15 with |
| apex | 22.17 | 21.86 |
| landing, no input | BOUNCE: vz +2.8 after -6.7, a 0.3 u hop, horizontal -0.3 u/s | not landed at all (sank through the block) |
| landing, with input | SLIDE: speed kept, vz folded into the floor plane (13.2 + 6.7 -> 19.0 u/s horizontal) | 1.6-2.1 u short |

The engine (marble.cc, OpenPQ-TGEMIT mbx): with no contact the acceleration is gravity (20) plus airAcceleration (5)
times the camera-frame move (up to sqrt 2 on the adapter's diagonal): no drag, no cap. On contact (velocityCancel):
a move that is not centred and an approach shallower than maxDotSlide (0.5, i.e. the normal speed under half the
total) slides, keeping |v|; otherwise a bounce with bounceRestitution 0.5 and a bounce-friction impulse (0.2).
The lateral error of the model was small all along (0.4 u mean): the "1.3 u drift" of 35.6 was the along-track and
timing error seen sideways on diagonal crossings.

### 36.2 The jump key acts one decision after the steering

Turn drill (steer at decision 0: the heading moves in state 1 -> 2) against the centre drill (jump sent at d: vz
rises in d + 2 -> d + 3, 183 of 187). The step model had learned it (it fires on the PREVIOUS reply's key, Jp, with
exactly the measured impulse) but the jump prior (JUMP_PRIOR, stage 5b) fired on the current key: every planned
flight ran one decision early and 0.7-0.9 u ahead, which JUMP_LIP_K (34.16) then papered over with a speed-scaled
margin. Also found: at the fire step beside a lip the step model predicted a 6.5 u/s horizontal loss (the game: +0.4).

### 36.3 Changes (planner.py)

* AIR_EXACT: airborne steps and flat-floor landings from the engine's physics (air_exact): flight under gravity and
  5 x the input, the landing resolved as slide / cancel / bounce at the contact time within the step, the hop of a
  bounce flown on, a bounce under 1.5 u/s up counts as settled. A step near a wall or lip (anything within 2.4 u above
  the marble at its end, the level under the midpoint higher than the landing floor) stays with the step model.
* AIR_DRAG_RESID 0.0009 -> 0 (it was compensating the model's weak air control).
* JUMP_PRIOR fires on Jp (the key sent one decision before the steering reply), the fire step's horizontal motion is
  the air step under the input still acting; JUMP_LIP_K 0.064 -> 0.
* On the same inputs the model's landing now matches the game's to 0.1 u and the same decision when there is an air
  input; without input the bounce and hop match to 0.05 u.

Remaining (ground model): braking after a landing ~12 u/s^2 against the engine's 14.4; acceleration on the run-up
~4 % high; the 7 u centre block at 11-13 u/s is at the physical limit of a brake-only landing (one fixed-program
success in the game survived only by a bumper under the pit, which no model has).

### 36.4 Drills with the corrected physics (10-01, 12:00-12:30)

Crossing drill, untouched TEST set (91 navigator pickup states, planner v5 = exact air + late key + air brake):
90/91, mean 1.87 s (v3 before the physics: 90/91, 1.91 s): no regression. New RING set (datasets/learned_nav/cross/
starts_ring.json: the 42 centredrill starts, aimed at the centre gem): v5 35/42 (3 falls, 4 timeouts), median
1.1 s against the route model's 1.2-2.6 s walk; by direction every start from 0/45/67.5/90/180/202.5/225/270/
315/337.5 deg makes it (one timeout at 0 deg 13 u/s: an approach plan jumped from 7 u out and overflew the gem
1.4 u up), the failures sit at 112.5-157.5 deg: fast slide landings at 18-19 u/s that cannot stop on the 7 u block
(falls), a landing into the groove whose far wall throws the marble 4 u up (timeout), and one start (135 deg,
13 u/s, 4 u to the lip) with no plan at all, where the marble ran off the lip: the approach branch picks the
least-cost plan even when every plan is vetoed, and the escape jump only triggers when the chosen plan jumps.
Air-brake proposals (v6): 34/42, 2 falls (one new: 180 deg 9 u/s, a pickup plan at p 0.26-0.31 followed to the
lip and then forbidden to jump by P_JUMP 0.35 > P_GO 0.25; P_JUMP = P_GO now).
Human jumps (18 recorded presses at 6+ u/s, inputs replayed): at the game's landing decision the model is
+0.4-0.7 u ahead along the line, 0.4 u lateral, 0.0 u in height (before: lateral 0.33, along +0.19 with the drag
residual hiding a short flight).

### 36.5 KOTM gates with the corrected physics (10-01, 12:20-13:10)

| batch | arm | rounds | mean (sd) | best | falls a round (nav / planner) |
|---|---|---|---|---|---|
| g6e_nav, g6l_nav (09-30, 10-01) | navigator alone | 8 + 8 | 153.8 (6.1), 155.4 (4.3) | 162 | 1.6 / - |
| g6q (10-01 morning, old physics) | hybrid: oracle + consult at the pickup | 8 | 152-154 | 164 | ~1.2 |
| g6r | + exact air, late key, air brake; BIG consults from any heading, window 24 | 8 | 141.3 (3.9) | 146 | 1.9 / 1.9 |
| g6s | + support guard, P_JUMP = P_GO; big consults only when aligned, window 16 | 8 | 147.1 (4.6) | 153 | 1.2 / 1.4 |

The big-detour consults do produce centre crossings (hybrid shortcut legs into the centre 1.8-2.8 s) but the
navigator with the oracle already takes the same legs in 1.4-2.0 s (it jumps on its own; walk 21 u legs in 1.47 s),
so the takeover gains nothing, and every failed big consult ends in the planner's "safe continuation" toward the
navigator's gem, where 10 of the 11 planner falls of g6s happened ('c' decisions after 8-12 'a' decisions with
p 0): the continuation drives the marble off the strips and the centre block's edges. The ground model near edges
is still the learned one; the support guard (no floor under the bottom = airborne) did not stop these, the marble
was on the floor and simply rolled over the edge while the rollout predicted a turn or a stop that the real brake
(14.4 u/s^2 against the model's ~12) should have made easier, not harder. BIG consults OFF (BIG_GAIN_U inf); g6t =
the g6q configuration with the new physics.

### 36.6 g6t: the g6q configuration with the new physics (10-01, 12:45-13:00)

g6t (oracle + consult at the pickup, BIG consults off, exact air, late key, air brake, support guard, P_JUMP 0.25),
8 rounds: 149.2 (sd 1.9), best 153; 3.0 oracle jumps and 2.4 shortcuts a round, falls 1.4 navigator / 0.4 planner.
g6q with the old physics: 152-154. Navigator alone 153.8 / 155.4. So the corrected physics made every rollout more
honest (the drills say so) and the hybrid's KOTM score no better: the crossings it adds are worth about what the
handovers cost, as before, and the spread is now tight. Goal (> 167 with the planner) NOT met today.
What is left in this direction: (1) the planner's turn-and-run-up from slow, misaligned states (its takeovers 1.8-2.8 s
against the navigator's 1.4-2.0 s on the same legs); (2) the learned ground model at edges (the continuation falls
of g6s: rollouts that stop or turn before an edge the real marble rolls over, while the real brake is stronger than
the model's); (3) the real-time path (a consult decision still 300-450 ms, ~1.1 s on the gate machines with four
games). Nothing committed in this repo.

## 37. Gem order: whole-spawn chooser and training on it (2026-10-01 evening - 10-02 night)

The operator (1x viewing, 10-01 evening): the marble takes grossly inefficient routes through some spawns, e.g. two
gems in the centre and two on the ring, entering the centre at speed: takes one centre gem, carries on to the ring,
and comes back for the second centre gem. Cause: the navigator's target comes from the GREEDY chooser
(nav/real_run.choose: straight-line distance + 1.6 x speed x (1 - cos turn), sticky at 60 %), which never looks past
the next gem; the policy only sees "target" and "next". The whole-spawn planner (nav/gems.plan_tour, log 28.50) was
parked on 09-26 after a 2.6 h training run on it gained nothing (the policy slowed and overshot gems, 28.52).

### 37.1 Choosers on the untrained policy (10-01, 19:30-21:00)

| gate | chooser | mean | u between pickups | s per gem | falls |
|---|---|---|---|---|---|
| g6u | greedy (the policy's training chooser) | 149.9 | 10.99 | 1.52 | 2.25 |
| g6v | route.Route.order, walk-or-jump legs (new) | 139.2 | 11.09 | 1.62 | 2.75 |
| g6w | route.Route.order, walk-only legs | 139.0 | 11.06 | 1.62 | 2.75 |

Same lesson as 28.50: on routes it was not trained on the policy falls (14 of 22 falls near the centre) and the
distance does not even shorten. route.py gained Route.order (whole-spawn order over the fitted walk/jump leg times,
ORDER_SWITCH_S 0.3, ORDER_JUMPS) for later; the hybrid's ORDER_TOUR = 'walk' now uses the TRAINER's own chooser
(gems.plan_tour over walk-only terrain fields, exactly vec_worker.pick with NAV_TOUR=walk) so training and play
see identical targets.

### 37.2 Training on the whole-spawn chooser (10-01 21:14 - 10-02 01:37)

Operator: "launch it ... train for at least 4 hours or until metrics drop precipitously and stay down; if metrics
improve keep training as long as they keep improving; if training does not help, think of something else". From
nav_latest = 26110, NAV_TOUR=walk; at 22:17 restarted with all 8 instances on KOTM (operator: Islands out,
"everything should be for kotm now"). Training-side KOTM rounds (8 games at speed): 50-round means 138 -> 143-145,
falls 2.8 -> 2.1 a round, speed 8.0 u/s throughout (the 09-26 run lost 0.4 u/s), sgem 1.47; best 50-round mean
145.2 at 26592 (nav_night_26592_145.pth), plateau from ~00:30 at 142-144; stopped at 01:37, endpoint
nav_tour_end_27004.pth. Dashboard fixed meanwhile (parsed all 137 logs per push: 7.4 MB / 9 s; now the 4 newest).

### 37.3 Hybrid gates on the trained policy, tour chooser (10-02 01:37-)

| gate | navigator | rounds | mean | best | u between pickups | s per gem | falls |
|---|---|---|---|---|---|---|---|
| g7a | snapshot 26592 | 4 | 153.2 | 162 | 10.40 | 1.47 | 2.25 |
| g7b | endpoint 27004 | 4 | 154.2 | 159 | 10.53 | 1.47 | 1.75 |
| g6u (ref) | 26110, greedy | 8 | 149.9 | 153 | 10.99 | 1.52 | 2.25 |
| navigator alone (ref) | 26110, greedy | 16 | 153.8 / 155.4 | 162 | 10.7 | 1.42-1.47 | 1.6 |

The route gain is real this time (5 % less distance per gem, no speed loss) and the hybrid is back at the
navigator's level with 4-round means; g7c = 8 more rounds on the endpoint.
g7c (endpoint 27004, 8 rounds): 141, 147, 159, 155, 162, 151, 155, 154 = 153.0; with g7b the endpoint stands at
153.4 over 12 rounds, best 162, falls ~1.9 a round. Equal to the navigator alone (153.8 / 155.4), not above it,
and no round above 167. The 160+ navigator rounds all had zero falls; the hybrid's rounds average ~2, worth 10-15
points, so falls are the lever now. 02:05: training RESUMED from the endpoint (same config: KOTM only, walk tour)
for the rest of the night with the 50-round snapshots; the training-side fall rate had been falling (2.8 -> 2.1)
when it was stopped. Morning: gate the best snapshot with the hybrid (NAV_CKPT=<snapshot> ORDER_TOUR 'walk').

### 37.4 Night 2 of training and the morning gate (10-02 02:01-06:40)

Training resumed 02:01 from 27004 (KOTM only, walk tour): training-side 50-round means 145-148 (new highs at 27117
146.3 and 27493 148.2), speed 8.1 u/s, falls 1.7-2.1; plateau; paused 06:16 at update 27928 (nav_tour_end_27928.pth).
Hybrid gate g7d on the best snapshot nav_night_27493_148.pth (ORDER_TOUR 'walk', 8 rounds): 159, 158, 158, 162,
162, 152, 150, 162 = 157.9 (sd 4.3), best 162. First hybrid batch ABOVE the navigator alone (153.8 / 155.4 on
greedy routes). Still no round above 167. Training resumed again at 06:40 from 27928.

### 37.5 Day 10-02: training continued, second snapshot gate (06:37-11:40)

Training 06:37-11:17 from 27928 (all KOTM, walk tour): 50-round highs 149.9 at 28328 and 150.7 at 28897, falls
1.5-1.9 a round, speed 8.2 u/s, sgem 1.43; endpoint nav_tour_end_28954.pth. Gate g7e, hybrid on
nav_night_28897_151.pth (8 rounds): 163, 163, 164, 152, 158, 157, 162, 152 = 158.9 (sd 4.5), best 164
(g7d on 27493: 157.9, best 162). The hybrid's score follows the training-side metric, so training stays on while
it trickles up (~1 point per 500 updates). Dashboard: per-map charts fed from the NAV line in single-map runs;
summary snapshot write via rename (OneDrive). Training resumed 11:35 from 28954.

### 37.6 Navigator alone vs the hybrid on the same snapshot (10-02, 14:55)

Operator: "is the only way we're making good scores the navigator-only runs?" and "please tell me we've been
training on hybrid and not navigator only". Facts: the trainer trains the NAVIGATOR; the planner has no trainable
part and is only added at play time. What changed in this run was what the navigator trains on (whole-spawn gem
order, KOTM only). A/B on nav_night_28897_151.pth, 8 rounds each, same chooser (ORDER_TOUR 'walk'):

| arm | rounds | mean (sd) | best | falls nav / planner | u/gem | s/gem |
|---|---|---|---|---|---|---|
| navigator alone (NAV_PLANNER_OFF=1, --shortcuts 0 --rescue 0) | 162 163 170 165 150 156 157 158 | 160.1 (5.8) | 170 | 1.00 / 0 | 10.56 | 1.429 |
| hybrid (oracle + consult at the pickup) | 163 163 164 152 158 157 162 152 | 158.9 (4.6) | 164 | 0.88 / 0.62 | 10.44 | 1.428 |

The planner is worth about -1 point here (inside noise): its 0.6 shortcuts and 1.5 oracle jumps a round save
~0.1 u per gem and no time, and its 0.6 falls a round give it back. The navigator alone on the trained policy
reached 170 (a round above 167, but without the planner, so not the operator's goal). Also found and fixed today:
the summary and the dashboard dropped every round with more than 130 gems as a "merged double round", hiding 17
of today's best training rounds (165-167); cut now 190. Training resumed 14:59 from 29700.

### 37.7 Afternoon of 10-02: the analytic jump check (stopped), training stopped, repo cleanup

Operator at 15:21: "try to train the navigator and planner together and save the best checkpoint so far".
Best checkpoint copied to models/nav/nav_best_20261002_tour_28897.pth. The learned rollout cannot run in the
trainer (80-400 ms a decision vs ~440 decisions/s over 8 games), so an analytic "jump now" check from the engine's
constants was written (nav/learned_nav/fastjump.py: the key fires 2 decisions later, the fire step adds JUMP_DZ /
JUMP_VZ, ballistic flight with or without the input toward the gem, landing on the gem's floor level with floor
around it, the gem reachable over continuous floor, stopping room after a slide landing; 0.2 ms a call). Against
the drills: no-input flights 10/11 recall with 10 false goes in 120 (the groove), planner-chosen crossings 62/90
on the test set and 18/34 on the ring set, 18 "jump" verdicts in 203 pickup states. The operator stopped the work
at 15:28; nothing is wired in. 15:30: training stopped on the operator's order; endpoint nav_tour_end_29794.pth,
nav_latest = 29794. Then: *.npz gitignored and untracked (1,276 files, 72 MB, kept on disk); the four unpushed
commits (2.9 GB, 2.5 GB of it stage-3 .npz in "Phase 3 done") rebuilt with plumbing (read-tree / rm --cached /
commit-tree, same authors, dates and messages) without the .npz and the intermediate .pth over 3 MB (kept:
nav_latest, nav_best_20261002_tour_28897, nav_night_28897_151, step3_0-2, flight3, guide_ttg), 150 MB, pushed as
7e142edcf; the old history is on backup/navigator-architecture-before-rewrite-20261002.

### 37.8 Where this leaves the project (10-02 16:30)

* KOTM: navigator alone (28897, whole-spawn order) 160.1 mean with a 170; hybrid 158.9 with a 164. The planner is
  not the lever on this map; the navigator's falls (1.0 a round, 70 % beside the holes at 7-8 u/s) are: every
  round above 165 had none.
* The planner's physics are right (log 36) and it crosses gaps the navigator cannot, which matters on maps with
  real jumps; on KOTM the navigator's own jumps and routes cover what it offers.
* Training with the whole-spawn chooser works (no speed loss) and trickles up; a continued run from 29794 is
  the cheapest next gain, judged by hybrid/navigator gates on snapshots (NAV_CKPT), not by the training bars.
* Open builds: the planner's jump hint inside the training loop (fastjump.py is the cheap form; the operator
  stopped it), the real-time consult path (features on the GPU), the learned ground model at edges.

## 38. Powerups (2026-10-02 evening; ml_agent/POWERUP_PLAN.md)

Operator: push KOTM past the 170 ceiling with powerups, aiming at ~180; usage must transfer to every map. In scope,
exactly seven: Super Speed, Super Jump, Helicopter, Super Bounce, Shock Absorber, Mega Marble, Blast; opponents out.
Inventory of the 60 Hunt mission files: Super Jump 253 items, Super Speed 234, Blast 222, Mega 186, Helicopter 127,
Shock Absorber 9, Super Bounce 8 (KOTM: 3 SJ, 3 SS, 2 Blast, 1 Mega; Bounce/Shock only on Citadel, Sweep,
ParkourPeaks, Apex, TripleTrail; Helicopter e.g. Promontory x12).

### 38.1 Phase 1: the bridge (observer.cs, mlAgent.cs, protocol.py, obs.py, env.py)

* Observation NAV_OBS_V7: 23 numbers appended after the spin (raw 38-60, protocol.RAW_POW_*): held type (1-7),
  blast meter 0..1, special-blast armed, mega active, mega seconds left, bounce / shock / helicopter seconds left,
  then the 3 nearest in-scope items x (type, camera-relative dx, dy, dz, seconds to respawn; 0 = there). Read on
  the listen server from ClientGroup.getObject(0).player (powerUpData, megaSchedule, powerupSchedule[id]) and
  from the mission's Item objects (scanned once per mission; _respawnSchedule gives the time left). The policy
  vector (obs.py VEC_POW, 26 numbers): held one-hot, meter, special, mega + time, effect times, the nearest item
  (type one-hot, bearing, distance, dz, respawn). VEC_DIM 61 -> 87; nav_latest and the best checkpoint migrated
  with zero columns (nav_v7_latest_29794.pth = nav_latest, nav_v7_best_28897.pth; V6 copies kept).
* Action: word 8 = powerup yaw, word 9 = fire the regular blast (input_useBlast); format_action(pow_yaw,
  use_blast). RADIUS control word (collision radius, mega flag, the yaws, meter) for the drills; YAWSET diagnostic.
* The use key, like the jump key, fires two decisions after it is sent; Super Speed boosts along the marble's camera
  yaw AT THE FIRE TICK, so the bridge latches the powerup yaw for $MLAgent::PowYawHoldTicks (12) ticks over the
  steering yaw (the steering keeps working: it is rotated by the current camera yaw). Before the latch every boost
  went along world +y (yaw 0). Yaw convention: forward = (sin yaw, cos yaw); yaw pi/2 fired along world +x.
* The engine's trigger only fires its own powerups (doPowerUp ids 1-5); the Mega Marble (id 6) is script-driven
  and is used through commandToServer('UsePowerup') (the mouse click's path), sent by the bridge once per use.

### 38.2 Phase 2 measurements on KOTM (nav/learned_nav/powdrill.py; logs/learned_nav/powerups/)

| powerup | measured |
|---|---|
| Super Jump | +20.0 u/s vertical impulse at the fire tick (18.6 seen one decision later); from the floor |
| Super Speed | +25.0 u/s ADDED to the velocity along the camera yaw, whatever the velocity (from rest 24.1 on the floor; at 6.3 u/s -> 31.3; in the air +25.0 with vz untouched); one -x trial gave no boost (to retest) |
| Blast, regular | meter fills 0 -> 1 in 25 s, usable from 0.20; impulse 10 x sqrt(meter) upward: vz seen one decision after the fire 3.11 (m 0.20) ... 8.61 (m 1.00) = 10 sqrt(m) - 1.28; apex 0.46 u (0.2) to 2.40 u (1.0); meter -> 0.03 after |
| Blast pickup (special) | fires at 10 x 1.03: vz 8.91 one decision later, apex 2.55 u; arms with meter 1 |
| Mega Marble | radius 0.18975 -> 0.6666 for 10 s (this mode; 5 s in competitive Hunt); activation pops the marble up when within 1 u of floor; floor acceleration slower (14.8 vs 18.2 u/s after 2.3 s), braking stronger (~17 vs 14 u/s^2); jump apex 1.79 vs 1.33 u; JUMP-THEN-MEGA: +25 u/s vertical when activated 1-2 decisions after the jump key (apex 22-24 u), +17-18 at 3-4 decisions (apex 12-14 u) |
| respawn | items reappear 7.0 s after a pickup; the same type is refused while held; a different type replaces the held one |

Not yet measured: Helicopter (gravity x0.25, air control x2 in the engine, 5 s), Super Bounce (restitution 0.9,
5 s), Shock Absorber (restitution 0.01, 5 s), the mega spin-launch on tarmac (Promontory), mega over narrow holes.

### 38.3 Phase 2 on the other maps (ParkourPeaks: bounce and shock; Promontory: helicopter, tarmac)

powdrill.py now reads the item positions from any Hunt mission file (every `new Item` block) and picks an item up by
dropping onto it; Promontory_Hunt geometry exported (special material friction_high, friction 1.5 = the tarmac).
* Super Bounce (ParkourPeaks, 2 items): after the use the engine state and the script schedule both show id 3 for
  4.8 s; a 3 u drop bounces with restitution 0.89 (plain marble 0.44-0.47). Shock Absorber (id 4, 4.8 s): restitution
  0.00 (the landing is dead). (An earlier run that showed both as "shock" was the drill picking up the neighbouring
  item while driving; fixed by dropping onto the item.)
* Helicopter (Promontory): 4.9 s; a 3 u drop reaches -4.16 u/s against -7.68 plain: gravity x0.29 measured (engine
  constant x0.25, the rest is the marble's air drag-free fall sampled at 64 ms). The jump and air-control readings
  on the item spots were spoiled by slopes and ledges; the engine constants stand (gravity x0.25, air control x2).
* Tarmac spin launch (Promontory, friction_high floor at (0, -126)): marble spun about +y, jump, mega activated in
  the air two decisions later, landing on tarmac: peak speed plain / mega 30 rad/s 3.8 / 4.2, 60 rad/s 4.1 / 6.1,
  90 rad/s 5.3 / 8.4 u/s (higher spins below).
  Higher spins (place() sets the angular velocity): plain / mega peak speed 150 rad/s 8.1 / 18.9, 250 rad/s 11.4 /
  37.0, 400 rad/s 19.3 / 37.6 u/s. The mega's launch saturates near 37 u/s (= its 0.667 u radius against the
  friction impulse), the plain marble's grows slowly. The operator's trick is real and large; how much spin a marble
  can carry into the air in play is measured next (a 15 u/s roll is 79 rad/s).

### 38.4 Spin, and the phase 2 summary (10-02 19:00)

Spin (KOTM): rolling at full throttle the marble reaches 17.8 u/s and 98 rad/s after 2.5 s (speed/spin 0.183 = R:
pure rolling); in the air, holding the roll direction spins it up by ~44 rad/s per second (19 -> 58 in 0.9 s),
holding the opposite direction reverses the spin at the same rate. So a fast roll plus a jump with the direction
held carries ~140 rad/s into a landing: with the mega on tarmac that is a 15-19 u/s launch (plain marble ~8); the
37 u/s saturation needs 250 rad/s, which play does not reach. KOTM has no high-friction surface (special_materials
[]), so the launch is for other maps.

Phase 2 is complete for all seven powerups (constants file nav/learned_nav/powerup_physics.py):
| powerup | engine / measured | duration |
|---|---|---|
| Super Jump (1) | +20 u/s along -gravity at the fire tick | instant |
| Super Speed (2) | +25 u/s along the camera yaw, projected on the contact plane, added to the velocity (24.1 from rest on floor; works in the air) | instant |
| Super Bounce (3) | landing restitution 0.9 (measured 0.89; plain 0.44-0.47) | 5 s |
| Shock Absorber (4) | landing restitution 0.01 (measured 0.00) | 5 s |
| Helicopter (5) | gravity x0.25, air control x2 (drop fall speed x0.54 measured) | 5 s |
| Mega Marble (6) | radius 0.19 -> 0.667; floor top speed ~80 % of normal, braking ~17 vs 14 u/s^2, jump apex 1.79 vs 1.33 u; activation within 1 u of floor pops the marble up; activated in the air right after a jump: +25 u/s vertical (apex 22-24 u), 3-4 decisions after: +17-18; spin launch on tarmac saturating at 37 u/s | 10 s (5 s in competitive Hunt) |
| Blast, regular meter | fills 0 -> 1 in 25 s, usable from 0.20; impulse 10 sqrt(meter) upward; apex 2.5 x meter u; meter -> ~0.03 after | |
| Blast pickup | arms the meter at 1 and fires at 10 x 1.03 | until used |
| rules | respawn 7.0 s; the same type is refused while held; a different type replaces it; the use key fires two decisions after it is sent (the yaw must still hold then: bridge latch) | |

## 39. Powerups, phase 3: into the planner (2026-10-02 evening)

* planner.Programs: `use` (0 none, 1 the held powerup, 2 the blast meter) and `use_yaw` per decision; shifted() carries
  them. simulate(..., pow=...) takes the marble's powerup state (held, meter, special, the uses sent in the last two
  decisions with their yaws, effect seconds left). A use fires two decisions after it is sent (as the key does):
  Super Jump +20 u/s up, Super Speed +25 u/s along (sin yaw, cos yaw), blast 10 sqrt(meter) (or 10 x 1.03 armed),
  then the meter 0.03 and refilling; Helicopter / Super Bounce / Shock Absorber set per-row gravity and air-control
  multipliers and the landing restitution that air_exact now takes; an impulse step is flown analytically. Mega is
  not planned (the policy's).
* FAST_GROUND: the step model brakes a 25 u/s marble 11 u/s in one step (never saw it); the game loses 0.9 u/s a
  decision with no input. Above 18 u/s on the floor the rollout applies the measured 14 u/s^2 along the velocity
  instead: model 24.2 -> 16.4 over 12 decisions vs game 24.3 -> 16.6.
* Planner.use_oracle(pow): fixed programs that fire now (Super Speed at 5 yaws round the gem's direction, with and
  without a brake after; Super Jump / Helicopter / blast with 4 air inputs) against the same programs without the
  use; returns the ranked gain and the best use. hybrid.py: POW_ORACLE every 4 decisions while holding one of them
  or with the meter usable, sets the use bit (+ yaw) on the navigator's action when the gain is >= POW_GAIN 0.1 and
  p_succ >= 0.75 (pow_uses, pow_log in the round record); Session.step carries use_pow / pow_yaw / use_blast.
* Gate g8a (hybrid + powerup oracle, 28897 V7, 8 rounds): 163 154 162 155 167 165 164 162 = 161.5, best 167
  (g7e without the oracle: 158.9; run-to-run noise ~sd 5). The oracle was asked ~560 times a round (every 4
  decisions, the blast meter is almost always usable), 206-231 ms each, and fired ONCE in 8 rounds (a Super Speed
  at 11.9 u from the gem, p 0.86). So the score is the navigator's; the questions now only go out at a lip while
  aligned (a crossing) or with a Super Speed held on an aligned 8 u+ run over floor (POW_LIP_U, POW_RUN_*).
* Gate g8b (the oracle only at a lip or on a Super Speed run, 8 rounds): 161 159 165 165 155 164 157 154 = 160.0,
  best 165; asked 80-123 times a round (145-221 ms each), fired once. Three gates of 8 rounds now sit at 158.9,
  161.5 and 160.0 around the navigator's own 160.1: the oracle neither helps nor costs, and the planner finds no
  situation on KOTM where a Super Jump, Super Speed or blast fired by rule beats the navigator's own line. The
  other half of phase 3 (powerup items as route waypoints, planner proposals that use them) is not worth building
  on top of a rule that never fires; phase 4 instead lets the policy learn the use.

## 40. Powerups, phase 4: the navigator learns to use them (2026-10-02 21:50)

* Sixth action element `use` (nav/model.py: ACTION_DIM 6, use_head, Bernoulli, initial logit USE_BIAS -3 =
  4.7 %, clamped -7..3, counted in the discrete entropy). Registered last, so an ACTION_DIM-5 checkpoint loads
  with the head at its init and its Adam state intact (train_nav pads the optimizer's parameter list).
* The worker (nav/vec_worker.py) turns the bit into the bridge words: a held powerup is used (a Super Speed along
  the commanded direction, yaw atan2(dx, dy); a Mega via UsePowerup), with nothing held a usable blast meter
  fires. The GAME line gets `pow=` with use decisions (u<type>, blast) and fires (f<type>: held -> none without a
  fall). hybrid.py POLICY_USE and real_run.py pass the navigator's bit through the same way (policy_uses in the
  round record). Reward unchanged: the game's points only; picking an item up earns nothing by itself.
* Training restarted 21:52 from 28897 (the V7 best, 160.1 navigator alone), all 8 instances on KOTM
  (start_training.ps1 defaults now KOTM x8); night_best.json and stdout.txt of the tour run archived with the
  suffix tour_run_20261002. First round with the fresh head: points 107 (the exploration noise of a 4.7 % use
  rate: 17 blasts, 5 Super Speeds, 3 Super Jumps, 1 Mega in 3 minutes).
* Gate for this run: snapshots at new 50-round highs (game_summary.sh), then 8 rounds navigator alone
  (NAV_CKPT, NAV_PLANNER_OFF=1) against 160.1 / 170 best, and the hybrid.

### 40.1 Phase 4a result (21:52-23:30, random uses, no prior)

* 50-round means by 20 minutes: 136.1, 138.3, 139.5 (snapshots nav_night_28966_136, nav_night_29142_140), falls 2.7
  a round (1.5 before powerups). Uses fell from 7 fires + 10 blasts a round in the first 100 rounds to about 1 Super
  Jump, 1.3 Super Speed, 0.2 Mega and 0.6 blasts, then stayed there for 40 minutes: that is the exploration floor
  (use logit clamped at -7 = 0.09 % x 2,800 decisions = 2.5 random attempts a round, counted only while something
  is held), so the head had learned "do not" within 20 minutes and nothing else. Random uses cannot land in a
  situation where a use pays often enough to teach "do".

### 40.2 Use prior (23:30; operator: "yeah go ahead")

* nav/model.py: use logit floor -9, plus USE_PRIOR 10 where `use_prior(vec)` approves (map-independent, from the
  rays, the gap block and the powerup block): Super Jump held, or the blast meter usable with nothing held, at a lip
  within 2.5 u along the heading with a gap the plain jump cannot cross (vec[VEC_GAP+2] off) but the boosted
  flight covers (lip + gap < speed x hang x 0.8; hang 2.0 s for the Super Jump, sqrt(meter) s for a blast);
  Super Speed held on the floor, heading within 30 deg of the waypoint, waypoint >= 8 u away, the edge ray to it
  clear. Approved: logit -9 + 10 = +1 (73 % sampled, fires in deterministic play) until the head learns to cancel
  it (up to -6). Synthetic test: approves the lip / run cases, refuses a run with an edge ahead.
* Training relaunched 23:36 from nav_latest (update 29295); phase 4a stdout archived as stdout_phase4a_20261002.txt.
  First rounds: 105-120 points, 6-8 Super Speeds, 1-5 Super Jumps, 7-11 blasts a round. Dashboard: two powerup
  charts (fires per game stacked by type; points of rounds with vs without a fire, use decisions).
* Overnight handoff for the watching model: `OVERNIGHT_2026-10-02_POWERUPS.md`; gate scripts copied to
  logs/nav/gate_scripts/.

### 40.3 Overnight, first hour (10-02 23:36 - 10-03 00:30; the loop model)

* Health: trainer and 8 games up all hour, 14 s an update, no traceback, watchdog quiet; updates 29295 -> 29516.
* 100-round blocks of sampled training rounds: 106.0, 110.3, 111.9, 113.0, 113.7, 112.1, 113.3 (up 7 points in the first
  half hour, flat since; best round 141); phase 4a ended at 138-140. Falls 3.7 a round all hour (phase 4a 1.6-2.9). Per
  round, Super Speed fires 7.0 -> 4.7, Super Jump fires 3.0 -> 2.8, blast decisions 9.2 -> 6.8, Mega 0.15.
* The head cannot cancel the prior: model.py adds USE_PRIOR after the -9..3 clamp, so an approved use sits at logit
  +1..+13, at least 73 % a decision in training and always in deterministic play (gates). The handoff and 40.2 assume the
  head cancels it where it hurts. The use head does train (its output layer moved 12 % in the first 40 updates, more than
  the jump head), but in approved situations only between 73 % and 100 %. The drop in fires comes from fewer approved
  situations: at 73 % or more a decision an approved window still ends in a fire.
* What the forced Super Speeds cost (trace; a fire = horizontal speed up 12 u/s or more in one decision): 25 % end in a
  fall within 3 s against 2.5-3.3 % for floor decisions at > 5 u/s, unchanged over the hour (25.0, 23.8, 25.7 % by
  20-minute thirds); 45-47 % reach the gem within 3 s. Phase 4a's random Super Speeds: 17 %. So far the policy fires
  fewer of them and handles the ones it fires no better.
* Proposed for the operator, not applied: where use_prior approves, floor the head at USE_PRIOR_FLOOR - USE_PRIOR
  (-4 - 10 = -14) instead of -9. Unchanged where the head is above -9; deterministic play fires only where the head keeps
  the logit above 0; sampled play keeps at least 1.8 % a decision in approved situations, so they stay explored. The loop
  model made the edit at 00:00 and reverted it at once: the auto-mode safety check refused to test it as a change to the
  shared training setup. model.py is as it was at 23:36.

### 40.4 Gates beside the trainer, and what the forced uses cost in deterministic play (10-03 00:32-01:09)

* Trap: the handoff's hybrid gate (gate_navonly.ps1, 4 games) does not fit beside the trainer. The RTX 3070 has 8 GB and
  the trainer with its 8 games uses 5.3; each hybrid process also loads the planner's CUDA ensemble. Update time went
  14 -> 22-80 s (the PPO step alone up to 63 s), no gate round finished in 14 minutes, and at 00:46:50 the watchdog took
  the slowdown for the throughput trap and killed every game (the trainer's loop relaunched its 8; back to 14 s at once).
  Also: hybrid.py POW_ORACLE is on even with NAV_PLANNER_OFF=1, so its navigator-alone gates still ask the planner.
* What works: nav/real_run.py on the CPU (CUDA_VISIBLE_DEVICES=-1; an empty value is dropped on Windows), NAV_TOUR=walk,
  one game per arm: 4 rounds in about 2 minutes, the trainer at 18 s an update meanwhile, the games closed after. Launcher
  in the loop model's scratchpad (rr_gate.sh). The harness differs from the hybrid's navigator-alone mode the 160.1
  baseline came from (same policy, chooser and stuck-breaker; not cross-checked tonight, since no gate can switch the
  prior off without a code change).
* 8 deterministic rounds each, the prior on (the current code):

  | checkpoint | rounds | mean | best | falls/round | use decisions/round (blast, SJ, SS) |
  |---|---|---|---|---|---|
  | 28897 V7, fresh use head (= the prior's rule on the best navigator) | 140 131 123 119 139 113 131 129 | 128.1 | 140 | 8.0 | 15.5, 5.7, 19.7 |
  | 29600 (this run, 1.5 h) | 140 140 139 137 131 125 134 134 | 135.0 | 140 | 4.6 | 8.7, 5.2, 9.6 |
  | reference: 28897 without uses (hybrid, 10-02, 37.6) | | 160.1 | 170 | 1.0 | |

  The prior's rule costs the best navigator about 32 points and 7 falls a round in deterministic play; 1.5 hours of
  training won back 7 of them. A gate always fires where the prior approves, so a checkpoint of this run reaches 160
  only if the policy learns to make those forced uses pay or to stay out of the approved situations.

### 40.5 Two hours in: where the falls come from, gate on 29875 (10-03 01:34-02:10)

* 200-round blocks of training rounds: 119.0 (01:16), 120.1, 121.1, 119.9 (02:04): the climb of the first two hours
  (107 -> 120) has slowed to about a point an hour; best training round 148. Falls flat at 3.6-3.7 a round. Super Speed
  fires 2.7 a round (7.0 at the start), Super Jump fires 2.4-2.8, blast decisions 7-8. By the handoff's measure (below
  130 after 2 hours) this is the bad case; training continues.
* Falls by the last event in the 60 decisions before them (training trace, last 40 minutes; the trace marks fewer falls
  than the GAME lines, so read shares): a blast-like vertical kick 44 %, no event (roll-off) 19 %, a Super Speed 15 %, a
  Super Jump 14 %, a plain jump 8 %. Kicks at more than 12 u/s within 30 decisions after a Super Speed: 2.5 a round, 26 %
  of them followed by a fall; all other kicks 2-7 %. Phase 4a, as a control with almost no blasts: 0.34 such kicks per
  Super Speed fire with 22 % falls; tonight 0.60 per fire. So the Super Speed itself is the danger: the marble passes the
  gem at about 30 u/s and runs on over lips and ramps (at that speed a ramp kicks it up like a blast), and the blast
  prior, which approves when lip + gap < speed x sqrt(meter) x 0.8, approves more blasts at that speed. The Super Speed
  approval checks the run TO the waypoint (aligned, 8 u+, edge ray clear) and not the 30 u or more the marble covers
  after the pickup.
* Gate on nav_029875 (CPU real_run, walk tour, 8 rounds): 130 150 128 154 127 146 142 129 = 138.3, best 154, falls 4.75
  a round, use decisions a round blast 9.6, Super Jump 5.4, Super Speed 5.8 (29600: 135.0, best 140). Stuck-breaks rose
  to 3.0 a round (28897 with the prior 1.9, 29600 2.25); they cluster within 8 u of the map centre (-25, 15), where the
  Mega item sits; the gates show no Mega uses, so the centre block itself is the likely place.

### 40.6 The 3.5-hour gate (10-03 03:05)

* Training rounds flat for an hour: 200-round blocks 122.6, 121.1, 121.8 (best 149), falls 3.4-3.5 a round.
* Gate on nav_030125 (CPU real_run, walk tour, 8 rounds, two games of 4): 145 150 142 143 / 127 130 131 136 = 138.0,
  best 150, falls 4.6 a round. The game with fewer Super Speed decisions (1.8 a round against 5.5) scored 145 against 131.
  Gates by checkpoint: 29600 135.0, 29875 138.3, 30125 138.0, so the deterministic score has also stopped rising, about
  22 points under the 160.1 of 28897 without uses. No round has beaten 154 tonight.
* 04:08: training blocks 124.0, 124.3, 125.0 (best 151), falls down to 2.8 a round, Super Speed fires 2.1 a round. Gate on
  nav_030375: 136 138 133 129 / 130 138 125 128 = 132.1, best 138, falls 4.75; use decisions a round in the gate still
  blast 8.7, Super Jump 7.0, Super Speed 6.3. The training rounds improve (fewer Super Speed fires, fewer falls); the
  deterministic gates (four checkpoints: 135.0, 138.3, 138.0, 132.1) do not move.
* 05:11: training blocks 122.0, 124.1, 121.5. Gate on nav_030650: 137 140 121 134 / 123 134 140 145 = 134.3, best 145,
  falls 4.1; use decisions a round blast 7.1, Super Jump 5.9, Super Speed 4.5.

### 40.7 Morning report (10-03 06:20; the loop model)

The run (phase 4 with USE_PRIOR, from update 29295 at 23:36) is still training: update 30916 at 06:18, 5,218 KOTM
training rounds, no crash and no restart. The watchdog fired once, at 00:46, on the slowdown caused by the first gate
(40.4). The operator did not answer the 01:09 question about the use floor fix, so nothing in the run was changed.

Training rounds by clock hour (sampled play; phase 4a ended at 138-140 with 1.6-2.9 falls, the tour run near 150):

| hour | rounds | mean | best | falls | Super Jump fires | Super Speed fires | Mega fires | blast decisions |
|---|---|---|---|---|---|---|---|---|
| 23 (from 23:36) | 302 | 109.4 | 138 | 3.71 | 2.9 | 6.2 | 0.15 | 8.3 |
| 00 | 685 | 114.0 | 142 | 3.74 | 2.7 | 4.8 | 0.13 | 7.1 |
| 01 | 836 | 119.3 | 148 | 3.68 | 2.7 | 3.3 | 0.15 | 7.2 |
| 02 | 820 | 121.7 | 149 | 3.57 | 2.5 | 2.7 | 0.15 | 7.3 |
| 03 | 807 | 123.3 | 151 | 3.17 | 2.7 | 2.6 | 0.15 | 6.0 |
| 04 | 782 | 123.7 | 149 | 3.03 | 2.7 | 2.0 | 0.16 | 5.4 |
| 05 | 762 | 124.5 | 149 | 2.90 | 2.4 | 1.7 | 0.14 | 5.3 |
| 06 (to 06:18) | 224 | 125.6 | 145 | 2.61 | 2.3 | 1.7 | 0.21 | 5.0 |

Gates, 8 deterministic rounds each (nav/real_run.py on the CPU, walk tour, the prior on; harness in 40.4):

| checkpoint | time | rounds | mean | best | falls/round |
|---|---|---|---|---|---|
| 28897 V7, fresh use head (the prior's rule alone) | 01:05 | 140 131 123 119 139 113 131 129 | 128.1 | 140 | 8.0 |
| 29600 | 01:05 | 140 140 139 137 131 125 134 134 | 135.0 | 140 | 4.6 |
| 29875 | 02:06 | 130 150 128 154 127 146 142 129 | 138.3 | 154 | 4.75 |
| 30125 | 03:07 | 145 150 142 143 127 130 131 136 | 138.0 | 150 | 4.6 |
| 30375 | 04:10 | 136 138 133 129 130 138 125 128 | 132.1 | 138 | 4.75 |
| 30650 | 05:13 | 137 140 121 134 123 134 140 145 | 134.3 | 145 | 4.1 |
| 30900 | 06:16 | 145 123 144 127 146 132 137 133 | 135.9 | 146 | 3.6 |
| 31275 (training rounds then 127-130) | 07:52 | 140 142 136 143 123 137 137 147 | 138.1 | 147 | 4.6 |
| 31725 (training rounds then 129-132) | 09:56 | 128 127 150 148 121 144 150 128 | 137.0 | 150 | 4.6 |
| 32150 (training rounds then 131-134) | 12:00 | 135 134 138 144 137 126 138 153 | 138.1 | 153 | 5.0 |
| reference: 28897 without uses (hybrid, navigator alone, 10-02) | | | 160.1 | 170 | 1.0 |

* Best snapshot: none. No 50-round mean passed phase 4a's 140.6 (night_best.json unchanged), so game_summary.sh took no
  snapshot; the best gated checkpoints are 29875 and 30125 (138). No round beat 170 (best: 154 in a gate, 151 in
  training) and no 8-round mean beat 160.1; nothing was copied to nav_v7pow_best.
* Verdict: neither. The policy did not learn to make the uses pay, and it could not learn to cancel the prior, because
  model.py adds USE_PRIOR after the clamp and an approved use always fires in deterministic play (40.3). Super Jump fires
  (2.3-2.9 a round) and blast decisions (5-8) stayed above the old floor; Super Speed fires fell from 6.2 to 1.7 a round
  as the sampled policy met fewer approved situations. Training points rose from 109 to 126 but stayed well under phase
  4a's 138-140, and the gates stayed at 132-138 all night, 22-28 points under the same navigator without uses. The cost
  is mostly the Super Speed (the marble passes the gem at about 30 u/s and runs on over lips and ramps) and the blasts the
  prior approves at that speed (40.5).
* Suggestions for the operator's decision:
  1. Let the head cancel the prior (40.3: where use_prior approves, floor the head at USE_PRIOR_FLOOR - USE_PRIOR, for
     example -4 - 10 = -14, instead of -9) and train on, judging by gates whether the uses that survive pay. Start from
     this run's endpoint (it already avoids some Super Speed situations) or from phase 4a's 29257.
  2. Tighten the approvals with the physics of 38.2: a Super Speed only when the run past the waypoint is clear for the
     boosted overshoot (about 30 u from 30 u/s at the ~14 u/s^2 the game brakes with) or the next waypoint lies on the
     same line; a blast with the measured lift (10 sqrt(meter) - 1.28 u/s), the two-decision lag, and not above about
     12 u/s.
  3. Gate beside the trainer with the CPU real_run (logs/nav/gate_scripts/rr_gate.sh, results with rr_read.py); the
     hybrid gate overflows the GPU and sets off the watchdog.
  4. On the 180 goal: tonight gives no sign that powerups add points on KOTM with this policy. The human's 172-177 rounds
     used none, and the measured gap to them is still route length (handoff section 3).
* Morning addendum (10:00): training rounds reached 129-132 a round (falls 2.2-2.4, best 153) while the gates stayed put
  (31275 138.1, 31725 137.0). The Mega, which the prior does not cover, is the one use the head explores on its own:
  use decisions 0.03 a round all night, 0.11 at 08:00, 0.49 at 09:50; rounds with a Mega use scored about 3 points under
  those without (08:00 hour), and the gates make no Mega use.

### 40.8 The use floor fix (10-03 13:10; operator: "Apply use floor fix")

* nav/model.py: USE_PRIOR_FLOOR -4.0. Where use_prior approves, the head's floor is USE_PRIOR_FLOOR - USE_PRIOR (-14)
  instead of -9, so an approved use can go down to logit -4 (1.8 % a decision, no fire in deterministic play); unchanged
  where the head is at or above -9 and wherever the prior is off. CPU test on nav_latest (32430): not approved identical
  to the old formula; approved identical for head outputs >= -9, below it -9.5 -> +0.5, -12 -> -2, -14 and less -> -4;
  gradient 1 through the new range; act() and evaluate_seq() run with finite outputs and a use-head gradient.
* The new rule on the policy as it was, before any training with it (nav_032425, 8 rounds, CPU real_run gate):
  156 127 143 145 / 152 152 140 141 = 144.5, best 156, falls 2.5 a round; use decisions a round blast 2.1, Super Jump
  2.1, Super Speed 0.1, Mega 0.5 (the last old-rule gate, 32150: 138.1, falls 5.0, Super Speed 11.6). The head had
  already pushed most Super Speed situations below the old floor, where they fired anyway; with the floor lowered they do
  not fire, falls halve and the mean rises about 6 points with no retraining.
* Restart (operator: "You can restart"), 13:19: game loop, trainer (pid 2888) and games stopped; the old run's endpoint
  kept as models/nav/nav_p4b_end_32445.pth; stdout / stderr archived as *_phase4b_20261003.txt and night_best.json as
  night_best_phase4b_20261003.json (reset, so game_summary.sh snapshots this run's highs); start_training.ps1 relaunched
  13:19:35 from nav_latest (update 32445) with the new rule.

### 40.9 Training with the use floor (10-03 from 13:19)

* First 96 rounds: 139.1, 140.3, 138.4 per 30 rounds (old rule at its end: 130.5), falls 1.3-2.4 a round. The head cancels
  most uses at once: per round Super Speed fires 0.6-0.8 (1.6 at the old rule's end), Super Jump fires 1.2-1.7 (2.2), blast
  decisions 1.0-1.7 (6.2), Mega 0.1-0.3.
* First new 50-round high 139.0 at 32473 (snapshot nav_night_32473_139.pth). Gate (8 rounds, CPU real_run): 149 143 149
  152 / 145 144 152 149 = 147.9, best 152, falls 3.4; use decisions in the gate: almost none (Super Jump 0.25 a round).
  The deterministic policy now leaves the powerups alone and gains 10 points on the old rule's 132-138; it is still 12
  under 28897's 160.1 without uses, so the hours of forced uses changed more than the use decisions.
* Stopped 13:36 at the operator's request ("can you completely stop training for a bit"): game loop, trainer and games
  stopped; endpoint models/nav/nav_p4c_stop_32510.pth (= nav_latest, update 32510). This run: 208 rounds at 138.9 (best
  155), falls 1.8 a round. To resume: start_training.ps1 (it resumes from nav_latest with the use floor). The helper
  scripts (game_watchdog.sh, game_summary.sh, the dashboard) were left running; idle without a trainer.

### 40.10 Review of the use floor fix, training resumed (10-03 14:13)

* The fix (40.8) is correct: where the prior approves the use logit is max(head, -14) + 10, so the head can take an
  approved use down to -4 (1.8 % sampled, never in deterministic play) and the unapproved path is unchanged (-9 floor).
  It is useful as a damage stop: the same checkpoint went 138.1 -> 144.5 with falls 5.0 -> 2.5 and no retraining, and
  the first gate after training with it read 147.9. It does not make the powerups pay: the head now cancels nearly
  every approved use (gate use decisions ~0), and the 12-point gap to 28897's 160.1 is the navigator itself, worn by
  14 hours of forced uses under the old rule. Making a use pay needs the tighter approvals of 40.7 (Super Speed only
  when the overshoot past the waypoint is clear or the next waypoint is on the line; blasts not above ~12 u/s), which
  the operator has not decided on.
* Resumed 14:13 from nav_latest (32510, the operator's stop point) with the rule as it stands; one of the two
  watchdogs stopped (two were running); stdout archived as stdout_phase4c_20261003.txt. First updates: speed 7.7,
  sgem 1.59, falls100 0.45.

### 40.11 Fresh start from 28897 with a survivable Super Speed approval (10-03 14:28; operator)

* Operator: pick up from 28897 and tighten the approval ("a Super Speed only when the run past the waypoint is clear
  for the overshoot or the next waypoint lies on the same line"); the model must learn to use Super Speeds on this
  map to get well above 160.
* Observation V8 (nav/obs.py, POW_DIM 27, VEC_DIM 88): vec[VEC_POW + 26] = Super Speed run clear, computed only while
  a Super Speed is held on the floor. The boosted speed is the marble's speed along the line + 25 u/s, at most 32;
  the floor's own braking (14 u/s^2, planner FAST_DECEL) stops it after v^2 / 28 u (8 u/s -> 32 -> 36.6 u), so the
  floor along the line to the waypoint (terrain gap_along, followed 48 u) must continue past max(stop, waypoint
  distance) + 2 u; or the next waypoint lies within 30 deg of the line, beyond this one, with floor all the way to it
  + 2 u. On KOTM: a 12 u ring run at 8 u/s has 37 u of floor and is refused alone (36.6 + 2), approved with the next
  gem on the line; centre runs (4.5 u) and west -> centre (14.5 u) refused.
* nav/model.py: the Super Speed branch of use_prior also needs that flag; USE_BIAS -11 for the fresh head, so an
  approved use starts at logit -1 (27 % sampled, never in deterministic play until the head learns it pays; was +7 =
  always). Super Jump / blast approvals unchanged (also at -1 now).
* Checkpoint: nav_v7_best_28897 migrated to V8 (migrate_obs_width, zero columns) = models/nav/nav_v8_28897.pth =
  nav_latest. The previous run's endpoint kept as nav_p4c_end_32560.pth (V7; V7 checkpoints now need migration or
  the old code), stdout as stdout_phase4c2_20261003.txt, night_best as night_best_phase4c_20261003.json. Trainer
  relaunched 14:28 (the first launch died on the V7 nav_latest: a locked file blocked the copy; redone).
* First rounds (sampled policy): 130-137 points, falls 1-4, per round Super Jump fires 1-3, Super Speed fires 1-2,
  blast 0-2: the exploration rate of the -1 logit. Speed 7.7, sgem 1.55.

### 40.12 The Super Speed redirect at the pickup (10-03 14:55; operator)

* Operator: a human's main Super Speed use on KOTM is to roll full speed into a corner gem and fire at an angle at the
  pickup so the velocity turns onto the line to the next gem, saving the brake-and-turn; and the model must KNOW
  during the approach that the redirect will work, or it keeps braking into corners. As coded (40.11) that move was
  impossible: the prior refused it (heading-to-waypoint test), the kick went along the policy's commanded direction
  (at the gem), and no check followed the new line.
* Observation V9 (nav/obs.py, POW_DIM 30, VEC_DIM 91): while a Super Speed is held and the next waypoint is known,
  at ANY distance: [27] redirect possible, [28-29] the kick aim. Predicted pickup velocity = the current speed along
  the line to the waypoint (at least 3 u/s); ss_redirect_aim solves the unit kick a with v + 25 a parallel to the
  line from the gem to the next gem (lam = n.v + sqrt((n.v)^2 + 625 - |v|^2)); survivable when the floor from the
  gem along that line (gap_along, 48 u) reaches the next gem + 2 u. The flag is right at the pickup and visible 10 u
  out, where the brake-or-not decision is made. Example: south along the ring at 8 u/s toward (-36, 2), next gem
  east: aim (0.95, 0.32), result along +x at 23 u/s, flag on; the same with the next gem off the map: off.
* Aim: obs.ss_aim(vec) = the redirect aim when the flag is on and the waypoint is within SS_REDIRECT_U 3 u (the
  2-decision key lag), else the line to the waypoint; the worker, hybrid.py and real_run.py all fire a Super Speed
  along it (was the policy's commanded direction, wrong for both branches). The policy decides whether and when; the
  physics aims, as with the jump prior.
* use_prior: a redirect branch (Super Speed held, on the floor, within 3 u of the waypoint, flag on) beside the run
  branch of 40.11. USE_BIAS -11 as before (approved -1).
* The V8 run (14:28-14:50, 28895 -> 28975, training rounds 130-140) migrated to V9 (zero columns) =
  nav_v9_from_run.pth = nav_latest; its endpoint kept as nav_p4d_end_v8.pth, stdout as stdout_phase4d_20261003.txt.
  Relaunched 14:55; first rounds 123-141, Super Speed fires 0-4 a round.
* What to watch (operator's test): Super Speed fires (f2) per round and whether rounds with them score more; the
  approach speed into corner gems with the flag on vs off (needs a trace analysis: the training trace has speed and
  goal, the flag can be recomputed from the terrain); then an 8-round gate against 160.1 / 170.

### 40.13 Redirect verified in the game; first 75 minutes (10-03 15:30-16:50)

* Worker diagnostics added 15:30 (restart, nothing lost): GAME line `pow=` now has `r2` (Super Speed uses aimed as a
  redirect, within 3 u of the gem with the flag on) beside `u2` (runs), `fss` (falls within 3 s of a Super Speed
  fire), and `pkon/pkoff/pknone` = mean pickup speed (u/s) / count for pickups with a Super Speed held and the
  redirect flag on, held with it off, and without one.
* powdrill --test redirect (one extra game, port 9951): on the west ring rolling south at 7.4 u/s with a Super Speed,
  fired with the aim ss_redirect_aim solves for a next gem due east (0.95, 0.32): the next decision reads 22.8 u/s
  at 3 deg (east), predicted 23.7 at 0; the mirror case 22.8 u/s at 177 deg; the straight run 30.6 u/s. The yaw
  word, the bridge latch and the physics agree; the mechanism is right.
* Training 15:30-16:45 (780 rounds, 29080 -> 29328) by quarters: points 138.9 / 139.9 / 139.2 / 138.7, falls
  2.0-2.3; per round redirect uses 1.6 / 1.2 / 1.2 / 1.5, run uses 0.3 / 0.2 / 0.2 / 0.2, Super Speed fires 1.9 /
  1.5 / 1.3 / 1.7, falls within 3 s of a fire 0.19 / 0.15 / 0.11 / 0.19 (about 10 % of fires); pickup speed with the
  flag on 9.3 u/s, off 8.1, no Super Speed 8.55, flat across the quarters. Rounds with a Super Speed fire score
  138.6 against 141-145 for the few without. So after 75 minutes the policy fires the redirect on most approaches
  the prior approves, one in ten ends in a fall, and it has not yet learned to arrive faster when the flag is on
  (the operator's test). 50-round high 140.9 at 29317 (snapshot nav_night_29317_141.pth).

### 40.14 Two hours of redirects: flat; the window was too early (10-03 17:45)

* 15:30-17:40 (1,360 rounds, 29080 -> 29513), by 200-round blocks: points 138.4 / 139.2 / 140.0 / 138.7, falls
  2.2-2.5, redirect uses 1.2-1.6 a round, fires 1.5-1.8, falls within 3 s of a fire 0.15-0.18 (a tenth of the
  fires), pickup speed with the flag on 9.3 -> 9.1 u/s (off 8.1 -> 7.9), rounds with a fire 138-139 against
  141-145 without. The policy fires the redirect on most approved approaches and nothing improves.
* Likely cause: the approval window began 3 u out, the use fires two decisions (0.128 s) later, so at 9 u/s a use
  sent at 2.5-3 u kicked the marble 1.3-1.8 u BEFORE the gem, swinging it off the line and past the pickup; the
  sampled policy (27 % per approved decision) fired on the first approved decisions most of the time. Second
  cost: the redirect arrives at the next gem at ~23 u/s with only 2 u of floor required beyond it.
* Changes (restart 17:50, nothing lost): the window is now dist <= 0.128 x speed + 0.8 u (obs.ss_redirect_window,
  the same in use_prior and the worker), so the kick lands within 0.8 u of the gem or past it; the flag needs
  SS_NEXT_MARGIN 6 u of floor past the next gem; the worker counts `rhit` / `rmiss` (a gem picked within 12
  decisions of a redirect fire or not), beside r2 / fss / pkon.

### 40.15 The redirect moves to after the pickup; one Super Speed rule (10-03 18:45)

* 17:45-18:35 with the tight pre-pickup window (556 rounds): points 141.7 / 141.7 / 142.9 / 141.8 (up from 139),
  falls after a fire 0.10-0.16 a round, pickup speed with the flag on 9.15 -> 9.41 u/s; but redirect fires still
  missed the gem 40 % of the time (rhit 0.26-0.32, rmiss 0.19 a round): a kick that starts before the pickup
  swings the marble off the gem however tight the window.
* So the kick now comes right AFTER the pickup, when the waypoint is already the next gem, and the aim is solved
  for the CURRENT waypoint: a = (lam n - v) / 25 with v + 25 a on the line to the waypoint (obs.ss_redirect_aim),
  stored in vec[VEC_POW + 28-29] whenever a Super Speed is held; vec[VEC_POW + 26] = that kick is survivable (floor
  along the waypoint line for max(stop, d) + 2 u with stop = lam^2 / 28, or the next gem on the line with floor to
  it + 6 u). use_prior's Super Speed branch: held, on the floor, waypoint >= 8 u, [26] on; no heading test (the aim
  handles the angle), no edge-ray test (the floor check is longer). The pre-pickup branch is gone; its flag [27]
  stays as approach information ("a redirect at the gem ahead toward the next gem would be survivable").
* Worker: 'r2' = a Super Speed use with the velocity more than 30 deg off the waypoint line (a turn), 'u2' a run;
  rhit / rmiss = the waypoint taken within 1.5 s of a turn's fire. ss_aim(vec) returns [28-29] whenever a Super
  Speed is held (the worker, hybrid.py and real_run.py fire along it).

### 40.16 One hour of the post-pickup rule (10-03 18:40-19:40, 29700 -> 29896)

* 616 rounds by quarters: points 141.5 / 141.1 / 141.9 / 142.4, falls 2.8 -> 2.45; Super Speed fires 0.9-1.15 a
  round, of them turns 0.41-0.56 (runs 0.14-0.19); turns that took the waypoint within 1.5 s 0.26-0.38 against
  0.12-0.14 that did not (about 72 % hits, was 60 % before the pickup); falls within 3 s of a fire 0.10-0.16; pickup
  speed with the approach flag on 9.3-9.7 u/s (off 8.0-8.2). Rounds with a fire now score the same as rounds
  without (142.1 / 141.7 and 142.3 / 142.6 in the last two quarters; they were 3-5 under all afternoon).
* So the uses no longer cost points, and they do not yet earn any: the sampled mean is flat at 142 and the fire
  rate is the prior's 27 % sampling, not a learned preference. Next: an 8-round deterministic gate of
  nav_p4e_29896.pth (CPU real_run; in deterministic play a use fires only where the head has learned it) against
  160.1 / 170.

### 40.17 Gate of 29896, and the arithmetic (10-03 19:45)

* nav_p4e_29896.pth, 8 deterministic rounds (CPU real_run, walk tour, two games): 152 159 151 148 / 162 158 156 161
  = 155.9, best 162, falls 1.6 a round, speed 8.4; use decisions: one Super Jump in 8 rounds, no Super Speed. So
  nothing has been learned to the point of firing in deterministic play (the head sits near -11 where approved,
  the prior lifts it to -1), and the navigator itself reads about 4 under 28897's 160.1 (8 rounds, sd 5: inside
  two standard errors).
* The arithmetic of a Super Speed on KOTM: a turn out of a pickup at 23 u/s toward a gem 16 u away saves ~0.8 s
  against rolling and turning at 8-9 u/s; at 1.46 s a gem that is half a point per use. The item respawns 7 s after
  a pickup and the marble holds one during ~40 % of its pickups, so the ceiling at a few free pickups a round is
  a few points, unless the fast approach the flag allows spreads to many gems. The gain the operator is after
  (20 points) cannot come from one fire a round; it needs the head to learn to fire at most approved moments and
  the approach speed to follow.
* Change (restart 20:00): USE_PRIOR 10 -> 11, so approved uses are sampled 50 % instead of 27 % (more data at no
  cost now that the uses are revenue-neutral). Nothing else.

### 40.18 Braking measured; the approval rate is the bottleneck (10-03 20:50)

* 19:44-20:44 at 50 % sampling (628 rounds, 29905 -> 30104): points 139.1 / 142.3 / 141.9 / 144.1 (50-round high
  144.8 at 30097, snapshot nav_night_30097_145.pth), falls 3.1 -> 2.2; Super Speed fires 1.1-1.2 a round, the SAME as
  at 27 %: the sampling rate is not what limits the fires, the number of approved moments is (2-3 a round).
* powdrill --test brake (one extra game): from 25 u/s on the ring floor, no input loses 7.7 u/s^2 (after a quick
  25 -> 20 in the first two decisions), the full brake 14.2 u/s^2 (stops in 1.7 s, 22 u), holding forward settles at
  ~19 u/s. So a Super Speed's resultant (17-33 u/s, never under 25 - |v|) needs ~20 u of floor or the next gem on
  its line, which after a pickup on KOTM is rare. The rule is the physics; the ceiling for the Super Speed on this
  map is a few points unless the kick is aimed at gems that lie far down a clear line (the route chooser could
  prefer such orders while a Super Speed is held: not built).
* Worker: `ap2` = approved Super Speed decisions a round (restart 20:52).

### 40.19 The operator's Super Speed demo (10-03 21:17-21:21; demos/demo_20261003_211632.npz)

* One round, 153 points, 123 pickups, 2 falls, recorded at 1x with the recorder now keeping the full 61-number
  observation (held powerup, meter, items) beside the inputs and the camera yaw (record_demos.py obs_full). The
  operator: not warmed up, not best play, but the idea of the Super Speed.
* 20 Super Speed fires in the round (the item at (-31, 3) taken at nearly every respawn; a Super Speed held 29 s of
  180, a Super Jump 35 s, a Mega 6 s). Timing: 12 of 19 fires come 5-11 ticks (0.1-0.2 s) after a pickup, the rest
  mid-leg (16-74 ticks). Every fire is a TURN: 77-151 deg between the velocity before and the line to the next
  gem (median ~110). Speed before 5.5-14.5 u/s (median 11), after the kick 11-23 u/s (median 16.7): a kick
  against most of the old velocity leaves a moderate speed, nothing like the 25-33 u/s of a straight run. The next
  gem lay 11-24 u away (median 12.2) and was reached 50-140 ticks later at 5-14 u/s; the backward key was held in
  the half second after the fire on 11 of 20 fires (braking into the gem). No fall followed any fire.
* The aim: the operator's camera yaw at the fire against obs.ss_redirect_aim (v + 25 a parallel to the line to the
  next gem): mean difference 2.6 deg, sd 8.5 deg over 19 fires; the resultant speed in the game vs the solver's lam:
  mean -1.7 u/s, sd 1.1 (the game loses a little to the floor). So the physics aim the worker fires along IS the
  human technique, and the post-pickup window (40.15) is the human timing.
* What the policy did differently: it had the same approvals (27-51 decisions a round) and declined them (use
  logit at the -4 floor) before it had learned the follow-up (brake and steer into the gem at 16-19 u/s); its turns
  still missed the gem 28 % of the time, and a miss at that speed costs more progress than a hit saves, so the
  per-use advantage read negative and the head gave up. The human never misses and brakes after the kick.
* Change (restart 21:45, from nav_latest 30200): USE_PRIOR_FLOOR -4 -> -1, so approved uses stay sampled at >= 27 %
  (the head cannot switch them off while it learns the control after the kick); deterministic play still fires
  nothing until the head learns a use is worth it. Approvals, aim and window unchanged (they match the demo).

### 40.20 The raised floor: more fires, more falls (10-03 21:24-22:23, 30195 -> 30394)

* 632 rounds by quarters: points 131.8 / 133.4 / 133.2 / 132.4 (was 142 at the -4 floor), falls 3.2-3.6 a round
  (2.5); Super Speed fires 3.5-4.3 a round (1.1), of them turns 3.9-5.1 with hits 2.1-2.8 and misses 1.0-1.3 (68 %
  hits); falls within 3 s of a fire 0.7-1.0 a round = about one fire in five; reward per decision in the 30 after a
  fire 1.18-1.34 against 1.55 overall. Approvals 16-20 decisions a round.
* So the head's refusal was right at this level of control: a fifth of the kicks end in a fall and a third of the
  turns miss, where the operator's 20 fires a round had no fall and no miss. The demo's kicks were all big turns
  (77-151 deg) with a resultant of 11-23 u/s (median 17); the prior approves any angle, so the policy also fires
  gentle turns and runs where the resultant is 25-33 u/s along a long clear line, and the floor check follows only
  the exact line (no lateral room). Worker diagnostics added (restart 22:27): the resultant speed 3 decisions after
  each fire bucketed A < 18, B 18-24, C > 24 u/s (`ksA/B/C`) and the falls within 3 s per bucket (`kfA/B/C`), to see
  whether the falls come from the fast kicks.

### 40.21 Watching at 1x; the reversal brake the solver could not produce (10-03 22:40)

* Operator watching the navigator at 1x (real_run with NAV_SAMPLE=1, new: the policy sampled as in training so the
  prior's uses fire; deterministic play has learned none): a Super Speed run through a line of two gems picks up a
  second Super Speed on the way, the momentum carries the marble far past the last gem, and the model never fires
  the second one backward to turn round, which is the right play.
* Cause: ss_redirect_aim solves v + 25 a parallel to the line to the waypoint; with the marble moving AWAY faster
  than 25 u/s there is no solution (disc < 0), it returned None, the flag stayed off, the use logit sat at -9, and
  the reversal was never sampled, so it could never be learned. Fix: in that case the aim is the pure brake
  (straight against the velocity, taking 25 u/s off on the spot; the marble keeps rolling away at |v| - 25) and
  the approval checks the floor along the OLD heading for the remaining stop. Tests: 28 u/s away -> aim backward,
  resultant 3 u/s away; 20 u/s away -> 5 u/s toward the gem; 30 u/s at 90 deg -> 5 u/s along the old heading.
  Physics only, no map data. Viewer restarted with it.

### 40.22 Only the kicks that can be braked (10-03 23:00; operator)

* Operator: the model uses the Super Speed as a short burst that cheeses the progress reward and then overshoots;
  the strategy is to go in fast KNOWING the kick can turn the velocity onto the next gem, and one model must make
  that choice. The channels exist (held type, the approach flag [27], the physics aim); what is missing is that
  the kicks the policy was allowed to take failed (a fifth fell, a third missed), so the fast approach could not
  earn credit.
* obs.py SS_LAM_MAX 24: the run/turn approval also requires the solver's resolved speed <= 24 u/s (the demo's
  kicks: 12-25 by the solver, ~1.7 less in the game, median 17; all big turns). Refused now: straight runs and gentle
  turns (8 u/s aligned resolves to 32, from rest 28). Kept: 90 deg turns at 8-12 u/s (24, 22), the demo's
  100-150 deg turns (13-19), the reversal brake (40.21). Floor checks unchanged.
* Training restarted ~23:00 from nav_latest (30420) with the floor at -1 (approved kicks sampled >= 27 %). Watch:
  turn hit rate (demo 100 %), falls within 3 s of a fire (demo 0), kick buckets ksA/B (no ksC expected), pickup
  speed with the approach flag on vs off.

### 40.23 Overnight loop picked up; the first ten minutes of the cap (10-03 23:10; the loop model)

* Trainer from 22:56 (update 30420, log train_nav_20261003_225620.log), 8 games, 15-16 s an update, no traceback.
* First 113 rounds (22:57-23:06) in two halves: points 139.2 / 136.6 (40.20 before the cap: 132-133), falls 2.2 / 2.5
  (3.2-3.6); approved Super Speed decisions 8.8 / 9.4 a round (16-20), fires 2.2 / 2.1 (3.5-4.3), turns 1.9 / 2.0, runs
  0.56 / 0.61; turns that took the waypoint 63 / 60 % (68 %); falls within 3 s of a fire 0.26 / 0.20 (0.7-1.0); kicks
  by the speed 3 decisions after the fire ksA 0.67 / 0.39, ksB 0.79 / 0.80, ksC 0.75 / 0.93 with falls kfA 0.02 / 0,
  kfB 0.18 / 0.11, kfC 0.07 / 0.09; pickup speed with the approach flag on 8.8, held with it off 7.3-7.4, none 8.1 u/s;
  reward per decision in the 30 after a fire 1.21 / 1.30 against 1.64 / 1.60 overall.
* ksC is not rare (about 40 % of the fires), and the code says why: in nav/obs.py the cap sits only in the first
  approval test (`ok = lip > need and v_after <= SS_LAM_MAX`); the override that follows (`if not ok` and the next
  waypoint lies within 30 deg of the line beyond this one with floor to it + 6 u: `ok = True`) has no cap. So straight
  runs and gentle turns toward two gems on one line are still approved at 25-32 u/s, the "burst" 40.22 meant to refuse.
  Those kicks fall rarely (kfC about one C kick in ten). Not changed (the approvals are the operator's); for the
  morning: put `v_after <= SS_LAM_MAX` on the override too, or keep it if the two-gem runs are wanted.

### 40.24 Tonight's goal: one round over 175; the best navigator measured, a stuck-breaker bug (10-03 23:10-23:40)

* Operator 23:10: "your goal overnight is to get a single round over 175 points. you are approved to make any edits you
  need to in order to accomplish the goal." The standing rules stay (no map hacks, the game's points as the only reward,
  nothing tuned on Horizon / Archipelago, demos are targets and never templates).
* nav/real_run.py NAV_NO_USE=1 (eval only): the use bit is ignored, so checkpoints from before the use head play as the
  navigator alone (their fresh head sits at the prior's threshold). 28897 migrated to V9 (models/nav/nav_v9_28897.pth;
  tour-run endpoints 28954 / 29700 / 29794 likewise, nav_v9_tour_<u>.pth).
* 28897, navigator alone, 8 rounds in the CPU gate: 155 154 160 159 / 168 156 164 163 = 159.9 (hybrid harness 10-02:
  160.1), falls 1.25, stuck-breaks 1.6 a round; tour end 29794: 157 164 158 156 = 158.8 (4 rounds); 28897 with
  NAV_STUCK_S 1.5: 161 165 149 162 = 159.2 (4 rounds, 4-5 stuck-breaks a round). Per round 127 pickups, 96 of them after
  1-2 s legs (131 s); falls cost ~5 s in long gaps; the game with 0-1 falls a round scored 168 / 156 / 164 / 163, the one
  with 1-4 falls 155 / 154 / 160 / 159: about 3 points a fall. All 20 falls in 16 rounds were roll-offs off a hole's lip
  near the target gem (0.5-10 u), the jump key pressed only after the marble had left the floor.
* Stuck-breaker bug (real_run): all 35 triggers in those 16 rounds had a gem picked up inside the 3 s window: the marble
  had looped through a gem cluster (the four centre gems sit on a 4 u square) and come back within 1 u of where it was
  3 s before, so "moved less than 1 u in 3 s" fired; the breaker then reset the memory and sampled 24 decisions along a
  path waypoint, next to the centre holes (4 of the 20 falls came within 40 decisions after one). Fix: NAV_STUCK_PICKUP
  (default on): a pickup inside the window is progress, the breaker does not fire. Map-independent; real_run only
  (hybrid.py has the same test, unchanged so far).
* Plan for the goal: the strongest navigator (28897, mean ~160, sd ~5) cannot be lifted 15 points in a night, so it
  plays many deterministic rounds beside the trainer (logs/nav/gate_scripts/many_rounds.sh: batches of two CPU games x
  4 rounds, ~3 rounds a minute, stops at the first round of 176+, prints every round of 170+; output
  logs/nav/many_rounds_28897.out; batch 1 from 23:19 with the old breaker, batches 2+ with the fix), while anything that
  raises its mean is measured on the same stream. Training continues as set up (40.22) and its snapshots are gated
  with uses allowed.

### 40.25 The goal is for the powerup model; demo vs the approval; training stopped (10-03 23:31-23:41; operator)

* Operator 23:31: the 175 must come from the model being trained, powerups live, not the navigator alone. The 28897
  runner was stopped after three batches (8 rounds each): 159.6 with the old stuck-breaker, then 162.1 and 162.0 with
  the fix (best 170, no stuck-breaker trigger at all); not a goal round. 23:32: nine game windows bogged the PC down;
  back to the trainer's 8, no extra windows while the operator uses the PC.
* The approval replayed over the operator's demo (scratch demo_approval.py: ObsBuilder + use_prior every 64 ms, the
  waypoint = the gem taken next): of his 20 Super Speed fires, 13 had an approved decision in the 0.26 s before them;
  the refused ones were the floor check (the floor along the exact line ends 4-4.5 u ahead, yet the kick worked) and
  twice the 24 u/s cap. Over the round, 36 of the 445 decisions with a Super Speed held were approved (the model sees
  ~9 a round). So the rule admits most of the human's kicks; what differs is the supply (he took a Super Speed about
  20 times a round, mostly the same item at every respawn; the model holds one unused for long stretches and never
  detours for a fresh one) and the kicks' outcome (the model's turns reach the gem 56-63 % of the time).
* Started, then reverted when training stopped: re-aiming at the fire. The aim is solved when the use is sent and the
  kick fires two decisions later; the bridge latch could take a re-solved yaw on the decision before the fire
  (mlAgent.cs: update $MLAgent::PowYaw while the hold runs; the worker sends ss_aim(vec) each decision while a fire is
  pending). Proposal, not in the code.
* Operator 23:41: stop training. Stopped (loop, trainer, games); endpoint models/nav/nav_p4f_stop_30575.pth
  (= nav_latest). The cap run (22:56-23:40, 488 rounds): 138.8 a round (best 156), falls 2.17, Super Speed fires 2.12,
  turns 1.76 with 56 % reaching the gem, falls within 3 s of a fire 0.26, kicks by resolved speed A 0.61 / B 0.74 /
  C 0.76 a round (falls 0.02 / 0.17 / 0.07), pickup speed with the approach flag on 8.73, off 7.35, none 8.09 u/s,
  reward per decision after a fire 1.11 against 1.62 overall.
* Code left from tonight: nav/real_run.py NAV_NO_USE (eval only) and NAV_STUCK_PICKUP (default on; the stuck-breaker
  ignores windows with a pickup); logs/nav/gate_scripts/many_rounds.sh; models/nav/nav_v9_28897.pth and
  nav_v9_tour_{28954,29700,29794}.pth (V9 migrations). Nothing else changed (the cap override of 40.23 is as it was).

### 40.26 The Super Speed curriculum, Stage 1, built and started (10-03 23:45 - 23:58; operator: implement the other agent's plan overnight)

The plan (the other agent's review, operator 23:45): fix what keeps the use decision from learning, then a reverse
curriculum inside ONE policy: (1) control after real kicks, (2) use or not shortly before, (3) the approach, (4) whole
rounds; isolate Super Speed; start from 28897; judge by matched comparisons and deterministic play. Built tonight:
* Use decision (nav/model.py): the physics approval is an action MASK (no use is possible where it is off); inside it
  p = USE_EPS 0.2 + 0.8 sigmoid(logit), a differentiable exploration floor (the old hard floor had zero gradient below
  it: a use that paid could never raise its own probability); the logit soft-bounded (12 tanh(raw / 12)), PPO scores
  this exact mixture, deterministic play fires only where the LEARNED part sigmoid(logit) > 0.5. USE_SS_ONLY: the mask
  keeps only the Super Speed branch; the worker and real_run fire only a held Super Speed. A fresh use head starts
  unsaturated (USE_BIAS_V10 -1: learned 0.27, sampled 0.42 where approved). Checks: the gradient reaches the head at
  raw -20, -1 and +2; p = 0.200 / 0.416 / 0.903 as specified; 1e-6 outside the mask.
* Residual (nav/model.py ss_res): a small head on [h, vec] adding to the steering (pre-tanh) and the brake logit, gated
  on (Super Speed held, a kick pending, or one within 2 s); zero-initialised, so 28897's driving is untouched at the
  start and ordinary driving keeps the base heads. One network, one action path, no external controller.
* Observation V10 (nav/obs.py, VEC_DIM 91 -> 98): [30] the kick's resulting speed; [31-36] the USE STATE (use sent in
  the last two decisions, a kick pending, the latched aim, time since the last Super Speed fire) via fill_use_state,
  filled the same way by the worker, real_run and (neutral) hybrid.
* One kick rule (nav/obs.py ss_kick_result / ss_kick_ok) for the fire approval [26] AND the approach flag [27]: the
  RESULTING velocity vector (the brake case keeps its old heading at |v| - 25; before, a scalar projection), the 24 u/s
  cap on every path (the next-gem-on-the-line override had skipped it: 8 u/s along the line + 25 = 33 u/s was approved;
  reproduced and now refused), the stop at SS_DECEL along the resulting heading. Unit checks pass (incl. a sideways
  30 u/s case the first version got wrong).
* Stage 1 starts (nav/ss_drill_starts.py): 3,349 real kicks in the agent's own traces of 10-03 18:40-23:40 (a
  horizontal velocity change >= 15 u/s in one decision on the floor): the state one decision after the kick, the gem
  it was aimed at, the gem after; plus the state 3 decisions before (for Stage 2). Split by hash: datasets/ss_drill/
  starts_dev.json 2,706 / starts_eval.json 643; with the speed after the kick <= 25 u/s (the kicks the approval now
  allows): 948 dev / 211 eval, median turn 116-125 deg, the gem 11.5 u away (the operator's demo: 77-151, 12 u).
* Drills (nav/vec_worker.py, nav/waypoints.py begin_drill / drill_mode): instances 0-3 (NAV_DRILL_INSTANCES) teleport
  into a recorded post-kick state (velocity and the rolling spin of the pre-kick velocity; engine variations: speed
  +-10 %, heading +-4 deg, position +-0.15 u) with the use state "kick fired one decision ago" and the held slot empty;
  targets: the intended gem, then the gem after it (synthetic arrival, the same shaped reward as always); a fall ends
  the drill, no chaining, 3 s per target. Instances 4-7 play ordinary rounds (Super Speed uses live through the mask).
  DRILL lines per drill, DRILLSUM per 100; logs/nav/gate_scripts/drill_stats.py.
* Not done (yet): the game-points-only objective as its own experiment (the shaped reward stays, so the drills are
  judged on the old objective); Stage 2-4; inventory-aware routing; re-aiming at the fire tick.
* Launched 23:57 from 28897 (models/nav/nav_v10_28897.pth = nav_latest; Adam padded for the 8 new tensors); the cap
  run archived as stdout_p4f_cap_20261003.txt / night_best_p4f_20261003.json (night_best reset). First 190 drills (the
  untrained policy): intended gem taken 92-97 % but median 1.8-2.4 s after the start (overshoot and come back); within
  1.0 s 22 %; the gem after 67 %; falls 17-21 %; timeouts 12-16 %.
* Deterministic drill evaluation (nav/ss_drill_eval.py, logs/nav/gate_scripts/ss_eval.sh <ckpt> <tag> <port> [--stage
  N] [--no-use]): the held-out starts in order, no variations, the worker's own drill code, one game window (~2 min).
  28897 (update 28895), stage 1, 211 starts: the kick's gem within 1.0 s 23 %, 1.5 s 29 %, at all 92 % (median 1.79 s),
  the gem after 72 %, falls 19 %, timeouts 9 %, return 56.4.

### 40.27 Stages 2 and 3 built (10-04 00:05-00:20)

* Bridge word GIVEPOW <ItemDatablock>|none (mlAgent.cs): the server's own pickup path (Marble::setPowerUp: powerUpData,
  the engine id the use key fires, the HUD) plus the client prediction; the .dso deleted and recompiled. Tested on one
  game (nav/givepow_test.py): held reads 2 after GIVEPOW SuperSpeedItem_MBU, the use fires it (+23.3 u/s along yaw 0,
  +25.0 along yaw pi; one trial met the start pad within 3 decisions), GIVEPOW none clears it. The training games of
  this run loaded the script before it, so stages 2-3 need the next restart.
* Stage 2 starts (datasets/ss_drill/starts2_*.json): 5 decisions (0.32 s) before each recorded kick, with every waypoint
  from there to the kick ('pre_goals': the pickup still ahead in 964 of 1,208 starts at <= 25 u/s, median 1.5 u away);
  Stage 3 (starts3_*.json): 20 decisions (1.3 s) before, 1-3 pickups on the way in. A stage 2/3 drill grants a Super
  Speed (GIVEPOW), teleports into the recorded state, and the POLICY decides whether and when to fire (the mask and the
  mixture as in rounds); DRILL2 / DRILL3 lines add fired / when. NAV_DRILL_PLAN 'idx:stage,...' mixes stages per
  instance; NAV_DRILL_NO_USE ignores the use bit (the matched comparison from the same starts).
* Untrained 28897, deterministic (no learned use): stage 2 (30 starts): kick's gem 93 % (median 2.53 s from the start),
  both gems 4.86 s, falls 3 %; sampled (fires 53 %): 93 %, 5.28 s, falls 10 %: firing does not pay before the
  control after the kick is learned. Stage 3 (20 starts): kick's gem 90 % (4.12 s), both 6.46 s, falls 5 %.

### 40.28 Stage 1 after 40 minutes; the arrival check; restart with a stage 2 instance (10-04 00:33-00:45)

* Training drills 23:57-00:33 (7,200): falls 21 % -> 14 % in the last six minutes, the gem after 60-65 -> 68 %, return
  53-54 -> 58.8; the kick's gem within 1.0 s flat at 21 %. Deterministic eval of 29010: within 1.0 s 22 %, falls 14 %
  (base 23 % / 19 %), timeouts 13 % (9 %).
* The recorded kicks point at their gems (median error 2.0 deg; a straight line from the start passes within the 0.65 u
  arrival radius for 57-60 % of the starts, within 1.5 u for 90 %). Two reasons the first pass is not taken:
  1. The synthetic arrival was tested on the sampled position once a decision; at 20 u/s a decision covers 1.28 u, so
     a pass 0.5 u off was counted 65 % of the time (0.6 u: 38 %; ~100 % at 8 u/s), while the game's pickup is a
     continuous collision. Fixed: waypoints._arrived tests the path since the last decision (closest approach, height
     interpolated). It changed the baseline little (28897 again 23 % / 92 % / falls 19 %), so this was not the main cause.
  2. The main cause is the navigator's habit (traced, 30 eval drills): right after the kick it steers a median 148 deg
     against its velocity (87 % of the first 8 decisions beyond 90 deg), braking 23 -> 17 u/s in 0.6 s, and the
     sideways part of that thrust bends it off the line; it then comes back for the gem (median 1.8 s). The brake
     before the gem is what Stage 1 has to unlearn (and then brake after it).
* Restart 00:37 (from 29025; stage 1a endpoint kept as nav_p4g_s1a_end_29025.pth, logs as *_s1a_20261004.txt) with the
  continuous arrival and NAV_DRILL_PLAN '0:1,1:1,2:1,3:2' (code default now): three post-kick drills, one stage 2 drill
  (fires in about half of them through the 0.2 floor), four ordinary rounds. The training games now have GIVEPOW.
  Re-measured with the continuous arrival: 28897 within 1.0 s 23 %, at all 92 %, falls 19 %, return 56.7; 29025: 25 %,
  92 %, falls 15 %, timeouts 16 %, return 57.3.

### 40.29 The residual moves after the tanh (10-04 00:47)

* In the first 6 decisions after a kick the base steering's pre-tanh output is saturated (|pre| median 1.54, p90 3.8,
  39 % above 2 where the tanh slope is under 0.07; 20 traced eval drills on 29025), so a residual added BEFORE the tanh
  got 7-20 % of its gradient exactly where it has to turn the 144-148 deg brake into a straight pass. nav/model.py now
  adds it after: mean = tanh(base) + residual (the Gaussian mean may leave [-1, 1]; action_to_joystick normalises the
  sampled direction); identical at zero (checks pass); the residual trained so far was tiny (last layer norm 0.11).
* Restart 00:47 from 29055 (endpoint nav_p4g_s1b_end_29055.pth, logs *_s1b_20261004.txt), same plan '0:1,1:1,2:1,3:2'.

### 40.30 Evals at 29205; exploration after the kick widened (10-04 01:33-01:44)

* Held-out, deterministic, 29205 (28897 in brackets): stage 1 (211): the kick's gem within 1.0 s 27 % (23), at all 91 %
  (92), both gems median 4.19 s (4.41), falls 16 % (19), timeouts 16 % (9), return 58.4 (56.7). Stage 2 (221): the
  policy fired in NONE (its learned use probability is under 0.5 everywhere), so with and without the kick read the
  same: both gems 4.67 s, falls 17-18 %, return 79.8 / 80.1. In training (sampled) stage 2, fired 302 vs not 341:
  both gems 67 / 63 %, median 4.67 / 4.99 s, falls 21 / 23 %, return 79.9 / 79.3: the kick is now about neutral.
  Ordinary rounds 00:49-01:41: 126 -> 128 -> 137 a round (the restarts' partial rounds included), falls 1.9 at the end.
* Why Stage 1 learns slowly: the navigator brakes ~145 deg against its velocity right after a kick and the steering
  noise (std 0.22) samples only +-12 deg around that, so a straight pass through the gem is hardly ever tried. Changes
  (all aimed at the 2 s after a kick; deterministic play unaffected): STD_POST_MULT 2.5 (steering noise x2.5 for 2 s
  after a fire; the entropy given to the controller excludes the boost, so it does not cut exploration elsewhere);
  SS_RES_SCALE 3 (the residual's output x3, so each Adam step moves it 3x as far; the trained residual was tiny);
  NAV_DRILL_PLAN '0:1,1:1,2:1,3:1,4:2' (4 post-kick, 1 pre-kick, 3 ordinary rounds).
* Restart 01:43 from 29235 (endpoint nav_p4g_s1c_end_29235.pth, logs *_s1c_20261004.txt).

### 40.31 Brake along the line: a post-kick brake floor (10-04 02:03-02:07)

* 20 minutes of 40.30 (4,100 drills): stage 1 flat (within 1.0 s 25 %, falls 17-20 %); stage 2 sampled: fired both
  gems 75 % in 4.67 s vs 72 % in 5.06 s not fired, falls 17 / 19 %, return 83.1 / 85.1. Traced 29300 (15 eval drills):
  still a median 151 deg against the velocity in the first 8 decisions (100 % beyond 90 deg); the residual IS training
  (output layer +75 % in 60 updates) but has not turned the brake; the post-kick line passes within 0.65 u of the gem
  in 73 % of these drills.
* The fix the physics suggests: braking EXACTLY against the velocity keeps the kick's line (the gem on the first pass,
  at a lower speed, then less overshoot); the operator held the backward key after 11 of his 20 kicks. The policy's
  brake action is exactly that (joystick: full thrust against the velocity) but its logit sits at -7 (0.09 %) and was
  never sampled. Now (BRAKE_EPS_POST 0.15): for 2 s after a fire p_brake = 0.15 + 0.85 sigmoid(logit), the same
  differentiable floor as the use decision, scored exactly by PPO; deterministic play brakes only where the learned
  part > 0.5; unchanged elsewhere. Checks: post-kick p 0.152, elsewhere 0.002-0.003 as before.
* Restart 02:06 from 29305 (endpoint nav_p4g_s1d_end_29305.pth, logs *_s1d_20261004.txt).

### 40.32 Braking along the line does not pay; the real gap is falls; both add-ons off (10-04 02:33-02:55)

* 27 minutes of the brake floor: training drills falls 22-23 % (17-20 % before), return ~50 (55-57); deterministic eval
  of 29395: within 1.0 s 27 %, falls 19 % (29205: 27 % / 16 %), return 56.2; the residual's brake output went slightly
  NEGATIVE (-0.024): PPO found the random brakes unhelpful.
* Direct test (nav/ss_drill_eval.py --force-brake K, eval only): 28897 braking for the first 4 / 8 decisions of every
  drill: within 1.0 s 23 / 22 %, falls 22 / 24 % (no brake: 23 %, 19 %). So braking along the kick line is not the
  missing skill; the straight-line estimate (60 % first-pass hits) ignored that the marble carries its pre-kick spin
  (a big-angle kick leaves spin across the new velocity, which bends the path).
* Measured against the operator's demo the first-pass rate is the wrong target: he reached the next gem 0.8-2.2 s after
  the kick (median ~1.5 s), the policy reaches it in a median 1.8-1.9 s (91 %); the gap is FALLS: 0 of 20 for him,
  14-19 % of the drills for the policy.
* Both exploration add-ons switched off (BRAKE_EPS_POST 0, STD_POST_MULT 1; the mechanisms stay in the code); the
  residual stays post-tanh x3, the plan stays '0:1,1:1,2:1,3:1,4:2'. Restart 02:42 from 29420 (endpoint
  nav_p4g_s1e_end_29420.pth, logs *_s1e_20261004.txt).
* The manoeuvre's value with the current control, matched (ss_drill_eval --force-use: fire at the first decision the
  approval allows, vs --no-use; 221 held-out stage 2 starts, 29420): with the kick (fired in 67 %): kick's gem 91 %
  (median 2.50 s), the gem after 80 %, both in a median 4.86 s, falls 17 %, timeouts 3 %, return 85.0; without: 91 %
  (2.69 s), 78 %, 4.80 s, falls 14 %, timeouts 8 %, return 83.2. About break-even: the kick reaches its gem ~0.2 s
  sooner and times out less, but adds 3 points of falls. The use decision cannot become worth learning before the
  falls after the kick come down.

### 40.33 The drills wore the navigator down; restart from 28897 with the navigator frozen (10-04 03:34-04:30)

* Evals of 29585 (deterministic, held-out starts). Stage 1 (211): within 1.0 s 24 %, gem at all 93 % (1.79 s), the
  gem after 79 % (base 72 %), falls 14 % (base 19 %), return 60.6 (base 56.4). Stage 2 matched (221): fire at the
  first approved decision (--force-use, fired 66 %): kick's gem 90 % (2.43 s), gem after 80 %, falls 19 %, timeouts
  1 %, return 84.1; no use: 86 % (2.69 s), 75 %, falls 19 %, timeouts 6 %, return 79.9. The policy's own deterministic
  choice: fired 0 %.
* Training stage 2 drills split by the pre-gem (pk) and fire time: pk0 fired (any time up to 0.7 s) return 61-63, hit2
  75-82 %, falls 10-13 % vs never fired 43.1, 57 %, 22 %; pk1 fired 93-100 vs 92.4. Firing pays in the drills.
* The learned preference where the mask approves (scratch use_probe.py: p_learn over 80 stage 2 starts): 28897 0.26,
  29025 0.19, 29235 0.07, 29420 0.06, 29585 0.10 (p90 0.27, max 0.37). It fell while only ordinary rounds trained it
  (sampled rounds: a fall within 3 s after 30-45 % of the fires, reward per decision after a fire 0.7-1.0 vs 1.5
  overall) and crept back once stage 2 drills ran.
* FULL ROUNDS (rr_gate, CPU, 16 rounds per arm, walk tour). 28897 without Super Speed: 159.0 (153-164), speed 8.38,
  falls 1.25. 29585: 141.5 without Super Speed (falls 2.94, speed 7.85); 142.4 firing at the first approved decision
  (new real_run flag NAV_FORCE_USE, diagnostic); residual zeroed (scratch rr_patch.py PATCH_RES=off) 145.1 (falls
  1.69); residual only after a kick + firing 141.3. 29235: 149.0. 29025: erratic (22, 33, 43 among 107-159).
  => Training the whole network on the drills wore the navigator down within ~130 updates (slower, more falls), and
  the residual's "Super Speed held" gate rewrote ordinary steering for the ~60 % of a round the item is just carried.
  The kick itself added nothing in full rounds with that control.
* Cause found in the trainer: every drill end (the second gem, a timeout, a fall) was TERMINAL. The critic saw the
  same-looking state with two futures (0 in a drill, the rest of the round in a round), and finishing a drill sooner
  was worth nothing beyond the per-gem speed bonus, while saving time is the whole point of Super Speed.
* Changes (restart 04:24 from 28897 V10; endpoint of the 40.26-40.32 run models/nav/nav_p4g_s1f_end_29735.pth, logs
  stdout_s1f_20261004.txt):
  - model.py FREEZE_BASE True: only use_head, ss_res and value_head train (55,941 of 857,404 parameters); the
    navigator stays exactly 28897. SS_RES_HELD False: the residual acts only while a kick is pending or within 2 s of
    one.
  - train_nav.py DRILL BOOTSTRAP: a drill's last transition gets + GAMMA * v_cont, the critic's running mean value over
    round decisions (logged as vcont=, ~163); the worker flags drill transitions ('drill' in the reply).
  - vec_worker.py NAV_DRILL_PLAN default '0:1,1:1,2:2,3:2,4:2': 2 post-kick, 3 pre-kick (the use decision), 3 rounds.
  - real_run.py NAV_FORCE_USE (default off).
* Reference on the frozen base: 28897 firing at the first approved decision (NAV_FORCE_USE, 16 rounds) 157.2, falls
  2.62 (without Super Speed 159.0, falls 1.25): the approval rule plus the 28897 control loses ~2 points a round to
  falls after the kick. That is what the residual has to fix. First 25 frozen updates: stage 1 sampled drills gem
  after 61 %, falls 27 % (the 28897 control); stage 2 pk0 fired return 62.4 vs 43.2 unfired.
* 04:34 restart (same run, from its own 28925): the policy moved by KL 0.000-0.001 per update, so the use head and
  residual got their own Adam group at LR x 10 (model.py SS_LR_MULT; the critic's head stays at LR; a 2-group
  optimizer state resumes only into a FREEZE_BASE run, other runs start a fresh Adam state). KL 0.002-0.005 after.
  Log train_nav_20261004_043426.log, watchdog_20261004c.out.
* Plan: use-probe and matched stage 2 evals at ~05:00 and ~05:30; full-round gates at ~05:45 (the learned use vs
  Super Speed off vs firing at the first approval) against 159.0.

### 40.34 Why the kicks fell: the rim counted as braking room; level-floor rule (10-04 04:46-05:15)

* Frozen run at 28966 (30 updates at LR x 10, drill ends bootstrapped): the learned use preference where approved went
  0.27 -> 1.00 (use_probe: mean 0.999, every approved decision > 0.5), so deterministic play fires at every approval.
  Post-kick control unchanged so far (stage 1 eval: falls 20 %, return 56.9; 28897 19 %, 56.7). Stage 2 matched at
  28966: the policy firing (61 %) return 79.9, falls 28 %; not firing 80.3, falls 19 %.
* What happens after a kick (28897 firing at every approval, 46 kicks in 16 rounds, scratch kick_falls.py /
  kick_falls2.py on the real_run traces): 18 fell within 3 s (39 %; 60 % when the kick left the marble at 22-26 u/s,
  23 % below 22). Every traced fall is the same sequence: the kick takes its gem in ~7 decisions at 16-19 u/s, the next
  gem is ~86 deg off, the marble overshoots ~10 u, climbs the platform's outer rim (z +1.5-2 u, off the floor) and
  rolls off it 2-3 s after the kick while the navigator turns toward the next gem.
* Cause: ss_kick_ok measured braking room with terrain gap_along, which follows a gradual bank step by step, so the
  rim counted as floor. At the two traced kicks: lip 21.0 u (needed 20.6 / 20.8 -> approved), LEVEL floor 15.5 u.
* Fix (general, no map data): obs.py SS_LEVEL_DZ 0.75 / level_run(): braking room is floor within 0.75 u of the
  kick's floor height (lip = min(lip, level run)); a terrain without height levels skips the check.
* Full rounds, 16 each unless noted (rr_gate, CPU, walk tour, deterministic navigator 28897):
  | arm | points | falls/round | kicks/round, falls within 3 s |
  |---|---|---|---|
  | old rule, no Super Speed | 159.0 | 1.25 | - |
  | old rule, fire at every approval | 157.2 | 2.62 | 2.9, 39 % |
  | level rule, fire at every approval (32 rounds) | 160.35 (160.9 inference patch, 159.8 obs.py) | 0.75 | 0.8, 1 of 26 |
  | level rule, no Super Speed | 160.2 | 0.44 | - |
  | frozen run 29003 (learned use, old rule) | 158.9 | 2.44 | 2.6, 39 % |
  | level rule + cap 30 instead of 24 (inference patch), fire at every approval | 151.5 | 2.31 | 5.3, 13 % |
  The level rule makes the kick safe (1 fall in 26 kicks) but rare, so it is worth about nothing yet; the navigator
  without kicks also falls less under the stricter flags (0.44 vs 1.25 a round, 7 vs 20 falls: it reads the approval
  flags, which phase 4 trained with a firing prior). Relaxing the cap brings back the fast straight runs: more kicks,
  more falls, 9 points lost. SS_LAM_MAX stays 24.
* 05:05 restart of the frozen run with the level rule (from its own 29020; old-rule endpoint
  nav_p4h_oldrule_end_29022.pth; log train_nav_20261004_050526.log, watchdog_20261004d.out).

### 40.35 Morning gate: the learned use on the frozen navigator (10-04 05:15-05:35)

* Training under the level rule (05:05-05:32, updates 29020-29098): stage 2 drills with a pre-gem fired return 101.8,
  falls 11 % vs not fired 94.9, 15 %; without a pre-gem 61.5 vs 47.1. Sampled rounds 146.7-147.1 (the 28897 navigator
  sampled; the 40.26-40.32 run sat at 127-135), 1.2 fires a round, falls within 3 s of a fire 0.04-0.09 a round
  (0.30-0.57 under the old rule).
* Matched stage 2 eval at 29058 (221 held-out starts, deterministic): the policy's own use (fired 48 %) return 86.8,
  kick's gem 95 % (median 2.24 s), the gem after 85 %, falls 14 %, timeouts 1 %; no use 80.6, 89 % (2.56 s), 76 %,
  17 %, 6 %. The first clearly positive matched result for the learned decision.
* FULL ROUNDS (deterministic, rr_gate, CPU, walk tour), pooled:
  | arm | rounds | points (se) | best | falls/round |
  |---|---|---|---|---|
  | frozen run 29069: learned use + post-kick residual, level rule | 48 | 161.25 (0.63) | 171 | 1.04 |
  | 28897, level rule, never fires | 32 | 160.19 (0.81) | 171 | 0.69 |
  | 28897, level rule, fires at every approval | 32 | 160.34 (0.79) | 171 | 0.75 |
  | 28897, old rule, never fires | 16 | 159.0 | 164 | 1.25 |
  | 28897, old rule, fires at every approval | 16 | 157.2 | 168 | 2.62 |
  29069 kicks 1.1 times a round, 2 of 35 kicks fell within 3 s. The learned policy is about +1 point over the same
  navigator without Super Speed (about one standard error: a direction, not yet a result) and +2 over the start of
  the night. Super Speed is now safe and slightly positive, but too rare to matter: the level rule approves ~1 kick
  a round where the operator's demo fires on most big turns.
* Training keeps running (frozen navigator, level rule, '0:1,1:1,2:2,3:2,4:2'). Snapshots tonight:
  nav_p4i_29058.pth / nav_p4i_29069.pth (frozen, level rule), nav_p4h_28966 / nav_p4h_29003 (frozen, old rule),
  nav_p4g_* (40.26-40.32 full-network run, worn down; endpoint nav_p4g_s1f_end_29735.pth).
* Suggestions (the operator's call): (1) keep the navigator frozen (or KL-anchored to 28897) for powerup work; the
  drills otherwise wear it down. (2) More safe kicks need the kick to end in less room: a kick aimed to resolve at a
  lower speed, or post-kick braking the policy can rely on (the operator brakes after 11 of 20 kicks); the rule then
  could price the stop at the full-brake rate only when the policy brakes. (3) Judge with 32+ rounds per arm: the
  differences are now 1-2 points. (4) The navigator reads the approval flags (no-use falls 1.25 -> 0.44 with the
  stricter flags): a navigator retrained with V10 flags may gain on its own.

### 40.36 Gate of the 50-round snapshot 29231 (10-04 06:48-06:53)

* Training healthy (update 29349 at 06:48; sampled rounds 147.2 over 366 since 05:05, 50-round high 148.1 at 29231,
  one sampled round of 167). Operator idle ~7 h, so one 4-window gate.
* nav_night_29231_148.pth, 32 deterministic rounds: **161.47** (se 1.00), best **172**, falls 1.12, 0.8 kicks a
  round (2 of 25 fell within 3 s). Pooled frozen-run gates (29069 + 29231, 80 rounds): ~161.3 vs 160.19 for the same
  navigator without Super Speed: still about +1 point, about one standard error.

### 40.37 Correction: no measurable Super Speed gain yet (10-04 07:24-07:36)

* nav_night_29377_149.pth (50-round high 148.8), 32 rounds: 158.72 (se 0.76), best 169, falls 1.47, 0.84 kicks a
  round (2 of 27 fell within 3 s).
* The no-Super-Speed base re-measured at the same time (28897, level-rule flags, NAV_NO_USE=1, 32 rounds): 162.00,
  best 173. Pooled base: **161.09 (se 0.55, 64 rounds)**, falls 0.73.
* Pooled learned use (29069, 29231, 29377; 112 rounds): **160.6**, falls ~1.2. So Super Speed adds nothing
  measurable yet (about -0.5, inside noise); the "+1" of 40.35 was the base's first 32 rounds reading low. The rounds
  with kicks fall more (1.0-1.5 vs 0.73 a round) although only ~2 of 27 kicks fall within 3 s: the extra falls come
  later or elsewhere (not yet traced; candidates: the frozen navigator after a kick beyond 3 s, or it reading the new
  V10 use-state inputs, which 28897 never trained with).
* What the night did gain: the navigator itself plays better under the level-rule approval flags, 161.09 (64) vs
  159.0 (16) under the old flags (falls 0.73 vs 1.25), and the 10-02 reference 160.1 (8).
* Next steps for the operator's decision: (1) trace the falls in kick rounds beyond 3 s; (2) zero the V10 use-state
  inputs for the navigator part (only the use head and residual need them) and re-gate; (3) the bigger lever is still
  more safe kicks (lower-speed kicks, reliable post-kick braking), as in 40.35.

### 40.38 The extra falls come from the trained residual (10-04 07:36-07:45)

* (2) is moot: 28897's weights on the V10 inputs (kick speed [30], use state [31-36]) are exactly zero in the vec
  encoder and the direction head, and the frozen run keeps them so. The frozen snapshots differ from 28897 only in
  value_head, use_head, ss_res and the value-normalisation buffers (checked tensor by tensor).
* Falls by time since the last kick (scratch fall_vs_kick.py), per round: learned arms (112 rounds) <3 s 0.05, 3-10 s
  0.04, >10 s 0.42, before any kick 0.68; fire-at-every-approval on 28897 (32) 0.03 / 0 / 0.12 / 0.59; no use 0.73.
  Use words per round are the same in both firing arms (2.2-2.8), so the two differ only by the residual.
* 29231 with the residual zeroed at inference (scratch rr_patch.py PATCH_RES=off, 32 rounds): **161.38** (se 0.78),
  falls **0.69**, 26 kicks, no fall within 10 s of a kick. With the residual: 161.47, falls 1.12. The residual (3 h
  of training, |w| 0.03 x SS_RES_SCALE 3) adds falls and no points; the kicks themselves are neutral (161.38 vs the
  no-use 161.09).
* Summary for the operator: Super Speed with the level rule is safe and neutral at ~0.8 kicks a round; the post-kick
  residual as trained does harm and could be switched off (SS_RES_SCALE 0) for play; the gain has to come from more
  safe kicks. The run keeps training (never stopped on my own).

### 40.39 Training stopped (operator); corrections; the night report (10-04 08:13-10:10)

* Operator 08:13: "keep training but at 10:00 AM go ahead and stop training and write up a nightly report". No gates or
  evals after that (the operator was at the PC). Stopped 10:06 (trainer, games, start_training, watchdog); endpoint
  update 29970 = nav_latest.pth = models/nav/nav_p4i_stop_29970.pth; stdout copied to stdout_p4i_20261004.txt. The
  frozen run's training rounds 05:05-10:06: 147.4 a round over 1,092 (best 167, falls 1.5), flat all morning.
* CORRECTION to 40.34, 40.35 and 40.37 ("the navigator plays better under the level-rule flags, 161.09 vs 159.0"):
  last evening's navigator-alone runner (rr_m28_*, V9 observation, NAV_NO_USE, stuck-breaker fix) scored **161.62**
  (se 0.62, 32 rounds, falls 0.69, best 170); 40.25 says three batches but five ran (batch 1 old breaker 159.62, then
  162.12, 162.00, 161.25, 161.12). So the navigator alone is ~161-162 under every flag set and the 159.0 of 04:00
  (16 rounds) was a low read. No evidence the stricter flags help the navigator.
* Highest rounds of the night: 173 (28897, Super Speed off, level rule, 07:35), 172 (frozen-run 29231, which did not
  fire in that round), 171 while firing (28897 at every approval), 170 with the learned model firing (29231). The goal
  (a round over 175 with the powerup model) was not reached. Training rounds: best 167.
* Housekeeping: 48 copies of the scratch monitor script (bash, every Monitor since 10-02 kept running after its expiry,
  each polling every 60 s) were found and stopped; the operator's game_summary.sh was left running. The night's
  diagnostics are now in logs/nav/gate_scripts/: rr_gate_patch.sh + rr_patch.py (PATCH_RES / PATCH_LAM / PATCH_LEVEL),
  probe_eval.sh + use_probe.py (learned use preference where approved), kick_falls.py, kick_falls2.py, fall_vs_kick.py.
* Night report (written for the operator): https://claude.ai/artifact/LPz1RsoFBgARwWbtTYME6a (private).

### 40.40 The second review implemented: heads that can see, the approach window, honest drill cuts (10-04 afternoon; operator: implement, do not train)

The operator passed on a second agent's review of the night (summary: Super Speed is safe and chosen but worth
nothing; the use head and the critic read only the frozen hidden state, whose powerup columns are zero; the residual
acts only after the kick so the fast approach cannot be learned; every drill end got the same mean-value bonus,
falls included; the spin in the starts was inferred, not recorded; the points-only objective was never built; the hit
counter does not say whether the intended gem was taken; judge by matched 12-15 s continuations from the agent's own
play). Checked against the code and the checkpoints: all of it holds. Built today, nothing trained:
* nav/model.py: `aux(vec)` = the goal block (vec[0:10]) + the whole powerup block (vec[VEC_POW:], 37) feeds the use
  head and the critic beside the hidden state (AUX_DIM 47); a checkpoint with the old heads loads with the critic's
  weights kept and zero aux columns (unchanged until trained), the use head fresh. The residual's gate adds the
  APPROACH window: Super Speed held, the approach flag [27] on, the gem within SS_APPROACH_U 10 u; still never while
  merely holding an item. Tests: 28897 V10 loads; the use logit now differs with the powerup block (same hidden
  state); gradients reach both heads; gate on only in the approach window / after a kick.
* nav/waypoints.py: DRILL_REWARD 'points' (default): a drill pays +1 per pickup and nothing else; DRILL_WINDOW_DEC
  188 (12 s) continuation drills in which a fall respawns and continues (outcome 'window'); 'shaped' stays available.
* nav/train_nav.py: a drill's cut is bootstrapped with the critic's value of the drill's ACTUAL final observation
  (sent by the worker; the hidden state after that step), x GAMMA, or x GAMMA^47 (the respawn) after a fall; v_cont
  only as a fallback. GRU warm-up: a live start's recorded observations run through the core before the drill.
* nav/vec_worker.py: LIVE STARTS: every round instance records each Super Speed fire of its own play (the raw
  observation 3 decisions before the use with the ACTUAL spin, the one after the kick, the waypoint and the next,
  32 observations of history, then the gems taken and the falls over the following 188 decisions) to
  logs/nav/live_starts_<stamp>_<idx>.jsonl. Stage 4 drills (NAV_DRILL_PLAN 'i:4'): GIVEPOW, teleport into the pre
  state, warm the GRU, run the 12 s window over the recorded chain of gems; DRILL4 lines carry id / picks / tpicks /
  falls for paired comparisons. The ended drill's final observation goes to the trainer with the done message.
* nav/ss_drill_starts.py --live [files]: live files -> datasets/ss_drill/starts_live_{dev,eval}.json (split by id).
  nav/ss_drill_eval.py --stage 4: window eval with warm-up; rows keyed by id; summary = pickups in 12 s, falls,
  first pickup time; branches --no-use / --force-use / learned on the same ids. nav/ss_live_smoke.py: the end-to-end
  check on one game.
* Not changed: SS_LAM_MAX 24, the level-floor rule, FREEZE_BASE True, SS_LR_MULT 10, USE_EPS 0.2; BRAKE_EPS_POST 0 and
  STD_POST_MULT 1 stay off. The residual starts at zero again (train from nav_v10_28897.pth, not from a night snapshot).
* Smoke test (17:12-17:20, one game): nav.ss_live_smoke recorded 1 fire with its continuation (real spin, 6 chained
  gems, 32 history observations), the converter made the live start files, and ss_drill_eval --stage 4 ran three 12 s
  window drills on it (warm-up, GIVEPOW, the window; 5-6 of 6 gems, one fall continued, fired 0 % deterministic, return =
  pickups). ss_drill_eval accepts --map (run_one.ps1 passes it). Worker default plan now '0:1,1:2,2:2' (run 1,
  collection). Handoff for the next run: HANDOFF_SUPERSPEED_MANEUVER_2026-10-04.md. Nothing trained, nothing committed.

### 40.41 Run 1 (collection) started (10-04 17:17; operator: start the training loop and monitor it)

* Code checked against the handoff: worker plan '0:1,1:2,2:2', LIVE_RECORD on, FREEZE_BASE True, SS_LR_MULT 10, USE_EPS
  0.2, AUX_DIM 47, SS_APPROACH_U 10, DRILL_REWARD 'points', DRILL_WINDOW_DEC 188, SS_LAM_MAX 24, SS_LEVEL_DZ 0.75.
* nav_latest.pth = models/nav/nav_v10_28897.pth (update 28895; the night's endpoint stays nav_p4i_stop_29970.pth);
  night_best.json archived as night_best_p4i_20261004.json (reset), stdout as stdout_before_run1_20261004.txt.
* Trainer from 17:17:36 (log train_nav_20261004_171736.log): FREEZE_BASE 61,957 trainable parameters (value head at LR,
  use head and residual at LR x 10, fresh Adam), 8 games, 16-17 s an update, KL 0.001 at the start. Watchdog
  watchdog_20261004e.out. First 3 minutes: 150 stage 1 drills, 259 stage 2 drills (return = pickups), 15 rounds, live
  start files from all five round instances (logs/nav/live_starts_20261004_17*_<3-7>.jsonl).
* The monitor script (scratchpad monitor2.sh) now exits by itself after 29 minutes: the old one ran on after every
  Monitor expiry (48 copies found at 10:06).
* Plan: ~18:20 convert the live files (python -m nav.ss_drill_starts --live) and restart as run 2 ('0:1,1:2,2:4,3:4').

### 40.42 Run 1 results; the live start set; run 2; the warm-start scoring bug fixed (10-04 18:20-18:30)

* Run 1 (17:17-18:20, updates 28895-29115, KL 0.0021 a update): sampled rounds 146.8 / 147.8 by half (falls 1.46 /
  1.47, Super Speed fires 1.42 / 1.58 a round, falls within 3 s of a fire 0.03, pickup speed with the approach flag on
  8.97 / 9.11 vs held-off 8.08 u/s). Stage 1 drills (3,444): the gem after 65 %, falls 24 %, 1.55 points. Stage 2
  drills, fired vs not (points-only returns): no pre-gem 1.71 vs 1.57 points, gem after 76 vs 71 %, falls 12 vs 16 %;
  with a pre-gem 2.85 vs 2.79, 86 vs 83 %, falls 12 vs 15 %. Endpoint models/nav/nav_run1_end_29115.pth.
* Live starts: 549 fires recorded in five files (+ the smoke file); ss_drill_starts --live kept 347 (speed after the kick
  8-32 u/s and a chain): datasets/ss_drill/starts_live_dev.json 260 / starts_live_eval.json 87; speed after median
  19.5 u/s, 6 chain gems, a fall in the 12 s after 14 % of them (sampled play).
* Run 2 from 18:21: vec_worker DRILL_PLAN default '0:1,1:2,2:4,3:4' (games 0 stage 1, 1 stage 2, 2-3 the 12 s live
  windows, 4-7 rounds still recording).
* BUG found in the first ten updates (KL 0.17-0.60 per update, PPO's early stop at 1 epoch of 3 in most of them, clip
  fraction only 0.01-0.03): a live-start drill warms the GRU in the trainer (40.40) but the step is still a reset, and
  model.evaluate_seq ZEROED the hidden state at every reset, so the update re-scored the window's first decisions
  (where the fire is decided) from a zero state while the actor had used the warmed one. Measured (scratch
  test_warm_reset.py, 28897): the old scoring is off by up to 22 in log-probability after a warm restart. Fix:
  evaluate_seq(..., h_reset) restarts from the STORED state of that step (ppo_recurrent passes the rollout's per-step
  states): zeros for an ordinary segment start (bit-identical to before), the warmed state for a live start (now exact
  at every step). The bugged ten updates were discarded: their state is models/nav/nav_run2a_warmbug_29125.pth (stdout
  stdout_run2a_warmbug_20261004.txt); run 2 restarted 18:26 from nav_run1_end_29115.pth (log
  train_nav_20261004_182610.log, watchdog_20261004g.out): KL 0.001-0.004, 3 epochs every update.

### 40.43 Run 2 after an hour; the first matched window eval (10-04 19:00-19:40)

* Training (18:26-19:31, updates 29115-29333): KL 0.0015 a update, 3 epochs, 8 games, no tracebacks. Sampled rounds
  146.4 / 147.4 by half hour (falls 1.51, Super Speed fires 1.1-1.4 a round, falls within 3 s of a fire 0.01-0.02,
  pickup speed with the approach flag on 9.0-9.1 vs off 8.1 u/s). Stage 1 drills improving slowly: the gem after 58 ->
  69 %, falls 28 -> 21 %, points 1.45 -> 1.60. Stage 4 windows (sampled, not matched): fired ~5.0 pickups / a fall in
  27-28 % vs not fired 5.1 / 22-27 %.
* Matched window eval, models/nav/nav_r2_29330.pth, 87 held-out live starts (starts_live_eval.json), deterministic, one
  game per branch in parallel (operator idle 2 h): learned 5.05 pickups in 12 s, falls 0.18, first pickup median 2.24 s,
  fired 60 %; no use 5.11, 0.14, 3.26 s; fire at every approval 4.98, 0.26, 2.11 s, fired 94 %. Paired over the 78 ids
  present in all three: learned - no use pickups -0.05 (se 0.11), falls +0.04 (0.06); force - no use -0.17 (0.12), falls
  +0.14 (0.07); learned - force +0.12 (0.14), falls -0.10 (0.07); where the learned policy fired (45): -0.08 (0.15).
  So the use head has become selective (it beats firing at every approval) but not yet better than not using: no
  full-round gate (the handoff's bar is learned > no use by more than one se).
* Note for the operator (reward is his call): drills pay points (+1 a pickup, ~0.03 a decision) and rounds the shaped
  reward (~1.7 a decision) into ONE critic, whose value on round states fell from ~145 (run 1) to ~125-138. A wrong level
  only adds variance (the baseline does not depend on the action), but it slows learning. Points for the rounds too, or
  shaped drills, would make the two consistent.

### 40.44 Run 2 at two hours; the live start set grown to 848 (10-04 20:34-20:40)

* 19:30-20:34 (updates 29333-29543): KL 0.0013, 3 epochs; sampled rounds 146.5 / 147.2 by half hour (falls 1.5-1.6,
  Super Speed fires 1.1-1.2 a round, falls within 3 s of a fire 0.00-0.02, pickup speed with the approach flag on 9.0-9.1
  vs off 8.1 u/s: +1.0, the handoff's mark is +1.4). Stage 1 flat at the gem after 65 %, falls 24 %, 1.55 points. Stage 2
  as before (fired better: 1.56 vs 1.29 points without a pre-gem). Stage 4 windows (sampled, not matched) by half hour:
  fired 4.93 / 5.03 / 5.21 pickups vs not fired 5.01 / 4.93 / 5.12. The critic's mean round value fell further (vcont
  ~120).
* The round instances recorded 822 more fires since 18:21. Converted all 14 live files (848 starts); the run 1 starts
  keep their ids and their dev / eval split (backups starts_live_{dev,eval}_run1.json), the newer ones get unique ids
  (file stamp + ':' + the old '<instance>-<step>' id, which repeats across launches because the step counter restarts)
  and the same 1-in-4 hash split: datasets/ss_drill/starts_live_dev.json 647 (260 + 387), starts_live_eval.json 201
  (87 + 114). No eval start is in dev.
* Restart 20:36 from 29545 (state before it: models/nav/nav_r2_starts260_end_29545.pth; stdout
  stdout_run2_part1_20261004.txt; log train_nav_20261004_203618.log, watchdog_20261004h.out); the window drills use the
  new starts at once.

### 40.45 Operator: run the window eval, then stop training (10-04 21:08-21:20)

* 21:08 status (log only): since 20:36 the window drills fired in 85 % (5.04 pickups vs 4.84 unfired, not matched);
  stage 1 the gem after 68 %, falls 22 %; stage 2 fired better (1.63 vs 1.35 points without a pre-gem, falls 16 vs 25 %).
* Operator 21:10: "run it now and then stop training". Training stopped 21:13 (trainer, games, watchdog; the monitor and
  its bash process too): endpoint models/nav/nav_r2_stop_29665.pth (= nav_latest.pth), stdout
  stdout_run2_part2_20261004.txt. Run 2 trained 18:26-21:13 (updates 29115-29665).
* Matched window eval on the endpoint, the 201 held-out live starts, three games in parallel (200 windows each; 167 ids
  in all three): learned 5.01 pickups in 12 s, falls 0.26, first pickup median 2.66 s, fired 51 %; no use 5.00, 0.20,
  3.26 s; fire at every approval 5.17, 0.24, 2.18 s, fired 98 %. Paired: learned - no use -0.02 (se 0.08), falls +0.07
  (0.04); force - no use **+0.17 (0.09, t 1.8)**, falls +0.06 (0.04); learned - force -0.18 (0.09, t -2.0). Where the
  learned policy held off (84 ids) firing would have gained +0.30 (0.14); where it fired (83) it did -0.11 (0.13) vs no
  use and force +0.03 (0.12). By subset: the run 1 eval ids (70) learned -0.26 (0.13), force +0.04 (0.13); the new ids
  (97) learned +0.16 (0.09), force +0.26 (0.13).
* Reading: with the trained residual, firing at every approval now gains about 0.17 gems in the 12 s after a kick (it
  lost 0.17 at 29330 on the 87 run 1 ids), while the learned use head skips kicks that pay and is no better than not
  using: it has learned to be selective but not yet the right selection (the critic's mixed reward scales, 40.43, are a
  likely reason its advantages are noisy). The handoff's bar for a full-round gate (learned > no use by one se) is not
  met, so no gate. At ~1 kick a round, +0.17 gems a kick would be worth well under one point a round.

### 40.46 Live play against people; the production bridge split from the training files (10-04 21:25-22:40)

* Operator: play against the model (account Alfonsus) on a real multiplayer server. The official PQ client has no
  agent bridge, so the repo build (marbleblast_mbx.exe) plays. It joined an online server in Oregon as a CLIENT (the
  server is remote); every bridge assumption that held on the local host of -autotrain failed there, one at a time:
  1. real time: nav.env TRAINING_MODE sends FIXEDSTEP / LOCKSTEP / RENDEREVERY, which would stall or change a shared
     server; live play needs the old asynchronous bridge (no control words, each decision held 4 x 16 ms, SPEED 1);
  2. the bridge acted only once the clock fell below MissionInfo.time (3:00) and only while $Game::Running, which only
     the hosting server's scripts set: on a joined client it never acted. Live: act from GO (clientCmdStartTimer or the
     game state "go"), play the Ready/Set countdown; the round ends when the server ends it; no automatic restart;
  3. gems: on a joined client datablocks arrive without their script fields (classname reads "ItemData"), so no gem
     matched and the model parked on the map's centre fallback. Live: recognise gems by datablock name (the server
     sends the names, commands.cs RecDataBlockNames) or a gem shape, excluding editor-only items (BackupGem);
  4. the camera swung ~15 times a second (the steering turns the marble camera each decision): fixed render view;
  5. the engine buffers console.log, so the bridge reports why it waits and what the gem scan sees as DEBUG|live| lines
     to the Python side.
  Result: the model saw gems on 2,834 of 2,839 decisions and scored 62 points in a full round against the operator
  (its offline rounds score ~160; a human taking gems and real-time control both cost points; one round only). One
  engine crash at 21:49 (crash_dumps/crash_2026-10-04_21-49-54, no symbols). As a joined player the powerup state is
  not readable (the observer reads it from the server's client list), so no Super Speed in live play yet.
* SPLIT (operator: "maintain both a production version ... and the training version ... keep the new changes in
  separate files"). The training files are back to exactly the 21:21 commit (8d6d83b1e; git diff empty):
  Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs, .../ai/observer.cs, ml_agent/nav/env.py,
  ml_agent/nav/real_run.py. Production, all new files:
  - Marble Blast Platinum/platinum/client/scripts/ai/live/agentLive.cs: the live settings and the live versions of
    MLAgent::startLoop / update / checkDone / onTimerStart / onGameEnd, MLAgent::liveWaitNote (new) and
    AIObserver::collectGems, verbatim from the version that played the 62-point round; it redefines them after the
    training bridge loads;
  - one hook in Marble Blast Platinum/platinum/client/init.cs (right after mlAgent.cs): with the launch argument -ailive
    it loads ai/live/agentLive.cs; training never passes -ailive;
  - ml_agent/nav/live_play.py: sets the production defaults (checkpoint nav_r2_stop_29665.pth, KOTM, port 8888, speed
    1, CPU), switches nav.env to real time at runtime, prints the DEBUG|live| lines, takes the map from NAV_MAP, and
    runs nav.real_run unchanged;
  - ml_agent/play_live.ps1: starts nav.live_play and marbleblast_mbx.exe -ailive (log logs/learned_nav/live_play.txt).
* Checks: training path, one deterministic -autotrain round through nav.real_run after the split: 155 points, 123 gems,
  1 fall (with -offline, see the next point). Live path not re-run after the split (operator: stop opening PQ); its
  functions are the ones that played.
* TRAP found: logging in as Alfonsus in the repo build saved the login (platinum/client/lbprefs.cs:
  $LBPref::Username / RememberPassword, gitignored). Since then every launch of the build logs in online, and an
  -autotrain launch stops at "Ready..." with the pause menu open (two training checks stalled: one froze after 1,198
  decisions, one never connected). With -offline the same launch trained normally. Before training restarts, either
  remove the saved login from lbprefs.cs (live play then needs a login each time) or add -offline to the training
  launchers (run_game_loop.ps1, logs/nav/gate_scripts/*.sh, run_one.ps1); the operator's call.

### 40.47 Watching at 1x with a free camera (10-04 22:45-23:30)
* Operator watched the model on Skatium_Hunt at 1x (nav_latest = update 29665; terrain map generated for it,
  terrain_maps/terrain_Skatium_Hunt.npz). Rounds 31 points (14 gems, 23 stuck-breaks) and 15 points (6 gems, ended
  after 1.2 min); both ended early right after the marble got stuck.
* Operator: "rotate the camera without affecting the model", "only unlock camera in the 1x speed version". Cause: the
  agent steers through the marble's camera yaw (executeAction sets it every decision; F/B/L/R and Super Speed follow
  it) while the picture is drawn at the render-only view yaw (Marble.setViewYaw, pinned at 0). The game's mouse handler
  adds to $mvYaw, which turned the STEERING yaw between decisions.
* Fix, new files only (training files unchanged, git diff empty):
  - Marble Blast Platinum/platinum/client/scripts/ai/watch/freeCam.cs: while the agent runs at 1x with every frame
    drawn (SPEED 1, RENDEREVERY 1), the mouse x axis and the left/right arrows turn the view yaw only (re-applied after
    Python's VIEWYAW at each new round and after respawns); pitch keeps the game's handler (render only: getMarbleAxis
    and Super Speed use the yaw, the model never reads pitch); both mouse buttons do nothing. Otherwise every handler
    behaves as in default.bind.cs;
  - client/init.cs hook: the launch argument -aifreecam loads it (next to -ailive);
  - ml_agent/watch_model.ps1 [map] (default KOTM): runner in watch mode (NAV_WATCH, speed 1, 4 view sub-steps, 50
    rounds, port 9961, CPU, nav_latest, NAV_TOUR walk) plus the game with -autotrain <map> -offline -aifreecam; closing
    the game window stops the runner. Log logs/learned_nav/rr_watch_<map>.txt.
* Operator tested it on KOTM: "it works".
* Then (operator: "add the free camera fix to the live server version so we can move the camera in lobby games"):
  play_live.ps1 now launches marbleblast_mbx.exe -ailive -aifreecam, and MLWatchCam::active() is true in live play
  ($MLAgent::Live) whenever the marble exists (live play is always real time, Python sends only SPEED 1, and
  agentLive.cs pins the view for the whole session). Not yet run in a lobby game.
