# Navigator training: handoff

> ## CURRENT STATE (2026-09-21 21:00) -- READ THIS BEFORE ANYTHING ELSE
>
> ### -> If you are picking this up cold, go straight to **section 25, "PICK UP HERE"**.
> **2026-09-22 00:10: read section 28 next.** The KOTM gap is a stop-and-go at every pickup (worst at the four
> centre gems), the reward prices a decision at 0.18 against ~0.8 in a scored round, and fast arrivals at the
> centre gems do NOT cause falls. Proposal in 28.4, awaiting the operator. Fresh 8-round eval: 106 points.
> It has the KOTM gap decomposed, the blockers ranked, the ordered list of what to do first, the
> table of everything already tried and dead, and the operating traps. This block is the summary;
> section 25 is the brief.
>
> Everything below section 2 is a dated running log. Where it disagrees with this block or with
> section 25, those two win. Sections are appended newest-last; 14-25 are 2026-09-20/21.
>
> **Goal: 150 points on KingOfTheMarble.** Human 143.5, so the target is 4.5 % above a strong
> human. At the current 1.253 points per gem that needs 39.6 gems/min against today's 25.98,
> a **1.52x** improvement.
>
> **Verified scores, 8 real rounds each on `models/nav/nav_eval_dirgain_1934.pth` (update 13,895),
> with the observer gem-source fix live:**
>
> | map | score | rounds | human | share |
> |---|---|---|---|---|
> | KingOfTheMarble | **102.9 points** | 106,106,97,96,102,104,107,105 | 143.5 | 72 % |
> | FlatGemTraining | **96.5 gems** | 99,99,95,91,96,97,96,99 | 104 | 93 % |
>
> History on KOTM: 90.8 -> 93.5 (fall-credit fix) -> 95.9 (`DIR_GOAL_GAIN = 30`) -> 98.4
> (observer gem source, section 22) -> **102.9** (legal quick respawn, section 27).
> FlatGem: 80.75 -> 84.9 -> 96.8 -> 96.5 (unchanged by section 27; that map has no falls, so it
> served as the control).
>
> | metric (FlatGem, 1 round, 100 gems) | agent | human | note |
> |---|---|---|---|
> | gems / min | 19.9 | 20.7 | 96 % of human |
> | mean speed | 10.08 u/s | 9.75 | **agent is already FASTER** |
> | distance per gem | 30.4 u | 28.3 u | **the live gap: route length** |
> | peak speed mid-leg | 12.8 u/s | 14.94 | **the live mechanism** |
> | speed at pickup | 10.55 u/s | 8.09 | agent carries more, human brakes |
> | blind decisions | 0.0 % | n/a | was 26-31 %, see section 22 |
> | falls / 100 u | 0.00 | 0.046 | not binding |
>
> **FALLS ARE NOT THE BOTTLENECK.** They were 0.585 on 2026-09-20; the fall-credit bug
> (section 17) was the cause. FlatGem now runs entire rounds at zero falls.
>
> **PACE IS NO LONGER THE BOTTLENECK EITHER, AND THE OLD PACE NUMBERS WERE MEASURED THROUGH A
> BROKEN OBSERVATION.** The 2026-09-21 morning block claimed "gems/min is the live gap" and
> "decelerates to 4.9 u/s at every pickup". Both were real symptoms of section 22's bug: the
> observation reported an EMPTY MAP for up to 1.9 s after a pickup, so the marble had nothing to
> steer at and coasted. With the live gem source the agent is now faster than the human on
> FlatGem and still slightly behind on score, because it covers 30.4 u per gem against 28.3.
>
> **THE LIVE PROBLEMS ARE (a) ROUTE LENGTH and (b) MID-LEG PEAK SPEED.** Route length is the
> larger term. Peak speed is understood and measured (section 23): thrust sits 57-63 deg off the
> marble's own velocity in the 9-18 u acceleration band where the human holds 25-31 deg, so only
> cos(57) = 0.54 of it adds speed against the human's 0.91.
>
> **Current config** (do not re-derive from older sections): `EDGE_K = 1.0`, `FALL = 25.0`,
> `TIME = 0.05`, `BRAKE = 0.05`, `BRAKE_SUPPRESS = 1.0`, `JUMP_DAMP = 3.0`, `BRAKE_ENABLED = True`,
> `THROTTLE_FLOOR = 0.90`, `CARRY = 0.0`, `TURN_COST = 0.0`, `DIR_GOAL_GAIN = 30.0`
> (`nav/model.py`, `NAV_DIR_GOAL_GAIN`), adaptive entropy in band 0.2-0.8
> (`ENTROPY_COEF` is a starting value only, the controller owns it), `DIRECT_GEM = 1`
> (real runs steer at the exact gem; no routing waypoints, no markers off a gem),
> `$AIObserver::GemSource = "server"` (observer.cs; see section 22, do NOT set back to
> `"itemarray"` except to reproduce the bug).
>
> **Training rotation:** 4 FlatGemTraining / 3 KingOfTheMarble / 1 FlatIslands, block-assigned by
> instance (0-3, 4-6, 7). `Sprawl` is generated but NOT in the rotation and its walk grid is NOT
> verified (2,150 slope-excluded cells, 19 % of walkable area, status unknown).
>
> **Not yet measured:** FlatIslands has never had a real-round evaluation. It regressed from 92 %
> to 81 % training arrivals during the `DIR_GOAL_GAIN` run and benefits from section 22's fix by
> an unknown amount.
>
> **Training is DOWN as of 20:16.** The last run reached update 13,895 on the rotation above.
> Nothing has been trained against the corrected observation yet, which is the single most
> obvious next move: every policy to date learned to cope with a map that went blank after every
> pickup.
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
