# PlatinumQuest ML Agent — Project Briefing

## What Is This Project?

We are training a reinforcement learning agent (PPO) to play **Marble Blast Platinum** — a 3D marble-rolling game where the player controls a marble that rolls around collecting gems, avoiding falling out of bounds (OOB), and competing against other players. The specific game mode is **Hunt Mode**: a 5-minute timed round where gems spawn on a flat arena, the marble rolls to collect them, and new gems respawn when collected. Gems are worth 1, 2, or 5 points depending on type.

The game is a legacy Torque Game Engine title. We control it by injecting AI scripts written in **TorqueScript** (the engine's scripting language), which communicate with a **Python PPO training server** over a TCP socket.

## The Goal

Train the marble to efficiently collect gems in Hunt mode. The primary performance metric is **gems per game** (points scored per 5-minute round). Currently training on a custom flat arena map with 7 gems that respawn upon collection. Current best: **~79 gems/game**. The main challenge now is getting the agent to take **tighter, more direct paths** to gems instead of overshooting and circling back.

## Architecture Overview

```
+-----------------------+     TCP Socket (port 8889)     +-----------------------+
|   Torque Engine       | <---------------------------> |  Python PPO Server    |
|   (PlatinumQuest)     |                               |  (train_ppo.py)       |
|                       |   Game -> Python:              |                       |
|  mlAgent.cs           |   "obs_json|gemDelta|oob|done" |  Actor NN (policy)    |
|  observer.cs          |                               |  Critic NN (value)    |
|  socketBridge.cs      |   Python -> Game:              |  RolloutBuffer        |
|  agent.cs             |   "F,B,L,R\n"                  |  PPOTrainer           |
+-----------------------+                               |  DashboardServer      |
                                                        +-----------------------+
```

**Data flow**: Game sends raw facts (observations, gem pickups, OOB events, done signal). Python computes ALL rewards and returns actions.

## File Inventory

### Python (ML side)

| File | Absolute Path | Purpose |
|------|---------------|---------|
| **train_ppo.py** | `ml_agent\train_ppo.py` | Main training server. Contains Actor/Critic networks, PPOTrainer, RolloutBuffer, PPOServer (TCP server + training loop + reward computation). ~1000 lines. |
| **play.py** | `ml_agent\play.py` | Inference-only script. Loads a checkpoint and plays without training (deterministic policy). |
| **dashboard.py** | `ml_agent\dashboard.py` | Real-time web dashboard (port 8889). SSE + Plotly.js charts. Runs as daemon thread during training. Shows reward, entropy, gems/hr, OOB, losses, gradient norms, gap penalty, etc. |
| **analyze_log.py** | `ml_agent\analyze_log.py` | Post-training log analysis. Parses .log files and prints metrics progression, problem detection, final state summary. Must be kept in sync when new metrics are added to logs. |
| **diagnostic.py** | `ml_agent\diagnostic.py` | Diagnostic tool (port 8888). Manual play verification — lets you play and see what observations/rewards the system generates. |

### TorqueScript (Game side)

| File | Absolute Path | Purpose |
|------|---------------|---------|
| **mlAgent.cs** | `Marble Blast Platinum\platinum\client\scripts\ai\mlAgent.cs` | Main ML agent logic. Step loop (every tick), assembles observations from observer.cs, receives actions from Python, applies them as joystick inputs. Handles episode boundaries. |
| **observer.cs** | `Marble Blast Platinum\platinum\client\scripts\ai\observer.cs` | Observation builder. Gathers all 61 observation dimensions: marble state, 5 nearest gems, 3 opponents, game state. All spatial data is in **camera-relative coordinates** (x=right, y=forward). |
| **socketBridge.cs** | `Marble Blast Platinum\platinum\client\scripts\ai\socketBridge.cs` | TCP socket client. Sends observations to Python, receives actions back. Handles connection/reconnection. |
| **agent.cs** | `Marble Blast Platinum\platinum\client\scripts\ai\agent.cs` | Base AI agent framework. Manages AI agent lifecycle, action application (setCustomAction for analog float inputs). |

### Game Data

| File | Absolute Path | Purpose |
|------|---------------|---------|
| **FlatGemTraining_Hunt.mcs** | `Marble Blast Platinum\platinum\data\multiplayer\hunt\custom\FlatGemTraining_Hunt.mcs` | Custom training map. Flat arena with 7 gem spawn positions. `gemGroups="0"` (all gems in one group for reliable spawning). 5-minute rounds. |
| **huntGems.cs** | `Marble Blast Platinum\platinum\server\scripts\huntGems.cs` | Server-side gem spawning logic. `makeGemGroup` scans for Items with `datablock.classname="Gem"`, `getCenterGems` finds spawn positions. |

### Training Output

| Location | Purpose |
|----------|---------|
| `ml_agent\logs\` | Training log files (timestamped .log files) |
| `ml_agent\models\checkpoints\` | Model checkpoints (update_N.pth, best.pth) |

## Neural Network Architecture

**Separate Actor and Critic networks** (previously shared, split for independent learning rates):

```
Actor Network:                    Critic Network:
  Input: 61 obs                     Input: 61 obs
  Linear(61, 256) -> ReLU           Linear(61, 256) -> ReLU
  Linear(256, 256) -> ReLU          Linear(256, 256) -> ReLU
  Linear(256, 128) -> ReLU          Linear(256, 128) -> ReLU
  Linear(128, 64) -> ReLU           Linear(128, 64) -> ReLU
  Linear(64, 2) -> (dx, dy)         Linear(64, 1) -> Value estimate
  atan2(dx, dy) -> angle
  Normal(angle, exp(log_std))
  Sample -> angle_to_joystick()
  -> (fwd, back, left, right) [0,1]
```

- **Continuous action space**: single angle in radians, always full magnitude (no idle)
- **Learnable log_std**: single scalar parameter, clamped to [-2.0, 1.0] (7.7 to 156 degrees)

## Observation Space (61 dimensions)

The game sends 37 raw observation dims. Python appends 24 dims of frame history (4 historical snapshots of position+velocity, each 4 frames apart) to give the model trajectory information (acceleration, jerk).

### Base observations (37 dims, from game)

| Indices | Count | Contents | Normalization |
|---------|-------|----------|---------------|
| 0-2 | 3 | Marble position (world x,y,z) | /100 |
| 3-5 | 3 | Marble velocity (camera-relative x,y,z) | /20 |
| 6 | 1 | Camera yaw (radians) | /pi |
| 7 | 1 | Camera pitch (radians) | /1.57 |
| 8-32 | 25 | 5 nearest gems x 5 (cam-rel direction unit vec, value, distance) | direction=unit vec, value/5, dist/100 |
| 33 | 1 | Time elapsed (ms) | /300000 |
| 34 | 1 | Time remaining (ms) | /300000 |
| 35 | 1 | My gem score | /100 |
| 36 | 1 | Gems remaining | /50 |

Removed from serialization (still collected by observer.cs for future use):
- Collision radius, powerup ID, mega marble active/time, powerup timer (5 dims)
- 3 opponent slots (18 dims)
- Opponent best score (1 dim)

### Frame history (24 dims, appended by Python)

| Indices | Count | Contents | Normalization |
|---------|-------|----------|---------------|
| 37-42 | 6 | Position+velocity from t-4 frames ago | same as obs[0:6] |
| 43-48 | 6 | Position+velocity from t-8 frames ago | same as obs[0:6] |
| 49-54 | 6 | Position+velocity from t-12 frames ago | same as obs[0:6] |
| 55-60 | 6 | Position+velocity from t-16 frames ago | same as obs[0:6] |

Frame history is parameterized: `FRAME_HISTORY_COUNT=4`, `FRAME_SKIP=4`. Filled with zeros until enough frames exist. Cleared on episode boundaries.

**Sentinel value**: `-999` for absent gems. Replaced during normalization with zeros (direction/value) and max distance (1.0).

**Camera-relative coordinates**: All spatial observations are rotated into camera space where x=right, y=forward. The agent's actions are relative to what it "sees."

## Communication Protocol

Game sends 4 pipe-delimited fields per step:
```
obs_json|gemDelta|oob|done
```
- `obs_json`: JSON array of 37 observation values (Python appends 24 frame-history dims -> 61 total)
- `gemDelta`: gem points collected this step (0, 1, 2, or 5)
- `oob`: 1 if marble went out of bounds this step, else 0
- `done`: 1 if episode ended, else 0

Python computes all rewards from these raw facts and returns:
```
F,B,L,R\n
```
Four float values [0,1] for forward, backward, left, right joystick inputs.

## Reward Structure (All Computed in Python)

| Signal | Raw Reward | After 0.1x scaling | Notes |
|--------|-----------|---------------------|-------|
| Gem collected | +200 per point | +20 to +100 | 1pt=+200, 2pt=+400, 5pt=+1000 |
| Out of bounds | -25 | -2.5 | Gap penalty intentionally NOT reset on OOB |
| Ramping gap penalty | -k * steps_since_gem per step | varies | k=0.001, resets to 0 on gem pickup |
| Distance shaping | P(d) = 20/(1+d/50) + 15/(1+d/3) | +0.05 to +3.5 | Broad gradient + steep near-gem well |

- **reward_scale = 0.1** applied before storing in buffer
- **reward clip = [-20, 20]** after scaling
- **5-step grace period** after gem pickup (no shaping penalty during respawn transition)
- **Ramping gap penalty** replaces the old flat time penalty. Cost per step grows linearly with time since last gem, creating quadratic total cost for long gaps. Normal 250-step gap costs ~31 raw. Overshooting a gem by 1.5 seconds (~94 steps) costs ~14% of one gem's reward.

## PPO Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Actor learning rate | 3e-5 | Adam optimizer, separate from critic |
| Critic learning rate | 1e-4 | Adam optimizer, higher than actor |
| Rollout size | 2048 | Steps per PPO update (~11% of one episode) |
| Batch size | 256 | 8 mini-batches per epoch |
| PPO epochs | 4 | Passes over buffer per update |
| Gamma | 0.99 | Discount factor |
| Lambda | 0.95 | GAE parameter |
| Clip epsilon | 0.2 | PPO clipping |
| Value coef | 0.5 | Critic loss weight |
| Entropy coef | 0.008 | Entropy bonus weight (tuned from 0.005) |
| Max grad norm | 1.0 | Gradient clipping (actor and critic) |
| VF clip | 20.0 | Value function clipping range |
| Target KL | 2.667 | KL early stopping (fires at 1.5x = 4.0) |
| Step cap | 15000 | Max steps per episode (in mlAgent.cs) |

## Training Protocol

1. Game runs at variable speed (typically 3-25x, adjustable during training)
2. Each Hunt round is 5 minutes game-time = ~18,875 steps at normal speed
3. Python server listens on port 8889, game connects via TCP
4. Every game tick: game sends obs -> Python returns action
5. Every 2048 steps: PPO update runs (4 epochs x 8 mini-batches)
6. Checkpoints saved every 10 updates (includes actor, critic, both optimizer states, log_std)
7. On checkpoint load: critic and optimizer states are restored (shape-filtered, with graceful fallback for old checkpoints)
8. Dashboard available at http://localhost:8889 during training

## Dashboard Charts

- Average Reward (rolling 100-game average)
- Policy Entropy
- Gems per Hour
- OOB Count
- Policy Loss / Value Loss
- Gradient Norm (actor + critic overlay)
- Average Gap Penalty Per Gem (lower = faster pickups)

## What We've Tried & Key Bugs Fixed

### Action Space Evolution
1. **Started with Categorical(9)**: 8 compass directions + idle. Worked but lacked analog precision.
2. **Switched to continuous angle**: Actor outputs (dx,dy) -> atan2 -> angle. Normal distribution for exploration. This is the current architecture.

### Network Architecture Evolution
1. **Started with shared ActorCritic**: Single feature extractor feeding both heads. Critic couldn't learn at a different rate than actor.
2. **Split to separate Actor/Critic networks**: Independent architectures and learning rates (actor 3e-5, critic 1e-4). Critic state + optimizer now saved in checkpoints and restored on load.

### Reward Evolution
1. **Reward computed in TorqueScript** -> Moved to **Python-side reward computation**. Game now sends raw facts only.
2. **Flat time penalty** (-0.02/step) -> Replaced with **ramping gap penalty** (k * steps_since_gem). Flat penalty was invisible (2.3% of signal, 10,000 steps to equal 1 gem). Ramping penalty creates meaningful pressure for faster gem collection.
3. **Velocity-alignment / dot product shaping** -> REMOVED. Caused entropy collapse and gem-orbiting behavior.
4. **Flat distance potential** (d/50) -> Added **steep near-gem well** (15/(1+d/3)). Without it, no incentive to close last few units.

### Entropy Coefficient History
- 0.01: too low with categorical actions, collapsed to 3 diagonal actions (84%)
- 0.05: too high, policy was nearly uniform
- 0.02: stable but plateaued
- 0.01: entropy was rising (too much exploration) with continuous actions
- 0.005: entropy falling too fast, policy over-narrowed, caused death spiral with gap penalty
- **0.008** (current): attempting to stabilize entropy near 0.41-0.45

### KL Divergence Tuning History
- target_kl=0.01: way too tight, 100% stop rate, blocked all learning
- target_kl=0.5: worked early but too tight later
- target_kl=1.5: 46% stop rate, too many skipped updates
- **target_kl=2.667** (current): fires at 4.0, ~5-6% stop rate

### Major Bugs (All Fixed)
- **Camera rotation in observer.cs**: Torque's coordinate system uses forward=(sin(yaw), cos(yaw)), right=(cos(yaw), -sin(yaw)). Was wrong in 4 places.
- **Gem spawning**: `gemGroups="2"` caused frequent spawn failures. Fixed to `gemGroups="0"`.
- **Observer gem filter**: Name-based blacklist let a phantom object through. Fixed to positive check: `datablock.classname $= "Gem"`.
- **Stuck episodes**: Dead zone after restartLevel. Fixed with `$MLAgent::TimerStarted` guard.
- **Short episode flood**: `gemCount >= maxGems` permanently true in Hunt mode. Removed.
- **Step cap too low**: 7000 was less than one full game (~18,875 steps). Raised to 15000.
- **Critic cold start**: Fresh critic every restart caused ~1000-update adaptation dips. Fixed: critic + optimizer states saved and restored from checkpoints.

## Current State & Known Issues

### What's Working
- Full pipeline: game connects, obs flow, training runs, dashboard works
- Agent collects ~72-79 gems per game
- Critic restoration eliminates cold-start dips
- KL early stopping at healthy ~5-6% rate
- Ramping gap penalty creates meaningful gem-collection pressure

### Current Problem: Entropy Drift
AvgRwd has been declining (~15,393 -> ~11,000) across recent runs. Root cause: entropy_coef=0.005 caused entropy to fall too aggressively (0.48 -> 0.41), PolicyStd narrowed from 22.4 to 20.9 degrees, creating a death spiral with the gap penalty — overly narrow policy can't course-correct on misses, misses become catastrophic, penalties spike, reward drops. Just changed entropy_coef to 0.008 to stabilize.

### Proposed but Not Yet Implemented: Overshoot Detection Penalty
A surgical fix for gem-missing precision: track nearest gem distance each step. If distance drops below ~3 units then rises past ~5 without a gem pickup (gradual increase, not a respawn jump), fire a one-time penalty scaled by how close the marble got. Targets overshoots directly without inflating the base gap penalty. Pure Python-side, uses existing obs[17] (nearest gem distance), no TorqueScript changes needed.

### Key Metrics (Latest Training)
- Best average reward: ~16,743
- Best gems/game: ~79
- PolicyStd: ~21 degrees
- Entropy: ~0.41 (declining, trying to stabilize)
- KL-STOP rate: ~5-6%
- Training at update ~20,490

## TorqueScript Gotchas
- `$= ""` checks empty, `$=` is string compare, `==` is numeric
- `%var` is local, `$var` is global
- `PlayGui.currentTime` counts UP from 0; `MissionInfo.time` is total (300000ms = 5 min)
- `PlayGui.gemCount` is cumulative gem **points** (not count) — gems can be worth 1, 2, or 5 points
- Compiled `.dso` files must be deleted when `.mcs` (script) files change
- Game runs server-side gem logic (huntGems.cs) — client scripts only read state

## Windows-Specific Issues
- **No Unicode in Python log output**: Windows cp1252 encoding crashes on characters like -> (U+2192). Use ASCII alternatives: `->` instead of `->`.
