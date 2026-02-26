# PlatinumQuest ML Agent — Project Briefing

## What Is This Project?

We are training a reinforcement learning agent (PPO) to play **Marble Blast Platinum** — a 3D marble-rolling game where the player controls a marble that rolls around collecting gems, avoiding falling out of bounds (OOB), and competing against other players. The specific game mode is **Hunt Mode**: a 5-minute timed round where gems spawn on a flat arena, the marble rolls to collect them, and new gems respawn when collected. Gems are worth 1, 2, or 5 points depending on type.

The game is a legacy Torque Game Engine title. We control it by injecting AI scripts written in **TorqueScript** (the engine's scripting language), which communicate with a **Python PPO training server** over a TCP socket.

## The Goal

Train the marble to efficiently collect gems in Hunt mode. The primary performance metric is **gems per hour** — how many gem points the agent collects per real-time hour of play. Currently training from scratch on a custom flat arena map with 7 gems that respawn upon collection.

## Architecture Overview

```
┌─────────────────────┐     TCP Socket (port 8888)     ┌─────────────────────┐
│   Torque Engine      │ ◄──────────────────────────►  │  Python PPO Server   │
│   (PlatinumQuest)    │                                │  (train_ppo.py)      │
│                      │   Game → Python:               │                      │
│  mlAgent.cs          │   "[obs]|reward|done|gems|oob" │  ActorCritic NN      │
│  observer.cs         │                                │  RolloutBuffer       │
│  socketBridge.cs     │   Python → Game:               │  PPOTrainer          │
│  agent.cs            │   "F,B,L,R\n"                  │  DashboardServer     │
└─────────────────────┘                                └─────────────────────┘
```

## File Inventory (Every File In the Pipeline)

### Python (ML side)

| File | Absolute Path | Purpose |
|------|---------------|---------|
| **train_ppo.py** | `c:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\train_ppo.py` | Main training server. Contains ActorCritic neural network, PPOTrainer, RolloutBuffer, PPOServer (TCP server + training loop). ~970 lines. |
| **play.py** | `c:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\play.py` | Inference-only script. Loads a checkpoint and plays without training (deterministic policy). |
| **dashboard.py** | `c:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\dashboard.py` | Real-time web dashboard (port 8889). SSE + Plotly.js charts. Runs as daemon thread during training. Shows reward, entropy, gems/hr, OOB, losses, etc. |
| **analyze_log.py** | `c:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\analyze_log.py` | Post-training log analysis. Parses .log files and prints metrics progression, problem detection, final state summary. |

### TorqueScript (Game side)

| File | Absolute Path | Purpose |
|------|---------------|---------|
| **mlAgent.cs** | `c:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\Marble Blast Platinum\platinum\client\scripts\ai\mlAgent.cs` | Main ML agent logic. Handles the step loop (called every tick at 3x game speed), assembles observations from observer.cs, receives actions from Python, applies them as joystick inputs. Handles episode boundaries (resetEpisode, checkDone, onGameEnd). |
| **observer.cs** | `c:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\Marble Blast Platinum\platinum\client\scripts\ai\observer.cs` | Observation builder. Gathers all 61 observation dimensions: marble state (position, velocity, camera angles, powerups), 5 nearest gems (camera-relative position, value, distance), 3 opponents (camera-relative position, velocity, mega status), game state (time, scores, gems remaining). All spatial data is transformed into **camera-relative coordinates** (x=right, y=forward). |
| **socketBridge.cs** | `c:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\Marble Blast Platinum\platinum\client\scripts\ai\socketBridge.cs` | TCP socket client. Sends observations to Python, receives actions back. Handles connection/reconnection. |
| **agent.cs** | `c:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\Marble Blast Platinum\platinum\client\scripts\ai\agent.cs` | Base AI agent framework. Manages AI agent lifecycle, action application (setCustomAction for analog float inputs). |
| **recorder.cs** | `c:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\Marble Blast Platinum\platinum\client\scripts\ai\recorder.cs` | (Not actively used in training) Data recorder for saving observation/action pairs. |

### Game Data

| File | Absolute Path | Purpose |
|------|---------------|---------|
| **FlatGemTraining_Hunt.mcs** | `c:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\Marble Blast Platinum\platinum\data\multiplayer\hunt\custom\FlatGemTraining_Hunt.mcs` | Custom training map. Flat arena with 7 gem spawn positions. `gemGroups="0"` (all gems in one group for reliable spawning). 5-minute rounds. |
| **huntGems.cs** | `c:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\Marble Blast Platinum\platinum\server\scripts\huntGems.cs` | Server-side gem spawning logic. `makeGemGroup` scans for Items with `datablock.classname="Gem"`, `getCenterGems` finds spawn positions, `spawnHuntGemGroup` / `doSpawnHuntGemGroup` handle respawning. |

### Training Output

| Location | Purpose |
|----------|---------|
| `c:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\logs\` | Training log files (timestamped .log files) |
| `c:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent\models\checkpoints\` | Model checkpoints (update_N.pth, best.pth) |

## Neural Network Architecture

**ActorCritic** (shared feature extractor, separate heads):

```
Input: 61 observations
  ↓
Shared Features:
  Linear(61, 256) → ReLU
  Linear(256, 256) → ReLU
  Linear(256, 128) → ReLU
  ↓                    ↓
Actor Head:          Critic Head:
  Linear(128, 64)      Linear(128, 64)
  ReLU                 ReLU
  Linear(64, 2)        Linear(64, 1)
  ↓                    ↓
  (dx, dy)             Value estimate
  ↓
  atan2(dx, dy) → angle (radians)
  Normal(angle, exp(log_std))
  ↓
  Sample angle → angle_to_joystick() → (fwd, back, left, right) floats [0,1]
```

- **Continuous action space**: single angle in radians, always full magnitude (no idle)
- **Learnable log_std**: single scalar parameter, clamped to [LOG_STD_MIN, LOG_STD_MAX]
- **LOG_STD_MIN = -1.0** (exp(-1.0) = 0.37 rad = 21 degrees, entropy floor)
- **LOG_STD_MAX = 0.0** initially (57 degrees), annealed down by 0.0005 per update

## Observation Space (61 dimensions)

| Indices | Count | Contents | Normalization |
|---------|-------|----------|---------------|
| 0-2 | 3 | Marble position (world x,y,z) | /100 |
| 3-5 | 3 | Marble velocity (camera-relative x,y,z) | /20 |
| 6 | 1 | Camera yaw (radians) | /π |
| 7 | 1 | Camera pitch (radians) | /1.57 |
| 8 | 1 | Collision radius | raw (~0.2) |
| 9 | 1 | Powerup ID (-1..5) | raw |
| 10 | 1 | Mega marble active (0/1) | raw |
| 11 | 1 | Mega marble time remaining | /20 |
| 12 | 1 | Powerup timer remaining | /20 |
| 13-37 | 25 | 5 nearest gems × 5 (cam-rel direction unit vec, value, distance) | direction=unit vec, value/5, dist/100 |
| 38-55 | 18 | 3 opponents × 6 (cam-rel pos, cam-rel vel, isMega) | pos/100, vel/20 |
| 56-60 | 5 | Game state (timeElapsed, timeRemaining, myScore, oppBestScore, gemsRemaining) | various |

**Sentinel value**: `-999` for all absent gems/opponents. Replaced during normalization with zeros (direction/value) and max distance (1.0) for gems, or all zeros for opponents.

**Camera-relative coordinates**: All spatial observations (velocity, gem positions, opponent positions/velocities) are rotated into camera space where x=right, y=forward. This means the agent's actions are relative to what it "sees." When you rotate the camera, the observations change, so the model does respond to camera rotation — this is intentional.

## Reward Structure

All reward computation currently happens in TorqueScript (mlAgent.cs), NOT in Python. The Python side receives a pre-computed reward float.

| Signal | Raw Reward | After 0.1x scaling |
|--------|-----------|---------------------|
| Gem collected | +200 per point (1pt gem = +200, 5pt gem = +1000) | +20 to +100 |
| Out of bounds (OOB) | -25 | -2.5 |
| Time penalty | -0.02 per step | -0.002 |
| Distance shaping (nearest gem) | P(d) = 20/(1+d/50) + 15/(1+d/3) | +0.05 to +3.5 |

- **reward_scale = 0.1** applied before storing in buffer
- **reward clip = [-20, 20]** after scaling
- Distance shaping has two components: a broad gradient (d/50 term) for far gems and a steep near-gem well (d/3 term) to incentivize closing the last few units

## PPO Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-4 | Adam optimizer |
| Rollout size | 2048 | Steps per PPO update (~11% of one episode) |
| Batch size | 256 | 8 mini-batches per epoch |
| PPO epochs | 4 | Passes over buffer per update |
| Gamma (γ) | 0.99 | Discount factor |
| Lambda (λ) | 0.95 | GAE parameter |
| Clip epsilon | 0.2 | PPO clipping |
| Value coef | 0.5 | Critic loss weight |
| Entropy coef | 0.01 | Entropy bonus weight |
| Max grad norm | 1.0 | Gradient clipping |
| VF clip | 20.0 | Value function clipping range |
| Target KL | 0.5 | KL early stopping (fires at 1.5x = 0.75) |
| Step cap | 15000 | Max steps per episode (in mlAgent.cs) |

## Training Protocol

1. Game runs at **3x speed** (fast-forward for faster training)
2. Each Hunt round is 5 minutes real-time = ~11,873 steps at 3x
3. Python server listens on port 8888, game connects via TCP
4. Every game tick: game sends obs → Python returns action (F,B,L,R floats)
5. Every 2048 steps: PPO update runs (4 epochs × 8 mini-batches)
6. Checkpoints saved every 10 updates
7. Dashboard available at http://localhost:8889 during training

## What We've Tried & Key Bugs Fixed

### Action Space Evolution
1. **Started with Categorical(9)**: 8 compass directions + idle. Worked but lacked analog precision.
2. **Switched to continuous angle**: Actor outputs (dx,dy) → atan2 → angle. Normal distribution for exploration. This is the current architecture. **Incompatible with old checkpoints** — must train from scratch.

### Major Bugs (All Fixed)
- **Camera rotation in observer.cs**: Torque's coordinate system uses forward=(sin(yaw), cos(yaw)), right=(cos(yaw), -sin(yaw)). Was wrong in 4 places (self velocity, gem pos, opp pos, opp velocity).
- **Gem spawning**: `gemGroups="2"` in the map file caused frequent spawn failures. Fixed to `gemGroups="0"`.
- **Observer gem filter**: Was using name-based blacklist for powerups; a phantom object slipped through. Fixed to positive check: `datablock.classname $= "Gem"`.
- **Stuck episodes**: Dead zone after restartLevel where timer=300000ms but game is "running". Fixed with a `$MLAgent::TimerStarted` guard.
- **Short episode flood**: `gemCount >= maxGems` check was permanently true in Hunt mode (cumulative score). Removed.
- **Orbiting gems**: Flat potential d/50 gave no incentive to close last few units. Added steep near-gem well 15/(1+d/3). A facing/dot-product bonus was tried but REMOVED — it caused entropy collapse by rewarding the orbiting pattern itself.
- **Entropy collapse**: entropy_coef=0.01 was too low with categorical actions, policy collapsed onto 3 diagonal actions (84%). Bumped to 0.05, then 0.02. With continuous actions, 0.01 is fine because the Normal distribution's entropy is naturally higher.
- **Time penalty too harsh**: -0.20/step buried the shaping gradient. Reduced to -0.02/step.
- **OOB penalty tuning**: -25 alone was too cheap, -50 too harsh. Current: -25 + implicit cost from time penalty on recovery steps.
- **Step cap too low**: 7000 was less than one full game (~11,873 steps). Raised to 15000.
- **Gems per game tracking**: checkDone() timer fires 5-10 seconds before the actual game end (clientCmdGameEnd). Fixed by having onGameEnd send an authoritative game-end signal with total gem count.

### KL Divergence Early Stopping
- Added KL early stopping (SB3-style): `approx_kl = mean((exp(log_ratio) - 1) - log_ratio)`
- Fires if approx_kl > 1.5 × target_kl
- Initially set target_kl=0.01 (way too tight — fired 100% of updates, blocked all learning)
- **Currently target_kl=0.5** (threshold 0.75, catches destructive KL >1.16 while allowing normal 0.02-0.33)

### Reward Shaping History
1. Started with velocity-alignment shaping (dot product of velocity and gem direction) → caused entropy collapse / orbiting
2. Tried various combinations of distance potential, facing bonus, time penalty
3. **Current approach**: distance potential P(d) = 20/(1+d/50) + 15/(1+d/3), no facing bonus, mild time penalty -0.02/step

## Current State & Open Questions

### What's Working
- The pipeline is stable: game connects, observations flow, training runs, dashboard works
- The agent has learned to move toward gems and collect them
- Gems per hour is the primary metric we track
- KL early stopping catches destructive updates without blocking normal learning

### TODO: Move Reward Computation to Python
Currently reward is computed in TorqueScript (mlAgent.cs). This requires deleting compiled .dso files whenever reward logic changes. The plan is to have CS send raw facts (obs, gemDelta, wasOOB) and have Python compute rewards. Benefits: no .dso deletion, dynamic reward tuning during training, reward scheduling, all reward logic in one place. This has NOT been implemented yet.

### Key Metrics From Recent Training
- The agent collects gems and achieves positive reward
- Entropy tends to hover in a healthy range with the continuous action space
- OOB events decrease as training progresses
- Log_std anneals from ~57 degrees down toward ~21 degrees over training

### TorqueScript Gotchas
- `$= ""` checks empty, `$=` is string compare, `==` is numeric
- `%var` is local, `$var` is global
- `PlayGui.currentTime` counts UP from 0; `MissionInfo.time` is total (300000ms = 5 min)
- `PlayGui.gemCount` is cumulative gem **points** (not count) — gems can be worth 1, 2, or 5 points
- Compiled `.dso` files must be deleted when `.mcs` (script) files change, otherwise the engine uses the stale compiled version
- Game runs server-side gem logic (huntGems.cs) — client scripts (observer.cs, mlAgent.cs) only read state

### Windows-Specific Issues
- **No Unicode in Python log output**: Windows cp1252 encoding crashes on characters like → (U+2192), ≈, ←. The degree symbol (°) is fine. Use ASCII alternatives: `->` instead of `→`.
