## Problem: Marble Overshoots Gems at Full Speed

### Context

We're training a reinforcement learning agent (PPO) to play Marble Blast Platinum Hunt mode. The marble rolls around a flat platform collecting gems that spawn one at a time. The agent controls direction (2D continuous) and throttle (continuous 0-1). It receives observations including its position, velocity, gem positions (all camera-relative), and frame history (4 snapshots of position+velocity at 8-frame intervals going back 32 frames).

The agent currently gets ~79 gems per 5-minute game. The problem: the marble approaches gems at full speed, collects them, then overshoots significantly due to momentum. It then wastes time decelerating and backtracking to reach the next gem (which spawned in a random direction). The optimal behavior would be to approach at full speed but brake in the final few units so it's nearly stopped at the moment of collection, allowing it to immediately redirect toward the next gem.

### Why the agent hasn't learned this organically

The gem reward is a flat +200 regardless of the marble's speed at pickup. There's no reward difference between collecting while stopped vs collecting at full speed. The agent has all the information it needs to brake (velocity in obs, frame history showing momentum), but no incentive to do so.

### Proposed Solution: Continuous Speed Bonus at Gem Pickup

Add a continuous bonus on top of the base gem reward, proportional to how slow the marble is moving at the moment of collection:

```
pickup_bonus = BONUS_MAX * max(0, 1 - speed / SPEED_CAP)
```

- At speed=0: full bonus (e.g., +100)
- At speed=SPEED_CAP (e.g., 15): zero bonus
- At speed > SPEED_CAP: zero bonus
- Linear ramp between 0 and SPEED_CAP

The base +200 gem reward is always awarded. The bonus is purely additive — the agent never earns LESS for collecting fast, it just earns MORE for collecting slow. Combined with the existing gap penalty (k=0.002 ramping cost per step without a gem), the incentive becomes: travel fast between gems, brake at the end, collect while nearly stopped for +300 instead of +200.

### Files Involved

- **`ml_agent/train_ppo.py`** — PPO training server. Contains the `compute_reward()` method where the bonus would be added. Gem pickup is detected via `gem_delta > 0`. Marble speed is available from the raw observation: `obs[3:6]` are velocity (vx, vy, vz) in camera-relative coordinates. Speed = sqrt(vx^2 + vy^2 + vz^2). The reward constants (GEM_REWARD=200, etc.) are defined around line 534.

- **`ml_agent/train_ppo.py` — Actor class** (lines 66-216) — The neural network that outputs 2D direction + throttle. Has a THROTTLE_FLOOR that was 0.15 (now drops to 0 after the model matures past 20 gems/game for 3 consecutive games). The agent can already brake by pushing opposite to its velocity vector — no architecture changes needed.

- **`Marble Blast Platinum/platinum/client/scripts/ai/observer.cs`** — Collects game state and sends to Python. Velocity (obs[3:6]) is world velocity rotated into camera space. Self position, gem positions, all camera-relative. The speed bonus uses these velocity values.

- **`Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs`** — Main game loop. Sends observations to Python via TCP, receives actions (fwd, back, left, right + optional camera yaw). No changes needed for the speed bonus.

- **`Marble Blast Platinum/platinum/client/scripts/ai/agent.cs`** — Low-level input handling. Sets $mvForwardAction etc. No changes needed.

- **`Marble Blast Platinum/platinum/client/scripts/ai/socketBridge.cs`** — TCP socket communication between game and Python. No changes needed.

- **`Marble Blast Platinum/platinum/data/multiplayer/hunt/custom/FlatGemTraining_Hunt.mcs`** — Training map definition. Flat 50x50 platform with 7 gem spawn positions. No changes needed.

- **`ml_agent/play.py`** — Inference-only server for watching the model play. No training. Would benefit from the speed bonus being visible in logs but no reward computation happens here.

- **`ml_agent/dashboard.py`** — Real-time training dashboard (Plotly.js + SSE). Could display a new metric for average speed at gem pickup.

- **`ml_agent/analyze_log.py`** — Post-training log analysis. Should be updated to parse and display the new speed bonus metric if added to log output.

### Current Reward Structure (all in Python, `compute_reward()`)

- Gem pickup: +200 per point (flat)
- OOB penalty: 0 (currently disabled)
- Gap penalty: 0.002 * steps_since_last_gem (ramping, resets on pickup)
- Distance shaping: 20/(1+d/50) + 15/(1+d/3) (potential-based, difference each step)
- Grace period: 20 steps after gem/OOB/episode start (no shaping)
- Reward scale: 0.1x before storing in buffer, clipped to [-20, 20]

### Observation Layout (59 dimensions)

- obs[0:3] — self position (camera-relative)
- obs[3:6] — self velocity (camera-relative) **<-- speed computed from these**
- obs[6:31] — 5 gem slots x 5 (dx, dy, dz, points, distance)
- obs[31:35] — game state (timeElapsed, timeRemaining, myScore, gemsRemaining)
- obs[35:59] — frame history: 4 snapshots x 6 (pos+vel), at t-8, t-16, t-24, t-32
