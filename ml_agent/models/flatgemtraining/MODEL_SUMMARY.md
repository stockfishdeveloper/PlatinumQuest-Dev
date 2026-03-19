# Flat Gem Training Model — Final Summary

## Map
**Flat Gem Training** (`FlatGemTraining_Hunt.mcs`) — 50x50 flat platform, 7 gem spawn positions, 1 gem at a time, no powerups, no obstacles, no opponents. 5-minute Hunt rounds.

## Performance
- **Average gems/game:** 79 (last 100 episodes)
- **Best single game:** 82 gems
- **Gems/minute:** 15.8
- **OOB rate:** ~13/game (no OOB penalty active)
- **Best avg reward:** 17,184.5

## Model Architecture
- **Actor:** Separate from Critic. Shared backbone (256-256-128-64 ReLU) -> actor_mean head (64->32->2 with tanh + small init) for 2D direction, throttle head (64->1, bias=2.0) for throttle via sigmoid.
- **Critic:** Independent network (256-256-128-64-1 ReLU).
- **Action space:** 2D continuous direction (dx, dy) + continuous throttle. Isotropic Gaussian on direction, Normal on throttle logit.
- **Observation dim:** 59 (35 base + 24 frame history). Base: self pos/vel (6, camera-relative), 5 gem slots (25), game state (4). Frame history: 4 snapshots of pos+vel at 8-frame intervals.

## Hyperparameters
- actor_lr: 3e-5, critic_lr: 1e-4
- entropy_coef: 0.01 (later reduced to 0.005)
- rollout: 2048, batch: 256, PPO epochs: 4
- clip_epsilon: 0.2, target_kl: 2.667
- gamma: 0.99, lambda: 0.95
- reward_scale: 0.1, clip [-20, 20]
- THROTTLE_FLOOR: 0.15 (with maturity gate to drop to 0 after 3 consecutive 20+ gem games)

## Reward Structure
- Gem pickup: +200 per point
- Distance shaping: 20/(1+d/50) + 15/(1+d/3) (potential-based)
- Grace period: 20 steps after gem/OOB/episode start
- OOB penalty: 0 (disabled)
- Gap penalty: 0 (disabled)
- Slow pickup bonus: 0 (tried +200, didn't work — model never learned to brake)

## Best Checkpoint
- **File:** best.pth (= update_5220)
- **Updates:** 5,220
- **Steps:** 10,627,103
- **Episodes:** 563
- **PolicyStd:** 21.7 deg
- **ThrottleStd:** 0.564
- **Training time:** ~20 hours

## Key Bugs Found & Fixed During This Model's Development

### 1. Tanh Saturation (Actor Mean)
The actor_mean head output (dx, dy) was passed through tanh for bounding. With default weight initialization, the pre-tanh values were large enough to saturate tanh at (+1, +1) from the start. The model output a fixed 45-degree direction regardless of input and relied entirely on stochastic sampling noise to collect gems.

**Fix:** Small initialization on the final actor_mean layer (weights * 0.01, bias = 0) so pre-tanh values start near zero where tanh is linear. Verified with probe_real_obs.py: pre-tanh values were +/-0.03 with real game observations — fully in the linear region.

### 2. Camera Yaw Desync (observer.cs)
Observer.cs read camera yaw from `$cameraYaw` (a TorqueScript global), but the Torque engine applies marble movement relative to `$MP::MyMarble.getCameraYaw()` (the marble object's internal camera). These two values diverge because playGui.cs modifies `$cameraYaw` every frame and the engine's Move struct processing modifies the marble's internal camera independently.

This caused observations to be rotated by a different angle than the engine used for movement. The model learned to compensate at its training camera angle, but performance degraded by ~15% at other angles, with a reproducible periodic dip pattern (0.97 correlation across test runs).

**Fix:** Changed all three rotation sites in observer.cs to read from `$MP::MyMarble.getCameraYaw()` instead of `$cameraYaw`. Verified with camera_obs_compare.py: gem at (3,17) correctly appeared as (-17, 3) at 90-degree camera (was showing (-12.68, 11.71) before fix).

**Consequence:** Required full retrain from scratch since obs values changed.

### 3. Camera Yaw Randomization
Added random camera yaw at episode start and after each OOB respawn during training. The yaw is sent as a 5th field in the action protocol, parsed by executeAction in mlAgent.cs. This trains the model to be camera-invariant by experiencing all orientations. Also zeroed $mvYaw/$mvYawLeftSpeed/$mvYawRightSpeed in AIAgent::clearInputs() to prevent camera drift.

### 4. Entropy Runaway with Continuous Actions
Any positive entropy_coef caused entropy to climb monotonically with the 3D continuous action space (dx, dy, throttle). The entropy bonus overwhelmed the reward gradient, pushing the policy toward zero-mean directions. Multiple training runs confirmed this pattern.

**Fix:** entropy_coef=0.01 with tanh-bounded outputs. The tanh prevents the mean from escaping to infinity (which was the old model's way of ignoring the entropy bonus). PolicyStd stabilized at ~22 degrees.

### 5. Hardcoded Learning Rate Override
Lines in checkpoint resume code hardcoded actor_lr=1.5e-5, critic_lr=5e-5, ignoring any changes to constructor defaults.

**Fix:** Read LR from optimizer.defaults and re-apply, so constructor values are always used.

## Failed Experiments
- **Slow pickup bonus (+200 for collecting gems at low speed):** Model never learned to brake. Throttle stayed pegged at 0.97-1.00 through 15+ hours of training. The reward signal only fires on the single gem collection frame — too sparse for the model to connect "I should have braked 20 frames ago" to the bonus.
- **Gap penalty (k=0.002):** Added ramping cost per step without a gem. Combined with slow pickup bonus, this was supposed to incentivize "sprint then brake." The gap penalty worked (penalized slow play) but the braking signal was too weak.
- **OOB penalty at various levels:** -25 was the working value with the old reward structure. Not needed on this map since camera randomization already provides diverse training.

## What This Model Learned
- Precise steering toward nearest gem in any camera orientation
- Full-speed gem collection with immediate redirect
- Edge awareness (low OOB rate without explicit penalty)
- Camera-invariant behavior (verified with 36-angle test)

## What This Model Did NOT Learn
- Speed control / braking before gem pickup
- Any concept of powerups, terrain, opponents, or jumping
- Multi-gem pathfinding (only 1 gem at a time on this map)

## Files
- **Training script:** ml_agent/train_ppo.py
- **Inference script:** ml_agent/play.py
- **Game observer:** Marble Blast Platinum/platinum/client/scripts/ai/observer.cs
- **Game agent loop:** Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs
- **Game input handling:** Marble Blast Platinum/platinum/client/scripts/ai/agent.cs
- **Training map:** Marble Blast Platinum/platinum/data/multiplayer/hunt/custom/FlatGemTraining_Hunt.mcs
- **Dashboard:** ml_agent/dashboard.py (port 8889)
- **Camera angle diagnostics:** ml_agent/anglediagnostic/
