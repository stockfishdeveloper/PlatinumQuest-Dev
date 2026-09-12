# Dashboard Chat Context

> This file is loaded into the system prompt of the dashboard's chat assistant on every message. Edit it as the project evolves — changes take effect on the next question, no restart needed.

---

## Project Overview

Training a PPO reinforcement-learning agent to play **Marble Blast Platinum** Hunt mode. The agent controls a marble that must collect gems that spawn one at a time on various maps. Final goal: the agent should play competitively against a human opponent on real Hunt maps (e.g. Prophetic Hunt) using powerups, terrain, and opponent awareness.

**Training happens in stages**, each adding new capabilities:

1. **Flat map, gem collection** — ✅ ~80 gems/game achieved
2. **Flat map with jumping** — 🔄 current stage (FlatWithJump map: 7 ground gems + 3 gems on raised blocks)
3. Simple terrain (ramps, drops)
4. Powerups (super speed, super jump, helicopter, blast, etc.)
5. Competing on the actual Prophetic Hunt map (solo)
6. Opponent awareness + self-play for competitive play

## Current Session State

- **Map:** `FlatWithJump_Hunt.mcs` — 50×50 flat platform with **4 SYMMETRIC raised blocks at (±10, ±10)** — one in each quadrant. **8 gem groups: 4 ground (cardinal axes at radius 17) + 4 elevated (on top of platforms) = 50/50 split**. Symmetric platform distribution prevents the model from learning XY-shortcut policies — previously with 3 platforms (2 at Y<0), the model learned "always jump when Y<0" as a shortcut that gave decent discrimination on average but wrong behavior at the (8,8) NE platform (4.74% jump prob there) and false positives elsewhere in the southern half. Symmetric layout forces the model to learn actual platform recognition, not Y-axis bias. Only 1 gem spawns at a time from a random group.
- **Resumed from:** `update_18770.pth` (fourth attempt — fresh jump pathway + fresh critic, fixed camera). After the separate `jump_features` architecture trained for 1.5 hours with random camera and showed per-observation variance growing to ~4pp but no actual discrimination (A-C gap stayed near 0), we identified two issues and fixed both at once: (1) turned off camera randomization so the jump network can learn XY-based discrimination, and (2) reset both the `jump_features`/`jump_head` (which had 920 updates of random-camera training to overwrite) and the **entire critic** (which had been trained across many reward structures over the journey and had a mushy value function). Direction/throttle/features backbone preserved.
- **Architecture change: jump pathway is now fully decoupled from the shared backbone.**
  - Removed `jump_norm` (BatchNorm) entirely.
  - Added **`self.jump_features`**: independent `Linear(59 → 64) → ReLU → Linear(64 → 32) → ReLU` MLP that takes the normalized observations directly (not the shared features).
  - Simplified `self.jump_head` to a single `Linear(32 → 1)` with bias = -1.0 (≈27% starting jump rate).
  - `forward()` routes: `jump_logit = clamp(jump_head(jump_features(state)), -3, 3)`.
  - Removed the eval/train-mode toggle in `get_action` — no longer needed without BatchNorm.
  - **Why this might work**: the shared features backbone is optimized for direction/throttle decisions (dense gradients, continuous rewards). Its 128-dim representation probably doesn't encode "near-platform XY" or "approaching elevated gem" — those aren't useful for steering. The new `jump_features` network can learn its own representation tailored to the jump decision.
  - Direction/throttle/features backbone/critic weights all intact — steering behavior should be unaffected.
  - **Trade-off**: jump pathway starts learning from scratch (no partial credit from old jump_head since input space changed from 128-dim features → 59-dim raw obs).
- **What to expect at training start:**
  - **Jump probability**: starts at ~27% uniformly (bias = -1). Should quickly become input-dependent as the new jump_features MLP learns a useful representation. Within the first few hundred episodes, if the theory is right, jump rate near platforms should rise and jump rate on flat ground should drop (net average may stay moderate, ~10–30%).
  - **Direction/throttle behavior unaffected** — gems/game should start at 50+ immediately (previous run's capability).
  - **Key metric: variance in jump probability across observations**, not just the average rate. Probe the model periodically with different XY positions and gem dz to see if discrimination is emerging.
  - **`avg_steps/gem`** should drop meaningfully if the model starts collecting elevated gems via learned jumping.
  - **Fallback if this still doesn't work**: the problem is more fundamental than representation — probably reward sparsity or credit assignment. Next options would be temporal-difference learning on jump-specific signals, curriculum progression (JumpOnly → mixed), or a denser reward signal (airborne-near-platform bonus, with the caveat that it's more map-specific).

---

## Action Space (4 dims total)

- **Direction (2D continuous):** dx, dy — sampled from an isotropic Gaussian with learnable scalar `log_std` clamped to `[-2.0, 1.0]`. Mean is output via tanh → normalized to unit circle (no wrap discontinuity).
- **Throttle (continuous):** sampled from a Normal in logit space, sigmoid → `[THROTTLE_FLOOR, 1]`. `THROTTLE_FLOOR = 0.15` initially, **drops to 0** after 3 consecutive games with ≥20 gems (maturity gate). Throttle `log_std` clamped to `[-3.0, 0.0]`.
- **Jump (Bernoulli):** sampled from `Bernoulli(sigmoid(clamp(jump_logit, -3, 3)))`. Jump is **excluded** from the entropy bonus — it learns purely from reward (Bernoulli entropy would push toward 50% and cause constant jumping).

The joystick layer converts `(dx, dy, throttle, jump)` to `(fwd, back, left, right, jump)` — throttle scales the forward/right components.

## Observation Space (59 dims, camera-relative)

| Range | Meaning |
|---|---|
| obs[0:3] | Self position (rotated into camera space) |
| obs[3:6] | Self velocity (rotated into camera space) |
| obs[6:11] | Gem 0: dx, dy, dz, point-value, distance |
| obs[11:16] | Gem 1 (slot — padded with -999 when empty) |
| obs[16:21] | Gem 2 |
| obs[21:26] | Gem 3 |
| obs[26:31] | Gem 4 |
| obs[31:35] | timeElapsed (ms), timeRemaining (ms), myScore, gemsRemaining |
| obs[35:59] | Frame history: 4 snapshots × (pos, vel) at t-8, t-16, t-24, t-32 |

Hunt mode only ever has 1 gem visible, so slots 1–4 are always sentinel (-999). Empty opponent slots and powerup fields are commented out in `observer.cs`.

Only obs[0:3] (self position) is camera-variant in an absolute sense — everything else is already relative. **Camera yaw is now FIXED at 0 radians throughout training and inference**, not randomized. Random camera was preventing the jump_features network from learning XY-based platform discrimination — the same world position produced different obs[0:2] every game. With fixed camera, "jump at these coordinates" becomes a learnable function. Since the camera doesn't move during real play either, this transfers fine to inference.

---

## Reward Structure (all computed in Python, `compute_reward()` in `train_ppo.py`)

**Currently active:**
- `GEM_REWARD = 200` per gem-point (flat — no speed/slow-pickup bonus).
- **Distance shaping** (potential-based, difference each step): `P(d) = 20/(1+d/50) + 15/(1+d/3)`. Broad attraction + steep near-gem well.
- **Grace period:** 20 steps after each gem pickup, OOB, or episode start — no shaping (prevents spurious penalty when the nearest gem changes).

**Currently active (re-enabled):**
- `GEM_GAP_PENALTY_K = 0.0005` — ramping per-step penalty `k * steps_since_gem * (steps_since_gem - 1) / 2`, resets on gem pickup. Re-enabled to give the model a reason to suppress unneeded jumps. Previously disabled because higher k values (0.002-0.003) destroyed training when the model couldn't reliably collect gems. With the current stable ~50 gems/game baseline, a low value creates pressure against wasted time without overwhelming the gem reward signal. At 370 steps/gem average, this adds ~17% of gem reward as cost — enough for meaningful gradient, not enough to dominate. The quadratic shape means brief inefficiencies are barely penalized but sustained waste explodes in cost.

**Currently active:**
- `JUMP_COST = 2.0` per step where jump action == 1 (per-jump reward delta = -2.0). Direct physical cost on jumping (encodes "jumps aren't free" — ground contact loss, OOB risk, momentum disruption). Escalation history: 0.3 (initial, dropped jump rate from 25% → ~5% but plateaued) → 0.6 (didn't move things, jump rate even crept up) → **2.0 ("nuke" mode)**. Diagnosis at 0.6: the model was jumping for the ENTIRE 50-frame platform approach (~88% per frame ≈ 44 jumps), accumulating ~26 reward cost per pickup but still netting +173. With 2.0, a 44-frame approach costs -88 (net +112 per pickup), but a 5-frame timed jump costs -10 (net +190). Strong gradient toward "jump only at the launch moment, not throughout the approach." Successful platform jumps remain clearly profitable (+112 net even with sloppy timing). Hard ceiling is ~4.0 above which platform jumps could go net-negative.

**Currently disabled (set to 0):**
- `OOB_PENALTY = 0`
- `SLOW_PICKUP_BONUS = 0` (tried; signal too sparse, didn't change behavior)
- `GEM_GAP_PENALTY_K = 0` — redundant with the more targeted JUMP_COST. Gap penalty punished any time-between-gems (which sometimes meant punishing useful slow approaches/braking); JUMP_COST punishes only the specific action we want to suppress.

**Post-reward pipeline:** `reward *= 0.1` (reward_scale), then clipped to `[-20, 20]` before storage in buffer.

---

## Architecture

Separate Actor and Critic networks, both with a 59→128→128 features backbone (ReLU).

**Actor heads**:
- `actor_mean`: consumes shared features. 128 → 64 → 2 → tanh. Small-weight init so tanh starts in linear region.
- `throttle_head`: consumes shared features. 128 → 64 → 1. Bias initialized to +2.0 so sigmoid(2)≈0.88 (near-full throttle at start).
- **`jump_features` + `jump_head`** — **independent pathway** that does NOT consume the shared features backbone. Takes the 59-dim normalized observation directly. Structure: `Linear(59 → 64) → ReLU → Linear(64 → 32) → ReLU → Linear(32 → 1)`. Logit clamped to `±3` in `forward()`. **Bias on final layer currently = -3.0 (sigmoid ≈ 5% initial jump rate)** — credit assignment experiment. At ~25% jump rate, ~7 jumps occur in the 30-frame window before each elevated gem pickup, so PPO+GAE spreads the +200 credit across many similar states (fuzzy lesson "jumping near platforms is good"). At ~5%, only ~0.6 jumps per pickup window, so the rare successful pickup events have clean credit assignment to a single jump. Compensates for fewer total samples by giving each one a stronger feature-discovery signal. Rationale for the dedicated pathway: the shared backbone is dominated by gradients from direction/throttle, so its 128-dim representation is optimized for steering decisions and may not encode signals the jump decision needs ("near a platform XY," "approaching an elevated gem").

`log_std` (direction) and `throttle_log_std` are learned scalars.

**Critic:** 59 → 128 → 128 → 1. Independent network.

---

## Training Hyperparameters

Current values in `train_ppo.py` as of 2026-09-11 (the training loop was corrected that day; see the list at the end of this section):

- `actor_lr = 3e-5`, actor grad clip `0.3`
- `critic_lr = 1e-4`, critic grad clip `5.0` (its own clip; it was throttled to 1e-5 / 0.3 before value-target normalization existed)
- **Value-target normalization (PopArt) in `Critic`.** The last layer predicts a normalized value; running mean/std of the GAE returns are refreshed every rollout with an output-preserving rescale of the last layer, so V(s) never jumps. `VL` on the Upd line is in normalized units and `Vstd=` is the scale (raw-scaled VL ~= VL * Vstd^2). Expect `CGN` around O(1) now, not ~100.
- **`ACTION_REPEAT = 4`.** Each decision is held for 4 game ticks (64 ms). Python still processes every tick: per-tick rewards are summed into the held decision, frame history stays tick-based, and a held brake is re-aimed against the current velocity each tick. The buffer stores one transition per decision. `jump_rate` / `brake_rate` on the GAME END line are now % of decisions.
- `rollout_size = 2048` decisions (= 8192 ticks, ~131 s of game time, ~20 gem events per rollout)
- `batch_size = 256`, `n_epochs = 4`
- `gamma = 0.996` per decision (== 0.999 per tick; horizon unchanged at ~16 s of game time), `lam = 0.95`
- `clip_epsilon = 0.1`, `target_kl = 1.0` (early stop fires at max_kl >= 1.5)
- `entropy_coef = -0.001` (direction + throttle only; jump and brake excluded). Direction log_std clamp `[-2.0, -0.5]` (35 deg max), throttle log_std clamp `[-3.0, -0.9]`.
- `vf_clip = 20.0` in raw-scaled units (converted to normalized units each update)
- `reward_scale = 0.1`, reward clip `[-100, 100]` per decision (was +/-20, which clipped a 2-point gem (40) and a 5-point gem (100) down to a 1-point gem's 20)
- `WARMUP_ROLLOUTS = 75` critic-only rollouts on a fresh warmup (was 300 at one tick per decision). `--warmup N` on the command line forces a fresh warmup of N rollouts; `--action-repeat N` overrides the repeat.

Training runs at **3x game speed**. One game tick is 16 ms of game time, so a 3-min KingOfTheMarble round is ~11,375 ticks / ~2,840 decisions, and a 5-min flat round is ~18,875 ticks.

### Training-loop fixes applied 2026-09-11 (already done, don't re-suggest)

- **Rollout-boundary bootstrap.** GAE used next_value = 0 at the end of every buffer even though the buffer is cut mid-episode. The last transition now bootstraps from V(s_next) of the following observation (the PPO update runs at the start of the next decision tick). Terminal transitions still use 0.
- **Game end marks done.** The `[]|-gems|0|1` game-end message finalizes the held decision with done=True, so returns no longer bootstrap across the restart teleport. It also clears the frame history.
- **Reward attached to the right action.** Each tick's reward is a consequence of earlier actions; it used to be stored with the action chosen after observing it. A decision's stored reward is now the sum over the ticks it was held.
- **Gem value is no longer clipped away** (reward clip +/-20 -> +/-100).
- **Critic un-throttled** via PopArt normalization instead of a tiny LR and a 0.3 clip.

---

## Metrics Glossary (for the dashboard)

| Metric | What it means / healthy range |
|---|---|
| **AvgRwd (100ep)** | Rolling 100-episode reward mean. Depends on reward scale. |
| **Entropy** | Direction + throttle entropy (jump excluded). Healthy 0.3 – 2.0. Below -0.5 = entropy collapse warning. |
| **PolicyStd** | Direction std in degrees. Healthy 12–25°. Hard clamp at 7.7° (log_std = -2). |
| **ThrottleStd** | Throttle std in logit space. Clamped [0.05, 1.0]. |
| **KL** | Policy KL divergence. Early-stop fires at max_kl ≥ 4.0; `target_kl = 2.667` is the soft target. |
| **KL-Stop %** | Percentage of last 100 updates where early stop triggered. Persistent high % means LR too high or policy unstable. |
| **GN / Actor grad norm** | Healthy 2–8. Very high = instability. |
| **CGN / Critic grad norm** | Healthy 5–20. Very low (<1) = critic not learning. |
| **PL (policy loss)** | Small negative values oscillating around 0 is normal. |
| **VL (value loss)** | Depends on reward scale; should decrease over time. |
| **Gems/hr** | Training-wide throughput of gem points. |
| **Avg Gems/Game** | Per-game average over recent games. This is the "how well is it playing" metric. |
| **Jump Rate** | % of steps where jump=1 in the most recent game. Not per-episode-averaged. |
| **Avg Steps/Gem** | Average steps between gem pickups. Lower = agent navigates/lands on gems faster. |
| **OOB** | Total out-of-bounds events across training. On flat maps this should trend down. |
| **Dry Rollouts** | Consecutive rollouts with 0 gems — diagnostic only, triggers `DRY` warnings. |
| **Best Avg Reward** | Best 100-ep average ever seen this run. Checkpoints write `best.pth` when a new best is hit. |

---

## Probe diagnostic gotcha (relevant for analysis, not training)

**Frame history must be in normalized scale.** The 24-dim frame history (obs[35:59]) is built during real training from `normalize_obs`'s already-normalized output, so historical pos/vel values are pre-scaled (positions ÷ 100, velocity ÷ 20). Probe scripts must mirror this — set frame history values pre-divided. Failing to do so causes `np.clip(-2, 2)` in normalize_obs to clip frame history positions to ±2 instead of leaving them in their natural normalized range, which made the model see wildly out-of-distribution inputs and produced phantom "quadrant patterns" in heatmap probes that DO NOT reflect real model behavior. Real training is unaffected; only the diagnostic probes had this bug.

## Bugs Already Fixed (don't re-suggest these)

- **Training-loop bugs (2026-09-11)**: rollout-boundary bootstrap of 0, game end not marked done, reward stored with the wrong action, gem value clipped to 1-point, critic throttled instead of normalized. All fixed; details in the Training Hyperparameters section.
- **Camera yaw mismatch**: `observer.cs` was reading the `$cameraYaw` TorqueScript global, which drifts from the engine's internal `$MP::MyMarble.getCameraYaw()`. Now reads directly from the marble.
- **Torque rotation formula**: `forward=(sin(yaw),cos(yaw))`, `right=(cos(yaw),-sin(yaw))` — fixed in 4 places in observer.cs (self vel, gem pos, opp pos, opp vel).
- **Mixed coordinate frames**: position was world-space while velocity was camera-relative. Fixed by rotating position into camera space too, and dropping redundant yaw/pitch from obs.
- **Gem spawning bug**: `gemGroups="2"` made getCenterGems search 1-gem subgroups and throw "Could not spawn a gem group!". Fixed by setting `gemGroups="0"`.
- **Phantom gem object** at fixed coords slipped through the gem filter. Fixed by using a positive `classname $= "Gem"` check.
- **Stuck episode dead zone**: restartLevel left `$Game::Running=true` briefly with stale timer. `$MLAgent::TimerStarted` guard skips the dead zone.
- **Short episode flood** from `gemCount >= maxGems` check in Hunt mode (cumulative score). Removed that check; timer + step cap suffice.
- **±π wrap discontinuity** in scalar angle action caused 1/4 of camera headings to systematically miss gems. Replaced with 2D isotropic Gaussian on (dx, dy) — no wrap.
- **Hardcoded LR override** was resetting LR to old values when resuming. Fixed to preserve optimizer state's LR.
- **Entropy runaway on 3D continuous actions**: any positive entropy_coef on Bernoulli jump pushed jump rate to 50%. Fixed by excluding jump from entropy bonus.
- **Jump head LayerNorm bug**: LayerNorm before jump_head washed out all useful signal because the features backbone had 2 dimensions dominated by raw time/score values (magnitude ~30k–48k), which swamped the per-sample normalization. Removed LayerNorm entirely; jump_head now consumes raw features.
- **Jump head positive-side sigmoid saturation (curriculum)**: jump_head weights grew large enough that logit pre-clamp was ~+12 on every input, gradients died, model output a constant ~100% jump regardless of input. First fix attempt: clamping logit to ±3 and reinitializing weights tiny. That prevented catastrophic saturation but did not solve the root cause.
- **Jump head negative-side sigmoid saturation (post-curriculum)**: After the clamp fix, feeding raw features (norm ~200k) into jump_head amplified every gradient step. 70% of rollout steps are on flat ground where jumping is slightly bad, so the weights drifted strongly negative until raw pre-clamp logit was around -98,907 for every input. Clamp held the logit at -3, so output was constant 4.74% and gradient couldn't flow. First fix attempt: `nn.BatchNorm1d(128)` (`jump_norm`) between features and jump_head. This did stop saturation, but didn't produce discrimination — the model still learned a uniform ~5% jump rate. Not because BN failed but because the shared features backbone doesn't encode the information the jump decision needs.
- **Jump decision representation mismatch**: The shared features backbone (59 → 128 → 128) is trained by gradients from all three heads, but direction/throttle gradients dominate (dense continuous rewards). Its 128-dim representation ends up optimized for steering ("unit vector to gem," "velocity along gem axis," "distance to gem"), not for the jump decision ("near a platform XY," "approaching an elevated gem," "haven't jumped yet"). Even with BN fixing the saturation, the jump_head couldn't extract a signal that wasn't in its input. Fixed by giving the jump pathway its own independent 2-layer MLP (`jump_features`) that takes the raw normalized observation directly, decoupled from the shared backbone. Also required **rebalancing the training map** from 7 ground + 3 elevated gems to 3 ground + 3 elevated (50/50) to improve the gradient signal ratio from ~8:1 against jumping to ~3:1.
- **Throttle clipped floor** preventing stops: tried removing via maturity gate; agent can already brake by pushing opposite to velocity, so no additional reward shaping is needed for deceleration.
- **Slow-pickup bonus** (+100 at low pickup speed, scaled): signal too sparse, agent never learned to brake. Removed.

## Design Principles We've Converged On

- **Never make code changes without explicit user approval** — present options and analysis, wait for a decision.
- **Verify both ends of data contracts** when a bug spans Python + TorqueScript.
- **Keep `analyze_log.py` in sync** when adding new training-log fields.
- **No Unicode in Python log output** — Windows cp1252 crashes on characters like →, ≈, ←. Use ASCII (`->`, `~`).
- **Camera is randomized per-episode** at training time for generalization; locked at inference for consistent play.

---

## Key Files

- `ml_agent/train_ppo.py` — PPO training server (port 8888)
- `ml_agent/dashboard.py` — this dashboard (port 8889, SSE + Plotly)
- `ml_agent/play.py` — inference-only server for watching the model play
- `ml_agent/analyze_log.py` — post-training log analysis (keep in sync with new log fields)
- `Marble Blast Platinum/platinum/client/scripts/ai/mlAgent.cs` — TorqueScript game loop
- `Marble Blast Platinum/platinum/client/scripts/ai/observer.cs` — obs collection
- `Marble Blast Platinum/platinum/client/scripts/ai/agent.cs` — low-level input handling
- `Marble Blast Platinum/platinum/data/multiplayer/hunt/custom/FlatWithJump_Hunt.mcs` — current training map
- `Marble Blast Platinum/platinum/data/multiplayer/hunt/custom/JumpOnly_Hunt.mcs` — jump-only curriculum map
- `Marble Blast Platinum/platinum/data/multiplayer/hunt/custom/FlatGemTraining_Hunt.mcs` — original flat map
