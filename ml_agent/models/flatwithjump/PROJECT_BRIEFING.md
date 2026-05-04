# FlatWithJump Project Briefing

**Saved**: 2026-05-03
**Reason**: pivoting model to KingOfTheMarble for transferable terrain training. Save context here so we can pick up the FlatWithJump brake-learning thread later.

---

## TL;DR — where we left off

The model trained on `FlatWithJump_Hunt` for ~49,060 PPO updates. It scores ~60-65 gems/game (peaked at 71) but **fundamentally cannot learn to brake**. The brake head fires at ~2-3% uniformly across the entire map, with no spatial or velocity discrimination. Reward shaping, γ extension to 0.999, and brake-head reset all failed to teach the conditional rule.

**The problem is not learning capacity, it's that on a flat map with sparse gems, "go full tilt at gem" is a stable local optimum that scores 64+ — the brake-aware policy (~102 gems) is unreachable from there via PPO exploration.**

User decided to switch to **KingOfTheMarble** (real terrain, slopes, smaller arena) so brake becomes load-bearing for survival, not just an optimization. Per memory rule: solutions must transfer to other maps. The flat-map oracle (`manual_brake.py`) and `target_speed = sqrt(2*A_MAX*d)` obs dim were rejected because they bake in flat-ground physics.

---

## Latest model state (update_49060)

- **Checkpoint**: `models/flatwithjump/checkpoints/update_49060.pth` (and earlier copies; 561 total)
- **Performance** (from log `training_20260503_195916.log`, last 3 games):
  - 47 / 42 / 34 gems (declining trend, best ever 69)
  - jump_rate: 3-7%
  - brake_rate: **2.0-2.2%** (matches sigmoid(-3) baseline post-reset; **never developed conditional structure**)
  - pickup_speed: 8.6-9.4 (high — model doesn't slow down for gems)
  - overshoot: 4-5% of steps
- **Critic** still adjusting to γ=0.999 change (value loss ranged 0.5 → 6.0 over the 999 updates after the change)
- **Entropy**: 1.6 (rising trend, model exploring but not finding brake)

## Brake heatmap result (the smoking gun)

Generated `brake_heatmap.html` via `generate_brake_heatmap.py`. With marble at every map cell moving full speed east at v=12:

```
Brake probability range: 2.53% to 3.01%   (across ENTIRE 50x50 map)
Approach (toward gem):   2.84%
Retreat  (away from gem): 2.79%
Approach − Retreat: +0.05pp        <- noise
Sensitivity (std across 8 dirs): 0.34pp   <- noise
```

The brake head has learned exactly nothing about position, velocity direction, or gem proximity. It's a state-blind random Bernoulli at ~2.8%.

---

## What was tried (in order)

| # | Intervention | Result |
|---|-------------|--------|
| 1 | Slow-pickup bonus (reward proportional to 1/speed at pickup) | Model didn't slow down; reward signal too weak |
| 2 | Increased gap penalty (k from 0.0008 → larger) | Model braked more globally then crashed performance |
| 3 | Decreased jump penalty | Model jumped more, didn't help brake |
| 4 | Hand-coded `manual_brake.py` controller | **Achieved 102.7 avg gems** at a_max=12, margin=0.00, min_target=0.05. Proved 102+ is reachable on this map. |
| 5 | Time-Optimal Control overshoot penalty (calibrated A_MAX=12 from manual_brake) | Mild improvement, brake rate still <5% |
| 6 | γ: 0.99 → **0.999** + brake_head reset (kaiming + bias=-3.0) | Best game improved 64→71, but **brake rate dropped from 4.7% baseline to 2.0%**. Critic relearning, brake didn't develop structure. |
| 7 | Brake heatmap probe | Confirmed brake head is state-blind everywhere. PPO failure to extract conditional structure. |

## What was NOT tried (rejected as non-transferable)

- **Behavior cloning from `manual_brake.py`** — bakes in flat-ground physics (`sqrt(2*A_MAX*d)` straight-line stopping), won't work on slopes
- **`target_speed` as obs dim** — same flat-ground assumption baked in
- **Higher brake baseline** (e.g., bias=0 → 50% brake rate) — would crash locomotion uniformly, advantages still go negative everywhere, equilibrium back at ~5% within 1k updates
- See `ml_agent/BRAKE_LEARNING_PROBLEM.md` for the full diagnosis written for outside consultation

## Key user constraint (in memory)

> "this needs to be FLEXIBLE AND TRANSFER TO OTHER MAPS"

→ saved as `feedback_must_transfer_to_other_maps.md`. Don't propose oracles, map-specific obs dims, or BC that assumes flat ground.

---

## Diagnostic finding: why PPO can't extract brake on flat training

The brake-aware policy (~102 gems) is *not reachable from* the brake-blind policy (~64 gems) via PPO exploration on this map because:

1. The brake-blind policy is a **stable local optimum** — going full tilt at gems scores 64. Stable.
2. To reach 102, the model needs the conditional rule "brake when fast AND close AND moving toward gem." That's 3+ features composed. Bernoulli sampling over 5% of frames doesn't generate enough correctly-conditioned brake events to extract it from sparse global reward.
3. Even with γ=0.999 (so future-gem cost factors in), the local optimum holds because random brakes hurt average performance more than they help.

**On a complex map**, simple full-tilt scoring stops working (marble OOBs constantly), so the brake-blind policy ceases to be a local optimum. PPO is then forced to find the conditional structure to survive.

That's the bet for KingOfTheMarble.

---

## Files saved here in `models/flatwithjump/`

```
checkpoints/         — 561 checkpoint .pth files (update_48510 .. update_49060)
                       update_49060.pth is the latest. Brake head reset at update_47530.
                       γ=0.999 change at update_47531.
code/                — snapshot of train_ppo.py, mlAgent.cs, observer.cs, agent.cs,
                       analyze_log.py, dashboard.py, play.py, manual_brake.py,
                       generate_jump_heatmap.py, generate_brake_heatmap.py,
                       FlatWithJump_Hunt.mcs (and .dso) — exact code at this point
PROJECT_BRIEFING.md  — this document
```

## Critical config at point of save

```python
# train_ppo.py — current values
self.gamma            = 0.999     # changed from 0.99 at update 47531
self.A_MAX            = 12.0      # calibrated from manual_brake auto-tuner
self.GEM_REWARD       = 200
self.GEM_GAP_PENALTY_K = 0.0008
self.JUMP_COST        = 0.3
self.OVERSHOOT_LAMBDA = 0.5
self.SLOW_PICKUP_BONUS = 0        # disabled
self.actor_lr         = 1.5e-5
self.critic_lr        = 5e-5
self.entropy_coef     = 0.005
self.rollout          = 2048
self.batch            = 256
self.target_kl        = 2.667
self.THROTTLE_FLOOR   = 0.15
```

Action space: 5-tuple `(dx, dy, throttle_logit, jump_binary, brake_binary)`.
Brake `brake_head.bias = -3.0` (~5% baseline). Init reset at update 47530.
Obs dim: 59 (35 base + 24 frame history, 4 frames at 8-frame intervals).

---

## When we come back to this thread

The most likely scenarios for returning:

### Scenario A: KingOfTheMarble training succeeds
The transferred policy learned brake on the complex map. Now we want to test whether that brake skill transfers BACK to FlatWithJump and improves the 64→102 gap.

**Plan:** load the KingOfTheMarble checkpoint, fine-tune briefly on FlatWithJump (or just probe), measure gems/game. If it crosses 80+, the multi-map approach is validated.

### Scenario B: KingOfTheMarble training fails to develop brake
Even on complex terrain, brake stays at ~2-3%. This means the problem is even deeper than local optima — it's a representation/exploration problem at the architecture level.

**Next interventions to consider** (in order of preference):
1. **Architecture upgrade**: replace MLP brake_features with a small recurrent core (GRU). Frame history helps but explicit recurrence lets the model integrate longer-term motion.
2. **Curriculum on multiple complex maps**: not just KingOfTheMarble. Force invariance across diverse physics.
3. **Add generic kinematic look-ahead obs**: predicted_pos[t+5/10/20] derived from current velocity + frame-history acceleration. Map-agnostic; just gives the model "where am I going" directly. (Acceptable per transfer rule — pure kinematics, no map facts.)

### Scenario C: User wants to revisit the brake reset / γ choice
Detailed history in this doc. The γ change worked (extends horizon ~10×), the brake reset re-enabled exploration but didn't break the local optimum. Don't repeat both — try architecture or look-ahead obs instead.

---

## Things to check first when resuming

1. `git log` since 2026-05-03 to see what changed in code
2. Latest log file in `ml_agent/logs/` — what's the current training state?
3. Run `analyze_log.py` on the latest log
4. Run `generate_brake_heatmap.py` on the current checkpoint to see if brake structure has emerged
5. Compare current `train_ppo.py` config to the snapshot in `code/` — any hyperparameter drift?

## Things NOT to do without re-reading this

- Don't reset brake head again. Done twice, doesn't help.
- Don't suggest BC from manual_brake or `target_speed` obs dim. Both are flat-only.
- Don't lower γ from 0.999 unless training diverges hard.
- Don't propose more reward shaping for brake before trying architecture/curriculum changes.
