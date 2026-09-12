"""
PPO Training Server for PlatinumQuest Hunt Mode

Receives observations from the game via TCP socket, computes rewards
in Python, runs PPO training updates, and sends actions back.

Protocol (game -> server):
    Each message is newline-delimited:
    "[obs_array]|gem_delta|oob|done"

    - obs_array: 35-float JSON array (see observer.cs for layout; Python appends 24 frame-history dims -> 59 total)
    - gem_delta: gem points collected this step (0 or positive int)
    - oob:       1 if out-of-bounds this step, 0 otherwise
    - done:      1 if episode ended, 0 otherwise

    Game-end signal: "[]|<neg_total_gems>|0|1"

Protocol (server -> game):
    "0.87,0.0,0.0,0.50\n"  (4 comma-separated float actions: F,B,L,R)

Usage:
    python train_ppo.py
    python train_ppo.py --load models/checkpoint_1000.pth
"""

import socket
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions
import time
import signal
import sys
import os
import math
from collections import deque
from datetime import datetime
from dashboard import DashboardServer

# ============================================================================
# Logging Helper
# ============================================================================

class DualLogger:
    """Prints to both console and file."""
    def __init__(self, log_file):
        self.log_file = log_file
        self.file = open(log_file, 'w', buffering=1)  # Line buffered

    def print(self, *args, **kwargs):
        """Print to both console and file."""
        message = ' '.join(map(str, args))
        print(message, **kwargs)  # To console
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.file.write(f"[{timestamp}] {message}\n")
        self.file.flush()

    def close(self):
        self.file.close()

# ============================================================================
# Neural Networks (Separate Actor and Critic)
# ============================================================================

class Actor(nn.Module):
    """Policy network for PPO with continuous 2D direction + throttle + binary jump.

    Action: (dx, dy, throttle, jump) where:
      - (dx, dy): unit-circle direction vector (0,1)=forward, (1,0)=right
      - throttle: [0.15, 1] force magnitude (floor prevents sit-still collapse)
      - jump: binary (0 or 1), sampled from Bernoulli
    Buffer stores (dx, dy, throttle_logit, jump_binary) — 4 dims.
    2D representation eliminates the scalar angle wrap discontinuity at +/-pi
    that caused systematic failures in 1/4 of camera headings.

    Separate from Critic to eliminate gradient interference: critic's high-variance
    value loss no longer contaminates policy gradient through shared weights.
    """

    LOG_STD_MIN = -2.0    # exp(-2.0) ~= 0.14 (tight directional spread)
    LOG_STD_MAX = -0.5    # exp(-0.5) ~= 0.61 rad ~= 35 deg (tightened from 1.0).
                          # On KingOfTheMarble the policy was drifting to ~50 deg
                          # spread despite entropy_coef reductions (0.005->0.002->0.001).
                          # Reward gradient itself was pushing log_std up — entropy_coef
                          # wasn't the lever. This is an architectural ceiling: forces
                          # log_std clamp at 35 deg, recovering navigation precision
                          # while keeping enough exploration to maintain brake learning.
    THROTTLE_LOG_STD_MIN = -3.0   # exp(-3.0) ~= 0.05 (tight throttle control)
    THROTTLE_FLOOR = 0.15         # Minimum throttle (disabled after model matures)
    THROTTLE_LOG_STD_MAX = -0.9   # exp(-0.9) ~= 0.41 (tightened from 0.0/=1.0).
                                  # Same reason: throttle stddev was creeping up
                                  # (0.231 -> 0.318) and contributing to runaway.
                                  # Cap at ~0.4 keeps reasonable headroom above
                                  # current 0.32 but prevents further widening.

    def __init__(self, obs_dim=61):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Outputs (mean_x, mean_y) as a unit-circle direction.
        # 2D output avoids the 0/2pi wraparound discontinuity.
        # Tanh bounds output to [-1,1] to prevent magnitude explosion:
        # without it, weights grow unbounded (norm reached 26+) since
        # normalization makes the loss magnitude-invariant, killing
        # gradient signal and collapsing the policy to a fixed direction.
        self.actor_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # (dx, dy) normalized to unit circle
            nn.Tanh(),
        )
        # Initialize final layer with small weights so tanh starts in its
        # linear region (near zero). Default init saturates tanh at +/-1,
        # killing gradient and making output input-independent.
        nn.init.uniform_(self.actor_mean[2].weight, -0.01, 0.01)
        nn.init.zeros_(self.actor_mean[2].bias)

        # Throttle head: outputs raw logit, passed through sigmoid -> [0, 1]
        self.throttle_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # raw logit -> sigmoid -> throttle
        )
        # Initialize throttle bias to +2.0 so sigmoid(2.0) ~= 0.88 (mostly full throttle)
        # This prevents the model from starting with 50% throttle which would halve gem collection
        nn.init.constant_(self.throttle_head[-1].bias, 2.0)

        # Jump pathway has its OWN feature extractor, decoupled from the shared
        # features backbone used by direction/throttle. The shared backbone is
        # dominated by gradients from direction/throttle (dense, continuous rewards)
        # and its 128-dim representation ends up optimized for steering decisions —
        # not for "am I near a platform, is this gem elevated, should I be airborne".
        # Giving jump its own 2-layer MLP lets it learn its own internal features
        # specifically tuned to the jump decision. Adds ~6K params, no impact on
        # direction/throttle quality.
        self.jump_features = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.jump_head = nn.Linear(32, 1)
        nn.init.constant_(self.jump_head.bias, -1.0)  # sigmoid(-1) ~= 27% starting

        # Brake pathway: explicit "decelerate now" action that overrides direction
        # to anti-velocity at full throttle for one frame. Same architectural pattern
        # as jump (independent feature MLP + Bernoulli head). Reason it's a separate
        # discrete action rather than relying on direction-head exploration: the
        # direction Gaussian (PolicyStd ~30°) puts ~0.003% probability on samples
        # 180° from gem direction, so the model never discovers braking via random
        # direction exploration. A Bernoulli on a single bit gives natural ~5%
        # baseline exploration via bias=-3 init.
        self.brake_features = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.brake_head = nn.Linear(32, 1)
        nn.init.constant_(self.brake_head.bias, -3.0)  # sigmoid(-3) ~= 4.7% baseline brake rate

        # Learnable log-std (state-independent, single scalar for angular spread)
        # Init to -1.5 -> exp(-1.5) ~= 0.22 rad ~= 12.8 deg
        self.log_std = nn.Parameter(torch.full((1,), -1.5))
        # Throttle log-std: init to -1.0 -> exp(-1.0) ~= 0.37 (moderate exploration)
        self.throttle_log_std = nn.Parameter(torch.full((1,), -1.0))

    def _get_direction_dist(self, mean_xy):
        """Get 2D isotropic Gaussian over (dx, dy) direction vectors.

        The mean is normalized to the unit circle. Both dx and dy share
        the same log_std, giving isotropic spread around the mean direction.
        No wrap discontinuity since we never convert to/from scalar angle.
        """
        # Normalize mean to unit circle
        norm = torch.norm(mean_xy, dim=1, keepdim=True).clamp(min=1e-6)
        mean_unit = mean_xy / norm
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp().expand_as(mean_unit)  # same std for both dims
        return torch.distributions.Normal(mean_unit, std)

    def _get_throttle_dist(self, throttle_mean):
        """Get throttle distribution in logit space (unbounded Normal)."""
        log_std = torch.clamp(self.throttle_log_std, self.THROTTLE_LOG_STD_MIN, self.THROTTLE_LOG_STD_MAX)
        return torch.distributions.Normal(throttle_mean.squeeze(-1), log_std.exp())

    def forward(self, state):
        features = self.features(state)
        mean_xy = self.actor_mean(features)
        throttle_logit = self.throttle_head(features)
        # Jump uses its own features extractor on raw (normalized) obs.
        jump_logit = self.jump_head(self.jump_features(state))
        # Asymmetric clamp. Negative side opened up to -7 (sigmoid = 0.09%) so the
        # JUMP_COST gradient can actually drive flat-ground jump prob below 1%.
        # Previous symmetric clamp at -3 (= 4.74% floor) was eating the suppression
        # gradient and making any JUMP_COST irrelevant on flat ground. Positive side
        # stays at +3 (= 95.3%) to prevent overconfidence at platforms.
        jump_logit = torch.clamp(jump_logit, -7.0, 3.0)
        # Brake: same pattern, same clamp. Allows model to suppress unneeded brakes
        # toward 0.09% on flat travel while committing strongly (95.3%) at slowdown moments.
        brake_logit = self.brake_head(self.brake_features(state))
        brake_logit = torch.clamp(brake_logit, -7.0, 3.0)
        return mean_xy, throttle_logit, jump_logit, brake_logit

    def get_action(self, state, deterministic=False):
        """Get action from state.

        Returns:
            action_for_buffer: (dx, dy, throttle_logit, jump_binary, brake_binary) - stored in buffer
            action_for_game: (dx, dy, throttle, jump_binary, brake_binary) - passed to action_to_joystick
            log_prob: joint log probability (direction + throttle + jump + brake)
        """
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0)
            mean_xy, throttle_logit, jump_logit, brake_logit = self.forward(state)

            # Direction (2D)
            dir_dist = self._get_direction_dist(mean_xy)
            if deterministic:
                norm = torch.norm(mean_xy, dim=1, keepdim=True).clamp(min=1e-6)
                direction = mean_xy / norm  # (1, 2)
            else:
                direction = dir_dist.sample()  # (1, 2)
            dir_log_prob = dir_dist.log_prob(direction).sum(dim=1)

            # Throttle (sample in logit space, then sigmoid)
            throttle_dist = self._get_throttle_dist(throttle_logit)
            if deterministic:
                throttle_logit_sample = throttle_logit.squeeze(-1)
            else:
                throttle_logit_sample = throttle_dist.sample()
            throttle_log_prob = throttle_dist.log_prob(throttle_logit_sample)
            throttle_raw = torch.sigmoid(throttle_logit_sample)
            throttle = self.THROTTLE_FLOOR + (1 - self.THROTTLE_FLOOR) * throttle_raw

            # Jump (Bernoulli from logit)
            jump_prob = torch.sigmoid(jump_logit.squeeze(-1))
            jump_dist = torch.distributions.Bernoulli(probs=jump_prob)
            if deterministic:
                jump_action = (jump_prob > 0.5).float()
            else:
                jump_action = jump_dist.sample()
            jump_log_prob = jump_dist.log_prob(jump_action)

            # Brake (Bernoulli from logit) — same pattern as jump
            brake_prob = torch.sigmoid(brake_logit.squeeze(-1))
            brake_dist = torch.distributions.Bernoulli(probs=brake_prob)
            if deterministic:
                brake_action = (brake_prob > 0.5).float()
            else:
                brake_action = brake_dist.sample()
            brake_log_prob = brake_dist.log_prob(brake_action)

            # Joint log prob includes all four heads
            log_prob = dir_log_prob + throttle_log_prob + jump_log_prob + brake_log_prob

        dx_val = direction[0, 0].item()
        dy_val = direction[0, 1].item()
        throttle_logit_val = throttle_logit_sample.item()
        throttle_val = throttle.item()
        jump_val = jump_action.item()
        brake_val = brake_action.item()
        # Buffer stores binary jump/brake actions (not logits) since Bernoulli log_prob needs 0/1
        return (dx_val, dy_val, throttle_logit_val, jump_val, brake_val), \
               (dx_val, dy_val, throttle_val, jump_val, brake_val), \
               log_prob.item()

    def evaluate_actions(self, states, actions):
        """Evaluate log_probs and entropy for stored actions.
        actions: (N, 5) tensor with columns [dx, dy, throttle_logit, jump_binary, brake_binary]."""
        mean_xy, throttle_logit, jump_logit, brake_logit = self.forward(states)

        # Direction (2D)
        dir_dist = self._get_direction_dist(mean_xy)
        dir_log_prob = dir_dist.log_prob(actions[:, 0:2]).sum(dim=1)
        dir_entropy = dir_dist.entropy().sum(dim=1)

        # Throttle (actions[:, 2] is stored in logit space)
        throttle_dist = self._get_throttle_dist(throttle_logit)
        throttle_log_prob = throttle_dist.log_prob(actions[:, 2])
        throttle_entropy = throttle_dist.entropy()

        # Jump (actions[:, 3] is the binary 0/1 action)
        jump_prob = torch.sigmoid(jump_logit.squeeze(-1))
        jump_dist = torch.distributions.Bernoulli(probs=jump_prob)
        jump_binary = actions[:, 3]
        jump_log_prob = jump_dist.log_prob(jump_binary)

        # Brake (actions[:, 4] is the binary 0/1 action)
        brake_prob = torch.sigmoid(brake_logit.squeeze(-1))
        brake_dist = torch.distributions.Bernoulli(probs=brake_prob)
        brake_binary = actions[:, 4]
        brake_log_prob = brake_dist.log_prob(brake_binary)

        # Joint log_prob includes ALL heads. Entropy EXCLUDES jump and brake —
        # Bernoulli entropy bonus pushes binary actions toward 50% probability,
        # which would cause constant jumping/braking. Jump and brake learn purely
        # from reward signal.
        log_prob_total = dir_log_prob + throttle_log_prob + jump_log_prob + brake_log_prob
        entropy_total = dir_entropy + throttle_entropy
        return log_prob_total, entropy_total

    @staticmethod
    def action_to_joystick(dx, dy, throttle, jump=0, brake=0, vx=0.0, vy=0.0):
        """Convert (dx, dy, throttle, jump, brake) action to joystick axes.

        Convention: dx>0 = right, dy>0 = forward.
        Direction is normalized to unit length; throttle scales the magnitude.

        When brake=1, the direction is OVERRIDDEN to anti-velocity at full throttle:
        the joystick points exactly opposite the marble's current 2D velocity (vx, vy
        in camera-relative frame), applying a maximum decelerating force for that frame.
        Falls back to no-op if the marble is already nearly stopped (|v_xy| < 0.1).

        The brake field is NOT sent in the wire protocol — its effect is already baked
        into fwd/back/left/right. The 5th return value remains `jump` to match the
        existing CS-side joystick handler that expects (fwd, back, left, right, jump).
        """
        import math
        if brake > 0.5:
            speed_xy = math.sqrt(vx * vx + vy * vy)
            if speed_xy > 0.1:
                dx = -vx / speed_xy
                dy = -vy / speed_xy
                throttle = 1.0
            # else: marble basically stopped — brake is no-op, fall through to normal handling
        else:
            norm = math.sqrt(dx * dx + dy * dy)
            if norm > 1e-6:
                dx = dx / norm
                dy = dy / norm

        move_x = dx * throttle  # positive = right
        move_y = dy * throttle  # positive = forward

        fwd   = round(max(move_y, 0.0), 6) + 0.0
        back  = round(max(-move_y, 0.0), 6) + 0.0
        right = round(max(move_x, 0.0), 6) + 0.0
        left  = round(max(-move_x, 0.0), 6) + 0.0

        return fwd, back, left, right, int(jump)


class Critic(nn.Module):
    """Value network for PPO. Separate from Actor to prevent gradient interference.

    The critic's value loss is high-variance (reward spikes from gem collection),
    so keeping it separate ensures its gradients never corrupt the policy weights.
    Uses a higher learning rate than the actor.

    Value-target normalization (PopArt, van Hasselt et al. 2016). The last layer
    predicts a NORMALIZED value v_n; the real value is
        V(s) = v_n * value_std + value_mean
    Running (mean, std) of the GAE returns are refreshed once per rollout.
    Whenever the stats change, the last layer's weight and bias are rescaled so
    that every V(s) is exactly preserved (output-preserving update) -- only the
    loss scale changes. This keeps the regression target O(1) regardless of
    gamma, reward scale or map, which is what previously forced critic_lr down
    to 1e-5 and a 0.3 grad clip (VL spiked to 27, CGN to 472 on the
    KingOfTheMarble switch). The stats are registered buffers, so they are
    saved in the checkpoint; checkpoints that predate them load with
    mean=0, std=1 (identity), i.e. the loaded critic behaves exactly as before
    until the first rollout initializes the stats.
    """

    POPART_BETA = 0.05      # per-rollout EMA rate for the running stats (~20-rollout memory)
    POPART_MIN_STD = 1e-2   # floor on value_std (degenerate all-equal returns)

    def __init__(self, obs_dim=61):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # PopArt running statistics of the value targets, in raw-scaled reward units.
        self.register_buffer('value_mean', torch.zeros(1))
        self.register_buffer('value_std', torch.ones(1))
        self.register_buffer('value_sq_mean', torch.ones(1))          # running E[G^2]
        self.register_buffer('value_stats_initialized', torch.zeros(1))  # 0 until first rollout

    def forward_normalized(self, state):
        """Raw last-layer output: the normalized value v_n. Used only by the value loss."""
        return self.net(state)

    def forward(self, state):
        """Real (denormalized) value V(s). Used by GAE, bootstrapping and all diagnostics."""
        return self.net(state) * self.value_std + self.value_mean

    def get_value(self, state):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0)
            return self.forward(state).item()

    @torch.no_grad()
    def update_value_stats(self, returns):
        """Refresh the running mean/std of the value targets and rescale the output
        layer so every V(s) prediction is unchanged (PopArt).

        Args:
            returns: 1-D tensor of GAE returns for this rollout (raw-scaled units).
        """
        old_mean = self.value_mean.clone()
        old_std = self.value_std.clone()

        batch_mean = returns.mean()
        batch_sq = (returns ** 2).mean()
        if self.value_stats_initialized.item() < 0.5:
            # First rollout: adopt the batch statistics directly rather than
            # crawling away from the (0, 1) placeholder via the EMA.
            self.value_mean.copy_(batch_mean.reshape(1))
            self.value_sq_mean.copy_(batch_sq.reshape(1))
            self.value_stats_initialized.fill_(1.0)
        else:
            b = self.POPART_BETA
            self.value_mean.mul_(1 - b).add_(b * batch_mean)
            self.value_sq_mean.mul_(1 - b).add_(b * batch_sq)
        var = (self.value_sq_mean - self.value_mean ** 2).clamp(min=0.0)
        self.value_std.copy_(var.sqrt().clamp(min=self.POPART_MIN_STD))

        # Output-preserving rescale of the final Linear(64, 1):
        #   v_n_new * std_new + mean_new  ==  v_n_old * std_old + mean_old
        out = self.net[-1]
        scale = old_std / self.value_std
        out.weight.mul_(scale)
        out.bias.mul_(scale).add_((old_mean - self.value_mean) / self.value_std)


# ============================================================================
# Experience Buffer
# ============================================================================

class RolloutBuffer:
    """Stores experience for PPO updates."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def compute_returns_and_advantages(self, gamma=0.99, lam=0.95, last_value=0.0):
        """Compute GAE advantages and discounted returns.

        last_value: V(s_{T+1}), the critic's estimate for the observation that
        follows the final stored transition. The buffer is cut every
        rollout_size decisions in the MIDDLE of an episode (truncation, not
        termination), so the last step must bootstrap from it. The old code
        used 0 here, which gave the final transition a TD error of roughly
        -V(s) and propagated a spurious negative advantage back ~50 steps
        into every single rollout. Ignored when dones[-1] is True.
        """
        advantages = []
        returns = []
        gae = 0

        # Work backwards through experience
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = last_value  # truncation: bootstrap from V(s_{T+1})
            else:
                next_value = self.values[t + 1]

            if self.dones[t]:
                next_value = 0
                gae = 0

            delta = self.rewards[t] + gamma * next_value - self.values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])

        return returns, advantages

    def get_batches(self, batch_size, gamma=0.99, lam=0.95, last_value=0.0,
                    precomputed=None):
        """Get training batches with computed advantages.

        precomputed: optional (returns, advantages) from
        compute_returns_and_advantages(), so the update can compute GAE once,
        use the returns for value-target normalization, and reuse them here.
        """
        if precomputed is not None:
            returns, advantages = precomputed
        else:
            returns, advantages = self.compute_returns_and_advantages(gamma, lam, last_value)

        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))  # (N, 5): [dx, dy, throttle_logit, jump_binary, brake_binary]
        old_log_probs = torch.FloatTensor(self.log_probs)
        old_values = torch.FloatTensor(self.values)
        returns_t = torch.FloatTensor(returns)
        advantages_t = torch.FloatTensor(advantages)

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Generate random mini-batches
        n_samples = len(self.states)
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            yield (
                states[batch_idx],
                actions[batch_idx],
                old_log_probs[batch_idx],
                returns_t[batch_idx],
                advantages_t[batch_idx],
                old_values[batch_idx],
            )

    def __len__(self):
        return len(self.states)


# ============================================================================
# PPO Trainer
# ============================================================================

class PPOTrainer:
    """Proximal Policy Optimization trainer with separate actor/critic optimizers.

    Actor and critic are trained independently with separate Adam optimizers,
    eliminating gradient interference between the policy and value objectives.
    The critic can use a higher LR since its loss is a simple regression target,
    while the actor uses a lower LR for stable policy updates.
    """

    def __init__(self, actor, critic,
                 actor_lr=3e-5, critic_lr=1e-4,
                 clip_epsilon=0.2, entropy_coef=0.01,
                 max_grad_norm=1.0, vf_clip=20.0, target_kl=1.5,
                 critic_max_grad_norm=None):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        # The critic gets its own clip. Sharing the actor's 0.3 clip against a
        # critic grad norm of ~100 meant clipping by 300x on every minibatch.
        self.critic_max_grad_norm = critic_max_grad_norm if critic_max_grad_norm is not None else max_grad_norm
        self.vf_clip = vf_clip   # in raw-scaled reward units; converted to normalized units per update
        self.target_kl = target_kl  # KL early stopping: fires at 1.5x (catches destructive updates)

    def update(self, buffer, n_epochs=4, batch_size=64, gamma=0.99, lam=0.95,
               freeze_actor=False, last_value=0.0):
        """Run PPO update with independent actor and critic steps.

        When freeze_actor=True (warmup phase), the actor's policy weights are
        not updated — only the critic is trained. KL/grad_norm are still
        computed for logging. Used after a map switch to let the critic
        relearn V(s) before the actor moves on potentially-garbage advantages.

        last_value: V(s_{T+1}) for bootstrapping the truncated rollout (see
        RolloutBuffer.compute_returns_and_advantages).
        """
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_grad_norm = 0
        total_critic_grad_norm = 0
        n_updates = 0
        max_kl = 0.0
        kl_early_stopped = False

        # GAE once per update (values are fixed for the whole update), then
        # refresh the PopArt value-target statistics from this rollout's returns
        # BEFORE any gradient step. The refresh is output-preserving, so the
        # buffer's stored old_values are still exact predictions of V(s).
        precomputed = buffer.compute_returns_and_advantages(gamma, lam, last_value)
        self.critic.update_value_stats(torch.FloatTensor(precomputed[0]))
        v_mean = self.critic.value_mean.item()
        v_std = self.critic.value_std.item()
        vf_clip_n = self.vf_clip / v_std

        for epoch in range(n_epochs):
            if kl_early_stopped:
                break
            for states, actions, old_log_probs, returns, advantages, old_values in \
                    buffer.get_batches(batch_size, gamma, lam, last_value, precomputed=precomputed):

                # --- Actor update ---
                new_log_probs, entropy = self.actor.evaluate_actions(states, actions)

                # KL early stopping: measures policy drift from start of this update
                with torch.no_grad():
                    log_ratio = new_log_probs - old_log_probs
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                max_kl = max(max_kl, approx_kl)
                if approx_kl > 1.5 * self.target_kl:
                    kl_early_stopped = True
                    break

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                actor_loss = policy_loss + self.entropy_coef * entropy_loss

                if not freeze_actor:
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    grad_norm = sum(
                        p.grad.norm().item() ** 2
                        for p in self.actor.parameters()
                        if p.grad is not None
                    ) ** 0.5
                    total_grad_norm += grad_norm
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()

                # --- Critic update (regression in PopArt-normalized units) ---
                values_n = self.critic.forward_normalized(states).squeeze(-1)
                returns_n = (returns - v_mean) / v_std
                old_values_n = (old_values - v_mean) / v_std
                values_clipped_n = old_values_n + torch.clamp(values_n - old_values_n, -vf_clip_n, vf_clip_n)
                vf_loss1 = (values_n - returns_n) ** 2
                vf_loss2 = (values_clipped_n - returns_n) ** 2
                value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                critic_grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in self.critic.parameters()
                    if p.grad is not None
                ) ** 0.5
                total_critic_grad_norm += critic_grad_norm
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_max_grad_norm)
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'grad_norm': total_grad_norm / max(n_updates, 1),
            'critic_grad_norm': total_critic_grad_norm / max(n_updates, 1),
            'kl_early_stopped': kl_early_stopped,
            'max_kl': max_kl,
            'value_std': v_std,    # PopArt scale: VL above is in units of this
            'value_mean': v_mean,
        }


# ============================================================================
# Training Server
# ============================================================================

class PPOServer:
    """TCP server that trains the model while communicating with the game."""

    def __init__(self, host='127.0.0.1', port=8888, model_path=None):
        self.host = host
        self.port = port
        self.running = True

        # Setup dual logging (console + file)
        os.makedirs('logs', exist_ok=True)
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"logs/training_{self.run_timestamp}.log"
        self.logger = DualLogger(log_filename)
        self.log = self.logger.print  # Shortcut

        # Frame history: feed historical position+velocity to give the model
        # trajectory information (acceleration, jerk). Each historical frame is
        # 6 dims (pos xyz + vel xyz), taken FRAME_SKIP steps apart.
        self.FRAME_HISTORY_COUNT = 4   # number of historical snapshots
        self.FRAME_SKIP = 8            # frames between each snapshot
        self.FRAME_HISTORY_DIMS = 6    # pos(3) + vel(3) per frame
        self.obs_dim_base = 35         # base obs dimensions from game
        self.obs_dim = self.obs_dim_base + self.FRAME_HISTORY_COUNT * self.FRAME_HISTORY_DIMS  # 59

        # Ring buffer of recent normalized obs (pos+vel only, indices 0:6).
        # Need current frame + FRAME_HISTORY_COUNT * FRAME_SKIP historical frames.
        self.frame_history_size = self.FRAME_HISTORY_COUNT * self.FRAME_SKIP + 1  # 33
        self.frame_history = deque(maxlen=self.frame_history_size)

        # Separate actor and critic networks — eliminates gradient interference.
        # The critic's high-variance value loss no longer corrupts policy weights.
        # Actor LR is lower (3e-5) for stable policy updates; critic LR higher (1e-4)
        # since value regression can afford bigger steps.
        self.actor = Actor(obs_dim=self.obs_dim)
        self.critic = Critic(obs_dim=self.obs_dim)
        # Trust region tightened for cross-map transfer (KingOfTheMarble switch):
        # target_kl 2.667->1.0 (early-stops at 1.5 instead of 4.0, catches bad updates 3x sooner),
        # clip_epsilon 0.2->0.1 (smaller policy ratio bounds), max_grad_norm 1.0->0.3
        # (3x tighter gradient clip). Combined with the critic warmup below, this
        # prevents grad-norm explosions when the critic's V(s) is wrong on a new map.
        #
        # critic_lr 1e-4 -> 1e-5: first KingOfTheMarble warmup attempt showed the
        # critic itself oscillating wildly (VL spiked 4 -> 27 -> 6 -> 16 -> ..., CGN hit 472
        # mid-warmup). The reward distribution on this map has much higher variance
        # than flat training (frequent OOBs + sparse gems = bimodal returns), and the
        # critic_lr was too high to stably absorb it. 10x lower critic_lr lets the
        # critic settle without exploding even with high target variance.
        # 2026-09-11: critic un-throttled. critic_lr back to 1e-4 and its own
        # grad clip of 5.0 (actor keeps 0.3). This is safe now because the value
        # targets are PopArt-normalized inside PPOTrainer/Critic, which removes
        # the target-variance problem that the 1e-5 / 0.3 throttling was
        # papering over. Expect VL to read ~0.1-1.0 (normalized units; multiply
        # by Vstd^2 from the Upd line for raw-scaled) and CGN to drop from ~100
        # to O(1).
        self.trainer = PPOTrainer(self.actor, self.critic, vf_clip=20.0,
                                   critic_lr=1e-4, critic_max_grad_norm=5.0,
                                   target_kl=1.0, clip_epsilon=0.1,
                                   max_grad_norm=0.3, entropy_coef=-0.001)
        # entropy_coef history on KingOfTheMarble:
        #   0.005: full runaway (Ent 2.15 -> 2.82 over 2854 updates, PolicyStd 30->50 deg)
        #   0.002: runaway slowed but not reversed (Ent 2.83 -> 2.89 over 940 updates)
        #   0.001: still rising slowly even after architectural log_std cap at -0.5.
        #          Throttle stddev was creeping up (0.318 -> 0.331). Diagnosis: reward
        #          gradient is correlating wider sampling with higher reward (mean
        #          direction policy is poorly calibrated for KOTM, so wide sampling
        #          finds gems by luck) — entropy bonus was on top of that, no
        #          downward pressure existed at all.
        #   -0.001: SIGN FLIPPED. Loss now PENALIZES entropy instead of rewarding it.
        #          Creates active downward pressure to balance the reward-gradient
        #          push upward. Watch: entropy should start trending DOWN. If it
        #          keeps rising even with this, reward gradient pressure is too
        #          strong — escalate to -0.002 or implement target-entropy
        #          (SAC-style adaptive). If entropy collapses below ~1.0 fast,
        #          back off to -0.0005 — too much pressure killing exploration.
        self.buffer = RolloutBuffer()

        # Critic warmup: when transitioning to a new map, the critic's V(s)
        # predictions are wildly wrong for the new state distribution. PPO
        # advantages = R - V(s) become garbage, the actor moves in bad directions,
        # grad norms explode. Solution: freeze the actor for the first
        # WARMUP_ROLLOUTS rollouts so the critic can relearn V(s) before the
        # policy starts moving on noisy gradients.
        self.WARMUP_ROLLOUTS = 75   # was 300 at one tick per decision; 75 covers
                                    # the same game time under ACTION_REPEAT=4.
                                    # Force a fresh warmup with --warmup N.
        self.warmup_start_update = None  # set on first PPO update; persists in checkpoint

        # Training config
        # rollout_size counts DECISIONS (see ACTION_REPEAT below). 2048 decisions
        # = 8192 ticks ~= 131 s of game time, ~70% of a 3-min KOTM round, so a
        # rollout now contains ~20 gem events instead of ~5.
        self.rollout_size = 2048
        self.n_epochs = 4
        self.batch_size = 256  # 2048/256 = 8 mini-batches per epoch
        self.gamma = 0.996       # Per DECISION. With ACTION_REPEAT=4 one decision spans 4 ticks, and
                                 # 0.996 ~= 0.999^4, so the effective horizon is unchanged at
                                 # ~1000 ticks / 16 s of game time. Original note (per-tick 0.999):
                                 # bumped from 0.99 so the next gem (~290 ticks away) keeps ~75% weight.
        self.lam = 0.95
        self.save_interval = 10  # Save every N updates

        # Mild reward scaling: 0.01 crushed all signal (VLoss=0.0001, GradNorm=0.06),
        # 1.0 caused wild VLoss spikes (163.6) and gradient clipping threw away 99%
        # of gradient info. 0.1 is the sweet spot: gem=+20, OOB=-2.5, shaping=±0.05-0.3/step.
        # Per-step rewards are clipped to [-5, 5] after scaling to prevent VLoss explosions.
        self.reward_scale = 0.1
        # Clip on the SCALED per-decision reward. Was +/-20, which is exactly a
        # 1-point gem (200 * 0.1): a 2-point gem (40) and a 5-point gem (100)
        # were being clipped down to 20, so the policy was never told that gem
        # value exists. +/-100 admits a 5-point gem unclipped. The value-target
        # normalization in the critic absorbs the larger spikes.
        self.REWARD_CLIP = 100.0

        # === Action repeat (frame skip) ===
        # One decision is held for ACTION_REPEAT consecutive game ticks (16 ms
        # each). Python still receives and processes EVERY tick: rewards are
        # computed per tick and summed into the held decision's window, frame
        # history and all diagnostics stay tick-based, and the held action is
        # re-aimed against the current velocity each tick so a held brake keeps
        # pointing anti-velocity. Only the buffer sees one transition per
        # decision. Why: at one tick per decision the GAE credit reaching a
        # decision 50 ticks before a fall was 0.949^50 = 7%; at 4 ticks per
        # decision it is 0.946^12.5 = 50%. Per-tick Gaussian noise was also
        # averaged away by the marble's inertia; a held sample actually moves
        # the trajectory. Purely temporal, no map content.
        self.ACTION_REPEAT = 4

        # === Reward parameters (all reward computation is in Python) ===
        self.GEM_REWARD = 200       # Per gem-point bonus (+200 raw, +20 scaled)
        # self.OOB_PENALTY_INIT = 10   # Starting OOB penalty (gentle for early training)
        # self.OOB_PENALTY_FULL = 25   # Full OOB penalty (activated at 30 avg gems/game)
        # self.OOB_PENALTY = self.OOB_PENALTY_INIT
        self.OOB_PENALTY = 25   # SMALL terminal event marker. The capped FREE_FALL
                                # penalty (5/step max, ~100 frames -> -500 raw) is the
                                # primary signal now. Keeping OOB at the original
                                # baseline -25 gives the critic a clear "trajectory
                                # ended badly" punctuation without re-triggering the
                                # panic-brake response we got at -50/-100. Total cost
                                # of a typical fall: ~500 (freefall) + 25 (OOB) = 525,
                                # roughly 2.5 gems worth — strong enough to matter,
                                # weak enough not to drown gem-seeking.

        # Free-fall penalty: dense per-step cost when vz indicates falling. The
        # model already has vz in obs (index 5) and frame history of vz over the
        # last 32 frames. Free-fall is detectable from vz alone, on any map:
        # bouncing on a normal surface keeps |vz| small (around -7), while
        # falling off any platform produces strong negative vz that grows with
        # gravity (we measured up to -68 on KOTM during a long fall).
        #
        # Per-step cost = min(MAX, max(0, -vz - threshold) * coef), so:
        #   vz = -2  (bouncing):       cost = 0
        #   vz = -10 (slope):          cost = 4/step
        #   vz = -15 (moderate fall):  cost = 5/step (CAP)
        #   vz = -68 (full fall):      cost = 5/step (CAP)
        # The CAP is critical: without it, a single long fall (100+ frames at
        # vz=-68) integrates to -58k raw and crashes training (we measured
        # critic VL spike to 3337 and policy collapsed to 92% brake / 0 gems).
        # With cap at 5/step, a 100-frame fall costs ~500 raw — comparable to
        # 2-3 gems, gives strong gradient signal, but can never drown reward.
        self.FREE_FALL_VZ_THRESHOLD = 8.0   # cost only above this magnitude
        self.FREE_FALL_COEF = 2.0           # raw units per (vz - threshold)
        self.FREE_FALL_MAX_PER_STEP = 0.75  # MIDPOINT (2026-05-09). Cap=1.5 caused
                                            # collapse (gems 30->3 over 200 games as
                                            # marble froze to avoid falls). Cap=0.5 was
                                            # ignored (OOB rate flat at 5/rollout, gems
                                            # climbed 28->34 organically without behavior
                                            # change toward edge avoidance). The transition
                                            # is sharp because of marginal-payoff dynamics:
                                            # too low and reducing FF saves trivial reward,
                                            # too high and reducing FF beats keeping gem
                                            # rate up. 0.75 splits the difference.
                                            # Per game expected: 28 OOBs * 100 frames * 0.75
                                            # = ~2100 raw freefall = ~32% of gem reward at
                                            # 33 gems. Hopefully creates measurable OOB
                                            # pressure without re-triggering collapse.
                                            # WATCH for collapse pattern: brake_rate
                                            # climbing past 50%, pickup_speed dropping
                                            # below 7, steps/gem above 600 — all early
                                            # warnings of the freeze trajectory.
        self.GEM_GAP_PENALTY_K = 0.0    # DISABLED for KingOfTheMarble transfer. On this map, avg_steps/gem hit 2838 (vs ~290 on flat). At k=0.0008, the cumulative gap penalty per inter-gem segment was ~3,200 raw — over 16x the gem reward (+200). This dominated the reward signal, made returns wildly bimodal (huge negative spikes between gems), and contributed directly to the critic instability (VL oscillating 4-27, CGN spiking to 472). With sparse gems on a complex map, ramping penalties teach the model nothing — they just punish exploration. Re-enable later (low value like 0.0001) once the model is reliably picking up gems on the new map. Old value: 0.0008. Original note kept for context: "Reverted from 0.0015 -> 0.0008 after the 0.0015 run produced ZERO behavior change."
        self.JUMP_COST = 0.3          # Per-step cost when jump action == 1. Halved from 0.6 -> 0.3 to make jump-over an economically viable alternative to bumping into platforms during flat traversal. Combined with GEM_GAP_PENALTY_K=0.0008, the model now has incentive to either steer around platforms (cheapest, ~0 cost) or jump over them (cheap, ~3 raw) instead of bumping (expensive, ~8.4 raw per second of waste). Per-platform jumps still profitable at this level: 30-frame approach jump = 9 raw cost vs 200 gem reward = +191 net.
        self.SLOW_PICKUP_BONUS = 0       # DISABLED for KingOfTheMarble transfer. Adds reward variance the critic can't absorb during transfer; brake never developed structure on flat anyway (heatmap proved state-blind). Was 100. Re-enable later if model is reliably picking up gems on the new map and we want to push for slower pickups.
        self.SLOW_PICKUP_SPEED_CAP = 30  # Speed at/above which bonus is zero.
        # === Time-Optimal Control: overshoot penalty (Phase 1) ===
        # At any step, if stopping_dist = v_radial^2 / (2*A_MAX) exceeds distance to gem,
        # the marble is committed to overshoot. Penalty fires continuously until the
        # marble brakes back into the safe envelope. This gives a DENSE gradient signal
        # (every step in the overshoot zone) instead of sparse pickup-time rewards,
        # solving the credit-assignment problem of "when to start braking".
        # See "Time-Optimal Control" / "Safe Velocity Envelope" — physics-grounded approach.
        self.A_MAX = 12.0           # Calibrated empirically via manual_brake.py benchmark (10 games × 2 configs). The aggressive config (a_max=12, margin=0.00, min_target=0.05) hit avg 102.7 gems with avg pickup speed 4.33. Was 20.0 (intuition guess), now 12.0 (measured). At v=25 with A_MAX=12: stopping_dist = 625/24 = 26.0 units. So safe zone shrinks from d>15.6 (old) to d>26 (new) — penalty fires earlier on approach, pushing the model to brake sooner. This matches actual marble physics where reverse-throttle decel + rolling friction ≈ 12 units/s², not 20.
        self.OVERSHOOT_LAMBDA = 0.0  # DISABLED for KingOfTheMarble transfer. The overshoot penalty was Time-Optimal Control shaping calibrated for flat ground (A_MAX=12). On a sloped map, the constant A_MAX assumption is wrong, so this penalty fires misleadingly. More importantly, it's sustained-negative shaping that adds variance to the reward signal during transfer — exactly what we're stripping out to let the critic stabilize. Was 0.5. Original calibration note: "At full commitment (v_radial=25, d=5): overshoot=10.6, penalty=5.3 raw/step."
        self.SHAPING_COEFF_1 = 20   # Broad attraction: 20/(1+d/50)
        self.SHAPING_RANGE_1 = 50   # Broad attraction range
        self.SHAPING_COEFF_2 = 15   # Near-gem well: 15/(1+d/3)
        self.SHAPING_RANGE_2 = 3    # Near-gem well range
        self.GRACE_PERIOD = 20      # Steps to skip shaping after gem/OOB/episode start
        self.DIST_SENTINEL = 999    # Sentinel value for "no gem"
        self.DIST_MAX_VALID = 900   # Max valid gem distance

        # Shaping state (mirrors CS globals that were removed)
        self.last_nearest_gem_dist = self.DIST_SENTINEL
        self.skip_potential_steps = 1  # Suppress sentinel spike on first step
        self.steps_since_gem = 0       # Ramping penalty counter, reset on gem pickup

        # Statistics (initialize before loading checkpoint)
        self.total_steps = 0
        self.total_updates = 0
        self.total_episodes = 0
        self.episode_rewards = deque(maxlen=100)
        self.current_episode_reward = 0
        self.best_avg_reward = -float('inf')

        # Load existing model if provided
        if model_path and os.path.exists(model_path):
            self.log(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, weights_only=False)
            # Load actor weights. Old checkpoints used a shared ActorCritic —
            # map compatible keys across (features.*, actor_mean.*, log_std).
            saved_state = checkpoint.get('model_state_dict', checkpoint.get('actor_state_dict', {}))
            actor_state = self.actor.state_dict()
            filtered = {}
            for key, val in saved_state.items():
                if key in actor_state:
                    if val.shape == actor_state[key].shape:
                        filtered[key] = val
                    else:
                        self.log(f"  Shape mismatch for {key}: checkpoint={val.shape} vs actor={actor_state[key].shape} -- skipping")
            self.actor.load_state_dict(filtered, strict=False)
            new_params = [k for k in actor_state if k not in filtered]
            if new_params:
                self.log(f"  New actor params (randomly init): {', '.join(new_params)}")

            # Restore critic if available (falls back to fresh init for old checkpoints)
            critic_state_saved = checkpoint.get('critic_state_dict')
            if critic_state_saved:
                critic_state = self.critic.state_dict()
                critic_filtered = {}
                for key, val in critic_state_saved.items():
                    if key in critic_state:
                        if val.shape == critic_state[key].shape:
                            critic_filtered[key] = val
                if critic_filtered:
                    self.critic.load_state_dict(critic_filtered, strict=False)
                    self.log(f"  Critic restored from checkpoint")
                else:
                    self.log(f"  Critic: no compatible weights found, starting fresh")
            else:
                self.log(f"  Critic: no saved state in checkpoint, starting fresh")

            # Restore optimizer states if available
            actor_opt = checkpoint.get('actor_optimizer_state_dict')
            if actor_opt:
                try:
                    self.trainer.actor_optimizer.load_state_dict(actor_opt)
                    self.log(f"  Actor optimizer restored")
                except Exception:
                    self.log(f"  Actor optimizer: incompatible, starting fresh")
            critic_opt = checkpoint.get('critic_optimizer_state_dict')
            if critic_opt and critic_state_saved:
                try:
                    self.trainer.critic_optimizer.load_state_dict(critic_opt)
                    self.log(f"  Critic optimizer restored")
                except Exception:
                    self.log(f"  Critic optimizer: incompatible, starting fresh")

            # Re-apply current LR after optimizer state restore (which carries old LR)
            actor_lr = self.trainer.actor_optimizer.defaults['lr']
            critic_lr = self.trainer.critic_optimizer.defaults['lr']
            self.trainer.actor_optimizer.param_groups[0]['lr'] = actor_lr
            self.trainer.critic_optimizer.param_groups[0]['lr'] = critic_lr
            self.log(f"  LR set: actor={actor_lr}, critic={critic_lr}")

            # Restore training progress
            self.total_steps = checkpoint.get('total_steps', 0)
            self.total_updates = checkpoint.get('total_updates', 0)
            self.total_episodes = checkpoint.get('total_episodes', 0)
            self.best_avg_reward = checkpoint.get('best_avg_reward', -float('inf'))
            saved_rewards = checkpoint.get('episode_rewards', [])
            self.episode_rewards = deque(saved_rewards, maxlen=100)

            # Restore warmup state (None if checkpoint pre-dates warmup feature)
            self.warmup_start_update = checkpoint.get('warmup_start_update', None)

            with torch.no_grad():
                std_deg = self.actor.log_std.exp().item() * 180 / 3.14159
                thr_std = self.actor.throttle_log_std.exp().item()
                self.log(f"  PolicyStd: {std_deg:.1f} deg (bounds: {self.actor.LOG_STD_MIN:.1f} to {self.actor.LOG_STD_MAX:.1f})")
                self.log(f"  ThrottleStd: {thr_std:.3f} (bounds: {self.actor.THROTTLE_LOG_STD_MIN:.1f} to {self.actor.THROTTLE_LOG_STD_MAX:.1f})")

            # Restore throttle floor state
            self.throttle_floor_active = checkpoint.get('throttle_floor_active', True)
            if not self.throttle_floor_active:
                self.actor.THROTTLE_FLOOR = 0.0
                self.consecutive_20gem_games = 3  # Already matured

            self.log(f"Model loaded successfully!")
            self.log(f"Resuming from: {self.total_steps} steps, {self.total_updates} updates, {self.total_episodes} episodes")
            self.log(f"Best avg reward restored: {self.best_avg_reward:.2f}")
            self.log(f"  Throttle floor: {'active (0.15)' if self.throttle_floor_active else 'disabled (mature)'}")

        self.actor.train()
        self.critic.train()
        self._session_start_step = self.total_steps

        # Recent actions (for angle/throttle stats display and dashboard)
        self.recent_actions = deque(maxlen=200)
        self.recent_throttles = deque(maxlen=200)

        # Per-episode counters (reset on done)
        self.episode_gem_pts = 0   # gem points collected this episode
        self.episode_oob = 0       # OOB events this episode
        self.episode_step = 0      # step index within the current episode
        self._frame_hist_logged = False  # has frame history debug logging started?
        self._frame_hist_log_end = 999999  # step to stop logging at

        # Action-repeat state. `pending` is the decision currently being held:
        # (obs_augmented, action_buf, value, log_prob). It is stored to the
        # buffer only when its window closes (next decision tick, done, or game
        # end), with `pending_reward` = sum of the per-tick rewards received
        # while it was held. `cached_action` is the (dx, dy, throttle, jump,
        # brake) tuple re-emitted on the intermediate ticks.
        self.pending = None
        self.pending_reward = 0.0
        self.tick_in_window = 0
        self.cached_action = None

        # Per-rollout counters (reset after each PPO update)
        self.rollout_gem_pts = 0   # gem points collected this rollout
        self.rollout_oob = 0       # OOB events this rollout
        self.rollout_positive = 0  # steps with positive reward this rollout
        self.rollout_steps = 0     # total steps this rollout
        # Free-fall penalty diagnostics (per rollout, surfaced on each Upd log line
        # so we can see in <12 sec wall-clock whether the new penalty is firing).
        self.rollout_freefall_frames = 0     # frames where free-fall penalty fired
        self.rollout_freefall_penalty = 0.0  # total free-fall penalty applied this rollout
        self.rollout_min_vz = 0.0            # most-negative vz seen this rollout
        # (action counts removed — continuous actions tracked via recent_actions deque)

        # Lifetime counters
        self.total_gem_pts = 0     # gem points across entire run
        self.total_oob = 0         # OOB events across entire run

        # Camera yaw is now FIXED at 0 radians for every episode and every respawn.
        # Random camera during training was preventing the jump_features network
        # from learning XY-based platform discrimination — the same world position
        # produced a different obs[0:2] every game with random camera, making
        # "jump at these coordinates" impossible to learn. At inference time the
        # camera is also locked (never moves during play), so training with a
        # fixed orientation transfers cleanly — the model never needs to deal
        # with rotation and doesn't overfit to one specific angle because there's
        # only ever one angle to experience.
        self.camera_yaw = 0.0

        # Entropy tracking for collapse detection
        self.entropy_history = deque(maxlen=10)

        # Run start time (for gems/hr)
        self.run_start_time = time.time()

        # Dry-rollout streak (rollouts with zero gems collected)
        self.dry_rollouts = 0

        # No-gem tracking (steps where no gem exists on the map)
        self.no_gem_steps = 0        # consecutive steps with no gem
        self.total_no_gem_steps = 0  # lifetime total
        self.episode_no_gem_steps = 0  # per-episode total
        self.no_gem_events = 0       # number of no-gem gaps

        # Rolling 100-episode gem points and lengths (for dashboard + flood detection)
        self.recent_episode_gems = deque(maxlen=100)
        self.recent_episode_lengths = deque(maxlen=100)

        # Throttle floor maturity gate: keep THROTTLE_FLOOR=0.15 until the model
        # gets 20+ gems for 3 consecutive games, then drop to 0 permanently.
        # This prevents sit-still collapse during early training when the model
        # hasn't learned that gems = reward, while allowing full speed control
        # once it knows what it's doing.
        self.throttle_floor_active = True
        self.consecutive_20gem_games = 0

        # Game-level gem tracking (a "game" = full 5-min Hunt round)
        # Episodes may be shorter than a game if step cap fires, so we
        # accumulate gems across episodes and only record when a full
        # game ends (episode_step >= 10000, meaning timer expired).
        self.game_gem_pts = 0          # accumulator across episodes within one game
        self.recent_game_gems = deque(maxlen=100)  # last 100 full games
        self.best_game_gems = 0        # all-time best gems in a single game
        self.game_gap_penalty_sum = 0.0   # sum of gap penalties across current game
        self.game_gem_pickups = 0         # number of gem pickups in current game
        self.recent_avg_gap_penalty = deque(maxlen=100)  # avg gap penalty per gem, per game

        # Overshoot penalty tracking (Time-Optimal Control)
        # Per-game accumulators reset at game end. Useful diagnostics:
        # - overshoot_penalty_sum: total raw cost paid this game (units of raw reward)
        # - overshoot_frames: count of frames where penalty fired (high = lots of bad commits)
        # - avg_overshoot_per_frame: severity per active frame (low = grazing the edge of safe zone; high = deep in overshoot)
        self.game_overshoot_penalty_sum = 0.0
        self.game_overshoot_frames = 0
        self.game_freefall_penalty_sum = 0.0
        self.game_freefall_frames = 0
        self.recent_overshoot_penalty = deque(maxlen=100)  # per-game total overshoot penalty
        self.recent_overshoot_frames = deque(maxlen=100)   # per-game count of overshoot frames

        # Near-miss and dwell-time tracking (per game)
        # "Near" = within 2 marble diameters of nearest gem (~0.8 units)
        self.NEAR_GEM_THRESHOLD = 0.8    # 2 marble diameters (radius ~0.2 * 4)
        self.near_gem = False             # currently within threshold
        self.game_near_misses = 0         # count: entered near zone, left without pickup
        self.game_dwell_steps = 0         # total steps spent within threshold
        self.recent_near_misses = deque(maxlen=100)    # per-game near-miss count
        self.recent_dwell_steps = deque(maxlen=100)    # per-game dwell steps

        # Jump rate and steps-per-gem tracking (per game)
        self.game_jumps = 0              # jump DECISIONS this game
        self.game_brakes = 0             # brake DECISIONS this game
        self.game_decisions = 0          # decisions this game (jump/brake rate denominator)
        self.game_steps = 0              # total ticks this game
        self.game_gem_steps_sum = 0      # sum of steps_since_gem at each pickup
        self.recent_jump_rate = deque(maxlen=100)        # jump % per game
        self.recent_brake_rate = deque(maxlen=100)       # brake % per game
        self.recent_avg_steps_per_gem = deque(maxlen=100) # avg steps between gems per game

        # Brake-effectiveness diagnostics (added to verify the brake action is doing
        # what we want vs just firing randomly). Three metrics:
        #  - avg_pickup_speed: speed at moment of gem pickup. The OUTCOME metric — if
        #    brake works, this drops over training.
        #  - brakes_near_pickup: number of brake events in last 30 frames before each
        #    pickup. The MECHANISM metric — if model learned timing, this rises.
        #  - slow_pickup_bonus_total: sum of bonus paid out per game. Verifies the
        #    reward signal is reaching the model.
        self.recent_brake_history = deque(maxlen=30)      # rolling 30-step brake action history
        self.game_pickup_speed_sum = 0.0                  # sum of pickup speeds this game
        self.game_brakes_near_pickup_sum = 0              # sum of brake counts in last 30 frames before each pickup
        self.game_slow_pickup_bonus_sum = 0.0             # sum of slow-pickup bonuses paid this game
        self.recent_avg_pickup_speed = deque(maxlen=100)
        self.recent_avg_brakes_near_pickup = deque(maxlen=100)
        self.recent_slow_pickup_bonus = deque(maxlen=100)

        # Socket setup
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.settimeout(1.0)
        self.socket.bind((self.host, self.port))

        # Create directories
        os.makedirs('models/checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

        # Real-time dashboard (daemon thread, zero training impact)
        self.dashboard = DashboardServer(self)
        self.dashboard.start()

    def run(self):
        """Main server loop."""
        self.log(f"=" * 60)
        self.log(f"PPO Training Server")
        self.log(f"=" * 60)
        self.log(f"Listening on {self.host}:{self.port}")
        self.log(f"Obs dim: {self.obs_dim} (base={self.obs_dim_base} + {self.FRAME_HISTORY_COUNT}x{self.FRAME_HISTORY_DIMS} history, skip={self.FRAME_SKIP})")
        self.log(f"Rollout size: {self.rollout_size} steps")
        self.log(f"PPO epochs: {self.n_epochs}")
        self.log(f"Batch size: {self.batch_size}")
        self.log(f"Press Ctrl+C to stop")
        self.log(f"=" * 60)
        self.log(f"Waiting for game to connect...")

        self.socket.listen(1)

        while self.running:
            try:
                conn, addr = self.socket.accept()
                self.log(f"\nGame connected from {addr}")
                self.handle_client(conn)
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                self.log("\nShutting down...")
                self.running = False
                break
            except Exception as e:
                if self.running:
                    self.log(f"Error: {e}")

        self.socket.close()
        self.log("Server stopped.")
        self.logger.close()

    def handle_client(self, conn):
        """Handle communication with the game."""
        buffer_str = ""

        try:
            while self.running:
                data = conn.recv(8192).decode('utf-8')
                if not data:
                    self.log("Game disconnected. Shutting down.")
                    self.running = False
                    break

                buffer_str += data

                while '\n' in buffer_str:
                    line, buffer_str = buffer_str.split('\n', 1)
                    line = line.strip()

                    if not line:
                        continue

                    # Parse message: obs_json|gem_delta|oob|done
                    action = self.process_message(line)

                    # Send action back (5th field = camera yaw for randomization)
                    action_str = ','.join(map(str, action)) + f',{self.camera_yaw:.6f}\n'
                    conn.sendall(action_str.encode('utf-8'))

        except Exception as e:
            self.log(f"Client error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()

    def normalize_obs(self, obs):
        """Normalize raw game observations to roughly [-1, 1] range.

        Observation layout (35 dims from game):
          [0-5]   Self state (6 dims: pos, vel — both camera-relative)
          [6-30]  5 nearest gems x 5 dims = 25 dims
          [31-34] Game state              =  4 dims
        Python appends 24 dims of frame history -> 59 total to model.
        """
        # Self state (indices 0-5)
        obs[0:3]  /= 100.0   # Position (camera-relative: x=right, y=forward, z=up)
        obs[3:6]  /= 20.0    # Velocity (camera-relative: x=right, y=forward, z=up)

        # Gems (indices 6-30: 5 gems x 5 dims = x, y, z, value, distance)
        # Fix sentinel values: -999 meant "no gem" but replacing with 0 told the
        # network a gem was AT the marble. Instead, mark absent gems as far away.
        gem_base = 6
        for i in range(5):
            b = gem_base + i * 5
            if obs[b+4] < -500:  # distance is sentinel -> gem absent
                obs[b:b+3] = 0.0  # no directional info
                obs[b+3]   = 0.0  # no value
                obs[b+4]   = 1.0  # max normalized distance (far away)
            else:
                dist = obs[b+4]
                if dist > 0.01:
                    obs[b:b+3] /= dist  # Unit direction vector (always magnitude ~1)
                else:
                    obs[b:b+3] = 0.0    # On top of gem, no direction needed
                obs[b+3]   /= 5.0      # Gem value (1-5 -> 0.2-1.0)
                obs[b+4]   /= 100.0    # Distance (0-100+ -> 0-1+)

        # Game state (indices 31-34)
        obs[31] /= 300000.0   # timeElapsed    (5-min hunt = 300,000 ms -> 0-1)
        obs[32] /= 300000.0   # timeRemaining
        obs[33] /= 100.0      # myGemScore
        obs[34] /= 50.0       # gemsRemaining

        # Safety clip: catch any remaining outliers.
        obs = np.clip(obs, -2.0, 2.0)

        return obs

    def compute_reward(self, raw_obs, gem_delta, oob):
        """Compute reward from raw game facts. Mirrors old mlAgent.cs::computeReward exactly.

        Args:
            raw_obs: raw observation list (before normalization), index 12 = nearest gem distance
            gem_delta: gem points collected this step (0 or positive int)
            oob: 1 if out-of-bounds this step, 0 otherwise

        Returns:
            float: raw reward (will be scaled/clipped later)
        """
        reward = 0.0

        # 0. OOB resets shaping state BEFORE reward computation.
        #    In CS, onOOB() fired before computeReward() — it set
        #    LastNearestGemDist=999 and SkipPotentialSteps=20 so the
        #    OOB step's distance check would be in grace period.
        if oob:
            self.last_nearest_gem_dist = self.DIST_SENTINEL
            self.skip_potential_steps = self.GRACE_PERIOD

        # 1. Gem collection reward
        if gem_delta > 0:
            reward += gem_delta * self.GEM_REWARD

            # Slow pickup bonus: reward collecting gems at low speed.
            # Encourages braking before collection so the marble can
            # immediately redirect toward the next gem instead of overshooting.
            # obs[3:6] = velocity (camera-relative vx, vy, vz)
            if len(raw_obs) > 5:
                vx, vy, vz = raw_obs[3], raw_obs[4], raw_obs[5]
                speed = (vx**2 + vy**2 + vz**2) ** 0.5
                bonus = self.SLOW_PICKUP_BONUS * max(0.0, 1.0 - speed / self.SLOW_PICKUP_SPEED_CAP)
                reward += bonus
                # Diagnostic accumulators: pickup speed, brakes-near-pickup, bonus paid.
                self.game_pickup_speed_sum += speed
                self.game_brakes_near_pickup_sum += sum(self.recent_brake_history)
                self.game_slow_pickup_bonus_sum += bonus

            self.skip_potential_steps = self.GRACE_PERIOD
            # Record gap penalty for this interval before resetting
            gap_penalty = self.GEM_GAP_PENALTY_K * self.steps_since_gem * (self.steps_since_gem - 1) / 2
            self.game_gap_penalty_sum += gap_penalty
            self.game_gem_pickups += 1
            self.game_gem_steps_sum += self.steps_since_gem
            self.steps_since_gem = 0  # Reset ramping penalty on gem pickup

        # 2. Distance shaping: potential-based P(d) = C1/(1+d/R1) + C2/(1+d/R2)
        # 2b. Overshoot penalty (Time-Optimal Control): -OVERSHOOT_LAMBDA * max(0, v_radial^2/(2*A_MAX) - d)
        #     Fires when stopping distance exceeds remaining distance — i.e., the marble is
        #     committed to overshoot. Continuous gradient signal at the "switching point".
        #     Skipped during grace period (just like distance shaping) because the gem just
        #     changed and the marble can't be expected to react instantly.
        nearest_dist = raw_obs[10] if len(raw_obs) > 10 else -1
        if nearest_dist > 0 and nearest_dist < self.DIST_MAX_VALID:
            if self.skip_potential_steps > 0:
                self.skip_potential_steps -= 1
            else:
                # 2a. Distance shaping
                new_potential = (self.SHAPING_COEFF_1 / (1 + nearest_dist / self.SHAPING_RANGE_1)
                               + self.SHAPING_COEFF_2 / (1 + nearest_dist / self.SHAPING_RANGE_2))
                old_potential = (self.SHAPING_COEFF_1 / (1 + self.last_nearest_gem_dist / self.SHAPING_RANGE_1)
                               + self.SHAPING_COEFF_2 / (1 + self.last_nearest_gem_dist / self.SHAPING_RANGE_2))
                reward += new_potential - old_potential

                # 2b. Overshoot penalty
                # raw_obs[3:6] = velocity (camera-relative); raw_obs[6:9] = gem-relative position
                if len(raw_obs) >= 9 and nearest_dist > 0.01:
                    vx, vy, vz = raw_obs[3], raw_obs[4], raw_obs[5]
                    gx_rel, gy_rel, gz_rel = raw_obs[6], raw_obs[7], raw_obs[8]
                    # Component of velocity in the gem direction (positive = approaching).
                    # Doing donuts at high speed (perpendicular motion) gives v_radial ~= 0
                    # so no penalty fires from non-approaching motion.
                    v_radial = (vx * gx_rel + vy * gy_rel + vz * gz_rel) / nearest_dist
                    v_radial = max(0.0, v_radial)
                    stopping_dist = (v_radial * v_radial) / (2.0 * self.A_MAX)
                    overshoot = max(0.0, stopping_dist - nearest_dist)
                    if overshoot > 0:
                        overshoot_penalty = self.OVERSHOOT_LAMBDA * overshoot
                        reward -= overshoot_penalty
                        self.game_overshoot_penalty_sum += overshoot_penalty
                        self.game_overshoot_frames += 1
            self.last_nearest_gem_dist = nearest_dist

        # 3. Ramping time penalty: grows with steps since last gem pickup.
        reward -= self.GEM_GAP_PENALTY_K * self.steps_since_gem
        self.steps_since_gem += 1

        # 4. OOB penalty
        if oob:
            reward -= self.OOB_PENALTY

        # 5. Free-fall penalty (dense per-step signal during a fall trajectory).
        # Replaces brute-force OOB-only terminal penalty. The marble's vz becomes
        # strongly negative when falling off a surface (gravity, no contact force),
        # well above the bouncing range on any normal floor / slope / curve.
        # Per-step cost is HARD-CAPPED to prevent the catastrophic drown-out we
        # measured when uncapped (vz=-68 -> -120/step -> -58k/rollout).
        if len(raw_obs) > 5:
            vz = raw_obs[5]
            if vz < self.rollout_min_vz:
                self.rollout_min_vz = vz
            fall_excess = -vz - self.FREE_FALL_VZ_THRESHOLD
            if fall_excess > 0:
                fall_cost = min(self.FREE_FALL_MAX_PER_STEP,
                                fall_excess * self.FREE_FALL_COEF)
                reward -= fall_cost
                self.game_freefall_penalty_sum += fall_cost
                self.game_freefall_frames += 1
                self.rollout_freefall_penalty += fall_cost
                self.rollout_freefall_frames += 1

        return reward

    def process_message(self, message):
        """Process a message from the game, return action."""
        try:
            # Parse: obs_json|gem_delta|oob|done
            parts = message.split('|')

            if len(parts) != 4:
                if self.total_steps < 3:
                    self.log(f"Malformed message (expected 4 parts, got {len(parts)}): {message[:100]}")
                return [0, 0, 0, 0, 0]

            obs_json, gem_delta_str, oob_str, done_str = parts

            # Parse fields
            obs = json.loads(obs_json)
            gem_delta = float(gem_delta_str)
            oob = int(float(oob_str))
            done = int(float(done_str))

            # Handle game-end signal: empty obs [] with done=1.
            # onGameEnd sends total game gems as negative gem_delta.
            # This is the authoritative game boundary — record to recent_game_gems.
            if len(obs) == 0 and done:
                total_game_gems = int(abs(gem_delta))  # CS sends negative to distinguish
                # The game is over: close the held decision's window and mark it
                # terminal so GAE does not bootstrap across the restart teleport.
                # (Previously the last stored transition kept done=False and its
                # return silently included V(spawn) of the NEXT game.)
                self._finalize_pending(done=True)
                self.cached_action = None
                self.recent_game_gems.append(total_game_gems)
                if total_game_gems > self.best_game_gems:
                    self.best_game_gems = total_game_gems
                avg_gap = self.game_gap_penalty_sum / max(self.game_gem_pickups, 1)
                self.recent_avg_gap_penalty.append(round(avg_gap, 2))
                self.recent_near_misses.append(self.game_near_misses)
                self.recent_dwell_steps.append(self.game_dwell_steps)
                # Jump rate, brake rate (% of DECISIONS), and avg steps per gem for this game
                jump_rate = (self.game_jumps / max(self.game_decisions, 1)) * 100
                brake_rate = (self.game_brakes / max(self.game_decisions, 1)) * 100
                self.recent_jump_rate.append(round(jump_rate, 2))
                self.recent_brake_rate.append(round(brake_rate, 2))
                avg_steps_per_gem = self.game_gem_steps_sum / max(self.game_gem_pickups, 1)
                self.recent_avg_steps_per_gem.append(round(avg_steps_per_gem, 1))

                # Overshoot stats for this game (Time-Optimal Control diagnostics)
                self.recent_overshoot_penalty.append(round(self.game_overshoot_penalty_sum, 1))
                self.recent_overshoot_frames.append(self.game_overshoot_frames)
                overshoot_pct = (self.game_overshoot_frames / max(self.game_steps, 1)) * 100
                avg_overshoot_per_frame = (self.game_overshoot_penalty_sum
                                           / max(self.game_overshoot_frames, 1))

                # Brake-effectiveness diagnostics (per-pickup averages + per-game total bonus)
                pickups = max(self.game_gem_pickups, 1)
                avg_pickup_speed = self.game_pickup_speed_sum / pickups
                avg_brakes_near_pickup = self.game_brakes_near_pickup_sum / pickups
                self.recent_avg_pickup_speed.append(round(avg_pickup_speed, 2))
                self.recent_avg_brakes_near_pickup.append(round(avg_brakes_near_pickup, 2))
                self.recent_slow_pickup_bonus.append(round(self.game_slow_pickup_bonus_sum, 1))

                freefall_pct = (self.game_freefall_frames / max(self.game_steps, 1)) * 100
                self.log(f"[GAME END] total gems this game: {total_game_gems}pts (best: {self.best_game_gems}) avg_gap_penalty: {avg_gap:.1f} jump_rate: {jump_rate:.1f}% brake_rate: {brake_rate:.1f}% avg_steps/gem: {avg_steps_per_gem:.0f} overshoot: {self.game_overshoot_penalty_sum:.0f} ({overshoot_pct:.1f}% of steps, avg {avg_overshoot_per_frame:.2f}/frame) pickup_speed: {avg_pickup_speed:.1f} brakes_near_pickup: {avg_brakes_near_pickup:.2f}/gem slow_pickup_bonus: {self.game_slow_pickup_bonus_sum:.0f} freefall: {self.game_freefall_penalty_sum:.0f} ({freefall_pct:.1f}% of steps)")

                # Refresh the dashboard's jump-probability heatmap in a background
                # thread. Probe takes ~1s on a snapshot of actor weights — never
                # blocks the training loop.
                try:
                    self.dashboard.trigger_heatmap_update()
                except Exception as e:
                    self.log(f"Dashboard heatmap trigger error: {e}")

                # Throttle floor maturity gate
                if self.throttle_floor_active:
                    if total_game_gems >= 20:
                        self.consecutive_20gem_games += 1
                        if self.consecutive_20gem_games >= 3:
                            self.actor.THROTTLE_FLOOR = 0.0
                            self.throttle_floor_active = False
                            self.log(f"[MATURITY] Throttle floor disabled (3 consecutive 20+ gem games)")
                    else:
                        self.consecutive_20gem_games = 0

                self.game_gem_pts = 0  # Reset accumulators for next game
                self.game_gap_penalty_sum = 0.0
                self.game_gem_pickups = 0
                self.game_near_misses = 0
                self.game_dwell_steps = 0
                self.game_jumps = 0
                self.game_brakes = 0
                self.game_decisions = 0
                self.game_steps = 0
                self.game_gem_steps_sum = 0
                self.game_overshoot_penalty_sum = 0.0
                self.game_overshoot_frames = 0
                self.game_freefall_penalty_sum = 0.0
                self.game_freefall_frames = 0
                self.game_pickup_speed_sum = 0.0
                self.game_brakes_near_pickup_sum = 0
                self.game_slow_pickup_bonus_sum = 0.0
                self.near_gem = False

                # Game-end is the authoritative episode boundary on this protocol.
                # The per-step done=1 path (checkDone in mlAgent.cs) rarely fires
                # because clientCmdGameEnd preempts the next agent tick by setting
                # $MLAgent::Enabled=false. Without this bookkeeping, total_episodes
                # and episode_rewards stay frozen at the values from the previous
                # working run -> AvgRwd appears stuck (was 12973.2 from FlatWithJump
                # for ~5 KingOfTheMarble runs in a row).
                self.total_episodes += 1
                self.episode_rewards.append(self.current_episode_reward)
                self.recent_episode_gems.append(self.episode_gem_pts)
                self.recent_episode_lengths.append(self.episode_step)
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                outcome = "SUCCESS" if self.current_episode_reward > 100 else "FAIL" if self.current_episode_reward < -20 else "NEUTRAL"
                self.log(f"Ep {self.total_episodes} [{outcome}] rwd={self.current_episode_reward:.1f} "
                         f"gems={total_game_gems}pts oob={self.episode_oob} "
                         f"steps={self.episode_step} | avg100={avg_reward:.1f}")
                self.current_episode_reward = 0
                self.episode_gem_pts = 0
                self.episode_oob = 0
                self.episode_step = 0
                self.episode_no_gem_steps = 0
                # Reset shaping state for next episode (mirrors the per-step done branch)
                self.last_nearest_gem_dist = self.DIST_SENTINEL
                self.skip_potential_steps = 1
                self.steps_since_gem = 0
                self.near_gem = False
                self.frame_history.clear()  # the marble teleports to spawn on restart
                self._frame_hist_logged = False
                self._frame_hist_log_end = 999999

                return [0, 0, 0, 0, 0]

            # ======================= Per-tick work (every 16 ms game tick) =======================
            # Track no-gem steps (sentinel distance at index 10 -- nearest gem dist)
            raw_gem0_dist = obs[10] if len(obs) > 10 else -1
            if raw_gem0_dist < -500:
                if self.no_gem_steps == 0:
                    self.no_gem_events += 1
                self.no_gem_steps += 1
                self.total_no_gem_steps += 1
                self.episode_no_gem_steps += 1
            elif self.no_gem_steps > 0:
                self.no_gem_steps = 0

            # Near-miss and dwell-time tracking
            if raw_gem0_dist > 0 and raw_gem0_dist < self.DIST_MAX_VALID:
                if raw_gem0_dist < self.NEAR_GEM_THRESHOLD:
                    # Inside near zone
                    if not self.near_gem:
                        self.near_gem = True  # Just entered
                    self.game_dwell_steps += 1
                else:
                    # Outside near zone
                    if self.near_gem:
                        # Was near, now left without picking up -> near miss
                        if gem_delta == 0:
                            self.game_near_misses += 1
                        self.near_gem = False
            # Reset near_gem flag on gem pickup (successful collection, not a miss)
            if gem_delta > 0:
                self.near_gem = False

            # Compute reward in Python (all tunable params live here now).
            # Rewards are computed EVERY tick and summed into the window of the
            # decision currently being held. The facts in this message (gem
            # pickup, OOB, position) are consequences of actions applied on
            # earlier ticks, so they belong to the pending decision -- never to
            # the one chosen below. (The old code stored each tick's reward with
            # the action chosen AFTER seeing it, i.e. with the wrong action.)
            reward = self.compute_reward(obs, gem_delta, oob)
            self.pending_reward += reward

            # Normalize observation (must happen AFTER compute_reward uses raw values)
            obs_array = self.normalize_obs(np.array(obs, dtype=np.float32))

            # Frame-history ring buffer is fed every tick, so the t-8/16/24/32
            # snapshots keep their tick-based meaning under action repeat.
            current_posvel = obs_array[0:6].copy()  # 6 dims: pos(3) + vel(3)
            self.frame_history.append(current_posvel)

            # Track events
            if gem_delta > 0:
                self.episode_gem_pts += int(gem_delta)
                self.rollout_gem_pts += int(gem_delta)
                self.total_gem_pts += int(gem_delta)
                self.log(f"[GEM] ep={self.total_episodes+1} step={self.total_steps} +{gem_delta:.0f}pts | ep_total={self.current_episode_reward + reward:.1f}")
            if oob:
                self.episode_oob += 1
                self.rollout_oob += 1
                self.total_oob += 1
                self.camera_yaw = 0.0  # fixed orientation (see __init__ for rationale)
            if reward > 0.1:
                self.rollout_positive += 1
            self.rollout_steps += 1
            self.episode_step += 1
            self.game_steps += 1
            self.total_steps += 1
            self.current_episode_reward += reward

            # ======================= Episode end (per-step done from checkDone) =======================
            if done:
                # The held decision's window ends here, terminally.
                self._finalize_pending(done=True)
                self.cached_action = None

                self.total_episodes += 1
                self.episode_rewards.append(self.current_episode_reward)
                self.recent_episode_gems.append(self.episode_gem_pts)
                self.recent_episode_lengths.append(self.episode_step)
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                outcome = "SUCCESS" if self.current_episode_reward > 100 else "FAIL" if self.current_episode_reward < -20 else "NEUTRAL"
                oob_str = f" OOB={self.episode_oob}" if self.episode_oob else ""
                gems_str = f" gems={self.episode_gem_pts}pts" if self.episode_gem_pts else ""
                self.log(f"Ep {self.total_episodes} [{outcome}] rwd={self.current_episode_reward:.1f}{gems_str}{oob_str} steps={self.episode_step} | avg100={avg_reward:.1f}")
                if self.episode_step <= 15:
                    self.log(f"  *** SHORT EPISODE ({self.episode_step} steps) - flood bug may still be active ***")

                self.current_episode_reward = 0
                self.episode_gem_pts = 0
                self.episode_oob = 0
                self.episode_step = 0
                self.episode_no_gem_steps = 0
                # Reset shaping state for new episode (mirrors CS resetEpisode)
                self.last_nearest_gem_dist = self.DIST_SENTINEL
                self.skip_potential_steps = self.GRACE_PERIOD
                self.steps_since_gem = 0
                self.near_gem = False
                self.frame_history.clear()  # Fresh history for new episode
                self._frame_hist_logged = False
                self._frame_hist_log_end = 999999
                self.camera_yaw = 0.0  # fixed orientation (see __init__ for rationale)
                self.recent_brake_history.append(0)
                return [0, 0, 0, 0, 0]

            # ======================= Decision tick (every ACTION_REPEAT ticks) =======================
            self.tick_in_window += 1
            if self.pending is None or self.tick_in_window >= self.ACTION_REPEAT:
                # 1. The previous decision's window is complete: store it.
                self._finalize_pending(done=False)

                # 2. Build the augmented observation for this decision.
                history_frames = []
                for i in range(1, self.FRAME_HISTORY_COUNT + 1):
                    # Current frame is at index len-1. Frame from i*FRAME_SKIP ticks ago
                    # is at index len-1 - i*FRAME_SKIP.
                    idx = len(self.frame_history) - 1 - i * self.FRAME_SKIP
                    if idx >= 0:
                        history_frames.append(self.frame_history[idx])
                    else:
                        history_frames.append(np.zeros(self.FRAME_HISTORY_DIMS, dtype=np.float32))
                obs_augmented = np.concatenate([obs_array] + history_frames)

                # Debug: log frame history to file only, after 5s of game time
                game_time_ms = obs[31] if len(obs) > 31 else 0
                if game_time_ms >= 5000 and self.episode_step < self._frame_hist_log_end:
                    if not self._frame_hist_logged:
                        self._frame_hist_logged = True
                        self._frame_hist_log_end = self.episode_step + 20
                    filled = sum(1 for f in history_frames if np.any(f != 0))
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.logger.file.write(
                        f"[{ts}]   [FRAME_HIST] step={self.episode_step} buf_len={len(self.frame_history)} "
                        f"filled={filled}/{self.FRAME_HISTORY_COUNT} "
                        f"cur_pos=({current_posvel[0]:.3f},{current_posvel[1]:.3f},{current_posvel[2]:.3f}) "
                        f"cur_vel=({current_posvel[3]:.3f},{current_posvel[4]:.3f},{current_posvel[5]:.3f})"
                        f"{' t-4=(' + ','.join(f'{v:.3f}' for v in history_frames[0]) + ')' if filled >= 1 else ''}"
                        f" obs_dim={len(obs_augmented)}\n"
                    )
                    self.logger.file.flush()

                # 3. Rollout boundary. The rollout is TRUNCATED here, not terminated:
                #    bootstrap its last transition with V(s_next) from the critic
                #    that produced the buffer's values, then update. (The old code
                #    updated right after buffer.add() with next_value=0.)
                if len(self.buffer) >= self.rollout_size:
                    bootstrap_value = 0.0 if self.buffer.dones[-1] else self.critic.get_value(obs_augmented)
                    self.run_ppo_update(last_value=bootstrap_value)

                # 4. Choose the new decision with the (possibly just-updated) networks.
                #    Returns buffer action (dx, dy, throttle_logit, jump_binary, brake_binary),
                #    game action (dx, dy, throttle, jump, brake), and joint log_prob.
                action_buf, action_game, log_prob = self.actor.get_action(obs_augmented)
                value = self.critic.get_value(obs_augmented)
                dx, dy, throttle, jump, brake = action_game
                angle = math.atan2(dx, dy)  # For logging/display only
                self.recent_actions.append(angle)
                self.recent_throttles.append(throttle)
                self.game_decisions += 1
                self.pending = (obs_augmented, action_buf, value, log_prob)
                self.pending_reward = 0.0
                self.tick_in_window = 0
                self.cached_action = action_game
                if jump > 0.5:
                    self.game_jumps += 1
                    # Per-decision physical cost of a jump, charged once per held
                    # decision. Each jump must "earn its keep" via a subsequent
                    # reward (a gem pickup gives +200, easily offsetting -0.3).
                    self.pending_reward -= self.JUMP_COST
                if brake > 0.5:
                    self.game_brakes += 1
                    # No explicit BRAKE_COST. Add a small per-brake cost later if
                    # the model spams it.

            # ======================= Emit the held action =======================
            # Rolling 30-tick brake history (brake flag of the action being held).
            # compute_reward reads this on the next tick's gem pickup to tally
            # "brakes in the last 30 ticks before pickup".
            dx, dy, throttle, jump, brake = self.cached_action
            self.recent_brake_history.append(1 if brake > 0.5 else 0)

            # Convert to joystick axes (F,B,L,R,jump). Brake's effect is baked into
            # F/B/L/R via the anti-velocity override, which is recomputed from the
            # CURRENT camera-relative velocity (obs[3:5]) on every tick so a held
            # brake keeps decelerating as the velocity vector changes.
            vx_raw = obs[3] if len(obs) > 5 else 0.0
            vy_raw = obs[4] if len(obs) > 5 else 0.0
            return list(Actor.action_to_joystick(dx, dy, throttle, jump, brake, vx_raw, vy_raw))

        except Exception as e:
            if self.total_steps < 5:
                self.log(f"Error processing message: {e}")
                self.log(f"Message (first 200 chars): {message[:200]}")
                import traceback
                traceback.print_exc()
            return [0, 0, 0, 0, 0]

    def _finalize_pending(self, done):
        """Store the held decision as one transition, with the reward summed over
        every tick of its window (scaled and clipped here). No-op when nothing
        is pending (first tick of a session / right after an episode boundary).
        """
        if self.pending is None:
            return
        obs_aug, action_buf, value, log_prob = self.pending
        scaled_reward = float(np.clip(self.pending_reward * self.reward_scale,
                                      -self.REWARD_CLIP, self.REWARD_CLIP))
        self.buffer.add(obs_aug, action_buf, scaled_reward, value, log_prob, done)
        self.pending = None
        self.pending_reward = 0.0
        self.tick_in_window = 0

    def run_ppo_update(self, last_value=0.0):
        """Run PPO training update.

        last_value: V(s_{T+1}) for the observation following the final stored
        transition (rollout truncation bootstrap). 0.0 if that transition was
        terminal.
        """
        self.actor.train()
        self.critic.train()

        # Initialize warmup window on first update of this run if not already set
        if self.warmup_start_update is None:
            self.warmup_start_update = self.total_updates
            self.log(f"  [WARMUP] Critic-only phase started at update {self.warmup_start_update}, "
                     f"will run for {self.WARMUP_ROLLOUTS} rollouts (until update "
                     f"{self.warmup_start_update + self.WARMUP_ROLLOUTS}). "
                     f"Actor frozen during this window.")

        warmup_active = (self.total_updates - self.warmup_start_update) < self.WARMUP_ROLLOUTS

        stats = self.trainer.update(
            self.buffer,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            gamma=self.gamma,
            lam=self.lam,
            freeze_actor=warmup_active,
            last_value=last_value,
        )
        self.total_updates += 1

        # Log warmup-end transition once
        if warmup_active and (self.total_updates - self.warmup_start_update) >= self.WARMUP_ROLLOUTS:
            self.log(f"  [WARMUP] Phase complete at update {self.total_updates}. Actor unfrozen.")
        self.entropy_history.append(stats['entropy'])

        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0

        # Detect entropy collapse for continuous Normal distribution
        # Entropy = 0.5*ln(2πe*σ²). At LOG_STD_MIN=-3 (σ=0.05rad≈2.9°): entropy≈-2.3
        # At LOG_STD_MAX=0.0 (σ=1.0rad≈57°): entropy≈1.4. Healthy range: -1.0 to 1.0
        collapse_warn = ""
        if stats['entropy'] < -0.5:
            collapse_warn = " *** ENTROPY COLLAPSE ***"
        elif stats['entropy'] < 0.3:
            collapse_warn = " (entropy low)"

        # Dry-rollout tracking
        if self.rollout_gem_pts == 0:
            self.dry_rollouts += 1
        else:
            self.dry_rollouts = 0

        dry_warn = f" DRY×{self.dry_rollouts}" if self.dry_rollouts >= 5 else ""
        kl_warn = " KL-STOP" if stats.get('kl_early_stopped') else ""

        # Average episode length (key metric for flood bug detection)
        avg_ep_len = np.mean(self.recent_episode_lengths) if self.recent_episode_lengths else 0

        # Action distribution (continuous angle + throttle)
        import math
        recent = list(self.recent_actions)[-min(200, len(self.recent_actions)):]
        recent_thr = list(self.recent_throttles)[-min(200, len(self.recent_throttles)):]
        if recent:
            angles_deg = [((a * 180 / math.pi) % 360) for a in recent]
            mean_deg = sum(angles_deg) / len(angles_deg)
            std_deg = (sum((d - mean_deg)**2 for d in angles_deg) / len(angles_deg)) ** 0.5
            log_std = torch.clamp(self.actor.log_std, self.actor.LOG_STD_MIN, self.actor.LOG_STD_MAX)
            policy_std_deg = log_std.exp().item() * 180 / math.pi
            act_str = f"MeanAngle:{mean_deg:.0f} StdDev:{std_deg:.0f} PolicyStd:{policy_std_deg:.1f}"
        else:
            act_str = "no actions yet"
        if recent_thr:
            mean_thr = sum(recent_thr) / len(recent_thr)
            min_thr = min(recent_thr)
            max_thr = max(recent_thr)
            thr_log_std = torch.clamp(self.actor.throttle_log_std,
                                      self.actor.THROTTLE_LOG_STD_MIN,
                                      self.actor.THROTTLE_LOG_STD_MAX)
            thr_policy_std = thr_log_std.exp().item()
            act_str += f" | Thr:{mean_thr:.2f}({min_thr:.2f}-{max_thr:.2f}) ThrStd:{thr_policy_std:.3f}"

        # Compact per-update line
        gems_str = f" gems={self.rollout_gem_pts}" if self.rollout_gem_pts else ""
        oob_str  = f" OOB={self.rollout_oob}" if self.rollout_oob else ""
        # Free-fall diagnostics — answers "is the penalty getting applied?".
        # FF=<frames>(-<penalty>) shows how many frames triggered it and total cost.
        # minVz= shows the most-negative vz seen — confirms the threshold is reached.
        ff_str = f" FF={self.rollout_freefall_frames}(-{self.rollout_freefall_penalty:.0f}) minVz={self.rollout_min_vz:.1f}"
        self.log(
            f"Upd {self.total_updates:4d} | "
            f"PL={stats['policy_loss']:.4f} VL={stats['value_loss']:.4f} "
            f"Ent={stats['entropy']:.3f} GN={stats['grad_norm']:.3f} CGN={stats['critic_grad_norm']:.3f} KL={stats['max_kl']:.4f} | "
            f"AvgRwd={avg_reward:.1f} AvgLen={avg_ep_len:.0f}{collapse_warn} |"
            f"{gems_str}{oob_str}{ff_str} Vstd={stats.get('value_std', 0.0):.1f}{dry_warn}{kl_warn}"
        )
        self.log(f"       {act_str}")

        # Push to live dashboard BEFORE resetting rollout counters
        # (snapshot reads rollout_gem_pts, rollout_oob, etc.)
        try:
            self.dashboard.push_snapshot(stats, avg_reward)
        except Exception as e:
            self.log(f"Dashboard push error: {e}")

        # Reset rollout counters
        self.rollout_gem_pts = 0
        self.rollout_oob = 0
        self.rollout_positive = 0
        self.rollout_steps = 0
        self.rollout_freefall_frames = 0
        self.rollout_freefall_penalty = 0.0
        self.rollout_min_vz = 0.0

        # Save checkpoint periodically
        if self.total_updates % self.save_interval == 0:
            path = f'models/checkpoints/update_{self.total_updates}.pth'
            self.save_model(path)

            # Save best model
            if avg_reward > self.best_avg_reward and len(self.episode_rewards) >= 10:
                self.best_avg_reward = avg_reward
                self.save_model('models/checkpoints/best.pth')
                self.log(f"  *** New best! avg_reward={avg_reward:.2f} ***")

            # Training summary every save_interval updates
            elapsed_hrs = (time.time() - self.run_start_time) / 3600
            gems_hr = self.total_gem_pts / max(elapsed_hrs, 1/3600)

            self.log(f"\n--- SUMMARY Upd {self.total_updates} | {elapsed_hrs:.2f}h | {self.total_steps:,} steps | {self.total_episodes} eps ---")
            laziness = avg_reward / gems_hr if gems_hr > 0 else 0
            log_std = torch.clamp(self.actor.log_std, self.actor.LOG_STD_MIN, self.actor.LOG_STD_MAX)
            policy_std_deg = log_std.exp().item() * 180 / 3.14159
            self.log(f"  Gems: {self.total_gem_pts}pts ({gems_hr:.0f}/hr) | OOB: {self.total_oob} | AvgRwd: {avg_reward:.1f} | Best: {self.best_avg_reward:.1f}")
            thr_log_std = torch.clamp(self.actor.throttle_log_std,
                                      self.actor.THROTTLE_LOG_STD_MIN,
                                      self.actor.THROTTLE_LOG_STD_MAX)
            thr_std = thr_log_std.exp().item()
            mean_thr = np.mean(self.recent_throttles) if self.recent_throttles else 0
            self.log(f"  AvgEpLen: {avg_ep_len:.0f} steps | Ent: {stats['entropy']:.3f} | GN: {stats['grad_norm']:.3f} | Lazy: {laziness:.2f} | Std: {policy_std_deg:.1f} | Thr: {mean_thr:.2f} ThrStd: {thr_std:.3f}")
            if self.recent_episode_gems:
                nonzero = sum(1 for g in self.recent_episode_gems if g > 0)
                self.log(f"  Last {len(self.recent_episode_gems)} eps: {nonzero} had gems")
            self.log(f"---")

        # Log to file
        self.log_stats(stats, avg_reward)

        # Clear buffer for next rollout
        self.buffer.clear()

    def save_model(self, path):
        """Save model checkpoint."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.trainer.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.trainer.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_updates': self.total_updates,
            'total_episodes': self.total_episodes,
            'best_avg_reward': self.best_avg_reward,
            'episode_rewards': list(self.episode_rewards),
            'throttle_floor_active': self.throttle_floor_active,
            'warmup_start_update': self.warmup_start_update,
        }, path)
        self.log(f"  Model saved to {path}")

    def log_stats(self, stats, avg_reward):
        """Log training statistics to CSV."""
        log_path = f'logs/training_{self.run_timestamp}.csv'

        # Write header if file doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write('update,total_steps,episodes,avg_reward,policy_loss,value_loss,entropy\n')

        with open(log_path, 'a') as f:
            f.write(f"{self.total_updates},{self.total_steps},{self.total_episodes},"
                    f"{avg_reward:.4f},{stats['policy_loss']:.6f},"
                    f"{stats['value_loss']:.6f},{stats['entropy']:.6f}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='PlatinumQuest PPO Training Server')
    parser.add_argument('--host', default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=8888, help='Server port')
    parser.add_argument('--rollout-size', type=int, default=2048, help='Steps per PPO update')
    parser.add_argument('--lr', type=float, default=3e-5, help='Actor learning rate (critic uses 1e-4)')
    parser.add_argument('--batch-size', type=int, default=256, help='Mini-batch size')
    parser.add_argument('--epochs', type=int, default=4, help='PPO epochs per update')
    parser.add_argument('--load', type=str, default=None, help='Load specific checkpoint (e.g. models/checkpoints/update_5070.pth)')
    parser.add_argument('--warmup', type=int, default=0,
                        help='Force a fresh critic-only warmup of N rollouts on this start (actor frozen). '
                             'Useful after changes that move the value targets (gamma, reward clip, bootstrap fix).')
    parser.add_argument('--action-repeat', type=int, default=None,
                        help='Override ACTION_REPEAT (ticks each decision is held; default 4)')

    args = parser.parse_args()

    model_path = None
    if args.load:
        # Explicit checkpoint specified
        if not os.path.exists(args.load):
            print(f"ERROR: Checkpoint not found: {args.load}")
            sys.exit(1)
        model_path = args.load
        print(f"Loading specified checkpoint: {args.load}")
    else:
        # Auto-resume from latest checkpoint if it exists
        import glob
        checkpoint_dir = 'models/checkpoints'
        update_files = glob.glob(f'{checkpoint_dir}/update_*.pth')

        if update_files:
            latest_file = max(update_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
            model_path = latest_file
            print(f"Resuming training from {latest_file}")
        elif os.path.exists(f'{checkpoint_dir}/best.pth'):
            model_path = f'{checkpoint_dir}/best.pth'
            print(f"Resuming training from best.pth")
        elif os.path.exists(f'{checkpoint_dir}/final.pth'):
            model_path = f'{checkpoint_dir}/final.pth'
            print(f"Resuming training from final.pth")
        else:
            print("Starting fresh training (no checkpoint found)")

    server = PPOServer(host=args.host, port=args.port, model_path=model_path)
    server.rollout_size = args.rollout_size
    server.batch_size = args.batch_size
    server.n_epochs = args.epochs
    server.trainer.actor_optimizer.param_groups[0]['lr'] = args.lr
    if args.action_repeat is not None:
        server.ACTION_REPEAT = max(1, args.action_repeat)
        server.log(f"ACTION_REPEAT overridden to {server.ACTION_REPEAT}")
    if args.warmup > 0:
        server.WARMUP_ROLLOUTS = args.warmup
        server.warmup_start_update = None   # (re)starts at the first update of this run
        server.log(f"Forced critic-only warmup: {args.warmup} rollouts")
    server.log(f"Action repeat: {server.ACTION_REPEAT} ticks/decision | gamma={server.gamma} | reward clip=+/-{server.REWARD_CLIP:.0f}")


    # Signal handler
    def signal_handler(sig, frame):
        server.log('\nReceived interrupt, saving and shutting down...')
        server.running = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        server.run()
    except KeyboardInterrupt:
        server.log("\nShutting down...")
    finally:
        server.socket.close()
        server.logger.close()


if __name__ == '__main__':
    main()
