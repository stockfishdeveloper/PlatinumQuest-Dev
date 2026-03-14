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
    """Policy network for PPO with continuous 2D direction + throttle output.

    Action: (dx, dy, throttle) where:
      - (dx, dy): unit-circle direction vector (0,1)=forward, (1,0)=right
      - throttle: [0.15, 1] force magnitude (floor prevents sit-still collapse)
    Buffer stores (dx, dy, throttle_logit) — 3 dims.
    2D representation eliminates the scalar angle wrap discontinuity at +/-pi
    that caused systematic failures in 1/4 of camera headings.

    Separate from Critic to eliminate gradient interference: critic's high-variance
    value loss no longer contaminates policy gradient through shared weights.
    """

    LOG_STD_MIN = -2.0    # exp(-2.0) ~= 0.14 (tight directional spread)
    LOG_STD_MAX = 1.0     # exp(1.0)  ~= 2.7  (wide exploration)
    THROTTLE_LOG_STD_MIN = -3.0   # exp(-3.0) ~= 0.05 (tight throttle control)
    THROTTLE_FLOOR = 0.15         # Minimum throttle to prevent "sit still" collapse
    THROTTLE_LOG_STD_MAX = 0.0    # exp(0.0)  ~= 1.0  (wide exploration)

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
        self.actor_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # (dx, dy) normalized to unit circle
        )

        # Throttle head: outputs raw logit, passed through sigmoid -> [0, 1]
        self.throttle_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # raw logit -> sigmoid -> throttle
        )
        # Initialize throttle bias to +2.0 so sigmoid(2.0) ~= 0.88 (mostly full throttle)
        # This prevents the model from starting with 50% throttle which would halve gem collection
        nn.init.constant_(self.throttle_head[-1].bias, 2.0)

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
        return mean_xy, throttle_logit

    def get_action(self, state, deterministic=False):
        """Get action from state.

        Returns:
            action_for_buffer: (dx, dy, throttle_logit) - stored in buffer for evaluate_actions
            action_for_game: (dx, dy, throttle) - direction + throttle for joystick
            log_prob: joint log probability
        """
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0)
            mean_xy, throttle_logit = self.forward(state)

            # Direction (2D)
            dir_dist = self._get_direction_dist(mean_xy)
            if deterministic:
                norm = torch.norm(mean_xy, dim=1, keepdim=True).clamp(min=1e-6)
                direction = mean_xy / norm  # (1, 2)
            else:
                direction = dir_dist.sample()  # (1, 2)
            # Sum log_prob over both dims for joint direction probability
            dir_log_prob = dir_dist.log_prob(direction).sum(dim=1)

            # Throttle (sample in logit space, then sigmoid)
            throttle_dist = self._get_throttle_dist(throttle_logit)
            if deterministic:
                throttle_logit_sample = throttle_logit.squeeze(-1)
            else:
                throttle_logit_sample = throttle_dist.sample()
            throttle_log_prob = throttle_dist.log_prob(throttle_logit_sample)
            throttle_raw = torch.sigmoid(throttle_logit_sample)
            # Remap [0,1] -> [THROTTLE_FLOOR, 1] to prevent sit-still collapse
            throttle = self.THROTTLE_FLOOR + (1 - self.THROTTLE_FLOOR) * throttle_raw

            # Joint log prob = direction + throttle
            log_prob = dir_log_prob + throttle_log_prob

        dx_val = direction[0, 0].item()
        dy_val = direction[0, 1].item()
        throttle_logit_val = throttle_logit_sample.item()
        throttle_val = throttle.item()
        return (dx_val, dy_val, throttle_logit_val), (dx_val, dy_val, throttle_val), log_prob.item()

    def evaluate_actions(self, states, actions):
        """Evaluate log_probs and entropy for stored actions.
        actions: (N, 3) tensor with columns [dx, dy, throttle_logit]."""
        mean_xy, throttle_logit = self.forward(states)

        # Direction (2D) — actions[:, 0:2] are stored (dx, dy)
        dir_dist = self._get_direction_dist(mean_xy)
        dir_log_prob = dir_dist.log_prob(actions[:, 0:2]).sum(dim=1)
        dir_entropy = dir_dist.entropy().sum(dim=1)

        # Throttle (actions[:, 2] is stored in logit space)
        throttle_dist = self._get_throttle_dist(throttle_logit)
        throttle_log_prob = throttle_dist.log_prob(actions[:, 2])
        throttle_entropy = throttle_dist.entropy()

        # Joint: sum of independent log probs and entropies
        return dir_log_prob + throttle_log_prob, dir_entropy + throttle_entropy

    @staticmethod
    def action_to_joystick(dx, dy, throttle):
        """Convert 2D direction + throttle to joystick axes (fwd, back, left, right).

        Convention: dx>0 = right, dy>0 = forward.
        Direction is normalized to unit length so magnitude is controlled
        purely by throttle. Sampled (dx,dy) from the 2D Gaussian can have
        magnitude != 1, which would otherwise leak into joystick values.
        Throttle scales the magnitude (0 = no force, 1 = full force).
        Values rounded to 6 decimal places to avoid scientific notation.
        """
        import math
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

        return fwd, back, left, right


class Critic(nn.Module):
    """Value network for PPO. Separate from Actor to prevent gradient interference.

    The critic's value loss is high-variance (reward spikes from gem collection),
    so keeping it separate ensures its gradients never corrupt the policy weights.
    Uses a higher learning rate than the actor.
    """

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

    def forward(self, state):
        return self.net(state)

    def get_value(self, state):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0)
            return self.forward(state).item()


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

    def compute_returns_and_advantages(self, gamma=0.99, lam=0.95):
        """Compute GAE advantages and discounted returns."""
        advantages = []
        returns = []
        gae = 0

        # Work backwards through experience
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0  # Terminal
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

    def get_batches(self, batch_size, gamma=0.99, lam=0.95):
        """Get training batches with computed advantages."""
        returns, advantages = self.compute_returns_and_advantages(gamma, lam)

        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))  # (N, 3): [dx, dy, throttle_logit]
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
                 actor_lr=1.5e-5, critic_lr=5e-5,
                 clip_epsilon=0.2, entropy_coef=0.01,
                 max_grad_norm=1.0, vf_clip=20.0, target_kl=1.5):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.vf_clip = vf_clip
        self.target_kl = target_kl  # KL early stopping: fires at 1.5x (catches destructive updates)

    def update(self, buffer, n_epochs=4, batch_size=64, gamma=0.99, lam=0.95):
        """Run PPO update with independent actor and critic steps."""
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_grad_norm = 0
        total_critic_grad_norm = 0
        n_updates = 0
        max_kl = 0.0
        kl_early_stopped = False

        for epoch in range(n_epochs):
            if kl_early_stopped:
                break
            for states, actions, old_log_probs, returns, advantages, old_values in \
                    buffer.get_batches(batch_size, gamma, lam):

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

                # --- Critic update ---
                values = self.critic(states).squeeze(-1)
                values_clipped = old_values + torch.clamp(values - old_values, -self.vf_clip, self.vf_clip)
                vf_loss1 = (values - returns) ** 2
                vf_loss2 = (values_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                critic_grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in self.critic.parameters()
                    if p.grad is not None
                ) ** 0.5
                total_critic_grad_norm += critic_grad_norm
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
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
        log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        self.trainer = PPOTrainer(self.actor, self.critic, vf_clip=20.0,
                                   target_kl=2.667, entropy_coef=0.005)
        self.buffer = RolloutBuffer()

        # Training config
        # 2048 steps ~= 11% of one episode (~18,875 steps).
        self.rollout_size = 2048
        self.n_epochs = 4
        self.batch_size = 256  # 2048/256 = 8 mini-batches per epoch
        self.gamma = 0.99
        self.lam = 0.95
        self.save_interval = 10  # Save every N updates

        # Mild reward scaling: 0.01 crushed all signal (VLoss=0.0001, GradNorm=0.06),
        # 1.0 caused wild VLoss spikes (163.6) and gradient clipping threw away 99%
        # of gradient info. 0.1 is the sweet spot: gem=+20, OOB=-2.5, shaping=±0.05-0.3/step.
        # Per-step rewards are clipped to [-5, 5] after scaling to prevent VLoss explosions.
        self.reward_scale = 0.1

        # === Reward parameters (all reward computation is in Python) ===
        self.GEM_REWARD = 200       # Per gem-point bonus (+200 raw, +20 scaled)
        # self.OOB_PENALTY_INIT = 10   # Starting OOB penalty (gentle for early training)
        # self.OOB_PENALTY_FULL = 25   # Full OOB penalty (activated at 30 avg gems/game)
        # self.OOB_PENALTY = self.OOB_PENALTY_INIT
        self.OOB_PENALTY = 0             # Disabled for fresh training
        # self.GEM_GAP_PENALTY_K = 0.001  # Ramping time penalty: cost = k * steps_since_gem
        self.GEM_GAP_PENALTY_K = 0        # Disabled for fresh training
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

            # Override learning rates for fine-tuning phase
            self.trainer.actor_optimizer.param_groups[0]['lr'] = 1.5e-5
            self.trainer.critic_optimizer.param_groups[0]['lr'] = 5e-5
            self.log(f"  LR override: actor=1.5e-5, critic=5e-5")

            # Restore training progress
            self.total_steps = checkpoint.get('total_steps', 0)
            self.total_updates = checkpoint.get('total_updates', 0)
            self.total_episodes = checkpoint.get('total_episodes', 0)
            self.best_avg_reward = checkpoint.get('best_avg_reward', -float('inf'))
            saved_rewards = checkpoint.get('episode_rewards', [])
            self.episode_rewards = deque(saved_rewards, maxlen=100)

            with torch.no_grad():
                std_deg = self.actor.log_std.exp().item() * 180 / 3.14159
                thr_std = self.actor.throttle_log_std.exp().item()
                self.log(f"  PolicyStd: {std_deg:.1f} deg (bounds: {self.actor.LOG_STD_MIN:.1f} to {self.actor.LOG_STD_MAX:.1f})")
                self.log(f"  ThrottleStd: {thr_std:.3f} (bounds: {self.actor.THROTTLE_LOG_STD_MIN:.1f} to {self.actor.THROTTLE_LOG_STD_MAX:.1f})")

            self.log(f"Model loaded successfully!")
            self.log(f"Resuming from: {self.total_steps} steps, {self.total_updates} updates, {self.total_episodes} episodes")
            self.log(f"Best avg reward restored: {self.best_avg_reward:.2f}")

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

        # Per-rollout counters (reset after each PPO update)
        self.rollout_gem_pts = 0   # gem points collected this rollout
        self.rollout_oob = 0       # OOB events this rollout
        self.rollout_positive = 0  # steps with positive reward this rollout
        self.rollout_steps = 0     # total steps this rollout
        # (action counts removed — continuous actions tracked via recent_actions deque)

        # Lifetime counters
        self.total_gem_pts = 0     # gem points across entire run
        self.total_oob = 0         # OOB events across entire run

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

        # Near-miss and dwell-time tracking (per game)
        # "Near" = within 2 marble diameters of nearest gem (~0.8 units)
        self.NEAR_GEM_THRESHOLD = 0.8    # 2 marble diameters (radius ~0.2 * 4)
        self.near_gem = False             # currently within threshold
        self.game_near_misses = 0         # count: entered near zone, left without pickup
        self.game_dwell_steps = 0         # total steps spent within threshold
        self.recent_near_misses = deque(maxlen=100)    # per-game near-miss count
        self.recent_dwell_steps = deque(maxlen=100)    # per-game dwell steps

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

                    # Send action back
                    action_str = ','.join(map(str, action)) + '\n'
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
            self.skip_potential_steps = self.GRACE_PERIOD
            # Record gap penalty for this interval before resetting
            gap_penalty = self.GEM_GAP_PENALTY_K * self.steps_since_gem * (self.steps_since_gem - 1) / 2
            self.game_gap_penalty_sum += gap_penalty
            self.game_gem_pickups += 1
            self.steps_since_gem = 0  # Reset ramping penalty on gem pickup

        # 2. Distance shaping: potential-based P(d) = C1/(1+d/R1) + C2/(1+d/R2)
        nearest_dist = raw_obs[10] if len(raw_obs) > 10 else -1
        if nearest_dist > 0 and nearest_dist < self.DIST_MAX_VALID:
            if self.skip_potential_steps > 0:
                self.skip_potential_steps -= 1
            else:
                new_potential = (self.SHAPING_COEFF_1 / (1 + nearest_dist / self.SHAPING_RANGE_1)
                               + self.SHAPING_COEFF_2 / (1 + nearest_dist / self.SHAPING_RANGE_2))
                old_potential = (self.SHAPING_COEFF_1 / (1 + self.last_nearest_gem_dist / self.SHAPING_RANGE_1)
                               + self.SHAPING_COEFF_2 / (1 + self.last_nearest_gem_dist / self.SHAPING_RANGE_2))
                reward += new_potential - old_potential
            self.last_nearest_gem_dist = nearest_dist

        # 3. Ramping time penalty: grows with steps since last gem pickup.
        reward -= self.GEM_GAP_PENALTY_K * self.steps_since_gem
        self.steps_since_gem += 1

        # 4. OOB penalty
        if oob:
            reward -= self.OOB_PENALTY

        return reward

    def process_message(self, message):
        """Process a message from the game, return action."""
        try:
            # Parse: obs_json|gem_delta|oob|done
            parts = message.split('|')

            if len(parts) != 4:
                if self.total_steps < 3:
                    self.log(f"Malformed message (expected 4 parts, got {len(parts)}): {message[:100]}")
                return [0, 0, 0, 0]

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
                self.recent_game_gems.append(total_game_gems)
                if total_game_gems > self.best_game_gems:
                    self.best_game_gems = total_game_gems
                avg_gap = self.game_gap_penalty_sum / max(self.game_gem_pickups, 1)
                self.recent_avg_gap_penalty.append(round(avg_gap, 2))
                self.recent_near_misses.append(self.game_near_misses)
                self.recent_dwell_steps.append(self.game_dwell_steps)
                self.log(f"[GAME END] total gems this game: {total_game_gems}pts (best: {self.best_game_gems}) avg_gap_penalty: {avg_gap:.1f} near_misses: {self.game_near_misses} dwell_steps: {self.game_dwell_steps}")
                self.game_gem_pts = 0  # Reset accumulators for next game
                self.game_gap_penalty_sum = 0.0
                self.game_gem_pickups = 0
                self.game_near_misses = 0
                self.game_dwell_steps = 0
                self.near_gem = False
                return [0, 0, 0, 0]

            # Track no-gem steps (sentinel distance at index 12 — nearest gem dist)
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

            # Compute reward in Python (all tunable params live here now)
            reward = self.compute_reward(obs, gem_delta, oob)

            # Normalize observation (must happen AFTER compute_reward uses raw values)
            obs_array = self.normalize_obs(np.array(obs, dtype=np.float32))

            # Build augmented observation with frame history.
            # Append current pos+vel (normalized) to ring buffer, then pull
            # snapshots at t-FRAME_SKIP, t-2*FRAME_SKIP, etc.
            current_posvel = obs_array[0:6].copy()  # 6 dims: pos(3) + vel(3)
            self.frame_history.append(current_posvel)

            history_frames = []
            for i in range(1, self.FRAME_HISTORY_COUNT + 1):
                # Current frame is at index len-1. Frame from i*FRAME_SKIP ago
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

            # Action: query model. Returns buffer action (dx, dy, throttle_logit),
            # game action (dx, dy, throttle), and joint log_prob.
            action_buf, action_game, log_prob = self.actor.get_action(obs_augmented)
            value = self.critic.get_value(obs_augmented)
            dx, dy, throttle = action_game
            import math
            angle = math.atan2(dx, dy)  # For logging/display only
            self.recent_actions.append(angle)
            self.recent_throttles.append(throttle)

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
            if reward > 0.1:
                self.rollout_positive += 1
            self.rollout_steps += 1
            self.episode_step += 1

            # Store experience (scale reward to keep critic targets small).
            # Clip scaled reward to [-20, 20] so gem spikes (+200 raw → +20 scaled)
            # are captured fully, while OOB (-25 raw → -2.5) remains distinct.
            # At 0.1 scale, ±20 allows raw rewards up to ±200 unclipped.
            scaled_reward = np.clip(reward * self.reward_scale, -20.0, 20.0)
            self.buffer.add(obs_augmented, action_buf, scaled_reward, value, log_prob, done)
            self.total_steps += 1
            self.current_episode_reward += reward

            # Episode end
            if done:
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
                    self.log(f"  *** SHORT EPISODE ({self.episode_step} steps) — flood bug may still be active ***")

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

            # PPO update when buffer is full
            if len(self.buffer) >= self.rollout_size:
                self.run_ppo_update()

            # Convert 2D direction + throttle to analog joystick axes (F,B,L,R)
            return list(Actor.action_to_joystick(dx, dy, throttle))

        except Exception as e:
            if self.total_steps < 5:
                self.log(f"Error processing message: {e}")
                self.log(f"Message (first 200 chars): {message[:200]}")
                import traceback
                traceback.print_exc()
            return [0, 0, 0, 0]

    def run_ppo_update(self):
        """Run PPO training update."""
        self.actor.train()
        self.critic.train()

        stats = self.trainer.update(
            self.buffer,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            gamma=self.gamma,
            lam=self.lam,
        )
        self.total_updates += 1
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
        self.log(
            f"Upd {self.total_updates:4d} | "
            f"PL={stats['policy_loss']:.4f} VL={stats['value_loss']:.4f} "
            f"Ent={stats['entropy']:.3f} GN={stats['grad_norm']:.3f} CGN={stats['critic_grad_norm']:.3f} KL={stats['max_kl']:.4f} | "
            f"AvgRwd={avg_reward:.1f} AvgLen={avg_ep_len:.0f}{collapse_warn} |"
            f"{gems_str}{oob_str}{dry_warn}{kl_warn}"
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
        }, path)
        self.log(f"  Model saved to {path}")

    def log_stats(self, stats, avg_reward):
        """Log training statistics to CSV."""
        log_path = 'logs/training_log.csv'

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
