"""
PPO Training Server for PlatinumQuest Hunt Mode

Receives observations from the game via TCP socket, computes rewards
in Python, runs PPO training updates, and sends actions back.

Protocol (game -> server):
    Each message is newline-delimited:
    "[obs_array]|gem_delta|oob|done"

    - obs_array: 61-float JSON array (see observer.cs for layout)
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
    """Policy network for PPO with continuous angle output.

    Action: a single angle in radians [0, 2pi) representing movement direction.
    Always applied at full magnitude. No idle action.
    Convention: 0 = forward (+Y), pi/2 = right (+X), pi = backward, 3pi/2 = left.

    Separate from Critic to eliminate gradient interference: critic's high-variance
    value loss no longer contaminates policy gradient through shared weights.
    """

    LOG_STD_MIN = -2.0    # exp(-2.0) ~= 0.14 rad ~= 7.7 deg (safety floor)
    LOG_STD_MAX = 1.0     # exp(1.0)  ~= 2.7 rad  ~= 156 deg (safety ceiling)

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

        # Learnable log-std (state-independent, single scalar for angular spread)
        # Init to -1.5 -> exp(-1.5) ~= 0.22 rad ~= 12.8 deg
        self.log_std = nn.Parameter(torch.full((1,), -1.5))

    def _get_dist(self, mean_xy):
        mean_angle = torch.atan2(mean_xy[:, 0], mean_xy[:, 1])  # atan2(x, y) so 0=forward
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return torch.distributions.Normal(mean_angle, log_std.exp())

    def forward(self, state):
        return self.actor_mean(self.features(state))

    def get_action(self, state, deterministic=False):
        """Get action from state. Returns (angle, log_prob)."""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0)
            mean_xy = self.forward(state)
            dist = self._get_dist(mean_xy)
            if deterministic:
                angle = torch.atan2(mean_xy[:, 0], mean_xy[:, 1])
            else:
                angle = dist.sample()
            log_prob = dist.log_prob(angle)
        return angle.item(), log_prob.item()

    def evaluate_actions(self, states, actions):
        """Evaluate log_probs and entropy for stored actions."""
        mean_xy = self.forward(states)
        dist = self._get_dist(mean_xy)
        return dist.log_prob(actions), dist.entropy()

    @staticmethod
    def angle_to_joystick(angle):
        """Convert angle (radians) to joystick axes (fwd, back, left, right).

        Convention: 0 = forward, pi/2 = right, pi = backward, 3pi/2 = left.
        Always full magnitude. Values rounded to 6 decimal places to avoid
        scientific notation (e.g. 1.2e-16) which TorqueScript may not parse.
        """
        import math
        move_x = math.sin(angle)  # positive = right
        move_y = math.cos(angle)  # positive = forward

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
        actions = torch.FloatTensor(np.array(self.actions))  # Continuous angle (radians)
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

        # Separate actor and critic networks — eliminates gradient interference.
        # The critic's high-variance value loss no longer corrupts policy weights.
        # Actor LR is lower (3e-5) for stable policy updates; critic LR higher (1e-4)
        # since value regression can afford bigger steps.
        self.actor = Actor(obs_dim=61)
        self.critic = Critic(obs_dim=61)
        self.trainer = PPOTrainer(self.actor, self.critic, vf_clip=20.0)
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
        self.OOB_PENALTY = 25       # Out-of-bounds penalty (-25 raw, -2.5 scaled)
        self.TIME_PENALTY = 0.02    # Per-step efficiency pressure
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
            # Handle architecture mismatch (e.g. old checkpoint had different network shape)
            # Load actor weights. Old checkpoints used a shared ActorCritic —
            # map compatible keys across (features.*, actor_mean.*, log_std).
            # Critic is always reset to reinit: old value estimates are stale
            # and would cause a gradient flood on the first update.
            saved_state = checkpoint.get('model_state_dict', checkpoint.get('actor_state_dict', {}))
            actor_state = self.actor.state_dict()
            filtered = {}
            for key, val in saved_state.items():
                if key in actor_state:
                    if val.shape == actor_state[key].shape:
                        filtered[key] = val
                    else:
                        self.log(f"  Shape mismatch for {key}: checkpoint={val.shape} vs actor={actor_state[key].shape} — skipping")
            self.actor.load_state_dict(filtered, strict=False)
            # Critic always starts fresh — don't restore optimizer states either.

            # Restore training progress
            self.total_steps = checkpoint.get('total_steps', 0)
            self.total_updates = checkpoint.get('total_updates', 0)
            self.total_episodes = checkpoint.get('total_episodes', 0)
            self.best_avg_reward = checkpoint.get('best_avg_reward', -float('inf'))
            saved_rewards = checkpoint.get('episode_rewards', [])
            self.episode_rewards = deque(saved_rewards, maxlen=100)

            with torch.no_grad():
                std_deg = self.actor.log_std.exp().item() * 180 / 3.14159
                self.log(f"  PolicyStd: {std_deg:.1f} deg (learnable, bounds: {self.actor.LOG_STD_MIN:.1f} to {self.actor.LOG_STD_MAX:.1f})")
            self.log(f"  Critic reinitialised from scratch (stale value estimates discarded)")

            self.log(f"Model loaded successfully!")
            self.log(f"Resuming from: {self.total_steps} steps, {self.total_updates} updates, {self.total_episodes} episodes")
            self.log(f"Best avg reward restored: {self.best_avg_reward:.2f}")

        self.actor.train()
        self.critic.train()
        self._session_start_step = self.total_steps

        # Recent actions (for angle stats display and dashboard)
        self.recent_actions = deque(maxlen=200)

        # Per-episode counters (reset on done)
        self.episode_gem_pts = 0   # gem points collected this episode
        self.episode_oob = 0       # OOB events this episode
        self.episode_step = 0      # step index within the current episode

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

        Observation layout (61 dims total):
          [0-12]  Self state (13 dims)
          [13-37] 5 nearest gems × 5 dims = 25 dims
          [38-55] 3 opponents × 6 dims   = 18 dims
          [56-60] Game state              =  5 dims
        """
        # Self state (indices 0-12) — no sentinels in self state
        obs[0:3]  /= 100.0   # Position (world units → ~[-1,1] for typical maps)
        obs[3:6]  /= 20.0    # Velocity (camera-relative: x=right, y=forward, z=up)
        # Wrap yaw to ±pi before normalizing (engine may return 0-2pi when AI doesn't move camera)
        while obs[6] > 3.14159:
            obs[6] -= 6.28318
        while obs[6] < -3.14159:
            obs[6] += 6.28318
        obs[6]    /= 3.14159 # Camera yaw  (radians, ±pi → ±1)
        obs[7]    /= 1.5708  # Camera pitch (radians, ±pi/2 → ±1)
        # obs[8]: collision radius (~0.2), already small
        # obs[9]: powerup id (-1..5), already small
        # obs[10]: megaMarbleActive (0/1)
        obs[11]   /= 20.0    # megaMarbleTimeRemaining (0-20 s → 0-1)
        obs[12]   /= 20.0    # powerupTimerRemaining   (0-20 s → 0-1)

        # Gems (indices 13-37: 5 gems × 5 dims = x, y, z, value, distance)
        # Fix sentinel values: -999 meant "no gem" but replacing with 0 told the
        # network a gem was AT the marble. Instead, mark absent gems as far away.
        gem_base = 13
        for i in range(5):
            b = gem_base + i * 5
            if obs[b+4] < -500:  # distance is sentinel → gem absent
                obs[b:b+3] = 0.0  # no directional info
                obs[b+3]   = 0.0  # no value
                obs[b+4]   = 1.0  # max normalized distance (far away)
            else:
                dist = obs[b+4]
                if dist > 0.01:
                    obs[b:b+3] /= dist  # Unit direction vector (always magnitude ~1)
                else:
                    obs[b:b+3] = 0.0    # On top of gem, no direction needed
                obs[b+3]   /= 5.0      # Gem value (1-5 → 0.2-1.0)
                obs[b+4]   /= 100.0    # Distance (0-100+ → 0-1+)

        # Opponents (indices 38-55: 3 opponents × 6 dims)
        opp_base = 38
        for i in range(3):
            b = opp_base + i * 6
            if obs[b] < -500:  # sentinel → opponent absent
                obs[b:b+3]   = 0.0  # no directional info
                obs[b+3:b+5] = 0.0  # no velocity
                obs[b+5]     = 0.0  # not mega
            else:
                obs[b:b+3]   /= 100.0  # Relative x, y, z positions
                obs[b+3:b+5] /= 20.0   # Relative velocities
                # obs[b+5]: isMega (0/1)

        # Game state (indices 56-60)
        obs[56] /= 300000.0   # timeElapsed    (5-min hunt = 300,000 ms → 0-1)
        obs[57] /= 300000.0   # timeRemaining
        obs[58] /= 100.0      # myGemScore
        obs[59] /= 100.0      # opponentBestScore
        obs[60] /= 50.0       # gemsRemaining

        # Safety clip: catch any remaining outliers.
        obs = np.clip(obs, -2.0, 2.0)

        return obs

    def compute_reward(self, raw_obs, gem_delta, oob):
        """Compute reward from raw game facts. Mirrors old mlAgent.cs::computeReward exactly.

        Args:
            raw_obs: raw observation list (before normalization), index 17 = nearest gem distance
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

        # 2. Distance shaping: potential-based P(d) = C1/(1+d/R1) + C2/(1+d/R2)
        nearest_dist = raw_obs[17] if len(raw_obs) > 17 else -1
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

        # 3. Time penalty
        reward -= self.TIME_PENALTY

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
                self.log(f"[GAME END] total gems this game: {total_game_gems}pts (best: {self.best_game_gems})")
                self.game_gem_pts = 0  # Reset accumulator for next game
                return [0, 0, 0, 0]

            # Track no-gem steps (sentinel distance at index 17)
            raw_gem0_dist = obs[17] if len(obs) > 17 else -1
            if raw_gem0_dist < -500:
                if self.no_gem_steps == 0:
                    self.no_gem_events += 1
                self.no_gem_steps += 1
                self.total_no_gem_steps += 1
                self.episode_no_gem_steps += 1
            elif self.no_gem_steps > 0:
                self.no_gem_steps = 0

            # Compute reward in Python (all tunable params live here now)
            reward = self.compute_reward(obs, gem_delta, oob)

            # Normalize observation (must happen AFTER compute_reward uses raw values)
            obs_array = self.normalize_obs(np.array(obs, dtype=np.float32))

            # Action repeat: query model every N frames, reuse last action otherwise.
            action, log_prob = self.actor.get_action(obs_array)
            value = self.critic.get_value(obs_array)
            self.recent_actions.append(action)

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
            self.buffer.add(obs_array, action, scaled_reward, value, log_prob, done)
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

            # PPO update when buffer is full
            if len(self.buffer) >= self.rollout_size:
                self.run_ppo_update()

            # Convert continuous angle to analog joystick axes (F,B,L,R)
            return list(Actor.angle_to_joystick(action))

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

        # Action distribution (continuous angle)
        import math
        recent = list(self.recent_actions)[-min(200, len(self.recent_actions)):]
        if recent:
            angles_deg = [((a * 180 / math.pi) % 360) for a in recent]
            mean_deg = sum(angles_deg) / len(angles_deg)
            std_deg = (sum((d - mean_deg)**2 for d in angles_deg) / len(angles_deg)) ** 0.5
            log_std = torch.clamp(self.actor.log_std, self.actor.LOG_STD_MIN, self.actor.LOG_STD_MAX)
            policy_std_deg = log_std.exp().item() * 180 / math.pi
            act_str = f"MeanAngle:{mean_deg:.0f}° StdDev:{std_deg:.0f}° PolicyStd:{policy_std_deg:.1f}°"
        else:
            act_str = "no actions yet"

        # Compact per-update line
        gems_str = f" gems={self.rollout_gem_pts}" if self.rollout_gem_pts else ""
        oob_str  = f" OOB={self.rollout_oob}" if self.rollout_oob else ""
        self.log(
            f"Upd {self.total_updates:4d} | "
            f"PL={stats['policy_loss']:.4f} VL={stats['value_loss']:.4f} "
            f"Ent={stats['entropy']:.3f} GN={stats['grad_norm']:.3f} KL={stats['max_kl']:.4f} | "
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
            self.log(f"  AvgEpLen: {avg_ep_len:.0f} steps | Ent: {stats['entropy']:.3f} | GN: {stats['grad_norm']:.3f} | Lazy: {laziness:.2f} | Std: {policy_std_deg:.1f}°")
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
