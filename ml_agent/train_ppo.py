"""
PPO Training Server for PlatinumQuest Hunt Mode

Receives observations + rewards from the game via TCP socket,
runs PPO training updates, and sends actions back.

Protocol (game -> server):
    Each message is newline-delimited:
    "[obs_array]|reward|done"

    Example: "[1.0,2.0,3.0,...,286 floats]|0.5|0"

Protocol (server -> game):
    "0,1,0,1,0,0\n"  (6 comma-separated binary actions)

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
from torch.distributions import Bernoulli
import time
import signal
import sys
import os
from collections import deque
from datetime import datetime

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
# Neural Network (Actor-Critic)
# ============================================================================

class ActorCritic(nn.Module):
    """Policy and value network for PPO."""

    def __init__(self, obs_dim=61, action_dim=6):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Actor head (action probabilities)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        # Critic head (state value)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action(self, state, deterministic=False):
        """Get action from state, return action, log_prob, value."""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0)

            action_logits, value = self.forward(state)
            probs = torch.sigmoid(action_logits)

            if deterministic:
                actions = (probs > 0.5).int()
            else:
                dist = Bernoulli(probs)
                actions = dist.sample().int()

            # Compute log probability
            dist = Bernoulli(probs)
            log_prob = dist.log_prob(actions.float()).sum(dim=-1)

        return actions.squeeze(0).numpy(), log_prob.item(), value.item()

    def evaluate_actions(self, states, actions):
        """Evaluate actions for PPO update."""
        action_logits, values = self.forward(states)
        probs = torch.sigmoid(action_logits)
        dist = Bernoulli(probs)

        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy


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
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(self.log_probs)
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
            )

    def __len__(self):
        return len(self.states)


# ============================================================================
# PPO Trainer
# ============================================================================

class PPOTrainer:
    """Proximal Policy Optimization trainer."""

    def __init__(self, model, lr=1e-4, clip_epsilon=0.2, value_coef=0.5,
                 entropy_coef=0.03, max_grad_norm=0.5):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def update(self, buffer, n_epochs=4, batch_size=64, gamma=0.99, lam=0.95):
        """Run PPO update on collected experience."""
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_grad_norm = 0
        n_updates = 0

        for epoch in range(n_epochs):
            for states, actions, old_log_probs, returns, advantages in \
                    buffer.get_batches(batch_size, gamma, lam):

                # Evaluate current policy
                new_log_probs, values, entropy = self.model.evaluate_actions(states, actions)

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, returns)

                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()

                # Measure gradient norm before clipping (useful for diagnosing exploding gradients)
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in self.model.parameters()
                    if p.grad is not None
                ) ** 0.5
                total_grad_norm += grad_norm

                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'grad_norm': total_grad_norm / max(n_updates, 1),
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

        # Model and trainer
        self.model = ActorCritic(obs_dim=61, action_dim=6)
        self.trainer = PPOTrainer(self.model)
        self.buffer = RolloutBuffer()

        # Training config
        self.rollout_size = 2048  # Steps before each PPO update — large enough to usually contain a full episode
        self.n_epochs = 4
        self.batch_size = 64
        self.gamma = 0.99
        self.lam = 0.95
        self.save_interval = 10  # Save every N updates

        # Reward scaling: divide rewards by this before storing in buffer.
        # Gem collection = +100 raw → +1.0 scaled. OOB = -100 → -1.0 scaled.
        # This keeps critic targets in a small range, preventing the
        # critic from diverging and flooding the shared network with
        # enormous gradients that collapse the actor.
        self.reward_scale = 0.01

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
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # Don't restore optimizer state — its momentum/variance was built at the
            # old value scale and causes bad updates on resume.

            # Reset the critic head so it relearns from the preserved shared features.
            # The critic saves a wildly wrong value estimate (e.g. -2849) that causes
            # a massive gradient flood into the shared layers on the first update,
            # corrupting the actor and collapsing the policy every time we reload.
            # The shared backbone and actor are untouched — the good policy is preserved.
            for layer in self.model.critic:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            # Restore training progress
            self.total_steps = checkpoint.get('total_steps', 0)
            self.total_updates = checkpoint.get('total_updates', 0)
            self.total_episodes = checkpoint.get('total_episodes', 0)
            self.best_avg_reward = checkpoint.get('best_avg_reward', -float('inf'))
            saved_rewards = checkpoint.get('episode_rewards', [])
            self.episode_rewards = deque(saved_rewards, maxlen=100)

            self.log(f"Model loaded successfully!")
            self.log(f"Resuming from: {self.total_steps} steps, {self.total_updates} updates, {self.total_episodes} episodes")
            self.log(f"Best avg reward restored: {self.best_avg_reward:.2f}")

        self.model.train()

        # Recent actions (for entropy-collapse detection)
        self.recent_actions = deque(maxlen=10)

        # Per-episode counters (reset on done)
        self.episode_gem_pts = 0   # gem points collected this episode
        self.episode_oob = 0       # OOB events this episode
        self.episode_step = 0      # step index within the current episode

        # Per-rollout counters (reset after each PPO update)
        self.rollout_gem_pts = 0   # gem points collected this rollout
        self.rollout_oob = 0       # OOB events this rollout
        self.rollout_positive = 0  # steps with positive reward this rollout
        self.rollout_steps = 0     # total steps this rollout
        self.rollout_action_sums = np.zeros(6)  # cumulative action activations this rollout

        # Lifetime counters
        self.total_gem_pts = 0     # gem points across entire run
        self.total_oob = 0         # OOB events across entire run

        # Entropy tracking for collapse detection
        self.entropy_history = deque(maxlen=10)

        # Run start time (for gems/hr)
        self.run_start_time = time.time()

        # Dry-rollout streak (rollouts with zero gems collected)
        self.dry_rollouts = 0

        # Rolling 20-episode gem points (for gems/episode trend)
        self.recent_episode_gems = deque(maxlen=20)

        # Socket setup
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.settimeout(1.0)
        self.socket.bind((self.host, self.port))

        # Create directories
        os.makedirs('models/checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

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
                    print("Game disconnected")
                    break

                buffer_str += data

                while '\n' in buffer_str:
                    line, buffer_str = buffer_str.split('\n', 1)
                    line = line.strip()

                    if not line:
                        continue

                    # Parse message: obs_json|reward|done
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
        # Replace -999 sentinels (absent gems/opponents) with 0 before scaling.
        obs = np.where(obs < -500, 0.0, obs)

        # Self state (indices 0-12)
        obs[0:3]  /= 100.0   # Position (world units → ~[-1,1] for typical maps)
        obs[3:6]  /= 20.0    # Velocity (max ~20 units/sec)
        obs[6]    /= 360.0   # Camera yaw  (0-360 → 0-1)
        obs[7]    /= 90.0    # Camera pitch (-90..90 → -1..1)
        # obs[8]: collision radius (~0.2), already small
        # obs[9]: powerup id (-1..5), already small
        # obs[10]: megaMarbleActive (0/1)
        obs[11]   /= 20.0    # megaMarbleTimeRemaining (0-20 s → 0-1)
        obs[12]   /= 20.0    # powerupTimerRemaining   (0-20 s → 0-1)

        # Gems (indices 13-37: 5 gems × 5 dims = x, y, z, value, distance)
        gem_base = 13
        for i in range(5):
            b = gem_base + i * 5
            obs[b:b+3] /= 100.0   # Relative x, y, z positions
            obs[b+3]   /= 5.0     # Gem value (1-5 → 0.2-1.0)
            obs[b+4]   /= 100.0   # Distance (0-100+ → 0-1+)

        # Opponents (indices 38-55: 3 opponents × 6 dims)
        opp_base = 38
        for i in range(3):
            b = opp_base + i * 6
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

    def process_message(self, message):
        """Process a message from the game, return action."""
        try:
            # Parse: obs_json|reward|done|gem_delta
            parts = message.split('|')

            if len(parts) != 4:
                if self.total_steps < 3:
                    self.log(f"Malformed message (expected 4 parts, got {len(parts)}): {message[:100]}")
                return [0, 0, 0, 0, 0, 0]

            obs_json, reward_str, done_str, gem_delta_str = parts

            # Parse observation
            obs = json.loads(obs_json)
            reward = float(reward_str)
            done = int(float(done_str))
            gem_delta = float(gem_delta_str)

            # Debug first message
            if self.total_steps == 0:
                self.log(f"First observation: {len(obs)} dims")
                raw = np.array(obs, dtype=np.float32)
                self.log(f"First obs raw min={raw.min():.1f} max={raw.max():.1f} mean={raw.mean():.1f}")
                self.log(f"First reward: {reward}")
                self.log(f"First done: {done}")

            # Normalize observation before passing to network.
            # Raw obs contains -999 sentinels (empty gem/opponent slots) and
            # millisecond time values (0-120000), which cause the critic to
            # output values in the thousands even with fresh random weights,
            # flooding the network with enormous gradients on the first update.
            obs_array = self.normalize_obs(np.array(obs, dtype=np.float32))

            if self.total_steps == 0:
                self.log(f"First obs normalized min={obs_array.min():.3f} max={obs_array.max():.3f} mean={obs_array.mean():.3f}")

            action, log_prob, value = self.model.get_action(obs_array)
            self.recent_actions.append(action.tolist())
            self.rollout_action_sums += action

            # Warn if first step of a new episode carries a suspiciously large reward
            # (indicates the sentinel-spike fix is not working or a new source of initial reward)
            if self.episode_step == 0 and self.total_episodes > 0 and abs(reward) > 1.0:
                self.log(f"  WARN: large first-step reward {reward:.2f} at ep={self.total_episodes+1} — check sentinel spike fix")

            # Track events
            if gem_delta > 0:
                self.episode_gem_pts += int(gem_delta)
                self.rollout_gem_pts += int(gem_delta)
                self.total_gem_pts += int(gem_delta)
                self.log(f"[GEM] ep={self.total_episodes+1} step={self.total_steps} +{gem_delta:.0f}pts | ep_total={self.current_episode_reward + reward:.1f}")
            if reward < -80:  # OOB penalty (-100) from game
                self.episode_oob += 1
                self.rollout_oob += 1
                self.total_oob += 1
                self.log(f"[OOB] ep={self.total_episodes+1} step={self.total_steps} | penalty={reward:.1f} | ep_so_far={self.current_episode_reward:.1f}")
            if reward > 0.1:
                self.rollout_positive += 1
            self.rollout_steps += 1
            self.episode_step += 1

            # Store experience (scale reward to keep critic targets small)
            self.buffer.add(obs_array, action, reward * self.reward_scale, value, log_prob, done)
            self.total_steps += 1
            self.current_episode_reward += reward

            # Episode end
            if done:
                self.total_episodes += 1
                self.episode_rewards.append(self.current_episode_reward)
                self.recent_episode_gems.append(self.episode_gem_pts)
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                outcome = "SUCCESS" if self.current_episode_reward > 50 else "FAIL" if self.current_episode_reward < -30 else "NEUTRAL"
                oob_str = f" | OOB={self.episode_oob}" if self.episode_oob else ""
                gems_str = f" | gems={self.episode_gem_pts}pts" if self.episode_gem_pts else ""
                self.log(f"Ep {self.total_episodes} [{outcome}] rwd={self.current_episode_reward:.1f}{gems_str}{oob_str} | avg100={avg_reward:.1f} | val={value:.3f}")
                if self.current_episode_reward > 50:
                    self.log(f"  *** SUCCESS: {self.episode_gem_pts}pts in this episode ***")
                self.current_episode_reward = 0
                self.episode_gem_pts = 0
                self.episode_oob = 0
                self.episode_step = 0

            # PPO update when buffer is full
            if len(self.buffer) >= self.rollout_size:
                self.run_ppo_update()

            return action.tolist()

        except Exception as e:
            if self.total_steps < 5:
                self.log(f"Error processing message: {e}")
                self.log(f"Message (first 200 chars): {message[:200]}")
                import traceback
                traceback.print_exc()
            return [0, 0, 0, 0, 0, 0]

    def run_ppo_update(self):
        """Run PPO training update."""
        self.model.train()

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
        pos_pct = (self.rollout_positive / max(self.rollout_steps, 1)) * 100

        # Detect entropy collapse
        collapse_warn = ""
        if stats['entropy'] < 1.0:
            collapse_warn = " *** ENTROPY COLLAPSE ***"
        elif stats['entropy'] < 2.0:
            collapse_warn = " (entropy low)"

        # Dry-rollout tracking
        if self.rollout_gem_pts == 0:
            self.dry_rollouts += 1
        else:
            self.dry_rollouts = 0

        dry_warn = f" *** DRY x{self.dry_rollouts} ***" if self.dry_rollouts >= 5 else ""

        # Action distribution: % of steps each action was pressed
        n = max(self.rollout_steps, 1)
        act_pct = self.rollout_action_sums / n * 100
        act_str = (f"F:{act_pct[0]:.0f}% B:{act_pct[1]:.0f}% "
                   f"L:{act_pct[2]:.0f}% R:{act_pct[3]:.0f}% "
                   f"J:{act_pct[4]:.0f}% P:{act_pct[5]:.0f}%")

        # Compact per-update line
        gems_str = f" gems={self.rollout_gem_pts}pts" if self.rollout_gem_pts else " gems=0"
        oob_str  = f" OOB={self.rollout_oob}" if self.rollout_oob else ""
        self.log(
            f"Upd {self.total_updates:4d} | "
            f"VLoss={stats['value_loss']:.4f} | "
            f"Ent={stats['entropy']:.3f}{collapse_warn} | "
            f"GradN={stats['grad_norm']:.3f} | "
            f"AvgRwd={avg_reward:.1f} | "
            f"+rwd={pos_pct:.1f}% |"
            f"{gems_str}{oob_str}{dry_warn}"
        )
        self.log(f"       Actions: {act_str}")

        # Repetitive action warning
        if len(self.recent_actions) >= 10:
            action_strings = [''.join(map(str, a)) for a in self.recent_actions]
            unique_actions = len(set(action_strings))
            if unique_actions <= 2:
                self.log(f"  WARNING: repetitive actions! Recent: {action_strings[-5:]}")

        # Reset rollout counters
        self.rollout_gem_pts = 0
        self.rollout_oob = 0
        self.rollout_positive = 0
        self.rollout_steps = 0
        self.rollout_action_sums = np.zeros(6)

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

            entropy_trend = ""
            if len(self.entropy_history) >= 5:
                recent = list(self.entropy_history)
                if recent[-1] < recent[0] - 0.5:
                    entropy_trend = " (falling)"
                elif recent[-1] > recent[0] + 0.5:
                    entropy_trend = " (rising)"

            self.log(f"\n{'#' * 55}")
            self.log(f"TRAINING SUMMARY - Update {self.total_updates}")
            self.log(f"{'#' * 55}")
            self.log(f"  Steps: {self.total_steps:,} | Episodes: {self.total_episodes} | Time: {elapsed_hrs:.2f}h")
            self.log(f"  Gems: {self.total_gem_pts}pts total | {gems_hr:.1f} pts/hr | OOB lifetime: {self.total_oob}")
            self.log(f"  AvgRwd: {avg_reward:.2f} | Best: {self.best_avg_reward:.2f}")
            self.log(f"  Entropy: {stats['entropy']:.4f}{entropy_trend}{collapse_warn}")
            self.log(f"  GradNorm: {stats['grad_norm']:.4f} | VLoss: {stats['value_loss']:.6f} | PLoss: {stats['policy_loss']:.6f}")
            if len(self.episode_rewards) >= 2:
                recent_10 = list(self.episode_rewards)[-10:]
                self.log(f"  Last 10 rewards: {[f'{r:.0f}' for r in recent_10]}")
            if self.recent_episode_gems:
                gem_list = list(self.recent_episode_gems)
                recent_gems_total = sum(gem_list)
                nonzero = sum(1 for g in gem_list if g > 0)
                self.log(f"  Last {len(gem_list)} eps gems: {recent_gems_total}pts total | {nonzero}/{len(gem_list)} eps had gems")
            if self.dry_rollouts >= 5:
                self.log(f"  *** {self.dry_rollouts} consecutive dry rollouts — policy may be stuck ***")
            self.log(f"{'#' * 55}\n")

        # Log to file
        self.log_stats(stats, avg_reward)

        # Clear buffer for next rollout
        self.buffer.clear()

    def save_model(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
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
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Mini-batch size')
    parser.add_argument('--epochs', type=int, default=4, help='PPO epochs per update')

    args = parser.parse_args()

    # Auto-resume from latest checkpoint if it exists
    # Check for update_N.pth files and find the highest N
    import glob
    checkpoint_dir = 'models/checkpoints'
    update_files = glob.glob(f'{checkpoint_dir}/update_*.pth')

    model_path = None
    if update_files:
        # Find the checkpoint with the highest update number
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
    server.trainer.optimizer.param_groups[0]['lr'] = args.lr

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
