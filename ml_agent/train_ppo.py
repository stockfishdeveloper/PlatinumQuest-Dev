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

    def __init__(self, obs_dim=286, action_dim=6):
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

    def __init__(self, model, lr=3e-4, clip_epsilon=0.2, value_coef=0.5,
                 entropy_coef=0.05, max_grad_norm=0.5):
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
        self.model = ActorCritic(obs_dim=286, action_dim=6)
        self.trainer = PPOTrainer(self.model)
        self.buffer = RolloutBuffer()

        # Load existing model if provided
        if model_path and os.path.exists(model_path):
            self.log(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.log("Model loaded successfully!")

        self.model.train()

        # Training config
        self.rollout_size = 512  # Steps before each PPO update (~25 seconds)
        self.n_epochs = 4
        self.batch_size = 64
        self.gamma = 0.99
        self.lam = 0.95
        self.save_interval = 10  # Save every N updates

        # Statistics
        self.total_steps = 0
        self.total_updates = 0
        self.total_episodes = 0
        self.episode_rewards = deque(maxlen=100)
        self.current_episode_reward = 0
        self.best_avg_reward = -float('inf')

        # Action tracking (for debugging)
        self.action_counts = np.zeros(6)  # Count how often each action is chosen
        self.episode_action_counts = np.zeros(6)  # Reset per episode

        # Reward tracking (for debugging)
        self.positive_rewards = 0
        self.negative_rewards = 0
        self.zero_rewards = 0

        # Recent actions (for debugging repetition)
        self.recent_actions = deque(maxlen=10)

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

        self.save_model('models/checkpoints/final.pth')
        self.socket.close()
        self.log("Server stopped. Final model saved.")
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

    def process_message(self, message):
        """Process a message from the game, return action."""
        try:
            # Parse: obs_json|reward|done
            parts = message.split('|')

            if len(parts) != 3:
                if self.total_steps < 3:
                    self.log(f"Malformed message (expected 3 parts, got {len(parts)}): {message[:100]}")
                return [0, 0, 0, 0, 0, 0]

            obs_json, reward_str, done_str = parts

            # Parse observation
            obs = json.loads(obs_json)
            reward = float(reward_str)
            done = int(float(done_str))

            # Debug first message
            if self.total_steps == 0:
                self.log(f"First observation: {len(obs)} dims")
                self.log(f"First reward: {reward}")
                self.log(f"First done: {done}")

            # Get action from model
            obs_array = np.array(obs, dtype=np.float32)
            action, log_prob, value = self.model.get_action(obs_array)

            # Track action usage
            self.action_counts += action
            self.episode_action_counts += action
            self.recent_actions.append(action.tolist())

            # Track reward distribution
            if reward > 0.1:
                self.positive_rewards += 1
            elif reward < -0.1:
                self.negative_rewards += 1
                # Log OOB penalties specifically
                if reward < -40:  # OOB penalty is -50 plus time penalty
                    self.log(f"[OOB PENALTY RECEIVED] Step {self.total_steps}: reward={reward:.2f} | Action taken: {action.tolist()}")
            else:
                self.zero_rewards += 1

            # Store experience
            self.buffer.add(obs_array, action, reward, value, log_prob, done)
            self.total_steps += 1
            self.current_episode_reward += reward

            # Track episodes
            if done:
                self.total_episodes += 1
                self.episode_rewards.append(self.current_episode_reward)
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0

                # Calculate action percentages for this episode
                episode_steps = np.sum(self.episode_action_counts) / 6.0
                action_pcts = (self.episode_action_counts / max(episode_steps, 1)) * 100

                self.log(f"\n--- Episode {self.total_episodes} Complete ---")
                self.log(f"  Episode Reward: {self.current_episode_reward:.2f}")
                self.log(f"  Avg Reward (last 100): {avg_reward:.2f}")
                self.log(f"  Total Steps: {self.total_steps}")
                self.log(f"  Buffer Size: {len(self.buffer)}")
                self.log(f"  Action Usage: F:{action_pcts[0]:.0f}% B:{action_pcts[1]:.0f}% L:{action_pcts[2]:.0f}% R:{action_pcts[3]:.0f}% J:{action_pcts[4]:.0f}% P:{action_pcts[5]:.0f}%")

                self.current_episode_reward = 0
                self.episode_action_counts = np.zeros(6)

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
        self.log(f"\n{'=' * 40}")
        self.log(f"PPO Update #{self.total_updates + 1}")
        self.log(f"{'=' * 40}")

        self.model.train()

        start_time = time.time()
        stats = self.trainer.update(
            self.buffer,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            gamma=self.gamma,
            lam=self.lam,
        )
        elapsed = time.time() - start_time

        self.total_updates += 1

        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0

        # Calculate overall action usage percentages
        total_action_uses = np.sum(self.action_counts)
        action_pcts = (self.action_counts / max(total_action_uses, 1)) * 100

        # Calculate reward distribution
        total_reward_steps = self.positive_rewards + self.negative_rewards + self.zero_rewards
        pos_pct = (self.positive_rewards / max(total_reward_steps, 1)) * 100
        neg_pct = (self.negative_rewards / max(total_reward_steps, 1)) * 100
        zero_pct = (self.zero_rewards / max(total_reward_steps, 1)) * 100

        self.log(f"  Policy Loss:  {stats['policy_loss']:.6f}")
        self.log(f"  Value Loss:   {stats['value_loss']:.6f}")
        self.log(f"  Entropy:      {stats['entropy']:.6f}")
        self.log(f"  Avg Reward:   {avg_reward:.2f}")
        self.log(f"  Update Time:  {elapsed:.2f}s")
        self.log(f"  Total Steps:  {self.total_steps}")
        self.log(f"  Episodes:     {self.total_episodes}")
        self.log(f"  Reward Distribution: +{pos_pct:.1f}%  -{neg_pct:.1f}%  0:{zero_pct:.1f}%")
        self.log(f"  Overall Action Usage:")
        self.log(f"    Forward: {action_pcts[0]:.1f}%  Backward: {action_pcts[1]:.1f}%")
        self.log(f"    Left:    {action_pcts[2]:.1f}%  Right:    {action_pcts[3]:.1f}%")
        self.log(f"    Jump:    {action_pcts[4]:.1f}%  Powerup:  {action_pcts[5]:.1f}%")

        # Check for repetitive behavior
        if len(self.recent_actions) >= 10:
            action_strings = [''.join(map(str, a)) for a in self.recent_actions]
            unique_actions = len(set(action_strings))
            self.log(f"  Action Diversity (last 10): {unique_actions}/10 unique")
            if unique_actions <= 2:
                self.log(f"  WARNING: Highly repetitive! Recent: {action_strings[-5:]}")

        self.log(f"{'=' * 40}\n")

        # Save checkpoint periodically
        if self.total_updates % self.save_interval == 0:
            path = f'models/checkpoints/update_{self.total_updates}.pth'
            self.save_model(path)

            # Save best model
            if avg_reward > self.best_avg_reward and len(self.episode_rewards) >= 10:
                self.best_avg_reward = avg_reward
                self.save_model('models/checkpoints/best.pth')
                self.log(f"  New best model! Avg reward: {avg_reward:.2f}")

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
    parser.add_argument('--load', help='Path to pre-trained model checkpoint')
    parser.add_argument('--rollout-size', type=int, default=512, help='Steps per PPO update')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Mini-batch size')
    parser.add_argument('--epochs', type=int, default=4, help='PPO epochs per update')

    args = parser.parse_args()

    server = PPOServer(host=args.host, port=args.port, model_path=args.load)
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
        server.save_model('models/checkpoints/final.pth')
        server.socket.close()
        server.logger.close()


if __name__ == '__main__':
    main()
