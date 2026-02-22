"""
Inference-only server for PlatinumQuest Hunt Mode

Loads a trained model and plays the game at normal speed (1x).
No training, no buffer, no dashboard — just picks the best action each step.

Usage:
    python play.py                              # loads best.pth
    python play.py --model models/checkpoints/update_7500.pth
    python play.py --stochastic                 # sample from policy instead of argmax
"""

import socket
import json
import numpy as np
import torch
import argparse
import os
import sys

# Reuse the model definition from the training script
from train_ppo import ActorCritic


def normalize_obs(obs):
    """Normalize raw game observations (mirrors PPOServer.normalize_obs exactly)."""
    obs[0:3]  /= 100.0
    obs[3:6]  /= 20.0
    while obs[6] > 3.14159:
        obs[6] -= 6.28318
    while obs[6] < -3.14159:
        obs[6] += 6.28318
    obs[6]    /= 3.14159
    obs[7]    /= 1.5708
    obs[11]   /= 20.0
    obs[12]   /= 20.0

    gem_base = 13
    for i in range(5):
        b = gem_base + i * 5
        if obs[b+4] < -500:
            obs[b:b+3] = 0.0
            obs[b+3]   = 0.0
            obs[b+4]   = 1.0
        else:
            dist = obs[b+4]
            if dist > 0.01:
                obs[b:b+3] /= dist
            else:
                obs[b:b+3] = 0.0
            obs[b+3]   /= 5.0
            obs[b+4]   /= 100.0

    opp_base = 38
    for i in range(3):
        b = opp_base + i * 6
        if obs[b] < -500:
            obs[b:b+3]   = 0.0
            obs[b+3:b+5] = 0.0
            obs[b+5]     = 0.0
        else:
            obs[b:b+3]   /= 100.0
            obs[b+3:b+5] /= 20.0

    obs[56] /= 300000.0
    obs[57] /= 300000.0
    obs[58] /= 100.0
    obs[59] /= 100.0
    obs[60] /= 50.0

    obs = np.clip(obs, -2.0, 2.0)
    return obs


def main():
    parser = argparse.ArgumentParser(description='PlatinumQuest Inference Server')
    parser.add_argument('--model', default='models/checkpoints/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--stochastic', action='store_true',
                        help='Sample from policy instead of taking argmax')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    # Load model
    model = ActorCritic(obs_dim=61, n_actions=9)
    checkpoint = torch.load(args.model, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    deterministic = not args.stochastic
    mode = "stochastic" if args.stochastic else "deterministic"
    updates = checkpoint.get('total_updates', '?')
    best = checkpoint.get('best_avg_reward', '?')
    print(f"Loaded {args.model} (update {updates}, best_avg={best})")
    print(f"Mode: {mode}")
    print(f"Listening on {args.host}:{args.port}")
    print(f"Start the game (make sure MLAgent uses 1x speed for inference)")
    print(f"Press Ctrl+C to stop")

    # Stats
    total_steps = 0
    episode_gems = 0
    episode_reward = 0.0
    total_episodes = 0

    # Start server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    sock.bind((args.host, args.port))
    sock.listen(1)

    try:
        while True:
            try:
                conn, addr = sock.accept()
                print(f"Game connected from {addr}")
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break

            buffer_str = ""
            try:
                while True:
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

                        parts = line.split('|')
                        if len(parts) != 4:
                            conn.sendall(b'0,0,0,0\n')
                            continue

                        obs_json, reward_str, done_str, gem_delta_str = parts
                        obs = np.array(json.loads(obs_json), dtype=np.float32)
                        reward = float(reward_str)
                        done = int(float(done_str))
                        gem_delta = float(gem_delta_str)

                        obs = normalize_obs(obs)
                        action, _, _ = model.get_action(obs, deterministic=deterministic)
                        total_steps += 1

                        episode_reward += reward
                        if gem_delta > 0:
                            episode_gems += int(gem_delta)
                            print(f"  +{gem_delta:.0f} gem pts (episode total: {episode_gems})")

                        if done:
                            total_episodes += 1
                            print(f"Ep {total_episodes} done | gems={episode_gems}pts rwd={episode_reward:.1f} steps={total_steps}")
                            episode_gems = 0
                            episode_reward = 0.0
                            total_steps = 0

                        action_tuple = ActorCritic.ACTION_MAP[action]
                        conn.sendall((','.join(map(str, action_tuple)) + '\n').encode('utf-8'))

            except Exception as e:
                print(f"Error: {e}")
            finally:
                conn.close()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sock.close()


if __name__ == '__main__':
    main()
