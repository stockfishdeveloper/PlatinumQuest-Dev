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
from collections import deque

# Reuse the model definition from the training script
from train_ppo import Actor


def normalize_obs(obs):
    """Normalize raw game observations (mirrors PPOServer.normalize_obs exactly).

    35-dim layout from game:
      [0-5]   Self: pos(3), vel(3) — both camera-relative
      [6-30]  5 gems x 5
      [31-34] Game: timeElapsed, timeRemaining, myScore, gemsRemaining
    """
    obs[0:3]  /= 100.0
    obs[3:6]  /= 20.0

    gem_base = 6
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

    obs[31] /= 300000.0
    obs[32] /= 300000.0
    obs[33] /= 100.0
    obs[34] /= 50.0

    obs = np.clip(obs, -2.0, 2.0)
    return obs


def main():
    # Frame history config (must match train_ppo.py)
    FRAME_HISTORY_COUNT = 4
    FRAME_SKIP = 8
    FRAME_HISTORY_DIMS = 6  # pos(3) + vel(3)
    OBS_DIM_BASE = 35
    OBS_DIM = OBS_DIM_BASE + FRAME_HISTORY_COUNT * FRAME_HISTORY_DIMS  # 59
    frame_history_size = FRAME_HISTORY_COUNT * FRAME_SKIP + 1  # 33
    frame_history = deque(maxlen=frame_history_size)

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
    model = Actor(obs_dim=OBS_DIM)
    checkpoint = torch.load(args.model, weights_only=False)
    state = checkpoint.get('actor_state_dict', checkpoint.get('model_state_dict', {}))
    model.load_state_dict(state, strict=False)
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

                        # Protocol: obs_json|gem_delta|oob|done
                        parts = line.split('|')
                        if len(parts) != 4:
                            conn.sendall(b'0,0,0,0\n')
                            continue

                        obs_json, gem_delta_str, oob_str, done_str = parts
                        obs_raw = json.loads(obs_json)

                        # Skip game-end signals (empty obs)
                        if len(obs_raw) == 0:
                            conn.sendall(b'0,0,0,0\n')
                            continue

                        obs = np.array(obs_raw, dtype=np.float32)
                        gem_delta = float(gem_delta_str)
                        done = int(float(done_str))

                        obs = normalize_obs(obs)

                        # Build augmented obs with frame history
                        current_posvel = obs[0:6].copy()
                        frame_history.append(current_posvel)
                        history_frames = []
                        for i in range(1, FRAME_HISTORY_COUNT + 1):
                            idx = len(frame_history) - 1 - i * FRAME_SKIP
                            if idx >= 0:
                                history_frames.append(frame_history[idx])
                            else:
                                history_frames.append(np.zeros(FRAME_HISTORY_DIMS, dtype=np.float32))
                        obs_augmented = np.concatenate([obs] + history_frames)

                        _, action_game, _ = model.get_action(obs_augmented, deterministic=deterministic)
                        dx, dy, throttle = action_game
                        total_steps += 1

                        if gem_delta > 0:
                            episode_gems += int(gem_delta)
                            print(f"  +{gem_delta:.0f} gem pts (episode total: {episode_gems})")

                        if done:
                            total_episodes += 1
                            print(f"Ep {total_episodes} done | gems={episode_gems}pts steps={total_steps}")
                            episode_gems = 0
                            total_steps = 0
                            frame_history.clear()

                        action_tuple = Actor.action_to_joystick(dx, dy, throttle)
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
