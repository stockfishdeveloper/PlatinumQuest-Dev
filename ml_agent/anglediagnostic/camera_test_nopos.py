"""
Camera Angle Sensitivity Test — with position zeroed out

Tests whether self-position in observations is the source of camera-angle
dependence. Zeroes out obs[0:3] (camera-relative position) before feeding
to the model. If results become uniform across angles, position is the culprit.

Usage:
    python camera_test_nopos.py
"""

import socket
import json
import math
import numpy as np
import torch
import argparse
import os
import sys
from collections import deque

from train_ppo import Actor


def normalize_obs(obs):
    """Normalize raw game observations (mirrors PPOServer.normalize_obs exactly)."""
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
    FRAME_HISTORY_COUNT = 4
    FRAME_SKIP = 8
    FRAME_HISTORY_DIMS = 6
    OBS_DIM_BASE = 35
    OBS_DIM = OBS_DIM_BASE + FRAME_HISTORY_COUNT * FRAME_HISTORY_DIMS
    frame_history_size = FRAME_HISTORY_COUNT * FRAME_SKIP + 1
    frame_history = deque(maxlen=frame_history_size)

    parser = argparse.ArgumentParser(description='Camera Test (no position)')
    parser.add_argument('--model', default='models/checkpoints/best.pth')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--step-deg', type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    model = Actor(obs_dim=OBS_DIM)
    checkpoint = torch.load(args.model, weights_only=False)
    state = checkpoint.get('actor_state_dict', checkpoint.get('model_state_dict', {}))
    model.load_state_dict(state, strict=False)
    model.eval()

    deterministic = not args.stochastic
    mode = "stochastic" if args.stochastic else "deterministic"
    print(f"Loaded {args.model}")
    print(f"Mode: {mode}")
    print(f"*** POSITION ZEROED OUT (obs[0:3] = 0) ***")

    angles_deg = list(range(0, 360, args.step_deg))
    num_games = len(angles_deg)
    results = {}

    current_game = 0
    current_angle_deg = angles_deg[0]
    current_angle_rad = math.radians(current_angle_deg)
    episode_gems = 0
    total_steps = 0

    print(f"Will test {num_games} camera angles: 0 to {360 - args.step_deg} degrees")
    print(f"Listening on {args.host}:{args.port}")
    print()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    sock.bind((args.host, args.port))
    sock.listen(1)

    try:
        while current_game < num_games:
            try:
                conn, addr = sock.accept()
                print(f"Game connected from {addr}")
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break

            buffer_str = ""
            try:
                while current_game < num_games:
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

                        obs_json, gem_delta_str, oob_str, done_str = parts
                        obs_raw = json.loads(obs_json)

                        if len(obs_raw) == 0:
                            conn.sendall(b'0,0,0,0\n')
                            continue

                        obs = np.array(obs_raw, dtype=np.float32)
                        gem_delta = float(gem_delta_str)
                        done = int(float(done_str))

                        obs = normalize_obs(obs)

                        # ZERO OUT POSITION (obs[0:3])
                        obs[0:3] = 0.0

                        # Frame history
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

                        action_tuple = Actor.action_to_joystick(dx, dy, throttle)
                        action_str = ','.join(map(str, action_tuple))
                        action_str += f',{current_angle_rad:.6f}'
                        conn.sendall((action_str + '\n').encode('utf-8'))

                        if done:
                            results[current_angle_deg] = episode_gems
                            print(f"  Game {current_game+1}/{num_games}: camera={current_angle_deg} deg -> {episode_gems} gems")

                            episode_gems = 0
                            total_steps = 0
                            frame_history.clear()
                            current_game += 1

                            if current_game < num_games:
                                current_angle_deg = angles_deg[current_game]
                                current_angle_rad = math.radians(current_angle_deg)

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                conn.close()

    except KeyboardInterrupt:
        print("\nStopping early...")
    finally:
        sock.close()

    if results:
        print("\n" + "=" * 50)
        print("RESULTS: Gems per camera angle (POSITION ZEROED)")
        print("=" * 50)
        gems_list = []
        for angle in sorted(results.keys()):
            gems = results[angle]
            gems_list.append(gems)
            bar = "#" * (gems // 2)
            print(f"  {angle:>3} deg: {gems:>3} gems  {bar}")

        avg = sum(gems_list) / len(gems_list)
        best_angle = max(results, key=results.get)
        worst_angle = min(results, key=results.get)
        print(f"\n  Average: {avg:.1f} gems")
        print(f"  Best:    {results[best_angle]} gems @ {best_angle} deg")
        print(f"  Worst:   {results[worst_angle]} gems @ {worst_angle} deg")
        print(f"  Spread:  {results[best_angle] - results[worst_angle]} gems")

        with open("camera_test_nopos_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to camera_test_nopos_results.json")


if __name__ == '__main__':
    main()
