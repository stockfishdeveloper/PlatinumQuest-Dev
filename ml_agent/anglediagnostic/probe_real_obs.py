"""
Probe the model with REAL observations from a live game.
Logs the pre-tanh and post-tanh values to see if tanh is truly saturated
with real (not synthetic) inputs.
"""

import socket
import json
import math
import numpy as np
import torch
import os
import sys
from collections import deque

from train_ppo import Actor


def normalize_obs(obs):
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
    OBS_DIM = 35 + FRAME_HISTORY_COUNT * FRAME_HISTORY_DIMS
    frame_history = deque(maxlen=FRAME_HISTORY_COUNT * FRAME_SKIP + 1)

    model = Actor(obs_dim=OBS_DIM)
    checkpoint = torch.load('models/checkpoints/best.pth', weights_only=False)
    model.load_state_dict(checkpoint['actor_state_dict'])
    model.eval()

    # Hook into actor_mean to capture pre-tanh values
    pre_tanh_vals = {}
    def hook_fn(module, input, output):
        pre_tanh_vals['last_input'] = input[0].detach().clone()
        pre_tanh_vals['last_output'] = output.detach().clone()

    # Register hook on the last linear layer of actor_mean
    # actor_mean is Sequential: Linear -> ReLU -> Linear
    # We want the output of the last Linear (index 2) BEFORE tanh in forward()
    last_linear = model.actor_mean[2]
    last_linear.register_forward_hook(hook_fn)

    print(f"Loaded model. Listening on 127.0.0.1:8888")
    print(f"Start game at 1x. Will log pre/post tanh for 300 steps then exit.")
    print()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    sock.bind(('127.0.0.1', 8888))
    sock.listen(1)

    step = 0
    MAX_STEPS = 300

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
                while step < MAX_STEPS:
                    data = conn.recv(8192).decode('utf-8')
                    if not data:
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

                        obs_raw = json.loads(parts[0])
                        if len(obs_raw) == 0:
                            conn.sendall(b'0,0,0,0\n')
                            continue

                        gem_delta = float(parts[1])
                        obs = np.array(obs_raw, dtype=np.float32)

                        # Get gem angle for context
                        gem_dx = obs[6]
                        gem_dy = obs[7]
                        gem_dist = obs[10]
                        gem_angle = math.degrees(math.atan2(gem_dx, gem_dy)) % 360 if gem_dist > 0 else -1

                        obs = normalize_obs(obs)

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

                        # Forward pass (triggers hook)
                        t_obs = torch.FloatTensor(obs_augmented).unsqueeze(0)
                        mean_xy, thr_logit = model(t_obs)

                        # Get pre-tanh values from hook
                        pre_tanh = pre_tanh_vals['last_output']
                        pre_dx = pre_tanh[0, 0].item()
                        pre_dy = pre_tanh[0, 1].item()

                        # Post-tanh (mean_xy already has tanh applied)
                        post_dx = mean_xy[0, 0].item()
                        post_dy = mean_xy[0, 1].item()

                        # Normalized direction
                        norm = math.sqrt(post_dx**2 + post_dy**2)
                        if norm > 1e-6:
                            dir_angle = math.degrees(math.atan2(post_dx/norm, post_dy/norm)) % 360
                        else:
                            dir_angle = -1

                        if step % 10 == 0 or gem_delta > 0:
                            prefix = "***GEM***" if gem_delta > 0 else "         "
                            print(f"{prefix} step={step:>4} gem@{gem_angle:>5.0f}deg dist={gem_dist:>5.1f} | "
                                  f"pre_tanh=({pre_dx:>+7.3f},{pre_dy:>+7.3f}) "
                                  f"post_tanh=({post_dx:>+6.4f},{post_dy:>+6.4f}) "
                                  f"output_dir={dir_angle:>5.1f}deg")

                        # Send action
                        _, action_game, _ = model.get_action(obs_augmented, deterministic=True)
                        dx, dy, throttle = action_game
                        action_tuple = Actor.action_to_joystick(dx, dy, throttle)
                        conn.sendall((','.join(map(str, action_tuple)) + '\n').encode('utf-8'))

                        step += 1

                print(f"\nDone after {step} steps")

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                conn.close()
            break

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


if __name__ == '__main__':
    main()
