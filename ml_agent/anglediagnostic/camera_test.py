"""
Camera Angle Sensitivity Test

Runs 36 games at different camera angles (0, 10, 20, ..., 350 degrees),
recording gems per game at each angle. Produces a polar chart showing
directional bias.

Usage:
    python camera_test.py
    python camera_test.py --model models/checkpoints/best.pth
    python camera_test.py --stochastic
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
    OBS_DIM = OBS_DIM_BASE + FRAME_HISTORY_COUNT * FRAME_HISTORY_DIMS  # 59
    frame_history_size = FRAME_HISTORY_COUNT * FRAME_SKIP + 1
    frame_history = deque(maxlen=frame_history_size)

    parser = argparse.ArgumentParser(description='Camera Angle Sensitivity Test')
    parser.add_argument('--model', default='models/checkpoints/best.pth')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--step-deg', type=int, default=10,
                        help='Degrees between each test angle (default: 10)')
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

    # Test angles
    angles_deg = list(range(0, 360, args.step_deg))
    num_games = len(angles_deg)
    results = {}  # angle_deg -> gems

    current_game = 0
    current_angle_deg = angles_deg[0]
    current_angle_rad = math.radians(current_angle_deg)
    episode_gems = 0
    total_steps = 0
    camera_set = False

    print(f"Will test {num_games} camera angles: 0 to {360 - args.step_deg} degrees (step {args.step_deg})")
    print(f"Listening on {args.host}:{args.port}")
    print(f"Start the game -- make sure MLAgent uses 1x speed")
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

                        # Get action
                        _, action_game, _ = model.get_action(obs_augmented, deterministic=deterministic)
                        dx, dy, throttle = action_game
                        total_steps += 1

                        if gem_delta > 0:
                            episode_gems += int(gem_delta)

                        # Build action string -- include camera yaw on every step
                        # to ensure camera stays locked at desired angle
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

    # Print results
    if results:
        print("\n" + "=" * 50)
        print("RESULTS: Gems per camera angle")
        print("=" * 50)
        for angle in sorted(results.keys()):
            gems = results[angle]
            bar = "#" * (gems // 2)
            print(f"  {angle:>3} deg: {gems:>3} gems  {bar}")

        avg = sum(results.values()) / len(results)
        best_angle = max(results, key=results.get)
        worst_angle = min(results, key=results.get)
        print(f"\n  Average: {avg:.1f} gems")
        print(f"  Best:    {results[best_angle]} gems @ {best_angle} deg")
        print(f"  Worst:   {results[worst_angle]} gems @ {worst_angle} deg")
        print(f"  Spread:  {results[best_angle] - results[worst_angle]} gems")

        # Save results to file for later analysis
        with open("camera_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to camera_test_results.json")

        # Generate HTML chart
        generate_chart(results)


def generate_chart(results):
    """Generate an HTML polar chart using inline SVG."""
    angles = sorted(results.keys())
    gems = [results[a] for a in angles]
    max_gems = max(gems) if gems else 1
    avg_gems = sum(gems) / len(gems) if gems else 0

    # SVG polar chart
    cx, cy = 300, 300  # center
    r_max = 250        # max radius
    svg_points = []
    svg_labels = []
    svg_dots = []

    for angle_deg, gem_count in zip(angles, gems):
        # SVG: 0 degrees = up (north), clockwise
        angle_rad = math.radians(angle_deg - 90)  # rotate so 0=right in SVG, then adjust
        # Actually: 0 deg camera = "north" in game. In SVG, up is -Y.
        # Let's map: 0 deg -> top, 90 deg -> right, etc.
        svg_angle = math.radians(angle_deg - 90)
        r = (gem_count / max_gems) * r_max if max_gems > 0 else 0
        x = cx + r * math.cos(svg_angle)
        y = cy + r * math.sin(svg_angle)
        svg_points.append(f"{x:.1f},{y:.1f}")
        svg_dots.append((x, y, gem_count, angle_deg))

    # Close the polygon
    polygon_points = " ".join(svg_points)

    # Grid circles
    grid_circles = ""
    for frac in [0.25, 0.5, 0.75, 1.0]:
        r = frac * r_max
        gem_val = int(frac * max_gems)
        grid_circles += f'<circle cx="{cx}" cy="{cy}" r="{r:.0f}" fill="none" stroke="#ddd" stroke-width="1"/>\n'
        grid_circles += f'<text x="{cx+4}" y="{cy-r+14}" font-size="11" fill="#999">{gem_val}</text>\n'

    # Angle lines and labels
    angle_lines = ""
    for deg in range(0, 360, 30):
        rad = math.radians(deg - 90)
        x2 = cx + r_max * math.cos(rad)
        y2 = cy + r_max * math.sin(rad)
        lx = cx + (r_max + 20) * math.cos(rad)
        ly = cy + (r_max + 20) * math.sin(rad)
        angle_lines += f'<line x1="{cx}" y1="{cy}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="#eee" stroke-width="1"/>\n'
        angle_lines += f'<text x="{lx:.0f}" y="{ly:.0f}" font-size="12" fill="#666" text-anchor="middle" dominant-baseline="middle">{deg}</text>\n'

    # Dots with labels
    dot_svg = ""
    for x, y, gem_count, angle_deg in svg_dots:
        color = "#e74c3c" if gem_count < avg_gems * 0.85 else "#2ecc71" if gem_count > avg_gems * 1.1 else "#3498db"
        dot_svg += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="white" stroke-width="1.5"/>\n'
        dot_svg += f'<title>{angle_deg} deg: {gem_count} gems</title>\n'

    # Average circle
    avg_r = (avg_gems / max_gems) * r_max if max_gems > 0 else 0

    html = f"""<!DOCTYPE html>
<html>
<head><title>Camera Angle Sensitivity Test</title></head>
<body style="font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; text-align: center;">
<h2>Camera Angle vs Gems Collected</h2>
<p>Each point = 1 game at that camera yaw. Distance from center = gem count.</p>
<svg width="620" height="620" viewBox="-10 -10 620 620">
{grid_circles}
{angle_lines}
<circle cx="{cx}" cy="{cy}" r="{avg_r:.0f}" fill="none" stroke="orange" stroke-width="2" stroke-dasharray="8,4"/>
<polygon points="{polygon_points}" fill="rgba(52,152,219,0.2)" stroke="#3498db" stroke-width="2"/>
{dot_svg}
</svg>
<p style="color:#999">
    <span style="color:orange">---</span> Average ({avg_gems:.0f} gems) |
    <span style="color:#2ecc71">Green</span> = above avg |
    <span style="color:#e74c3c">Red</span> = below avg |
    <span style="color:#3498db">Blue</span> = near avg
</p>
<h3>Raw Data</h3>
<table style="margin: 0 auto; border-collapse: collapse;">
<tr><th style="padding:4px 12px; border-bottom:2px solid #333;">Angle</th><th style="padding:4px 12px; border-bottom:2px solid #333;">Gems</th><th style="padding:4px 12px; border-bottom:2px solid #333;">vs Avg</th></tr>
"""
    for angle_deg in sorted(results.keys()):
        g = results[angle_deg]
        diff = g - avg_gems
        color = "#e74c3c" if g < avg_gems * 0.85 else "#2ecc71" if g > avg_gems * 1.1 else "#333"
        html += f'<tr><td style="padding:2px 12px;">{angle_deg}</td><td style="padding:2px 12px;">{g}</td><td style="padding:2px 12px; color:{color}">{diff:+.0f}</td></tr>\n'

    html += f"""</table>
<p>Best: {max(gems)} gems @ {angles[gems.index(max(gems))]} deg | Worst: {min(gems)} gems @ {angles[gems.index(min(gems))]} deg | Spread: {max(gems)-min(gems)}</p>
</body></html>"""

    with open("camera_test_results.html", "w") as f:
        f.write(html)
    print(f"  Chart saved to camera_test_results.html")


if __name__ == '__main__':
    main()
