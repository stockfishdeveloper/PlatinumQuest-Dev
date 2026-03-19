"""
Camera Observation Comparison Test

At two different camera angles, drive the marble to the same world position,
with a gem at the same world position, and compare the raw observations.
If rotation is correct, the camera-relative observations should be identical
(except for absolute position which will differ).

This script:
1. Sets camera to angle A, records obs for 30 steps
2. Waits for next game, sets camera to angle B, records obs for 30 steps
3. Compares the observations field by field

Usage: python camera_obs_compare.py
"""

import socket
import json
import math
import numpy as np

HOST = '127.0.0.1'
PORT = 8888

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    sock.bind((HOST, PORT))
    sock.listen(1)

    print(f"Listening on {HOST}:{PORT}")
    print("Will compare raw observations at two camera angles")
    print("Start game at 1x speed")
    print()

    # Test: at step 0, marble is at origin, gem spawns somewhere
    # Just compare the FIRST observation at two different camera angles
    # Both should start at same world pos (0,0) with same gem layout

    ANGLE_A = 0
    ANGLE_B = 90  # 90 degrees different

    obs_a = None
    obs_b = None
    game = 0

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
            step = 0

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

                        obs_raw = json.loads(parts[0])
                        done = int(float(parts[3]))

                        if len(obs_raw) == 0:
                            conn.sendall(b'0,0,0,0\n')
                            continue

                        angle_deg = ANGLE_A if game == 0 else ANGLE_B
                        angle_rad = math.radians(angle_deg)

                        # Set camera and don't move
                        action_str = f"0,0,0,0,{angle_rad:.6f}\n"
                        conn.sendall(action_str.encode('utf-8'))

                        if step == 5:  # After a few frames to settle camera
                            obs = np.array(obs_raw, dtype=np.float32)
                            if game == 0:
                                obs_a = obs.copy()
                                print(f"\nGame {game+1} (camera={angle_deg} deg), captured obs at step {step}")
                                print(f"  Raw obs (first 35): {obs[:35]}")
                            else:
                                obs_b = obs.copy()
                                print(f"\nGame {game+1} (camera={angle_deg} deg), captured obs at step {step}")
                                print(f"  Raw obs (first 35): {obs[:35]}")

                        step += 1

                        if done:
                            game += 1
                            step = 0
                            print(f"\n--- Game {game} ended ---")

                            if game >= 2 and obs_a is not None and obs_b is not None:
                                print("\n" + "=" * 70)
                                print(f"COMPARISON: camera {ANGLE_A} deg vs {ANGLE_B} deg")
                                print("=" * 70)

                                labels = [
                                    "selfPosX", "selfPosY", "selfPosZ",
                                    "selfVelX", "selfVelY", "selfVelZ",
                                ]
                                for i in range(5):
                                    labels.extend([f"gem{i}_dx", f"gem{i}_dy", f"gem{i}_dz", f"gem{i}_pts", f"gem{i}_dist"])
                                labels.extend(["timeElapsed", "timeRemaining", "myScore", "gemsRemaining"])

                                print(f"\n{'Field':>15} | {'Angle A':>12} | {'Angle B':>12} | {'Diff':>12}")
                                print("-" * 60)

                                for i in range(min(35, len(obs_a), len(obs_b))):
                                    label = labels[i] if i < len(labels) else f"obs[{i}]"
                                    diff = obs_b[i] - obs_a[i]
                                    flag = " ***" if abs(diff) > 0.01 else ""
                                    print(f"{label:>15} | {obs_a[i]:>12.4f} | {obs_b[i]:>12.4f} | {diff:>+12.4f}{flag}")

                                # Now check: are vel and gem fields correctly rotated?
                                # If marble is at rest at origin, vel should be ~0 in both
                                # Gem relative position should be rotated by (B-A) degrees
                                print("\n--- Gem direction analysis ---")
                                for g in range(1):  # Just first gem
                                    b = 6 + g * 5
                                    dx_a, dy_a = obs_a[b], obs_a[b+1]
                                    dx_b, dy_b = obs_b[b], obs_b[b+1]
                                    dist_a = obs_a[b+4]
                                    dist_b = obs_b[b+4]
                                    angle_a_gem = math.degrees(math.atan2(dx_a, dy_a)) % 360
                                    angle_b_gem = math.degrees(math.atan2(dx_b, dy_b)) % 360
                                    angle_diff = angle_a_gem - angle_b_gem
                                    if angle_diff > 180: angle_diff -= 360
                                    if angle_diff < -180: angle_diff += 360

                                    print(f"  Gem 0 @ angle A: ({dx_a:.2f}, {dy_a:.2f}) = {angle_a_gem:.1f} deg, dist={dist_a:.1f}")
                                    print(f"  Gem 0 @ angle B: ({dx_b:.2f}, {dy_b:.2f}) = {angle_b_gem:.1f} deg, dist={dist_b:.1f}")
                                    print(f"  Gem angle difference: {angle_diff:.1f} deg (expected: {ANGLE_A - ANGLE_B} deg)")
                                    print(f"  Distance difference: {dist_a - dist_b:.4f} (should be ~0)")

                                # Check position rotation
                                print("\n--- Position analysis ---")
                                px_a, py_a = obs_a[0], obs_a[1]
                                px_b, py_b = obs_b[0], obs_b[1]
                                # At camera A (0 deg): cam pos = world pos * rotation(0) = world pos
                                # At camera B (90 deg): cam pos = world pos * rotation(90)
                                # If marble at same world pos, cam pos B should be A rotated by 90
                                print(f"  Pos @ A: ({px_a:.4f}, {py_a:.4f})")
                                print(f"  Pos @ B: ({px_b:.4f}, {py_b:.4f})")

                                # Verify: rotate A by (B-A) degrees, should get B
                                rot = math.radians(ANGLE_B - ANGLE_A)
                                expected_bx = px_a * math.cos(rot) - py_a * math.sin(rot)
                                expected_by = px_a * math.sin(rot) + py_a * math.cos(rot)
                                print(f"  Expected B (A rotated by {ANGLE_B-ANGLE_A} deg): ({expected_bx:.4f}, {expected_by:.4f})")
                                print(f"  Actual B - Expected B: ({px_b - expected_bx:.4f}, {py_b - expected_by:.4f})")

                                conn.close()
                                sock.close()
                                return

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                conn.close()

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


if __name__ == '__main__':
    main()
