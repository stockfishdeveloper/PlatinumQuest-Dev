"""
Camera Action Alignment Test

Tests whether $mvForwardAction pushes the marble in the exact direction
the camera is facing. Sets camera to various angles, sends pure forward
action, and logs the resulting velocity direction vs camera direction.

If there's a systematic offset between camera yaw and actual force direction,
this will reveal it.

Usage:
    python camera_action_test.py
"""

import socket
import json
import math
import time

HOST = '127.0.0.1'
PORT = 8888

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    sock.bind((HOST, PORT))
    sock.listen(1)

    print(f"Listening on {HOST}:{PORT}")
    print("Tests whether forward action aligns with camera direction")
    print("Start the game at 1x speed")
    print()

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
            test_angles = list(range(0, 360, 10))
            current_test = 0
            test_step = 0
            SETTLE_STEPS = 30   # let marble stop
            PUSH_STEPS = 60     # push forward
            MEASURE_STEP = 50   # measure velocity at this step within push phase
            phase = "settle"    # settle -> push -> settle -> push ...

            results = {}

            try:
                while current_test < len(test_angles):
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

                        obs_json = parts[0]
                        obs_raw = json.loads(obs_json)
                        if len(obs_raw) == 0:
                            conn.sendall(b'0,0,0,0\n')
                            continue

                        angle_deg = test_angles[current_test]
                        angle_rad = math.radians(angle_deg)

                        if phase == "settle":
                            # No movement, just set camera
                            action_str = f"0,0,0,0,{angle_rad:.6f}\n"
                            conn.sendall(action_str.encode('utf-8'))
                            test_step += 1
                            if test_step >= SETTLE_STEPS:
                                phase = "push"
                                test_step = 0
                        elif phase == "push":
                            # Pure forward at full throttle
                            action_str = f"1.0,0,0,0,{angle_rad:.6f}\n"
                            conn.sendall(action_str.encode('utf-8'))

                            if test_step == MEASURE_STEP:
                                # Read world velocity from obs
                                # obs[3:6] are camera-relative vel
                                # We need world vel to compare with camera yaw
                                # cam_vel_x = world_vx * cos(yaw) - world_vy * sin(yaw)
                                # cam_vel_y = world_vx * sin(yaw) + world_vy * cos(yaw)
                                # Invert: world_vx = cam_vx * cos(yaw) + cam_vy * sin(yaw)
                                #          world_vy = -cam_vx * sin(yaw) + cam_vy * cos(yaw)
                                cam_vx = obs_raw[3]
                                cam_vy = obs_raw[4]

                                # But wait - obs was collected BEFORE this action
                                # The velocity we see is from the PREVIOUS step's action
                                # That's fine for settle->push transition measurement

                                # Camera-relative velocity direction
                                cam_vel_angle = math.degrees(math.atan2(cam_vx, cam_vy)) % 360

                                # In camera space, pure forward should give vel direction ~0 degrees
                                # Any offset from 0 is a misalignment
                                offset = cam_vel_angle
                                if offset > 180:
                                    offset -= 360

                                vel_mag = math.sqrt(cam_vx**2 + cam_vy**2)

                                # Also reconstruct world vel direction
                                cos_y = math.cos(angle_rad)
                                sin_y = math.sin(angle_rad)
                                world_vx = cam_vx * cos_y + cam_vy * sin_y
                                world_vy = -cam_vx * sin_y + cam_vy * cos_y
                                world_vel_angle = math.degrees(math.atan2(world_vx, world_vy)) % 360

                                # Expected world direction = camera yaw
                                # Torque forward = (sin(yaw), cos(yaw))
                                # atan2(sin(yaw), cos(yaw)) = yaw
                                expected = angle_deg
                                world_offset = world_vel_angle - expected
                                if world_offset > 180: world_offset -= 360
                                if world_offset < -180: world_offset += 360

                                results[angle_deg] = {
                                    'cam_vel_angle': cam_vel_angle,
                                    'cam_offset': offset,
                                    'world_vel_angle': world_vel_angle,
                                    'world_offset': world_offset,
                                    'vel_mag': vel_mag,
                                    'cam_vx': cam_vx,
                                    'cam_vy': cam_vy,
                                }

                                print(f"  cam={angle_deg:>3} deg: cam_vel={cam_vel_angle:>6.1f} (offset={offset:>+6.1f}) world_vel={world_vel_angle:>6.1f} (offset={world_offset:>+6.1f}) |vel|={vel_mag:.1f}")

                            test_step += 1
                            if test_step >= PUSH_STEPS:
                                phase = "settle"
                                test_step = 0
                                current_test += 1

                        step += 1

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                conn.close()

            # Summary
            if results:
                print("\n" + "=" * 60)
                print("SUMMARY: Camera-relative velocity offset from pure forward")
                print("=" * 60)
                offsets = [r['cam_offset'] for r in results.values()]
                world_offsets = [r['world_offset'] for r in results.values()]
                print(f"  Cam offset:   mean={sum(offsets)/len(offsets):+.1f} min={min(offsets):+.1f} max={max(offsets):+.1f} spread={max(offsets)-min(offsets):.1f}")
                print(f"  World offset: mean={sum(world_offsets)/len(world_offsets):+.1f} min={min(world_offsets):+.1f} max={max(world_offsets):+.1f} spread={max(world_offsets)-min(world_offsets):.1f}")

                with open("camera_action_test_results.json", "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  Saved to camera_action_test_results.json")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sock.close()


if __name__ == '__main__':
    main()
