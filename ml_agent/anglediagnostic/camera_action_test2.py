"""
Camera Action Alignment Test v2

Same as v1 but with much longer settle period and measures velocity
at multiple points during the push to get a clean reading.
Also tests left, right, and backward actions.

Usage:
    python camera_action_test2.py
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
    print("Uses LONG settle time between tests")
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
            # Only test a few angles but with clean measurements
            test_angles = list(range(0, 360, 30))
            current_test = 0
            test_step = 0
            SETTLE_STEPS = 180    # 3 seconds at 60Hz - marble fully stopped
            PUSH_STEPS = 10       # very short push from standstill
            MEASURE_STEP = 9      # measure at end of push
            phase = "settle"

            results = {}
            prev_cam_vx = 0
            prev_cam_vy = 0

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

                        cam_vx = obs_raw[3]
                        cam_vy = obs_raw[4]

                        if phase == "settle":
                            action_str = f"0,0,0,0,{angle_rad:.6f}\n"
                            conn.sendall(action_str.encode('utf-8'))
                            test_step += 1

                            # Log velocity during settle to verify it reaches zero
                            if test_step == SETTLE_STEPS - 1:
                                vel_mag = math.sqrt(cam_vx**2 + cam_vy**2)
                                if vel_mag > 0.1:
                                    print(f"  WARNING: cam={angle_deg} still moving at settle end: vel={vel_mag:.2f}")

                            if test_step >= SETTLE_STEPS:
                                phase = "push"
                                test_step = 0

                        elif phase == "push":
                            # Pure forward
                            action_str = f"1.0,0,0,0,{angle_rad:.6f}\n"
                            conn.sendall(action_str.encode('utf-8'))

                            if test_step == MEASURE_STEP:
                                vel_mag = math.sqrt(cam_vx**2 + cam_vy**2)

                                if vel_mag > 0.01:
                                    cam_vel_angle = math.degrees(math.atan2(cam_vx, cam_vy)) % 360
                                    offset = cam_vel_angle
                                    if offset > 180: offset -= 360
                                else:
                                    cam_vel_angle = 0
                                    offset = 0

                                results[angle_deg] = {
                                    'cam_vel_angle': cam_vel_angle,
                                    'cam_offset': offset,
                                    'vel_mag': vel_mag,
                                    'cam_vx': cam_vx,
                                    'cam_vy': cam_vy,
                                }

                                print(f"  cam={angle_deg:>3} deg: vel_dir={cam_vel_angle:>6.1f} (offset={offset:>+6.1f}) |vel|={vel_mag:.3f} raw=({cam_vx:.4f},{cam_vy:.4f})")

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

            if results:
                print("\n" + "=" * 60)
                print("SUMMARY")
                print("=" * 60)
                offsets = [r['cam_offset'] for r in results.values()]
                print(f"  Offset: mean={sum(offsets)/len(offsets):+.2f} min={min(offsets):+.2f} max={max(offsets):+.2f}")
                print(f"  Spread: {max(offsets)-min(offsets):.2f} degrees")

                for angle in sorted(results.keys()):
                    r = results[angle]
                    print(f"    {angle:>3} deg: offset={r['cam_offset']:>+6.1f}  |v|={r['vel_mag']:.3f}")

                with open("camera_action_test2_results.json", "w") as f:
                    json.dump(results, f, indent=2)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sock.close()


if __name__ == '__main__':
    main()
