"""Turn calibration test.

Sends turn commands to ESP32 and measures actual rotation via ArUco.
Compares commanded angle vs actual angle to calibrate direction and gain.

Press SPACE to execute next test turn.
Press 'r' to reset IMU yaw.
Press 'q' to quit.

Usage:
    python calibrate_turn.py
    python calibrate_turn.py --esp32 192.168.4.113
"""

import argparse
import math
import struct
import socket
import time

import cv2
import numpy as np
import requests

from tracker import create_camera, ArUcoTracker, draw_overlay


def send_turn(sock, addr, delta_deg):
    """Send a gyro-turn command (mode 1, 8-byte packet)."""
    delta_units = int(delta_deg * 100)
    delta_units = max(-32767, min(32767, delta_units))
    packet = struct.pack(">hhBBh", 0, 0, 0, 1, delta_units)
    sock.sendto(packet, addr)


def send_stop(sock, addr):
    """Send zero command (mode 0, 5-byte)."""
    sock.sendto(struct.pack(">hhB", 0, 0, 0), addr)


def get_imu_yaw(esp32_ip):
    """Poll IMU yaw from ESP32."""
    try:
        r = requests.get(f"http://{esp32_ip}/api/imu", timeout=0.3)
        return r.json().get("yaw", None)
    except:
        return None


def reset_imu_yaw(esp32_ip):
    """Reset IMU yaw to 0."""
    try:
        requests.post(f"http://{esp32_ip}/api/imu", data={"resetYaw": "1"}, timeout=0.5)
        print("[imu] Yaw reset to 0")
    except:
        print("[imu] Failed to reset yaw")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esp32", default="192.168.4.113")
    parser.add_argument("--mono", action="store_true", default=True,
                        help="Use OAK-D mono camera (120fps global shutter)")
    parser.add_argument("--fps", type=float, default=120.0)
    args = parser.parse_args()

    esp32_ip = args.esp32
    esp32_addr = (esp32_ip, 4210)

    print("=== Turn Calibration Test ===")
    print(f"ESP32: {esp32_ip}")
    print()

    # Camera + tracker
    cam_type = "mono 120fps" if args.mono else "color 60fps"
    print(f"Opening OAK-D Pro ({cam_type})...")
    camera = create_camera(src=0, resolution_index=1, use_oakd=True,
                           use_mono=args.mono, target_fps=args.fps).start()
    time.sleep(1.0)
    tracker = ArUcoTracker(use_clahe=True)

    # Wait for marker
    print("Waiting for ArUco marker...")
    deadline = time.time() + 15
    while time.time() < deadline:
        frame = camera.read()
        if frame is not None and tracker.auto_calibrate(frame, marker_id=0):
            break
        time.sleep(0.05)

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Test sequence: small and large turns in both directions
    test_angles = [45, 90, -45, -90, 180, -180, 30, -30]
    test_index = 0

    # Results
    results = []

    # Reset IMU
    reset_imu_yaw(esp32_ip)
    time.sleep(0.5)

    print()
    print("Controls:")
    print("  SPACE = execute next turn")
    print("  r     = reset IMU yaw")
    print("  q     = quit and show results")
    print()
    print(f"Test sequence: {test_angles}")
    print()

    turning = False
    turn_start_aruco = None
    turn_start_imu = None
    turn_cmd_deg = 0
    turn_start_time = 0

    while True:
        frame = camera.read()
        if frame is None:
            continue

        # Detect marker
        pose = tracker.get_robot_pose(frame, marker_id=0)
        aruco_heading = math.degrees(pose.heading_rad) if pose else None
        imu_yaw = get_imu_yaw(esp32_ip)

        # Heartbeat to prevent failsafe
        # During turn: re-send same turn command (ESP32 deduplicates by value)
        # When idle: send stop
        if turning:
            send_turn(sock, esp32_addr, turn_cmd_deg)  # deduplicated on ESP32
        else:
            send_stop(sock, esp32_addr)

        # Log during turn at 4Hz
        if turning:
            elapsed = time.time() - turn_start_time
            if not hasattr(main, '_last_turn_log'):
                main._last_turn_log = 0
            if elapsed - main._last_turn_log > 0.25:
                main._last_turn_log = elapsed
                aruco_d = (aruco_heading - turn_start_aruco + 180) % 360 - 180 if aruco_heading is not None else None
                imu_d = imu_yaw - turn_start_imu if imu_yaw is not None else None
                a_str = f"{aruco_d:+.1f}" if aruco_d is not None else "N/A"
                i_str = f"{imu_d:+.1f}" if imu_d is not None else "N/A"
                print(f"  t={elapsed:.1f}s | aruco_delta={a_str} imu_delta={i_str} | raw_aruco={aruco_heading} raw_imu={imu_yaw}")

        # Check if turn completed
        if turning and time.time() - turn_start_time > 0.3:
            elapsed = time.time() - turn_start_time
            if elapsed > 4.0:
                # Timeout — record what we got
                turning = False
                main._last_turn_log = 0
                aruco_delta = (aruco_heading - turn_start_aruco + 180) % 360 - 180 if aruco_heading is not None else 0
                imu_delta = imu_yaw - turn_start_imu if imu_yaw is not None else 0

                results.append({
                    "cmd": turn_cmd_deg,
                    "aruco_delta": aruco_delta,
                    "imu_delta": imu_delta,
                    "time": elapsed,
                })
                print(f"  RESULT: cmd={turn_cmd_deg:+.0f}  aruco={aruco_delta:+.1f}  imu={imu_delta:+.1f}  time={elapsed:.1f}s")
                print()
                if test_index < len(test_angles):
                    print(f"  Next: {test_angles[test_index]} (press SPACE)")

        # Draw display
        display = frame.copy()
        draw_overlay(display, pose)

        # Status
        if aruco_heading is not None:
            cv2.putText(display, f"ArUco: {aruco_heading:.1f} deg", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "ArUco: N/A", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if imu_yaw is not None:
            cv2.putText(display, f"IMU: {imu_yaw:.1f} deg", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if turning:
            elapsed = time.time() - turn_start_time
            cv2.putText(display, f"TURNING {turn_cmd_deg:+.0f} deg... ({elapsed:.1f}s)",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif test_index < len(test_angles):
            cv2.putText(display, f"Next: {test_angles[test_index]:+.0f} deg (SPACE)",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(display, "All tests done! (q to quit)", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show results so far
        y = 140
        for r in results:
            err = r["aruco_delta"] - r["cmd"]
            color = (0, 255, 0) if abs(err) < 10 else (0, 165, 255) if abs(err) < 20 else (0, 0, 255)
            cv2.putText(display, f"cmd={r['cmd']:+4.0f}  aruco={r['aruco_delta']:+6.1f}  imu={r['imu_delta']:+6.1f}  err={err:+.1f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y += 22

        cv2.imshow("Turn Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_imu_yaw(esp32_ip)
        elif key == ord(' ') and not turning and test_index < len(test_angles):
            # Start next turn
            if aruco_heading is None:
                print("  ERROR: ArUco not visible, can't measure turn")
                continue

            turn_cmd_deg = test_angles[test_index]
            turn_start_aruco = aruco_heading
            turn_start_imu = imu_yaw if imu_yaw is not None else 0
            turn_start_time = time.time()
            turning = True
            test_index += 1

            print(f"  Sending turn: {turn_cmd_deg:+.0f}°")
            send_turn(sock, esp32_addr, turn_cmd_deg)

    # Stop motors
    for _ in range(5):
        send_stop(sock, esp32_addr)
    sock.close()
    camera.stop()
    cv2.destroyAllWindows()

    # Print summary
    print()
    print("=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"{'CMD':>6}  {'ArUco':>8}  {'IMU':>8}  {'ArErr':>8}  {'IMUErr':>8}")
    for r in results:
        aruco_err = r["aruco_delta"] - r["cmd"]
        imu_err = r["imu_delta"] - r["cmd"]
        print(f"{r['cmd']:+6.0f}  {r['aruco_delta']:+8.1f}  {r['imu_delta']:+8.1f}  {aruco_err:+8.1f}  {imu_err:+8.1f}")

    if results:
        aruco_errs = [abs(r["aruco_delta"] - r["cmd"]) for r in results]
        imu_errs = [abs(r["imu_delta"] - r["cmd"]) for r in results]
        print(f"\nArUco mean abs error: {sum(aruco_errs)/len(aruco_errs):.1f}°")
        print(f"IMU mean abs error:   {sum(imu_errs)/len(imu_errs):.1f}°")

        # Check if IMU direction is inverted
        imu_signs = [1 if (r["imu_delta"] * r["cmd"]) > 0 else -1 for r in results if abs(r["cmd"]) > 10]
        if imu_signs and sum(imu_signs) < 0:
            print("\n!! IMU direction appears INVERTED -- gyro Z sign needs flipping")
        else:
            print("\nOK: IMU direction matches commanded turns")


if __name__ == "__main__":
    main()
