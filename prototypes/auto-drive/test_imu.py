"""IMU vs ArUco heading comparison test.

Spins the robot slowly and compares:
  - ESP32 gyro yaw (from /api/imu)
  - ArUco marker heading (from camera)

Shows both headings on the CV window in real-time.
Press SPACE to start/stop spin. Press 'y' to reset IMU yaw. Press 'q' to quit.

Usage:
    python test_imu.py                        # OAK-D + ESP32
    python test_imu.py --camera 0             # built-in webcam + ESP32
    python test_imu.py --esp32 192.168.4.113  # custom ESP32 IP
"""

import argparse
import math
import time
import struct
import socket
import threading

import cv2
import numpy as np
import requests

from tracker import create_camera, ArUcoTracker, draw_overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esp32", default="192.168.4.113")
    parser.add_argument("--camera", type=int, default=-1)
    parser.add_argument("--spin-speed", type=float, default=0.3,
                        help="Spin steering value 0-1 (default 0.3)")
    args = parser.parse_args()

    use_oakd = args.camera == -1
    cam_src = 0 if args.camera == -1 else args.camera

    print("=== IMU vs ArUco Heading Test ===")
    print(f"ESP32: {args.esp32}")
    print(f"Camera: {'OAK-D Pro' if use_oakd else f'index {cam_src}'}")
    print()

    # Start camera
    print("Opening camera...")
    camera = create_camera(src=cam_src, resolution_index=1,
                           use_oakd=use_oakd, target_fps=60.0).start()
    time.sleep(1.0)

    tracker = ArUcoTracker(use_clahe=True)

    # UDP socket for motor commands
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    esp32_addr = (args.esp32, 4210)

    # Test ESP32 connection
    print(f"Testing ESP32 at {args.esp32}...")
    try:
        r = requests.get(f"http://{args.esp32}/api/imu", timeout=2)
        imu_data = r.json()
        print(f"  IMU ready: {imu_data['ready']}")
        print(f"  Yaw: {imu_data['yaw']:.1f}°")
        print()
    except Exception as e:
        print(f"  ESP32 not reachable: {e}")
        print("  Continuing without IMU data...")
        print()

    # Reset IMU yaw
    try:
        requests.post(f"http://{args.esp32}/api/imu", data={"resetYaw": "1"}, timeout=1)
        print("IMU yaw reset to 0")
    except:
        pass

    # Auto-calibrate tracker
    print("Waiting for ArUco marker...")
    deadline = time.time() + 10
    while time.time() < deadline:
        frame = camera.read()
        if frame is not None and tracker.auto_calibrate(frame, marker_id=0):
            break
        time.sleep(0.05)

    spinning = False
    spin_speed = args.spin_speed

    # History for plotting
    imu_history = []   # (time, yaw_deg)
    aruco_history = [] # (time, heading_deg)
    t0 = time.time()

    # Background IMU poller
    imu_yaw = [0.0]
    imu_lock = threading.Lock()

    def poll_imu():
        while True:
            try:
                r = requests.get(f"http://{args.esp32}/api/imu", timeout=0.5)
                data = r.json()
                with imu_lock:
                    imu_yaw[0] = data["yaw"]
            except:
                pass
            time.sleep(0.05)  # ~20Hz polling

    imu_thread = threading.Thread(target=poll_imu, daemon=True)
    imu_thread.start()

    print()
    print("Controls:")
    print("  SPACE = start/stop spin")
    print("  y     = reset IMU yaw to 0")
    print("  +/-   = adjust spin speed")
    print("  q     = quit")
    print()

    while True:
        frame = camera.read()
        if frame is None:
            continue

        # Detect ArUco
        pose = tracker.get_robot_pose(frame, marker_id=0)

        # Get current values
        t = time.time() - t0
        with imu_lock:
            current_imu_yaw = imu_yaw[0]

        aruco_heading_deg = None
        if pose is not None:
            aruco_heading_deg = math.degrees(pose.heading_rad)
            aruco_history.append((t, aruco_heading_deg))

        imu_history.append((t, current_imu_yaw))

        # Send motor command
        if spinning:
            # Slow spin: throttle=0, steering=spin_speed
            # Negate for ESP32 inversion
            steer_val = int(-spin_speed * 32767)
            packet = struct.pack(">hhB", 0, steer_val, 0)
        else:
            packet = struct.pack(">hhB", 0, 0, 0)
        udp_sock.sendto(packet, esp32_addr)

        # Draw display
        display = frame.copy()
        draw_overlay(display, pose)

        # Draw heading comparison
        h, w = display.shape[:2]

        # IMU heading indicator (red)
        imu_rad = math.radians(current_imu_yaw)
        cx, cy = w // 2, 120
        r_len = 80
        ix = int(cx + r_len * math.cos(imu_rad))
        iy = int(cy + r_len * math.sin(imu_rad))
        cv2.arrowedLine(display, (cx, cy), (ix, iy), (0, 0, 255), 3, tipLength=0.3)
        cv2.putText(display, f"IMU: {current_imu_yaw:.1f} deg", (cx - 80, cy - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ArUco heading indicator (green)
        if aruco_heading_deg is not None:
            ar_rad = math.radians(aruco_heading_deg)
            ax = int(cx + r_len * math.cos(ar_rad))
            ay = int(cy + r_len * math.sin(ar_rad))
            cv2.arrowedLine(display, (cx, cy), (ax, ay), (0, 255, 0), 3, tipLength=0.3)
            cv2.putText(display, f"ArUco: {aruco_heading_deg:.1f} deg", (cx - 80, cy + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Error
            err = current_imu_yaw - aruco_heading_deg
            cv2.putText(display, f"Error: {err:.1f} deg", (cx - 60, cy + 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Status
        status = f"{'SPINNING' if spinning else 'STOPPED'} | Speed: {spin_speed:.2f}"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Simple heading trace at bottom
        trace_y = h - 60
        trace_h = 50
        cv2.rectangle(display, (0, trace_y - trace_h), (w, trace_y + 10), (30, 30, 30), -1)

        # Draw last 200 points of history
        for hist, color in [(imu_history, (0, 0, 255)), (aruco_history, (0, 255, 0))]:
            pts = hist[-200:]
            if len(pts) < 2:
                continue
            t_min = pts[0][0]
            t_max = pts[-1][0]
            t_range = max(t_max - t_min, 0.1)
            for i in range(1, len(pts)):
                x1 = int((pts[i-1][0] - t_min) / t_range * w)
                x2 = int((pts[i][0] - t_min) / t_range * w)
                # Normalize heading to [-180, 180] for display
                y1 = int(trace_y - (pts[i-1][1] % 360) / 360 * trace_h)
                y2 = int(trace_y - (pts[i][1] % 360) / 360 * trace_h)
                cv2.line(display, (x1, y1), (x2, y2), color, 2)

        cv2.putText(display, "RED=IMU  GREEN=ArUco", (10, trace_y - trace_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow("IMU vs ArUco Test", display)

        # Log at 2Hz
        if not hasattr(main, '_log_t'):
            main._log_t = 0
        if t - main._log_t > 0.5:
            main._log_t = t
            aruco_str = f"{aruco_heading_deg:.1f}" if aruco_heading_deg is not None else "N/A"
            err_str = f"{current_imu_yaw - aruco_heading_deg:.1f}" if aruco_heading_deg is not None else "N/A"
            print(f"[t={t:.1f}s] IMU={current_imu_yaw:.1f}° ArUco={aruco_str}° Err={err_str}° {'SPIN' if spinning else 'STOP'}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            spinning = not spinning
            print(f"Spin: {'ON' if spinning else 'OFF'} (speed={spin_speed:.2f})")
        elif key == ord('y'):
            try:
                requests.post(f"http://{args.esp32}/api/imu",
                              data={"resetYaw": "1"}, timeout=1)
                print("IMU yaw reset to 0")
            except:
                print("Failed to reset IMU yaw")
        elif key == ord('+') or key == ord('='):
            spin_speed = min(1.0, spin_speed + 0.05)
            print(f"Spin speed: {spin_speed:.2f}")
        elif key == ord('-'):
            spin_speed = max(0.05, spin_speed - 0.05)
            print(f"Spin speed: {spin_speed:.2f}")

    # Stop motors
    for _ in range(5):
        udp_sock.sendto(struct.pack(">hhB", 0, 0, 0), esp32_addr)
    udp_sock.close()
    camera.stop()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
