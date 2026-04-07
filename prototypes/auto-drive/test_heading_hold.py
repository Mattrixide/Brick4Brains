"""Test ESP32 heading hold mode (Phase 3 — straight-line driving test).

Two tests:
1. HOLD TEST: heading=0, speed=0 — push robot by hand, it should resist
2. DRIVE TEST: heading=0, speed=0.5 for 3 seconds — measure ArUco deviation

Logs ArUco position each frame to measure lateral deviation from a straight line.

Usage:
    python test_heading_hold.py --oakd             # run with OAK-D Pro
    python test_heading_hold.py --oakd --drive      # skip hold test, go straight to drive
    python test_heading_hold.py --dry-run           # no motors, just camera tracking
"""

import argparse
import math
import sys
import time

from comms import RobotComms
from tracker import ArUcoTracker, RobotPose, create_camera
from sensor_fusion import TelemetryReceiver

try:
    import cv2
except ImportError:
    print("OpenCV required: pip install opencv-python")
    sys.exit(1)


def run_hold_test(comms, camera, tracker, telemetry, duration=10.0):
    """Test 1: Hold heading at 0, speed 0. Push the robot — it should resist."""
    print("\n" + "=" * 60)
    print("TEST 1: HEADING HOLD (push resistance)")
    print(f"  Sending heading=0, speed=0 for {duration:.0f}s")
    print("  Push the robot — it should resist turning")
    print("  Press 'q' to skip")
    print("=" * 60)

    # Reset IMU yaw first by sending button Y (bit 3)
    comms.send(0.0, 0.0, 0x08)
    time.sleep(0.1)
    comms.send(0.0, 0.0, 0)
    time.sleep(0.3)

    # Quick motor check — nudge forward in direct mode to confirm connectivity
    print("  Motor check: forward pulse...")
    # Wait for telemetry to come alive first
    for _ in range(50):
        comms.send(0.0, 0.0, 0)
        time.sleep(0.02)
        if telemetry.is_active:
            break
    time.sleep(0.5)  # let ESCs arm at neutral

    hdg_before = telemetry.get()["heading"] if telemetry.is_active else None
    for _ in range(20):  # ~0.4s at 50Hz
        comms.send(0.5, 0.0)
        time.sleep(0.02)
    comms.send(0.0, 0.0, 0)
    time.sleep(0.3)
    hdg_after = telemetry.get()["heading"] if telemetry.is_active else None

    if hdg_before is not None and hdg_after is not None:
        moved = abs(hdg_after - hdg_before) > 0.5
        if moved:
            print(f"  Motor check OK (heading moved {hdg_before:.1f} -> {hdg_after:.1f})")
        else:
            print(f"  WARNING: Motors did not respond! Heading stayed at {hdg_before:.1f}")
            print("  >>> Please power-cycle the ESP32 and try again <<<")
            comms.send(0.0, 0.0, 0)
            return
    else:
        print("  WARNING: No telemetry — cannot verify motor response")
        print("  >>> Please check ESP32 is powered on <<<")
        return

    # Reset IMU yaw after motor check so heading mode starts from 0°
    comms.send(0.0, 0.0, 0x08)  # button Y = reset yaw
    time.sleep(0.1)
    comms.send(0.0, 0.0, 0)
    time.sleep(0.3)
    print("  IMU yaw reset. Starting heading hold...")

    start = time.monotonic()
    while time.monotonic() - start < duration:
        frame = camera.read()
        if frame is None:
            continue

        # Send heading hold command
        comms.send_heading(target_heading_deg=0.0, speed_norm=0.0)

        # Track ArUco
        poses = tracker.detect(frame)
        our_pose = next((d for d in poses if d["id"] == 1), None)  # marker ID 1

        # Show telemetry
        telem = telemetry.get()
        elapsed = time.monotonic() - start
        heading_str = f"hdg={telem['heading']:+.1f}" if telemetry.is_active else "hdg=N/A"
        gyro_str = f"gyro={telem['gyro_z']:+.1f}" if telemetry.is_active else "gyro=N/A"
        pos_str = ""
        if our_pose:
            cx, cy = our_pose["center"]
            pos_str = f" pos=({cx:.0f},{cy:.0f})"

        print(f"\r  [{elapsed:5.1f}s] {heading_str}  {gyro_str}{pos_str}    ", end="", flush=True)

        # Show camera feed
        cv2.imshow("Heading Hold Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n  Skipped.")
            break

    # Stop
    comms.send(0.0, 0.0, 0)
    print()


def run_drive_test(comms, camera, tracker, telemetry, speed=0.5, duration=3.0, max_travel_px=400):
    """Test 2: Drive straight at speed, measure ArUco lateral deviation."""
    print("\n" + "=" * 60)
    print("TEST 2: STRAIGHT-LINE DRIVE")
    print(f"  Sending heading=0, speed={speed} for {duration:.0f}s")
    print(f"  Safety cutoff: stops if marker moves >{max_travel_px}px from start")
    print("  Press 'q' to abort (motors stop immediately)")
    print("=" * 60)
    print("  Starting in 3 seconds — position robot now!")
    time.sleep(3)

    # Reset IMU yaw
    comms.send(0.0, 0.0, 0x08)
    time.sleep(0.1)
    comms.send(0.0, 0.0, 0)
    time.sleep(0.3)

    # Record start position
    positions = []  # (time, x, y, heading_deg)
    start_pose = None

    # Capture a few frames to get start position
    for _ in range(10):
        frame = camera.read()
        if frame is None:
            continue
        poses = tracker.detect(frame)
        det = next((d for d in poses if d["id"] == 1), None)
        if det is not None:
            start_pose = det
            break

    if start_pose is None:
        print("  ERROR: Cannot detect robot marker (ID 1) — aborting")
        return

    start_x, start_y = start_pose["center"]
    start_heading = start_pose["heading_rad"]
    print(f"  Start position: ({start_x:.0f}, {start_y:.0f})")

    # Drive!
    start = time.monotonic()
    aborted = False
    while time.monotonic() - start < duration:
        frame = camera.read()
        if frame is None:
            continue

        comms.send_heading(target_heading_deg=0.0, speed_norm=speed)

        poses = tracker.detect(frame)
        our_pose = next((d for d in poses if d["id"] == 1), None)
        elapsed = time.monotonic() - start

        if our_pose:
            cx, cy = our_pose["center"]
            hdg = our_pose["heading_rad"]
            positions.append((elapsed, cx, cy, hdg))
            # Lateral deviation: perpendicular distance from start heading line
            dx = cx - start_x
            dy = cy - start_y
            perp_angle = start_heading + math.pi / 2
            lateral = dx * math.cos(perp_angle) + dy * math.sin(perp_angle)
            dist = math.sqrt(dx**2 + dy**2)
            print(f"\r  [{elapsed:5.1f}s] pos=({cx:.0f},{cy:.0f}) "
                  f"dist={dist:.0f}px lateral={lateral:+.1f}px    ", end="", flush=True)

            # Safety cutoff: stop if too far from start
            if dist > max_travel_px:
                print(f"\n  SAFETY STOP: traveled {dist:.0f}px (limit {max_travel_px}px)")
                break

        cv2.imshow("Drive Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n  ABORTED by user")
            aborted = True
            break

    # Stop immediately
    comms.send(0.0, 0.0, 0)
    time.sleep(0.05)
    comms.send(0.0, 0.0, 0)
    print()

    if aborted or len(positions) < 5:
        print("  Not enough data to analyze.")
        return

    # Analyze deviation
    laterals = []
    for t, x, y, h in positions:
        dx = x - start_x
        dy = y - start_y
        perp_angle = start_heading + math.pi / 2
        lateral = dx * math.cos(perp_angle) + dy * math.sin(perp_angle)
        laterals.append(lateral)

    max_dev = max(abs(l) for l in laterals)
    avg_dev = sum(abs(l) for l in laterals) / len(laterals)
    final_dx = positions[-1][1] - start_x
    final_dy = positions[-1][2] - start_y
    total_dist = math.sqrt(final_dx**2 + final_dy**2)

    print("\n  RESULTS:")
    print(f"  Frames tracked:     {len(positions)}")
    print(f"  Distance traveled:  {total_dist:.1f} px")
    print(f"  Max lateral dev:    {max_dev:.1f} px")
    print(f"  Avg lateral dev:    {avg_dev:.1f} px")
    print(f"  Final lateral dev:  {laterals[-1]:+.1f} px")


def main():
    parser = argparse.ArgumentParser(description="Test ESP32 heading hold mode")
    parser.add_argument("--ip", default="192.168.4.194", help="ESP32 IP address")
    parser.add_argument("--oakd", action="store_true", help="Use OAK-D Pro camera")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (non-OAK-D)")
    parser.add_argument("--drive", action="store_true", help="Skip hold test, run drive test only")
    parser.add_argument("--speed", type=float, default=0.3, help="Drive test speed (0.0-1.0)")
    parser.add_argument("--duration", type=float, default=2.0, help="Drive test duration (seconds)")
    parser.add_argument("--dry-run", action="store_true", help="No motor commands")
    args = parser.parse_args()

    # Initialize comms
    comms = RobotComms(host=args.ip, port=4210)
    if args.dry_run:
        comms._dry_run = True
    comms.connect()

    # Initialize telemetry receiver
    telemetry = TelemetryReceiver()
    telemetry.start()

    # Initialize camera
    camera = create_camera(
        src=args.camera,
        use_oakd=args.oakd,
        target_fps=60.0,
    )
    camera.start()

    # Wait for camera
    print("Waiting for camera...")
    for _ in range(30):
        frame = camera.read()
        if frame is not None:
            break
        time.sleep(0.1)
    else:
        print("ERROR: Camera not ready")
        return

    # Initialize ArUco tracker
    tracker = ArUcoTracker()

    print(f"Camera ready. Telemetry: {'active' if telemetry.is_active else 'waiting...'}")

    try:
        if not args.drive:
            run_hold_test(comms, camera, tracker, telemetry)

        run_drive_test(comms, camera, tracker, telemetry,
                       speed=args.speed, duration=args.duration)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        # Always stop motors
        comms.send(0.0, 0.0, 0)
        comms.send(0.0, 0.0, 0)
        comms.close()
        telemetry.stop()
        camera.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
