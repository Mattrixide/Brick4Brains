"""Spin test: rotate left 360, then rotate right 360.
Manual keyboard control for testing TX15 + robot.

Uses external webcam to track ArUco ID #1 (50mm marker) and sends
steering/throttle commands to the robot via TX15 transmitter (SBUS).

TX15 Setup:
  1. Connect TX15 via USB
  2. EdgeTX: System -> USB -> Serial
  3. EdgeTX: Model -> Trainer -> Mode: Master/SBUS
  4. Note COM port from Device Manager

Usage:
  python main.py                        # dry-run (no TX15)
  python main.py --port COM3            # live with TX15 on COM3
  python main.py --port COM3 --camera 1 # specify camera index

Auto-test (SPACE):
  Rotate left 360 degrees, then rotate right 360 degrees.

Manual controls (hold key):
  W / UP    - Throttle forward
  S / DOWN  - Throttle reverse
  A / LEFT  - Steer left
  D / RIGHT - Steer right
  X         - Stop (center all)

Other:
  SPACE - Start/pause auto spin test
  r     - Reset
  +/-   - Adjust manual speed (10% steps)
  ESC/q - Quit
"""

import argparse
import math
import time

import cv2
import numpy as np

from tracker import ArucoTracker, ThreadedCamera
from tx_control import TX15Controller, list_serial_ports

# Auto-test parameters
SPIN_TARGET_DEG = 360.0
SPIN_STEER = 0.35            # steering value during auto-spin
SPIN_TOLERANCE_DEG = 10.0    # close enough to 360

# Manual control
MANUAL_SPEED_DEFAULT = 0.30  # default manual speed (normalized)
MANUAL_SPEED_STEP = 0.10


class SpinTest:
    """State machine: spin left 360, then spin right 360."""

    WAIT = "WAITING"
    SPIN_LEFT = "SPIN LEFT"
    SPIN_RIGHT = "SPIN RIGHT"
    DONE = "DONE"

    def __init__(self, tx):
        self.tx = tx
        self.state = self.WAIT
        self.prev_heading = None
        self.total_turn = 0.0
        self._state_enter_time = 0.0

    def start(self):
        self.state = self.WAIT
        self.prev_heading = None
        self.total_turn = 0.0
        print("[TEST] Waiting for marker to start spin test...")

    def reset(self):
        self.tx.stop()
        self.state = self.WAIT
        self.prev_heading = None
        self.total_turn = 0.0
        print("[TEST] Reset")

    def update(self, result):
        if result is None:
            self.tx.stop()
            return

        heading = result.heading_rad

        if self.state == self.WAIT:
            self.prev_heading = heading
            self.total_turn = 0.0
            self._state_enter_time = time.perf_counter()
            self.state = self.SPIN_LEFT
            print(f"[TEST] Initial heading: {math.degrees(heading):.1f} deg")
            print(f"[TEST] >>> SPINNING LEFT 360 degrees...")

        elif self.state == self.SPIN_LEFT:
            if self.prev_heading is not None:
                delta = ArucoTracker.angle_diff(heading, self.prev_heading)
                self.total_turn += delta
            self.prev_heading = heading

            # Spinning left = negative total_turn (counter-clockwise)
            turned_deg = abs(math.degrees(self.total_turn))

            if turned_deg >= (SPIN_TARGET_DEG - SPIN_TOLERANCE_DEG):
                self.tx.stop()
                elapsed = time.perf_counter() - self._state_enter_time
                print(f"[TEST] Left spin complete ({turned_deg:.1f} deg, {elapsed:.1f}s)")
                # Reset for right spin
                self.prev_heading = heading
                self.total_turn = 0.0
                self._state_enter_time = time.perf_counter()
                self.state = self.SPIN_RIGHT
                print(f"[TEST] >>> SPINNING RIGHT 360 degrees...")
                time.sleep(0.5)  # brief pause between spins
            else:
                # Steer left, no throttle (spin in place)
                self.tx.steer(-SPIN_STEER)
                self.tx.throttle(0.0)

        elif self.state == self.SPIN_RIGHT:
            if self.prev_heading is not None:
                delta = ArucoTracker.angle_diff(heading, self.prev_heading)
                self.total_turn += delta
            self.prev_heading = heading

            turned_deg = abs(math.degrees(self.total_turn))

            if turned_deg >= (SPIN_TARGET_DEG - SPIN_TOLERANCE_DEG):
                self.tx.stop()
                elapsed = time.perf_counter() - self._state_enter_time
                print(f"[TEST] Right spin complete ({turned_deg:.1f} deg, {elapsed:.1f}s)")
                self.state = self.DONE
                print("[TEST] >>> DONE!")
            else:
                # Steer right, no throttle
                self.tx.steer(SPIN_STEER)
                self.tx.throttle(0.0)

        elif self.state == self.DONE:
            self.tx.stop()


def draw_hud(frame, test, result, px_per_mm, fps, manual_mode, manual_speed):
    """Draw heads-up display overlay."""
    h, w = frame.shape[:2]

    lines = [
        f"State: {test.state}" if not manual_mode else "State: MANUAL",
        f"FPS: {fps:.0f}",
        f"TX: steer={test.tx.steer_value:+.2f} thr={test.tx.throttle_value:+.2f}",
    ]

    if manual_mode:
        lines.append(f"Manual speed: {manual_speed:.0%}")

    if result and px_per_mm > 0:
        lines.append(f"Scale: {px_per_mm:.2f} px/mm")
        lines.append(f"Heading: {math.degrees(result.heading_rad):.1f} deg")

        if test.state in (SpinTest.SPIN_LEFT, SpinTest.SPIN_RIGHT):
            turned = abs(math.degrees(test.total_turn))
            direction = "L" if test.state == SpinTest.SPIN_LEFT else "R"
            lines.append(f"Turned ({direction}): {turned:.1f} / {SPIN_TARGET_DEG:.0f} deg")
    else:
        lines.append("NO MARKER DETECTED")

    for i, line in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(frame, line, (11, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Controls help at bottom
    if manual_mode:
        msg = "WASD=drive  X=stop  +/-=speed  SPACE=auto  Q=quit"
    elif test.state == test.WAIT:
        msg = "SPACE=auto spin  WASD=manual  Q=quit"
    elif test.state == test.DONE:
        msg = "Done! R=reset  WASD=manual  Q=quit"
    else:
        msg = "SPACE=pause  R=reset  Q=quit"
    cv2.putText(frame, msg, (11, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(frame, msg, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="Spin test + manual control")
    parser.add_argument("--port", type=str, default=None,
                        help="TX15 serial port (e.g. COM3). Omit for dry-run.")
    parser.add_argument("--camera", type=int, default=1,
                        help="Camera index (default: 1 for external webcam)")
    parser.add_argument("--marker-id", type=int, default=1,
                        help="ArUco marker ID to track (default: 1)")
    parser.add_argument("--marker-size", type=float, default=50.0,
                        help="Marker side length in mm (default: 50)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    # List available serial ports
    print("Available serial ports:")
    for port, desc in list_serial_ports():
        print(f"  {port}: {desc}")
    print()

    # Initialize
    print(f"[INIT] Camera {args.camera} at {args.width}x{args.height}")
    cam = ThreadedCamera(src=args.camera, width=args.width, height=args.height).start()
    time.sleep(0.5)

    tracker = ArucoTracker(marker_id=args.marker_id, marker_size_mm=args.marker_size)

    tx = TX15Controller(port=args.port)
    tx.connect()

    test = SpinTest(tx)
    auto_running = False
    manual_mode = False
    manual_speed = MANUAL_SPEED_DEFAULT
    px_per_mm = 0.0

    # FPS tracking
    frame_count = 0
    fps_time = time.perf_counter()
    fps = 0.0

    print("\nSpin Test + Manual Control Ready")
    print(f"  Marker: ArUco 4x4 ID #{args.marker_id}, {args.marker_size}mm")
    print(f"  TX15: {'DRY RUN' if args.port is None else args.port}")
    print()
    print("  SPACE     = start auto spin test (left 360, right 360)")
    print("  W/A/S/D   = manual forward/left/back/right (hold)")
    print("  Arrow keys = same as WASD")
    print("  X         = stop")
    print("  +/-       = adjust manual speed")
    print("  R         = reset")
    print("  Q/ESC     = quit")
    print()

    while True:
        frame = cam.read()
        if frame is None:
            continue

        # Track
        result = tracker.detect(frame)
        tracker.draw(frame, result)

        if result:
            px_per_mm = result.px_per_mm

        # Update auto test if running
        if auto_running and not manual_mode and test.state != SpinTest.DONE:
            test.update(result)

        # FPS
        frame_count += 1
        now = time.perf_counter()
        if now - fps_time >= 0.5:
            fps = frame_count / (now - fps_time)
            frame_count = 0
            fps_time = now

        # Draw HUD
        draw_hud(frame, test, result, px_per_mm, fps, manual_mode, manual_speed)

        cv2.imshow("Drive Test", frame)

        # --- Keyboard handling ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            break

        elif key == ord(' '):
            if auto_running:
                auto_running = False
                manual_mode = False
                tx.stop()
                print("[TEST] Paused")
            else:
                auto_running = True
                manual_mode = False
                test.start()
                print("[TEST] Auto spin started!")

        elif key == ord('r'):
            auto_running = False
            manual_mode = False
            test.reset()

        # --- Manual drive keys (hold to send, release stops) ---
        elif key == ord('w') or key == 82:  # W or UP arrow
            manual_mode = True
            auto_running = False
            tx.throttle(manual_speed)

        elif key == ord('s') or key == 84:  # S or DOWN arrow
            manual_mode = True
            auto_running = False
            tx.throttle(-manual_speed)

        elif key == ord('a') or key == 81:  # A or LEFT arrow
            manual_mode = True
            auto_running = False
            tx.steer(-manual_speed)

        elif key == ord('d') or key == 83:  # D or RIGHT arrow
            manual_mode = True
            auto_running = False
            tx.steer(manual_speed)

        elif key == ord('x'):
            tx.stop()
            print("[MANUAL] Stop")

        elif key == ord('+') or key == ord('='):
            manual_speed = min(1.0, manual_speed + MANUAL_SPEED_STEP)
            print(f"[MANUAL] Speed: {manual_speed:.0%}")

        elif key == ord('-'):
            manual_speed = max(0.1, manual_speed - MANUAL_SPEED_STEP)
            print(f"[MANUAL] Speed: {manual_speed:.0%}")

        elif key == 255:
            # No key pressed -- if manual mode, stop sending
            if manual_mode:
                tx.stop()

    # Cleanup
    tx.disconnect()
    cam.stop()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
