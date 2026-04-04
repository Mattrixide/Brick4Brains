"""Debug enemy detection — standalone viewer.

Shows the camera feed with foreground mask overlay.
Press 'r' to capture reference frame (empty arena).
Press 'q' to quit.

Usage:
    python debug_enemy.py          # OAK-D Pro
    python debug_enemy.py --camera 0  # built-in webcam
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from tracker import create_camera
from enemy_tracker import EnemyDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=-1)
    parser.add_argument("--oakd", action="store_true", default=True)
    parser.add_argument("--threshold", type=int, default=30,
                        help="Reference diff threshold (0-255, default 30)")
    args = parser.parse_args()

    # Use webcam if --camera specified
    use_oakd = args.camera == -1
    cam_src = 0 if args.camera == -1 else args.camera

    print("Starting camera...")
    camera = create_camera(
        src=cam_src, resolution_index=1,
        use_oakd=use_oakd, target_fps=60.0
    ).start()
    time.sleep(1.0)

    detector = EnemyDetector(diff_threshold=args.threshold)

    # Load arena corners
    floor_path = os.path.join(os.path.dirname(__file__), "floor_calibration.json")
    if os.path.exists(floor_path):
        with open(floor_path) as f:
            data = json.load(f)
        corners = data.get("corners_px")
        if corners:
            detector.set_arena_corners(corners)
            print(f"Arena mask: {len(corners)} corners")

    print()
    print("Controls:")
    print("  r = capture reference frame (do this with empty arena)")
    print("  +/- = adjust diff threshold")
    print("  q = quit")
    print()
    print(f"Diff threshold: {detector._diff_threshold}")

    reference_captured = False

    while True:
        frame = camera.read()
        if frame is None:
            continue

        # Run detection (no robot to exclude since we're debugging)
        detection = detector.detect(frame, our_robot_corners=None)

        # Build display frame
        display = frame.copy()

        # Draw arena boundary
        if hasattr(detector, '_arena_pts') and detector._arena_pts is not None:
            cv2.polylines(display, [detector._arena_pts], True, (0, 255, 0), 2)

        # Draw detection
        if detection is not None:
            cx, cy = int(detection[0]), int(detection[1])
            cv2.circle(display, (cx, cy), 12, (0, 0, 255), 3)
            cv2.putText(display, "ENEMY", (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show FG mask alongside
        fg = detector.fg_mask
        if fg is not None:
            # Color the mask: white = foreground
            fg_color = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
            # Draw arena boundary on mask too
            if hasattr(detector, '_arena_pts') and detector._arena_pts is not None:
                cv2.polylines(fg_color, [detector._arena_pts], True, (0, 255, 0), 1)
            if detection is not None:
                cv2.circle(fg_color, (cx, cy), 12, (0, 0, 255), 3)

            # Stack side by side (resize mask to match)
            h, w = display.shape[:2]
            fg_resized = cv2.resize(fg_color, (w, h))
            combined = np.hstack([display, fg_resized])
        else:
            combined = display

        # Status text
        status = f"Threshold: {detector._diff_threshold} | Ref: {'YES' if reference_captured else 'NO (press r)'}"
        if detection:
            status += f" | DETECTED at ({int(detection[0])}, {int(detection[1])})"
        cv2.putText(combined, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Enemy Detection Debug", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.capture_reference(frame)
            reference_captured = True
            print(f"Reference captured! Threshold: {detector._diff_threshold}")
        elif key == ord('+') or key == ord('='):
            detector._diff_threshold = min(255, detector._diff_threshold + 5)
            print(f"Threshold: {detector._diff_threshold}")
        elif key == ord('-'):
            detector._diff_threshold = max(5, detector._diff_threshold - 5)
            print(f"Threshold: {detector._diff_threshold}")

    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
