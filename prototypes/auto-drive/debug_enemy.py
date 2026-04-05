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

        # Draw ALL contours that pass size filter (even if they fail other filters)
        fg = detector.fg_mask
        if fg is not None:
            contours, _ = cv2.findContours(fg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area > 100:  # show anything bigger than 100px
                    x, y, w, h = cv2.boundingRect(c)
                    passed = detector.MIN_AREA < area < detector.MAX_AREA
                    color = (0, 255, 0) if passed else (0, 0, 180)
                    cv2.rectangle(display, (x, y), (x+w, y+h), color, 1)
                    cv2.putText(display, f"{int(area)}", (x, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Draw detection + log state changes
        if detection is not None:
            cx, cy = int(detection[0]), int(detection[1])
            cv2.circle(display, (cx, cy), 12, (0, 0, 255), 3)
            cv2.putText(display, "ENEMY", (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not hasattr(main, '_was_detected'):
                main._was_detected = False
            if not main._was_detected:
                print(f"[detect] LOCKED at ({cx},{cy})")
                main._was_detected = True
        else:
            if hasattr(main, '_was_detected') and main._was_detected:
                # Count contours that passed size filter
                n_contours = 0
                if fg is not None:
                    for c in contours:
                        if cv2.contourArea(c) > 100:
                            n_contours += 1
                print(f"[detect] LOST — {n_contours} contours visible")
                main._was_detected = False

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
            # Save detection screenshot (once per detection, not every frame)
            if not hasattr(main, '_last_det_save'):
                main._last_det_save = 0
            now_t = time.time()
            if now_t - main._last_det_save > 2.0:
                main._last_det_save = now_t
                save_dir = os.path.dirname(os.path.abspath(__file__))
                cv2.imwrite(os.path.join(save_dir, "debug_detection.png"), combined)
                print(f"  Detection screenshot saved")
        cv2.putText(combined, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Enemy Detection Debug", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.capture_reference(frame)
            reference_captured = True
            # Save screenshot of the reference frame + current view
            save_dir = os.path.dirname(os.path.abspath(__file__))
            cv2.imwrite(os.path.join(save_dir, "debug_reference.png"), frame)
            cv2.imwrite(os.path.join(save_dir, "debug_combined.png"), combined)
            print(f"Reference captured! Threshold: {detector._diff_threshold}")
            print(f"  Saved debug_reference.png and debug_combined.png")
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
