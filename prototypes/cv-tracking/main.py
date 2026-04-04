"""Brick for Brains -- main entry point.

Validates ArUco detection, color tracking, background subtraction,
Kalman filtering, and floor plane grid using a USB webcam or OAK-D Pro.

Controls:
  1 - ArUco marker detection mode
  2 - Color-based tracking mode
  3 - Background subtraction mode
  4 - Combined mode (ArUco + background subtraction)
  5 - Floor grid mode (ArUco + floor grid overlay)
  t - Toggle trail drawing
  k - Toggle Kalman filter overlay
  p - Calibrate floor plane (requires OAK-D Pro, arena must be empty)
  f - Cycle camera resolution (480p / 720p / 1080p)
  e - Decrease exposure (shorter = less motion blur)
  d - Increase exposure (longer = brighter but more blur)
  a - Toggle auto-exposure on/off
  g - Increase camera gain (brighter, compensates for short exposure)
  b - Decrease camera gain
  s - Toggle frame sharpening (unsharp mask)
  i - Query and save camera capabilities to camera_profiles.json
  q / ESC - Quit
"""

import json
import math
import os
import time

import cv2
import numpy as np

from capture import create_camera, DepthAICamera
from detectors import ArUcoDetector, ColorDetector, BackgroundSubDetector
from floor_plane import (FloorPlaneDetector, run_single_marker_calibration,
                         run_corner_calibration)
from kalman_tracker import KalmanTracker

MODE_NAMES = {1: "ArUco", 2: "Color", 3: "Background Sub", 4: "Combined", 5: "Floor Grid"}
PROFILES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'camera_profiles.json')


def save_camera_profile(caps):
    """Save camera capabilities to camera_profiles.json.

    Matches by device_index + backend. Replaces existing entry for the
    same camera, appends if new.
    """
    profiles = []
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE, 'r') as f:
            profiles = json.load(f)

    updated = False
    for i, p in enumerate(profiles):
        if (p.get('device_index') == caps['device_index'] and
                p.get('backend') == caps['backend']):
            profiles[i] = caps
            updated = True
            break

    if not updated:
        profiles.append(caps)

    with open(PROFILES_FILE, 'w') as f:
        json.dump(profiles, f, indent=2)

    return PROFILES_FILE


REPORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'camera_report.html')


def generate_camera_report():
    """Generate a standalone HTML report from camera_profiles.json."""
    if not os.path.exists(PROFILES_FILE):
        return

    with open(PROFILES_FILE, 'r') as f:
        profiles = json.load(f)

    cards_html = []
    for p in profiles:
        # Resolution rows
        res_rows = []
        for r in p.get('supported_resolutions', []):
            note = f' <span style="color:#999">(mapped from {r["requested"]})</span>' if r.get('requested') else ''
            fps_str = f'{r["fps"]:.1f}' if r['fps'] > 0 else '—'
            codec = r.get('fourcc', '—') or '—'
            res_rows.append(
                f'<tr><td>{r["width"]}x{r["height"]}</td><td>{codec}</td><td>{fps_str}</td><td>{note}</td></tr>'
            )

        # Property rows — skip -1 values (unsupported)
        prop_rows = []
        for k, v in p.get('properties', {}).items():
            if v is None or v == -1 or v == -1.0:
                continue
            if isinstance(v, float) and v == int(v):
                v = int(v)
            prop_rows.append(f'<tr><td>{k}</td><td>{v}</td></tr>')

        ts = p.get('timestamp', 'Unknown')
        codecs = ', '.join(p.get('supported_codecs', [])) or 'unknown'
        cur_res = p.get('current_resolution', [0, 0])
        cards_html.append(f'''
    <div class="cam-card">
      <h2>Camera {p.get('device_index', '?')} &mdash; {p.get('backend', '?')}</h2>
      <p class="timestamp">Last queried: {ts}</p>
      <p class="timestamp">Current resolution: {cur_res[0]}x{cur_res[1]} &bull; Supported codecs: {codecs}</p>
      <h3>Supported Resolutions</h3>
      <table>
        <thead><tr><th>Resolution</th><th>Codec</th><th>Measured FPS</th><th>Notes</th></tr></thead>
        <tbody>{''.join(res_rows)}</tbody>
      </table>
      <h3>Properties</h3>
      <table>
        <thead><tr><th>Property</th><th>Value</th></tr></thead>
        <tbody>{''.join(prop_rows)}</tbody>
      </table>
    </div>''')

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Camera Profiles &mdash; Brick for Brains</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #f4f6f9;
      color: #202124;
      line-height: 1.6;
      padding: 32px;
      max-width: 960px;
      margin: 0 auto;
    }}
    h1 {{
      font-size: 1.5rem;
      margin-bottom: 4px;
    }}
    .subtitle {{
      font-size: .875rem;
      color: #5f6368;
      margin-bottom: 24px;
    }}
    .cam-card {{
      background: #fff;
      border: 1px solid #e8eaed;
      border-left: 4px solid #1a73e8;
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,.08);
    }}
    .cam-card h2 {{
      font-size: 1.125rem;
      margin-bottom: 4px;
    }}
    .cam-card h3 {{
      font-size: .8125rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: #5f6368;
      margin: 16px 0 8px;
    }}
    .timestamp {{
      font-size: .8125rem;
      color: #5f6368;
      margin-bottom: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: .875rem;
      margin-bottom: 8px;
    }}
    thead th {{
      background: #f4f6f9;
      border-bottom: 2px solid #dadce0;
      padding: 8px 12px;
      text-align: left;
      font-weight: 600;
    }}
    tbody td {{
      padding: 6px 12px;
      border-bottom: 1px solid #e8eaed;
    }}
    tbody tr:hover {{ background: #e8f0fe; }}
    .count {{
      font-size: .875rem;
      color: #5f6368;
      margin-bottom: 20px;
    }}
    a {{ color: #1a73e8; }}
  </style>
</head>
<body>
  <h1>Camera Profiles</h1>
  <p class="subtitle">Generated by cv-tracking prototype &mdash; press <kbd>i</kbd> to query a camera</p>
  <p class="count">{len(profiles)} camera(s) profiled</p>
{''.join(cards_html)}
  <p style="margin-top:24px;font-size:.8125rem;color:#5f6368;">
    <a href="camera_profiles.json">Raw JSON data</a> &mdash;
    <a href="../../dashboard/prototypes.html">Back to Prototypes</a>
  </p>
</body>
</html>'''

    with open(REPORT_FILE, 'w') as f:
        f.write(html)

    return REPORT_FILE


def print_capabilities(caps):
    """Print camera capabilities to console."""
    print(f"\n{'=' * 50}")
    print(f"  Camera {caps['device_index']} - {caps['backend']}")
    print(f"  Queried: {caps['timestamp']}")
    print(f"{'=' * 50}")

    codecs = ', '.join(caps.get('supported_codecs', [])) or 'unknown'
    print(f"\n  Supported codecs: {codecs}")
    print(f"\n  Supported Resolutions:")
    print(f"    {'Resolution':<12} {'Codec':<6} {'FPS':>8}")
    print(f"    {'-'*12} {'-'*6} {'-'*8}")
    for r in caps['supported_resolutions']:
        note = f"  (from {r['requested']})" if r.get('requested') else ""
        codec = r.get('fourcc', '?') or '?'
        print(f"    {r['width']:>4}x{r['height']:<4}   {codec:<6} {r['fps']:>5.1f} FPS{note}")

    print(f"\n  Properties:")
    for k, v in caps['properties'].items():
        if v is not None:
            if isinstance(v, float) and v == int(v):
                v = int(v)
            print(f"    {k:<20} {v}")

    print(f"{'=' * 50}\n")


class ChaserBot:
    """Simulated tank-drive robot that rotates and drives to get behind the ArUco marker."""

    def __init__(self, width=40, length=50, forward_speed=1.5, turn_rate=0.04,
                 offset_distance=120, arrive_deadzone=25, angle_deadzone=0.15):
        self.width = width
        self.length = length
        self.forward_speed = forward_speed  # pixels per frame (slower)
        self.turn_rate = turn_rate  # radians per frame (slower)
        self.offset_distance = offset_distance
        self.arrive_deadzone = arrive_deadzone  # stop moving when within this many px
        self.angle_deadzone = angle_deadzone  # stop turning when within this many radians (~8 deg)
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0  # radians, 0 = pointing right
        self.initialized = False
        self.arrived = False
        self.trail = []
        self.max_trail = 80

    def _angle_diff(self, target, current):
        """Shortest signed angle from current to target."""
        diff = (target - current + math.pi) % (2 * math.pi) - math.pi
        return diff

    def update(self, detection):
        """Rotate toward the back of the ArUco marker, then drive forward."""
        if detection is None:
            return

        cx, cy = detection.center_px
        marker_heading = detection.heading_rad

        # Target: behind the marker (opposite of its heading)
        target_x = cx - self.offset_distance * math.cos(marker_heading)
        target_y = cy - self.offset_distance * math.sin(marker_heading)

        if not self.initialized:
            self.x = float(target_x)
            self.y = float(target_y)
            self.heading = math.atan2(target_y - self.y, target_x - self.x)
            self.initialized = True
            return

        # Distance and angle to the behind-target position
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.hypot(dx, dy)

        # Check if we've arrived at the behind position
        if dist < self.arrive_deadzone:
            self.arrived = True
            # We're in position — face the ArUco marker itself
            face_dx = cx - self.x
            face_dy = cy - self.y
            angle_to_marker = math.atan2(face_dy, face_dx)
            angle_error = self._angle_diff(angle_to_marker, self.heading)

            # Turn to face marker, but stop if within deadzone
            if abs(angle_error) > self.angle_deadzone:
                if abs(angle_error) > self.turn_rate:
                    self.heading += self.turn_rate * (1 if angle_error > 0 else -1)
                else:
                    self.heading = angle_to_marker
                self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi
            # Don't move — hold position
        else:
            self.arrived = False
            angle_to_target = math.atan2(dy, dx)
            angle_error = self._angle_diff(angle_to_target, self.heading)

            # Turn toward target (clamped by turn rate)
            if abs(angle_error) > self.angle_deadzone:
                if abs(angle_error) > self.turn_rate:
                    self.heading += self.turn_rate * (1 if angle_error > 0 else -1)
                else:
                    self.heading = angle_to_target

            # Normalize heading
            self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi

            # Only drive forward if roughly facing the target
            if abs(angle_error) < math.pi / 4:
                speed = min(self.forward_speed, dist * 0.3)  # slow down on approach
                speed = max(speed, 0.5)  # minimum creep speed
                self.x += speed * math.cos(self.heading)
                self.y += speed * math.sin(self.heading)

        # Trail
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

    def draw(self, frame):
        if not self.initialized:
            return

        # Draw trail
        for i in range(1, len(self.trail)):
            alpha = i / len(self.trail)
            color = (0, 0, int(180 * alpha + 75))
            cv2.line(frame, self.trail[i - 1], self.trail[i], color, 1)

        # Build rotated rectangle corners
        cx, cy = int(self.x), int(self.y)
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        hl = self.length / 2
        hw = self.width / 2

        # 4 corners relative to center, rotated by heading
        corners = []
        for lx, ly in [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]:
            rx = cx + int(lx * cos_h - ly * sin_h)
            ry = cy + int(lx * sin_h + ly * cos_h)
            corners.append((rx, ry))

        pts = np.array(corners, dtype=np.int32)

        # Draw filled body
        cv2.fillPoly(frame, [pts], (0, 0, 200))
        cv2.polylines(frame, [pts], True, (255, 255, 255), 2)

        # Draw "front" indicator (line from center toward heading)
        front_x = cx + int(hl * cos_h)
        front_y = cy + int(hl * sin_h)
        cv2.arrowedLine(frame, (cx, cy), (front_x, front_y), (0, 255, 255), 2, tipLength=0.4)

        # Draw tank treads (two lines along the sides)
        for sign in [-1, 1]:
            t1x = cx + int(-hl * cos_h - sign * hw * sin_h)
            t1y = cy + int(-hl * sin_h + sign * hw * cos_h)
            t2x = cx + int(hl * cos_h - sign * hw * sin_h)
            t2y = cy + int(hl * sin_h + sign * hw * cos_h)
            cv2.line(frame, (t1x, t1y), (t2x, t2y), (80, 80, 80), 4)

        # Label
        cv2.putText(frame, "CHASER", (cx - 25, cy - int(hw) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)


def main():
    # Initialize camera (auto-detect OAK-D, fall back to webcam at 720p)
    cam = create_camera(src=0, resolution_index=1, backend=cv2.CAP_DSHOW).start()
    # Let camera warm up
    time.sleep(0.5)

    # Initialize detectors
    aruco_det = ArUcoDetector()
    color_det = ColorDetector()
    bgsub_det = BackgroundSubDetector()

    # Kalman trackers -- one per tracked "slot"
    kalman_primary = KalmanTracker()    # "our robot" or primary tracked object
    kalman_secondary = KalmanTracker()  # "enemy" in combined mode

    # Floor plane detector (loads saved calibration if available)
    floor_det = FloorPlaneDetector()

    # Simulated chaser bot
    chaser = ChaserBot()

    # State
    mode = 1
    show_trail = True
    show_kalman = True
    process_fps = 0.0
    frame_count = 0
    fps_time = time.perf_counter()

    num_cameras = 2  # Camera 0 = built-in, Camera 1 = Logitech C270

    print("Brick for Brains")
    print("Controls: 1=ArUco  2=Color  3=BGSub  4=Combined  5=FloorGrid")
    print("          t=trail  k=kalman  p=calibrate floor (marker ID 10 on floor + click)")
    print("          f=resolution")
    print("          e/d=exposure-/+  a=auto-exposure  g/b=gain+/-  s=sharpen  c=camera")
    print("          i=camera info  q=quit")
    if floor_det.calibrated:
        print("Floor calibration loaded. Press '5' for grid mode.")

    while True:
        frame = cam.read()
        if frame is None:
            continue

        t_start = time.perf_counter()
        detections_primary = []
        detections_secondary = []

        # --- Detection based on mode ---
        if mode == 1:
            detections_primary = aruco_det.detect_and_draw(frame)
        elif mode == 2:
            detections_primary = color_det.detect(frame)
            color_det.draw(frame, detections_primary)
        elif mode == 3:
            detections_primary = bgsub_det.detect(frame)
            bgsub_det.draw(frame, detections_primary)
        elif mode == 4:
            detections_primary = aruco_det.detect_and_draw(frame)
            detections_secondary = bgsub_det.detect(frame)
            # In combined mode, filter out BGSub detections that overlap ArUco
            if detections_primary:
                aruco_centers = [d.center_px for d in detections_primary]
                filtered = []
                for det in detections_secondary:
                    overlap = False
                    for ac in aruco_centers:
                        dist = ((det.center_px[0] - ac[0]) ** 2 +
                                (det.center_px[1] - ac[1]) ** 2) ** 0.5
                        if dist < max(det.bbox[2], det.bbox[3]):
                            overlap = True
                            break
                    if not overlap:
                        filtered.append(det)
                detections_secondary = filtered
            bgsub_det.draw(frame, detections_secondary)
        elif mode == 5:
            # Floor grid mode: ArUco detection + floor grid overlay
            detections_primary = aruco_det.detect_and_draw(frame)
            floor_det.draw_grid(frame)
            # Show world coordinates for each detection
            for det in detections_primary:
                floor_det.draw_world_position(frame, det)

        # --- Kalman filter ---
        if show_kalman:
            # Primary tracker: use first detection
            pred_primary = kalman_primary.predict()
            if detections_primary:
                d = detections_primary[0]
                corrected = kalman_primary.update(d.center_px[0], d.center_px[1])
                kalman_primary.draw_prediction(frame, d.center_px, pred_primary)
            elif pred_primary:
                # No detection -- show prediction only
                cv2.circle(frame, pred_primary, 8, (0, 0, 255), 1)

            # Secondary tracker (combined mode)
            if mode == 4:
                pred_secondary = kalman_secondary.predict()
                if detections_secondary:
                    d = detections_secondary[0]
                    corrected = kalman_secondary.update(d.center_px[0], d.center_px[1])
                    kalman_secondary.draw_prediction(frame, d.center_px, pred_secondary)

        # --- Chaser bot (simulated enemy) ---
        if detections_primary and (mode == 1 or mode == 4):
            chaser.update(detections_primary[0])
        chaser.draw(frame)

        # --- Trails ---
        if show_trail:
            kalman_primary.draw_trail(frame, color=(0, 255, 0), predicted_color=(0, 100, 255))
            if mode == 4:
                kalman_secondary.draw_trail(frame, color=(255, 100, 0), predicted_color=(255, 0, 100))

        # --- Processing FPS ---
        t_process = (time.perf_counter() - t_start) * 1000  # ms
        frame_count += 1
        now = time.perf_counter()
        if now - fps_time >= 0.5:
            process_fps = frame_count / (now - fps_time)
            frame_count = 0
            fps_time = now

        # --- HUD overlay ---
        res = cam.resolution
        exp_label = "AUTO" if cam.auto_exposure else f"{cam.exposure} ({cam.exposure_us:.0f}us)"
        hud_lines = [
            f"Mode: {MODE_NAMES[mode]}",
            f"Capture: {cam.capture_fps:.0f} FPS | Process: {process_fps:.0f} FPS",
            f"Frame: {t_process:.1f}ms | Res: {res[0]}x{res[1]} | Cam: {cam.camera_index}",
            f"Exposure: {exp_label} | Gain: {cam.gain}",
            f"Trail: {'ON' if show_trail else 'OFF'} | Kalman: {'ON' if show_kalman else 'OFF'}"
            f" | Sharpen: {'ON' if cam.sharpen_enabled else 'OFF'}",
            f"Floor: {'CALIBRATED' if floor_det.calibrated else 'not calibrated (press p)'}",
        ]
        for i, line in enumerate(hud_lines):
            y = 22 + i * 22
            # Shadow for readability
            cv2.putText(frame, line, (11, y + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Detection count
        n_det = len(detections_primary) + len(detections_secondary)
        if n_det > 0:
            det_text = f"Detections: {n_det}"
            cv2.putText(frame, det_text, (11, res[1] - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.putText(frame, det_text, (10, res[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        cv2.imshow("Brick for Brains", frame)

        # --- Keyboard handling ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord('1'):
            mode = 1
            kalman_primary.reset()
            kalman_secondary.reset()
        elif key == ord('2'):
            mode = 2
            kalman_primary.reset()
        elif key == ord('3'):
            mode = 3
            kalman_primary.reset()
        elif key == ord('4'):
            mode = 4
            kalman_primary.reset()
            kalman_secondary.reset()
        elif key == ord('5'):
            if floor_det.calibrated:
                mode = 5
                kalman_primary.reset()
            else:
                print("Floor not calibrated. Press 'p' to calibrate first.")
        elif key == ord('p'):
            # Floor plane calibration
            # Try: single marker pose -> 4-corner ArUco -> manual click
            if frame is not None:
                camera_matrix, dist_coeffs = cam.get_intrinsics()
                result = run_single_marker_calibration(
                    cam, camera_matrix, dist_coeffs
                )
                if result is None:
                    print("Trying 4-corner ArUco (IDs 0-3)...")
                    test_det = FloorPlaneDetector()
                    if test_det.calibrate_from_aruco(frame):
                        result = test_det
                if result is None:
                    print("Falling back to manual click...")
                    result = run_corner_calibration(frame, rgb_size=cam.resolution)
                if result is not None:
                    floor_det = result
                    print("Calibration successful! Press '5' for grid mode.")
                else:
                    print("Calibration cancelled.")
        elif key == ord('t'):
            show_trail = not show_trail
        elif key == ord('k'):
            show_kalman = not show_kalman
        elif key == ord('f'):
            new_res = cam.cycle_resolution()
            kalman_primary.reset()
            kalman_secondary.reset()
            print(f"Resolution: {new_res[0]}x{new_res[1]}")
        # --- Exposure / gain / sharpen controls ---
        elif key == ord('e'):
            val = cam.adjust_exposure(-1)  # shorter exposure = less blur
            print(f"Exposure: {val} (~{cam.exposure_us:.0f}us)")
        elif key == ord('d'):
            val = cam.adjust_exposure(1)   # longer exposure = brighter
            print(f"Exposure: {val} (~{cam.exposure_us:.0f}us)")
        elif key == ord('a'):
            auto = cam.set_auto_exposure(not cam.auto_exposure)
            print(f"Auto-exposure: {'ON' if auto else 'OFF'}")
        elif key == ord('g'):
            val = cam.adjust_gain(16)
            print(f"Gain: {val}")
        elif key == ord('b'):
            val = cam.adjust_gain(-16)
            print(f"Gain: {val}")
        elif key == ord('s'):
            sharpening = cam.toggle_sharpen()
            print(f"Sharpening: {'ON' if sharpening else 'OFF'}")
        elif key == ord('i'):
            print("\nQuerying camera capabilities (brief pause)...")
            caps = cam.get_capabilities()
            print_capabilities(caps)
            path = save_camera_profile(caps)
            report = generate_camera_report()
            print(f"Saved to {path}")
            print(f"Report: {report}")
        elif key == ord('c'):
            if not isinstance(cam.camera_index, int):
                print(f"Camera switching not available for {cam.camera_index}")
            else:
                new_idx = (cam.camera_index + 1) % num_cameras
                print(f"Switching to camera {new_idx}...")
                if cam.switch_camera(new_idx):
                    print(f"Camera {new_idx}: {cam.resolution[0]}x{cam.resolution[1]}")
                    kalman_primary.reset()
                    kalman_secondary.reset()
                    chaser.initialized = False
                else:
                    print(f"Camera {new_idx} failed, staying on camera {cam.camera_index}")

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
