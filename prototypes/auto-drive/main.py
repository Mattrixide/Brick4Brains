"""
Auto-Drive Prototype — CV-guided autonomous robot control.

Combines ArUco marker tracking with ESP32 motor control, Xbox controller
override, and a web dashboard for mission management.

Usage:
    python main.py                                  # dry-run, camera 1
    python main.py --esp32 esp32wifi.local          # with ESP32
    python main.py --esp32 192.168.4.65 --camera 0  # custom camera
    python main.py --esp32 esp32wifi.local --show-cv # show OpenCV window
    python main.py --help
"""

import argparse
import json
import math
import os
import time
import threading
from collections import deque

import cv2
import numpy as np

from tracker import ThreadedCamera, DepthAICamera, ArUcoTracker, RobotPose, draw_overlay, create_camera
from comms import RobotComms
from controller import XboxController
from autonomy import (
    PathFollower,
    IMUAssistedPathFollower,
    get_available_missions,
    generate_square,
    generate_forward_back,
    generate_circle,
    generate_goto,
)
from dashboard_server import DashboardServer, create_shared_state
from sensor_fusion import HeadingFusion, TelemetryReceiver, IMUPoller
from enemy_tracker import EnemyTracker
from intercept import (
    compute_intercept_point,
    proportional_navigation,
    pure_pursuit,
    SmoothedIntercept,
    PursuitFSM,
    PursuitState,
)
from state_machine import BattleController, BattleContext
from battle_config import BattleConfig
from match_timer import MatchTimer, PinTimer
from keyboard_poll import KeyboardPoller

# Voice system (optional — graceful fallback if not available)
try:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "robot-voice"))
    from main import RobotAnnouncer
    _HAS_VOICE = True
except ImportError:
    _HAS_VOICE = False


# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------
MODE_IDLE = "idle"
MODE_AUTO = "auto"
MODE_MANUAL = "manual"
MODE_CALIBRATING = "calibrating"
MODE_INTERCEPT = "intercept"       # tracking enemy, waiting for trigger
MODE_INTERCEPT_CHARGE = "charging"  # actively pursuing enemy
MODE_PIN = "pinning"               # pinning enemy to wall
MODE_REVERSE = "reversing"         # backing away after pin
MODE_BATTLE = "battle"             # HSM combat state machine
MODE_READY = "ready"               # standing by, waiting for battle start
MODE_VERIFY = "verify"             # running system verification

# System mode lifecycle (overlays on top of robot mode)
SYSTEM_CONFIG = "config"
SYSTEM_PREMATCH = "prematch"
SYSTEM_BATTLE = "battle"
SYSTEM_POSTMATCH = "postmatch"


# ---------------------------------------------------------------------------
# Mission factory
# ---------------------------------------------------------------------------
MISSION_GENERATORS = {
    "square": lambda p: generate_square(p.get("size_cm", 60.0)),
    "drive_square": lambda p: generate_square(p.get("size_cm", 60.0)),
    "forward_back": lambda p: generate_forward_back(p.get("distance_cm", 60.0)),
    "circle": lambda p: generate_circle(
        p.get("radius_cm", 30.0), int(p.get("num_points", 8))
    ),
    "drive_circle": lambda p: generate_circle(
        p.get("radius_cm", 30.0), int(p.get("num_points", 8))
    ),
    "goto": lambda p: generate_goto(p.get("x_cm", 0.0), p.get("y_cm", 0.0)),
}


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class AutoDriveApp:
    def __init__(self, args):
        self.args = args
        self.mode = MODE_IDLE
        self.running = True
        self._system_mode = SYSTEM_CONFIG
        self._flourish_timer = None  # for victory dance

        # Voice system
        if _HAS_VOICE:
            try:
                self._voice = RobotAnnouncer(mode='vocoder')
                print("[voice] Brick's voice system online")
            except Exception:
                self._voice = None
        else:
            self._voice = None

        # Keyboard input — bypasses pygame/OpenCV message queue race
        self._keyboard = KeyboardPoller()

        # Xbox button edge detection
        self._prev_ctrl_buttons = 0

        # Verification state
        self._verified = False
        self._ready_log_t = 0.0

        # Frame logger (jsonl)
        self._frame_log_file = None
        self._frame_count = 0

        # Trail for dashboard visualization
        self.trail = deque(maxlen=200)

        # Timing
        self._last_update = time.perf_counter()
        self._loop_fps = 0.0
        self._fps_count = 0
        self._fps_timer = time.perf_counter()

        # Components
        self.camera = None
        self.tracker = ArUcoTracker(use_clahe=True)
        self.comms = RobotComms(host=args.esp32 or None, port=args.udp_port)
        self.controller = XboxController(deadzone=0.08)
        self.follower = PathFollower()

        # Heading EMA filter (kills ArUco jitter before it hits PID)
        self._heading_alpha = 0.35  # lower = smoother, higher = more responsive
        self._filtered_heading = None
        self._filtered_x = None
        self._filtered_y = None

        # IMU sensor fusion + telemetry
        self._heading_fusion = HeadingFusion()
        self._telemetry = TelemetryReceiver(port=4211)
        self._imu_poller = None  # started later if ESP32 is connected
        self._last_telemetry_time = 0.0

        # Heading-hold PID (IMU-based micro-corrections for straight driving)
        self._heading_hold_enabled = False
        self._heading_hold_target = None  # target heading in degrees (IMU frame)
        self._heading_hold_kp = 0.04
        self._heading_hold_ki = 0.0004
        self._heading_hold_kd = 0.003
        self._heading_hold_bias = 0.0
        self._heading_hold_integral = 0.0
        self._heading_hold_prev_error = 0.0
        self._load_drive_calibration()

        # Enemy tracking + interception
        self._enemy_tracker = EnemyTracker(dt=1/60, sigma_a=5.0, sigma_meas_cm=5.0)
        self._pursuit_fsm = PursuitFSM()
        self._smoothed_intercept = SmoothedIntercept(alpha=0.3)
        self._our_velocity = (0.0, 0.0)  # estimated from position changes
        self._prev_pos = None
        self._our_max_speed_cm_s = 50.0  # tune based on robot capability
        self._intercept_prev_steering = 0.0  # slew rate limiter for intercept

        # Battle state machine
        self._battle_config = BattleConfig.load(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "battle_config.json")
        )
        self._match_timer = MatchTimer(
            self._battle_config.match_duration_s,
            self._battle_config.urgency_ramp_start_s,
        )
        self._pin_timer = PinTimer(self._battle_config.pin_duration_s)
        self._battle_controller = BattleController(
            self._battle_config, self._match_timer, self._pin_timer
        )

        # Click-to-point state
        self._click_target_px = None

        # Pit calibration state (2-click workflow)
        self._pit_calibrating = False
        self._pit_corner1_px = None  # first click pixel coords
        self._pit_corner1_cm = None  # first click world coords

        # Measurement overlay (persists on frame until cleared)
        self._measure_line = None  # ((x1_px,y1_px), (x2_px,y2_px), dist_cm, label)

        # Stream encoding rate limiter (browser can't display >30fps anyway)
        self._stream_interval = 1.0 / 30.0
        self._last_stream_time = 0.0

        # Dashboard shared state
        self.shared_state = create_shared_state(
            esp32_host=args.esp32 or "(dry-run)",
        )
        self.dashboard = DashboardServer(self.shared_state, port=args.port)

        # Marker size and fallback scale
        self.tracker.set_marker_size(args.marker_size / 10.0)  # mm -> cm
        self.tracker.set_scale(args.px_per_cm)

        # Generate ChArUco board image for printing
        self._charuco_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "charuco_board.png"
        )
        if not os.path.exists(self._charuco_path):
            self.tracker.generate_charuco_board(self._charuco_path)

    def _say(self, text):
        """Non-blocking voice announcement (no-op if voice unavailable)."""
        if self._voice:
            try:
                self._voice.say(text)
            except Exception:
                pass

    def _verify_show(self, lines):
        """Show verification status on the OpenCV window."""
        frame = self.camera.read() if self.camera else None
        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        overlay = frame.copy()
        # Dark background box
        h, w = overlay.shape[:2]
        cv2.rectangle(overlay, (w//4, h//6), (3*w//4, 5*h//6), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        # Title
        y = h//6 + 40
        cv2.putText(frame, "SYSTEM VERIFICATION", (w//4 + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        y += 40
        for line in lines:
            color = (0, 255, 0) if "PASS" in line else (0, 0, 255) if "FAIL" in line else (200, 200, 200)
            if "..." in line:
                color = (255, 255, 100)
            cv2.putText(frame, line, (w//4 + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y += 30
        if self.args.show_cv:
            cv2.imshow("Auto-Drive", frame)
            cv2.waitKey(1)

    def _run_verification(self):
        """Run system verification — fast, non-blocking. Returns True if all pass."""
        results = []
        passed = 0

        # 1. ESP32
        esp_ok = self.comms.connected and not self.comms._dry_run
        results.append(f"[1/4] ESP32: {'PASS' if esp_ok else 'FAIL'}")
        passed += esp_ok

        # 2. Motors — tiny nudge (no sleep)
        if esp_ok:
            for _ in range(5):
                self.comms.send(0.15, 0.0)
            self.comms.stop()
            results.append("[2/4] Motors: PASS")
            passed += 1
        else:
            results.append("[2/4] Motors: FAIL (no ESP32)")

        # 3. IMU
        imu_ok = self._telemetry.is_active
        results.append(f"[3/4] IMU: {'PASS' if imu_ok else 'FAIL'}")
        passed += imu_ok

        # 4. Camera
        cam_ok = self.camera is not None and self.camera.read() is not None
        results.append(f"[4/4] Camera: {'PASS' if cam_ok else 'FAIL'}")
        passed += cam_ok

        # Show on screen + console
        if passed == 4:
            results.append("")
            results.append("VERIFIED - 4/4 passed")
        else:
            results.append("")
            results.append(f"INCOMPLETE - {passed}/4 passed")
        self._verify_show(results)

        print("=" * 40)
        for r in results:
            if r:
                print(f"  {r}")
        print("=" * 40)

        self._verified = passed == 4
        if self._verified:
            self._say("Systems verified.")
        else:
            self._say(f"Verification incomplete. {4 - passed} failed.")

        return self._verified

    def _emergency_stop(self):
        """Halt all systems immediately."""
        self.comms.stop()
        self.mode = MODE_IDLE
        self._system_mode = SYSTEM_CONFIG
        self._match_timer.reset()
        self._battle_controller.reset()
        self._flourish_timer = None
        self._say("Emergency stop")
        print("[EMERGENCY STOP] All systems halted")

    def _enter_ready(self):
        """Enter ready/standing-by mode."""
        self.mode = MODE_READY
        self._system_mode = SYSTEM_PREMATCH
        self._ready_log_t = 0.0
        self.comms.stop()


        print("[ready] READY — press B (Xbox) or Space to start battle")
        self._say("Ready.")

    def _start_battle(self):
        """Start battle mode from ready. Keep enemy tracking — don't reset it."""
        self._system_mode = SYSTEM_BATTLE
        self._battle_controller.reset()
        self._match_timer.reset()
        self._match_timer.start()
        self._pin_timer.reset()
        self._flourish_timer = None
        # Don't reset enemy tracker — preserve tracking from ready mode
        self.mode = MODE_BATTLE
        print("[battle] FIGHT! Match started!")
        self._say("Fight!")

    def start(self):
        """Initialize all components and run the main loop."""
        print("=" * 60)
        print("  Auto-Drive Prototype")
        print("=" * 60)

        # Start camera
        cam_label = "OAK-D Pro" if self.args.oakd else f"camera {self.args.camera}"
        target_fps = self.args.fps
        if self.args.mono:
            cam_label += f" (mono {target_fps:.0f}fps)"
        print(f"\n[camera] Opening {cam_label} ...")
        self.camera = create_camera(
            src=self.args.camera,
            resolution_index=1,  # 720p
            use_oakd=self.args.oakd,
            use_mono=self.args.mono,
            target_fps=target_fps,
        ).start()
        time.sleep(0.5)  # let camera warm up
        print(f"[camera] Running at {self.camera.fps:.0f} fps")

        # Pass real camera intrinsics to tracker if available (OAK-D)
        if hasattr(self.camera, 'intrinsics') and self.camera.intrinsics:
            intr = self.camera.intrinsics
            frame = self.camera.read()
            if frame is not None:
                fh, fw = frame.shape[:2]
                self.tracker.set_camera_matrix(
                    fw, fh,
                    fx=intr['fx'], fy=intr['fy'],
                    cx=intr['cx'], cy=intr['cy'],
                )

        # Connect ESP32
        print(f"\n[comms] Connecting to ESP32 ...")
        self.comms.connect()

        # Init Xbox controller
        print(f"\n[controller] Looking for Xbox controller ...")
        has_controller = self.controller.init()
        if not has_controller:
            print("[controller] No controller found — manual override disabled")

        # Set origin to center of first detected frame
        print(f"\n[tracker] Waiting for first marker detection ...")
        self._calibrate_origin()

        # Try to load saved calibration (floor plane first, then legacy homography)
        floor_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "floor_calibration.json"
        )
        legacy_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "homography.json"
        )
        self._floor_det = None
        if os.path.exists(floor_path):
            self.tracker.load_floor_plane(floor_path)
            from floor_plane import FloorPlaneDetector
            self._floor_det = FloorPlaneDetector()  # loads from floor_calibration.json
        elif os.path.exists(legacy_path):
            self.tracker.load_homography(legacy_path)

        # Load arena corners for enemy detection masking
        floor_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "floor_calibration.json"
        )
        if os.path.exists(floor_path):
            with open(floor_path) as f:
                floor_data = json.load(f)
            corners_px = floor_data.get("corners_px")
            if corners_px:
                self._enemy_tracker.detector.set_arena_corners(corners_px)

        # Load saved reference frame (empty arena) for static enemy detection
        # Press 'r' during runtime to capture a new one when arena is empty
        ref_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "arena_reference.png"
        )
        if os.path.exists(ref_path):
            saved_ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if saved_ref is not None:
                self._enemy_tracker.detector._reference_gray = cv2.GaussianBlur(
                    saved_ref, (5, 5), 0
                )
                print(f"[enemy] Loaded saved reference frame from {ref_path}")
        else:
            print("[enemy] No saved reference — press 'r' with empty arena to capture one")

        # Start IMU telemetry receiver
        self._telemetry.start()

        # Start IMU HTTP poller if ESP32 is connected
        if args.esp32:
            esp32_ip = self.comms._addr[0] if self.comms._addr else args.esp32
            self._imu_poller = IMUPoller(host=esp32_ip)
            self._imu_poller.start()

        # Upgrade to IMU-assisted follower if ESP32 is connected
        if args.esp32:
            self.follower = IMUAssistedPathFollower(
                comms=self.comms,
                sensor_fusion=self._heading_fusion,
                telemetry=self._telemetry,
            )
            print("[main] IMU-assisted path follower enabled")

        # Start dashboard with video feed callback
        self._latest_jpeg = None
        self._jpeg_lock = threading.Lock()
        self.dashboard._frame_callback = self._get_jpeg
        self.dashboard.start()
        print(f"\n[dashboard] Running at http://localhost:{self.args.port}")

        print(f"\n[main] Ready — mode: {self.mode.upper()}")
        if self.args.show_cv:
            print("[main] Press 'q' in CV window to quit")
            print("[main] Click on CV window to set goto target")
        print("[main] Press Ctrl+C to stop\n")

        # Main loop
        try:
            self._run_loop()
        except KeyboardInterrupt:
            print("\n[main] Shutting down ...")
        finally:
            self._shutdown()

    def _calibrate_origin(self):
        """Auto-calibrate using solvePnP with known marker size.

        Falls back to simple px_per_cm if marker not found.
        """
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.05)
                continue
            if self.tracker.auto_calibrate(frame, marker_id=self.args.marker_id):
                return
            time.sleep(0.05)

        # Fallback: use simple scale + frame center origin
        print("[tracker] Auto-calibration failed — using simple px_per_cm fallback")
        frame = self.camera.read()
        if frame is not None:
            h, w = frame.shape[:2]
            self.tracker.set_origin(w / 2, h / 2)
            print(f"[tracker] Origin set to frame center ({w//2}, {h//2})")
        else:
            print("[tracker] WARNING: no frames available")

    def _load_drive_calibration(self):
        """Load heading-hold PID calibration from drive_calibration.json."""
        cal_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "drive_calibration.json"
        )
        if os.path.exists(cal_path):
            try:
                with open(cal_path) as f:
                    cal = json.load(f)
                self._heading_hold_kp = cal.get("heading_hold_kp", self._heading_hold_kp)
                self._heading_hold_ki = cal.get("heading_hold_ki", self._heading_hold_ki)
                self._heading_hold_kd = cal.get("heading_hold_kd", self._heading_hold_kd)
                self._heading_hold_bias = cal.get("steering_bias", 0.0)
                self._heading_hold_enabled = True
                print(f"[heading-hold] Loaded calibration: "
                      f"Kp={self._heading_hold_kp:.4f} Ki={self._heading_hold_ki:.6f} "
                      f"Kd={self._heading_hold_kd:.4f} bias={self._heading_hold_bias:+.4f}")
            except Exception as e:
                print(f"[heading-hold] Failed to load calibration: {e}")
        else:
            print("[heading-hold] No drive_calibration.json found — run calibrate_drive.py first")

    def _heading_hold_correction(self, throttle: float, steering: float, dt: float) -> float:
        """Apply IMU-based heading-hold micro-correction to steering.

        When driving forward (throttle > 0), locks the IMU heading at the start
        of the drive and applies PID corrections to maintain that heading.
        The navigation PID from autonomy.py sets the desired heading via
        waypoint targeting; this layer keeps the robot tracking that heading
        between camera frames using fast IMU feedback.

        Returns adjusted steering value.
        """
        if not self._heading_hold_enabled:
            return steering
        if not self._imu_poller or not self._imu_poller.is_active:
            return steering

        imu_heading = self._imu_poller.get_yaw()

        # Only apply heading-hold when driving mostly straight
        # If nav PID is commanding large steering (turning), don't fight it
        if abs(steering) > 0.3:
            # Large turn commanded — release heading lock, let nav PID handle it
            self._heading_hold_target = None
            self._heading_hold_integral = 0.0
            return steering

        # When throttle goes from 0 to positive, lock the current heading
        if abs(throttle) > 0.1:
            if self._heading_hold_target is None:
                # New drive segment — lock heading to current IMU reading
                self._heading_hold_target = imu_heading
                self._heading_hold_integral = 0.0
                self._heading_hold_prev_error = 0.0
            else:
                # Slowly slew target toward current heading when nav steers
                # This lets the nav PID adjust course without fighting
                if abs(steering) > 0.02:
                    heading_rate = steering * 60.0  # deg/s at full steering
                    self._heading_hold_target += heading_rate * dt
                    # Keep target near actual heading (don't let it run away)
                    drift = self._heading_hold_target - imu_heading
                    drift = (drift + 180.0) % 360.0 - 180.0
                    if abs(drift) > 20.0:
                        self._heading_hold_target = imu_heading
        else:
            # Not driving — release heading lock
            self._heading_hold_target = None
            self._heading_hold_integral = 0.0
            return steering

        # PID on heading error
        error = self._heading_hold_target - imu_heading
        # Wrap to [-180, 180]
        error = (error + 180.0) % 360.0 - 180.0

        self._heading_hold_integral += error * dt
        # Tight anti-windup
        self._heading_hold_integral = max(-5.0, min(5.0, self._heading_hold_integral))

        derivative = (error - self._heading_hold_prev_error) / dt if dt > 0.001 else 0.0
        self._heading_hold_prev_error = error

        correction = (self._heading_hold_kp * error +
                      self._heading_hold_ki * self._heading_hold_integral +
                      self._heading_hold_kd * derivative +
                      self._heading_hold_bias)

        # Clamp correction to small micro-adjustments only
        # This is a trim layer, not a steering controller
        max_correction = 0.15
        correction = max(-max_correction, min(max_correction, correction))

        # Blend: add correction to nav steering
        adjusted = steering + correction
        return max(-1.0, min(1.0, adjusted))

    def _run_loop(self):
        """Core loop: track, decide, act."""
        while self.running:
            now = time.perf_counter()
            dt = now - self._last_update
            self._last_update = now

            # 1. Read camera and track
            frame = self.camera.read()
            pose = None
            if frame is not None:
                pose = self.tracker.get_robot_pose(frame, marker_id=self.args.marker_id)

            # 1b. Update sensor fusion with IMU
            if self._imu_poller and self._imu_poller.is_active:
                self._heading_fusion.update_imu(self._imu_poller.get_yaw())
            elif self._telemetry.is_active:
                tel = self._telemetry.get()
                self._heading_fusion.update_gyro(tel["gyro_z"], dt)

            # 2. Convert to world coordinates + EMA filter
            # When ArUco lost, use IMU-fused heading instead of defaulting to 0
            if self._heading_fusion.is_calibrated:
                heading_rad = self._heading_fusion.heading_rad
            else:
                heading_rad = 0.0
            x_cm, y_cm = 0.0, 0.0
            detected = False
            if pose is not None:
                detected = True
                raw_x, raw_y = self.tracker.px_to_cm(pose.x_px, pose.y_px)

                # Compute heading in world coordinates by transforming
                # two points along the marker's forward direction
                hdist = 20.0  # pixel offset along heading
                fwd_px_x = pose.x_px + hdist * math.cos(pose.heading_rad)
                fwd_px_y = pose.y_px + hdist * math.sin(pose.heading_rad)
                try:
                    fwd_x, fwd_y = self.tracker.px_to_cm(fwd_px_x, fwd_px_y)
                    raw_heading = math.atan2(fwd_y - raw_y, fwd_x - raw_x)
                except (ValueError, cv2.error):
                    raw_heading = pose.heading_rad

                # EMA filter on position and heading
                a = self._heading_alpha
                if self._filtered_heading is None:
                    self._filtered_x = raw_x
                    self._filtered_y = raw_y
                    self._filtered_heading = raw_heading
                else:
                    self._filtered_x = a * raw_x + (1 - a) * self._filtered_x
                    self._filtered_y = a * raw_y + (1 - a) * self._filtered_y
                    # Angle-aware EMA: use angle_diff to avoid wrapping issues
                    hdiff = math.atan2(
                        math.sin(raw_heading - self._filtered_heading),
                        math.cos(raw_heading - self._filtered_heading),
                    )
                    self._filtered_heading += a * hdiff
                    # Wrap to [-pi, pi] to prevent unbounded drift
                    self._filtered_heading = math.atan2(
                        math.sin(self._filtered_heading),
                        math.cos(self._filtered_heading),
                    )

                x_cm = self._filtered_x
                y_cm = self._filtered_y
                heading_rad = self._filtered_heading
                self.trail.append((x_cm, y_cm))

                # Update sensor fusion with CV heading
                self._heading_fusion.update_cv(heading_rad)

                # Estimate our velocity from position changes
                if self._prev_pos is not None:
                    vx = (x_cm - self._prev_pos[0]) / max(dt, 0.001)
                    vy = (y_cm - self._prev_pos[1]) / max(dt, 0.001)
                    self._our_velocity = (vx, vy)
                self._prev_pos = (x_cm, y_cm)

            if pose is None and detected is False:
                self._heading_fusion.update_no_cv()

            # 2b. Enemy tracking (run in ALL intercept-related modes)
            if frame is not None and self.mode in (MODE_READY, MODE_INTERCEPT, MODE_INTERCEPT_CHARGE, MODE_PIN, MODE_REVERSE, MODE_BATTLE):
                # Pass ArUco corners for color-based classification (not exclusion mask)
                our_corners = pose.corners if pose is not None else None
                self._enemy_tracker.update(
                    frame, our_corners,
                    px_to_cm=self.tracker.px_to_cm,
                )

            # 3. Read controller + keyboard
            ctrl = self.controller.read()
            keys = self._keyboard.poll()  # hardware key state, no message queue

            # 3a. Handle keyboard input (GetAsyncKeyState — never eaten by pygame)
            if 'q' in keys:
                self.running = False
                break
            if 'v' in keys:
                if time.perf_counter() - getattr(self, '_last_verify_t', 0) > 2.0:
                    self._last_verify_t = time.perf_counter()
                    self._run_verification()
            if 'b' in keys:
                if self.mode == MODE_BATTLE:
                    self._emergency_stop()
                elif self.mode == MODE_READY:
                    self._start_battle()
                else:
                    self._start_battle()
            if ' ' in keys:
                if self.mode == MODE_READY:
                    self._start_battle()

            # 3b. Xbox button edge detection (rising edge = press)
            btn_pressed = ctrl.buttons & ~self._prev_ctrl_buttons
            self._prev_ctrl_buttons = ctrl.buttons
            BTN_B = 0x02       # bit 1
            BTN_BACK = 0x40    # bit 6
            BTN_START = 0x80   # bit 7

            # Back button → emergency stop from any mode
            if btn_pressed & BTN_BACK:
                self._emergency_stop()

            # Start button
            elif btn_pressed & BTN_START:
                if self.mode in (MODE_IDLE, MODE_MANUAL):
                    self._enter_ready()
                elif self.mode in (MODE_BATTLE, MODE_READY):
                    self._emergency_stop()

            # B button → start battle from ready
            elif btn_pressed & BTN_B:
                if self.mode == MODE_READY:
                    self._start_battle()

            # 4. Process dashboard commands
            self._process_dashboard_commands()

            # 5. State machine
            throttle, steering, buttons = 0.0, 0.0, 0

            if self.mode == MODE_MANUAL:
                # Pass through controller input
                throttle = ctrl.throttle
                steering = ctrl.steering
                buttons = ctrl.buttons

            elif self.mode == MODE_AUTO:
                # Check for controller override — require deliberate input
                # (higher threshold than deadzone to avoid stick drift triggering)
                override = (abs(ctrl.throttle) > 0.25 or
                            abs(ctrl.steering) > 0.25 or
                            ctrl.buttons != 0)
                if override:
                    print("[main] Xbox override detected — switching to MANUAL")
                    self.mode = MODE_MANUAL
                    self.follower = PathFollower()  # reset
                    throttle = ctrl.throttle
                    steering = ctrl.steering
                    buttons = ctrl.buttons
                elif detected:
                    # Run autonomy
                    throttle, steering, done, status = self.follower.update(
                        x_cm, y_cm, heading_rad, dt
                    )
                    # Log at 2Hz
                    if not hasattr(self, '_auto_log_t'):
                        self._auto_log_t = 0
                    if now - self._auto_log_t > 0.5:
                        self._auto_log_t = now
                        print(f"[auto] robot=({x_cm:.0f},{y_cm:.0f}) hdg={math.degrees(heading_rad):.0f}° "
                              f"thr={throttle:.2f} str={steering:.2f} | {status}")
                    if done:
                        print(f"[auto] {status}")
                        self.mode = MODE_IDLE
                else:
                    # Marker lost — keep driving with last known heading if IMU available
                    if (self._heading_fusion.is_calibrated and
                        self._heading_fusion.frames_without_cv < 30 and
                        self.follower.active):
                        # Use last known position + IMU heading to continue
                        # Keep last throttle/steering — don't stop
                        if not hasattr(self, '_lost_log_t'):
                            self._lost_log_t = 0
                        if now - self._lost_log_t > 1.0:
                            self._lost_log_t = now
                            print(f"[auto] MARKER LOST — coasting on IMU "
                                  f"(hdg={self._heading_fusion.heading_deg:.0f}°, "
                                  f"frames={self._heading_fusion.frames_without_cv})")
                    else:
                        # No IMU or lost too long — stop
                        throttle = 0.0
                        steering = 0.0
                        if not hasattr(self, '_lost_log_t'):
                            self._lost_log_t = 0
                        if now - self._lost_log_t > 1.0:
                            self._lost_log_t = now
                            print("[auto] MARKER LOST — stopping (no IMU or timeout)")

            elif self.mode == MODE_CALIBRATING:
                # Drive with controller while collecting calibration points
                throttle = ctrl.throttle
                steering = ctrl.steering
                buttons = ctrl.buttons
                # Collect calibration data each frame
                if frame is not None:
                    captured, total = self.tracker.update_calibration_drive(
                        frame, marker_id=self.args.marker_id
                    )
                    with self.shared_state["lock"]:
                        self.shared_state["calib_points"] = total

            elif self.mode == MODE_INTERCEPT:
                # Tracking mode — detect enemy but don't move
                # SPACE triggers the charge
                throttle = 0.0
                steering = 0.0
                if self._enemy_tracker.is_tracking:
                    enemy_pos = self._enemy_tracker.position_cm
                    distance = math.hypot(enemy_pos[0] - x_cm, enemy_pos[1] - y_cm) if detected else 0
                    if not hasattr(self, '_track_log_t'):
                        self._track_log_t = 0
                    if now - self._track_log_t > 1.0:
                        self._track_log_t = now
                        print(f"[track] enemy=({enemy_pos[0]:.0f},{enemy_pos[1]:.0f}) "
                              f"dist={distance:.0f}cm — press SPACE to charge")

            elif self.mode == MODE_INTERCEPT_CHARGE:
                # Check for controller override
                override = (abs(ctrl.throttle) > 0.25 or
                            abs(ctrl.steering) > 0.25 or
                            ctrl.buttons != 0)
                if override:
                    print("[main] Xbox override — switching to MANUAL")
                    self.mode = MODE_IDLE
                    self._pursuit_fsm.reset()
                    throttle = ctrl.throttle
                    steering = ctrl.steering
                elif detected and self._enemy_tracker.is_tracking:
                    enemy_pos = self._enemy_tracker.position_cm
                    enemy_vel = self._enemy_tracker.velocity_cm_s
                    our_pos = (x_cm, y_cm)
                    distance = math.hypot(enemy_pos[0] - x_cm, enemy_pos[1] - y_cm)

                    state = self._pursuit_fsm.update_with_distance(
                        self._enemy_tracker.enemy_detected,
                        self._enemy_tracker.is_tracking,
                        self._enemy_tracker.kalman.frames_without_detection,
                        distance,
                    )

                    # Pure pursuit arc driving — never stop, always drive + steer
                    from autonomy import angle_diff
                    desired_heading = math.atan2(
                        enemy_pos[1] - y_cm, enemy_pos[0] - x_cm
                    )
                    # Use the same heading source as MODE_AUTO (CV + EMA filtered)
                    # Previously used IMU-fused heading which could diverge
                    use_heading = math.atan2(math.sin(heading_rad), math.cos(heading_rad))
                    alpha = angle_diff(desired_heading, use_heading)

                    # HIT — check if enemy is near wall for pin
                    if distance < 15.0:
                        # Check if enemy is near arena wall
                        # (arena roughly ±120cm, wall if any axis > 90cm)
                        ex, ey = enemy_pos[0], enemy_pos[1]
                        near_wall = abs(ex) > 80 or abs(ey) > 80

                        if near_wall:
                            # PIN: enemy already at wall — hold
                            throttle = 0.2
                            steering = 0.0
                            self._pin_start_time = now
                            self.mode = MODE_PIN
                            print(f"[intercept] PIN! enemy at wall ({ex:.0f},{ey:.0f}) — 5s hold")
                        else:
                            # Mid-arena ram — FULL POWER push to wall
                            throttle = 1.0
                            steering = 0.0
                            self._pin_start_time = now
                            self.mode = MODE_PIN
                            print(f"[intercept] RAM! dist={distance:.0f}cm — PUSHING to wall")
                    elif state == PursuitState.LOST:
                        # Keep driving toward Kalman-predicted enemy position
                        # (don't stop just because detection dropped for a few frames)
                        lookahead = max(20.0, distance * 0.5)
                        track_width = 15.0
                        turn_factor = track_width * math.sin(alpha) / lookahead
                        steering = max(-0.5, min(0.5, turn_factor * 1.2))
                        throttle = 0.6  # reduced speed while coasting
                    elif state == PursuitState.SEARCH:
                        throttle = 0.0
                        steering = 0.0
                    elif abs(alpha) > 1.0:
                        # Way off (>57°) — spin to face enemy (no slew needed)
                        throttle = 0.0
                        steering = 0.6 if alpha > 0 else -0.6
                        self._charge_prev_steer = 0.0
                    else:
                        # PURE PURSUIT: drive forward + steer arc toward enemy
                        lookahead = max(20.0, distance * 0.5)
                        track_width = 15.0
                        turn_factor = track_width * math.sin(alpha) / lookahead
                        raw_steering = max(-0.5, min(0.5, turn_factor * 1.2))

                        # Slew rate limit — smooth curves
                        if not hasattr(self, '_charge_prev_steer'):
                            self._charge_prev_steer = 0.0
                        max_slew = 0.08
                        delta_s = raw_steering - self._charge_prev_steer
                        delta_s = max(-max_slew, min(max_slew, delta_s))
                        steering = self._charge_prev_steer + delta_s
                        self._charge_prev_steer = steering

                        # Throttle: full speed, ease off when steering hard
                        if distance < 15.0:
                            throttle = 0.3 + 0.5 * (distance / 15.0)
                        else:
                            throttle = 0.8 * (1.0 - abs(steering) * 0.3)

                    heading_error = alpha  # for debug log

                    # Debug log at 2Hz
                    if not hasattr(self, '_intercept_log_t'):
                        self._intercept_log_t = 0
                    if now - self._intercept_log_t > 0.5:
                        self._intercept_log_t = now
                        phase = "TURN" if abs(heading_error) > 2.6 else "ARC"
                        src = "CV"
                        print(f"[intercept] {state} {phase} | "
                              f"robot=({x_cm:.0f},{y_cm:.0f}) hdg={math.degrees(use_heading):.0f}°[{src}] | "
                              f"enemy=({enemy_pos[0]:.0f},{enemy_pos[1]:.0f}) | "
                              f"dist={distance:.0f}cm herr={math.degrees(heading_error):.0f}° | "
                              f"thr={throttle:.2f} str={steering:.2f}")
                elif not self._enemy_tracker.is_tracking:
                    # No valid track — drive toward last known position
                    # Don't stop immediately, coast forward for a bit
                    if self._enemy_tracker.kalman.frames_without_detection < 180:
                        # Coast toward last known enemy position
                        throttle = 0.4
                        steering = 0.0
                    else:
                        throttle = 0.0
                        steering = 0.0

                # Check: ArUco lost AND robot not moving → off visible area → reverse
                if not detected:
                    if not hasattr(self, '_charge_aruco_lost'):
                        self._charge_aruco_lost = 0
                    self._charge_aruco_lost += 1

                    # Robot position frozen = not moving (stuck at wall or off camera)
                    if self._charge_aruco_lost > 45:  # ~0.75s at 60fps
                        # Save enemy lock for reacquisition after reverse
                        self._saved_enemy_lock = self._enemy_tracker.detector._track_lock_px
                        self._reverse_start_time = now
                        self.mode = MODE_REVERSE
                        self._charge_aruco_lost = 0
                        print(f"[charge] ArUco lost & not moving — REVERSING to reacquire")
                else:
                    self._charge_aruco_lost = 0

            elif self.mode == MODE_PIN:
                # Pinning enemy — push forward for 5 seconds
                pin_elapsed = now - self._pin_start_time
                pin_remaining = 5.0 - pin_elapsed

                # Track how long ArUco has been lost during pin
                if not hasattr(self, '_pin_aruco_lost_frames'):
                    self._pin_aruco_lost_frames = 0
                if detected:
                    self._pin_aruco_lost_frames = 0
                else:
                    self._pin_aruco_lost_frames += 1

                if pin_remaining <= 0:
                    self._saved_enemy_lock = self._enemy_tracker.detector._track_lock_px
                    self._reverse_start_time = now
                    self.mode = MODE_REVERSE
                    self._pin_aruco_lost_frames = 0
                    print("[pin] 5s complete — REVERSING 2ft")
                elif self._pin_aruco_lost_frames > 30:
                    self._saved_enemy_lock = self._enemy_tracker.detector._track_lock_px
                    self._reverse_start_time = now
                    self.mode = MODE_REVERSE
                    self._pin_aruco_lost_frames = 0
                    print(f"[pin] ArUco lost — REVERSING to reacquire")
                else:
                    # Check if enemy is at wall yet
                    enemy_at_wall = False
                    if self._enemy_tracker.is_tracking:
                        epos = self._enemy_tracker.position_cm
                        enemy_at_wall = abs(epos[0]) > 80 or abs(epos[1]) > 80

                    if enemy_at_wall:
                        # At wall — soft hold
                        throttle = 0.2
                        steering = 0.0
                    else:
                        # Not at wall — FULL POWER push
                        throttle = 1.0
                        steering = 0.0

                    # Check if enemy escaped (moved away)
                    if self._enemy_tracker.is_tracking:
                        enemy_pos = self._enemy_tracker.position_cm
                        pin_dist = math.hypot(enemy_pos[0] - x_cm, enemy_pos[1] - y_cm) if detected else 0
                        if detected and pin_dist > 25:
                            # Enemy escaped — re-engage immediately
                            self.mode = MODE_INTERCEPT_CHARGE
                            self._pursuit_fsm._acquire_count = self._pursuit_fsm.ACQUIRE_FRAMES + 1
                            self._charge_prev_steer = 0.0
                            self._pin_aruco_lost_frames = 0
                            print(f"[pin] Enemy escaped! dist={pin_dist:.0f}cm — RE-ENGAGING")

                    if not hasattr(self, '_pin_log_t'):
                        self._pin_log_t = 0
                    if now - self._pin_log_t > 1.0:
                        self._pin_log_t = now
                        aruco_status = "OK" if detected else f"LOST({self._pin_aruco_lost_frames})"
                        print(f"[pin] {pin_remaining:.0f}s remaining... ArUco: {aruco_status}")

            elif self.mode == MODE_REVERSE:
                # Reverse 2 feet (~60cm) — drive backward for ~2 seconds
                reverse_elapsed = now - self._reverse_start_time

                if reverse_elapsed > 2.0:
                    # Done reversing — back to tracking
                    throttle = 0.0
                    steering = 0.0
                    self.comms.stop()
                    self.mode = MODE_INTERCEPT
                    self._pursuit_fsm.reset()
                    # Restore enemy track lock so it reacquires near last known position
                    if hasattr(self, '_saved_enemy_lock') and self._saved_enemy_lock is not None:
                        self._enemy_tracker.detector._track_lock_px = self._saved_enemy_lock
                        self._saved_enemy_lock = None
                    print("[reverse] Done — back to tracking")
                else:
                    # Drive backward
                    throttle = -0.5
                    steering = 0.0

            elif self.mode == MODE_BATTLE:
                # HSM combat state machine

                # Postmatch flourish — victory spin then stop
                if self._system_mode == SYSTEM_POSTMATCH and self._flourish_timer is not None:
                    flourish_elapsed = now - self._flourish_timer
                    if flourish_elapsed < 3.0:
                        throttle = 0.0
                        steering = 0.8
                    else:
                        self.mode = MODE_IDLE
                        self._system_mode = SYSTEM_CONFIG
                        self._flourish_timer = None
                        self.comms.stop()
                        print("[battle] Flourish complete — idle")
                # Normal battle mode
                else:
                    # Check for controller override
                    override = (abs(ctrl.throttle) > 0.25 or
                                abs(ctrl.steering) > 0.25 or
                                ctrl.buttons != 0)
                    if override:
                        print("[battle] Xbox override — switching to IDLE")
                        self.mode = MODE_IDLE
                        self._system_mode = SYSTEM_CONFIG
                        self._match_timer.reset()
                        self._battle_controller.reset()
                        self.comms.stop()
                        throttle = ctrl.throttle
                        steering = ctrl.steering
                        buttons = ctrl.buttons
                    elif self._match_timer.is_expired:
                        print("[battle] Match timer expired — victory flourish!")
                        self._system_mode = SYSTEM_POSTMATCH
                        self._flourish_timer = time.perf_counter()
                        self._battle_controller.reset()
                    else:
                        # Build context from current sensor data
                        enemy_pos = None
                        enemy_heading = self._enemy_tracker.heading_rad  # from orientation estimator
                        enemy_vel = None
                        e_detected = self._enemy_tracker.enemy_detected
                        e_tracking = self._enemy_tracker.is_tracking
                        e_frames_lost = self._enemy_tracker.kalman.frames_without_detection if hasattr(self._enemy_tracker, 'kalman') else 999
                        dist = 999.0

                        if e_tracking and self._enemy_tracker.position_cm is not None:
                            enemy_pos = self._enemy_tracker.position_cm
                            if detected:
                                dist = math.hypot(
                                    enemy_pos[0] - x_cm, enemy_pos[1] - y_cm
                                )
                            vel_arr = self._enemy_tracker.velocity_cm_s
                            if vel_arr is not None:
                                enemy_vel = (float(vel_arr[0]), float(vel_arr[1]))
                            else:
                                enemy_vel = (0.0, 0.0)

                        # Get IMU accelerometer data
                        accel_x, accel_y = 0.0, 0.0
                        if self._telemetry.is_active:
                            tel = self._telemetry.get()
                            accel_x = tel.get("accel_x", 0.0)
                            accel_y = tel.get("accel_y", 0.0)

                        ctx = BattleContext(
                            our_pos=(x_cm, y_cm),
                            our_heading_rad=heading_rad,
                            our_velocity=self._our_velocity,
                            enemy_pos=enemy_pos,
                            enemy_heading_rad=enemy_heading,
                            enemy_velocity=enemy_vel,
                            enemy_detected=e_detected,
                            enemy_tracking=e_tracking,
                            frames_without_detection=e_frames_lost,
                            distance_cm=dist,
                            dt=dt,
                            our_detected=detected,
                            accel_x_mg=accel_x,
                            accel_y_mg=accel_y,
                            throttle_cmd=throttle,
                        )
                        output = self._battle_controller.tick(ctx)
                        if output.target_omega_dps is not None:
                            # Rate mode: ESP32 holds angular velocity at 3.33kHz
                            self.comms.send_rate(
                                output.target_omega_dps,
                                output.target_speed,
                                output.buttons,
                            )
                            throttle = 0.0
                            steering = 0.0
                            buttons = output.buttons
                            self._rate_mode_active = True
                        else:
                            # Legacy direct mode
                            throttle = output.throttle
                            steering = output.steering
                            buttons = output.buttons
                            self._rate_mode_active = False

            elif self.mode == MODE_READY:
                # Standing by — show status, wait for battle start
                now_t = time.perf_counter()

                # Log enemy position every frame for debugging
                e_tracking = self._enemy_tracker.is_tracking if hasattr(self._enemy_tracker, 'is_tracking') else False
                if e_tracking and self._enemy_tracker.position_cm is not None:
                    ep = self._enemy_tracker.position_cm
                    if not hasattr(self, '_ready_enemy_log_t') or now_t - self._ready_enemy_log_t > 0.1:
                        self._ready_enemy_log_t = now_t
                        print(f"[ready] enemy=({ep[0]:+.1f},{ep[1]:+.1f})")

                if now_t - self._ready_log_t > 1.0:
                    self._ready_log_t = now_t
                    esp_ok = self.comms.connected and not self.comms._dry_run
                    imu_ok = self._telemetry.is_active
                    cam_ok = self._loop_fps > 30
                    aruco_ok = detected
                    print(f"[ready] STANDING BY — ESP:{'OK' if esp_ok else 'FAIL'}"
                          f" IMU:{'OK' if imu_ok else 'FAIL'}"
                          f" CAM:{'OK' if cam_ok else 'FAIL'}"
                          f" ArUco:{'OK' if aruco_ok else 'FAIL'}"
                          f" Enemy:{'TRACKED' if e_tracking else 'NO'}")

                # Stick override → manual
                if abs(ctrl.throttle) > 0.25 or abs(ctrl.steering) > 0.25:
                    print("[main] Stick override — switching to MANUAL")
                    self.mode = MODE_MANUAL
                    throttle = ctrl.throttle
                    steering = ctrl.steering

            elif self.mode == MODE_IDLE:
                # Check for controller override — require deliberate input
                override = (abs(ctrl.throttle) > 0.25 or
                            abs(ctrl.steering) > 0.25 or
                            ctrl.buttons != 0)
                if override:
                    print("[main] Xbox input detected — switching to MANUAL")
                    self.mode = MODE_MANUAL
                    throttle = ctrl.throttle
                    steering = ctrl.steering
                    buttons = ctrl.buttons

            # 6. Apply inversions for ESP32 (invertThrottle=-1, invertSteering=-1)
            # The ESP32 applies its own invertThrottle=-1 and invertSteering=-1
            # so we pre-negate to cancel that out, then the ESC gets the right sign
            if self.mode in (MODE_AUTO, MODE_INTERCEPT, MODE_INTERCEPT_CHARGE, MODE_PIN, MODE_REVERSE, MODE_BATTLE):
                steering = -steering

            if self.mode == MODE_AUTO:
                with self.shared_state["lock"]:
                    t_mix = self.shared_state.get("throttle_mix", 0.6)
                    s_mix = self.shared_state.get("steering_mix", 0.8)
                # Full throttle when driving straight — no mix scaling
                # Only scale steering to prevent wild swerves
                steering *= s_mix

                # Boost past ESC dead zone
                min_throttle = 0.25
                if 0 < abs(throttle) < min_throttle:
                    throttle = min_throttle if throttle > 0 else -min_throttle

                # Steering: only boost when heading error is large (turning)
                # When error is small, let the PID output stand (even if below dead zone)
                heading_err = abs(heading_rad - math.atan2(
                    self.follower._mission.waypoints[self.follower.current_waypoint_index].y - y_cm,
                    self.follower._mission.waypoints[self.follower.current_waypoint_index].x - x_cm
                )) if (self.follower.active and self.follower._mission and
                       self.follower.current_waypoint_index < len(self.follower._mission.waypoints)
                       ) else 0.0
                # Wrap heading error
                heading_err = abs(math.atan2(math.sin(heading_err), math.cos(heading_err)))

                if heading_err > 0.5:  # >30deg — boost steering for sure
                    min_steering = 0.30
                elif heading_err > 0.2:  # 10-30deg — moderate boost
                    min_steering = 0.20
                else:
                    min_steering = 0.0  # <10deg — let PID handle it naturally

                if min_steering > 0 and 0 < abs(steering) < min_steering:
                    steering = min_steering if steering > 0 else -min_steering

            # 6b. Heading-hold micro-corrections (IMU-based)
            # NOT in MODE_BATTLE — the state machine has its own steering controller
            if self.mode in (MODE_AUTO, MODE_INTERCEPT_CHARGE):
                steering = self._heading_hold_correction(throttle, steering, dt)

            # 6c. ESC dead zone boost for battle mode
            if self.mode == MODE_BATTLE:
                if 0 < abs(throttle) < 0.25:
                    throttle = 0.25 if throttle > 0 else -0.25
                if 0 < abs(steering) < 0.20:
                    steering = 0.20 if steering > 0 else -0.20

            # 7. Send motor command
            if self.mode in (MODE_AUTO, MODE_INTERCEPT_CHARGE, MODE_BATTLE) and (abs(throttle) > 0.01 or abs(steering) > 0.01):
                if not hasattr(self, '_cmd_log_t'):
                    self._cmd_log_t = 0
                if now - self._cmd_log_t > 1.0:
                    self._cmd_log_t = now
                    state_info = f" state={self._battle_controller.state}" if self.mode == MODE_BATTLE else ""
                    aruco_info = f" aruco={'Y' if detected else 'N'}"
                    accel_info = ""
                    if self.mode == MODE_BATTLE and self._telemetry.is_active:
                        tel = self._telemetry.get()
                        amag = math.hypot(tel.get("accel_x", 0), tel.get("accel_y", 0))
                        accel_info = f" accel={amag:.0f}mg"
                    print(f"[cmd] thr={throttle:.2f} str={steering:.2f} pos=({x_cm:.0f},{y_cm:.0f}){aruco_info}{accel_info}{state_info}")
            if not getattr(self, '_rate_mode_active', False):
                self.comms.send(throttle, steering, buttons)

            # 7b. Frame logging (READY + BATTLE modes)
            if self.mode in (MODE_READY, MODE_BATTLE):
                if self._frame_log_file is None:
                    log_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "logs",
                        f"frames_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
                    )
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    self._frame_log_file = open(log_path, "w")
                    print(f"[log] Frame log: {log_path}")

                e_pos = self._enemy_tracker.position_cm if self._enemy_tracker.is_tracking else None
                has_enemy = e_pos is not None and self._enemy_tracker.is_tracking
                rec = {
                    "f": self._frame_count,
                    "t": round(now, 4),
                    "mode": self.mode,
                    "bs": self._battle_controller.state if self.mode == MODE_BATTLE else "ready",
                    "ox": round(x_cm, 1), "oy": round(y_cm, 1),
                    "oh": round(heading_rad, 3),
                    "od": detected,
                    "ex": round(float(e_pos[0]), 1) if has_enemy else None,
                    "ey": round(float(e_pos[1]), 1) if has_enemy else None,
                    "ed": self._enemy_tracker.enemy_detected,
                    "et": self._enemy_tracker.is_tracking,
                    "dist": round(math.hypot(float(e_pos[0]) - x_cm, float(e_pos[1]) - y_cm), 1) if has_enemy and detected else 999.0,
                    "thr": round(throttle, 3), "str": round(steering, 3),
                }
                self._frame_log_file.write(json.dumps(rec, separators=(",", ":")) + "\n")
                self._frame_count += 1
            elif self._frame_log_file is not None:
                self._frame_log_file.close()
                print(f"[log] Frame log closed ({self._frame_count} frames)")
                self._frame_log_file = None
                self._frame_count = 0

            # 8. Update dashboard state
            self._update_shared_state(
                x_cm, y_cm, heading_rad, detected, throttle, steering
            )

            # 9. Encode for dashboard stream at 30fps (control loop runs at full speed)
            if frame is not None:
                with self.shared_state["lock"]:
                    self.shared_state["frame_w"] = frame.shape[1]
                    self.shared_state["frame_h"] = frame.shape[0]

                if now - self._last_stream_time >= self._stream_interval:
                    self._last_stream_time = now

                    # Draw overlays only on stream frames (expensive)
                    stream_frame = frame.copy()

                    with self.shared_state["lock"]:
                        show_grid = self.shared_state.get("show_grid", True)
                    if show_grid:
                        # Use floor plane grid if calibrated, else old grid
                        if hasattr(self, '_floor_det') and self._floor_det and self._floor_det.calibrated:
                            self._floor_det.draw_grid(stream_frame)
                        else:
                            self._draw_floor_grid(stream_frame)

                    wp_px = self._get_waypoint_pixels()
                    draw_overlay(stream_frame, pose, waypoints=wp_px)

                    mode_color = {
                        MODE_IDLE: (200, 200, 200),
                        MODE_AUTO: (0, 255, 0),
                        MODE_MANUAL: (0, 200, 255),
                        MODE_CALIBRATING: (0, 255, 255),
                        MODE_INTERCEPT: (0, 165, 255),
                        MODE_INTERCEPT_CHARGE: (0, 0, 255),
                        MODE_PIN: (0, 255, 255),
                        MODE_REVERSE: (255, 0, 255),
                        MODE_BATTLE: (0, 255, 100),
                    }.get(self.mode, (200, 200, 200))
                    cv2.putText(
                        stream_frame, f"Mode: {self.mode.upper()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2,
                    )
                    cv2.putText(
                        stream_frame,
                        f"FPS: {self.camera.fps:.0f}  Thr: {throttle:+.2f}  Str: {steering:+.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    )

                    if self._measure_line is not None:
                        p1, p2, dist, label = self._measure_line
                        cv2.line(stream_frame, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.circle(stream_frame, p1, 6, (0, 255, 255), -1)
                        cv2.circle(stream_frame, p2, 6, (0, 255, 255), -1)
                        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                        cv2.putText(stream_frame, label, (mid[0] + 8, mid[1] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                                    cv2.LINE_AA)

                    if self.tracker.is_calibrating:
                        for px, py in self.tracker._calib_points_px:
                            px_i, py_i = int(px), int(py)
                            if 0 <= px_i < stream_frame.shape[1] and 0 <= py_i < stream_frame.shape[0]:
                                cv2.circle(stream_frame, (px_i, py_i), 4, (255, 255, 0), -1, cv2.LINE_AA)

                    _, jpeg = cv2.imencode('.jpg', stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    with self._jpeg_lock:
                        self._latest_jpeg = jpeg.tobytes()

                    # Draw enemy detection debug mask (small thumbnail, top-right)
                    fg = self._enemy_tracker.detector.fg_mask
                    if fg is not None:
                        thumb_h = 120
                        thumb_w = int(fg.shape[1] * thumb_h / fg.shape[0])
                        thumb = cv2.resize(fg, (thumb_w, thumb_h))
                        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
                        x_off = stream_frame.shape[1] - thumb_w - 10
                        stream_frame[10:10+thumb_h, x_off:x_off+thumb_w] = thumb_bgr
                        cv2.putText(stream_frame, "FG MASK", (x_off, 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

                    # Draw enemy tracking overlay
                    if self._enemy_tracker.is_tracking or self._enemy_tracker.enemy_detected:
                        self._enemy_tracker.draw_overlay(
                            stream_frame,
                            cm_to_px=self.tracker.cm_to_px,
                        )
                        if self.mode == MODE_BATTLE:
                            # Show HSM state instead of old PursuitFSM
                            bstate = self._battle_controller.state
                            state_colors = {
                                "scan": (150, 150, 150), "acquire": (0, 200, 255),
                                "charge_pursue": (0, 100, 255), "charge_flank": (0, 150, 255),
                                "charge_ram": (0, 0, 255), "charge_pin": (0, 255, 255),
                                "pit_position": (0, 200, 100), "pit_push": (0, 255, 0),
                                "pit_commit": (0, 255, 50), "pit_abort": (0, 200, 200),
                                "evade_retreat": (255, 0, 200), "evade_reposition": (200, 100, 255),
                                "unstick": (0, 165, 255), "lost_target": (100, 100, 100),
                            }
                            sc = state_colors.get(bstate, (200, 200, 200))
                            cv2.putText(stream_frame, bstate.upper().replace("_", " "),
                                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)
                            # Match timer
                            rem = self._match_timer.remaining_s
                            mins = int(rem) // 60
                            secs = int(rem) % 60
                            timer_text = f"Match: {mins}:{secs:02d}"
                            urg = self._match_timer.urgency
                            urg_color = (0, 255, 255) if urg < 0.5 else (0, 165, 255) if urg < 0.8 else (0, 0, 255)
                            cv2.putText(stream_frame, timer_text, (10, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, urg_color, 1)
                            cv2.putText(stream_frame, f"Urgency: {urg:.0%}", (180, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, urg_color, 1)
                        else:
                            # Old pursuit FSM state for legacy modes
                            state_text = f"Pursuit: {self._pursuit_fsm.state.upper()}"
                            cv2.putText(stream_frame, state_text, (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

                    # PIN countdown overlay (both old MODE_PIN and new battle charge_pin)
                    if self.mode == MODE_PIN and hasattr(self, '_pin_start_time'):
                        remaining = max(0, 5.0 - (time.perf_counter() - self._pin_start_time))
                        h_f, w_f = stream_frame.shape[:2]
                        countdown_text = f"{remaining:.1f}"
                        text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 4, 8)[0]
                        tx = (w_f - text_size[0]) // 2
                        ty = (h_f + text_size[1]) // 2
                        cv2.putText(stream_frame, countdown_text, (tx+3, ty+3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)
                        cv2.putText(stream_frame, countdown_text, (tx, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
                        cv2.putText(stream_frame, "PINNING", (tx, ty - 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    elif self.mode == MODE_BATTLE and self._battle_controller.state == "charge_pin":
                        remaining = max(0, self._pin_timer.remaining_s)
                        h_f, w_f = stream_frame.shape[:2]
                        countdown_text = f"{remaining:.1f}"
                        text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 4, 8)[0]
                        tx = (w_f - text_size[0]) // 2
                        ty = (h_f + text_size[1]) // 2
                        cv2.putText(stream_frame, countdown_text, (tx+3, ty+3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)
                        cv2.putText(stream_frame, countdown_text, (tx, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 100), 8)
                        cv2.putText(stream_frame, "PINNING", (tx, ty - 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 100), 3)

                    elif self.mode == MODE_REVERSE:
                        h_f, w_f = stream_frame.shape[:2]
                        cv2.putText(stream_frame, "REVERSING", (w_f//2 - 150, h_f//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)

                    # Pit calibration in-progress overlay
                    if self._pit_calibrating:
                        h_f, w_f = stream_frame.shape[:2]
                        if self._pit_corner1_px is None:
                            cv2.putText(stream_frame, "PIT CAL: Click corner 1", (10, h_f - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            cv2.circle(stream_frame, self._pit_corner1_px, 8, (0, 0, 255), -1)
                            cv2.putText(stream_frame, "PIT CAL: Click opposite corner", (10, h_f - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Draw pit overlay if configured
                    if (self._battle_config.pit_x_cm != 0 or self._battle_config.pit_y_cm != 0):
                        try:
                            pit_half = self._battle_config.pit_radius_cm
                            pit_cx, pit_cy = self._battle_config.pit_x_cm, self._battle_config.pit_y_cm
                            # Draw pit rectangle using corners
                            corners_cm = [
                                (pit_cx - pit_half, pit_cy - pit_half),
                                (pit_cx + pit_half, pit_cy - pit_half),
                                (pit_cx + pit_half, pit_cy + pit_half),
                                (pit_cx - pit_half, pit_cy + pit_half),
                            ]
                            corners_px = []
                            for cx_cm, cy_cm in corners_cm:
                                px, py = self.tracker.cm_to_px(cx_cm, cy_cm)
                                corners_px.append((int(px), int(py)))
                            pts = np.array(corners_px, dtype=np.int32)
                            cv2.polylines(stream_frame, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(stream_frame, "PIT", (corners_px[0][0], corners_px[0][1] - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        except (ValueError, cv2.error):
                            pass  # calibration not loaded yet

                    # Draw IMU fusion status
                    if self._telemetry.is_active:
                        tel = self._telemetry.get()
                        imu_text = f"IMU: {tel['heading']:.1f}deg gyro:{tel['gyro_z']:.1f}dps"
                        cv2.putText(stream_frame, imu_text, (10, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

                    if self.args.show_cv:
                        cv2.imshow("Auto-Drive", stream_frame)

                        # Register click callback (once)
                        if not hasattr(self, '_cv_callback_set'):
                            cv2.setMouseCallback("Auto-Drive", self._on_cv_click)
                            self._cv_callback_set = True

                        # waitKey pumps OpenCV HighGUI — needed for imshow/mouse
                        # Key handling is done via KeyboardPoller above (GetAsyncKeyState)
                        # Only calibration keys (p, a, r, i, t) remain here
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("p"):
                            self._handle_calibration({"action": "floor_plane"})
                        elif key == ord("a"):
                            # Auto-detect arena walls
                            if frame is not None:
                                from floor_plane import auto_detect_arena, FloorPlaneDetector
                                points = auto_detect_arena(frame, num_points=8)
                                if points:
                                    detector = FloorPlaneDetector()
                                    rgb_size = (frame.shape[1], frame.shape[0])
                                    if detector.calibrate_from_corners(points, rgb_size=rgb_size):
                                        floor_path = os.path.join(
                                            os.path.dirname(os.path.abspath(__file__)),
                                            "floor_calibration.json"
                                        )
                                        self.tracker.load_floor_plane(floor_path)
                                        self._floor_det = detector
                                        self._filtered_heading = None
                                        self._filtered_x = None
                                        self._filtered_y = None
                                        # Update enemy detector arena mask
                                        with open(floor_path) as f:
                                            corners_px = json.load(f).get("corners_px")
                                        if corners_px:
                                            self._enemy_tracker.detector.set_arena_corners(corners_px)
                                        print("[main] Auto-detected arena walls — calibration applied")
                                    else:
                                        print("[main] Auto-detect found walls but calibration failed")
                                else:
                                    print("[main] Auto-detect failed — try manual calibration (p)")
                        elif key == ord("r"):
                            # Capture empty arena reference frame + save to disk
                            if frame is not None:
                                self._enemy_tracker.detector.capture_reference(frame)
                                self._enemy_tracker.reset()
                                ref_path = os.path.join(
                                    os.path.dirname(os.path.abspath(__file__)),
                                    "arena_reference.png"
                                )
                                ref_gray = self._enemy_tracker.detector._reference_gray
                                if ref_gray is not None:
                                    cv2.imwrite(ref_path, ref_gray)
                                    print(f"[main] Reference frame saved to {ref_path}")
                                print("[main] Reference frame captured — static enemies now detectable")
                        elif key == ord("i"):
                            # Toggle intercept mode
                            if self.mode in (MODE_INTERCEPT, MODE_INTERCEPT_CHARGE, MODE_PIN, MODE_REVERSE):
                                self.mode = MODE_IDLE
                                self._pursuit_fsm.reset()
                                self.comms.stop()
                                print("[main] Intercept mode OFF")
                            else:
                                self.mode = MODE_INTERCEPT
                                self._pursuit_fsm.reset()
                                self._enemy_tracker.reset()
                                print("[main] Intercept TRACKING — press SPACE to charge")
                        elif key == ord("t"):
                            # Pit calibration — mark 2 opposite corners
                            if self._pit_calibrating:
                                self._pit_calibrating = False
                                self._pit_corner1_px = None
                                self._pit_corner1_cm = None
                                print("[pit-cal] Pit calibration cancelled")
                            else:
                                self._pit_calibrating = True
                                self._pit_corner1_px = None
                                self._pit_corner1_cm = None
                                print("[pit-cal] Click two OPPOSITE corners of the pit on the camera view")
                        elif key == ord(" "):
                            # SPACE — trigger charge (battle/ready start handled by KeyboardPoller)
                            if self.mode == MODE_INTERCEPT and self._enemy_tracker.is_tracking:
                                self.mode = MODE_INTERCEPT_CHARGE
                                self._pursuit_fsm.reset()
                                # Skip ACQUIRE — enemy already tracked, go straight to INTERCEPT
                                self._pursuit_fsm._acquire_count = self._pursuit_fsm.ACQUIRE_FRAMES + 1
                                self._charge_prev_steer = 0.0
                                print("[main] CHARGE! Pursuing enemy")
                            elif self.mode == MODE_INTERCEPT_CHARGE:
                                # SPACE again — stop charging, go back to tracking
                                self.mode = MODE_INTERCEPT
                                self.comms.stop()
                                print("[main] Charge stopped — back to tracking")

            # 9. FPS tracking
            self._fps_count += 1
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._loop_fps = self._fps_count / elapsed
                self._fps_count = 0
                self._fps_timer = now

            # No sleep — run as fast as camera provides frames
            # The camera read() blocks naturally when no new frame is available

    def _get_jpeg(self):
        """Return latest JPEG-encoded frame for MJPEG stream."""
        with self._jpeg_lock:
            return self._latest_jpeg

    def _process_dashboard_commands(self):
        """Check for and process pending dashboard commands."""
        lock = self.shared_state["lock"]
        with lock:
            cmd = self.shared_state.get("pending_command")
            if cmd is None:
                return
            self.shared_state["pending_command"] = None

        cmd_type = cmd.get("type")

        if cmd_type == "mission":
            name = cmd["name"]
            params = cmd.get("params", {})
            # Convert param values to float
            params = {k: float(v) for k, v in params.items()}
            gen = MISSION_GENERATORS.get(name)
            if gen:
                mission = gen(params)
                self.follower.start_mission(mission)
                self.mode = MODE_AUTO
                print(f"[main] Starting mission: {name} {params}")
                for i, wp in enumerate(mission.waypoints):
                    print(f"  waypoint {i}: ({wp.x:.1f}, {wp.y:.1f}) heading={wp.heading}")
            else:
                print(f"[main] Unknown mission: {name}")

        elif cmd_type == "set_mode":
            new_mode = cmd["mode"]
            print(f"[main] Dashboard set mode: {new_mode}")
            self.mode = new_mode
            if new_mode == MODE_IDLE:
                self.follower = PathFollower()
                self._match_timer.reset()
                self._battle_controller.reset()
                self.comms.stop()

        elif cmd_type == "emergency_stop":
            print("[main] EMERGENCY STOP")
            self.mode = MODE_IDLE
            self.follower = PathFollower()
            self.comms.stop()

        elif cmd_type == "measure":
            x1_px, y1_px = cmd["x1_px"], cmd["y1_px"]
            x2_px, y2_px = cmd["x2_px"], cmd["y2_px"]
            try:
                x1_cm, y1_cm = self.tracker.px_to_cm(x1_px, y1_px)
                x2_cm, y2_cm = self.tracker.px_to_cm(x2_px, y2_px)
                dist_cm = math.hypot(x2_cm - x1_cm, y2_cm - y1_cm)
                label = f"{dist_cm:.1f} cm"
                self._measure_line = (
                    (int(x1_px), int(y1_px)),
                    (int(x2_px), int(y2_px)),
                    dist_cm,
                    label,
                )
                # Store result in shared state so dashboard can read it
                with self.shared_state["lock"]:
                    self.shared_state["measure_result"] = {
                        "dist_cm": round(dist_cm, 1),
                        "p1_cm": (round(x1_cm, 1), round(y1_cm, 1)),
                        "p2_cm": (round(x2_cm, 1), round(y2_cm, 1)),
                    }
                print(f"[measure] ({x1_cm:.1f},{y1_cm:.1f}) -> ({x2_cm:.1f},{y2_cm:.1f}) = {dist_cm:.1f} cm")
            except ValueError as e:
                print(f"[measure] Failed: {e}")

        elif cmd_type == "click_goto":
            # Convert fractional click to pixel, then to world cm via tracker
            frame = self.camera.read()
            if frame is not None:
                fh, fw = frame.shape[:2]
                px_x = cmd["x_frac"] * fw
                px_y = cmd["y_frac"] * fh
                try:
                    x_cm, y_cm = self.tracker.px_to_cm(px_x, px_y)
                    mission = generate_goto(x_cm, y_cm)
                    self.follower.start_mission(mission)
                    self.mode = MODE_AUTO
                    print(f"[main] Click goto: pixel=({px_x:.0f},{px_y:.0f}) -> ({x_cm:.1f},{y_cm:.1f})cm")
                except ValueError as e:
                    print(f"[main] Click goto failed: {e}")

        elif cmd_type == "start_battle":
            print("[main] Starting battle!")
            self._system_mode = SYSTEM_BATTLE
            self._battle_controller.reset()
            self._match_timer.reset()
            self._match_timer.start()
            self._pin_timer.reset()
            self._flourish_timer = None
            self._enemy_tracker.reset()
            self.mode = MODE_BATTLE

        elif cmd_type == "stop_battle":
            print("[main] Stopping battle")
            self._system_mode = SYSTEM_CONFIG
            self._match_timer.reset()
            self._battle_controller.reset()
            self._flourish_timer = None
            self.mode = MODE_IDLE
            self.comms.stop()

        elif cmd_type == "battle_config":
            config_data = cmd.get("config", {})
            self._battle_config.update(**config_data)
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "battle_config.json"
            )
            self._battle_config.save(config_path)
            # Reinit timers if durations changed
            self._match_timer.duration_s = self._battle_config.match_duration_s
            self._pin_timer.duration_s = self._battle_config.pin_duration_s
            print(f"[main] Battle config updated: {config_data}")

        elif cmd_type == "calibrate":
            self._handle_calibration(cmd)

    def _handle_calibration(self, cmd):
        """Process calibration commands."""
        action = cmd.get("action")
        calib_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "homography.json"
        )
        floor_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "floor_calibration.json"
        )

        if action == "floor_plane":
            # Single-marker floor plane calibration (interactive)
            from floor_plane import run_single_marker_calibration, run_corner_calibration
            frame = self.camera.read()
            if frame is None:
                print("[calib] No camera frame available")
                return
            # Get camera intrinsics
            if hasattr(self.camera, 'get_intrinsics'):
                camera_matrix, dist_coeffs = self.camera.get_intrinsics()
            elif hasattr(self.camera, 'intrinsics') and self.camera.intrinsics:
                intr = self.camera.intrinsics
                camera_matrix = np.array([
                    [intr['fx'], 0, intr['cx']],
                    [0, intr['fy'], intr['cy']],
                    [0, 0, 1],
                ], dtype=np.float64)
                dist_coeffs = np.zeros(5, dtype=np.float64)
            else:
                h, w = frame.shape[:2]
                fx = w * 0.6
                camera_matrix = np.array([
                    [fx, 0, w/2], [0, fx, h/2], [0, 0, 1]
                ], dtype=np.float64)
                dist_coeffs = np.zeros(5, dtype=np.float64)

            result = run_single_marker_calibration(
                self.camera, camera_matrix, dist_coeffs
            )
            if result is None:
                print("[calib] Falling back to manual click...")
                result = run_corner_calibration(frame, rgb_size=(frame.shape[1], frame.shape[0]))

            if result is not None:
                # Load the saved floor plane into the tracker's homography
                self.tracker.load_floor_plane(floor_path)
                self._floor_det = result  # for grid drawing
                self._filtered_heading = None
                self._filtered_x = None
                self._filtered_y = None
                print("[calib] Floor plane calibration applied")
            else:
                print("[calib] Floor plane calibration cancelled")

        elif action == "drive_start":
            self.tracker.start_calibration_drive()
            self.mode = MODE_CALIBRATING
            print("[main] Calibration drive started — use Xbox controller to drive")

        elif action == "drive_finish":
            success = self.tracker.finish_calibration_drive()
            self.mode = MODE_IDLE
            if success:
                print("[main] Calibration drive complete — homography computed")
                self._filtered_heading = None
                self._filtered_x = None
                self._filtered_y = None
            else:
                print("[main] Calibration drive failed — not enough points")

        elif action == "charuco":
            frame = self.camera.read()
            if frame is not None:
                success = self.tracker.calibrate_from_charuco(frame)
                if success:
                    self._filtered_heading = None
                    self._filtered_x = None
                    self._filtered_y = None
                    print("[calib] ChArUco floor calibration successful")
                else:
                    print("[calib] ChArUco calibration failed — is the board visible?")
            else:
                print("[calib] No camera frame available")

        elif action == "auto":
            # Re-run auto-calibration
            frame = self.camera.read()
            if frame is not None:
                success = self.tracker.auto_calibrate(frame, marker_id=self.args.marker_id)
                if success:
                    self._filtered_heading = None
                    self._filtered_x = None
                    self._filtered_y = None
                    print("[calib] Auto-calibration successful")
                else:
                    print("[calib] Auto-calibration failed — marker not visible")

        elif action == "capture":
            # Capture current marker pixel position as a calibration point
            frame = self.camera.read()
            if frame is not None:
                pose = self.tracker.get_robot_pose(frame, marker_id=self.args.marker_id)
                if pose is not None:
                    x_cm = float(cmd.get("x_cm", 0.0))
                    y_cm = float(cmd.get("y_cm", 0.0))
                    count = self.tracker.add_calibration_point(
                        pose.x_px, pose.y_px, x_cm, y_cm
                    )
                    print(f"[calib] Point {count}: pixel=({pose.x_px:.0f},{pose.y_px:.0f}) -> world=({x_cm},{y_cm})cm")
                else:
                    print("[calib] No marker detected — place marker and try again")
            else:
                print("[calib] No camera frame available")

        elif action == "compute":
            success = self.tracker.compute_homography()
            if success:
                print("[calib] Homography computed successfully")
                # Reset EMA filters since coordinate system changed
                self._filtered_heading = None
                self._filtered_x = None
                self._filtered_y = None

        elif action == "clear":
            self.tracker.clear_calibration_points()
            self.tracker._homography = None
            self.tracker._homography_inv = None
            print("[calib] Calibration cleared")

        elif action == "save":
            self.tracker.save_homography(calib_path)

        elif action == "load":
            if self.tracker.load_homography(calib_path):
                self._filtered_heading = None
                self._filtered_x = None
                self._filtered_y = None

    def _on_cv_click(self, event, x, y, flags, param):
        """Mouse callback for click-to-point or pit calibration on the CV window."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Pit calibration mode — 2-click workflow
        if self._pit_calibrating:
            try:
                x_cm, y_cm = self.tracker.px_to_cm(x, y)
            except (ValueError, cv2.error) as e:
                print(f"[pit-cal] Failed to convert pixel ({x},{y}): {e}")
                return

            if self._pit_corner1_px is None:
                # First corner
                self._pit_corner1_px = (x, y)
                self._pit_corner1_cm = (x_cm, y_cm)
                print(f"[pit-cal] Corner 1: pixel=({x},{y}) -> ({x_cm:.1f},{y_cm:.1f})cm")
                print("[pit-cal] Now click the OPPOSITE corner of the pit")
            else:
                # Second corner — compute pit bounds
                c1 = self._pit_corner1_cm
                c2 = (x_cm, y_cm)
                pit_min_x = min(c1[0], c2[0])
                pit_max_x = max(c1[0], c2[0])
                pit_min_y = min(c1[1], c2[1])
                pit_max_y = max(c1[1], c2[1])
                pit_cx = (pit_min_x + pit_max_x) / 2
                pit_cy = (pit_min_y + pit_max_y) / 2
                pit_w = pit_max_x - pit_min_x
                pit_h = pit_max_y - pit_min_y
                pit_radius = max(pit_w, pit_h) / 2

                print(f"[pit-cal] Corner 2: pixel=({x},{y}) -> ({x_cm:.1f},{y_cm:.1f})cm")
                print(f"[pit-cal] Pit: center=({pit_cx:.1f},{pit_cy:.1f}) size={pit_w:.1f}x{pit_h:.1f}cm")

                # Update battle config
                self._battle_config.pit_x_cm = pit_cx
                self._battle_config.pit_y_cm = pit_cy
                self._battle_config.pit_radius_cm = pit_radius
                self._battle_config.pit_danger_radius_cm = pit_radius + 15.0

                # Save to disk
                config_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "battle_config.json"
                )
                self._battle_config.save(config_path)
                print(f"[pit-cal] Saved to {config_path}")

                # Done
                self._pit_calibrating = False
                self._pit_corner1_px = None
            return

        # Normal click-to-point
        try:
            x_cm, y_cm = self.tracker.px_to_cm(x, y)
        except (ValueError, cv2.error) as e:
            print(f"[click] Failed to convert pixel ({x},{y}): {e}")
            return

        print(f"[click] Target: pixel=({x},{y}) -> ({x_cm:.1f},{y_cm:.1f})cm")
        mission = generate_goto(x_cm, y_cm)
        self.follower.start_mission(mission)
        self.mode = MODE_AUTO

    def _update_shared_state(self, x_cm, y_cm, heading_rad, detected, throttle, steering):
        """Push current state to dashboard."""
        lock = self.shared_state["lock"]
        with lock:
            self.shared_state["mode"] = self.mode
            self.shared_state["x_cm"] = x_cm
            self.shared_state["y_cm"] = y_cm
            self.shared_state["heading_rad"] = heading_rad
            self.shared_state["detected"] = detected
            self.shared_state["fps"] = self.camera.fps if self.camera else 0.0
            self.shared_state["trail"] = list(self.trail)
            self.shared_state["px_per_cm"] = self.tracker._px_per_cm or 5.0
            self.shared_state["origin_x"] = self.tracker._origin_x
            self.shared_state["origin_y"] = self.tracker._origin_y

            # Battle state machine fields
            self.shared_state["system_mode"] = self._system_mode
            if self.mode == MODE_BATTLE:
                self.shared_state["battle_state"] = self._battle_controller.state
                self.shared_state["match_remaining_s"] = round(self._match_timer.remaining_s, 1)
                self.shared_state["pin_remaining_s"] = (
                    round(self._pin_timer.remaining_s, 1) if self._pin_timer.is_running else None
                )
                self.shared_state["urgency"] = round(self._match_timer.urgency, 2)
            else:
                self.shared_state["battle_state"] = None
                self.shared_state["match_remaining_s"] = None
                self.shared_state["pin_remaining_s"] = None
                self.shared_state["urgency"] = None

            if self.follower.active:
                self.shared_state["mission_progress"] = self.follower.mission_progress
                self.shared_state["mission_name"] = (
                    self.follower._mission.name if self.follower._mission else ""
                )
                # Build waypoint status list for dashboard
                wps = []
                if self.follower._mission:
                    for i, wp in enumerate(self.follower._mission.waypoints):
                        if i < self.follower.current_waypoint_index:
                            status = "reached"
                        elif i == self.follower.current_waypoint_index:
                            status = "current"
                        else:
                            status = "pending"
                        wps.append({"x": wp.x, "y": wp.y, "status": status})
                self.shared_state["waypoints"] = wps
            else:
                if self.mode != MODE_AUTO:
                    self.shared_state["mission_name"] = ""
                    self.shared_state["mission_progress"] = 0.0
                    self.shared_state["waypoints"] = []

    def _get_waypoint_pixels(self):
        """Convert current mission waypoints to pixel coordinates for CV overlay."""
        if not self.follower.active or not self.follower._mission:
            return None
        result = []
        for wp in self.follower._mission.waypoints:
            try:
                px, py = self.tracker.cm_to_px(wp.x, wp.y)
                result.append((px, py))
            except ValueError:
                pass
        return result if result else None

    def _draw_floor_grid(self, frame):
        """Draw a CRT-green perspective floor grid on the camera frame."""
        h, w = frame.shape[:2]
        grid_cm = 30  # 30cm grid spacing
        grid_range = 300  # -300cm to +300cm
        step_cm = 10  # sample every 10cm along each line for smooth curves

        # CRT green palette
        grid_color = (0, 100, 0)       # dark green grid lines
        origin_color = (0, 200, 0)     # brighter green for axes
        label_color = (0, 180, 0)      # green labels

        def _world_polyline(world_pts):
            """Convert world points to pixel polyline, clipping to frame bounds."""
            px_pts = []
            for xc, yc in world_pts:
                try:
                    px, py = self.tracker.cm_to_px(xc, yc)
                    px_i, py_i = int(px), int(py)
                    # Keep points within extended frame bounds
                    if -500 < px_i < w + 500 and -500 < py_i < h + 500:
                        px_pts.append((px_i, py_i))
                    else:
                        # Break the polyline if point goes way off
                        if px_pts:
                            px_pts.append(None)  # sentinel
                except (ValueError, cv2.error):
                    pass
            return px_pts

        def _draw_polyline(frame, px_pts, color, thickness):
            """Draw a polyline that may have breaks (None sentinels)."""
            for i in range(len(px_pts) - 1):
                if px_pts[i] is not None and px_pts[i + 1] is not None:
                    cv2.line(frame, px_pts[i], px_pts[i + 1], color, thickness,
                             cv2.LINE_AA)

        y_values = list(range(-grid_range, grid_range + 1, step_cm))
        x_values = list(range(-grid_range, grid_range + 1, step_cm))

        # Vertical lines (constant x, varying y)
        for x_cm in range(-grid_range, grid_range + 1, grid_cm):
            pts = [(x_cm, y) for y in y_values]
            px_pts = _world_polyline(pts)
            if len(px_pts) >= 2:
                is_axis = (x_cm == 0)
                _draw_polyline(frame, px_pts, origin_color if is_axis else grid_color,
                               2 if is_axis else 1)
                # Label
                for pt in reversed(px_pts):
                    if pt is not None and 0 <= pt[0] < w and 0 <= pt[1] < h:
                        cv2.putText(frame, f"{x_cm}",
                                    (pt[0] + 4, pt[1] - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1,
                                    cv2.LINE_AA)
                        break

        # Horizontal lines (constant y, varying x)
        for y_cm in range(-grid_range, grid_range + 1, grid_cm):
            pts = [(x, y_cm) for x in x_values]
            px_pts = _world_polyline(pts)
            if len(px_pts) >= 2:
                is_axis = (y_cm == 0)
                _draw_polyline(frame, px_pts, origin_color if is_axis else grid_color,
                               2 if is_axis else 1)
                for pt in px_pts:
                    if pt is not None and 0 <= pt[0] < w and 0 <= pt[1] < h:
                        cv2.putText(frame, f"{y_cm}",
                                    (pt[0] + 4, pt[1] - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1,
                                    cv2.LINE_AA)
                        break

        # Origin crosshair
        try:
            ox_px, oy_px = self.tracker.cm_to_px(0.0, 0.0)
            ox_i, oy_i = int(ox_px), int(oy_px)
            if 0 <= ox_i < w and 0 <= oy_i < h:
                cv2.drawMarker(frame, (ox_i, oy_i), (0, 255, 0),
                               cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
                cv2.putText(frame, "ORIGIN", (ox_i + 12, oy_i - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
                            cv2.LINE_AA)
        except (ValueError, cv2.error):
            pass

        # Draw calibration sample points only during active calibration
        if self.tracker.is_calibrating:
            for px, py in self.tracker._calib_points_px:
                px_i, py_i = int(px), int(py)
                if 0 <= px_i < w and 0 <= py_i < h:
                    cv2.circle(frame, (px_i, py_i), 4, (255, 255, 0), -1, cv2.LINE_AA)

    def _shutdown(self):
        """Clean up all resources."""
        self.running = False
        print("[main] Stopping motors ...")
        self.comms.close()
        print("[main] Stopping telemetry ...")
        self._telemetry.stop()
        if self._imu_poller:
            self._imu_poller.stop()
        print("[main] Stopping camera ...")
        if self.camera:
            self.camera.stop()
        print("[main] Closing controller ...")
        self.controller.close()
        if self._voice:
            self._voice.shutdown()
        if self.args.show_cv:
            cv2.destroyAllWindows()
        print("[main] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-Drive: CV-guided autonomous robot control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--esp32",
        default="",
        help="ESP32 hostname or IP (default: dry-run mode)",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=4210,
        help="ESP32 UDP port (default: 4210)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Camera index (default: 1 = external webcam)",
    )
    parser.add_argument(
        "--oakd",
        action="store_true",
        help="Use OAK-D Pro camera via DepthAI",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Use mono camera (OV9282 global shutter, up to 120fps)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Target camera FPS (default: 60, mono supports up to 130 at 800p)",
    )
    parser.add_argument(
        "--marker-id",
        type=int,
        default=0,
        help="ArUco marker ID for our robot (default: 0)",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=50.0,
        help="ArUco marker physical size in mm (default: 50)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Dashboard HTTP port (default: 5000)",
    )
    parser.add_argument(
        "--show-cv",
        action="store_true",
        help="Show OpenCV debug window with tracking overlay",
    )
    parser.add_argument(
        "--px-per-cm",
        type=float,
        default=5.0,
        help="Pixels per cm calibration factor (default: 5.0 for 720p at ~2m)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = AutoDriveApp(args)
    app.start()
