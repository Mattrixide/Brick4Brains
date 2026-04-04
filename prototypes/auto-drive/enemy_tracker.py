"""Enemy robot detection and tracking without ArUco markers.

Uses MOG2 background subtraction + contour filtering + Kalman filter
to detect, track, and estimate velocity of the enemy robot.
Our robot is excluded using its known ArUco bounding box.
"""

import math
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Enemy Kalman filter (constant-velocity, position+velocity state)
# ---------------------------------------------------------------------------

class EnemyKalmanFilter:
    """4-state Kalman filter: [x, y, vx, vy] with position-only measurements.

    Process noise tuned for adversarial combat robots (sigma_a = 5 m/s²).
    Handles detection dropouts via predict-only coasting.
    """

    def __init__(self, dt=1/60, sigma_a=5.0, sigma_meas=0.01):
        self.dt = dt
        self.x = np.zeros(4, dtype=np.float64)       # [x, y, vx, vy]
        self.P = np.eye(4, dtype=np.float64) * 100.0  # high initial uncertainty

        # State transition (constant velocity)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement: observe position only
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise (discrete white noise acceleration)
        q = sigma_a ** 2
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        self.Q_baseline = np.array([
            [dt4/4, 0,     dt3/2, 0    ],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0    ],
            [0,     dt3/2, 0,     dt2  ],
        ], dtype=np.float64) * q
        self.Q = self.Q_baseline.copy()

        # Measurement noise
        self.R = np.eye(2, dtype=np.float64) * sigma_meas**2

        self.frames_without_detection = 0
        self.max_coast = 60  # ~1s at 60fps — stationary objects may flicker
        self._initialized = False

    def predict(self):
        """Predict step — call every frame."""
        if not self._initialized:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """Update step — call with [x, y] when detection available, None otherwise."""
        if measurement is not None:
            z = np.array(measurement, dtype=np.float64)

            if not self._initialized:
                self.x[:2] = z
                self.x[2:] = 0.0
                self._initialized = True
                self.frames_without_detection = 0
                return

            # Innovation
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)

            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ self.H) @ self.P

            # Adaptive process noise (NIS check)
            nis = y.T @ np.linalg.inv(S) @ y
            if nis > 6.0:
                # Enemy maneuvering hard — boost Q
                self.Q = self.Q_baseline * 4.0
            elif nis < 1.0:
                # Tracking well — decay toward baseline
                self.Q = self.Q * 0.9 + self.Q_baseline * 0.1

            self.frames_without_detection = 0
        else:
            self.frames_without_detection += 1

    @property
    def position(self):
        """Estimated [x, y] in world units."""
        return self.x[:2].copy()

    @property
    def velocity(self):
        """Estimated [vx, vy] in world units/s."""
        return self.x[2:].copy()

    @property
    def speed(self):
        return np.linalg.norm(self.x[2:])

    @property
    def is_tracking(self):
        return self._initialized and self.frames_without_detection < self.max_coast

    def reset(self):
        self._initialized = False
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self.Q = self.Q_baseline.copy()
        self.frames_without_detection = 0


# ---------------------------------------------------------------------------
# Enemy detector (MOG2 + contour filtering)
# ---------------------------------------------------------------------------

class EnemyDetector:
    """Detects the enemy robot using reference frame differencing + MOG2.

    Two detection methods combined:
    1. Reference frame diff: captures empty arena, detects anything new (even stationary)
    2. MOG2 background subtraction: detects moving objects (complements reference diff)

    Either method triggering counts as a detection.

    Pipeline: diff/MOG2 → morphology → contour filter → exclude our robot → centroid
    """

    # Contour filter thresholds (720p, overhead camera)
    MIN_AREA = 800       # px² (~28×28) — raised to reject small noise
    MAX_AREA = 8000      # px² (~90×90)
    MIN_ASPECT = 0.3
    MAX_ASPECT = 3.0
    MIN_SOLIDITY = 0.5   # raised — real robots are solid shapes

    def __init__(self, diff_threshold=30):
        """Initialize detector.

        Args:
            diff_threshold: pixel intensity difference to count as foreground (0-255).
                            Higher = less sensitive, fewer false positives.
        """
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=30,
            detectShadows=False,
        )
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        self._diff_threshold = diff_threshold

        # Reference frame (empty arena)
        self._reference_gray = None
        self._last_fg_mask = None

        # Arena mask — only detect within the arena polygon
        self._arena_mask = None

        # Target lock — once tracking, stick with nearest detection
        self._track_lock_px = None

        # ROI tracker — once locked, track a small region instead of full frame
        self._roi_tracker = None
        self._roi_bbox = None  # (x, y, w, h)
        self._roi_frames_ok = 0

    def set_arena_corners(self, corners_px):
        """Set the arena boundary polygon for masking detections.

        Args:
            corners_px: list of [x, y] pixel coordinates of arena corners
        """
        if corners_px and len(corners_px) >= 3:
            pts = np.array(corners_px, dtype=np.int32)
            # We'll create the mask lazily once we know the frame size
            self._arena_pts = pts
            self._arena_mask = None  # reset, will be created on next detect()
            print(f"[enemy] Arena mask set ({len(corners_px)} corners)")

    def capture_reference(self, frame):
        """Capture the current frame as the empty arena reference.

        Call this when the arena is empty (no enemy, our robot can be present
        since it's excluded by ArUco mask anyway).
        """
        self._reference_gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0
        )
        print(f"[enemy] Reference frame captured ({frame.shape[1]}x{frame.shape[0]})")

    @property
    def has_reference(self) -> bool:
        return self._reference_gray is not None

    def detect(self, frame, our_robot_corners=None):
        """Detect enemy in frame. Returns (cx, cy) in pixels or None.

        Uses ROI tracker if locked on, otherwise reference diff + MOG2.
        """
        # If we have an active ROI tracker, use it (much more stable)
        if self._roi_tracker is not None:
            ok, bbox = self._roi_tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                self._roi_bbox = (x, y, w, h)
                cx, cy = x + w // 2, y + h // 2
                self._track_lock_px = (cx, cy)
                self._roi_frames_ok += 1
                self._last_fg_mask = None
                return (cx, cy)
            else:
                # Tracker lost — fall back to detection
                print("[enemy] ROI tracker lost — falling back to detection")
                self._roi_tracker = None
                self._roi_bbox = None
                self._roi_frames_ok = 0

        fg_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Method 1: Reference frame differencing (detects stationary objects)
        # This is the PRIMARY method — it sees anything new vs the empty arena
        if self._reference_gray is not None:
            current_gray = cv2.GaussianBlur(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0
            )
            diff = cv2.absdiff(current_gray, self._reference_gray)
            _, ref_mask = cv2.threshold(diff, self._diff_threshold, 255, cv2.THRESH_BINARY)
            # Pre-exclude our robot from reference diff (it will always differ)
            if our_robot_corners is not None:
                center = our_robot_corners.mean(axis=0).astype(int)
                cv2.circle(ref_mask, tuple(center), 100, 0, -1)
            fg_mask = cv2.bitwise_or(fg_mask, ref_mask)

        # Method 2: MOG2 (detects motion, supplements reference diff)
        # Very low learning rate so stationary objects don't get absorbed
        mog_mask = self.bg_sub.apply(frame, learningRate=0.001)
        _, mog_mask = cv2.threshold(mog_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.bitwise_or(fg_mask, mog_mask)

        # Morphological cleanup
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._kernel_close)

        # Mask to arena bounds only
        if hasattr(self, '_arena_pts') and self._arena_pts is not None:
            if self._arena_mask is None or self._arena_mask.shape != fg_mask.shape:
                self._arena_mask = np.zeros(fg_mask.shape, dtype=np.uint8)
                cv2.fillPoly(self._arena_mask, [self._arena_pts], 255)
            fg_mask = cv2.bitwise_and(fg_mask, self._arena_mask)

        # Exclude our robot (generous radius to cover full body, not just marker)
        if our_robot_corners is not None:
            center = our_robot_corners.mean(axis=0).astype(int)
            cv2.circle(fg_mask, tuple(center), 100, 0, -1)

        self._last_fg_mask = fg_mask

        # Find and filter contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.MIN_AREA < area < self.MAX_AREA):
                continue

            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h if h > 0 else 0
            if not (self.MIN_ASPECT < aspect < self.MAX_ASPECT):
                continue

            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < self.MIN_SOLIDITY:
                continue

            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                candidates.append((cx, cy, area))

        if not candidates:
            if hasattr(self, '_consecutive_detections'):
                self._consecutive_detections = 0
            return None

        # If we have a tracked position, pick the candidate closest to it
        # (target lock — don't jump to a new object)
        if self._track_lock_px is not None:
            lx, ly = self._track_lock_px
            candidates.sort(key=lambda c: (c[0]-lx)**2 + (c[1]-ly)**2)
        else:
            # No lock yet — pick largest
            candidates.sort(key=lambda c: c[2], reverse=True)

        best = (candidates[0][0], candidates[0][1])
        best_area = candidates[0][2]
        self._track_lock_px = best  # update lock position

        # Initialize ROI tracker after several consistent detections
        if self._roi_tracker is None and best is not None:
            if not hasattr(self, '_consecutive_detections'):
                self._consecutive_detections = 0
            self._consecutive_detections += 1
            if self._consecutive_detections < 10:
                return best  # don't lock on yet, just return raw detection

            # Enough consistent detections — lock on
            side = int(math.sqrt(best_area)) + 20  # pad a bit
            cx, cy = int(best[0]), int(best[1])
            x = max(0, cx - side // 2)
            y = max(0, cy - side // 2)
            w = min(side, frame.shape[1] - x)
            h = min(side, frame.shape[0] - y)
            if w > 10 and h > 10:
                self._roi_tracker = cv2.TrackerMIL_create()
                self._roi_tracker.init(frame, (x, y, w, h))
                self._roi_bbox = (x, y, w, h)
                self._roi_frames_ok = 0
                print(f"[enemy] ROI tracker initialized at ({x},{y} {w}x{h})")

        return best

    @property
    def fg_mask(self):
        """Return the latest foreground mask (for debug overlay)."""
        return self._last_fg_mask


# ---------------------------------------------------------------------------
# Combined enemy tracker
# ---------------------------------------------------------------------------

class EnemyTracker:
    """Combines detection + Kalman filtering for enemy robot tracking.

    Usage:
        tracker = EnemyTracker()
        # Each frame:
        tracker.update(frame, our_corners, px_to_cm_func)
        if tracker.is_tracking:
            pos = tracker.position_cm
            vel = tracker.velocity_cm_s
    """

    def __init__(self, dt=1/60, sigma_a=5.0, sigma_meas_cm=1.0):
        self.detector = EnemyDetector()
        self.kalman = EnemyKalmanFilter(dt=dt, sigma_a=sigma_a,
                                         sigma_meas=sigma_meas_cm / 100.0)
        self._last_detection_px = None
        self._last_detection_cm = None

    def update(self, frame, our_robot_corners=None, px_to_cm=None):
        """Run detection + Kalman update for this frame.

        Args:
            frame: BGR camera frame
            our_robot_corners: ArUco corners of our robot (for exclusion)
            px_to_cm: callable(px_x, px_y) -> (x_cm, y_cm)
        """
        # Detect enemy in pixel space
        det_px = self.detector.detect(frame, our_robot_corners)

        # Convert to world coordinates if detection available
        det_cm = None
        if det_px is not None and px_to_cm is not None:
            try:
                x_cm, y_cm = px_to_cm(det_px[0], det_px[1])
                det_cm = (x_cm / 100.0, y_cm / 100.0)  # cm -> m for Kalman
            except (ValueError, cv2.error):
                det_cm = None

        self._last_detection_px = det_px
        self._last_detection_cm = det_cm

        # Kalman predict + update
        self.kalman.predict()
        self.kalman.update(det_cm)

    @property
    def is_tracking(self) -> bool:
        return self.kalman.is_tracking

    @property
    def enemy_detected(self) -> bool:
        return self._last_detection_px is not None

    @property
    def detection_px(self):
        """Last raw detection in pixel coordinates, or None."""
        return self._last_detection_px

    @property
    def position_cm(self):
        """Kalman-filtered enemy position in cm."""
        pos_m = self.kalman.position
        return pos_m * 100.0  # m -> cm

    @property
    def velocity_cm_s(self):
        """Kalman-filtered enemy velocity in cm/s."""
        vel_m = self.kalman.velocity
        return vel_m * 100.0  # m/s -> cm/s

    @property
    def speed_cm_s(self) -> float:
        return self.kalman.speed * 100.0

    @property
    def position_m(self):
        """Kalman-filtered position in meters."""
        return self.kalman.position

    @property
    def velocity_m_s(self):
        """Kalman-filtered velocity in m/s."""
        return self.kalman.velocity

    def draw_overlay(self, frame, px_to_cm=None, cm_to_px=None):
        """Draw enemy detection and tracking overlay on frame."""
        # Raw detection (yellow circle)
        if self._last_detection_px is not None:
            cx, cy = int(self._last_detection_px[0]), int(self._last_detection_px[1])
            cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 2)
            cv2.putText(frame, "ENEMY", (cx + 12, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Kalman-filtered position + velocity arrow (red)
        if self.is_tracking and cm_to_px is not None:
            pos_cm = self.position_cm
            try:
                px, py = cm_to_px(pos_cm[0], pos_cm[1])
                px_i, py_i = int(px), int(py)
                cv2.circle(frame, (px_i, py_i), 10, (0, 0, 255), 2)

                # Velocity arrow
                vel = self.velocity_cm_s
                if np.linalg.norm(vel) > 1.0:  # >1 cm/s
                    end_cm = pos_cm + vel * 0.5  # 0.5s prediction
                    try:
                        ex, ey = cm_to_px(end_cm[0], end_cm[1])
                        cv2.arrowedLine(frame, (px_i, py_i), (int(ex), int(ey)),
                                        (0, 0, 255), 2, tipLength=0.3)
                    except (ValueError, cv2.error):
                        pass
            except (ValueError, cv2.error):
                pass

    def reset(self):
        self.kalman.reset()
        self.detector._track_lock_px = None
        self.detector._roi_tracker = None
        self.detector._roi_bbox = None
        self.detector._roi_frames_ok = 0
        self.detector._consecutive_detections = 0
        self._last_detection_px = None
        self._last_detection_cm = None
