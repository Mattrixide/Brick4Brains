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
        # Clamp position to arena bounds (2.0m = 200cm)
        self.x[0] = np.clip(self.x[0], -2.0, 2.0)
        self.x[1] = np.clip(self.x[1], -2.0, 2.0)
        self.P = self.F @ self.P @ self.F.T + self.Q

    # Mahalanobis distance gate — reject measurements too far from prediction
    GATE_THRESHOLD = 3.0  # tighter gate — reject more aggressively

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

            # Mahalanobis gating — reject teleporting measurements
            mahal_sq = float(y.T @ np.linalg.inv(S) @ y)
            if mahal_sq > self.GATE_THRESHOLD ** 2:
                # Measurement is too far from prediction — reject it
                self.frames_without_detection += 1
                return

            K = self.P @ self.H.T @ np.linalg.inv(S)

            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ self.H) @ self.P

            # Adaptive process noise (NIS check)
            if mahal_sq > 6.0:
                self.Q = self.Q_baseline * 4.0
            elif mahal_sq < 1.0:
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
    MIN_AREA = 500       # px² (~22×22)
    MAX_AREA = 25000     # px² (~158×158) — larger to handle close-up perspective
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

        # Static detection filter — ignore blobs that don't move
        self._static_positions = []   # list of (x, y, frames_static)
        self._static_threshold = 15   # pixels — if blob hasn't moved this much, it's static
        self._static_max_frames = 30  # after this many frames, mark as static ghost

        # ROI tracker — once locked, track a small region instead of full frame
        self._roi_tracker = None
        self._roi_bbox = None  # (x, y, w, h)
        self._roi_frames_ok = 0

    def set_arena_corners(self, corners_px, expand_px=30):
        """Set the arena boundary polygon for masking detections.

        Args:
            corners_px: list of [x, y] pixel coordinates of arena corners
            expand_px: expand the polygon by this many pixels (catches edges)
        """
        if corners_px and len(corners_px) >= 3:
            pts = np.array(corners_px, dtype=np.float32)
            # Expand polygon outward to catch detections near arena edges
            center = pts.mean(axis=0)
            for i in range(len(pts)):
                direction = pts[i] - center
                length = np.linalg.norm(direction)
                if length > 0:
                    pts[i] = pts[i] + direction / length * expand_px
            self._arena_pts = pts.astype(np.int32)
            self._arena_mask = None  # reset, will be created on next detect()
            print(f"[enemy] Arena mask set ({len(corners_px)} corners, +{expand_px}px expansion)")

    def capture_reference(self, frame):
        """Capture the current frame as the empty arena reference.

        Call this when the arena is empty (no enemy, our robot can be present
        since it's excluded by ArUco mask anyway).
        """
        self._reference_gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0
        )
        # Reset track lock — old lock from false detections would block real enemy
        self._track_lock_px = None
        print(f"[enemy] Reference frame captured ({frame.shape[1]}x{frame.shape[0]})")

    @property
    def has_reference(self) -> bool:
        return self._reference_gray is not None

    def detect(self, frame, our_robot_corners=None, use_reference_diff=True):
        """Detect enemy in frame. Returns (cx, cy) in pixels or None.

        Args:
            frame: BGR camera frame
            our_robot_corners: ArUco corners for exclusion
            use_reference_diff: if False, skip reference diff (use during charge
                                when our robot moving creates too many artifacts)
        """
        fg_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Method 1: Reference frame differencing (only when robot is stationary)
        if self._reference_gray is not None and use_reference_diff:
            current_gray = cv2.GaussianBlur(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0
            )
            diff = cv2.absdiff(current_gray, self._reference_gray)
            _, ref_mask = cv2.threshold(diff, self._diff_threshold, 255, cv2.THRESH_BINARY)
            # Pre-exclude our robot from reference diff (it will always differ)
            if our_robot_corners is not None:
                center = our_robot_corners.mean(axis=0).astype(int)
                cv2.circle(ref_mask, tuple(center), 150, 0, -1)
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

        # Exclude our robot (very generous radius — robot appears large when close to camera)
        if our_robot_corners is not None:
            center = our_robot_corners.mean(axis=0).astype(int)
            cv2.circle(fg_mask, tuple(center), 150, 0, -1)

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

        # Reject teleporting detections — if closest candidate jumped too far
        # from last known position, it's probably our own robot or a ghost
        MAX_JUMP_PX = 60  # max pixels between frames — tighter to reject more noise

        # If we have a tracked position, pick nearest candidate + reject teleports
        if self._track_lock_px is not None:
            lx, ly = self._track_lock_px
            candidates.sort(key=lambda c: (c[0]-lx)**2 + (c[1]-ly)**2)

            # Check if nearest candidate is within jump limit
            nearest = candidates[0]
            dist = math.sqrt((nearest[0]-lx)**2 + (nearest[1]-ly)**2)
            if dist > MAX_JUMP_PX:
                # All candidates are too far — enemy probably occluded by our robot
                # Don't update lock, return None (Kalman will coast)
                return None
        else:
            # No lock yet — pick largest
            candidates.sort(key=lambda c: c[2], reverse=True)

        best = (candidates[0][0], candidates[0][1])
        self._track_lock_px = best

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

    def update(self, frame, our_robot_corners=None, px_to_cm=None,
               use_reference_diff=True):
        """Run detection + Kalman update for this frame.

        Args:
            frame: BGR camera frame
            our_robot_corners: ArUco corners of our robot (for exclusion)
            px_to_cm: callable(px_x, px_y) -> (x_cm, y_cm)
            use_reference_diff: pass False during charge to avoid self-detection
        """
        # Detect enemy in pixel space
        det_px = self.detector.detect(frame, our_robot_corners,
                                       use_reference_diff=use_reference_diff)

        # Convert to world coordinates if detection available
        det_cm = None
        if det_px is not None and px_to_cm is not None:
            try:
                x_cm, y_cm = px_to_cm(det_px[0], det_px[1])
                # Arena bounds check — reject detections outside the arena
                # (arena is ~244cm = 8ft, allow some margin)
                ARENA_MAX_CM = 300.0
                if abs(x_cm) > ARENA_MAX_CM or abs(y_cm) > ARENA_MAX_CM:
                    det_px = None  # outside arena — reject
                    # Uncomment for debug: print(f"[enemy] Rejected: ({x_cm:.0f},{y_cm:.0f})cm outside arena")
                else:
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
        self.detector._consecutive_detections = 0
        self.detector._static_positions = []
        self._last_detection_px = None
        self._last_detection_cm = None
