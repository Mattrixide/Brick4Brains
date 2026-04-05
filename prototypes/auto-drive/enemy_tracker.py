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
    GATE_THRESHOLD = 2.5  # tight gate — reject jumps aggressively

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
# Enemy detector — "Not-Us" with color assist
# ---------------------------------------------------------------------------

# HSV ranges for yellow ArUco marker (tune per venue if needed)
YELLOW_LOW = np.array([18, 80, 80], dtype=np.uint8)
YELLOW_HIGH = np.array([38, 255, 255], dtype=np.uint8)

# Fraction thresholds for blob classification
YELLOW_FRAC_OURS = 0.15    # >15% yellow = our robot
YELLOW_FRAC_ENEMY = 0.03   # <3% yellow = definitely enemy


class EnemyDetector:
    """Detects enemy by elimination: find all blobs, exclude ours by color.

    No reference frame needed. Uses MOG2 for foreground detection +
    yellow color classification to identify our robot vs enemy.
    When blobs merge, splits by yellow vs non-yellow pixels.

    Pipeline: MOG2 → morphology → arena mask → color classify → enemy centroid
    """

    MIN_AREA = 400
    MAX_AREA = 30000
    MIN_SOLIDITY = 0.4

    def __init__(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=40,
            detectShadows=False,
        )
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        self._arena_mask = None
        self._arena_pts = None
        self._last_fg_mask = None
        self._track_lock_px = None

        # MOG2 warmup
        self._warmup_frames = 0
        self._warmup_needed = 90  # 1.5s at 60fps

    def set_arena_corners(self, corners_px, expand_px=30):
        """Set arena boundary polygon."""
        if corners_px and len(corners_px) >= 3:
            pts = np.array(corners_px, dtype=np.float32)
            center = pts.mean(axis=0)
            for i in range(len(pts)):
                d = pts[i] - center
                length = np.linalg.norm(d)
                if length > 0:
                    pts[i] = pts[i] + d / length * expand_px
            self._arena_pts = pts.astype(np.int32)
            self._arena_mask = None
            print(f"[enemy] Arena mask set ({len(corners_px)} corners, +{expand_px}px)")

    def capture_reference(self, frame):
        """Reset MOG2 and track lock (replaces old reference frame capture)."""
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False,
        )
        self._warmup_frames = 0
        self._track_lock_px = None
        print(f"[enemy] MOG2 reset — warming up ({self._warmup_needed} frames)")

    @property
    def has_reference(self) -> bool:
        return self._warmup_frames >= self._warmup_needed

    def detect(self, frame, our_robot_corners=None, use_reference_diff=True):
        """Detect enemy. Returns (cx, cy) in pixels or None.

        Args:
            frame: BGR camera frame
            our_robot_corners: ArUco corners (used for proximity check, not exclusion mask)
            use_reference_diff: ignored (kept for API compatibility)
        """
        h, w = frame.shape[:2]

        # MOG2 foreground
        # Fast learning during warmup to build background, then nearly frozen
        # so stationary enemies stay visible for the whole match
        if self._warmup_frames < self._warmup_needed:
            lr = 0.02  # fast warmup
        else:
            lr = 0.0002  # nearly frozen — 83 seconds to absorb
        fg_mask = self.bg_sub.apply(frame, learningRate=lr)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphology
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._kernel_close)

        # Arena mask
        if self._arena_pts is not None:
            if self._arena_mask is None or self._arena_mask.shape != fg_mask.shape:
                self._arena_mask = np.zeros(fg_mask.shape, dtype=np.uint8)
                cv2.fillPoly(self._arena_mask, [self._arena_pts], 255)
            fg_mask = cv2.bitwise_and(fg_mask, self._arena_mask)

        self._last_fg_mask = fg_mask

        # Warmup
        self._warmup_frames += 1
        if self._warmup_frames < self._warmup_needed:
            return None

        # Color classification
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)

        # ArUco center for proximity check
        aruco_center = None
        if our_robot_corners is not None:
            aruco_center = our_robot_corners.mean(axis=0)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        enemy_candidates = []
        merged_enemy_pos = None

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.MIN_AREA or area > self.MAX_AREA:
                continue

            hull_area = cv2.contourArea(cv2.convexHull(c))
            if hull_area > 0 and (area / hull_area) < self.MIN_SOLIDITY:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # Color analysis: what fraction of this blob is yellow?
            blob_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(blob_mask, [c], -1, 255, -1)
            yellow_in_blob = cv2.bitwise_and(yellow_mask, blob_mask)
            yellow_count = cv2.countNonZero(yellow_in_blob)
            blob_pixels = cv2.countNonZero(blob_mask)
            yellow_frac = yellow_count / blob_pixels if blob_pixels > 0 else 0

            # Near ArUco?
            near_aruco = False
            if aruco_center is not None:
                dist = math.sqrt((cx - aruco_center[0])**2 + (cy - aruco_center[1])**2)
                near_aruco = dist < 120

            # Classification
            if near_aruco and yellow_frac > YELLOW_FRAC_ENEMY:
                # Our robot — skip
                continue

            if yellow_frac > YELLOW_FRAC_OURS:
                # Mostly yellow — it's ours even without ArUco
                continue

            if yellow_frac > YELLOW_FRAC_ENEMY and area > 5000:
                # MERGED BLOB: contains some yellow (ours) + mostly not-yellow (enemy)
                # Split: find centroid of non-yellow pixels
                not_yellow = cv2.bitwise_and(blob_mask, cv2.bitwise_not(yellow_mask))
                not_yellow = cv2.erode(not_yellow, self._kernel_open)
                nM = cv2.moments(not_yellow)
                if nM["m00"] > 0:
                    merged_enemy_pos = (nM["m10"] / nM["m00"], nM["m01"] / nM["m00"])
                continue

            # Not yellow, not near ArUco → enemy candidate
            enemy_candidates.append((cx, cy, area))

        # Pick best candidate
        result = None

        if enemy_candidates:
            if self._track_lock_px is not None:
                lx, ly = self._track_lock_px
                enemy_candidates.sort(key=lambda c: (c[0]-lx)**2 + (c[1]-ly)**2)
                best = enemy_candidates[0]
                dist = math.sqrt((best[0]-lx)**2 + (best[1]-ly)**2)
                if dist < 250:
                    result = (best[0], best[1])
            else:
                enemy_candidates.sort(key=lambda c: c[2], reverse=True)
                result = (enemy_candidates[0][0], enemy_candidates[0][1])

        # Fallback to merged blob split
        if result is None and merged_enemy_pos is not None:
            result = merged_enemy_pos

        if result is not None:
            self._track_lock_px = result

        return result

    @property
    def fg_mask(self):
        return self._last_fg_mask

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
        """Run detection + Kalman update for this frame."""
        det_px = self.detector.detect(frame, our_robot_corners)

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
        # Raw detection — box + crosshair
        if self._last_detection_px is not None:
            cx, cy = int(self._last_detection_px[0]), int(self._last_detection_px[1])
            # Draw targeting box
            box_size = 40
            cv2.rectangle(frame, (cx - box_size, cy - box_size),
                          (cx + box_size, cy + box_size), (0, 255, 255), 2)
            # Crosshair
            cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
            cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)
            cv2.putText(frame, "ENEMY", (cx + box_size + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        elif self.is_tracking:
            # No raw detection but Kalman is tracking — show predicted position
            if cm_to_px is not None:
                pos_cm = self.position_cm
                try:
                    px, py = cm_to_px(pos_cm[0], pos_cm[1])
                    px_i, py_i = int(px), int(py)
                    cv2.rectangle(frame, (px_i - 30, py_i - 30),
                                  (px_i + 30, py_i + 30), (0, 100, 255), 1)
                    cv2.putText(frame, "PREDICT", (px_i + 35, py_i - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
                except (ValueError, cv2.error):
                    pass

        # Kalman-filtered position + velocity arrow (red)
        if self.is_tracking and cm_to_px is not None:
            pos_cm = self.position_cm
            try:
                px, py = cm_to_px(pos_cm[0], pos_cm[1])
                px_i, py_i = int(px), int(py)
                cv2.circle(frame, (px_i, py_i), 10, (0, 0, 255), 2)

                # Distance label
                if self._last_detection_px is not None:
                    dcx = int(self._last_detection_px[0])
                    cv2.putText(frame, f"{np.linalg.norm(pos_cm):.0f}cm",
                                (px_i + 15, py_i + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # Velocity arrow
                vel = self.velocity_cm_s
                if np.linalg.norm(vel) > 1.0:
                    end_cm = pos_cm + vel * 0.5
                    try:
                        ex, ey = cm_to_px(end_cm[0], end_cm[1])
                        cv2.arrowedLine(frame, (px_i, py_i), (int(ex), int(ey)),
                                        (0, 0, 255), 2, tipLength=0.3)
                    except (ValueError, cv2.error):
                        pass
            except (ValueError, cv2.error):
                pass

        # Show track lock position (small green dot)
        lock = self.detector._track_lock_px
        if lock is not None:
            cv2.circle(frame, (int(lock[0]), int(lock[1])), 4, (0, 255, 0), -1)

    def reset(self):
        self.kalman.reset()
        self.detector._track_lock_px = None
        self.detector._consecutive_detections = 0
        self.detector._static_positions = []
        self._last_detection_px = None
        self._last_detection_cm = None
