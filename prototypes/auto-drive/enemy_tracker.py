"""Enemy robot detection and tracking without ArUco markers.

Uses MOG2 background subtraction + contour filtering + Kalman filter
to detect, track, and estimate velocity of the enemy robot.
Our robot is excluded using ArUco bounding box proximity (works on mono/grayscale).
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
        self.max_coast = 120  # ~2s at 60fps — coast longer during charge
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
    GATE_THRESHOLD = 4.0  # loosened — enemy gets rammed and flies across arena

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
# Enemy detector — "Not-Us" via ArUco exclusion (works on mono/grayscale)
# ---------------------------------------------------------------------------

# Exclusion radius around ArUco marker center (pixels)
ARUCO_EXCLUSION_RADIUS_PX = 80


class EnemyDetector:
    """Detects enemy by elimination: find all foreground blobs, exclude ours by ArUco overlap.

    Works on both color and mono/grayscale frames. Uses MOG2 for foreground
    detection. Our robot is identified by proximity to its ArUco marker
    bounding box — any blob overlapping the ArUco region is excluded.

    Pipeline: (MOG2 | reference diff) → merge → morphology → arena mask → ArUco exclusion → enemy centroid

    Reference frame mode: capture an empty arena frame, then absdiff against it
    to detect anything that's not floor — works for stationary objects.
    MOG2 catches moving objects. The two masks are OR'd together.
    """

    MIN_AREA = 400
    MAX_AREA = 30000
    MIN_SOLIDITY = 0.4
    REF_DIFF_THRESHOLD = 35  # pixel intensity difference to count as foreground

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
        self._lock_stale_frames = 0  # how long lock hasn't moved
        self._lock_stale_threshold = 300  # ~5 seconds at 60fps — enemy may be stationary

        # MOG2 warmup
        self._warmup_frames = 0
        self._warmup_needed = 90  # 1.5s at 60fps

        # Reference frame (empty arena snapshot for static object detection)
        self._reference_gray = None

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
        """Capture empty arena as reference frame AND reset MOG2.

        Call this when the arena is empty (no robots). The reference frame
        is used for absdiff-based static object detection. MOG2 is also
        reset so it learns the current background.
        """
        # Store grayscale reference
        if len(frame.shape) == 3:
            self._reference_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            self._reference_gray = frame.copy()

        # Apply light blur to reduce noise in reference
        self._reference_gray = cv2.GaussianBlur(self._reference_gray, (5, 5), 0)

        # Reset MOG2
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False,
        )
        self._warmup_frames = 0
        self._track_lock_px = None
        print(f"[enemy] Reference frame captured + MOG2 reset — warming up ({self._warmup_needed} frames)")

    @property
    def has_reference_frame(self) -> bool:
        """True if an empty-arena reference frame has been captured."""
        return self._reference_gray is not None

    @property
    def has_reference(self) -> bool:
        return self._warmup_frames >= self._warmup_needed

    def detect(self, frame, our_robot_corners=None, use_reference_diff=True):
        """Detect enemy. Returns (cx, cy) in pixels or None.

        Works on both BGR and grayscale frames. Uses ArUco bounding box
        to exclude our robot — no color information needed.

        Args:
            frame: camera frame (BGR or grayscale)
            our_robot_corners: ArUco corners array for our robot exclusion
            use_reference_diff: ignored (kept for API compatibility)
        """
        h, w = frame.shape[:2]

        # Convert to grayscale for MOG2 if needed (consistent 1-channel input)
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # MOG2 foreground
        # Fast learning during warmup to build background, then nearly frozen
        # so stationary enemies stay visible for the whole match
        if self._warmup_frames < self._warmup_needed:
            lr = 0.02  # fast warmup
        else:
            lr = 0.0002  # nearly frozen — 83 seconds to absorb
        fg_mask = self.bg_sub.apply(gray, learningRate=lr)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Reference frame diff — catches stationary objects MOG2 misses
        if self._reference_gray is not None and self._reference_gray.shape == gray.shape:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            diff = cv2.absdiff(blurred, self._reference_gray)
            _, ref_mask = cv2.threshold(diff, self.REF_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            # Merge: anything detected by EITHER method is foreground
            fg_mask = cv2.bitwise_or(fg_mask, ref_mask)

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

        # Build ArUco exclusion zone from marker corners
        aruco_center = None
        aruco_bbox = None  # (x_min, y_min, x_max, y_max) expanded bounding box
        if our_robot_corners is not None:
            corners = our_robot_corners.reshape(-1, 2)
            aruco_center = corners.mean(axis=0)
            # Expand bounding box around ArUco corners
            x_min = corners[:, 0].min() - ARUCO_EXCLUSION_RADIUS_PX
            x_max = corners[:, 0].max() + ARUCO_EXCLUSION_RADIUS_PX
            y_min = corners[:, 1].min() - ARUCO_EXCLUSION_RADIUS_PX
            y_max = corners[:, 1].max() + ARUCO_EXCLUSION_RADIUS_PX
            aruco_bbox = (x_min, y_min, x_max, y_max)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        enemy_candidates = []

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

            # ArUco exclusion: skip blob if centroid is inside our robot's zone
            if aruco_bbox is not None:
                if aruco_bbox[0] <= cx <= aruco_bbox[2] and aruco_bbox[1] <= cy <= aruco_bbox[3]:
                    continue

            # Also check distance to ArUco center for blobs just outside bbox
            if aruco_center is not None:
                dist = math.sqrt((cx - aruco_center[0])**2 + (cy - aruco_center[1])**2)
                if dist < ARUCO_EXCLUSION_RADIUS_PX:
                    continue

            # Passed all filters — enemy candidate
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

        if result is not None:
            # Check if lock is stale (hasn't moved in ~2 seconds)
            if self._track_lock_px is not None:
                dx = result[0] - self._track_lock_px[0]
                dy = result[1] - self._track_lock_px[1]
                if abs(dx) < 5 and abs(dy) < 5:
                    self._lock_stale_frames += 1
                else:
                    self._lock_stale_frames = 0

                if self._lock_stale_frames > self._lock_stale_threshold:
                    self._track_lock_px = None
                    self._lock_stale_frames = 0
                    print("[enemy] Track lock stale — reacquiring")
                    if enemy_candidates:
                        enemy_candidates.sort(key=lambda c: c[2], reverse=True)
                        result = (enemy_candidates[0][0], enemy_candidates[0][1])
                        self._track_lock_px = result
                    return result

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
                # Arena bounds check
                ARENA_MAX_CM = 300.0
                if abs(x_cm) > ARENA_MAX_CM or abs(y_cm) > ARENA_MAX_CM:
                    det_px = None
                else:
                    det_m = (x_cm / 100.0, y_cm / 100.0)
                    # Speed gate — enemy can't teleport, but can get rammed far
                    # 150cm allows for being launched across the 8ft arena
                    if self.kalman._initialized:
                        pred = self.kalman.position
                        jump_m = math.sqrt((det_m[0]-pred[0])**2 + (det_m[1]-pred[1])**2)
                        if jump_m > 1.5:  # 150cm max between frames
                            det_px = None  # reject — physically impossible
                            det_m = None
                        else:
                            det_cm = det_m
                    else:
                        det_cm = det_m
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
