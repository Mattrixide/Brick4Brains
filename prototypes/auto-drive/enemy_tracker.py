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
        self.max_coast = 30  # ~0.5s at 60fps
        self._initialized = False

    def predict(self):
        """Predict step — call every frame."""
        if not self._initialized:
            return
        # Decay velocity when coasting (no detections)
        if self.frames_without_detection > 0:
            self.x[2] *= 0.80
            self.x[3] *= 0.80
        self.x = self.F @ self.x
        # Clamp to arena bounds, zero velocity if clamped
        for i, vi in [(0, 2), (1, 3)]:
            if self.x[i] < -2.0 or self.x[i] > 2.0:
                self.x[i] = np.clip(self.x[i], -2.0, 2.0)
                self.x[vi] = 0.0
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

            # Standard Kalman update (gating handled by EnemyTracker speed gate)
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)

            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ self.H) @ self.P

            # Clamp velocity to physically possible range (500 cm/s = 5 m/s)
            MAX_VEL = 5.0  # m/s
            speed = math.sqrt(self.x[2]**2 + self.x[3]**2)
            if speed > MAX_VEL:
                scale = MAX_VEL / speed
                self.x[2] *= scale
                self.x[3] *= scale

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

# Exclusion radius around ArUco marker center (pixels) — legacy fallback
ARUCO_EXCLUSION_RADIUS_PX = 80

# Robot body dimensions relative to ArUco marker size (50mm marker)
# From screenshot: robot body is roughly 2.5x marker width, 1.5x marker length
# Plus small margin for position error
ROBOT_WIDTH_MARKER_SCALE = 4.5   # side-to-side (increased to cover wheels/shadow)
ROBOT_LENGTH_MARKER_SCALE = 3.5  # front-to-back (increased to cover weapon/wedge)


def _compute_robot_footprint(aruco_corners):
    """Compute rotated rectangle polygon of our robot body from ArUco corners.

    Returns 4x2 int32 array of polygon vertices, or None.
    """
    if aruco_corners is None:
        return None
    corners = aruco_corners.reshape(-1, 2).astype(np.float64)
    if len(corners) != 4:
        return None

    center = corners.mean(axis=0)

    # Marker size in pixels (average edge length)
    edge1 = np.linalg.norm(corners[1] - corners[0])
    edge2 = np.linalg.norm(corners[2] - corners[1])
    marker_px = (edge1 + edge2) / 2

    # Heading from top edge of marker
    top_edge = corners[1] - corners[0]
    heading = math.atan2(top_edge[1], top_edge[0])

    # Robot half-dimensions in pixels
    half_w = marker_px * ROBOT_WIDTH_MARKER_SCALE / 2
    half_l = marker_px * ROBOT_LENGTH_MARKER_SCALE / 2

    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    # Local corners: width along marker top-edge direction, length perpendicular
    local = np.array([
        [-half_w, -half_l],
        [half_w, -half_l],
        [half_w, half_l],
        [-half_w, half_l],
    ])

    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    rotated = (R @ local.T).T + center
    return rotated.astype(np.int32)


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
    REF_DIFF_THRESHOLD = 20  # pixel intensity difference to count as foreground

    def __init__(self):
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        self._arena_mask = None
        self._arena_pts = None
        self._last_fg_mask = None
        self._track_lock_px = None

        # Previous frame for temporal diff (detects moving objects)
        self._prev_gray = None

        # Last winning contour (for orientation estimation)
        self._last_contour = None

        # Our robot footprint polygon (for debug overlay)
        self._robot_footprint = None

        # Reference frame (empty arena snapshot — primary detection method)
        self._reference_gray = None

        # Warmup counter (allow a few frames for camera auto-exposure to settle)
        self._warmup_frames = 0
        self._warmup_needed = 30  # 0.5s at 60fps

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
        """Capture empty arena as reference frame.

        Call this when the arena is empty (no robots). The reference frame
        is used for absdiff-based detection of anything that isn't floor.
        """
        if len(frame.shape) == 3:
            self._reference_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            self._reference_gray = frame.copy()

        self._reference_gray = cv2.GaussianBlur(self._reference_gray, (5, 5), 0)
        self._warmup_frames = 0
        self._track_lock_px = None
        self._prev_gray = None
        print(f"[enemy] Reference frame captured — detection active")

    @property
    def has_reference_frame(self) -> bool:
        """True if an empty-arena reference frame has been captured."""
        return self._reference_gray is not None

    @property
    def has_reference(self) -> bool:
        return self._warmup_frames >= self._warmup_needed

    def detect(self, frame, our_robot_corners=None, use_reference_diff=True,
               predicted_px=None):
        """Detect enemy. Returns (cx, cy) in pixels or None.

        Uses reference frame absdiff as primary detection (no MOG2).
        Frame-to-frame diff tags moving objects to break ghost ties.

        Args:
            frame: camera frame (BGR or grayscale)
            our_robot_corners: ArUco corners array for our robot exclusion
            use_reference_diff: ignored (kept for API compatibility)
        """
        h, w = frame.shape[:2]

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Primary: reference frame diff (detects anything not floor)
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        if self._reference_gray is not None and self._reference_gray.shape == gray.shape:
            diff = cv2.absdiff(blurred, self._reference_gray)
            _, fg_mask = cv2.threshold(diff, self.REF_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Secondary: frame-to-frame diff (tags moving objects)
        temporal_mask = np.zeros((h, w), dtype=np.uint8)
        if self._prev_gray is not None and self._prev_gray.shape == blurred.shape:
            tdiff = cv2.absdiff(blurred, self._prev_gray)
            _, temporal_mask = cv2.threshold(tdiff, 15, 255, cv2.THRESH_BINARY)
            temporal_mask = cv2.morphologyEx(temporal_mask, cv2.MORPH_CLOSE, self._kernel_close)
        self._prev_gray = blurred.copy()

        # Morphology
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._kernel_close)

        # Arena mask
        if self._arena_pts is not None:
            if self._arena_mask is None or self._arena_mask.shape != fg_mask.shape:
                self._arena_mask = np.zeros(fg_mask.shape, dtype=np.uint8)
                cv2.fillPoly(self._arena_mask, [self._arena_pts], 255)
            fg_mask = cv2.bitwise_and(fg_mask, self._arena_mask)

        # Subtract our robot's footprint from foreground mask
        # This replaces the old per-contour ArUco exclusion and handles
        # the blob-merge case when robots are in contact.
        # When ArUco is lost (corners=None), keep the last valid footprint
        # to prevent the enemy tracker from locking onto our robot.
        if our_robot_corners is not None:
            new_fp = _compute_robot_footprint(our_robot_corners)
            if new_fp is not None:
                self._robot_footprint = new_fp
            else:
                # Fallback: simple circle exclusion
                corners = our_robot_corners.reshape(-1, 2)
                center = corners.mean(axis=0).astype(np.int32)
                cv2.circle(fg_mask, tuple(center), ARUCO_EXCLUSION_RADIUS_PX, 0, -1)

        # Apply footprint mask (current or stale from last detection)
        if self._robot_footprint is not None:
            cv2.fillPoly(fg_mask, [self._robot_footprint], 0)

        # Progressive reference healing: slowly blend current frame into reference
        # where our robot and enemy AREN'T. Ghost blobs at old positions fade in ~1s.
        if self._reference_gray is not None and self._robot_footprint is not None:
            heal_mask = np.ones((h, w), dtype=np.uint8) * 255
            cv2.fillPoly(heal_mask, [self._robot_footprint], 0)  # exclude current robot pos
            # Exclude enemy detection area (use previous frame's detection)
            if self._track_lock_px is not None:
                ex, ey = int(self._track_lock_px[0]), int(self._track_lock_px[1])
                cv2.circle(heal_mask, (ex, ey), 60, 0, -1)
            if self._arena_pts is not None and self._arena_mask is not None:
                heal_mask = cv2.bitwise_and(heal_mask, self._arena_mask)
            alpha_heal = 0.05  # ~1s to fully heal at 60fps (0.95^60 ≈ 0.05)
            blended = cv2.addWeighted(
                self._reference_gray, 1.0 - alpha_heal,
                blurred, alpha_heal, 0
            )
            self._reference_gray = np.where(
                heal_mask > 0, blended, self._reference_gray
            )

        self._last_fg_mask = fg_mask

        # Warmup
        self._warmup_frames += 1
        if self._warmup_frames < self._warmup_needed:
            return None

        # Find contours (our robot already masked out)
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

            # Check if this blob overlaps with temporal diff (is it moving?)
            # Sample temporal mask at centroid ± small region
            cx_i, cy_i = int(cx), int(cy)
            r = 20  # sample radius
            y0, y1 = max(0, cy_i-r), min(h, cy_i+r)
            x0, x1 = max(0, cx_i-r), min(w, cx_i+r)
            is_moving = False
            if y1 > y0 and x1 > x0:
                region = temporal_mask[y0:y1, x0:x1]
                is_moving = np.mean(region) > 30  # >30/255 means significant motion

            enemy_candidates.append((cx, cy, area, is_moving, c))

        # Pick best candidate using scoring:
        # - Moving blobs get priority (real enemy, not ghost)
        # - Among same-moving-status, prefer largest blob
        # - Track lock gives a small proximity bonus, not hard lock
        result = None
        self._last_contour = None

        if enemy_candidates:
            scored = []
            # Get our robot center for anti-self-detection
            our_center = None
            if self._robot_footprint is not None:
                pts = self._robot_footprint.reshape(-1, 2).astype(np.float64)
                our_center = pts.mean(axis=0)

            for cx, cy, area, moving, contour in enemy_candidates:
                score = 0.0
                score += 5000 if moving else 0
                score += area
                # Penalize distance from Kalman prediction (if available)
                if predicted_px is not None:
                    dist_to_pred = math.sqrt((cx - predicted_px[0])**2 + (cy - predicted_px[1])**2)
                    score -= dist_to_pred * 2.0
                # Penalize detections near our robot — likely self-detection leak
                if our_center is not None:
                    dist_to_us = math.sqrt((cx - our_center[0])**2 + (cy - our_center[1])**2)
                    if dist_to_us < 150:  # within ~150px of our robot
                        score -= 8000  # strong penalty, overrides area bonus
                scored.append((cx, cy, area, moving, contour, score))
            scored.sort(key=lambda c: c[5], reverse=True)

            result = (scored[0][0], scored[0][1])
            self._last_contour = scored[0][4]

            # Save debug info: all candidates with scores
            self._last_candidates = [
                {"px": (round(s[0], 1), round(s[1], 1)), "area": int(s[2]),
                 "moving": bool(s[3]), "score": round(float(s[5]), 0)}
                for s in scored[:5]  # top 5
            ]
        else:
            self._last_candidates = []

        # Update lock to follow the detection
        if result is not None:
            self._track_lock_px = result

        return result

    @property
    def fg_mask(self):
        """Return the latest foreground mask (for debug overlay)."""
        return self._last_fg_mask


# ---------------------------------------------------------------------------
# Enemy orientation estimation (3-layer: velocity → shape → hold)
# ---------------------------------------------------------------------------

class EnemyOrientationEstimator:
    """Estimates enemy heading from velocity and contour shape analysis.

    Layer 1: Velocity heading (when speed > threshold)
    Layer 2: minAreaRect + half-split disambiguation (when stationary)
    Layer 3: Hold last known heading with decaying confidence
    """

    def __init__(self, velocity_threshold_cm_s: float = 5.0, ema_alpha: float = 0.3):
        self._vel_threshold = velocity_threshold_cm_s
        self._ema_alpha = ema_alpha
        self._heading_rad: float | None = None  # current best estimate
        self._confidence: float = 0.0           # 0=unknown, 1=high
        self._method: str = "none"              # last method used
        self._angular_vel_deg_s: float = 0.0    # for spin detection
        self._prev_shape_angle: float | None = None
        self._is_spinning: bool = False

    def update(self, velocity_cm_s: tuple[float, float] | None,
               contour: np.ndarray | None) -> None:
        """Update heading estimate with new frame data.

        Uses velocity heading when moving (reliable).
        Holds last known heading when stationary (shape analysis is too noisy
        at 50px blob size to determine front from back).
        """

        heading_candidate = None
        method = "none"
        conf = 0.0

        # Velocity heading — only reliable signal at this resolution
        if velocity_cm_s is not None:
            vx, vy = velocity_cm_s
            speed = math.hypot(vx, vy)
            if speed > self._vel_threshold:
                heading_candidate = math.atan2(vy, vx)
                method = "velocity"
                conf = min(1.0, speed / (self._vel_threshold * 3))

        # Apply heading update or hold
        if heading_candidate is not None:
            if self._heading_rad is not None:
                # EMA filter with angle wrapping
                diff = self._angle_diff(heading_candidate, self._heading_rad)
                self._heading_rad = self._heading_rad + self._ema_alpha * diff
                self._heading_rad = math.atan2(
                    math.sin(self._heading_rad), math.cos(self._heading_rad)
                )
            else:
                self._heading_rad = heading_candidate
            self._confidence = conf
            self._method = method
        else:
            # Hold last heading, slow confidence decay
            self._confidence = max(0.0, self._confidence - 0.005)
            self._method = "hold" if self._heading_rad is not None else "none"

        # Spin detection from shape angle changes
        if contour is not None and len(contour) >= 5:
            self._detect_spin(contour)

    def _shape_heading(self, contour: np.ndarray) -> float | None:
        """Estimate heading from minAreaRect + half-split disambiguation."""
        rect = cv2.minAreaRect(contour)
        (rcx, rcy), (w, h), angle_deg = rect

        # Ensure angle represents the long axis (side-to-side for our robots)
        if w < h:
            angle_deg += 90.0
        axis_rad = math.radians(angle_deg)

        # Centroid from moments
        M = cv2.moments(contour)
        if M["m00"] < 1:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Principal axis direction (perpendicular to long axis = front-back axis)
        # Long axis = side-to-side, so front-back is perpendicular
        front_back_rad = axis_rad + math.pi / 2
        fb_dx = math.cos(front_back_rad)
        fb_dy = math.sin(front_back_rad)

        # Half-split: project contour points onto front-back axis
        pts = contour.reshape(-1, 2).astype(np.float64)
        centered = pts - np.array([rcx, rcy])
        projections = centered[:, 0] * fb_dx + centered[:, 1] * fb_dy

        n_pos = np.sum(projections > 0)
        n_neg = np.sum(projections <= 0)

        if min(n_pos, n_neg) < 3:
            return None

        ratio = max(n_pos, n_neg) / max(1, min(n_pos, n_neg))
        if ratio < 1.1:
            # Too symmetric — also check centroid offset
            offset_x = cx - rcx
            offset_y = cy - rcy
            offset_proj = offset_x * fb_dx + offset_y * fb_dy
            if abs(offset_proj) < 1.0:
                return None  # truly symmetric, can't determine front

            # Front is opposite the centroid shift (centroid is toward heavy/back end)
            # Flip Y for pixel→world conversion
            if offset_proj > 0:
                return math.atan2(fb_dy, -fb_dx)
            else:
                return math.atan2(-fb_dy, fb_dx)

        # Front = lighter half (fewer points = tapered wedge)
        if n_pos < n_neg:
            heading = math.atan2(fb_dy, fb_dx)
        else:
            heading = math.atan2(-fb_dy, -fb_dx)

        # Validate with centroid offset if available
        offset_x = cx - rcx
        offset_y = cy - rcy
        offset_proj = offset_x * fb_dx + offset_y * fb_dy
        if abs(offset_proj) > 1.5:
            # Centroid offset disagrees — front is away from centroid
            centroid_heading = math.atan2(-offset_y, -offset_x)
            # Use centroid direction if it conflicts with half-split
            dot = math.cos(heading) * math.cos(centroid_heading) + \
                  math.sin(heading) * math.sin(centroid_heading)
            if dot < 0:
                heading += math.pi  # flip

        # Flip Y axis: pixel coords have Y-down, world coords have Y-up
        heading = math.atan2(-math.sin(heading), math.cos(heading))
        return heading

    def _detect_spin(self, contour: np.ndarray) -> None:
        """Detect if enemy is spinning based on shape angle change rate."""
        rect = cv2.minAreaRect(contour)
        _, (w, h), angle_deg = rect
        if w < h:
            angle_deg += 90.0

        if self._prev_shape_angle is not None:
            diff = abs(angle_deg - self._prev_shape_angle)
            if diff > 90:
                diff = 180 - diff  # handle wrap
            # At 60fps, >20 deg/frame = 1200 deg/s = spinning
            self._is_spinning = diff > 20
            self._angular_vel_deg_s = diff * 60  # rough estimate
        self._prev_shape_angle = angle_deg

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Shortest signed angle from b to a."""
        d = a - b
        return (d + math.pi) % (2.0 * math.pi) - math.pi

    @property
    def heading_rad(self) -> float | None:
        """Current heading estimate in radians, or None if unknown."""
        if self._confidence < 0.1:
            return None
        return self._heading_rad

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def method(self) -> str:
        return self._method

    @property
    def is_spinning(self) -> bool:
        return self._is_spinning


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

    def __init__(self, dt=1/60, sigma_a=5.0, sigma_meas_cm=8.0):
        self.detector = EnemyDetector()
        self.kalman = EnemyKalmanFilter(dt=dt, sigma_a=sigma_a,
                                         sigma_meas=sigma_meas_cm / 100.0)
        self.orientation = EnemyOrientationEstimator()
        self._last_detection_px = None
        self._last_detection_cm = None

    def update(self, frame, our_robot_corners=None, px_to_cm=None,
               use_reference_diff=True):
        """Run detection + Kalman update for this frame."""
        # Skip detection when our ArUco is lost — stale footprint mask
        # causes self-detection (our robot detected as enemy)
        if our_robot_corners is None:
            det_px = None
            self._last_detection_px = None  # clear stale prediction bias
            self._aruco_reacquire_cooldown = 2
        elif getattr(self, '_aruco_reacquire_cooldown', 0) > 0:
            # Skip 2 frames after ArUco reacquires (ghost transient)
            det_px = None
            self._aruco_reacquire_cooldown -= 1
        else:
            predicted_px = self._last_detection_px
            det_px = self.detector.detect(frame, our_robot_corners,
                                           predicted_px=predicted_px)

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
                    # Speed gate — adaptive, widens when filter is coasting
                    if self.kalman._initialized:
                        pred = self.kalman.position
                        jump_m = math.sqrt((det_m[0]-pred[0])**2 + (det_m[1]-pred[1])**2)
                        # Base gate 100cm, widens with missed frames up to 300cm
                        gate_m = 1.0 + 0.3 * self.kalman.frames_without_detection
                        gate_m = min(gate_m, 3.0)
                        if jump_m > gate_m:
                            # Too far — but if we've been coasting, reinit filter
                            if self.kalman.frames_without_detection >= 5:
                                det_cm = det_m
                                # Reinit: snap position, zero velocity, inflate P
                                self.kalman.x[:2] = np.array(det_m)
                                self.kalman.x[2:] = 0.0
                                self.kalman.P = np.eye(4) * 10.0
                                self.kalman.frames_without_detection = 0
                            else:
                                det_px = None
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

        # Orientation estimation (velocity + shape)
        vel = self.velocity_cm_s if self.kalman._initialized else None
        contour = self.detector._last_contour
        self.orientation.update(vel, contour)

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
    def heading_rad(self) -> float | None:
        """Estimated enemy heading in radians, or None if unknown."""
        return self.orientation.heading_rad

    @property
    def heading_confidence(self) -> float:
        """Heading confidence 0-1."""
        return self.orientation.confidence

    @property
    def heading_method(self) -> str:
        """How heading was determined: 'velocity', 'shape', 'hold', 'none'."""
        return self.orientation.method

    @property
    def is_spinning(self) -> bool:
        """True if enemy appears to be spinning in place."""
        return self.orientation.is_spinning

    def contact_estimate_cm(self, our_pos_cm, our_heading_rad,
                             contact_distance_cm: float = 15.0):
        """Estimate enemy position during contact (when vision is occluded).

        Assumes enemy is directly in front of us at contact_distance.
        Use when is_tracking but enemy_detected is False (Kalman coasting).
        """
        ex = our_pos_cm[0] + contact_distance_cm * math.cos(our_heading_rad)
        ey = our_pos_cm[1] + contact_distance_cm * math.sin(our_heading_rad)
        return np.array([ex, ey])

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

        # Heading arrow (green = velocity, cyan = shape, gray = hold)
        if self.heading_rad is not None and self._last_detection_px is not None:
            cx, cy = int(self._last_detection_px[0]), int(self._last_detection_px[1])
            h_rad = self.heading_rad
            arrow_len = 50
            hx = cx + int(arrow_len * math.cos(h_rad))
            hy = cy - int(arrow_len * math.sin(h_rad))  # negate Y for screen coords
            method = self.heading_method
            h_color = (0, 255, 0) if method == "velocity" else \
                      (255, 255, 0) if method == "shape" else (150, 150, 150)
            cv2.arrowedLine(frame, (cx, cy), (hx, hy), h_color, 2, tipLength=0.3)
            cv2.putText(frame, f"H:{method[:3]} {math.degrees(h_rad):.0f}°",
                        (cx + 45, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, h_color, 1)
            if self.is_spinning:
                cv2.putText(frame, "SPIN", (cx + 45, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Show track lock position (small green dot)
        lock = self.detector._track_lock_px
        if lock is not None:
            cv2.circle(frame, (int(lock[0]), int(lock[1])), 4, (0, 255, 0), -1)

        # Show our robot footprint mask (magenta outline)
        footprint = self.detector._robot_footprint
        if footprint is not None:
            cv2.polylines(frame, [footprint], True, (255, 0, 255), 2)

    def reset(self):
        self.kalman.reset()
        self.detector._track_lock_px = None
        self._last_detection_px = None
        self._last_detection_cm = None
