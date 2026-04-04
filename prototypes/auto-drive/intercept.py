"""Interception and pursuit algorithms for combat robot.

Provides:
  - Intercept point calculation (quadratic solver)
  - Proportional Navigation guidance (N=4)
  - Pure pursuit fallback
  - Pursuit state machine (SEARCH/ACQUIRE/INTERCEPT/CLOSE/LOST)
  - Smoothed intercept point (EMA filter)
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Intercept point solver
# ---------------------------------------------------------------------------

def compute_intercept_point(our_pos, our_speed, enemy_pos, enemy_vel):
    """Compute where to drive to intercept a constant-velocity enemy.

    Args:
        our_pos:    (x, y) our position (any consistent units)
        our_speed:  scalar max speed (same units/s)
        enemy_pos:  (x, y) enemy position
        enemy_vel:  (vx, vy) enemy velocity

    Returns:
        ((ix, iy), t) intercept point and time, or (None, None)
    """
    ox = enemy_pos[0] - our_pos[0]
    oy = enemy_pos[1] - our_pos[1]
    evx, evy = enemy_vel[0], enemy_vel[1]

    a = evx**2 + evy**2 - our_speed**2
    b = 2.0 * (ox * evx + oy * evy)
    c = ox**2 + oy**2

    t = None

    if abs(a) < 1e-10:
        # Linear case (equal speeds)
        if abs(b) < 1e-10:
            return (None, None)
        t = -c / b
    else:
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return (None, None)

        sqrt_d = math.sqrt(discriminant)
        t1 = (-b + sqrt_d) / (2 * a)
        t2 = (-b - sqrt_d) / (2 * a)

        candidates = [t for t in [t1, t2] if t > 0.001]
        if not candidates:
            return (None, None)
        t = min(candidates)

    if t is None or t < 0:
        return (None, None)

    ix = enemy_pos[0] + evx * t
    iy = enemy_pos[1] + evy * t

    return ((ix, iy), t)


def predict_with_walls(enemy_pos, enemy_vel, dt,
                       arena_min=(0, 0), arena_max=(244, 244)):
    """Predict enemy position accounting for wall bounces.

    Args:
        arena_min/max: arena bounds in cm
    """
    pred_x = enemy_pos[0] + enemy_vel[0] * dt
    pred_y = enemy_pos[1] + enemy_vel[1] * dt
    vel_x, vel_y = enemy_vel[0], enemy_vel[1]

    if pred_x < arena_min[0]:
        pred_x = 2 * arena_min[0] - pred_x
        vel_x = -vel_x
    elif pred_x > arena_max[0]:
        pred_x = 2 * arena_max[0] - pred_x
        vel_x = -vel_x

    if pred_y < arena_min[1]:
        pred_y = 2 * arena_min[1] - pred_y
        vel_y = -vel_y
    elif pred_y > arena_max[1]:
        pred_y = 2 * arena_max[1] - pred_y
        vel_y = -vel_y

    return (pred_x, pred_y), (vel_x, vel_y)


# ---------------------------------------------------------------------------
# Proportional Navigation
# ---------------------------------------------------------------------------

def proportional_navigation(our_pos, our_vel, enemy_pos, enemy_vel,
                            N=4.0, max_omega=10.0):
    """Compute steering command using Proportional Navigation.

    Args:
        our_pos:   (x, y) in cm
        our_vel:   (vx, vy) in cm/s
        enemy_pos: (x, y) in cm
        enemy_vel: (vx, vy) in cm/s
        N:         navigation constant (3-5, default 4)
        max_omega: max turn rate (rad/s)

    Returns:
        (throttle, steering) normalized to [-1, 1]
    """
    # Relative position and velocity
    Rx = enemy_pos[0] - our_pos[0]
    Ry = enemy_pos[1] - our_pos[1]
    Vx = enemy_vel[0] - our_vel[0]
    Vy = enemy_vel[1] - our_vel[1]

    R_sq = Rx**2 + Ry**2
    R_mag = math.sqrt(R_sq)

    if R_mag < 5.0:  # 5cm — we've hit them
        return (0.0, 0.0)

    # LOS rotation rate (2D cross product / range²)
    lambda_dot = (Rx * Vy - Ry * Vx) / R_sq

    # Closing velocity
    V_closing = -(Rx * Vx + Ry * Vy) / R_mag

    our_speed = math.sqrt(our_vel[0]**2 + our_vel[1]**2)

    if V_closing < 0.5:
        # Target moving away — fall back to pure pursuit heading
        desired = math.atan2(Ry, Rx)
        our_heading = math.atan2(our_vel[1], our_vel[0]) if our_speed > 1.0 else 0.0
        heading_err = _angle_diff(desired, our_heading)
        steering = max(-1.0, min(1.0, heading_err * 2.0))
        return (1.0, steering)

    # PN lateral acceleration
    a_n = N * lambda_dot * V_closing

    # Convert to angular velocity
    V_robot = max(our_speed, 5.0)  # avoid divide by zero
    omega = a_n / V_robot
    omega = max(-max_omega, min(max_omega, omega))

    # Normalize to steering [-1, 1]
    steering = omega / max_omega

    return (1.0, max(-1.0, min(1.0, steering)))


# ---------------------------------------------------------------------------
# Pure pursuit
# ---------------------------------------------------------------------------

def pure_pursuit(our_pos, our_heading_rad, target_pos):
    """Simple pure pursuit: steer toward target's current position.

    Returns:
        (throttle, steering) normalized to [-1, 1]
    """
    dx = target_pos[0] - our_pos[0]
    dy = target_pos[1] - our_pos[1]
    dist = math.hypot(dx, dy)

    if dist < 5.0:  # 5cm
        return (0.0, 0.0)

    desired = math.atan2(dy, dx)
    heading_err = _angle_diff(desired, our_heading_rad)

    steering = max(-1.0, min(1.0, heading_err * 1.5))
    throttle = min(1.0, dist / 30.0)  # proportional to distance

    return (throttle, steering)


# ---------------------------------------------------------------------------
# Smoothed intercept point (EMA)
# ---------------------------------------------------------------------------

class SmoothedIntercept:
    """EMA filter on intercept point to prevent jittery steering."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.smoothed = None

    def update(self, new_intercept):
        """Update with new intercept point. Returns smoothed point."""
        if new_intercept is None:
            return self.smoothed

        pt = np.array(new_intercept, dtype=np.float64)
        if self.smoothed is None:
            self.smoothed = pt
        else:
            self.smoothed = self.alpha * pt + (1 - self.alpha) * self.smoothed

        return self.smoothed

    def reset(self):
        self.smoothed = None


# ---------------------------------------------------------------------------
# Pursuit state machine
# ---------------------------------------------------------------------------

class PursuitState:
    SEARCH = "search"
    ACQUIRE = "acquire"
    INTERCEPT = "intercept"
    CLOSE = "close"
    LOST = "lost"


class PursuitFSM:
    """State machine for pursuit strategy selection.

    States:
        SEARCH:    No enemy detected — spin or patrol
        ACQUIRE:   Enemy detected, building velocity estimate (~10 frames)
        INTERCEPT: Full PN guidance active
        CLOSE:     Within striking distance — pure pursuit, max throttle
        LOST:      Had track, lost it — coast on Kalman prediction
    """

    CLOSE_RANGE_CM = 30.0       # switch to pure pursuit
    ACQUIRE_FRAMES = 10         # frames to build velocity estimate
    LOST_TIMEOUT_FRAMES = 15    # ~250ms at 60fps

    def __init__(self):
        self.state = PursuitState.SEARCH
        self._acquire_count = 0

    def update(self, enemy_detected, enemy_tracker):
        """Update state based on detection status and tracker state.

        Args:
            enemy_detected: bool, was enemy detected this frame
            enemy_tracker: EnemyTracker instance

        Returns:
            new state string
        """
        is_tracking = enemy_tracker.is_tracking
        frames_lost = enemy_tracker.kalman.frames_without_detection

        if not enemy_detected and frames_lost > self.LOST_TIMEOUT_FRAMES:
            self.state = PursuitState.SEARCH
            self._acquire_count = 0
            return self.state

        if not enemy_detected and is_tracking:
            self.state = PursuitState.LOST
            return self.state

        if enemy_detected:
            distance = np.linalg.norm(enemy_tracker.position_cm)
            # This is distance from origin — we need distance from US
            # Caller should provide our position for proper distance calc

            if self._acquire_count < self.ACQUIRE_FRAMES:
                self._acquire_count += 1
                self.state = PursuitState.ACQUIRE
                return self.state

            self.state = PursuitState.INTERCEPT
            return self.state

        return self.state

    def update_with_distance(self, enemy_detected, is_tracking,
                              frames_lost, distance_cm):
        """Update with explicit distance calculation.

        Args:
            distance_cm: distance from our robot to enemy
        """
        if not enemy_detected and frames_lost > self.LOST_TIMEOUT_FRAMES:
            self.state = PursuitState.SEARCH
            self._acquire_count = 0
            return self.state

        if not enemy_detected and is_tracking:
            self.state = PursuitState.LOST
            return self.state

        if enemy_detected:
            self._acquire_count = min(self._acquire_count + 1,
                                       self.ACQUIRE_FRAMES + 1)

            if distance_cm < self.CLOSE_RANGE_CM:
                self.state = PursuitState.CLOSE
            elif self._acquire_count >= self.ACQUIRE_FRAMES:
                self.state = PursuitState.INTERCEPT
            else:
                self.state = PursuitState.ACQUIRE

            return self.state

        return self.state

    def reset(self):
        self.state = PursuitState.SEARCH
        self._acquire_count = 0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _angle_diff(a, b):
    """Shortest signed angle from b to a, wrapped to [-pi, pi]."""
    d = a - b
    return (d + math.pi) % (2.0 * math.pi) - math.pi
