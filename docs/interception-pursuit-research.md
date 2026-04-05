# Interception & Pursuit Research: Practical Solutions

Research for B4B autonomous combat robot -- overhead camera tracking, enemy detection, smooth pursuit, and Kalman filter hardening.

## Table of Contents
1. [Enemy Detection Without Self-Interference](#1-enemy-detection-without-self-interference)
2. [Smooth Pursuit Curves for Tank Drive](#2-smooth-pursuit-curves-for-tank-drive)
3. [Kalman Filter Gating for Outlier Rejection](#3-kalman-filter-gating-for-outlier-rejection)
4. [OAK-D Depth for Object Detection](#4-oak-d-depth-for-object-detection)
5. [Concrete Code Changes for B4B](#5-concrete-code-changes-for-b4b)

---

## 1. Enemy Detection Without Self-Interference

### Problem Recap
When our robot moves fast, ArUco detection fails (motion blur), so we can't mask ourselves out of the reference diff. The enemy detection then latches onto our own robot's blob. When we get close to the enemy, the two blobs merge.

### Solution A: Color-Based Self-Identification (Primary Recommendation)

Put a bright, distinctive color marker on top of our robot (e.g., neon green tape, bright orange lid). Then detect ourselves via HSV color segmentation *in addition to* ArUco. When ArUco fails due to motion blur, the color mask still works because color detection is blur-resistant.

```python
import cv2
import numpy as np

class ColorSelfTracker:
    """Detect our robot by its distinctive color when ArUco fails.
    
    Put bright green tape/lid on top of our robot. HSV detection
    works even with motion blur because color survives blur.
    """
    
    def __init__(self):
        # Neon green in HSV (tune these with a trackbar tool)
        self.hsv_lower = np.array([35, 100, 100])
        self.hsv_upper = np.array([85, 255, 255])
        # Morphology kernels
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self._last_mask = None
    
    def detect(self, frame):
        """Returns (center_xy, bounding_radius) or (None, None).
        
        Works even under motion blur because color survives blur.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Morphology cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)
        self._last_mask = mask
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        
        # Largest green blob = our robot
        biggest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest)
        if area < 200:  # too small, noise
            return None, None
        
        ((cx, cy), radius) = cv2.minEnclosingCircle(biggest)
        return (cx, cy), radius
    
    def get_exclusion_mask(self, frame_shape, center, radius, expand=1.5):
        """Create a mask that blacks out our robot's region.
        
        Use this to exclude our robot from enemy detection.
        expand: multiply radius by this factor for safety margin.
        """
        mask = np.ones(frame_shape[:2], dtype=np.uint8) * 255
        if center is not None:
            cv2.circle(mask, (int(center[0]), int(center[1])),
                      int(radius * expand), 0, -1)
        return mask
```

**Integration with existing enemy_tracker.py:**

```python
class EnemyDetector:
    def __init__(self, ...):
        # ... existing init ...
        self._color_self = ColorSelfTracker()
    
    def detect(self, frame, our_robot_corners=None):
        # Try ArUco corners first, fall back to color detection
        exclusion_center = None
        exclusion_radius = 150  # default
        
        if our_robot_corners is not None:
            exclusion_center = our_robot_corners.mean(axis=0).astype(int)
        else:
            # ArUco failed (motion blur) -- use color fallback
            color_center, color_radius = self._color_self.detect(frame)
            if color_center is not None:
                exclusion_center = np.array([int(color_center[0]), int(color_center[1])])
                exclusion_radius = int(color_radius * 1.5)
        
        # ... rest of detection uses exclusion_center/exclusion_radius ...
```

**Why this works:** Color survives motion blur. A neon green lid on your robot will still show as a green blob even at high speed. ArUco markers have sharp edges that blur destroys, but a solid color patch is robust to blur.

### Solution B: SORT-Style Multi-Object Tracking with Assignment

Instead of detecting "the enemy" directly, detect ALL moving objects and use track assignment to figure out which is which. The SORT algorithm (Simple Online Realtime Tracking) does exactly this:

1. Detect all foreground blobs (both robots)
2. Kalman filter predicts where each tracked object should be
3. Hungarian algorithm matches detections to tracks by distance
4. Track IDs persist across frames

This solves the merge problem because even when blobs merge, the Kalman prediction keeps the two tracks separate, and when they separate again the assignment recovers.

Reference implementation: [abewley/sort](https://github.com/abewley/sort) -- runs at 260 Hz, uses `filterpy.kalman.KalmanFilter` and `scipy.optimize.linear_sum_assignment`.

**Lightweight SORT adapted for our 2-robot arena (centroid-based, not bbox-based):**

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

class CentroidKalmanTracker:
    """Kalman tracker for a single object using centroid position."""
    _id_counter = 0
    
    def __init__(self, centroid, dt=1/60):
        self.id = CentroidKalmanTracker._id_counter
        CentroidKalmanTracker._id_counter += 1
        
        # State: [x, y, vx, vy]
        self.x = np.array([centroid[0], centroid[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 100.0, 100.0])
        
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        
        sigma_a = 500.0  # px/s^2 -- combat robots accelerate hard
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        self.Q = np.array([
            [dt4/4, 0, dt3/2, 0],
            [0, dt4/4, 0, dt3/2],
            [dt3/2, 0, dt2, 0],
            [0, dt3/2, 0, dt2],
        ]) * sigma_a**2
        
        self.R = np.eye(2) * 5.0**2  # measurement noise: 5px
        
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
    
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.x[:2]
    
    def update(self, centroid):
        z = np.array(centroid, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.hits += 1
        self.time_since_update = 0
    
    @property
    def position(self):
        return self.x[:2]
    
    @property
    def velocity(self):
        return self.x[2:]


class MultiObjectTracker:
    """SORT-style tracker for the 2-robot arena.
    
    Tracks all foreground blobs, assigns persistent IDs.
    Use track assignment to determine which is our robot vs enemy.
    """
    
    def __init__(self, max_lost=30, dt=1/60):
        self.trackers = []
        self.max_lost = max_lost  # frames before dropping a track
        self.dt = dt
        # Gate: max distance (px) for a detection to match a track
        self.gate_distance = 120.0
    
    def update(self, detections):
        """Update with list of centroid detections [(x,y), ...].
        
        Returns list of (track_id, position, velocity) for active tracks.
        """
        # 1. Predict all existing tracks
        predictions = []
        for trk in self.trackers:
            predictions.append(trk.predict())
        
        # 2. Build cost matrix (Euclidean distance)
        if len(predictions) > 0 and len(detections) > 0:
            cost = np.zeros((len(detections), len(predictions)))
            for d, det in enumerate(detections):
                for t, pred in enumerate(predictions):
                    cost[d, t] = np.linalg.norm(np.array(det) - pred)
            
            # 3. Hungarian assignment
            row_idx, col_idx = linear_sum_assignment(cost)
            
            matched_dets = set()
            matched_trks = set()
            
            for d, t in zip(row_idx, col_idx):
                if cost[d, t] < self.gate_distance:
                    self.trackers[t].update(detections[d])
                    matched_dets.add(d)
                    matched_trks.add(t)
            
            # 4. Create new tracks for unmatched detections
            for d, det in enumerate(detections):
                if d not in matched_dets:
                    self.trackers.append(
                        CentroidKalmanTracker(det, dt=self.dt)
                    )
        elif len(detections) > 0:
            # No existing tracks -- create all new
            for det in detections:
                self.trackers.append(
                    CentroidKalmanTracker(det, dt=self.dt)
                )
        
        # 5. Remove dead tracks
        self.trackers = [t for t in self.trackers
                        if t.time_since_update <= self.max_lost]
        
        # 6. Return active tracks
        results = []
        for trk in self.trackers:
            if trk.hits >= 3 or trk.time_since_update == 0:
                results.append((trk.id, trk.position.copy(), trk.velocity.copy()))
        
        return results
    
    def identify_robots(self, tracks, our_aruco_pos=None, our_color_pos=None):
        """Given tracks and our known position, identify which track is us vs enemy.
        
        Returns (our_track_id, enemy_track_id) or (None, None).
        """
        if not tracks:
            return None, None
        
        our_pos = our_aruco_pos if our_aruco_pos is not None else our_color_pos
        if our_pos is None:
            return None, None
        
        our_pos = np.array(our_pos)
        
        # Find track closest to our known position
        best_dist = float('inf')
        our_id = None
        for tid, pos, vel in tracks:
            dist = np.linalg.norm(pos - our_pos)
            if dist < best_dist:
                best_dist = dist
                our_id = tid
        
        # Enemy = the other track
        enemy_id = None
        for tid, pos, vel in tracks:
            if tid != our_id:
                enemy_id = tid
                break
        
        return our_id, enemy_id
```

### Solution C: Blob Merge Handling with Predicted Separation

When our robot gets close to the enemy, the two blobs merge into one. Instead of trying to separate them, accept the merge and use predictions:

```python
def handle_merged_blob(merged_centroid, our_predicted_pos, enemy_predicted_pos):
    """When blobs merge, estimate individual positions from the merged blob.
    
    The merged centroid is roughly the area-weighted average of the two robots.
    Use Kalman predictions to estimate where each robot is within the merged blob.
    """
    separation = np.linalg.norm(our_predicted_pos - enemy_predicted_pos)
    
    if separation < 50:  # pixels -- robots are very close
        # Trust Kalman predictions, don't update with merged measurement
        # Just coast both tracks on prediction alone
        return None, None  # signal: don't update either track
    
    # If somewhat separated, assign merged centroid to nearest track
    d_our = np.linalg.norm(merged_centroid - our_predicted_pos)
    d_enemy = np.linalg.norm(merged_centroid - enemy_predicted_pos)
    
    if d_our < d_enemy:
        return merged_centroid, None  # update our track, coast enemy
    else:
        return None, merged_centroid  # coast ours, update enemy
```

### Recommendation

**Use Solution A (color) + Solution B (SORT) together.** Put neon green tape on your robot. The color tracker gives you self-position even during motion blur. SORT gives you persistent track IDs that survive brief occlusions and merges. The combination solves all three failure modes:

1. **Motion blur kills ArUco** -> color fallback masks our robot
2. **Blobs merge** -> SORT's Kalman predictions coast through the merge
3. **Detection teleports** -> SORT's Hungarian assignment + gating rejects impossible jumps

---

## 2. Smooth Pursuit Curves for Tank Drive

### Problem Recap
Current behavior: turn in place until facing target, drive straight, repeat. This creates a jerky stop-start pattern because the turn-in-place threshold is 90 degrees and there's no simultaneous turn+drive for smaller angles.

### Solution: Pure Pursuit with Continuous Arc Driving

Pure pursuit computes a circular arc from your current position to a lookahead point, then converts that arc to differential wheel speeds. There is NO stopping. The robot always drives forward while steering.

**References:**
- [CMU original paper (Coulter 1992)](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)
- [FRC Team 1712 Python implementation](https://github.com/arimb/PurePursuit)
- [PythonRobotics pure_pursuit.py](https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/pure_pursuit/pure_pursuit.py)
- [Purdue SIGBots Wiki](https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit)

**The core math for tank/differential drive:**

```
Given:
  - Robot at (rx, ry) facing heading theta
  - Lookahead point at (gx, gy)
  - Lookahead distance L_d = distance to goal point
  - Robot wheelbase (track width) W

1. Compute angle to goal relative to robot heading:
   alpha = atan2(gy - ry, gx - rx) - theta

2. Compute signed curvature of the arc:
   kappa = 2 * sin(alpha) / L_d

3. Compute angular velocity from curvature:
   omega = v * kappa    (where v = forward velocity)

4. Compute wheel speeds:
   v_left  = v - omega * W / 2
   v_right = v + omega * W / 2
```

From the Purdue SIGBots wiki, for a differential drive robot the turn velocity is:
```
turnVel = (W * sin(alpha) / L_d) * linearVel
left_speed  = linearVel - turnVel
right_speed = linearVel + turnVel
```

**Complete implementation for our combat robot:**

```python
import math
import numpy as np


class PurePursuitController:
    """Smooth arc-based pursuit for tank/differential drive robots.
    
    Instead of turn-stop-drive, this computes a continuous arc from
    the robot's current pose to a lookahead point. The robot always
    drives forward while steering -- no stopping.
    
    For combat: the "lookahead point" is the intercept point or
    the enemy's predicted position.
    """
    
    def __init__(self, 
                 wheelbase_cm=15.0,      # track width of our robot
                 min_lookahead_cm=15.0,   # minimum lookahead distance
                 max_lookahead_cm=60.0,   # maximum lookahead distance
                 lookahead_gain=0.5,      # lookahead = gain * speed + min
                 max_speed=1.0):
        self.wheelbase = wheelbase_cm
        self.min_lookahead = min_lookahead_cm
        self.max_lookahead = max_lookahead_cm
        self.lookahead_gain = lookahead_gain
        self.max_speed = max_speed
    
    def compute(self, our_pos, our_heading_rad, our_speed, target_pos):
        """Compute throttle and steering for smooth arc toward target.
        
        Args:
            our_pos: (x, y) in cm
            our_heading_rad: robot heading in radians
            our_speed: current speed in cm/s (for adaptive lookahead)
            target_pos: (x, y) target point in cm
        
        Returns:
            (throttle, steering) in [-1, 1] range
            throttle: forward/back
            steering: negative=left, positive=right
        """
        dx = target_pos[0] - our_pos[0]
        dy = target_pos[1] - our_pos[1]
        distance = math.hypot(dx, dy)
        
        if distance < 3.0:  # 3cm -- we've arrived
            return 0.0, 0.0
        
        # Adaptive lookahead: faster = look further ahead = smoother arcs
        lookahead = self.min_lookahead + self.lookahead_gain * abs(our_speed)
        lookahead = min(lookahead, self.max_lookahead)
        # Don't look past the target
        lookahead = min(lookahead, distance)
        
        # Angle to target relative to our heading
        angle_to_target = math.atan2(dy, dx)
        alpha = _angle_wrap(angle_to_target - our_heading_rad)
        
        # --- Core pure pursuit: signed curvature ---
        # kappa = 2 * sin(alpha) / L_d
        # This is the curvature of the arc from us to the lookahead point
        if lookahead > 0.01:
            curvature = 2.0 * math.sin(alpha) / lookahead
        else:
            curvature = 0.0
        
        # --- Convert curvature to throttle + steering ---
        
        # Throttle: proportional to distance, reduced when turning sharp
        # Sharp turns (high curvature) need lower speed for stability
        sharpness = abs(curvature) * self.wheelbase  # dimensionless turn intensity
        speed_factor = 1.0 / (1.0 + 2.0 * sharpness)  # smooth reduction
        
        throttle = self.max_speed * speed_factor
        # Scale down near target
        if distance < 20.0:
            throttle *= distance / 20.0
        throttle = max(0.1, min(self.max_speed, throttle))
        
        # Steering: from curvature
        # For tank drive: steering = omega / max_omega
        # omega = v * kappa, but we normalize to [-1, 1]
        # At max curvature, one wheel stops and the other goes full
        # That happens when omega * W/2 = v, so kappa_max = 2/W
        kappa_max = 2.0 / self.wheelbase
        steering = curvature / kappa_max
        steering = max(-1.0, min(1.0, steering))
        
        return throttle, steering
    
    def compute_wheel_speeds(self, our_pos, our_heading_rad, our_speed, target_pos):
        """Compute individual left/right wheel speeds (alternative output).
        
        Returns:
            (v_left, v_right) normalized to [-1, 1]
        """
        dx = target_pos[0] - our_pos[0]
        dy = target_pos[1] - our_pos[1]
        distance = math.hypot(dx, dy)
        
        if distance < 3.0:
            return 0.0, 0.0
        
        lookahead = self.min_lookahead + self.lookahead_gain * abs(our_speed)
        lookahead = min(lookahead, self.max_lookahead, distance)
        
        angle_to_target = math.atan2(dy, dx)
        alpha = _angle_wrap(angle_to_target - our_heading_rad)
        
        curvature = 2.0 * math.sin(alpha) / max(lookahead, 0.01)
        
        # Forward velocity (reduce for sharp turns)
        v = self.max_speed / (1.0 + 2.0 * abs(curvature) * self.wheelbase)
        if distance < 20.0:
            v *= distance / 20.0
        
        # Differential wheel speeds
        omega = v * curvature
        v_left = v - omega * self.wheelbase / 2.0
        v_right = v + omega * self.wheelbase / 2.0
        
        # Normalize so neither wheel exceeds [-1, 1]
        max_v = max(abs(v_left), abs(v_right), 0.01)
        if max_v > 1.0:
            v_left /= max_v
            v_right /= max_v
        
        return v_left, v_right


class CombatPursuitController:
    """High-level pursuit controller combining intercept + pure pursuit.
    
    Replaces the turn-stop-drive pattern with continuous arc driving.
    Uses proportional navigation at range, pure pursuit when close.
    """
    
    def __init__(self, wheelbase_cm=15.0):
        self.pursuit = PurePursuitController(wheelbase_cm=wheelbase_cm)
        self._smoothed_target = None
        self._alpha = 0.3  # EMA smoothing for target point
        
        # PN parameters
        self.N = 4.0  # navigation constant
        self.pn_range_threshold = 50.0  # cm -- switch to pure pursuit below this
    
    def compute_command(self, our_pos, our_heading, our_vel,
                        enemy_pos, enemy_vel, our_speed_max=100.0):
        """Compute smooth pursuit command.
        
        At long range: uses proportional navigation to compute an intercept
        point, then does pure pursuit arc toward that point.
        
        At close range: pure pursuit directly toward enemy position.
        
        Returns (throttle, steering) in [-1, 1].
        """
        our_pos = np.array(our_pos, dtype=np.float64)
        enemy_pos = np.array(enemy_pos, dtype=np.float64)
        enemy_vel = np.array(enemy_vel, dtype=np.float64)
        
        distance = np.linalg.norm(enemy_pos - our_pos)
        enemy_speed = np.linalg.norm(enemy_vel)
        
        # Choose target point
        if distance > self.pn_range_threshold and enemy_speed > 5.0:
            # Long range + enemy moving: intercept point
            target = self._compute_intercept(our_pos, our_speed_max,
                                              enemy_pos, enemy_vel)
        else:
            # Close range or enemy stationary: go directly to enemy
            target = enemy_pos
        
        # Smooth the target point (prevents jitter)
        if self._smoothed_target is None:
            self._smoothed_target = target.copy()
        else:
            self._smoothed_target = (self._alpha * target + 
                                      (1 - self._alpha) * self._smoothed_target)
        
        # Our current speed estimate (from velocity if available)
        our_speed = np.linalg.norm(our_vel) if our_vel is not None else 0.0
        
        # Pure pursuit arc toward smoothed target
        return self.pursuit.compute(
            our_pos, our_heading, our_speed, self._smoothed_target
        )
    
    def _compute_intercept(self, our_pos, our_speed, enemy_pos, enemy_vel):
        """Compute intercept point using quadratic solver."""
        ox = enemy_pos[0] - our_pos[0]
        oy = enemy_pos[1] - our_pos[1]
        evx, evy = enemy_vel[0], enemy_vel[1]
        
        a = evx**2 + evy**2 - our_speed**2
        b = 2.0 * (ox * evx + oy * evy)
        c = ox**2 + oy**2
        
        if abs(a) < 1e-10:
            if abs(b) < 1e-10:
                return enemy_pos.copy()
            t = -c / b
        else:
            disc = b**2 - 4*a*c
            if disc < 0:
                return enemy_pos.copy()
            sqrt_d = math.sqrt(disc)
            t1 = (-b + sqrt_d) / (2*a)
            t2 = (-b - sqrt_d) / (2*a)
            candidates = [t for t in [t1, t2] if t > 0.001]
            if not candidates:
                return enemy_pos.copy()
            t = min(candidates)
        
        if t < 0:
            return enemy_pos.copy()
        
        # Clamp intercept time to prevent chasing far-future predictions
        t = min(t, 2.0)  # max 2 seconds ahead
        
        return np.array([
            enemy_pos[0] + evx * t,
            enemy_pos[1] + evy * t,
        ])
    
    def reset(self):
        self._smoothed_target = None


def _angle_wrap(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
```

### Key Differences from Current Code

| Current `PathFollower` | New `PurePursuitController` |
|---|---|
| `turn_in_place_threshold = 1.57 rad` (90 deg) -- stops and rotates for large angles | Never stops. Drives a wide arc for large angles, tight arc for small angles |
| Separate turn phase and drive phase | Single continuous command: always throttle > 0 with steering |
| Heading PID with slew limiting | Geometric curvature computation -- no PID tuning needed |
| Fixed throttle (1.0 or proportional near target) | Speed adapts to curvature -- slows for sharp turns automatically |
| Jittery: constant heading corrections | Smooth: curvature is continuous function of position |

### The "Carrot on a Stick" Principle

The lookahead distance is the key parameter:
- **Short lookahead (15cm):** Robot makes tight corrections, follows path precisely but can oscillate
- **Long lookahead (60cm):** Robot makes wide gentle arcs, very smooth but cuts corners
- **Adaptive lookahead:** `L = k * speed + L_min` -- faster = look further = smoother curves

For combat: use short lookahead (aggressive tracking) when close, long lookahead (smooth approach) when far.

### Why Not Just Lower the Turn-in-Place Threshold?

Setting `turn_in_place_threshold_rad` to a smaller value (e.g., 0.3 rad / 17 degrees) would make the robot drive more and stop less, but it still has a discontinuity: below the threshold it drives while steering, above it stops and rotates. Pure pursuit has no discontinuity -- the curvature smoothly increases as the angle increases. At 180 degrees behind, the curvature is maximal (tightest turn) but there is still forward motion.

---

## 3. Kalman Filter Gating for Outlier Rejection

### Problem Recap
When enemy detection teleports (our robot detected as enemy, or detection jumps to noise), the Kalman filter follows because it trusts measurements. The existing NIS check (line 89-95 in enemy_tracker.py) boosts process noise when NIS > 6.0, but that makes the filter MORE responsive to outliers, not less.

### Solution: Mahalanobis Distance Gating (Reject Before Update)

The fix is simple: compute the Mahalanobis distance (normalized innovation squared, NIS) BEFORE the Kalman update. If it's too large, REJECT the measurement entirely. Don't update. Just coast on prediction.

This is standard practice in aerospace tracking (missile guidance, radar tracking) and is used in SORT, DeepSORT, and the Ultralytics tracker. Reference: [Georgia Tech Robust KF paper](https://acds-lab.gatech.edu/papers/IROS2007_RobustKF.pdf), [Ultralytics gating_distance](https://docs.ultralytics.com/reference/trackers/utils/kalman_filter/).

**Chi-squared thresholds for 2D measurements (position x,y):**

| Confidence | Chi-squared threshold | Meaning |
|---|---|---|
| 95% | 5.99 | Rejects 5% of valid measurements (conservative) |
| 99% | 9.21 | Rejects 1% of valid (good default) |
| 99.5% | 10.60 | Very permissive -- only rejects extreme outliers |

**Updated EnemyKalmanFilter with proper gating:**

```python
class EnemyKalmanFilter:
    """4-state Kalman filter with Mahalanobis distance gating.
    
    Rejects measurements that are statistically impossible given
    the current state estimate. Prevents teleporting detections
    from corrupting the track.
    """
    
    def __init__(self, dt=1/60, sigma_a=5.0, sigma_meas=0.01):
        # ... same init as before ...
        
        # Gating threshold: chi-squared with 2 DOF
        # 9.21 = 99% confidence -- rejects measurements that are 
        # >99% unlikely given current state
        self.gate_threshold = 9.21
        
        # Consecutive rejections counter
        self._consecutive_rejects = 0
        self._max_consecutive_rejects = 10  # after this, maybe we're lost
    
    def update(self, measurement):
        """Gated update -- rejects outlier measurements."""
        if measurement is not None:
            z = np.array(measurement, dtype=np.float64)
            
            if not self._initialized:
                self.x[:2] = z
                self.x[2:] = 0.0
                self._initialized = True
                self.frames_without_detection = 0
                self._consecutive_rejects = 0
                return
            
            # --- GATING: Reject before update ---
            innovation = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            S_inv = np.linalg.inv(S)
            
            # Mahalanobis distance squared (= NIS for Kalman filter)
            mahal_sq = float(innovation.T @ S_inv @ innovation)
            
            if mahal_sq > self.gate_threshold:
                # Measurement is an outlier -- REJECT, don't update
                self._consecutive_rejects += 1
                self.frames_without_detection += 1
                
                if self._consecutive_rejects > self._max_consecutive_rejects:
                    # We've been rejecting too long -- maybe the track is wrong
                    # Reset and re-acquire on this measurement
                    self.x[:2] = z
                    self.x[2:] = 0.0
                    self.P = np.eye(4) * 100.0
                    self._consecutive_rejects = 0
                    self.frames_without_detection = 0
                
                return  # Don't update filter
            
            # --- ACCEPTED: Normal Kalman update ---
            K = self.P @ self.H.T @ S_inv
            self.x = self.x + K @ innovation
            self.P = (np.eye(4) - K @ self.H) @ self.P
            
            self._consecutive_rejects = 0
            self.frames_without_detection = 0
            
            # Adaptive Q (only for accepted measurements)
            if mahal_sq > 4.0:
                self.Q = self.Q_baseline * 2.0  # mild boost for maneuvering
            elif mahal_sq < 1.0:
                self.Q = self.Q * 0.95 + self.Q_baseline * 0.05
        else:
            self.frames_without_detection += 1
            self._consecutive_rejects = 0  # no measurement isn't a reject
```

### What Changed from Current Code

The current code (line 82-95 in enemy_tracker.py) does this:
```python
# CURRENT (BUG): Update first, THEN check NIS
y = z - self.H @ self.x
S = self.H @ self.P @ self.H.T + self.R
K = self.P @ self.H.T @ np.linalg.inv(S)
self.x = self.x + K @ y          # <-- already corrupted by outlier!
self.P = (np.eye(4) - K @ self.H) @ self.P
nis = y.T @ np.linalg.inv(S) @ y
if nis > 6.0:
    self.Q = self.Q_baseline * 4.0  # <-- makes it WORSE next frame
```

The fix:
1. Compute innovation and NIS **BEFORE** the update
2. If NIS > threshold, **skip the update entirely** (coast on prediction)
3. Only increase Q mildly for accepted measurements that show maneuvering
4. After too many consecutive rejections, reset the track (we're probably lost)

### Complementary: Max Jump Distance (Already Exists)

Your existing `MAX_JUMP_PX = 80` in the detector (line 302) is a good first line of defense in pixel space. The Kalman gating adds a second line of defense in world-coordinate space that accounts for velocity and uncertainty.

---

## 4. OAK-D Depth for Object Detection

### Why Depth Solves Your Problems

The OAK-D Pro has stereo depth cameras. An overhead depth camera sees the arena floor at a known depth, and robots are 2-4 inches (~5-10cm) tall. This means:

1. **Floor subtraction via depth:** Any pixel significantly closer to the camera than the floor is a robot. No reference frame needed, immune to lighting changes, works whether objects are moving or stationary.
2. **Self vs enemy separation:** Even when blobs overlap in 2D, they're at different (x,y) positions in 3D space.
3. **No ArUco dependency for detection:** Depth-based detection doesn't care about markers at all.

### Implementation: Depth-Based Floor Subtraction

References:
- [oakmower project](https://alemamm.github.io/oakmower/) -- OAK-D obstacle detection using RANSAC floor plane segmentation
- [pyRANSAC-3D](https://github.com/leomariga/pyRANSAC-3D) -- `pip install pyransac3d`, fits planes to point clouds
- [DepthAI StereoDepth docs](https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/)
- [DepthAI Spatial Location Calculator](https://oak-web.readthedocs.io/en/stable/samples/SpatialDetection/spatial_location_calculator/)

```python
import depthai as dai
import numpy as np
import cv2


def create_depth_pipeline():
    """Create DepthAI pipeline for overhead depth-based object detection."""
    pipeline = dai.Pipeline()
    
    # Stereo cameras
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    
    # Also get RGB for ArUco + visualization
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    
    # Outputs
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    xout_rgb.setStreamName("rgb")
    
    # Configure mono cameras
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setCamera("left")
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setCamera("right")
    
    # Configure stereo depth
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align to RGB
    stereo.setOutputSize(640, 400)
    
    # Depth range limits for overhead arena viewing
    stereo.setDepthLowerThreshold(500)   # 0.5m min
    stereo.setDepthUpperThreshold(3000)  # 3.0m max
    
    # Filters for cleaner depth
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    config = stereo.initialConfig.get()
    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.speckleFilter.speckleRange = 50
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.enable = True
    stereo.initialConfig.set(config)
    
    # Configure RGB
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(640, 400)
    cam_rgb.setInterleaved(False)
    
    # Links
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    stereo.depth.link(xout_depth.input)
    cam_rgb.preview.link(xout_rgb.input)
    
    return pipeline


class DepthFloorDetector:
    """Detect objects above the floor plane using depth data.
    
    For overhead camera: the floor is the FARTHEST surface.
    Robots are CLOSER to the camera (lower depth values).
    
    Steps:
    1. Calibrate floor depth (average depth of empty arena)
    2. Each frame: threshold depth to find pixels significantly
       closer than the floor
    3. Contour detection on the thresholded mask
    """
    
    def __init__(self, height_threshold_mm=40, min_area=200):
        """
        Args:
            height_threshold_mm: objects must be at least this many mm
                above the floor to be detected. 40mm = ~1.5 inches,
                well below any robot's height.
            min_area: minimum contour area in pixels
        """
        self.height_threshold_mm = height_threshold_mm
        self.min_area = min_area
        self.floor_depth = None  # median floor depth in mm
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    def calibrate_floor(self, depth_frame):
        """Calibrate with empty arena. Call once at startup.
        
        Args:
            depth_frame: uint16 depth image from OAK-D (values in mm)
        """
        valid = depth_frame[depth_frame > 0]
        if len(valid) == 0:
            print("[depth] WARNING: no valid depth pixels for calibration")
            return
        
        self.floor_depth = np.median(valid)
        print(f"[depth] Floor calibrated at {self.floor_depth:.0f}mm "
              f"({self.floor_depth/25.4:.1f} inches)")
    
    def detect(self, depth_frame, arena_mask=None):
        """Detect objects above the floor.
        
        Args:
            depth_frame: uint16 depth from OAK-D (mm)
            arena_mask: optional uint8 mask (255=inside arena)
        
        Returns:
            list of dicts with centroid, area, bbox, depth_mm, height_mm
        """
        if self.floor_depth is None:
            return []
        
        # Objects are CLOSER to camera than the floor
        # depth_frame < floor_depth - threshold means "above the floor"
        valid_mask = depth_frame > 0
        above_floor = depth_frame < (self.floor_depth - self.height_threshold_mm)
        
        object_mask = (valid_mask & above_floor).astype(np.uint8) * 255
        
        if arena_mask is not None:
            object_mask = cv2.bitwise_and(object_mask, arena_mask)
        
        # Morphology cleanup
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, self._kernel)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, self._kernel)
        
        contours, _ = cv2.findContours(
            object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                bbox = cv2.boundingRect(c)
                
                x, y, w, h = bbox
                roi_depth = depth_frame[y:y+h, x:x+w]
                valid_roi = roi_depth[roi_depth > 0]
                obj_depth = np.median(valid_roi) if len(valid_roi) > 0 else 0
                
                detections.append({
                    'centroid': (cx, cy),
                    'area': area,
                    'bbox': bbox,
                    'depth_mm': obj_depth,
                    'height_mm': self.floor_depth - obj_depth,
                })
        
        return detections
```

### Practical OAK-D Notes

- **Camera height:** For an 8x8ft arena, camera at ~6-8ft height gives good depth resolution. At 6ft (~1800mm), a 3-inch (~75mm) tall robot is at depth ~1725mm. The 75mm difference is easily detectable.
- **Depth resolution:** OAK-D stereo at 1-2m range has ~2-5mm depth resolution. More than enough.
- **Frame alignment:** Use `stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)` so depth and RGB pixel coordinates match.
- **Temporal filter:** Smooth depth noise across frames, reduces flickering detections.
- **Invalid pixels:** Depth is 0 for invalid/out-of-range. Always filter these out.
- **Alternative to RANSAC:** For a flat overhead view, simple median depth calibration works. RANSAC is needed when the camera is tilted and sees the floor at varying distances. Since our camera looks straight down, the floor is at a nearly constant depth.

### Depth + RGB Fusion Pipeline

The most robust approach combines depth (finding objects) with RGB (ArUco tracking, color ID):

```python
class FusedDetector:
    """Combines depth-based and RGB-based detection.
    
    Depth finds objects above the floor (immune to reference frame issues).
    RGB provides ArUco tracking and color identification.
    """
    
    def __init__(self):
        self.depth_detector = DepthFloorDetector(height_threshold_mm=40)
        self.color_self = ColorSelfTracker()
        self.multi_tracker = MultiObjectTracker()
    
    def update(self, rgb_frame, depth_frame, aruco_corners=None):
        """Full detection pipeline.
        
        Returns:
            our_pos: (x, y) in pixels or None
            enemy_pos: (x, y) in pixels or None
            enemy_vel: (vx, vy) in pixels/frame or None
        """
        # 1. Depth-based detection (finds ALL objects above floor)
        depth_detections = self.depth_detector.detect(depth_frame)
        centroids = [d['centroid'] for d in depth_detections]
        
        # 2. Multi-object tracking (assigns persistent IDs)
        tracks = self.multi_tracker.update(centroids)
        
        # 3. Identify our robot (ArUco preferred, color fallback)
        our_pos_known = None
        if aruco_corners is not None:
            our_pos_known = aruco_corners.mean(axis=0)
        else:
            color_center, _ = self.color_self.detect(rgb_frame)
            if color_center is not None:
                our_pos_known = np.array(color_center)
        
        # 4. Assign tracks
        our_id, enemy_id = self.multi_tracker.identify_robots(
            tracks, our_aruco_pos=our_pos_known
        )
        
        our_pos = None
        enemy_pos = None
        enemy_vel = None
        
        for tid, pos, vel in tracks:
            if tid == our_id:
                our_pos = pos
            elif tid == enemy_id:
                enemy_pos = pos
                enemy_vel = vel
        
        return our_pos, enemy_pos, enemy_vel
```

---

## 5. Concrete Code Changes for B4B

### Priority Order

1. **Kalman gating (30 minutes):** Change `enemy_tracker.py` `EnemyKalmanFilter.update()` to gate before updating. This is the single highest-impact fix -- stops detection teleportation from corrupting the track. See Section 3.

2. **Pure pursuit controller (1-2 hours):** Replace the turn-stop-drive PathFollower with the PurePursuitController in `autonomy.py`. Wire it into the intercept mode in `main.py`. See Section 2.

3. **Color self-identification (1 hour):** Add neon green tape to robot, add `ColorSelfTracker` class, use as fallback in `EnemyDetector.detect()` when `our_robot_corners` is None. See Section 1, Solution A.

4. **SORT multi-tracker (2-3 hours):** Replace single-object `EnemyDetector` with `MultiObjectTracker` + `identify_robots()`. This is the most architectural change but gives the most robustness. See Section 1, Solution B.

5. **OAK-D depth detection (half day):** Switch from USB webcam to OAK-D, add depth pipeline, replace reference-frame diffing with depth floor subtraction. This is the production solution. See Section 4.

### File Changes Summary

| File | Change |
|---|---|
| `enemy_tracker.py` | Add Mahalanobis gating to `EnemyKalmanFilter.update()` |
| `enemy_tracker.py` | Add `ColorSelfTracker` class, integrate into `EnemyDetector.detect()` |
| `enemy_tracker.py` | (Later) Replace with `MultiObjectTracker` |
| `autonomy.py` | Add `PurePursuitController` and `CombatPursuitController` classes |
| `intercept.py` | Wire `CombatPursuitController` into pursuit FSM |
| `main.py` | Replace PathFollower calls with PurePursuitController |
| (new) `depth_detector.py` | OAK-D depth pipeline + DepthFloorDetector |

---

## Sources

- [SORT: Simple Online Realtime Tracking (abewley/sort)](https://github.com/abewley/sort)
- [SORT Explained (Roboflow)](https://blog.roboflow.com/sort-explained-real-time-object-tracking-in-python/)
- [deep-sort-realtime PyPI](https://pypi.org/project/deep-sort-realtime/)
- [Pure Pursuit - Algorithms for Automated Driving](https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/PurePursuit.html)
- [Pure Pursuit - Purdue SIGBots Wiki](https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit)
- [Pure Pursuit - CMU original paper (Coulter 1992)](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)
- [FRC Team 1712 Pure Pursuit (Python)](https://github.com/arimb/PurePursuit)
- [PythonRobotics Pure Pursuit](https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/pure_pursuit/pure_pursuit.py)
- [Differential Drive Pure Pursuit Simulation](https://github.com/SamShue/Pure-Pursuit)
- [FRC Team 1712 Pure Pursuit Wiki](https://github.com/Dawgma-1712/FRC-2018/wiki/pure-pursuit)
- [Proportional Navigation (Python)](https://github.com/iwishiwasaneagle/proportional_navigation)
- [propNav - 3DOF PN Missile Sim](https://github.com/gedeschaines/propNav)
- [Robust Kalman Filter for Outlier Detection (Georgia Tech)](https://acds-lab.gatech.edu/papers/IROS2007_RobustKF.pdf)
- [Robust Kalman Filtering via Mahalanobis Distance (Springer)](https://link.springer.com/article/10.1007/s00190-013-0690-8)
- [Ultralytics Kalman Filter with Gating Distance](https://docs.ultralytics.com/reference/trackers/utils/kalman_filter/)
- [filterpy - Python Kalman Filter Library](https://github.com/rlabbe/filterpy)
- [oakmower - OAK-D Obstacle Detection](https://alemamm.github.io/oakmower/)
- [DepthAI Spatial Location Calculator](https://oak-web.readthedocs.io/en/stable/samples/SpatialDetection/spatial_location_calculator/)
- [DepthAI StereoDepth Node](https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/)
- [pyRANSAC-3D](https://github.com/leomariga/pyRANSAC-3D)
- [Detect Planes from Depth Image (ROS)](https://github.com/felixchenfy/ros_detect_planes_from_depth_img)
- [NU-MSR Overhead Mobile Tracker](https://github.com/NU-MSR/overhead_mobile_tracker)
- [Three Methods of Lateral Control (Pure Pursuit, Stanley, MPC)](https://dingyan89.medium.com/three-methods-of-vehicle-lateral-control-pure-pursuit-stanley-and-mpc-db8cc1d32081)
