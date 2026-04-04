# Moving Object Interception & Pursuit Research

**Project**: Brick for Brains - Autonomous Combat Robot
**Context**: Differential-drive beetleweight in 8x8ft arena, ArUco-tracked (own robot), CV-detected enemy (no markers), overhead camera
**Goal**: Intercept and ram enemy robot with <30ms decision latency at 60 FPS

---

## Table of Contents

1. [Proportional Navigation Guidance](#1-proportional-navigation-guidance)
2. [Intercept Point Calculation](#2-intercept-point-calculation)
3. [Kalman Filter for Enemy Tracking](#3-kalman-filter-for-enemy-tracking)
4. [Pure Pursuit vs Lead Pursuit](#4-pure-pursuit-vs-lead-pursuit)
5. [Enemy Detection Without Markers](#5-enemy-detection-without-markers)
6. [Prediction Under Uncertainty](#6-prediction-under-uncertainty)
7. [Recommended Architecture](#7-recommended-architecture)

---

## 1. Proportional Navigation Guidance

### 1.1 Core Principle

Proportional Navigation (PN) is the dominant guidance law used in homing missiles, adapted here for ground robot interception. The fundamental insight: **two objects are on a collision course when their line-of-sight (LOS) angle does not change**. PN commands steering proportional to the LOS rotation rate to drive it to zero.

Source: [Wikipedia - Proportional Navigation](https://en.wikipedia.org/wiki/Proportional_navigation)

### 1.2 Line-of-Sight (LOS) Rate Calculation

Given our robot at position `(x_r, y_r)` and enemy at `(x_e, y_e)`:

```
LOS angle:      lambda = atan2(y_e - y_r, x_e - x_r)

LOS rate:       lambda_dot = (lambda_current - lambda_previous) / dt
```

For a more robust calculation using relative velocity:

```
R = [x_e - x_r, y_e - y_r]          # relative position vector
V_rel = [vx_e - vx_r, vy_e - vy_r]  # relative velocity vector

lambda_dot = (R x V_rel) / |R|^2
           = (R_x * V_rel_y - R_y * V_rel_x) / (R_x^2 + R_y^2)
```

Where `R x V_rel` is the 2D cross product (scalar). This avoids atan2 discontinuities and numerical differentiation noise.

### 1.3 The PN Steering Command (2D)

**Pure Proportional Navigation (PPN):**

```
a_n = N * lambda_dot * V_c
```

Where:
- `a_n` = commanded lateral acceleration (perpendicular to our velocity vector)
- `N` = navigation constant (dimensionless), typically **3 to 5**
- `lambda_dot` = LOS rotation rate (rad/s)
- `V_c` = closing velocity = -dR/dt (rate of range decrease)

**Closing velocity calculation:**

```
V_c = -(R . V_rel) / |R|
    = -(R_x * V_rel_x + R_y * V_rel_y) / sqrt(R_x^2 + R_y^2)
```

Where `R . V_rel` is the dot product. Positive V_c means objects are approaching.

### 1.4 Navigation Constant N Selection

| N Value | Behavior | Use Case |
|---------|----------|----------|
| 2       | Minimum for collision course, tail-chase only | Not practical |
| 3       | Classical PN, optimal for non-maneuvering targets | **Default starting point** |
| 4       | More aggressive correction, handles moderate maneuvers | **Recommended for combat** |
| 5       | Very aggressive, handles high-g maneuvers | Risk of oscillation |
| >5      | Diminishing returns, amplifies noise | Avoid |

**For combat robotics: Start with N=4.** Combat robots maneuver unpredictably. N=3 is optimal against constant-velocity targets but sluggish against maneuvering ones. N=5 risks oscillation with noisy CV measurements.

Source: [JHU APL - Basic Principles of Homing Guidance](https://secwww.jhuapl.edu/techdigest/content/techdigest/pdf/V29-N01/29-01-Palumbo_Principles_Rev2018.pdf)

### 1.5 Augmented Proportional Navigation (APN)

APN adds a term to compensate for estimated target acceleration:

```
a_n = N * lambda_dot * V_c + (N/2) * a_t
```

Where `a_t` is the estimated target lateral acceleration (perpendicular to LOS). This is the **optimal guidance law against a maneuvering target** and can be derived from optimal control theory.

For our application, `a_t` can be estimated from the Kalman filter's velocity change between frames.

### 1.6 Adapting PN to Differential Drive (Non-Holonomic)

Missiles steer with lateral acceleration. A differential-drive robot steers with angular velocity (omega). The adaptation:

**Step 1: Compute desired lateral acceleration from PN:**
```
a_n = N * lambda_dot * V_c
```

**Step 2: Convert acceleration to angular velocity command:**
```
omega = a_n / V_robot
```

Where `V_robot` is our robot's current forward speed. This gives the turn rate needed.

**Step 3: Convert to differential drive wheel speeds:**
```
V_left  = V_robot - omega * (track_width / 2)
V_right = V_robot + omega * (track_width / 2)
```

**Step 4: Clamp to physical limits:**
```python
V_left  = clamp(V_left, -V_max, V_max)
V_right = clamp(V_right, -V_max, V_max)
```

### 1.7 PN Pseudocode for Combat Robot

```python
def proportional_navigation(our_pos, our_vel, enemy_pos, enemy_vel,
                             N=4.0, track_width=0.15):
    """
    Compute differential drive commands using Proportional Navigation.
    
    Args:
        our_pos: (x, y) in meters
        our_vel: (vx, vy) in m/s
        enemy_pos: (x, y) in meters
        enemy_vel: (vx, vy) in m/s (from Kalman filter)
        N: navigation constant (3-5)
        track_width: wheel-to-wheel distance in meters
    
    Returns:
        (V_left, V_right) wheel speed commands
    """
    # Relative position and velocity
    Rx = enemy_pos[0] - our_pos[0]
    Ry = enemy_pos[1] - our_pos[1]
    Vx = enemy_vel[0] - our_vel[0]
    Vy = enemy_vel[1] - our_vel[1]
    
    R_sq = Rx**2 + Ry**2
    R_mag = math.sqrt(R_sq)
    
    if R_mag < 0.05:  # 5cm -- we've hit them
        return (0.0, 0.0)
    
    # LOS rotation rate (2D cross product / range squared)
    lambda_dot = (Rx * Vy - Ry * Vx) / R_sq
    
    # Closing velocity (negative dot product / range)
    V_closing = -(Rx * Vx + Ry * Vy) / R_mag
    
    if V_closing < 0.01:  # Target moving away faster than we approach
        # Fall back to pure pursuit (steer directly toward target)
        lambda_dot = compute_heading_error(our_pos, our_vel, enemy_pos)
        omega = 2.0 * lambda_dot  # proportional control fallback
    else:
        # PN lateral acceleration
        a_n = N * lambda_dot * V_closing
        
        # Convert to angular velocity
        V_robot = math.sqrt(our_vel[0]**2 + our_vel[1]**2)
        V_robot = max(V_robot, 0.1)  # avoid divide by zero
        omega = a_n / V_robot
    
    # Convert to differential drive
    V_forward = V_MAX  # full speed ahead during intercept
    V_left  = V_forward - omega * (track_width / 2)
    V_right = V_forward + omega * (track_width / 2)
    
    # Clamp to motor limits
    V_left  = max(-V_MAX, min(V_MAX, V_left))
    V_right = max(-V_MAX, min(V_MAX, V_right))
    
    return (V_left, V_right)
```

Source: [ResearchGate - Proportional navigation guidance for robotic interception](https://www.researchgate.net/publication/230444899_Proportional_navigation_guidance_for_robotic_interception_of_moving_objects)

---

## 2. Intercept Point Calculation

### 2.1 Problem Statement

Given:
- Our robot at position **P_r** with speed **s** (scalar, we control direction)
- Enemy at position **P_e** with velocity vector **V_e** (from Kalman filter)

Find: The point **I** where our robot can arrive at the same time as the enemy, assuming the enemy maintains constant velocity.

### 2.2 Quadratic Equation Derivation

Let **o** = P_e - P_r (offset vector from us to enemy).

At intercept time `t`:
- Enemy is at: P_e + V_e * t
- We travel distance: s * t

Setting distances equal (we can reach the intercept point):

```
|o + V_e * t| = s * t
```

Squaring both sides:

```
(o_x + V_e_x * t)^2 + (o_y + V_e_y * t)^2 = s^2 * t^2
```

Expanding and collecting terms in t:

```
(V_e_x^2 + V_e_y^2 - s^2) * t^2 + 2*(o_x*V_e_x + o_y*V_e_y) * t + (o_x^2 + o_y^2) = 0
```

This is a standard quadratic `a*t^2 + b*t + c = 0` with:

```
a = |V_e|^2 - s^2          = (enemy speed)^2 - (our speed)^2
b = 2 * (o . V_e)          = 2 * dot(offset, enemy_velocity)
c = |o|^2                  = (distance to enemy)^2
```

Source: [Calculating an intercept course](http://jaran.de/goodbits/2011/07/17/calculating-an-intercept-course-to-a-target-with-constant-direction-and-velocity-in-a-2-dimensional-plane/), [AI Projectile Intercept Formula](https://medium.com/andys-coding-blog/ai-projectile-intercept-formula-for-gaming-without-trigonometry-37b70ef5718b)

### 2.3 Solution Cases

**Discriminant: D = b^2 - 4*a*c**

| Case | Condition | Meaning |
|------|-----------|---------|
| D < 0 | No real roots | **Enemy is faster and moving away -- intercept impossible** |
| D = 0 | One root | Tangential intercept (barely reachable) |
| D > 0 | Two roots | Two possible intercept times |
| a = 0 | Linear case | Equal speeds: t = -c / b (if b != 0) |
| a = 0, b = 0 | Degenerate | Already at same position or parallel equal-speed |

**Selection rule**: Take the **smallest positive** root. Negative roots represent intercepts "in the past."

### 2.4 Complete Implementation

```python
def compute_intercept_point(our_pos, our_speed, enemy_pos, enemy_vel):
    """
    Compute the intercept point assuming enemy maintains constant velocity.
    
    Returns:
        (intercept_point, intercept_time) or (None, None) if impossible
    """
    # Offset from us to enemy
    ox = enemy_pos[0] - our_pos[0]
    oy = enemy_pos[1] - our_pos[1]
    
    # Quadratic coefficients
    evx, evy = enemy_vel
    a = evx**2 + evy**2 - our_speed**2
    b = 2.0 * (ox * evx + oy * evy)
    c = ox**2 + oy**2
    
    t = None
    
    if abs(a) < 1e-10:
        # Linear case: equal speeds
        if abs(b) < 1e-10:
            return (None, None)  # degenerate
        t = -c / b
    else:
        # Quadratic case
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return (None, None)  # no intercept possible
        
        sqrt_d = math.sqrt(discriminant)
        t1 = (-b + sqrt_d) / (2 * a)
        t2 = (-b - sqrt_d) / (2 * a)
        
        # Pick smallest positive time
        candidates = [t for t in [t1, t2] if t > 0.001]
        if not candidates:
            return (None, None)
        t = min(candidates)
    
    if t is None or t < 0:
        return (None, None)
    
    # Intercept point is where the enemy will be at time t
    intercept_x = enemy_pos[0] + evx * t
    intercept_y = enemy_pos[1] + evy * t
    
    return ((intercept_x, intercept_y), t)
```

### 2.5 Handling "No Intercept" Cases

When the quadratic has no solution (enemy is faster and diverging):

1. **Fall back to pure pursuit** -- steer directly toward current enemy position
2. **Cut-off strategy** -- steer toward the point on the arena boundary the enemy is heading toward (arena is only 8x8ft, walls force direction changes)
3. **Predictive wall bounce** -- if enemy is heading toward a wall, predict the bounce point

### 2.6 Iterative Refinement

The closed-form solution assumes constant enemy velocity. For a maneuvering target:

```python
def iterative_intercept(our_pos, our_speed, enemy_pos, enemy_vel, 
                         enemy_accel, max_iterations=3):
    """
    Iteratively refine intercept point accounting for enemy acceleration.
    """
    intercept, t = compute_intercept_point(our_pos, our_speed, 
                                            enemy_pos, enemy_vel)
    if intercept is None:
        return None, None
    
    for _ in range(max_iterations):
        # Predict enemy position at time t with acceleration
        pred_x = enemy_pos[0] + enemy_vel[0]*t + 0.5*enemy_accel[0]*t**2
        pred_y = enemy_pos[1] + enemy_vel[1]*t + 0.5*enemy_accel[1]*t**2
        pred_vx = enemy_vel[0] + enemy_accel[0]*t
        pred_vy = enemy_vel[1] + enemy_accel[1]*t
        
        # Recompute with updated prediction
        new_intercept, new_t = compute_intercept_point(
            our_pos, our_speed, (pred_x, pred_y), (pred_vx, pred_vy))
        
        if new_intercept is None:
            break
        
        if abs(new_t - t) < 0.01:  # converged (10ms)
            break
        
        intercept, t = new_intercept, new_t
    
    return intercept, t
```

---

## 3. Kalman Filter for Enemy Tracking

### 3.1 State Vector (Constant Velocity Model)

The enemy robot has no markers, so we track it via CV detection (position only, no direct velocity measurement). The Kalman filter estimates velocity from position observations.

```
State vector x = [x, y, vx, vy]^T

x, y   = enemy position in arena coordinates (meters)
vx, vy = enemy velocity (m/s), estimated by the filter
```

Source: [Cookie Robotics - Kalman Filter for 2D Motion](https://cookierobotics.com/071/)

### 3.2 State Transition Matrix F

Constant velocity model with time step `dt`:

```
F = | 1   0   dt  0  |
    | 0   1   0   dt |
    | 0   0   1   0  |
    | 0   0   0   1  |
```

This encodes: `x_new = x + vx*dt`, `vx_new = vx` (velocity persists).

### 3.3 Measurement Matrix H

We only measure position (from CV detection), not velocity:

```
H = | 1  0  0  0 |
    | 0  1  0  0 |
```

Maps state to measurement: `z = H * x = [x, y]^T`

### 3.4 Process Noise Covariance Q

The process noise models **unmodeled acceleration** -- the enemy changing direction/speed. This is the critical tuning parameter for combat.

Using the "discrete white noise acceleration" model:

```
Q = | dt^4/4    0      dt^3/2   0     |
    | 0         dt^4/4  0       dt^3/2 |         * sigma_a^2
    | dt^3/2    0       dt^2    0     |
    | 0         dt^3/2  0       dt^2  |
```

Where `sigma_a` is the **assumed maximum acceleration** of the enemy (m/s^2).

**Tuning sigma_a for combat robots:**

| sigma_a | Behavior | Scenario |
|---------|----------|----------|
| 1-2 m/s^2 | Smooth tracking, slow to adapt | Predictable movement |
| 3-5 m/s^2 | Balanced -- good for most combat | **Recommended starting point** |
| 8-15 m/s^2 | Very responsive, noisy velocity estimate | Erratic spinner/flipper |
| 20+ m/s^2 | Essentially trusts measurement over prediction | Emergency fallback |

**Start with sigma_a = 5.0 m/s^2** for beetleweight combat. This accommodates sudden direction changes (beetles can do ~2-3g turns at full speed) while maintaining useful velocity estimates.

Source: [IEEE - MSE Design of NCV Kalman Filters for Tracking Maneuvering Targets](https://ieeexplore.ieee.org/document/10032801/)

### 3.5 Measurement Noise Covariance R

The measurement noise reflects **how noisy our CV detection is** -- centroid jitter from contour detection, background subtraction artifacts, etc.

```
R = | sigma_x^2    0       |
    | 0            sigma_y^2 |
```

For overhead camera CV detection in an 8x8ft arena at 720p:
- Arena is ~244cm across ~900 pixels = ~2.7mm/pixel
- Contour centroid accuracy: typically +/- 3-5 pixels = ~8-14mm
- **sigma_x = sigma_y = 0.01 m (1cm)** is a reasonable starting point

Measure empirically: place a stationary object, detect it for 100 frames, compute standard deviation of centroid positions.

### 3.6 Prediction and Update Equations

**Prediction step** (every frame, even without detection):

```
x_predicted = F * x_estimated
P_predicted = F * P_estimated * F^T + Q
```

**Update step** (only when detection is available):

```
y = z - H * x_predicted                          # innovation (measurement residual)
S = H * P_predicted * H^T + R                     # innovation covariance
K = P_predicted * H^T * S^(-1)                    # Kalman gain
x_estimated = x_predicted + K * y                  # updated state
P_estimated = (I - K * H) * P_predicted            # updated covariance
```

**Joseph form** (numerically more stable for the covariance update):
```
P_estimated = (I - K*H) * P_predicted * (I - K*H)^T + K * R * K^T
```

### 3.7 Handling Detection Dropouts

Combat robots frequently disappear from detection (occlusion, blur, reflection):

```python
class EnemyTracker:
    def __init__(self, dt=1/60, sigma_a=5.0, sigma_meas=0.01):
        self.dt = dt
        self.x = np.zeros(4)        # [x, y, vx, vy]
        self.P = np.eye(4) * 100    # high initial uncertainty
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        
        # Process noise (discrete white noise acceleration)
        q = sigma_a**2
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        self.Q = np.array([
            [dt4/4, 0,     dt3/2, 0    ],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0    ],
            [0,     dt3/2, 0,     dt2  ]
        ]) * q
        
        self.R = np.eye(2) * sigma_meas**2
        self.frames_without_detection = 0
        self.MAX_COAST = 15  # ~250ms at 60fps
    
    def predict(self):
        """Always call predict each frame."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement):
        """Call when detection is available. measurement = [x, y]."""
        if measurement is not None:
            z = np.array(measurement)
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ self.H) @ self.P
            self.frames_without_detection = 0
        else:
            self.frames_without_detection += 1
    
    def is_tracking_valid(self):
        """Is the track still trustworthy?"""
        return self.frames_without_detection < self.MAX_COAST
    
    @property
    def position(self):
        return self.x[:2]
    
    @property
    def velocity(self):
        return self.x[2:]
    
    @property
    def speed(self):
        return np.linalg.norm(self.x[2:])
    
    @property
    def position_uncertainty(self):
        """Returns 1-sigma position uncertainty in meters."""
        return np.sqrt(self.P[0,0] + self.P[1,1])
```

### 3.8 Adaptive Process Noise

For adversarial targets that switch between cruising and maneuvering:

```python
def adaptive_process_noise(self, innovation):
    """
    Increase process noise when the filter is surprised by measurements.
    Based on innovation-based adaptive estimation (IAE).
    """
    # Normalized innovation squared (NIS)
    S = self.H @ self.P @ self.H.T + self.R
    nis = innovation.T @ np.linalg.inv(S) @ innovation
    
    # If NIS is much larger than expected (chi-squared, 2 DOF, 95% = 5.99)
    if nis > 6.0:
        # Enemy is maneuvering -- boost process noise temporarily
        self.Q *= 4.0  # quadruple Q
    elif nis < 1.0:
        # Tracking well -- can reduce process noise toward baseline
        self.Q = self.Q * 0.9 + self.Q_baseline * 0.1
```

Source: [MDPI - Self-Tuning Process Noise in Variational Bayesian Adaptive Kalman Filter](https://www.mdpi.com/2079-9292/12/18/3887)

---

## 4. Pure Pursuit vs Lead Pursuit

### 4.1 Pure Pursuit

**Definition**: Steer directly toward the target's **current position**. The velocity vector always points at the target.

**Equation:**
```
desired_heading = atan2(enemy_y - our_y, enemy_x - our_x)
heading_error = normalize_angle(desired_heading - our_heading)
omega = Kp * heading_error
```

**Curvature formulation** (from CMU's Coulter 1992):
```
kappa = 2 * sin(alpha) / L_d
```
Where:
- `kappa` = curvature (1/turning_radius)
- `alpha` = angle from robot heading to target point
- `L_d` = lookahead distance (distance to target)

For differential drive:
```
omega = kappa * V_robot = 2 * V_robot * sin(alpha) / L_d
```

Source: [CMU - Implementation of Pure Pursuit Path Tracking (Coulter 1992)](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf), [Algorithms for Automated Driving - Pure Pursuit](https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/PurePursuit.html)

### 4.2 Lead Pursuit

**Definition**: Steer toward a point **ahead of the target** along its velocity vector. The aim point "leads" the target.

**Equation:**
```
lead_point = enemy_pos + enemy_vel * lead_time
desired_heading = atan2(lead_point_y - our_y, lead_point_x - our_x)
heading_error = normalize_angle(desired_heading - our_heading)
omega = Kp * heading_error
```

**Lead time selection**: Use the intercept time from Section 2, or a fixed time horizon:
```
lead_time = distance_to_enemy / our_speed
```

This is essentially a simplified version of intercept point calculation.

### 4.3 Comparison

| Property | Pure Pursuit | Lead Pursuit | Proportional Navigation |
|----------|-------------|--------------|------------------------|
| **Complexity** | Trivial | Low | Moderate |
| **Requires enemy velocity?** | No | Yes | Yes |
| **Path shape** | Curved (tail-chase spiral) | Straighter intercept | Near-optimal intercept |
| **Against moving target** | Always behind | Close to intercept | Provably optimal |
| **Against stationary** | Converges directly | Same as pure | Degenerates to pure |
| **Noise sensitivity** | Low | Medium | Higher (uses V_c, lambda_dot) |
| **Best for** | Close range, slow targets | Medium range | Long range, fast targets |

### 4.4 Lookahead Distance Tuning

For the pure pursuit curvature equation, the lookahead distance `L_d` controls stability:

```
L_d = K_dd * V_robot
```

Where K_dd is a tuning constant.

| L_d | Effect |
|-----|--------|
| Small (< 0.3m) | Aggressive tracking, oscillation risk, good accuracy |
| Medium (0.3-0.6m) | Balanced -- **recommended for 8x8ft arena** |
| Large (> 0.6m) | Smooth but cuts corners, slow to react |

For our 8x8ft (2.44m) arena, the maximum meaningful L_d is about 1.0m. Recommended: **L_d = 0.3 * V_robot + 0.1** (velocity-scaled with minimum).

Source: [Purdue SIGBots - Basic Pure Pursuit](https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit)

### 4.5 Recommended Strategy by Situation

```
if distance > 1.0m and enemy_speed > threshold and velocity_estimate_confident:
    use Proportional Navigation (N=4)
elif distance > 0.5m and have_velocity_estimate:
    use Lead Pursuit with intercept point
else:
    use Pure Pursuit (always works, no velocity needed)
```

---

## 5. Enemy Detection Without Markers

### 5.1 MOG2 Background Subtraction

OpenCV's MOG2 (Mixture of Gaussians, version 2) is the recommended approach for detecting the enemy robot. It models each pixel as a mixture of Gaussians and classifies pixels that don't fit the background model as foreground.

Source: [OpenCV - BackgroundSubtractorMOG2](https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)

**Key parameters and recommended values for combat:**

```python
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=120,          # ~2 seconds at 60fps (default 500 is too slow for combat)
    varThreshold=25,      # higher than default 16 to reduce noise
    detectShadows=False   # shadows waste CPU and create false detections
)
```

| Parameter | Default | Combat Setting | Rationale |
|-----------|---------|----------------|-----------|
| `history` | 500 | **120** | 2s of frames. Shorter = adapts faster to lighting changes. Longer = more stable background. 500 (8s) means a stopped robot stays "foreground" too long |
| `varThreshold` | 16 | **25-40** | Higher threshold = less sensitive = fewer false positives. Arena floor texture creates noise at default 16 |
| `detectShadows` | True | **False** | Shadow detection costs ~20% performance. Overhead lighting in arena creates complex shadows. Better to handle via morphological ops |
| `backgroundRatio` | 0.9 | **0.7** | Fraction of data to consider background. Lower = more aggressive foreground detection |
| `nmixtures` | 5 | **3** | Fewer Gaussians per pixel = faster. 3 is sufficient for arena floor |
| `varInit` | 15 | 15 | Initial variance for new Gaussians |
| `varMin` | 4 | 4 | Minimum allowed variance |
| `varMax` | 75 | **50** | Maximum variance. Lower = more sensitive to changes |

Source: [Simon Wenkel - Background Subtraction with OpenCV](https://www.simonwenkel.com/notes/software_libraries/opencv/background-subtraction-using-opencv.html)

### 5.2 Post-Processing Pipeline

Raw MOG2 output is noisy. Apply morphological operations:

```python
def detect_enemy(frame, bg_subtractor, our_robot_mask):
    # 1. Background subtraction
    fg_mask = bg_subtractor.apply(frame, learningRate=0.005)
    
    # 2. Threshold to binary (remove shadow pixels if detectShadows=True)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    # 3. Morphological cleanup
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)   # remove noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close) # fill gaps
    
    # 4. Exclude our own robot
    fg_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(our_robot_mask))
    
    # 5. Find and filter contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    enemy_detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (MIN_AREA < area < MAX_AREA):
            continue
        
        # Aspect ratio filter
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        if not (0.3 < aspect_ratio < 3.0):  # robots are roughly square-ish
            continue
        
        # Solidity filter (ratio of contour area to convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < 0.4:  # robots are fairly solid shapes
            continue
        
        # Centroid
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            enemy_detections.append((cx, cy, area))
    
    # Return largest detection (most likely the enemy robot)
    if enemy_detections:
        enemy_detections.sort(key=lambda d: d[2], reverse=True)
        return enemy_detections[0][:2]
    
    return None
```

Source: [OpenCV - Contour Properties](https://docs.opencv.org/3.4/d1/d32/tutorial_py_contour_properties.html)

### 5.3 Contour Filtering Parameters

For a beetleweight (3lb / 1.36kg) robot in an 8x8ft arena at 720p:

```
Arena: 2.44m x 2.44m mapped to ~900 x 900 pixels
Robot size: ~10cm x 10cm = ~37 x 37 pixels at center
Scale: ~2.7mm/pixel

MIN_AREA = 400 pixels^2    (~20x20px, smallest reasonable robot blob)
MAX_AREA = 8000 pixels^2   (~90x90px, largest reasonable + some margin)
```

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| Area min | 400 px^2 | Reject noise specks |
| Area max | 8000 px^2 | Reject arena features, large shadows |
| Aspect ratio | 0.3 - 3.0 | Reject long thin reflections/shadows |
| Solidity | > 0.4 | Reject irregular noise shapes |

### 5.4 Excluding Our Own Robot

Two approaches, use both:

**Approach 1: ArUco-based mask.** Our robot's ArUco detection gives its position and orientation. Expand the marker bounding box to cover the full robot body:

```python
def create_robot_mask(frame_shape, aruco_corners, expansion=40):
    """Create a mask around our detected ArUco marker + body."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if aruco_corners is not None:
        center = aruco_corners.mean(axis=0).astype(int)
        # Expanded bounding box covers full robot body
        cv2.circle(mask, tuple(center), expansion, 255, -1)
    return mask
```

**Approach 2: Color/appearance filtering.** If our robot has a distinctive color (e.g., bright tape), HSV filtering can further exclude it.

### 5.5 Frame Differencing as Fast Alternative

For extremely low-latency needs, frame differencing is ~3x faster than MOG2 but less robust:

```python
def frame_difference_detect(prev_gray, curr_gray, threshold=30):
    """Simple frame differencing -- fast but noisy."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask
```

**Frame differencing limitations:**
- Detects motion, not objects -- a stopped enemy is invisible
- "Ghosting" artifact: detects where the object was AND where it is
- Double-blob effect at high speed
- Aperture problem: uniform-colored robot interior produces hollow detection

**When to use**: As a fast pre-filter to confirm "something is moving" before running MOG2 on the ROI, or as a fallback when MOG2 background model is corrupted.

Source: [LearnOpenCV - Moving Object Detection](https://learnopencv.com/moving-object-detection-with-opencv/), [DEV.to - Motion Detection and Tracking in OpenCV](https://dev.to/jarvissan22/blog-cv2-video-and-motion-detection-and-tracking-j4c)

### 5.6 Handling Shadows and Reflections

Arena-specific challenges:
- **Polycarbonate walls** create reflections (especially under LED lighting)
- **Overhead lighting** creates hard shadows that move with robots
- **Arena floor** (plywood) has texture that MOG2 may classify as foreground

Mitigation strategies:

1. **Increase varThreshold to 25-40**: Reduces sensitivity to subtle shadows
2. **Morphological opening (5x5)**: Removes small shadow/reflection artifacts
3. **Area filtering**: Shadows are usually smaller or larger than the expected robot size
4. **Arena boundary mask**: Mask out wall reflection zones (known regions near arena edges)
5. **Solidity filter**: Shadows tend to be elongated with low solidity
6. **CLAHE preprocessing**: Normalizes uneven lighting before background subtraction

```python
# CLAHE preprocessing to handle uneven arena lighting
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
```

---

## 6. Prediction Under Uncertainty

### 6.1 The Core Problem

Enemy robots are adversarial -- they actively try to evade or attack unpredictably. No motion model is accurate for more than a fraction of a second. The system must handle:

- Sudden 180-degree turns
- Stops and starts
- Spinning in place (spinners)
- Erratic evasive movement

### 6.2 Prediction Horizon

**Rule of thumb**: Only trust predictions up to `T_trust` seconds ahead.

```
T_trust = min(0.5s, distance_to_enemy / closing_speed)
```

At 60 FPS with <30ms latency, we recompute every ~16ms. Even a bad 500ms prediction barely matters because we correct 30 times within that window.

**Prediction confidence decays exponentially:**
```python
confidence = exp(-t / tau)
# tau = 0.3s for combat robots (direction changes every ~0.3-0.5s)
```

### 6.3 Low-Pass Filtering the Intercept Point

Raw intercept point calculations jump around frame-to-frame due to:
- Measurement noise in enemy position
- Velocity estimate noise from Kalman filter
- Discrete-time LOS rate calculation noise

**Exponential Moving Average (EMA) on intercept point:**

```python
class SmoothedIntercept:
    def __init__(self, alpha=0.3):
        """
        alpha: smoothing factor (0-1). 
               Lower = smoother but more lag.
               Higher = responsive but jittery.
        """
        self.alpha = alpha
        self.smoothed = None
    
    def update(self, new_intercept):
        if new_intercept is None:
            return self.smoothed
        
        if self.smoothed is None:
            self.smoothed = np.array(new_intercept)
        else:
            self.smoothed = (self.alpha * np.array(new_intercept) + 
                           (1 - self.alpha) * self.smoothed)
        return self.smoothed
```

**Recommended alpha values:**

| alpha | Behavior | Use when |
|-------|----------|----------|
| 0.1-0.2 | Very smooth, significant lag | Enemy moving predictably |
| 0.3-0.4 | Balanced | **Default for combat** |
| 0.5-0.7 | Responsive, some jitter | Enemy maneuvering actively |
| 0.8-1.0 | Near-raw signal | Close range, need maximum reactivity |

**Adaptive alpha based on distance:**
```python
alpha = 0.2 + 0.5 * (1.0 - distance / max_arena_diagonal)
# Close range: alpha ~0.7 (responsive)
# Far range: alpha ~0.2 (smooth)
```

Source: [mbedded.ninja - EMA Filters](https://blog.mbedded.ninja/programming/signal-processing/digital-filters/exponential-moving-average-ema-filter/), [ResearchGate - Double Exponential Smoothing for Predictive Tracking](https://www.researchgate.net/publication/268217162_Double_Exponential_Smoothing_for_Predictive_Vision_Based_Target_Tracking_of_a_Wheeled_Mobile_Robot)

### 6.4 Re-planning Frequency

At 60 FPS, we have 16.7ms per frame. The full pipeline:

```
Frame capture:           ~2ms
ArUco detection (ours):  ~3ms
Background sub (enemy):  ~4ms
Kalman filter update:    ~0.1ms
Intercept calculation:   ~0.05ms
PN steering command:     ~0.05ms
Motor command TX:        ~1ms
------------------------------------
Total:                   ~10ms (well under 30ms budget)
```

**Replan every frame (60 Hz)**. The math is cheap (~0.2ms for Kalman + intercept + PN). There is no reason to replan less frequently. The smoothing filter handles jitter.

### 6.5 When to Switch Strategies

Implement a finite state machine:

```python
class PursuitStateMachine:
    SEARCH   = 0  # No enemy detected
    ACQUIRE  = 1  # Enemy detected, building velocity estimate
    INTERCEPT = 2  # Full PN guidance active
    CLOSE    = 3  # Within striking distance, pure pursuit
    LOST     = 4  # Had track, lost it, coasting on prediction
    
    def update(self, enemy_detected, track_age, distance, 
               velocity_confidence):
        
        if not enemy_detected and track_age > 15:  # 250ms
            return self.SEARCH
        
        if not enemy_detected and track_age <= 15:
            return self.LOST
        
        if distance < 0.3:  # 30cm -- striking distance
            return self.CLOSE
        
        if velocity_confidence < 0.5:  # Need ~10 frames to estimate velocity
            return self.ACQUIRE
        
        return self.INTERCEPT
```

**Strategy per state:**

| State | Strategy | Speed | Notes |
|-------|----------|-------|-------|
| SEARCH | Spin in place or patrol pattern | Low | Looking for enemy |
| ACQUIRE | Pure pursuit toward detection | Medium | Building Kalman velocity estimate |
| INTERCEPT | Proportional Navigation (N=4) | **Full** | Primary combat mode |
| CLOSE | Pure pursuit, max throttle | **Full** | PN degenerates at close range |
| LOST | Drive toward last predicted position | Medium | Coast on Kalman prediction |

### 6.6 Handling Wall Bounces

In an 8x8ft enclosed arena, wall collisions are frequent. Predict wall bounces:

```python
def predict_with_walls(enemy_pos, enemy_vel, dt, 
                        arena_min=(0,0), arena_max=(2.44, 2.44)):
    """Predict enemy position accounting for wall bounces."""
    pred_x = enemy_pos[0] + enemy_vel[0] * dt
    pred_y = enemy_pos[1] + enemy_vel[1] * dt
    vel_x, vel_y = enemy_vel
    
    # Simple elastic bounce off walls
    if pred_x < arena_min[0]:
        pred_x = 2*arena_min[0] - pred_x
        vel_x = -vel_x
    elif pred_x > arena_max[0]:
        pred_x = 2*arena_max[0] - pred_x
        vel_x = -vel_x
    
    if pred_y < arena_min[1]:
        pred_y = 2*arena_min[1] - pred_y
        vel_y = -vel_y
    elif pred_y > arena_max[1]:
        pred_y = 2*arena_max[1] - pred_y
        vel_y = -vel_y
    
    return (pred_x, pred_y), (vel_x, vel_y)
```

---

## 7. Recommended Architecture

### 7.1 Full Pipeline Summary

```
Camera Frame (60fps)
    |
    +---> ArUco Detection ----> Our position + heading
    |                                |
    +---> MOG2 Background Sub        |
    |         |                      |
    |    Contour Filter              |
    |    Exclude Our Robot <---------+
    |         |
    |    Enemy Detection (or None)
    |         |
    +---> Kalman Filter (predict always, update when detected)
              |
         Enemy position + velocity estimate
              |
         Intercept Point Calculation (quadratic)
              |
         EMA Smoothing on intercept point
              |
         Strategy FSM (SEARCH/ACQUIRE/INTERCEPT/CLOSE/LOST)
              |
         Guidance Law:
           - INTERCEPT: Proportional Navigation (N=4)
           - CLOSE: Pure Pursuit
           - ACQUIRE: Pure Pursuit
           - LOST: Coast to prediction
           - SEARCH: Patrol pattern
              |
         Differential Drive Commands
              |
         Motor Output (UDP to ESP32)
```

### 7.2 Key Parameters Cheat Sheet

| Parameter | Value | Section |
|-----------|-------|---------|
| Navigation constant N | 4 | 1.4 |
| Kalman sigma_a | 5.0 m/s^2 | 3.4 |
| Kalman sigma_meas | 0.01 m | 3.5 |
| Kalman max coast frames | 15 (~250ms) | 3.7 |
| MOG2 history | 120 | 5.1 |
| MOG2 varThreshold | 25-40 | 5.1 |
| Contour min area | 400 px^2 | 5.3 |
| Contour max area | 8000 px^2 | 5.3 |
| Contour solidity min | 0.4 | 5.3 |
| EMA alpha (intercept) | 0.3 | 6.3 |
| Pure pursuit L_d | 0.3*V + 0.1 m | 4.4 |
| Close range threshold | 0.3 m | 6.5 |
| Prediction trust horizon | 0.5 s | 6.2 |

### 7.3 Estimated Latency Budget

| Stage | Time | Cumulative |
|-------|------|------------|
| Frame capture | 2 ms | 2 ms |
| ArUco detection | 3 ms | 5 ms |
| MOG2 + contours | 4 ms | 9 ms |
| Kalman + intercept + PN | 0.3 ms | 9.3 ms |
| EMA smoothing | 0.01 ms | 9.3 ms |
| Motor command TX (UDP) | 1 ms | 10.3 ms |
| **Total** | **~10 ms** | Well under 30ms target |

---

## References

- [Wikipedia - Proportional Navigation](https://en.wikipedia.org/wiki/Proportional_navigation)
- [JHU APL - Basic Principles of Homing Guidance (Palumbo)](https://secwww.jhuapl.edu/techdigest/content/techdigest/pdf/V29-N01/29-01-Palumbo_Principles_Rev2018.pdf)
- [ResearchGate - PN guidance for robotic interception of moving objects](https://www.researchgate.net/publication/230444899_Proportional_navigation_guidance_for_robotic_interception_of_moving_objects)
- [ResearchGate - Augmented ideal PN guidance for robotic interception](https://www.researchgate.net/publication/3412043_Robotic_interception_of_moving_objects_using_an_augmented_ideal_proportional_navigation_guidance_technique)
- [Calculating an intercept course (jaran.de)](http://jaran.de/goodbits/2011/07/17/calculating-an-intercept-course-to-a-target-with-constant-direction-and-velocity-in-a-2-dimensional-plane/)
- [AI Projectile Intercept Formula (Medium)](https://medium.com/andys-coding-blog/ai-projectile-intercept-formula-for-gaming-without-trigonometry-37b70ef5718b)
- [Cookie Robotics - Kalman Filter for 2D Motion](https://cookierobotics.com/071/)
- [KalmanFilter.net - Examples](https://kalmanfilter.net/)
- [IEEE - MSE Design of NCV Kalman Filters for Maneuvering Targets](https://ieeexplore.ieee.org/document/10032801/)
- [MDPI - Self-Tuning Process Noise in Adaptive Kalman Filter](https://www.mdpi.com/2079-9292/12/18/3887)
- [CMU - Pure Pursuit Path Tracking (Coulter 1992)](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)
- [Algorithms for Automated Driving - Pure Pursuit](https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/PurePursuit.html)
- [Purdue SIGBots - Basic Pure Pursuit](https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit)
- [OpenCV - BackgroundSubtractorMOG2](https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)
- [Simon Wenkel - Background Subtraction with OpenCV](https://www.simonwenkel.com/notes/software_libraries/opencv/background-subtraction-using-opencv.html)
- [LearnOpenCV - Moving Object Detection](https://learnopencv.com/moving-object-detection-with-opencv/)
- [OpenCV - Contour Properties](https://docs.opencv.org/3.4/d1/d32/tutorial_py_contour_properties.html)
- [mbedded.ninja - EMA Filters](https://blog.mbedded.ninja/programming/signal-processing/digital-filters/exponential-moving-average-ema-filter/)
- [ResearchGate - Double Exponential Smoothing for Predictive Tracking](https://www.researchgate.net/publication/268217162_Double_Exponential_Smoothing_for_Predictive_Vision_Based_Target_Tracking_of_a_Wheeled_Mobile_Robot)
