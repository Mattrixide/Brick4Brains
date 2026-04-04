# IMU-Assisted Turning, Click-to-Point & Interception Research

**Project**: Brick for Brains - Autonomous Combat Robot
**Context**: Beetleweight (3lb) differential drive, ESP32 MCU, SparkFun ISM330DHCX 6DoF IMU, ArUco CV tracking, 8x8ft arena
**Goal**: Precise high-speed turning (<0.5s for 90°), click-to-point navigation, moving object interception

---

## Quick Wins

| Change | Impact | Effort | Status |
|--------|--------|--------|--------|
| Add ISM330DHCX gyro to ESP32 (Qwiic) | 1kHz heading feedback vs 30Hz camera | Wire + firmware | **Research complete** |
| On-ESP32 feedforward+PID turn control | <0.5s 90° turns, <5° error | Firmware (~200 lines) | **Research complete** |
| Mode-switched sensor fusion | Survive ArUco occlusion for ~5s | ~50 lines Python | **Research complete** |
| cv2.setMouseCallback for click-to-point | Click → goto in one click | ~20 lines Python | **Trivial** |
| MOG2 background subtraction for enemy | Detect enemy without markers | ~80 lines Python | **Research complete** |
| Kalman filter + PN guidance for intercept | Optimal interception trajectory | ~150 lines Python | **Research complete** |

---

## Table of Contents

1. [ISM330DHCX + ESP32 Integration](#1-ism330dhcx--esp32-integration)
2. [Feedforward + PID Turn Control](#2-feedforward--pid-turn-control)
3. [Trapezoidal Motion Profiling](#3-trapezoidal-motion-profiling)
4. [Mode-Switched Sensor Fusion](#4-mode-switched-sensor-fusion)
5. [Click-to-Point Navigation](#5-click-to-point-navigation)
6. [Enemy Detection Without Markers](#6-enemy-detection-without-markers)
7. [Interception & Pursuit Algorithms](#7-interception--pursuit-algorithms)
8. [Recommended Architecture](#8-recommended-architecture)

---

## 1. ISM330DHCX + ESP32 Integration

### 1.1 Sensor Specifications

The SparkFun Micro 6DoF IMU uses the ST ISM330DHCX — a high-performance 6-axis IMU with gyroscope + accelerometer. Unlike the BNO055/085 (which has onboard fusion), this is a raw sensor — we do fusion ourselves. This is actually **advantageous** for combat robots: no magnetometer means no interference from drive motors and weapon magnets.

**Gyroscope:**

| Parameter | Value |
|-----------|-------|
| Full-scale ranges | ±125, ±250, ±500, ±1000, ±2000, ±4000 dps |
| ODR options | 12.5, 26, 52, 104, 208, 416, 833, 1666, 3332, 6667 Hz |
| Rate noise density | 3.8 mdps/√Hz typical |
| Zero-rate level | ±1 dps typical |
| Temp sensitivity | ±0.005 dps/°C typical |
| Sensitivity @ ±2000dps | 70 mdps/LSB |

**Accelerometer:**

| Parameter | Value |
|-----------|-------|
| Full-scale ranges | ±2, ±4, ±8, ±16 g |
| ODR options | 1.6 to 6667 Hz |
| Noise density | 60 µg/√Hz typical |
| Sensitivity @ ±4g | 0.122 mg/LSB |

Source: [ISM330DHCX Datasheet (ST)](https://www.st.com/resource/en/datasheet/ism330dhcx.pdf)

### 1.2 SparkFun Micro Board

| Parameter | Value |
|-----------|-------|
| Dimensions | 0.75 × 0.30 in (19 × 7.6 mm) |
| Connector | 1× Qwiic JST-SH (GND, 3.3V, SDA, SCL) |
| Default I2C address | 0x6B (alternate 0x6A via jumper) |
| Pull-ups | 2.2kΩ on SDA/SCL (included, disable via jumper) |
| Interrupt | INT1 only (no INT2 on micro board) |
| SPI | Not available (I2C only) |

**ESP32 wiring:** Qwiic connector → GPIO21 (SDA) / GPIO22 (SCL). Both are 3.3V — no level shifter needed. The board's 2.2kΩ pull-ups are sufficient for 400 kHz I2C.

Source: [SparkFun Micro 6DoF Product Page](https://www.sparkfun.com/sparkfun-micro-6dof-imu-ism330dhcx-qwiic.html)

### 1.3 Arduino Library

**Library:** `SparkFun 6DoF ISM330DHCX` — wraps ST's official C driver.
**GitHub:** [sparkfun/SparkFun_6DoF_ISM330DHCX_Arduino_Library](https://github.com/sparkfun/SparkFun_6DoF_ISM330DHCX_Arduino_Library)

```cpp
#include <Wire.h>
#include <SparkFun_ISM330DHCX.h>

SparkFun_ISM330DHCX myISM;
sfe_ism_data_t gyroData;
sfe_ism_data_t accelData;

void setup() {
    Wire.begin();
    Wire.setClock(400000);  // 400 kHz Fast Mode
    
    if (!myISM.begin()) { /* handle error */ }
    
    myISM.deviceReset();
    while (!myISM.getDeviceReset()) { delay(1); }
    
    // Gyro: 416 Hz, ±2000 dps
    myISM.setGyroDataRate(ISM_GY_ODR_416Hz);
    myISM.setGyroFullScale(ISM_2000dps);
    myISM.setGyroFilterLP1();
    myISM.setGyroLP1Bandwidth(ISM_MEDIUM);
    
    // Accel: 416 Hz, ±4g (for impact detection)
    myISM.setAccelDataRate(ISM_XL_ODR_416Hz);
    myISM.setAccelFullScale(ISM_4g);
}

void loop() {
    if (myISM.checkStatus()) {
        myISM.getGyro(&gyroData);   // .xData/.yData/.zData in mdps
        myISM.getAccel(&accelData); // .xData/.yData/.zData in mg
    }
}
```

**Key ODR enums:** `ISM_GY_ODR_104Hz`, `ISM_GY_ODR_208Hz`, `ISM_GY_ODR_416Hz`, `ISM_GY_ODR_833Hz`, `ISM_GY_ODR_1666Hz`
**Key FS enums:** `ISM_125dps`, `ISM_250dps`, `ISM_500dps`, `ISM_1000dps`, `ISM_2000dps`, `ISM_4000dps`

### 1.4 Recommended Settings for Combat

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Gyro ODR | **416 Hz** | 2.4ms period, fast enough for PID, within I2C bandwidth at 400 kHz |
| Gyro FS | **±2000 dps** | 70 mdps/LSB resolution, covers most combat. Use ±4000 only if saturation observed |
| Accel ODR | **416 Hz** | Match gyro for synchronized reads |
| Accel FS | **±4g** | Detects impacts without saturating on normal motion |
| I2C clock | **400 kHz** | Fast Mode — reliable on ESP32 with Qwiic pull-ups |
| Control loop | **416 Hz** | Match IMU ODR for one read per loop |

### 1.5 Gyro Integration for Heading

**Basic integration:**
```cpp
float gyroZ_dps = (gyroData.zData - biasZ) / 1000.0f;  // mdps → dps
heading_deg += gyroZ_dps * dt;  // integrate
```

**Calibration (critical — single most impactful step):**

At power-on, keep robot stationary for ~1 second. Average gyro Z readings to estimate bias:

```cpp
float biasZ = 0;
const int N = 416;  // 1 second at 416 Hz
for (int i = 0; i < N; i++) {
    while (!myISM.checkStatus()) {}
    myISM.getGyro(&gyroData);
    biasZ += gyroData.zData;
}
biasZ /= N;
```

**Dead-zone threshold (reduces drift by ~50%):**
```cpp
float threshold_dps = 0.3f;  // below noise floor
if (fabsf(gyroZ_dps) < threshold_dps) gyroZ_dps = 0;
```

**Expected drift after calibration:**

| Method | Drift Rate |
|--------|------------|
| Raw (no calibration) | 1-5 °/min |
| Bias calibration only | 0.1-0.5 °/min |
| Bias + threshold filter | **0.05-0.2 °/min** |

At 0.1°/min drift, heading error after 5 seconds of ArUco occlusion is only ~0.008° — negligible.

**Handling gyro saturation during combat:**
- At ±2000 dps, saturation means the robot is spinning >5.5 rev/s — extreme impacts only
- Detect: `if (fabsf(gyroZ_dps) > 1900) saturated = true;`
- During saturation: hold last heading, flag for recalibration
- After impact settles: rapid bias re-estimation (~200ms) if robot is momentarily still

Sources:
- [SparkFun ISM330DHCX Hookup Guide](https://learn.sparkfun.com/tutorials/qwiic-6dof---ism330dhcx-hookup-guide/all)
- [Minimal-Drift Heading with MEMS Gyro (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3787445/)

---

## 2. Feedforward + PID Turn Control

### 2.1 Why Feedforward + PID

Pure PID reacts to error. **Feedforward predicts** the motor command needed, and PID corrects the residual. This is near time-optimal — the feedforward gets you 80-90% of the way, PID handles the rest.

```
                                                         +-------+
desired_omega ──> [Motion Profile] ──> omega_setpoint ──>| FF    |──> V_ff ──+
                                            |            +-------+           |
                                            |                                +(sum)──> Motor
                                            |            +-------+           |
                                            +──> error ──| PID   |──> V_pid─+
                                            |            +-------+
                                   gyro_omega (measured) ─┘
```

### 2.2 Motor Model for Feedforward

Simplified DC motor steady-state:
```
V_ff = kV * omega_setpoint + kS * sign(omega_setpoint) + kA * alpha_setpoint
```

| Term | Purpose | How to measure |
|------|---------|----------------|
| `kV` (velocity FF) | Volts per dps at steady state | Spin robot at known speeds, measure voltage |
| `kS` (static friction FF) | Minimum voltage to start moving | Ramp voltage slowly, note when motion begins |
| `kA` (acceleration FF) | Voltage for angular acceleration | Optional — improves transient response |

**Starting values (tune empirically):**
- `kV ≈ 0.05` (V per dps — characterize your motors)
- `kS ≈ 0.8` (V — minimum to overcome friction)
- `kA ≈ 0.001` (V per dps/s — often negligible)

### 2.3 PID on Angular Velocity Error

```cpp
float error = omega_setpoint - omega_measured;  // dps
integral += error * dt;
integral = constrain(integral, -max_integral, max_integral);  // anti-windup
float derivative = (error - prev_error) / dt;

float V_pid = Kp * error + Ki * integral + Kd * derivative;
float V_total = V_ff + V_pid;

// Differential drive: opposite directions for rotation
setMotors(-V_total, V_total);
```

**PID starting points for beetleweight:**

| Gain | Starting Value | Notes |
|------|---------------|-------|
| Kp | 0.01-0.05 | V per dps error. Increase until oscillation, back off 30% |
| Ki | 0.001-0.01 | Eliminates steady-state error from friction. Keep low |
| Kd | 0.0001-0.001 | Damps oscillation. Often unnecessary with good FF |

**Tuning procedure:**
1. Set Ki = Kd = 0, enable feedforward
2. Increase Kp until oscillation, set to 60-70% of that
3. Add small Ki to eliminate steady-state error
4. Add Kd only if oscillation persists
5. Implement integral windup clamping

### 2.4 Differential Drive Kinematics

For pure rotation (zero forward velocity):
```
v_left  = -(L/2) * omega
v_right =  (L/2) * omega
```

For combined forward + turning:
```
v_left  = v_forward - (L/2) * omega
v_right = v_forward + (L/2) * omega
```

Where `L` = wheel track width (wheel-to-wheel distance).

Sources:
- [Feedforward+PID for Agricultural Robot Turning (SAGE)](https://journals.sagepub.com/doi/10.1177/1729881419897678)
- [Differential Drive Kinematics (Columbia)](https://www.cs.columbia.edu/~allen/F15/NOTES/icckinematics.pdf)

---

## 3. Trapezoidal Motion Profiling

### 3.1 Profile Phases

A trapezoidal angular velocity profile for a turn:

```
omega
  ^
  |     ___________
  |    /           \         omega_max
  |   /             \
  |  /               \
  | /                 \
  +----+----+----+----+---> time
    t0   t1   t2   t3
   accel cruise decel
```

1. **Accelerate:** Ramp from 0 to omega_max at rate alpha
2. **Cruise:** Hold omega_max until deceleration must begin
3. **Decelerate:** Ramp from omega_max to 0 at rate alpha

### 3.2 Computing Switch Points

Given target angle `theta_target`, max angular velocity `omega_max`, and angular acceleration `alpha`:

```
t_accel = omega_max / alpha
theta_accel = omega_max² / (2 * alpha)     // angle during accel phase
```

**If `2 * theta_accel >= theta_target`:** Triangular profile (never reaches omega_max)
```
t_accel = sqrt(theta_target / alpha)
omega_peak = alpha * t_accel
t_cruise = 0
t_total = 2 * t_accel
```

**Otherwise:** Full trapezoidal
```
theta_cruise = theta_target - 2 * theta_accel
t_cruise = theta_cruise / omega_max
t_total = 2 * t_accel + t_cruise
```

### 3.3 Implementation

```cpp
struct TrapezoidalProfile {
    float theta_target;  // degrees (absolute value)
    float omega_max;     // dps
    float alpha;         // dps/s (angular acceleration)
    float t_accel, t_cruise, t_total;
    float omega_peak;
    float direction;     // +1 or -1
    
    void compute(float target_deg, float max_omega, float max_alpha) {
        direction = (target_deg >= 0) ? 1.0f : -1.0f;
        theta_target = fabsf(target_deg);
        omega_max = max_omega;
        alpha = max_alpha;
        
        t_accel = omega_max / alpha;
        float theta_accel = 0.5f * alpha * t_accel * t_accel;
        
        if (2.0f * theta_accel >= theta_target) {
            // Triangular — never reach omega_max
            t_accel = sqrtf(theta_target / alpha);
            omega_peak = alpha * t_accel;
            t_cruise = 0;
        } else {
            // Full trapezoidal
            omega_peak = omega_max;
            t_cruise = (theta_target - 2.0f * theta_accel) / omega_max;
        }
        t_total = 2.0f * t_accel + t_cruise;
    }
    
    float getOmega(float t) const {
        if (t < 0) return 0;
        if (t < t_accel) return direction * alpha * t;
        if (t < t_accel + t_cruise) return direction * omega_peak;
        if (t < t_total) return direction * alpha * (t_total - t);
        return 0;
    }
    
    bool isDone(float t) const { return t >= t_total; }
};
```

### 3.4 Practical Parameters for Beetleweight

| Parameter | Starting Value | Notes |
|-----------|---------------|-------|
| omega_max | 360-720 dps | 1-2 rev/s — tune based on motor capability |
| alpha | 1800-3600 dps/s | Reaches 360 dps in 100-200ms |
| Min angle for trapezoid | ~36° at 720/3600 | Below this → triangular profile |

**Example: 90° turn at 720 dps, 3600 dps/s:**
- t_accel = 720/3600 = 0.2s
- theta_accel = 720²/(2×3600) = 72°
- 2×72 = 144 > 90 → **triangular profile**
- t_accel = √(90/3600) = 0.158s
- omega_peak = 3600 × 0.158 = 569 dps
- t_total = 0.316s — **90° turn in ~320ms**

### 3.5 Combining Profile with FF+PID

```cpp
TrapezoidalProfile profile;
profile.compute(90.0, 720.0, 3600.0);
float t_start = millis() / 1000.0f;

void controlLoop() {
    float t = millis() / 1000.0f - t_start;
    
    float omega_setpoint = profile.getOmega(t);
    float omega_measured = (gyroData.zData - biasZ) / 1000.0f;  // dps
    
    // Feedforward
    float V_ff = kV * omega_setpoint + kS * sign(omega_setpoint);
    
    // PID on velocity error
    float V_pid = pid.compute(omega_setpoint - omega_measured, dt);
    
    // Optional: outer position loop for final angle accuracy
    float angle_error = profile.getTheta(t) - heading_deg;
    float V_pos = Kp_pos * angle_error;
    
    float V_total = V_ff + V_pid + V_pos;
    setMotors(-V_total, V_total);
}
```

Sources:
- [Trapezoidal Velocity Profiles (Medium)](https://medium.com/@christian_lozoya/trapezoidal-velocity-profile-f1892c720cd7)
- [WPILib Trapezoidal Profiles (FIRST Robotics)](https://docs.wpilib.org/en/stable/docs/software/advanced-controls/controllers/trapezoidal-profiles.html)

---

## 4. Mode-Switched Sensor Fusion

### 4.1 Why Not a Simple Complementary Filter

A standard complementary filter (`heading = 0.98*gyro + 0.02*cv`) blends continuously. This is suboptimal for combat because:
- When ArUco IS visible, CV heading is **ground truth** — why trust it only 2%?
- When ArUco is LOST, there's nothing to blend with — the 0.02 term is zero
- On re-acquisition, gradual correction wastes time correcting drift that we know exactly

### 4.2 Mode-Switched Architecture

Three modes with hard transitions:

**Mode 1: ArUco Visible (normal operation)**
- CV heading = ground truth (absolute, no drift)
- Gyro interpolates between camera frames (30Hz camera → 1kHz gyro fills the gaps)
- On each CV frame: `gyro_heading = cv_heading` (reset gyro to CV)
- Between CV frames: `heading = gyro_heading + ∫gyro_z dt` (gyro provides interpolation)

**Mode 2: ArUco Lost (occlusion, collision)**
- Gyro-only heading: `heading += gyro_z * dt`
- Expected drift: ~0.05-0.2°/min → **<0.02° error over 5 seconds**
- Increment `frames_without_cv` counter

**Mode 3: ArUco Re-acquired**
- **Instant snap-correction**: `gyro_heading = cv_heading` immediately
- No gradual blend — CV is absolute truth
- Reset `frames_without_cv = 0`

### 4.3 Implementation

```python
class HeadingFusion:
    def __init__(self):
        self.heading = 0.0           # fused heading (degrees)
        self.gyro_heading = 0.0      # gyro-integrated heading
        self.cv_heading = None       # last CV heading
        self.frames_without_cv = 0
        self.cv_available = False
    
    def update_gyro(self, gyro_z_dps: float, dt: float):
        """Called at IMU rate (~416 Hz via telemetry at ~100 Hz)."""
        self.gyro_heading += gyro_z_dps * dt
        self.heading = self.gyro_heading
    
    def update_cv(self, cv_heading_deg: float):
        """Called when ArUco is detected (~30 Hz)."""
        self.cv_heading = cv_heading_deg
        self.cv_available = True
        
        # INSTANT SNAP: trust CV absolutely
        self.gyro_heading = cv_heading_deg
        self.heading = cv_heading_deg
        self.frames_without_cv = 0
    
    def update_no_cv(self):
        """Called when ArUco detection fails this frame."""
        self.frames_without_cv += 1
        self.cv_available = False
        # heading continues from gyro integration (Mode 2)
    
    @property
    def is_cv_tracking(self) -> bool:
        return self.frames_without_cv < 5  # <~160ms
    
    @property
    def heading_confidence(self) -> float:
        """1.0 when CV is fresh, decays with time."""
        if self.frames_without_cv == 0:
            return 1.0
        # Decay based on expected gyro drift
        return max(0.0, 1.0 - self.frames_without_cv * 0.01)
```

### 4.4 Why This Works for Combat

| Scenario | Duration | Heading Error | Acceptable? |
|----------|----------|---------------|-------------|
| Normal tracking | Continuous | 0° (CV ground truth) | Yes |
| Brief occlusion (collision) | 0.5-2s | <0.03° | Yes |
| Extended occlusion | 5s | <0.15° | Yes |
| Very long occlusion | 30s | ~1° | Marginal |

Combat occlusions are typically <2 seconds. Gyro drift is irrelevant at this timescale.

---

## 5. Click-to-Point Navigation

### 5.1 OpenCV Mouse Callback

```python
click_target = None

def on_mouse(event, x, y, flags, param):
    global click_target
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert pixel to world coordinates via homography
        x_cm, y_cm = tracker.px_to_cm(x, y)
        click_target = (x_cm, y_cm)
        print(f"Click target: ({x_cm:.1f}, {y_cm:.1f}) cm")

cv2.setMouseCallback("Tracking", on_mouse)
```

The existing `tracker.px_to_cm()` in `prototypes/auto-drive/tracker.py` (lines 400-414) handles the homography transform.

### 5.2 Velocity Profiling for Maximum Speed

The current `PathFollower` uses distance-proportional throttle: `throttle = min(distance/30, 1.0)`. This is conservative. For "as fast as possible":

**Bang-coast-bang profile:**
1. **Full throttle** from start until braking distance
2. **Full brake** to stop at target

```python
def compute_throttle(distance_cm, speed_cm_s, max_speed, max_decel):
    braking_distance = speed_cm_s**2 / (2 * max_decel)
    
    if distance_cm <= braking_distance:
        return -1.0  # full brake
    else:
        return 1.0   # full throttle
```

With IMU accelerometer feedback, we can measure actual deceleration and tune the braking point precisely.

### 5.3 Full Click-to-Point Sequence

1. Click on camera feed → pixel to world coords
2. Compute heading to target: `atan2(dy, dx)`
3. **Turn phase**: Send `turn(heading_delta)` to ESP32 → gyro-assisted turn at 1kHz → done in <0.5s
4. **Drive phase**: Full throttle with IMU heading correction → brake at computed distance
5. **Fine adjust**: If heading specified, rotate to final heading

**Expected performance:** 60cm point-to-point in <3 seconds (0.3s turn + 2s drive + 0.5s settle).

---

## 6. Enemy Detection Without Markers

### 6.1 MOG2 Background Subtraction

```python
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=120,          # 2s at 60fps (adapts to lighting changes)
    varThreshold=30,      # higher = fewer false positives
    detectShadows=False   # save ~20% CPU
)
```

| Parameter | Combat Setting | Default | Rationale |
|-----------|---------------|---------|-----------|
| history | **120** | 500 | 2s adaptation, stopped robot stays foreground |
| varThreshold | **25-40** | 16 | Arena floor texture creates noise at 16 |
| detectShadows | **False** | True | Saves CPU, handle via morphology instead |
| nmixtures | **3** | 5 | Fewer Gaussians = faster, 3 sufficient for arena |

### 6.2 Detection Pipeline

```python
def detect_enemy(frame, bg_sub, our_robot_corners):
    # 1. Background subtraction
    fg_mask = bg_sub.apply(frame, learningRate=0.005)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    # 2. Morphological cleanup
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
    
    # 3. Exclude our robot (expand ArUco bbox)
    if our_robot_corners is not None:
        center = our_robot_corners.mean(axis=0).astype(int)
        cv2.circle(fg_mask, tuple(center), 40, 0, -1)  # mask out
    
    # 4. Contour filtering
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if not (400 < area < 8000): continue  # size filter
        
        x, y, w, h = cv2.boundingRect(c)
        if not (0.3 < w/h < 3.0): continue   # aspect ratio
        
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area > 0 and area/hull_area < 0.4: continue  # solidity
        
        M = cv2.moments(c)
        if M["m00"] > 0:
            return (M["m10"]/M["m00"], M["m01"]/M["m00"])
    
    return None  # no enemy detected
```

### 6.3 Contour Filter Thresholds

For 720p overhead camera, arena ~900px across:

| Filter | Value | Purpose |
|--------|-------|---------|
| Min area | 400 px² (~20×20) | Reject noise |
| Max area | 8000 px² (~90×90) | Reject arena features |
| Aspect ratio | 0.3 - 3.0 | Reject elongated shadows |
| Solidity | > 0.4 | Reject irregular shapes |

Sources:
- [OpenCV BackgroundSubtractorMOG2](https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)
- [OpenCV Contour Properties](https://docs.opencv.org/3.4/d1/d32/tutorial_py_contour_properties.html)

---

## 7. Interception & Pursuit Algorithms

> Full equations and implementations are in `docs/interception-pursuit-research.md`. This section summarizes the key algorithms.

### 7.1 Strategy Selection (FSM)

```
if distance > 1.0m and velocity_estimate_confident:
    → Proportional Navigation (N=4)
elif distance > 0.3m and have_velocity_estimate:
    → Lead Pursuit with intercept point
elif distance <= 0.3m:
    → Pure Pursuit (close range, max throttle)
elif enemy_lost and coast_frames < 15:
    → Drive to last predicted position
else:
    → Search pattern (spin/patrol)
```

### 7.2 Proportional Navigation (PN)

**Core equation:**
```
a_n = N × λ̇ × V_c
```

Where N=4 (navigation constant), λ̇ = LOS rotation rate, V_c = closing velocity.

**LOS rate (robust, avoids atan2):**
```
R = enemy_pos - our_pos
V_rel = enemy_vel - our_vel
λ̇ = (Rx × Vy_rel - Ry × Vx_rel) / |R|²
```

**Adapt to differential drive:**
```
omega = a_n / V_robot                    # lateral accel → turn rate
V_left  = V_forward - omega × L/2       # differential drive
V_right = V_forward + omega × L/2
```

### 7.3 Intercept Point (Quadratic Solver)

Given enemy at P_e with velocity V_e, our robot at P_r with max speed s:

```
a = |V_e|² - s²
b = 2 × (offset · V_e)        where offset = P_e - P_r
c = |offset|²

t = smallest positive root of at² + bt + c = 0
intercept_point = P_e + V_e × t
```

If discriminant < 0: no intercept possible → fall back to pure pursuit.

### 7.4 Kalman Filter for Enemy

4-state constant-velocity model: `x = [x, y, vx, vy]ᵀ`

| Parameter | Value |
|-----------|-------|
| Process noise σ_a | 5.0 m/s² (handles ~2-3g combat maneuvers) |
| Measurement noise σ | 0.01 m (CV centroid accuracy) |
| Max coast frames | 15 (~250ms at 60fps) |

**Adaptive process noise:** Monitor normalized innovation squared (NIS). If NIS > 6.0, enemy is maneuvering → quadruple Q temporarily. If NIS < 1.0, tracking well → decay Q toward baseline.

### 7.5 Smoothing the Intercept Point

EMA filter on intercept point to prevent jittery steering:
```python
smoothed = alpha * new_intercept + (1-alpha) * smoothed
```

| Distance | alpha | Behavior |
|----------|-------|----------|
| Far (>1m) | 0.2 | Smooth, predictable path |
| Medium | 0.3-0.4 | **Default** |
| Close (<0.3m) | 0.7+ | Maximum reactivity |

### 7.6 Wall Bounce Prediction

In the 8x8ft enclosed arena, enemies will hit walls. Simple elastic bounce model:
```python
if pred_x < 0 or pred_x > 2.44:
    vel_x = -vel_x  # bounce off side wall
if pred_y < 0 or pred_y > 2.44:
    vel_y = -vel_y  # bounce off end wall
```

---

## 8. Recommended Architecture

### 8.1 Full Pipeline

```
Camera Frame (30-60fps)
    │
    ├──> ArUco Detection ──> Our position + heading ──> Sensor Fusion
    │                             │                         │
    ├──> MOG2 Background Sub      │                    Fused Heading
    │         │                   │                    (1kHz via gyro)
    │    Contour Filter           │                         │
    │    Exclude Our Robot <──────┘                         │
    │         │                                             │
    │    Enemy Detection (or None)                          │
    │         │                                             │
    └──> Kalman Filter (predict always, update when detected)
              │
         Enemy pos + velocity
              │
    ┌─────────┴──────────┐
    │  Intercept Point   │
    │  (quadratic solver)│
    │  + EMA smoothing   │
    └─��───────┬──────────┘
              │
    Strategy FSM: SEARCH → ACQUIRE → INTERCEPT → CLOSE
              │
    Guidance Law (PN / Lead Pursuit / Pure Pursuit)
              │
    ┌─────────┴──────────┐
    │   ESP32 Command    │
    │  Mode 0: direct    │
    │  Mode 1: turn(Δθ)  │
    └──��──────┬──────────┘
              │
    ESP32 (1kHz loop): Gyro PID → Motor PWM
```

### 8.2 Command Protocol Extension

Extend existing 5-byte UDP packet to 8 bytes:

| Byte | Field | Type | Description |
|------|-------|------|-------------|
| 0-1 | throttle | int16 BE | -32767 to 32767, forward positive |
| 2-3 | steering | int16 BE | -32767 to 32767, right positive |
| 4 | buttons | uint8 | Bitmask |
| 5 | mode | uint8 | 0=direct, 1=gyro-turn |
| 6-7 | heading_delta | int16 BE | 0.01° units (Mode 1 only) |

Backward-compatible: if only 5 bytes received, treat as Mode 0 (direct).

### 8.3 Telemetry Protocol (ESP32 → PC)

UDP port 4211, 100 Hz, 20 bytes:

| Byte | Field | Type | Description |
|------|-------|------|-------------|
| 0-3 | heading | float32 LE | Integrated heading (degrees) |
| 4-7 | gyro_z | float32 LE | Raw gyro Z (dps) |
| 8-11 | accel_x | float32 LE | Accelerometer X (mg) |
| 12-15 | accel_y | float32 LE | Accelerometer Y (mg) |
| 16-19 | timestamp | uint32 LE | millis() on ESP32 |

### 8.4 Key Parameters Cheat Sheet

| Parameter | Value | Section |
|-----------|-------|---------|
| Gyro ODR | 416 Hz | 1.4 |
| Gyro FS | ±2000 dps | 1.4 |
| FF kV | ~0.05 (tune) | 2.2 |
| FF kS | ~0.8 (tune) | 2.2 |
| PID Kp | 0.01-0.05 | 2.3 |
| omega_max | 720 dps | 3.4 |
| alpha (angular accel) | 3600 dps/s | 3.4 |
| 90° turn time | ~320ms | 3.4 |
| PN constant N | 4 | 7.2 |
| Kalman σ_a | 5.0 m/s² | 7.4 |
| Kalman σ_meas | 0.01 m | 7.4 |
| Max coast frames | 15 | 7.4 |
| MOG2 history | 120 | 6.1 |
| MOG2 varThreshold | 25-40 | 6.1 |
| Intercept EMA alpha | 0.3 | 7.5 |
| Close range threshold | 0.3 m | 7.1 |

### 8.5 Latency Budget

| Stage | Time | Cumulative |
|-------|------|------------|
| Frame capture | 2 ms | 2 ms |
| ArUco detection | 3 ms | 5 ms |
| MOG2 + contours | 4 ms | 9 ms |
| Kalman + intercept + PN | 0.3 ms | 9.3 ms |
| UDP command TX | 1 ms | 10.3 ms |
| ESP32 gyro PID response | 2.4 ms | 12.7 ms |
| **Total** | **~13 ms** | **Well under 30ms target** |

---

## References

**IMU & Control:**
- [ISM330DHCX Datasheet (ST)](https://www.st.com/resource/en/datasheet/ism330dhcx.pdf)
- [SparkFun Micro 6DoF Product Page](https://www.sparkfun.com/sparkfun-micro-6dof-imu-ism330dhcx-qwiic.html)
- [SparkFun ISM330DHCX Hookup Guide](https://learn.sparkfun.com/tutorials/qwiic-6dof---ism330dhcx-hookup-guide/all)
- [SparkFun ISM330DHCX Arduino Library](https://github.com/sparkfun/SparkFun_6DoF_ISM330DHCX_Arduino_Library)
- [Minimal-Drift Heading with MEMS Gyro (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3787445/)
- [Feedforward+PID for Robot Turning (SAGE)](https://journals.sagepub.com/doi/10.1177/1729881419897678)
- [Differential Drive Kinematics (Columbia)](https://www.cs.columbia.edu/~allen/F15/NOTES/icckinematics.pdf)
- [WPILib Trapezoidal Profiles](https://docs.wpilib.org/en/stable/docs/software/advanced-controls/controllers/trapezoidal-profiles.html)

**Interception & Pursuit:**
- [Proportional Navigation (Wikipedia)](https://en.wikipedia.org/wiki/Proportional_navigation)
- [JHU APL - Homing Guidance Principles](https://secwww.jhuapl.edu/techdigest/content/techdigest/pdf/V29-N01/29-01-Palumbo_Principles_Rev2018.pdf)
- [PN Guidance for Robotic Interception (ResearchGate)](https://www.researchgate.net/publication/230444899)
- [Intercept Course Calculation (jaran.de)](http://jaran.de/goodbits/2011/07/17/calculating-an-intercept-course-to-a-target-with-constant-direction-and-velocity-in-a-2-dimensional-plane/)
- [Pure Pursuit Path Tracking (CMU, Coulter 1992)](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)

**Enemy Detection:**
- [OpenCV BackgroundSubtractorMOG2](https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)
- [OpenCV Contour Properties](https://docs.opencv.org/3.4/d1/d32/tutorial_py_contour_properties.html)
- [Kalman Filter for 2D Motion (cookierobotics.com)](https://cookierobotics.com/071/)
- [Adaptive Kalman Filter (MDPI)](https://www.mdpi.com/2079-9292/12/18/3887)
- [EMA Filters (mbedded.ninja)](https://blog.mbedded.ninja/programming/signal-processing/digital-filters/exponential-moving-average-ema-filter/)

> **Detailed implementations** with full pseudocode for interception algorithms: see `docs/interception-pursuit-research.md`
