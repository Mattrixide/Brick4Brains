# Implementation Research: ESP32 Firmware, DepthAI Pipelines & Reference Projects

This document covers implementation-level research for the Combat Robot Tracker project, focusing on concrete code patterns, example repositories, and reference architectures.

---

## 1. ESP32 Firmware Patterns for Combat Robot Motor Control

### 1.1 Reference Repositories

#### VectorSpaceHQ/VS_combat_robot
- **URL**: https://github.com/VectorSpaceHQ/VS_combat_robot
- **What it is**: Open-source, low-cost DIY combat robot kit for antweight (1lb) and beetleweight (3lb) classes
- **Architecture**: Transmitter/receiver pattern using ESP32 on both ends
- **Contents**: PCB designs (transmitter + receiver), firmware (C++, 66.4%), 3D-printable chassis (OpenSCAD, 22%)
- **Relevance**: Closest match to our project -- a complete ESP32-based combat robot control system. Good reference for PCB design and mechanical integration.

#### Roboost-Robotics/firmware
- **URL**: https://github.com/Roboost-Robotics/firmware
- **What it is**: Modular micro-ROS motor control firmware for ESP32, PlatformIO-based
- **Communication**: UDP over WiFi (micro-ROS transport layer)
- **Motor drivers supported**: L298N (implemented), VESC (in development)
- **Controllers**: Simple direct control + PID with half-quad encoder feedback
- **Kinematics**: 4-wheeled mecanum drive (implemented), 3-wheeled swerve (in development)
- **ROS integration**: Subscribes to `/cmd_vel`, publishes to `/odom`
- **Relevance**: Good reference for modular firmware architecture, but uses micro-ROS which adds complexity we don't need. The motor control and PID patterns are directly applicable.
- **Note**: Repository is marked as undergoing restructuring.

#### Ezward/Esp32CameraRover2
- **URL**: https://github.com/Ezward/Esp32CameraRover2
- **What it is**: Differential drive framework with closed-loop speed control, pose estimation, and go-to-goal behavior
- **Hardware**: ESP32Cam + L9110S motor driver + cheap robot car chassis
- **Key features**: PID speed control, wheel encoder feedback (LM393 optocoupler), pose estimation, GoToGoal behavior
- **Limitation**: Encoder pins share serial TX/RX -- must disable serial output when using encoders (`USE_WHEEL_ENCODERS=1`, `SERIAL_DISABLE=1`)
- **Relevance**: The closed-loop PID + pose estimation architecture is very relevant. GoToGoal behavior pattern could inform our strategy engine's motor command generation.

#### ESP32 Combat Robot Controller (Hackaday.io)
- **URL**: https://hackaday.io/project/202469-esp32-combat-robot-controller
- **What it is**: Custom PCB using ESP32-C3-MINI with BLE gamepad control
- **Motor driver**: Two onboard DRV8871 drivers (up to 2A each per channel)
- **Communication**: BLE with Xbox gamepad
- **Hardware features**: Debug LED, tactile pushbutton, I/O headers with 3.3V/GND
- **PCB**: Designed in EasyEDA, manufactured by JLCPCB
- **Relevance**: Hardware design reference for motor driver integration. DRV8871 is a good driver choice for beetleweight robots (compact, simple PWM control, 2A per channel).

### 1.2 Differential Drive Motor Control

#### Command Format Pattern
The standard approach for differential drive is to receive `(left_speed, right_speed)` commands:

```
// Typical command struct
struct MotorCommand {
    uint8_t  cmd_type;      // 0x01 = motor, 0x02 = config, 0xFF = e-stop
    uint16_t sequence;      // For detecting packet loss
    int16_t  left_speed;    // -1000 to +1000
    int16_t  right_speed;   // -1000 to +1000
    uint8_t  checksum;      // Simple XOR or CRC8
};
```

For ROS-based systems, the pattern is to receive `/cmd_vel` (linear.x, angular.z) and convert to wheel speeds:
```
left_speed  = (linear_x - angular_z * track_width / 2) / wheel_radius
right_speed = (linear_x + angular_z * track_width / 2) / wheel_radius
```

Our system does NOT need ROS, so direct `(left, right)` speed commands are simpler and lower latency.

#### Motor Driver Options for Beetleweight

| Driver | Current | Channels | Interface | Notes |
|--------|---------|----------|-----------|-------|
| **DRV8871** | 3.6A peak, 2A continuous | 1 (need 2) | 2-pin PWM (IN1/IN2) | Compact, simple, used in Hackaday combat robot project |
| **L298N** | 2A per channel | 2 | 4-pin (IN1-IN4 + ENA/ENB) | Cheap, widely available, but inefficient (1.4V drop) |
| **DRV8833** | 1.5A per channel | 2 | 4-pin PWM | Dual H-bridge, efficient, good for smaller motors |
| **L9110S** | 800mA per channel | 2 | 2-pin per motor | Low current, used in Ezward rover project |
| **TB6612FNG** | 1.2A per channel | 2 | 6-pin | Efficient MOSFET driver, good middle ground |

**Recommendation**: **DRV8871** (2x) for our beetleweight. 2A continuous is sufficient for 3lb class drive motors, and the simple 2-pin PWM interface minimizes firmware complexity.

### 1.3 PID Speed Control

#### ESP-IDF Official Example
- **URL**: https://github.com/espressif/esp-idf/tree/v5.2.1/examples/peripherals/mcpwm/mcpwm_bdc_speed_control
- **What it does**: Brushed DC motor speed control using MCPWM + PCNT (pulse counter) + PID
- **PWM frequency**: 25kHz (above audible range)
- **Timer resolution**: 10MHz
- **PID parameters**: Kp=0.6, Ki=0.4, Kd=0.2 (incremental calculation mode)
- **Control loop rate**: 100Hz (10ms timer callback)
- **Encoder**: Quadrature via PCNT peripheral with 1000ns glitch filter
- **Target**: 400 pulses per 10ms period

Key code pattern for PID update:
```c
// Every 10ms via timer callback:
int cur_pulse_count = pcnt_get_count();
int real_pulses = cur_pulse_count - last_pulse_count;
float error = target_speed - real_pulses;
float new_speed;
pid_compute(pid_ctrl, error, &new_speed);
bdc_motor_set_speed(motor, (uint32_t)new_speed);
last_pulse_count = cur_pulse_count;
```

**Recommendation**: Use the ESP-IDF MCPWM + PCNT approach for motor control if using ESP-IDF framework. For Arduino framework, use the ArduPID library with `analogWrite()` or `ledcWrite()` for PWM output.

#### PID Tuning Reference Values
From the Hackster.io differential drive project:
```
// Left wheel PID
float kp_l = 1.8, ki_l = 5.0, kd_l = 0.01;
// Right wheel PID
float kp_r = 2.25, ki_r = 5.0, kd_r = 0.01;
```
Note: PID values are motor-specific. These provide a starting point but will need tuning for our specific motors.

### 1.4 IMU Reading (BNO085 / BNO055)

#### BNO085 on ESP32 -- Critical Compatibility Issue
- **The BNO085's I2C implementation violates the I2C protocol** in some cases and does not work reliably with ESP32 or ESP32-S3 due to an I2C driver silicon bug.
- **SPI is the only reliable connection method** for BNO085 on ESP32.
- Best library: **esp32_BNO08x** (https://github.com/myles-parfeniuk/esp32_BNO08x) -- C++ esp-idf driver component, SPI only.
- The Adafruit BNO08x Arduino library can work but may have intermittent I2C failures on ESP32.

#### BNO055 on ESP32 -- More Reliable via I2C
- I2C connection works reliably (pins D21=SDA, D22=SCL on standard ESP32)
- Library: **Adafruit_BNO055** (Arduino) or **Bosch BNO055 driver** (esp-idf)
- Provides fused orientation data (Euler angles, quaternions) on-chip
- 100Hz output rate for fused data

#### Recommendation for Our Robot
Since the PRD specifies an onboard IMU on the ESP32 robot:
- **If using BNO085**: Connect via **SPI only**. Use the esp32_BNO08x library. This gives access to rotation vector, game rotation vector, and raw gyro/accel at up to 400Hz.
- **If using BNO055**: Connect via **I2C**. Simpler to wire, proven reliable on ESP32. On-chip sensor fusion provides heading directly. 100Hz fused output.
- **Simpler alternative**: MPU6050 via I2C (used in the Impacto_24 project). Much cheaper, 6-axis only (no magnetometer), requires host-side fusion. Good enough if we're doing sensor fusion on the host PC anyway.

### 1.5 UDP Command Parsing Pattern

Using the AsyncUDP library for non-blocking packet reception:

```cpp
#include <WiFi.h>
#include <AsyncUDP.h>

AsyncUDP udp;
const int CMD_PORT = 4210;
const int TELEMETRY_PORT = 4211;

unsigned long lastCmdTime = 0;
const unsigned long FAILSAFE_TIMEOUT_MS = 200;

void setup() {
    WiFi.begin(ssid, password);
    // ... wait for connection ...

    // Listen for motor commands
    udp.listen(CMD_PORT);
    udp.onPacket([](AsyncUDPPacket packet) {
        if (packet.length() == sizeof(MotorCommand)) {
            MotorCommand cmd;
            memcpy(&cmd, packet.data(), sizeof(cmd));
            if (validateChecksum(cmd)) {
                setMotorSpeeds(cmd.left_speed, cmd.right_speed);
                lastCmdTime = millis();
            }
        }
    });
}

void loop() {
    // Failsafe: stop motors if no command received
    if (millis() - lastCmdTime > FAILSAFE_TIMEOUT_MS) {
        setMotorSpeeds(0, 0);  // Emergency stop
    }

    // Send telemetry at 100Hz
    sendIMUTelemetry();
    delay(10);
}
```

### 1.6 Watchdog and Failsafe Patterns

#### Software Failsafe (Recommended for MVP)
Track `millis()` since last valid command. If timeout exceeds 200ms, set motors to zero. This is the simplest and most common pattern for RC/combat robots.

#### Hardware Watchdog (Additional Safety Layer)
ESP32 provides Task Watchdog Timer (TWDT):
```c
#include "esp_task_wdt.h"

// Initialize with 3-second timeout, panic on timeout
esp_task_wdt_init(3, true);
esp_task_wdt_add(NULL);  // Add current task

// In main loop, reset watchdog only when comms are healthy
if (comms_healthy) {
    esp_task_wdt_reset();
}
// If not reset within 3 seconds, ESP32 reboots (motors stop on reset)
```

**Recommendation**: Implement BOTH:
1. **Software failsafe** (200ms timeout) -- stops motors gracefully, allows quick recovery when comms resume
2. **Hardware watchdog** (3-second timeout) -- catches firmware crashes/lockups, forces full reboot

### 1.7 Recommended Firmware Architecture

```
ESP32 Firmware Architecture
============================

Main Task (Core 1)
  +-- WiFi connection manager (auto-reconnect)
  +-- UDP command listener (AsyncUDP, port 4210)
  |     +-- Parse binary command struct
  |     +-- Validate checksum
  |     +-- Update motor setpoints
  |     +-- Reset failsafe timer
  +-- Failsafe checker (200ms timeout)
  +-- LED status indicator

Motor Task (Core 0, 100Hz timer)
  +-- Read encoder pulses (PCNT)
  +-- PID compute per wheel
  +-- Set MCPWM duty cycle
  +-- Acceleration ramping

Telemetry Task (Core 0, 100Hz)
  +-- Read IMU (SPI for BNO085, I2C for BNO055)
  +-- Pack telemetry struct
  +-- Send UDP packet (port 4211)
  +-- Include battery voltage (ADC read)

Framework: PlatformIO + Arduino or ESP-IDF
```

---

## 2. DepthAI Pipeline Examples

### 2.1 Key Reference Repository

#### luxonis/depthai-python
- **URL**: https://github.com/luxonis/depthai-python
- **What it is**: Official Python API and examples for OAK-D cameras
- **Key examples for our project**:
  - `examples/StereoDepth/rgb_depth_aligned.py` -- RGB + depth alignment pipeline
  - `examples/ColorCamera/rgb_camera_control.py` -- Manual exposure, focus, white balance control
  - `examples/ColorCamera/rgb_preview.py` -- Basic camera streaming
  - `examples/host_side/latency_measurement.py` -- Measuring pipeline latency

#### ArchilChovatiya/ArUco-marker-detection-with-DepthAi
- **URL**: https://github.com/ArchilChovatiya/ArUco-marker-detection-with-DepthAi
- **What it is**: ArUco marker detection with OAK-D camera, generating 3D spatial position for markers
- **Dependencies**: depthai, opencv-contrib-python, numpy, scikit-learn
- **Architecture**: Camera streams to host, OpenCV ArUco detection runs on host, depth data provides 3D position
- **Relevance**: Directly demonstrates our exact use case -- ArUco detection + depth from an OAK-D camera

### 2.2 RGB + Stereo Depth Aligned Pipeline

This is the core pipeline pattern our system needs. From the official Luxonis example (`rgb_depth_aligned.py`):

```python
import depthai as dai
import cv2

# Configuration
FPS = 30  # Can set to 60 for our use case
MONO_RES = dai.MonoCameraProperties.SensorResolution.THE_720_P

# Create pipeline
pipeline = dai.Pipeline()

# --- Color Camera ---
camRgb = pipeline.create(dai.node.Camera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setSize(1280, 720)     # 720p for lower latency; use 1920,1080 for 1080p
camRgb.setFps(FPS)

# Fixed focus for overhead mounting (no auto-focus hunting)
# Read lens position from calibration data
calibData = device.readCalibration2()
lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
camRgb.initialControl.setManualFocus(lensPosition)

# --- Stereo Depth ---
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

left.setResolution(MONO_RES)
left.setCamera("left")
left.setFps(FPS)
right.setResolution(MONO_RES)
right.setCamera("right")
right.setFps(FPS)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align depth to RGB

# --- Output streams ---
rgbOut = pipeline.create(dai.node.XLinkOut)
rgbOut.setStreamName("rgb")
camRgb.video.link(rgbOut.input)

depthOut = pipeline.create(dai.node.XLinkOut)
depthOut.setStreamName("depth")
stereo.disparity.link(depthOut.input)

left.out.link(stereo.left)
right.out.link(stereo.right)
```

### 2.3 Low-Latency Pipeline Configuration

Key optimizations from Luxonis documentation:

```python
# 1. Set XLink chunk size to 0 for immediate transfer
# (transfers data as soon as available, no buffer filling)
pipeline.setXLinkChunkSize(0)

# 2. Use non-blocking queues with small max size
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

    # 3. Always grab the latest frame, discard old ones
    while True:
        rgbFrame = qRgb.tryGet()
        depthFrame = qDepth.tryGet()

        if rgbFrame is not None:
            frame = rgbFrame.getCvFrame()
            # Process with OpenCV...

# 3. Measure latency
latency = (dai.Clock.now() - imgFrame.getTimestamp()).total_seconds() * 1000
```

#### Latency Budget (Based on Luxonis Documentation)

| Component | Typical Latency | Notes |
|-----------|----------------|-------|
| Camera capture + ISP | ~16ms at 60FPS | One frame period |
| XLink transfer (USB3) | ~5ms at 1080p | Set chunk size to 0 |
| Host queue wait | ~0ms | Non-blocking, maxSize=1 |
| **Total pipeline** | **~21ms at 60FPS** | Best case |
| ArUco detection (host) | ~15-30ms | At 1080p, depends on CPU |
| **Total capture-to-detection** | **~36-51ms** | Within PRD 50ms target |

**Critical**: At 1080p, ISP output (YUV420) at 60FPS has ~33ms latency. Using `preview` output instead of `isp`/`video` can reduce this but changes the format.

#### Bandwidth Requirements

| Resolution | Format | FPS | Bandwidth |
|-----------|--------|-----|-----------|
| 1080p | NV12/YUV420 | 30 | 747 Mbps |
| 1080p | RGB | 30 | 1.5 Gbps |
| 1080p | NV12/YUV420 | 60 | 1.5 Gbps |
| 720p | NV12/YUV420 | 60 | 663 Mbps |

USB3 provides 5 Gbps, so 1080p/60fps is feasible but leaves less headroom for simultaneous depth streaming. **720p/60fps is safer** if running RGB + depth simultaneously.

### 2.4 Host-Side ArUco Detection with DepthAI Streaming

This is the pattern our system will use -- camera streams from OAK-D, ArUco detection runs on the host CPU:

```python
import depthai as dai
import cv2
import numpy as np

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
# Disable corner refinement for speed during combat
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Camera calibration matrix (from OAK-D calibration)
# Used for pose estimation
camera_matrix = np.array(...)  # From calibration
dist_coeffs = np.array(...)    # From calibration

def process_frame(frame, depth_frame):
    """Detect ArUco markers and get their positions."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            # Get marker center
            center = corners[i][0].mean(axis=0).astype(int)

            # Get depth at marker center (if depth frame available)
            if depth_frame is not None:
                depth_value = depth_frame[center[1], center[0]]

            # Estimate pose (rotation + translation)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[i:i+1], marker_length=0.076,  # 3 inches = 0.076m
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs
            )

            # rvecs gives orientation, tvecs gives 3D position
            # Convert to arena coordinates via homography

    return corners, ids
```

### 2.5 Manual Exposure Control for Motion Blur

Critical for fast-moving robot tracking:

```python
# Set manual exposure during pipeline setup
camRgb.initialControl.setManualExposure(500, 800)
# Parameters: exposure_time_us=500 (microseconds), iso=800

# Or adjust dynamically via control queue:
controlIn = pipeline.create(dai.node.XLinkIn)
controlIn.setStreamName("control")
controlIn.out.link(camRgb.inputControl)

# Then at runtime:
ctrl = dai.CameraControl()
ctrl.setManualExposure(500, 800)  # 500us exposure, ISO 800
device.getInputQueue("control").send(ctrl)
```

**Exposure time calculation**: At 8 ft/s robot speed and ~240 px/ft sensor resolution:
- Robot moves 1920 px/s on sensor
- 500us exposure = 0.96 pixels of motion blur (acceptable)
- May need ISO 800-1600 to compensate for reduced light

### 2.6 Pipeline Architecture for Our System

```
OAK-D Pro Pipeline
==================

On-Device (Myriad X VPU):
  ColorCamera (1080p or 720p, 60FPS)
    +-- video output --> XLinkOut "rgb"
    +-- [Optional] preview output --> NeuralNetwork (enemy detection, Phase 3)
  MonoCamera Left (720p, 60FPS)  --+
  MonoCamera Right (720p, 60FPS) --+--> StereoDepth
                                         +-- disparity --> XLinkOut "depth"
  [Optional] IR dot projector ON for active stereo

On-Host (Python):
  Thread 1: Frame Receiver
    +-- Get RGB frame from "rgb" queue
    +-- Get depth frame from "depth" queue
    +-- Timestamp + latency measurement

  Thread 2: ArUco Detector
    +-- Receive RGB frames
    +-- cv2.aruco.detectMarkers()
    +-- Estimate pose (rvec, tvec)
    +-- Transform to arena coordinates via homography
    +-- Publish robot position + heading

  Thread 3: Enemy Detector
    +-- Receive RGB + depth frames
    +-- Background subtraction (MOG2 or frame diff)
    +-- Depth thresholding
    +-- Exclude our robot's known position
    +-- Contour analysis for enemy position
    +-- Publish enemy position + velocity estimate

  Thread 4: Telemetry Receiver (UDP from ESP32)
    +-- Receive IMU data at 100Hz
    +-- Feed to sensor fusion / Kalman filter
```

---

## 3. Open-Source Combat Robot Autonomy & Tracking Projects

### 3.1 Directly Relevant Projects

#### pasinduanuradhaperera/Impacto_24 -- ArUco Robot Arena Navigation
- **URL**: https://github.com/pasinduanuradhaperera/Impacto_24
- **License**: MIT
- **What it is**: Autonomous robot using ArUco markers for arena navigation. Raspberry Pi-based with OpenCV, motor control, and IMU integration.
- **Architecture**:
  - `game.py` -- Main orchestrator
  - `detection.py` -- ArUco marker detection with OpenCV
  - `motor.py` -- Motor control via L298N driver
  - `encoder.py` -- Wheel encoder feedback (distance and direction)
  - `rotation.py` -- IMU-based rotation (MPU6050)
  - `utils.py` -- Utility functions
- **Key patterns**:
  - Camera calibration for accurate distance estimation from ArUco markers
  - Multi-stage navigation: detect marker -> calculate distance -> path correct -> search if lost
  - IMU for rotation correction during navigation
  - Encoder feedback for precise distance measurement
  - GPIO switch for calibration mode activation
  - Systemd service for startup automation
- **Relevance**: **HIGH** -- Very similar architecture to our system. Demonstrates ArUco detection + IMU fusion + motor control in an arena setting. Differences: their camera is onboard (not overhead), and they navigate TO markers rather than tracking robots.

#### Cymplecy/trackmqtt -- Overhead Arena Robot Tracking
- **URL**: https://simplesi.net/tracking-robot/
- **What it is**: Overhead camera tracking system using ArUco markers for robot position in an arena
- **Key architecture**:
  - 4 ArUco markers define arena corners (exactly our calibration approach)
  - Additional ArUco markers on robots for tracking
  - Overhead camera provides top-down view
  - Position data sent via MQTT to robots
- **Relevance**: **VERY HIGH** -- This is the closest existing project to our overhead tracking system. Same ArUco corner calibration + robot tracking pattern. Differences: uses MQTT instead of UDP, no combat/strategy component, no depth sensing.

#### yishaiSilver/aruco-slam -- ArUco SLAM
- **URL**: https://github.com/yishaiSilver/aruco-slam
- **What it is**: Detects ArUco markers, builds a map, uses EKF and Factor Graphs to localize camera position relative to markers
- **Techniques**: Extended Kalman Filters, Factor Graphs for optimization
- **Relevance**: The EKF implementation for ArUco-based localization is directly applicable to our sensor fusion approach.

### 3.2 Combat Robot Autonomy Projects

#### NHRL Autonomous Classes
- NHRL and some RCE events offer autonomous combat robot classes
- Autonomous robots at 1lb and 3lb weights fight on dedicated arenas
- Most autonomous combat bots use **onboard sensors** (IR proximity, ultrasonic, line sensors) rather than overhead vision
- The overhead camera approach is novel for combat robotics -- no well-known open-source projects doing exactly what we're doing

#### Autonomous Sumo Robots (Closest Analog)
Several open-source sumo robots provide relevant patterns:

| Project | URL | Sensors | Strategy |
|---------|-----|---------|----------|
| **Omus** | https://github.com/advra/Omus | IR sensors, custom PCB, PIC32, brushless motors | Autonomous sumo with dual IR detection |
| **BatmanAmman** | https://github.com/mozaloom/sumo-robot-batman-amman | IR, line sensors | Opponent detection + strategic movement |
| **AIMovement/sumo** | https://github.com/AIMovement/sumo | Various | AI/ML/RL techniques for sumo strategy |

**Key difference from our approach**: Sumo bots use onboard sensors and reactive strategies. Our system uses external overhead vision with a centralized strategy engine -- a fundamentally different (and potentially superior) architecture since we have a global view of the arena.

### 3.3 General Autonomous Robot Vision Projects

#### Team254/CheesyVision
- **URL**: https://github.com/Team254/CheesyVision
- **What it is**: FRC team's system to signal robots during autonomous mode using webcam + computer vision
- **Relevance**: Demonstrates the pattern of using an external camera to communicate state to a robot during competition. Different domain (FRC) but similar concept.

#### sergionr2/RacingRobot
- **URL**: https://github.com/sergionr2/RacingRobot
- **What it is**: Autonomous racing robot with Raspberry Pi + camera
- **Relevance**: Computer vision-based autonomous navigation at speed. Demonstrates handling motion blur and real-time decision making.

### 3.4 Resource Lists

#### IainIsCreative/awesome-robot-combat
- **URL**: https://github.com/IainIsCreative/awesome-robot-combat
- **What it is**: Curated list of combat robot resources, parts suppliers, events, and communities
- **Includes**:
  - Parts suppliers: Itgresa (Black Frost kit), Absolute Chaos (beetle parts), Ranglebox (featherweight/beetle components)
  - Events: BattleBots, NHRL, RoboGames
  - Communities: Out Of The Arena Discord, NHRL Discord
- **Relevance**: Useful for sourcing parts and connecting with the combat robot community

#### NHRL Combat Robot Design Handbook
- **URL**: https://wiki.nhrl.io/wiki/index.php/The_Combat_Robot_Design_Handbook
- **Relevance**: Official design guide from the largest US robot combat league. Covers mechanical design, electronics, and safety requirements.

---

## 4. Key Takeaways and Implementation Recommendations

### 4.1 ESP32 Firmware

1. **Use PlatformIO + Arduino framework** for faster development. ESP-IDF is more powerful but has a steeper learning curve. Arduino's AsyncUDP and Wire libraries make WiFi + IMU integration straightforward.

2. **Motor control**: DRV8871 (2x) + MCPWM or ledc PWM at 25kHz. If using encoders, use the ESP32's hardware PCNT peripheral for accurate pulse counting.

3. **PID control**: Start with Kp=1.5, Ki=3.0, Kd=0.01 and tune from there. Run PID at 100Hz in a timer interrupt. Consider starting with open-loop control for MVP and adding PID in Phase 2.

4. **IMU**: If using BNO085, MUST use SPI (not I2C) due to ESP32 silicon bug. BNO055 works fine on I2C. MPU6050 is cheapest option if host-side fusion is planned.

5. **Failsafe**: Implement both software timeout (200ms, stops motors) AND hardware watchdog (3s, reboots ESP32). The software timeout should be the primary safety mechanism.

6. **Disable WiFi power saving** (`WiFi.setSleep(false)` in Arduino, or `esp_wifi_set_ps(WIFI_PS_NONE)` in ESP-IDF) to prevent latency spikes.

### 4.2 DepthAI Pipeline

1. **Use 720p/60fps** for the initial pipeline to maximize frame rate while keeping bandwidth under USB3 limits when running RGB + depth simultaneously. Upgrade to 1080p/60fps if the USB link supports it.

2. **Set XLink chunk size to 0** and use non-blocking queues with `maxSize=1` for lowest latency.

3. **ArUco detection runs on host** -- the VPU's Script Node cannot run OpenCV ArUco detection. This is the correct architecture; don't try to run it on-device.

4. **Manual exposure** at 400-500us with ISO 800-1600 is critical for reducing motion blur. Tune ISO based on arena lighting conditions.

5. **Fixed focus** (not auto-focus) for overhead mounting. Set manual focus to the calibrated lens position at startup.

6. **Active stereo** (IR dot projector) should be enabled to improve depth accuracy on the textureless white floor.

### 4.3 Architecture

1. Our **overhead camera + centralized strategy** approach is novel in combat robotics. No existing open-source project does exactly this. We are combining:
   - Overhead ArUco tracking (from projects like trackmqtt)
   - IMU sensor fusion (from projects like aruco-slam and Impacto_24)
   - Autonomous combat strategy (from sumo bot AI projects)
   - ESP32 motor control (from combat robot controller projects)

2. The **Impacto_24** project is the closest architectural reference, but with an onboard camera. Our overhead approach gives us a significant advantage: global view of both robots, no camera occlusion during collisions, and centralized processing power.

3. The **modular firmware architecture** from Roboost (separate motor control, communication, and telemetry tasks on different cores) is the right pattern for our ESP32 firmware.
