# Product Requirements Document: Brick for Brains — Autonomous Combat Robot System

## Overview

**Brick for Brains** — an autonomous combat robotics system that uses computer vision to track robots in an arena, make strategic decisions, and control our robot to push the opponent into a corner pit — all without human intervention during a match.

---

## 1. System Context

### Arena Specifications
- **Dimensions**: 8ft × 8ft enclosed arena
- **Walls**: Polycarbonate, 4ft tall
- **Ceiling**: Opaque, with LED lighting from above (top-down illumination)
- **Floor**: White surface with a logo in the center
- **Pit**: Located in one corner; robots pushed into the pit are eliminated

### Hardware
| Component | Details |
|-----------|---------|
| **Camera** | OAK-D Pro (Luxonis) — RGB + stereo depth, mounted on freestanding rig above arena wall height, looking down at ~24-degree angle |
| **PC** | Host machine running the vision pipeline, strategy engine, and web dashboard |
| **Our Robot** | ESP32 microcontroller, WiFi-controlled, onboard IMU, ArUco tags on top and bottom |
| **Enemy Robot** | Human-controlled, no ArUco tags, unknown appearance |

### Communication
- PC ↔ ESP32: WiFi (UDP or TCP commands)
- ESP32 → PC: IMU telemetry stream (orientation, acceleration)
- PC: Runs all vision processing, strategy, and the web dashboard

---

## 2. Functional Requirements

### 2.1 Arena Calibration

**On initialization**, the operator manually places a single ArUco tag at each of the four corners sequentially. The system captures each placement to establish:

- Arena boundary coordinates in camera space
- Homography transform: camera pixels → arena coordinate system (0,0 to 8,8 in feet)
- Pit location and boundary (identified by a dedicated ArUco tag placed at/near the pit)

**Requirements:**
- Guided calibration flow: system prompts operator for each corner placement in order
- Visual confirmation that each corner was captured correctly
- Persist calibration data so it survives restarts within a session
- Manual adjustment of bounds via the web dashboard post-calibration
- Re-calibration option without full restart

### 2.2 Robot Tracking — Our Robot (ArUco-based)

Our robot has ArUco markers on its top and bottom surfaces.

**Requirements:**
- Detect ArUco tag from the overhead camera to determine:
  - **Position** (x, y) in arena coordinates
  - **Orientation** (heading angle) from tag rotation
- Fuse with IMU data from the ESP32 for higher-frequency orientation updates between camera frames
- Target tracking rate: ≥30 Hz effective position updates (camera + IMU fusion)
- Handle temporary occlusion (e.g., robots colliding, dust) with prediction/extrapolation
- Bottom ArUco tag serves as backup if robot flips

### 2.3 Robot Tracking — Enemy Robot (Vision-based)

The enemy robot has no markers. Detection must rely on visual features.

**Requirements:**
- Detect the enemy robot using contrast against the white floor
- Use depth data from OAK-D Pro stereo to distinguish robot from floor features (logo, shadows)
- Track enemy position (x, y) and estimate heading from motion vector
- Handle cases where robots are overlapping or adjacent (use our robot's known position to disambiguate)
- Estimate enemy velocity and acceleration for strategic prediction
- Must work with varying robot shapes/colors across different opponents

### 2.4 Autonomous Strategy Engine

The system decides how to move our robot to push the opponent into the pit.

**Requirements:**
- **Inputs**: Our robot pose (x, y, heading), enemy pose (x, y, estimated heading), enemy velocity, pit location, arena bounds
- **Outputs**: Motor commands sent to ESP32 (direction, speed, rotation)
- **Core behaviors**:
  - **Pursue**: Close distance to enemy robot
  - **Position**: Maneuver to get behind/beside the enemy relative to the pit
  - **Push**: Drive into the enemy to push it toward the pit
  - **Retreat**: Back away if in a disadvantageous position
  - **Avoid pit**: Never drive our own robot into the pit
- Strategy should be configurable via the web dashboard (aggression level, preferred approach angle, retreat thresholds)
- State machine or behavior tree architecture for clear strategy logic
- Real-time strategy decisions at ≥10 Hz

### 2.5 ESP32 Communication

**Requirements:**
- WiFi link between PC and ESP32
- Command protocol for motor control:
  - Differential drive or mecanum commands (depending on robot drivetrain)
  - Speed and direction per motor/side
  - Emergency stop command
- Telemetry stream from ESP32 → PC:
  - IMU data (gyro, accelerometer) at high rate (≥100 Hz)
  - Battery voltage
  - Connection health/heartbeat
- Latency target: <20ms round-trip for motor commands
- Auto-reconnect on WiFi dropout
- Failsafe: robot stops if communication lost for >200ms

### 2.6 Web-Based Admin Dashboard

A browser-based interface for configuration, monitoring, and debugging.

#### Pages:

**Live View**
- Real-time top-down arena visualization showing:
  - Arena boundaries and pit location
  - Our robot position, heading, and trail
  - Enemy robot position, heading, and trail
  - Current strategy state (pursue/position/push/retreat)
- Raw camera feed overlay option
- FPS and latency indicators

**Calibration**
- Guided calibration wizard
- Manual adjustment of corner positions (drag-and-drop on arena map)
- Pit boundary adjustment
- Save/load calibration profiles

**Strategy Configuration**
- Adjust strategy parameters (aggression, approach angles, retreat thresholds, speed limits)
- Select between strategy presets
- Enable/disable specific behaviors
- Visualize strategy decision boundaries on the arena map

**Testing & Debug**
- Manual motor control (virtual joystick / WASD) — override autonomous mode
- Raw sensor data display (IMU values, camera FPS, depth map)
- ArUco detection visualization (show detected markers with IDs and poses)
- Enemy detection visualization (show contours, bounding boxes, confidence)
- Latency and timing graphs
- Log viewer with filtering
- Record/playback for match analysis

**Connection Status**
- ESP32 connection state and signal strength
- Camera connection state
- System resource usage (CPU, memory, GPU)

---

## 3. Non-Functional Requirements

### Performance
- Vision pipeline must process at ≥30 FPS on the host PC
- End-to-end latency (camera capture → motor command sent): <50ms
- Web dashboard updates at ≥10 FPS for live visualization

### Reliability
- System must handle and recover from: camera frame drops, WiFi interruptions, ArUco detection failures
- Failsafe behaviors must activate within 200ms of any critical failure
- No single dropped frame should cause erratic robot behavior (use smoothing/prediction)

### Usability
- Calibration process completable in <60 seconds
- Dashboard accessible from any device on the local network
- System operational with minimal setup between matches

---

## 4. High-Level Architecture

### 4.1 System Overview

The system uses a **multiprocessing pipeline architecture** where CPU-bound stages run in separate OS processes to bypass Python's GIL. Frame data is shared via shared memory (zero-copy), and structured messages flow through lock-free queues.

```
+------------------------------------------------------------------------+
|                              Host PC                                    |
|                                                                         |
|  PROCESS GROUP A: Real-Time Pipeline (3 processes)                      |
|  +-------------+  shared   +--------------+  queue  +---------------+   |
|  | P1: Camera   |--memory-->| P2: Detection |------->| P3: Strategy   |  |
|  |   Capture    |           |   + Tracking  |        |   + Control    |  |
|  |              |           |               |        |                |  |
|  | - DepthAI    |           | - ArUco Det.  |        | - State Est.   |  |
|  | - RGB+Depth  |           | - Enemy Det.  |        | - Behavior     |  |
|  | - IMU ingest |           | - Homography  |        |   Tree         |  |
|  | - Frame ring |           | - BG Sub.     |        | - Motion Plan  |  |
|  |   buffer     |           | - MOSSE track |        | - Motor Cmds   |  |
|  +------+-------+           +------+--------+        +------+---------+  |
|         |                          |                        |            |
|         | telemetry                | detections             | commands   |
|         v                          v                        v            |
|  +------------------------------------------------------------------+   |
|  |                     Shared Message Bus (queues)                    |   |
|  +----------+---------------------------+-----------+----------------+   |
|             |                           |           |                    |
|  PROCESS GROUP B: Support Services                  |                    |
|  +----------v--------+  +--------------v---+  +-----v--------------+    |
|  | P4: Web Dashboard  |  | P5: Comms Layer  |  | P6: Telemetry      |   |
|  |   Backend          |  |   (ESP32 I/O)    |  |   & Logging        |   |
|  |                    |  |                  |  |                    |   |
|  | - FastAPI          |  | - UDP send/recv  |  | - Metrics          |   |
|  | - WebSocket        |  | - Heartbeat      |  | - Recording        |   |
|  | - REST API         |  | - Failsafe       |  | - Replay engine    |   |
|  +--------------------+  +--------+---------+  +--------------------+   |
|                                    |                                     |
+------------------------------------+-------------------------------------+
                                     | WiFi (UDP)
                               +-----v-----+
                               |  ESP32     |
                               | - Motors   |
                               | - IMU      |
                               | - Battery  |
                               +-----------+
```

### 4.2 Module Dependency Diagram

```
                      +-------------------+
                      | Arena Calibration  |
                      | (run once/setup)   |
                      +--------+----------+
                               | homography matrix
                               v
+----------+  frame  +--------------+  pixel coords  +--------------+
| Camera   |-------->| ArUco        |--------------->| State        |
| Capture  |         | Detector     |                | Estimator /  |
|          |---+     +--------------+                | Tracker      |
+----------+   |                                     |              |
               |     +--------------+  pixel coords  | - Kalman     |
               +---->| Enemy        |--------------->|   Filter     |
                     | Detector     |                | - IMU Fusion |
                     +--------------+                +------+-------+
                                                            | world state
                                                            v
                                                   +---------------+
                                                   | Strategy      |
                                                   | Engine        |
                                                   | (Behav. Tree) |
                                                   +-------+-------+
                                                           | desired motion
                                                           v
                                              +--------------------+
                                              | Motion Planner     |
                                              | (Pure Pursuit +    |
                                              |  Potential Fields)  |
                                              +--------+-----------+
                                                       | velocity cmds
                                                       v
                                              +--------------------+
                                              | Motor Controller   |
                                              | (PID per wheel)    |
                                              +--------+-----------+
                                                       | motor PWM
                                                       v
                                              +--------------------+
                                              | Comms Layer        |
                                              | (UDP to ESP32)     |
                                              +--------------------+
```

### 4.3 Data Flow -- Real-Time Pipeline

The critical path from camera frame to motor command flows through 3 processes. Pipeline stages overlap so each frame's total latency is ~25ms while throughput matches the camera frame rate:

```
Time --------------------------------------------------------------------->

P1 Camera:  [capture F1]  [capture F2]  [capture F3]  ...
               |              |              |
               v (shared mem) v              v
P2 Detect:     [detect F1]  [detect F2]  [detect F3]  ...
                  |              |              |
                  v (queue)      v              v
P3 Strategy:      [decide F1]  [decide F2]  [decide F3]  ...
                     |              |              |
                     v (queue)      v              v
P5 Comms:            [send F1]  [send F2]  [send F3]  ...

Pipeline latency: ~25ms total (each stage overlaps with the next)
Pipeline throughput: 1 frame per ~17ms at 60 FPS capture
```

### 4.4 Shared Memory Layout

```
SharedMemory "arena_frames" (preallocated ring buffer):
+----------------------------------------------------------+
| Header (64 bytes)                                        |
|  - write_index: uint32 (which slot the producer writes)  |
|  - frame_count: uint64 (monotonic counter)               |
|  - timestamp_ns: uint64 (capture time)                   |
|  - flags: uint32 (ready, dropped, etc.)                  |
+----------------------------------------------------------+
| Slot 0: RGB frame (640x480x3 = 921,600 bytes)            |
|         Depth frame (640x480x2 = 614,400 bytes)          |
|         Metadata (64 bytes: timestamp, seq, exposure)     |
+----------------------------------------------------------+
| Slot 1: (same layout)                                     |
+----------------------------------------------------------+
| Slot 2: (same layout)                                     |
+----------------------------------------------------------+
| Slot 3: (same layout)                                     |
+----------------------------------------------------------+
Total: ~6 MB for 4-slot ring buffer (RGB + Depth)
```

Note: Resolution shown is for the detection pipeline input. The camera captures at 1080p but frames are downscaled to 640x480 for detection processing to minimize latency. Full-resolution frames are optionally stored for the dashboard and recording.

### 4.5 Process / Thread Architecture

| Process | Threads | CPU Affinity | Priority | Purpose |
|---------|---------|-------------|----------|---------|
| P1: Camera Capture | 2 (DepthAI callback + IMU listener) | Core 0 | High | Frame acquisition, IMU ingest |
| P2: Detection | 2 (ArUco + Enemy, parallel) | Core 1-2 | High | All computer vision detection |
| P3: Strategy + Control | 1 (tight loop) | Core 3 | Realtime | State estimation, decisions, motor output |
| P4: Web Dashboard | 3 (FastAPI uvicorn workers) | Any | Normal | HTTP/WebSocket serving |
| P5: Comms Layer | 2 (send thread + recv thread) | Any | High | UDP to/from ESP32 |
| P6: Telemetry | 1 (async writer) | Any | Low | Logging, metrics, recording |

### 4.6 Interface Contracts Summary

All inter-module data uses **typed dataclasses** with `msgpack` serialization for queue transport, or **NumPy arrays via shared memory** for frame data.

```python
# Core data types flowing between modules:

@dataclass
class FramePacket:
    frame_id: int
    timestamp_ns: int
    rgb: np.ndarray          # via shared memory, not copied
    depth: np.ndarray        # via shared memory, not copied

@dataclass
class RobotDetection:
    robot_id: str            # "ours" or "enemy"
    pixel_xy: Tuple[float, float]
    arena_xy: Tuple[float, float]
    heading_rad: float
    confidence: float
    timestamp_ns: int

@dataclass
class WorldState:
    our_robot: RobotState    # position, velocity, heading, angular_vel
    enemy_robot: RobotState
    pit_location: Tuple[float, float]
    arena_bounds: Tuple[float, float, float, float]
    timestamp_ns: int

@dataclass
class MotorCommand:
    left_speed: float        # -1.0 to 1.0
    right_speed: float       # -1.0 to 1.0
    timestamp_ns: int
    ttl_ms: int              # command expires after this many ms
```

### 4.7 Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Vision Pipeline | Python + DepthAI SDK | Native OAK-D Pro support, OpenCV ArUco built-in |
| Frame Transport | `multiprocessing.shared_memory` | Zero-copy frame sharing between processes |
| Message Transport | `multiprocessing.Queue` | Structured data between processes, thread-safe |
| State Estimation | FilterPy (Python) | EKF/UKF implementations, NumPy-based, fast enough at 30-60 Hz |
| Strategy Engine | Python + `py_trees` | Behavior tree library with tick-based execution |
| ESP32 Comms | Python (asyncio UDP) | Low-latency, async-friendly |
| Backend API | FastAPI (Python) | WebSocket support for live data, REST for config |
| Web Dashboard | React + TypeScript | Rich interactive UI, WebSocket for real-time updates |
| ESP32 Firmware | Arduino/PlatformIO (C++) | Standard ESP32 toolchain |
| Serialization | `msgpack` | Fast binary serialization for queue messages |
| Metrics | Prometheus client + custom | Latency histograms, FPS counters, pipeline health |

---

## 5. MVP Scope (Phase 1)

To get a working system on the arena floor:

1. **Arena calibration** via ArUco corner placement
2. **Our robot tracking** via ArUco tag detection
3. **Enemy robot tracking** via background subtraction on white floor + depth
4. **Basic pursuit strategy** — drive toward enemy, push toward pit
5. **ESP32 motor commands** over WiFi
6. **Minimal dashboard** — live arena view + manual override joystick + connection status

### Phase 2 Additions
- Full strategy engine with configurable behaviors
- IMU sensor fusion
- Match recording and playback
- Strategy configuration UI
- Advanced enemy tracking (Kalman filter, multi-hypothesis)

### Phase 3 Additions
- Machine learning-based enemy behavior prediction
- Automated strategy tuning from recorded matches
- Multi-camera support for redundancy

---

## 6. Research Findings

### 6A. Robot Combat Events (RCE)

#### Weight Classes and Event Format
- Robot Combat Events (robotcombatevents.com) hosts events across multiple weight classes, including a "Gladiator" class for Viper kits and sportsman antweights. The most common competitive classes at local/regional events are **1lb (antweight)** and **3lb (beetleweight)**.
- NHRL (National Havoc Robot League) — a major US league — runs **3lb, 12lb, and 30lb** classes.
- Registration at RCE events typically costs $25/bot, max 4 bots per team, max 2 per class, different drivers required per class.

#### Arena Specifications
- **Beetleweight (3lb) arenas are typically 8ft x 8ft** with a steel floor, confirming the PRD assumption. Walls are double-layered polycarbonate for safety.
- Antweight arenas are smaller (5ft x 6ft or 4ft x 4ft). Some events use even smaller arenas (2ft x 2ft with hazards) for the lightest classes.
- Pits are located in corners with a pronounced lip to prevent accidental drive-ins. The minimum recommended pit gap is 140mm (~5.5 inches). Some arenas feature a 1ft-square modular pit.
- At least 25% of the arena edge must be un-walled in some rule sets, allowing robots to be pushed off the edge.
- Walls are typically 3-4ft tall polycarbonate with steel framing.

#### Match Rules
- **Match duration: 3 minutes** (may be shortened to 2 minutes under time pressure at some events).
- A robot touching the bottom of the pit has lost. If a robot bounces into and out of the pit without touching the bottom, the fight continues.
- If a robot is immobile or lacks controlled motion, a judge starts a **10-second countdown**. If motion is not restored, that robot loses.
- Judging criteria (when both robots survive full duration): **Damage, Control, and Aggression** — each worth 1 point, winner must take at least 2 of 3 categories.
- Pinning is limited to **5 seconds**; grappling limited to **15 seconds**.

#### Autonomous Robot Rules
- Some events offer a separate autonomous class (1lb and 3lb). Autonomous robots fight on a **3ft diameter raised black surface** with a 1-inch white border (different from standard arenas).
- Autonomous robots must be **fully autonomous** — a radio-control device is allowed ONLY for failsafe (emergency stop), not for any control input.
- **Key finding**: The standard beetleweight arena (8x8, white floor, corner pit) is for human-controlled robots. If entering an autonomous class, the arena format may differ. Confirm with the specific event whether autonomous bots fight in the standard arena or a dedicated autonomous arena.

#### Common Robot Designs (Beetleweight)
- **Wedge/Lifter**: Low-profile wedge to get under opponents, sometimes with an active lifter mechanism. Good for pushing opponents into pits.
- **Horizontal Spinner**: Spinning bar or disc weapon. High damage potential but vulnerable to being flipped.
- **Vertical Spinner**: Drum or disc spinning vertically. Popular and effective at all weight classes.
- **Flipper**: Pneumatic or spring-loaded mechanism to launch opponents.
- Typical beetleweight drive speed: **50-100 inches/second (4-8 ft/s)**. This is critical for tracking system design — objects move fast.

#### PRD Assumption Check
- **8x8ft arena: CONFIRMED** for beetleweight class.
- **Corner pit: CONFIRMED** as standard feature.
- **White floor: CONFIRMED** as common (though some events use other surfaces).
- **Polycarbonate walls: CONFIRMED**, typically double-layered.
- **No restrictions found** against external overhead cameras or sensors for standard remote-controlled classes, but autonomous class rules may impose constraints. Verify with specific event organizers.

---

### 6B. OAK-D Pro Camera Capabilities

#### RGB Camera
- **Sensor**: IMX378, 12MP (4056x3040)
- **Video encoding**: 4K/30FPS, 1080p/60FPS, H.264/H.265/MJPEG
- **Standard FOV (OAK-D Pro)**: 81 DFOV, 69 HFOV, 55 VFOV
- **Wide-angle FOV (OAK-D Pro W)**: 150 DFOV for stereo pair; 120 or 150 for RGB depending on variant
- **Auto-focus and fixed-focus** variants available. Fixed-focus recommended for overhead mounting (no hunting).
- **Manual exposure control**: 1 to 33,000 microseconds. Critical for reducing motion blur on fast-moving robots.

#### Stereo Depth
- **Baseline**: 7.5cm between left and right stereo cameras
- **Stereo sensors**: OV9282 (global shutter, 1280x800)
- **Depth range**: Ideal 70cm - 12m; MinZ ~20cm (400P + extended disparity), ~35cm (400P or 800P + extended disparity), ~70cm (800P standard)
- **Depth accuracy**: Below 2% error at 70cm with good texture (~1.5cm error)
- **Active stereo**: IR laser dot projector improves depth on textureless surfaces (like the white arena floor)
- **IR illumination LED**: Enables night-vision mode
- **Depth resolutions**: 400P (640x400) and 800P (1280x800)

#### IMU
- **Sensor**: BNO085, 9-axis IMU (accelerometer + gyroscope + magnetometer)
- **Max frequencies**: 500Hz raw accelerometer, 1000Hz raw gyroscope, 500Hz combined/synced
- **Note**: This is the camera's onboard IMU, separate from the robot's onboard IMU. Could be used for camera vibration compensation if needed.

#### VPU (Myriad X)
- **Processing power**: 4 TOPS total, 1.4 TOPS for AI inference
- **On-device capabilities**: Neural network inference (any OpenVINO model), H.264/H.265 encoding, image manipulation (warp, dewarp, resize, crop), edge detection, feature tracking
- **Script Node**: Runs lightweight Python scripts on-device for pipeline flow control, but NOT suitable for heavy CV operations (no ArUco detection on-device)
- **ArUco detection**: Must run on the HOST, not on the VPU. The VPU's Script Node is too limited for OpenCV ArUco detection. The camera streams frames to the host where OpenCV processes them.
- **Pre-trained model zoo**: 200+ models available (MobileNet-SSD, YOLO, etc.) that can run on the VPU for object detection

#### Pipeline Latency
- **Best case**: ~5.2ms average latency (frame capture to host receipt) at lower resolutions
- **1080p streaming**: Low latency achievable with USB3 (5-10Gbps bandwidth)
- **4K streaming**: Latency increases significantly — 150ms at 8 FPS, 530ms at 10 FPS when USB link is saturated
- **Recommendation**: Use 1080p/60FPS or 720p for lowest latency. Avoid 4K for real-time tracking.
- **Low-latency tips from Luxonis**: Disable power saving, use USB3, minimize pipeline complexity, use the "zero-copy" branch if available

#### Lens Selection for Overhead Arena Coverage
- **Standard lens (OAK-D Pro)**: 69 HFOV. At an 8ft mounting height above an 8ft arena, the horizontal coverage would be approximately `2 * 8ft * tan(69/2)` = ~11.3ft. This covers the full 8ft arena width with margin.
- **Wide-angle lens (OAK-D Pro W)**: 120+ HFOV. Covers a much larger area but introduces barrel distortion at edges, reducing marker detection accuracy at the periphery. Also, pixels-per-foot decreases, making ArUco markers harder to resolve.
- **Mounting height calculation (standard lens)**: To cover 8ft width with 69 HFOV, minimum height = `4ft / tan(34.5)` = ~5.8ft above the arena floor. With margin, **6-7ft above the arena floor** is optimal. Since walls are 4ft tall, this means mounting 10-11ft above the ground (4ft walls + 6-7ft above floor).

#### Power and Connectivity
- USB-C, supports USB2 and USB3
- Total power: up to ~7.5W at full utilization
- Powered via USB-C from host PC

---

### 6C. CV Motion Tracking for Fast Robots

#### Tracking Speed Requirements
- Beetleweight robots move at **4-8 ft/s** (48-96 inches/second).
- At 30 FPS, a robot moving at 8 ft/s travels **3.2 inches per frame**. At 60 FPS, this halves to **1.6 inches per frame**.
- Higher frame rates significantly improve tracking continuity. 60 FPS is strongly recommended.

#### ArUco Detection

**Dictionary Selection:**
- **DICT_4X4_50 recommended** for this application. Reasons:
  - Fewest bits per marker = largest effective cell size = more robust detection at distance
  - Only need a handful of unique IDs (our robot top, our robot bottom, 4 calibration corners, pit marker = ~7 markers total). 50 IDs is more than sufficient.
  - Higher inter-marker distance (less chance of false ID matches) with smaller dictionaries
  - Better detection range and robustness to blur/rotation than 5x5 or 6x6

**Marker Size:**
- Minimum detectable marker: ~20-30 pixels per side on the image sensor
- At 1080p (1920x1080) covering an 8ft arena from ~7ft height, each foot of arena = ~240 pixels horizontally
- A **3-inch (7.6cm) marker** would appear as ~60 pixels across — well above the 30-pixel minimum, providing robust detection with margin for blur and angle
- A **4-inch (10cm) marker** would appear as ~80 pixels — even more robust, but may be too large for small beetleweight robots (typical size 6-8 inches across)
- **Recommendation**: Use 3-inch ArUco markers from DICT_4X4_50

**Detection Performance:**
- OpenCV ArUco detection at 1080p typically runs at 15-30ms per frame on a modern CPU
- Corner refinement (`CORNER_REFINE_SUBPIX`) improves pose accuracy but adds ~5ms
- For speed, disable corner refinement during active combat; enable during calibration

#### Motion Blur Mitigation
- Motion blur occurs when an object moves more than **0.5 pixels during exposure time**
- At 8 ft/s robot speed and ~240 px/ft resolution, the robot moves ~1920 px/s on the sensor
- To keep blur under 1 pixel: exposure time must be < **1/1920s = ~520 microseconds**
- OAK-D Pro supports manual exposure down to **1 microsecond**, so 500us is easily achievable
- **Trade-off**: Short exposure = less light = darker image. The arena has LED top-down illumination, which helps. May need to increase ISO sensitivity.
- The OV9282 stereo cameras have a **global shutter** (no rolling shutter artifacts), which is ideal for fast-moving objects

#### Background Subtraction for Enemy Detection
- **MOG2 (Mixture of Gaussians)** from OpenCV is well-suited:
  - Adapts to gradual lighting changes
  - Can detect and separate shadows (important with overhead LED lighting)
  - Key parameters: `history=500` (frames for background model), `varThreshold=16`, `detectShadows=True`
- **Challenge**: The center logo on the white floor will confuse simple background subtraction. Solutions:
  - Capture a clean background frame during calibration (no robots) and use frame differencing against that reference
  - Use the homography transform to create a static background model of just the floor
  - Combine with depth data: the floor is at a known depth, robots are elevated above it
- **Depth-based detection** is more robust than pure appearance:
  - Set a depth threshold (anything above the floor plane by >1 inch is a robot)
  - OAK-D Pro's active stereo (IR dot projector) helps get depth on the textureless white floor
  - At 7ft mounting height, floor is at ~213cm depth; robots are ~5-10cm above the floor
  - Depth accuracy at 2m: approximately 2-4% = 4-8cm, which is marginal for detecting robots only a few cm tall. Combine depth with appearance for best results.

#### State Estimation: Kalman Filter vs Particle Filter
- **Extended Kalman Filter (EKF) recommended** for this application:
  - Robot motion is largely linear (differential drive on a flat surface) — EKF handles this well
  - Computationally efficient: runs in <1ms per update, suitable for real-time 60Hz fusion
  - Well-established for IMU + camera fusion in robotics
  - State vector: `[x, y, vx, vy, heading, angular_velocity]`
- **Particle filter** is overkill here:
  - Better for multi-modal distributions (e.g., "robot could be in multiple places") — not our case since we have ArUco giving direct position
  - 10-100x more computationally expensive than EKF
  - Use particle filter only if tracking fails frequently and multiple hypotheses are needed (Phase 2/3)
- For the enemy robot (no ArUco), a Kalman filter on the background-subtracted centroid position works well for smooth tracking

#### IMU + Camera Sensor Fusion
- **EKF-based fusion** is the standard approach:
  - Camera provides absolute position + orientation at 30-60 Hz (via ArUco detection)
  - IMU provides relative orientation changes at 100-500 Hz
  - EKF prediction step uses IMU data between camera frames
  - EKF correction step uses camera measurements when available
  - Result: smooth, high-frequency pose estimates that handle brief occlusions
- **Complementary filter** is a simpler alternative:
  - High-pass filter on IMU gyro (good for fast changes) + low-pass filter on camera heading (good for absolute reference)
  - Less optimal than EKF but easier to implement and debug
  - Good option for MVP, upgrade to EKF in Phase 2
- **Asynchronous fusion**: IMU and camera data arrive at different rates. The EKF naturally handles this — predict at IMU rate, correct at camera rate.

#### Enemy Robot Detection Without Markers
- **Hybrid approach recommended** (depth + appearance):
  1. **Depth thresholding**: Anything elevated above the floor plane by >2cm is a candidate object
  2. **Background subtraction**: MOG2 or frame differencing against calibration reference to detect moving objects
  3. **Exclusion masking**: Subtract our robot's known position/footprint from candidates
  4. **Contour analysis**: Find the largest remaining contour — that is the enemy
  5. **Fallback**: If depth is unreliable at this range, rely on appearance-based detection with the logo area masked out
- **On-device neural inference** (Phase 2/3): Train a custom object detector (MobileNet-SSD or YOLO) on combat robot images, deploy to the Myriad X VPU for on-device enemy detection at ~30 FPS with minimal host CPU load
- **Heading estimation**: Track the enemy centroid over time; the velocity vector direction estimates heading

---

### 6D. ESP32 Communication

#### Protocol Comparison

| Protocol | Median Latency | Pros | Cons |
|----------|---------------|------|------|
| **WiFi UDP** | ~5-10ms RTT (optimized) | Simple, standard networking, works with any AP | Requires WiFi AP, no delivery guarantee |
| **WiFi TCP** | ~5-10ms RTT (optimized) | Reliable delivery, similar latency to UDP | Slightly more overhead, Nagle's algorithm must be disabled |
| **ESP-NOW** | ~1-3ms RTT | Lowest latency, peer-to-peer, no AP needed | 250-byte payload limit, requires second ESP32 on host side, **cannot coexist with WiFi reliably** (80%+ packet loss when WiFi is active) |
| **WebSockets** | ~10-20ms RTT | Full duplex, easy to integrate with web stack | Runs over TCP, slightly higher overhead |

#### Optimization Requirements for WiFi
- **Disable WiFi power saving** on the ESP32 (modem sleep causes 100ms+ latency spikes)
- **Place WiFi and LwIP stacks in IRAM** for faster interrupt handling
- **Disable Nagle's Algorithm** if using TCP (`TCP_NODELAY` flag)
- **Use a dedicated WiFi AP** (not shared with spectators/other devices) to minimize contention
- **Best achievable WiFi latency**: ~2ms one-way under ideal conditions (no other clients, strong signal, minimal beacons)
- **Realistic arena latency**: 5-15ms RTT including processing, which meets the PRD's <20ms target

#### ESP-NOW Consideration
- ESP-NOW achieves the lowest latency (~1-3ms RTT) and is Espressif's proprietary protocol
- However, **ESP-NOW and WiFi cannot coexist reliably** on the same ESP32 — packet loss exceeds 80% when both are active
- If using ESP-NOW for motor commands, telemetry would need to go over the same ESP-NOW link (limiting bandwidth) OR use a separate radio
- **Recommendation**: Use WiFi UDP for this project. The latency is sufficient (<20ms target), and it allows the host PC to communicate without needing a second ESP32 as a bridge

#### Motor Control Patterns for Differential Drive
- **Command format**: `(left_speed, right_speed)` as signed integers (-255 to +255 for 8-bit PWM, or -1000 to +1000 for finer control)
- **Command rate**: Send motor commands at 50-100 Hz for smooth control
- **Heartbeat/watchdog**: ESP32 firmware should stop motors if no command received within 200ms (matches PRD failsafe requirement)
- **Motor driver**: DRV8871 (2A per channel) or L298N are common choices for beetleweight robots
- **PWM frequency**: 20-25 kHz for silent motor operation (above audible range)
- **Acceleration ramping**: Implement on ESP32 firmware to prevent wheel spin and current spikes. Limit acceleration to ~1000 units/second.

#### Polycarbonate Arena Interference
- Polycarbonate is **RF-transparent** at 2.4 GHz — it does not significantly attenuate WiFi signals
- The steel floor and frame may cause **multipath reflections**, but this primarily affects range, not latency
- **Recommendation**: Position the WiFi AP directly above or beside the arena with line-of-sight to minimize multipath effects. The host PC running the camera can also serve as the AP (or use a dedicated router mounted overhead).

---

## 7. Technical Recommendations

### 7.1 Camera Selection and Mounting

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Camera model | **OAK-D Pro** (standard FOV, fixed-focus) | Standard 69 HFOV covers the 8ft arena from 7ft height with margin. Wide-angle (Pro W) introduces distortion and reduces pixels-per-foot, making ArUco detection harder at edges. Fixed-focus avoids auto-focus hunting. |
| Resolution | **1080p at 60 FPS** | Best balance of resolution and frame rate. 4K adds too much latency (150-530ms). 720p is acceptable but reduces ArUco detection margin. |
| Mounting height | **9ft above the arena floor** on a freestanding rig, 5ft above the 4ft walls | Provides ~151-212 px/ft resolution at 1080p across the arena (near to far side). Angled view at ~24 deg from vertical keeps ArUco detection within reliable range. See section 7.8 for full analysis. |
| Exposure | **Manual exposure, 400-500 microseconds** | Freezes motion of robots moving at 8 ft/s to <1 pixel of blur. Increase ISO to compensate for reduced light. |
| Depth mode | **400P resolution, active stereo (IR projector ON)** | IR projector helps with textureless white floor. 400P is sufficient for depth thresholding and reduces processing load. |

### 7.2 ArUco Markers

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Dictionary | **DICT_4X4_50** | Fewest bits = most robust at distance. Only need ~7 unique IDs. Highest inter-marker distance for error resistance. |
| Marker size | **4 inches (10cm)** | At 9ft angled mounting, appears as ~50-70px at 1080p (far to near side) — well above 30px minimum even at worst case. Fits on beetleweight robots (6-8 inch footprint). Larger size compensates for oblique viewing angle. |
| Marker IDs | ID 0-3: arena corners, ID 4: pit, ID 5: our robot (top), ID 6: our robot (bottom) | Simple, memorable assignment. |
| Corner refinement | **Off during combat, on during calibration** | Saves ~5ms per frame during combat. Calibration needs precision, not speed. |
| Printing | **Matte finish, high contrast black on white** | Gloss causes specular reflections under overhead LEDs. |

### 7.3 Enemy Detection Strategy

| Phase | Approach | Details |
|-------|----------|---------|
| MVP | **Reference frame differencing + depth thresholding** | Capture clean background during calibration. Subtract it from live frames. Combine with depth data (anything >2cm above floor plane). Mask out our robot's known position. |
| Phase 2 | **Add Kalman filter tracking** | Smooth noisy detections, predict through brief occlusions, estimate velocity for strategy engine. |
| Phase 3 | **On-device neural detector** | Train MobileNet-SSD or YOLOv5-nano on combat robot images. Deploy to Myriad X VPU. Runs at ~30 FPS with zero host CPU cost. |

### 7.4 State Estimation and Sensor Fusion

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Our robot tracking | **Complementary filter (MVP), upgrade to EKF (Phase 2)** | Complementary filter is simpler to implement and debug. EKF provides better accuracy but requires careful tuning. |
| IMU rate | **100-200 Hz from ESP32** | BNO085 on OAK-D can run up to 500Hz but that data is for camera stabilization. Robot's onboard IMU at 100-200Hz provides heading between camera frames. |
| State vector | **[x, y, vx, vy, heading, omega]** | Position, velocity, heading, and angular velocity. 6 states is manageable for EKF. |
| Enemy tracking | **Kalman filter on centroid position** | State: [x, y, vx, vy]. Simpler than our robot since no orientation from markers. |
| Occlusion handling | **Predict forward using last known velocity for up to 500ms** | Beyond 500ms of no detection, flag enemy position as uncertain to the strategy engine. |

### 7.5 Communication Protocol

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Motor commands | **WiFi UDP** | 5-15ms RTT meets <20ms target. No AP bridge needed (unlike ESP-NOW). Simple to implement with Python asyncio. |
| IMU telemetry | **WiFi UDP** (same link) | Piggyback on existing WiFi connection. Separate UDP port from motor commands. |
| Packet format | **Binary struct, 12-20 bytes** | Motor command: `[cmd_type(1), seq(2), left_speed(2), right_speed(2), checksum(1)]` = 8 bytes. Telemetry: `[type(1), seq(2), ax(2), ay(2), az(2), gx(2), gy(2), gz(2), heading(2), battery(2), checksum(1)]` = 20 bytes. |
| Command rate | **50 Hz** for motor commands | Sufficient for smooth differential drive control. Higher rates waste bandwidth without benefit. |
| Failsafe | **200ms timeout on ESP32** | If no valid motor command received in 200ms, set both motors to zero. |
| WiFi optimization | **Disable modem sleep, use dedicated AP, disable Nagle's** | Critical for consistent low latency. |

### 7.6 Frame Rate vs Resolution Trade-offs

| Resolution | Max FPS | Px/ft (at 7ft height) | ArUco 3" marker size | Latency | Recommendation |
|-----------|---------|----------------------|----------------------|---------|----------------|
| 4K (3840x2160) | 30 | ~480 | ~120px | 150-530ms | Do not use — latency too high |
| 1080p (1920x1080) | 60 | ~240 | ~60px | ~5-10ms | **Primary choice** — best balance |
| 720p (1280x720) | 60+ | ~160 | ~40px | ~3-5ms | Fallback if CPU constrained |

### 7.7 Key Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Depth accuracy marginal at 2m range | Enemy detection may miss low-profile robots | Combine depth with appearance-based detection; do not rely on depth alone |
| Motion blur despite short exposure | ArUco detection fails during fast maneuvers | Set exposure to 400-500us; increase ISO; accept slightly noisier image |
| WiFi latency spikes in arena | Motor commands delayed >50ms | Use dedicated AP; disable power saving; implement command prediction on ESP32 |
| ArUco occluded during collisions | Lose our robot position for several frames | IMU-based dead reckoning during occlusion (up to 500ms); bottom marker as backup |
| Logo on floor confuses background subtraction | False positives in enemy detection | Mask logo region using calibration data; prioritize depth-based detection |
| Autonomous class uses different arena format | 3ft circular arena, not 8x8 square | Verify with event organizers; system architecture should support configurable arena geometry |

### 7.8 Camera Mounting Position (External to Arena) -- REVISED

The camera MUST be mounted OUTSIDE the arena. Robots would destroy anything inside. The ceiling is opaque plywood and **CANNOT be modified** -- we do not own the arena. This section evaluates all viable mounting positions given this hard constraint.

#### Arena Physical Constraints
- **Walls**: 4ft tall polycarbonate, double-layered (2 sheets of 1/4" Lexan with 1" air gap between them)
- **Ceiling**: Opaque -- composed of 2 sheets of 3/4" plywood on a 2x4 wood frame bolted to a welded steel frame. LED lights mounted underneath facing down. Additional plywood above the 2x4 frame provides space for NHRL production equipment. **We CANNOT cut, drill, or modify the ceiling in any way.**
- **Floor**: White surface with center logo, steel base
- **Wall material**: Lexan (polycarbonate brand) -- inherently **clear/transparent**, not frosted. Standard across NHRL and most combat robotics arenas for spectator visibility.
- **Wall-ceiling junction**: The 2x4 ceiling frame bolts to the top of the welded steel frame. The polycarbonate wall sheets are retained by steel plates bolted to the frame. There may be small gaps (<1/8") but **no usable opening** for inserting a camera between walls and ceiling.
- **We do NOT own the arena** -- it is competition venue equipment. All mounting must be external, portable, and leave no marks.

#### Previous Recommendation (Invalidated)

The previous version of this section recommended Option A2 (polycarbonate window insert in the ceiling). **This is not possible.** We cannot modify the ceiling. All "Option A" variants (open hole, window insert, recessed box) are eliminated.

#### Option B: Viewing Through the Polycarbonate Walls

**Concept**: Mount camera outside the arena, shooting through the clear polycarbonate walls to see the floor and robots inside.

**Polycarbonate optical analysis**:
- NHRL arena walls use clear Lexan (polycarbonate), which is transparent to visible light (~88% transmission for single sheet)
- However, the walls are **double-layered** with a 1" air gap, creating two refractive surfaces plus an air cavity
- Polycarbonate refractive index: 1.586. Double layer amplifies refraction, creating ghost images
- Polycarbonate transmits near-infrared (700-1500nm) at ~100%, so the OAK-D Pro's 940nm IR dot projector would pass through -- but the double-layer air gap creates IR reflections between the sheets

**Critical problems**:
1. **Viewing angle**: From wall height (4ft), looking across the 8ft arena floor, the camera angle to the far side is extremely oblique (~63 degrees from perpendicular at the far wall). ArUco markers on the far side would be viewed nearly edge-on.
2. **Double-layer distortion**: Two sheets with an air gap create double images, chromatic aberration, and ghosting that cannot be fully corrected with calibration
3. **Occlusion**: Robots, wall framing, and steel retention plates obstruct the view. One robot hides behind the other.
4. **ArUco tags are on TOP of robots**: A side view through the wall cannot see markers mounted on the robot's top surface at all -- they face upward, not sideways.

**Verdict**: NOT VIABLE. Cannot see top-mounted ArUco tags from a side view. Double-layer distortion is severe. Occlusion is guaranteed.

#### Option C: Elevated Angled Overhead from One Side (RECOMMENDED)

**Concept**: Mount the camera on a freestanding boom/truss above one side of the arena, looking down at an angle into the arena. The camera sits above the wall height (above 4ft) and looks down over the wall top, past the ceiling edge, into the arena interior.

**Key insight**: The ceiling does not extend beyond the arena frame. The camera can look DOWN past the edge of the ceiling and into the arena from above, provided it is positioned outside the arena perimeter and high enough.

**Geometry analysis (camera centered on one 8ft side)**:

Mounting the camera centered above one 8ft wall, offset horizontally from the arena edge by ~1ft (to clear the frame), looking down at an angle:

| Camera Height (above floor) | Tilt from Vertical | Near Edge Angle | Far Edge Angle | Notes |
|-----|-----|-----|-----|-----|
| 7ft (3ft above walls) | ~30 deg | ~8 deg from vertical | ~52 deg from vertical | Far side at limit; tight coverage |
| 8ft (4ft above walls) | ~27 deg | ~7 deg | ~47 deg | Better -- far side within reliable ArUco range |
| 9ft (5ft above walls) | ~24 deg | ~6 deg | ~42 deg | Good -- all angles within reliable detection |
| 10ft (6ft above walls) | ~22 deg | ~6 deg | ~39 deg | Best angles; tallest practical rig |

Angle calculations (camera 1ft outside arena edge, looking at arena center):
- Near wall (directly below): angle from perpendicular = arctan(1ft / height) -- nearly vertical
- Far wall (9ft horizontal away): angle from perpendicular = arctan(9ft / height)
- At 9ft height: far wall angle = arctan(9/9) = 45 deg from perpendicular
- At 10ft height: far wall angle = arctan(9/10) = 42 deg from perpendicular

**ArUco detection reliability at angle**:
- ArUco markers are reliably detected up to ~60-70 degrees from perpendicular (i.e., from the marker's normal vector)
- Pose estimation accuracy degrades beyond ~45 degrees but detection still works
- At 9ft camera height: maximum angle to far edge is ~42 deg -- **within reliable detection AND pose estimation range**
- Research confirms that at 45 deg tilt, ArUco pose estimation errors are manageable, especially with multiple markers and homography-based correction
- At 10ft height: maximum angle is ~39 deg -- comfortably within reliable range

**FOV coverage calculation (at 9ft height, OAK-D Pro standard 69H x 55V)**:
- Camera aimed at arena center (4ft from near wall, 4ft from far wall, 9ft above floor)
- Slant distance to arena center: sqrt(4^2 + 9^2) = ~9.85ft (3.0m)
- Slant distance to near edge: sqrt(1^2 + 9^2) = ~9.05ft (2.76m)
- Slant distance to far edge: sqrt(9^2 + 9^2) = ~12.7ft (3.87m)
- The 69-degree HFOV at 9.85ft slant covers ~12.5ft horizontally -- more than the 8ft arena width (2.25ft margin each side)
- The 55-degree VFOV covers the depth from near to far wall when the camera is tilted ~24 deg from vertical

**Resolution analysis (1080p at 9ft height)**:
- Near side (directly below): slant distance ~9.05ft, resolution ~212 px/ft
- Center: slant distance ~9.85ft, resolution ~195 px/ft
- Far side: slant distance ~12.7ft, resolution ~151 px/ft
- **Resolution ratio near:far = 1.4:1** -- much better than corner mounting (6:1)
- A 4-inch ArUco marker: ~63px near side, ~50px far side -- both well above 30px minimum
- A 3-inch ArUco marker: ~47px near side, ~38px far side -- adequate but tighter margin on far side

**Perspective distortion**:
- The angled view creates trapezoidal distortion (near side wider than far side in the image)
- This is correctable via homography transform during arena calibration (the 4-corner ArUco calibration procedure handles this automatically)
- The calibration already computes a homography from camera pixels to arena coordinates -- an angled view just means a different homography matrix, not a fundamentally different approach
- Position accuracy after homography correction: ~0.5 inch near side, ~1 inch far side (acceptable for robot tracking)

**Depth sensor considerations**:
- Stereo depth works best perpendicular to surfaces. At an angle, floor depth data is less reliable.
- Near side: depth to floor ~2.76m at ~8 deg angle -- good depth accuracy
- Far side: depth to floor ~3.87m at ~42 deg angle -- depth accuracy degrades (~5-10% error)
- **Recommendation**: Use depth primarily for robot height detection (distinguishing robots from floor), not for precise distance measurement. The homography provides better position accuracy than depth at these angles.
- The IR dot projector is unobstructed (no polycarbonate in the optical path) and will improve depth on the textureless white floor

**Occlusion analysis**:
- Near-overhead viewing at ~24 deg tilt provides excellent occlusion resistance
- Robot-to-robot occlusion: minimal. A 4-inch-tall robot at the near edge could theoretically occlude a robot 1.5ft behind it, but the steep downward angle makes this unlikely in practice
- Wall occlusion: the camera looks OVER the walls, so walls do not obstruct the view
- Ceiling edge: the ceiling terminates at the arena frame boundary; the camera looks past the ceiling edge into the arena. Some ceiling edge may occlude a small strip of the near wall area -- **mount the camera close enough to the arena edge to minimize this blind spot**

**Mounting the far-side pit corner**: If the pit is in a corner, position the camera on the **opposite side** from the pit. This puts the pit at the far side where the camera has a clear downward view, and avoids the near-side blind strip being at the critical pit area.

**Physical mounting options**:

| Mounting System | Height Range | Portability | Setup Time | Load Capacity | Cost |
|---|---|---|---|---|---|
| ProX T-LS35C crank truss (2 stands + triangle truss) | 9-10ft | Two stands pack into bags | ~20 min | 360 lbs | ~$500 |
| Two heavy-duty light stands + crossbar + super clamp | 8-10ft | Fold into carry bags | ~10 min | 20-30 lbs | ~$200 |
| Glide Gear OH 150 modular overhead rig | 4-12ft | Modular, compact | ~5 min | 20 lbs | ~$300 |
| Custom 80/20 aluminum extrusion rig | Custom | Disassembles | ~15 min | Custom | ~$150 |
| C-stand with boom arm | 7-10ft | Standard photo gear | ~5 min | 10 lbs | ~$100 |

**Recommended rig**: Two heavy-duty light stands (10ft max height) with a crossbar spanning ~3ft, camera mounted on the crossbar via super clamp and ball head. Position the stands flanking one side of the arena, crossbar extending slightly over the arena edge. Total weight under 15 lbs, fits in a gear bag. **Alternative**: C-stand with boom arm for simpler setup -- one stand, arm extends over the arena edge, camera at the end with counterweight.

**Vibration isolation**: The camera rig is freestanding (not attached to the arena), so it does NOT vibrate when robots impact. This is a significant advantage over ceiling mounting. The only vibration concern is floor vibration from very heavy impacts, which is negligible for 3lb beetleweight class.

#### Option D: Corner-Mounted High, Looking Down at Angle

**Concept**: Mount camera above one corner, angled to see the entire arena.

**Geometry analysis**:
- Camera above one corner, needs to see the opposite corner 11.3ft away (diagonal of 8x8 arena)
- At 9ft above floor: far corner angle = arctan(11.3/9) = ~51 deg from perpendicular -- at the edge of reliable ArUco pose estimation
- Near corner: extremely high resolution (~400+ px/ft), wasted sensor area
- Far corner: very low resolution (~70 px/ft) -- barely adequate for ArUco
- **Resolution ratio near:far = 6:1** -- extremely non-uniform

**Verdict**: NOT RECOMMENDED as primary. The non-uniform resolution and extreme angle to the far corner make this inferior to the side-mounted overhead approach. However, it could serve as a **secondary camera** for redundancy (see Option F).

#### Option E: Mounted on Top of a Wall Panel

**Concept**: Mount the camera directly on top of a polycarbonate wall panel (at 4ft height), looking down into the arena.

**Problems**:
- At only 4ft height, the camera is nearly at floor level relative to the arena interior
- Cannot see ArUco tags on top of robots at this angle -- viewing angle is too oblique
- Extreme perspective distortion (near robots are 100x larger than far robots in the image)
- Wall vibrates severely during robot impacts
- Mounting hardware could come loose and fall into the arena (safety hazard)

**Verdict**: NOT VIABLE. Camera is too low for any useful overhead perspective.

#### Option F: Two Cameras from Different Angles

**Concept**: Use two OAK-D cameras mounted at different positions to get combined coverage.

**Configuration options**:
1. **Two opposite sides**: Cameras on opposite sides of the arena, each covering their half with good angles. Cross-calibrated via shared ArUco corner markers.
2. **Side + corner**: Primary camera on one side (Option C), secondary in a corner for redundancy.

**Advantages**:
- Eliminates the near/far resolution asymmetry -- each camera's near side is the other's far side
- Redundancy: if one camera is bumped or fails, the other continues tracking
- Better occlusion handling from two viewpoints

**Disadvantages**:
- Doubles hardware cost (2x OAK-D Pro = ~$600 more)
- Requires cross-camera calibration and data fusion
- Doubles USB bandwidth and processing requirements
- More complex setup at competitions (two rigs to position)
- Synchronization between cameras adds latency (~5ms for frame alignment)

**Verdict**: NOT RECOMMENDED for Phase 1. A single camera from one side (Option C) provides adequate coverage. Reserve dual-camera as a Phase 3 enhancement if single-camera tracking proves insufficient.

#### Option G: Sliding Camera into Wall-Ceiling Gap

**Concept**: Exploit any gap between the polycarbonate walls and the plywood ceiling to insert a small camera.

**Analysis**:
- The NHRL cage has the ceiling's 2x4 frame bolted directly to the top of the welded steel frame
- Polycarbonate wall retention plates bolt into the same steel frame
- Gap between wooden components and steel: less than 1/8 inch
- Even the smallest camera module (e.g., OAK-D Lite at 91mm x 28mm) cannot fit through a 1/8" gap
- Attempting to force anything into the gap risks damaging competition equipment

**Verdict**: NOT VIABLE. No usable gap exists.

#### How Reference Systems Solve This

**RoboCup Small Size League (SSL)**:
- Cameras mounted on a bar 4 meters (~13ft) above the open field -- no ceiling to deal with
- 2-4 cameras (FLIR Blackfly S, USB3) depending on field size (up to 12m x 9m)
- The field has NO ceiling or enclosure -- cameras simply look straight down
- Uses colored dot patterns on robots (analogous to ArUco markers)
- ssl-vision open-source software (https://github.com/RoboCup-SSL/ssl-vision)
- Our arena is enclosed with a solid ceiling, so we cannot replicate this exact approach

**NHRL broadcast cameras**:
- NHRL mounts production cameras in the space above the ceiling plywood (the "production equipment space")
- These cameras can see into the arena through custom openings that NHRL installs in their own equipment
- As a competitor, we do NOT have access to this space or permission to mount equipment there

**Key takeaway**: Most overhead vision tracking systems (SSL, NHRL production) assume either an open field or ownership of the overhead structure. Our situation -- enclosed arena with inaccessible opaque ceiling that we don't own -- is unusual. The freestanding elevated rig (Option C) is the practical adaptation.

#### Comparison Matrix (Revised)

| Criterion | C: Side Overhead (RECOMMENDED) | D: Corner High | E: Wall Top | F: Two Cameras | G: Gap Insert |
|-----------|------|------|------|------|------|
| ArUco detection reliability | Good (max 42 deg at 9ft) | Marginal (51 deg far corner) | Not viable | Very Good (redundant) | Not viable |
| Arena floor coverage | Full | Full but non-uniform | Partial | Full, uniform | N/A |
| Depth sensor usability | Moderate (near: good, far: degraded) | Poor (3m+ range, steep angle) | Not viable | Good (cross-coverage) | N/A |
| Occlusion resistance | Very Good | Moderate | Poor | Excellent | N/A |
| Resolution uniformity | 1.4:1 near/far ratio | 6:1 near/far ratio | >10:1 ratio | ~1:1 combined | N/A |
| Vibration sensitivity | Very Low (freestanding) | Very Low | Very High | Very Low | N/A |
| Mounting complexity | Moderate (rig setup) | Easy | Easy but risky | High (two rigs) | Not possible |
| Portability | Good (gear bags) | Good | Minimal | Moderate | N/A |
| Setup time | ~10-20 min | ~10 min | ~5 min | ~30 min | N/A |
| Arena modification required | None | None | None | None | N/A |
| LED glare | None (above ceiling) | None | None | None | N/A |

#### Final Recommendation: Option C (Elevated Angled Overhead from One Side)

**Primary choice**: Mount the OAK-D Pro on a freestanding rig above one side of the arena, looking down at a steep angle (~24 deg from vertical) to cover the full 8x8ft arena floor.

**Specifications**:
- **Mounting height**: 9ft above arena floor (5ft above the 4ft walls)
- **Horizontal offset**: ~1ft outside arena edge (to clear the frame/ceiling overhang)
- **Camera position**: Centered on one 8ft side of the arena
- **Tilt angle**: ~24 degrees from vertical (aimed at arena center)
- **Camera model**: OAK-D Pro (standard FOV 69H x 55V, fixed-focus)
- **Mounting rig**: Two 10ft light stands + crossbar + super clamp + ball head, OR C-stand with boom arm
- **Vibration isolation**: Not needed -- freestanding rig does not couple to arena vibrations
- **Orientation**: Position camera on the side OPPOSITE the pit corner for best pit coverage

**Tradeoffs to acknowledge**:
- **This is NOT as good as true overhead**: A perpendicular overhead view would give uniform resolution, zero perspective distortion, and optimal ArUco detection. The angled view introduces a 1.4:1 resolution gradient, trapezoidal distortion (correctable), and slightly degraded ArUco pose estimation at the far edge.
- **Depth sensor is less useful**: At the viewing angle, depth data on the far side is inaccurate. Use homography-based position estimation as the primary method; use depth only for robot-vs-floor discrimination.
- **Near-wall blind strip**: A narrow strip along the near wall (directly below the camera) may be partially occluded by the ceiling edge. This is ~6-12 inches and can be minimized by positioning the camera as close to the arena edge as possible.
- **Calibration is more complex**: The angled view requires careful homography calibration, but this is already part of our 4-corner calibration procedure and is handled automatically by the software.

**Fallback**: If an event organizer grants permission to access the ceiling or mount equipment above the arena, revert to the overhead approach (camera looking straight down through an opening). Always ask -- some events may accommodate this for autonomous classes.

**Phase 3 enhancement**: Add a second camera (Option F) from the opposite side or a corner for redundancy and improved accuracy. The software architecture should support multiple camera inputs from the start.

#### Key Numbers Summary (Revised)

| Parameter | Value |
|-----------|-------|
| Mounting height above floor | 9ft (2.74m) |
| Mounting height above ground | 9ft (freestanding rig, not on the walls) |
| Camera horizontal offset from arena | ~1ft (0.3m) |
| Tilt angle from vertical | ~24 degrees |
| Slant distance to arena center | ~9.85ft (3.0m) |
| Slant distance to near edge | ~9.05ft (2.76m) |
| Slant distance to far edge | ~12.7ft (3.87m) |
| Horizontal coverage (1080p) | ~12.5ft at center (8ft arena + 2.25ft margin each side) |
| Resolution at near side (1080p) | ~212 px/ft |
| Resolution at center (1080p) | ~195 px/ft |
| Resolution at far side (1080p) | ~151 px/ft |
| Near/far resolution ratio | 1.4:1 |
| 4-inch ArUco marker at near side | ~70px (excellent) |
| 4-inch ArUco marker at far side | ~50px (good, above 30px minimum) |
| 3-inch ArUco marker at far side | ~38px (adequate, tighter margin) |
| Max viewing angle (far edge) | ~42 deg from perpendicular (within ArUco reliable range) |
| Depth range to floor | 2.76m - 3.87m (within OAK-D Pro 0.7-12m range) |
| Depth accuracy at far side | ~5-10% error (~15-30cm) -- use homography instead |
| Vibration coupling | None (freestanding rig) |
| Setup time | ~10-20 minutes |
| Equipment weight | ~12-15 lbs total |

**Marker recommendation**: Use **4-inch (10cm) ArUco markers** for robust detection across the full arena. At the far side (worst case), a 4-inch marker yields ~50px -- solid margin above the 30px minimum. This fits on beetleweight robots with 6-8 inch footprints. Consider using DICT_4X4_50 dictionary for maximum detection robustness at oblique angles (simpler patterns are more tolerant of perspective distortion).

---

## 8. Tracking Algorithm Options

### 8.1 Comparison Table

| Algorithm | Latency (per frame) | Accuracy | Occlusion Handling | CPU Cost | GPU Required | Best For |
|-----------|-------------------|----------|-------------------|----------|-------------|----------|
| **Linear Kalman Filter** | <0.1ms | High (linear systems) | Good (predicts through gaps) | Negligible | No | Our robot state estimation with ArUco measurements |
| **Extended Kalman Filter (EKF)** | <0.5ms | High (nonlinear) | Good | Very Low | No | IMU + camera sensor fusion, nonlinear motion models |
| **Unscented Kalman Filter (UKF)** | <1ms | Slightly better than EKF | Good | Low | No | Alternative to EKF when Jacobians are hard to derive |
| **Particle Filter** | 5-50ms (depends on # particles) | Excellent (multi-modal) | Excellent | Medium-High | No | Multi-hypothesis tracking when detection is very noisy |
| **MOSSE Tracker** | <1ms (~450+ FPS) | Low-Moderate | Poor | Negligible | No | Ultra-fast single-object tracking between detections |
| **KCF Tracker** | ~3ms (~300 FPS) | Moderate | Poor | Very Low | No | Fast correlation tracking with HOG features |
| **CSRT Tracker** | ~25ms (~40 FPS) | High | Moderate | Medium | No | Accurate single-object tracking (too slow for our pipeline) |
| **MOG2 Background Subtraction** | ~2ms (640x480) | Moderate | N/A (detection only) | Low | No | Enemy detection against known background |
| **KNN Background Subtraction** | ~4ms (640x480) | Moderate-High (sharper edges) | N/A (detection only) | Low-Medium | No | Alternative to MOG2 with better edge detection |
| **SORT** | ~1ms (association only) | High | Poor (loses tracks quickly) | Very Low | No | Multi-object tracking with Kalman + Hungarian |
| **DeepSORT** | ~15-30ms (with re-ID) | Very High | Good (re-identification) | Medium-High | Recommended | Multi-object tracking with appearance features |
| **ByteTrack** | ~2ms (association only) | Very High | Good | Low | No | State-of-art MOT, uses low-confidence detections |
| **Optical Flow (Lucas-Kanade)** | ~2-5ms (sparse) | Moderate | Poor | Low | No | Motion estimation between frames, feature tracking |
| **Optical Flow (Farneback)** | ~15-30ms (dense) | High | Moderate | High | Optional | Dense motion field (overkill for 2-object tracking) |

### 8.2 Recommendation

**Primary approach: EKF + MOSSE + MOG2 hybrid**

The system uses different tracking techniques at different pipeline stages:

1. **Our Robot (ArUco-based)**: Linear Kalman Filter for state estimation. ArUco detection provides direct position and heading measurements at camera frame rate. Between camera frames, predict forward using a constant-velocity model. When IMU fusion is added (Phase 2), upgrade to EKF to handle the nonlinear IMU measurement model.

2. **Enemy Robot (markerless)**: MOG2 background subtraction + depth thresholding for detection. MOSSE correlation tracker for frame-to-frame association between detections (runs at 450+ FPS, adding <1ms overhead). Linear Kalman Filter on the detected centroid for smoothing and velocity estimation.

3. **Frame-to-frame bridging**: MOSSE tracker runs on ROI around last known enemy position, providing sub-millisecond updates even when the main detector has a noisy frame. The MOSSE result is treated as a low-confidence measurement that the Kalman filter can weight appropriately.

**Rationale**: This combination achieves the lowest possible latency while providing smooth, robust tracking. Total tracking overhead per frame: ~5ms (MOG2) + <1ms (MOSSE) + <1ms (Kalman updates) = ~7ms, well within the detection budget. DeepSORT and particle filters are reserved for Phase 2/3 if tracking robustness proves insufficient.

**Phase evolution**:
- Phase 1 (MVP): Kalman filter + MOG2 + MOSSE
- Phase 2: EKF with IMU fusion, add DeepSORT if enemy re-identification is needed
- Phase 3: On-device neural detector (Myriad X VPU) replaces MOG2 for enemy detection

---

## 9. Control Algorithm Options

### 9.1 Comparison Table

| Algorithm | Response Time | Compute Cost | Adversarial Handling | Configurability | Complexity | Best For |
|-----------|-------------|-------------|---------------------|----------------|-----------|----------|
| **PID Control** | <1ms | Negligible | Poor (reactive only) | Low (3 gains) | Very Low | Low-level motor speed control, wheel velocity tracking |
| **Pure Pursuit** | <1ms | Very Low | Poor (path following only) | Low (lookahead distance) | Low | Path following to a target point for differential drive |
| **Model Predictive Control (MPC)** | 5-50ms (depends on horizon) | High | Good (plans ahead) | High (many parameters) | High | Optimal trajectory planning with constraints |
| **Potential Fields** | <1ms | Very Low | Moderate (reactive avoidance) | Medium (field weights) | Low | Obstacle avoidance, pit avoidance, goal attraction |
| **Behavior Trees** | <1ms per tick | Very Low | Good (structured reactions) | High (tree structure) | Medium | Strategic decision making, composable behaviors |
| **Finite State Machine** | <0.1ms | Negligible | Moderate (predefined states) | Low (fixed transitions) | Low | Simple strategy with clear state transitions |
| **Hybrid: BT + Potential Fields + PID** | <2ms total | Low | Good | High | Medium | Full combat strategy with layered control |

### 9.2 Recommendation

**Hybrid layered control architecture: Behavior Tree + Potential Fields + PID**

The control system is organized into three layers that run at different frequencies:

```
Layer 1: STRATEGY (Behavior Tree)          -- ticks at 30 Hz
  |  Selects high-level behavior: PURSUE, POSITION, PUSH, RETREAT, AVOID_PIT
  |  Outputs: target_point, behavior_mode, aggression_level
  v
Layer 2: MOTION PLANNING (Potential Fields) -- runs at 60 Hz
  |  Computes velocity vector from attractive/repulsive fields
  |  Attractive: target point from strategy
  |  Repulsive: pit boundary, arena walls
  |  Outputs: desired_linear_velocity, desired_angular_velocity
  v
Layer 3: MOTOR CONTROL (PID)               -- runs at 50 Hz (command rate)
  |  Converts desired velocities to differential drive wheel speeds
  |  PID loop on each wheel for speed tracking
  |  Outputs: left_speed, right_speed (-1.0 to 1.0)
  v
  Comms Layer --> ESP32
```

**Why this combination**:

- **Behavior Tree over FSM**: BTs are modular, composable, and handle priority interrupts naturally. A "check pit proximity" safety node can interrupt any behavior at every tick. FSMs scale poorly as strategies grow complex (state explosion). BTs can be hot-reloaded from the dashboard for tuning.

- **Potential Fields over Pure Pursuit**: Pure Pursuit follows a predefined path, but in combat the target moves unpredictably. Potential fields generate reactive velocity commands that naturally combine goal-seeking with obstacle avoidance. The pit acts as a strong repulsive field for our robot, preventing self-elimination.

- **PID over MPC**: MPC computes optimal trajectories but costs 5-50ms per solve, consuming the entire latency budget. PID runs in <1ms and is sufficient for differential drive motor control. MPC is overkill for a 2-wheel robot in a small arena where reaction time matters more than path optimality.

**Behavior Tree structure (MVP)**:

```
Root (Priority Selector)
  |
  +-- [SAFETY] Sequence
  |     +-- Check: too close to pit?
  |     +-- Action: emergency retreat from pit
  |
  +-- [COMBAT] Selector
        +-- [PUSH] Sequence
        |     +-- Check: behind enemy relative to pit?
        |     +-- Check: close enough to push?
        |     +-- Action: drive forward into enemy
        |
        +-- [POSITION] Sequence
        |     +-- Check: enemy detected?
        |     +-- Action: maneuver behind enemy (relative to pit)
        |
        +-- [PURSUE] Sequence
              +-- Check: enemy detected?
              +-- Action: drive toward enemy
```

**Phase evolution**:
- Phase 1 (MVP): Simple FSM (PURSUE -> PUSH -> RETREAT) with potential fields for pit avoidance, PID motor control
- Phase 2: Replace FSM with full Behavior Tree, add configurable aggression parameters
- Phase 3: Add predictive elements (anticipate enemy movement), potential MPC for optimal push trajectories

---

## 10. Module Specifications

### 10.1 Camera Capture Module

| Property | Value |
|----------|-------|
| **Process** | P1 (dedicated process) |
| **Input** | OAK-D Pro USB3 stream |
| **Output** | RGB + Depth frames in shared memory ring buffer; IMU data in a queue |
| **Performance Budget** | <5ms from sensor readout to shared memory write |
| **Dependencies** | None (first in pipeline) |
| **Library** | `depthai` SDK |

**Input Interface**:
- OAK-D Pro camera via USB3 (DepthAI pipeline config)
- IMU telemetry from ESP32 via UDP (received by comms layer, forwarded through queue)

**Output Interface**:
- `SharedMemory("arena_frames")`: Ring buffer with RGB (640x480x3) + Depth (640x480x2) frames
- `Queue("imu_data")`: IMU readings `{timestamp_ns, ax, ay, az, gx, gy, gz, heading}`
- `Queue("frame_metadata")`: Frame metadata `{frame_id, timestamp_ns, slot_index}`

**Test Strategy**:
- Unit test: Mock DepthAI pipeline, verify frames are written to shared memory correctly
- Integration test: Connect real camera, measure capture-to-memory latency
- Replay test: Read frames from a recorded file and write to shared memory at original rate

---

### 10.2 Arena Calibration Module

| Property | Value |
|----------|-------|
| **Process** | Runs in P2 during setup, then provides static data |
| **Input** | Camera frames (shared memory), operator commands (via dashboard) |
| **Output** | Homography matrix, arena bounds, pit location |
| **Performance Budget** | Not real-time; runs once during setup (<60s total) |
| **Dependencies** | Camera Capture (for frames) |
| **Library** | `cv2.findHomography`, `cv2.aruco` |

**Input Interface**:
- RGB frames from shared memory
- Operator commands: `{action: "capture_corner", corner_id: 0-3}`, `{action: "capture_pit"}`

**Output Interface**:
- `CalibrationData`: `{homography_3x3, arena_corners_px, arena_corners_world, pit_center_world, pit_radius_ft}`
- Persisted to `config/calibration.json`
- Loaded at startup if valid calibration exists

**Test Strategy**:
- Unit test: Given known pixel/world point pairs, verify homography computation
- Simulated test: Use a synthetic top-down image with ArUco markers at known positions
- Manual test: Calibration wizard in dashboard with visual feedback

---

### 10.3 ArUco Detector Module

| Property | Value |
|----------|-------|
| **Process** | P2 (Detection process, Thread A) |
| **Input** | RGB frame from shared memory |
| **Output** | `RobotDetection` for our robot (pixel + arena coordinates, heading) |
| **Performance Budget** | <8ms per frame at 640x480 |
| **Dependencies** | Camera Capture (frames), Arena Calibration (homography) |
| **Library** | `cv2.aruco` |

**Input Interface**:
- RGB frame via shared memory slot (referenced by `frame_metadata` from queue)
- Homography matrix from calibration (loaded once at startup, updated on recalibration)

**Output Interface**:
- `Queue("our_robot_detections")`: `RobotDetection` dataclass
- On detection failure: sends `RobotDetection` with `confidence=0.0` so downstream knows to rely on prediction

**Configuration**:
- `dictionary`: DICT_4X4_50
- `marker_ids`: [5, 6] (top and bottom of our robot)
- `corner_refinement`: False during combat, True during calibration
- `min_marker_perimeter_rate`: 0.03 (filter small false positives)

**Test Strategy**:
- Unit test: Feed synthetic images with known ArUco markers, verify detection and pose
- Benchmark test: Measure detection time across resolutions (640x480, 1280x720, 1920x1080)
- Robustness test: Test with motion-blurred, rotated, and partially occluded markers

---

### 10.4 Enemy Detector Module

| Property | Value |
|----------|-------|
| **Process** | P2 (Detection process, Thread B, runs in parallel with ArUco) |
| **Input** | RGB + Depth frames from shared memory |
| **Output** | `RobotDetection` for enemy robot |
| **Performance Budget** | <10ms per frame at 640x480 |
| **Dependencies** | Camera Capture (frames), Arena Calibration (homography, background reference), ArUco Detector (our robot position for exclusion mask) |
| **Library** | `cv2.BackgroundSubtractorMOG2`, `cv2.TrackerMOSSE` |

**Input Interface**:
- RGB + Depth frames via shared memory
- `our_robot_detection` from ArUco detector (for exclusion masking)
- Background reference frame (captured during calibration)

**Output Interface**:
- `Queue("enemy_detections")`: `RobotDetection` dataclass
- Includes bounding box and contour area for dashboard visualization

**Detection Pipeline (per frame)**:
1. Depth threshold: mask pixels where depth < (floor_depth - 2cm)
2. MOG2 foreground mask on RGB
3. Bitwise AND of depth mask and MOG2 mask
4. Erode/dilate to remove noise
5. Subtract our robot's bounding box from mask
6. Find contours, select largest as enemy
7. MOSSE tracker update on enemy ROI for frame-to-frame continuity
8. Apply homography to convert centroid to arena coordinates

**Test Strategy**:
- Unit test: Synthetic frames with known object positions, verify detection
- Integration test: Record arena footage, replay through detector, compare with ground truth
- Edge case test: Overlapping robots, robot on logo, robot near wall

---

### 10.5 State Estimator / Tracker Module

| Property | Value |
|----------|-------|
| **Process** | P3 (Strategy process, first stage) |
| **Input** | Robot detections from P2, IMU data from P1 |
| **Output** | `WorldState` with filtered positions, velocities, headings |
| **Performance Budget** | <1ms per update |
| **Dependencies** | ArUco Detector, Enemy Detector, Camera Capture (IMU) |
| **Library** | `filterpy` (KalmanFilter) |

**Input Interface**:
- `Queue("our_robot_detections")`: ArUco-based detections
- `Queue("enemy_detections")`: Background-subtraction-based detections
- `Queue("imu_data")`: IMU readings for heading fusion (Phase 2)

**Output Interface**:
- `WorldState` dataclass passed directly to Strategy Engine (same process)
- Published to `Queue("world_state")` for dashboard and telemetry

**State Vectors**:
- Our robot: `[x, y, vx, vy, heading, omega]` (6-state Kalman filter)
- Enemy robot: `[x, y, vx, vy]` (4-state Kalman filter)

**Kalman Filter Configuration**:
```
Process noise (Q): tuned for expected robot acceleration (~2 ft/s^2)
Measurement noise (R): tuned for ArUco detection noise (~0.5 inch std dev)
Prediction-only mode: activated when no detection received (occlusion)
Max prediction-only duration: 500ms before flagging position as uncertain
```

**Test Strategy**:
- Unit test: Feed synthetic detections with known noise, verify filter output convergence
- Simulation test: Generate a robot trajectory, add noise, verify filter tracks it
- Latency test: Measure time per predict/update cycle

---

### 10.6 Strategy Engine Module

| Property | Value |
|----------|-------|
| **Process** | P3 (Strategy process, second stage) |
| **Input** | `WorldState` from State Estimator |
| **Output** | Target point, behavior mode, aggression parameters |
| **Performance Budget** | <2ms per tick |
| **Dependencies** | State Estimator |
| **Library** | `py_trees` (Phase 2), custom FSM (Phase 1 MVP) |

**Input Interface**:
- `WorldState` directly from State Estimator (same process, no serialization)

**Output Interface**:
- `StrategyOutput`: `{target_xy, behavior_mode, aggression, max_speed}`
- Passed directly to Motion Planner (same process)
- Published to `Queue("strategy_state")` for dashboard visualization

**Behaviors (MVP FSM)**:
- `PURSUE`: Target point = enemy position. Drive toward enemy.
- `POSITION`: Target point = point behind enemy relative to pit direction. Maneuver to get pushing angle.
- `PUSH`: Target point = pit center (through enemy). Full speed ahead.
- `RETREAT`: Target point = center of arena (away from pit). Activate when too close to pit.
- `SEARCH`: Target point = sweep pattern. Activate when enemy not detected for >1s.

**Configurable Parameters** (via dashboard):
- `aggression`: 0.0-1.0 (affects how directly we attack vs. position carefully)
- `retreat_pit_distance`: distance from pit that triggers retreat (default: 1.5 ft)
- `push_alignment_threshold`: angle tolerance for POSITION->PUSH transition (default: 30 deg)
- `pursuit_speed`: max speed during pursuit (default: 0.8)
- `push_speed`: max speed during push (default: 1.0)

**Test Strategy**:
- Unit test: Given specific WorldState snapshots, verify correct behavior selection
- Simulation test: Run strategy against a simulated enemy with simple motion patterns
- Replay test: Feed recorded match WorldState data, verify strategy decisions

---

### 10.7 Motion Planner Module

| Property | Value |
|----------|-------|
| **Process** | P3 (Strategy process, third stage) |
| **Input** | `StrategyOutput` from Strategy Engine, `WorldState` |
| **Output** | Desired linear and angular velocity |
| **Performance Budget** | <1ms per update |
| **Dependencies** | Strategy Engine, State Estimator |
| **Library** | Custom (NumPy-based potential fields) |

**Input Interface**:
- `StrategyOutput` (target point, behavior mode)
- `WorldState` (current positions, pit location, arena bounds)

**Output Interface**:
- `MotionCommand`: `{linear_vel, angular_vel}` passed to Motor Controller

**Potential Field Configuration**:
```
Attractive field: target point (from strategy), strength proportional to distance
Repulsive field: pit boundary, strength = k / distance^2, range = 2 ft
Repulsive field: arena walls, strength = k / distance^2, range = 0.5 ft
Resultant: vector sum of all fields, clamped to max velocity
```

**Test Strategy**:
- Unit test: Given a target point and obstacle positions, verify correct velocity vector
- Visualization test: Plot potential field over arena map in dashboard
- Boundary test: Verify robot never gets a velocity pointing into the pit

---

### 10.8 Motor Controller Module

| Property | Value |
|----------|-------|
| **Process** | P3 (outputs to comms queue) |
| **Input** | `MotionCommand` (linear_vel, angular_vel) |
| **Output** | `MotorCommand` (left_speed, right_speed) |
| **Performance Budget** | <0.5ms per conversion |
| **Dependencies** | Motion Planner |
| **Library** | Custom (differential drive kinematics) |

**Input Interface**:
- `MotionCommand` from Motion Planner

**Output Interface**:
- `Queue("motor_commands")`: `MotorCommand` dataclass consumed by Comms Layer
- Override input: `Queue("manual_override")` from dashboard (manual joystick mode)

**Differential Drive Kinematics**:
```
left_speed  = (linear_vel - angular_vel * wheel_base / 2) / max_wheel_speed
right_speed = (linear_vel + angular_vel * wheel_base / 2) / max_wheel_speed
Clamp both to [-1.0, 1.0]
```

**Test Strategy**:
- Unit test: Verify kinematics for straight, turn-in-place, arc motions
- Integration test: Send known commands, verify ESP32 receives correct values

---

### 10.9 Comms Layer Module

| Property | Value |
|----------|-------|
| **Process** | P5 (dedicated process) |
| **Input** | `MotorCommand` from queue; UDP packets from ESP32 |
| **Output** | UDP packets to ESP32; IMU data to queue |
| **Performance Budget** | <2ms send latency; <5ms total RTT |
| **Dependencies** | Motor Controller (commands), Camera Capture (IMU forwarding) |
| **Library** | Python `asyncio`, `socket` |

**Input Interface**:
- `Queue("motor_commands")`: Motor commands from P3
- `Queue("manual_override")`: Manual commands from dashboard
- UDP socket: Telemetry packets from ESP32

**Output Interface**:
- UDP socket: Motor command packets to ESP32
- `Queue("imu_data")`: IMU readings forwarded to State Estimator
- `Queue("connection_status")`: Heartbeat status, latency, signal quality

**Protocol**:
- Motor command packet (8 bytes): `[0x01, seq_hi, seq_lo, left_hi, left_lo, right_hi, right_lo, checksum]`
- Telemetry packet (20 bytes): See section 7.5
- Heartbeat: ESP32 sends every 100ms; PC sends motor commands at 50 Hz (doubles as heartbeat)
- Failsafe: If no command sent in 200ms, send explicit stop command

**Test Strategy**:
- Unit test: Verify packet serialization/deserialization
- Loopback test: Send commands to ESP32 simulator, verify round-trip
- Latency test: Measure RTT with real ESP32 on local WiFi
- Failsafe test: Simulate dropped connection, verify stop command is sent

---

### 10.10 Web Dashboard Backend Module

| Property | Value |
|----------|-------|
| **Process** | P4 (dedicated process) |
| **Input** | WorldState, strategy state, connection status, frames (all via queues) |
| **Output** | HTTP REST API, WebSocket streams to browser |
| **Performance Budget** | Dashboard updates at 10+ FPS; API response <50ms |
| **Dependencies** | All other modules (consumes their published state) |
| **Library** | FastAPI, uvicorn, WebSocket |

**Input Interface**:
- `Queue("world_state")`: Robot positions, velocities
- `Queue("strategy_state")`: Current behavior, target point
- `Queue("connection_status")`: ESP32 link health
- `Queue("telemetry_metrics")`: FPS, latency histograms
- `Queue("frame_jpeg")`: Compressed frame for live camera view (optional, at reduced rate)

**Output Interface**:
- REST API: Configuration CRUD, calibration commands, strategy parameter updates
- WebSocket `/ws/arena`: Real-time arena state at 10-30 Hz
- WebSocket `/ws/telemetry`: Performance metrics at 5 Hz
- WebSocket `/ws/camera`: JPEG frames at 5-10 FPS for live camera view

**API Endpoints**:
```
GET  /api/status              -- system health, connection status
GET  /api/calibration         -- current calibration data
POST /api/calibration/start   -- begin calibration wizard
POST /api/calibration/capture -- capture a calibration point
GET  /api/strategy/params     -- current strategy parameters
PUT  /api/strategy/params     -- update strategy parameters
POST /api/control/override    -- switch to manual control mode
POST /api/control/autonomous  -- switch to autonomous mode
POST /api/control/estop       -- emergency stop
GET  /api/recording/list      -- list recorded matches
POST /api/recording/start     -- start recording
POST /api/recording/stop      -- stop recording
```

**Test Strategy**:
- Unit test: Test each endpoint with mock data
- Integration test: Run with real pipeline, verify WebSocket delivers correct data
- Load test: Verify dashboard handles 10+ concurrent browser connections

---

### 10.11 Telemetry & Logging Module

| Property | Value |
|----------|-------|
| **Process** | P6 (dedicated process, low priority) |
| **Input** | All published queues from other modules |
| **Output** | Log files, metrics endpoint, recording files |
| **Performance Budget** | No real-time constraint; must not back-pressure producers |
| **Dependencies** | All other modules (read-only consumer) |
| **Library** | Python `logging`, `msgpack` for recording, Prometheus client |

**Input Interface**:
- Subscribes to all published queues (non-blocking reads)
- Each queue message includes a timestamp for ordering

**Output Interface**:
- Log files: `logs/tracker_YYYYMMDD_HHMMSS.log` (structured JSON lines)
- Recording files: `recordings/match_YYYYMMDD_HHMMSS.msgpack` (all events in sequence)
- Prometheus metrics endpoint: `/metrics` on port 9090

**Metrics Collected**:
- `pipeline_fps`: Frames processed per second
- `pipeline_latency_ms`: End-to-end latency histogram (capture to command send)
- `detection_latency_ms`: Per-module detection time
- `aruco_detection_rate`: Fraction of frames with successful ArUco detection
- `enemy_detection_rate`: Fraction of frames with enemy detected
- `comms_rtt_ms`: ESP32 round-trip time
- `comms_packet_loss`: Fraction of dropped packets
- `strategy_state`: Current behavior mode (as label)

**Recording Format**:
- Sequence of timestamped events: `{timestamp_ns, module, event_type, data}`
- Can be replayed through any module for debugging
- Approximately 1-2 MB per minute of match recording

**Test Strategy**:
- Unit test: Verify recording writes and reads back correctly
- Replay test: Record a session, replay through state estimator and strategy engine, verify identical outputs
- Performance test: Verify logging does not add >1ms to any pipeline stage

---

## 11. Performance Architecture

### 11.1 Pipeline Parallelism Strategy

The system uses **pipeline parallelism** across OS processes to maximize throughput while minimizing latency:

```
Process    | Core Affinity | GIL Impact | Parallelism Method
-----------|---------------|------------|-------------------
P1 Camera  | Core 0        | Bypassed   | Separate process, shared memory output
P2 Detect  | Core 1-2      | Bypassed   | Separate process, 2 threads (ArUco + Enemy)
P3 Strategy| Core 3        | Bypassed   | Separate process, tight compute loop
P4 Web     | Any           | Minimal    | Separate process, async I/O
P5 Comms   | Any           | Minimal    | Separate process, async I/O
P6 Telemetry| Any          | None       | Separate process, non-blocking reads
```

**Why multiprocessing over threading**: Python's GIL means threads cannot execute CPU-bound code in parallel. Detection (OpenCV ArUco, MOG2) and strategy (Kalman filter, behavior tree) are CPU-bound. By placing them in separate OS processes, they truly run in parallel on different CPU cores.

**Why shared memory over queues for frames**: A 640x480 RGB frame is ~900KB. Serializing and deserializing this through `multiprocessing.Queue` (which uses pickle) adds 2-5ms per frame. Shared memory provides zero-copy access: the producer writes the frame once, the consumer reads it directly from the same physical memory location.

**Queue usage**: Structured data (detections, world state, commands) is small (<1KB) and is sent through `multiprocessing.Queue` with `msgpack` serialization. Overhead: <0.1ms per message.

### 11.2 Latency Budget Breakdown

Target: Camera capture to motor command sent in **<30ms** (below the PRD's 50ms target, leaving margin).

| Pipeline Stage | Budget | Description |
|---------------|--------|-------------|
| Camera capture + USB transfer | 5ms | DepthAI frame acquisition to shared memory |
| ArUco detection | 5ms | OpenCV ArUco on 640x480 (no corner refinement) |
| Enemy detection | 8ms | MOG2 + depth threshold + contour + MOSSE (parallel with ArUco) |
| Detection-to-strategy queue | 0.5ms | msgpack serialize, enqueue, dequeue, deserialize |
| State estimation (Kalman) | 0.5ms | Predict + update for both robots |
| Strategy decision (BT/FSM) | 1ms | Behavior tree tick or FSM transition |
| Motion planning (potential fields) | 0.5ms | Compute velocity from field sum |
| Motor controller (kinematics) | 0.2ms | Differential drive math |
| Strategy-to-comms queue | 0.3ms | Enqueue motor command |
| UDP send to ESP32 | 2ms | Network transmission (one-way) |
| **Total** | **~18ms** | Well under 30ms budget |

Note: ArUco and Enemy detection run in parallel threads within P2, so the detection stage time is `max(ArUco, Enemy)` = ~8ms, not their sum.

### 11.3 Memory Management

| Resource | Size | Lifetime | Cleanup |
|----------|------|----------|---------|
| Shared memory ring buffer | ~6 MB | Process lifetime | Explicitly unlinked on shutdown |
| Per-frame detection results | <1 KB each | Consumed and freed by queue reader | GC or explicit del |
| MOG2 background model | ~5 MB | Enemy detector lifetime | Freed on detector shutdown |
| MOSSE tracker state | ~100 KB | Enemy detector lifetime | Re-initialized on track loss |
| Kalman filter matrices | <10 KB per filter | State estimator lifetime | Constant size |
| Recording buffer | 1-2 MB/min | Match duration | Flushed to disk periodically |
| Web dashboard frame JPEG | ~50 KB | Replaced each frame | Circular buffer |

**Memory safety**:
- Shared memory is created by P1 and attached by P2. If P2 crashes, the shared memory remains valid and P2 can reattach on restart.
- Ring buffer uses a write index with memory barriers (via `multiprocessing.Value`) to prevent torn reads.
- Queues have `maxsize` set to prevent unbounded memory growth if a consumer falls behind. Oldest items are dropped.

### 11.4 Profiling and Benchmarking Strategy

**Built-in instrumentation**: Every module wraps its main processing loop with timing code:

```python
class ModuleBase:
    def process(self, input_data):
        start = time.perf_counter_ns()
        result = self._do_work(input_data)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        self.metrics.observe("processing_time_ms", elapsed_ms)
        return result
```

**Benchmarking tools**:
- `scripts/benchmark_pipeline.py`: Runs the full pipeline with recorded frames, measures per-stage latency
- `scripts/benchmark_detection.py`: Tests ArUco and enemy detection at various resolutions and parameters
- `scripts/benchmark_kalman.py`: Measures Kalman filter predict/update cycle time with varying state sizes

**Profiling approach**:
1. Use `cProfile` for per-function profiling within each process
2. Use `py-spy` for sampling profiler across all processes (no code changes needed)
3. Use Prometheus + Grafana for live latency dashboards during testing
4. Record per-frame timestamps at each pipeline stage for offline waterfall analysis

**Performance regression testing**:
- CI runs benchmark scripts on every merge
- Alert if any stage exceeds its latency budget by >20%
- Track FPS over time to catch gradual degradation

**Optimization priorities** (when latency budget is exceeded):
1. Reduce frame resolution (640x480 -> 320x240) for detection
2. Skip frames (process every 2nd frame) for enemy detection if ArUco is fast enough
3. Move OpenCV operations to GPU via `cv2.cuda` if available
4. Consider Cython or C extension for hot inner loops (Kalman filter, potential fields)
