# Shared Context — Brick for Brains

This file is read and updated by all agents. Keep it organized by topic.

---

## Architecture Decisions

- **Pipeline**: Camera Capture → Detection → Tracking → Strategy → Motor Control
- **Multi-process**: Each pipeline stage runs in its own process with shared memory (production)
- **Single-process**: Prototypes use threading for simplicity
- **Tracking**: ArUco 4x4_50 for our robot, background subtraction + depth for enemy
- **Control**: Behavior tree strategy engine (Phase 2), basic pursuit (Phase 1)
- **Communication**: WiFi UDP to ESP32 (production), TX15 SBUS (prototyping)

## Hardware

- **Camera**: OAK-D Pro, 1080p/60fps, fixed-focus, mounted outside arena at 6-7ft
- **Transmitter**: RadioMaster TX15 with EdgeTX, Master/SBUS trainer mode
- **Robot**: Beetleweight (3lb), tank drive, ESP32 microcontroller, BNO055/085 IMU
- **Arena**: 8x8ft, 4ft polycarbonate walls, plywood ceiling with LEDs, corner pit

## Research Findings

- ArUco detection: 7-20ms per frame, DICT_4X4_50 is fastest, disable corner refinement for speed
- Threaded webcam capture: 379% FPS improvement over blocking reads
- MOG2 background subtraction: 64 FPS, fastest OpenCV background subtractor
- MOSSE tracker: 450+ FPS for frame-to-frame tracking between detections
- Camera exposure: 400-500us freezes motion at 8ft/s to <1px blur
- WiFi UDP latency: 5-15ms RTT (meets <20ms target)

### ArUco Optimization Deep-Dive (see `docs/aruco-optimization-research.md`)

- **MJPG codec**: Switch from YUY2 to MJPG for 2-3x higher capture FPS at 720p+. Set fourcc before resolution.
- **Adaptive threshold tuning**: Reduce `adaptiveThreshWinSizeMax` from 23 to 15, step from 10 to 6 — gives 3 passes instead of 3 (same count but tighter range). Single-pass (Min=Max=5) is fastest but risky.
- **Marker perimeter rates**: Set `minMarkerPerimeterRate=0.04`, `maxMarkerPerimeterRate=0.25` for 720p to reject false candidates before expensive decode.
- **720p is the sweet spot**: 50mm markers appear as 17-33px/side at 720p (1.5-3m range). Min detectable is ~13px/side. 1080p gives 25-50px but costs 2x processing.
- **Corner refinement**: CORNER_REFINE_NONE is fastest (1x). APRILTAG doubles detection range but is 5-8x slower. Use NONE for tracking, SUBPIX only for calibration.
- **CLAHE preprocessing**: Useful for uneven arena LED lighting. Apply to grayscale, ~1.5ms cost. Needs testing under actual conditions.
- **ROI cropping**: Crop to arena region for proportional speedup. Combine with Kalman prediction for tighter ROI.
- **Do NOT use**: Gaussian blur (hurts borders), global histogram EQ (amplifies noise), Python multiprocessing with ArUco (known hang bug #11140), UMat/GPU for ArUco (only 5-6% GPU utilization).
- **Default exposure should be -11 (~488us)** not -7 (~7.8ms) for combat speeds.
- **DICT_4X4_50 confirmed optimal**: Lowest pixel requirement, maximum inter-marker distance for <=50 IDs, fastest decode.
- **Projected end-to-end latency**: 8-15ms at 720p with tuned parameters (well under 30ms target).

## Dashboard

- **Milestones section** added to `dashboard/index.html` — expandable accordion cards for Milestones 0-6 with work items and status dots. Replaces the old simple timeline.
- TPM should update milestone work item statuses after planning sessions (look for the comment "work items updated after planning sessions" in the HTML).

## Known Issues

- TX15 USB serial driver fails on Windows 11 (STM32 CDC composite device error code 10)
- Plywood ceiling blocks overhead camera mounting — must mount outside arena
- Depth accuracy is marginal at 2m range — need hybrid depth + appearance detection

## Prototypes

### prototypes/cv-tracking/ (CV Tracking - COMPLETE)
- Validates: ArUco detection, color tracking, BGSub, Kalman filtering, threaded capture
- Status: Working, tested with multiple cameras and resolutions

### prototypes/drive-test/ (TX15 Drive Test - IN PROGRESS)
- Validates: SBUS motor control via TX15, ArUco-guided autonomous driving
- Status: Code complete, blocked on TX15 USB driver issue
- Test plan: Spin left 360, spin right 360 + manual WASD control
