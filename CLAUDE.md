# Brick for Brains

Autonomous combat robotics system — computer vision tracking, strategy engine, and ESP32 motor control for beetleweight arena combat.

## Project Structure

```
B4B/
  docs/              # PRD, research, prototype plans
  dashboard/         # Static HTML project dashboard (open index.html)
  prototype/         # Webcam CV tracking prototype (ArUco, color, BGSub, Kalman)
  drive-test/        # TX15 SBUS drive test prototype (ArUco tracking + motor control)
  .claude/
    agents/          # Agent role definitions (prototyper, researcher, architect, pm)
    shared-context.md # Shared knowledge base — all agents read/write this
```

## Key Technical Details

- **Arena**: 8x8ft beetleweight, 4ft polycarbonate walls, plywood ceiling (no ceiling access)
- **Our robot**: ArUco 4x4_50 markers (50mm), ID #1
- **Camera**: OAK-D Pro (production), USB webcam index 1 (prototyping)
- **Transmitter**: RadioMaster TX15, Master/SBUS trainer mode, USB serial
- **Motor control**: ESP32 via WiFi UDP (production), TX15 SBUS (prototyping)
- **Latency target**: <30ms end-to-end (camera frame to motor command)
- **FPS target**: 60 FPS minimum

## Conventions

- Python 3.12 at `C:\Users\mattr\AppData\Local\Programs\Python\Python312\python.exe`
- Each prototype is self-contained with its own `requirements.txt`
- Use `cv2.CAP_DSHOW` backend on Windows for camera access
- External webcam = camera index 1, built-in = camera index 0
- All agents read `.claude/shared-context.md` before starting work

## Current Blockers

- TX15 USB serial driver not working on Windows 11 (STM32 VID:0483 PID:5740 CDC composite device, usbser fails code 10)
