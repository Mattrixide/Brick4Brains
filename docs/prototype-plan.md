# CV Tracking Prototype Plan

A webcam-based prototype to validate the core tracking approach for the combat robot tracker before committing to the full OAK-D Pro implementation.

---

## 1. Research Findings

### 1.1 OpenCV ArUco Detection

**Performance:** `detectMarkers()` takes 7-20ms per frame on typical hardware depending on resolution and lighting. At 480p, detection is comfortably under 10ms, leaving budget for other processing.

**Key optimizations discovered:**
- Use `DICT_4X4_50` -- smallest dictionary, fewest bits to decode, fastest detection
- Tune `minMarkerPerimeterRate` (default 0.03) -- setting too low causes the detector to evaluate far more contours, killing performance
- Disable corner subpixel refinement (`cornerRefinementMethod = CORNER_REFINE_NONE`) for speed, enable only when accuracy matters more than FPS
- Process at 480p for detection, display at native resolution
- Convert to grayscale once, reuse for all detection

**API (OpenCV 4.x):**
```python
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
corners, ids, rejected = detector.detectMarkers(gray_frame)
```

**Sources:**
- [OpenCV ArUco Tutorial](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [PyImageSearch ArUco Detection](https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/)
- [OpenCV Issue #26686 -- performance under challenging conditions](https://github.com/opencv/opencv/issues/26686)

### 1.2 High-FPS Threaded Webcam Capture

**The problem:** `cv2.VideoCapture.read()` is blocking. The main thread stalls waiting for the camera hardware, limiting throughput.

**The solution:** Move frame grabbing to a background thread. PyImageSearch benchmarks show:
- Without threading: **29.97 FPS**
- With threading: **143.71 FPS** (379% improvement)
- With threading + display: **39.93 FPS** (38% improvement over non-threaded with display)

**Windows-specific:** Use `cv2.CAP_DSHOW` backend for DirectShow access. Set `CAP_PROP_BUFFERSIZE = 1` to avoid stale buffered frames.

**Pattern to reuse** (from PyImageSearch / allskyee gist):
```python
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = Lock()

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        with self.lock:
            return self.frame.copy()
```

**Sources:**
- [PyImageSearch -- Increasing webcam FPS](https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/)
- [allskyee threaded capture gist](https://gist.github.com/allskyee/7749b9318e914ca45eb0a1000a81bf56)
- [tobybreckon/python-examples-cv camera_stream.py](https://github.com/tobybreckon/python-examples-cv/blob/master/camera_stream.py)

### 1.3 Fast Object Trackers (MOSSE / KCF)

**MOSSE:** 450+ FPS. Extremely fast but less accurate. Best for frame-to-frame tracking between heavier detection cycles. Handles lighting changes and occlusion detection via peak-to-sidelobe ratio.

**KCF:** Slower than MOSSE but more accurate. Uses correlation filtering on overlapping patches. Good balance of speed and precision.

**Usage pattern:** Run full detection (ArUco or MOG2) every N frames. Between detections, use MOSSE/KCF to track the bounding box cheaply. Re-initialize tracker when detection finds the object.

**API:**
```python
tracker = cv2.legacy.TrackerMOSSE_create()
tracker.init(frame, bbox)
success, bbox = tracker.update(frame)
```

**Sources:**
- [LearnOpenCV -- Object Tracking](https://learnopencv.com/object-tracking-using-opencv-cpp-python/)
- [PyImageSearch -- OpenCV Object Tracking](https://pyimagesearch.com/2018/07/30/opencv-object-tracking/)

### 1.4 MOG2 Background Subtraction

**Performance:** 64 FPS processing 10,000 frames. 3x faster than MOG, 10x faster than GMG. The fastest OpenCV background subtractor.

**Key parameters:**
- `history`: Number of frames for background model (default 500; lower = faster adaptation)
- `varThreshold`: Sensitivity (default 16; higher = less sensitive)
- `detectShadows`: Set to `False` for speed -- shadow detection adds significant overhead

**Post-processing for clean contours:**
```python
bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)
mask = bg_sub.apply(frame)
# morphological cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

**Sources:**
- [Medium -- Detecting Moving Objects with Background Subtractors](https://medium.com/@siromermer/detecting-and-tracking-moving-objects-with-background-subtractors-using-opencv-f2ff7f94586f)
- [OpenCV MOG2 Class Reference](https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)

### 1.5 Kalman Filter for Tracking

**OpenCV's built-in KalmanFilter** is sufficient for prototype 2D tracking with a constant-velocity model:
- State: `[x, y, vx, vy]` (4 states)
- Measurement: `[x, y]` (2 measurements)
- Transition matrix models constant velocity between frames

**For the full system:** FilterPy library provides EKF/UKF needed for camera+IMU sensor fusion, but for the webcam prototype, OpenCV's linear Kalman filter is enough.

**Sources:**
- [OpenCV Kalman sample](https://github.com/opencv/opencv/blob/master/samples/python/kalman.py)
- [RahmadSadli/2-D-Kalman-Filter](https://github.com/RahmadSadli/2-D-Kalman-Filter)
- [MachinelearningSpace -- 2D Object Tracking with Kalman Filter](https://machinelearningspace.com/2d-object-tracking-using-kalman-filter/)

### 1.6 Related Projects

**RoboCup SSL Vision** ([GitHub](https://github.com/RoboCup-SSL/ssl-vision)) -- Shared vision system for small robot soccer. Uses overhead cameras at 60Hz to track robots via colored dot patterns. Architecture: capture -> detect -> track -> broadcast. Closest existing system to our use case. Written in C++ but the architecture pattern is directly applicable.

---

## 2. Architecture

The prototype mirrors the PRD's pipeline architecture but simplified for a single process with threads:

```
+-----------------+     +-----------------+     +------------------+
| CameraCapture   |---->| Detector        |---->| Display          |
| (background     |     | (ArUco, Color,  |     | (overlays, FPS,  |
|  thread)        |     |  BGSub, or      |     |  trails, debug   |
|                 |     |  Combined)      |     |  info)           |
+-----------------+     +--------+--------+     +------------------+
                                 |
                        +--------v--------+
                        | KalmanTracker   |
                        | (smoothing,     |
                        |  prediction)    |
                        +-----------------+
```

**Module responsibilities:**

| Module | PRD Equivalent | Responsibility |
|--------|---------------|----------------|
| `CameraCapture` | P1: Camera Capture | Threaded frame acquisition, FPS measurement, resolution control |
| `Detector` | P2: Detection + Tracking | ArUco detection, color tracking, MOG2 background subtraction |
| `KalmanTracker` | State Estimator | Kalman filter smoothing, velocity estimation, prediction |
| `Display` | P4: Web Dashboard (simplified) | Visualization, overlays, debug info, keyboard controls |

**Data flow types** (matching PRD dataclass patterns):

```python
@dataclass
class Detection:
    label: str              # "aruco_0", "color", "motion"
    center_px: tuple        # (x, y) in pixels
    bbox: tuple             # (x, y, w, h)
    heading_rad: float      # rotation (ArUco only)
    confidence: float
    timestamp: float

@dataclass
class TrackedObject:
    detection: Detection
    predicted_px: tuple     # Kalman-predicted position
    velocity_px: tuple      # estimated (vx, vy) in px/frame
    trail: list             # last N positions
```

---

## 3. File Structure

```
tracker/
  prototype/
    main.py              # Entry point, mode switching, keyboard controls, display loop
    capture.py           # ThreadedCamera class (threaded webcam capture)
    detectors.py         # ArUcoDetector, ColorDetector, BackgroundSubDetector classes
    kalman_tracker.py    # KalmanTracker class wrapping cv2.KalmanFilter
```

| File | Lines (est.) | Description |
|------|-------------|-------------|
| `main.py` | ~150 | Main loop, mode switching (keys 1-4), display rendering, FPS overlay, trail drawing, keyboard controls |
| `capture.py` | ~60 | ThreadedCamera: background thread capture, resolution setting, FPS counting, CAP_DSHOW on Windows |
| `detectors.py` | ~130 | Three detector classes sharing a common interface: `detect(frame) -> list[Detection]`. ArUco with pose, HSV color with configurable range, MOG2 with contour filtering |
| `kalman_tracker.py` | ~60 | Wraps cv2.KalmanFilter for 2D constant-velocity tracking. predict() and update() methods. Maintains trail history. |
| **Total** | **~400** | |

---

## 4. Implementation Steps

Build order is designed so each step produces a runnable, testable result:

### Step 1: Threaded Camera Capture (`capture.py` + minimal `main.py`)
- Implement `ThreadedCamera` class with `start()`, `read()`, `stop()`
- Use `cv2.CAP_DSHOW` on Windows, configurable resolution
- Add FPS counter (capture thread FPS + display loop FPS)
- **Test:** Run and verify live webcam feed with FPS overlay, try 720p vs 480p

### Step 2: ArUco Detection (`detectors.py` -- ArUcoDetector)
- Create `ArUcoDetector` class using `DICT_4X4_50`
- Detect markers, extract corners, IDs, compute center position
- Compute heading from corner geometry (atan2 of top-right vs top-left)
- Draw detected markers with `cv2.aruco.drawDetectedMarkers()`
- Draw axis/heading arrow on each marker
- **Test:** Print an ArUco 4x4 marker (ID 0-5), hold in front of webcam, verify detection + heading

### Step 3: Color-Based Detection (`detectors.py` -- ColorDetector)
- HSV-based color detection with configurable range
- Default to bright green or orange (easy to find objects for)
- Find contours, filter by minimum area, compute bounding box + center
- Add trackbar UI for live HSV range tuning (optional, via keyboard toggle)
- **Test:** Hold a brightly colored object, verify bounding box tracking

### Step 4: Background Subtraction (`detectors.py` -- BackgroundSubDetector)
- MOG2 with `detectShadows=False`
- Morphological cleanup (open + close)
- Contour filtering by minimum area
- Compute bounding box + center for each moving object
- **Test:** Move hand or object across static background, verify detection

### Step 5: Kalman Filter Tracker (`kalman_tracker.py`)
- 4-state constant-velocity Kalman filter `[x, y, vx, vy]`
- `predict()` returns predicted position
- `update(measured_x, measured_y)` corrects the filter
- Maintain trail (deque of last 50 positions)
- Visualize: draw predicted position (different color) vs measured position
- **Test:** Track ArUco marker or colored object with Kalman overlay, observe smoothing

### Step 6: Mode Switching + Combined Mode (`main.py` finalization)
- Keyboard controls:
  - `1` -- ArUco mode
  - `2` -- Color tracking mode
  - `3` -- Background subtraction mode
  - `4` -- Combined mode (ArUco = "our robot", BGSub = "enemy")
  - `t` -- Toggle trail drawing
  - `k` -- Toggle Kalman filter overlay
  - `f` -- Cycle through resolutions (480p / 720p / 1080p)
  - `q` / `ESC` -- Quit
- Combined mode: run ArUco detector + BGSub detector simultaneously, different colors for each
- Display: FPS counters (capture + processing), mode label, active overlays
- **Test:** Switch modes, verify all work, check combined mode with ArUco card + moving hand

---

## 5. Dependencies

```
opencv-python>=4.8.0       # ArUco, MOG2, Kalman, drawing, video capture
opencv-contrib-python>=4.8.0  # legacy trackers (MOSSE/KCF) if needed
numpy>=1.24.0              # array operations
```

**Install:** `pip install opencv-contrib-python numpy`

(opencv-contrib-python includes everything in opencv-python plus the contrib modules like legacy trackers)

No other dependencies required. The prototype deliberately avoids heavy dependencies to stay lightweight.

---

## 6. Test Plan

| # | Test | How to verify | Pass criteria |
|---|------|--------------|---------------|
| 1 | Camera capture | Run `main.py`, observe live feed | Feed displays without lag, FPS counter shows > 25 FPS |
| 2 | Resolution switching | Press `f` to cycle resolutions | Resolution changes visible, FPS counter updates accordingly |
| 3 | ArUco detection | Print ArUco 4x4 marker (IDs 0-5), hold in front of camera | Marker outlined, ID displayed, heading arrow drawn |
| 4 | ArUco at speed | Move ArUco marker quickly across frame | Detection maintains at moderate speed; note speed where detection drops |
| 5 | Color tracking | Hold bright colored object in frame | Bounding box follows object, center point drawn |
| 6 | Background subtraction | Start with static scene, then move object/hand | Moving objects detected with bounding boxes; static scene produces no false positives after warmup |
| 7 | Kalman smoothing | Track object with Kalman enabled (`k` key) | Predicted position (red) is smoother than measured position (green); prediction leads slightly during fast motion |
| 8 | Combined mode | Hold ArUco card in one hand, wave other hand | ArUco tracked as "ours" (blue), moving hand tracked as "enemy" (red), both trails visible |
| 9 | FPS under load | Run combined mode at 480p | Processing FPS stays above 20 FPS |
| 10 | Mode switching | Press 1-4 rapidly | Mode switches cleanly without crashes or stale state |

### Manual ArUco marker generation

Generate printable markers with:
```python
import cv2
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker = cv2.aruco.generateImageMarker(dictionary, 0, 200)  # ID=0, 200px
cv2.imwrite("aruco_marker_0.png", marker)
```

---

## 7. What This Prototype Validates

| Technical Risk | What we learn |
|---------------|---------------|
| **ArUco detection speed at resolution** | Can we detect ArUco markers fast enough at 480p? What's the FPS ceiling? Does detection survive motion blur? |
| **Threading model for capture** | Does the threaded capture pattern give us enough FPS headroom? Is the lock-based frame sharing fast enough? |
| **MOG2 for enemy detection** | Can background subtraction reliably detect a moving object against a static background? How much morphological cleanup is needed? |
| **Kalman filter value** | Does Kalman smoothing meaningfully improve tracking quality? Is the constant-velocity model good enough, or do we need acceleration? |
| **Combined pipeline budget** | Can we run ArUco + BGSub + Kalman + display all within a single frame budget at 30+ FPS? |
| **Module interfaces** | Do the Detection and TrackedObject dataclass patterns work well for passing data between pipeline stages? |
| **Resolution vs FPS tradeoff** | What resolution gives us the best balance of detection reliability and processing speed? |
| **Windows webcam behavior** | Does CAP_DSHOW behave well? Any buffering issues? Any resolution limitations? |

### What this does NOT validate (deferred to OAK-D Pro phase)
- Depth-based detection (no depth sensor on webcam)
- Multi-process shared memory pipeline (prototype is single-process)
- IMU sensor fusion (no IMU in prototype)
- Network latency to ESP32
- DepthAI SDK integration

---

## 8. Estimated Lines of Code

| File | Lines | Complexity |
|------|-------|-----------|
| `main.py` | ~150 | Medium -- mode switching logic, display rendering, keyboard handling |
| `capture.py` | ~60 | Low -- well-understood threaded pattern |
| `detectors.py` | ~130 | Medium -- three detector classes, each ~40 lines |
| `kalman_tracker.py` | ~60 | Low -- thin wrapper around cv2.KalmanFilter |
| **Total** | **~400** | Buildable in a single focused session |
