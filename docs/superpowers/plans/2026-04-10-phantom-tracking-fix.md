# Phantom Tracking Fix Implementation Plan (v5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate phantom static-blob tracking (40% of battle frames) by breaking the self-perpetuating healing exclusion cycle and adding a raw-detection displacement gate with "was ever moving" discrimination.

**Architecture:** Three changes to `enemy_tracker.py`:

1. **Raw-detection displacement gate** — Accumulate raw detection positions (pre-Kalman, `det_cm`) over 1-second windows and compare window means. Raw detections have sigma_meas=8cm, so the mean of ~50 detections has std ≈ 1.1cm. Two consecutive window means have displacement noise ≈ 1.6cm — well below the 5cm threshold. (Previous versions used Kalman position, which has a 5.8cm displacement noise floor due to high process noise sigma_a=5.0.)

2. **"Was ever moving" discrimination** — Phantoms were NEVER moving; real opponents almost always move before stopping. Never-moved tracks get a 1-second gate. Was-moving tracks are NEVER dropped by the displacement gate (only by Kalman coast timeout after detection stops). This makes pins completely safe.

3. **Tracking-state healing protection** — Protect tracked region from reference healing while `is_tracking=True`. When the gate kills a never-moved track, protection stops and the phantom dissolves. Real enemies that were ever moving are never killed by the gate, so they're always protected.

**Tech Stack:** Python 3.12, OpenCV, NumPy

**Key design decisions from expert review rounds 1-5:**
- Kalman position displacement noise floor is 5.8cm (sigma_a=5.0). Raw detection mean over 1s has noise ≈ 1.6cm. Use raw detections.
- First Kalman check after init has 10.6cm mean displacement — would falsely set was_ever_moving 50% of the time. Raw detection means avoid this.
- Was-moving tracks should NEVER be dropped by the gate (combat tactics review: pins, pushes, spinner spin-up all create legitimate 1-3s stationary periods). Only Kalman coast timeout (0.5s after detection stops) kills was-moving tracks.
- Healing protection tied to `is_tracking` — simpler and handles any pin duration.
- Post-gate cooldown: after gate kills a phantom, suppress re-detection at that position for 20 frames so healing can dissolve the blob. Log data confirms re-detection in 1-42 frames without cooldown; healing needs ~70 frames.
- Camera is static (overhead OAK-D Pro) — no parallax from bot movement.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `prototypes/auto-drive/enemy_tracker.py` | Modify | All tracking changes |
| `prototypes/auto-drive/main.py` | Modify | Frame logging of new debug signals |

---

### Task 1: Add Raw Detection Accumulator and Flags to EnemyTracker

**Files:**
- Modify: `prototypes/auto-drive/enemy_tracker.py:8-10` (imports)
- Modify: `prototypes/auto-drive/enemy_tracker.py:661-668` (EnemyTracker `__init__`)

- [ ] **Step 1: Add imports at module level**

After line 10 (`import cv2`), add:

```python
import logging
import time
```

Verify neither is already imported. Skip if present.

- [ ] **Step 2: Add new state to EnemyTracker.__init__()**

After line 667 (`self._last_detection_cm = None`), add:

```python
# Phantom displacement gate — uses raw detection means (not Kalman)
self._det_accum = []                # list of (x_m, y_m) raw detections in current window
self._det_window_start = 0.0        # time.perf_counter() when current window started
self._prev_window_mean = None       # mean position of previous window (meters)
self._was_ever_moving = False        # True if displacement ever exceeded 10cm between windows
self._stationary_windows = 0         # consecutive windows with displacement < 5cm
self._gate_cooldown = 0              # frames to suppress re-detection after gate fires
self._gate_cooldown_px = None        # pixel position to suppress (from _track_lock_px)
```

- [ ] **Step 3: Commit**

```bash
git add prototypes/auto-drive/enemy_tracker.py
git commit -m "Add raw detection accumulator and phantom gate state"
```

---

### Task 2: Implement Raw-Detection Displacement Gate

After each Kalman update, accumulate raw detections. Every 1 second, compute window mean, compare to previous window mean, and check displacement. Only never-moved tracks get dropped.

**Files:**
- Modify: `prototypes/auto-drive/enemy_tracker.py:727-734` (after Kalman update in `update()`)

- [ ] **Step 1: Add displacement gate after Kalman update**

In `update()`, BEFORE the `detect()` call (before line 683), add the cooldown suppression:

```python
# Post-gate cooldown: suppress re-detection at old phantom position
# Gives healing ~20 frames to dissolve the phantom from the reference
if self._gate_cooldown > 0:
    self._gate_cooldown -= 1
```

Then after line 729 (`self.kalman.update(det_cm)`) and before line 731 (`vel = ...`), insert:

```python
# Raw-detection displacement gate — detect stationary phantom tracks
# Uses pre-Kalman detections (noise ~1.6cm between window means)
# instead of Kalman position (noise ~5.8cm due to high process noise)
_log = logging.getLogger(__name__)
now_t = time.perf_counter()
if det_cm is not None:
    self._det_accum.append(det_cm)

    # Initialize window start on first detection
    if self._det_window_start == 0.0:
        self._det_window_start = now_t

    # Every 1 second, compute window mean and check displacement
    if now_t - self._det_window_start >= 1.0 and len(self._det_accum) >= 25:
        # Compute mean of raw detections in this window
        accum = np.array(self._det_accum)
        window_mean = accum.mean(axis=0)  # [x_m, y_m]

        if self._prev_window_mean is not None:
            displacement_cm = np.linalg.norm(window_mean - self._prev_window_mean) * 100.0

            # Track if this object has ever moved significantly
            if displacement_cm > 10.0:
                self._was_ever_moving = True

            if displacement_cm < 5.0:
                self._stationary_windows += 1
            else:
                self._stationary_windows = 0

            # Only drop never-moved tracks (phantoms)
            # Was-moving tracks (real enemies) are NEVER dropped by the gate
            # They only die via Kalman coast timeout when detection stops
            if not self._was_ever_moving and self._stationary_windows >= 1:
                _log.info(
                    "[tracker] PHANTOM GATE — displacement %.1fcm, never moved, dropping",
                    displacement_cm)
                # Save phantom position for cooldown suppression
                self._gate_cooldown = 20  # suppress re-detection for ~0.4s
                self._gate_cooldown_px = (
                    self.detector._track_lock_px if self.detector._track_lock_px is not None
                    else None
                )
                self.kalman.reset()
                self._det_accum = []
                self._det_window_start = 0.0
                self._prev_window_mean = None
                self._was_ever_moving = False
                self._stationary_windows = 0
                self.detector._track_lock_px = None
                self._last_detection_px = None
                self._last_detection_cm = None
            else:
                self._prev_window_mean = window_mean.copy()
                self._det_accum = []
                self._det_window_start = now_t
        else:
            # First window — just record the mean, no comparison yet
            self._prev_window_mean = window_mean.copy()
            self._det_accum = []
            self._det_window_start = now_t
```

Finally, in the `update()` method, AFTER `det_px` is obtained from `detect()` (around line 684) but BEFORE the coordinate conversion block (line 688), add the cooldown rejection:

```python
# Suppress re-detection near old phantom position during cooldown
if self._gate_cooldown > 0 and det_px is not None and self._gate_cooldown_px is not None:
    dist_to_old = math.sqrt(
        (det_px[0] - self._gate_cooldown_px[0])**2 +
        (det_px[1] - self._gate_cooldown_px[1])**2
    )
    if dist_to_old < 80:  # within 80px of old phantom — suppress
        det_px = None
```

- [ ] **Step 2: Verify the change compiles**

Run:
```bash
cd prototypes/auto-drive && "C:\Users\mattr\AppData\Local\Programs\Python\Python312\python.exe" -c "from enemy_tracker import EnemyTracker; t = EnemyTracker(); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add prototypes/auto-drive/enemy_tracker.py
git commit -m "Add raw-detection displacement gate for phantom elimination

Accumulates raw detections (pre-Kalman) over 1s windows and compares
window means. Displacement noise is ~1.6cm (vs 5.8cm for Kalman position).
Only drops never-moved tracks. Was-moving tracks are never gate-killed,
making pins and pushing matches completely safe."
```

---

### Task 3: Tie Healing Exclusion to Active Tracking State

Protect tracked objects from reference healing while `is_tracking=True`. When the displacement gate kills a never-moved track, `is_tracking` becomes False, protection stops, and the phantom dissolves.

**Files:**
- Modify: `prototypes/auto-drive/enemy_tracker.py:210-220` (EnemyDetector `__init__`)
- Modify: `prototypes/auto-drive/enemy_tracker.py:345-361` (healing block in `detect()`)
- Modify: `prototypes/auto-drive/enemy_tracker.py:669-684` (EnemyTracker `update()`)

- [ ] **Step 1: Add healing control attribute to EnemyDetector**

In `EnemyDetector.__init__()` (after line 217: `self._track_lock_px = None`), add:

```python
self._heal_protect = False  # set by EnemyTracker: True = protect from healing
```

- [ ] **Step 2: Condition the healing exclusion**

Replace lines 355-357 in `detect()`:

```python
# OLD:
if self._track_lock_px is not None:
    ex, ey = int(self._track_lock_px[0]), int(self._track_lock_px[1])
    cv2.circle(heal_mask, (ex, ey), 60, 0, -1)
```

With:

```python
# Only protect tracked enemy from healing while actively tracked
# When displacement gate kills a phantom track, is_tracking→False → heals away
if self._track_lock_px is not None and self._heal_protect:
    ex, ey = int(self._track_lock_px[0]), int(self._track_lock_px[1])
    cv2.circle(heal_mask, (ex, ey), 60, 0, -1)
```

- [ ] **Step 3: Set `_heal_protect` in EnemyTracker.update()**

Before the `detect()` call (before line 683), add:

```python
# Healing protection: protect tracked region while Kalman is actively tracking
self.detector._heal_protect = self.kalman.is_tracking
```

- [ ] **Step 4: Verify compiles**

Run:
```bash
cd prototypes/auto-drive && "C:\Users\mattr\AppData\Local\Programs\Python\Python312\python.exe" -c "from enemy_tracker import EnemyTracker; t = EnemyTracker(); print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add prototypes/auto-drive/enemy_tracker.py
git commit -m "Tie healing exclusion to active tracking state

Protect tracked region from reference healing while is_tracking=True.
Phantom gate kills track → protection stops → phantom heals away in ~0.5s.
Real enemies (was_ever_moving) are never gate-killed, always protected."
```

---

### Task 4: Log New Signals for Debugging

**Files:**
- Modify: `prototypes/auto-drive/main.py:1488-1520` (frame logging block)

- [ ] **Step 1: Add new fields to frame log**

In the frame logging section of `main.py`, after the IMU telemetry read (around line 1496), add:

```python
e_stw = self._enemy_tracker._stationary_windows
e_ever_moved = self._enemy_tracker._was_ever_moving
e_disp = 0.0
if self._enemy_tracker._prev_window_mean is not None and self._enemy_tracker.kalman._initialized:
    pos_m = self._enemy_tracker.kalman.position
    e_disp = round(float(np.linalg.norm(pos_m - self._enemy_tracker._prev_window_mean)) * 100.0, 1)
```

Then in the `rec` dict (after the `"imu_fails_total"` line), add:

```python
"e_stw": e_stw, "e_moved": e_ever_moved, "e_disp": e_disp,
```

- [ ] **Step 2: Verify numpy is imported in main.py**

Check that `import numpy as np` exists. If not, add it near the top.

- [ ] **Step 3: Commit**

```bash
git add prototypes/auto-drive/main.py
git commit -m "Log enemy stationary windows, was-ever-moving, displacement in frame JSONL"
```

---

### Task 5: Validate Against Saved Log Data

**Files:**
- Read: `prototypes/auto-drive/logs/good_runs/frames_20260409_203620.jsonl`

- [ ] **Step 1: Write and run validation script**

Create `prototypes/auto-drive/validate_phantom_fix.py`:

```python
"""Validate phantom tracking fix against saved battle log.

Simulates the raw-detection displacement gate using logged enemy position data.
Raw detections are approximated by (ex, ey) when ed=True.
"""
import json, math

log_path = "logs/good_runs/frames_20260409_203620.jsonl"
with open(log_path) as f:
    frames = [json.loads(l) for l in f]

t0 = frames[0]["t"]

DISP_THRESH_CM = 5.0
EVER_MOVING_THRESH_CM = 10.0
MIN_DETECTIONS = 25

det_accum = []
window_start = 0.0
prev_mean = None
was_ever_moving = False
stationary_windows = 0
gate_events = []

for f in frames:
    ex, ey, t, ed = f.get("ex"), f.get("ey"), f["t"], f.get("ed", False)
    if ex is None or ey is None or not ed:
        continue
    det_accum.append((ex / 100.0, ey / 100.0))  # cm to m
    if window_start == 0.0:
        window_start = t
    if t - window_start >= 1.0 and len(det_accum) >= MIN_DETECTIONS:
        mean_x = sum(d[0] for d in det_accum) / len(det_accum)
        mean_y = sum(d[1] for d in det_accum) / len(det_accum)
        window_mean = (mean_x, mean_y)
        if prev_mean is not None:
            disp_cm = math.sqrt((window_mean[0]-prev_mean[0])**2 + (window_mean[1]-prev_mean[1])**2) * 100
            if disp_cm > EVER_MOVING_THRESH_CM:
                was_ever_moving = True
            if disp_cm < DISP_THRESH_CM:
                stationary_windows += 1
            else:
                stationary_windows = 0
            if not was_ever_moving and stationary_windows >= 1:
                gate_events.append((f["f"], t - t0, disp_cm, len(det_accum)))
                was_ever_moving = False
                stationary_windows = 0
                prev_mean = None
                det_accum = []
                window_start = 0.0
                continue
        prev_mean = window_mean
        det_accum = []
        window_start = t

print(f"Frames: {len(frames)}, Duration: {frames[-1]['t'] - t0:.1f}s")
print(f"Phantom gate events: {len(gate_events)}")
for frame, t, disp, n_dets in gate_events:
    print(f"  f={frame} t={t:.1f}s disp={disp:.1f}cm ({n_dets} detections in window)")
```

Run:
```bash
cd prototypes/auto-drive && "C:\Users\mattr\AppData\Local\Programs\Python\Python312\python.exe" validate_phantom_fix.py
```

- [ ] **Step 2: Clean up**

```bash
rm prototypes/auto-drive/validate_phantom_fix.py
```

- [ ] **Step 3: Final commit**

```bash
git add prototypes/auto-drive/enemy_tracker.py prototypes/auto-drive/main.py
git commit -m "Phantom tracking fix: raw-detection displacement gate + tracking-state healing

Break the self-perpetuating cycle where tracked phantoms are excluded from
reference healing. Uses raw detection means (1.6cm noise) not Kalman position
(5.8cm noise) for displacement measurement. Only drops never-moved tracks;
was-moving tracks (real enemies) are never gate-killed, making pins safe.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Summary of Changes

| Change | Location | Effect |
|--------|----------|--------|
| Raw detection accumulator | `enemy_tracker.py` EnemyTracker | Accumulate det_cm over 1s windows, compute means |
| Displacement gate | `enemy_tracker.py` EnemyTracker.update() | Compare window means, drop never-moved tracks at <5cm |
| Post-gate cooldown | `enemy_tracker.py` EnemyTracker.update() | Suppress re-detection within 80px of phantom for 20 frames |
| Was-ever-moving flag | `enemy_tracker.py` EnemyTracker | Phantom (never moved) = 1s gate; real enemy = never dropped |
| Tracking-state healing | `enemy_tracker.py` EnemyDetector.detect() | Protect while `is_tracking=True` |
| Frame logging | `main.py` | Log stationary_windows, was_ever_moving, displacement |

**Total lines changed:** ~85 lines across 2 files
**Risk:** Low — changes are additive. No changes to detection pipeline, scoring, or state machine.

### Why Raw Detections, Not Kalman Position

| Signal | Noise floor (1s displacement) | 5cm threshold margin |
|--------|------|------|
| Kalman position | 5.8cm mean | **56% false positive** |
| Raw detection mean | ~1.6cm | **<0.1% false positive** |

The Kalman filter's high process noise (sigma_a=5.0 m/s²) prevents it from averaging down measurement noise for stationarity detection. Raw detection means bypass this entirely.

### Why Never-Move-Only Gating

Was-moving tracks are NEVER dropped by the displacement gate. This eliminates all tactical risks:
- Pins: safe (enemy was moving before pin, track persists)
- Pushing matches: safe (both bots were moving)
- Disabled opponents: safe (they moved before being disabled)
- Spinner spin-up: safe (they drove to position first)

Only tracks that were NEVER observed moving (born stationary = phantom) get the 1-second gate.
