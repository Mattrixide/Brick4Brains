"""Mode-switched heading fusion: CV ground truth + gyro interpolation.

Three modes:
  1. ArUco visible:      CV heading is ground truth, gyro interpolates between frames
  2. ArUco lost:         Gyro-only heading (drifts ~0.05-0.2 deg/min)
  3. ArUco re-acquired:  Instant snap-correction to CV heading

Also receives IMU telemetry from ESP32 via UDP port 4211.
"""

import math
import struct
import socket
import threading
import time


# ---------------------------------------------------------------------------
# Telemetry receiver (background thread)
# ---------------------------------------------------------------------------

class IMUPoller:
    """Polls ESP32 /api/imu via HTTP in a background thread (~20 Hz)."""

    def __init__(self, host="192.168.4.113"):
        self._url = f"http://{host}/api/imu"
        self._thread = None
        self._running = False
        self._lock = threading.Lock()

        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.ready = False
        self.last_recv_time = 0.0
        self.polls = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"[imu-poll] Polling {self._url}")

    def _poll_loop(self):
        import requests
        while self._running:
            try:
                r = requests.get(self._url, timeout=0.3)
                data = r.json()
                with self._lock:
                    self.yaw = data.get("yaw", 0.0)
                    self.pitch = data.get("pitch", 0.0)
                    self.roll = data.get("roll", 0.0)
                    self.ready = data.get("ready", False)
                    self.last_recv_time = time.monotonic()
                    self.polls += 1
            except Exception:
                pass
            time.sleep(0.04)  # ~25 Hz

    @property
    def is_active(self):
        with self._lock:
            if self.last_recv_time == 0:
                return False
            return (time.monotonic() - self.last_recv_time) < 1.0

    def get_yaw(self):
        with self._lock:
            return self.yaw

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def reset_yaw(self):
        """Send reset yaw command to ESP32."""
        import requests
        try:
            requests.post(self._url, data={"resetYaw": "1"}, timeout=0.5)
        except Exception:
            pass


class TelemetryReceiver:
    """Receives IMU telemetry from ESP32 on UDP port 4211.

    Packet format (32 bytes, little-endian):
        float32 heading           (degrees)
        float32 gyro_z            (dps)
        float32 accel_x           (mg)
        float32 accel_y           (mg)
        uint32  timestamp         (millis on ESP32)
        uint16  imu_check_fails   (consecutive checkStatus failures)
        uint16  imu_dt_skips      (consecutive dt > 10ms skips)
        uint32  imu_fails_total   (lifetime checkStatus failures)
        uint32  reserved
    Also accepts legacy 20-byte packets (no IMU health fields).
    """

    def __init__(self, port=4211):
        self._port = port
        self._sock = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()

        # Latest telemetry values
        self.heading = 0.0
        self.gyro_z = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.esp_timestamp = 0
        self.last_recv_time = 0.0
        self.packets_received = 0

        # IMU health debug counters
        self.imu_check_fails = 0
        self.imu_dt_skips = 0
        self.imu_fails_total = 0

    def start(self):
        """Start background receiver thread."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", self._port))
        self._sock.settimeout(0.1)

        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[telemetry] Listening on UDP port {self._port}")

    def _recv_loop(self):
        while self._running:
            try:
                data, addr = self._sock.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < 20:
                continue

            heading, gyro_z, accel_x, accel_y, timestamp = struct.unpack(
                "<ffffI", data[:20]
            )

            # Parse IMU health fields from expanded 32-byte packet
            imu_fails = 0
            imu_dt = 0
            imu_fails_total = 0
            if len(data) >= 32:
                imu_fails, imu_dt, imu_fails_total = struct.unpack(
                    "<HHI", data[20:28]
                )

            with self._lock:
                self.heading = heading
                self.gyro_z = gyro_z
                self.accel_x = accel_x
                self.accel_y = accel_y
                self.esp_timestamp = timestamp
                self.imu_check_fails = imu_fails
                self.imu_dt_skips = imu_dt
                self.imu_fails_total = imu_fails_total
                self.last_recv_time = time.monotonic()
                self.packets_received += 1

    def get(self):
        """Return latest telemetry as a dict (thread-safe)."""
        with self._lock:
            return {
                "heading": self.heading,
                "gyro_z": self.gyro_z,
                "accel_x": self.accel_x,
                "accel_y": self.accel_y,
                "esp_timestamp": self.esp_timestamp,
                "age_ms": (time.monotonic() - self.last_recv_time) * 1000
                          if self.last_recv_time > 0 else float("inf"),
                "imu_check_fails": self.imu_check_fails,
                "imu_dt_skips": self.imu_dt_skips,
                "imu_fails_total": self.imu_fails_total,
            }

    @property
    def is_active(self):
        """True if we've received telemetry in the last 500ms."""
        with self._lock:
            if self.last_recv_time == 0:
                return False
            return (time.monotonic() - self.last_recv_time) < 0.5

    def stop(self):
        self._running = False
        if self._sock:
            self._sock.close()
        if self._thread:
            self._thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Mode-switched heading fusion
# ---------------------------------------------------------------------------

class HeadingFusion:
    """Fuses CV heading (absolute, slow) with IMU yaw (fast, drifts).

    Uses IMU yaw from ESP32 /api/imu with offset calibration against ArUco.
    When CV is available:  Calibrate offset, output = IMU yaw - offset
    When CV is lost:       IMU-only heading (drifts slowly but stable at speed)
    When CV re-acquired:   Instant recalibrate offset
    """

    def __init__(self):
        self.heading_deg = 0.0         # fused heading output (degrees)
        self._imu_yaw_deg = 0.0        # raw IMU yaw from ESP32
        self._offset_deg = 0.0         # IMU_yaw - ArUco_heading offset
        self._offset_calibrated = False
        self._cv_heading_deg = None
        self._frames_without_cv = 0
        self._has_cv = False

    def update_imu(self, imu_yaw_deg: float):
        """Called with raw IMU yaw from ESP32 (~20 Hz via HTTP polling).

        If offset is calibrated, updates fused heading.
        """
        self._imu_yaw_deg = imu_yaw_deg
        if self._offset_calibrated and self._frames_without_cv > 0:
            self.heading_deg = imu_yaw_deg - self._offset_deg

    def update_gyro(self, gyro_z_dps: float, dt: float):
        """Called at IMU telemetry rate (UDP, if available).

        Fallback if HTTP polling isn't fast enough.
        """
        self._imu_yaw_deg += gyro_z_dps * dt
        if self._offset_calibrated and self._frames_without_cv > 0:
            self.heading_deg = self._imu_yaw_deg - self._offset_deg

    def update_cv(self, cv_heading_rad: float):
        """Called when ArUco is detected (~30 Hz).

        Calibrates the offset between IMU yaw and ArUco heading.
        """
        cv_deg = math.degrees(cv_heading_rad)
        self._cv_heading_deg = cv_deg
        self._has_cv = True
        self._frames_without_cv = 0

        # Calibrate offset: offset = IMU_yaw - ArUco_heading
        self._offset_deg = self._imu_yaw_deg - cv_deg
        self._offset_calibrated = True
        # Smooth blend instead of hard snap — prevents oscillation during turns
        diff = cv_deg - self.heading_deg
        diff = (diff + 180) % 360 - 180  # wrap to [-180, 180]
        self.heading_deg = self.heading_deg + diff * 0.7

    def update_no_cv(self):
        """Called when ArUco detection fails this frame."""
        self._frames_without_cv += 1
        self._has_cv = False
        # heading continues from IMU with last calibrated offset

    @property
    def heading_rad(self) -> float:
        """Fused heading in radians, normalized to [-pi, pi]."""
        rad = math.radians(self.heading_deg)
        return math.atan2(math.sin(rad), math.cos(rad))

    @property
    def is_cv_tracking(self) -> bool:
        """True if CV heading was available recently."""
        return self._frames_without_cv < 5

    @property
    def is_calibrated(self) -> bool:
        return self._offset_calibrated

    @property
    def frames_without_cv(self) -> int:
        return self._frames_without_cv

    @property
    def confidence(self) -> float:
        """1.0 when CV is fresh, decays with gyro-only time."""
        if not self._offset_calibrated:
            return 0.0
        if self._frames_without_cv == 0:
            return 1.0
        return max(0.0, 1.0 - self._frames_without_cv * 0.01)


# ---------------------------------------------------------------------------
# Robot position Kalman filter (bridges ArUco dropouts)
# ---------------------------------------------------------------------------

import numpy as np


class RobotPositionKF:
    """4-state Kalman filter [x, y, vx, vy] for our robot position (cm units).

    Constant-velocity process model (no IMU accel — vibration causes divergence).
    ArUco position as measurement. Bridges 50%+ dropout gracefully.
    """

    def __init__(self, dt=1/60, sigma_a_cm=500.0, sigma_meas_cm=5.0,
                 arena_half_cm=122.0, max_coast=90):
        self.dt = dt
        self.x = np.zeros(4, dtype=np.float64)       # [x, y, vx, vy] in cm
        self.P = np.eye(4, dtype=np.float64) * 1000.0  # high initial uncertainty

        # State transition (constant velocity)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement: observe position only
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise (discrete white noise acceleration, cm units)
        q = sigma_a_cm ** 2
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        self.Q_baseline = np.array([
            [dt4/4, 0,     dt3/2, 0    ],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0    ],
            [0,     dt3/2, 0,     dt2  ],
        ], dtype=np.float64) * q
        self.Q = self.Q_baseline.copy()

        # Reduced Q for stationary hint (pin state)
        self.Q_stationary = self.Q_baseline * 0.1

        # Measurement noise (cm)
        self.R = np.eye(2, dtype=np.float64) * sigma_meas_cm**2

        self.arena_half = arena_half_cm
        self.max_coast = max_coast
        self.frames_without_measurement = 0
        self._initialized = False
        self._stationary = False

        # Innovation gate threshold (chi-squared, 2 DOF, 99%)
        self._gate_threshold = 9.21
        self._reinit_coast_threshold = 5  # reinit if coasting this many frames

    def predict(self):
        """Predict step — call every frame."""
        if not self._initialized:
            return

        # Velocity decay during coast — fast decay so wall-stuck detector works
        # 0.70: reaches 5 cm/s from 120 in ~15 frames (0.25s)
        # 0.90 was too slow: took 44 frames (0.7s)
        if self.frames_without_measurement > 0:
            decay = 0.70
            self.x[2] *= decay
            self.x[3] *= decay

        # Stationary hint: zero velocity
        if self._stationary:
            self.x[2] = 0.0
            self.x[3] = 0.0

        self.x = self.F @ self.x

        # Arena bounds clamping — use generous bounds (arena_half + 30cm margin)
        # The robot CAN be at -142 if pressed against a wall at -122.
        # Tight clamping at -122 creates phantom velocity (clamp to -122, then
        # update pulls back to -142 = 20cm jump every frame = 48 cm/s phantom).
        clamp_limit = self.arena_half + 30.0
        for i, vi in [(0, 2), (1, 3)]:
            if abs(self.x[i]) > clamp_limit:
                self.x[i] = np.clip(self.x[i], -clamp_limit, clamp_limit)
                self.x[vi] = 0.0

        Q = self.Q_stationary if self._stationary else self.Q
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, x_cm, y_cm):
        """Update with ArUco measurement. Call when detected."""
        z = np.array([x_cm, y_cm], dtype=np.float64)

        if not self._initialized:
            self.x[:2] = z
            self.x[2:] = 0.0
            self._initialized = True
            self.frames_without_measurement = 0
            return

        # Innovation (residual)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R

        # Innovation gate — reject outliers
        try:
            S_inv = np.linalg.inv(S)
            mahal_dist = float(y.T @ S_inv @ y)
        except np.linalg.LinAlgError:
            # Singular S matrix — skip this measurement entirely
            self.frames_without_measurement += 1
            return

        if mahal_dist > self._gate_threshold:
            # Measurement rejected as outlier
            if self.frames_without_measurement >= self._reinit_coast_threshold:
                # Been coasting too long — snap to measurement (reinitialize)
                self.x[:2] = z
                self.x[2:] = 0.0
                self.P = np.eye(4, dtype=np.float64) * 100.0
                self.frames_without_measurement = 0
                return
            else:
                # Short coast — discard this measurement
                self.frames_without_measurement += 1
                return

        # Standard Kalman update
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        # Velocity clamp (physical max for beetleweight)
        MAX_VEL = 120.0  # cm/s
        speed = math.sqrt(self.x[2]**2 + self.x[3]**2)
        if speed > MAX_VEL:
            scale = MAX_VEL / speed
            self.x[2] *= scale
            self.x[3] *= scale

        self.frames_without_measurement = 0

    def update_no_measurement(self):
        """Call when ArUco is not detected this frame."""
        self.frames_without_measurement += 1

    def set_stationary_hint(self, stationary: bool):
        """Call with True during pin state, False otherwise. Idempotent."""
        self._stationary = stationary

    @property
    def position_cm(self):
        """Estimated (x, y) in cm. Always valid after first detection."""
        if not self._initialized:
            return (0.0, 0.0)
        return (float(self.x[0]), float(self.x[1]))

    @property
    def velocity_cm_s(self):
        """Estimated (vx, vy) in cm/s."""
        if not self._initialized:
            return (0.0, 0.0)
        return (float(self.x[2]), float(self.x[3]))

    @property
    def speed_cm_s(self):
        return math.sqrt(self.x[2]**2 + self.x[3]**2) if self._initialized else 0.0

    @property
    def is_tracking(self):
        return self._initialized and self.frames_without_measurement < self.max_coast

    @property
    def confidence(self):
        """1.0 when fresh, decays with coast frames."""
        if not self._initialized:
            return 0.0
        if self.frames_without_measurement == 0:
            return 1.0
        return max(0.0, 1.0 - self.frames_without_measurement / self.max_coast)

    def reset(self):
        self._initialized = False
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1000.0
        self.frames_without_measurement = 0
        self._stationary = False
