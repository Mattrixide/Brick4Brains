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

class TelemetryReceiver:
    """Receives IMU telemetry from ESP32 on UDP port 4211.

    Packet format (20 bytes, little-endian):
        float32 heading     (degrees)
        float32 gyro_z      (dps)
        float32 accel_x     (mg)
        float32 accel_y     (mg)
        uint32  timestamp   (millis on ESP32)
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

            with self._lock:
                self.heading = heading
                self.gyro_z = gyro_z
                self.accel_x = accel_x
                self.accel_y = accel_y
                self.esp_timestamp = timestamp
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
    """Fuses CV heading (absolute, slow) with gyro heading (fast, drifts).

    When CV is available:  CV is ground truth, gyro fills gaps between frames.
    When CV is lost:       Gyro-only integration.
    When CV re-acquired:   Instant snap to CV heading (no gradual blend).
    """

    def __init__(self):
        self.heading_deg = 0.0         # fused heading output
        self._gyro_heading_deg = 0.0   # accumulated gyro heading
        self._cv_heading_deg = None    # last CV heading
        self._frames_without_cv = 0
        self._has_cv = False
        self._last_gyro_time = 0.0

    def update_gyro(self, gyro_z_dps: float, dt: float):
        """Called at IMU telemetry rate (~100 Hz from ESP32).

        Integrates gyro into heading between CV frames.
        """
        self._gyro_heading_deg += gyro_z_dps * dt
        self.heading_deg = self._gyro_heading_deg

    def update_cv(self, cv_heading_rad: float):
        """Called when ArUco is detected (~30 Hz).

        Instant snap: reset gyro heading to match CV.
        """
        cv_deg = math.degrees(cv_heading_rad)
        self._cv_heading_deg = cv_deg
        self._has_cv = True

        # INSTANT SNAP — trust CV absolutely
        self._gyro_heading_deg = cv_deg
        self.heading_deg = cv_deg
        self._frames_without_cv = 0

    def update_no_cv(self):
        """Called when ArUco detection fails this frame."""
        self._frames_without_cv += 1
        self._has_cv = False
        # heading continues from gyro integration

    @property
    def heading_rad(self) -> float:
        """Fused heading in radians."""
        return math.radians(self.heading_deg)

    @property
    def is_cv_tracking(self) -> bool:
        """True if CV heading was available recently."""
        return self._frames_without_cv < 5

    @property
    def frames_without_cv(self) -> int:
        return self._frames_without_cv

    @property
    def confidence(self) -> float:
        """1.0 when CV is fresh, decays with gyro-only time."""
        if self._frames_without_cv == 0:
            return 1.0
        return max(0.0, 1.0 - self._frames_without_cv * 0.01)
