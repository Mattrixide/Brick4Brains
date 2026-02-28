"""ArUco marker tracker with real-world position measurement.

Tracks ArUco 4x4_50 marker ID #1 and converts pixel positions to
millimeters using the known marker size (50mm) for scale calibration.
"""

import math
import sys
import time
from dataclasses import dataclass
from threading import Thread, Lock

import cv2
import numpy as np


@dataclass
class TrackResult:
    """Result of tracking a single frame."""
    center_px: tuple          # (x, y) in pixels
    heading_rad: float        # marker heading in radians (0 = right)
    px_per_mm: float          # scale factor from this frame
    corners: np.ndarray       # 4 corner points
    timestamp: float = 0.0


class ArucoTracker:
    """Track ArUco marker ID #1 with real-world distance measurement.

    Uses the known 50mm marker side length to compute a pixel-to-mm
    scale factor each frame. Position tracking is done in pixel space
    and converted to mm on demand.
    """

    def __init__(self, marker_id=1, marker_size_mm=50.0):
        self.marker_id = marker_id
        self.marker_size_mm = marker_size_mm

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        self._detector = cv2.aruco.ArucoDetector(dictionary, params)

    def detect(self, frame):
        """Detect marker and return TrackResult, or None if not found."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None:
            return None

        for i, mid in enumerate(ids.flatten()):
            if mid == self.marker_id:
                pts = corners[i][0]  # shape (4, 2)
                center = pts.mean(axis=0)

                # Heading: angle from corner[0] to corner[1]
                dx = pts[1][0] - pts[0][0]
                dy = pts[1][1] - pts[0][1]
                heading = math.atan2(dy, dx)

                # Scale: average of all 4 side lengths
                side_lengths = [
                    np.linalg.norm(pts[(j + 1) % 4] - pts[j])
                    for j in range(4)
                ]
                avg_side_px = sum(side_lengths) / 4.0
                px_per_mm = avg_side_px / self.marker_size_mm

                return TrackResult(
                    center_px=(float(center[0]), float(center[1])),
                    heading_rad=heading,
                    px_per_mm=px_per_mm,
                    corners=pts,
                    timestamp=time.perf_counter(),
                )
        return None

    def draw(self, frame, result):
        """Draw marker overlay on frame."""
        if result is None:
            return

        cx, cy = int(result.center_px[0]), int(result.center_px[1])
        pts = result.corners.astype(np.int32)

        # Draw marker outline
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # Center dot
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Heading arrow
        arrow_len = 40
        ex = int(cx + arrow_len * math.cos(result.heading_rad))
        ey = int(cy + arrow_len * math.sin(result.heading_rad))
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.3)

        # Label
        cv2.putText(frame, f"ID#{self.marker_id}", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    @staticmethod
    def distance_mm(pos_a_px, pos_b_px, px_per_mm):
        """Compute distance in mm between two pixel positions."""
        dx = pos_a_px[0] - pos_b_px[0]
        dy = pos_a_px[1] - pos_b_px[1]
        dist_px = math.hypot(dx, dy)
        return dist_px / px_per_mm

    @staticmethod
    def angle_diff(target, current):
        """Shortest signed angle from current to target (radians)."""
        diff = (target - current + math.pi) % (2 * math.pi) - math.pi
        return diff


class ThreadedCamera:
    """Threaded webcam capture (simplified from prototype/capture.py)."""

    def __init__(self, src=1, width=640, height=480):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.stream = cv2.VideoCapture(src, backend)
        if not self.stream.isOpened():
            raise RuntimeError(f"Cannot open camera {src}")

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.stream.read()
        if not self.grabbed:
            raise RuntimeError("Cannot read from camera")

        self.stopped = False
        self._lock = Lock()

    def start(self):
        Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self._lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()
