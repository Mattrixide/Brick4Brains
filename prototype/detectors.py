"""Detection modules: ArUco markers, HSV color tracking, MOG2 background subtraction.

All detectors share a common interface: detect(frame) -> list[Detection].
"""

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Detection:
    """A single detected object in a frame."""
    label: str              # "aruco_<id>", "color", "motion"
    center_px: tuple        # (x, y) in pixels
    bbox: tuple             # (x, y, w, h)
    heading_rad: float = 0.0
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.perf_counter)


class ArUcoDetector:
    """Detect ArUco 4x4_50 markers with heading estimation."""

    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        # Speed optimizations
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        self._detector = cv2.aruco.ArucoDetector(dictionary, params)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        detections = []
        if ids is None:
            return detections

        for i, marker_id in enumerate(ids.flatten()):
            pts = corners[i][0]  # 4 corner points
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            # Heading: angle from top-left to top-right corner
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
            heading = math.atan2(dy, dx)
            # Bounding box
            x_min, y_min = pts.min(axis=0).astype(int)
            x_max, y_max = pts.max(axis=0).astype(int)
            detections.append(Detection(
                label=f"aruco_{marker_id}",
                center_px=(cx, cy),
                bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                heading_rad=heading,
            ))
        return detections

    def draw(self, frame, detections, corners_raw=None):
        """Draw marker outlines, IDs, and heading arrows."""
        for det in detections:
            cx, cy = det.center_px
            bx, by, bw, bh = det.bbox
            # Bounding box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 200, 0), 2)
            # Label
            cv2.putText(frame, det.label, (bx, by - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            # Heading arrow
            arrow_len = max(bw, bh) * 0.7
            ex = int(cx + arrow_len * math.cos(det.heading_rad))
            ey = int(cy + arrow_len * math.sin(det.heading_rad))
            cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.3)
            # Center dot
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    def detect_and_draw(self, frame):
        """Detect markers and draw overlays. Returns detections."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        detections = []
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                pts = corners[i][0]
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                dx = pts[1][0] - pts[0][0]
                dy = pts[1][1] - pts[0][1]
                heading = math.atan2(dy, dx)
                x_min, y_min = pts.min(axis=0).astype(int)
                x_max, y_max = pts.max(axis=0).astype(int)
                det = Detection(
                    label=f"aruco_{marker_id}",
                    center_px=(cx, cy),
                    bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                    heading_rad=heading,
                )
                detections.append(det)
                # Heading arrow
                arrow_len = max(det.bbox[2], det.bbox[3]) * 0.7
                ex = int(cx + arrow_len * math.cos(heading))
                ey = int(cy + arrow_len * math.sin(heading))
                cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.3)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        return detections


class ColorDetector:
    """Detect objects by HSV color range."""

    # Default: bright green
    DEFAULT_LOWER = np.array([35, 80, 80])
    DEFAULT_UPPER = np.array([85, 255, 255])

    def __init__(self, lower_hsv=None, upper_hsv=None, min_area=500):
        self.lower = lower_hsv if lower_hsv is not None else self.DEFAULT_LOWER.copy()
        self.upper = upper_hsv if upper_hsv is not None else self.DEFAULT_UPPER.copy()
        self.min_area = min_area

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            detections.append(Detection(
                label="color",
                center_px=(cx, cy),
                bbox=(x, y, w, h),
                confidence=min(area / 5000.0, 1.0),
            ))
        return detections

    def draw(self, frame, detections):
        for det in detections:
            x, y, w, h = det.bbox
            cx, cy = det.center_px
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"color ({det.confidence:.1f})", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


class BackgroundSubDetector:
    """Detect moving objects via MOG2 background subtraction."""

    def __init__(self, history=300, var_threshold=40, min_area=800):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )
        self.min_area = min_area
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame):
        mask = self.bg_sub.apply(frame)
        # Morphological cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            detections.append(Detection(
                label="motion",
                center_px=(cx, cy),
                bbox=(x, y, w, h),
                confidence=min(area / 10000.0, 1.0),
            ))
        return detections

    def draw(self, frame, detections):
        for det in detections:
            x, y, w, h = det.bbox
            cx, cy = det.center_px
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, "motion", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
