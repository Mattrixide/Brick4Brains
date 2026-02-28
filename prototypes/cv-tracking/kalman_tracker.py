"""Kalman filter tracker for 2D object smoothing and prediction.

Uses OpenCV's cv2.KalmanFilter with a constant-velocity model:
  State:       [x, y, vx, vy]
  Measurement: [x, y]
"""

from collections import deque

import cv2
import numpy as np


class KalmanTracker:
    """Wraps cv2.KalmanFilter for 2D constant-velocity tracking."""

    def __init__(self, process_noise=1e-2, measurement_noise=1e-1, trail_length=60):
        # 4 state variables (x, y, vx, vy), 2 measurements (x, y)
        self.kf = cv2.KalmanFilter(4, 2)

        # State transition: constant velocity model
        # [x]   [1 0 dt 0 ] [x]
        # [y] = [0 1 0  dt] [y]
        # [vx]  [0 0 1  0 ] [vx]
        # [vy]  [0 0 0  1 ] [vy]
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement: we observe x, y directly
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise covariance
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # Error covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self._initialized = False
        self.trail = deque(maxlen=trail_length)
        self.predicted_trail = deque(maxlen=trail_length)

    def predict(self):
        """Predict next state. Returns (x, y) or None if not initialized."""
        if not self._initialized:
            return None
        prediction = self.kf.predict()
        px, py = int(prediction[0, 0]), int(prediction[1, 0])
        self.predicted_trail.append((px, py))
        return (px, py)

    def update(self, x, y):
        """Correct the filter with a measurement. Returns corrected (x, y)."""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self._initialized:
            # Initialize state with first measurement, zero velocity
            self.kf.statePost = np.array([
                [np.float32(x)], [np.float32(y)], [0.0], [0.0]
            ], dtype=np.float32)
            self._initialized = True
            self.trail.append((x, y))
            return (x, y)

        corrected = self.kf.correct(measurement)
        cx, cy = int(corrected[0, 0]), int(corrected[1, 0])
        self.trail.append((cx, cy))
        return (cx, cy)

    @property
    def velocity(self):
        """Return estimated velocity (vx, vy) in pixels/frame."""
        if not self._initialized:
            return (0.0, 0.0)
        state = self.kf.statePost
        return (float(state[2, 0]), float(state[3, 0]))

    def reset(self):
        """Reset the filter state."""
        self._initialized = False
        self.trail.clear()
        self.predicted_trail.clear()
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def draw_trail(self, frame, color=(0, 255, 0), predicted_color=(0, 0, 255)):
        """Draw measured trail and predicted trail on the frame."""
        # Measured trail (green)
        pts = list(self.trail)
        for i in range(1, len(pts)):
            alpha = i / len(pts)  # fade in
            thickness = max(1, int(alpha * 3))
            cv2.line(frame, pts[i - 1], pts[i], color, thickness)

        # Predicted trail (red)
        pts = list(self.predicted_trail)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            thickness = max(1, int(alpha * 2))
            cv2.line(frame, pts[i - 1], pts[i], predicted_color, thickness)

    def draw_prediction(self, frame, measured_pos, predicted_pos):
        """Draw measured vs predicted positions."""
        if measured_pos:
            cv2.circle(frame, measured_pos, 6, (0, 255, 0), 2)  # green = measured
        if predicted_pos:
            cv2.circle(frame, predicted_pos, 6, (0, 0, 255), 2)  # red = predicted
            # Velocity arrow
            vx, vy = self.velocity
            if abs(vx) > 0.5 or abs(vy) > 0.5:
                ex = int(predicted_pos[0] + vx * 5)
                ey = int(predicted_pos[1] + vy * 5)
                cv2.arrowedLine(frame, predicted_pos, (ex, ey),
                                (0, 100, 255), 2, tipLength=0.3)
