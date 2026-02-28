"""Threaded webcam capture for high-FPS frame acquisition.

Moves camera I/O to a background thread so the main thread never blocks
waiting for the camera hardware. Based on the PyImageSearch threaded
capture pattern and allskyee's gist.

Includes manual exposure control for motion blur reduction and optional
frame sharpening post-processing.
"""

import time
import sys
from threading import Thread, Lock

import cv2
import numpy as np

RESOLUTIONS = [
    (640, 480),    # 480p
    (1280, 720),   # 720p
    (1920, 1080),  # 1080p
]

# Windows exposure values are log2: EXP_TIME = 2^(-value) seconds.
# value  -5 => ~31ms,  -7 => ~7.8ms,  -9 => ~1.95ms,
# value -10 => ~977us, -11 => ~488us, -12 => ~244us, -13 => ~122us
EXPOSURE_MIN = -13
EXPOSURE_MAX = -1
EXPOSURE_DEFAULT = -7  # ~7.8ms, reasonable starting point

GAIN_MIN = 0
GAIN_MAX = 255
GAIN_DEFAULT = 0


def exposure_to_us(exp_val):
    """Convert Windows log2 exposure index to approximate microseconds.

    Windows uses EXP_TIME = 2^(exp_val) seconds, where exp_val is negative.
    e.g. exp_val=-7 => 2^(-7) = 0.0078s = 7812us
    """
    return 2 ** exp_val * 1_000_000


class ThreadedCamera:
    """Threaded webcam capture with FPS counting, resolution, and exposure control."""

    def __init__(self, src=0, resolution_index=0):
        self._src = src
        self._backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.stream = cv2.VideoCapture(src, self._backend)
        if not self.stream.isOpened():
            raise RuntimeError(f"Cannot open camera {src}")

        self._resolution_index = resolution_index
        self._apply_resolution()

        # Try to minimize internal buffer to get latest frames
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Exposure state
        self._exposure = EXPOSURE_DEFAULT
        self._auto_exposure = True
        self._gain = GAIN_DEFAULT
        self._sharpen_enabled = False
        # The unsharp-mask kernel: identity + Laplacian sharpening
        self._sharpen_strength = 1.0  # 0.5 = mild, 1.0 = moderate, 2.0 = strong

        self.grabbed, self.frame = self.stream.read()
        if not self.grabbed:
            raise RuntimeError("Cannot read from camera")

        self.stopped = False
        self.lock = Lock()

        # FPS tracking for capture thread
        self._capture_fps = 0.0
        self._capture_frame_count = 0
        self._capture_fps_time = time.perf_counter()

    def _apply_resolution(self):
        w, h = RESOLUTIONS[self._resolution_index]
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    @property
    def resolution(self):
        w = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    @property
    def capture_fps(self):
        return self._capture_fps

    @property
    def exposure(self):
        return self._exposure

    @property
    def auto_exposure(self):
        return self._auto_exposure

    @property
    def gain(self):
        return self._gain

    @property
    def sharpen_enabled(self):
        return self._sharpen_enabled

    @property
    def exposure_us(self):
        """Current exposure time in microseconds (approximate)."""
        return exposure_to_us(self._exposure)

    def set_auto_exposure(self, enabled):
        """Enable or disable auto-exposure."""
        self._auto_exposure = enabled
        if enabled:
            # 0.75 = auto exposure mode (DirectShow / MSMF)
            self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        else:
            # 0.25 = manual exposure mode
            self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self._apply_exposure()
        return enabled

    def set_exposure(self, value):
        """Set manual exposure (negative int, -13=shortest to -1=longest).

        Automatically disables auto-exposure.
        """
        value = max(EXPOSURE_MIN, min(EXPOSURE_MAX, int(value)))
        self._exposure = value
        if self._auto_exposure:
            self.set_auto_exposure(False)
        else:
            self._apply_exposure()
        return value

    def adjust_exposure(self, delta):
        """Adjust exposure by delta steps. Negative = shorter (less blur)."""
        return self.set_exposure(self._exposure + delta)

    def _apply_exposure(self):
        """Write the current exposure value to the camera."""
        self.stream.set(cv2.CAP_PROP_EXPOSURE, self._exposure)

    def set_gain(self, value):
        """Set camera gain (0-255). Higher = brighter but noisier."""
        value = max(GAIN_MIN, min(GAIN_MAX, int(value)))
        self._gain = value
        self.stream.set(cv2.CAP_PROP_GAIN, value)
        return value

    def adjust_gain(self, delta):
        """Adjust gain by delta steps."""
        return self.set_gain(self._gain + delta)

    def set_sharpen(self, enabled):
        """Toggle frame sharpening post-processing."""
        self._sharpen_enabled = enabled
        return enabled

    def toggle_sharpen(self):
        """Toggle frame sharpening on/off."""
        self._sharpen_enabled = not self._sharpen_enabled
        return self._sharpen_enabled

    @property
    def camera_index(self):
        return self._src

    def cycle_resolution(self):
        """Cycle to the next resolution. Returns the new (w, h)."""
        self._resolution_index = (self._resolution_index + 1) % len(RESOLUTIONS)
        self._apply_resolution()
        return self.resolution

    def switch_camera(self, new_src):
        """Switch to a different camera index. Returns True on success."""
        self.stopped = True
        time.sleep(0.1)  # let capture thread exit
        self.stream.release()

        self._src = new_src
        self.stream = cv2.VideoCapture(new_src, self._backend)
        if not self.stream.isOpened():
            # Fall back to previous
            self._src = 0
            self.stream = cv2.VideoCapture(0, self._backend)
            self._apply_resolution()
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.stopped = False
            Thread(target=self._update, daemon=True).start()
            return False

        self._apply_resolution()
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Re-apply exposure settings
        if not self._auto_exposure:
            self.set_auto_exposure(False)

        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        Thread(target=self._update, daemon=True).start()
        return True

    def start(self):
        Thread(target=self._update, daemon=True).start()
        return self

    def _sharpen_frame(self, frame):
        """Apply unsharp mask sharpening to reduce perceived motion blur.

        Unsharp mask: sharpened = original + strength * (original - blurred)
        """
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        sharpened = cv2.addWeighted(
            frame, 1.0 + self._sharpen_strength,
            blurred, -self._sharpen_strength,
            0,
        )
        return sharpened

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()

            # Apply sharpening in capture thread to keep main thread free
            if grabbed and frame is not None and self._sharpen_enabled:
                frame = self._sharpen_frame(frame)

            with self.lock:
                self.grabbed, self.frame = grabbed, frame

            # Update capture FPS counter
            self._capture_frame_count += 1
            now = time.perf_counter()
            elapsed = now - self._capture_fps_time
            if elapsed >= 0.5:
                self._capture_fps = self._capture_frame_count / elapsed
                self._capture_frame_count = 0
                self._capture_fps_time = now

    def read(self):
        """Return the latest frame (copied to avoid race conditions)."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()
