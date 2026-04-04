"""Threaded webcam capture for high-FPS frame acquisition.

Moves camera I/O to a background thread so the main thread never blocks
waiting for the camera hardware. Based on the PyImageSearch threaded
capture pattern and allskyee's gist.

Includes manual exposure control for motion blur reduction and optional
frame sharpening post-processing.
"""

import math
import time
import sys
from datetime import datetime
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

# OAK-D exposure/gain constants (microseconds + ISO, not Windows log2 scale)
OAKD_EXPOSURE_MIN = 100        # 100us
OAKD_EXPOSURE_MAX = 33000      # 33ms
OAKD_EXPOSURE_DEFAULT = 8000   # 8ms, reasonable starting point
OAKD_EXPOSURE_STEP = 500       # 500us per adjust step

OAKD_GAIN_MIN = 100            # ISO 100
OAKD_GAIN_MAX = 1600           # ISO 1600
OAKD_GAIN_DEFAULT = 100        # ISO 100
OAKD_GAIN_STEP = 100           # ISO per adjust step


def exposure_to_us(exp_val):
    """Convert Windows log2 exposure index to approximate microseconds.

    Windows uses EXP_TIME = 2^(exp_val) seconds, where exp_val is negative.
    e.g. exp_val=-7 => 2^(-7) = 0.0078s = 7812us
    """
    return 2 ** exp_val * 1_000_000


class ThreadedCamera:
    """Threaded webcam capture with FPS counting, resolution, and exposure control."""

    def __init__(self, src=0, resolution_index=1, backend=None):
        self._src = src
        if backend is not None:
            self._backend = backend
        elif sys.platform == "win32":
            # MSMF gets better FPS than DSHOW on many cameras (e.g. Lumina 4K)
            self._backend = cv2.CAP_MSMF
        else:
            self._backend = cv2.CAP_ANY
        self.stream = cv2.VideoCapture(src, self._backend)
        if not self.stream.isOpened():
            raise RuntimeError(f"Cannot open camera {src}")

        # Request 30 FPS and MJPG codec for bandwidth efficiency
        self.stream.set(cv2.CAP_PROP_FPS, 30.0)
        self.stream.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

        self._resolution_index = resolution_index
        self._apply_resolution()

        # Minimize internal buffer to get latest frames
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Disable autofocus for fixed overhead mount (prevents focus hunting)
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)

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

        # Enable auto-exposure after first read so camera has initialized fully.
        # DirectShow cameras may use 0.75 or 3 for auto mode — try both.
        self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

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
        prev_sum = None
        while not self.stopped:
            grabbed, frame = self.stream.read()

            # Apply sharpening in capture thread to keep main thread free
            if grabbed and frame is not None and self._sharpen_enabled:
                frame = self._sharpen_frame(frame)

            with self.lock:
                self.grabbed, self.frame = grabbed, frame

            # Only count unique frames (not duplicate buffered reads)
            is_new = True
            if grabbed and frame is not None:
                frame_sum = int(cv2.sumElems(frame[:8, :8, 0])[0])
                if frame_sum == prev_sum:
                    is_new = False
                prev_sum = frame_sum

            if is_new:
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

    def get_intrinsics(self):
        """Estimate camera intrinsics from resolution and assumed FOV.

        Returns (camera_matrix, dist_coeffs) as numpy arrays.
        These are rough estimates -- for precise calibration use OAK-D Pro.
        """
        w, h = self.resolution
        # Assume ~60 degree horizontal FOV (typical for webcams)
        hfov_rad = math.radians(60)
        fx = w / (2 * math.tan(hfov_rad / 2))
        fy = fx  # assume square pixels
        cx, cy = w / 2.0, h / 2.0
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros(5, dtype=np.float64)
        return camera_matrix, dist_coeffs

    def get_capabilities(self):
        """Query all capabilities of the current camera.

        Briefly pauses the capture thread to test supported resolutions,
        then resumes. Returns a dict with device info, supported resolutions,
        and all queryable OpenCV properties.
        """
        # Pause capture thread
        self.stopped = True
        time.sleep(0.15)

        caps = {
            'device_index': self._src,
            'backend': self.stream.getBackendName(),
            'timestamp': datetime.now().isoformat(),
            'current_resolution': list(self.resolution),
        }

        # Test common resolutions
        test_resolutions = [
            (320, 240), (640, 360), (640, 480),
            (800, 600), (960, 540), (1280, 720),
            (1920, 1080), (2560, 1440), (3840, 2160),
        ]
        # Release current stream so we can reopen at each resolution
        self.stream.release()

        def _measure_fps(src, backend, w, h, fourcc=None):
            """Open a fresh capture at the given resolution and measure actual FPS."""
            cap = cv2.VideoCapture(src, backend)
            if not cap.isOpened():
                cap.release()
                return None
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            # Use short exposure so it doesn't bottleneck FPS
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, -11)  # ~488us
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Read actual fourcc to see if codec was accepted
            actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            actual_fourcc = ''.join(
                chr((actual_fourcc_int >> 8 * i) & 0xFF) for i in range(4)
            ) if actual_fourcc_int > 0 else 'unknown'
            # Warm up
            for _ in range(3):
                cap.read()
            # Measure
            n_frames = 15
            t0 = time.perf_counter()
            for _ in range(n_frames):
                cap.read()
            elapsed = time.perf_counter() - t0
            fps = n_frames / elapsed if elapsed > 0 else 0
            reported_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return {
                'width': actual_w,
                'height': actual_h,
                'measured_fps': round(fps, 1),
                'reported_fps': round(reported_fps, 1) if reported_fps > 0 else None,
                'fourcc': actual_fourcc,
                'requested_fourcc': fourcc,
            }

        seen = set()
        supported = []
        supported_codecs = set()
        codecs_to_test = ['YUY2', 'MJPG']
        for w, h in test_resolutions:
            for codec in codecs_to_test:
                result = _measure_fps(self._src, self._backend, w, h, codec)
                if result is None:
                    continue
                # Skip if camera didn't actually switch to requested codec
                if codec and result['fourcc'] != codec:
                    continue
                key = (result['width'], result['height'], result['fourcc'])
                if key in seen:
                    continue
                seen.add(key)
                supported_codecs.add(result['fourcc'])
                supported.append({
                    'width': result['width'],
                    'height': result['height'],
                    'fps': result['measured_fps'],
                    'reported_fps': result['reported_fps'],
                    'fourcc': result['fourcc'],
                    'requested': f"{w}x{h}" if (result['width'], result['height']) != (w, h) else None,
                })
        caps['supported_resolutions'] = supported
        caps['supported_codecs'] = sorted(supported_codecs)

        # Reopen stream at original resolution
        orig_w, orig_h = caps['current_resolution']
        self.stream = cv2.VideoCapture(self._src, self._backend)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, orig_w)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, orig_h)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self._auto_exposure:
            self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        else:
            self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.stream.set(cv2.CAP_PROP_EXPOSURE, self._exposure)
        if self._gain != GAIN_DEFAULT:
            self.stream.set(cv2.CAP_PROP_GAIN, self._gain)

        # Query all properties
        prop_list = [
            ('fps', cv2.CAP_PROP_FPS),
            ('brightness', cv2.CAP_PROP_BRIGHTNESS),
            ('contrast', cv2.CAP_PROP_CONTRAST),
            ('saturation', cv2.CAP_PROP_SATURATION),
            ('hue', cv2.CAP_PROP_HUE),
            ('exposure', cv2.CAP_PROP_EXPOSURE),
            ('gain', cv2.CAP_PROP_GAIN),
            ('auto_exposure', cv2.CAP_PROP_AUTO_EXPOSURE),
            ('white_balance', cv2.CAP_PROP_WB_TEMPERATURE),
            ('auto_white_balance', cv2.CAP_PROP_AUTO_WB),
            ('focus', cv2.CAP_PROP_FOCUS),
            ('autofocus', cv2.CAP_PROP_AUTOFOCUS),
            ('zoom', cv2.CAP_PROP_ZOOM),
            ('sharpness', cv2.CAP_PROP_SHARPNESS),
            ('gamma', cv2.CAP_PROP_GAMMA),
            ('backlight', cv2.CAP_PROP_BACKLIGHT),
            ('buffer_size', cv2.CAP_PROP_BUFFERSIZE),
        ]
        properties = {}
        for name, prop_id in prop_list:
            val = self.stream.get(prop_id)
            properties[name] = val

        # FourCC codec
        fourcc_int = int(self.stream.get(cv2.CAP_PROP_FOURCC))
        if fourcc_int > 0:
            properties['fourcc'] = ''.join(
                chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)
            )
        else:
            properties['fourcc'] = None

        caps['properties'] = properties

        # Resume capture thread
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        Thread(target=self._update, daemon=True).start()

        return caps

    def stop(self):
        self.stopped = True
        self.stream.release()


class DepthAICamera:
    """OAK-D camera capture using DepthAI v3 SDK.

    Same interface as ThreadedCamera for drop-in use via create_camera().
    OAK-D cameras are not UVC devices — they require the DepthAI pipeline.
    """

    def __init__(self, resolution_index=1):
        import depthai as dai
        self._dai = dai

        self._resolution_index = resolution_index
        self._exposure_val = OAKD_EXPOSURE_DEFAULT
        self._auto_exposure = True
        self._gain_val = OAKD_GAIN_DEFAULT
        self._sharpen_enabled = False
        self._sharpen_strength = 1.0

        self.stopped = False
        self.lock = Lock()
        self.frame = None
        self.grabbed = False

        self._capture_fps = 0.0
        self._capture_frame_count = 0
        self._capture_fps_time = time.perf_counter()

        self._device = None
        self._pipeline = None
        self._cam_node = None
        self._queue = None

        self._build_pipeline()

        # Wait for first frame
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            img = self._queue.tryGet()
            if img is not None:
                self.frame = img.getCvFrame()
                self.grabbed = True
                break
            time.sleep(0.01)
        if not self.grabbed:
            self.stop()
            raise RuntimeError("OAK-D: no frames received within 5s")

    def _build_pipeline(self):
        dai = self._dai
        w, h = RESOLUTIONS[self._resolution_index]

        self._device = dai.Device()
        self._pipeline = dai.Pipeline(self._device)
        self._cam_node = self._pipeline.create(dai.node.Camera).build()
        output = self._cam_node.requestOutput(
            (w, h), type=dai.ImgFrame.Type.BGR888p, fps=60.0
        )
        self._queue = output.createOutputQueue(maxSize=2, blocking=False)
        self._pipeline.start()

    @property
    def resolution(self):
        w, h = RESOLUTIONS[self._resolution_index]
        return (w, h)

    @property
    def capture_fps(self):
        return self._capture_fps

    @property
    def exposure(self):
        return self._exposure_val

    @property
    def auto_exposure(self):
        return self._auto_exposure

    @property
    def gain(self):
        return self._gain_val

    @property
    def sharpen_enabled(self):
        return self._sharpen_enabled

    @property
    def exposure_us(self):
        """Current exposure time in microseconds."""
        return self._exposure_val

    @property
    def camera_index(self):
        return "OAK-D"

    def set_auto_exposure(self, enabled):
        """Enable or disable auto-exposure."""
        self._auto_exposure = enabled
        dai = self._dai
        ctrl = dai.CameraControl()
        if enabled:
            ctrl.setAutoExposureEnable()
        else:
            ctrl.setManualExposure(self._exposure_val, self._gain_val)
        self._cam_node.inputControl.send(ctrl)
        return enabled

    def set_exposure(self, value):
        """Set manual exposure in microseconds. Disables auto-exposure."""
        value = max(OAKD_EXPOSURE_MIN, min(OAKD_EXPOSURE_MAX, int(value)))
        self._exposure_val = value
        if self._auto_exposure:
            self.set_auto_exposure(False)
        else:
            self._apply_exposure()
        return value

    def adjust_exposure(self, delta):
        """Adjust exposure by delta steps (each step = 500us)."""
        return self.set_exposure(self._exposure_val + delta * OAKD_EXPOSURE_STEP)

    def _apply_exposure(self):
        """Send current exposure+gain to the camera."""
        dai = self._dai
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(self._exposure_val, self._gain_val)
        self._cam_node.inputControl.send(ctrl)

    def set_gain(self, value):
        """Set camera gain as ISO (100-1600)."""
        value = max(OAKD_GAIN_MIN, min(OAKD_GAIN_MAX, int(value)))
        self._gain_val = value
        if not self._auto_exposure:
            self._apply_exposure()
        return value

    def adjust_gain(self, delta):
        """Adjust gain by ISO steps. delta sign determines direction."""
        step = OAKD_GAIN_STEP if delta > 0 else -OAKD_GAIN_STEP
        return self.set_gain(self._gain_val + step)

    def set_sharpen(self, enabled):
        """Toggle frame sharpening post-processing."""
        self._sharpen_enabled = enabled
        return enabled

    def toggle_sharpen(self):
        """Toggle frame sharpening on/off."""
        self._sharpen_enabled = not self._sharpen_enabled
        return self._sharpen_enabled

    def cycle_resolution(self):
        """Cycle to the next resolution. Requires pipeline restart."""
        self.stopped = True
        time.sleep(0.1)
        self._resolution_index = (self._resolution_index + 1) % len(RESOLUTIONS)
        # Tear down current pipeline
        try:
            self._pipeline.stop()
        except Exception:
            pass
        try:
            self._device.close()
        except Exception:
            pass
        # Rebuild at new resolution
        self._build_pipeline()
        # Wait for first frame at new resolution
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            img = self._queue.tryGet()
            if img is not None:
                with self.lock:
                    self.frame = img.getCvFrame()
                    self.grabbed = True
                break
            time.sleep(0.01)
        # Re-apply exposure if manual
        if not self._auto_exposure:
            self._apply_exposure()
        self.stopped = False
        Thread(target=self._update, daemon=True).start()
        return self.resolution

    def switch_camera(self, new_src):
        """OAK-D doesn't support indexed camera switching."""
        return False

    def start(self):
        Thread(target=self._update, daemon=True).start()
        return self

    def _sharpen_frame(self, frame):
        """Apply unsharp mask sharpening."""
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        sharpened = cv2.addWeighted(
            frame, 1.0 + self._sharpen_strength,
            blurred, -self._sharpen_strength,
            0,
        )
        return sharpened

    def _update(self):
        while not self.stopped:
            img = self._queue.tryGet()
            if img is None:
                time.sleep(0.001)
                continue

            frame = img.getCvFrame()

            if self._sharpen_enabled:
                frame = self._sharpen_frame(frame)

            with self.lock:
                self.grabbed = True
                self.frame = frame

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

    def get_intrinsics(self):
        """Get camera intrinsics from OAK-D factory calibration.

        Returns (camera_matrix, dist_coeffs) as numpy arrays.
        OAK-D ISP output is already undistorted, so dist_coeffs are zero.
        """
        w, h = self.resolution
        try:
            calib = self._device.readCalibration()
            intrinsics = calib.getCameraIntrinsics(
                self._dai.CameraBoardSocket.CAM_A,
                self._dai.Size2f(w, h),
            )
            camera_matrix = np.array(intrinsics, dtype=np.float64)
        except Exception:
            try:
                calib = self._device.readCalibration2()
                intrinsics = calib.getCameraIntrinsics(
                    self._dai.CameraBoardSocket.CAM_A,
                    self._dai.Size2f(w, h),
                )
                camera_matrix = np.array(intrinsics, dtype=np.float64)
            except Exception as e:
                print(f"Could not read OAK-D calibration ({e}), estimating")
                hfov_rad = math.radians(69)  # OAK-D Pro ~69 deg HFOV
                fx = w / (2 * math.tan(hfov_rad / 2))
                camera_matrix = np.array([
                    [fx, 0, w / 2.0],
                    [0, fx, h / 2.0],
                    [0,  0,  1],
                ], dtype=np.float64)
        # ISP output is undistorted, so use zero distortion
        dist_coeffs = np.zeros(5, dtype=np.float64)
        return camera_matrix, dist_coeffs

    def get_capabilities(self):
        """Return OAK-D capabilities dict."""
        return {
            'device_index': 'OAK-D',
            'backend': 'DepthAI v3',
            'timestamp': datetime.now().isoformat(),
            'current_resolution': list(self.resolution),
            'supported_resolutions': [
                {'width': w, 'height': h, 'fps': 60.0, 'reported_fps': 60.0,
                 'fourcc': 'BGR', 'requested': None}
                for w, h in RESOLUTIONS
            ],
            'supported_codecs': ['BGR (ISP)'],
            'properties': {
                'fps': round(self._capture_fps, 1),
                'exposure_us': self._exposure_val,
                'gain_iso': self._gain_val,
                'auto_exposure': self._auto_exposure,
            },
        }

    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        try:
            if self._pipeline:
                self._pipeline.stop()
        except Exception:
            pass
        try:
            if self._device:
                self._device.close()
        except Exception:
            pass


def create_camera(src=0, resolution_index=1, backend=None):
    """Auto-detect OAK-D camera, fall back to webcam.

    Tries DepthAI first. If no OAK-D is found or depthai isn't installed,
    falls back to ThreadedCamera (OpenCV webcam).
    """
    try:
        cam = DepthAICamera(resolution_index=resolution_index)
        print("OAK-D Pro detected")
        return cam
    except ImportError:
        print("depthai not installed, using webcam")
    except Exception as e:
        print(f"OAK-D not available ({e}), using webcam")

    return ThreadedCamera(src=src, resolution_index=resolution_index, backend=backend)
