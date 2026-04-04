"""
ArUco marker tracking module for autonomous robot prototype.

Provides threaded camera capture and ArUco marker detection with
pose estimation, coordinate calibration, and overlay drawing.
"""

import math
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Resolution presets
# ---------------------------------------------------------------------------
RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]


# ---------------------------------------------------------------------------
# RobotPose dataclass
# ---------------------------------------------------------------------------
@dataclass
class RobotPose:
    """Detected robot pose from a single ArUco marker."""
    x_px: float
    y_px: float
    heading_rad: float
    corners: np.ndarray
    timestamp: float = field(default_factory=time.perf_counter)


# ---------------------------------------------------------------------------
# ThreadedCamera
# ---------------------------------------------------------------------------
class ThreadedCamera:
    """Background-threaded camera capture with minimal latency."""

    def __init__(self, src: int = 1, resolution_index: int = 1):
        self.src = src
        self.resolution_index = resolution_index
        w, h = RESOLUTIONS[resolution_index]

        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(src, backend)

        # Request MJPG codec
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Disable autofocus
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # FPS tracking
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = time.perf_counter()

    @property
    def fps(self) -> float:
        """Return the measured capture FPS."""
        return self._fps

    def start(self) -> "ThreadedCamera":
        """Start the background capture thread."""
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self._lock:
                self._frame = frame

            # Update FPS counter
            self._frame_count += 1
            now = time.perf_counter()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = now

    def read(self) -> Optional[np.ndarray]:
        """Return a copy of the latest frame, or None if no frame yet."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self) -> None:
        """Stop capture thread and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.cap.isOpened():
            self.cap.release()


# ---------------------------------------------------------------------------
# OAK-D Pro Camera (DepthAI v3)
# ---------------------------------------------------------------------------
class DepthAICamera:
    """OAK-D camera capture using DepthAI v3 SDK.

    Same interface as ThreadedCamera (fps, read, start, stop) for drop-in use.
    Supports mono camera (CAM_B/CAM_C) at up to 120fps with global shutter,
    or color camera (CAM_A) at up to 60fps.
    """

    def __init__(self, resolution_index: int = 1, use_mono: bool = False,
                 target_fps: float = 60.0):
        import depthai as dai
        self._dai = dai
        self._use_mono = use_mono

        self.resolution_index = resolution_index
        w, h = RESOLUTIONS[resolution_index]

        # Mono cameras are 1280x800 native — adjust if requesting 720p
        if use_mono:
            # OV9282 FPS depends on resolution:
            #   640x400: 120fps, 640x480: 90fps, 800x600: 60fps, 1280x800: 30fps
            if target_fps > 90:
                w, h = 640, 400   # 120fps
            elif target_fps > 60:
                w, h = 640, 480   # 90fps
            elif target_fps > 30:
                w, h = 800, 600   # 60fps
            else:
                w, h = 1280, 800  # 30fps

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._fps: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = time.perf_counter()

        # Build and start DepthAI pipeline
        self._device = dai.Device()

        # Choose camera socket
        if use_mono:
            socket = dai.CameraBoardSocket.CAM_B  # left mono
            img_type = dai.ImgFrame.Type.GRAY8
            sensor_fps = min(target_fps, 120.0)
            print(f"[camera] Using mono camera (CAM_B) at {sensor_fps:.0f}fps, global shutter")
        else:
            socket = dai.CameraBoardSocket.CAM_A  # color
            img_type = dai.ImgFrame.Type.BGR888p
            sensor_fps = min(target_fps, 60.0)

        # Read camera intrinsics before starting pipeline
        self.intrinsics = None
        try:
            calib = self._device.readCalibration()
            intr = calib.getCameraIntrinsics(socket, w, h)
            self.intrinsics = {
                'fx': intr[0][0], 'fy': intr[1][1],
                'cx': intr[0][2], 'cy': intr[1][2],
            }
            print(f"[camera] OAK-D intrinsics ({socket.name}): "
                  f"fx={intr[0][0]:.1f} fy={intr[1][1]:.1f} "
                  f"cx={intr[0][2]:.1f} cy={intr[1][2]:.1f}")
        except Exception as e:
            print(f"[camera] Could not read OAK-D intrinsics: {e}")

        self._pipeline = dai.Pipeline(self._device)
        self._pipeline.setXLinkChunkSize(0)  # minimum latency
        cam_node = self._pipeline.create(dai.node.Camera).build(
            socket, sensorFps=int(sensor_fps)
        )
        output = cam_node.requestOutput((w, h), type=img_type, fps=sensor_fps)
        self._queue = output.createOutputQueue(maxSize=1, blocking=False)
        self._pipeline.start()

        # Wait for first frame
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            img = self._queue.tryGet()
            if img is not None:
                self._frame = img.getCvFrame()
                break
            time.sleep(0.01)
        if self._frame is None:
            self.stop()
            raise RuntimeError("OAK-D: no frames received within 5s")

    @property
    def fps(self) -> float:
        return self._fps

    def start(self) -> "DepthAICamera":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        while self._running:
            img = self._queue.tryGet()
            if img is None:
                time.sleep(0.001)
                continue
            frame = img.getCvFrame()
            # Convert mono to BGR so overlays can use color
            if self._use_mono and frame is not None and len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            with self._lock:
                self._frame = frame
            self._frame_count += 1
            now = time.perf_counter()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = now

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._pipeline.stop()
        except Exception:
            pass
        try:
            self._device.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Camera factory
# ---------------------------------------------------------------------------
def create_camera(src: int = 0, resolution_index: int = 1, use_oakd: bool = False,
                   use_mono: bool = False, target_fps: float = 60.0):
    """Create the appropriate camera.

    If use_oakd is True, uses DepthAICamera (with optional mono for 120fps).
    Otherwise falls back to ThreadedCamera (USB webcam via OpenCV).
    """
    if use_oakd:
        try:
            cam = DepthAICamera(
                resolution_index=resolution_index,
                use_mono=use_mono,
                target_fps=target_fps,
            )
            cam_type = "mono (global shutter)" if use_mono else "color"
            print(f"[camera] OAK-D Pro {cam_type} — target {target_fps:.0f}fps")
            return cam
        except ImportError:
            print("[camera] depthai not installed — falling back to webcam")
        except Exception as e:
            print(f"[camera] OAK-D not available ({e}) — falling back to webcam")

    return ThreadedCamera(src=src, resolution_index=resolution_index)


# ---------------------------------------------------------------------------
# ArUcoTracker
# ---------------------------------------------------------------------------
class ArUcoTracker:
    """ArUco marker detector with tuned parameters for arena tracking."""

    def __init__(self, use_clahe: bool = True):
        self.use_clahe = use_clahe

        # Dictionary
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        # Tuned detector parameters
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 15
        params.adaptiveThreshWinSizeStep = 6
        params.minMarkerPerimeterRate = 0.04
        params.maxMarkerPerimeterRate = 4.0
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        self.params = params

        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)

        # CLAHE for preprocessing
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Simple calibration (fallback)
        self._px_per_cm: Optional[float] = None
        self._origin_x: float = 0.0
        self._origin_y: float = 0.0

        # Homography calibration (perspective-correct)
        self._homography: Optional[np.ndarray] = None  # pixel -> world
        self._homography_inv: Optional[np.ndarray] = None  # world -> pixel
        self._calib_points_px: list = []
        self._calib_points_cm: list = []

        # Marker physical size for auto-calibration
        self._marker_size_cm: float = 5.0  # 50mm default
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale, optionally apply CLAHE."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.use_clahe:
            gray = self._clahe.apply(gray)
        return gray

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect all ArUco markers in the frame.

        Returns a list of dicts with keys: id, corners, center, heading_rad.
        """
        gray = self._preprocess(frame)
        corners_list, ids, _ = self.detector.detectMarkers(gray)

        detections = []
        if ids is None:
            return detections

        for i, marker_id in enumerate(ids.flatten()):
            corners = corners_list[i][0]  # shape (4, 2)
            center = corners.mean(axis=0)

            # Heading: vector from midpoint of bottom edge to midpoint of top edge
            top_mid = (corners[0] + corners[1]) / 2.0
            bottom_mid = (corners[2] + corners[3]) / 2.0
            direction = top_mid - bottom_mid
            heading = math.atan2(direction[1], direction[0])

            detections.append({
                "id": int(marker_id),
                "corners": corners,
                "center": center,
                "heading_rad": heading,
            })

        return detections

    def get_robot_pose(
        self, frame: np.ndarray, marker_id: int = 1
    ) -> Optional[RobotPose]:
        """Return a RobotPose for the specified marker, or None if not found."""
        detections = self.detect(frame)
        for det in detections:
            if det["id"] == marker_id:
                return RobotPose(
                    x_px=float(det["center"][0]),
                    y_px=float(det["center"][1]),
                    heading_rad=det["heading_rad"],
                    corners=det["corners"],
                )
        return None

    # -- Calibration ---------------------------------------------------------

    def set_scale(self, px_per_cm: float) -> None:
        """Set the pixels-per-cm conversion factor."""
        self._px_per_cm = px_per_cm

    def set_origin(self, x_px: float, y_px: float) -> None:
        """Set the origin point for coordinate conversion."""
        self._origin_x = x_px
        self._origin_y = y_px

    def px_to_cm(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Convert pixel coordinates to cm.

        Uses homography if calibrated, otherwise falls back to simple scale.
        """
        if self._homography is not None:
            pt = np.array([[[x_px, y_px]]], dtype=np.float64)
            result = cv2.perspectiveTransform(pt, self._homography)
            return (float(result[0][0][0]), float(result[0][0][1]))

        if self._px_per_cm is None:
            raise ValueError("Scale not set. Call set_scale() first.")
        x_cm = (x_px - self._origin_x) / self._px_per_cm
        y_cm = (y_px - self._origin_y) / self._px_per_cm
        return (x_cm, y_cm)

    def cm_to_px(self, x_cm: float, y_cm: float) -> tuple[float, float]:
        """Convert world cm coordinates to pixel coordinates.

        Uses inverse homography if calibrated, otherwise simple scale.
        """
        if self._homography_inv is not None:
            pt = np.array([[[x_cm, y_cm]]], dtype=np.float64)
            result = cv2.perspectiveTransform(pt, self._homography_inv)
            return (float(result[0][0][0]), float(result[0][0][1]))

        if self._px_per_cm is None:
            raise ValueError("Scale not set. Call set_scale() first.")
        px_x = self._origin_x + x_cm * self._px_per_cm
        px_y = self._origin_y + y_cm * self._px_per_cm
        return (px_x, px_y)

    # -- Homography calibration ------------------------------------------------

    def set_marker_size(self, size_cm: float):
        """Set the physical marker size in cm."""
        self._marker_size_cm = size_cm

    def set_camera_matrix(self, frame_w: int, frame_h: int,
                          fx: Optional[float] = None, fy: Optional[float] = None,
                          cx: Optional[float] = None, cy: Optional[float] = None):
        """Set or auto-detect camera intrinsic matrix.

        Tries to read real intrinsics from OAK-D via DepthAI calibration data.
        Falls back to provided values or estimation from frame size.
        """
        # No auto-detection here — caller should provide values
        # (DepthAICamera reads intrinsics from the device on init)

        if fx is None:
            # Fallback estimate: focal length ~ 0.6 * frame_width
            fx = float(frame_w) * 0.6
        if fy is None:
            fy = fx
        if cx is None:
            cx = frame_w / 2.0
        if cy is None:
            cy = frame_h / 2.0

        self._camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ], dtype=np.float64)
        self._dist_coeffs = np.zeros(5, dtype=np.float64)

    def auto_calibrate(self, frame: np.ndarray, marker_id: int = 0) -> bool:
        """Compute floor homography using solvePnP + camera intrinsics.

        Uses the known marker size and camera intrinsics to solve the camera
        pose, then builds a homography where world axes align with the camera
        image (X=right, Y=down-into-floor) rather than with the marker rotation.
        """
        h, w = frame.shape[:2]
        if self._camera_matrix is None:
            self.set_camera_matrix(w, h)

        detections = self.detect(frame)
        target = None
        for det in detections:
            if det["id"] == marker_id:
                target = det
                break
        if target is None:
            print("[tracker] Auto-calibrate: marker not detected")
            return False

        corners = target["corners"]
        half = self._marker_size_cm / 2.0

        # 3D marker corners on Z=0 plane
        obj_points = np.array([
            [-half, -half, 0],
            [ half, -half, 0],
            [ half,  half, 0],
            [-half,  half, 0],
        ], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            obj_points, corners.astype(np.float64),
            self._camera_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            print("[tracker] Auto-calibrate: solvePnP failed")
            return False

        # Find the marker's X-axis direction in image space
        center_img, _ = cv2.projectPoints(
            np.array([[0, 0, 0]], dtype=np.float64), rvec, tvec,
            self._camera_matrix, self._dist_coeffs)
        xaxis_img, _ = cv2.projectPoints(
            np.array([[1, 0, 0]], dtype=np.float64), rvec, tvec,
            self._camera_matrix, self._dist_coeffs)
        c2d = center_img.reshape(2)
        x2d = xaxis_img.reshape(2)
        marker_angle = math.atan2(x2d[1] - c2d[1], x2d[0] - c2d[0])

        # Counter-rotate so world X points right in image
        cos_a = math.cos(-marker_angle)
        sin_a = math.sin(-marker_angle)

        # Build grid of camera-aligned world points, rotate to marker frame,
        # project to image, then compute homography
        grid_range = 200
        step = 25
        aligned_pts = []
        for x in range(-grid_range, grid_range + 1, step):
            for y in range(-grid_range, grid_range + 1, step):
                aligned_pts.append([float(x), float(y)])
        aligned_pts = np.array(aligned_pts, dtype=np.float64)

        # Rotate aligned -> marker coords for projection
        marker_3d = np.zeros((len(aligned_pts), 3), dtype=np.float64)
        for i, (wx, wy) in enumerate(aligned_pts):
            marker_3d[i] = [cos_a * wx + sin_a * wy,
                            -sin_a * wx + cos_a * wy, 0]

        img_proj, _ = cv2.projectPoints(
            marker_3d, rvec, tvec,
            self._camera_matrix, self._dist_coeffs)
        img_proj = img_proj.reshape(-1, 2)

        # Homography: image pixels -> camera-aligned world cm
        self._homography, status = cv2.findHomography(
            img_proj, aligned_pts, cv2.RANSAC, 5.0)
        if self._homography is None:
            print("[tracker] Auto-calibrate: homography failed")
            return False

        self._homography_inv = np.linalg.inv(self._homography)

        origin_px = self.cm_to_px(0.0, 0.0)
        self._origin_x = origin_px[0]
        self._origin_y = origin_px[1]

        side_px = np.mean([np.linalg.norm(corners[j] - corners[(j+1) % 4])
                           for j in range(4)])
        self._px_per_cm = side_px / self._marker_size_cm

        self._calib_points_px = []
        self._calib_points_cm = []

        inliers = int(status.sum()) if status is not None else len(aligned_pts)
        print(f"[tracker] Auto-calibrated: marker {marker_id}, "
              f"size={self._marker_size_cm}cm, "
              f"px_per_cm={self._px_per_cm:.1f}, "
              f"marker ~{side_px:.0f}px, "
              f"angle={math.degrees(marker_angle):.1f}deg, "
              f"{inliers}/{len(aligned_pts)} inliers")
        return True

    # -- Drive-around calibration -----------------------------------------------

    def start_calibration_drive(self):
        """Begin drive-around calibration. Drive the robot around the floor."""
        self._calib_driving = True
        self._calib_points_px = []
        self._calib_points_cm = []
        self._calib_last_px = None
        self._calib_world_x = 0.0
        self._calib_world_y = 0.0
        self._calib_min_move_px = 25  # min pixel movement between captures
        # Clear existing homography so we start fresh
        self._homography = None
        self._homography_inv = None
        print("[tracker] Drive calibration started — drive the robot around")

    def update_calibration_drive(self, frame: np.ndarray,
                                  marker_id: int = 0) -> tuple[bool, int]:
        """Called each frame during calibration drive.

        Auto-captures points when the marker moves enough.
        Uses the marker's pixel size to convert pixel displacement to cm.
        Returns (captured_this_frame, total_points).
        """
        if not self._calib_driving:
            return False, len(self._calib_points_px)

        detections = self.detect(frame)
        target = None
        for det in detections:
            if det["id"] == marker_id:
                target = det
                break
        if target is None:
            return False, len(self._calib_points_px)

        corners = target["corners"]
        center = target["center"]
        cx, cy = float(center[0]), float(center[1])

        # Compute current px_per_cm from marker size in this frame
        side_px = float(np.mean([
            np.linalg.norm(corners[j] - corners[(j + 1) % 4])
            for j in range(4)
        ]))
        local_px_per_cm = side_px / self._marker_size_cm

        if self._calib_last_px is None:
            # First point — this becomes the origin (0, 0)
            self._calib_last_px = (cx, cy)
            self._calib_world_x = 0.0
            self._calib_world_y = 0.0
            self._calib_points_px.append((cx, cy))
            self._calib_points_cm.append((0.0, 0.0))
            self._px_per_cm = local_px_per_cm
            self._origin_x = cx
            self._origin_y = cy
            print(f"[calib] Origin set: pixel=({cx:.0f},{cy:.0f}), "
                  f"px_per_cm={local_px_per_cm:.1f}")
            return True, len(self._calib_points_px)

        # Check if we've moved enough
        dx_px = cx - self._calib_last_px[0]
        dy_px = cy - self._calib_last_px[1]
        dist_px = math.hypot(dx_px, dy_px)

        if dist_px < self._calib_min_move_px:
            return False, len(self._calib_points_px)

        # Convert pixel displacement to cm using local marker scale
        dx_cm = dx_px / local_px_per_cm
        dy_cm = dy_px / local_px_per_cm

        # Accumulate world position
        self._calib_world_x += dx_cm
        self._calib_world_y += dy_cm

        # Store the point
        self._calib_points_px.append((cx, cy))
        self._calib_points_cm.append((self._calib_world_x, self._calib_world_y))
        self._calib_last_px = (cx, cy)

        n = len(self._calib_points_px)
        if n % 5 == 0:
            print(f"[calib] {n} points collected, "
                  f"latest=({self._calib_world_x:.1f},{self._calib_world_y:.1f})cm")
        return True, n

    def finish_calibration_drive(self) -> bool:
        """End drive-around calibration and compute homography."""
        self._calib_driving = False
        n = len(self._calib_points_px)
        print(f"[calib] Drive calibration finished with {n} points")
        if n < 4:
            print("[calib] Need at least 4 points for homography")
            return False
        return self.compute_homography()

    @property
    def is_calibrating(self) -> bool:
        return getattr(self, '_calib_driving', False)

    @property
    def calib_point_count(self) -> int:
        return len(self._calib_points_px)

    def add_calibration_point(self, px_x: float, px_y: float,
                               x_cm: float, y_cm: float) -> int:
        """Add a pixel↔world calibration point pair. Returns total point count."""
        self._calib_points_px.append((px_x, px_y))
        self._calib_points_cm.append((x_cm, y_cm))
        return len(self._calib_points_px)

    def clear_calibration_points(self):
        """Clear all calibration points."""
        self._calib_points_px.clear()
        self._calib_points_cm.clear()

    def compute_homography(self) -> bool:
        """Compute homography from collected calibration points.

        Requires at least 4 points. Returns True on success.
        """
        n = len(self._calib_points_px)
        if n < 4:
            print(f"[tracker] Need at least 4 calibration points, have {n}")
            return False

        pts_px = np.array(self._calib_points_px, dtype=np.float64)
        pts_cm = np.array(self._calib_points_cm, dtype=np.float64)

        # pixel -> world
        self._homography, status = cv2.findHomography(pts_px, pts_cm)
        if self._homography is None:
            print("[tracker] Homography computation failed")
            return False

        # world -> pixel (inverse)
        self._homography_inv = np.linalg.inv(self._homography)

        # Also update origin to (0,0) in world = some pixel
        origin_px = self.cm_to_px(0.0, 0.0)
        self._origin_x = origin_px[0]
        self._origin_y = origin_px[1]

        inliers = int(status.sum()) if status is not None else n
        print(f"[tracker] Homography computed from {n} points ({inliers} inliers)")
        return True

    @property
    def has_homography(self) -> bool:
        return self._homography is not None

    def save_homography(self, path: str):
        """Save homography and calibration points to a JSON file."""
        import json
        data = {
            "points_px": self._calib_points_px,
            "points_cm": self._calib_points_cm,
            "homography": self._homography.tolist() if self._homography is not None else None,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[tracker] Homography saved to {path}")

    def load_homography(self, path: str) -> bool:
        """Load homography from a JSON file. Returns True on success."""
        import json
        try:
            with open(path) as f:
                data = json.load(f)
            self._calib_points_px = data.get("points_px", [])
            self._calib_points_cm = data.get("points_cm", [])
            h = data.get("homography")
            if h is not None:
                self._homography = np.array(h, dtype=np.float64)
                self._homography_inv = np.linalg.inv(self._homography)
                origin_px = self.cm_to_px(0.0, 0.0)
                self._origin_x = origin_px[0]
                self._origin_y = origin_px[1]
                print(f"[tracker] Homography loaded from {path} ({len(self._calib_points_px)} points)")
                return True
        except (FileNotFoundError, json.JSONDecodeError, np.linalg.LinAlgError) as e:
            print(f"[tracker] Failed to load homography: {e}")
        return False

    # -- Floor plane calibration (single marker + click) ----------------------

    def load_floor_plane(self, path: str = None) -> bool:
        """Load floor plane calibration and set the tracker's homography.

        The floor plane calibration (from floor_plane.py) uses centimeters
        as the world unit, matching this tracker's px_to_cm/cm_to_px system.
        """
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "floor_calibration.json",
            )
        import json
        try:
            with open(path) as f:
                data = json.load(f)
            h = data.get("homography")
            if h is None:
                return False
            self._homography = np.array(h, dtype=np.float64)
            self._homography_inv = np.linalg.inv(self._homography)
            origin_px = self.cm_to_px(0.0, 0.0)
            self._origin_x = origin_px[0]
            self._origin_y = origin_px[1]
            corners = data.get("corners_ft", [])  # named _ft but actually cm in auto-drive
            print(f"[tracker] Floor plane calibration loaded from {path}")
            print(f"[tracker]   {len(corners)} arena corners")
            return True
        except Exception as e:
            print(f"[tracker] Failed to load floor plane: {e}")
            return False

    # -- ChArUco board calibration --------------------------------------------

    # Board spec: 8x6 grid, 50mm squares, 25mm markers on 4 letter pages
    # Physical size: 8*50mm=400mm x 6*50mm=300mm (~15.7" x 11.8")
    CHARUCO_COLS = 8
    CHARUCO_ROWS = 6
    CHARUCO_SQUARE_M = 0.050   # 50mm in meters
    CHARUCO_MARKER_M = 0.025   # 25mm in meters

    def generate_charuco_board(self, output_path: str = "charuco_board.png") -> str:
        """Generate a printable ChArUco board image.

        Board is 8x6 squares at 50mm each = 400mm x 300mm.
        Fits on 4 letter (8.5x11") pages taped together (2 wide x 2 tall).
        Uses DICT_4X4_100 to avoid conflicts with robot marker (ID 0 in DICT_4X4_50).
        """
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        board = cv2.aruco.CharucoBoard(
            (self.CHARUCO_COLS, self.CHARUCO_ROWS),
            self.CHARUCO_SQUARE_M,
            self.CHARUCO_MARKER_M,
            dictionary,
        )
        # Generate at 300 DPI: 400mm = 15.75" -> 4724px, 300mm = 11.81" -> 3543px
        img = board.generateImage((4724, 3543), marginSize=50)
        cv2.imwrite(output_path, img)
        print(f"[tracker] ChArUco board saved to {output_path} "
              f"({self.CHARUCO_COLS}x{self.CHARUCO_ROWS}, "
              f"{self.CHARUCO_SQUARE_M*1000:.0f}mm squares)")
        return output_path

    def calibrate_from_charuco(self, frame: np.ndarray) -> bool:
        """Detect ChArUco board on the floor and compute floor homography.

        The board's coordinate system becomes the world coordinate system:
        - Origin at the board's top-left corner
        - X along the board's long edge (columns)
        - Y along the board's short edge (rows)
        - Units: centimeters

        Requires camera_matrix to be set.
        """
        if self._camera_matrix is None:
            h, w = frame.shape[:2]
            self.set_camera_matrix(w, h)

        # Use DICT_4X4_100 for the ChArUco board
        charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        board = cv2.aruco.CharucoBoard(
            (self.CHARUCO_COLS, self.CHARUCO_ROWS),
            self.CHARUCO_SQUARE_M,
            self.CHARUCO_MARKER_M,
            charuco_dict,
        )

        # Detect ArUco markers first
        charuco_params = cv2.aruco.DetectorParameters()
        charuco_detector = cv2.aruco.ArucoDetector(charuco_dict, charuco_params)

        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = charuco_detector.detectMarkers(gray)

        if marker_ids is None or len(marker_ids) < 4:
            n = 0 if marker_ids is None else len(marker_ids)
            print(f"[tracker] ChArUco: only {n} markers detected (need >= 4)")
            return False

        # Interpolate ChArUco corners (sub-pixel accurate)
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )

        if not ret or charuco_corners is None or len(charuco_corners) < 6:
            n = 0 if charuco_corners is None else len(charuco_corners)
            print(f"[tracker] ChArUco: only {n} corners interpolated (need >= 6)")
            return False

        n_corners = len(charuco_corners)

        # Estimate board pose (camera extrinsics relative to board)
        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board,
            self._camera_matrix, self._dist_coeffs,
            None, None,
        )

        if not success:
            print("[tracker] ChArUco: pose estimation failed")
            return False

        # Build floor homography from camera extrinsics
        # For Z=0 plane: H = K * [r1 | r2 | t] (drop 3rd column of R)
        R, _ = cv2.Rodrigues(rvec)
        extrinsic = np.hstack([R, tvec])           # 3x4
        P = self._camera_matrix @ extrinsic         # 3x4 projection
        H_world_to_pixel = P[:, [0, 1, 3]]         # drop Z column → 3x3

        # Invert: pixel → world (in meters)
        H_pixel_to_world_m = np.linalg.inv(H_world_to_pixel)
        H_pixel_to_world_m /= H_pixel_to_world_m[2, 2]  # normalize

        # Scale meters → centimeters
        scale = np.diag([100.0, 100.0, 1.0])
        self._homography = scale @ H_pixel_to_world_m    # pixel → cm
        self._homography_inv = np.linalg.inv(self._homography)  # cm → pixel

        # Update origin and px_per_cm
        origin_px = self.cm_to_px(0.0, 0.0)
        self._origin_x = origin_px[0]
        self._origin_y = origin_px[1]

        # Estimate px_per_cm from 1cm offset
        p0 = np.array(self.cm_to_px(0, 0))
        p1 = np.array(self.cm_to_px(1, 0))
        self._px_per_cm = float(np.linalg.norm(p1 - p0))

        # Compute reprojection error
        obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
        if obj_pts is not None and len(obj_pts) > 0:
            reproj, _ = cv2.projectPoints(obj_pts, rvec, tvec,
                                          self._camera_matrix, self._dist_coeffs)
            error = np.sqrt(np.mean((img_pts - reproj.reshape(-1, 1, 2)) ** 2))
        else:
            error = -1

        self._calib_points_px = []
        self._calib_points_cm = []

        print(f"[tracker] ChArUco calibrated: {n_corners} corners, "
              f"reproj error={error:.2f}px, px_per_cm={self._px_per_cm:.1f}")
        return True


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------
def draw_overlay(
    frame: np.ndarray,
    pose: Optional[RobotPose],
    waypoints: Optional[list[tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Draw tracking overlay on the frame (modifies in-place, also returns it).

    - Marker bounding box (green)
    - Center dot (red)
    - Heading arrow (blue)
    - Position and heading text
    - Optional waypoints as cyan circles
    """
    if pose is not None:
        corners = pose.corners.astype(np.int32)

        # Bounding box
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

        # Center dot
        cx, cy = int(pose.x_px), int(pose.y_px)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Heading arrow
        arrow_len = 50
        ax = int(cx + arrow_len * math.cos(pose.heading_rad))
        ay = int(cy + arrow_len * math.sin(pose.heading_rad))
        cv2.arrowedLine(frame, (cx, cy), (ax, ay), (255, 0, 0), 2, tipLength=0.3)

        # Text info
        heading_deg = math.degrees(pose.heading_rad)
        text = f"({cx}, {cy})  {heading_deg:.1f}deg"
        cv2.putText(
            frame, text, (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # Waypoints
    if waypoints:
        for wx, wy in waypoints:
            cv2.circle(frame, (int(wx), int(wy)), 8, (255, 255, 0), 2)

    return frame
