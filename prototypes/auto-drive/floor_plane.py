"""Floor plane detection and arena grid.

Projects a grid onto the arena floor for robot navigation.
Uses a single ArUco marker on the floor to establish the camera's 3D pose,
then lets you click 4 arena corners on the live stream. Each click is
projected onto the real floor plane with accurate world coordinates.

The arena can be any quadrilateral shape -- it doesn't have to be square.

Calibration methods (in priority order):
  1. Single-marker pose + click: drop one large ArUco marker (ID 10) on
     the floor, press 'p', then click 4 corners on the live stream.
     Each click shows real-world coordinates. Best method.
  2. 4-corner ArUco: place markers ID 0-3 at the arena corners.
  3. Manual click: click 4 corners on a frozen frame (no 3D awareness).

Usage:
  - Print the calibration marker: python generate_markers.py
  - Press 'p' in main.py to calibrate
  - Press '5' for grid mode
  - Any pixel coordinate can be converted to world cm and back

Standalone calibration:
  python floor_plane.py
"""

import json
import math
import os
import time

import cv2
import numpy as np

CALIBRATION_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "floor_calibration.json"
)
_CALIB_WINDOW = "Floor Calibration"


def _destroy_calib_window():
    try:
        cv2.destroyWindow(_CALIB_WINDOW)
    except cv2.error:
        pass


# Arena defaults (overridden by calibration data)
# Auto-drive uses centimeters as the world unit
ARENA_WIDTH_FT = 243.84   # 8ft = 243.84cm
ARENA_HEIGHT_FT = 243.84  # 8ft = 243.84cm
GRID_DIVISIONS = 8  # ~30cm squares

# Single calibration marker config (must match generate_markers.py)
CALIB_MARKER_ID = 10
# IMPORTANT: This must be the ACTUAL printed size of the marker.
# Measure the outer black square edge-to-edge with a ruler.
# The PNG is designed for 7.5" but most printers scale it down.
CALIB_MARKER_MM = 174.6  # 6-7/8" as measured -- update if yours differs

# 4-corner marker IDs (legacy method)
CORNER_MARKERS = {
    0: (0, 0),
    1: (ARENA_WIDTH_FT, 0),
    2: (ARENA_WIDTH_FT, ARENA_HEIGHT_FT),
    3: (0, ARENA_HEIGHT_FT),
}

# World unit conversion: mm from solvePnP -> output unit
# cv-tracking version uses feet (304.8), auto-drive uses centimeters (10.0)
MM_PER_UNIT = 10.0  # millimeters per centimeter


# ======================================================================
# 3D pose math
# ======================================================================

def estimate_marker_pose(frame, camera_matrix, dist_coeffs,
                         marker_id=CALIB_MARKER_ID,
                         marker_size_mm=CALIB_MARKER_MM):
    """Detect a single ArUco marker and estimate its 3D pose.

    The marker defines the floor plane: its center is the origin,
    X-axis along the marker's right edge, Y-axis along the top edge,
    Z-axis pointing up from the floor.

    Returns:
        (rvec, tvec, corners) if marker found, or (None, None, None).
        rvec/tvec are the camera pose relative to the marker.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    all_corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, None, None

    # Find our calibration marker
    ids_flat = ids.flatten().tolist()
    if marker_id not in ids_flat:
        return None, None, None

    idx = ids_flat.index(marker_id)
    corners_2d = all_corners[idx][0]  # shape (4, 2)

    # 3D object points for the marker corners (in marker coordinate system)
    # ArUco corner order: top-left, top-right, bottom-right, bottom-left
    half = marker_size_mm / 2.0
    obj_pts = np.array([
        [-half,  half, 0],  # top-left
        [ half,  half, 0],  # top-right
        [ half, -half, 0],  # bottom-right
        [-half, -half, 0],  # bottom-left
    ], dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, corners_2d, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not success:
        return None, None, None

    return rvec, tvec, corners_2d


def pixel_to_floor_3d(px, py, camera_matrix, rvec, tvec):
    """Project a pixel onto the floor plane (Z=0) using the camera pose.

    Returns (world_x_mm, world_y_mm) on the floor, or None if the ray
    doesn't intersect the floor (pointing above horizon).
    """
    K_inv = np.linalg.inv(camera_matrix)
    R, _ = cv2.Rodrigues(rvec)

    # Camera position in world (marker) coordinates
    cam_pos = (-R.T @ tvec).flatten()

    # Ray from camera through the pixel, in world coordinates
    ray_cam = (K_inv @ np.array([px, py, 1.0], dtype=np.float64)).flatten()
    ray_world = (R.T @ ray_cam).flatten()

    # Intersect with Z=0 plane
    if abs(ray_world[2]) < 1e-10:
        return None  # ray parallel to floor

    t = -cam_pos[2] / ray_world[2]
    if t < 0:
        return None  # intersection behind camera (above horizon)

    point = cam_pos + t * ray_world
    return (float(point[0]), float(point[1]))


# ======================================================================
# FloorPlaneDetector
# ======================================================================

class FloorPlaneDetector:
    """Arena floor plane with pixel <-> world coordinate mapping.

    Supports arbitrary quadrilateral arenas. Calibration produces a
    homography for fast runtime coordinate conversion and grid drawing.
    Optionally stores the full 3D camera pose for advanced use.
    """

    def __init__(self):
        self.calibrated = False
        self.homography = None       # pixel -> world transform
        self.inv_homography = None   # world -> pixel transform
        self.floor_corners_px = None  # 4 arena corners in pixel space
        self.floor_corners_ft = None  # 4 arena corners in world cm
        self._grid_lines_px = None   # precomputed grid lines for fast drawing
        self._rgb_size = (1280, 720)
        # 3D pose (optional, from single-marker calibration)
        self._rvec = None
        self._tvec = None
        self._camera_matrix = None

        # Try to load existing calibration
        self.load_calibration()

    def load_calibration(self, path=None):
        """Load calibration from JSON file."""
        path = path or CALIBRATION_FILE
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self.floor_corners_px = np.array(data["corners_px"], dtype=np.float32)
            self.floor_corners_ft = np.array(data["corners_ft"], dtype=np.float32)
            self.homography = np.array(data["homography"], dtype=np.float64)
            self.inv_homography = np.array(data["inv_homography"], dtype=np.float64)
            self._rgb_size = tuple(data.get("rgb_size", [1280, 720]))
            if "rvec" in data and data["rvec"] is not None:
                self._rvec = np.array(data["rvec"], dtype=np.float64)
                self._tvec = np.array(data["tvec"], dtype=np.float64)
                self._camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
            self.calibrated = True
            self._precompute_grid()
            print(f"Floor calibration loaded ({path})")
            return True
        except Exception as e:
            print(f"Failed to load floor calibration: {e}")
            return False

    def save_calibration(self, path=None):
        """Save calibration to JSON file."""
        path = path or CALIBRATION_FILE
        data = {
            "corners_px": self.floor_corners_px.tolist(),
            "corners_ft": self.floor_corners_ft.tolist(),
            "homography": self.homography.tolist(),
            "inv_homography": self.inv_homography.tolist(),
            "rgb_size": list(self._rgb_size),
            "rvec": self._rvec.tolist() if self._rvec is not None else None,
            "tvec": self._tvec.tolist() if self._tvec is not None else None,
            "camera_matrix": self._camera_matrix.tolist() if self._camera_matrix is not None else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Floor calibration saved to {path}")

    # ------------------------------------------------------------------
    # Calibration methods
    # ------------------------------------------------------------------

    def calibrate_from_pose(self, corners_px, corners_mm, rvec, tvec,
                            camera_matrix, rgb_size):
        """Calibrate from clicked corners with known 3D pose.

        Args:
            corners_px: list of 4 (x, y) pixel coordinates
            corners_mm: list of 4 (x_mm, y_mm) world coordinates from ray projection
            rvec, tvec: camera pose from marker detection
            camera_matrix: camera intrinsics
            rgb_size: (width, height)

        Returns True on success.
        """
        # Convert mm to cm
        corners_ft = [[x / MM_PER_UNIT, y / MM_PER_UNIT] for x, y in corners_mm]

        self._rvec = rvec.copy()
        self._tvec = tvec.copy()
        self._camera_matrix = camera_matrix.copy()

        return self._compute_homography(corners_px, corners_ft, rgb_size)

    def calibrate_from_aruco(self, frame):
        """Calibrate using ArUco markers ID 0-3 at the 4 arena corners."""
        rgb_size = (frame.shape[1], frame.shape[0])

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            return False

        found_ids = set(ids.flatten().tolist())
        missing = set(CORNER_MARKERS.keys()) - found_ids
        if missing:
            print(f"Missing corner markers: {sorted(missing)}")
            return False

        pixel_corners = []
        world_corners = []
        for marker_id in sorted(CORNER_MARKERS.keys()):
            idx = list(ids.flatten()).index(marker_id)
            pts = all_corners[idx][0]
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            pixel_corners.append([cx, cy])
            world_corners.append(list(CORNER_MARKERS[marker_id]))
            wx, wy = CORNER_MARKERS[marker_id]
            print(f"  ID {marker_id} at pixel ({cx:.0f}, {cy:.0f}) -> ({wx}, {wy}) ft")

        return self._compute_homography(pixel_corners, world_corners, rgb_size)

    def calibrate_from_corners(self, corners, rgb_size=(1280, 720)):
        """Calibrate from clicked corners (at least 4, assumes rectangular arena).

        If exactly 4 corners: maps to TL, TR, BR, BL of arena.
        If more: distributes points evenly along the arena perimeter.
        """
        pixel_corners = [[float(c[0]), float(c[1])] for c in corners]
        n = len(pixel_corners)

        if n == 4:
            world_corners = [
                [0, 0],
                [ARENA_WIDTH_FT, 0],
                [ARENA_WIDTH_FT, ARENA_HEIGHT_FT],
                [0, ARENA_HEIGHT_FT],
            ]
        else:
            # Distribute points along arena perimeter
            w, h = ARENA_WIDTH_FT, ARENA_HEIGHT_FT
            perimeter = 2 * (w + h)
            # Build perimeter path: TL→TR→BR→BL→TL
            edges = [
                ([0, 0], [w, 0], w),          # top
                ([w, 0], [w, h], h),           # right
                ([w, h], [0, h], w),           # bottom
                ([0, h], [0, 0], h),           # left
            ]
            world_corners = []
            for i in range(n):
                t = i / n * perimeter
                # Walk along edges to find position at distance t
                walked = 0
                for (start, end, length) in edges:
                    if walked + length >= t:
                        frac = (t - walked) / length
                        wx = start[0] + frac * (end[0] - start[0])
                        wy = start[1] + frac * (end[1] - start[1])
                        world_corners.append([wx, wy])
                        break
                    walked += length

        return self._compute_homography(pixel_corners, world_corners, rgb_size)

    def _compute_homography(self, pixel_corners, world_corners_ft, rgb_size):
        """Compute homography from pixel corners to world coordinates in cm."""
        self._rgb_size = rgb_size

        px = np.array(pixel_corners, dtype=np.float32)
        wx = np.array(world_corners_ft, dtype=np.float32)

        if len(px) < 4 or len(wx) < 4:
            print("Need at least 4 corner points")
            return False

        # Use convex hull for the arena boundary polygon (for grid drawing)
        # This ensures all points are included, not just the first 4
        hull_idx = cv2.convexHull(px.reshape(-1, 1, 2), returnPoints=False).flatten()
        self.floor_corners_px = px[hull_idx]
        self.floor_corners_ft = wx[hull_idx]

        # Homography uses ALL points for best fit
        self.homography, _ = cv2.findHomography(px, wx)
        self.inv_homography, _ = cv2.findHomography(wx, px)

        if self.homography is None or self.inv_homography is None:
            print("Failed to compute homography")
            return False

        self.calibrated = True
        self._precompute_grid()
        self.save_calibration()

        # Print arena dimensions
        d01 = np.linalg.norm(wx[0] - wx[1])
        d12 = np.linalg.norm(wx[1] - wx[2])
        d23 = np.linalg.norm(wx[2] - wx[3])
        d30 = np.linalg.norm(wx[3] - wx[0])
        print(f"Floor calibration complete!")
        print(f"  Arena edges: {d01:.1f} x {d12:.1f} x {d23:.1f} x {d30:.1f} ft")
        return True

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def pixel_to_world(self, px, py):
        """Convert pixel to world coordinates (cm). Returns (x, y) or None."""
        if not self.calibrated:
            return None
        pt = np.array([[[float(px), float(py)]]], dtype=np.float64)
        world = cv2.perspectiveTransform(pt, self.homography)
        return (float(world[0][0][0]), float(world[0][0][1]))

    def world_to_pixel(self, wx, wy):
        """Convert world (cm) to pixel coordinates. Returns (x, y) or None."""
        if not self.calibrated:
            return None
        pt = np.array([[[float(wx), float(wy)]]], dtype=np.float64)
        pixel = cv2.perspectiveTransform(pt, self.inv_homography)
        return (int(round(pixel[0][0][0])), int(round(pixel[0][0][1])))

    # ------------------------------------------------------------------
    # Grid drawing
    # ------------------------------------------------------------------

    def _precompute_grid(self):
        """Precompute grid lines covering the arena area.

        Uses world coordinate bounding box — works with any number of corners.
        """
        if not self.calibrated or self.floor_corners_ft is None:
            return

        # World coordinate bounding box
        wx_min = float(self.floor_corners_ft[:, 0].min())
        wx_max = float(self.floor_corners_ft[:, 0].max())
        wy_min = float(self.floor_corners_ft[:, 1].min())
        wy_max = float(self.floor_corners_ft[:, 1].max())

        lines = []
        n = GRID_DIVISIONS

        # Vertical lines (constant x, varying y)
        for i in range(n + 1):
            wx = wx_min + (wx_max - wx_min) * i / n
            p1 = self.world_to_pixel(wx, wy_min)
            p2 = self.world_to_pixel(wx, wy_max)
            if p1 and p2:
                lines.append((p1, p2))

        # Horizontal lines (constant y, varying x)
        for i in range(n + 1):
            wy = wy_min + (wy_max - wy_min) * i / n
            p1 = self.world_to_pixel(wx_min, wy)
            p2 = self.world_to_pixel(wx_max, wy)
            if p1 and p2:
                lines.append((p1, p2))

        self._grid_lines_px = lines

    def draw_grid(self, frame, color=(0, 200, 0), thickness=1, alpha=0.25):
        """Draw the arena grid overlay on the frame."""
        if not self.calibrated or not self._grid_lines_px:
            return

        # Semi-transparent floor fill
        if alpha > 0 and self.floor_corners_px is not None:
            overlay = frame.copy()
            pts = self.floor_corners_px.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 40, 0))
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Grid lines
        for p1, p2 in self._grid_lines_px:
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

        # Arena border (thicker)
        if self.floor_corners_px is not None:
            pts = self.floor_corners_px.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, color, thickness + 1, cv2.LINE_AA)

        # Corner dots
        if self.floor_corners_px is not None:
            for i, corner in enumerate(self.floor_corners_px):
                cx, cy = int(corner[0]), int(corner[1])
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 0), 1)
                # Show world coordinates at each corner
                if self.floor_corners_ft is not None:
                    wx, wy = self.floor_corners_ft[i]
                    label = f"({wx:.1f},{wy:.1f})"
                    cv2.putText(frame, label, (cx + 8, cy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
                    cv2.putText(frame, label, (cx + 8, cy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    def draw_world_position(self, frame, detection, color=(255, 0, 255)):
        """Annotate a detection with its world-coordinate position."""
        if not self.calibrated:
            return
        world = self.pixel_to_world(*detection.center_px)
        if world is None:
            return
        wx, wy = world
        cx, cy = detection.center_px
        label = f"({wx:.1f},{wy:.1f})cm"
        cv2.putText(frame, label, (cx + 10, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, label, (cx + 10, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ======================================================================
# Auto-detect arena walls
# ======================================================================

def auto_detect_arena(frame, min_area_frac=0.05, max_area_frac=0.8,
                      num_points=8, debug=False):
    """Detect arena walls from a camera frame.

    Finds the largest rectangular-ish contour that looks like an arena floor.
    Returns list of (x, y) pixel points along the boundary, or None.

    Args:
        frame: BGR camera frame
        min_area_frac: minimum contour area as fraction of frame
        max_area_frac: maximum contour area as fraction of frame
        num_points: number of points to place along the boundary
        debug: if True, show intermediate steps
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_area = h * w

    # Try multiple approaches and pick the best contour

    # Approach 1: Adaptive threshold (good for textured floors)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 51, 5)

    # Approach 2: Canny edges + dilation to close gaps
    edges = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_closed = cv2.dilate(edges, kernel, iterations=2)
    edges_closed = cv2.erode(edges_closed, kernel, iterations=1)

    # Try both and pick whichever gives a better arena contour
    best_contour = None
    best_score = 0

    for mask in [thresh, edges_closed]:
        # Invert if needed (arena should be the filled region)
        for inv in [False, True]:
            m = cv2.bitwise_not(mask) if inv else mask

            # Morphological close to fill small gaps
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                area = cv2.contourArea(c)
                frac = area / total_area

                if frac < min_area_frac or frac > max_area_frac:
                    continue

                # Approximate to polygon
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                # Score: prefer 4-sided, large, convex shapes
                n_sides = len(approx)
                if n_sides < 4:
                    continue

                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                # Rectangularity: area / bounding rect area
                rect = cv2.minAreaRect(c)
                rect_area = rect[1][0] * rect[1][1]
                rectangularity = area / rect_area if rect_area > 0 else 0

                score = frac * solidity * rectangularity
                if n_sides == 4:
                    score *= 1.5  # bonus for quadrilateral

                if score > best_score:
                    best_score = score
                    best_contour = c

    if best_contour is None:
        print("[auto-detect] No arena contour found")
        return None

    # Get the convex hull for clean boundary
    hull = cv2.convexHull(best_contour)

    # Sample points evenly along the hull perimeter
    peri = cv2.arcLength(hull, True)
    points = []

    # Walk along the hull contour and sample at even intervals
    hull_pts = hull.reshape(-1, 2)
    n_hull = len(hull_pts)

    # Calculate cumulative distances along hull
    cum_dist = [0.0]
    for i in range(1, n_hull):
        d = np.linalg.norm(hull_pts[i].astype(float) - hull_pts[i-1].astype(float))
        cum_dist.append(cum_dist[-1] + d)
    # Close the loop
    cum_dist.append(cum_dist[-1] + np.linalg.norm(
        hull_pts[0].astype(float) - hull_pts[-1].astype(float)))
    total_dist = cum_dist[-1]

    for i in range(num_points):
        target_dist = total_dist * i / num_points

        # Find which segment this falls on
        for j in range(len(cum_dist) - 1):
            if cum_dist[j] <= target_dist <= cum_dist[j+1]:
                seg_len = cum_dist[j+1] - cum_dist[j]
                if seg_len < 0.01:
                    frac = 0
                else:
                    frac = (target_dist - cum_dist[j]) / seg_len
                p1 = hull_pts[j % n_hull].astype(float)
                p2 = hull_pts[(j+1) % n_hull].astype(float)
                pt = p1 + frac * (p2 - p1)
                points.append((int(pt[0]), int(pt[1])))
                break

    if debug:
        vis = frame.copy()
        cv2.drawContours(vis, [best_contour], -1, (255, 0, 0), 2)
        cv2.drawContours(vis, [hull], -1, (0, 255, 0), 2)
        for i, (px, py) in enumerate(points):
            cv2.circle(vis, (px, py), 8, (0, 255, 255), -1)
            cv2.putText(vis, str(i+1), (px+10, py-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("Auto-Detect Debug", vis)
        cv2.waitKey(0)
        cv2.destroyWindow("Auto-Detect Debug")

    area_frac = cv2.contourArea(best_contour) / total_area
    print(f"[auto-detect] Found arena: {len(points)} points, "
          f"{area_frac*100:.0f}% of frame, score={best_score:.3f}")

    return points


# ======================================================================
# Interactive single-marker calibration (live stream + click)
# ======================================================================

_click_points = []
_click_world = []  # world coords for each click


_pose_drag_state = {"dragging": -1, "hover": -1}
_POSE_GRAB_DIST = 15

def _pose_click_callback(event, x, y, flags, param):
    """Mouse callback for corner selection with 3D projection. Supports drag."""
    global _click_points, _click_world
    camera_matrix, rvec, tvec = param
    s = _pose_drag_state

    if event == cv2.EVENT_MOUSEMOVE:
        s["hover"] = -1
        for i, (px, py) in enumerate(_click_points):
            if abs(x - px) < _POSE_GRAB_DIST and abs(y - py) < _POSE_GRAB_DIST:
                s["hover"] = i
                break
        if s["dragging"] >= 0:
            # Update dragged point
            world = pixel_to_floor_3d(x, y, camera_matrix, rvec, tvec)
            if world is not None:
                _click_points[s["dragging"]] = (x, y)
                _click_world[s["dragging"]] = world

    elif event == cv2.EVENT_LBUTTONDOWN:
        # Check if grabbing existing point
        for i, (px, py) in enumerate(_click_points):
            if abs(x - px) < _POSE_GRAB_DIST and abs(y - py) < _POSE_GRAB_DIST:
                s["dragging"] = i
                return
        # New point
        world = pixel_to_floor_3d(x, y, camera_matrix, rvec, tvec)
        if world is None:
            print(f"  Click ({x}, {y}) -- above horizon, try clicking lower")
            return
        wx_mm, wy_mm = world
        wx_ft, wy_ft = wx_mm / MM_PER_UNIT, wy_mm / MM_PER_UNIT
        _click_points.append((x, y))
        _click_world.append((wx_mm, wy_mm))
        print(f"  Point {len(_click_points)}: pixel ({x}, {y}) -> "
              f"({wx_ft:.2f}, {wy_ft:.2f}) ft  ({wx_mm:.0f}, {wy_mm:.0f}) mm")

    elif event == cv2.EVENT_LBUTTONUP:
        s["dragging"] = -1


def run_single_marker_calibration(cam, camera_matrix, dist_coeffs):
    """Interactive calibration using a single ArUco marker + click.

    Phase 1: Detect calibration marker (ID 10) on the floor.
             Shows live feed with pose axes drawn on the marker.
             Press SPACE to lock the pose.

    Phase 2: Click 4 arena corners on the live feed.
             Each click shows real-world coordinates.
             Press ENTER to confirm, BACKSPACE to undo.

    Args:
        cam: camera object with .read() method
        camera_matrix: 3x3 intrinsics
        dist_coeffs: distortion coefficients

    Returns:
        A calibrated FloorPlaneDetector, or None if cancelled.
    """
    global _click_points, _click_world

    # Get resolution from camera (handle different camera APIs)
    if hasattr(cam, 'resolution'):
        rgb_size = cam.resolution
    else:
        frame = cam.read()
        rgb_size = (frame.shape[1], frame.shape[0]) if frame is not None else (1280, 720)
    rvec = None
    tvec = None
    pose_locked = False
    _click_points = []
    _click_world = []

    cv2.namedWindow(_CALIB_WINDOW)

    print("\n--- Floor Plane Calibration (single marker) ---")
    print(f"Place calibration marker (ID {CALIB_MARKER_ID}, {CALIB_MARKER_MM:.1f}mm) "
          f"flat on the arena floor.")
    print("Press SPACE when marker is detected to lock the camera pose.")
    print("Press ESC to cancel.\n")

    # Phase 1: Detect marker and lock pose
    while not pose_locked:
        frame = cam.read()
        if frame is None:
            continue

        vis = frame.copy()
        r, t, corners = estimate_marker_pose(
            frame, camera_matrix, dist_coeffs
        )

        if r is not None:
            # Draw marker outline
            pts = corners.astype(np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

            # Draw 3D axes on the marker
            cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, r, t, 100.0)

            # Compute camera height
            R, _ = cv2.Rodrigues(r)
            cam_pos = (-R.T @ t).flatten()
            height_mm = cam_pos[2]
            status = (f"Marker found! Height: {height_mm:.0f}mm "
                      f"({height_mm/MM_PER_UNIT:.1f}ft)  |  SPACE=lock pose")
        else:
            status = f"Looking for marker ID {CALIB_MARKER_ID}..."

        # HUD
        cv2.putText(vis, "PHASE 1: Detect calibration marker", (11, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        cv2.putText(vis, "PHASE 1: Detect calibration marker", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
        cv2.putText(vis, status, (11, 61),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(vis, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(_CALIB_WINDOW, vis)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            _destroy_calib_window()
            print("Calibration cancelled.")
            return None
        elif key == ord(" ") and r is not None:
            rvec = r.copy()
            tvec = t.copy()
            pose_locked = True
            print("Camera pose locked! Click arena corners (at least 4, drag to adjust).")
            print("You can remove the marker from the floor.\n")

    # Phase 2: Click 4 arena corners on live feed
    cv2.setMouseCallback(
        _CALIB_WINDOW, _pose_click_callback,
        param=(camera_matrix, rvec, tvec),
    )

    while True:
        frame = cam.read()
        if frame is None:
            continue

        vis = frame.copy()

        # Draw clicked corners with world coordinates
        for i, ((cx, cy), (wmx, wmy)) in enumerate(zip(_click_points, _click_world)):
            ft_x, ft_y = wmx / MM_PER_UNIT, wmy / MM_PER_UNIT
            color = (0, 255, 255)
            cv2.circle(vis, (cx, cy), 8, (0, 0, 0), 2)
            cv2.circle(vis, (cx, cy), 8, color, 1)
            cv2.circle(vis, (cx, cy), 3, color, -1)
            label = f"{i+1}: ({ft_x:.1f},{ft_y:.1f})cm"
            cv2.putText(vis, label, (cx + 12, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(vis, label, (cx + 12, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if i > 0:
                cv2.line(vis, _click_points[i-1], (cx, cy), (0, 200, 0), 2, cv2.LINE_AA)

        # Close polygon and fill
        n_pts = len(_click_points)
        if n_pts >= 3:
            cv2.line(vis, _click_points[-1], _click_points[0], (0, 200, 0), 2, cv2.LINE_AA)
            overlay = vis.copy()
            pts = np.array(_click_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 60, 0))
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

        # Highlight hovered point
        if _pose_drag_state["hover"] >= 0:
            hx, hy = _click_points[_pose_drag_state["hover"]]
            cv2.circle(vis, (hx, hy), 12, (0, 200, 255), 2)

        # HUD
        if n_pts < 4:
            instruction = f"Click point {n_pts+1} (need at least 4)"
        else:
            instruction = f"{n_pts} points | ENTER=confirm  Click=add more"

        cv2.putText(vis, "PHASE 2: Click arena corners (drag to adjust)", (11, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        cv2.putText(vis, "PHASE 2: Click arena corners (drag to adjust)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
        cv2.putText(vis, instruction, (11, 61),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(vis, instruction, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, "ESC=cancel  BACKSPACE=undo", (11, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(vis, "ESC=cancel  BACKSPACE=undo", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow(_CALIB_WINDOW, vis)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            _destroy_calib_window()
            print("Calibration cancelled.")
            return None
        elif key == 8 and _click_points:  # BACKSPACE
            removed_px = _click_points.pop()
            removed_w = _click_world.pop()
            print(f"  Undid corner at ({removed_px[0]}, {removed_px[1]})")
        elif key == 13 and len(_click_points) >= 4:  # ENTER
            break

    # Build initial detector from clicked corners
    detector = FloorPlaneDetector()
    if not detector.calibrate_from_pose(
        corners_px=_click_points,
        corners_mm=_click_world,
        rvec=rvec,
        tvec=tvec,
        camera_matrix=camera_matrix,
        rgb_size=rgb_size,
    ):
        _destroy_calib_window()
        return None

    # Phase 3: Verify & refine
    # Place marker at different spots on the arena. Green = accurate,
    # red = off. Press SPACE to add the marker's position as a correction
    # point, ENTER to finish.

    cv2.setMouseCallback(_CALIB_WINDOW, lambda *a: None)  # disable clicks

    GOOD_THRESHOLD_MM = 50.0  # within 50mm (~2") = green
    correction_points_px = list(_click_points)
    correction_points_mm = list(_click_world)

    print("\n--- Phase 3: Verify & Refine ---")
    print("Move the marker around the arena to check accuracy.")
    print("GREEN = good (<2\" error), RED = needs correction.")
    print("SPACE = add correction point, ENTER = done.\n")

    while True:
        frame = cam.read()
        if frame is None:
            continue

        vis = frame.copy()

        # Draw the current grid
        detector.draw_grid(vis)

        # Detect marker
        r, t, corners = estimate_marker_pose(
            frame, camera_matrix, dist_coeffs
        )

        error_mm = None
        marker_status = "Place marker on arena floor..."

        if r is not None:
            # Get the marker center in 3D (it's at the origin of its own frame)
            # But we need to use the ORIGINAL locked pose to project where
            # the marker center SHOULD be according to the calibration
            marker_center_px = np.mean(corners, axis=0).astype(int)
            mcx, mcy = int(marker_center_px[0]), int(marker_center_px[1])

            # Where the calibration THINKS this pixel is (cm)
            calib_world = detector.pixel_to_world(mcx, mcy)

            # Where the marker ACTUALLY is (from its own pose, in the
            # original coordinate system)
            actual_world = pixel_to_floor_3d(
                mcx, mcy, camera_matrix, rvec, tvec
            )

            if calib_world and actual_world:
                # Compare in mm
                actual_ft = (actual_world[0] / MM_PER_UNIT,
                             actual_world[1] / MM_PER_UNIT)
                dx = (calib_world[0] - actual_ft[0]) * MM_PER_UNIT
                dy = (calib_world[1] - actual_ft[1]) * MM_PER_UNIT
                error_mm = math.hypot(dx, dy)

                is_good = error_mm < GOOD_THRESHOLD_MM
                color = (0, 255, 0) if is_good else (0, 0, 255)

                # Draw marker outline
                pts = corners.astype(np.int32)
                cv2.polylines(vis, [pts], True, color, 3)

                # Draw crosshair at marker center
                cv2.drawMarker(vis, (mcx, mcy), color, cv2.MARKER_CROSS, 30, 2)

                # Error label
                err_label = (f"Error: {error_mm:.0f}mm ({error_mm/25.4:.1f}in)"
                             f"  {'OK' if is_good else 'SPACE to correct'}")
                cv2.putText(vis, err_label, (mcx + 20, mcy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                cv2.putText(vis, err_label, (mcx + 20, mcy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

                # Show actual vs calibrated position
                act_label = f"Actual: ({actual_ft[0]:.2f},{actual_ft[1]:.2f})cm"
                cal_label = f"Calib:  ({calib_world[0]:.2f},{calib_world[1]:.2f})cm"
                cv2.putText(vis, act_label, (mcx + 20, mcy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(vis, act_label, (mcx + 20, mcy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(vis, cal_label, (mcx + 20, mcy + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(vis, cal_label, (mcx + 20, mcy + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                marker_status = err_label

        # HUD
        n_corr = len(correction_points_px) - 4
        hud = f"PHASE 3: Verify & Refine  ({n_corr} correction points added)"
        cv2.putText(vis, hud, (11, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(vis, hud, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(vis, "SPACE=add correction  ENTER=done  ESC=cancel", (11, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(vis, "SPACE=add correction  ENTER=done  ESC=cancel", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow(_CALIB_WINDOW, vis)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            _destroy_calib_window()
            print("Calibration cancelled.")
            return None
        elif key == 13:  # ENTER - done verifying
            break
        elif key == ord(" ") and r is not None and error_mm is not None:
            # Add the marker center as a correction point
            mcx, mcy = int(np.mean(corners[:, 0])), int(np.mean(corners[:, 1]))
            actual = pixel_to_floor_3d(mcx, mcy, camera_matrix, rvec, tvec)
            if actual:
                correction_points_px.append((mcx, mcy))
                correction_points_mm.append(actual)
                print(f"  Added correction point: pixel ({mcx},{mcy}) -> "
                      f"({actual[0]/MM_PER_UNIT:.2f},{actual[1]/MM_PER_UNIT:.2f})ft  "
                      f"(error was {error_mm:.0f}mm)")

                # Recompute homography with all points (corners + corrections)
                all_px = [[float(p[0]), float(p[1])] for p in correction_points_px]
                all_ft = [[m[0]/MM_PER_UNIT, m[1]/MM_PER_UNIT] for m in correction_points_mm]
                px_arr = np.array(all_px, dtype=np.float32)
                ft_arr = np.array(all_ft, dtype=np.float32)

                H, _ = cv2.findHomography(px_arr, ft_arr)
                H_inv, _ = cv2.findHomography(ft_arr, px_arr)
                if H is not None and H_inv is not None:
                    detector.homography = H
                    detector.inv_homography = H_inv
                    detector._precompute_grid()
                    print(f"  Homography refined with {len(all_px)} points")

    _destroy_calib_window()

    # Save final refined calibration
    detector.save_calibration()
    return detector


# ======================================================================
# Simple click calibration (no marker, frozen frame, assumes rect arena)
# ======================================================================

_simple_click_corners = []


_drag_state = {
    "points": [],
    "dragging": -1,     # index of point being dragged, -1 = none
    "hover": -1,        # index of point under cursor
}
_DRAG_RADIUS = 12       # pixels — click within this to grab a point


def _corner_mouse_callback(event, x, y, flags, param):
    s = _drag_state
    grab_dist = _DRAG_RADIUS

    if event == cv2.EVENT_MOUSEMOVE:
        # Check hover
        s["hover"] = -1
        for i, (px, py) in enumerate(s["points"]):
            if abs(x - px) < grab_dist and abs(y - py) < grab_dist:
                s["hover"] = i
                break
        # Drag
        if s["dragging"] >= 0:
            s["points"][s["dragging"]] = (x, y)

    elif event == cv2.EVENT_LBUTTONDOWN:
        # Check if clicking on existing point to drag
        for i, (px, py) in enumerate(s["points"]):
            if abs(x - px) < grab_dist and abs(y - py) < grab_dist:
                s["dragging"] = i
                return
        # New point
        s["points"].append((x, y))
        print(f"  Point {len(s['points'])}: ({x}, {y})")

    elif event == cv2.EVENT_LBUTTONUP:
        s["dragging"] = -1


def run_corner_calibration(frame, rgb_size=None):
    """Click arena corners on a frozen frame. Supports dragging and unlimited points.

    Controls:
        Left click:  Place new point / grab existing point to drag
        Backspace:   Remove last point
        ENTER:       Confirm (need at least 4 points)
        ESC:         Cancel
    """
    if rgb_size is None:
        rgb_size = (frame.shape[1], frame.shape[0])

    _drag_state["points"] = []
    _drag_state["dragging"] = -1
    _drag_state["hover"] = -1
    frozen = frame.copy()

    cv2.namedWindow(_CALIB_WINDOW)
    cv2.setMouseCallback(_CALIB_WINDOW, _corner_mouse_callback)

    print("\n--- Floor Calibration ---")
    print("Click corners around the arena perimeter (at least 4).")
    print("Drag points to adjust. BACKSPACE=undo, ENTER=confirm, ESC=cancel.\n")

    while True:
        vis = frozen.copy()
        pts = _drag_state["points"]
        n = len(pts)

        # Draw polygon fill if >= 3 points
        if n >= 3:
            overlay = vis.copy()
            poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [poly], (0, 60, 0))
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

        # Draw edges
        for i in range(n):
            j = (i + 1) % n
            if j < n:
                cv2.line(vis, pts[i], pts[j], (0, 200, 0), 2)
        if n >= 3:
            cv2.line(vis, pts[-1], pts[0], (0, 200, 0), 2)

        # Draw points
        for i, (cx, cy) in enumerate(pts):
            color = (0, 200, 255) if i == _drag_state["hover"] else (0, 255, 255)
            radius = 10 if i == _drag_state["hover"] else 7
            cv2.circle(vis, (cx, cy), radius, color, -1)
            cv2.circle(vis, (cx, cy), radius, (0, 0, 0), 1)
            cv2.putText(vis, str(i+1), (cx+14, cy-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Status
        if n < 4:
            label = f"Click point {n+1} (need at least 4)"
        else:
            label = f"{n} points | ENTER=confirm  BACKSPACE=undo  Click=add more"
        cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        if _drag_state["dragging"] >= 0:
            cv2.putText(vis, "DRAGGING", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        cv2.imshow(_CALIB_WINDOW, vis)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            _destroy_calib_window()
            return None
        elif key == 8 and pts:  # backspace
            pts.pop()
            _drag_state["dragging"] = -1
        elif key == 13 and n >= 4:  # enter
            break

    _destroy_calib_window()

    detector = FloorPlaneDetector()
    if detector.calibrate_from_corners(pts, rgb_size=rgb_size):
        return detector
    return None


# ======================================================================
# Standalone entry point
# ======================================================================

if __name__ == "__main__":
    print("Floor Plane Calibration (standalone)")
    print("=" * 40)

    from capture import create_camera
    cam = create_camera(src=0, resolution_index=1).start()
    time.sleep(1.0)

    camera_matrix, dist_coeffs = cam.get_intrinsics()
    print(f"Camera intrinsics:\n{camera_matrix}")

    result = run_single_marker_calibration(cam, camera_matrix, dist_coeffs)
    if result is None:
        print("\nFalling back to manual click calibration...")
        frame = cam.read()
        if frame is not None:
            result = run_corner_calibration(frame, rgb_size=cam.resolution)

    if result is not None:
        print(f"\nCalibration successful! Saved to: {CALIBRATION_FILE}")
        for i, (px_c, ft_c) in enumerate(
            zip(result.floor_corners_px, result.floor_corners_ft)
        ):
            print(f"  Corner {i+1}: pixel ({px_c[0]:.0f}, {px_c[1]:.0f}) "
                  f"-> ({ft_c[0]:.1f}, {ft_c[1]:.1f}) ft")
    else:
        print("\nCalibration failed or cancelled.")

    cam.stop()
