# OAK-D Pro Floor Plane Detection Research

**Date**: 2026-04-03
**Use case**: Detect the 8x8ft plywood arena floor from an OAK-D Pro on a tripod ~4ft high, looking down at the arena.

---

## Summary of Approaches (Ranked by Simplicity)

| Approach | Dependencies | Complexity | Reliability | Latency |
|----------|-------------|-----------|------------|---------|
| 1. Depth Thresholding | depthai, numpy, cv2 | ~40 lines | High (known setup) | <2ms |
| 2. SpatialLocationCalculator ROI Grid | depthai, numpy | ~60 lines | High (on-device) | <1ms |
| 3. PointCloud + Open3D RANSAC | depthai, open3d, numpy | ~80 lines | Very High | ~5-15ms |
| 4. Numpy-only RANSAC on depth map | depthai, numpy, scipy | ~70 lines | High | ~3-8ms |
| 5. On-device PointCloud node + Open3D | depthai, open3d | ~100 lines | Very High | ~10-20ms |

**Recommendation**: Start with Approach 1 (depth thresholding) for prototyping. It is dead simple and perfectly suited for a fixed camera setup. Graduate to Approach 3 (Open3D RANSAC) only if you need to handle camera vibration or need precise plane equation for 3D coordinate transforms.

---

## Approach 1: Depth Thresholding (Simplest, Recommended)

Since the camera is at a known height (~4ft / ~1220mm) looking down at a flat floor, the floor will appear as a band of consistent depth values. Simply threshold the depth map.

### Code

```python
#!/usr/bin/env python3
"""
OAK-D Pro floor detection via depth thresholding.
Camera at ~4ft (1220mm) looking down at 8x8ft arena floor.
"""
import cv2
import numpy as np
import depthai as dai

# Floor distance config (mm) - tune these for your setup
FLOOR_DEPTH_MIN = 1000   # 1.0m minimum (closest floor edge)
FLOOR_DEPTH_MAX = 2000   # 2.0m maximum (farthest floor edge)
# At 4ft height looking down at angle, depth to floor varies by viewing angle

pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
xoutDepth = pipeline.create(dai.node.XLinkOut)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

# Depth filters for cleaner output
config = stereo.initialConfig.get()
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.temporalFilter.enable = True
config.postProcessing.thresholdFilter.minRange = 400    # 0.4m
config.postProcessing.thresholdFilter.maxRange = 5000   # 5.0m
stereo.initialConfig.set(config)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)
xoutDepth.setStreamName("depth")

with dai.Device(pipeline) as device:
    # Enable IR laser dot projector for better depth on textureless surfaces
    device.setIrLaserDotProjectorBrightness(1200)

    depthQ = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        inDepth = depthQ.get()
        depthFrame = inDepth.getFrame()  # uint16, values in millimeters

        # Create floor mask: pixels within the expected depth range
        floor_mask = cv2.inRange(depthFrame.astype(np.uint16),
                                  FLOOR_DEPTH_MIN, FLOOR_DEPTH_MAX)

        # Clean up mask with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel)

        # Find the floor contour (should be the largest blob)
        contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            # This contour IS your arena floor boundary in pixel coords
            area = cv2.contourArea(largest)
            print(f"Floor area: {area} px, contour points: {len(largest)}")

        # Visualize
        depth_vis = cv2.normalize(depthFrame, None, 255, 0,
                                   cv2.NORM_INF, cv2.CV_8UC1)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        floor_vis = cv2.cvtColor(floor_mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([depth_vis, floor_vis])
        cv2.imshow("Depth | Floor Mask", combined)

        if cv2.waitKey(1) == ord('q'):
            break
```

### Key Notes
- **Depth values are in millimeters** (uint16 from StereoDepth node)
- `FLOOR_DEPTH_MIN` / `FLOOR_DEPTH_MAX` must be calibrated for your exact camera angle and height. Run the depth view first, hover over the floor in different spots, and note the depth range.
- **IR dot projector is critical** for plywood: `device.setIrLaserDotProjectorBrightness(1200)` -- without this, textureless surfaces produce noisy/missing depth. The OAK-D Pro has an IR laser dot projector specifically for this.
- The depth filters (spatial, temporal, threshold) clean up the raw depth significantly.
- Latency: essentially free (<2ms for the threshold + morphology).

### Gotchas
- If the camera is tilted (not straight down), depth to the floor varies across the frame -- the near edge is closer, the far edge is farther. You need a wider depth range or need to account for the gradient.
- Robots on the floor will have DIFFERENT depth (they stick up above the floor), so they will naturally be excluded from the floor mask. This is actually useful for segmentation.
- Polycarbonate walls may reflect IR and create depth artifacts at the edges.

---

## Approach 2: SpatialLocationCalculator Grid (On-Device)

Use the OAK-D's built-in `SpatialLocationCalculator` node to sample depth at a grid of ROIs across the frame. This runs on-device and gives you X/Y/Z coordinates in mm at each grid point. You can then determine which ROIs are "at floor level."

### Code

```python
#!/usr/bin/env python3
"""
Use SpatialLocationCalculator to sample a grid of depth points.
Points at floor depth = arena surface.
"""
import cv2
import numpy as np
import depthai as dai

GRID_SIZE = 8  # 8x8 grid of ROIs
FLOOR_Z_MIN = 1000  # mm
FLOOR_Z_MAX = 2000  # mm
ROI_SIZE = 1.0 / GRID_SIZE

pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialCalc = pipeline.create(dai.node.SpatialLocationCalculator)
xoutSpatial = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

# Create grid of ROIs
for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 100
        config.depthThresholds.upperThreshold = 10000
        config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
        topLeft = dai.Point2f(col * ROI_SIZE, row * ROI_SIZE)
        bottomRight = dai.Point2f((col + 1) * ROI_SIZE, (row + 1) * ROI_SIZE)
        config.roi = dai.Rect(topLeft, bottomRight)
        spatialCalc.initialConfig.addROI(config)

spatialCalc.inputConfig.setWaitForMessage(False)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(spatialCalc.inputDepth)
spatialCalc.out.link(xoutSpatial.input)
spatialCalc.passthroughDepth.link(xoutDepth.input)

xoutSpatial.setStreamName("spatialData")
xoutDepth.setStreamName("depth")

with dai.Device(pipeline) as device:
    device.setIrLaserDotProjectorBrightness(1200)

    spatialQ = device.getOutputQueue("spatialData", maxSize=4, blocking=False)
    depthQ = device.getOutputQueue("depth", maxSize=4, blocking=False)

    while True:
        spatialData = spatialQ.get().getSpatialLocations()
        inDepth = depthQ.get()
        depthFrame = inDepth.getFrame()

        depth_vis = cv2.normalize(depthFrame, None, 255, 0,
                                   cv2.NORM_INF, cv2.CV_8UC1)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        for i, data in enumerate(spatialData):
            roi = data.config.roi
            roi = roi.denormalize(depth_vis.shape[1], depth_vis.shape[0])
            x1, y1 = int(roi.topLeft().x), int(roi.topLeft().y)
            x2, y2 = int(roi.bottomRight().x), int(roi.bottomRight().y)

            z = data.spatialCoordinates.z  # depth in mm
            is_floor = FLOOR_Z_MIN <= z <= FLOOR_Z_MAX

            color = (0, 255, 0) if is_floor else (0, 0, 255)
            cv2.rectangle(depth_vis, (x1, y1), (x2, y2), color, 1)
            cv2.putText(depth_vis, f"{int(z)}",
                       (x1 + 2, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, color, 1)

        cv2.imshow("Floor Grid", depth_vis)
        if cv2.waitKey(1) == ord('q'):
            break
```

### Key Notes
- Runs spatial calculations on the VPU (on-device), very fast
- Gives you X/Y/Z in mm for each grid cell
- Green cells = floor, Red cells = not floor (wall, robot, out of range)
- Can dynamically reconfigure ROIs at runtime via `inputConfig`

---

## Approach 3: PointCloud + Open3D RANSAC Plane Fitting (Most Robust)

Use the OAK-D's built-in `PointCloud` node to generate a 3D pointcloud, then use Open3D's `segment_plane()` (RANSAC) to find the dominant plane. This is the most geometrically correct approach.

### Code

```python
#!/usr/bin/env python3
"""
OAK-D Pro pointcloud + Open3D RANSAC plane fitting.
Finds the dominant plane (floor) in the scene.
"""
import cv2
import numpy as np
import time
import depthai as dai

try:
    import open3d as o3d
except ImportError:
    import sys
    sys.exit("Install open3d: pip install open3d")

FPS = 30

pipeline = dai.Pipeline()

# Camera nodes
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
pointcloud = pipeline.create(dai.node.PointCloud)
sync = pipeline.create(dai.node.Sync)
xOut = pipeline.create(dai.node.XLinkOut)

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setIspScale(1, 3)  # 360p output for speed
camRgb.setFps(FPS)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoLeft.setFps(FPS)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")
monoRight.setFps(FPS)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(pointcloud.inputDepth)
camRgb.isp.link(sync.inputs["rgb"])
pointcloud.outputPointCloud.link(sync.inputs["pcl"])
sync.out.link(xOut.input)
xOut.setStreamName("out")
xOut.input.setBlocking(False)

with dai.Device(pipeline) as device:
    device.setIrLaserDotProjectorBrightness(1200)

    q = device.getOutputQueue(name="out", maxSize=4, blocking=False)

    while True:
        inMessage = q.get()
        if inMessage is None:
            continue

        inPointCloud = inMessage["pcl"]
        if inPointCloud is None:
            continue

        points = inPointCloud.getPoints().astype(np.float64)
        if len(points) < 100:
            continue

        # Build Open3D pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Optional: downsample for speed (voxel size in mm since OAK outputs mm)
        pcd_down = pcd.voxel_down_sample(voxel_size=10.0)  # 10mm voxels

        t0 = time.time()
        # RANSAC plane fitting
        # distance_threshold: max distance (mm) from plane to be an inlier
        # ransac_n: points to sample per iteration
        # num_iterations: RANSAC iterations
        plane_model, inliers = pcd_down.segment_plane(
            distance_threshold=15.0,   # 15mm tolerance
            ransac_n=3,
            num_iterations=200
        )
        dt = (time.time() - t0) * 1000

        [a, b, c, d] = plane_model
        print(f"Plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.1f} = 0  "
              f"({len(inliers)} inliers, {dt:.1f}ms)")

        # The plane normal (a, b, c) tells you the floor orientation
        # d / norm(a,b,c) gives distance from camera to plane
        normal = np.array([a, b, c])
        distance_to_plane = abs(d) / np.linalg.norm(normal)
        print(f"  Distance to floor: {distance_to_plane:.0f}mm")

        # Inlier points = floor, outlier points = robots/walls/objects
        floor_pcd = pcd_down.select_by_index(inliers)
        objects_pcd = pcd_down.select_by_index(inliers, invert=True)

        # Get RGB frame for display
        inColor = inMessage["rgb"]
        cvFrame = inColor.getCvFrame()
        cv2.imshow("RGB", cvFrame)

        if cv2.waitKey(1) == ord('q'):
            break
```

### Key Notes
- `segment_plane()` returns `[a, b, c, d]` plane equation and list of inlier indices
- **distance_threshold**: Set to ~15mm for a plywood floor (accounts for slight warping). In mm because OAK depth is in mm.
- **Voxel downsampling** is important: raw 400p depth gives ~150K points. Downsampling to 10mm voxels reduces to ~5K-15K points and makes RANSAC much faster.
- With downsampling + 200 iterations, plane fitting takes ~5-15ms.
- The plane normal vector tells you the floor orientation (useful for coordinate transforms).
- `open3d` is a large dependency (~100MB) but extremely well-tested for this.

### Gotchas
- Open3D's visualizer conflicts with OpenCV's `imshow` on some systems. If you only need the plane equation (not 3D visualization), skip the Open3D visualizer entirely.
- The `PointCloud` node was added in depthai 2.22+. Make sure you have a recent version.
- First frame may have garbage data; skip frames where `len(points) < 100`.

---

## Approach 4: Numpy-Only RANSAC on Depth Map (No Open3D)

If you want to avoid the Open3D dependency, you can do RANSAC plane fitting directly on the depth map using numpy. This converts pixel coordinates to 3D using the camera intrinsics, then fits a plane.

### Code

```python
#!/usr/bin/env python3
"""
Lightweight plane fitting on OAK-D depth map using numpy only.
No Open3D dependency.
"""
import cv2
import numpy as np
import depthai as dai
import time

def depth_to_3d_points(depth_frame, intrinsics, subsample=8):
    """Convert depth map to 3D points using camera intrinsics."""
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]

    h, w = depth_frame.shape
    # Subsample for speed
    ys, xs = np.mgrid[0:h:subsample, 0:w:subsample]
    zs = depth_frame[0:h:subsample, 0:w:subsample].astype(np.float64)

    # Filter out zero/invalid depth
    valid = zs > 0
    xs = xs[valid].flatten()
    ys = ys[valid].flatten()
    zs = zs[valid].flatten()

    # Unproject to 3D (in mm)
    x3d = (xs - cx) * zs / fx
    y3d = (ys - cy) * zs / fy
    z3d = zs

    return np.column_stack([x3d, y3d, z3d])


def ransac_plane_fit(points, threshold=15.0, iterations=200):
    """
    RANSAC plane fitting. Returns (plane_model, inlier_mask).
    plane_model = [a, b, c, d] where ax + by + cz + d = 0
    """
    best_inliers = None
    best_model = None
    best_count = 0
    n_points = len(points)

    for _ in range(iterations):
        # Sample 3 random points
        idx = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[idx]

        # Compute plane normal via cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal = normal / norm_len

        # Plane equation: normal . (p - p1) = 0
        # => a*x + b*y + c*z + d = 0 where d = -normal . p1
        d = -np.dot(normal, p1)

        # Distance from all points to plane
        distances = np.abs(points @ normal + d)

        # Count inliers
        inlier_mask = distances < threshold
        count = np.sum(inlier_mask)

        if count > best_count:
            best_count = count
            best_model = np.append(normal, d)
            best_inliers = inlier_mask

    return best_model, best_inliers


# Pipeline setup (same as Approach 1)
pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
xoutDepth = pipeline.create(dai.node.XLinkOut)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

config = stereo.initialConfig.get()
config.postProcessing.spatialFilter.enable = True
config.postProcessing.temporalFilter.enable = True
stereo.initialConfig.set(config)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)
xoutDepth.setStreamName("depth")

with dai.Device(pipeline) as device:
    device.setIrLaserDotProjectorBrightness(1200)

    depthQ = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    # Get camera intrinsics for 3D unprojection
    calibData = device.readCalibration()
    intrinsics = calibData.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_C,  # right mono camera
        dai.Size2f(640, 400)           # 400p resolution
    )

    while True:
        inDepth = depthQ.get()
        depthFrame = inDepth.getFrame()

        # Convert depth to 3D points (subsampled 8x for speed)
        points = depth_to_3d_points(depthFrame, intrinsics, subsample=8)

        if len(points) < 100:
            continue

        t0 = time.time()
        plane_model, inlier_mask = ransac_plane_fit(
            points, threshold=15.0, iterations=200
        )
        dt = (time.time() - t0) * 1000

        if plane_model is not None:
            a, b, c, d = plane_model
            n_inliers = np.sum(inlier_mask)
            distance = abs(d) / np.linalg.norm([a, b, c])
            print(f"Plane: [{a:.3f}, {b:.3f}, {c:.3f}, {d:.1f}]  "
                  f"inliers={n_inliers}/{len(points)}  "
                  f"dist={distance:.0f}mm  {dt:.1f}ms")

        # Visualize
        depth_vis = cv2.normalize(depthFrame, None, 255, 0,
                                   cv2.NORM_INF, cv2.CV_8UC1)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imshow("Depth", depth_vis)

        if cv2.waitKey(1) == ord('q'):
            break
```

### Key Notes
- Zero external dependencies beyond numpy (already needed for depthai)
- `subsample=8` reduces ~150K points to ~2-3K, making RANSAC very fast (~3-8ms)
- Camera intrinsics retrieved from device calibration data automatically
- Returns the same plane equation as Open3D for downstream use

---

## Approach 5: Calibration-Time Floor Capture (Simplest Production Path)

For a fixed camera setup like yours, the most bulletproof approach is a one-time calibration step: capture the empty arena, fit the plane once, then use it forever (or until the camera moves).

### Code

```python
#!/usr/bin/env python3
"""
One-time floor calibration: capture empty arena, fit plane, save.
Then at runtime, just threshold depth relative to the saved plane.
"""
import cv2
import numpy as np
import json
import depthai as dai

def calibrate_floor(device, depthQ, intrinsics, n_frames=30):
    """Capture n_frames of depth, average them, fit a plane."""
    print(f"Calibrating floor... capturing {n_frames} frames")
    depth_accum = None

    for i in range(n_frames):
        frame = depthQ.get().getFrame().astype(np.float64)
        if depth_accum is None:
            depth_accum = frame
        else:
            depth_accum += frame

    avg_depth = (depth_accum / n_frames)

    # Convert to 3D
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]
    h, w = avg_depth.shape
    ys, xs = np.mgrid[0:h:4, 0:w:4]
    zs = avg_depth[0:h:4, 0:w:4]
    valid = zs > 0
    xs_v = xs[valid].flatten()
    ys_v = ys[valid].flatten()
    zs_v = zs[valid].flatten()
    x3d = (xs_v - cx) * zs_v / fx
    y3d = (ys_v - cy) * zs_v / fy
    points = np.column_stack([x3d, y3d, zs_v])

    # Least-squares plane fit on averaged depth (no RANSAC needed for empty arena)
    # Solve: ax + by + c = z  =>  [x y 1] @ [a b c]^T = z
    A = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    result = np.linalg.lstsq(A, points[:, 2], rcond=None)
    a, b, c = result[0]

    # Convert to standard form: ax + by - z + c = 0
    plane = {"a": float(a), "b": float(b), "c": float(c),
             "avg_depth_min": float(np.percentile(zs_v, 5)),
             "avg_depth_max": float(np.percentile(zs_v, 95))}

    with open("floor_calibration.json", "w") as f:
        json.dump(plane, f, indent=2)

    print(f"Floor calibration saved!")
    print(f"  Depth range: {plane['avg_depth_min']:.0f} - {plane['avg_depth_max']:.0f} mm")
    return plane


def load_floor_calibration():
    with open("floor_calibration.json") as f:
        return json.load(f)

# Usage at runtime:
# plane = load_floor_calibration()
# floor_mask = cv2.inRange(depthFrame, plane["avg_depth_min"] - 50,
#                           plane["avg_depth_max"] + 50)
```

---

## Critical OAK-D Pro Notes

### IR Dot Projector
The OAK-D Pro has an active IR stereo system with a dot projector. This is **essential** for your use case:
```python
device.setIrLaserDotProjectorBrightness(1200)  # 0-1200 mA
```
Without it, the plain plywood floor will produce very poor depth data because stereo matching needs texture. The dot projector adds invisible IR texture to featureless surfaces. **Always enable this for floor detection.**

### IR Flood Light
The OAK-D Pro also has an IR flood illuminator for low-light scenarios:
```python
device.setIrFloodLightBrightness(500)  # 0-1500 mA
```
Useful if arena lighting is dim, but the dot projector is more important for depth quality.

### Depth Output Format
- StereoDepth node outputs `uint16` depth in **millimeters**
- 0 = invalid/no data
- Maximum range depends on baseline and resolution (~20m theoretical, ~10m practical)
- At 400p with HIGH_DENSITY preset, expect good depth from ~0.4m to ~5m

### Recommended Depth Filters
For floor detection, enable these post-processing filters:
```python
config = stereo.initialConfig.get()
config.postProcessing.spatialFilter.enable = True        # Smooths depth spatially
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.temporalFilter.enable = True       # Smooths depth over time
config.postProcessing.thresholdFilter.minRange = 400     # Min 0.4m
config.postProcessing.thresholdFilter.maxRange = 3000    # Max 3.0m
config.postProcessing.speckleFilter.enable = True        # Removes small noise
config.postProcessing.speckleFilter.speckleRange = 50
stereo.initialConfig.set(config)
```

### PointCloud Node
- Available in depthai 2.22+
- Takes depth input, outputs `dai.PointCloudData` with `.getPoints()` returning Nx3 numpy array (in mm)
- Must be synced with RGB via `Sync` node if you want colored pointcloud
- Runs on VPU, minimal host CPU overhead

---

## Official Luxonis References

| Resource | URL | Notes |
|----------|-----|-------|
| PointCloud visualize example | `depthai-python/examples/PointCloud/visualize_pointcloud.py` | Official, uses Open3D viz |
| RGBD pointcloud experiment | `depthai-experiments/gen2-pointcloud/rgbd-pointcloud/` | Host-side Open3D projection |
| Spatial Location Calculator | `depthai-python/examples/SpatialDetection/spatial_location_calculator.py` | On-device depth ROI sampling |
| Stereo depth video | `depthai-python/examples/StereoDepth/stereo_depth_video.py` | Full stereo config options |
| Depth filters | StereoDepth `initialConfig.get().postProcessing` | Spatial, temporal, threshold, speckle, decimation |

GitHub repos:
- https://github.com/luxonis/depthai-python (main SDK examples)
- https://github.com/luxonis/depthai-experiments (community experiments)
- Latest SDK: `depthai 3.5.0` (March 2026), requires Python >= 3.9

---

## Recommendation for B4B

**For prototyping (now):** Use Approach 1 (depth thresholding). Set up the OAK-D Pro on the tripod, enable the IR dot projector, view the raw depth, note the floor depth range, and threshold. This gives you a floor mask in <2ms with zero complexity. Combine with your existing ArUco tracking pipeline.

**For production:** Use Approach 5 (calibration capture) + Approach 1. One-time calibration captures the exact floor depth range for your setup. Then at runtime, a simple threshold gives you the floor mask. If you later need the actual 3D plane equation (e.g., for projecting robot positions onto the floor plane), upgrade to Approach 3 or 4.

**What NOT to do:** Don't start with the full pointcloud + Open3D pipeline unless you specifically need 3D coordinates. It adds ~100MB of dependencies and 5-15ms of latency for something that depth thresholding does in <2ms for a fixed camera setup.
