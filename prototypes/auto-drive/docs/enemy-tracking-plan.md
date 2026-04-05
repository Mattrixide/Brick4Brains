# Enemy Tracking Plan — Mono Camera Migration

Date: 2026-04-05

## Problem
Switched from color camera (IMX378, rolling shutter, 1280x720) to mono camera (OV9282, global shutter, 1280x800) for better marker tracking. Enemy detection broke because it relied on yellow color detection (HSV) to distinguish our robot from enemy.

## Approach 1: MOG2 + ArUco Exclusion (CURRENT PRIORITY)
- Keep MOG2 background subtraction — works great in grayscale (faster too, 1 channel vs 3)
- Remove all HSV/yellow color classification logic
- Exclude blobs by ArUco bounding box overlap instead of yellow fraction
- Our robot has ArUco markers → any blob overlapping ArUco corners is "us", everything else is enemy
- Global shutter benefit: sharper foreground masks, no motion blur on fast-moving objects
- Estimated effort: modify ~30 lines in enemy_tracker.py

## Approach 2: Stereo Depth Detection (NEXT)
- Use both OAK-D Pro mono cameras (CAM_B + CAM_C) for on-device stereo depth
- Arena floor is at known depth from ceiling camera (~150cm)
- Robots are 5-8cm tall → depth anomaly detection
- Threshold depth map: anything at "robot height" is a candidate
- Runs on OAK-D VPU, zero host CPU cost
- Combine with ArUco exclusion for robust enemy-only detection
- Consideration: using StereoDepth consumes both mono cameras, tracking frame comes from rectified left output

## Approach 3: Motion Confidence Layer (SUPPLEMENT)
- Frame differencing (frame_t - frame_{t-1}) as a supplementary signal
- Global shutter makes temporal differencing very clean (no rolling shutter smear)
- Adds "is this blob currently moving?" confidence to MOG2 detections
- Useful for target priority: pursue the moving enemy, not debris
- Very cheap: <1ms per frame

## Approach 4: YOLO Deep Learning (IF NEEDED)
- YOLOv8n with TensorRT: ~5-8ms inference on GPU
- Requires training data: hundreds of labeled overhead beetleweight images
- Generic "robot" detector, not opponent-specific
- Only pursue if MOG2+depth proves unreliable at competition
- High setup cost, diminishing returns vs simpler approaches

## Current Enemy Tracker Architecture
- MOG2 background subtraction (history=300, varThreshold=40, frozen learning rate)
- Morphological cleanup (open 5x5, close 15x15)
- Arena polygon mask (excludes outside-arena detections)
- Contour filtering (area 400-30000px, solidity >= 0.4)
- Kalman filter tracking (constant velocity, Mahalanobis gating 2.5)
- Track lock with stale reacquire (120 frames, 5px movement threshold)
