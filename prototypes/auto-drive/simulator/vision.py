"""Vision simulation — converts ground-truth physics to noisy BattleContext."""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

# Import will work once __init__.py sets up sys.path
from state_machine import BattleContext


@dataclass
class VisionConfig:
    detection_range_cm: float = 180.0
    detection_steepness: float = 0.05
    base_drop_rate: float = 0.07
    position_noise_std_cm: float = 2.0
    heading_noise_std_rad: float = 0.03
    latency_frames: int = 2
    aruco_edge_margin_cm: float = 15.0
    tracking_memory_frames: int = 10
    seed: int | None = None


class VisionSimulator:
    """Simulates noisy camera + ArUco detection for both robots."""

    def __init__(self, config: VisionConfig | None = None,
                 arena_half_w: float = 122.0, arena_half_h: float = 122.0):
        self.cfg = config or VisionConfig()
        self.arena_half_w = arena_half_w
        self.arena_half_h = arena_half_h
        self._rng = np.random.default_rng(self.cfg.seed)

        # Enemy tracking state
        self._frames_without_detection = 999
        self._last_detected_pos: tuple[float, float] | None = None
        self._prev_detected_pos: tuple[float, float] | None = None
        self._last_detected_heading: float | None = None
        self._detection_buffer: deque = deque(maxlen=max(1, self.cfg.latency_frames + 1))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._frames_without_detection = 999
        self._last_detected_pos = None
        self._prev_detected_pos = None
        self._last_detected_heading = None
        self._detection_buffer.clear()

    def _detection_probability(self, distance_cm: float) -> float:
        """Sigmoid-based detection probability that drops with distance."""
        x = -(distance_cm - self.cfg.detection_range_cm) * self.cfg.detection_steepness
        sigmoid = 1.0 / (1.0 + math.exp(-x))
        return sigmoid * (1.0 - self.cfg.base_drop_rate)

    def _is_near_wall(self, pos: np.ndarray) -> bool:
        """Check if position is near arena edge (ArUco detection fails)."""
        margin = self.cfg.aruco_edge_margin_cm
        return (abs(pos[0]) > self.arena_half_w - margin or
                abs(pos[1]) > self.arena_half_h - margin)

    def generate_context(self, our_body, enemy_body, dt: float) -> BattleContext:
        """Generate a BattleContext from ground-truth physics state."""
        from simulator.physics import RobotBody  # avoid circular at module level

        # --- Our ArUco detection ---
        our_detected = not self._is_near_wall(our_body.pos)
        our_pos = (float(our_body.pos[0]), float(our_body.pos[1]))
        our_heading = our_body.heading
        our_vel = (float(our_body.vel[0]), float(our_body.vel[1]))

        # --- Enemy detection ---
        true_dist = float(np.linalg.norm(enemy_body.pos - our_body.pos))

        # Buffer the ground-truth detection (for latency)
        self._detection_buffer.append((
            float(enemy_body.pos[0]), float(enemy_body.pos[1]),
            enemy_body.heading, true_dist,
        ))

        # Pull from latency buffer
        if len(self._detection_buffer) > self.cfg.latency_frames:
            buffered = self._detection_buffer[0]
        else:
            buffered = self._detection_buffer[-1]

        buf_x, buf_y, buf_heading, buf_dist = buffered

        # Roll for detection
        p = self._detection_probability(buf_dist)
        enemy_detected = bool(self._rng.random() < p)

        enemy_pos = None
        enemy_heading = None
        enemy_vel = None
        enemy_tracking = False

        if enemy_detected:
            self._frames_without_detection = 0

            # Add noise
            noisy_x = buf_x + self._rng.normal(0, self.cfg.position_noise_std_cm)
            noisy_y = buf_y + self._rng.normal(0, self.cfg.position_noise_std_cm)
            noisy_heading = buf_heading + self._rng.normal(0, self.cfg.heading_noise_std_rad)

            enemy_pos = (noisy_x, noisy_y)

            # Velocity and heading from finite difference (mimics real tracker)
            # Enemy heading is only reliable when the enemy is actually moving
            if self._last_detected_pos is not None:
                vx = (noisy_x - self._last_detected_pos[0]) / max(dt, 0.001)
                vy = (noisy_y - self._last_detected_pos[1]) / max(dt, 0.001)
                enemy_vel = (vx, vy)
                speed = math.hypot(vx, vy)
                if speed > 3.0:  # only trust heading when moving > 3 cm/s
                    enemy_heading = math.atan2(vy, vx)
                else:
                    enemy_heading = None  # heading unreliable for slow/stationary
            else:
                enemy_heading = None

            self._prev_detected_pos = self._last_detected_pos
            self._last_detected_pos = enemy_pos
            self._last_detected_heading = noisy_heading
            enemy_tracking = True
        else:
            self._frames_without_detection += 1

            # Still tracking if detected recently
            if self._frames_without_detection < self.cfg.tracking_memory_frames:
                enemy_tracking = True
                enemy_pos = self._last_detected_pos
                enemy_heading = self._last_detected_heading

        # Distance
        if enemy_pos is not None and our_detected:
            distance_cm = math.hypot(
                enemy_pos[0] - our_pos[0],
                enemy_pos[1] - our_pos[1],
            )
        else:
            distance_cm = 999.0

        return BattleContext(
            our_pos=our_pos,
            our_heading_rad=our_heading,
            our_velocity=our_vel,
            enemy_pos=enemy_pos,
            enemy_heading_rad=enemy_heading,
            enemy_velocity=enemy_vel,
            enemy_detected=enemy_detected,
            enemy_tracking=enemy_tracking,
            frames_without_detection=self._frames_without_detection,
            distance_cm=distance_cm,
            dt=dt,
            our_detected=our_detected,
        )
