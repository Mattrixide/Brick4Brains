"""Enemy AI controller with switchable behavior modes."""
import json
import math
import os

from sim.arena import SimRobot, _load_json
from sim.config import SimConfig


MODES = ["manual", "sit", "circle", "charge", "flee", "ai"]


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def _point_to_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _nearest_wall_info(x, y, corners):
    """Returns (min_distance, wall_normal_angle) to nearest wall segment."""
    min_dist = float('inf')
    wall_nx, wall_ny = 0, 0
    n = len(corners)
    for i in range(n):
        ax, ay = corners[i]
        bx, by = corners[(i + 1) % n]
        d = _point_to_segment_dist(x, y, ax, ay, bx, by)
        if d < min_dist:
            min_dist = d
            # Normal pointing inward (away from wall)
            dx, dy = bx - ax, by - ay
            seg_len = math.hypot(dx, dy)
            if seg_len > 0:
                # Perpendicular to segment, pointing toward robot
                nx, ny = -dy / seg_len, dx / seg_len
                # Make sure normal points toward the robot
                mid_to_pt_x = x - (ax + bx) / 2
                mid_to_pt_y = y - (ay + by) / 2
                if nx * mid_to_pt_x + ny * mid_to_pt_y < 0:
                    nx, ny = -nx, -ny
                wall_nx, wall_ny = nx, ny
    return min_dist, math.atan2(wall_ny, wall_nx)


class EnemyController:
    """Switchable enemy AI with multiple behavior modes."""

    def __init__(self):
        self.mode = "manual"
        self._ai_bridge = None
        self._ai_match_started = False
        # Load arena corners for wall avoidance
        floor_cal = _load_json("floor_calibration.json")
        if floor_cal and "corners_ft" in floor_cal:
            self._corners = [tuple(c) for c in floor_cal["corners_ft"]]
        else:
            h = 122
            self._corners = [(-h, -h), (h, -h), (h, h), (-h, h)]
        # Load pit location for pit avoidance
        battle_cfg = _load_json("battle_config.json")
        if battle_cfg and "pit_x_cm" in battle_cfg:
            self._pit = (battle_cfg["pit_x_cm"], battle_cfg["pit_y_cm"])
            self._pit_danger = battle_cfg.get("pit_danger_radius_cm",
                                              battle_cfg.get("pit_radius_cm", 20) + 15)
        else:
            self._pit = None
            self._pit_danger = 0

    def set_mode(self, mode):
        if mode not in MODES:
            print(f"Unknown mode: {mode}")
            return
        if mode != self.mode:
            self.mode = mode
            self._ai_bridge = None
            self._ai_match_started = False
            print(f"Enemy mode: {mode.upper()}")

    def reset(self):
        """Reset state (call on arena reset)."""
        self._ai_bridge = None
        self._ai_match_started = False

    def get_drive(self, enemy: SimRobot, brick: SimRobot, dt: float, cfg: SimConfig):
        """Return (throttle, steering) or None if AI mode handles forces internally."""
        if self.mode == "manual":
            return (0, 0)
        elif self.mode == "sit":
            return (0, 0)
        elif self.mode == "circle":
            return (0.5, 0.3)
        elif self.mode == "charge":
            return self._charge(enemy, brick)
        elif self.mode == "flee":
            return self._flee(enemy, brick)
        elif self.mode == "ai":
            self._tick_ai(enemy, brick, dt, cfg)
            return None
        return (0, 0)

    def _charge(self, enemy: SimRobot, brick: SimRobot):
        ex, ey = enemy.position
        bx, by = brick.position
        target_angle = math.atan2(by - ey, bx - ex)
        angle_diff = _wrap_to_pi(target_angle - enemy.heading_rad)
        steering = _clamp(angle_diff * 2.0, -1, 1)
        return (0.8, steering)

    def _flee(self, enemy: SimRobot, brick: SimRobot):
        ex, ey = enemy.position
        bx, by = brick.position
        dist_to_brick = math.hypot(ex - bx, ey - by)

        # Primary: angle away from Brick
        flee_angle = math.atan2(ey - by, ex - bx)

        # Pit avoidance: strong repulsion from pit
        if self._pit:
            pit_dx = ex - self._pit[0]
            pit_dy = ey - self._pit[1]
            pit_dist = math.hypot(pit_dx, pit_dy)
            if pit_dist < self._pit_danger + 40:
                pit_away = math.atan2(pit_dy, pit_dx)
                pit_weight = 1.0 - max(0, (pit_dist - self._pit_danger) / 40.0)
                pit_weight = min(1.0, pit_weight)
                flee_dx = math.cos(flee_angle) * (1 - pit_weight) + math.cos(pit_away) * pit_weight
                flee_dy = math.sin(flee_angle) * (1 - pit_weight) + math.sin(pit_away) * pit_weight
                flee_angle = math.atan2(flee_dy, flee_dx)

        # Wall avoidance: when near a wall, turn to run ALONG the wall
        # instead of into it. Pick the tangent direction that's most away from Brick.
        wall_dist, wall_away_angle = _nearest_wall_info(ex, ey, self._corners)
        wall_threshold = 50.0

        if wall_dist < wall_threshold:
            wall_weight = (1.0 - wall_dist / wall_threshold) ** 0.5

            # Two tangent options: +90° and -90° from wall normal
            tang1 = wall_away_angle + math.pi / 2
            tang2 = wall_away_angle - math.pi / 2

            # Pick the tangent that leads more away from Brick
            score1 = math.cos(tang1 - flee_angle)
            score2 = math.cos(tang2 - flee_angle)
            tangent = tang1 if score1 > score2 else tang2

            # Near wall: blend between flee and tangent
            # Very near wall (< 20cm): mostly tangent + push away from wall
            if wall_dist < 20:
                # Almost touching wall — run along it + push away
                blend_dx = math.cos(tangent) * 0.7 + math.cos(wall_away_angle) * 0.3
                blend_dy = math.sin(tangent) * 0.7 + math.sin(wall_away_angle) * 0.3
            else:
                blend_dx = math.cos(flee_angle) * (1 - wall_weight) + math.cos(tangent) * wall_weight
                blend_dy = math.sin(flee_angle) * (1 - wall_weight) + math.sin(tangent) * wall_weight

            target_angle = math.atan2(blend_dy, blend_dx)
        else:
            target_angle = flee_angle

        angle_diff = _wrap_to_pi(target_angle - enemy.heading_rad)
        steering = _clamp(angle_diff * 2.5, -1, 1)

        # Trapped detection: close to wall AND close to Brick = pinned
        speed = math.hypot(*enemy.velocity)
        if wall_dist < 25 and dist_to_brick < 35 and speed < 5:
            # Pinned! Drive perpendicular to Brick (sidestep along wall)
            brick_angle = math.atan2(by - ey, bx - ex)
            perp1 = brick_angle + math.pi / 2
            perp2 = brick_angle - math.pi / 2
            s1 = abs(math.cos(perp1 - (wall_away_angle + math.pi / 2)))
            s2 = abs(math.cos(perp2 - (wall_away_angle + math.pi / 2)))
            escape_angle = perp1 if s1 > s2 else perp2
            angle_diff = _wrap_to_pi(escape_angle - enemy.heading_rad)
            steering = _clamp(angle_diff * 3.0, -1, 1)
            if abs(angle_diff) > math.pi / 2:
                return (-1.5, steering)  # reverse hard to escape
            return (1.5, steering)  # boost to break free

        # Speed: 1.5x boost so enemy can outrun Brick and force a chase
        SPEED_BOOST = 1.5
        if dist_to_brick < 30:
            throttle = 1.0 * SPEED_BOOST
        elif dist_to_brick < 80:
            throttle = 0.8 * SPEED_BOOST
        else:
            throttle = 0.6 * SPEED_BOOST

        # When very close and facing wrong way, reverse
        if dist_to_brick < 25 and abs(angle_diff) > math.pi * 0.6:
            return (-0.8, steering)

        # Slow for sharp turns but keep minimum speed
        if abs(angle_diff) > 1.0:
            throttle = max(0.3, throttle * 0.4)

        return (throttle, steering)

    def _tick_ai(self, enemy: SimRobot, brick: SimRobot, dt: float, cfg: SimConfig):
        """Lazily create a SimBridge for the enemy and tick it."""
        if self._ai_bridge is None:
            from sim.bridge import SimBridge
            self._ai_bridge = SimBridge(enemy, cfg, strategy_override="charge")
            self._ai_match_started = False

        if not self._ai_match_started:
            self._ai_bridge.start_match(brick)
            self._ai_match_started = True

        self._ai_bridge.tick(dt, brick)
