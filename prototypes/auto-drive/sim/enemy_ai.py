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


def _ray_segment_intersect(rx, ry, rex, rey, ax, ay, bx, by):
    """Ray from (rx,ry)->(rex,rey) vs segment (ax,ay)-(bx,by).
    Returns (hit_x, hit_y, t) where t is fraction along ray, or None."""
    dx, dy = rex - rx, rey - ry
    sx, sy = bx - ax, by - ay
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-10:
        return None
    t = ((ax - rx) * sy - (ay - ry) * sx) / denom
    u = ((ax - rx) * dy - (ay - ry) * dx) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (rx + t * dx, ry + t * dy, t)
    return None


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
        """Reynolds steering: wall whiskers + pit repulsion + flee + wander."""
        ex, ey = enemy.position
        bx, by = brick.position
        heading = enemy.heading_rad
        dist_to_brick = math.hypot(ex - bx, ey - by)

        # === Steering forces (dx, dy vectors) ===
        fx, fy = 0.0, 0.0  # accumulated force

        # 1. WALL AVOIDANCE (whiskers) — highest priority
        # Cast 3 rays: forward, +30deg, -30deg
        whisker_len = 60.0  # lookahead distance
        for angle_offset in [0, 0.5, -0.5]:  # ~0, +30deg, -30deg
            ray_angle = heading + angle_offset
            ray_ex = ex + math.cos(ray_angle) * whisker_len
            ray_ey = ey + math.sin(ray_angle) * whisker_len
            # Test against each wall segment
            for j in range(len(self._corners)):
                hit = _ray_segment_intersect(
                    ex, ey, ray_ex, ray_ey,
                    self._corners[j][0], self._corners[j][1],
                    self._corners[(j+1) % len(self._corners)][0],
                    self._corners[(j+1) % len(self._corners)][1],
                )
                if hit is not None:
                    hx, hy, t = hit
                    dist = t * whisker_len
                    if dist < whisker_len:
                        # Push away from wall, stronger when closer
                        strength = 5.0 * (1.0 - dist / whisker_len) ** 2
                        # Wall normal (from hit point back toward robot)
                        nx = ex - hx
                        ny = ey - hy
                        nl = math.hypot(nx, ny)
                        if nl > 0.01:
                            fx += (nx / nl) * strength
                            fy += (ny / nl) * strength

        # 2. PIT AVOIDANCE — strong repulsion
        if self._pit:
            pit_dx = ex - self._pit[0]
            pit_dy = ey - self._pit[1]
            pit_dist = math.hypot(pit_dx, pit_dy)
            danger = self._pit_danger + 30
            if pit_dist < danger and pit_dist > 0.1:
                strength = 4.0 * ((danger - pit_dist) / danger) ** 2
                fx += (pit_dx / pit_dist) * strength
                fy += (pit_dy / pit_dist) * strength

        # 3. FLEE from Brick — scaled by distance
        if dist_to_brick > 0.1:
            flee_dx = (ex - bx) / dist_to_brick
            flee_dy = (ey - by) / dist_to_brick
            if dist_to_brick < 40:
                flee_strength = 3.0
            elif dist_to_brick < 100:
                flee_strength = 1.5
            else:
                flee_strength = 0.5
            fx += flee_dx * flee_strength
            fy += flee_dy * flee_strength

        # 4. WANDER — Reynolds wander: jitter a target on a circle ahead
        import random
        if not hasattr(self, '_wander_angle'):
            self._wander_angle = heading
        self._wander_angle += random.gauss(0, 0.3)  # smooth random jitter
        # Wander is the primary movement force — keeps robot moving always
        wander_strength = 2.0 if dist_to_brick > 80 else 1.0
        fx += math.cos(self._wander_angle) * wander_strength
        fy += math.sin(self._wander_angle) * wander_strength

        # === Convert force to throttle/steering ===
        target_angle = math.atan2(fy, fx)
        angle_diff = _wrap_to_pi(target_angle - heading)
        steering = _clamp(angle_diff * 2.5, -1, 1)

        # Wall-jam escape: if stopped near a wall, reverse to pull free
        cur_speed = math.hypot(*enemy.velocity)
        wall_dist, _ = _nearest_wall_info(ex, ey, self._corners)
        if cur_speed < 3 and wall_dist < 25:
            # Jammed against wall — reverse hard to pull away, then can turn
            return (-1.0 * 1.5, steering)

        # Speed: 1.5x boost, always moving (minimum 0.4)
        SPEED_BOOST = 1.5
        throttle = 0.7 * SPEED_BOOST
        if dist_to_brick < 40:
            throttle = 1.0 * SPEED_BOOST
        if abs(angle_diff) > 1.2:
            throttle *= 0.5
        throttle = max(0.4, throttle)

        # Reverse when facing wrong way and Brick is close
        if dist_to_brick < 25 and abs(angle_diff) > math.pi * 0.6:
            return (-1.0, steering)

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
