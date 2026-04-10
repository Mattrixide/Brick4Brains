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
        self._was_pinned = False  # track if we were just pinned
        self._pin_escape_timer = 0  # frames of boost after pin release
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
        if hasattr(self, '_waypoints'):
            del self._waypoints
        if hasattr(self, '_wander_angle'):
            del self._wander_angle

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
        """Waypoint patrol + flee from Brick. Always moving, never gets stuck."""
        import random
        ex, ey = enemy.position
        bx, by = brick.position
        heading = enemy.heading_rad
        dist_to_brick = math.hypot(ex - bx, ey - by)
        cur_speed = math.hypot(*enemy.velocity)

        # === Pin detection + escape burst ===
        # Detect: we're pinned when near wall + near Brick + nearly stopped
        wall_dist, wall_away = _nearest_wall_info(ex, ey, self._corners)
        being_pinned = wall_dist < 20 and dist_to_brick < 30 and cur_speed < 5

        if being_pinned:
            self._was_pinned = True
        elif self._was_pinned and dist_to_brick > 25:
            # Just released from pin! Start escape burst
            self._was_pinned = False
            self._pin_escape_timer = 90  # 1.5 seconds of boost at 60fps

        # During escape burst: max speed away from wall, ignore waypoints
        if self._pin_escape_timer > 0:
            self._pin_escape_timer -= 1
            # Drive hard away from wall toward arena center
            corners = self._corners
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            center_x = (min(xs) + max(xs)) / 2
            center_y = (min(ys) + max(ys)) / 2
            escape_angle = math.atan2(center_y - ey, center_x - ex)
            diff = _wrap_to_pi(escape_angle - heading)
            steering = _clamp(diff * 3.0, -1, 1)
            if abs(diff) > math.pi / 2:
                return (-2.0, steering)  # reverse at 2x if facing wrong way
            return (2.0, steering)  # 2x forward boost toward center

        # === Generate safe waypoints on first call ===
        if not hasattr(self, '_waypoints') or not self._waypoints:
            self._generate_waypoints()
            self._wp_idx = 0

        # === Pick target: current waypoint or flee from Brick ===
        wp = self._waypoints[self._wp_idx]

        # If Brick is close, pick the waypoint farthest from Brick
        if dist_to_brick < 60:
            best_idx = max(range(len(self._waypoints)),
                          key=lambda i: math.hypot(self._waypoints[i][0] - bx,
                                                    self._waypoints[i][1] - by))
            wp = self._waypoints[best_idx]
            self._wp_idx = best_idx

        # Reached waypoint? Move to next (skip if path crosses pit)
        dist_to_wp = math.hypot(ex - wp[0], ey - wp[1])
        if dist_to_wp < 20:
            # Try next waypoints, skip any that require passing near pit
            for attempt in range(len(self._waypoints)):
                self._wp_idx = (self._wp_idx + 1) % len(self._waypoints)
                next_wp = self._waypoints[self._wp_idx]
                if self._pit:
                    # Check if midpoint of path is too close to pit
                    mid_x = (ex + next_wp[0]) / 2
                    mid_y = (ey + next_wp[1]) / 2
                    mid_pit = math.hypot(mid_x - self._pit[0], mid_y - self._pit[1])
                    if mid_pit < self._pit_danger + 50:
                        continue  # skip this waypoint
                break
            wp = self._waypoints[self._wp_idx]
            dist_to_wp = math.hypot(ex - wp[0], ey - wp[1])

        # === HARD PIT GEOFENCE (absolute highest priority) ===
        # Predict where we'll be in 0.5s and hard-brake if it's in the pit zone
        if self._pit:
            vx, vy = enemy.velocity
            speed = math.hypot(vx, vy)
            for lookahead in [0.15, 0.3, 0.5, 0.8, 1.2]:
                future_x = ex + vx * lookahead
                future_y = ey + vy * lookahead
                future_pit = math.hypot(future_x - self._pit[0], future_y - self._pit[1])
                if future_pit < self._pit_danger + 35:
                    # Will enter pit zone — hard brake + steer away
                    pit_dx = ex - self._pit[0]
                    pit_dy = ey - self._pit[1]
                    away = math.atan2(pit_dy, pit_dx)
                    diff = _wrap_to_pi(away - heading)
                    return (-1.5, _clamp(diff * 3.0, -1, 1))

        # === Wall-jam escape FIRST (highest priority) ===
        cur_speed = math.hypot(*enemy.velocity)
        wall_dist, wall_away = _nearest_wall_info(ex, ey, self._corners)
        if wall_dist < 25 and cur_speed < 10:
            away_diff = _wrap_to_pi(wall_away - heading)
            if abs(away_diff) > math.pi / 2:
                return (-1.5, _clamp(away_diff * 2.0, -1, 1))
            return (1.0, _clamp(away_diff * 2.0, -1, 1))

        # === Pit avoidance: override target if anywhere near pit ===
        if self._pit:
            pit_dx = ex - self._pit[0]
            pit_dy = ey - self._pit[1]
            pit_dist = math.hypot(pit_dx, pit_dy)
            # Check if heading TOWARD pit
            heading_to_pit = math.atan2(self._pit[1] - ey, self._pit[0] - ex)
            heading_toward = abs(_wrap_to_pi(heading - heading_to_pit)) < math.pi / 2

            danger_zone = self._pit_danger + 80  # very wide
            if pit_dist < danger_zone and heading_toward:
                # Heading toward pit — steer away, slow down
                away_angle = math.atan2(pit_dy, pit_dx)
                angle_diff = _wrap_to_pi(away_angle - heading)
                steering = _clamp(angle_diff * 3.0, -1, 1)
                if pit_dist < self._pit_danger + 15:
                    return (-1.5, steering)  # emergency reverse
                # Gradual slowdown
                frac = (pit_dist - self._pit_danger) / 60.0
                pit_throttle = 0.3 + 0.5 * frac
                return (pit_throttle, steering)
            elif pit_dist < self._pit_danger + 50:
                # Very close regardless of heading — always avoid
                away_angle = math.atan2(pit_dy, pit_dx)
                angle_diff = _wrap_to_pi(away_angle - heading)
                steering = _clamp(angle_diff * 3.0, -1, 1)
                return (-1.5, steering)

        # === Steer toward waypoint ===
        target_angle = math.atan2(wp[1] - ey, wp[0] - ex)
        angle_diff = _wrap_to_pi(target_angle - heading)
        steering = _clamp(angle_diff * 2.5, -1, 1)

        # === Speed ===
        SPEED_BOOST = 1.2
        if dist_to_brick < 40:
            throttle = 0.8 * SPEED_BOOST  # escape
        else:
            throttle = 0.5 * SPEED_BOOST  # cruise (manageable speed for wall avoidance)

        # Slow for sharp turns
        if abs(angle_diff) > 1.0:
            throttle *= 0.5
        throttle = max(0.4, throttle)

        return (throttle, steering)

    def _generate_waypoints(self):
        """Generate safe patrol waypoints inside the arena, avoiding pit."""
        # Compute arena center and safe interior points
        xs = [c[0] for c in self._corners]
        ys = [c[1] for c in self._corners]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        half_w = (max(xs) - min(xs)) / 2
        half_h = (max(ys) - min(ys)) / 2

        # Generate points at ~60% of the way from center to each corner
        # plus midpoints of edges, all pulled inward
        self._waypoints = []
        n = len(self._corners)
        for i in range(n):
            # Midpoint of each edge, pulled 40cm inward
            ax, ay = self._corners[i]
            bx, by = self._corners[(i + 1) % n]
            mx = (ax + bx) / 2
            my = (ay + by) / 2
            # Pull toward center
            dx, dy = cx - mx, cy - my
            d = math.hypot(dx, dy)
            if d > 0:
                mx += dx / d * 50  # 50cm inward from wall midpoint
                my += dy / d * 50

            # Skip if too close to pit
            if self._pit:
                if math.hypot(mx - self._pit[0], my - self._pit[1]) < self._pit_danger + 50:
                    continue
            self._waypoints.append((mx, my))

        # Add center
        if self._pit:
            if math.hypot(cx - self._pit[0], cy - self._pit[1]) > self._pit_danger + 50:
                self._waypoints.append((cx, cy))
        else:
            self._waypoints.append((cx, cy))

        # Add a few random interior points
        import random
        for _ in range(4):
            px = cx + random.uniform(-half_w * 0.5, half_w * 0.5)
            py = cy + random.uniform(-half_h * 0.5, half_h * 0.5)
            if self._pit:
                if math.hypot(px - self._pit[0], py - self._pit[1]) < self._pit_danger + 50:
                    continue
            self._waypoints.append((px, py))

        # Shuffle for unpredictability
        random.shuffle(self._waypoints)
        if not self._waypoints:
            self._waypoints = [(cx, cy)]  # fallback

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
