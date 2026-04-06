"""2D physics engine for combat robot arena simulation.

Robots are oriented rectangles with tank drive (differential left/right tracks).
"""

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PhysicsConfig:
    max_speed_cm_s: float = 75.0      # half speed (~30 in/s)
    max_turn_rate_rad_s: float = 4.0
    max_accel_cm_s2: float = 250.0
    floor_friction: float = 3.0
    collision_restitution: float = 0.3
    wall_restitution: float = 0.1
    robot_mass_kg: float = 1.36       # 3 lbs
    # Robot dimensions (beetleweight)
    robot_length_cm: float = 15.2     # 6" front-to-back
    robot_width_cm: float = 25.4      # 10" side-to-side
    track_width_cm: float = 22.0      # distance between left/right tracks


@dataclass
class RobotBody:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heading: float = 0.0
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angular_vel: float = 0.0
    mass: float = 1.36
    length: float = 15.2   # front-to-back
    width: float = 25.4    # side-to-side
    speed_mult: float = 1.0   # multiplier on max speed
    accel_mult: float = 1.0   # multiplier on max acceleration/force

    @property
    def radius(self) -> float:
        """Bounding circle radius for broad-phase collision."""
        return math.hypot(self.length, self.width) / 2

    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    def corners(self) -> np.ndarray:
        """Get the 4 corners of the robot rectangle in world coords.

        Returns shape (4, 2) array: front-left, front-right, back-right, back-left.
        """
        c = math.cos(self.heading)
        s = math.sin(self.heading)
        # Forward and right vectors
        fwd = np.array([c, s])
        right = np.array([s, -c])  # perpendicular, rightward
        hl = self.length / 2  # half-length (front-back)
        hw = self.width / 2   # half-width (side-side)
        return np.array([
            self.pos + fwd * hl - right * hw,  # front-left
            self.pos + fwd * hl + right * hw,  # front-right
            self.pos - fwd * hl + right * hw,  # back-right
            self.pos - fwd * hl - right * hw,  # back-left
        ])

    def copy(self) -> "RobotBody":
        return RobotBody(
            pos=self.pos.copy(), heading=self.heading,
            vel=self.vel.copy(), angular_vel=self.angular_vel,
            mass=self.mass, length=self.length, width=self.width,
        )


@dataclass
class Arena:
    half_w: float = 122.0
    half_h: float = 122.0
    # Square pit: defined by min/max corners
    has_pit: bool = False
    pit_min: tuple[float, float] = (0.0, 0.0)
    pit_max: tuple[float, float] = (0.0, 0.0)
    pit_lip_cm: float = 1.9

    # Legacy circle fields (for compat with battle_config)
    pit_center: tuple[float, float] = (0.0, 0.0)
    pit_radius: float = 20.0

    @staticmethod
    def with_corner_pit(corner: str = "upper_right",
                        size_cm: float = 45.7,
                        inset_cm: float = 7.6,
                        lip_cm: float = 1.9,
                        half_w: float = 122.0,
                        half_h: float = 122.0) -> "Arena":
        """Create arena with a square pit inset from a corner."""
        if corner == "upper_right":
            x_max = half_w - inset_cm
            y_max = half_h - inset_cm
            x_min = x_max - size_cm
            y_min = y_max - size_cm
        elif corner == "upper_left":
            x_min = -half_w + inset_cm
            y_max = half_h - inset_cm
            x_max = x_min + size_cm
            y_min = y_max - size_cm
        elif corner == "lower_right":
            x_max = half_w - inset_cm
            y_min = -half_h + inset_cm
            x_min = x_max - size_cm
            y_max = y_min + size_cm
        elif corner == "lower_left":
            x_min = -half_w + inset_cm
            y_min = -half_h + inset_cm
            x_max = x_min + size_cm
            y_max = y_min + size_cm
        else:
            raise ValueError(f"Unknown corner: {corner}")

        center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        return Arena(
            half_w=half_w, half_h=half_h,
            has_pit=True,
            pit_min=(x_min, y_min),
            pit_max=(x_max, y_max),
            pit_lip_cm=lip_cm,
            pit_center=center,
            pit_radius=size_cm / 2,
        )


@dataclass
class StepResult:
    a_in_pit: bool = False
    b_in_pit: bool = False
    collision: bool = False


# ---------------------------------------------------------------------------
# Tank drive model
# ---------------------------------------------------------------------------

def apply_tank_drive(body: RobotBody, throttle: float, steering: float,
                     cfg: PhysicsConfig, dt: float) -> None:
    """Tank drive: throttle = forward power, steering = differential.

    Maps throttle/steering to left/right track speeds:
      left_track  = throttle + steering
      right_track = throttle - steering

    This gives:
      - throttle only → straight line
      - steering only → spin in place
      - both → arc turn
    """
    throttle = max(-1.0, min(1.0, throttle))
    steering = max(-1.0, min(1.0, steering))

    left = throttle + steering
    right = throttle - steering

    # Clamp individual tracks
    left = max(-1.0, min(1.0, left))
    right = max(-1.0, min(1.0, right))

    # Per-robot speed and force scaling
    max_speed = cfg.max_speed_cm_s * body.speed_mult
    max_accel = cfg.max_accel_cm_s2 * body.accel_mult

    # Track speeds → linear and angular velocity
    track_width = cfg.track_width_cm
    v_linear = (left + right) / 2.0 * max_speed
    omega = (left - right) / track_width * max_speed

    # Clamp turn rate
    omega = max(-cfg.max_turn_rate_rad_s, min(cfg.max_turn_rate_rad_s, omega))

    # Update heading
    body.heading += omega * dt
    body.heading = math.atan2(math.sin(body.heading), math.cos(body.heading))
    body.angular_vel = omega

    # Forward acceleration in heading direction
    forward = np.array([math.cos(body.heading), math.sin(body.heading)])
    target_vel = forward * v_linear

    # Accelerate toward target velocity (not instant)
    diff = target_vel - body.vel
    max_dv = max_accel * dt
    dv_mag = float(np.linalg.norm(diff))
    if dv_mag > max_dv:
        diff = diff * (max_dv / dv_mag)
    body.vel += diff

    # Floor friction
    friction_factor = max(0.0, 1.0 - cfg.floor_friction * dt)
    body.vel *= friction_factor

    # Clamp speed
    speed = float(np.linalg.norm(body.vel))
    if speed > max_speed:
        body.vel *= max_speed / speed

    # Integrate position
    body.pos += body.vel * dt


# ---------------------------------------------------------------------------
# Collision detection (rectangle approximated by bounding circle + SAT)
# ---------------------------------------------------------------------------

def _project_corners(corners: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    """Project polygon corners onto axis, return (min, max)."""
    dots = corners @ axis
    return float(dots.min()), float(dots.max())


def _sat_overlap(corners_a: np.ndarray, corners_b: np.ndarray) -> tuple[bool, float, np.ndarray]:
    """Separating Axis Theorem for two convex polygons.

    Returns (overlapping, min_overlap, push_axis).
    """
    min_overlap = float('inf')
    push_axis = np.zeros(2)

    for corners in [corners_a, corners_b]:
        n = len(corners)
        for i in range(n):
            edge = corners[(i + 1) % n] - corners[i]
            axis = np.array([-edge[1], edge[0]])  # perpendicular
            length = np.linalg.norm(axis)
            if length < 1e-8:
                continue
            axis = axis / length

            min_a, max_a = _project_corners(corners_a, axis)
            min_b, max_b = _project_corners(corners_b, axis)

            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap <= 0:
                return False, 0.0, np.zeros(2)

            if overlap < min_overlap:
                min_overlap = overlap
                push_axis = axis

    # Ensure push direction is from A to B
    center_diff = corners_b.mean(axis=0) - corners_a.mean(axis=0)
    if np.dot(push_axis, center_diff) < 0:
        push_axis = -push_axis

    return True, float(min_overlap), push_axis


def resolve_collision(a: RobotBody, b: RobotBody, cfg: PhysicsConfig) -> bool:
    """Resolve rectangle-rectangle collision using SAT."""
    # Broad phase: bounding circle check
    dist = float(np.linalg.norm(b.pos - a.pos))
    if dist > a.radius + b.radius:
        return False

    # Narrow phase: SAT
    corners_a = a.corners()
    corners_b = b.corners()
    overlapping, overlap, axis = _sat_overlap(corners_a, corners_b)

    if not overlapping:
        return False

    # Separate bodies along push axis
    total_mass = a.mass + b.mass
    a.pos -= axis * overlap * (b.mass / total_mass)
    b.pos += axis * overlap * (a.mass / total_mass)

    # Impulse-based velocity exchange
    rel_vel = b.vel - a.vel
    vel_along_normal = float(np.dot(rel_vel, axis))

    if vel_along_normal > 0:
        return True  # separating

    e = cfg.collision_restitution
    j = -(1 + e) * vel_along_normal / (1.0 / a.mass + 1.0 / b.mass)

    impulse = axis * j
    a.vel -= impulse / a.mass
    b.vel += impulse / b.mass

    return True


# ---------------------------------------------------------------------------
# Wall collision (uses bounding circle for simplicity)
# ---------------------------------------------------------------------------

def resolve_walls(body: RobotBody, arena: Arena, cfg: PhysicsConfig) -> None:
    """Clamp robot to arena bounds using bounding circle."""
    r = body.radius
    min_x, max_x = -arena.half_w + r, arena.half_w - r
    min_y, max_y = -arena.half_h + r, arena.half_h - r

    if body.pos[0] < min_x:
        body.pos[0] = min_x
        body.vel[0] *= -cfg.wall_restitution
    elif body.pos[0] > max_x:
        body.pos[0] = max_x
        body.vel[0] *= -cfg.wall_restitution

    if body.pos[1] < min_y:
        body.pos[1] = min_y
        body.vel[1] *= -cfg.wall_restitution
    elif body.pos[1] > max_y:
        body.pos[1] = max_y
        body.vel[1] *= -cfg.wall_restitution


# ---------------------------------------------------------------------------
# Pit detection (square pit with lip)
# ---------------------------------------------------------------------------

def _point_in_rect(x: float, y: float,
                    x_min: float, y_min: float,
                    x_max: float, y_max: float) -> bool:
    return x_min <= x <= x_max and y_min <= y <= y_max


def _dist_to_rect_edge(x: float, y: float,
                        x_min: float, y_min: float,
                        x_max: float, y_max: float) -> float:
    """Distance from point to nearest edge of rectangle.

    Positive = outside, negative (magnitude) = depth inside.
    """
    dx = max(x_min - x, 0, x - x_max)
    dy = max(y_min - y, 0, y - y_max)
    if dx > 0 or dy > 0:
        return math.hypot(dx, dy)
    return -min(x - x_min, x_max - x, y - y_min, y_max - y)


def apply_pit_lip(body: RobotBody, arena: Arena, cfg: PhysicsConfig) -> None:
    """Apply lip resistance force at the square pit edge."""
    if not arena.has_pit:
        return

    lip = arena.pit_lip_cm
    x, y = body.pos[0], body.pos[1]
    x_min, y_min = arena.pit_min
    x_max, y_max = arena.pit_max

    outer_min = (x_min - lip, y_min - lip)
    outer_max = (x_max + lip, y_max + lip)

    in_outer = _point_in_rect(x, y, *outer_min, *outer_max)
    in_pit = _point_in_rect(x, y, x_min, y_min, x_max, y_max)

    if in_outer and not in_pit:
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        dx = x - cx
        dy = y - cy
        dist = math.hypot(dx, dy)
        if dist < 0.01:
            return
        nx, ny = dx / dist, dy / dist

        dist_to_inner = _dist_to_rect_edge(x, y, x_min, y_min, x_max, y_max)
        depth = max(0, 1.0 - dist_to_inner / lip) if lip > 0 else 0

        lip_force = depth * 400.0
        body.vel[0] += nx * lip_force * (1.0 / 120.0)
        body.vel[1] += ny * lip_force * (1.0 / 120.0)


def check_pit(body: RobotBody, arena: Arena) -> bool:
    """Check if robot center has fallen into the square pit (past the lip)."""
    if not arena.has_pit:
        return False
    x, y = body.pos[0], body.pos[1]
    x_min, y_min = arena.pit_min
    x_max, y_max = arena.pit_max
    shrink = arena.pit_lip_cm * 1.5
    return _point_in_rect(x, y,
                          x_min + shrink, y_min + shrink,
                          x_max - shrink, y_max - shrink)


# ---------------------------------------------------------------------------
# Physics world
# ---------------------------------------------------------------------------

class PhysicsWorld:
    """Manages the physics simulation for two robots in an arena."""

    def __init__(self, arena: Arena, config: PhysicsConfig | None = None):
        self.arena = arena
        self.cfg = config or PhysicsConfig()

    def step(self, a: RobotBody, b: RobotBody,
             out_a: tuple[float, float], out_b: tuple[float, float],
             dt: float) -> StepResult:
        """Advance one physics step. out_a/out_b are (throttle, steering)."""
        apply_tank_drive(a, out_a[0], out_a[1], self.cfg, dt)
        apply_tank_drive(b, out_b[0], out_b[1], self.cfg, dt)

        collision = resolve_collision(a, b, self.cfg)

        resolve_walls(a, self.arena, self.cfg)
        resolve_walls(b, self.arena, self.cfg)

        apply_pit_lip(a, self.arena, self.cfg)
        apply_pit_lip(b, self.arena, self.cfg)

        return StepResult(
            a_in_pit=check_pit(a, self.arena),
            b_in_pit=check_pit(b, self.arena),
            collision=collision,
        )
