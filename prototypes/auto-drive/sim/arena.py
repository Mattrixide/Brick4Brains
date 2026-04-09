"""Physics arena and robot bodies for the combat simulator."""
import json
import math
import os

import pymunk

from .config import SimConfig

AUTO_DRIVE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_json(filename):
    path = os.path.join(AUTO_DRIVE_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


class SimRobot:
    """A rectangular robot body in the pymunk physics space."""

    def __init__(self, space, x, y, heading_deg, width, depth, mass, name="robot"):
        self.name = name
        self.width = width    # Y-axis (lateral)
        self.depth = depth    # X-axis (forward)
        self.mass = mass
        self.alive = True

        moment = pymunk.moment_for_box(mass, (depth, width))
        self.body = pymunk.Body(mass, moment)
        self.body.position = (x, y)
        self.body.angle = math.radians(heading_deg)

        # Vertices: front-right, front-left, back-left, back-right
        hw = width / 2.0
        hd = depth / 2.0
        vertices = [
            (hd, -hw),   # front-right
            (hd, hw),    # front-left
            (-hd, hw),   # back-left
            (-hd, -hw),  # back-right
        ]
        self.shape = pymunk.Poly(self.body, vertices)
        self.shape.collision_type = 1
        self.shape.elasticity = 0.3
        self.shape.friction = 0.8

        space.add(self.body, self.shape)

    @property
    def position(self):
        return tuple(self.body.position)

    @property
    def heading_rad(self):
        return self.body.angle

    @property
    def velocity(self):
        return tuple(self.body.velocity)

    @property
    def angular_velocity(self):
        return self.body.angular_velocity

    def apply_drive(self, throttle, steering, cfg):
        """Apply forward force and steering torque."""
        if not self.alive:
            return
        # Forward force in local X direction
        force_mag = throttle * cfg.max_forward_force
        local_force = (force_mag, 0)
        self.body.apply_force_at_local_point(local_force, (0, 0))

        # Steering torque
        torque = steering * cfg.max_torque
        self.body.torque += torque

    def apply_friction_forces(self, cfg):
        """Coulomb friction opposing forward motion. Applied BEFORE substeps."""
        if not self.alive:
            return
        vx, vy = self.body.velocity
        speed = math.sqrt(vx * vx + vy * vy)
        if speed < 0.01:
            return

        # Decompose velocity into forward component
        cos_a = math.cos(self.body.angle)
        sin_a = math.sin(self.body.angle)
        forward_speed = vx * cos_a + vy * sin_a

        if abs(forward_speed) < 0.01:
            return

        # Coulomb friction force = mu * m * g, opposing forward motion
        friction_mag = cfg.ground_friction_mu * self.mass * cfg.gravity_cms2
        # Direction opposing forward motion
        sign = -1.0 if forward_speed > 0 else 1.0
        fx = sign * friction_mag * cos_a
        fy = sign * friction_mag * sin_a
        self.body.apply_force_at_local_point((sign * friction_mag, 0), (0, 0))

    def apply_velocity_damping(self, cfg):
        """Lateral and angular damping. Applied AFTER substeps."""
        if not self.alive:
            return
        vx, vy = self.body.velocity
        speed = math.sqrt(vx * vx + vy * vy)

        if speed < 1.0:
            self.body.velocity = (vx * 0.8, vy * 0.8)
            self.body.angular_velocity *= 0.8
            return

        # Decompose into forward/lateral
        cos_a = math.cos(self.body.angle)
        sin_a = math.sin(self.body.angle)
        forward_speed = vx * cos_a + vy * sin_a
        lateral_speed = -vx * sin_a + vy * cos_a

        # Damp lateral
        lateral_speed *= cfg.lateral_damping

        # Recompose
        self.body.velocity = (
            forward_speed * cos_a - lateral_speed * sin_a,
            forward_speed * sin_a + lateral_speed * cos_a,
        )

        # Damp angular
        self.body.angular_velocity *= cfg.angular_damping

    def get_corners_world(self):
        """Return 4 corners in world cm coordinates."""
        hw = self.width / 2.0
        hd = self.depth / 2.0
        local = [
            (hd, -hw),
            (hd, hw),
            (-hd, hw),
            (-hd, -hw),
        ]
        cos_a = math.cos(self.body.angle)
        sin_a = math.sin(self.body.angle)
        px, py = self.body.position
        world = []
        for lx, ly in local:
            wx = px + lx * cos_a - ly * sin_a
            wy = py + lx * sin_a + ly * cos_a
            world.append((wx, wy))
        return world

    def freeze(self):
        """Mark robot as eliminated."""
        self.alive = False
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0

    def reset(self, x, y, heading_deg):
        """Reset robot to starting position."""
        self.body.position = (x, y)
        self.body.angle = math.radians(heading_deg)
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0
        self.alive = True


class SimArena:
    """Physics world containing walls, pit, and two robots."""

    def __init__(self):
        self.cfg = SimConfig.load()
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # top-down, no gravity

        # Load calibration data
        floor_cal = _load_json("floor_calibration.json")
        battle_cfg = _load_json("battle_config.json")

        # Build arena walls
        if floor_cal and "corners_ft" in floor_cal:
            # corners_ft is actually in cm
            self.arena_corners = [tuple(c) for c in floor_cal["corners_ft"]]
        else:
            # Fallback: 244cm square centered at origin
            h = self.cfg.arena_cm / 2.0
            self.arena_corners = [(-h, -h), (h, -h), (h, h), (-h, h)]

        self._create_walls()

        # Pit sensor
        self.pit_center = None
        self.pit_radius = None
        if battle_cfg and "pit_x_cm" in battle_cfg:
            self.pit_center = (battle_cfg["pit_x_cm"], battle_cfg["pit_y_cm"])
            self.pit_radius = battle_cfg["pit_radius_cm"]
            self._create_pit()

        # Create robots
        self.brick = SimRobot(
            self.space, -40, 0, 0,
            self.cfg.brick_width_cm, self.cfg.brick_depth_cm,
            self.cfg.brick_mass_kg, name="brick",
        )
        self.brick.shape.elasticity = self.cfg.robot_elasticity
        self.brick.shape.friction = self.cfg.robot_friction

        self.enemy = SimRobot(
            self.space, 40, 0, 180,
            self.cfg.enemy_width_cm, self.cfg.enemy_depth_cm,
            self.cfg.enemy_mass_kg, name="enemy",
        )
        self.enemy.shape.elasticity = self.cfg.robot_elasticity
        self.enemy.shape.friction = self.cfg.robot_friction

    def _create_walls(self):
        """Create static wall segments around the arena perimeter."""
        self.walls = []
        n = len(self.arena_corners)
        for i in range(n):
            a = self.arena_corners[i]
            b = self.arena_corners[(i + 1) % n]
            seg = pymunk.Segment(self.space.static_body, a, b, 1.0)
            seg.elasticity = self.cfg.wall_elasticity
            seg.friction = self.cfg.wall_friction
            self.walls.append(seg)
            self.space.add(seg)

    def _create_pit(self):
        """Create a circular pit sensor using a polygon approximation."""
        cx, cy = self.pit_center
        r = self.pit_radius
        # Approximate circle with 16 segments
        num_pts = 16
        vertices = []
        for i in range(num_pts):
            angle = 2 * math.pi * i / num_pts
            vertices.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

        pit_body = self.space.static_body
        pit_shape = pymunk.Poly(pit_body, vertices)
        pit_shape.sensor = True
        pit_shape.collision_type = 2
        self.space.add(pit_shape)
        self.pit_shape = pit_shape

        # Collision handler: robot (1) vs pit (2)
        if self.cfg.pit_elimination:
            handler = self.space.add_collision_handler(1, 2)

            def _pit_begin(arbiter, space, data):
                # Find which robot fell in and freeze it
                for shape in arbiter.shapes:
                    if shape.collision_type == 1:
                        # Find the robot that owns this shape
                        for robot in (self.brick, self.enemy):
                            if robot.shape is shape:
                                robot.freeze()
                return False

            handler.begin = _pit_begin

    def step(self, dt=None):
        """Advance physics by one render frame."""
        if dt is None:
            dt = 1.0 / self.cfg.render_fps

        substeps = self.cfg.physics_fps // self.cfg.render_fps
        sub_dt = dt / substeps

        # Apply friction forces BEFORE substeps
        self.brick.apply_friction_forces(self.cfg)
        self.enemy.apply_friction_forces(self.cfg)

        # Run substeps
        for _ in range(substeps):
            self.space.step(sub_dt)

        # Apply velocity damping AFTER substeps
        self.brick.apply_velocity_damping(self.cfg)
        self.enemy.apply_velocity_damping(self.cfg)

    def reset(self):
        """Reset both robots to starting positions."""
        self.brick.reset(-40, 0, 0)
        self.enemy.reset(40, 0, 180)
