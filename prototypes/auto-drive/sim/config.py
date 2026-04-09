"""Tunable physics parameters for the combat simulator.
Runtime config loads from sim_config.json; these are defaults."""
import json
import os
from dataclasses import dataclass, asdict


@dataclass
class SimConfig:
    # Arena
    arena_cm: float = 244.0

    # Robot dimensions (inches to cm)
    brick_width_cm: float = 22.86     # 9 inches
    brick_depth_cm: float = 17.78     # 7 inches
    enemy_width_cm: float = 15.24     # 6 inches
    enemy_depth_cm: float = 22.86     # 9 inches

    # Masses
    brick_mass_kg: float = 1.36       # 3lb beetleweight
    enemy_mass_kg: float = 1.36

    # Drive forces (calibrate.py populates these)
    max_forward_force: float = 400.0
    max_torque: float = 1500.0
    motor_lag_s: float = 0.030        # ESC response lag (~30ms)

    # Friction
    ground_friction_mu: float = 0.6   # rubber on plywood
    lateral_damping: float = 0.85     # per-frame lateral velocity retention
    angular_damping: float = 0.93     # per-frame angular velocity retention
    gravity_cms2: float = 980.0       # g in cm/s^2

    # Collision
    wall_elasticity: float = 0.2
    wall_friction: float = 1.0
    robot_elasticity: float = 0.3
    robot_friction: float = 0.8

    # Simulation
    physics_fps: int = 240            # substeps per second
    render_fps: int = 60
    scale_px_per_cm: float = 2.5

    # Pit
    pit_elimination: bool = True

    def save(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "sim_config.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "sim_config.json")
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()
