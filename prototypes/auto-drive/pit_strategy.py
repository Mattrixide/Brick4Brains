"""Pit strategy utilities for combat state machine.

Provides pit zone geometry, herding calculations, and self-pit avoidance.
Used by the BattleController's PIT_* states.
"""

import math
from dataclasses import dataclass, field


@dataclass
class PitZone:
    """Defines a pit hazard zone in arena coordinates (cm)."""
    center_x_cm: float = 0.0
    center_y_cm: float = 0.0
    radius_cm: float = 20.0
    danger_radius_cm: float = 40.0

    @property
    def center(self) -> tuple[float, float]:
        return (self.center_x_cm, self.center_y_cm)


def herding_point(
    enemy_pos: tuple[float, float],
    pit: PitZone,
    offset_cm: float = 25.0,
) -> tuple[float, float]:
    """Compute position behind enemy relative to pit for herding.

    Returns the point we should drive to in order to push
    the enemy toward the pit.
    """
    dx = enemy_pos[0] - pit.center_x_cm
    dy = enemy_pos[1] - pit.center_y_cm
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        dist = 1.0
    nx, ny = dx / dist, dy / dist
    return (
        enemy_pos[0] + nx * offset_cm,
        enemy_pos[1] + ny * offset_cm,
    )


def push_target(
    enemy_pos: tuple[float, float],
    pit: PitZone,
    overshoot_factor: float = 0.3,
) -> tuple[float, float]:
    """Compute a target point past the enemy toward the pit.

    Used during the PUSH phase — aim through the enemy.
    """
    return (
        enemy_pos[0] - (enemy_pos[0] - pit.center_x_cm) * overshoot_factor,
        enemy_pos[1] - (enemy_pos[1] - pit.center_y_cm) * overshoot_factor,
    )


def is_near_pit(
    pos: tuple[float, float],
    pit: PitZone,
    margin_cm: float = 0.0,
) -> bool:
    """Check if a position is within the pit's danger zone."""
    dist = math.hypot(pos[0] - pit.center_x_cm, pos[1] - pit.center_y_cm)
    return dist < (pit.danger_radius_cm + margin_cm)


def should_abort(
    our_pos: tuple[float, float],
    pit: PitZone,
    margin_cm: float = 15.0,
) -> bool:
    """Check if our robot is dangerously close to the pit (self-preservation)."""
    dist = math.hypot(our_pos[0] - pit.center_x_cm, our_pos[1] - pit.center_y_cm)
    return dist < (pit.danger_radius_cm + margin_cm)


def is_enemy_near_pit(
    enemy_pos: tuple[float, float],
    pit: PitZone,
) -> bool:
    """Check if the enemy is within the pit's danger radius."""
    dist = math.hypot(enemy_pos[0] - pit.center_x_cm, enemy_pos[1] - pit.center_y_cm)
    return dist < pit.danger_radius_cm


def pit_speed_limit(
    our_pos: tuple[float, float],
    pit: PitZone,
    base_throttle: float,
    slow_zone_multiplier: float = 1.5,
) -> float:
    """Reduce throttle when near the pit for safety.

    Within slow_zone_multiplier * danger_radius, throttle is halved.
    """
    dist = math.hypot(our_pos[0] - pit.center_x_cm, our_pos[1] - pit.center_y_cm)
    if dist < pit.danger_radius_cm * slow_zone_multiplier:
        return base_throttle * 0.5
    return base_throttle
