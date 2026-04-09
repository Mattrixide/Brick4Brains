"""Enemy side classification and safe approach vector computation."""

import math

FRONT = "front"
BACK = "back"
LEFT = "left"
RIGHT = "right"


def angle_diff(a: float, b: float) -> float:
    """Shortest signed angle from b to a, wrapped to [-pi, pi]."""
    d = a - b
    return (d + math.pi) % (2.0 * math.pi) - math.pi


def classify_approach_side(
    our_pos: tuple[float, float],
    enemy_pos: tuple[float, float],
    enemy_heading_rad: float,
) -> str:
    """Determine which side of the enemy we are currently on.

    Returns one of: 'front', 'back', 'left', 'right'.
    Quadrants are ±45 degrees from each cardinal direction.
    """
    dx = our_pos[0] - enemy_pos[0]
    dy = our_pos[1] - enemy_pos[1]
    approach_angle = math.atan2(dy, dx)

    # Relative angle: 0 = we're directly in front of enemy
    rel = angle_diff(approach_angle, enemy_heading_rad)

    if abs(rel) < math.pi / 4:
        return FRONT
    elif abs(rel) > 3 * math.pi / 4:
        return BACK
    elif rel > 0:
        return LEFT
    else:
        return RIGHT


def heading_from_velocity(
    vx: float, vy: float, min_speed: float = 2.0
) -> float | None:
    """Estimate heading from velocity vector.

    Returns None if speed is below min_speed (unreliable).
    """
    speed = math.hypot(vx, vy)
    if speed < min_speed:
        return None
    return math.atan2(vy, vx)


_SIDE_OFFSETS = {
    FRONT: 0.0,
    BACK: math.pi,
    LEFT: -math.pi / 2,
    RIGHT: math.pi / 2,
}


def get_safe_approach_position(
    enemy_pos: tuple[float, float],
    enemy_heading_rad: float,
    safe_side: str,
    distance_cm: float = 25.0,
) -> tuple[float, float]:
    """Compute a target position on the enemy's safe side.

    Returns the (x, y) point at `distance_cm` from the enemy,
    on the specified side.
    """
    offset_angle = enemy_heading_rad + _SIDE_OFFSETS.get(safe_side, 0.0)
    return (
        enemy_pos[0] + distance_cm * math.cos(offset_angle),
        enemy_pos[1] + distance_cm * math.sin(offset_angle),
    )


def is_approach_safe(
    our_pos: tuple[float, float],
    enemy_pos: tuple[float, float],
    enemy_heading_rad: float,
    safe_side: str,
    tolerance_rad: float = math.pi / 4,
) -> bool:
    """Check if we are currently approaching from the safe side.

    Returns True if our position is within tolerance of the safe side.
    """
    dx = our_pos[0] - enemy_pos[0]
    dy = our_pos[1] - enemy_pos[1]
    approach_angle = math.atan2(dy, dx)

    ideal_angle = enemy_heading_rad + _SIDE_OFFSETS.get(safe_side, 0.0)
    diff = abs(angle_diff(approach_angle, ideal_angle))
    return diff <= tolerance_rad


def needs_flanking(
    our_pos: tuple[float, float],
    enemy_pos: tuple[float, float],
    enemy_heading_rad: float,
    safe_side: str,
    distance_cm: float = 999.0,
    enemy_heading_confidence: float = 0.0,
) -> bool:
    """Check if we should arc around to the safe side.

    Only returns True when flanking is highly confident:
    - Far enough away to maneuver (>80cm)
    - Enemy heading is reliable (confidence > 0.6)
    - We're approaching a clearly dangerous side (>60 degrees off safe side)

    At close range or with uncertain heading, always pursue directly.
    """
    # Too close to flank — just charge
    if distance_cm < 80.0:
        return False

    # Don't trust flank decisions with low heading confidence
    if enemy_heading_confidence < 0.6:
        return False

    # How far off the safe approach angle are we?
    dx = our_pos[0] - enemy_pos[0]
    dy = our_pos[1] - enemy_pos[1]
    approach_angle = math.atan2(dy, dx)
    ideal_angle = enemy_heading_rad + _SIDE_OFFSETS.get(safe_side, 0.0)
    off_angle = abs(angle_diff(approach_angle, ideal_angle))

    # Only flank if clearly on the wrong side (>60 degrees off)
    # Within 60 degrees, pursue directly — close enough to safe side
    return off_angle > math.pi / 3
