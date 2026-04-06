"""Enemy AI behaviors for combat robot simulation."""

import math
from abc import ABC, abstractmethod

import numpy as np

from simulator.physics import RobotBody, Arena


def _angle_to(src: np.ndarray, dst: np.ndarray) -> float:
    """Angle from src to dst."""
    d = dst - src
    return math.atan2(d[1], d[0])


def _angle_diff(a: float, b: float) -> float:
    """Shortest signed angle from b to a."""
    d = a - b
    return (d + math.pi) % (2.0 * math.pi) - math.pi


class EnemyAI(ABC):
    """Base class for enemy robot AI."""

    @abstractmethod
    def tick(self, own: RobotBody, opponent: RobotBody,
             arena: Arena, dt: float) -> tuple[float, float]:
        """Returns (throttle, steering)."""

    def reset(self, rng: np.random.Generator | None = None) -> None:
        pass


class StationaryAI(EnemyAI):
    """Does nothing. Baseline for testing pin mechanics."""

    def tick(self, own, opponent, arena, dt):
        return (0.0, 0.0)


class RandomWalkAI(EnemyAI):
    """Random direction changes every 0.5-2s with wall avoidance."""

    def __init__(self):
        self._target_heading = 0.0
        self._change_timer = 0.0
        self._change_interval = 1.0
        self._rng: np.random.Generator | None = None

    def reset(self, rng=None):
        self._rng = rng or np.random.default_rng()
        self._target_heading = self._rng.uniform(-math.pi, math.pi)
        self._change_timer = 0.0
        self._change_interval = self._rng.uniform(0.5, 2.0)

    def tick(self, own, opponent, arena, dt):
        if self._rng is None:
            self._rng = np.random.default_rng()

        self._change_timer += dt
        if self._change_timer >= self._change_interval:
            self._change_timer = 0.0
            self._change_interval = self._rng.uniform(0.5, 2.0)
            self._target_heading = self._rng.uniform(-math.pi, math.pi)

        # Wall avoidance: if near wall, turn toward center
        margin = 20.0
        if (abs(own.pos[0]) > arena.half_w - margin or
                abs(own.pos[1]) > arena.half_h - margin):
            self._target_heading = _angle_to(own.pos, np.zeros(2))

        alpha = _angle_diff(self._target_heading, own.heading)
        steering = max(-1.0, min(1.0, alpha * 1.5))
        throttle = 0.4

        return (throttle, steering)


class AggressivePursuitAI(EnemyAI):
    """Always drives toward opponent. Full throttle."""

    def tick(self, own, opponent, arena, dt):
        desired = _angle_to(own.pos, opponent.pos)
        alpha = _angle_diff(desired, own.heading)

        if abs(alpha) > 1.0:
            return (0.0, 0.6 if alpha > 0 else -0.6)

        steering = max(-0.8, min(0.8, alpha * 1.0))
        throttle = 0.8 * (1.0 - abs(steering) * 0.3)
        return (throttle, steering)


class DefensiveAI(EnemyAI):
    """Evades when opponent is close, random walk otherwise."""

    def __init__(self, flee_range_cm: float = 40.0):
        self._flee_range = flee_range_cm
        self._random = RandomWalkAI()

    def reset(self, rng=None):
        self._random.reset(rng)

    def tick(self, own, opponent, arena, dt):
        dist = float(np.linalg.norm(opponent.pos - own.pos))

        if dist < self._flee_range:
            # Run away
            away = _angle_to(opponent.pos, own.pos)
            alpha = _angle_diff(away, own.heading)
            steering = max(-1.0, min(1.0, alpha * 1.5))
            return (0.6, steering)

        return self._random.tick(own, opponent, arena, dt)


class WedgeAI(EnemyAI):
    """Always faces opponent and drives forward. Tests flanking."""

    def tick(self, own, opponent, arena, dt):
        desired = _angle_to(own.pos, opponent.pos)
        alpha = _angle_diff(desired, own.heading)
        steering = max(-1.0, min(1.0, alpha * 2.0))
        throttle = 0.6 * (1.0 - abs(alpha) / math.pi)
        return (throttle, steering)


# Registry for lookup by name
AI_REGISTRY: dict[str, type[EnemyAI]] = {
    "stationary": StationaryAI,
    "random_walk": RandomWalkAI,
    "aggressive": AggressivePursuitAI,
    "defensive": DefensiveAI,
    "wedge": WedgeAI,
}


def create_enemy_ai(name: str, **kwargs) -> EnemyAI:
    """Create an enemy AI by name."""
    cls = AI_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown AI: {name}. Available: {list(AI_REGISTRY.keys())}")
    return cls(**kwargs)
