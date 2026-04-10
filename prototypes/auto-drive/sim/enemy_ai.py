"""Enemy AI controller with switchable behavior modes."""
import math

from sim.arena import SimRobot
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


class EnemyController:
    """Switchable enemy AI with multiple behavior modes."""

    def __init__(self):
        self.mode = "manual"
        self._ai_bridge = None
        self._ai_match_started = False

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
        # Angle AWAY from brick
        target_angle = math.atan2(ey - by, ex - bx)
        angle_diff = _wrap_to_pi(target_angle - enemy.heading_rad)
        steering = _clamp(angle_diff * 2.0, -1, 1)
        return (0.7, steering)

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
