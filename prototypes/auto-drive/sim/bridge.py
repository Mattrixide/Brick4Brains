"""Bridge connecting a SimRobot to the real BattleController."""

import math
import os

from state_machine import BattleController, BattleContext, BattleOutput
from battle_config import BattleConfig
from match_timer import MatchTimer, PinTimer

from sim.arena import SimRobot
from sim.config import SimConfig

AUTO_DRIVE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class SimBridge:
    """Connects a SimRobot to a BattleController so the AI can drive it."""

    def __init__(self, robot: SimRobot, cfg: SimConfig, battle_config_path=None):
        self.robot = robot
        self.cfg = cfg

        if battle_config_path is None:
            battle_config_path = os.path.join(AUTO_DRIVE_DIR, "battle_config.json")

        self.battle_config = BattleConfig.load(battle_config_path)
        self._battle_config_path = battle_config_path

        self._build_controller()

        self._prev_vx = 0.0
        self._prev_vy = 0.0
        self._last_output = BattleOutput()

    def _build_controller(self):
        """Create timer, pin_timer, and controller from current config."""
        bc = self.battle_config
        self.match_timer = MatchTimer(
            duration_s=bc.match_duration_s,
            urgency_ramp_s=bc.urgency_ramp_start_s,
            phase_start_s=bc.phase_start_s,
            phase_final_s=bc.phase_final_s,
        )
        self._pin_timer = PinTimer(max_duration_s=bc.pin_duration_s)
        self.controller = BattleController(bc, self.match_timer, self._pin_timer)

    def start_match(self):
        """Start the match timer."""
        self.match_timer.start()

    def reset(self):
        """Recreate controller and timers from config."""
        self._build_controller()
        self._prev_vx = 0.0
        self._prev_vy = 0.0
        self._last_output = BattleOutput()

    def tick(self, dt: float, enemy: SimRobot) -> BattleOutput:
        """Run one frame of the BattleController and apply forces to the robot."""
        # Our state
        ox, oy = self.robot.position
        our_heading = self.robot.heading_rad
        ovx, ovy = self.robot.velocity

        # Enemy state
        ex, ey = enemy.position
        enemy_heading = enemy.heading_rad
        evx, evy = enemy.velocity

        # Acceleration from velocity delta (cm/s^2 -> milligravity)
        if dt > 0:
            ax = (ovx - self._prev_vx) / dt
            ay = (ovy - self._prev_vy) / dt
        else:
            ax = 0.0
            ay = 0.0
        self._prev_vx = ovx
        self._prev_vy = ovy

        # Convert to body-frame acceleration and then to milligravity
        cos_a = math.cos(our_heading)
        sin_a = math.sin(our_heading)
        accel_fwd = ax * cos_a + ay * sin_a   # forward
        accel_lat = -ax * sin_a + ay * cos_a   # lateral
        accel_fwd_mg = accel_fwd / 9.80  # cm/s^2 to milligravity (0.98 cm/s^2 per mg)
        accel_lat_mg = accel_lat / 9.80

        # Distance
        dx = ex - ox
        dy = ey - oy
        distance = math.sqrt(dx * dx + dy * dy)

        # Pack context
        ctx = BattleContext(
            our_pos=(ox, oy),
            our_heading_rad=our_heading,
            our_velocity=(ovx, ovy),
            enemy_pos=(ex, ey),
            enemy_heading_rad=enemy_heading,
            enemy_velocity=(evx, evy),
            enemy_detected=enemy.alive,
            enemy_tracking=enemy.alive,
            frames_without_detection=0 if enemy.alive else 999,
            distance_cm=distance,
            dt=dt,
            our_detected=True,
            accel_x_mg=accel_fwd_mg,
            accel_y_mg=accel_lat_mg,
            throttle_cmd=self._last_output.throttle,
        )

        # Tick the battle controller
        output = self.controller.tick(ctx)
        self._last_output = output

        # Apply output to pymunk body
        if not self.robot.alive:
            return output

        if output.target_omega_dps is not None:
            # Rate mode: P-controller on angular velocity
            current_omega_dps = math.degrees(self.robot.angular_velocity)
            omega_error = output.target_omega_dps - current_omega_dps
            torque = _clamp(
                omega_error * self.cfg.max_torque * 0.005,
                -self.cfg.max_torque,
                self.cfg.max_torque,
            )
            self.robot.body.torque += torque

            # Forward force from target_speed
            if abs(output.target_speed) > 0.01:
                self.robot.body.apply_force_at_local_point(
                    (output.target_speed * self.cfg.max_forward_force, 0), (0, 0)
                )
        else:
            # Direct mode
            self.robot.apply_drive(output.throttle, output.steering, self.cfg)

        return output

    @property
    def state(self) -> str:
        """Current battle controller state name."""
        return str(self.controller.state)

    @property
    def last_output(self) -> BattleOutput:
        return self._last_output
