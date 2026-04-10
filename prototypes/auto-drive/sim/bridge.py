"""Bridge connecting a SimRobot to the real BattleController."""

import math
import os
import time

from state_machine import BattleController, BattleContext, BattleOutput
from battle_config import BattleConfig
from match_timer import MatchTimer, PinTimer


# --- Sim clock patch ---
# BattleController and MatchTimer use time.perf_counter() internally.
# In headless/fast sim, wall clock doesn't match sim time.
# This patches time.perf_counter to return sim-accumulated time.
_sim_time = 0.0
_real_perf_counter = time.perf_counter


def _sim_perf_counter():
    return _sim_time


def _enable_sim_clock():
    global _sim_time
    _sim_time = _real_perf_counter()
    time.perf_counter = _sim_perf_counter


def _advance_sim_clock(dt):
    global _sim_time
    _sim_time += dt


def _disable_sim_clock():
    time.perf_counter = _real_perf_counter

from sim.arena import SimRobot
from sim.config import SimConfig

AUTO_DRIVE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class SimBridge:
    """Connects a SimRobot to a BattleController so the AI can drive it."""

    def __init__(self, robot: SimRobot, cfg: SimConfig, battle_config_path=None,
                 strategy_override=None):
        self.robot = robot
        self.cfg = cfg

        if battle_config_path is None:
            battle_config_path = os.path.join(AUTO_DRIVE_DIR, "battle_config.json")

        self.battle_config = BattleConfig.load(battle_config_path)
        self._battle_config_path = battle_config_path

        # Override strategy without modifying the real config file
        if strategy_override:
            self.battle_config.strategy = strategy_override
            self.battle_config.opening_strategy = strategy_override

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
        # Load arena corners for accurate wall detection
        import json
        arena_corners = None
        floor_cal_path = os.path.join(AUTO_DRIVE_DIR, "floor_calibration.json")
        if os.path.exists(floor_cal_path):
            with open(floor_cal_path) as f:
                floor_cal = json.load(f)
            if "corners_ft" in floor_cal:
                arena_corners = [tuple(c) for c in floor_cal["corners_ft"]]
        self.controller = BattleController(bc, self.match_timer, self._pin_timer,
                                           arena_corners=arena_corners)

    def start_match(self, enemy: SimRobot):
        """Start the match timer and transition controller out of 'wait'."""
        _enable_sim_clock()
        self.match_timer.start()
        # Build a context so controller.start_match can decide opening strategy
        ox, oy = self.robot.position
        ex, ey = enemy.position
        ctx = BattleContext(
            our_pos=(ox, oy),
            our_heading_rad=self.robot.heading_rad,
            our_velocity=(0, 0),
            enemy_pos=(ex, ey) if enemy.alive else None,
            enemy_detected=enemy.alive,
            enemy_tracking=enemy.alive,
            distance_cm=math.hypot(ex - ox, ey - oy) if enemy.alive else 999.0,
            dt=0.016,
            our_detected=True,
        )
        self.controller.start_match(ctx)

    def reset(self):
        """Recreate controller and timers from config."""
        self._build_controller()
        self._prev_vx = 0.0
        self._prev_vy = 0.0
        self._last_output = BattleOutput()

    def tick(self, dt: float, enemy: SimRobot) -> BattleOutput:
        """Run one frame of the BattleController and apply forces to the robot."""
        _advance_sim_clock(dt)

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
        # 1g = 980 cm/s^2, 1mg = 0.98 cm/s^2, so mg = cm_s2 / 0.98
        cos_a = math.cos(our_heading)
        sin_a = math.sin(our_heading)
        accel_fwd = ax * cos_a + ay * sin_a   # forward
        accel_lat = -ax * sin_a + ay * cos_a   # lateral
        accel_fwd_mg = accel_fwd / 0.98
        accel_lat_mg = accel_lat / 0.98
        # Add simulated motor/floor vibration (~200mg baseline) so
        # IMU-based stuck detection works (real robot has constant vibration)
        import random
        accel_fwd_mg += random.gauss(0, 200)
        accel_lat_mg += random.gauss(0, 150)

        # Distance — edge-to-edge, not center-to-center
        # Subtract half-depths along the line between robots so thresholds
        # work the same as real CV (where ArUco is near robot center)
        dx = ex - ox
        dy = ey - oy
        center_dist = math.sqrt(dx * dx + dy * dy)
        edge_offset = self.robot.depth / 2 + enemy.depth / 2
        distance = max(0.0, center_dist - edge_offset)

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
            throttle_cmd=self._last_output.throttle if self._last_output.target_omega_dps is None else self._last_output.target_speed,
        )

        # Tick the battle controller
        output = self.controller.tick(ctx)
        self._last_output = output

        # Apply output to pymunk body
        if not self.robot.alive:
            return output

        if output.target_omega_dps is not None:
            # Rate mode: store target for post-step application
            # Negate: BattleController omega sign assumes CW=positive (inverted IMU)
            # but pymunk uses standard math convention (CCW=positive)
            self.robot._rate_mode_omega = -math.radians(output.target_omega_dps)
            self.robot._rate_mode_speed = output.target_speed

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
