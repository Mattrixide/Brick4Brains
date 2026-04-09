"""Single match simulation runner with simulated time clock."""

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

import simulator  # ensure sys.path setup
from battle_config import BattleConfig
from match_timer import MatchTimer, PinTimer
from state_machine import BattleController, BattleContext

from simulator.physics import (
    Arena, PhysicsConfig, PhysicsWorld, RobotBody,
)
from simulator.vision import VisionConfig, VisionSimulator
from simulator.enemy_ai import create_enemy_ai


# ---------------------------------------------------------------------------
# Simulated clock — patches time.perf_counter for faster-than-realtime
# ---------------------------------------------------------------------------

class SimClock:
    """Replaces time.perf_counter for simulated time."""

    def __init__(self, start: float = 0.0):
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


@contextmanager
def simulated_time(clock: SimClock):
    """Context manager that patches time.perf_counter with a SimClock."""
    original = time.perf_counter
    time.perf_counter = clock
    try:
        yield clock
    finally:
        time.perf_counter = original


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    battle_config: BattleConfig = field(default_factory=BattleConfig)
    physics_config: PhysicsConfig = field(default_factory=PhysicsConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    enemy_ai_type: str = "random_walk"
    match_duration_s: float = 180.0
    physics_hz: int = 120
    our_start_pos: tuple[float, float] = (-80.0, 0.0)
    our_start_heading: float = 0.0
    enemy_start_pos: tuple[float, float] = (80.0, 0.0)
    enemy_start_heading: float = 3.14159
    seed: int = 0
    pin_win_duration_s: float = 5.0  # how long a pin must hold to score


@dataclass
class SimResult:
    outcome: str = "timeout"
    duration_s: float = 0.0
    total_pins: int = 0
    time_to_first_pin: float | None = None
    pit_events: int = 0
    states_visited: dict[str, float] = field(default_factory=dict)
    avg_distance_cm: float = 0.0
    collision_count: int = 0
    seed: int = 0


# ---------------------------------------------------------------------------
# Pin detector (physics-level authority)
# ---------------------------------------------------------------------------

class PinDetector:
    """Detects pin events based on controller state + physics proximity.

    Uses a hybrid approach: the controller entering pin state means
    it believes a pin is happening. The physics validates proximity and
    wall contact. A pin "scores" when the controller has accumulated
    enough time in pin state.
    """

    def __init__(self, wall_threshold_cm: float = 80.0,
                 pin_score_duration_s: float = 5.0,
                 proximity_cm: float = 25.0):
        self.wall_threshold = wall_threshold_cm
        self.pin_score_duration = pin_score_duration_s
        self.proximity = proximity_cm
        self._our_cumulative_pin = 0.0
        self._enemy_pin_time = 0.0
        self.our_pin_count = 0
        self.enemy_pin_count = 0
        self.first_pin_time: float | None = None

    def _is_at_wall(self, pos: np.ndarray) -> bool:
        return abs(pos[0]) >= self.wall_threshold or abs(pos[1]) >= self.wall_threshold

    def update(self, our: RobotBody, enemy: RobotBody,
               controller_state: str, dt: float, sim_time: float) -> str | None:
        """Returns 'win_pin' or 'loss_pin' if a pin scores, else None."""
        dist = float(np.linalg.norm(our.pos - enemy.pos))
        close = dist < self.proximity

        # Our robot pinning enemy (controller in pin + physics confirms)
        if controller_state == "pin" and close and self._is_at_wall(enemy.pos):
            self._our_cumulative_pin += dt
            if self._our_cumulative_pin >= self.pin_score_duration:
                self.our_pin_count += 1
                if self.first_pin_time is None:
                    self.first_pin_time = sim_time
                self._our_cumulative_pin = 0.0
                return "win_pin"

        # Enemy pinning us (physics only — enemy AI doesn't have state machine)
        if (close and self._is_at_wall(our.pos) and
                our.speed() < 8.0 and enemy.speed() > 2.0):
            self._enemy_pin_time += dt
            if self._enemy_pin_time >= self.pin_score_duration:
                self.enemy_pin_count += 1
                self._enemy_pin_time = 0.0
                return "loss_pin"
        else:
            self._enemy_pin_time = 0.0

        return None


# ---------------------------------------------------------------------------
# Simulation recorder
# ---------------------------------------------------------------------------

class SimRecorder:
    """Accumulates per-tick telemetry and produces a SimResult."""

    def __init__(self, seed: int):
        self._seed = seed
        self._state_times: dict[str, float] = {}
        self._distances: list[float] = []
        self._collision_count = 0
        self._pit_events = 0
        self._total_pins = 0
        self._first_pin_time: float | None = None
        self._duration = 0.0

    def record(self, state: str, distance: float, collision: bool, dt: float) -> None:
        self._state_times[state] = self._state_times.get(state, 0.0) + dt
        self._distances.append(distance)
        if collision:
            self._collision_count += 1
        self._duration += dt

    def record_pin(self, sim_time: float) -> None:
        self._total_pins += 1
        if self._first_pin_time is None:
            self._first_pin_time = sim_time

    def record_pit(self) -> None:
        self._pit_events += 1

    def finalize(self, outcome: str) -> SimResult:
        avg_dist = sum(self._distances) / max(1, len(self._distances))
        return SimResult(
            outcome=outcome,
            duration_s=self._duration,
            total_pins=self._total_pins,
            time_to_first_pin=self._first_pin_time,
            pit_events=self._pit_events,
            states_visited=dict(self._state_times),
            avg_distance_cm=avg_dist,
            collision_count=self._collision_count,
            seed=self._seed,
        )


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def run_single(config: SimConfig, visualizer=None) -> SimResult:
    """Run a single complete match simulation."""
    clock = SimClock(start=1000.0)  # start at 1000s to avoid zero-time edge cases

    with simulated_time(clock):
        rng = np.random.default_rng(config.seed)

        # Create arena
        bc = config.battle_config
        if bc.pit_radius_cm > 0 and (bc.pit_x_cm != 0 or bc.pit_y_cm != 0):
            arena = Arena.with_corner_pit(
                corner="upper_right",
                size_cm=bc.pit_radius_cm * 2,
                inset_cm=7.6,
                lip_cm=1.9,
            )
            # Sync battle config pit coords with actual arena pit center
            # so the state machine's herding logic aims at the right spot
            bc.pit_x_cm = arena.pit_center[0]
            bc.pit_y_cm = arena.pit_center[1]
        else:
            arena = Arena()

        # Create physics
        phys_cfg = config.physics_config
        world = PhysicsWorld(arena, phys_cfg)

        # Create robot bodies (6" x 10" rectangles)
        # Our robot is 1.5x speed and force vs enemy
        our_body = RobotBody(
            pos=np.array(config.our_start_pos, dtype=float),
            heading=config.our_start_heading,
            vel=np.zeros(2),
            mass=phys_cfg.robot_mass_kg,
            length=phys_cfg.robot_length_cm,
            width=phys_cfg.robot_width_cm,
            speed_mult=1.5,
            accel_mult=2.0,
        )
        enemy_body = RobotBody(
            pos=np.array(config.enemy_start_pos, dtype=float),
            heading=config.enemy_start_heading,
            vel=np.zeros(2),
            mass=phys_cfg.robot_mass_kg,
            length=phys_cfg.robot_length_cm,
            width=phys_cfg.robot_width_cm,
            speed_mult=1.0,
            accel_mult=1.0,
        )

        # Create vision
        vis_cfg = VisionConfig(seed=int(rng.integers(0, 2**31)))
        vision = VisionSimulator(vis_cfg, arena.half_w, arena.half_h)

        # Create enemy AI
        enemy_ai = create_enemy_ai(config.enemy_ai_type)
        enemy_ai.reset(np.random.default_rng(int(rng.integers(0, 2**31))))

        # Create battle controller
        config.battle_config.match_duration_s = config.match_duration_s
        match_timer = MatchTimer(config.match_duration_s,
                                 config.battle_config.urgency_ramp_start_s)
        pin_timer = PinTimer(config.battle_config.pin_duration_s)
        controller = BattleController(config.battle_config, match_timer, pin_timer)

        # Start match
        match_timer.start()
        controller.reset()

        # Pin detection and recording
        pin_detector = PinDetector(
            wall_threshold_cm=config.battle_config.wall_threshold_cm,
        )
        recorder = SimRecorder(config.seed)

        dt = 1.0 / config.physics_hz
        max_steps = int(config.match_duration_s * config.physics_hz) + 10
        outcome = "timeout"

        for step in range(max_steps):
            sim_time = clock() - 1000.0  # relative to match start

            # Generate sensor context
            ctx = vision.generate_context(our_body, enemy_body, dt)

            # Run our controller
            output = controller.tick(ctx)

            # Run enemy AI
            enemy_out = enemy_ai.tick(enemy_body, our_body, arena, dt)

            # Physics step
            step_result = world.step(
                our_body, enemy_body,
                (output.throttle, output.steering),
                enemy_out,
                dt,
            )

            # Compute distance
            dist = float(np.linalg.norm(our_body.pos - enemy_body.pos))

            # Record telemetry
            recorder.record(controller.state, dist, step_result.collision, dt)

            # Check pin
            pin_result = pin_detector.update(
                our_body, enemy_body, controller.state, dt, sim_time
            )
            if pin_result == "win_pin":
                recorder.record_pin(sim_time)
                # Don't end match on first pin — keep fighting (like real rules)
                # But count it as a "win" outcome if match ends
            elif pin_result == "loss_pin":
                outcome = "loss_pin"
                break

            # Check pits
            if step_result.a_in_pit:
                recorder.record_pit()
                outcome = "loss_pit"
                break
            if step_result.b_in_pit:
                recorder.record_pit()
                outcome = "win_pit"
                break

            # Check match timer
            if match_timer.is_expired:
                if pin_detector.our_pin_count > 0:
                    outcome = "win_pin"
                else:
                    outcome = "timeout"
                break

            # Visualization callback
            if visualizer is not None:
                visualizer.draw_frame(
                    our_body, enemy_body,
                    controller.state, sim_time,
                    match_timer.remaining_s,
                    ctx.enemy_detected,
                    ctx.enemy_tracking,
                )
                if not visualizer.handle_events():
                    outcome = "aborted"
                    break

            # Advance clock
            clock.advance(dt)

        return recorder.finalize(outcome)
