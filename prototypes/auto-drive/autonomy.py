"""Autonomous mission and path-following module for combat robot prototype.

Provides waypoint-based missions (square, forward-back, circle, goto) and a
PID-driven PathFollower that outputs throttle/steering commands normalized
to [-1, 1].

Also provides IMUAssistedPathFollower that uses gyro-assisted turns via
ESP32 and trapezoidal velocity profiling for maximum-speed navigation.

No external dependencies beyond the stdlib.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    x: float  # cm from origin
    y: float  # cm from origin
    heading: Optional[float] = None  # target heading in radians, None = don't care


@dataclass
class Mission:
    name: str
    waypoints: list[Waypoint]
    description: str


# ---------------------------------------------------------------------------
# Mission generators
# ---------------------------------------------------------------------------

def generate_square(size_cm: float = 60.0) -> Mission:
    """4 waypoints forming a square with 90-degree turns at each corner."""
    waypoints = [
        Waypoint(size_cm, 0.0, heading=0.0),
        Waypoint(size_cm, size_cm, heading=math.pi / 2),
        Waypoint(0.0, size_cm, heading=math.pi),
        Waypoint(0.0, 0.0, heading=-math.pi / 2),
    ]
    return Mission(
        name="square",
        waypoints=waypoints,
        description=f"Drive a {size_cm}cm square and return to start",
    )


def generate_forward_back(distance_cm: float = 60.0) -> Mission:
    """Go forward, turn 180, come back."""
    waypoints = [
        Waypoint(distance_cm, 0.0, heading=0.0),
        Waypoint(distance_cm, 0.0, heading=math.pi),
        Waypoint(0.0, 0.0, heading=math.pi),
    ]
    return Mission(
        name="forward_back",
        waypoints=waypoints,
        description=f"Drive {distance_cm}cm forward, 180-turn, drive back",
    )


def generate_circle(radius_cm: float = 30.0, num_points: int = 8) -> Mission:
    """Approximate circle with N waypoints, heading tangent at each point."""
    waypoints: list[Waypoint] = []
    for i in range(num_points):
        angle = 2.0 * math.pi * i / num_points
        x = radius_cm * math.cos(angle)
        y = radius_cm * math.sin(angle)
        tangent = angle + math.pi / 2  # tangent direction
        waypoints.append(Waypoint(x, y, heading=tangent))
    # Close the loop — return to first point
    waypoints.append(Waypoint(waypoints[0].x, waypoints[0].y, heading=waypoints[0].heading))
    return Mission(
        name="circle",
        waypoints=waypoints,
        description=f"Drive a circle of radius {radius_cm}cm with {num_points} waypoints",
    )


def generate_goto(x_cm: float = 0.0, y_cm: float = 0.0) -> Mission:
    """Go to a specific point on the floor."""
    waypoints = [Waypoint(x_cm, y_cm, heading=None)]
    return Mission(
        name="goto",
        waypoints=waypoints,
        description=f"Drive to ({x_cm:.1f}, {y_cm:.1f}) cm",
    )


def get_available_missions() -> dict:
    """Return dict of mission name -> {description, params with defaults}."""
    return {
        "square": {
            "description": "Drive a square and return to start",
            "params": {"size_cm": 60.0},
        },
        "forward_back": {
            "description": "Drive forward, 180-turn, drive back",
            "params": {"distance_cm": 60.0},
        },
        "circle": {
            "description": "Drive an approximate circle",
            "params": {"radius_cm": 30.0, "num_points": 8},
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def angle_diff(a: float, b: float) -> float:
    """Shortest signed angle from *b* to *a*, wrapped to [-pi, pi]."""
    d = a - b
    return (d + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# PID controller
# ---------------------------------------------------------------------------

class PIDController:
    def __init__(self, kp: float, kd: float = 0.0, output_limit: float = 1.0):
        self.kp = kp
        self.kd = kd
        self.output_limit = output_limit
        self._prev_error: Optional[float] = None

    def update(self, error: float, dt: float) -> float:
        """Return clamped control output for the given error and timestep.

        The derivative term uses per-frame difference (not divided by dt)
        to avoid noise amplification at high loop rates.
        """
        derivative = 0.0
        if self._prev_error is not None:
            derivative = error - self._prev_error  # per-frame, not /dt
        self._prev_error = error

        output = self.kp * error + self.kd * derivative
        return max(-self.output_limit, min(self.output_limit, output))

    def reset(self) -> None:
        self._prev_error = None


# ---------------------------------------------------------------------------
# Path follower
# ---------------------------------------------------------------------------

class PathFollower:
    def __init__(
        self,
        heading_kp: float = 0.8,
        heading_kd: float = 0.02,
        throttle_kp: float = 1.0,
        waypoint_threshold_cm: float = 5.0,
        heading_threshold_rad: float = 0.26,
        max_steer_slew: float = 0.15,
        turn_in_place_threshold_rad: float = 0.5,  # ~30 degrees
    ):
        self._heading_pid = PIDController(kp=heading_kp, kd=heading_kd)
        self._throttle_kp = throttle_kp
        self._waypoint_threshold_cm = waypoint_threshold_cm
        self._heading_threshold_rad = heading_threshold_rad
        self._max_steer_slew = max_steer_slew
        self._turn_in_place_threshold = turn_in_place_threshold_rad
        self._prev_steering: float = 0.0

        self._mission: Optional[Mission] = None
        self._wp_index: int = 0
        self._active: bool = False

    # -- public API ----------------------------------------------------------

    def start_mission(self, mission: Mission) -> None:
        """Load a mission, reset state, begin following."""
        self._mission = mission
        self._wp_index = 0
        self._active = True
        self._prev_steering = 0.0
        self._heading_pid.reset()

    def _slew_limit(self, desired: float) -> float:
        """Apply slew rate limiting to steering output."""
        delta = desired - self._prev_steering
        delta = max(-self._max_steer_slew, min(self._max_steer_slew, delta))
        result = self._prev_steering + delta
        self._prev_steering = result
        return result

    def update(
        self, x_cm: float, y_cm: float, heading_rad: float, dt: float
    ) -> tuple[float, float, bool, str]:
        """Compute control outputs from current pose.

        Returns (throttle, steering, done, status_str) where throttle and
        steering are in [-1, 1].
        """
        if not self._active or self._mission is None:
            return 0.0, 0.0, True, "no active mission"

        waypoints = self._mission.waypoints
        if self._wp_index >= len(waypoints):
            self._active = False
            return 0.0, 0.0, True, "mission complete"

        wp = waypoints[self._wp_index]

        dx = wp.x - x_cm
        dy = wp.y - y_cm
        distance = math.hypot(dx, dy)

        # Desired heading toward waypoint
        desired_heading = math.atan2(dy, dx)
        heading_error = angle_diff(desired_heading, heading_rad)

        close_enough = distance < self._waypoint_threshold_cm

        # When close and waypoint specifies a target heading, rotate to it
        if close_enough and wp.heading is not None:
            target_heading_error = angle_diff(wp.heading, heading_rad)
            if abs(target_heading_error) < self._heading_threshold_rad:
                # Waypoint achieved — advance
                self._wp_index += 1
                self._heading_pid.reset()
                if self._wp_index >= len(waypoints):
                    self._active = False
                    return 0.0, 0.0, True, "mission complete"
                return 0.0, 0.0, False, f"advancing to waypoint {self._wp_index}"
            # Rotate in place toward target heading
            steering = self._heading_pid.update(target_heading_error, dt)
            steering = max(-1.0, min(1.0, steering))
            steering = self._slew_limit(steering)
            return 0.0, steering, False, (
                f"rotating at wp {self._wp_index} "
                f"herr={math.degrees(target_heading_error):.1f}°"
            )

        if close_enough:
            # No heading requirement — just advance
            self._wp_index += 1
            self._heading_pid.reset()
            if self._wp_index >= len(waypoints):
                self._active = False
                return 0.0, 0.0, True, "mission complete"
            return 0.0, 0.0, False, f"advancing to waypoint {self._wp_index}"

        # Tank drive: rotate in place first, then drive straight
        if abs(heading_error) > self._turn_in_place_threshold:
            # Phase 1: Rotate in place to face the waypoint (no throttle)
            steering = self._heading_pid.update(heading_error, dt)
            steering = max(-1.0, min(1.0, steering))
            steering = self._slew_limit(steering)
            status = (
                f"wp {self._wp_index} TURNING dist={distance:.1f}cm "
                f"herr={math.degrees(heading_error):.1f}°"
            )
            return 0.0, steering, False, status

        # Phase 2: Heading roughly aligned — drive forward with minor corrections
        steering = self._heading_pid.update(heading_error, dt)
        steering = max(-1.0, min(1.0, steering))
        steering = self._slew_limit(steering)

        # Throttle proportional to distance, capped
        throttle = self._throttle_kp * min(distance / 30.0, 1.0)
        throttle = max(-1.0, min(1.0, throttle))

        status = (
            f"wp {self._wp_index} DRIVING dist={distance:.1f}cm "
            f"herr={math.degrees(heading_error):.1f}° "
            f"thr={throttle:.2f} str={steering:.2f}"
        )
        return throttle, steering, False, status

    # -- properties ----------------------------------------------------------

    @property
    def current_waypoint_index(self) -> int:
        return self._wp_index

    @property
    def mission_progress(self) -> float:
        """Fraction of waypoints completed, 0.0 to 1.0."""
        if self._mission is None or not self._mission.waypoints:
            return 0.0
        return self._wp_index / len(self._mission.waypoints)

    @property
    def active(self) -> bool:
        return self._active


# ---------------------------------------------------------------------------
# IMU-assisted path follower
# ---------------------------------------------------------------------------

class IMUAssistedPathFollower(PathFollower):
    """Path follower that uses ESP32 gyro-assisted turns for fast, precise turning.

    Turn phase:  Sends turn(heading_delta) command to ESP32 → executes at 1kHz
    Drive phase: Trapezoidal velocity profile for maximum-speed navigation

    Requires a comms object with send_turn() and a sensor_fusion object
    for heading feedback.
    """

    # Turn states
    PHASE_TURN = "turning"
    PHASE_DRIVE = "driving"
    PHASE_WAIT_TURN = "wait_turn"  # waiting for ESP32 turn to complete

    def __init__(
        self,
        comms=None,
        sensor_fusion=None,
        telemetry=None,
        heading_kp: float = 0.8,
        heading_kd: float = 0.02,
        throttle_kp: float = 1.0,
        waypoint_threshold_cm: float = 5.0,
        heading_threshold_rad: float = 0.26,
        max_steer_slew: float = 0.15,
        turn_in_place_threshold_rad: float = 0.5,
        max_speed: float = 1.0,
        decel_distance_cm: float = 15.0,
    ):
        super().__init__(
            heading_kp=heading_kp,
            heading_kd=heading_kd,
            throttle_kp=throttle_kp,
            waypoint_threshold_cm=waypoint_threshold_cm,
            heading_threshold_rad=heading_threshold_rad,
            max_steer_slew=max_steer_slew,
            turn_in_place_threshold_rad=turn_in_place_threshold_rad,
        )
        self._comms = comms
        self._fusion = sensor_fusion
        self._telemetry = telemetry
        self._max_speed = max_speed
        self._decel_distance = decel_distance_cm
        self._phase = self.PHASE_TURN
        self._turn_target_heading = None
        self._turn_start_time = None
        self._turn_timeout = 3.0  # seconds

    def start_mission(self, mission: Mission) -> None:
        super().start_mission(mission)
        self._phase = self.PHASE_TURN

    def _use_imu_turn(self) -> bool:
        """Check if IMU-assisted turning is available.

        Disabled until ESP32 firmware supports 8-byte turn packets.
        Falls back to base PathFollower (vision-only PID steering).
        """
        return False

    def update(
        self, x_cm: float, y_cm: float, heading_rad: float, dt: float
    ) -> tuple[float, float, bool, str]:
        """Compute control outputs, using IMU-assisted turns when available.

        Falls back to base PathFollower behavior when IMU is not connected.
        """
        if not self._use_imu_turn():
            # Fallback: use base class (vision-only PID)
            return super().update(x_cm, y_cm, heading_rad, dt)

        if not self._active or self._mission is None:
            return 0.0, 0.0, True, "no active mission"

        waypoints = self._mission.waypoints
        if self._wp_index >= len(waypoints):
            self._active = False
            return 0.0, 0.0, True, "mission complete"

        wp = waypoints[self._wp_index]
        dx = wp.x - x_cm
        dy = wp.y - y_cm
        distance = math.hypot(dx, dy)
        desired_heading = math.atan2(dy, dx)
        heading_error = angle_diff(desired_heading, heading_rad)

        close_enough = distance < self._waypoint_threshold_cm

        # Check waypoint reached
        if close_enough:
            if wp.heading is not None:
                target_err = angle_diff(wp.heading, heading_rad)
                if abs(target_err) > self._heading_threshold_rad:
                    # Rotate to final heading using IMU turn
                    if self._phase != self.PHASE_WAIT_TURN:
                        delta_deg = math.degrees(target_err)
                        self._comms.send_turn(delta_deg)
                        self._phase = self.PHASE_WAIT_TURN
                        self._turn_start_time = __import__('time').perf_counter()
                        return 0.0, 0.0, False, f"IMU turn to wp heading {math.degrees(wp.heading):.0f}°"

                    # Waiting for turn to complete
                    elapsed = __import__('time').perf_counter() - self._turn_start_time
                    if abs(target_err) < self._heading_threshold_rad or elapsed > self._turn_timeout:
                        self._phase = self.PHASE_TURN
                        self._wp_index += 1
                        self._heading_pid.reset()
                        if self._wp_index >= len(waypoints):
                            self._active = False
                            return 0.0, 0.0, True, "mission complete"
                    return 0.0, 0.0, False, f"waiting for IMU turn ({elapsed:.1f}s)"

            # No heading requirement or heading achieved
            self._wp_index += 1
            self._heading_pid.reset()
            self._phase = self.PHASE_TURN
            if self._wp_index >= len(waypoints):
                self._active = False
                return 0.0, 0.0, True, "mission complete"
            return 0.0, 0.0, False, f"advancing to waypoint {self._wp_index}"

        # PHASE: Turn to face waypoint (IMU-assisted)
        if abs(heading_error) > self._turn_in_place_threshold:
            if self._phase != self.PHASE_WAIT_TURN:
                delta_deg = math.degrees(heading_error)
                self._comms.send_turn(delta_deg)
                self._phase = self.PHASE_WAIT_TURN
                self._turn_start_time = __import__('time').perf_counter()
                return 0.0, 0.0, False, (
                    f"wp {self._wp_index} IMU TURN {delta_deg:+.1f}°"
                )

            # Waiting for turn
            elapsed = __import__('time').perf_counter() - self._turn_start_time
            if abs(heading_error) < self._turn_in_place_threshold or elapsed > self._turn_timeout:
                self._phase = self.PHASE_DRIVE
            return 0.0, 0.0, False, f"wp {self._wp_index} waiting turn ({elapsed:.1f}s)"

        # PHASE: Drive toward waypoint with velocity profiling
        self._phase = self.PHASE_DRIVE
        steering = self._heading_pid.update(heading_error, dt)
        steering = max(-1.0, min(1.0, steering))
        steering = self._slew_limit(steering)

        # Trapezoidal-ish velocity profile: full speed then decel near target
        if distance < self._decel_distance:
            throttle = self._max_speed * (distance / self._decel_distance)
            throttle = max(0.15, throttle)  # keep above dead zone
        else:
            throttle = self._max_speed

        status = (
            f"wp {self._wp_index} DRIVING dist={distance:.1f}cm "
            f"herr={math.degrees(heading_error):.1f}° "
            f"thr={throttle:.2f} str={steering:.2f}"
        )
        return throttle, steering, False, status
