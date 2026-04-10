"""Hierarchical State Machine for combat mode using the transitions library."""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

from transitions.extensions import HierarchicalMachine

from battle_config import BattleConfig
from enemy_sides import (
    classify_approach_side,
    get_safe_approach_position,
    is_approach_safe,
    needs_flanking,
)
from match_timer import MatchTimer, PinTimer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BattleContext:
    """Sensor snapshot passed to tick() every frame."""
    our_pos: tuple[float, float] = (0.0, 0.0)
    our_heading_rad: float = 0.0
    our_velocity: tuple[float, float] = (0.0, 0.0)
    enemy_pos: tuple[float, float] | None = None
    enemy_heading_rad: float | None = None
    enemy_velocity: tuple[float, float] | None = None
    enemy_detected: bool = False
    enemy_tracking: bool = False
    frames_without_detection: int = 999
    distance_cm: float = 999.0
    dt: float = 0.016
    our_detected: bool = False  # our ArUco visible
    accel_x_mg: float = 0.0    # forward acceleration in milligravity
    accel_y_mg: float = 0.0    # lateral acceleration in milligravity
    throttle_cmd: float = 0.0  # what we're commanding (for stuck detection)
    imu_pitch_deg: float = 0.0   # IMU pitch (for flip detection)
    imu_roll_deg: float = 0.0    # IMU roll (for flip detection)
    imu_heading_deg: float = 0.0  # IMU heading (for dead-reckoning)


@dataclass
class BattleOutput:
    """Motor command output from the state machine."""
    throttle: float = 0.0
    steering: float = 0.0
    buttons: int = 0
    # Rate mode: when set, ESP32 holds this angular velocity at 3.33kHz
    target_omega_dps: float | None = None  # None = legacy direct mode
    target_speed: float = 0.0              # forward speed for rate mode


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _angle_diff(a: float, b: float) -> float:
    """Shortest signed angle from b to a, wrapped to [-pi, pi]."""
    d = a - b
    return (d + math.pi) % (2.0 * math.pi) - math.pi


def _point_to_segment_dist(px, py, ax, ay, bx, by):
    """Distance from point (px,py) to line segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _near_wall(x: float, y: float, threshold: float, arena_corners=None) -> bool:
    """Check if a position is near the arena wall.

    Uses actual distance to wall polygon if arena_corners is provided,
    otherwise falls back to simple coordinate threshold.

    When using real wall distance, threshold is clamped to max 40cm
    because the old abs(x) > N semantics don't translate to wall distance
    (80cm from a wall is the middle of the arena, not "near wall").
    """
    if arena_corners and len(arena_corners) >= 3:
        wall_threshold = min(threshold, 15.0)  # real distance: robot half-depth from wall
        min_dist = float('inf')
        n = len(arena_corners)
        for i in range(n):
            ax, ay = arena_corners[i]
            bx, by = arena_corners[(i + 1) % n]
            d = _point_to_segment_dist(x, y, ax, ay, bx, by)
            if d < min_dist:
                min_dist = d
        return min_dist < wall_threshold
    return abs(x) > threshold or abs(y) > threshold


# ---------------------------------------------------------------------------
# BattleController — the HSM model
# ---------------------------------------------------------------------------

# State definitions for HierarchicalMachine
_STATES = [
    "wait",
    "goto_center",
    "acquire",
    {
        "name": "charge",
        "children": ["pursue", "flank", "reorient"],
        "initial": "pursue",
    },
    "pin",
    {
        "name": "pit",
        "children": ["position", "push", "commit", "abort"],
        "initial": "position",
    },
    {
        "name": "evade",
        "children": ["retreat", "reposition"],
        "initial": "retreat",
    },
    "wall_reverse",
    "unstick",
    "lost_target",
    "lost_aruco",
    "victory_dance",
]


class BattleController:
    """HSM-based combat controller.

    Call tick(ctx) every frame. It evaluates the current state,
    runs its action function, and returns a BattleOutput with
    throttle/steering/buttons.
    """

    def __init__(
        self,
        config: BattleConfig,
        match_timer: MatchTimer,
        pin_timer: PinTimer,
        arena_corners=None,
    ):
        self.cfg = config
        self.match_timer = match_timer
        self.pin_timer = pin_timer
        self._arena_corners = arena_corners  # calibrated wall polygon for _near_wall

        # Internal tracking
        self._acquire_count = 0
        self._prev_steer = 0.0
        self._lost_timer: float | None = None
        self._lost_rotating = False
        self._unstick_timer: float | None = None
        self._unstick_phase = 1  # +1 or -1
        self._unstick_toggle_t = 0.0
        self._retreat_timer: float | None = None
        self._aruco_lost_frames = 0
        self._last_positions: deque[tuple[float, float, float]] = deque()
        self._last_enemy_pos: tuple[float, float] | None = None
        self._reposition_timer: float | None = None
        self._reorient_timer: float | None = None
        self._log_t = 0.0

        # Pit strategy state
        self._pit_abort_timer: float | None = None

        # Stuck detection
        self._stuck_accel_frames = 0
        self._flank_reversing = False
        self._unstick_start_pos: tuple[float, float] | None = None
        self._last_retreat_reason: str | None = None

        # Pin state
        self._pin_entry_time: float = 0.0

        # Passive impact cooldown
        self._last_impact_time: float = 0.0

        # Phase tracking
        self._prev_phase = "start"

        # Stall detection (3 named floats — zero allocation)
        self._speed_t0 = 0.0
        self._speed_t1 = 0.0
        self._speed_t2 = 0.0
        self._push_commit_timer: float | None = None
        self._push_commit_enemy_close = False

        # ArUco loss recovery
        self._last_aruco_x = 0.0
        self._last_aruco_y = 0.0
        self._last_aruco_heading = 0.0
        self._aruco_dead_reckon_start: float | None = None

        # Wall reverse
        self._wall_reverse_timer: float | None = None
        self._wall_reverse_start_pos: tuple[float, float] | None = None

        # Passive impact detection
        self._hit_count = 0

        # Victory dance
        self._victory_start: float | None = None
        self._dance_finished = False
        self._pin_count = 0

        # Recovery cycle breaker
        self._recovery_cycle_count = 0

        # Goto center
        self._goto_center_target: tuple[float, float] | None = None

        # Lost ArUco recovery
        self._lost_aruco_t: float = 0.0
        self._lost_aruco_phase: int = -1
        self._lost_aruco_target_heading: float = 0.0

        # Reorient cycle detection
        self._reorient_count: int = 0
        self._reorient_window_start: float = 0.0
        self._reorient_heading_at_entry: float = 0.0

        # Wall stuck counter
        self._wall_stuck_frames: int = 0

        # Build the HSM
        self.machine = HierarchicalMachine(
            model=self,
            states=_STATES,
            initial="wait",
            auto_transitions=False,
        )

        # Action map — built once, not per-tick
        self._action_map = {
            "wait": self._action_wait,
            "goto_center": self._action_goto_center,
            "acquire": self._action_acquire,
            "charge_pursue": self._action_charge_pursue,
            "charge_flank": self._action_charge_flank,
            "charge_reorient": self._action_charge_reorient,
            "pin": self._action_pin,
            "pit_position": self._action_pit_position,
            "pit_push": self._action_pit_push,
            "pit_commit": self._action_pit_commit,
            "pit_abort": self._action_pit_abort,
            "evade_retreat": self._action_evade_retreat,
            "evade_reposition": self._action_evade_reposition,
            "wall_reverse": self._action_wall_reverse,
            "unstick": self._action_unstick,
            "lost_target": self._action_lost_target,
            "lost_aruco": self._action_lost_aruco,
            "victory_dance": self._action_victory_dance,
        }

    # -- Public API ---------------------------------------------------------

    def tick(self, ctx: BattleContext) -> BattleOutput:
        """Main entry point — call once per frame."""
        now = time.perf_counter()

        # Victory dance: match expired and we fought (had pins)
        current = self.state
        if (self.match_timer.is_expired
                and current != "victory_dance"
                and current != "wait"
                and self._pin_count > 0):
            self.machine.set_state("victory_dance")
            self._victory_start = now
            log.info("[battle] VICTORY DANCE — match over with %d pins", self._pin_count)
            return self._action_victory_dance(ctx, now)

        # Track ArUco visibility
        if ctx.our_detected:
            self._aruco_lost_frames = 0
            self._last_aruco_x = ctx.our_pos[0]
            self._last_aruco_y = ctx.our_pos[1]
            self._last_aruco_heading = ctx.our_heading_rad
            self._aruco_dead_reckon_start = None
        else:
            self._aruco_lost_frames += 1

        # Track position history for stuck detection (deque, no per-tick alloc)
        if ctx.our_detected:
            self._last_positions.append((ctx.our_pos[0], ctx.our_pos[1], now))
            cutoff = now - 1.5
            while self._last_positions and self._last_positions[0][2] < cutoff:
                self._last_positions.popleft()

        # Track velocity for stall detection (3 named floats, clamped to physical max)
        self._speed_t2 = self._speed_t1
        self._speed_t1 = self._speed_t0
        self._speed_t0 = min(math.hypot(ctx.our_velocity[0], ctx.our_velocity[1]), 120.0)

        # Remember last enemy position
        if ctx.enemy_detected and ctx.enemy_pos is not None:
            self._last_enemy_pos = ctx.enemy_pos

        # Phase tracking
        phase = self.match_timer.phase
        if phase != self._prev_phase:
            log.info("[battle] PHASE: %s -> %s", self._prev_phase, phase)
            self._prev_phase = phase

        # --- Global transitions (checked every tick) ---
        current = self.state

        # Skip global guards for non-combat states
        if current in ("wait", "victory_dance", "lost_aruco"):
            pass
        else:
            # Passive impact detection — "we got hit" while not commanding throttle
            # 5000mg threshold (vibration is 1000-3000mg), cooldown 200ms, exclude high-energy states
            accel_mag = math.hypot(ctx.accel_x_mg, ctx.accel_y_mg)
            if (accel_mag > 5000 and abs(ctx.throttle_cmd) < 0.2
                    and current not in ("charge_pursue", "charge_reorient", "pin", "pit_push", "pit_commit")
                    and (now - self._last_impact_time) > 0.2):
                self._last_impact_time = now
                self._hit_count += 1
                log.info("[battle] PASSIVE IMPACT: %.0fmg (hit #%d)", accel_mag, self._hit_count)
                if current == "acquire":
                    self._enter_retreat(reason="passive_impact")
                    return self._action_evade_retreat(ctx, now)
                elif current == "lost_target":
                    # Enemy found us
                    self.machine.set_state("acquire")
                    self._acquire_count = 5
                    return BattleOutput()

            # Stall detection — velocity-based
            stall = (self._speed_t2 > 15.0
                     and self._speed_t1 < self.cfg.stall_speed_threshold
                     and self._speed_t0 < self.cfg.stall_speed_threshold)

            if stall and abs(ctx.throttle_cmd) > 0.3:
                in_push_state = current in ("charge_pursue", "pit_push", "pit_commit")
                enemy_close = ctx.enemy_detected and ctx.distance_cm < self.cfg.charge_close_range_cm * 1.5

                if in_push_state and enemy_close:
                    # Push commit — keep pushing, start timer if not already
                    if self._push_commit_timer is None:
                        self._push_commit_timer = now
                        self._push_commit_enemy_close = True
                        log.info("[battle] PUSH COMMIT — stall detected, committing for %.1fs", self.cfg.push_commit_s)
                elif _near_wall(ctx.our_pos[0], ctx.our_pos[1], self.cfg.wall_threshold_cm, self._arena_corners):
                    # No enemy nearby + at wall = wall crash
                    self._enter_wall_reverse(ctx, now)
                    return self._action_wall_reverse(ctx, now)

            # Push commit timer evaluation
            if self._push_commit_timer is not None:
                # Moving again? Cancel commit
                if self._speed_t0 > 15.0:
                    log.info("[battle] PUSH COMMIT cancelled — moving again")
                    self._push_commit_timer = None
                elif now - self._push_commit_timer > self.cfg.push_commit_s:
                    # Timer expired — evaluate context
                    self._push_commit_timer = None
                    if ctx.enemy_detected and ctx.enemy_pos is not None:
                        ex, ey = ctx.enemy_pos
                        if _near_wall(ex, ey, self.cfg.wall_threshold_cm, self._arena_corners):
                            # Pin!
                            self._enter_pin(now)
                            return self._action_pin(ctx, now)
                    # No pin — reorient to try a different angle
                    # (wall_reverse makes no sense if we're not at a wall)
                    our_near_wall = _near_wall(ctx.our_pos[0], ctx.our_pos[1],
                                               self.cfg.wall_threshold_cm, self._arena_corners)
                    if our_near_wall:
                        self._enter_wall_reverse(ctx, now)
                        return self._action_wall_reverse(ctx, now)
                    else:
                        log.info("[battle] PUSH stall in open — reorienting")
                        self.machine.set_state("charge_reorient")
                        self._reorient_timer = now
                        return self._action_charge_reorient(ctx, now)

            # ArUco lost recovery — dead-reckon reverse (cheap frame counter check)
            # Pin + charge_pursue excluded: ArUco loss during engagement is expected
            if current not in ("evade_retreat", "evade_reposition", "wall_reverse", "unstick", "pin", "charge_pursue"):
                if self._aruco_lost_frames > 60:
                    if self._aruco_lost_frames > 180:
                        # Give up — enter lost_aruco
                        if current != "lost_aruco":
                            self.machine.set_state("lost_aruco")
                            self._lost_aruco_t = 0.0  # reset so _action_lost_aruco starts fresh
                            log.info("[battle] LOST ARUCO — giving up, waiting for re-acquisition")
                        return self._action_lost_aruco(ctx, now)
                    else:
                        # Dead-reckon reverse phase (frames 60-180)
                        return self._dead_reckon_reverse(ctx, now)

            # Enemy lost → lost_target (from combat states — cheap check)
            # Pin excluded: enemy hidden under us during pin is expected, pin timer handles exit
            if current in ("charge_pursue", "charge_flank",
                           "charge_reorient",
                           "pit_position", "pit_push"):
                if not ctx.enemy_detected and ctx.frames_without_detection > 30:
                    self._enter_lost_target(now)
                    return self._action_lost_target(ctx, now)

            # Stuck at wall — uses position history displacement, NOT KF velocity
            # (KF velocity has phantom readings from ArUco jitter + arena clamping)
            if (current not in ("unstick", "evade_retreat", "evade_reposition", "wall_reverse", "pin")
                    and abs(ctx.throttle_cmd) > 0.15
                    and len(self._last_positions) >= 10):
                oldest = self._last_positions[0]
                newest = self._last_positions[-1]
                dt_pos = newest[2] - oldest[2]
                if dt_pos > 0.5:  # need at least 0.5s of position history
                    displacement = math.hypot(newest[0] - oldest[0], newest[1] - oldest[1])
                    at_wall = _near_wall(newest[0], newest[1], self.cfg.wall_threshold_cm, self._arena_corners)
                    if at_wall and displacement < 5.0:
                        self._wall_stuck_frames += 1
                        if self._wall_stuck_frames > 5:  # confirmed stuck (already have 0.5s of history)
                            self._wall_stuck_frames = 0
                            log.info("[battle] STUCK AT WALL — pos=(%.0f,%.0f) disp=%.1fcm in %.1fs",
                                     newest[0], newest[1], displacement, dt_pos)
                            self._enter_wall_reverse(ctx, now)
                            return self._action_wall_reverse(ctx, now)
                    else:
                        self._wall_stuck_frames = 0
                else:
                    pass  # not enough history yet, don't reset counter

            # Stuck detection — expensive (position history scan), check last
            if (current not in ("unstick", "evade_retreat", "evade_reposition", "wall_reverse", "pin",
                               "pit_position", "pit_push", "pit_commit")
                    and self._push_commit_timer is None):
                if self._is_stuck(ctx):
                    self._enter_unstick()
                    return self._action_unstick(ctx, now)

        # --- State-specific action + transitions ---
        action = self._action_map.get(current)
        if action:
            result = action(ctx, now)
            # Global throttle cap (tune down for testing)
            MAX_THROTTLE = 0.75
            result.throttle = max(-MAX_THROTTLE, min(MAX_THROTTLE, result.throttle))
            return result

        # Fallback — should not reach here
        return BattleOutput()

    @property
    def debug_info(self) -> dict:
        """Expose internal state for frame logging."""
        return {
            "stuck_frames": self._stuck_accel_frames,
            "unstick_phase": self._unstick_phase,
            "aruco_lost": self._aruco_lost_frames,
            "retreat_reason": self._last_retreat_reason,
            "phase": self.match_timer.phase,
            "opening": self.cfg.opening_strategy,
            "push_commit_active": self._push_commit_timer is not None,
            "hit_count": self._hit_count,
            "stall_speed": round(self._speed_t0, 1),
        }

    @property
    def is_dance_finished(self) -> bool:
        """For main.py to read — True when victory dance is complete."""
        return self._dance_finished

    def start_match(self, ctx: BattleContext) -> None:
        """Called by main.py when match begins. Routes based on opening strategy."""
        opening = self.cfg.opening_strategy
        if opening == "center":
            # Guard: if pit is near center, skip
            pit_center_dist = math.hypot(self.cfg.pit_x_cm, self.cfg.pit_y_cm)
            if pit_center_dist < self.cfg.pit_danger_radius_cm:
                log.warning("[battle] Center opening skipped — pit near center, falling back to charge")
                opening = "charge"
            else:
                self._goto_center_target = (0.0, 0.0)
                self.machine.set_state("goto_center")
                log.info("[battle] START — opening: center")
                return

        if ctx.enemy_detected:
            self.machine.set_state("acquire")
            self._acquire_count = 1
            log.info("[battle] START — opening: %s, enemy detected", opening)
        else:
            self._enter_lost_target(time.perf_counter())
            log.info("[battle] START — opening: %s, no enemy yet", opening)

    def enter_victory_dance(self) -> None:
        """Called by main.py when match timer expires."""
        self.machine.set_state("victory_dance")
        self._victory_start = time.perf_counter()
        self._dance_finished = False
        log.info("[battle] VICTORY DANCE — match over")

    def reset(self) -> None:
        """Reset to wait state for a new match."""
        self.machine.set_state("wait")
        self._acquire_count = 0
        self._prev_steer = 0.0
        self._lost_timer = None
        self._lost_rotating = False
        self._unstick_timer = None
        self._retreat_timer = None
        self._aruco_lost_frames = 0
        self._last_positions.clear()
        self._last_enemy_pos = None
        self._reposition_timer = None
        self._reorient_timer = None
        self._pit_abort_timer = None
        self._push_commit_timer = None
        self._aruco_dead_reckon_start = None
        self._wall_reverse_timer = None
        self._hit_count = 0
        self._victory_start = None
        self._dance_finished = False
        self._speed_t0 = self._speed_t1 = self._speed_t2 = 0.0
        self._prev_phase = "start"
        self._stuck_accel_frames = 0
        self._last_retreat_reason = None
        self._pin_entry_time = 0.0
        self._last_impact_time = 0.0
        self._log_t = 0.0
        self._lost_aruco_t = 0.0
        self._reorient_count = 0
        self._reorient_window_start = 0.0
        self._reorient_heading_at_entry = 0.0
        self.pin_timer.reset()

    def _pit_distance(self, x, y):
        """Distance from position to pit center."""
        return math.hypot(x - self.cfg.pit_x_cm, y - self.cfg.pit_y_cm)

    def _pit_away_angle(self, x, y):
        """Angle pointing away from pit center."""
        return math.atan2(y - self.cfg.pit_y_cm, x - self.cfg.pit_x_cm)

    # -- Smart re-engagement ------------------------------------------------

    def _reengage(self, ctx: BattleContext) -> None:
        """Route to the best state based on current tracking status."""
        self._recovery_cycle_count = 0
        if ctx.enemy_tracking and ctx.enemy_pos is not None:
            if self.cfg.strategy == "pit":
                self.machine.set_state("pit_position")
            else:
                self.machine.set_state("charge_pursue")
                self._prev_steer = 0.0
                self._acquire_count = self.cfg.acquire_frames + 1
            log.info("[battle] Re-engaging — enemy still tracked")
        elif ctx.enemy_detected:
            self.machine.set_state("acquire")
            self._acquire_count = max(self._acquire_count, 1)
        else:
            self._enter_lost_target(time.perf_counter())

    # -- Stuck detection (IMU + position) -----------------------------------

    def _is_stuck(self, ctx: BattleContext) -> bool:
        """Detect if robot is stuck using IMU + position history."""
        if abs(ctx.throttle_cmd) > 0.3:
            accel_mag = math.hypot(ctx.accel_x_mg, ctx.accel_y_mg)
            if accel_mag < 80:
                self._stuck_accel_frames += 1
            else:
                self._stuck_accel_frames = 0
            if self._stuck_accel_frames > 30:
                self._stuck_accel_frames = 0
                return True
        else:
            self._stuck_accel_frames = 0

        if abs(ctx.throttle_cmd) < 0.1:
            return False
        if len(self._last_positions) < 10:
            return False
        oldest = self._last_positions[0]
        newest = self._last_positions[-1]
        dt = newest[2] - oldest[2]
        if dt < 0.8:
            return False
        displacement = math.hypot(newest[0] - oldest[0], newest[1] - oldest[1])
        return displacement < 3.0

    # -- Dead-reckon reverse (ArUco lost) -----------------------------------

    def _dead_reckon_reverse(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Reverse toward last known ArUco position using IMU heading."""
        if self._aruco_dead_reckon_start is None:
            self._aruco_dead_reckon_start = now
            log.info("[battle] DEAD RECKON — reversing toward last ArUco pos")

        # Compute heading FROM current estimated position TO last known position
        # Use IMU heading since ArUco is lost
        imu_heading_rad = math.radians(ctx.imu_heading_deg)

        # We want to go TOWARD _last_aruco_pos, but in reverse
        # Desired heading = toward last pos, drive backward
        dx = self._last_aruco_x - ctx.our_pos[0]
        dy = self._last_aruco_y - ctx.our_pos[1]
        desired_heading = math.atan2(dy, dx)

        # Reverse: we want our tail pointing toward the target
        reverse_heading = desired_heading + math.pi
        alpha = _angle_diff(reverse_heading, imu_heading_rad)

        # Check if flipped — invert steering if so
        flipped = abs(ctx.imu_roll_deg) > 90 or abs(ctx.imu_pitch_deg) > 90
        steer_sign = -1.0 if flipped else 1.0

        omega = steer_sign * (-150.0 * alpha)
        omega = max(-200.0, min(200.0, omega))

        return BattleOutput(target_omega_dps=omega, target_speed=-0.3)

    # -- State entry helpers ------------------------------------------------

    def _enter_unstick(self) -> None:
        self._recovery_cycle_count += 1
        if self._recovery_cycle_count >= 3:
            # Break the wall_reverse/unstick loop — retreat to open space
            log.info("[battle] RECOVERY LOOP BREAK — forcing retreat after %d cycles",
                     self._recovery_cycle_count)
            self._recovery_cycle_count = 0
            self._enter_retreat(reason="recovery_loop")
            return
        self.machine.set_state("unstick")
        self._unstick_timer = time.perf_counter()
        self._unstick_phase = 1
        self._unstick_toggle_t = time.perf_counter()
        self._last_positions.clear()
        log.info("[battle] UNSTICK — oscillating to free (cycle %d)", self._recovery_cycle_count)

    def _enter_retreat(self, reason: str = "aruco_lost") -> None:
        self.machine.set_state("evade_retreat")
        self._retreat_timer = time.perf_counter()
        self._last_retreat_reason = reason
        if reason == "aruco_lost":
            self._aruco_lost_frames = 0
        log.info("[battle] RETREAT — %s", reason)

    def _enter_lost_target(self, now: float) -> None:
        self.machine.set_state("lost_target")
        self._lost_timer = now
        self._lost_rotating = False
        log.info("[battle] LOST TARGET — driving to last known position")

    def _enter_wall_reverse(self, ctx: BattleContext, now: float) -> None:
        self._recovery_cycle_count += 1
        self.machine.set_state("wall_reverse")
        self._wall_reverse_timer = now
        self._wall_reverse_start_pos = (ctx.our_pos[0], ctx.our_pos[1]) if ctx.our_detected else None
        self._push_commit_timer = None
        log.info("[battle] WALL REVERSE — backing up (cycle %d)", self._recovery_cycle_count)

    def _enter_pin(self, now: float) -> None:
        self.machine.set_state("pin")
        self.pin_timer.start()
        self._pin_entry_time = now
        self._pin_count += 1
        # Don't reset recovery counter here — short micro-pins (0.5s) along walls
        # shouldn't count as "successful recovery" for the cycle breaker
        log.info("[battle] PIN — holding (#%d)", self._pin_count)

    # -- Action functions ---------------------------------------------------

    def _action_wait(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Pre-match idle — motors off."""
        return BattleOutput()

    def _action_goto_center(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Drive to arena center (opening strategy)."""
        # Enemy detected while driving — acquire immediately
        if ctx.enemy_detected:
            self.machine.set_state("acquire")
            self._acquire_count = 1
            return BattleOutput()

        # ArUco loss guard
        if self._aruco_lost_frames > 60:
            return self._dead_reckon_reverse(ctx, now)

        target = self._goto_center_target or (0.0, 0.0)
        dx = target[0] - ctx.our_pos[0]
        dy = target[1] - ctx.our_pos[1]
        dist = math.hypot(dx, dy)

        if dist < 15.0:
            # Arrived at center
            self._enter_lost_target(now)
            return BattleOutput()

        desired_heading = math.atan2(dy, dx)
        alpha = _angle_diff(desired_heading, ctx.our_heading_rad)
        steering = max(-0.4, min(0.4, alpha * 0.4))
        return BattleOutput(throttle=0.5, steering=steering)

    def _action_acquire(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Validate detection over multiple frames before committing."""
        urgency = self.match_timer.urgency
        phase = self.match_timer.phase

        # Phase-adjusted acquire frames
        if phase == "final":
            required = max(5, int(self.cfg.acquire_frames * (1.0 - 0.7 * urgency)))
        else:
            required = max(5, int(self.cfg.acquire_frames * (1.0 - 0.5 * urgency)))

        # Fast_pin opening: reduced acquire
        opening = self.cfg.opening_strategy
        if phase == "start" and opening == "fast_pin":
            required = max(5, required // 2)

        if ctx.enemy_detected:
            self._acquire_count += 1
        else:
            self._acquire_count = max(0, self._acquire_count - 2)

        if self._acquire_count <= 0:
            self._enter_lost_target(now)
            return BattleOutput()

        if self._acquire_count >= required:
            # Locked on — choose routing based on phase + strategy
            if phase == "start" and opening == "avoid":
                self.machine.set_state("evade_reposition")
                self._reposition_timer = now
            elif self.cfg.strategy == "pit":
                self.machine.set_state("pit_position")
            elif self.cfg.strategy == "evade":
                self.machine.set_state("evade_reposition")
                self._reposition_timer = now
            else:
                self.machine.set_state("charge_pursue")
                self._prev_steer = 0.0
            return BattleOutput()

        return BattleOutput()

    def _action_charge_pursue(self, ctx: BattleContext, now: float) -> BattleOutput:
        """PN guidance toward enemy — throttle scales with distance, max at close range."""
        if ctx.enemy_pos is None:
            # Brief dropout — drive toward last known position instead of stopping
            if self._last_enemy_pos is not None:
                desired = math.atan2(
                    self._last_enemy_pos[1] - ctx.our_pos[1],
                    self._last_enemy_pos[0] - ctx.our_pos[0],
                )
                alpha = _angle_diff(desired, ctx.our_heading_rad)
                omega = -200.0 * alpha
                omega = max(-300.0, min(300.0, omega))
                return BattleOutput(target_omega_dps=omega, target_speed=0.4)
            return BattleOutput()

        urgency = self.match_timer.urgency
        phase = self.match_timer.phase

        # Close range + near wall → PIN
        if ctx.distance_cm < self.cfg.charge_close_range_cm:
            ex, ey = ctx.enemy_pos
            if _near_wall(ex, ey, self.cfg.wall_threshold_cm, self._arena_corners):
                self._enter_pin(now)
                return self._action_pin(ctx, now)

        # Pure pursuit arc
        desired_heading = math.atan2(
            ctx.enemy_pos[1] - ctx.our_pos[1],
            ctx.enemy_pos[0] - ctx.our_pos[0],
        )
        alpha = _angle_diff(desired_heading, ctx.our_heading_rad)

        # Rate mode: compute desired angular velocity
        Kp_omega = 200.0
        omega = -Kp_omega * alpha
        omega = max(-300.0, min(300.0, omega))

        alpha_abs = abs(alpha)

        # Diagnostic logging every 0.5s
        if now - self._log_t > 0.5:
            self._log_t = now
            log.info("[pursue] pos=(%.0f,%.0f) h=%.0f° enemy=(%.0f,%.0f) alpha=%.0f° omega=%.0f dist=%.0f",
                     ctx.our_pos[0], ctx.our_pos[1], math.degrees(ctx.our_heading_rad),
                     ctx.enemy_pos[0], ctx.enemy_pos[1],
                     math.degrees(alpha), omega, ctx.distance_cm)

        # Reorient if facing away from enemy
        if alpha_abs > math.radians(110):
            # Track reorient cycles — if too many in a short window, give up
            if now - self._reorient_window_start > 8.0:
                self._reorient_count = 0
                self._reorient_window_start = now
            self._reorient_count += 1
            if self._reorient_count >= 3:
                log.info("[pursue] REORIENT cycling %d times in 8s — dropping to lost_target",
                         self._reorient_count)
                self._reorient_count = 0
                self._enter_lost_target(now)
                return self._action_lost_target(ctx, now)

            log.info("[pursue] REORIENT triggered: alpha=%.0f°", math.degrees(alpha))
            self.machine.set_state("charge_reorient")
            self._reorient_timer = now
            self._reorient_heading_at_entry = ctx.our_heading_rad
            return self._action_charge_reorient(ctx, now)

        # Throttle scaling: faster at close range, PN guidance at all distances
        # cos² base ensures we slow for turns
        speed = math.cos(alpha) ** 2
        speed *= (1.0 - min(abs(omega) / 300.0, 1.0) * 0.3)

        # Distance-based throttle boost — max at close range
        if ctx.distance_cm < self.cfg.charge_close_range_cm * 3:
            speed = max(speed, 0.8)
        if ctx.distance_cm < self.cfg.charge_close_range_cm:
            speed = 1.0

        # Fast pin opening: always max throttle
        if phase == "start" and self.cfg.opening_strategy == "fast_pin":
            speed = max(speed, 0.9)

        # FINAL phase: more aggressive
        if phase == "final":
            speed = max(speed, 0.6)

        speed = max(0.15, speed)

        return BattleOutput(target_omega_dps=omega, target_speed=speed)

    def _action_charge_reorient(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Back up briefly, spin to face enemy, then resume pursuit."""
        if self._reorient_timer is None:
            self._reorient_timer = now
            self._reorient_heading_at_entry = ctx.our_heading_rad

        elapsed = now - self._reorient_timer

        # Abort if enemy closes to contact
        if ctx.enemy_detected and ctx.distance_cm < 10:
            self._reorient_timer = None
            self.machine.set_state("charge_pursue")
            return BattleOutput()  # next tick runs charge_pursue (avoid recursion)

        # Phase 1: Backup (0-0.25s)
        if elapsed < 0.25:
            return BattleOutput(target_omega_dps=0.0, target_speed=-0.35)

        # Hard timeout — applies regardless of enemy_pos
        if elapsed > 1.5:
            self._reorient_timer = None
            self.machine.set_state("charge_pursue")
            return self._action_charge_pursue(ctx, now)

        # Stall detection: commanding spin but heading hasn't changed
        # If heading moved less than 15° after 0.75s of spinning, we're stuck
        if elapsed > 0.75:
            heading_delta = abs(_angle_diff(ctx.our_heading_rad, self._reorient_heading_at_entry))
            if heading_delta < math.radians(15):
                log.info("[reorient] STALLED — heading moved only %.0f° in %.1fs, entering unstick",
                         math.degrees(heading_delta), elapsed)
                self._reorient_timer = None
                self._enter_unstick()
                return self._action_unstick(ctx, now)

        # Phase 2: Spin to face enemy
        if ctx.enemy_pos is not None:
            desired = math.atan2(
                ctx.enemy_pos[1] - ctx.our_pos[1],
                ctx.enemy_pos[0] - ctx.our_pos[0],
            )
            alpha = _angle_diff(desired, ctx.our_heading_rad)

            if abs(alpha) < math.radians(40):
                self._reorient_timer = None
                self._reorient_count = 0  # successful reorient — reset cycle counter
                self.machine.set_state("charge_pursue")
                return self._action_charge_pursue(ctx, now)

            omega = -300.0 if alpha > 0 else 300.0
            return BattleOutput(target_omega_dps=omega, target_speed=0.25)

        return BattleOutput(target_omega_dps=300.0, target_speed=0.25)

    def _action_charge_flank(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Arc around to the enemy's safe side before committing."""
        if ctx.enemy_pos is None or ctx.enemy_heading_rad is None:
            self.machine.set_state("charge_pursue")
            return self._action_charge_pursue(ctx, now)

        if is_approach_safe(ctx.our_pos, ctx.enemy_pos, ctx.enemy_heading_rad, self.cfg.safe_side):
            self.machine.set_state("charge_pursue")
            self._prev_steer = 0.0
            return self._action_charge_pursue(ctx, now)

        target = get_safe_approach_position(
            ctx.enemy_pos, ctx.enemy_heading_rad, self.cfg.safe_side, distance_cm=40.0
        )
        desired_heading = math.atan2(
            target[1] - ctx.our_pos[1],
            target[0] - ctx.our_pos[0],
        )
        alpha = _angle_diff(desired_heading, ctx.our_heading_rad)

        Kp_omega = 120.0
        omega = -Kp_omega * alpha
        omega = max(-300.0, min(300.0, omega))

        alpha_abs = abs(alpha)
        if alpha_abs > math.radians(100):
            self._flank_reversing = True
        elif alpha_abs < math.radians(80):
            self._flank_reversing = False

        if not self._flank_reversing:
            speed = math.cos(alpha) * 0.8
        else:
            reverse_alpha = math.pi - alpha_abs
            speed = -math.cos(reverse_alpha) * 0.8

        speed *= (1.0 - min(abs(omega) / 300.0, 1.0) * 0.2)
        if speed >= 0:
            speed = max(0.15, speed)
        else:
            speed = min(-0.15, speed)

        return BattleOutput(target_omega_dps=omega, target_speed=speed)

    def _action_pin(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Hold enemy against wall with low forward pressure."""
        # Pin timer expired → back up to re-acquire ArUco
        if self.pin_timer.is_expired:
            self.pin_timer.reset()
            # Always retreat after pin — this backs us away from the wall
            # so ArUco becomes visible again (marker was obscured during pin)
            self._enter_retreat(reason="pin_release")
            return self._action_evade_retreat(ctx, now)

        # NO ArUco-loss exit during pin — ArUco loss is EXPECTED
        # (marker pressed against wall/opponent). Pin timer is the only timeout.

        # Enemy escaped or no longer at wall? Exit pin early.
        # (0.5s grace period after pin entry to avoid bounce oscillation)
        pin_elapsed = now - self._pin_entry_time
        if pin_elapsed > 0.5 and ctx.enemy_tracking and ctx.our_detected:
            escaped = ctx.distance_cm > self.cfg.pin_escape_range_cm
            enemy_left_wall = False
            if ctx.enemy_pos is not None:
                enemy_left_wall = not _near_wall(
                    ctx.enemy_pos[0], ctx.enemy_pos[1],
                    self.cfg.wall_threshold_cm, self._arena_corners)
            if escaped or enemy_left_wall:
                self.pin_timer.reset()
                self.machine.set_state("charge_pursue")
                self._prev_steer = 0.0
                self._acquire_count = self.cfg.acquire_frames + 1
                reason = "escaped" if escaped else "left wall"
                log.info("[battle] PIN — enemy %s (%.0fcm), re-engaging", reason, ctx.distance_cm)
                return self._action_charge_pursue(ctx, now)

        # Hold with low forward pressure (0.2) — opponent fights back
        # Pulse to 0.4 if enemy starting to escape
        if ctx.enemy_tracking and ctx.enemy_pos is not None:
            ex, ey = ctx.enemy_pos
            at_wall = _near_wall(ex, ey, self.cfg.wall_threshold_cm, self._arena_corners)
            if at_wall:
                # Escaping? (distance increasing but still in range)
                if ctx.distance_cm > self.cfg.charge_close_range_cm * 1.5:
                    speed = 0.4  # pulse harder
                else:
                    speed = 0.2  # light hold
            else:
                speed = 1.0  # not at wall yet, full push
        else:
            speed = 0.2

        return BattleOutput(target_omega_dps=0.0, target_speed=speed)

    # -- Pit strategy actions -----------------------------------------------

    def _action_pit_position(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Navigate to herding position behind enemy relative to pit."""
        if ctx.enemy_pos is None:
            return BattleOutput()

        pit = (self.cfg.pit_x_cm, self.cfg.pit_y_cm)

        self_dist_to_pit = math.hypot(
            ctx.our_pos[0] - pit[0], ctx.our_pos[1] - pit[1]
        )
        if self_dist_to_pit < self.cfg.pit_danger_radius_cm + 25.0:
            self.machine.set_state("pit_abort")
            self._pit_abort_timer = now
            log.info("[battle] PIT ABORT — too close to pit (%.0fcm)", self_dist_to_pit)
            return self._action_pit_abort(ctx, now)

        dx = ctx.enemy_pos[0] - pit[0]
        dy = ctx.enemy_pos[1] - pit[1]
        dist_enemy_pit = math.hypot(dx, dy)
        if dist_enemy_pit < 1.0:
            dist_enemy_pit = 1.0
        nx, ny = dx / dist_enemy_pit, dy / dist_enemy_pit

        offset = max(35.0, min(60.0, dist_enemy_pit * 0.4))
        herd_x = ctx.enemy_pos[0] + nx * offset
        herd_y = ctx.enemy_pos[1] + ny * offset

        half_w = self.cfg.arena_width_cm / 2 - 10
        half_h = self.cfg.arena_height_cm / 2 - 10
        herd_x = max(-half_w, min(half_w, herd_x))
        herd_y = max(-half_h, min(half_h, herd_y))

        desired_heading = math.atan2(
            herd_y - ctx.our_pos[1], herd_x - ctx.our_pos[0]
        )
        alpha = _angle_diff(desired_heading, ctx.our_heading_rad)

        dist_to_herd = math.hypot(
            herd_x - ctx.our_pos[0], herd_y - ctx.our_pos[1]
        )

        if dist_to_herd < 20.0 and abs(alpha) < 0.6:
            self.machine.set_state("pit_push")
            log.info("[battle] PIT PUSH — in herding position")
            return self._action_pit_push(ctx, now)

        # Rate mode: omega from heading error, speed from distance
        Kp_omega = 200.0
        omega = -Kp_omega * alpha
        omega = max(-300.0, min(300.0, omega))

        if abs(alpha) > 1.0:
            # Large heading error — spin in place
            speed = 0.0
        else:
            speed = min(0.8, 0.4 + dist_to_herd / 100.0)
            speed *= math.cos(alpha) ** 2  # slow for turns
            # Slow down near pit to avoid overshooting into it
            if self_dist_to_pit < self.cfg.pit_danger_radius_cm * 2.5:
                speed = min(speed, 0.4)

        return BattleOutput(target_omega_dps=omega, target_speed=speed)

    def _action_pit_push(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Push enemy toward pit."""
        if not hasattr(self, '_pit_push_entry'):
            self._pit_push_entry = now
        if ctx.enemy_pos is None:
            self._pit_push_entry = None
            self.machine.set_state("pit_position")
            return BattleOutput()

        pit = (self.cfg.pit_x_cm, self.cfg.pit_y_cm)

        self_dist = math.hypot(
            ctx.our_pos[0] - pit[0], ctx.our_pos[1] - pit[1]
        )
        # Abort well before pit edge — robot body extends ~11cm past center,
        # plus need stopping distance at current speed
        abort_dist = self.cfg.pit_danger_radius_cm + 25.0
        if self_dist < abort_dist:
            self.machine.set_state("pit_abort")
            self._pit_abort_timer = now
            return self._action_pit_abort(ctx, now)

        enemy_dist = math.hypot(
            ctx.enemy_pos[0] - pit[0], ctx.enemy_pos[1] - pit[1]
        )
        if enemy_dist < self.cfg.pit_danger_radius_cm:
            self.machine.set_state("pit_commit")
            log.info("[battle] PIT COMMIT — enemy near pit (%.0fcm)", enemy_dist)
            return self._action_pit_commit(ctx, now)

        # Push toward enemy, not directly toward pit — this keeps Brick
        # on a path that goes through the enemy instead of around it
        if ctx.enemy_detected and ctx.enemy_pos is not None:
            target_x, target_y = ctx.enemy_pos
        else:
            target_x, target_y = pit

        desired_heading = math.atan2(
            target_y - ctx.our_pos[1], target_x - ctx.our_pos[0]
        )
        alpha = _angle_diff(desired_heading, ctx.our_heading_rad)

        Kp_omega = 200.0
        omega = -Kp_omega * alpha
        omega = max(-300.0, min(300.0, omega))

        # Brief brake phase on entry (0.15s) to kill approach momentum
        push_elapsed = now - self._pit_push_entry
        if push_elapsed < 0.15:
            return BattleOutput(target_omega_dps=omega, target_speed=0.0)

        # Speed: slow and controlled, scaled by distance to pit
        danger = self.cfg.pit_danger_radius_cm
        if self_dist < danger * 2.0:
            speed = 0.2   # crawl when close to pit
        else:
            speed = 0.5   # moderate push
        speed *= math.cos(alpha) ** 2

        return BattleOutput(target_omega_dps=omega, target_speed=speed)

    def _action_pit_commit(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Max power push at pit edge."""
        pit = (self.cfg.pit_x_cm, self.cfg.pit_y_cm)
        self_dist = math.hypot(
            ctx.our_pos[0] - pit[0], ctx.our_pos[1] - pit[1]
        )
        if self_dist < self.cfg.pit_radius_cm + 20:
            self.machine.set_state("pit_abort")
            self._pit_abort_timer = now
            log.info("[battle] PIT ABORT — self too close during commit")
            return self._action_pit_abort(ctx, now)

        # Rate mode: full speed, hold heading toward pit
        pit = (self.cfg.pit_x_cm, self.cfg.pit_y_cm)
        desired = math.atan2(pit[1] - ctx.our_pos[1], pit[0] - ctx.our_pos[0])
        alpha = _angle_diff(desired, ctx.our_heading_rad)
        omega = -200.0 * alpha
        omega = max(-300.0, min(300.0, omega))
        return BattleOutput(target_omega_dps=omega, target_speed=1.0)

    def _action_pit_abort(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Retreat away from pit."""
        if self._pit_abort_timer is None:
            self._pit_abort_timer = now

        elapsed = now - self._pit_abort_timer
        if elapsed > 1.5:
            self._pit_abort_timer = None
            self._reengage(ctx)
            return BattleOutput()

        pit = (self.cfg.pit_x_cm, self.cfg.pit_y_cm)
        away_angle = math.atan2(
            ctx.our_pos[1] - pit[1], ctx.our_pos[0] - pit[0]
        )
        alpha = _angle_diff(away_angle, ctx.our_heading_rad)

        # Rate mode: drive away from pit
        Kp_omega = 150.0
        omega = -Kp_omega * alpha
        omega = max(-200.0, min(200.0, omega))

        if abs(alpha) > math.pi / 2:
            # Facing pit — reverse
            return BattleOutput(target_omega_dps=omega, target_speed=-0.5)
        else:
            return BattleOutput(target_omega_dps=omega, target_speed=0.5)

    # -- Evade actions ------------------------------------------------------

    def _action_evade_retreat(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Reverse away from threat."""
        if self._retreat_timer is None:
            self._retreat_timer = now

        elapsed = now - self._retreat_timer

        retreat_time = self.cfg.reverse_duration_s
        if ctx.enemy_tracking and self.cfg.strategy != "evade":
            retreat_time = min(retreat_time, 0.8)

        if elapsed > retreat_time:
            self._retreat_timer = None
            # If retreat was due to ArUco loss and still blind, don't drive blind
            if self._last_retreat_reason == "aruco_lost" and not ctx.our_detected:
                self.machine.set_state("lost_aruco")
                self._lost_aruco_t = 0.0  # reset so _action_lost_aruco starts fresh
                log.info("[battle] RETREAT ended blind — waiting for ArUco")
                return BattleOutput()
            elif self.cfg.strategy == "evade":
                self.machine.set_state("evade_reposition")
                self._reposition_timer = now
            else:
                self._reengage(ctx)
            return BattleOutput()

        return BattleOutput(throttle=-0.8, steering=0.0)

    def _action_evade_reposition(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Drive toward arena center for safety."""
        if self._reposition_timer is None:
            self._reposition_timer = now

        elapsed = now - self._reposition_timer

        # Counter-attack window for avoid opening: if enemy facing away, strike
        if (self.cfg.opening_strategy == "avoid"
                and self.match_timer.phase == "start"
                and ctx.enemy_detected and ctx.enemy_heading_rad is not None):
            # Check if enemy is facing away from us
            angle_to_us = math.atan2(
                ctx.our_pos[1] - ctx.enemy_pos[1],
                ctx.our_pos[0] - ctx.enemy_pos[0],
            )
            facing_diff = abs(_angle_diff(ctx.enemy_heading_rad, angle_to_us))
            if facing_diff > math.radians(90):
                # Enemy facing away — strike from behind
                log.info("[battle] AVOID counter-attack — enemy facing away")
                self._reposition_timer = None
                self.machine.set_state("charge_pursue")
                self._prev_steer = 0.0
                return self._action_charge_pursue(ctx, now)

        if elapsed > 3.0:
            self._reposition_timer = None
            self._reengage(ctx)
            return BattleOutput()

        desired_heading = math.atan2(-ctx.our_pos[1], -ctx.our_pos[0])
        alpha = _angle_diff(desired_heading, ctx.our_heading_rad)

        dist_to_center = math.hypot(ctx.our_pos[0], ctx.our_pos[1])
        if dist_to_center < 20:
            self._reposition_timer = None
            self._reengage(ctx)
            return BattleOutput()

        steering = max(-0.4, min(0.4, alpha * 0.4))
        return BattleOutput(throttle=0.4, steering=steering)

    # -- Wall reverse -------------------------------------------------------

    def _action_wall_reverse(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Quick backup after wall crash: stop 100ms, reverse 400ms, re-engage."""
        if self._wall_reverse_timer is None:
            self._wall_reverse_timer = now
            self._wall_reverse_start_pos = (ctx.our_pos[0], ctx.our_pos[1]) if ctx.our_detected else None

        elapsed = now - self._wall_reverse_timer

        # Phase 1: Stop (absorb impact)
        if elapsed < 0.1:
            return BattleOutput()

        # Phase 2: Reverse (with pit avoidance)
        if elapsed < 0.5:
            steering = 0.0
            # If reversing toward pit, steer away
            pit_dist = self._pit_distance(ctx.our_pos[0], ctx.our_pos[1])
            if pit_dist < self.cfg.pit_danger_radius_cm + 25:
                away = self._pit_away_angle(ctx.our_pos[0], ctx.our_pos[1])
                alpha = _angle_diff(away, ctx.our_heading_rad)
                steering = max(-0.6, min(0.6, alpha * 0.5))
            return BattleOutput(throttle=-0.6, steering=steering)

        # Phase 3: Done — check if we actually moved
        self._wall_reverse_timer = None
        if self._wall_reverse_start_pos is not None and ctx.our_detected:
            disp = math.hypot(
                ctx.our_pos[0] - self._wall_reverse_start_pos[0],
                ctx.our_pos[1] - self._wall_reverse_start_pos[1],
            )
            if disp < 3.0:
                # Still wedged — escalate to unstick
                self._enter_unstick()
                return self._action_unstick(ctx, now)

        # Route through reorient (not direct _reengage) to re-evaluate approach
        # angle and avoid immediately charging back into the same wall
        if ctx.enemy_tracking and ctx.enemy_pos is not None:
            self.machine.set_state("charge_reorient")
            self._reorient_timer = now
            return self._action_charge_reorient(ctx, now)

        # No enemy — go to acquire or lost_target
        if ctx.enemy_detected:
            self.machine.set_state("acquire")
            self._acquire_count = max(self._acquire_count, 1)
        else:
            self._enter_lost_target(now)
        return BattleOutput()

    # -- Unstick action -----------------------------------------------------

    def _action_unstick(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Oscillate forward/reverse to free from stuck position."""
        if self._unstick_timer is None:
            self._unstick_timer = now
            self._unstick_start_pos = (ctx.our_pos[0], ctx.our_pos[1]) if ctx.our_detected else None

        elapsed = now - self._unstick_timer

        if self._unstick_start_pos is not None and ctx.our_detected:
            disp = math.hypot(ctx.our_pos[0] - self._unstick_start_pos[0],
                              ctx.our_pos[1] - self._unstick_start_pos[1])
            if disp > 10.0:
                log.info("[battle] UNSTICK — freed (%.0fcm moved)", disp)
                self._unstick_timer = None
                self._last_positions.clear()
                self._reengage(ctx)
                return BattleOutput()

        if elapsed > self.cfg.unstick_oscillate_s:
            log.info("[battle] UNSTICK — timeout (%.1fs)", elapsed)
            self._unstick_timer = None
            self._last_positions.clear()
            self._reengage(ctx)
            return BattleOutput()

        if now - self._unstick_toggle_t > 0.3:
            self._unstick_phase *= -1
            self._unstick_toggle_t = now

        # Pit avoidance: bias oscillation away from pit
        steering = 0.0
        pit_dist = self._pit_distance(ctx.our_pos[0], ctx.our_pos[1])
        if pit_dist < self.cfg.pit_danger_radius_cm + 25:
            away = self._pit_away_angle(ctx.our_pos[0], ctx.our_pos[1])
            alpha = _angle_diff(away, ctx.our_heading_rad)
            steering = max(-0.5, min(0.5, alpha * 0.4))

        return BattleOutput(throttle=0.5 * self._unstick_phase, steering=steering)

    # -- Lost target action -------------------------------------------------

    def _action_lost_target(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Drive toward last known enemy position, then slow rotate to search."""
        if self._lost_timer is None:
            self._lost_timer = now
            self._lost_rotating = False

        elapsed = now - self._lost_timer

        # Re-acquired?
        if ctx.enemy_tracking:
            self._lost_timer = None
            self._lost_rotating = False
            self._reengage(ctx)
            return BattleOutput()

        # After 2s of driving to last known pos, switch to slow rotation
        if elapsed > 2.0 and not self._lost_rotating:
            self._lost_rotating = True
            log.info("[battle] LOST TARGET — rotating to search")

        if self._lost_rotating:
            urgency = self.match_timer.urgency
            spin_speed = 0.3 + 0.2 * urgency
            return BattleOutput(throttle=0.0, steering=spin_speed)

        # Drive toward last known position
        if self._last_enemy_pos is not None:
            desired_heading = math.atan2(
                self._last_enemy_pos[1] - ctx.our_pos[1],
                self._last_enemy_pos[0] - ctx.our_pos[0],
            )
            alpha = _angle_diff(desired_heading, ctx.our_heading_rad)
            steering = max(-0.4, min(0.4, alpha * 0.4))
            return BattleOutput(throttle=0.4, steering=steering)

        return BattleOutput(throttle=0.0, steering=0.3)

    # -- Lost ArUco action --------------------------------------------------

    def _action_lost_aruco(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Own position lost — drive toward arena center to get back into
        camera view.

        Uses last known position to compute heading toward (0, 0) and drives
        forward at 0.5 power for up to 2 seconds.  If that doesn't recover
        the marker, spin slowly to search.
        """
        # ArUco re-acquired → snap back to normal
        if ctx.our_detected:
            log.info("[battle] LOST ARUCO — re-acquired, re-engaging")
            self._lost_aruco_t = 0.0
            self._reengage(ctx)
            return BattleOutput()

        # First frame — compute heading toward arena center from last known pos
        if self._lost_aruco_t == 0.0:
            self._lost_aruco_t = now
            dx = 0.0 - self._last_aruco_x
            dy = 0.0 - self._last_aruco_y
            self._lost_aruco_target_heading = math.atan2(dy, dx)
            log.info("[battle] LOST ARUCO — driving toward center from (%.0f, %.0f), heading %.0f°",
                     self._last_aruco_x, self._last_aruco_y,
                     math.degrees(self._lost_aruco_target_heading))

        elapsed = now - self._lost_aruco_t

        # Phase 1 (0-2s): drive toward arena center at 0.5 power
        if elapsed < 2.0:
            imu_heading_rad = math.radians(ctx.imu_heading_deg)
            alpha = _angle_diff(self._lost_aruco_target_heading, imu_heading_rad)
            omega = -150.0 * alpha
            omega = max(-200.0, min(200.0, omega))
            return BattleOutput(target_omega_dps=omega, target_speed=0.5)

        # Phase 2 (2-4s): slow spin to search for marker
        if elapsed < 4.0:
            return BattleOutput(target_omega_dps=120.0, target_speed=0.0)

        # Phase 3 (4s+): stop
        return BattleOutput(target_omega_dps=0.0, target_speed=0.0)

    # -- Victory dance action -----------------------------------------------

    def _action_victory_dance(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Post-match: drive to arena center, then spin."""
        if self._victory_start is None:
            self._victory_start = now

        elapsed = now - self._victory_start

        # Phase 1 (0-3s): drive to arena center
        if elapsed < 3.0:
            dist_to_center = math.hypot(ctx.our_pos[0], ctx.our_pos[1])
            if dist_to_center < 15:
                # Close enough — start spinning early
                return BattleOutput(target_omega_dps=360.0, target_speed=0.0)

            # Wall-stuck escape: reverse if near arena edge
            if _near_wall(ctx.our_pos[0], ctx.our_pos[1], 15.0, self._arena_corners):
                return BattleOutput(target_omega_dps=0.0, target_speed=-0.5)

            desired = math.atan2(-ctx.our_pos[1], -ctx.our_pos[0])
            alpha = _angle_diff(desired, ctx.our_heading_rad)
            omega = -200.0 * alpha
            omega = max(-300.0, min(300.0, omega))
            speed = min(0.6, dist_to_center / 100.0 + 0.3)
            return BattleOutput(target_omega_dps=omega, target_speed=speed)

        # Phase 2 (3-6s): victory spin
        if elapsed < 3.0 + self.cfg.victory_dance_duration_s:
            return BattleOutput(target_omega_dps=360.0, target_speed=0.0)

        self._dance_finished = True
        return BattleOutput()
