"""Hierarchical State Machine for combat mode using the transitions library."""

import logging
import math
import time
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


# ---------------------------------------------------------------------------
# BattleController — the HSM model
# ---------------------------------------------------------------------------

# State definitions for HierarchicalMachine
_STATES = [
    "scan",
    "acquire",
    {
        "name": "charge",
        "children": ["pursue", "flank", "ram", "pin", "reorient"],
        "initial": "pursue",
    },
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
    "unstick",
    "lost_target",
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
    ):
        self.cfg = config
        self.match_timer = match_timer
        self.pin_timer = pin_timer

        # Internal tracking
        self._acquire_count = 0
        self._prev_steer = 0.0
        self._lost_timer: float | None = None
        self._unstick_timer: float | None = None
        self._unstick_phase = 1  # +1 or -1
        self._unstick_toggle_t = 0.0
        self._retreat_timer: float | None = None
        self._aruco_lost_frames = 0
        self._last_positions: list[tuple[float, float, float]] = []  # (x, y, t)
        self._last_enemy_pos: tuple[float, float] | None = None
        self._reposition_timer: float | None = None
        self._reorient_timer: float | None = None
        self._log_t = 0.0

        # Pit strategy state
        self._pit_abort_timer: float | None = None

        # Build the HSM — transitions evaluates only the current state's
        # transitions, so this is O(1) per tick regardless of total transitions.
        self.machine = HierarchicalMachine(
            model=self,
            states=_STATES,
            initial="scan",
            auto_transitions=False,
            queued=True,
        )

    # -- Public API ---------------------------------------------------------

    def tick(self, ctx: BattleContext) -> BattleOutput:
        """Main entry point — call once per frame."""
        now = time.perf_counter()

        # Track ArUco loss for our robot
        if ctx.our_detected:
            self._aruco_lost_frames = 0
        else:
            self._aruco_lost_frames += 1

        # Track position history for stuck detection
        if ctx.our_detected:
            self._last_positions.append((ctx.our_pos[0], ctx.our_pos[1], now))
            # Keep last 1.5 seconds
            cutoff = now - 1.5
            self._last_positions = [
                p for p in self._last_positions if p[2] > cutoff
            ]

        # Remember last enemy position
        if ctx.enemy_detected and ctx.enemy_pos is not None:
            self._last_enemy_pos = ctx.enemy_pos

        # --- Global transitions (checked every tick) ---
        current = self.state

        # Wall impact detection — sudden deceleration spike (>1500mg = ~1.5g)
        accel_mag = math.hypot(ctx.accel_x_mg, ctx.accel_y_mg)
        if (current.startswith("charge") and accel_mag > 1500
                and abs(ctx.throttle_cmd) > 0.3):
            log.info("[battle] IMPACT detected (%.0fmg) — reversing", accel_mag)
            self._enter_retreat(reason="impact")
            return self._action_evade_retreat(ctx, now)

        # Stuck detection (not while already unsticking or retreating)
        if current not in ("unstick", "evade_retreat", "evade_reposition"):
            if self._is_stuck(ctx):
                self._enter_unstick()
                return self._action_unstick(ctx, now)

        # ArUco lost → we can't see ourselves (probably at a wall)
        # Give it time — brief ArUco drops are normal near walls
        # Only retreat after sustained loss (~1 second)
        if current.startswith("charge") or current.startswith("pit"):
            if self._aruco_lost_frames > 60:  # ~1 second at 60fps
                self._enter_retreat(reason="aruco_lost_wall")
                return self._action_evade_retreat(ctx, now)

        # Enemy lost → lost_target (from combat states, not evade/unstick)
        if current in ("charge_pursue", "charge_flank", "charge_ram",
                        "charge_reorient", "pit_position", "pit_push"):
            if not ctx.enemy_detected and ctx.frames_without_detection > 30:
                self._enter_lost_target(now)
                return self._action_lost_target(ctx, now)

        # --- State-specific action + transitions ---
        action_map = {
            "scan": self._action_scan,
            "acquire": self._action_acquire,
            "charge_pursue": self._action_charge_pursue,
            "charge_flank": self._action_charge_flank,
            "charge_ram": self._action_charge_ram,
            "charge_reorient": self._action_charge_reorient,
            "charge_pin": self._action_charge_pin,
            "pit_position": self._action_pit_position,
            "pit_push": self._action_pit_push,
            "pit_commit": self._action_pit_commit,
            "pit_abort": self._action_pit_abort,
            "evade_retreat": self._action_evade_retreat,
            "evade_reposition": self._action_evade_reposition,
            "unstick": self._action_unstick,
            "lost_target": self._action_lost_target,
        }

        action = action_map.get(current)
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
            "stuck_frames": getattr(self, '_stuck_accel_frames', 0),
            "unstick_phase": getattr(self, '_unstick_phase', 0),
            "aruco_lost": self._aruco_lost_frames,
            "retreat_reason": getattr(self, '_last_retreat_reason', None),
        }

    def reset(self) -> None:
        """Reset to scan state for a new match."""
        # Force state back to scan
        self.machine.set_state("scan")
        self._acquire_count = 0
        self._prev_steer = 0.0
        self._lost_timer = None
        self._unstick_timer = None
        self._retreat_timer = None
        self._aruco_lost_frames = 0
        self._last_positions.clear()
        self._last_enemy_pos = None
        self._reposition_timer = None
        self._reorient_timer = None
        self._pit_abort_timer = None
        self.pin_timer.reset()

    # -- Smart re-engagement (skip scan/acquire if enemy still tracked) -----

    def _reengage(self, ctx: BattleContext) -> None:
        """Route to the best state based on current tracking status.

        If enemy is still tracked, go straight to pursue/pit — no scan/acquire.
        Only fall back to scan when we truly have no idea where the enemy is.
        """
        if ctx.enemy_tracking and ctx.enemy_pos is not None:
            # Still tracking — jump straight to combat
            if self.cfg.strategy == "pit":
                self.machine.set_state("pit_position")
            else:
                self.machine.set_state("charge_pursue")
                self._prev_steer = 0.0
                self._acquire_count = self.cfg.acquire_frames + 1
            log.info("[battle] Re-engaging — enemy still tracked")
        elif ctx.enemy_detected:
            # Just detected — quick acquire
            self.machine.set_state("acquire")
            self._acquire_count = max(self._acquire_count, 1)
        else:
            # Truly lost — scan
            self.machine.set_state("scan")
            self._acquire_count = 0

    # -- Stuck detection (IMU + position) ------------------------------------

    def _is_stuck(self, ctx: BattleContext) -> bool:
        """Detect if robot is stuck using IMU + position history.

        Two signals:
        1. Position-based: no movement over ~0.8s while commanding throttle
        2. IMU-based: commanding throttle but acceleration is near zero
           (hitting a wall — motors spin but robot doesn't move)
        """
        # IMU-based: commanding forward but no forward acceleration
        # accel_x_mg is in body frame — forward acceleration when driving
        if abs(ctx.throttle_cmd) > 0.3:
            accel_mag = math.hypot(ctx.accel_x_mg, ctx.accel_y_mg)
            # When driving freely, accel is typically 100-500mg
            # When stuck against a wall, accel drops to <50mg despite throttle
            if not hasattr(self, '_stuck_accel_frames'):
                self._stuck_accel_frames = 0
            if accel_mag < 80:
                self._stuck_accel_frames += 1
            else:
                self._stuck_accel_frames = 0
            # Stuck if low accel for 30 frames (~0.5s) while commanding throttle
            if self._stuck_accel_frames > 30:
                self._stuck_accel_frames = 0
                return True
        else:
            self._stuck_accel_frames = 0

        # Position-based fallback (works even without IMU)
        # Only check when actually commanding throttle (not during acquire/standstill)
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

    # -- State entry helpers ------------------------------------------------

    def _enter_unstick(self) -> None:
        self.machine.set_state("unstick")
        self._unstick_timer = time.perf_counter()
        self._unstick_phase = 1
        self._unstick_toggle_t = time.perf_counter()
        self._last_positions.clear()
        log.info("[battle] UNSTICK — oscillating to free")

    def _enter_retreat(self, reason: str = "aruco_lost") -> None:
        self.machine.set_state("evade_retreat")
        self._retreat_timer = time.perf_counter()
        self._last_retreat_reason = reason
        if reason == "aruco_lost":
            self._aruco_lost_frames = 0
        # Don't clear _last_enemy_pos — we want reengage to know where enemy was
        log.info("[battle] RETREAT — %s", reason)

    def _enter_lost_target(self, now: float) -> None:
        self.machine.set_state("lost_target")
        self._lost_timer = now
        log.info("[battle] LOST TARGET — driving to last known position")

    # -- Action functions ---------------------------------------------------

    def _action_scan(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Spin slowly to find enemy."""
        urgency = self.match_timer.urgency
        spin_speed = 0.3 + 0.2 * urgency  # faster scan when urgent

        # Transition: enemy detected → acquire
        if ctx.enemy_detected:
            self.machine.set_state("acquire")
            self._acquire_count = 1
            return BattleOutput(throttle=0.0, steering=0.0)

        return BattleOutput(throttle=0.0, steering=spin_speed)

    def _action_acquire(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Validate detection over multiple frames before committing."""
        urgency = self.match_timer.urgency
        required = max(5, int(self.cfg.acquire_frames * (1.0 - 0.5 * urgency)))

        if ctx.enemy_detected:
            self._acquire_count += 1
        else:
            # Detection dropped — decay
            self._acquire_count = max(0, self._acquire_count - 2)

        if self._acquire_count <= 0:
            self.machine.set_state("scan")
            return BattleOutput(throttle=0.0, steering=0.3)

        if self._acquire_count >= required:
            # Locked on — choose strategy
            if self.cfg.strategy == "pit":
                self.machine.set_state("pit_position")
            elif self.cfg.strategy == "evade":
                self.machine.set_state("evade_reposition")
                self._reposition_timer = now
            else:
                self.machine.set_state("charge_pursue")
                self._prev_steer = 0.0
            return BattleOutput(throttle=0.0, steering=0.0)

        return BattleOutput(throttle=0.0, steering=0.0)

    def _action_charge_pursue(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Pure pursuit arc driving toward enemy, respecting safe side."""
        if ctx.enemy_pos is None:
            return BattleOutput()

        urgency = self.match_timer.urgency

        # Flanking disabled — direct charge only for now
        # if (ctx.enemy_heading_rad is not None
        #         and ctx.distance_cm > self.cfg.charge_close_range_cm * 2):
        #     if needs_flanking(ctx.our_pos, ctx.enemy_pos, ctx.enemy_heading_rad, self.cfg.safe_side):
        #         self.machine.set_state("charge_flank")
        #         return self._action_charge_flank(ctx, now)

        # Close range → RAM
        if ctx.distance_cm < self.cfg.charge_close_range_cm:
            # Check if at wall for PIN
            ex, ey = ctx.enemy_pos
            near_wall = abs(ex) > self.cfg.wall_threshold_cm or abs(ey) > self.cfg.wall_threshold_cm
            if near_wall:
                self.machine.set_state("charge_pin")
                self.pin_timer.start()
                log.info("[battle] PIN — enemy at wall (%.0f, %.0f)", ex, ey)
                return BattleOutput(throttle=0.2, steering=0.0)
            else:
                self.machine.set_state("charge_ram")
                log.info("[battle] RAM — close range %.0fcm", ctx.distance_cm)
                return BattleOutput(throttle=1.0, steering=0.0)

        # Pure pursuit arc
        desired_heading = math.atan2(
            ctx.enemy_pos[1] - ctx.our_pos[1],
            ctx.enemy_pos[0] - ctx.our_pos[0],
        )
        alpha = _angle_diff(desired_heading, ctx.our_heading_rad)

        # Rate mode: compute desired angular velocity (frame-invariant)
        # Negate because positive omega = CW on our robot (IMU mounted inverted)
        Kp_omega = 200.0  # dps per radian — increased for tighter pursuit arcs
        omega = -Kp_omega * alpha
        omega = max(-300.0, min(300.0, omega))  # clamp to 300 dps

        alpha_abs = abs(alpha)

        # Diagnostic logging every 0.5s
        if now - self._log_t > 0.5:
            self._log_t = now
            log.info("[pursue] pos=(%.0f,%.0f) h=%.0f° enemy=(%.0f,%.0f) desired=%.0f° alpha=%.0f° omega=%.0f speed=%.2f dist=%.0f",
                     ctx.our_pos[0], ctx.our_pos[1], math.degrees(ctx.our_heading_rad),
                     ctx.enemy_pos[0], ctx.enemy_pos[1],
                     math.degrees(desired_heading), math.degrees(alpha),
                     omega, math.cos(alpha), ctx.distance_cm)

        # Reorient if facing away from enemy — back up + spin, attack front-first
        if alpha_abs > math.radians(110):
            log.info("[pursue] REORIENT triggered: alpha=%.0f°", math.degrees(alpha))
            self.machine.set_state("charge_reorient")
            self._reorient_timer = now
            return self._action_charge_reorient(ctx, now)

        # Forward pursuit: cos² scaling — slows more at moderate error for tighter arcs
        speed = math.cos(alpha) ** 2
        speed *= (1.0 - min(abs(omega) / 300.0, 1.0) * 0.3)
        speed = max(0.15, speed)

        return BattleOutput(target_omega_dps=omega, target_speed=speed)

    def _action_charge_reorient(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Back up briefly, spin to face enemy, then resume pursuit."""
        if self._reorient_timer is None:
            self._reorient_timer = now

        elapsed = now - self._reorient_timer

        # Abort if enemy closes to contact
        if ctx.enemy_detected and ctx.distance_cm < 10:
            self._reorient_timer = None
            self.machine.set_state("charge_ram")
            return BattleOutput(throttle=1.0, steering=0.0)

        # Phase 1: Backup (0-0.25s) — straight reverse, gyro holds heading
        if elapsed < 0.25:
            return BattleOutput(target_omega_dps=0.0, target_speed=-0.35)

        # Phase 2: Spin to face enemy
        if ctx.enemy_pos is not None:
            desired = math.atan2(
                ctx.enemy_pos[1] - ctx.our_pos[1],
                ctx.enemy_pos[0] - ctx.our_pos[0],
            )
            alpha = _angle_diff(desired, ctx.our_heading_rad)

            # Aligned enough — resume pursuit
            if abs(alpha) < math.radians(40):
                self._reorient_timer = None
                self.machine.set_state("charge_pursue")
                return self._action_charge_pursue(ctx, now)

            # Spin shortest path (negated — positive omega = CW on our robot)
            omega = -300.0 if alpha > 0 else 300.0
            return BattleOutput(target_omega_dps=omega, target_speed=0.25)

        # Timeout after 1.5s total
        if elapsed > 1.5:
            self._reorient_timer = None
            self.machine.set_state("charge_pursue")
            return self._action_charge_pursue(ctx, now)

        # Blind spin (no enemy tracking)
        return BattleOutput(target_omega_dps=300.0, target_speed=0.25)

    def _action_charge_flank(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Arc around to the enemy's safe side before committing."""
        if ctx.enemy_pos is None or ctx.enemy_heading_rad is None:
            # Lost heading info — just pursue directly
            self.machine.set_state("charge_pursue")
            return self._action_charge_pursue(ctx, now)

        # Check if we've reached the safe side
        if is_approach_safe(ctx.our_pos, ctx.enemy_pos, ctx.enemy_heading_rad, self.cfg.safe_side):
            self.machine.set_state("charge_pursue")
            self._prev_steer = 0.0
            return self._action_charge_pursue(ctx, now)

        # Drive to the safe approach position
        target = get_safe_approach_position(
            ctx.enemy_pos, ctx.enemy_heading_rad, self.cfg.safe_side, distance_cm=40.0
        )
        desired_heading = math.atan2(
            target[1] - ctx.our_pos[1],
            target[0] - ctx.our_pos[0],
        )
        alpha = _angle_diff(desired_heading, ctx.our_heading_rad)
        alpha_abs = abs(alpha)

        # Rate mode with bidirectional pursuit (same as charge_pursue)
        Kp_omega = 120.0
        omega = -Kp_omega * alpha  # negated — positive omega = CW on our robot
        omega = max(-300.0, min(300.0, omega))

        if not hasattr(self, '_flank_reversing'):
            self._flank_reversing = False
        if alpha_abs > math.radians(100):
            self._flank_reversing = True
        elif alpha_abs < math.radians(80):
            self._flank_reversing = False

        if not self._flank_reversing:
            speed = math.cos(alpha) * 0.8  # slightly slower than pursue
        else:
            reverse_alpha = math.pi - alpha_abs
            speed = -math.cos(reverse_alpha) * 0.8

        speed *= (1.0 - min(abs(omega) / 300.0, 1.0) * 0.2)
        if speed >= 0:
            speed = max(0.15, speed)
        else:
            speed = min(-0.15, speed)

        return BattleOutput(target_omega_dps=omega, target_speed=speed)

    def _action_charge_ram(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Full throttle final approach — push enemy to wall."""
        if ctx.enemy_pos is None:
            return BattleOutput(throttle=1.0, steering=0.0)

        ex, ey = ctx.enemy_pos
        near_wall = abs(ex) > self.cfg.wall_threshold_cm or abs(ey) > self.cfg.wall_threshold_cm

        if near_wall:
            self.machine.set_state("charge_pin")
            self.pin_timer.start()
            log.info("[battle] PIN — rammed to wall (%.0f, %.0f)", ex, ey)
            return BattleOutput(throttle=0.2, steering=0.0)

        # Still pushing — maintain heading toward enemy
        desired_heading = math.atan2(
            ctx.enemy_pos[1] - ctx.our_pos[1],
            ctx.enemy_pos[0] - ctx.our_pos[0],
        )
        alpha = _angle_diff(desired_heading, ctx.our_heading_rad)

        # Overshot — facing away and distance increasing, drop to pursue→reorient
        if abs(alpha) > math.radians(90) and ctx.distance_cm > self.cfg.charge_close_range_cm:
            self.machine.set_state("charge_pursue")
            return self._action_charge_pursue(ctx, now)

        steering = max(-0.3, min(0.3, alpha * 0.3))

        urgency = self.match_timer.urgency
        throttle = min(1.0, 1.0 + 0.0 * urgency)  # already max
        return BattleOutput(throttle=throttle, steering=steering)

    def _action_charge_pin(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Hold enemy against wall for configured pin duration."""
        # Pin timer expired → short back-off then re-engage
        if self.pin_timer.is_expired:
            self.pin_timer.reset()
            self._enter_retreat(reason="pin_release")
            return self._action_evade_retreat(ctx, now)

        # ArUco lost too long during pin → retreat
        if self._aruco_lost_frames > 30:
            self.pin_timer.reset()
            self._enter_retreat(reason="aruco_lost")
            return self._action_evade_retreat(ctx, now)

        # Enemy escaped?
        if ctx.enemy_tracking and ctx.our_detected and ctx.distance_cm > self.cfg.pin_escape_range_cm:
            self.pin_timer.reset()
            self.machine.set_state("charge_pursue")
            self._prev_steer = 0.0
            self._acquire_count = self.cfg.acquire_frames + 1  # skip re-acquire
            log.info("[battle] PIN — enemy escaped (%.0fcm), re-engaging", ctx.distance_cm)
            return self._action_charge_pursue(ctx, now)

        # Hold — soft at wall, full power if not at wall yet
        if ctx.enemy_tracking and ctx.enemy_pos is not None:
            ex, ey = ctx.enemy_pos
            at_wall = abs(ex) > self.cfg.wall_threshold_cm or abs(ey) > self.cfg.wall_threshold_cm
            speed = 0.2 if at_wall else 1.0
        else:
            speed = 0.2

        # Rate mode: omega=0 (drive straight, gyro resists impacts)
        return BattleOutput(target_omega_dps=0.0, target_speed=speed)

    # -- Pit strategy actions -----------------------------------------------

    def _action_pit_position(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Navigate to herding position behind enemy relative to pit."""
        if ctx.enemy_pos is None:
            return BattleOutput(throttle=0.0, steering=0.3)

        pit = (self.cfg.pit_x_cm, self.cfg.pit_y_cm)

        # Self-pit avoidance
        self_dist_to_pit = math.hypot(
            ctx.our_pos[0] - pit[0], ctx.our_pos[1] - pit[1]
        )
        if self_dist_to_pit < self.cfg.pit_danger_radius_cm:
            self.machine.set_state("pit_abort")
            self._pit_abort_timer = now
            log.info("[battle] PIT ABORT — too close to pit (%.0fcm)", self_dist_to_pit)
            return self._action_pit_abort(ctx, now)

        # Herding point: far side of enemy from pit (swing wide to get behind)
        dx = ctx.enemy_pos[0] - pit[0]
        dy = ctx.enemy_pos[1] - pit[1]
        dist_enemy_pit = math.hypot(dx, dy)
        if dist_enemy_pit < 1.0:
            dist_enemy_pit = 1.0
        nx, ny = dx / dist_enemy_pit, dy / dist_enemy_pit

        # Offset distance scales with how far enemy is from pit
        # Farther enemy = wider swing to get behind
        offset = max(35.0, min(60.0, dist_enemy_pit * 0.4))
        herd_x = ctx.enemy_pos[0] + nx * offset
        herd_y = ctx.enemy_pos[1] + ny * offset

        # Clamp herding point to arena bounds
        half_w = self.cfg.arena_width_cm / 2 - 10
        half_h = self.cfg.arena_height_cm / 2 - 10
        herd_x = max(-half_w, min(half_w, herd_x))
        herd_y = max(-half_h, min(half_h, herd_y))

        # Drive to herding point
        desired_heading = math.atan2(
            herd_y - ctx.our_pos[1], herd_x - ctx.our_pos[0]
        )
        alpha = _angle_diff(desired_heading, ctx.our_heading_rad)

        dist_to_herd = math.hypot(
            herd_x - ctx.our_pos[0], herd_y - ctx.our_pos[1]
        )

        # Check if we're in position (behind enemy relative to pit)
        if dist_to_herd < 20.0 and abs(alpha) < 0.6:
            self.machine.set_state("pit_push")
            log.info("[battle] PIT PUSH — in herding position")
            return self._action_pit_push(ctx, now)

        if abs(alpha) > 1.0:
            return BattleOutput(
                throttle=0.0,
                steering=0.6 if alpha > 0 else -0.6,
            )

        # Drive faster when far from herding point
        throttle = min(0.8, 0.4 + dist_to_herd / 100.0)
        steering = max(-0.6, min(0.6, alpha * 0.6))
        return BattleOutput(throttle=throttle, steering=steering)

    def _action_pit_push(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Push enemy toward pit."""
        if ctx.enemy_pos is None:
            self.machine.set_state("pit_position")
            return BattleOutput()

        pit = (self.cfg.pit_x_cm, self.cfg.pit_y_cm)

        # Self-pit avoidance
        self_dist = math.hypot(
            ctx.our_pos[0] - pit[0], ctx.our_pos[1] - pit[1]
        )
        if self_dist < self.cfg.pit_danger_radius_cm:
            self.machine.set_state("pit_abort")
            self._pit_abort_timer = now
            return self._action_pit_abort(ctx, now)

        # Enemy near pit? → commit
        enemy_dist = math.hypot(
            ctx.enemy_pos[0] - pit[0], ctx.enemy_pos[1] - pit[1]
        )
        if enemy_dist < self.cfg.pit_danger_radius_cm:
            self.machine.set_state("pit_commit")
            log.info("[battle] PIT COMMIT — enemy near pit (%.0fcm)", enemy_dist)
            return self._action_pit_commit(ctx, now)

        # Aim AT the pit through the enemy — full commitment
        desired_heading = math.atan2(
            pit[1] - ctx.our_pos[1], pit[0] - ctx.our_pos[0]
        )
        alpha = _angle_diff(desired_heading, ctx.our_heading_rad)
        steering = max(-0.5, min(0.5, alpha * 0.6))

        # Full power push — only slow down if dangerously close to pit ourselves
        throttle = 1.0
        if self_dist < self.cfg.pit_danger_radius_cm * 1.2:
            throttle = 0.5

        return BattleOutput(throttle=throttle, steering=steering)

    def _action_pit_commit(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Max power push at pit edge — no corrections."""
        pit = (self.cfg.pit_x_cm, self.cfg.pit_y_cm)

        # Self-preservation: abort if too close
        self_dist = math.hypot(
            ctx.our_pos[0] - pit[0], ctx.our_pos[1] - pit[1]
        )
        if self_dist < self.cfg.pit_radius_cm + 5:
            self.machine.set_state("pit_abort")
            self._pit_abort_timer = now
            log.info("[battle] PIT ABORT — self too close during commit")
            return self._action_pit_abort(ctx, now)

        return BattleOutput(throttle=1.0, steering=0.0)

    def _action_pit_abort(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Retreat away from pit."""
        if self._pit_abort_timer is None:
            self._pit_abort_timer = now

        elapsed = now - self._pit_abort_timer
        if elapsed > 1.5:
            self._pit_abort_timer = None
            self._reengage(ctx)
            return BattleOutput()

        # Drive away from pit
        pit = (self.cfg.pit_x_cm, self.cfg.pit_y_cm)
        away_angle = math.atan2(
            ctx.our_pos[1] - pit[1], ctx.our_pos[0] - pit[0]
        )
        alpha = _angle_diff(away_angle, ctx.our_heading_rad)

        if abs(alpha) > math.pi / 2:
            # Facing pit — reverse
            return BattleOutput(throttle=-0.6, steering=0.0)
        else:
            steering = max(-0.4, min(0.4, alpha * 0.4))
            return BattleOutput(throttle=0.5, steering=steering)

    # -- Evade actions ------------------------------------------------------

    def _action_evade_retreat(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Reverse away from threat."""
        if self._retreat_timer is None:
            self._retreat_timer = now

        elapsed = now - self._retreat_timer

        # Short retreat if enemy still tracked — just create separation
        retreat_time = self.cfg.reverse_duration_s
        if ctx.enemy_tracking and self.cfg.strategy != "evade":
            retreat_time = min(retreat_time, 0.8)  # quick 0.8s back-off

        if elapsed > retreat_time:
            self._retreat_timer = None
            if self.cfg.strategy == "evade":
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

        # After 3 seconds, re-engage
        if elapsed > 3.0:
            self._reposition_timer = None
            self._reengage(ctx)
            return BattleOutput()

        # Drive toward center (0, 0)
        desired_heading = math.atan2(
            -ctx.our_pos[1], -ctx.our_pos[0]
        )
        alpha = _angle_diff(desired_heading, ctx.our_heading_rad)

        dist_to_center = math.hypot(ctx.our_pos[0], ctx.our_pos[1])
        if dist_to_center < 20:
            # Close enough to center — re-engage
            self._reposition_timer = None
            self._reengage(ctx)
            return BattleOutput()

        steering = max(-0.4, min(0.4, alpha * 0.4))
        return BattleOutput(throttle=0.4, steering=steering)

    # -- Unstick action -----------------------------------------------------

    def _action_unstick(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Oscillate forward/reverse to free from stuck position."""
        if self._unstick_timer is None:
            self._unstick_timer = now
            self._unstick_start_pos = (ctx.our_pos[0], ctx.our_pos[1]) if ctx.our_detected else None

        elapsed = now - self._unstick_timer

        # Early exit: moved enough to be free (>10cm from start)
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

        # Toggle direction every 0.3s
        if now - self._unstick_toggle_t > 0.3:
            self._unstick_phase *= -1
            self._unstick_toggle_t = now

        return BattleOutput(throttle=0.5 * self._unstick_phase, steering=0.0)

    # -- Lost target action -------------------------------------------------

    def _action_lost_target(self, ctx: BattleContext, now: float) -> BattleOutput:
        """Drive toward last known enemy position, then scan."""
        if self._lost_timer is None:
            self._lost_timer = now

        elapsed = now - self._lost_timer

        # Re-acquired?
        if ctx.enemy_tracking:
            self._lost_timer = None
            self._reengage(ctx)
            return BattleOutput()

        # Timeout — truly lost, scan
        if elapsed > 2.0:
            self._lost_timer = None
            self.machine.set_state("scan")
            self._acquire_count = 0
            return BattleOutput()

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
