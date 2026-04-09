"""Full state machine simulation — runs BattleController against an AI enemy.

Produces a JSONL + arena metadata file compatible with replay.html.
No physical hardware or CV required.

Usage:
    python sim_full_match.py [--duration 180] [--opening charge] [--strategy charge]

The AI enemy uses behavioral scripting to trigger different state transitions:
- Charges at our robot (triggers evasion)
- Retreats to walls (triggers pursuit + pin)
- Stands still near pit (triggers pit strategy)
- Moves erratically (triggers lost_target)
- Disappears briefly (triggers ArUco loss simulation)
"""

import argparse
import json
import math
import os
import random
import time
from collections import deque

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from battle_config import BattleConfig
from match_timer import MatchTimer, PinTimer
from state_machine import BattleController, BattleContext, BattleOutput

random.seed(42)


# ---------------------------------------------------------------------------
# Sim clock — patches time.perf_counter for deterministic simulation
# ---------------------------------------------------------------------------

class SimClock:
    """Deterministic clock that replaces time.perf_counter for simulation."""

    def __init__(self):
        self._time = 0.0
        self._real_perf_counter = time.perf_counter

    def advance(self, dt: float):
        self._time += dt

    def perf_counter(self) -> float:
        return self._time

    def install(self):
        time.perf_counter = self.perf_counter

    def uninstall(self):
        time.perf_counter = self._real_perf_counter

ARENA_HALF = 122.0  # cm
FPS = 60
DT = 1.0 / FPS


# ---------------------------------------------------------------------------
# Simple 2D physics for robots
# ---------------------------------------------------------------------------

class SimRobot:
    """Minimal 2D robot body with velocity integration."""

    def __init__(self, x: float, y: float, heading: float):
        self.x = x
        self.y = y
        self.heading = heading  # radians
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0  # scalar speed
        self.max_speed_cm_s = 120.0  # typical beetleweight at full power

    def step(self, throttle: float, omega_dps: float | None, dt: float):
        """Integrate one timestep given motor commands."""
        # Heading update — rate mode has fast response (ESP32 at 3.33kHz)
        if omega_dps is not None:
            self.heading += math.radians(omega_dps) * dt
        # Wrap heading
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi

        # Speed from throttle — fast acceleration (beetleweight response)
        target_speed = throttle * self.max_speed_cm_s
        self.speed += (target_speed - self.speed) * min(1.0, 8.0 * dt)

        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed

        self.x += self.vx * dt
        self.y += self.vy * dt

        # Wall collisions — bounce back
        if abs(self.x) > ARENA_HALF:
            self.x = math.copysign(ARENA_HALF, self.x)
            self.vx *= -0.3
            self.speed *= 0.3
        if abs(self.y) > ARENA_HALF:
            self.y = math.copysign(ARENA_HALF, self.y)
            self.vy *= -0.3
            self.speed *= 0.3

    @property
    def pos(self):
        return (self.x, self.y)

    @property
    def velocity(self):
        return (self.vx, self.vy)


# ---------------------------------------------------------------------------
# AI Enemy behaviors
# ---------------------------------------------------------------------------

class AIEnemy:
    """Scripted AI enemy that cycles through behaviors to exercise all states."""

    def __init__(self, x: float = -90.0, y: float = 90.0):
        self.robot = SimRobot(x, y, -math.pi / 4)  # facing toward center
        self._behavior = "wander"
        self._behavior_timer = 0.0
        self._behavior_duration = 5.0
        self._wander_angle = 0.0

        # Behavior schedule — triggers different state transitions
        self._schedule = [
            ("stand_still", 3.0),    # Easy first target
            ("rush_wall", 6.0),      # Goes to wall -> enables pin
            ("wander", 5.0),         # Move around
            ("rush_wall", 5.0),      # Another pin opportunity
            ("evade_fast", 4.0),     # Runs away -> lost_target
            ("hide", 3.0),           # Disappear -> lost detection
            ("wander", 4.0),
            ("rush_wall", 5.0),      # Pin
            ("stand_still", 3.0),
            ("rush_us", 4.0),        # Charge at us -> passive impact
            ("wander", 5.0),
            ("rush_wall", 5.0),      # Pin
            ("near_pit", 5.0),       # Near pit
            ("evade_fast", 3.0),
            ("rush_wall", 6.0),      # Pin
            ("wander", 5.0),
            ("stand_still", 3.0),
            ("rush_wall", 6.0),      # Pin
            ("rush_us", 3.0),        # Hit us
            ("rush_wall", 8.0),      # Final pins
            ("stand_still", 5.0),
            ("rush_wall", 5.0),
            ("wander", 10.0),
            ("rush_wall", 8.0),
        ]
        self._schedule_idx = 0

    def step(self, our_pos: tuple, dt: float, sim_time: float):
        """Update enemy position based on current behavior."""
        self._behavior_timer += dt

        # Advance behavior
        if self._behavior_timer >= self._behavior_duration:
            self._behavior_timer = 0.0
            self._schedule_idx = (self._schedule_idx + 1) % len(self._schedule)
            self._behavior, self._behavior_duration = self._schedule[self._schedule_idx]

        r = self.robot
        ox, oy = our_pos

        if self._behavior == "wander":
            self._wander_angle += (random.random() - 0.5) * 0.2
            r.heading += (self._wander_angle - r.heading) * 0.05
            r.step(0.3, None, dt)

        elif self._behavior == "rush_wall":
            # Drive to nearest wall
            nearest_wall_x = ARENA_HALF if r.x > 0 else -ARENA_HALF
            nearest_wall_y = ARENA_HALF if r.y > 0 else -ARENA_HALF
            # Pick whichever axis is closer to wall
            if abs(abs(r.x) - ARENA_HALF) < abs(abs(r.y) - ARENA_HALF):
                target_heading = math.atan2(0, nearest_wall_x - r.x)
            else:
                target_heading = math.atan2(nearest_wall_y - r.y, 0)
            r.heading += (target_heading - r.heading) * 0.1
            r.step(0.6, None, dt)

        elif self._behavior == "stand_still":
            r.step(0.0, None, dt)

        elif self._behavior == "evade_fast":
            # Run away from our robot
            away_angle = math.atan2(r.y - oy, r.x - ox)
            r.heading += (away_angle - r.heading) * 0.15
            r.step(0.8, None, dt)

        elif self._behavior == "hide":
            # Move to corner (simulates detection loss)
            corner_x = ARENA_HALF * 0.9
            corner_y = ARENA_HALF * 0.9
            target = math.atan2(corner_y - r.y, corner_x - r.x)
            r.heading += (target - r.heading) * 0.1
            r.step(0.4, None, dt)

        elif self._behavior == "near_pit":
            # Move near the pit
            pit_x, pit_y = 78.9, 109.3
            target = math.atan2(pit_y - r.y, pit_x - r.x)
            dist_to_pit = math.hypot(pit_x - r.x, pit_y - r.y)
            r.heading += (target - r.heading) * 0.1
            if dist_to_pit > 30:
                r.step(0.5, None, dt)
            else:
                r.step(0.1, None, dt)

        elif self._behavior == "rush_us":
            # Charge at our robot
            target = math.atan2(oy - r.y, ox - r.x)
            r.heading += (target - r.heading) * 0.15
            r.step(0.9, None, dt)

    @property
    def visible(self):
        """Whether enemy is 'detectable' by CV."""
        # Hide behavior = intermittent detection
        if self._behavior == "hide":
            return random.random() > 0.6  # 40% detection rate
        return True


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(duration: float, opening: str, strategy: str) -> list[str]:
    """Run a full match simulation and return JSONL lines."""

    # Install sim clock so time.perf_counter() is deterministic
    clock = SimClock()
    clock.install()

    try:
        return _run_simulation_inner(duration, opening, strategy, clock)
    finally:
        clock.uninstall()


def _run_simulation_inner(duration: float, opening: str, strategy: str, clock: SimClock) -> list[str]:

    cfg = BattleConfig(
        match_duration_s=duration,
        opening_strategy=opening,
        strategy=strategy,
        phase_start_s=30.0,
        phase_final_s=30.0,
        push_commit_s=1.0,
        stall_speed_threshold=8.0,
        pit_x_cm=78.9,
        pit_y_cm=109.3,
        pit_radius_cm=24.2,
        pit_danger_radius_cm=39.2,
    )
    match_timer = MatchTimer(
        duration_s=duration,
        phase_start_s=cfg.phase_start_s,
        phase_final_s=cfg.phase_final_s,
    )
    pin_timer = PinTimer()
    controller = BattleController(cfg, match_timer, pin_timer)

    our_robot = SimRobot(90.0, -90.0, 3 * math.pi / 4)  # lower-right, facing upper-left
    enemy = AIEnemy(x=-90.0, y=90.0)  # upper-left corner
    frames = []
    sim_time = 0.0
    enemy_frames_lost = 0

    # Log initial wait state
    frames.append(json.dumps({
        "f": 0, "t": 0.0, "mode": "battle", "bs": "wait", "mp": "start",
        "ox": round(our_robot.x, 1), "oy": round(our_robot.y, 1),
        "oh": round(our_robot.heading, 3), "od": True,
        "ovx": 0.0, "ovy": 0.0,
        "ex": round(enemy.robot.x, 1), "ey": round(enemy.robot.y, 1),
        "eh": round(enemy.robot.heading, 3),
        "evx": 0.0, "evy": 0.0, "ed": True, "et": True,
        "dist": round(math.hypot(enemy.robot.x - our_robot.x, enemy.robot.y - our_robot.y), 1),
        "thr": 0.0, "str": 0.0, "mr": duration, "urg": 0.0,
        "ax": 0.0, "ay": 0.0, "rm": 0, "fps": 60.0,
        **controller.debug_info,
    }, separators=(",", ":")))
    frame_num = 1

    # Start match
    clock.advance(DT)
    match_timer.start()
    initial_ctx = BattleContext(
        enemy_detected=True,
        enemy_tracking=True,
    )
    controller.start_match(initial_ctx)

    frame_num = 1  # frame 0 was the wait frame

    # ArUco loss simulation
    aruco_loss_trigger_time = 45.0 + random.random() * 30
    aruco_loss_duration = 4.0

    prev_state = controller.state

    while sim_time < duration + 5.0:  # +5s for victory dance
        # Advance sim clock FIRST so time.perf_counter() returns sim_time
        clock.advance(DT)

        # Check for victory dance entry
        if match_timer.is_expired and controller.state != "victory_dance" and not controller.is_dance_finished:
            controller.enter_victory_dance()

        # Check for dance completion
        if controller.is_dance_finished:
            if sim_time > duration + 3.5:
                break

        # Simulate ArUco loss
        our_detected = True
        if aruco_loss_trigger_time <= sim_time < aruco_loss_trigger_time + aruco_loss_duration:
            our_detected = False

        # Enemy AI step
        enemy.step(our_robot.pos, DT, sim_time)
        enemy_visible = enemy.visible

        # Track enemy frames lost (like real CV system)
        if enemy_visible:
            enemy_frames_lost = 0
        else:
            enemy_frames_lost += 1

        enemy_pos = enemy.robot.pos if enemy_visible else None
        dist = math.hypot(enemy.robot.x - our_robot.x, enemy.robot.y - our_robot.y) if enemy_visible else 999.0

        # Simulate accel (noise + impact spikes)
        base_accel = abs(our_robot.speed) * 3.0 + random.gauss(0, 30)
        # Spike when near wall and moving fast
        if (abs(our_robot.x) > ARENA_HALF - 5 or abs(our_robot.y) > ARENA_HALF - 5) and our_robot.speed > 30:
            base_accel += 1500  # wall impact spike
        # Spike when enemy rushes us and is close
        if enemy._behavior == "rush_us" and dist < 15:
            base_accel += 2500  # passive impact

        accel_x = base_accel * math.cos(our_robot.heading) + random.gauss(0, 20)
        accel_y = base_accel * math.sin(our_robot.heading) + random.gauss(0, 20)

        # Build context
        ctx = BattleContext(
            our_pos=our_robot.pos if our_detected else (our_robot.x + random.gauss(0, 2), our_robot.y + random.gauss(0, 2)),
            our_heading_rad=our_robot.heading,
            our_velocity=our_robot.velocity if our_detected else (0.0, 0.0),
            enemy_pos=enemy_pos,
            enemy_heading_rad=enemy.robot.heading if enemy_visible else None,
            enemy_velocity=enemy.robot.velocity if enemy_visible else None,
            enemy_detected=enemy_visible,
            enemy_tracking=enemy_visible,
            frames_without_detection=enemy_frames_lost,
            distance_cm=dist,
            dt=DT,
            our_detected=our_detected,
            accel_x_mg=accel_x,
            accel_y_mg=accel_y,
            throttle_cmd=our_robot.speed / our_robot.max_speed_cm_s,
            imu_heading_deg=math.degrees(our_robot.heading),
            imu_pitch_deg=0.0,
            imu_roll_deg=0.0,
        )

        # Tick the state machine
        output = controller.tick(ctx)

        # Log state transitions
        current_state = controller.state
        if current_state != prev_state:
            print(f"  [{sim_time:6.1f}s] {prev_state} -> {current_state}")
            prev_state = current_state

        # Apply output to our robot physics
        if output.target_omega_dps is not None:
            our_robot.step(output.target_speed, output.target_omega_dps, DT)
        else:
            our_robot.step(output.throttle, None, DT)
            # Legacy steering -> heading change (stronger response for sim)
            our_robot.heading += output.steering * 3.0 * DT

        # Build frame record
        mr = match_timer.remaining_s if match_timer.is_running else 0.0
        urg = match_timer.urgency

        rec = {
            "f": frame_num,
            "t": round(sim_time, 4),
            "mode": "battle",
            "bs": current_state,
            "mp": match_timer.phase,
            "ox": round(our_robot.x, 1),
            "oy": round(our_robot.y, 1),
            "oh": round(our_robot.heading, 3),
            "od": our_detected,
            "ovx": round(our_robot.vx, 1),
            "ovy": round(our_robot.vy, 1),
            "ex": round(enemy.robot.x, 1) if enemy_visible else None,
            "ey": round(enemy.robot.y, 1) if enemy_visible else None,
            "eh": round(enemy.robot.heading, 3) if enemy_visible else None,
            "evx": round(enemy.robot.vx, 1) if enemy_visible else None,
            "evy": round(enemy.robot.vy, 1) if enemy_visible else None,
            "ed": enemy_visible,
            "et": enemy_visible,
            "dist": round(dist, 1),
            "thr": round(output.throttle, 3),
            "str": round(output.steering, 3),
            "mr": round(mr, 1),
            "urg": round(urg, 3),
            "ax": round(accel_x, 0),
            "ay": round(accel_y, 0),
            "rm": 1 if output.target_omega_dps is not None else 0,
            "fps": 60.0,
            "ehm": "velocity" if enemy_visible else None,
            "ehc": round(0.7 + 0.3 * random.random(), 2) if enemy_visible else 0.0,
            **controller.debug_info,
        }
        frames.append(json.dumps(rec, separators=(",", ":")))

        sim_time += DT
        frame_num += 1

    return frames


def gen_arena_meta():
    return {
        "origin_x": 640,
        "origin_y": 400,
        "px_per_cm": 2.5,
        "frame_w": 1280,
        "frame_h": 800,
        "arena_width_cm": 244.0,
        "arena_height_cm": 244.0,
        "pit_x_cm": 78.9,
        "pit_y_cm": 109.3,
        "pit_radius_cm": 24.2,
        "pit_danger_radius_cm": 39.2,
    }


def gen_arena_image(meta):
    import numpy as np
    try:
        import cv2
    except ImportError:
        print("cv2 not available — skipping arena image")
        return None

    w, h = meta["frame_w"], meta["frame_h"]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 25, 20)
    ox, oy = int(meta["origin_x"]), int(meta["origin_y"])
    s = meta["px_per_cm"]
    half_w = int(meta["arena_width_cm"] / 2 * s)
    half_h = int(meta["arena_height_cm"] / 2 * s)
    cv2.rectangle(img, (ox - half_w, oy - half_h), (ox + half_w, oy + half_h), (60, 60, 60), 2)
    for gcm in range(-120, 121, 30):
        gpx = int(ox + gcm * s)
        cv2.line(img, (gpx, oy - half_h), (gpx, oy + half_h), (35, 30, 25), 1)
        gpy = int(oy + gcm * s)
        cv2.line(img, (ox - half_w, gpy), (ox + half_w, gpy), (35, 30, 25), 1)
    cv2.drawMarker(img, (ox, oy), (0, 80, 0), cv2.MARKER_CROSS, 20, 1)
    pit_cx = int(ox + meta["pit_x_cm"] * s)
    pit_cy = int(oy + meta["pit_y_cm"] * s)
    pit_r = int(meta["pit_radius_cm"] * s)
    cv2.rectangle(img, (pit_cx - pit_r, pit_cy - pit_r), (pit_cx + pit_r, pit_cy + pit_r), (0, 0, 120), 2)
    cv2.putText(img, "PIT", (pit_cx - 15, pit_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1)
    noise = np.random.randint(0, 8, (h, w), dtype=np.uint8)
    img[:, :, 0] = np.clip(img[:, :, 0].astype(int) + noise, 0, 255).astype(np.uint8)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full state machine simulation")
    parser.add_argument("--duration", type=float, default=180.0, help="Match duration in seconds")
    parser.add_argument("--opening", type=str, default="charge", help="Opening strategy")
    parser.add_argument("--strategy", type=str, default="charge", help="Main strategy")
    args = parser.parse_args()

    print(f"=== Full Match Simulation ===")
    print(f"Duration: {args.duration}s | Opening: {args.opening} | Strategy: {args.strategy}")
    print()

    t0 = time.perf_counter()
    lines = run_simulation(args.duration, args.opening, args.strategy)
    elapsed = time.perf_counter() - t0

    print(f"\nSimulation complete: {len(lines)} frames in {elapsed:.2f}s ({len(lines)/elapsed:.0f} fps)")

    # Count states
    from collections import Counter
    states = Counter()
    phases = Counter()
    for line in lines:
        rec = json.loads(line)
        states[rec["bs"]] += 1
        if rec.get("mp"):
            phases[rec["mp"]] += 1

    print(f"\nStates hit ({len(states)}):")
    for s, c in states.most_common():
        print(f"  {s}: {c} frames ({c/len(lines)*100:.1f}%)")
    print(f"\nPhases: {dict(phases)}")

    # Write output
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "sim_full_match.jsonl")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {len(lines)} frames to {out_path}")

    meta = gen_arena_meta()
    meta_path = out_path.replace(".jsonl", "_arena.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote arena metadata to {meta_path}")

    img = gen_arena_image(meta)
    if img is not None:
        import cv2
        img_path = out_path.replace(".jsonl", "_arena.png")
        cv2.imwrite(img_path, img)
        print(f"Wrote arena image to {img_path}")

    # Validation
    missing_states = {"wait", "acquire", "charge_pursue", "pin", "evade_retreat",
                      "lost_target", "victory_dance"} - set(states.keys())
    if missing_states:
        print(f"\nWARNING: Expected states not hit: {missing_states}")
    else:
        print(f"\nOK: All core states exercised")

    missing_phases = {"start", "mid", "final", "post"} - set(phases.keys())
    if missing_phases:
        print(f"WARNING: Expected phases not hit: {missing_phases}")
    else:
        print(f"OK: All phases exercised")
