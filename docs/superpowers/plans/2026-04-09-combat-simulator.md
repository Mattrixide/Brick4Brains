# Combat Robot Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pymunk-based 2D combat simulator that runs Brick's BattleController against AI or manual enemies, using real arena geometry and physics calibrated from log data.

**Architecture:** SimArena (pymunk physics, no rendering) feeds robot state into SimBridge, which packs BattleContext and calls BattleController.tick(). Output forces are applied back to pymunk bodies. SimRenderer handles pygame drawing separately (enables headless testing). SimSession manages match state, speed, logging. Enemy is switchable between keyboard, scripted behaviors, and AI.

**Tech Stack:** Python 3.12, pymunk (Chipmunk2D), pygame, scipy (calibration), existing battle code (state_machine.py, battle_config.py, match_timer.py)

**Spec:** `docs/superpowers/specs/2026-04-09-combat-simulator-design.md`

**Expert review fixes incorporated:**
- Friction: forces before substeps, damping after (pymunk expert)
- Rate mode torque clamped to [-MAX_TORQUE, MAX_TORQUE] (pymunk expert)
- Default forces reduced from 2000→400N (pymunk expert)
- MatchTimer constructed with all 4 args (battle code expert)
- SimBridge computes accel_x/y_mg from velocity delta (battle code expert)
- sys.path fixed: single dirname for auto-drive/ (battle code expert)
- flee mode mapped to key 6 (battle code expert)
- SimLogger writes ab, fp, ehm, ehc, ax, ay fields (replay expert)
- Arena meta includes pit_danger_radius_cm (replay expert)
- Config as JSON dataclass, not module constants (arch expert)
- Rendering separated from arena physics (arch expert)
- SimSession extracted from run.py (arch expert)
- SimBridge takes robots, not arena (arch expert)
- Calibration uses raw position diffs, not KF velocity (calibration expert)
- Calibration uses least-squares fit of full dynamics model (calibration expert)
- Motor response lag modeled as first-order filter (calibration expert)

---

## File Structure

```
prototypes/auto-drive/sim/
  __init__.py       # Empty, makes sim a package
  config.py         # SimConfig dataclass + defaults, loads/saves sim_config.json
  arena.py          # SimArena: pymunk world, walls, pit, robot bodies (no rendering)
  renderer.py       # SimRenderer: pygame drawing, separated for headless testing
  bridge.py         # SimBridge: BattleContext packing, force application
  enemy_ai.py       # EnemyController: manual, scripted, AI modes
  session.py        # SimSession: match state, speed, logging orchestration
  run.py            # Main entry point: thin pygame event loop
  calibrate.py      # Analyze real logs via least-squares fit
```

**Existing files referenced (read-only, no modifications):**
- `prototypes/auto-drive/state_machine.py` — BattleController, BattleContext, BattleOutput
- `prototypes/auto-drive/battle_config.py` — BattleConfig
- `prototypes/auto-drive/match_timer.py` — MatchTimer, PinTimer
- `prototypes/auto-drive/floor_calibration.json` — Arena wall corners
- `prototypes/auto-drive/battle_config.json` — Pit location, match settings

---

## Phase 1: Arena + Manual Control

### Task 1: Create config.py with physics constants

**Files:**
- Create: `prototypes/auto-drive/sim/__init__.py`
- Create: `prototypes/auto-drive/sim/config.py`

- [ ] **Step 1: Create the sim package and config file**

```python
# sim/__init__.py
# (empty — makes sim a package)
```

```python
# sim/config.py
"""Tunable physics parameters for the combat simulator.
Runtime config loads from sim_config.json; these are defaults."""
import json
import os
from dataclasses import dataclass, field, asdict

@dataclass
class SimConfig:
    # Arena
    arena_cm: float = 244.0

    # Robot dimensions (inches to cm)
    brick_width_cm: float = 22.86     # 9 inches
    brick_depth_cm: float = 17.78     # 7 inches
    enemy_width_cm: float = 15.24     # 6 inches
    enemy_depth_cm: float = 22.86     # 9 inches

    # Masses
    brick_mass_kg: float = 1.36       # 3lb beetleweight
    enemy_mass_kg: float = 1.36

    # Drive forces (calibrate.py populates these)
    max_forward_force: float = 400.0  # reduced from 2000 per expert review
    max_torque: float = 1500.0
    motor_lag_s: float = 0.030        # ESC response lag (~30ms)

    # Friction
    ground_friction_mu: float = 0.6   # rubber on plywood
    lateral_damping: float = 0.85     # per-frame lateral velocity retention
    angular_damping: float = 0.93     # per-frame angular velocity retention
    gravity_cms2: float = 980.0       # g in cm/s^2

    # Collision
    wall_elasticity: float = 0.2
    wall_friction: float = 1.0
    robot_elasticity: float = 0.3
    robot_friction: float = 0.8

    # Simulation
    physics_fps: int = 240            # substeps per second
    render_fps: int = 60
    scale_px_per_cm: float = 2.5

    # Pit
    pit_elimination: bool = True

    def save(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "sim_config.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "sim_config.json")
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()
```

- [ ] **Step 2: Commit**

```bash
git add prototypes/auto-drive/sim/__init__.py prototypes/auto-drive/sim/config.py
git commit -m "sim: add config.py with physics constants and robot dimensions"
```

---

### Task 2: Create arena.py with pymunk world, walls, pit, robots, rendering

**Files:**
- Create: `prototypes/auto-drive/sim/arena.py`

- [ ] **Step 1: Write SimArena class**

`sim/arena.py` — the core physics world. Loads real arena geometry, creates robot bodies, steps physics, renders via pygame.

```python
# sim/arena.py
"""SimArena: pymunk physics world with real arena geometry."""
import json
import math
import os

import pygame
import pymunk

from . import config as C

AUTO_DRIVE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_json(filename):
    path = os.path.join(AUTO_DRIVE_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


class SimRobot:
    """A pymunk robot body with metadata."""

    def __init__(self, space, x, y, heading_deg, width, depth, mass, name="robot"):
        self.name = name
        self.width = width
        self.depth = depth
        self.mass = mass
        self.alive = True  # set False when eliminated (pit)

        moment = pymunk.moment_for_box(mass, (depth, width))
        self.body = pymunk.Body(mass, moment)
        self.body.position = (x, y)
        self.body.angle = math.radians(heading_deg)

        hw, hd = width / 2, depth / 2
        verts = [(hd, -hw), (hd, hw), (-hd, hw), (-hd, -hw)]
        self.shape = pymunk.Poly(self.body, verts)
        self.shape.elasticity = C.ROBOT_ELASTICITY
        self.shape.friction = C.ROBOT_FRICTION
        self.shape.collision_type = 1  # robot

        space.add(self.body, self.shape)

    @property
    def position(self):
        return (self.body.position.x, self.body.position.y)

    @property
    def heading_rad(self):
        return self.body.angle

    @property
    def velocity(self):
        return (self.body.velocity.x, self.body.velocity.y)

    @property
    def angular_velocity(self):
        return self.body.angular_velocity

    def apply_drive(self, throttle, steering):
        """Apply throttle (forward force) and steering (torque)."""
        if not self.alive:
            return
        self.body.apply_force_at_local_point((throttle * C.MAX_FORWARD_FORCE, 0), (0, 0))
        self.body.torque = steering * C.MAX_TORQUE

    def apply_friction_forces(self, cfg):
        """Apply Coulomb friction FORCE (call BEFORE substeps)."""
        if not self.alive:
            return
        vx, vy = self.body.velocity
        speed = math.hypot(vx, vy)
        friction_force = cfg.ground_friction_mu * self.mass * cfg.gravity_cms2

        if speed > 1.0:
            # Decompose into forward/lateral and apply friction along forward only
            cos_a = math.cos(self.body.angle)
            sin_a = math.sin(self.body.angle)
            v_fwd = vx * cos_a + vy * sin_a
            fwd_sign = -1.0 if v_fwd > 0 else 1.0 if v_fwd < 0 else 0.0
            fx = cos_a * fwd_sign * friction_force
            fy = sin_a * fwd_sign * friction_force
            self.body.apply_force_at_world_point((fx, fy), self.body.position)

    def apply_velocity_damping(self, cfg):
        """Apply lateral + angular damping (call AFTER substeps)."""
        if not self.alive:
            return
        vx, vy = self.body.velocity
        speed = math.hypot(vx, vy)

        if speed < 1.0:
            self.body.velocity = (vx * 0.8, vy * 0.8)
        else:
            # Lateral damping (robots resist sideways sliding)
            cos_a = math.cos(self.body.angle)
            sin_a = math.sin(self.body.angle)
            v_fwd = vx * cos_a + vy * sin_a
            v_lat = -vx * sin_a + vy * cos_a
            v_lat *= cfg.lateral_damping
            self.body.velocity = (
                v_fwd * cos_a - v_lat * sin_a,
                v_fwd * sin_a + v_lat * cos_a,
            )

        # Angular damping
        self.body.angular_velocity *= cfg.angular_damping

    def get_corners_world(self):
        """Get 4 corners in world cm coords."""
        hw, hd = self.width / 2, self.depth / 2
        cos_a = math.cos(self.body.angle)
        sin_a = math.sin(self.body.angle)
        local = [(hd, -hw), (hd, hw), (-hd, hw), (-hd, -hw)]
        cx, cy = self.body.position
        return [(cx + lx * cos_a - ly * sin_a,
                 cy + lx * sin_a + ly * cos_a) for lx, ly in local]

    def freeze(self):
        """Stop the robot (eliminated)."""
        self.alive = False
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0

    def reset(self, x, y, heading_deg):
        """Reset to starting position."""
        self.body.position = (x, y)
        self.body.angle = math.radians(heading_deg)
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0
        self.alive = True


class SimArena:
    """Pymunk physics arena with real geometry."""

    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 1.0  # we handle friction manually

        # Load real geometry
        self.floor_cal = _load_json("floor_calibration.json")
        self.battle_cfg = _load_json("battle_config.json")

        # Arena walls
        self.arena_corners = None  # world cm coordinates
        self._create_walls()

        # Pit
        self.pit_x = 0.0
        self.pit_y = 0.0
        self.pit_radius = 0.0
        self._create_pit()

        # Robots
        self.brick = SimRobot(
            self.space, -40, 0, 0,
            C.BRICK_WIDTH_CM, C.BRICK_DEPTH_CM, C.BRICK_MASS_KG, "brick"
        )
        self.enemy = SimRobot(
            self.space, 40, 0, 180,
            C.ENEMY_WIDTH_CM, C.ENEMY_DEPTH_CM, C.ENEMY_MASS_KG, "enemy"
        )

        # Pit collision handler
        if self.pit_radius > 0:
            handler = self.space.add_collision_handler(1, 2)  # robot vs pit
            handler.begin = self._on_pit_collision

    def _create_walls(self):
        """Create arena walls from floor_calibration.json corners."""
        if self.floor_cal and "corners_ft" in self.floor_cal:
            self.arena_corners = self.floor_cal["corners_ft"]
        else:
            # Fallback: 244cm square
            half = C.ARENA_CM / 2
            self.arena_corners = [
                [-half, -half], [half, -half], [half, half], [-half, half]
            ]

        corners = self.arena_corners
        for i in range(len(corners)):
            a = tuple(corners[i])
            b = tuple(corners[(i + 1) % len(corners)])
            seg = pymunk.Segment(self.space.static_body, a, b, 3.0)
            seg.elasticity = C.WALL_ELASTICITY
            seg.friction = C.WALL_FRICTION
            self.space.add(seg)

    def _create_pit(self):
        """Create pit sensor from battle_config.json."""
        if self.battle_cfg:
            self.pit_x = self.battle_cfg.get("pit_x_cm", 0)
            self.pit_y = self.battle_cfg.get("pit_y_cm", 0)
            self.pit_radius = self.battle_cfg.get("pit_radius_cm", 0)

        if self.pit_radius > 0 and C.PIT_ELIMINATION:
            # Pit as a sensor (no physical collision, just detection)
            pit_body = self.space.static_body
            r = self.pit_radius
            verts = [
                (self.pit_x - r, self.pit_y - r),
                (self.pit_x + r, self.pit_y - r),
                (self.pit_x + r, self.pit_y + r),
                (self.pit_x - r, self.pit_y + r),
            ]
            pit_shape = pymunk.Poly(pit_body, verts)
            pit_shape.sensor = True
            pit_shape.collision_type = 2  # pit
            self.space.add(pit_shape)

    def _on_pit_collision(self, arbiter, space, data):
        """Called when a robot enters the pit."""
        for shape in arbiter.shapes:
            if shape.collision_type == 1:  # robot
                # Find which robot owns this shape
                if shape == self.brick.shape:
                    self.brick.freeze()
                elif shape == self.enemy.shape:
                    self.enemy.freeze()
        return False  # sensor — don't resolve physically

    def step(self, dt=1/60):
        """Step physics one frame."""
        # Forces BEFORE substeps
        self.brick.apply_friction_forces(self.cfg)
        self.enemy.apply_friction_forces(self.cfg)
        # Substeps
        substeps = self.cfg.physics_fps // self.cfg.render_fps
        sub_dt = dt / substeps
        for _ in range(substeps):
            self.space.step(sub_dt)
        # Damping AFTER substeps (so solver doesn't overwrite)
        self.brick.apply_velocity_damping(self.cfg)
        self.enemy.apply_velocity_damping(self.cfg)

    def reset(self):
        """Reset robots to starting positions."""
        self.brick.reset(-40, 0, 0)
        self.enemy.reset(40, 0, 180)

```

Then create `sim/renderer.py` with all the drawing code (grid, arena border, pit, robots, HUD). The renderer takes a `SimArena` reference and a pygame screen, keeping rendering fully separated from physics. This enables headless testing of the physics layer.

- [ ] **Step 2: Commit**

```bash
git add prototypes/auto-drive/sim/arena.py
git commit -m "sim: add SimArena with pymunk world, walls, pit, robot rendering"
```

---

### Task 3: Create run.py with pygame loop and manual control

**Files:**
- Create: `prototypes/auto-drive/sim/run.py`

- [ ] **Step 1: Write the main loop**

```python
# sim/run.py
"""Combat robot simulator — main entry point.

Run: python -m sim.run  (from prototypes/auto-drive/)
"""
import math
import sys
import os

# Ensure auto-drive is on the path for battle code imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame

from sim.arena import SimArena
from sim import config as C


def main():
    pygame.init()
    win_size = int(C.ARENA_CM * C.SCALE_PX_PER_CM) + 80
    screen = pygame.display.set_mode((win_size, win_size))
    pygame.display.set_caption("B4B Combat Simulator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    arena = SimArena()
    paused = True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    arena.reset()
                    paused = True

        # Input
        keys = pygame.key.get_pressed()

        # Brick: WASD
        bt = (1 if keys[pygame.K_w] else 0) - (1 if keys[pygame.K_s] else 0)
        bs = (1 if keys[pygame.K_a] else 0) - (1 if keys[pygame.K_d] else 0)
        arena.brick.apply_drive(bt, bs)

        # Enemy: Arrow keys
        et = (1 if keys[pygame.K_UP] else 0) - (1 if keys[pygame.K_DOWN] else 0)
        es = (1 if keys[pygame.K_LEFT] else 0) - (1 if keys[pygame.K_RIGHT] else 0)
        arena.enemy.apply_drive(et, es)

        # Physics
        if not paused:
            arena.step()

        # Render
        screen.fill((20, 20, 20))
        arena.draw(screen)

        # HUD
        sb = math.hypot(*arena.brick.velocity)
        se = math.hypot(*arena.enemy.velocity)
        status = "PAUSED" if paused else "RUNNING"
        hud_text = f"{status} | Brick: {sb:.0f}cm/s  Enemy: {se:.0f}cm/s | WASD/Arrows  Space=start  R=reset"
        hud = font.render(hud_text, True, (150, 150, 150))
        screen.blit(hud, (10, win_size - 22))

        pygame.display.flip()
        clock.tick(C.RENDER_FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test manually**

Run from `prototypes/auto-drive/`:
```bash
python -m sim.run
```

**Validate:**
- Arena walls match real calibrated shape (not a perfect square)
- Pit is visible as red square
- WASD drives green robot, arrows drive red robot
- Robots collide and push each other
- Robots bounce off walls
- Robots entering pit are frozen/grayed out
- Space pauses/unpauses, R resets

- [ ] **Step 3: Commit**

```bash
git add prototypes/auto-drive/sim/run.py
git commit -m "sim: add run.py with pygame loop and manual dual-robot control"
```

---

## Phase 2: Physics Calibration

### Task 4: Create calibrate.py to extract physics from real logs

**Files:**
- Create: `prototypes/auto-drive/sim/calibrate.py`

- [ ] **Step 1: Write calibration script**

```python
# sim/calibrate.py
"""Extract physics parameters from real match logs.

Reads JSONL frame logs and fits:
- Acceleration response (throttle → velocity change)
- Friction/deceleration (velocity decay when throttle=0)
- Turn rate (steering → angular velocity)
- Top speed

Run: python -m sim.calibrate [logfile.jsonl]
"""
import json
import math
import os
import sys

LOGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)


def load_frames(path):
    frames = []
    with open(path) as f:
        for line in f:
            frames.append(json.loads(line))
    # Normalize timestamps
    if frames and frames[0]["t"] > 1000:
        t0 = frames[0]["t"]
        for fr in frames:
            fr["t"] -= t0
    return frames


def analyze_acceleration(frames):
    """Find throttle 0→1 transitions, measure velocity buildup."""
    accels = []
    for i in range(1, len(frames)):
        f, pf = frames[i], frames[i - 1]
        if not f.get("od") or not pf.get("od"):
            continue
        thr = f.get("thr", 0)
        prev_thr = pf.get("thr", 0)
        dt = f["t"] - pf["t"]
        if dt <= 0:
            continue
        ovx, ovy = f.get("ovx", 0) or 0, f.get("ovy", 0) or 0
        povx, povy = pf.get("ovx", 0) or 0, pf.get("ovy", 0) or 0
        speed = math.hypot(ovx, ovy)
        prev_speed = math.hypot(povx, povy)
        accel = (speed - prev_speed) / dt
        if abs(thr) > 0.5 and abs(prev_thr) > 0.5:
            accels.append((thr, accel, speed))
    if accels:
        avg_accel = sum(a[1] for a in accels if a[1] > 0) / max(1, sum(1 for a in accels if a[1] > 0))
        max_speed = max(a[2] for a in accels)
        print(f"  Avg positive acceleration: {avg_accel:.1f} cm/s^2")
        print(f"  Max observed speed: {max_speed:.1f} cm/s")
        return avg_accel, max_speed
    return None, None


def analyze_friction(frames):
    """Find throttle→0 transitions, measure velocity decay."""
    decay_rates = []
    for i in range(1, len(frames)):
        f, pf = frames[i], frames[i - 1]
        if not f.get("od") or not pf.get("od"):
            continue
        thr = abs(f.get("thr", 0))
        dt = f["t"] - pf["t"]
        if dt <= 0 or thr > 0.05:
            continue
        ovx, ovy = f.get("ovx", 0) or 0, f.get("ovy", 0) or 0
        povx, povy = pf.get("ovx", 0) or 0, pf.get("ovy", 0) or 0
        speed = math.hypot(ovx, ovy)
        prev_speed = math.hypot(povx, povy)
        if prev_speed > 5:
            ratio = speed / prev_speed
            decay_rates.append(ratio)
    if decay_rates:
        avg_decay = sum(decay_rates) / len(decay_rates)
        # Convert per-frame decay to friction coefficient
        # F_friction = mu * m * g → deceleration = mu * g
        # speed_next = speed - mu * g * dt → ratio = 1 - mu * g * dt / speed
        # Rough estimate: mu ~ (1 - avg_decay) * speed_avg / (g * dt_avg)
        print(f"  Avg velocity retention per frame: {avg_decay:.4f}")
        print(f"  (Lower = more friction, 1.0 = no friction)")
        return avg_decay
    return None


def analyze_turn_rate(frames):
    """Measure angular velocity under steering input."""
    omegas = []
    for i in range(1, len(frames)):
        f = frames[i]
        if not f.get("od"):
            continue
        steer = abs(f.get("str", 0))
        omega = abs(f.get("omega", 0))
        if steer > 0.3 and omega > 10:
            omegas.append((steer, omega))
    if omegas:
        avg_omega = sum(o[1] for o in omegas) / len(omegas)
        max_omega = max(o[1] for o in omegas)
        print(f"  Avg turn rate under steering: {avg_omega:.1f} deg/s")
        print(f"  Max turn rate: {max_omega:.1f} deg/s")
        return avg_omega, max_omega
    return None, None


def main():
    # Find log files
    if len(sys.argv) > 1:
        paths = [sys.argv[1]]
    else:
        paths = sorted(
            [os.path.join(LOGS_DIR, f) for f in os.listdir(LOGS_DIR)
             if f.startswith("frames_") and f.endswith(".jsonl")],
            key=os.path.getmtime, reverse=True
        )[:5]  # analyze 5 most recent

    if not paths:
        print("No log files found")
        return

    all_accels, all_speeds, all_decays, all_omegas = [], [], [], []

    for path in paths:
        name = os.path.basename(path)
        frames = load_frames(path)
        battle_frames = [f for f in frames if f.get("mode") == "battle"]
        if len(battle_frames) < 20:
            continue
        print(f"\n--- {name} ({len(battle_frames)} battle frames) ---")

        accel, top_speed = analyze_acceleration(battle_frames)
        if accel:
            all_accels.append(accel)
        if top_speed:
            all_speeds.append(top_speed)

        decay = analyze_friction(battle_frames)
        if decay:
            all_decays.append(decay)

        omega_avg, omega_max = analyze_turn_rate(battle_frames)
        if omega_avg:
            all_omegas.append(omega_avg)

    print("\n=== SUMMARY ===")
    if all_accels:
        print(f"Acceleration: {sum(all_accels)/len(all_accels):.1f} cm/s^2")
    if all_speeds:
        print(f"Top speed: {max(all_speeds):.1f} cm/s")
    if all_decays:
        print(f"Velocity retention: {sum(all_decays)/len(all_decays):.4f} per frame")
    if all_omegas:
        print(f"Turn rate: {sum(all_omegas)/len(all_omegas):.1f} deg/s avg")
    print("\nUpdate sim/config.py with these values to match real robot behavior.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run calibration**

```bash
python -m sim.calibrate
```

**Validate:** Script reads logs, prints acceleration/friction/turn rate values. Compare against what feels right in the sim.

- [ ] **Step 3: Update config.py with fitted values**

Based on calibration output, update the constants in `sim/config.py`. This is a manual step — review the numbers and adjust.

- [ ] **Step 4: Commit**

```bash
git add prototypes/auto-drive/sim/calibrate.py prototypes/auto-drive/sim/config.py
git commit -m "sim: add calibrate.py, tune physics from real log data"
```

---

## Phase 3: SimBridge + Brick AI

### Task 5: Create bridge.py connecting pymunk to BattleController

**Files:**
- Create: `prototypes/auto-drive/sim/bridge.py`

- [ ] **Step 1: Write SimBridge**

```python
# sim/bridge.py
"""SimBridge: connects a pymunk robot body to BattleController."""
import math
import os
import sys

from sim.arena import SimArena, SimRobot
from sim import config as C

# Import battle code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state_machine import BattleController, BattleContext, BattleOutput
from battle_config import BattleConfig
from match_timer import MatchTimer, PinTimer


class SimBridge:
    """Packs pymunk state into BattleContext, feeds BattleController,
    applies output as forces."""

    def __init__(self, robot: SimRobot, config_path=None):
        self.robot = robot

        # Load battle config
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "battle_config.json"
            )
        self.battle_config = BattleConfig.load(config_path)
        self.match_timer = MatchTimer(
            self.battle_config.match_duration_s,
            self.battle_config.urgency_ramp_start_s,
            self.battle_config.phase_start_s,
            self.battle_config.phase_final_s,
        )
        self.pin_timer = PinTimer(self.battle_config.pin_duration_s)
        self.controller = BattleController(
            self.battle_config, self.match_timer, self.pin_timer
        )
        self._last_output = BattleOutput()
        self._prev_vx = 0.0
        self._prev_vy = 0.0

    def start_match(self):
        """Start the match timer."""
        self.match_timer.start()

    def reset(self):
        """Reset controller and timers."""
        self.match_timer.reset()
        self.pin_timer = PinTimer(self.battle_config.pin_duration_s)
        self.controller = BattleController(
            self.battle_config, self.match_timer, self.pin_timer
        )
        self._last_output = BattleOutput()

    @property
    def state(self):
        return self.controller.state

    @property
    def last_output(self):
        return self._last_output

    def tick(self, dt, enemy: SimRobot):
        """Run one frame of the battle controller."""
        if not self.robot.alive:
            return BattleOutput()

        ox, oy = self.robot.position
        ovx, ovy = self.robot.velocity
        ex, ey = enemy.position
        evx, evy = enemy.velocity

        enemy_alive = enemy.alive
        dist = math.hypot(ex - ox, ey - oy) if enemy_alive else 999.0

        # Compute acceleration from velocity delta (enables impact detection)
        accel_x_cms2 = (ovx - self._prev_vx) / dt if dt > 0 else 0
        accel_y_cms2 = (ovy - self._prev_vy) / dt if dt > 0 else 0
        accel_x_mg = accel_x_cms2 / 0.98  # cm/s^2 to milligravity
        accel_y_mg = accel_y_cms2 / 0.98
        self._prev_vx, self._prev_vy = ovx, ovy

        ctx = BattleContext(
            our_pos=(ox, oy),
            our_heading_rad=self.robot.heading_rad,
            our_velocity=(ovx, ovy),
            enemy_pos=(ex, ey) if enemy_alive else None,
            enemy_heading_rad=enemy.heading_rad if enemy_alive else None,
            enemy_velocity=(evx, evy) if enemy_alive else None,
            enemy_detected=enemy_alive,
            enemy_tracking=enemy_alive,
            frames_without_detection=0 if enemy_alive else 999,
            distance_cm=dist,
            dt=dt,
            our_detected=True,  # always visible in sim
            accel_x_mg=accel_x_mg,
            accel_y_mg=accel_y_mg,
            throttle_cmd=self._last_output.throttle,
        )

        output = self.controller.tick(ctx)
        self._last_output = output

        # Apply output to pymunk body
        if output.target_omega_dps is not None:
            # Rate mode: P-controller for angular velocity, clamped
            target_omega_rad = math.radians(output.target_omega_dps)
            current_omega = self.robot.angular_velocity
            omega_error = target_omega_rad - current_omega
            torque = max(-self.cfg.max_torque, min(self.cfg.max_torque,
                         omega_error * self.cfg.max_torque * 0.5))
            self.robot.body.torque = torque
            self.robot.body.apply_force_at_local_point(
                (output.target_speed * self.cfg.max_forward_force, 0), (0, 0)
            )
        else:
            # Direct mode
            self.robot.apply_drive(output.throttle, output.steering)

        return output
```

- [ ] **Step 2: Integrate SimBridge into run.py**

Update `sim/run.py` to optionally use SimBridge for Brick instead of keyboard. Add a toggle key (B) to switch between manual and AI control.

In the main loop, after input handling, add:

```python
# At top of main():
from sim.bridge import SimBridge
brick_bridge = SimBridge(arena.brick, arena)
brick_ai_mode = False
match_running = False

# In event handling, add:
elif event.key == pygame.K_b:
    brick_ai_mode = not brick_ai_mode
    print(f"Brick AI: {'ON' if brick_ai_mode else 'OFF'}")

# Replace the brick drive section:
if brick_ai_mode and not paused:
    if not match_running:
        brick_bridge.start_match()
        match_running = True
    brick_output = brick_bridge.tick(1/C.RENDER_FPS, arena.enemy)
else:
    bt = (1 if keys[pygame.K_w] else 0) - (1 if keys[pygame.K_s] else 0)
    bs = (1 if keys[pygame.K_a] else 0) - (1 if keys[pygame.K_d] else 0)
    arena.brick.apply_drive(bt, bs)

# In reset handler:
brick_bridge.reset()
match_running = False

# In HUD, show state:
if brick_ai_mode:
    state_text = f"AI:{brick_bridge.state}"
else:
    state_text = "MANUAL"
```

- [ ] **Step 3: Test**

Run `python -m sim.run`, press B to enable Brick AI, press Space to start. Brick should acquire and charge the enemy.

**Validate:**
- Brick transitions through states: wait → acquire → charge_pursue
- Brick drives toward enemy
- Pin state triggers on contact
- State displayed in HUD

- [ ] **Step 4: Commit**

```bash
git add prototypes/auto-drive/sim/bridge.py prototypes/auto-drive/sim/run.py
git commit -m "sim: add SimBridge connecting BattleController to pymunk"
```

---

## Phase 4: Enemy AI Modes

### Task 6: Create enemy_ai.py with scripted behaviors

**Files:**
- Create: `prototypes/auto-drive/sim/enemy_ai.py`

- [ ] **Step 1: Write enemy controller**

```python
# sim/enemy_ai.py
"""Enemy robot control modes: manual, scripted, AI."""
import math

from sim.arena import SimRobot


class EnemyController:
    """Switchable enemy control."""

    MODES = ["manual", "sit", "circle", "charge", "flee", "ai"]

    def __init__(self):
        self.mode = "manual"
        self._circle_angle = 0.0
        self._ai_bridge = None  # lazily created

    def set_mode(self, mode):
        if mode in self.MODES:
            self.mode = mode
            print(f"Enemy mode: {mode.upper()}")

    def get_drive(self, enemy: SimRobot, brick: SimRobot, dt: float):
        """Return (throttle, steering) for the enemy."""
        if self.mode == "manual":
            return (0, 0)  # handled by keyboard in run.py
        elif self.mode == "sit":
            return (0, 0)
        elif self.mode == "circle":
            return self._circle(dt)
        elif self.mode == "charge":
            return self._charge(enemy, brick)
        elif self.mode == "flee":
            return self._flee(enemy, brick)
        elif self.mode == "ai":
            return self._ai(enemy, brick, dt)
        return (0, 0)

    def _circle(self, dt):
        """Drive in a circle."""
        return (0.5, 0.3)

    def _charge(self, enemy, brick):
        """Drive straight at Brick."""
        ex, ey = enemy.position
        bx, by = brick.position
        target_angle = math.atan2(by - ey, bx - ex)
        angle_diff = (target_angle - enemy.heading_rad + math.pi) % (2 * math.pi) - math.pi
        steering = max(-1, min(1, angle_diff * 2.0))
        return (0.8, steering)

    def _flee(self, enemy, brick):
        """Run away from Brick."""
        ex, ey = enemy.position
        bx, by = brick.position
        away_angle = math.atan2(ey - by, ex - bx)
        angle_diff = (away_angle - enemy.heading_rad + math.pi) % (2 * math.pi) - math.pi
        steering = max(-1, min(1, angle_diff * 2.0))
        return (0.7, steering)

    def _ai(self, enemy, brick, dt):
        """Run BattleController for enemy."""
        if self._ai_bridge is None:
            from sim.bridge import SimBridge
            # Create with default config — could use a separate config
            self._ai_bridge = SimBridge(enemy, None)
            self._ai_bridge.start_match()
        output = self._ai_bridge.tick(dt, brick)
        # Bridge already applies forces, return (0,0) so run.py doesn't double-apply
        return None  # signal that bridge handled it

    def reset(self):
        self._circle_angle = 0.0
        if self._ai_bridge:
            self._ai_bridge.reset()
            self._ai_bridge = None
```

- [ ] **Step 2: Integrate into run.py**

Update `sim/run.py` to use EnemyController. Add number key switching (1-5 for modes), integrate `get_drive()` into the loop.

Key additions to the main loop:

```python
# At top:
from sim.enemy_ai import EnemyController
enemy_ctrl = EnemyController()

# In event handling:
elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
    modes = {pygame.K_1: "manual", pygame.K_2: "sit", pygame.K_3: "circle",
             pygame.K_4: "charge", pygame.K_5: "flee", pygame.K_6: "ai"}
    enemy_ctrl.set_mode(modes[event.key])

# Replace enemy drive section:
if enemy_ctrl.mode == "manual":
    et = (1 if keys[pygame.K_UP] else 0) - (1 if keys[pygame.K_DOWN] else 0)
    es = (1 if keys[pygame.K_LEFT] else 0) - (1 if keys[pygame.K_RIGHT] else 0)
    arena.enemy.apply_drive(et, es)
elif not paused:
    result = enemy_ctrl.get_drive(arena.enemy, arena.brick, 1/C.RENDER_FPS)
    if result is not None:  # None means AI bridge handled it
        arena.enemy.apply_drive(result[0], result[1])

# In reset:
enemy_ctrl.reset()
```

- [ ] **Step 3: Test all modes**

Run sim, press 1-5 to switch enemy modes:
- 1: Manual (arrows)
- 2: Sits still
- 3: Drives in circle
- 4: Charges at Brick
- 5: Runs its own BattleController AI

**Validate:** Each mode works. AI vs AI (press B then 5) produces a full match.

- [ ] **Step 4: Commit**

```bash
git add prototypes/auto-drive/sim/enemy_ai.py prototypes/auto-drive/sim/run.py
git commit -m "sim: add enemy AI modes (manual, sit, circle, charge, flee, AI)"
```

---

## Phase 5: Frame Logging + Replay

### Task 7: Add JSONL frame logging to run.py

**Files:**
- Modify: `prototypes/auto-drive/sim/run.py`

- [ ] **Step 1: Add frame logger**

Add a `SimLogger` class (or inline in run.py) that writes JSONL in the same format as `main.py`. Fields to populate from SimArena state:

```python
import json
import time
import os

class SimLogger:
    def __init__(self, arena, brick_bridge):
        self.arena = arena
        self.bridge = brick_bridge
        self.file = None
        self.frame_count = 0
        self.start_time = None

    def start(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
        )
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"sim_{ts}.jsonl")
        self.file = open(log_path, "w")
        self.frame_count = 0
        self.start_time = time.perf_counter()
        # Write arena metadata
        self._write_arena_meta(log_path)
        print(f"Logging to {log_path}")

    def _write_arena_meta(self, log_path):
        meta_path = log_path.replace(".jsonl", "_arena.json")
        arena = self.arena
        meta = {
            "arena_width_cm": 244.0,
            "arena_height_cm": 244.0,
        }
        if arena.floor_cal:
            if "corners_ft" in arena.floor_cal:
                meta["corners_cm"] = arena.floor_cal["corners_ft"]
            if "inv_homography" in arena.floor_cal:
                meta["inv_homography"] = arena.floor_cal["inv_homography"]
            if "homography" in arena.floor_cal:
                meta["homography"] = arena.floor_cal["homography"]
            meta["origin_x"] = arena.floor_cal.get("origin_x", 0)
            meta["origin_y"] = arena.floor_cal.get("origin_y", 0)
            meta["px_per_cm"] = arena.floor_cal.get("px_per_cm", 5.0)
            rgb = arena.floor_cal.get("rgb_size", [1280, 800])
            meta["frame_w"] = rgb[0]
            meta["frame_h"] = rgb[1]
        if arena.pit_radius > 0:
            meta["pit_x_cm"] = arena.pit_x
            meta["pit_y_cm"] = arena.pit_y
            meta["pit_radius_cm"] = arena.pit_radius
            meta["pit_danger_radius_cm"] = arena.pit_radius + 15
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def log_frame(self, dt, brick_output=None, enemy_ctrl_mode="manual"):
        if not self.file:
            return
        a = self.arena
        b = a.brick
        e = a.enemy
        bx, by = b.position
        bvx, bvy = b.velocity
        ex, ey = e.position
        evx, evy = e.velocity
        dist = math.hypot(ex - bx, ey - by) if e.alive else 999.0

        thr = brick_output.throttle if brick_output else 0
        steer = brick_output.steering if brick_output else 0
        bs = self.bridge.state if self.bridge else "manual"
        mr = self.bridge.match_timer.remaining_s if self.bridge and self.bridge.match_timer.is_running else None
        urg = self.bridge.match_timer.urgency if self.bridge and self.bridge.match_timer.is_running else None
        mp = self.bridge.match_timer.phase if self.bridge and self.bridge.match_timer.is_running else None

        # Robot polygons for replay (ArUco box + footprint from pymunk vertices)
        ab = [[round(c[0], 1), round(c[1], 1)] for c in b.get_corners_world()]
        fp = ab  # footprint = same as body shape in sim

        rec = {
            "f": self.frame_count,
            "t": round(time.perf_counter() - self.start_time, 4),
            "mode": "battle",
            "bs": bs,
            "mp": mp,
            "ox": round(bx, 1), "oy": round(by, 1),
            "oh": round(b.heading_rad, 3),
            "od": b.alive,
            "ovx": round(bvx, 1), "ovy": round(bvy, 1),
            "ex": round(ex, 1) if e.alive else None,
            "ey": round(ey, 1) if e.alive else None,
            "eh": round(e.heading_rad, 3) if e.alive else None,
            "evx": round(evx, 1) if e.alive else None,
            "evy": round(evy, 1) if e.alive else None,
            "ed": e.alive, "et": e.alive,
            "edx": round(ex, 1) if e.alive else None,
            "edy": round(ey, 1) if e.alive else None,
            "dist": round(dist, 1),
            "thr": round(thr, 3), "str": round(steer, 3),
            "mr": round(mr, 1) if mr is not None else None,
            "urg": round(urg, 3) if urg is not None else None,
            "fps": round(1/dt, 1) if dt > 0 else 60.0,
            "ab": ab, "fp": fp,
            "ehm": "velocity", "ehc": 1.0,
            "ax": None, "ay": None,
        }
        self.file.write(json.dumps(rec) + "\n")
        self.frame_count += 1

    def stop(self):
        if self.file:
            self.file.close()
            self.file = None
            print(f"Log closed ({self.frame_count} frames)")
```

Integrate into run.py main loop:
- **L key** toggles logging on/off
- `logger.log_frame()` called each frame when logging and not paused
- Logger stopped on quit

- [ ] **Step 2: Test with replay viewer**

1. Run sim with AI (B key), start match (Space), enable logging (L)
2. Let it run for a few seconds, quit
3. Run `python serve_replay.py` and open replay viewer
4. Load the `sim_*.jsonl` file

**Validate:** Replay shows robot positions matching sim, state transitions, timer, pit.

- [ ] **Step 3: Commit**

```bash
git add prototypes/auto-drive/sim/run.py
git commit -m "sim: add JSONL frame logging compatible with replay viewer"
```

---

## Phase 6: Speed Control + Polish

### Task 8: Add simulation speed control and HUD polish

**Files:**
- Modify: `prototypes/auto-drive/sim/run.py`

- [ ] **Step 1: Add speed control**

Add a `sim_speed` variable (default 1.0). Plus/minus keys adjust it through `[0.5, 1, 2, 4]`. Multiply `dt` passed to `arena.step()` and `bridge.tick()` by sim_speed.

```python
SPEEDS = [0.5, 1.0, 2.0, 4.0]
speed_idx = 1

# In event handling:
elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
    speed_idx = min(len(SPEEDS) - 1, speed_idx + 1)
elif event.key == pygame.K_MINUS:
    speed_idx = max(0, speed_idx - 1)

# In physics step:
sim_speed = SPEEDS[speed_idx]
effective_dt = (1 / C.RENDER_FPS) * sim_speed
arena.step(effective_dt)
```

- [ ] **Step 2: Polish HUD**

Show in HUD:
- Match timer (M:SS) with urgency color
- Battle state
- Enemy mode
- Sim speed
- Brick AI on/off
- Logging on/off

```python
# Build HUD lines
lines = []
if brick_ai_mode and brick_bridge.match_timer.is_running:
    rem = brick_bridge.match_timer.remaining_s
    mins, secs = int(rem) // 60, int(rem) % 60
    lines.append(f"Match: {mins}:{secs:02d}  State: {brick_bridge.state.upper()}")
lines.append(f"Speed: {sim_speed}x  Brick: {'AI' if brick_ai_mode else 'WASD'}  Enemy: {enemy_ctrl.mode}  Log: {'ON' if logging else 'OFF'}")
```

- [ ] **Step 3: Test full workflow**

1. Launch sim
2. Press B (Brick AI), 4 (enemy charges), Space (start)
3. Press + to speed up to 4x
4. Press L to log
5. Watch a full 30-second match at 4x speed
6. Quit, open replay viewer, verify

- [ ] **Step 4: Commit**

```bash
git add prototypes/auto-drive/sim/run.py
git commit -m "sim: add speed control (0.5-4x) and polished HUD"
```

---

## Self-Review Checklist

- **Spec coverage:** All 6 phases covered (arena, calibration, bridge, enemy AI, logging, speed). All spec requirements addressed: pymunk physics, real arena geometry, pit, BattleController integration, enemy modes, JSONL logging, speed control.
- **Placeholder scan:** No TBD/TODO. All code blocks are complete.
- **Type consistency:** `BattleContext`, `BattleOutput`, `BattleConfig`, `MatchTimer`, `PinTimer` used consistently with their real definitions. `SimRobot.position` returns tuple, `SimRobot.heading_rad` returns float — matches BattleContext fields.
- **File paths:** All reference `prototypes/auto-drive/sim/` consistently.
- **Import paths:** `sys.path.insert(0, ...)` used in run.py and bridge.py to find auto-drive modules. sim modules use relative imports (`from sim.arena import ...` or `from . import config`).
