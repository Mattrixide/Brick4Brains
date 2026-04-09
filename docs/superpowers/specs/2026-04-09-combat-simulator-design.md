# Combat Robot Simulator — Design Spec

**Date:** 2026-04-09
**Status:** Approved
**Location:** `prototypes/auto-drive/sim/`

## Purpose

A pymunk-based 2D physics simulator that lets Brick's battle strategy code run against an enemy robot without needing the real camera, CV pipeline, or hardware. Enables rapid iteration on strategy, tuning, and testing.

## Architecture

Three layers, loosely coupled:

```
SimArena (pymunk physics)  →  SimBridge (packs BattleContext)  →  BattleController.tick()
     ↑                                                                    ↓
     └──────────────── applies throttle/steering as forces ←──────── BattleOutput
```

### SimArena (`arena.py`)

Owns the pymunk space, steps physics, renders via pygame.

**World setup:**
- Arena walls loaded from `floor_calibration.json` (`corners_ft`) as pymunk static segments — the real calibrated polygon, not an assumed rectangle
- Pit loaded from `battle_config.json` (`pit_x_cm`, `pit_y_cm`, `pit_radius_cm`) as a pymunk sensor shape — robots entering the pit are eliminated
- Two rectangular robot bodies:
  - Brick (our robot): 9" wide x 7" front-to-back (22.86cm x 17.78cm)
  - Enemy: 6" wide x 9" front-to-back (15.24cm x 22.86cm)

**Physics model:**
- Top-down 2D, zero gravity
- Ground friction: Coulomb model (`F = mu * m * g`) with enhanced lateral damping (robots resist sideways sliding)
- Collision: pymunk rigid body impulse resolution with tunable elasticity and surface friction
- Motor model: force applied along heading direction proportional to throttle, torque proportional to steering

**Physics parameters** derived from real log data (see Calibration section below):
- Max forward force, max torque
- Ground friction coefficient (mu)
- Lateral damping ratio
- Angular damping
- Wall/robot elasticity
- Robot masses

**Rendering:**
- Pygame top-down view
- Green grid with arena outline (matching replay viewer style)
- Red pit square
- Robot rectangles with heading indicators
- State labels, distance lines, match timer, pin countdown

### SimBridge (`bridge.py`)

Connects a pymunk robot body to a `BattleController`. Each frame:

1. Read position, heading, velocity from the pymunk body
2. Read enemy state from SimArena
3. Compute distance, relative angles
4. Pack into `BattleContext` (the same dataclass used by real code)
5. Call `BattleController.tick(ctx)` → receive `BattleOutput`
6. Convert `BattleOutput.throttle`/`steering` to pymunk forces on the body
7. Handle `target_omega_dps` (rate mode) by applying torque to match target angular velocity

Reuses the existing `BattleController`, `BattleConfig`, `MatchTimer`, `PinTimer` classes directly — no copies or modifications.

### Enemy AI (`enemy_ai.py`)

Switchable enemy control modes:

1. **Manual** — keyboard (arrow keys) or gamepad control
2. **Sit** — stationary target, useful for testing approach/charge
3. **Circle** — drives in a circle at configurable speed/radius
4. **Charge** — drives straight at Brick
5. **Flee** — runs away from Brick
6. **BattleController AI** — runs a second `BattleController` instance with its own config, so two strategies fight each other

Each mode implements a simple interface:
```python
def get_drive(self, arena_state) -> (throttle: float, steering: float)
```

### Main Entry Point (`run.py`)

Pygame main loop:

- Initializes SimArena, SimBridge, enemy AI
- Handles keyboard input and mode switching
- Steps physics and controllers each frame
- Renders and logs

**Controls:**
- **Space** — start/pause match
- **R** — reset robots to starting positions
- **1-5** — switch enemy mode (manual, sit, circle, charge, AI)
- **Arrow keys** — manual enemy control (when in manual mode)
- **+/-** — simulation speed (0.5x, 1x, 2x, 4x)
- **ESC** — quit
- **L** — toggle frame logging on/off

### Config (`config.py`)

All tunable physics parameters in one place:

```python
# Robot dimensions
BRICK_WIDTH_CM = 22.86      # 9 inches
BRICK_DEPTH_CM = 17.78      # 7 inches
ENEMY_WIDTH_CM = 15.24      # 6 inches
ENEMY_DEPTH_CM = 22.86      # 9 inches

# Physics (populated by calibration script)
BRICK_MASS_KG = 1.36        # 3lb beetleweight
ENEMY_MASS_KG = 1.36
GROUND_FRICTION_MU = 0.6    # rubber on plywood
LATERAL_DAMPING = 0.85
ANGULAR_DAMPING = 0.93
MAX_FORWARD_FORCE = 2000.0
MAX_TORQUE = 4000.0
WALL_ELASTICITY = 0.2
ROBOT_ELASTICITY = 0.3
```

## Physics Calibration from Log Data

A one-time analysis script (`sim/calibrate.py`) that reads real match logs and extracts physics parameters:

**Inputs:** JSONL frame logs from `prototypes/auto-drive/logs/`

**What it extracts:**

1. **Acceleration response** — Find frames where throttle jumps from 0 to 1.0, measure velocity change over time → max forward force
2. **Friction/deceleration** — Find frames where throttle drops to 0, measure velocity decay curve → ground friction coefficient
3. **Turn rate response** — Find frames with constant steering input, measure angular velocity → max torque and angular damping
4. **Top speed** — Maximum observed velocity under sustained throttle → validates force/friction balance
5. **Lateral behavior** — Measure sideways velocity component during turns → lateral damping ratio

**Output:** Prints fitted constants and optionally writes them to `sim/config.py`.

## Frame Logging

The simulator writes JSONL frame logs in the same format as real runs:

```json
{
  "f": 0, "t": 0.0,
  "mode": "battle", "bs": "charge_pursue",
  "mp": "start", "mr": 30.0, "urg": 0.5,
  "ox": 10.5, "oy": -20.3, "oh": 1.57, "od": true,
  "ovx": 15.0, "ovy": 2.1,
  "ex": -40.0, "ey": 30.0, "eh": 3.14,
  "evx": -10.0, "evy": 5.0,
  "ed": true, "et": true, "dist": 65.3,
  "thr": 0.75, "str": 0.15,
  "fps": 60.0
}
```

- `od` is always `true` in simulation (no ArUco dropout)
- `ed`/`et` are always `true` when enemy is alive (no detection failures)
- File written to `logs/sim_*.jsonl` with companion `_arena.json` (includes pit + corners)
- Replay viewer loads these automatically — same arena overlay, grid, pit, robot visualization

## File Structure

```
prototypes/auto-drive/sim/
  __init__.py
  run.py           # Main entry point, pygame loop, controls
  arena.py         # SimArena: pymunk world, walls, pit, rendering
  bridge.py        # SimBridge: BattleContext ↔ pymunk forces
  enemy_ai.py      # Scripted enemy behaviors + AI mode
  config.py        # Tunable physics parameters
  calibrate.py     # Extract physics params from real logs
```

## What This Does NOT Change

- No modifications to `state_machine.py`, `battle_config.py`, `match_timer.py`, or any existing battle code
- The simulator imports and uses these as-is
- `BattleContext` and `BattleOutput` are the only interface — if the strategy code works in sim, it works on the real robot (minus CV noise)

## Implementation Phases

### Phase 1: Arena + Manual Control
Build SimArena with pymunk, load real arena walls from `floor_calibration.json`, pit from `battle_config.json`. Two rectangular robots, both keyboard-controlled. Pygame rendering with grid/pit.

**Validate:** Drive both robots around, push each other, fall in pit. Physics feel reasonable.

### Phase 2: Physics Calibration
Write `calibrate.py` to extract friction, acceleration, turn rate from real logs. Populate `config.py` with fitted values.

**Validate:** Sim robot acceleration/top speed/turn rate matches real robot behavior from logs.

### Phase 3: SimBridge + Brick AI
Build SimBridge that packs pymunk state into `BattleContext` and feeds `BattleController.tick()`. Brick drives itself.

**Validate:** Brick acquires and charges a stationary enemy. State machine transitions look correct.

### Phase 4: Enemy AI Modes
Add scripted enemy behaviors (sit, circle, charge, flee) and second BattleController mode.

**Validate:** Run full matches against each enemy type. Strategy responds correctly to different opponents.

### Phase 5: Frame Logging + Replay
Write JSONL frame logs matching real format. Include arena metadata with pit + corners.

**Validate:** Open sim logs in replay viewer. Grid, pit, robot positions, state transitions all render correctly.

### Phase 6: Speed Control + Polish
Add simulation speed control (0.5x-4x), match timer integration, HUD polish.

**Validate:** Run a 30-second match at 4x speed, replay looks correct.

## Success Criteria

1. Two rectangular robots moving in a 244cm arena with real calibrated walls and pit
2. Brick runs `BattleController` strategy code unmodified
3. Enemy switchable between manual, scripted, and AI modes
4. Physics feel similar to real robot behavior (validated by calibration from logs)
5. Frame logs load in replay viewer and look correct
6. Can run matches at 1x-4x speed for rapid iteration
