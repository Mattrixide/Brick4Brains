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

Owns the pymunk space, steps physics. **Does not render** — rendering is in `renderer.py` (separated for headless testing).

**World setup:**
- Arena walls loaded from `floor_calibration.json` (`corners_ft` — named "ft" but values are in cm) as pymunk static segments — the real calibrated polygon, not an assumed rectangle
- Pit loaded from `battle_config.json` (`pit_x_cm`, `pit_y_cm`, `pit_radius_cm`) as a pymunk sensor shape — robots entering the pit are eliminated
- Two rectangular robot bodies:
  - Brick (our robot): 9" wide x 7" front-to-back (22.86cm x 17.78cm)
  - Enemy: 6" wide x 9" front-to-back (15.24cm x 22.86cm)

**Physics model:**
- Top-down 2D, zero gravity
- Ground friction: Coulomb model (`F = mu * m * g`) applied along forward axis only. Separate lateral damping for sideways sliding resistance. Forces applied *before* substeps; velocity/angular damping applied *after* all substeps complete (not before, to avoid solver overwriting damped values).
- Collision: pymunk rigid body impulse resolution with tunable elasticity and surface friction
- Motor model: force applied along heading direction proportional to throttle, torque proportional to steering. Motor response lag (~30ms first-order filter) to approximate real ESC ramp time.
- Rate mode: P-controller on angular velocity with **clamped output** to `[-MAX_TORQUE, MAX_TORQUE]`

**Physics parameters** derived from real log data (see Calibration section below):
- Max forward force, max torque
- Ground friction coefficient (mu)
- Lateral damping ratio
- Angular damping
- Wall/robot elasticity
- Robot masses
- Motor lag time constant

### SimRenderer (`renderer.py`)

Pygame top-down rendering, separated from physics for headless testing:
- Green grid with arena outline (matching replay viewer style)
- Red pit square
- Robot rectangles with heading indicators
- State labels, distance lines, match timer, pin countdown

### SimBridge (`bridge.py`)

Connects a pymunk robot body to a `BattleController`. Does **not** depend on `SimArena` — only takes two `SimRobot` references (ours + enemy). Each frame:

1. Read position, heading, velocity from the pymunk body
2. Read enemy state from enemy SimRobot
3. Compute distance, relative angles
4. Compute `accel_x_mg`/`accel_y_mg` from velocity delta between frames (enables impact detection)
5. Pack into `BattleContext` with **all fields** (the same dataclass used by real code)
6. Call `BattleController.tick(ctx)` → receive `BattleOutput`
7. Convert `BattleOutput.throttle`/`steering` to pymunk forces on the body
8. Handle `target_omega_dps` (rate mode) by applying torque P-controller, **clamped** to `[-MAX_TORQUE, MAX_TORQUE]`

`MatchTimer` constructed with all 4 args from BattleConfig: `duration_s`, `urgency_ramp_start_s`, `phase_start_s`, `phase_final_s`.

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

### Main Entry Point (`run.py`) + Session (`session.py`)

`run.py` is a thin pygame event loop. Match state, speed control, logging, and bridge orchestration live in `SimSession` (in `session.py`) to keep run.py small.

**Controls:**
- **Space** — start/pause match
- **R** — reset robots to starting positions
- **1-6** — switch enemy mode (manual, sit, circle, charge, flee, AI)
- **Arrow keys** — manual enemy control (when in manual mode)
- **+/-** — simulation speed (0.5x, 1x, 2x, 4x)
- **B** — toggle Brick AI on/off
- **L** — toggle frame logging on/off
- **ESC** — quit

### Config (`sim_config.json` + `config.py`)

`config.py` provides default constants. Runtime config is a `SimConfig` dataclass that loads/saves from `sim_config.json` — this lets `calibrate.py` write fitted values and enables A/B comparisons with different physics tunings. Module constants are fallback defaults only.

## Physics Calibration from Log Data

A one-time analysis script (`sim/calibrate.py`) that reads real match logs and extracts physics parameters.

**Inputs:** JSONL frame logs from `prototypes/auto-drive/logs/`

**Important data notes:**
- `ovx`/`ovy` are Kalman-filtered with a 0.70 velocity decay baked in — **do not use for friction analysis**
- Use raw position differences `(ox_next - ox) / dt` instead for velocity estimation
- `omega` is raw gyro data and usable directly
- Skip frames where `od=False` (ArUco lost) — creates position discontinuities

**Approach:** Least-squares fit of the full discrete dynamics model using scipy:
```
x(t+1) = f(x(t), u(t), params)
```
where state = `(ox, oy, oh, vx, vy, omega)`, input = `(thr, str)`, and fitted params = `MAX_FORWARD_FORCE`, `GROUND_FRICTION_MU`, `MAX_TORQUE`, `ANGULAR_DAMPING`, `LATERAL_DAMPING`, `MOTOR_LAG_S`.

**What it fits:**

1. **Longitudinal acceleration** — project velocity change onto heading direction when throttle is nonzero → max forward force
2. **Friction/deceleration** — velocity decay from raw position diffs when throttle=0 → ground friction coefficient
3. **Angular acceleration** — measure `d(omega)/dt` during steering changes → max torque (not steady-state omega)
4. **Motor lag** — measure delay between throttle change and velocity response → first-order time constant
5. **Lateral behavior** — sideways velocity component during turns → lateral damping ratio
6. **Top speed** — validates force/friction balance

**Output:** Prints fitted constants and writes them to `sim_config.json`.

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
- `ab` (ArUco box) populated from Brick pymunk body vertices
- `fp` (footprint) populated from Brick pymunk body vertices
- `ehm` set to `"velocity"`, `ehc` set to `1.0` (perfect heading in sim)
- `ax`/`ay` computed from velocity delta (enables impact detection testing)
- File written to `logs/sim_*.jsonl` with companion `_arena.json` (includes pit + `pit_danger_radius_cm` + corners)
- Replay viewer loads these automatically — same arena overlay, grid, pit, robot visualization

## File Structure

```
prototypes/auto-drive/sim/
  __init__.py
  config.py        # SimConfig dataclass + defaults, loads/saves sim_config.json
  arena.py         # SimArena: pymunk world, walls, pit (no rendering)
  renderer.py      # SimRenderer: pygame drawing (separated for headless testing)
  bridge.py        # SimBridge: BattleContext ↔ pymunk forces
  enemy_ai.py      # EnemyController: manual, scripted, AI modes
  session.py       # SimSession: match state, speed, logging orchestration
  run.py           # Main entry point: thin pygame event loop
  calibrate.py     # Extract physics params from real logs via least-squares fit
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
