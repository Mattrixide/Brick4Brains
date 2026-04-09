"""Calibrate simulator physics from real robot log data.

Usage:
    python -m sim.calibrate [--save] [logfile.jsonl ...]
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys

from .config import SimConfig

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

# Speed above this (cm/s) is almost certainly ArUco jitter, not real motion.
# A 3lb beetleweight tops out around 200 cm/s on plywood.
MAX_PLAUSIBLE_SPEED = 300.0
MAX_PLAUSIBLE_ANG_ACCEL = 5000.0  # deg/s^2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_recent_logs(n: int = 5) -> list[str]:
    """Return the n most recent frames_*.jsonl files by modification time."""
    pattern = os.path.join(LOGS_DIR, "frames_*.jsonl")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[:n]


def load_frames(path: str) -> list[dict]:
    """Load JSONL, normalize timestamps, return list of frame dicts."""
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
    if not frames:
        return frames
    # Normalize timestamps if absolute (> 1000)
    t0 = frames[0]["t"]
    if t0 > 1000:
        for fr in frames:
            fr["t"] -= t0
    return frames


def filter_battle_detected(frames: list[dict]) -> list[dict]:
    """Keep only battle-mode frames with ArUco detection."""
    return [f for f in frames if f.get("mode") == "battle" and f.get("od") is True]


# ---------------------------------------------------------------------------
# Raw velocity from position diffs
# ---------------------------------------------------------------------------

def compute_raw_velocities(frames: list[dict]) -> list[dict]:
    """Compute raw velocity from consecutive position diffs.

    Returns list of dicts with keys: t, vx, vy, speed, oh, thr, str, omega, dt.
    Each entry corresponds to the midpoint between frame[i] and frame[i+1].
    Both frames must have od=True. Outlier speeds are filtered.
    """
    results = []
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        if not a.get("od") or not b.get("od"):
            continue
        dt = b["t"] - a["t"]
        if dt < 0.005 or dt > 0.2:  # skip bad gaps (< 5ms or > 200ms)
            continue
        vx = (b["ox"] - a["ox"]) / dt
        vy = (b["oy"] - a["oy"]) / dt
        speed = math.hypot(vx, vy)
        if speed > MAX_PLAUSIBLE_SPEED:
            continue  # ArUco position jump — discard
        results.append({
            "t": (a["t"] + b["t"]) / 2,
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "oh": a["oh"],
            "thr": a["thr"],
            "str": a.get("str", 0),
            "omega": a.get("omega", 0),
            "dt": dt,
        })
    return results


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _find_runs(values: list, predicate, min_len: int = 3):
    """Yield (start, end) index pairs for consecutive runs where predicate is True."""
    run_start = None
    for i, v in enumerate(values):
        if predicate(v):
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= min_len:
                yield run_start, i
            run_start = None
    if run_start is not None and (len(values) - run_start) >= min_len:
        yield run_start, len(values)


def analyze_acceleration(frames_per_file: list[list[dict]]) -> dict:
    """Find segments with sustained throttle and measure forward acceleration."""
    accels = []
    for frames in frames_per_file:
        vels = compute_raw_velocities(frames)
        if len(vels) < 4:
            continue
        for start, end in _find_runs(vels, lambda v: abs(v["thr"]) > 0.3, 3):
            for j in range(start, end - 1):
                a, b = vels[j], vels[j + 1]
                dt = b["t"] - a["t"]
                if dt <= 0 or dt > 0.2:
                    continue
                heading = a["oh"]
                hx, hy = math.cos(heading), math.sin(heading)
                fwd_a = a["vx"] * hx + a["vy"] * hy
                fwd_b = b["vx"] * hx + b["vy"] * hy
                accel = (fwd_b - fwd_a) / dt
                # Only count accel in the direction of throttle
                if a["thr"] > 0 and accel > 0:
                    accels.append(accel)
                elif a["thr"] < 0 and accel < 0:
                    accels.append(-accel)

    if not accels:
        return {"avg_accel_cm_s2": None, "median_accel_cm_s2": None, "samples": 0}

    return {
        "avg_accel_cm_s2": statistics.mean(accels),
        "median_accel_cm_s2": statistics.median(accels),
        "samples": len(accels),
    }


def analyze_friction(frames_per_file: list[list[dict]]) -> dict:
    """Measure speed decay when throttle is near zero."""
    decay_ratios = []
    decay_speeds = []   # speed at start of each decay pair, for mu estimation
    decay_dts = []      # dt for each decay pair
    for frames in frames_per_file:
        vels = compute_raw_velocities(frames)
        if len(vels) < 4:
            continue
        for start, end in _find_runs(vels, lambda v: abs(v["thr"]) < 0.05, 3):
            for j in range(start, end - 1):
                a, b = vels[j], vels[j + 1]
                if a["speed"] > 5.0 and b["speed"] > 0:
                    ratio = b["speed"] / a["speed"]
                    if ratio < 2.0:  # filter obvious jitter spikes
                        decay_ratios.append(ratio)
                        decay_speeds.append(a["speed"])
                        decay_dts.append(b["t"] - a["t"])

    if not decay_ratios:
        return {"avg_decay_ratio": None, "median_decay_ratio": None,
                "avg_decay_speed": None, "avg_decay_dt": None, "samples": 0}

    return {
        "avg_decay_ratio": statistics.mean(decay_ratios),
        "median_decay_ratio": statistics.median(decay_ratios),
        "avg_decay_speed": statistics.mean(decay_speeds),
        "avg_decay_dt": statistics.mean(decay_dts),
        "samples": len(decay_ratios),
    }


def analyze_turn_rate(frames_per_file: list[list[dict]]) -> dict:
    """Measure angular acceleration and steady-state turn rate during steering."""
    ang_accels = []      # d(omega)/dt during steering onset
    steady_omegas = []   # abs(omega) at steady state with steering

    for frames in frames_per_file:
        for i in range(len(frames) - 1):
            a, b = frames[i], frames[i + 1]
            if not a.get("od") or not b.get("od"):
                continue
            if "omega" not in a or "omega" not in b:
                continue
            if abs(a.get("str", 0)) < 0.3:
                continue
            dt = b["t"] - a["t"]
            if dt < 0.005 or dt > 0.2:
                continue

            # Steady-state: collect omega magnitude during steering
            if abs(a["omega"]) > 5:  # ignore near-zero
                steady_omegas.append(abs(a["omega"]))

            # Angular acceleration: look for onset (omega changing significantly)
            d_omega = b["omega"] - a["omega"]
            ang_accel = abs(d_omega / dt)
            if 10 < ang_accel < MAX_PLAUSIBLE_ANG_ACCEL:  # filter noise floor
                ang_accels.append(ang_accel)

    result = {"samples": len(ang_accels)}

    if ang_accels:
        result["avg_ang_accel_deg_s2"] = statistics.mean(ang_accels)
        result["median_ang_accel_deg_s2"] = statistics.median(ang_accels)
    else:
        result["avg_ang_accel_deg_s2"] = None
        result["median_ang_accel_deg_s2"] = None

    if steady_omegas:
        result["avg_steady_omega_deg_s"] = statistics.mean(steady_omegas)
        result["median_steady_omega_deg_s"] = statistics.median(steady_omegas)
    else:
        result["avg_steady_omega_deg_s"] = None
        result["median_steady_omega_deg_s"] = None

    return result


def analyze_top_speed(frames_per_file: list[list[dict]]) -> dict:
    """Find maximum observed speed from raw position diffs."""
    speeds = []
    for frames in frames_per_file:
        vels = compute_raw_velocities(frames)
        speeds.extend(v["speed"] for v in vels)

    if not speeds:
        return {"max_speed_cm_s": None, "p95_speed_cm_s": None, "median_speed_cm_s": None}

    speeds_sorted = sorted(speeds)
    p95_idx = int(len(speeds_sorted) * 0.95)
    return {
        "max_speed_cm_s": speeds_sorted[-1],
        "p95_speed_cm_s": speeds_sorted[min(p95_idx, len(speeds_sorted) - 1)],
        "median_speed_cm_s": speeds_sorted[len(speeds_sorted) // 2],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_calibration(log_files: list[str], save: bool = False):
    """Run calibration analysis on the given log files."""
    frames_per_file: list[list[dict]] = []
    total_battle = 0
    for path in log_files:
        raw = load_frames(path)
        battle = filter_battle_detected(raw)
        print(f"  {os.path.basename(path)}: {len(raw)} total, {len(battle)} battle+detected")
        if battle:
            frames_per_file.append(battle)
            total_battle += len(battle)

    if not frames_per_file:
        print("\nNo battle frames found. Cannot calibrate.")
        return

    print(f"\nTotal battle frames: {total_battle} across {len(frames_per_file)} file(s)")

    # Run analyses — each processes files independently to avoid cross-file dt issues
    accel = analyze_acceleration(frames_per_file)
    friction = analyze_friction(frames_per_file)
    turn = analyze_turn_rate(frames_per_file)
    top_speed = analyze_top_speed(frames_per_file)

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)

    print("\nAcceleration:")
    if accel["avg_accel_cm_s2"] is not None:
        print(f"  Average forward accel: {accel['avg_accel_cm_s2']:.1f} cm/s^2")
        print(f"  Median forward accel:  {accel['median_accel_cm_s2']:.1f} cm/s^2")
    else:
        print("  No data")
    print(f"  Samples: {accel['samples']}")

    print("\nFriction (speed decay at zero throttle):")
    if friction["avg_decay_ratio"] is not None:
        print(f"  Average decay ratio: {friction['avg_decay_ratio']:.4f}")
        print(f"  Median decay ratio:  {friction['median_decay_ratio']:.4f}")
    else:
        print("  No data")
    print(f"  Samples: {friction['samples']}")

    print("\nTurn rate:")
    if turn["avg_ang_accel_deg_s2"] is not None:
        print(f"  Average angular accel: {turn['avg_ang_accel_deg_s2']:.1f} deg/s^2")
        print(f"  Median angular accel:  {turn['median_ang_accel_deg_s2']:.1f} deg/s^2")
    else:
        print("  Angular accel: no data")
    if turn.get("avg_steady_omega_deg_s") is not None:
        print(f"  Average steady omega:  {turn['avg_steady_omega_deg_s']:.1f} deg/s")
        print(f"  Median steady omega:   {turn['median_steady_omega_deg_s']:.1f} deg/s")
    else:
        print("  Steady-state omega: no data")
    print(f"  Samples: {turn['samples']}")

    print("\nTop speed:")
    if top_speed["max_speed_cm_s"] is not None:
        print(f"  Max: {top_speed['max_speed_cm_s']:.1f} cm/s")
        print(f"  95th percentile: {top_speed['p95_speed_cm_s']:.1f} cm/s")
        print(f"  Median: {top_speed['median_speed_cm_s']:.1f} cm/s")
    else:
        print("  No data")

    # Compute suggested config values
    cfg = SimConfig()
    mass = cfg.brick_mass_kg  # 1.36 kg

    print("\n" + "=" * 60)
    print("SUGGESTED CONFIG VALUES")
    print("=" * 60)

    # Use median values — more robust to outliers than mean
    use_accel = accel.get("median_accel_cm_s2")
    use_decay = friction.get("median_decay_ratio")
    if use_accel is not None:
        # pymunk force = mass * accel (all in consistent units: kg, cm, s)
        suggested_force = use_accel * mass
        print(f"  max_forward_force: {suggested_force:.0f}  (current: {cfg.max_forward_force})")
    else:
        suggested_force = None
        print(f"  max_forward_force: no data  (keeping {cfg.max_forward_force})")

    if use_decay is not None:
        # Model: v(t+dt) = v(t) * decay_ratio
        # In the sim, friction force = mu * m * g opposes motion:
        #   v_next = v - mu*g*dt  =>  decay_ratio = 1 - mu*g*dt/v
        # Rearranging: mu = (1 - decay_ratio) * v / (g * dt)
        # Use the actual speed and dt from friction measurements
        v_typ = friction.get("avg_decay_speed", 30.0)
        avg_dt = friction.get("avg_decay_dt", 1.0 / 40.0)
        if v_typ and v_typ > 1 and avg_dt > 0:
            suggested_mu = (1.0 - use_decay) * v_typ / (cfg.gravity_cms2 * avg_dt)
        else:
            suggested_mu = 0.6
        suggested_mu = max(0.1, min(2.0, suggested_mu))
        print(f"  ground_friction_mu: {suggested_mu:.3f}  (current: {cfg.ground_friction_mu})")
    else:
        suggested_mu = None
        print(f"  ground_friction_mu: no data  (keeping {cfg.ground_friction_mu})")

    # For torque: prefer median angular accel; fall back to mean
    use_ang_accel = turn.get("median_ang_accel_deg_s2") or turn.get("avg_ang_accel_deg_s2")
    if use_ang_accel is not None and use_ang_accel > 0:
        # torque = I * alpha
        # I for rectangle = mass * (w^2 + d^2) / 12 (in cm units)
        I_cm = mass * (cfg.brick_width_cm**2 + cfg.brick_depth_cm**2) / 12
        alpha_rad = math.radians(use_ang_accel)
        suggested_torque = I_cm * alpha_rad
        print(f"  max_torque: {suggested_torque:.0f}  (current: {cfg.max_torque})")
    else:
        suggested_torque = None
        print(f"  max_torque: no data  (keeping {cfg.max_torque})")

    if save:
        if suggested_force is not None:
            cfg.max_forward_force = suggested_force
        if suggested_mu is not None:
            cfg.ground_friction_mu = suggested_mu
        if suggested_torque is not None:
            cfg.max_torque = suggested_torque
        cfg.save()
        out_path = os.path.join(os.path.dirname(__file__), "sim_config.json")
        print(f"\nSaved to {out_path}")
    else:
        print("\n(pass --save to write these values to sim_config.json)")


def main():
    parser = argparse.ArgumentParser(description="Calibrate sim physics from real robot logs")
    parser.add_argument("logfiles", nargs="*", help="Specific JSONL log files to use")
    parser.add_argument("--save", action="store_true", help="Save results to sim_config.json")
    args = parser.parse_args()

    print("=" * 60)
    print("B4B Combat Simulator -- Physics Calibration")
    print("=" * 60)

    if args.logfiles:
        log_files = args.logfiles
    else:
        log_files = find_recent_logs(5)

    if not log_files:
        print("No log files found!")
        sys.exit(1)

    print(f"\nUsing {len(log_files)} log file(s):")
    run_calibration(log_files, save=args.save)


if __name__ == "__main__":
    main()
