"""Drive calibration tool — measures motor bias and tunes heading-hold PID.

Connects to ESP32, reads IMU gyro, and runs three calibration phases.
The robot automatically detects wall hits (sudden heading spikes) and
reverses + turns 180 to keep running in the arena.

  Phase 1: BIAS TEST
    Drive straight with zero steering, measure yaw drift per leg.
    Multiple legs (wall bounces) are averaged to get a stable drift rate.
    Result: steering_bias (constant offset to counteract motor pull).

  Phase 2: RELAY AUTO-TUNE
    Drive with bang-bang steering correction based on heading error.
    Robot oscillates around target heading while bouncing off walls.
    Measures oscillation period (Tu) and amplitude to compute Ku.
    Result: PID gains via Ziegler-Nichols "no overshoot" formulas.

  Phase 3: VERIFY
    Drive straight using the tuned heading-hold PID.
    Log max heading deviation across multiple legs. Should stay within +/-3 deg.

Saves calibration to drive_calibration.json for use by main.py.

Usage:
    python calibrate_drive.py --esp32 esp32wifi.local
    python calibrate_drive.py --esp32 192.168.4.122 --throttle 0.5
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field

from comms import RobotComms
from sensor_fusion import IMUPoller


# ---------------------------------------------------------------------------
# Wall detection threshold
# ---------------------------------------------------------------------------
# If heading changes by more than this many degrees between two samples,
# we assume the robot hit a wall (impact deflection or wheels stalling).
WALL_HIT_THRESHOLD_DEG = 8.0

# If heading drifts more than this from the leg's starting heading,
# assume robot is stuck against a wall and spinning.
WALL_DRIFT_THRESHOLD_DEG = 40.0

# Reverse duration and turn duration after wall hit
REVERSE_TIME_S = 0.6
TURN_TIME_S = 0.5


# ---------------------------------------------------------------------------
# IMU reader
# ---------------------------------------------------------------------------

class IMUReader:
    """Reads heading from ESP32 IMU via HTTP polling."""

    def __init__(self, host: str):
        self._poller = IMUPoller(host=host)
        self._baseline_yaw = None

    def start(self):
        self._poller.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self._poller.is_active:
                break
            time.sleep(0.05)
        if not self._poller.is_active:
            raise RuntimeError("No IMU data received within 3 seconds")
        print(f"[imu] Connected, yaw={self._poller.get_yaw():.1f} deg")

    def reset_heading(self):
        """Zero the heading reference."""
        self._poller.reset_yaw()
        time.sleep(0.2)
        self._baseline_yaw = self._poller.get_yaw()

    def get_heading(self) -> float:
        """Get heading relative to baseline, in degrees."""
        yaw = self._poller.get_yaw()
        if self._baseline_yaw is None:
            self._baseline_yaw = yaw
        return yaw - self._baseline_yaw

    @property
    def is_active(self) -> bool:
        return self._poller.is_active

    def stop(self):
        self._poller.stop()


# ---------------------------------------------------------------------------
# Wall bounce helper
# ---------------------------------------------------------------------------

def detect_wall_hit(headings: list[float], threshold_delta: float = WALL_HIT_THRESHOLD_DEG,
                    threshold_drift: float = WALL_DRIFT_THRESHOLD_DEG) -> bool:
    """Check if the latest heading samples indicate a wall hit."""
    if len(headings) < 3:
        return False

    # Sudden heading change between consecutive samples
    delta = abs(headings[-1] - headings[-2])
    if delta > threshold_delta:
        return True

    # Large accumulated drift from leg start (index 0)
    drift = abs(headings[-1] - headings[0])
    if drift > threshold_drift:
        return True

    return False


def wall_bounce(comms: RobotComms, imu: IMUReader, throttle: float):
    """Reverse, turn ~180, prepare for next leg."""
    # Stop
    comms.stop()
    time.sleep(0.1)

    # Reverse
    start = time.monotonic()
    while time.monotonic() - start < REVERSE_TIME_S:
        comms.send(-throttle, 0.0)
        time.sleep(0.02)

    # Turn ~180 (spin in place)
    # Read heading before turn
    heading_before = imu.get_heading()
    turn_start = time.monotonic()
    while time.monotonic() - turn_start < TURN_TIME_S:
        comms.send(0.0, 0.8)  # spin right
        time.sleep(0.02)

    comms.stop()
    heading_after = imu.get_heading()
    turned = heading_after - heading_before
    print(f"  [bounce] Reversed + turned {turned:+.0f} deg")
    time.sleep(0.2)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BiasResult:
    steering_bias: float
    drift_rate_dps: float
    total_drift_deg: float
    samples: int
    legs: int


@dataclass
class PIDResult:
    kp: float
    ki: float
    kd: float
    ku: float
    tu: float
    relay_amplitude: float
    oscillation_amplitude: float


@dataclass
class VerifyResult:
    max_deviation_deg: float
    rms_deviation_deg: float
    mean_deviation_deg: float
    samples: int
    passed: bool
    legs: int


# ---------------------------------------------------------------------------
# Phase 1: Bias test with wall bouncing
# ---------------------------------------------------------------------------

def phase1_bias_test(
    comms: RobotComms,
    imu: IMUReader,
    throttle: float = 0.5,
    duration: float = 10.0,
    dt: float = 0.02,
) -> BiasResult:
    """Drive straight with zero steering, bounce off walls, measure drift."""
    print(f"\n{'='*60}")
    print(f"  PHASE 1: BIAS TEST")
    print(f"  Throttle={throttle:.0%}, Duration={duration:.1f}s total")
    print(f"  Robot drives straight, bounces off walls, measures veer")
    print(f"{'='*60}")
    print("Starting in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)

    imu.reset_heading()
    time.sleep(0.3)

    # Collect data from multiple legs (wall-to-wall drives)
    all_drift_rates = []
    all_headings = []
    leg_headings = []
    leg_start_time = time.monotonic()
    total_start = time.monotonic()
    leg_count = 0

    print("[bias] Driving...")
    while time.monotonic() - total_start < duration:
        comms.send(throttle, 0.0)
        h = imu.get_heading()
        leg_headings.append(h)
        all_headings.append(h)

        # Check for wall hit
        if detect_wall_hit(leg_headings):
            # Compute drift rate for this leg
            leg_elapsed = time.monotonic() - leg_start_time
            if len(leg_headings) > 10 and leg_elapsed > 0.3:
                # Use middle portion (skip accel and wall impact)
                trim = max(3, len(leg_headings) // 5)
                stable = leg_headings[trim:-3] if len(leg_headings) > trim + 3 else leg_headings
                if len(stable) > 3:
                    drift = stable[-1] - stable[0]
                    stable_elapsed = len(stable) * dt
                    rate = drift / stable_elapsed if stable_elapsed > 0.05 else 0.0
                    all_drift_rates.append(rate)
                    leg_count += 1
                    print(f"  [leg {leg_count}] drift={drift:+.1f} deg in {leg_elapsed:.1f}s "
                          f"(rate={rate:+.1f} deg/s)")

            # Bounce off wall
            wall_bounce(comms, imu, throttle)

            # Reset for new leg
            imu.reset_heading()
            time.sleep(0.1)
            leg_headings = []
            leg_start_time = time.monotonic()
            continue

        time.sleep(dt)

    # Process final leg if it has data
    leg_elapsed = time.monotonic() - leg_start_time
    if len(leg_headings) > 10 and leg_elapsed > 0.3:
        trim = max(3, len(leg_headings) // 5)
        stable = leg_headings[trim:] if len(leg_headings) > trim else leg_headings
        if len(stable) > 3:
            drift = stable[-1] - stable[0]
            stable_elapsed = len(stable) * dt
            rate = drift / stable_elapsed if stable_elapsed > 0.05 else 0.0
            all_drift_rates.append(rate)
            leg_count += 1
            print(f"  [leg {leg_count}] drift={drift:+.1f} deg in {leg_elapsed:.1f}s "
                  f"(rate={rate:+.1f} deg/s)")

    comms.stop()
    print("[bias] Stopped.")

    if not all_drift_rates:
        print("[bias] WARNING: No valid legs recorded")
        return BiasResult(0.0, 0.0, 0.0, len(all_headings), 0)

    # Average drift rate across all legs
    avg_drift_rate = sum(all_drift_rates) / len(all_drift_rates)

    # Compute steering bias
    steering_bias = -avg_drift_rate * 0.01
    steering_bias = max(-0.15, min(0.15, steering_bias))

    print(f"\n[bias] Results:")
    print(f"  Legs completed:  {leg_count}")
    print(f"  Drift rates:     [{', '.join(f'{r:+.1f}' for r in all_drift_rates)}] deg/s")
    print(f"  Avg drift rate:  {avg_drift_rate:+.2f} deg/s")
    print(f"  Steering bias:   {steering_bias:+.4f}")
    print(f"  Total samples:   {len(all_headings)}")

    return BiasResult(steering_bias, avg_drift_rate,
                      avg_drift_rate * duration, len(all_headings), leg_count)


# ---------------------------------------------------------------------------
# Phase 2: Relay auto-tune with wall bouncing
# ---------------------------------------------------------------------------

def phase2_relay_autotune(
    comms: RobotComms,
    imu: IMUReader,
    steering_bias: float = 0.0,
    throttle: float = 0.5,
    relay_amplitude: float = 0.20,
    duration: float = 15.0,
    dt: float = 0.02,
) -> PIDResult:
    """Relay (bang-bang) test with wall bouncing to find PID gains."""
    print(f"\n{'='*60}")
    print(f"  PHASE 2: RELAY AUTO-TUNE")
    print(f"  Throttle={throttle:.0%}, Relay=+/-{relay_amplitude:.2f}, Bias={steering_bias:+.4f}")
    print(f"  Duration={duration:.1f}s total")
    print(f"  Robot oscillates side-to-side while bouncing off walls")
    print(f"{'='*60}")
    print("Starting in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)

    imu.reset_heading()
    time.sleep(0.3)

    # Collect heading data across all legs (only during forward driving)
    all_headings = []
    all_timestamps = []
    leg_headings = []
    zero_crossings = []
    prev_heading = 0.0
    leg_start_time = time.monotonic()
    total_start = time.monotonic()
    leg_count = 0

    print("[relay] Driving with relay control...")
    while time.monotonic() - total_start < duration:
        h = imu.get_heading()
        t = time.monotonic() - total_start
        leg_headings.append(h)

        # Bang-bang: steer opposite to heading error
        if h > 0:
            steer = -relay_amplitude + steering_bias
        else:
            steer = relay_amplitude + steering_bias

        comms.send(throttle, steer)
        all_headings.append(h)
        all_timestamps.append(t)

        # Detect zero crossings
        if len(all_headings) > 1 and prev_heading * h < 0:
            zero_crossings.append(t)
        prev_heading = h

        # Check for wall hit
        if detect_wall_hit(leg_headings):
            leg_count += 1
            print(f"  [leg {leg_count}] wall hit at heading {h:+.1f} deg")
            wall_bounce(comms, imu, throttle)

            # DON'T reset heading — relay test needs continuous heading
            # to measure oscillation across legs
            leg_headings = []
            leg_start_time = time.monotonic()
            # Re-read heading after bounce (it's now in a new direction)
            # Reset baseline so relay oscillates around 0
            imu.reset_heading()
            time.sleep(0.1)
            prev_heading = 0.0
            continue

        time.sleep(dt)

    comms.stop()
    print("[relay] Stopped.")

    if len(all_headings) < 20:
        print("[relay] WARNING: Too few samples, using defaults")
        return PIDResult(0.03, 0.0003, 0.005, 0.0, 0.0, relay_amplitude, 0.0)

    # Only analyze data from non-bounce segments
    # Skip first 30% (startup transient)
    trim_start = int(len(all_headings) * 0.3)
    trimmed = all_headings[trim_start:]

    if len(trimmed) < 10:
        trimmed = all_headings[len(all_headings) // 2:]

    osc_amplitude = (max(trimmed) - min(trimmed)) / 2.0

    # Oscillation period from zero crossings
    trim_time = all_timestamps[trim_start] if trim_start < len(all_timestamps) else 0
    steady_crossings = [t for t in zero_crossings if t > trim_time]

    if len(steady_crossings) >= 3:
        half_periods = [
            steady_crossings[i + 1] - steady_crossings[i]
            for i in range(len(steady_crossings) - 1)
        ]
        tu = 2.0 * sum(half_periods) / len(half_periods)
    elif len(steady_crossings) == 2:
        tu = 2.0 * (steady_crossings[1] - steady_crossings[0])
    elif len(zero_crossings) >= 2:
        half_periods = [
            zero_crossings[i + 1] - zero_crossings[i]
            for i in range(len(zero_crossings) - 1)
        ]
        tu = 2.0 * sum(half_periods) / len(half_periods)
    else:
        print("[relay] WARNING: Few zero crossings, using estimated period")
        tu = 1.0

    # Ku = 4 * d / (pi * a)
    if osc_amplitude > 0.5:
        ku = 4.0 * relay_amplitude / (math.pi * math.radians(osc_amplitude))
    else:
        print("[relay] WARNING: Very small oscillation, using conservative default")
        ku = 0.1

    # Ziegler-Nichols "no overshoot"
    kp = 0.2 * ku
    ki = 0.4 * kp / tu if tu > 0 else 0.0
    kd = kp * tu / 3.0

    # Safety caps
    kp = min(kp, 0.05)
    ki = min(ki, 0.005)
    kd = min(kd, 0.02)

    print(f"\n[relay] Results:")
    print(f"  Legs: {leg_count}, Zero crossings: {len(zero_crossings)}")
    print(f"  Oscillation amplitude: +/-{osc_amplitude:.2f} deg")
    print(f"  Oscillation period Tu: {tu:.3f}s")
    print(f"  Ultimate gain Ku:      {ku:.4f}")
    print(f"  --- PID gains (Z-N no overshoot) ---")
    print(f"  Kp = {kp:.6f}")
    print(f"  Ki = {ki:.6f}")
    print(f"  Kd = {kd:.6f}")

    # Short trace
    trace_points = min(25, len(all_headings))
    step = max(1, len(all_headings) // trace_points)
    trace = [f"{all_headings[i]:+.1f}" for i in range(0, len(all_headings), step)]
    print(f"  Heading trace: [{', '.join(trace)}] deg")

    return PIDResult(kp, ki, kd, ku, tu, relay_amplitude, osc_amplitude)


# ---------------------------------------------------------------------------
# Phase 3: Verify with wall bouncing
# ---------------------------------------------------------------------------

def phase3_verify(
    comms: RobotComms,
    imu: IMUReader,
    kp: float,
    ki: float,
    kd: float,
    steering_bias: float = 0.0,
    throttle: float = 0.5,
    duration: float = 10.0,
    dt: float = 0.02,
) -> VerifyResult:
    """Drive straight with tuned PID, bounce off walls, measure performance."""
    print(f"\n{'='*60}")
    print(f"  PHASE 3: VERIFY")
    print(f"  Throttle={throttle:.0%}, PID=({kp:.4f}, {ki:.6f}, {kd:.4f})")
    print(f"  Bias={steering_bias:+.4f}, Duration={duration:.1f}s total")
    print(f"  Robot should drive STRAIGHT with PID corrections")
    print(f"{'='*60}")
    print("Starting in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)

    imu.reset_heading()
    time.sleep(0.3)

    all_headings = []
    leg_headings = []
    integral = 0.0
    prev_error = 0.0
    leg_start_time = time.monotonic()
    total_start = time.monotonic()
    leg_count = 0

    print("[verify] Driving with heading-hold PID...")
    while time.monotonic() - total_start < duration:
        h = imu.get_heading()
        leg_headings.append(h)

        # PID: target heading is 0 (straight)
        error = -h
        integral += error * dt
        integral = max(-10.0, min(10.0, integral))
        derivative = (error - prev_error) / dt if dt > 0.001 else 0.0
        prev_error = error

        steer = kp * error + ki * integral + kd * derivative + steering_bias
        steer = max(-1.0, min(1.0, steer))

        comms.send(throttle, steer)
        all_headings.append(h)

        # Check for wall hit
        if detect_wall_hit(leg_headings):
            leg_elapsed = time.monotonic() - leg_start_time
            leg_max = max(abs(x) for x in leg_headings) if leg_headings else 0
            leg_count += 1
            print(f"  [leg {leg_count}] max_dev={leg_max:.1f} deg, duration={leg_elapsed:.1f}s")

            wall_bounce(comms, imu, throttle)

            # Reset heading for new leg
            imu.reset_heading()
            time.sleep(0.1)
            leg_headings = []
            integral = 0.0
            prev_error = 0.0
            leg_start_time = time.monotonic()
            continue

        time.sleep(dt)

    # Final leg
    if leg_headings:
        leg_max = max(abs(x) for x in leg_headings) if leg_headings else 0
        leg_count += 1
        leg_elapsed = time.monotonic() - leg_start_time
        print(f"  [leg {leg_count}] max_dev={leg_max:.1f} deg, duration={leg_elapsed:.1f}s")

    comms.stop()
    print("[verify] Stopped.")

    if len(all_headings) < 5:
        return VerifyResult(0.0, 0.0, 0.0, len(all_headings), False, 0)

    max_dev = max(abs(h) for h in all_headings)
    mean_dev = sum(all_headings) / len(all_headings)
    rms_dev = math.sqrt(sum(h ** 2 for h in all_headings) / len(all_headings))
    passed = max_dev < 5.0  # 5 deg threshold (3 is very tight for wall-bounce)

    print(f"\n[verify] Results:")
    print(f"  Legs completed:  {leg_count}")
    print(f"  Max deviation:   {max_dev:.2f} deg")
    print(f"  RMS deviation:   {rms_dev:.2f} deg")
    print(f"  Mean deviation:  {mean_dev:+.2f} deg")
    print(f"  Samples:         {len(all_headings)}")
    print(f"  PASS:            {'YES' if passed else 'NO (>5 deg)'}")

    trace_points = min(25, len(all_headings))
    step = max(1, len(all_headings) // trace_points)
    trace = [f"{all_headings[i]:+.1f}" for i in range(0, len(all_headings), step)]
    print(f"  Heading trace:   [{', '.join(trace)}] deg")

    return VerifyResult(max_dev, rms_dev, mean_dev, len(all_headings), passed, leg_count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Drive calibration -- measure motor bias and tune heading-hold PID"
    )
    parser.add_argument(
        "--esp32", required=True,
        help="ESP32 hostname or IP (e.g., esp32wifi.local)"
    )
    parser.add_argument(
        "--throttle", type=float, default=0.5,
        help="Throttle level for tests (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--relay-amplitude", type=float, default=0.20,
        help="Relay steering amplitude for auto-tune (default: 0.20)"
    )
    parser.add_argument(
        "--duration", type=float, default=12.0,
        help="Duration per phase in seconds (default: 12.0)"
    )
    parser.add_argument(
        "--skip-bias", action="store_true",
        help="Skip bias test (use existing calibration or zero)"
    )
    parser.add_argument(
        "--skip-tune", action="store_true",
        help="Skip relay auto-tune (use existing calibration)"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only run verification with existing calibration"
    )
    parser.add_argument(
        "--output", default="drive_calibration.json",
        help="Output calibration file (default: drive_calibration.json)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  DRIVE CALIBRATION TOOL")
    print("  Measures motor bias and tunes heading-hold PID")
    print("  Robot will bounce off walls automatically")
    print("=" * 60)

    # Connect to ESP32
    comms = RobotComms(host=args.esp32)
    comms.connect()
    if not comms.connected:
        print("[error] Cannot connect to ESP32")
        return

    # Start IMU reader
    imu = IMUReader(host=comms._addr[0])
    try:
        imu.start()
    except RuntimeError as e:
        print(f"[error] {e}")
        comms.close()
        return

    # Load existing calibration if available
    cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    existing_cal = {}
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            existing_cal = json.load(f)
        print(f"[cal] Loaded existing calibration from {cal_path}")

    steering_bias = existing_cal.get("steering_bias", 0.0)
    kp = existing_cal.get("heading_hold_kp", 0.03)
    ki = existing_cal.get("heading_hold_ki", 0.0003)
    kd = existing_cal.get("heading_hold_kd", 0.005)

    try:
        # Phase 1: Bias test
        if not args.skip_bias and not args.verify_only:
            bias_result = phase1_bias_test(
                comms, imu,
                throttle=args.throttle,
                duration=args.duration,
            )
            steering_bias = bias_result.steering_bias
            print(f"\n[cal] Waiting 3s before next phase...")
            time.sleep(3.0)

        # Phase 2: Relay auto-tune
        if not args.skip_tune and not args.verify_only:
            pid_result = phase2_relay_autotune(
                comms, imu,
                steering_bias=steering_bias,
                throttle=args.throttle,
                relay_amplitude=args.relay_amplitude,
                duration=args.duration + 3.0,  # extra time for bounces
            )
            kp = pid_result.kp
            ki = pid_result.ki
            kd = pid_result.kd
            print(f"\n[cal] Waiting 3s before verification...")
            time.sleep(3.0)

        # Phase 3: Verify
        verify_result = phase3_verify(
            comms, imu,
            kp=kp, ki=ki, kd=kd,
            steering_bias=steering_bias,
            throttle=args.throttle,
            duration=args.duration,
        )

        # Save calibration
        calibration = {
            "steering_bias": round(steering_bias, 6),
            "heading_hold_kp": round(kp, 6),
            "heading_hold_ki": round(ki, 6),
            "heading_hold_kd": round(kd, 6),
            "throttle_tested": args.throttle,
            "verify_max_deviation_deg": round(verify_result.max_deviation_deg, 2),
            "verify_rms_deviation_deg": round(verify_result.rms_deviation_deg, 2),
            "verify_passed": verify_result.passed,
            "verify_legs": verify_result.legs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(cal_path, "w") as f:
            json.dump(calibration, f, indent=2)
        print(f"\n[cal] Saved calibration to {cal_path}")
        print(json.dumps(calibration, indent=2))

        if verify_result.passed:
            print("\n[PASS] CALIBRATION COMPLETE -- heading hold within +/-5 deg")
        else:
            print(f"\n[FAIL] Heading deviation too high ({verify_result.max_deviation_deg:.1f} deg)")
            print("  Try: --throttle 0.3 (slower speed)")
            print("  Or:  --relay-amplitude 0.30 (stronger oscillation)")

    except KeyboardInterrupt:
        print("\n[cal] Interrupted by user")
        comms.stop()

    finally:
        imu.stop()
        comms.close()


if __name__ == "__main__":
    main()
