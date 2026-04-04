"""Automatic PID turn tuner.

Sends turn commands, measures actual result via IMU, adjusts gains,
and repeats until error is within target.

Updates ESP32 gains via HTTP API, no reflashing needed.

Usage:
    python auto_tune_turn.py
    python auto_tune_turn.py --target-error 5  # tune until <5 deg error
"""

import argparse
import math
import struct
import socket
import time
import json

import cv2
import requests

from tracker import create_camera, ArUcoTracker


ESP32_IP = "192.168.4.113"
UDP_PORT = 4210


def send_turn(sock, addr, delta_deg):
    delta_units = int(delta_deg * 100)
    delta_units = max(-32767, min(32767, delta_units))
    packet = struct.pack(">hhBBh", 0, 0, 0, 1, delta_units)
    sock.sendto(packet, addr)


def send_stop(sock, addr):
    sock.sendto(struct.pack(">hhB", 0, 0, 0), addr)


def get_imu_yaw(ip):
    try:
        r = requests.get(f"http://{ip}/api/imu", timeout=0.5)
        return r.json().get("yaw", None)
    except:
        return None


def reset_imu_yaw(ip):
    try:
        requests.post(f"http://{ip}/api/imu", data={"resetYaw": "1"}, timeout=0.5)
    except:
        pass


def do_turn_test(sock, addr, ip, cmd_deg, timeout=4.0):
    """Execute a turn and measure the result.

    Returns dict with cmd, imu_result, error, settle_time, overshot.
    """
    # Get starting yaw
    start_yaw = get_imu_yaw(ip)
    if start_yaw is None:
        return None

    # Send turn command
    send_turn(sock, addr, cmd_deg)
    t0 = time.time()

    # Poll IMU while turning
    samples = []
    prev_yaw = start_yaw
    settled_time = None

    while time.time() - t0 < timeout:
        # Keep sending as heartbeat
        send_turn(sock, addr, cmd_deg)

        yaw = get_imu_yaw(ip)
        if yaw is not None:
            delta = yaw - start_yaw
            samples.append((time.time() - t0, delta))

            # Detect when it first settles (stops changing)
            if settled_time is None and len(samples) > 5:
                recent = [s[1] for s in samples[-5:]]
                if max(recent) - min(recent) < 2.0:
                    settled_time = time.time() - t0

        time.sleep(0.04)  # ~25Hz polling

    # Send stop
    for _ in range(5):
        send_stop(sock, addr)

    time.sleep(0.3)

    # Final measurement
    final_yaw = get_imu_yaw(ip)
    if final_yaw is None:
        final_yaw = samples[-1][1] + start_yaw if samples else start_yaw

    imu_delta = final_yaw - start_yaw
    error = imu_delta - cmd_deg

    # Detect overshoot: did it pass the target and come back?
    overshot = False
    max_delta = max(s[1] for s in samples) if samples else 0
    min_delta = min(s[1] for s in samples) if samples else 0
    if cmd_deg > 0 and max_delta > cmd_deg + 5:
        overshot = True
    if cmd_deg < 0 and min_delta < cmd_deg - 5:
        overshot = True

    # Count oscillations (sign changes in error relative to target)
    oscillations = 0
    if len(samples) > 2:
        errors = [s[1] - cmd_deg for s in samples]
        for i in range(1, len(errors)):
            if errors[i] * errors[i-1] < 0:
                oscillations += 1

    return {
        "cmd": cmd_deg,
        "result": imu_delta,
        "error": error,
        "abs_error": abs(error),
        "settle_time": settled_time,
        "overshot": overshot,
        "oscillations": oscillations,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esp32", default=ESP32_IP)
    parser.add_argument("--target-error", type=float, default=8.0,
                        help="Target max error in degrees (default 8)")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--mono", action="store_true", default=True)
    args = parser.parse_args()

    ip = args.esp32
    addr = (ip, UDP_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("=== Auto Turn Tuner ===")
    print(f"ESP32: {ip}")
    print(f"Target error: <{args.target_error} deg")
    print()

    # Current gains (read from what we know is flashed)
    gains = {
        "kP": 4.0,
        "kD": 0.035,
        "min_delta_far": 120,
        "min_delta_near": 60,
        "max_delta": 350,
        "tolerance": 4.0,
    }

    # Test angles
    test_angles = [45, 90, -45, -90]

    reset_imu_yaw(ip)
    time.sleep(0.5)

    for round_num in range(1, args.max_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num} | kP={gains['kP']:.2f} kD={gains['kD']:.4f} "
              f"minFar={gains['min_delta_far']} minNear={gains['min_delta_near']} "
              f"maxD={gains['max_delta']}")
        print(f"{'='*60}")

        results = []
        for angle in test_angles:
            reset_imu_yaw(ip)
            time.sleep(0.5)

            result = do_turn_test(sock, addr, ip, angle, timeout=3.5)
            if result is None:
                print(f"  {angle:+4d} deg: FAILED (no IMU)")
                continue

            results.append(result)
            status = "OVERSHOOT" if result["overshot"] else "OK"
            osc = f" osc={result['oscillations']}" if result["oscillations"] > 0 else ""
            print(f"  {angle:+4d} deg -> {result['result']:+6.1f} deg | "
                  f"err={result['error']:+5.1f} | {status}{osc}")

        if not results:
            print("No results — ESP32 unreachable?")
            break

        # Analyze results
        avg_error = sum(r["abs_error"] for r in results) / len(results)
        avg_signed = sum(r["error"] for r in results) / len(results)
        any_overshoot = any(r["overshot"] for r in results)
        total_osc = sum(r["oscillations"] for r in results)
        max_error = max(r["abs_error"] for r in results)

        print(f"\n  Avg error: {avg_error:.1f} deg | Max: {max_error:.1f} | "
              f"Signed avg: {avg_signed:+.1f} | Overshoots: {any_overshoot} | "
              f"Oscillations: {total_osc}")

        # Check if we're done
        if avg_error < args.target_error and not any_overshoot and total_osc < 2:
            print(f"\n*** TARGET REACHED! avg_error={avg_error:.1f} < {args.target_error} ***")
            break

        # Adjust gains based on results
        old_gains = gains.copy()

        if any_overshoot or total_osc > 2:
            # Bouncy — increase D, decrease max power
            gains["kD"] = min(0.08, gains["kD"] * 1.3)
            gains["max_delta"] = max(200, gains["max_delta"] - 25)
            print(f"  -> Bouncy: kD {old_gains['kD']:.4f}->{gains['kD']:.4f}, "
                  f"maxD {old_gains['max_delta']}->{gains['max_delta']}")

        if avg_signed > 5:
            # Consistently overshooting — reduce kP or increase D
            gains["kP"] = max(1.0, gains["kP"] * 0.9)
            gains["kD"] = min(0.08, gains["kD"] * 1.15)
            print(f"  -> Overshoot: kP {old_gains['kP']:.2f}->{gains['kP']:.2f}, "
                  f"kD {old_gains['kD']:.4f}->{gains['kD']:.4f}")

        elif avg_signed < -5:
            # Consistently undershooting — increase kP or min_delta
            gains["kP"] = min(8.0, gains["kP"] * 1.1)
            gains["min_delta_far"] = min(200, gains["min_delta_far"] + 10)
            print(f"  -> Undershoot: kP {old_gains['kP']:.2f}->{gains['kP']:.2f}, "
                  f"minFar {old_gains['min_delta_far']}->{gains['min_delta_far']}")

        elif avg_error > args.target_error * 1.5:
            # Large error but no consistent direction — adjust kP
            if max_error > 30:
                gains["kP"] = max(1.0, gains["kP"] * 0.85)
                print(f"  -> Large error: kP {old_gains['kP']:.2f}->{gains['kP']:.2f}")
            else:
                gains["min_delta_near"] = min(100, gains["min_delta_near"] + 10)
                print(f"  -> Near error: minNear {old_gains['min_delta_near']}->{gains['min_delta_near']}")

        # Apply new gains to ESP32 (we can't do this via HTTP yet,
        # so we update the firmware constants and reflash)
        # For now, just report what the gains should be
        print(f"\n  Recommended firmware constants:")
        print(f"    TURN_KP = {gains['kP']:.2f}f;")
        print(f"    TURN_KD = {gains['kD']:.4f}f;")
        print(f"    TURN_MIN_DELTA_FAR = {gains['min_delta_far']};")
        print(f"    TURN_MIN_DELTA_NEAR = {gains['min_delta_near']};")
        print(f"    TURN_MAX_DELTA = {gains['max_delta']};")

        # Send new gains to ESP32 instantly via UDP
        send_gains(sock, addr, gains)
        print("  -> Gains sent to ESP32 (live update)")
        time.sleep(1)  # brief settle

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL GAINS:")
    print(f"  TURN_KP = {gains['kP']:.2f}f;")
    print(f"  TURN_KD = {gains['kD']:.4f}f;")
    print(f"  TURN_MIN_DELTA_FAR = {gains['min_delta_far']};")
    print(f"  TURN_MIN_DELTA_NEAR = {gains['min_delta_near']};")
    print(f"  TURN_MAX_DELTA = {gains['max_delta']};")
    print(f"{'='*60}")

    sock.close()


def send_gains(sock, addr, gains):
    """Send gains to ESP32 via UDP mode 2 — instant, no reflash."""
    # 28-byte packet: 8-byte header + 5 floats (little-endian)
    header = struct.pack(">hhBBh", 0, 0, 0, 2, 0)  # mode=2
    payload = struct.pack("<fffff",
        gains["kP"],
        gains["kD"],
        float(gains["min_delta_far"]),
        float(gains["min_delta_near"]),
        float(gains["max_delta"]),
    )
    sock.sendto(header + payload, addr)


def update_and_flash(gains):
    """No longer needed — gains are sent live via UDP."""
    pass


if __name__ == "__main__":
    main()
