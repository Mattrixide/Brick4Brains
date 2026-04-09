"""Generate a synthetic 3-minute match replay for testing the replay dashboard.

Updated for new state machine: uses wait, acquire, charge_pursue, pin, pit,
evade, wall_reverse, unstick, lost_target, lost_aruco, victory_dance.
Removed: scan, charge_ram, charge_pin.
Added: match phase (mp field), new states in script.
"""

import json
import math
import os
import random
import time

random.seed(42)

DURATION = 180.0  # 3 minutes
FPS = 60
ARENA_HALF = 122.0  # 244cm / 2
PHASE_START_S = 30.0
PHASE_FINAL_S = 30.0

# State sequence with approximate durations (seconds)
# Simulates a realistic match flow through all states and phases
SCRIPT = [
    # START phase — opening
    ("wait", 1.0),
    ("acquire", 2.0),
    ("charge_pursue", 5.0),
    ("pin", 5.0),
    ("evade_retreat", 1.5),
    ("acquire", 1.0),
    ("charge_pursue", 4.0),
    ("wall_reverse", 0.5),
    ("charge_reorient", 1.0),
    ("charge_pursue", 3.0),
    ("pin", 5.0),
    # MID phase
    ("evade_retreat", 1.2),
    ("lost_target", 3.0),
    ("acquire", 1.5),
    ("charge_pursue", 6.0),
    ("unstick", 1.5),
    ("charge_pursue", 4.0),
    ("pin", 5.0),
    ("evade_retreat", 1.0),
    ("charge_pursue", 8.0),
    ("charge_flank", 3.0),
    ("charge_pursue", 3.0),
    ("pin", 5.0),
    ("evade_retreat", 1.5),
    ("lost_aruco", 2.0),
    ("acquire", 2.0),
    ("charge_pursue", 6.0),
    ("lost_target", 2.5),
    ("acquire", 1.0),
    ("charge_pursue", 5.0),
    ("pin", 5.0),
    ("evade_retreat", 1.0),
    ("charge_pursue", 5.0),
    ("wall_reverse", 0.5),
    ("unstick", 1.5),
    ("charge_pursue", 4.0),
    ("charge_flank", 2.0),
    ("charge_pursue", 3.0),
    ("pin", 5.0),
    ("evade_retreat", 1.2),
    # Pit strategy sequence
    ("pit_position", 4.0),
    ("pit_push", 3.0),
    ("pit_abort", 1.5),
    ("charge_pursue", 5.0),
    ("pit_position", 3.0),
    ("pit_push", 2.0),
    ("pit_commit", 1.5),
    # FINAL phase — aggressive
    ("acquire", 1.0),
    ("charge_pursue", 8.0),
    ("pin", 5.0),
    ("evade_retreat", 1.0),
    ("charge_pursue", 6.0),
    ("pin", 5.0),
    ("evade_retreat", 1.0),
    ("charge_pursue", 5.0),
    # POST — victory dance
    ("victory_dance", 3.0),
]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def compute_phase(elapsed):
    """Compute match phase from elapsed time."""
    if elapsed >= DURATION:
        return "post"
    if elapsed < PHASE_START_S:
        return "start"
    if elapsed >= DURATION - PHASE_FINAL_S:
        return "final"
    return "mid"


def gen_frames():
    frames = []
    dt = 1.0 / FPS
    t = 0.0
    frame_num = 0

    # Robot positions
    ox, oy = 0.0, -60.0
    oh = math.pi / 2  # facing up
    ovx, ovy = 0.0, 0.0

    ex, ey = 0.0, 60.0
    eh = -math.pi / 2  # facing down
    evx, evy = 0.0, 0.0

    match_remaining = DURATION

    # Walk through script
    script_idx = 0
    state_start = 0.0
    state = SCRIPT[0][0]
    state_dur = SCRIPT[0][1]

    enemy_wander_angle = 0.0

    while t < DURATION:
        # Advance state
        if t - state_start >= state_dur:
            script_idx += 1
            if script_idx >= len(SCRIPT):
                script_idx = len(SCRIPT) - 1  # stay on last state
            state = SCRIPT[script_idx][0]
            state_dur = SCRIPT[script_idx][1]
            state_start = t

        progress = (t - state_start) / max(state_dur, 0.01)
        match_remaining = max(0, DURATION - t)
        urgency = max(0, min(1, 1.0 - match_remaining / 60.0)) if match_remaining < 60 else 0.0

        # Enemy wanders semi-randomly
        enemy_wander_angle += (random.random() - 0.5) * 0.3
        e_speed = 15.0 + random.random() * 10  # cm/s
        evx = math.cos(enemy_wander_angle) * e_speed
        evy = math.sin(enemy_wander_angle) * e_speed

        # State-dependent behavior
        throttle = 0.0
        steering = 0.0
        our_detected = True
        enemy_detected = True
        enemy_tracking = True

        if state == "wait":
            throttle = 0.0
            steering = 0.0
            ovx, ovy = 0.0, 0.0

        elif state == "goto_center":
            angle_to_center = math.atan2(-oy, -ox)
            oh += (angle_to_center - oh) * 0.1
            speed = 30
            ovx = math.cos(oh) * speed
            ovy = math.sin(oh) * speed
            throttle = 0.5
            steering = clamp((angle_to_center - oh) * 0.4, -0.4, 0.4)

        elif state == "acquire":
            throttle = 0.0
            steering = 0.0
            ovx, ovy = 0.0, 0.0
            enemy_detected = progress > 0.3
            enemy_tracking = progress > 0.5

        elif state == "charge_pursue":
            angle_to_enemy = math.atan2(ey - oy, ex - ox)
            oh += (angle_to_enemy - oh) * 0.1
            dist = math.hypot(ex - ox, ey - oy)
            # Throttle scales with distance
            speed_factor = min(1.0, max(0.4, 1.0 - dist / 200.0))
            speed = 40 + 30 * speed_factor
            ovx = math.cos(oh) * speed
            ovy = math.sin(oh) * speed
            throttle = 0.5 + 0.5 * speed_factor
            steering = clamp((angle_to_enemy - oh) * 0.3, -0.5, 0.5)
            # Close distance when near
            if dist < 30:
                ox += (ex - ox) * 0.1
                oy += (ey - oy) * 0.1

        elif state == "charge_flank":
            angle_to_enemy = math.atan2(ey - oy, ex - ox)
            flank_angle = angle_to_enemy + math.pi / 3
            speed = 35
            ovx = math.cos(flank_angle) * speed
            ovy = math.sin(flank_angle) * speed
            oh += 0.05
            throttle = 0.8
            steering = 0.3

        elif state == "charge_reorient":
            # Backing up and spinning
            if progress < 0.3:
                throttle = -0.35
                ovx, ovy = -10, 0
            else:
                throttle = 0.25
                oh += 0.15
                ovx = math.cos(oh) * 15
                ovy = math.sin(oh) * 15
            steering = 0.5

        elif state == "pin":
            throttle = 0.2
            steering = 0.0
            ovx, ovy = 1.0, 0.0
            # Push enemy to nearest wall
            wall_x = ARENA_HALF if ex > 0 else -ARENA_HALF
            ex += (wall_x - ex) * 0.05
            ox += (ex - ox) * 0.08
            oy += (ey - oy) * 0.08
            evx, evy = 0.0, 0.0
            e_speed = 0

        elif state == "evade_retreat":
            throttle = -0.8
            steering = 0.0
            retreat_angle = math.atan2(oy - ey, ox - ex)
            speed = 30
            ovx = math.cos(retreat_angle) * speed
            ovy = math.sin(retreat_angle) * speed
            oh = retreat_angle + math.pi

        elif state == "evade_reposition":
            # Drive toward center
            angle_to_center = math.atan2(-oy, -ox)
            oh += (angle_to_center - oh) * 0.1
            speed = 25
            ovx = math.cos(oh) * speed
            ovy = math.sin(oh) * speed
            throttle = 0.4
            steering = 0.2

        elif state == "wall_reverse":
            if progress < 0.2:
                throttle = 0.0
                ovx, ovy = 0.0, 0.0
            else:
                throttle = -0.6
                ovx = -math.cos(oh) * 25
                ovy = -math.sin(oh) * 25

        elif state == "unstick":
            phase = int((t - state_start) / 0.3) % 2
            throttle = 0.5 if phase == 0 else -0.5
            steering = 0.0
            ovx = 5.0 * (1 if phase == 0 else -1)
            ovy = 0.0

        elif state == "lost_target":
            if progress < 0.6:
                # Drive to last known
                throttle = 0.4
                steering = 0.1
                angle_to_last = math.atan2(ey - oy, ex - ox)
                oh += (angle_to_last - oh) * 0.05
                speed = 15
                ovx = math.cos(oh) * speed
                ovy = math.sin(oh) * speed
            else:
                # Slow rotation
                throttle = 0.0
                steering = 0.35
                ovx, ovy = 0.0, 0.0
            enemy_detected = False
            enemy_tracking = False

        elif state == "lost_aruco":
            throttle = 0.0
            steering = 0.0
            ovx, ovy = 0.0, 0.0
            our_detected = False
            # Drift slightly
            ox += random.gauss(0, 0.3)
            oy += random.gauss(0, 0.3)

        elif state == "pit_position":
            # Drive to herding position
            pit_x, pit_y = 78.9, 109.3
            angle_to_herd = math.atan2(ey + 40 - oy, ex + 40 - ox)
            oh += (angle_to_herd - oh) * 0.08
            speed = 30
            ovx = math.cos(oh) * speed
            ovy = math.sin(oh) * speed
            throttle = 0.6
            steering = 0.3

        elif state == "pit_push":
            pit_x, pit_y = 78.9, 109.3
            angle_to_pit = math.atan2(pit_y - oy, pit_x - ox)
            oh += (angle_to_pit - oh) * 0.1
            speed = 40
            ovx = math.cos(oh) * speed
            ovy = math.sin(oh) * speed
            throttle = 1.0
            steering = clamp((angle_to_pit - oh) * 0.5, -0.5, 0.5)
            # Push enemy toward pit
            ex += (pit_x - ex) * 0.02
            ey += (pit_y - ey) * 0.02

        elif state == "pit_commit":
            throttle = 1.0
            steering = 0.0
            speed = 50
            ovx = math.cos(oh) * speed
            ovy = math.sin(oh) * speed

        elif state == "pit_abort":
            throttle = -0.6
            steering = 0.0
            ovx = -math.cos(oh) * 25
            ovy = -math.sin(oh) * 25

        elif state == "victory_dance":
            throttle = 0.0
            steering = 0.8
            oh += 0.3
            ovx = math.cos(oh) * 5
            ovy = math.sin(oh) * 5

        # Update positions
        if state not in ("pin",):
            ox += ovx * dt
            oy += ovy * dt
            ex += evx * dt
            ey += evy * dt

        # Clamp to arena
        ox = clamp(ox, -ARENA_HALF + 5, ARENA_HALF - 5)
        oy = clamp(oy, -ARENA_HALF + 5, ARENA_HALF - 5)
        ex = clamp(ex, -ARENA_HALF + 5, ARENA_HALF - 5)
        ey = clamp(ey, -ARENA_HALF + 5, ARENA_HALF - 5)

        # Enemy heading from velocity
        if e_speed > 1:
            eh = math.atan2(evy, evx)

        dist = math.hypot(ex - ox, ey - oy)
        accel_x = random.gauss(0, 50) + abs(throttle) * 200
        accel_y = random.gauss(0, 30)

        rec = {
            "f": frame_num,
            "t": round(t, 4),
            "mode": "battle",
            "bs": state,
            "mp": compute_phase(t),
            "ox": round(ox, 1),
            "oy": round(oy, 1),
            "oh": round(oh, 3),
            "od": our_detected,
            "ovx": round(ovx, 1),
            "ovy": round(ovy, 1),
            "ex": round(ex, 1) if enemy_tracking else None,
            "ey": round(ey, 1) if enemy_tracking else None,
            "eh": round(eh, 3) if enemy_tracking else None,
            "evx": round(evx, 1) if enemy_tracking else None,
            "evy": round(evy, 1) if enemy_tracking else None,
            "ed": enemy_detected,
            "et": enemy_tracking,
            "dist": round(dist, 1) if enemy_tracking else 999.0,
            "thr": round(throttle, 3),
            "str": round(steering, 3),
            "mr": round(match_remaining, 1),
            "urg": round(urgency, 2),
            "ax": round(accel_x, 0),
            "ay": round(accel_y, 0),
            "rm": 0,
            "fps": 60.0,
            "phase": compute_phase(t),
            "ehm": "velocity",
            "ehc": round(0.5 + 0.5 * random.random(), 2),
            "stuck_frames": 0,
            "unstick_phase": 1 if state == "unstick" else 0,
            "aruco_lost": 90 if state == "lost_aruco" else 0,
            "retreat_reason": "pin_release" if state == "evade_retreat" else None,
            "hit_count": 0,
            "push_commit_active": False,
            "stall_speed": round(math.hypot(ovx, ovy), 1),
        }
        frames.append(json.dumps(rec, separators=(",", ":")))

        t += dt
        frame_num += 1

    return frames


def gen_arena_meta():
    """Generate test arena metadata matching the battle_config."""
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
    """Generate a synthetic arena image for testing."""
    import numpy as np
    try:
        import cv2
    except ImportError:
        print("cv2 not available — skipping arena image generation")
        return None

    w, h = meta["frame_w"], meta["frame_h"]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 25, 20)  # dark floor

    ox, oy = int(meta["origin_x"]), int(meta["origin_y"])
    s = meta["px_per_cm"]

    # Arena border
    half_w = int(meta["arena_width_cm"] / 2 * s)
    half_h = int(meta["arena_height_cm"] / 2 * s)
    cv2.rectangle(img, (ox - half_w, oy - half_h), (ox + half_w, oy + half_h), (60, 60, 60), 2)

    # Grid lines every 30cm
    for gcm in range(-120, 121, 30):
        gpx = int(ox + gcm * s)
        cv2.line(img, (gpx, oy - half_h), (gpx, oy + half_h), (35, 30, 25), 1)
        gpy = int(oy + gcm * s)
        cv2.line(img, (ox - half_w, gpy), (ox + half_w, gpy), (35, 30, 25), 1)

    # Center cross
    cv2.drawMarker(img, (ox, oy), (0, 80, 0), cv2.MARKER_CROSS, 20, 1)

    # Pit zone
    pit_cx = int(ox + meta["pit_x_cm"] * s)
    pit_cy = int(oy + meta["pit_y_cm"] * s)
    pit_r = int(meta["pit_radius_cm"] * s)
    cv2.rectangle(img, (pit_cx - pit_r, pit_cy - pit_r), (pit_cx + pit_r, pit_cy + pit_r), (0, 0, 120), 2)
    cv2.putText(img, "PIT", (pit_cx - 15, pit_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1)

    # Texture noise
    noise = np.random.randint(0, 8, (h, w), dtype=np.uint8)
    img[:, :, 0] = np.clip(img[:, :, 0].astype(int) + noise, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(img[:, :, 1].astype(int) + noise, 0, 255).astype(np.uint8)

    return img


if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "test_match_3min.jsonl")

    print(f"Generating {DURATION}s match at {FPS}fps ...")
    lines = gen_frames()
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} frames to {out_path}")

    # Generate arena metadata
    meta = gen_arena_meta()
    meta_path = out_path.replace(".jsonl", "_arena.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote arena metadata to {meta_path}")

    # Generate arena image
    img = gen_arena_image(meta)
    if img is not None:
        import cv2
        img_path = out_path.replace(".jsonl", "_arena.png")
        cv2.imwrite(img_path, img)
        print(f"Wrote arena image to {img_path}")
