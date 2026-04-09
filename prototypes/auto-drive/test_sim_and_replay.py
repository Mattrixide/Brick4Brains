"""Full verification test: runs simulation, generates test replay, then loads both
into replay.html via Playwright and validates all systems.

Usage:
    python test_sim_and_replay.py

Requires: playwright (pip install playwright && playwright install chromium)
"""
import json
import math
import os
import subprocess
import sys
import time

PYTHON = sys.executable
BASE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(BASE, "logs")
SCREENSHOT_DIR = os.path.join(LOGS, "test_screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

PASS_COUNT = 0
FAIL_COUNT = 0
WARNINGS = []


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS: {name}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL: {name} -- {detail}")


def warn(msg):
    WARNINGS.append(msg)
    print(f"  WARN: {msg}")


# ======================================================================
# Phase 1: Code integrity
# ======================================================================
print("\n" + "=" * 60)
print("PHASE 1: Code Integrity")
print("=" * 60)

import ast
for f in ["state_machine.py", "match_timer.py", "battle_config.py",
          "main.py", "generate_test_replay.py", "sim_full_match.py",
          "dashboard_server.py"]:
    path = os.path.join(BASE, f)
    try:
        ast.parse(open(path).read())
        check(f"Parse {f}", True)
    except SyntaxError as e:
        check(f"Parse {f}", False, str(e))


# ======================================================================
# Phase 2: No removed state references
# ======================================================================
print("\n" + "=" * 60)
print("PHASE 2: No Removed State References")
print("=" * 60)

removed_patterns = ['"scan"', '"charge_ram"', '"charge_pin"']
found_refs = []
for root, dirs, files in os.walk(BASE):
    if "logs" in root or "__pycache__" in root or "node_modules" in root:
        continue
    for f in files:
        if not f.endswith(".py") or f == "test_sim_and_replay.py":
            continue
        path = os.path.join(root, f)
        for i, line in enumerate(open(path), 1):
            if line.strip().startswith("#"):
                continue
            for pat in removed_patterns:
                if pat in line:
                    found_refs.append(f"{path}:{i}: {pat}")

check("No removed state refs in Python files", len(found_refs) == 0,
      f"Found: {found_refs[:5]}")


# ======================================================================
# Phase 3: Unit tests — MatchTimer, BattleConfig, BattleController
# ======================================================================
print("\n" + "=" * 60)
print("PHASE 3: Unit Tests")
print("=" * 60)

sys.path.insert(0, BASE)
from match_timer import MatchTimer, PinTimer
from battle_config import BattleConfig
from state_machine import BattleController, BattleContext

# MatchTimer phase logic
mt = MatchTimer(duration_s=180, phase_start_s=30, phase_final_s=30)
check("MatchTimer phase before start", mt.phase == "start")

# Phase boundary validation (60s match with 30+30 should clamp)
mt_short = MatchTimer(duration_s=60, phase_start_s=30, phase_final_s=30)
total_phase = mt_short._phase_start_s + mt_short._phase_final_s
check("Phase boundaries clamped for short match",
      total_phase < 60,
      f"start={mt_short._phase_start_s:.1f} + final={mt_short._phase_final_s:.1f} = {total_phase:.1f}")

# BattleConfig new fields
cfg = BattleConfig()
check("BattleConfig opening_strategy default", cfg.opening_strategy == "charge")
check("BattleConfig push_commit_s default", cfg.push_commit_s == 1.0)
check("BattleConfig stall_speed_threshold", cfg.stall_speed_threshold == 8.0)
check("BattleConfig victory_dance_duration_s", cfg.victory_dance_duration_s == 3.0)

# BattleConfig validation
cfg_bad = BattleConfig(opening_strategy="invalid")
check("BattleConfig invalid opening falls back", cfg_bad.opening_strategy == "charge")

# BattleConfig backward compat load
cfg_loaded = BattleConfig.load(os.path.join(BASE, "battle_config.json"))
check("BattleConfig loads old config without crash",
      cfg_loaded.strategy in ("charge", "pit", "evade"))
check("BattleConfig new fields have defaults after old load",
      cfg_loaded.opening_strategy == "charge")

# BattleController state machine
mt2 = MatchTimer(duration_s=60)
pt = PinTimer()
cfg2 = BattleConfig()
bc = BattleController(cfg2, mt2, pt)

check("BattleController starts in wait", bc.state == "wait")

# Start match
mt2.start()
ctx = BattleContext(enemy_detected=True, enemy_tracking=True)
bc.start_match(ctx)
check("start_match transitions to acquire", bc.state == "acquire")

# Victory dance
bc.enter_victory_dance()
check("enter_victory_dance transitions", bc.state == "victory_dance")
check("is_dance_finished initially False", not bc.is_dance_finished)

# Reset
bc.reset()
check("reset goes to wait", bc.state == "wait")

# Debug info has all expected fields
mt3 = MatchTimer(duration_s=60)
pt3 = PinTimer()
bc3 = BattleController(BattleConfig(), mt3, pt3)
mt3.start()
info = bc3.debug_info
expected_keys = {"stuck_frames", "unstick_phase", "aruco_lost", "retreat_reason",
                 "phase", "opening", "push_commit_active", "hit_count", "stall_speed"}
check("debug_info has all keys", expected_keys.issubset(set(info.keys())),
      f"Missing: {expected_keys - set(info.keys())}")

# Opening strategies
for opening in ["charge", "fast_pin", "center", "avoid", "pit"]:
    mt_o = MatchTimer(duration_s=60)
    pt_o = PinTimer()
    cfg_o = BattleConfig(opening_strategy=opening)
    bc_o = BattleController(cfg_o, mt_o, pt_o)
    mt_o.start()
    ctx_o = BattleContext(enemy_detected=True, enemy_tracking=True)
    try:
        bc_o.start_match(ctx_o)
        check(f"Opening '{opening}' starts without crash", True)
    except Exception as e:
        check(f"Opening '{opening}' starts without crash", False, str(e))


# ======================================================================
# Phase 4: Generate test replay
# ======================================================================
print("\n" + "=" * 60)
print("PHASE 4: Generate Test Replay")
print("=" * 60)

result = subprocess.run(
    [PYTHON, os.path.join(BASE, "generate_test_replay.py")],
    capture_output=True, text=True, cwd=BASE
)
check("generate_test_replay.py runs", result.returncode == 0, result.stderr[:200] if result.stderr else "")

test_jsonl = os.path.join(LOGS, "test_match_3min.jsonl")
check("Test replay JSONL created", os.path.exists(test_jsonl))

if os.path.exists(test_jsonl):
    with open(test_jsonl) as f:
        test_frames = [json.loads(l) for l in f]
    test_states = set(f["bs"] for f in test_frames)
    test_phases = set(f["mp"] for f in test_frames)

    check("Test replay has 16+ states", len(test_states) >= 16,
          f"Got {len(test_states)}: {sorted(test_states)}")
    check("Test replay has mp field", all("mp" in f for f in test_frames))
    check("Test replay has start/mid/final phases",
          {"start", "mid", "final"}.issubset(test_phases),
          f"Got: {test_phases}")
    check("Test replay has wait state", "wait" in test_states)
    check("Test replay has victory_dance state", "victory_dance" in test_states)
    check("Test replay has pin state (not charge_pin)", "pin" in test_states and "charge_pin" not in test_states)


# ======================================================================
# Phase 5: Run full simulation
# ======================================================================
print("\n" + "=" * 60)
print("PHASE 5: Full Simulation")
print("=" * 60)

result = subprocess.run(
    [PYTHON, os.path.join(BASE, "sim_full_match.py")],
    capture_output=True, text=True, cwd=BASE, timeout=60
)
check("sim_full_match.py runs", result.returncode == 0,
      result.stderr[:300] if result.stderr else result.stdout[-300:] if result.stdout else "")

sim_jsonl = os.path.join(LOGS, "sim_full_match.jsonl")
check("Sim replay JSONL created", os.path.exists(sim_jsonl))

if os.path.exists(sim_jsonl):
    with open(sim_jsonl) as f:
        sim_frames = [json.loads(l) for l in f]
    sim_states = set(f["bs"] for f in sim_frames)
    sim_phases = set(f.get("mp") for f in sim_frames if f.get("mp"))

    check("Sim has 5+ states", len(sim_states) >= 5,
          f"Got {len(sim_states)}: {sorted(sim_states)}")
    check("Sim has all 4 phases",
          {"start", "mid", "final", "post"}.issubset(sim_phases),
          f"Got: {sim_phases}")
    check("Sim has victory_dance", "victory_dance" in sim_states)
    check("Sim has wait", "wait" in sim_states)
    check("Sim has mp field in frames", all("mp" in f for f in sim_frames))
    check("Sim has debug_info fields",
          "phase" in sim_frames[0] and "hit_count" in sim_frames[0],
          f"Keys: {list(sim_frames[0].keys())[:10]}")

    # Verify frame timestamps are monotonic
    times = [f["t"] for f in sim_frames]
    check("Sim timestamps monotonic",
          all(times[i] <= times[i+1] for i in range(len(times)-1)))

    # Verify arena metadata
    sim_arena = os.path.join(LOGS, "sim_full_match_arena.json")
    check("Sim arena JSON exists", os.path.exists(sim_arena))
    if os.path.exists(sim_arena):
        with open(sim_arena) as f:
            arena = json.load(f)
        required_keys = {"origin_x", "origin_y", "px_per_cm", "frame_w", "frame_h",
                         "arena_width_cm", "arena_height_cm", "pit_x_cm", "pit_y_cm"}
        check("Arena metadata has required keys",
              required_keys.issubset(set(arena.keys())),
              f"Missing: {required_keys - set(arena.keys())}")


# ======================================================================
# Phase 6: Playwright — Replay Viewer with simulation data
# ======================================================================
print("\n" + "=" * 60)
print("PHASE 6: Playwright Replay Viewer Tests")
print("=" * 60)

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("  SKIP: Playwright not installed")
    WARNINGS.append("Playwright not installed — skipped replay viewer tests")
else:
    replay_html = os.path.join(BASE, "replay.html")
    replay_url = f"file:///{replay_html.replace(os.sep, '/')}"

    js_errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.on("pageerror", lambda err: js_errors.append(str(err)))
        page.on("console", lambda msg: js_errors.append(f"[{msg.type}] {msg.text}")
                if msg.type == "error" else None)

        # ------------------------------------------------------------------
        # Test A: Load simulation replay (new format with mp field)
        # ------------------------------------------------------------------
        print("\n  --- Test A: Simulation Replay (new format) ---")
        page.goto(replay_url)
        page.wait_for_timeout(500)

        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "01_drop_zone.png"))

        # Load sim files
        sim_files = [
            os.path.join(LOGS, "sim_full_match.jsonl"),
            os.path.join(LOGS, "sim_full_match_arena.json"),
            os.path.join(LOGS, "sim_full_match_arena.png"),
        ]
        sim_files = [f for f in sim_files if os.path.exists(f)]

        if len(sim_files) >= 1:
            file_input = page.locator("#file-input")
            file_input.set_input_files(sim_files)
            page.wait_for_timeout(1500)

            # Check frames loaded
            frame_count = page.evaluate("() => frames.length")
            check("Sim replay loaded frames", frame_count > 0, f"Got {frame_count}")

            # Check state events built
            event_count = page.evaluate("() => stateEvents.length")
            check("State events built", event_count > 0, f"Got {event_count}")

            # Check phase events built
            phase_event_count = page.evaluate("() => phaseEvents.length")
            check("Phase events built", phase_event_count > 0, f"Got {phase_event_count}")

            # Check hasPhaseData
            has_phase = page.evaluate("() => hasPhaseData")
            check("hasPhaseData is true for new data", has_phase)

            # Check STATE_COLORS has new states
            has_pin = page.evaluate("() => STATE_COLORS['pin'] !== undefined")
            has_wait = page.evaluate("() => STATE_COLORS['wait'] !== undefined")
            has_vd = page.evaluate("() => STATE_COLORS['victory_dance'] !== undefined")
            has_wr = page.evaluate("() => STATE_COLORS['wall_reverse'] !== undefined")
            has_la = page.evaluate("() => STATE_COLORS['lost_aruco'] !== undefined")
            check("STATE_COLORS has 'pin'", has_pin)
            check("STATE_COLORS has 'wait'", has_wait)
            check("STATE_COLORS has 'victory_dance'", has_vd)
            check("STATE_COLORS has 'wall_reverse'", has_wr)
            check("STATE_COLORS has 'lost_aruco'", has_la)

            # Check backward compat aliases
            has_cp = page.evaluate("() => STATE_COLORS['charge_pin'] !== undefined")
            has_cr = page.evaluate("() => STATE_COLORS['charge_ram'] !== undefined")
            has_scan = page.evaluate("() => STATE_COLORS['scan'] !== undefined")
            check("Backward compat: charge_pin alias", has_cp)
            check("Backward compat: charge_ram alias", has_cr)
            check("Backward compat: scan alias", has_scan)

            # Check PHASE_COLORS
            for ph in ["start", "mid", "final", "post"]:
                has = page.evaluate(f"() => PHASE_COLORS['{ph}'] !== undefined")
                check(f"PHASE_COLORS has '{ph}'", has)

            # Screenshot frame 0
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "02_sim_frame0.png"))

            # Check phase badge in data panel
            phase_el = page.locator("#d-phase")
            phase_text = phase_el.inner_text()
            check("Phase badge shows value (not --)", phase_text != "--",
                  f"Got: '{phase_text}'")

            # Check battle state badge
            state_el = page.locator("#d-battle-state")
            state_text = state_el.inner_text()
            check("Battle state badge shows value", state_text != "--",
                  f"Got: '{state_text}'")

            # Step through frames — verify no JS errors
            for _ in range(50):
                page.keyboard.press("ArrowRight")
            page.wait_for_timeout(200)
            cur_frame = page.evaluate("() => currentFrame")
            check("Arrow stepping works", cur_frame == 50, f"Got frame {cur_frame}")

            # Jump to middle (should be MID phase)
            total = page.evaluate("() => frames.length")
            mid_frame = total // 2
            page.evaluate(f"() => {{ renderFrame({mid_frame}); drawStateBar(); }}")
            page.wait_for_timeout(300)
            mid_phase = page.evaluate(f"() => frames[{mid_frame}].mp")
            check("Mid-match frame has 'mid' phase", mid_phase == "mid",
                  f"Got: '{mid_phase}'")
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "03_sim_middle.png"))

            # Jump to near end (should be FINAL or POST phase)
            end_frame = total - 100 if total > 200 else total - 10
            page.evaluate(f"() => {{ renderFrame({end_frame}); drawStateBar(); }}")
            page.wait_for_timeout(300)
            end_phase = page.evaluate(f"() => frames[{end_frame}].mp")
            check("Near-end frame has final/post phase",
                  end_phase in ("final", "post"),
                  f"Got: '{end_phase}'")
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "04_sim_end.png"))

            # Jump to last frame (victory dance)
            page.evaluate(f"() => {{ renderFrame({total - 1}); drawStateBar(); }}")
            page.wait_for_timeout(300)
            last_state = page.evaluate(f"() => frames[{total - 1}].bs")
            check("Last frame is victory_dance", last_state == "victory_dance",
                  f"Got: '{last_state}'")
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "05_sim_victory.png"))

            # Play for 2 seconds and verify no crash
            page.evaluate("() => { renderFrame(0); drawStateBar(); }")
            page.keyboard.press("Space")
            page.wait_for_timeout(2000)
            page.keyboard.press("Space")
            played_frame = page.evaluate("() => currentFrame")
            check("Playback advances frames", played_frame > 10,
                  f"At frame {played_frame}")
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "06_sim_after_play.png"))

            # Check state bar canvas is rendered (not blank)
            bar_pixels = page.evaluate("""() => {
                const c = document.getElementById('state-bar');
                const ctx = c.getContext('2d');
                const data = ctx.getImageData(0, 0, c.width, 1).data;
                let nonZero = 0;
                for (let i = 0; i < data.length; i += 4) {
                    if (data[i] > 0 || data[i+1] > 0 || data[i+2] > 0) nonZero++;
                }
                return nonZero;
            }""")
            check("State bar has rendered content", bar_pixels > 10,
                  f"Non-zero pixels: {bar_pixels}")

        # ------------------------------------------------------------------
        # Test B: Load OLD replay file (backward compat — no mp field)
        # ------------------------------------------------------------------
        print("\n  --- Test B: Old Replay (backward compat) ---")
        page.goto(replay_url)
        page.wait_for_timeout(500)

        # Find an old log file
        old_files = []
        for fname in sorted(os.listdir(LOGS)):
            if fname.startswith("frames_") and fname.endswith(".jsonl"):
                old_files.append(fname)

        if old_files:
            old_base = old_files[0].replace(".jsonl", "")
            old_jsonl = os.path.join(LOGS, f"{old_base}.jsonl")
            old_arena_json = os.path.join(LOGS, f"{old_base}_arena.json")
            old_arena_png = os.path.join(LOGS, f"{old_base}_arena.png")
            old_load = [f for f in [old_jsonl, old_arena_json, old_arena_png] if os.path.exists(f)]

            if old_load:
                file_input = page.locator("#file-input")
                file_input.set_input_files(old_load)
                page.wait_for_timeout(1500)

                old_frame_count = page.evaluate("() => frames.length")
                check("Old replay loads", old_frame_count > 0, f"Got {old_frame_count}")

                old_has_phase = page.evaluate("() => hasPhaseData")
                check("Old replay: hasPhaseData is false", not old_has_phase)

                old_phase_events = page.evaluate("() => phaseEvents.length")
                check("Old replay: no phase events", old_phase_events == 0)

                # Phase badge should show --
                old_phase_text = page.locator("#d-phase").inner_text()
                check("Old replay: phase badge shows --", old_phase_text == "--",
                      f"Got: '{old_phase_text}'")

                # Old charge_pin state should render (not crash)
                # Check if any frame has charge_pin
                has_old_pin = page.evaluate("""() => {
                    return frames.some(f => f.bs === 'charge_pin');
                }""")
                if has_old_pin:
                    # Jump to a charge_pin frame
                    pin_frame = page.evaluate("""() => {
                        return frames.findIndex(f => f.bs === 'charge_pin');
                    }""")
                    page.evaluate(f"() => {{ renderFrame({pin_frame}); drawStateBar(); }}")
                    page.wait_for_timeout(200)
                    # Check state color resolves (not gray fallback)
                    pin_color = page.evaluate("() => stateColor('charge_pin')")
                    check("Old charge_pin gets a color (not fallback)",
                          pin_color != "#888", f"Got: {pin_color}")

                page.screenshot(path=os.path.join(SCREENSHOT_DIR, "07_old_replay.png"))
        else:
            warn("No old replay files found in logs/ — skipping backward compat test")

        # ------------------------------------------------------------------
        # Test C: Load scripted test replay
        # ------------------------------------------------------------------
        print("\n  --- Test C: Scripted Test Replay ---")
        page.goto(replay_url)
        page.wait_for_timeout(500)

        test_files = [
            os.path.join(LOGS, "test_match_3min.jsonl"),
            os.path.join(LOGS, "test_match_3min_arena.json"),
            os.path.join(LOGS, "test_match_3min_arena.png"),
        ]
        test_files = [f for f in test_files if os.path.exists(f)]

        if test_files:
            file_input = page.locator("#file-input")
            file_input.set_input_files(test_files)
            page.wait_for_timeout(1500)

            test_fc = page.evaluate("() => frames.length")
            check("Test replay loads", test_fc > 0)

            test_has_phase = page.evaluate("() => hasPhaseData")
            check("Test replay: hasPhaseData is true", test_has_phase)

            # Check all unique states rendered
            unique_states = page.evaluate("""() => {
                const s = new Set(frames.map(f => f.bs));
                return Array.from(s);
            }""")
            check("Test replay has 16+ unique states in frames",
                  len(unique_states) >= 16,
                  f"Got {len(unique_states)}: {sorted(unique_states)}")

            # Verify each new state has a non-fallback color
            for state in ["wait", "pin", "victory_dance", "wall_reverse",
                          "lost_aruco", "goto_center", "charge_reorient"]:
                color = page.evaluate(f"() => stateColor('{state}')")
                check(f"State '{state}' has color (not fallback)",
                      color != "#888", f"Got: {color}")

            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "08_test_replay.png"))

            # Speed test — play at 8x for 3 seconds
            # Set max speed
            for _ in range(6):
                page.evaluate("() => { speedIdx = (speedIdx + 1) % SPEEDS.length; playSpeed = SPEEDS[speedIdx]; $('btn-speed').textContent = SPEEDS[speedIdx] + 'x'; }")
            page.keyboard.press("Space")
            page.wait_for_timeout(3000)
            page.keyboard.press("Space")
            fast_frame = page.evaluate("() => currentFrame")
            check("8x playback advances frames", fast_frame > 20,
                  f"Reached frame {fast_frame}")

        # ------------------------------------------------------------------
        # Final JS error check
        # ------------------------------------------------------------------
        print("\n  --- JS Error Check ---")
        # Filter out benign errors (file:// fetch, favicon)
        real_errors = [e for e in js_errors
                       if "error" in e.lower()
                       and "favicon" not in e.lower()
                       and "file:///" not in e]
        check("No JS errors during all tests", len(real_errors) == 0,
              f"Errors: {real_errors[:5]}")

        browser.close()


# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 60)
print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed, {len(WARNINGS)} warnings")
print("=" * 60)

if WARNINGS:
    print("\nWarnings:")
    for w in WARNINGS:
        print(f"  - {w}")

if FAIL_COUNT > 0:
    print(f"\nFAILED — {FAIL_COUNT} tests failed")
    sys.exit(1)
else:
    print("\nALL TESTS PASSED")
    print(f"\nScreenshots saved to: {SCREENSHOT_DIR}")
    sys.exit(0)
