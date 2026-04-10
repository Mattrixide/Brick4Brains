"""Regression test seeds for the combat simulator.

Each seed reproduces a specific bug found during development.
Run: python -m sim.test_seeds  (from prototypes/auto-drive/)

All seeds should pass — if any fail, a previously fixed bug has regressed.
"""
import math
import random
import sys
import os
import logging

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.arena import SimArena
from sim.bridge import SimBridge
from sim.enemy_ai import EnemyController


def run_match(seed, strategy="charge", duration_s=180, enemy_mode="flee"):
    """Run a single match and return stats."""
    random.seed(seed)
    arena = SimArena()
    bridge = SimBridge(arena.brick, arena.cfg, strategy_override=strategy)
    bridge.start_match(arena.enemy)
    ec = EnemyController()
    ec.set_mode(enemy_mode)

    pins = 0
    wr = 0
    us = 0
    prev = ""
    stuck_frames = 0
    total = 0
    states = set()

    frames = int(duration_s * 60)
    for i in range(frames):
        r = ec.get_drive(arena.enemy, arena.brick, 1 / 60, arena.cfg)
        if r:
            arena.enemy.apply_drive(r[0], r[1], arena.cfg)
        bridge.tick(1 / 60, arena.enemy)
        arena.step(1 / 60)
        bs = bridge.state
        states.add(bs)
        if bs == "pin" and bs != prev:
            pins += 1
        if bs == "wall_reverse" and bs != prev:
            wr += 1
        if bs == "unstick" and bs != prev:
            us += 1
        bspeed = math.hypot(*arena.brick.velocity)
        if bspeed < 2 and bs in ("charge_pursue", "charge_reorient"):
            stuck_frames += 1
        total += 1
        prev = bs
        if not arena.brick.alive or not arena.enemy.alive:
            break

    return {
        "seed": seed,
        "duration": i / 60,
        "pins": pins,
        "wall_reverses": wr,
        "unsticks": us,
        "stuck_pct": stuck_frames / max(1, total) * 100,
        "brick_alive": arena.brick.alive,
        "enemy_alive": arena.enemy.alive,
        "brick_pitted": not arena.brick.alive,
        "enemy_pitted": not arena.enemy.alive,
        "victory": "victory_dance" in states,
        "states": states,
    }


# =============================================================================
# Regression test seeds
# Each test documents: the bug, the seed that reproduced it, and the pass criteria
# =============================================================================

TESTS = [
    # --- Head-to-head deadlock (bug #13) ---
    # Brick and enemy in contact at speed=0 in open arena for 10+ seconds.
    # Fixed by contact stalemate detector (2s timeout → retreat).
    {
        "name": "Head-to-head deadlock (seed 652)",
        "seed": 652,
        "check": lambda r: r["stuck_pct"] < 20,
        "fail_msg": "stuck_pct={stuck_pct:.0f}% (should be <20%, was 43% before fix)",
    },
    {
        "name": "Open arena deadlock (seed 117)",
        "seed": 117,
        "check": lambda r: r["stuck_pct"] < 20,
        "fail_msg": "stuck_pct={stuck_pct:.0f}% (should be <20%, was 82% before fix)",
    },
    {
        "name": "Open arena deadlock (seed 750)",
        "seed": 750,
        "check": lambda r: r["stuck_pct"] < 20,
        "fail_msg": "stuck_pct={stuck_pct:.0f}% (should be <20%, was 80% before fix)",
    },
    {
        "name": "Open arena deadlock (seed 683)",
        "seed": 683,
        "check": lambda r: r["stuck_pct"] < 20,
        "fail_msg": "stuck_pct={stuck_pct:.0f}% (should be <20%, was 72% before fix)",
    },

    # --- Wall-follow grinding (bug #11, seed 53) ---
    # Brick repeatedly charges along wall: charge→wall_reverse 10+ times.
    # Fixed by recovery cycle breaker + not resetting counter in _enter_pin.
    {
        "name": "Wall-follow grinding (seed 53)",
        "seed": 53,
        "check": lambda r: r["wall_reverses"] < 5,
        "fail_msg": "wall_reverses={wall_reverses} (should be <5, was 10 before fix)",
    },

    # --- Wall-reverse/unstick death loop (bug #9, seed 89 from first 100-match batch) ---
    # 123 unsticks, 0 pins — pure wall-grinding for 3 minutes.
    # Fixed by recovery cycle breaker (3 cycles → forced retreat).
    {
        "name": "Recovery death loop (seed 89)",
        "seed": 89,
        "check": lambda r: r["unsticks"] < 30,
        "fail_msg": "unsticks={unsticks} (should be <30, was 123 before fix)",
    },

    # --- Brick self-pitting (various seeds) ---
    # Brick charges into pit. Unstick oscillation reverses into pit.
    # Fixed by pit avoidance in recovery states.
    {
        "name": "Brick self-pit early (seed 75)",
        "seed": 75,
        "check": lambda r: r["brick_alive"],
        "fail_msg": "Brick pitted itself (should survive)",
    },
    {
        "name": "Brick self-pit early (seed 269)",
        "seed": 269,
        "check": lambda r: r["brick_alive"],
        "fail_msg": "Brick pitted itself (should survive)",
    },

    # --- Victory dance should trigger on full matches ---
    # Fixed by adding match_timer.is_expired + pin_count check in tick().
    {
        "name": "Victory dance triggers (seed 0)",
        "seed": 0,
        "check": lambda r: r["victory"] if not r["brick_pitted"] and not r["enemy_pitted"] else True,
        "fail_msg": "No victory dance on full match (should celebrate)",
    },

    # --- Max wall_reverse/unstick (seed 550 from 1000-match batch) ---
    # Worst offender: 64 wall_reverses, 66 unsticks.
    {
        "name": "Excessive recovery (seed 550)",
        "seed": 550,
        "check": lambda r: r["wall_reverses"] < 70 and r["unsticks"] < 70,
        "fail_msg": "wall_reverses={wall_reverses} unsticks={unsticks} (should be <70 each, was 64/66 originally)",
    },
]


def main():
    print(f"Running {len(TESTS)} regression tests...\n")
    passed = 0
    failed = 0

    for test in TESTS:
        result = run_match(test["seed"])
        ok = test["check"](result)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
            print(f"  {status}  {test['name']}")
        else:
            failed += 1
            msg = test["fail_msg"].format(**result)
            print(f"  {status}  {test['name']} — {msg}")

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed out of {len(TESTS)} tests")
    if failed:
        print(f"  *** REGRESSIONS DETECTED ***")
        sys.exit(1)
    else:
        print(f"  All regression tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
