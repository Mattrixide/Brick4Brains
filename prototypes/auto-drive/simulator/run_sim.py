"""CLI entry point for combat robot simulation."""

import argparse
import sys
import os

# Ensure parent paths are set up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulator  # noqa: trigger sys.path setup

from battle_config import BattleConfig
from simulator.sim_runner import SimConfig, run_single
from simulator.batch import BatchConfig, run_batch
from simulator.physics import PhysicsConfig
from simulator.vision import VisionConfig


def _make_battle_config(args) -> BattleConfig:
    """Build BattleConfig from CLI args."""
    bc = BattleConfig(strategy=args.strategy, safe_side=args.safe_side)
    if args.pit:
        bc.pit_x_cm = args.pit_x
        bc.pit_y_cm = args.pit_y
        bc.pit_radius_cm = args.pit_radius
        bc.pit_danger_radius_cm = args.pit_radius + 15.0
    return bc


def cmd_single(args):
    """Run a single simulation."""
    cfg = SimConfig(
        enemy_ai_type=args.enemy,
        match_duration_s=args.duration,
        seed=args.seed,
        battle_config=_make_battle_config(args),
    )
    result = run_single(cfg)
    print(f"Outcome:     {result.outcome}")
    print(f"Duration:    {result.duration_s:.1f}s")
    print(f"Pins:        {result.total_pins}")
    if result.time_to_first_pin:
        print(f"First pin:   {result.time_to_first_pin:.1f}s")
    print(f"Avg dist:    {result.avg_distance_cm:.0f}cm")
    print(f"Collisions:  {result.collision_count}")
    print(f"States:")
    for state, t in sorted(result.states_visited.items(), key=lambda x: -x[1]):
        print(f"  {state:20s} {t:.2f}s")


def cmd_batch(args):
    """Run a batch of simulations."""
    sweep = {}
    if args.enemies:
        sweep["enemy_ai_type"] = args.enemies.split(",")

    batch = BatchConfig(
        base_config=SimConfig(
            enemy_ai_type=args.enemy,
            match_duration_s=args.duration,
            battle_config=_make_battle_config(args),
        ),
        num_runs=args.runs,
        sweep_params=sweep,
        num_workers=args.workers,
        output_path=args.output,
    )
    run_batch(batch)


def cmd_sweep(args):
    """Run a parameter sweep."""
    values = [float(v) for v in args.values.split(",")]
    sweep = {args.param: values}

    if args.enemies:
        sweep["enemy_ai_type"] = args.enemies.split(",")

    batch = BatchConfig(
        base_config=SimConfig(
            enemy_ai_type=args.enemy,
            match_duration_s=args.duration,
            battle_config=_make_battle_config(args),
        ),
        num_runs=args.runs,
        sweep_params=sweep,
        num_workers=args.workers,
        output_path=args.output,
    )
    run_batch(batch)


def cmd_visual(args):
    """Run a single simulation with visualization."""
    try:
        from simulator.visualizer import Visualizer
        from simulator.physics import Arena
    except ImportError:
        print("pygame is required for visualization: pip install pygame")
        return

    bc = _make_battle_config(args)
    if args.pit:
        arena = Arena.with_corner_pit(
            corner="upper_right", size_cm=45.7, inset_cm=7.6, lip_cm=1.9
        )
    else:
        arena = Arena()
    vis = Visualizer(arena, scale=args.scale, speed=args.speed)

    cfg = SimConfig(
        enemy_ai_type=args.enemy,
        match_duration_s=args.duration,
        seed=args.seed,
        battle_config=bc,
    )
    result = run_single(cfg, visualizer=vis)
    vis.close()

    print(f"\nOutcome: {result.outcome}")
    print(f"Duration: {result.duration_s:.1f}s, Pins: {result.total_pins}")


def main():
    parser = argparse.ArgumentParser(description="B4B Combat Robot Simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    # Common args
    def add_common(p):
        p.add_argument("--enemy", default="random_walk",
                       help="Enemy AI type (stationary, random_walk, aggressive, defensive, wedge)")
        p.add_argument("--duration", type=float, default=180.0,
                       help="Match duration in seconds")
        p.add_argument("--strategy", default="charge",
                       help="Our strategy (charge, pit, evade)")
        p.add_argument("--safe-side", default="front", dest="safe_side",
                       choices=["front", "back", "left", "right"],
                       help="Which side of the enemy to approach from (default: front)")
        p.add_argument("--pit", action="store_true",
                       help="Enable pit hazard (upper-right, 1.5ft square)")
        p.add_argument("--pit-x", type=float, default=80.0,
                       help="Pit center X in cm (default: 80)")
        p.add_argument("--pit-y", type=float, default=80.0,
                       help="Pit center Y in cm (default: 80)")
        p.add_argument("--pit-radius", type=float, default=23.0,
                       help="Pit radius in cm (default: 23, ~1.5ft square)")

    # single
    p = sub.add_parser("single", help="Run one simulation")
    add_common(p)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_single)

    # batch
    p = sub.add_parser("batch", help="Run batch simulations")
    add_common(p)
    p.add_argument("--runs", type=int, default=100, help="Runs per config")
    p.add_argument("--workers", type=int, default=0, help="Parallel workers (0=auto)")
    p.add_argument("--output", default="sim_results.csv", help="Output CSV path")
    p.add_argument("--enemies", default=None,
                   help="Comma-separated enemy types to sweep")
    p.set_defaults(func=cmd_batch)

    # sweep
    p = sub.add_parser("sweep", help="Parameter sweep")
    add_common(p)
    p.add_argument("--param", required=True, help="Parameter to sweep (e.g. battle_config.charge_close_range_cm)")
    p.add_argument("--values", required=True, help="Comma-separated values")
    p.add_argument("--runs", type=int, default=50, help="Runs per value")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--output", default="sweep_results.csv")
    p.add_argument("--enemies", default=None)
    p.set_defaults(func=cmd_sweep)

    # visual
    p = sub.add_parser("visual", help="Run with visualization")
    add_common(p)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scale", type=float, default=3.0, help="Pixels per cm")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    p.set_defaults(func=cmd_visual)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
