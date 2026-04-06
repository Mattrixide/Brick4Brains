"""Batch simulation runner with parallel execution and parameter sweeps."""

import copy
import csv
import itertools
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass, field

from simulator.sim_runner import SimConfig, SimResult, run_single


@dataclass
class BatchConfig:
    base_config: SimConfig = field(default_factory=SimConfig)
    num_runs: int = 100
    sweep_params: dict[str, list] = field(default_factory=dict)
    num_workers: int = 0  # 0 = cpu_count
    output_path: str = "sim_results.csv"


def _expand_sweeps(batch: BatchConfig) -> list[SimConfig]:
    """Expand parameter sweeps into individual SimConfig instances."""
    if not batch.sweep_params:
        # No sweeps — just vary seeds
        configs = []
        for i in range(batch.num_runs):
            cfg = copy.deepcopy(batch.base_config)
            cfg.seed = i
            configs.append(cfg)
        return configs

    # Build all combinations
    param_names = list(batch.sweep_params.keys())
    param_values = list(batch.sweep_params.values())
    combos = list(itertools.product(*param_values))

    configs = []
    for combo in combos:
        for run_i in range(batch.num_runs):
            cfg = copy.deepcopy(batch.base_config)
            cfg.seed = run_i

            # Apply swept parameters
            for name, value in zip(param_names, combo):
                # Support nested params like "battle_config.charge_close_range_cm"
                parts = name.split(".")
                obj = cfg
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)

            configs.append(cfg)

    return configs


def _result_to_row(config: SimConfig, result: SimResult) -> dict:
    """Convert a config + result pair to a flat dict for CSV."""
    row = {
        "seed": result.seed,
        "enemy_ai": config.enemy_ai_type,
        "strategy": config.battle_config.strategy,
        "outcome": result.outcome,
        "duration_s": round(result.duration_s, 2),
        "total_pins": result.total_pins,
        "time_to_first_pin": (
            round(result.time_to_first_pin, 2)
            if result.time_to_first_pin is not None else None
        ),
        "pit_events": result.pit_events,
        "avg_distance_cm": round(result.avg_distance_cm, 1),
        "collision_count": result.collision_count,
        "match_duration_s": config.match_duration_s,
        "charge_close_range_cm": config.battle_config.charge_close_range_cm,
        "pin_duration_s": config.battle_config.pin_duration_s,
        "wall_threshold_cm": config.battle_config.wall_threshold_cm,
        "safe_side": config.battle_config.safe_side,
    }
    # Add state times
    for state, t in result.states_visited.items():
        row[f"state_{state}_s"] = round(t, 2)
    return row


def run_batch(batch: BatchConfig) -> list[dict]:
    """Run a batch of simulations, optionally in parallel."""
    configs = _expand_sweeps(batch)
    total = len(configs)
    workers = batch.num_workers or os.cpu_count() or 4

    print(f"Running {total} simulations on {workers} workers...")
    t0 = time.perf_counter()

    if workers == 1:
        results = [run_single(c) for c in configs]
    else:
        with multiprocessing.Pool(workers) as pool:
            results = pool.map(run_single, configs)

    elapsed = time.perf_counter() - t0
    print(f"Completed in {elapsed:.1f}s ({elapsed/total*1000:.0f}ms/match)")

    # Build rows
    rows = [_result_to_row(c, r) for c, r in zip(configs, results)]

    # Save CSV
    if batch.output_path and rows:
        fieldnames = list(rows[0].keys())
        # Collect all possible state columns
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())
        fieldnames = sorted(all_keys)

        with open(batch.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to {batch.output_path}")

    # Print summary
    _print_summary(rows)

    return rows


def _print_summary(rows: list[dict]) -> None:
    """Print aggregate statistics."""
    total = len(rows)
    if total == 0:
        print("No results.")
        return

    wins = sum(1 for r in rows if "win" in r["outcome"])
    losses = sum(1 for r in rows if "loss" in r["outcome"])
    timeouts = sum(1 for r in rows if r["outcome"] == "timeout")
    pins = sum(r["total_pins"] for r in rows)

    pin_times = [r["time_to_first_pin"] for r in rows if r["time_to_first_pin"] is not None]
    avg_pin_time = sum(pin_times) / len(pin_times) if pin_times else None
    avg_duration = sum(r["duration_s"] for r in rows) / total
    avg_distance = sum(r["avg_distance_cm"] for r in rows) / total

    print(f"\n{'='*50}")
    print(f"  Batch Results ({total} runs)")
    print(f"{'='*50}")
    print(f"  Win rate:          {wins/total:.1%}")
    print(f"  Loss rate:         {losses/total:.1%}")
    print(f"  Timeout rate:      {timeouts/total:.1%}")
    print(f"  Total pins:        {pins}")
    if avg_pin_time is not None:
        print(f"  Avg time to pin:   {avg_pin_time:.1f}s")
    print(f"  Avg match length:  {avg_duration:.1f}s")
    print(f"  Avg distance:      {avg_distance:.0f}cm")

    # By enemy type
    ai_types = set(r["enemy_ai"] for r in rows)
    if len(ai_types) > 1:
        print(f"\n  By enemy type:")
        for ai in sorted(ai_types):
            ai_rows = [r for r in rows if r["enemy_ai"] == ai]
            ai_wins = sum(1 for r in ai_rows if "win" in r["outcome"])
            ai_pin_times = [r["time_to_first_pin"] for r in ai_rows if r["time_to_first_pin"] is not None]
            avg_pt = sum(ai_pin_times) / len(ai_pin_times) if ai_pin_times else None
            pt_str = f"avg_pin={avg_pt:.1f}s" if avg_pt else "no pins"
            print(f"    {ai:15s} win={ai_wins/len(ai_rows):.0%}  {pt_str}")

    print(f"{'='*50}\n")
