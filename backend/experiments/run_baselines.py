# backend/experiments/run_baselines.py
"""
Phase 6 — Classical Scheduler Baseline Experiments

Runs FCFS, RR, and MLFQ across all workload types for n_runs seeds.
Each (run_id, workload_type) pair uses the same seed everywhere so
results are directly comparable to run_rl.py output.

Usage:
    python -m backend.experiments.run_baselines
    python -m backend.experiments.run_baselines --runs 20 --processes 30
"""

import os
import argparse
import copy
import csv
from typing import Dict, List

from backend.experiments.workload_factory import generate_workload, WORKLOAD_TYPES
from backend.simulator.multi_core_simulator import MultiCoreSimulator
from backend.simulator.metrics import MetricsEngine

# Classical schedulers
from backend.simulator.traditional.fcfs import FCFSScheduler
from backend.simulator.traditional.round_robin import RoundRobinScheduler
from backend.simulator.traditional.mlfq import MLFQScheduler

# Output path
RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "baselines.csv")

CSV_COLUMNS = [
    "algorithm",
    "workload_type",
    "run_id",
    "avg_waiting_time",
    "avg_turnaround_time",
    "avg_response_time",
    "throughput",
    "cpu_utilization",
    "fairness_index",
    "context_switches",
]


# ------------------------------------------------------------
# Core helpers
# ------------------------------------------------------------

def _make_schedulers() -> Dict[str, object]:
    """
    Return fresh scheduler instances for a single run.
    Instantiated fresh each run to clear any internal state (e.g. MLFQ queues).
    """
    return {
        "FCFS": FCFSScheduler(),
        "RR":   RoundRobinScheduler(quantum=2),
        "MLFQ": MLFQScheduler(),
    }


def run_single_experiment(
    scheduler,
    processes: List,
    num_cores: int = 2
) -> Dict:
    """
    Run one classical scheduler on a given process list.

    Processes are deep-copied before the run so the originals stay pristine
    for the next scheduler in the same run.

    Args:
        scheduler:  An instantiated classical scheduler (FCFS / RR / MLFQ).
        processes:  List of Process objects (will be deep-copied internally).
        num_cores:  Number of CPU cores to simulate.

    Returns:
        Dict of metric keys matching CSV_COLUMNS (minus algorithm/workload/run).
    """
    procs_copy = copy.deepcopy(processes)

    sim = MultiCoreSimulator(
        processes=procs_copy,
        scheduler=scheduler,
        num_cores=num_cores,
    )

    result = sim.run()

    metrics = MetricsEngine.summarize(
        processes=result["processes"],
        cores=result["cores"],
        total_time=result["total_time"],
    )

    return {
        "avg_waiting_time":    metrics["avg_waiting_time"],
        "avg_turnaround_time": metrics["avg_turnaround_time"],
        "avg_response_time":   metrics["avg_response_time"],
        "throughput":          metrics["throughput"],
        "cpu_utilization":     metrics["cpu_utilization"],
        "fairness_index":      metrics["fairness_index"],
        "context_switches":    metrics["context_switches"],
    }


def run_all_baselines(
    n_runs: int = 10,
    n_processes: int = 30,
    num_cores: int = 2
) -> List[Dict]:
    """
    Main experiment loop.

    For each (run_id, workload_type):
      1. Generate one seeded workload (seed = run_id).
      2. Run every classical scheduler on a deep copy of that workload.
      3. Record row.

    This guarantees identical inputs across all algorithms and workload types.

    Returns:
        List of row dicts ready for CSV serialisation.
    """
    rows = []

    total_experiments = n_runs * len(WORKLOAD_TYPES)
    done = 0

    for run_id in range(n_runs):
        for workload_type in WORKLOAD_TYPES:

            # --- Generate ONE seeded workload for this (run_id, workload_type)
            processes = generate_workload(
                workload_type=workload_type,
                n_processes=n_processes,
                seed=run_id
            )

            # --- Run each scheduler on a copy of *the same* workload
            schedulers = _make_schedulers()

            for algo_name, scheduler in schedulers.items():
                try:
                    metrics = run_single_experiment(
                        scheduler=scheduler,
                        processes=processes,
                        num_cores=num_cores,
                    )

                    row = {
                        "algorithm":    algo_name,
                        "workload_type": workload_type,
                        "run_id":       run_id,
                        **metrics,
                    }
                    rows.append(row)

                except Exception as e:
                    print(
                        f"  ⚠ [{algo_name}] run_id={run_id} "
                        f"workload={workload_type} FAILED: {e}"
                    )

            done += 1
            pct = (done / total_experiments) * 100
            print(
                f"  ✓ run_id={run_id:>2}  workload={workload_type:<10}  "
                f"({pct:.0f}% complete)"
            )

    return rows


# ------------------------------------------------------------
# Persistence helpers
# ------------------------------------------------------------

def save_results(rows: List[Dict], filepath: str = OUTPUT_FILE):
    """
    Save experiment rows to a CSV file.

    Creates the parent directory if it doesn't exist.
    Overwrites any previous file at the same path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Results saved → {filepath}  ({len(rows)} rows)")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run classical scheduler baseline experiments."
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of independent runs (seeds) per workload type (default: 10)"
    )
    parser.add_argument(
        "--processes", type=int, default=30,
        help="Number of processes per workload (default: 30)"
    )
    parser.add_argument(
        "--cores", type=int, default=2,
        help="Number of CPU cores to simulate (default: 2)"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_FILE,
        help=f"Output CSV path (default: {OUTPUT_FILE})"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 6 — Classical Scheduler Baselines")
    print("=" * 60)
    print(f"  Schedulers : FCFS, RR (q=2), MLFQ")
    print(f"  Workloads  : {', '.join(WORKLOAD_TYPES)}")
    print(f"  Runs       : {args.runs}  (seed = run_id)")
    print(f"  Processes  : {args.processes}")
    print(f"  Cores      : {args.cores}")
    print(f"  Output     : {args.output}")
    print("=" * 60)
    print()

    rows = run_all_baselines(
        n_runs=args.runs,
        n_processes=args.processes,
        num_cores=args.cores,
    )

    save_results(rows, filepath=args.output)

    expected = args.runs * len(WORKLOAD_TYPES) * 3   # 3 classical schedulers
    print(f"\n  Expected rows : {expected}")
    print(f"  Actual rows   : {len(rows)}")
    if len(rows) < expected:
        print("  ⚠ Some experiments failed — check logs above.")


if __name__ == "__main__":
    main()
