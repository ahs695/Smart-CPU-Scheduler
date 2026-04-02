# backend/experiments/run_rl.py
"""
Phase 6 — RL Scheduler Experiments

Evaluates PPO and Hybrid (PPO + LSTM) on the EXACT same seeded workloads
used by run_baselines.py, enabling apples-to-apples statistical comparison.

Usage:
    python -m backend.experiments.run_rl
    python -m backend.experiments.run_rl --runs 20 --mode both
    python -m backend.experiments.run_rl --mode ppo
    python -m backend.experiments.run_rl --mode hybrid
"""

import os
import argparse
import copy
import csv
from typing import Dict, List, Optional

import torch
from stable_baselines3 import PPO

from backend.experiments.workload_factory import generate_workload, WORKLOAD_TYPES
from backend.ml.lstm_model import BurstPredictorLSTM
from backend.rl.env import SchedulingEnv
from backend.hybrid.hybrid_scheduler import HybridSchedulingEnv
from backend.simulator.metrics import MetricsEngine

# Model paths
PPO_MODEL_PATH    = "models/ppo_scheduler_ppo.zip"
HYBRID_MODEL_PATH = "models/ppo_scheduler_hybrid.zip"
HYBRID_LEGACY_PATH = "models/ppo_scheduler.zip"
LSTM_MODEL_PATH   = "models/lstm_model.pt"

# Output
RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "rl_results.csv")

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

# RL env config — must match training settings
MAX_QUEUE_SIZE = 50
MAX_STEPS      = 100_000


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

def load_ppo_model(path: str) -> Optional[object]:
    if os.path.exists(path):
        print(f"  ✓ Loaded PPO model from {path}")
        return PPO.load(path)
    print(f"  ⚠ PPO model not found: {path}")
    return None


def load_lstm(path: str) -> BurstPredictorLSTM:
    lstm = BurstPredictorLSTM()
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            lstm.load_state_dict(checkpoint["model_state_dict"])
        else:
            lstm.load_state_dict(checkpoint)
        print(f"  ✓ Loaded LSTM model from {path}")
    else:
        print(f"  ⚠ LSTM model not found: {path}  (using untrained weights)")
    lstm.eval()
    return lstm


# ------------------------------------------------------------
# Single RL evaluation
# ------------------------------------------------------------

def run_rl_experiment(
    model,
    processes: List,
    mode: str,
    lstm: Optional[BurstPredictorLSTM],
    num_cores: int = 2,
) -> Optional[Dict]:
    """
    Run one RL episode on a deep-copied process list.

    Args:
        model:     Loaded SB3 PPO model.
        processes: Process list (will be deep-copied internally).
        mode:      'ppo' or 'hybrid'.
        lstm:      BurstPredictorLSTM instance (used only for hybrid).
        num_cores: Number of simulated CPU cores.

    Returns:
        Metrics dict, or None if the episode failed/was truncated.
    """
    procs_copy = copy.deepcopy(processes)

    if mode == "hybrid":
        env = HybridSchedulingEnv(
            processes=procs_copy,
            num_cores=num_cores,
            max_queue_size=MAX_QUEUE_SIZE,
            max_steps=MAX_STEPS,
            lstm_model=lstm,
            use_predictions=True,
        )
    else:
        env = SchedulingEnv(
            processes=procs_copy,
            num_cores=num_cores,
            max_queue_size=MAX_QUEUE_SIZE,
            max_steps=MAX_STEPS,
        )

    state, _ = env.reset()
    steps = 0

    while True:
        action, _ = model.predict(state, deterministic=True)
        state, _, terminated, truncated, _ = env.step(action)
        steps += 1

        if terminated:
            break

        if truncated:
            # Episode hit max_steps — some processes may be incomplete
            print(
                f"    ⚠ [{mode.upper()}] Episode truncated at {steps} steps "
                f"(some processes may be incomplete)"
            )
            break

    sim = env.simulator

    metrics = MetricsEngine.summarize(
        processes=sim.processes,
        cores=sim.cores,
        total_time=sim.time,
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


# ------------------------------------------------------------
# Main experiment loop
# ------------------------------------------------------------

def run_all_rl(
    n_runs: int = 10,
    n_processes: int = 30,
    num_cores: int = 2,
    run_ppo: bool = True,
    run_hybrid: bool = True,
) -> List[Dict]:
    """
    Evaluate enabled RL schedulers on all seeded workloads.

    Uses the same (workload_type, run_id → seed) mapping as run_baselines.py
    so results can be directly merged and compared.
    """
    # --- Load models once
    ppo_model    = load_ppo_model(PPO_MODEL_PATH) if run_ppo else None
    hybrid_model = None
    if run_hybrid:
        hybrid_model = load_ppo_model(HYBRID_MODEL_PATH)
        if hybrid_model is None:
            hybrid_model = load_ppo_model(HYBRID_LEGACY_PATH)

    lstm = load_lstm(LSTM_MODEL_PATH)
    print()

    # --- Decide which algorithms to actually run
    agents = {}
    if run_ppo and ppo_model:
        agents["PPO"] = ("ppo", ppo_model)
    elif run_ppo:
        print("  ⚠ Skipping PPO — model not available.")

    if run_hybrid and hybrid_model:
        agents["Hybrid"] = ("hybrid", hybrid_model)
    elif run_hybrid:
        print("  ⚠ Skipping Hybrid — model not available.")

    if not agents:
        print("  ❌ No RL models available. Exiting.")
        return []

    rows = []
    total_experiments = n_runs * len(WORKLOAD_TYPES)
    done = 0

    for run_id in range(n_runs):
        for workload_type in WORKLOAD_TYPES:

            # --- Same seeded workload as baselines
            processes = generate_workload(
                workload_type=workload_type,
                n_processes=n_processes,
                seed=run_id,
            )

            for algo_name, (mode, model) in agents.items():
                try:
                    metrics = run_rl_experiment(
                        model=model,
                        processes=processes,
                        mode=mode,
                        lstm=lstm if mode == "hybrid" else None,
                        num_cores=num_cores,
                    )

                    if metrics is not None:
                        row = {
                            "algorithm":     algo_name,
                            "workload_type": workload_type,
                            "run_id":        run_id,
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
# Persistence
# ------------------------------------------------------------

def save_results(rows: List[Dict], filepath: str = OUTPUT_FILE):
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
        description="Evaluate RL schedulers (PPO / Hybrid) on seeded workloads."
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
        help="Number of CPU cores (default: 2)"
    )
    parser.add_argument(
        "--mode", choices=["ppo", "hybrid", "both"], default="both",
        help="Which RL agent(s) to evaluate (default: both)"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_FILE,
        help=f"Output CSV path (default: {OUTPUT_FILE})"
    )
    args = parser.parse_args()

    run_ppo    = args.mode in ("ppo", "both")
    run_hybrid = args.mode in ("hybrid", "both")

    print("=" * 60)
    print("  Phase 6 — RL Scheduler Evaluation")
    print("=" * 60)
    active = []
    if run_ppo:    active.append("PPO")
    if run_hybrid: active.append("Hybrid (PPO+LSTM)")
    print(f"  Agents     : {', '.join(active)}")
    print(f"  Workloads  : {', '.join(WORKLOAD_TYPES)}")
    print(f"  Runs       : {args.runs}  (seed = run_id)")
    print(f"  Processes  : {args.processes}")
    print(f"  Cores      : {args.cores}")
    print(f"  Output     : {args.output}")
    print("=" * 60)
    print()

    rows = run_all_rl(
        n_runs=args.runs,
        n_processes=args.processes,
        num_cores=args.cores,
        run_ppo=run_ppo,
        run_hybrid=run_hybrid,
    )

    if rows:
        save_results(rows, filepath=args.output)

    n_agents = (1 if run_ppo else 0) + (1 if run_hybrid else 0)
    expected = args.runs * len(WORKLOAD_TYPES) * n_agents
    print(f"\n  Expected rows : {expected}")
    print(f"  Actual rows   : {len(rows)}")
    if len(rows) < expected:
        print("  ⚠ Some experiments failed or model files were missing.")


if __name__ == "__main__":
    main()
