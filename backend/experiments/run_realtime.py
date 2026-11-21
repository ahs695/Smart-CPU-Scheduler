# backend/experiments/run_realtime.py
"""
Phase 2 — Real-time scheduler experiments.

Two suites, every scheduler (FCFS, SJF, SRTF, RR, MLFQ, EDF, RMS, PPO, Hybrid):

  aperiodic  the standard seeded workload types with generated deadlines
             (deadline = arrival + ceil(burst × slack)) at three slack levels
  periodic   random implicit-deadline periodic task sets (UUniFast utilisations)
             at several total utilisations, scaled by the core count

PPO and Hybrid were trained to minimise turnaround and ignore deadlines; they
are included as-is to show how a throughput-oriented learned policy fares on
SLA metrics.

Usage:
    python -m backend.experiments.run_realtime [--runs 5] [--cores 1 2 4]
"""

import argparse
import copy
import math
import csv
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from backend.experiments.workload_factory import WORKLOAD_TYPES, generate_workload
from backend.experiments.run_rl import load_lstm, load_ppo_model
from backend.hybrid.hybrid_scheduler import HybridSchedulingEnv
from backend.rl.env import SchedulingEnv
from backend.paths import LSTM_MODEL_PATH, RESULTS_DIR
from backend.simulator.metrics import MetricsEngine
from backend.simulator.multi_core_simulator import MultiCoreSimulator
from backend.simulator.realtime import (
    assign_deadlines, expand_periodic, hyperperiod, random_task_set, utilisation,
)
from backend.simulator.traditional.edf import EDFScheduler
from backend.simulator.traditional.fcfs import FCFSScheduler
from backend.simulator.traditional.mlfq import MLFQScheduler
from backend.simulator.traditional.rms import RMSScheduler
from backend.simulator.traditional.round_robin import RoundRobinScheduler
from backend.simulator.traditional.sjf import SJFScheduler

OUTPUT_FILE = os.path.join(RESULTS_DIR, "realtime.csv")
SLACK_LEVELS = [1.5, 3.0, 6.0]
UTILISATIONS = [0.5, 0.7, 0.9, 1.0]      # per core
PERIODIC_HORIZON_CAP = 400
CSV_COLUMNS = ["suite", "workload", "level", "actual_utilisation", "num_cores", "run_id", "algorithm",
               "deadline_miss_rate", "avg_lateness", "avg_turnaround_time"]

TRADITIONAL = {
    "FCFS": FCFSScheduler, "SJF": SJFScheduler, "SRTF": lambda: SJFScheduler(preemptive=True),
    "RR": lambda: RoundRobinScheduler(quantum=2), "MLFQ": MLFQScheduler,
    "EDF": EDFScheduler, "RMS": RMSScheduler,
}


def periodic_task_set(n_tasks: int, utilisation: float, rng, tolerance: float = 0.03) -> List[Dict]:
    return random_task_set(n_tasks, utilisation, rng, tolerance)


def _metrics(sim_processes, cores, total_time) -> Dict:
    m = MetricsEngine.summarize(sim_processes, cores, total_time)
    return {k: m[k] for k in ("deadline_miss_rate", "avg_lateness", "avg_turnaround_time")}


def run_all(processes, num_cores, rl_models, lstm) -> Dict[str, Dict]:
    out = {}
    for name, make in TRADITIONAL.items():
        sim = MultiCoreSimulator(copy.deepcopy(processes), make(), num_cores)
        sim.run()
        out[name] = _metrics(sim.processes, sim.cores, sim.time)
    for name, (mode, model) in rl_models.items():
        env = (HybridSchedulingEnv(processes=copy.deepcopy(processes), num_cores=num_cores, lstm_model=lstm)
               if mode == "hybrid" else SchedulingEnv(processes=copy.deepcopy(processes), num_cores=num_cores))
        state, _ = env.reset()
        while True:
            action, _ = model.predict(state, deterministic=True)
            state, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        out[name] = _metrics(env.simulator.processes, env.simulator.cores, env.simulator.time)
    return out


def main():
    parser = argparse.ArgumentParser(description="Real-time (deadline) scheduler experiments")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--cores", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output", type=str, default=OUTPUT_FILE)
    args = parser.parse_args()

    lstm = load_lstm(LSTM_MODEL_PATH)
    rows = []
    for cores in args.cores:
        rl_models = {}
        for name, mode in (("PPO", "ppo"), ("Hybrid", "hybrid")):
            model = load_ppo_model(mode, cores)
            if model is not None:
                rl_models[name] = (mode, model)

        # Suite 1: aperiodic workloads with generated deadlines
        for slack in SLACK_LEVELS:
            for wt in WORKLOAD_TYPES:
                for run in range(args.runs):
                    procs = generate_workload(wt, 30, seed=run)
                    assign_deadlines(procs, slack)
                    for algo, m in run_all(procs, cores, rl_models, lstm).items():
                        rows.append({"suite": "aperiodic", "workload": wt, "level": slack,
                                     "actual_utilisation": "", "num_cores": cores,
                                     "run_id": run, "algorithm": algo, **m})
            print(f"  ✓ cores={cores} aperiodic slack={slack}")

        # Suite 2: periodic task sets (total U = per-core U × cores)
        for u in UTILISATIONS:
            for run in range(args.runs):
                rng = np.random.default_rng(10_000 + run * 97 + int(u * 100) + cores)
                tasks = periodic_task_set(int(rng.integers(3, 6)) + 2 * (cores - 1), u * cores, rng,
                                          tolerance=0.03 * cores)
                procs = expand_periodic(tasks, min(2 * hyperperiod(tasks), PERIODIC_HORIZON_CAP))
                actual_u = utilisation(tasks) / cores
                for algo, m in run_all(procs, cores, rl_models, lstm).items():
                    rows.append({"suite": "periodic", "workload": "periodic", "level": u,
                                 "actual_utilisation": round(actual_u, 4), "num_cores": cores,
                                 "run_id": run, "algorithm": algo, **m})
            print(f"  ✓ cores={cores} periodic U={u}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ {len(rows)} rows → {args.output}")

    # Summary: mean deadline miss rate (%) per suite / level / cores
    algos = list(TRADITIONAL) + ["PPO", "Hybrid"]
    table = defaultdict(list)
    for r in rows:
        table[(r["suite"], r["level"], r["num_cores"], r["algorithm"])].append(r["deadline_miss_rate"])
    print("\nMean deadline miss rate (%)")
    print(f"{'suite':<10}{'level':>6}{'cores':>6}  " + "".join(f"{a:>8}" for a in algos))
    for suite, levels in (("aperiodic", SLACK_LEVELS), ("periodic", UTILISATIONS)):
        for level in levels:
            for cores in args.cores:
                vals = [np.mean(table[(suite, level, cores, a)]) * 100 if table[(suite, level, cores, a)] else float("nan")
                        for a in algos]
                print(f"{suite:<10}{level:>6}{cores:>6}  " + "".join(f"{v:8.1f}" for v in vals))


if __name__ == "__main__":
    main()
