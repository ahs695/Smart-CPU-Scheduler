# backend/experiments/plot_results.py
"""
Phase 6 — Result Visualisation

Generates publication-quality matplotlib bar charts from the experiment CSVs.

Charts produced:
  1. Waiting time comparison  — all algorithms per workload type
  2. Turnaround time      — same layout
  3. Fairness index       — same layout
  4. Throughput           — same layout

Output directory: results/plots/

Usage:
    python -m backend.experiments.plot_results
    python -m backend.experiments.plot_results --metric avg_waiting_time
"""

import os
import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

# Lazy-import matplotlib so the module can be imported even without it
try:
    import matplotlib
    matplotlib.use("Agg")          # Non-interactive backend — safe in all envs
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# File paths
BASELINES_CSV = "results/baselines.csv"
RL_CSV        = "results/rl_results.csv"
PLOTS_DIR     = "results/plots"

# Metric → display label mapping
METRICS = {
    "avg_waiting_time":    "Average Waiting Time (units)",
    "avg_turnaround_time": "Average Turnaround Time (units)",
    "fairness_index":      "Jain's Fairness Index (0–1, higher=better)",
    "throughput":          "Throughput (processes / time unit)",
}

WORKLOAD_TYPES = ["cpu_heavy", "io_heavy", "mixed", "random"]

# Consistent colour palette — ordered: FCFS, RR, MLFQ, PPO, Hybrid
PALETTE = {
    "FCFS":   "#6c7a89",   # slate grey
    "RR":     "#3d9970",   # teal green
    "MLFQ":   "#2980b9",   # blue
    "PPO":    "#e67e22",   # amber
    "Hybrid": "#8e44ad",   # purple
}

ALGO_ORDER = ["FCFS", "RR", "MLFQ", "PPO", "Hybrid"]


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def load_csv(filepath: str) -> List[Dict]:
    if not os.path.exists(filepath):
        return []
    rows = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows


def _group(rows: List[Dict], algo: str, workload: str, metric: str) -> np.ndarray:
    return np.array(
        [float(r[metric]) for r in rows
         if r["algorithm"] == algo and r["workload_type"] == workload],
        dtype=float
    )


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_metric(
    all_rows: List[Dict],
    metric: str,
    output_dir: str = PLOTS_DIR,
):
    """
    Produce one grouped bar chart per metric showing all algorithms
    across all workload types in a single figure.

    Layout: 4 groups (workloads) × N bars (algorithms).
    Error bars show ±1 standard deviation.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  ⚠ matplotlib not installed. Skipping plots.")
        return

    label = METRICS.get(metric, metric)

    # Determine which algorithms are actually present in the data
    present_algos = [
        a for a in ALGO_ORDER
        if any(r["algorithm"] == a for r in all_rows)
    ]

    n_workloads = len(WORKLOAD_TYPES)
    n_algos     = len(present_algos)

    bar_width  = 0.75 / n_algos
    group_gap  = 1.0
    x_centers  = np.arange(n_workloads) * group_gap

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for i, algo in enumerate(present_algos):
        means, stds = [], []
        for workload in WORKLOAD_TYPES:
            vals = _group(all_rows, algo, workload, metric)
            means.append(np.mean(vals) if len(vals) else 0.0)
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

        offsets = x_centers + (i - n_algos / 2 + 0.5) * bar_width
        color   = PALETTE.get(algo, "#ffffff")

        ax.bar(
            offsets, means,
            width=bar_width * 0.9,
            color=color, alpha=0.88,
            label=algo,
            yerr=stds,
            capsize=3,
            error_kw={"ecolor": "white", "alpha": 0.6, "linewidth": 1.2},
        )

    # Axes styling
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [w.replace("_", "\n") for w in WORKLOAD_TYPES],
        color="white", fontsize=11
    )
    ax.tick_params(axis="y", colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color("white")
    ax.set_ylabel(label, color="white", fontsize=12)
    ax.set_xlabel("Workload Type", color="white", fontsize=12)
    ax.set_title(
        f"Scheduler Comparison — {label}",
        color="white", fontsize=14, fontweight="bold", pad=16
    )
    ax.grid(axis="y", color="#334", linestyle="--", linewidth=0.8, alpha=0.6)

    # Legend
    legend = ax.legend(
        facecolor="#0f3460", edgecolor="#444",
        labelcolor="white", fontsize=10,
        loc="upper right"
    )

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{metric}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved → {out_path}")


def plot_all(
    all_rows: List[Dict],
    metrics: Optional[List[str]] = None,
    output_dir: str = PLOTS_DIR,
):
    """Generate one chart per metric."""
    if not MATPLOTLIB_AVAILABLE:
        print("  ⚠ matplotlib is not installed — skipping all plots.")
        print("       Install with:  pip install matplotlib")
        return

    target_metrics = metrics or list(METRICS.keys())
    for m in target_metrics:
        plot_metric(all_rows, m, output_dir)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison charts from experiment results."
    )
    parser.add_argument(
        "--baselines", type=str, default=BASELINES_CSV,
        help=f"Path to baselines CSV (default: {BASELINES_CSV})"
    )
    parser.add_argument(
        "--rl", type=str, default=RL_CSV,
        help=f"Path to RL results CSV (default: {RL_CSV})"
    )
    parser.add_argument(
        "--metric", type=str, default=None,
        choices=list(METRICS.keys()),
        help="Plot only a specific metric (default: all)"
    )
    parser.add_argument(
        "--output", type=str, default=PLOTS_DIR,
        help=f"Output directory for plots (default: {PLOTS_DIR})"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  Phase 6 — Result Visualisation")
    print("=" * 55)

    baseline_rows = load_csv(args.baselines)
    rl_rows       = load_csv(args.rl)
    all_rows      = baseline_rows + rl_rows

    print(f"  Loaded {len(baseline_rows)} baseline rows + {len(rl_rows)} RL rows")
    print(f"  Output dir: {args.output}")
    print()

    if not all_rows:
        print("  ❌ No data found. Run the experiment scripts first:")
        print("       python -m backend.experiments.run_baselines")
        print("       python -m backend.experiments.run_rl")
        return

    metrics = [args.metric] if args.metric else None
    plot_all(all_rows, metrics=metrics, output_dir=args.output)

    print(f"\n✅ All charts saved to {args.output}/")


if __name__ == "__main__":
    main()
