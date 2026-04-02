# backend/experiments/statistical_tests.py
"""
Phase 6 — Statistical Validation

Loads baselines.csv and rl_results.csv, then performs pairwise
two-sample t-tests (scipy.stats.ttest_ind) to determine whether
the RL schedulers show statistically significant improvements over
classical baselines.

Usage:
    python -m backend.experiments.statistical_tests
    python -m backend.experiments.statistical_tests --alpha 0.01
"""

import os
import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import ttest_ind

# Default file paths
BASELINES_CSV = "results/baselines.csv"
RL_CSV        = "results/rl_results.csv"
SUMMARY_CSV   = "results/summary.csv"

# Which metrics to run t-tests on
TEST_METRICS = [
    "avg_waiting_time",
    "avg_turnaround_time",
    "fairness_index",
]

# Friendly display names for metrics
METRIC_LABELS = {
    "avg_waiting_time":    "Waiting Time",
    "avg_turnaround_time": "Turnaround Time",
    "fairness_index":      "Fairness Index",
}

# Pairwise comparisons: (RL algo, classical baseline)
COMPARISONS = [
    ("PPO",    "RR"),
    ("PPO",    "MLFQ"),
    ("Hybrid", "RR"),
    ("Hybrid", "MLFQ"),
]

# Workload labels (matches workload_factory.py)
WORKLOAD_TYPES = ["cpu_heavy", "io_heavy", "mixed", "random"]


# ------------------------------------------------------------
# CSV Loading
# ------------------------------------------------------------

def load_csv(filepath: str) -> List[Dict]:
    """
    Load a results CSV into a list of row dicts.
    Numeric columns are cast to float automatically.
    Returns an empty list (with a printed warning) if the file is missing.
    """
    if not os.path.exists(filepath):
        print(f"  ⚠ File not found (skipping): {filepath}")
        return []

    rows = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    pass   # Keep as string (algorithm, workload_type)
            rows.append(row)
    return rows


def _group_by(rows: List[Dict], algorithm: str, workload: str, metric: str) -> np.ndarray:
    """
    Extract a 1-D array of metric values for a specific (algorithm, workload) slice.
    """
    values = [
        float(row[metric])
        for row in rows
        if row["algorithm"] == algorithm and row["workload_type"] == workload
    ]
    return np.array(values, dtype=float)


# ------------------------------------------------------------
# t-test runner
# ------------------------------------------------------------

def run_comparison(
    all_rows: List[Dict],
    algo_a: str,
    algo_b: str,
    workload: str,
    metric: str,
    alpha: float,
) -> Dict:
    """
    Run an independent two-sample t-test between algo_a and algo_b
    on a single metric for a given workload type.

    Returns a result dict with means, stds, t-statistic, p-value,
    and a significance flag.
    """
    a_vals = _group_by(all_rows, algo_a, workload, metric)
    b_vals = _group_by(all_rows, algo_b, workload, metric)

    if len(a_vals) < 2 or len(b_vals) < 2:
        return {
            "algo_a": algo_a, "algo_b": algo_b,
            "workload": workload, "metric": metric,
            "mean_a": None, "std_a": None,
            "mean_b": None, "std_b": None,
            "t_stat": None, "p_value": None,
            "significant": None,
            "n_a": len(a_vals), "n_b": len(b_vals),
            "error": "Insufficient data (< 2 samples per group)",
        }

    t_stat, p_value = ttest_ind(a_vals, b_vals, equal_var=False)  # Welch's t-test

    return {
        "algo_a":      algo_a,
        "algo_b":      algo_b,
        "workload":    workload,
        "metric":      metric,
        "mean_a":      float(np.mean(a_vals)),
        "std_a":       float(np.std(a_vals, ddof=1)),
        "mean_b":      float(np.mean(b_vals)),
        "std_b":       float(np.std(b_vals, ddof=1)),
        "t_stat":      float(t_stat),
        "p_value":     float(p_value),
        "significant": bool(p_value < alpha),
        "n_a":         len(a_vals),
        "n_b":         len(b_vals),
        "error":       None,
    }


# ------------------------------------------------------------
# Pretty printer
# ------------------------------------------------------------

def _sig_label(p_value: float, alpha: float, significant: bool) -> str:
    if not significant:
        return f"NOT significant (p ≥ {alpha})"
    if p_value < 0.001:
        return "★★★ Highly significant (p < 0.001)"
    if p_value < 0.01:
        return "★★  Significant (p < 0.01)"
    return "★   Significant (p < 0.05)"


def print_results(results: List[Dict], alpha: float):
    """
    Pretty-print all comparison results grouped by (comparison, workload).
    """
    # Group: (algo_a, algo_b, workload) → list of per-metric results
    grouped = defaultdict(list)
    for r in results:
        key = (r["algo_a"], r["algo_b"], r["workload"])
        grouped[key].append(r)

    for (algo_a, algo_b, workload) in grouped:
        print()
        print("─" * 62)
        print(f"  Algorithm Comparison: {algo_a} vs {algo_b}  [{workload}]")
        print("─" * 62)

        for r in grouped[(algo_a, algo_b, workload)]:
            label = METRIC_LABELS.get(r["metric"], r["metric"])

            if r["error"]:
                print(f"\n  {label}:")
                print(f"    ⚠  {r['error']}")
                continue

            print(f"\n  {label}:")
            print(f"    Mean ± Std  ({algo_a:<8}):  {r['mean_a']:.4f} ± {r['std_a']:.4f}   (n={r['n_a']})")
            print(f"    Mean ± Std  ({algo_b:<8}):  {r['mean_b']:.4f} ± {r['std_b']:.4f}   (n={r['n_b']})")
            print(f"    t-statistic : {r['t_stat']:+.4f}")
            print(f"    p-value     : {r['p_value']:.6f}")
            print(f"    Verdict     : {_sig_label(r['p_value'], alpha, r['significant'])}")

    print()
    print("=" * 62)
    print("  SUMMARY — Significant Improvements (p < α)")
    print("=" * 62)

    sig = [r for r in results if r["significant"] and not r["error"]]
    if not sig:
        print("  No statistically significant differences found.")
    else:
        for r in sig:
            label = METRIC_LABELS.get(r["metric"], r["metric"])
            direction = "↓ lower" if r["mean_a"] < r["mean_b"] else "↑ higher"
            print(
                f"  {r['algo_a']:<8} vs {r['algo_b']:<6}"
                f"  [{r['workload']:<10}]"
                f"  {label:<20}"
                f"  p={r['p_value']:.4f}  {direction}"
            )


# ------------------------------------------------------------
# Summary CSV
# ------------------------------------------------------------

def save_summary(results: List[Dict], filepath: str = SUMMARY_CSV):
    """
    Save all t-test results to a structured CSV for further analysis.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fieldnames = [
        "algo_a", "algo_b", "workload", "metric",
        "mean_a", "std_a", "mean_b", "std_b",
        "t_stat", "p_value", "significant", "n_a", "n_b", "error",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Summary saved → {filepath}  ({len(results)} comparisons)")


# ------------------------------------------------------------
# Aggregated per-algorithm summary
# ------------------------------------------------------------

def print_aggregate_summary(all_rows: List[Dict]):
    """
    Print mean ± std for every (algorithm, workload) combination
    across all three tested metrics — useful for the results table
    in a research paper.
    """
    algos     = sorted(set(r["algorithm"]     for r in all_rows))
    workloads = sorted(set(r["workload_type"] for r in all_rows))
    metrics   = ["avg_waiting_time", "avg_turnaround_time", "fairness_index"]

    print()
    print("=" * 62)
    print("  AGGREGATE RESULTS TABLE")
    print("=" * 62)

    for workload in workloads:
        print(f"\n  Workload: {workload}")
        header = f"  {'Algorithm':<12}" + "".join(f"  {METRIC_LABELS[m]:<22}" for m in metrics)
        print(header)
        print("  " + "-" * (len(header) - 2))

        for algo in algos:
            row_str = f"  {algo:<12}"
            for m in metrics:
                vals = _group_by(all_rows, algo, workload, m)
                if len(vals) == 0:
                    row_str += f"  {'N/A':<22}"
                else:
                    row_str += f"  {np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}        "
            print(row_str)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Statistical validation of scheduler comparison results."
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
        "--alpha", type=float, default=0.05,
        help="Significance level α for t-tests (default: 0.05)"
    )
    parser.add_argument(
        "--summary", type=str, default=SUMMARY_CSV,
        help=f"Output summary CSV path (default: {SUMMARY_CSV})"
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="Skip writing the summary CSV"
    )
    args = parser.parse_args()

    print("=" * 62)
    print("  Phase 6 — Statistical Validation (Welch's t-test)")
    print("=" * 62)
    print(f"  Baselines CSV : {args.baselines}")
    print(f"  RL CSV        : {args.rl}")
    print(f"  α (alpha)     : {args.alpha}")
    print(f"  Comparisons   : {len(COMPARISONS)} pairs × {len(WORKLOAD_TYPES)} workloads × {len(TEST_METRICS)} metrics")
    print("=" * 62)

    # --- Load data
    print("\nLoading results...")
    baseline_rows = load_csv(args.baselines)
    rl_rows       = load_csv(args.rl)
    all_rows      = baseline_rows + rl_rows
    print(f"  Baseline rows : {len(baseline_rows)}")
    print(f"  RL rows       : {len(rl_rows)}")
    print(f"  Total rows    : {len(all_rows)}")

    # --- Run all t-tests
    results = []
    for algo_a, algo_b in COMPARISONS:
        for workload in WORKLOAD_TYPES:
            for metric in TEST_METRICS:
                r = run_comparison(
                    all_rows=all_rows,
                    algo_a=algo_a,
                    algo_b=algo_b,
                    workload=workload,
                    metric=metric,
                    alpha=args.alpha,
                )
                results.append(r)

    # --- Print results
    print_results(results, alpha=args.alpha)
    print_aggregate_summary(all_rows)

    # --- Save summary CSV
    if not args.no_summary:
        save_summary(results, filepath=args.summary)


if __name__ == "__main__":
    main()
