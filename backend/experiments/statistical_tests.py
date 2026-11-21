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
from scipy.stats import ttest_rel

from backend.paths import results_path

# Default file paths
BASELINES_CSV = results_path("baselines.csv")
RL_CSV        = results_path("rl_results.csv")
SUMMARY_CSV   = results_path("summary.csv")

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

# Pairwise comparisons: (learned scheduler, reference). SRTF knows true
# bursts, so "Hybrid vs SRTF" measures the remaining gap to the oracle.
COMPARISONS = [
    ("PPO",    "FCFS"),
    ("PPO",    "RR"),
    ("PPO",    "MLFQ"),
    ("Hybrid", "PPO"),
    ("Hybrid", "FCFS"),
    ("Hybrid", "RR"),
    ("Hybrid", "MLFQ"),
    ("Hybrid", "SRTF"),
]

from backend.experiments.workload_factory import WORKLOAD_TYPES


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


def _by_run(rows: List[Dict], algorithm: str, workload: str, metric: str, cores=None) -> Dict[float, float]:
    """{run_id: value} for one (algorithm, workload[, num_cores]) slice."""
    return {
        row["run_id"]: float(row[metric])
        for row in rows
        if row["algorithm"] == algorithm and row["workload_type"] == workload
        and (cores is None or row.get("num_cores") == cores)
    }


def _group_by(rows: List[Dict], algorithm: str, workload: str, metric: str, cores=None) -> np.ndarray:
    """1-D array of metric values for a specific (algorithm, workload[, num_cores]) slice."""
    return np.array(list(_by_run(rows, algorithm, workload, metric, cores).values()), dtype=float)


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
    cores=None,
) -> Dict:
    """
    Paired t-test between algo_a and algo_b on one metric for a workload type
    (and core count). Both algorithms ran on the same seeded workloads, so runs
    are paired by run_id.

    Returns a result dict with means, stds, t-statistic, p-value,
    and a significance flag.
    """
    a_map = _by_run(all_rows, algo_a, workload, metric, cores)
    b_map = _by_run(all_rows, algo_b, workload, metric, cores)
    common = sorted(set(a_map) & set(b_map))
    a_vals = np.array([a_map[k] for k in common], dtype=float)
    b_vals = np.array([b_map[k] for k in common], dtype=float)

    if len(common) < 2:
        return {
            "algo_a": algo_a, "algo_b": algo_b, "num_cores": cores,
            "workload": workload, "metric": metric,
            "mean_a": None, "std_a": None,
            "mean_b": None, "std_b": None,
            "t_stat": None, "p_value": None,
            "significant": None,
            "n_a": len(a_vals), "n_b": len(b_vals),
            "error": "Insufficient data (< 2 samples per group)",
        }

    if np.allclose(a_vals, b_vals):
        t_stat, p_value = 0.0, 1.0      # identical results: no difference
    else:
        t_stat, p_value = ttest_rel(a_vals, b_vals)

    return {
        "algo_a":      algo_a,
        "algo_b":      algo_b,
        "num_cores":   cores,
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
        key = (r["algo_a"], r["algo_b"], r["workload"], r.get("num_cores"))
        grouped[key].append(r)

    for (algo_a, algo_b, workload, cores) in grouped:
        print()
        print("─" * 62)
        print(f"  Algorithm Comparison: {algo_a} vs {algo_b}  [{workload}, {cores} cores]")
        print("─" * 62)

        for r in grouped[(algo_a, algo_b, workload, cores)]:
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
                f"  [{r['workload']:<10} {r.get('num_cores')}c]"
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
        "algo_a", "algo_b", "num_cores", "workload", "metric",
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
    cores_list = sorted(set(r["num_cores"] for r in all_rows if "num_cores" in r)) or [None]
    metrics   = ["avg_waiting_time", "avg_turnaround_time", "fairness_index"]

    print()
    print("=" * 62)
    print("  AGGREGATE RESULTS TABLE")
    print("=" * 62)

    for cores in cores_list:
        for workload in workloads:
            print(f"\n  Workload: {workload}   Cores: {cores}")
            header = f"  {'Algorithm':<12}" + "".join(f"  {METRIC_LABELS[m]:<22}" for m in metrics)
            print(header)
            print("  " + "-" * (len(header) - 2))

            for algo in algos:
                row_str = f"  {algo:<12}"
                for m in metrics:
                    vals = _group_by(all_rows, algo, workload, m, cores)
                    if len(vals) == 0:
                        row_str += f"  {'N/A':<22}"
                    else:
                        row_str += f"  {np.mean(vals):.3f} ± {np.std(vals, ddof=1) if len(vals) > 1 else 0:.3f}        "
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
    print("  Phase 6 — Statistical Validation (paired t-test)")
    print("=" * 62)
    print(f"  Baselines CSV : {args.baselines}")
    print(f"  RL CSV        : {args.rl}")
    print(f"  α (alpha)     : {args.alpha}")
    print(f"  Comparisons   : {len(COMPARISONS)} pairs × {len(WORKLOAD_TYPES)} workloads × cores × {len(TEST_METRICS)} metrics (paired t-test)")
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
    cores_list = sorted(set(r["num_cores"] for r in all_rows if "num_cores" in r)) or [None]
    for cores in cores_list:
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
                        cores=cores,
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
