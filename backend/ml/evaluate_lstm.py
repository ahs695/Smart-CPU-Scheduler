# backend/ml/evaluate_lstm.py
"""
Evaluate the LSTM burst predictor on held-out workloads, per burst pattern,
against two naive predictors:
  - last burst:  the previous job's burst
  - window mean: mean of the previous SEQ_LENGTH bursts

    python -m backend.ml.evaluate_lstm
"""

import torch

from backend.ml.dataset import DatasetBuilder
from backend.ml.lstm_model import MAX_BURST, SEQ_LENGTH, load_lstm
from backend.paths import LSTM_MODEL_PATH

TEST_SEED = 3_000
GLOBAL_MEAN_BURST = 15.0  # fallback prediction for a job with no history


def evaluate(model, num_workloads: int = 1000, seed: int = TEST_SEED):
    """Returns {pattern: {"n", "lstm_mae", "last_mae", "mean_mae"}} plus an "all" row (MAE in ticks)."""
    groups = DatasetBuilder.generate(num_workloads, seed)
    rows, totals = {}, {"n": 0, "lstm": 0.0, "last": 0.0, "mean": 0.0}

    for pattern, (X, y, histories) in groups.items():
        if not X:
            continue
        with torch.no_grad():
            preds = model(torch.tensor(X, dtype=torch.float32)).clamp(0, 1).numpy() * MAX_BURST
        actual = [t * MAX_BURST for t in y]
        last = [h[-1] if h else GLOBAL_MEAN_BURST for h in histories]
        mean = [sum(h[-SEQ_LENGTH:]) / len(h[-SEQ_LENGTH:]) if h else GLOBAL_MEAN_BURST for h in histories]

        err = lambda p: sum(abs(a - b) for a, b in zip(actual, p))
        e_lstm, e_last, e_mean = err(preds), err(last), err(mean)
        n = len(actual)
        rows[pattern] = {"n": n, "lstm_mae": e_lstm / n, "last_mae": e_last / n, "mean_mae": e_mean / n}
        totals["n"] += n
        totals["lstm"] += e_lstm
        totals["last"] += e_last
        totals["mean"] += e_mean

    n = totals["n"]
    rows["all"] = {"n": n, "lstm_mae": totals["lstm"] / n, "last_mae": totals["last"] / n, "mean_mae": totals["mean"] / n}
    return rows


def main():
    model = load_lstm(LSTM_MODEL_PATH)
    if model is None:
        raise SystemExit(f"No LSTM checkpoint at {LSTM_MODEL_PATH}")
    rows = evaluate(model)
    print(f"\n{'pattern':<10} {'n':>7} {'LSTM':>8} {'last':>8} {'mean':>8}   (MAE, ticks)")
    for pattern, r in rows.items():
        print(f"{pattern:<10} {r['n']:>7} {r['lstm_mae']:>8.2f} {r['last_mae']:>8.2f} {r['mean_mae']:>8.2f}")


if __name__ == "__main__":
    main()
