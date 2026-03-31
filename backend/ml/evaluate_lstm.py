import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

from backend.ml.dataset import DatasetBuilder
from backend.ml.lstm_model import BurstPredictorLSTM


class Evaluator:
    """
    Evaluates LSTM burst predictor.
    Computes MAE, RMSE, and compares with baseline.
    """

    def __init__(
        self,
        model_path: str,
        seq_length: int = 5,
        dataset_size: int = 1000,
        device: str = None
    ):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print("Using device:", self.device)

        # Load dataset
        self.dataset = DatasetBuilder.build_dataset(
            num_sequences=dataset_size,
            seq_length=seq_length
        )

        # Load model
        self.model = BurstPredictorLSTM().to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle both formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

    # ------------------------------------------------------------

    def evaluate(self):

        mae = 0
        mse = 0

        baseline_mae = 0
        baseline_mse = 0

        predictions = []
        targets = []

        with torch.no_grad():

            for X, y in self.dataset:

                X = X.unsqueeze(0).to(self.device)
                y = y.to(self.device)

                pred = self.model(X).squeeze()

                # LSTM errors
                error = abs(pred - y).item()
                mae += error
                mse += error ** 2

                # Baseline: last value in sequence
                baseline_pred = X[0, -1].item()

                b_error = abs(baseline_pred - y.item())
                baseline_mae += b_error
                baseline_mse += b_error ** 2

                predictions.append(pred.item())
                targets.append(y.item())

        n = len(self.dataset)

        results = {
            "LSTM_MAE": mae / n,
            "LSTM_RMSE": math.sqrt(mse / n),
            "Baseline_MAE": baseline_mae / n,
            "Baseline_RMSE": math.sqrt(baseline_mse / n),
        }

        return results, predictions, targets

    # ------------------------------------------------------------

    def plot(self, predictions, targets, num_points=100):

        plt.figure(figsize=(10, 5))

        plt.plot(targets[:num_points], label="True")
        plt.plot(predictions[:num_points], label="Predicted")

        plt.title("LSTM Burst Prediction vs Ground Truth")
        plt.xlabel("Sample")
        plt.ylabel("Normalized Burst")

        plt.legend()
        plt.grid()

        plt.show()


# ------------------------------------------------------------


def main():

    evaluator = Evaluator(
        model_path="models/lstm_model.pt",  # or specific timestamp model
        dataset_size=1000
    )

    results, preds, targets = evaluator.evaluate()

    print("\nEvaluation Results:")
    print("====================")

    for k, v in results.items():
        print(f"{k}: {v:.5f}")

    evaluator.plot(preds, targets)


if __name__ == "__main__":
    main()