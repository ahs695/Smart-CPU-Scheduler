# backend/ml/train_lstm.py
"""
Train the LSTM burst predictor.

    python -m backend.ml.train_lstm [--workloads 4000] [--epochs 60]

Train / validation sets use disjoint seeds; the checkpoint with the lowest
validation loss is saved to LSTM_MODEL_PATH (early stopping).
"""

import argparse
import copy
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.ml.dataset import DatasetBuilder
from backend.ml.lstm_model import BurstPredictorLSTM, MAX_BURST
from backend.paths import LSTM_MODEL_PATH, MODELS_DIR

TRAIN_SEED = 1_000
VAL_SEED = 2_000


class Trainer:

    def __init__(
        self,
        num_workloads: int = 4000,
        batch_size: int = 256,
        epochs: int = 60,
        learning_rate: float = 1e-3,
        patience: int = 8,
        device: str = "cpu",
    ):
        self.epochs = epochs
        self.patience = patience
        self.device = device

        train_set = DatasetBuilder.build_dataset(num_workloads, seed=TRAIN_SEED)
        val_set = DatasetBuilder.build_dataset(max(200, num_workloads // 5), seed=VAL_SEED)
        print(f"Samples: train={len(train_set)}  val={len(val_set)}")

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=1024)

        self.model = BurstPredictorLSTM().to(device)
        # Huber: robust to the occasional regime switch, which no model can foresee
        self.criterion = nn.SmoothL1Loss(beta=0.05)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=3)

    def _run(self, loader, train: bool):
        self.model.train(train)
        total, mae, n = 0.0, 0.0, 0
        with torch.set_grad_enabled(train):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.model(X)
                loss = self.criterion(preds, y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                total += loss.item() * len(y)
                mae += (preds - y).abs().sum().item()
                n += len(y)
        return total / n, mae / n * MAX_BURST

    def train(self):
        best_loss, best_state, stale = float("inf"), None, 0
        for epoch in range(1, self.epochs + 1):
            train_loss, _ = self._run(self.train_loader, train=True)
            val_loss, val_mae = self._run(self.val_loader, train=False)
            self.scheduler.step(val_loss)
            improved = val_loss < best_loss - 1e-6
            if improved:
                best_loss, best_state, stale = val_loss, copy.deepcopy(self.model.state_dict()), 0
            else:
                stale += 1
            print(f"Epoch {epoch:02d} | train {train_loss:.5f} | val {val_loss:.5f} | "
                  f"val MAE {val_mae:.2f} ticks{'  *' if improved else ''}")
            if stale >= self.patience:
                print(f"Early stopping (no improvement for {self.patience} epochs)")
                break
        self.model.load_state_dict(best_state)
        return best_loss

    def save_model(self, path=LSTM_MODEL_PATH):
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print("\nModel saved to:", path)


def main():
    parser = argparse.ArgumentParser(description="Train the LSTM burst predictor")
    parser.add_argument("--workloads", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output", type=str, default=LSTM_MODEL_PATH)
    args = parser.parse_args()

    torch.manual_seed(0)
    trainer = Trainer(num_workloads=args.workloads, epochs=args.epochs)
    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
