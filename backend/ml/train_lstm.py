import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from backend.ml.dataset import DatasetBuilder
from backend.ml.lstm_model import BurstPredictorLSTM


class Trainer:
    """
    Training pipeline for LSTM burst predictor.
    """

    def __init__(
        self,
        seq_length: int = 5,
        dataset_size: int = 5000,
        batch_size: int = 64,
        epochs: int = 30,
        learning_rate: float = 0.001,
        device: str = None
    ):

        self.seq_length = seq_length
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print("Using device:", self.device)

        # Build dataset
        dataset = DatasetBuilder.build_dataset(
            num_sequences=dataset_size,
            seq_length=seq_length
        )

        # Train/validation split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size]
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size
        )

        # Model
        self.model = BurstPredictorLSTM().to(self.device)

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

    # ------------------------------------------------------------

    def train_epoch(self):

        self.model.train()

        total_loss = 0

        for X, y in self.train_loader:

            X = X.to(self.device)
            y = y.to(self.device)

            preds = self.model(X)

            loss = self.criterion(preds, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    # ------------------------------------------------------------

    def validate(self):

        self.model.eval()

        total_loss = 0

        with torch.no_grad():

            for X, y in self.val_loader:

                X = X.to(self.device)
                y = y.to(self.device)

                preds = self.model(X)

                loss = self.criterion(preds, y)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    # ------------------------------------------------------------

    def train(self):

        print("\nStarting training...\n")

        for epoch in range(1, self.epochs + 1):

            train_loss = self.train_epoch()
            val_loss = self.validate()

            print(
                f"Epoch {epoch:02d} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f}"
            )

    # ------------------------------------------------------------

    def save_model(self, path="models/lstm_model.pt"):

        os.makedirs("models", exist_ok=True)

        torch.save(self.model.state_dict(), path)

        print("\nModel saved to:", path)


# ------------------------------------------------------------


def main():

    trainer = Trainer()

    trainer.train()

    trainer.save_model()


if __name__ == "__main__":
    main()