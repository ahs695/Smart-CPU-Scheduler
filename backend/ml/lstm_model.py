import torch
import torch.nn as nn


class BurstPredictorLSTM(nn.Module):
    """
    LSTM model for CPU burst prediction.

    Input:
        sequence of past CPU bursts

    Output:
        predicted next burst length
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------

    def forward(self, x):
        """
        Forward pass.

        x shape:
            (batch_size, sequence_length)

        Convert to:
            (batch_size, sequence_length, input_size)
        """

        # Add feature dimension
        x = x.unsqueeze(-1)

        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Take output of last timestep
        last_output = lstm_out[:, -1, :]

        # Predict next burst
        prediction = self.fc(last_output)

        return prediction.squeeze(-1)

# ------------------------------------------------------------
# Shared constants — used by the dataset, training, the Hybrid env and the API,
# so every consumer feeds the model identically scaled inputs.
# ------------------------------------------------------------

SEQ_LENGTH = 8      # number of previous bursts the model sees
MAX_BURST = 50.0    # bursts are divided by this (and clipped to [0, 1])


def normalize_burst(burst: float) -> float:
    return min(max(burst / MAX_BURST, 0.0), 1.0)


def history_window(previous_bursts) -> list:
    """Last SEQ_LENGTH normalised bursts, left-padded with 0.0 when history is short."""
    window = [normalize_burst(b) for b in list(previous_bursts)[-SEQ_LENGTH:]]
    return [0.0] * (SEQ_LENGTH - len(window)) + window


def load_lstm(path: str, device: str = "cpu"):
    """Load a BurstPredictorLSTM checkpoint (plain state_dict or {'model_state_dict': ...})."""
    import os

    model = BurstPredictorLSTM()
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint)
    else:
        model = None
    if model is not None:
        model.to(device).eval()
    return model
