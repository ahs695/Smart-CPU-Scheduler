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