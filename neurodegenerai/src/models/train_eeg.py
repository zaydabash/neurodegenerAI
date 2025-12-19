"""
EEG Signal Classification Model for NeuroDegenerAI.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet1D(nn.Module):
    """Simple 1D CNN for EEG signal classification."""

    def __init__(
        self, n_channels: int = 8, n_classes: int = 3, input_length: int = 250
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 16, kernel_size=15, stride=1, padding=7)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(5)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(5)

        # Calculate flattened size
        self.flat_size = 32 * (input_length // 5 // 5)

        self.fc1 = nn.Linear(self.flat_size, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EEGPredictor:
    """Wrapper for EEG model prediction."""

    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EEGNet1D().to(self.device)
        self.classes = ["Normal", "Sleep", "Anomalous"]

        if model_path:
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
            except Exception:
                # Fallback to untrained or random initialization for demo
                pass
        self.model.eval()

    def predict(self, data: np.ndarray) -> dict[str, Any]:
        """
        Predict state from EEG data.
        Input data shape: (batch, channels, length) or (channels, length)
        """
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]

        x = torch.FloatTensor(data).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            prob = probabilities[0][prediction].item()

        return {
            "prediction_idx": prediction,
            "prediction": self.classes[prediction],
            "probability": prob,
            "confidence": prob,  # Simple mapping for demo
            "model_name": "EEGNet1D",
            "model_type": "eeg",
        }


def train_mock_model(save_path: str = "neurodegenerai/models/eeg_model.pth"):
    """Quickly train/save a mock model on synthetic data."""
    from ..data.eeg_gen import EEGGenerator

    gen = EEGGenerator()
    X, y = gen.generate_batch(n_samples_per_state=50)

    X_train = torch.FloatTensor(X)
    y_train = torch.LongTensor(y)

    model = EEGNet1D()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), save_path)
    return save_path


if __name__ == "__main__":
    train_mock_model()
