"""
Synthetic EEG signal generation for NeuroDegenerAI.
"""


from typing import Any

import numpy as np


class EEGGenerator:
    """Generates synthetic multi-channel EEG signals."""

    def __init__(self, sfreq: int = 250, n_channels: int = 8):
        self.sfreq = sfreq
        self.n_channels = n_channels
        self.channel_names = [f"CH{i+1}" for i in range(n_channels)]

    def generate_signal(
        self, duration: float = 1.0, state: str = "normal"
    ) -> dict[str, Any]:
        """
        Generate synthetic EEG data for a given duration and state.

        States:
        - normal: Predominantly alpha and beta waves.
        - anomalous: Includes high-frequency spikes and irregular patterns.
        - sleep: Predominantly delta and theta waves.
        """
        n_samples = int(self.sfreq * duration)
        t = np.arange(n_samples) / self.sfreq

        data = np.zeros((self.n_channels, n_samples))

        for i in range(self.n_channels):
            if state == "normal":
                # Alpha (8-13 Hz) and Beta (13-30 Hz)
                s = 0.5 * np.sin(2 * np.pi * 10 * t + np.random.rand())  # Alpha
                s += 0.3 * np.sin(2 * np.pi * 20 * t + np.random.rand())  # Beta
                s += 0.2 * np.random.randn(n_samples)  # Noise
            elif state == "sleep":
                # Delta (0.5-4 Hz) and Theta (4-8 Hz)
                s = 0.8 * np.sin(2 * np.pi * 2 * t + np.random.rand())  # Delta
                s += 0.4 * np.sin(2 * np.pi * 6 * t + np.random.rand())  # Theta
                s += 0.1 * np.random.randn(n_samples)  # Low noise
            elif state == "anomalous":
                # Irregular with spikes
                s = 0.5 * np.sin(2 * np.pi * 15 * t + np.random.rand())
                # Add random spikes
                spikes = np.zeros(n_samples)
                spike_idx = np.random.choice(
                    n_samples, size=int(n_samples * 0.02), replace=False
                )
                spikes[spike_idx] = np.random.uniform(2.0, 5.0, size=len(spike_idx))
                s += spikes
                s += 0.4 * np.random.randn(n_samples)
            else:
                s = np.random.randn(n_samples)

            data[i, :] = s

        return {
            "data": data.tolist(),
            "sfreq": self.sfreq,
            "channels": self.channel_names,
            "state": state,
            "duration": duration,
        }

    def generate_normal_state(self, duration: float = 1.0) -> np.ndarray[Any, Any]:
        """Generate normal state EEG data."""
        return np.array(self.generate_signal(duration, "normal")["data"])

    def generate_sleep_state(self, duration: float = 1.0) -> np.ndarray[Any, Any]:
        """Generate sleep state EEG data."""
        return np.array(self.generate_signal(duration, "sleep")["data"])

    def generate_anomalous_state(self, duration: float = 1.0) -> np.ndarray[Any, Any]:
        """Generate anomalous state EEG data."""
        return np.array(self.generate_signal(duration, "anomalous")["data"])

    def generate_batch(
        self, n_samples_per_state: int = 10, duration: float = 1.0
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Generate a batch of labeled EEG data for training/evaluation."""
        states = ["normal", "sleep", "anomalous"]
        X = []
        y = []

        for i, state in enumerate(states):
            for _ in range(n_samples_per_state):
                signal = self.generate_signal(duration=duration, state=state)
                X.append(signal["data"])
                y.append(i)

        return np.array(X), np.array(y)
