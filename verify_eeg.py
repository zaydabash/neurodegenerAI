"""
Verification script for EEG API endpoint.
"""

import os
import sys

import requests

# Add src to path to use EEGGenerator
sys.path.append(os.path.join(os.getcwd()))
from neurodegenerai.src.data.eeg_gen import EEGGenerator  # noqa: E402


def test_eeg_prediction() -> None:
    url = "http://127.0.0.1:8000/v1/neuro/eeg"
    gen = EEGGenerator(n_channels=8)

    states = ["normal", "sleep", "anomalous"]

    for state in states:
        print(f"\nTesting state: {state}")
        signal = gen.generate_signal(duration=1.0, state=state)

        payload = {
            "data": signal["data"],
            "sfreq": signal["sfreq"],
            "metadata": {"test": True, "original_state": state},
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(
                    f"Success! Prediction: {result['explanation']['state']}, Probability: {result['probability']:.4f}"
                )
            else:
                print(f"Failed with status code: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Error connecting to API: {e}")
            print("Is the NeuroDegenerAI API running? (make neuro-api)")


if __name__ == "__main__":
    test_eeg_prediction()
