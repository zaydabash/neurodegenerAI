#!/usr/bin/env python3
"""
Build the real EEG Alzheimer's-vs-control model.

Downloads preprocessed EEG recordings from OpenNeuro ds004504, extracts
band-power features, trains the band-power CNN with subject-level evaluation,
and caches the model under the neuro model directory.

Usage:
    PYTHONPATH=. python scripts/train_eeg_real.py --max-per-class 20 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import sys

from neurodegenerai.src.models.eeg_real import train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=20,
        help="Max Alzheimer's and control subjects to download (each).",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument(
        "--test-fraction", type=float, default=0.3, help="Held-out subject fraction."
    )
    args = parser.parse_args()

    metrics = train(
        max_per_class=args.max_per_class,
        epochs=args.epochs,
        test_fraction=args.test_fraction,
    )
    if metrics is None:
        print("Training failed: real EEG dataset could not be built.", file=sys.stderr)
        return 1

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
