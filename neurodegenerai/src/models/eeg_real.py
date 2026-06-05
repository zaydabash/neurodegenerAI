"""
Real EEG Alzheimer's-vs-control model (band-power CNN).

A compact 1D CNN consumes per-epoch band-power maps (channels x bands) from
real ds004504 recordings. Training uses **subject-level** train/test splits so
epochs from the same person never leak across the split - the single most
important correctness detail for this kind of dataset.

At inference, a raw EEG array is epoched, band power is computed per epoch, the
CNN scores each epoch, and the per-epoch probabilities are averaged into a
subject-level Alzheimer's probability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from shared.lib.config import get_model_dir
from shared.lib.logging import get_logger

from ..data.eeg_openneuro import (
    BAND_NAMES,
    N_BANDS,
    N_CHANNELS,
    N_FEATURES,
    build_dataset,
    epochs_features,
)

logger = get_logger(__name__)


def _build_model():
    import torch.nn as nn

    class BandPowerCNN(nn.Module):
        """1D CNN over EEG channels with per-band features as input channels.

        Regularized (BatchNorm + dropout + average pooling) because the
        dataset has relatively few subjects and the real risk is overfitting
        to individual subjects rather than learning the AD signature.
        """

        def __init__(self, n_features: int = N_FEATURES, num_classes: int = 2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(32, num_classes),
            )

        def forward(self, x):  # x: (batch, features, channels)
            return self.classifier(self.features(x))

    return BandPowerCNN()


def _to_tensor(feats: np.ndarray):
    """(n, channels, features) -> tensor (n, features, channels)."""
    import torch

    return torch.from_numpy(np.asarray(feats, dtype=np.float32)).permute(0, 2, 1)


def _standardize(feats: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (feats - mean) / (std + 1e-6)


def _model_paths() -> tuple[Path, Path]:
    model_dir = Path(get_model_dir("neuro"))
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "eeg_real_bandpower.pth", model_dir / "eeg_real_bandpower.json"


def _fit_model(X, y, epochs, seed):
    """Standardize, train a fresh CNN, and return (model, mean, std)."""
    import torch
    import torch.nn as nn

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    xb = _to_tensor(_standardize(X, mean, std))
    yb = torch.from_numpy(np.asarray(y)).long()

    torch.manual_seed(seed)
    model = _build_model()
    counts = np.bincount(y, minlength=2).astype(np.float32)
    weights = torch.tensor(
        counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    model.train()
    batch = 128
    for _ in range(epochs):
        perm = torch.randperm(len(xb))
        for i in range(0, len(xb), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss_fn(model(xb[idx]), yb[idx]).backward()
            opt.step()
    model.eval()
    return model, mean, std


def _subject_scores(model, X, y, groups, mean, std):
    """Return (subject_true, subject_prob) on the given set."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        prob = F.softmax(model(_to_tensor(_standardize(X, mean, std))), dim=1)[
            :, 1
        ].numpy()
    return _aggregate_by_subject(y, prob, groups)


def _cross_validate(X, y, groups, epochs, seed, n_splits=5):
    """GroupKFold CV; returns mean held-out subject-level AUC and accuracy."""
    from sklearn.model_selection import GroupKFold

    uniq = np.array(sorted(set(groups.tolist())))
    n_splits = min(n_splits, len(uniq))
    gkf = GroupKFold(n_splits=n_splits)
    all_true, all_prob = [], []
    for tr, te in gkf.split(X, y, groups):
        model, mean, std = _fit_model(X[tr], y[tr], epochs, seed)
        st, sp = _subject_scores(model, X[te], y[te], groups[te], mean, std)
        all_true.extend(st)
        all_prob.extend(sp)
    auc = _safe_auc(all_true, all_prob)
    acc = float(np.mean((np.array(all_prob) >= 0.5) == np.array(all_true)))
    return auc, acc, len(all_true)


def train(
    max_per_class: int = 20,
    epochs: int = 60,
    test_fraction: float = 0.3,  # retained for API compatibility
    seed: int = 42,
) -> dict[str, Any] | None:
    """Download real data, evaluate with GroupKFold CV, train + cache a model.

    Returns a metrics dict, or None if the dataset could not be built.
    """
    import torch

    dataset = build_dataset(max_per_class=max_per_class)
    if dataset is None:
        logger.warning("Real EEG dataset unavailable; skipping training")
        return None

    X, y, groups = dataset["X"], dataset["y"], dataset["groups"]
    if len(set(groups)) < 6:
        logger.warning("Too few subjects to train a real EEG model")
        return None

    # Honest generalization estimate via subject-level cross-validation.
    cv_auc, cv_acc, n_eval = _cross_validate(X, y, groups, epochs, seed)
    logger.info(
        f"EEG CV: subject AUC={cv_auc:.3f} acc={cv_acc:.3f} over {n_eval} subjects"
    )

    # Final model trained on all subjects, cached for serving.
    model, mean, std = _fit_model(X, y, epochs, seed)

    model_path, meta_path = _model_paths()
    torch.save(model.state_dict(), model_path)
    meta = {
        "channels": N_CHANNELS,
        "bands": BAND_NAMES,
        "n_features": N_FEATURES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_subjects": int(len(set(groups))),
        "n_epochs": int(len(X)),
        "cv_subject_auc": cv_auc,
        "cv_subject_accuracy": cv_acc,
        "subject_auc": cv_auc,  # headline = cross-validated estimate
        "dataset": "openneuro-ds004504",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(
        f"Trained real EEG model on {meta['n_subjects']} subjects; "
        f"cross-validated subject AUC={cv_auc:.3f}"
    )
    return meta


def _safe_auc(y_true, y_score) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _aggregate_by_subject(y_true, y_prob, groups):
    subj_true, subj_prob = [], []
    for subject in sorted(set(groups.tolist())):
        mask = groups == subject
        subj_true.append(int(np.round(y_true[mask].mean())))
        subj_prob.append(float(y_prob[mask].mean()))
    return subj_true, subj_prob


class RealEEGPredictor:
    """Predict Alzheimer's vs control from a raw EEG array."""

    def __init__(self) -> None:
        import torch

        self.model_path, self.meta_path = _model_paths()
        self.available = self.model_path.exists() and self.meta_path.exists()
        self.model = None
        self.meta: dict[str, Any] = {}
        if self.available:
            self.meta = json.loads(self.meta_path.read_text())
            self.mean = np.array(self.meta["mean"], dtype=np.float32)
            self.std = np.array(self.meta["std"], dtype=np.float32)
            self.model = _build_model()
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            self.model.eval()

    def predict(self, data: np.ndarray, sfreq: float) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        if not self.available:
            raise FileNotFoundError("Real EEG model not trained")

        feats = epochs_features(np.asarray(data, dtype=np.float32), float(sfreq))
        if feats.shape[0] == 0:
            raise ValueError("EEG segment too short for band-power analysis")

        std_feats = _standardize(feats, self.mean, self.std)
        with torch.no_grad():
            probs = F.softmax(self.model(_to_tensor(std_feats)), dim=1)[:, 1].numpy()
        probability = float(np.mean(probs))
        prediction = int(probability >= 0.5)
        confidence = float(max(probability, 1 - probability))

        # Mean relative power per band (first N_BANDS feature columns).
        band_means = feats[:, :, :N_BANDS].mean(axis=(0, 1))
        return {
            "prediction_idx": prediction,
            "prediction": "Alzheimer's" if prediction == 1 else "Control",
            "probability": probability,
            "confidence": confidence,
            "model_name": "eeg_bandpower_cnn[real:openneuro-ds004504]",
            "model_type": "eeg_alzheimer",
            "explanation": {
                "band_power": {
                    name: round(float(val), 4)
                    for name, val in zip(BAND_NAMES, band_means, strict=False)
                },
                "n_epochs": int(feats.shape[0]),
                "subject_auc": self.meta.get("subject_auc"),
            },
        }


_predictor: RealEEGPredictor | None = None


def get_real_eeg_predictor() -> RealEEGPredictor:
    """Return a cached RealEEGPredictor (its ``available`` flag may be False)."""
    global _predictor
    if _predictor is None:
        _predictor = RealEEGPredictor()
    return _predictor
