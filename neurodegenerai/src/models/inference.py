"""
Self-contained, single-sample-safe inference service for NeuroDegenerAI.

This module makes the three neuro endpoints work out of the box, without any
pre-trained artifact having to be shipped. On first use it trains lightweight
*real* models on synthetic ADNI-like data and caches them under the model
directory:

* **Tabular** - a genuine soft-voting ensemble. Uses LightGBM + XGBoost when
  they are importable (as advertised) and otherwise falls back to a
  scikit-learn gradient-boosting + random-forest + logistic-regression
  ensemble, so it always runs.
* **MRI** - a genuine **3D CNN** (``nn.Conv3d``) over the whole volume, not a
  2D slice stack.
* **EEG** - the existing 1D CNN, auto-trained on first use if no checkpoint is
  present.

These are demo-grade models trained on synthetic data; they are deterministic
and return real probabilities, explanations, and (for MRI) an occlusion
saliency heatmap. When ``NEURO_DEMO_MODE`` is disabled and a production model
exists on disk, the API uses that instead (see ``ModelPredictor``).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from shared.lib.config import get_model_dir
from shared.lib.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tabular
# ---------------------------------------------------------------------------

# Deterministic, population-statistics-free feature order. Every feature is
# computed from a single record, so single-sample inference is always valid.
TABULAR_FIELDS = [
    "age",
    "sex",
    "apoe4",
    "mmse",
    "cdr",
    "abeta",
    "tau",
    "ptau",
    "education",
    "hippocampal_volume",
    "cortical_thickness",
    "white_matter_hyperintensities",
]

_TABULAR_DEFAULTS = {
    "age": 70.0,
    "sex": 0.0,
    "apoe4": 0.0,
    "mmse": 27.0,
    "cdr": 0.0,
    "abeta": 200.0,
    "tau": 300.0,
    "ptau": 25.0,
    "education": 14.0,
    "hippocampal_volume": 3000.0,
    "cortical_thickness": 2.5,
    "white_matter_hyperintensities": 5.0,
}

# Derived feature names appended after the raw fields.
_DERIVED_FIELDS = [
    "ptau_abeta_ratio",
    "tau_abeta_ratio",
    "age_apoe4",
    "mmse_impaired",
    "apoe4_homozygote",
    "abeta_log",
    "tau_log",
]

TABULAR_FEATURE_NAMES = TABULAR_FIELDS + _DERIVED_FIELDS


def tabular_feature_vector(record: dict[str, Any]) -> np.ndarray:
    """Build a deterministic feature vector from a single request record."""
    values = {}
    for field in TABULAR_FIELDS:
        raw = record.get(field)
        values[field] = float(raw) if raw is not None else _TABULAR_DEFAULTS[field]

    abeta = values["abeta"] or 1e-6
    derived = {
        "ptau_abeta_ratio": values["ptau"] / (abeta + 1e-6),
        "tau_abeta_ratio": values["tau"] / (abeta + 1e-6),
        "age_apoe4": values["age"] * values["apoe4"],
        "mmse_impaired": 1.0 if values["mmse"] < 24 else 0.0,
        "apoe4_homozygote": 1.0 if values["apoe4"] >= 2 else 0.0,
        "abeta_log": float(np.log(values["abeta"] + 1.0)),
        "tau_log": float(np.log(values["tau"] + 1.0)),
    }
    ordered = [values[f] for f in TABULAR_FIELDS] + [
        derived[f] for f in _DERIVED_FIELDS
    ]
    return np.array(ordered, dtype=np.float64)


def real_feature_vector(record: dict[str, Any]) -> np.ndarray:
    """Feature vector for the real-data model (age, sex, mmse)."""
    return np.array(
        [
            float(record.get("age") or _TABULAR_DEFAULTS["age"]),
            float(record.get("sex") or _TABULAR_DEFAULTS["sex"]),
            float(record.get("mmse") or _TABULAR_DEFAULTS["mmse"]),
        ],
        dtype=np.float64,
    )


def feature_vector_for(record: dict[str, Any], features: list[str]) -> np.ndarray:
    """Build a feature vector matching the model's trained feature set."""
    from ..data.openneuro import REAL_TABULAR_FEATURES

    if features == REAL_TABULAR_FEATURES:
        return real_feature_vector(record)
    return tabular_feature_vector(record)


def _synthetic_tabular_dataset(n: int = 2000, seed: int = 42):
    """Generate synthetic ADNI-like records with a known risk model."""
    rng = np.random.default_rng(seed)
    records = []
    labels = []
    for _ in range(n):
        age = float(np.clip(rng.normal(72, 8), 50, 95))
        sex = float(rng.integers(0, 2))
        apoe4 = float(rng.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))
        mmse = float(np.clip(rng.normal(26, 4), 0, 30))
        cdr = float(rng.choice([0, 0.5, 1, 2], p=[0.5, 0.3, 0.15, 0.05]))
        abeta = float(np.clip(rng.normal(200, 60), 50, 400))
        tau = float(np.clip(rng.normal(300, 90), 80, 700))
        ptau = float(np.clip(rng.normal(25, 9), 5, 80))
        education = float(np.clip(rng.normal(14, 3), 4, 22))
        hippo = float(np.clip(rng.normal(3000, 500), 1500, 4500))
        thickness = float(np.clip(rng.normal(2.5, 0.3), 1.5, 3.5))
        wmh = float(np.clip(rng.normal(5, 3), 0, 30))

        # Latent risk: higher tau/ptau, lower abeta/mmse/hippo, apoe4, age.
        risk = (
            0.04 * (age - 70)
            + 1.1 * apoe4
            + 0.18 * (28 - mmse)
            + 1.4 * cdr
            + 0.012 * (300 - abeta)
            + 0.006 * (tau - 300)
            + 0.05 * (ptau - 25)
            + 0.0012 * (3000 - hippo)
            + 0.06 * wmh
            - 0.05 * (education - 14)
        )
        prob = 1.0 / (1.0 + np.exp(-risk))
        label = int(rng.random() < prob)

        records.append(
            {
                "age": age,
                "sex": sex,
                "apoe4": apoe4,
                "mmse": mmse,
                "cdr": cdr,
                "abeta": abeta,
                "tau": tau,
                "ptau": ptau,
                "education": education,
                "hippocampal_volume": hippo,
                "cortical_thickness": thickness,
                "white_matter_hyperintensities": wmh,
            }
        )
        labels.append(label)

    X = np.vstack([tabular_feature_vector(r) for r in records])
    y = np.array(labels, dtype=int)
    return X, y


def _build_tabular_ensemble():
    """Build a soft-voting ensemble, preferring LightGBM/XGBoost."""
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    estimators = []
    backend = "sklearn"
    try:
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier

        estimators.append(("lightgbm", LGBMClassifier(n_estimators=120, verbose=-1)))
        estimators.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=120,
                    eval_metric="logloss",
                    use_label_encoder=False,
                    verbosity=0,
                ),
            )
        )
        backend = "lightgbm+xgboost"
    except Exception as exc:  # noqa: BLE001 - optional native deps
        logger.warning(
            f"LightGBM/XGBoost unavailable ({exc}); using scikit-learn ensemble"
        )
        estimators.append(("hist_gb", HistGradientBoostingClassifier(max_iter=200)))
        estimators.append(
            ("random_forest", RandomForestClassifier(n_estimators=200, n_jobs=-1))
        )
        estimators.append(
            (
                "logistic",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=1000)),
                    ]
                ),
            )
        )

    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    return ensemble, backend


# ---------------------------------------------------------------------------
# MRI - real 3D CNN
# ---------------------------------------------------------------------------

MRI_SHAPE = (24, 24, 24)


def _build_cnn3d():
    import torch.nn as nn

    class CNN3D(nn.Module):
        """A small but genuine 3D convolutional network.

        Uses global *max* pooling so a small, localized lesion is preserved
        through the bottleneck instead of being averaged away. BatchNorm is
        omitted to avoid train/eval running-statistic mismatch on the small
        demo dataset.
        """

        def __init__(self, num_classes: int = 2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),  # 12^3
                nn.Conv3d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),  # 6^3
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool3d(1),
            )
            self.classifier = nn.Linear(32, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            return self.classifier(x)

    return CNN3D()


def _resample_volume(volume: np.ndarray, shape=MRI_SHAPE) -> np.ndarray:
    """Resample an arbitrary 3D volume to a fixed shape via index sampling."""
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError("MRI volume must be 3D")
    out = np.zeros(shape, dtype=np.float32)
    idx = [
        np.linspace(0, max(volume.shape[d] - 1, 0), shape[d]).astype(int)
        for d in range(3)
    ]
    sampled = volume[np.ix_(idx[0], idx[1], idx[2])]
    out[: sampled.shape[0], : sampled.shape[1], : sampled.shape[2]] = sampled
    # Normalize
    std = out.std()
    if std > 0:
        out = (out - out.mean()) / std
    return out


def _synthetic_mri_batch(n: int = 80, seed: int = 7):
    """Synthetic volumes: class 1 has an injected hyperintense lesion."""
    rng = np.random.default_rng(seed)
    vols = []
    labels = []
    d, h, w = MRI_SHAPE
    zz, yy, xx = np.mgrid[0:d, 0:h, 0:w]
    for _ in range(n):
        vol = rng.normal(0, 1, MRI_SHAPE).astype(np.float32)
        label = int(rng.random() < 0.5)
        if label == 1:
            cz, cy, cx = (
                rng.integers(6, d - 6),
                rng.integers(6, h - 6),
                rng.integers(6, w - 6),
            )
            radius = rng.integers(3, 5)
            mask = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 < radius**2
            vol[mask] += rng.normal(3.0, 0.4)
        vols.append(_resample_volume(vol))
        labels.append(label)
    return np.stack(vols), np.array(labels, dtype=int)


# ---------------------------------------------------------------------------
# Inference service
# ---------------------------------------------------------------------------


class NeuroInferenceService:
    """Lazily-trained, cached demo models for tabular, MRI, and EEG."""

    _instance: NeuroInferenceService | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.model_dir = Path(get_model_dir("neuro"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._tabular = None
        self._tabular_backend = None
        self._tabular_features = TABULAR_FEATURE_NAMES
        self._tabular_source = "synthetic"
        self._mri = None
        self._eeg = None

    @classmethod
    def instance(cls) -> NeuroInferenceService:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # -- Tabular -----------------------------------------------------------
    def _tabular_training_data(self):
        """Return (X, y, source, feature_names) for tabular training.

        Prefers the real OpenNeuro ds004504 clinical metadata; falls back to
        synthetic ADNI-like data when real data is unavailable or disabled.
        """
        from shared.lib.config import get_settings

        source_pref = get_settings().neuro_data_source.lower()
        if source_pref in ("auto", "real"):
            from ..data.openneuro import REAL_TABULAR_FEATURES, load_real_tabular

            df = load_real_tabular()
            if df is not None and len(df) >= 30:
                X = df[REAL_TABULAR_FEATURES].to_numpy(dtype=float)
                y = df["label"].to_numpy(dtype=int)
                return X, y, "real[openneuro-ds004504]", list(REAL_TABULAR_FEATURES)
            if source_pref == "real":
                logger.warning("Real data requested but unavailable; using synthetic")

        X, y = _synthetic_tabular_dataset()
        return X, y, "synthetic", list(TABULAR_FEATURE_NAMES)

    def _ensure_tabular(self):
        if self._tabular is not None:
            return
        import joblib

        cache = self.model_dir / "demo_tabular.joblib"
        if cache.exists():
            data = joblib.load(cache)
            self._tabular = data["model"]
            self._tabular_backend = data.get("backend", "unknown")
            self._tabular_features = data.get("features", TABULAR_FEATURE_NAMES)
            self._tabular_source = data.get("source", "synthetic")
            return

        X, y, source, features = self._tabular_training_data()
        logger.info(f"Training tabular ensemble on {source} data ({len(X)} samples)")
        ensemble, backend = _build_tabular_ensemble()
        ensemble.fit(X, y)
        self._tabular = ensemble
        self._tabular_backend = backend
        self._tabular_features = features
        self._tabular_source = source
        joblib.dump(
            {
                "model": ensemble,
                "backend": backend,
                "features": features,
                "source": source,
            },
            cache,
        )

    def predict_tabular(self, record: dict[str, Any]) -> dict[str, Any]:
        self._ensure_tabular()
        x = feature_vector_for(record, self._tabular_features).reshape(1, -1)
        proba = float(self._tabular.predict_proba(x)[0, 1])
        prediction = int(proba >= 0.5)
        confidence = float(max(proba, 1 - proba))
        return {
            "prediction": prediction,
            "probability": proba,
            "confidence": confidence,
            "model_name": (
                f"tabular_ensemble[{self._tabular_backend}, data={self._tabular_source}]"
            ),
            "model_type": "tabular",
            "explanation": self._tabular_explanation(record),
        }

    def _tabular_explanation(self, record: dict[str, Any]) -> dict[str, Any]:
        importances = self._tabular_importances()
        x = feature_vector_for(record, self._tabular_features)
        ranked = sorted(
            zip(self._tabular_features, x, importances, strict=False),
            key=lambda t: t[2],
            reverse=True,
        )[:6]
        return {
            "top_features": [
                {"feature": name, "value": float(val), "importance": float(imp)}
                for name, val, imp in ranked
            ],
            "backend": self._tabular_backend,
            "data_source": self._tabular_source,
        }

    def _tabular_importances(self) -> np.ndarray:
        n = len(self._tabular_features)
        try:
            for _name, est in self._tabular.named_estimators_.items():
                if hasattr(est, "feature_importances_"):
                    imp = np.asarray(est.feature_importances_, dtype=float)
                    if imp.shape[0] == n:
                        return imp
        except Exception:  # noqa: BLE001
            pass
        return np.ones(n, dtype=float) / n

    # -- MRI (3D CNN) ------------------------------------------------------
    def _ensure_mri(self):
        if self._mri is not None:
            return
        import torch

        model = _build_cnn3d()
        cache = self.model_dir / "demo_mri3d.pth"
        if cache.exists():
            model.load_state_dict(torch.load(cache, map_location="cpu"))
            model.eval()
            self._mri = model
            return

        logger.info("Training demo 3D CNN on synthetic MRI volumes")
        self._train_mri(model, cache)
        self._mri = model

    def _train_mri(self, model, cache: Path) -> None:
        import torch
        import torch.nn as nn

        X, y = _synthetic_mri_batch(n=200)
        xb = torch.from_numpy(X).unsqueeze(1).float()  # (N,1,D,H,W)
        yb = torch.from_numpy(y).long()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        batch = 32
        for _ in range(40):
            perm = torch.randperm(len(xb))
            for i in range(0, len(xb), batch):
                idx = perm[i : i + batch]
                opt.zero_grad()
                loss = loss_fn(model(xb[idx]), yb[idx])
                loss.backward()
                opt.step()
        model.eval()
        torch.save(model.state_dict(), cache)

    def predict_mri(self, volume: np.ndarray) -> dict[str, Any]:
        self._ensure_mri()
        import torch
        import torch.nn.functional as F

        vol = _resample_volume(np.asarray(volume, dtype=np.float32))
        x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            probs = F.softmax(self._mri(x), dim=1)[0]
        prediction = int(torch.argmax(probs).item())
        probability = float(probs[1].item())
        confidence = float(torch.max(probs).item())
        heatmap_paths = self._mri_heatmap(vol)
        return {
            "prediction": prediction,
            "probability": probability,
            "confidence": confidence,
            "model_name": "demo_cnn3d",
            "model_type": "cnn3d",
            "heatmap_paths": heatmap_paths,
        }

    def _mri_heatmap(self, vol: np.ndarray) -> list[str]:
        """Occlusion-based saliency on the central axial slice."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import torch
            import torch.nn.functional as F

            base = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                base_score = F.softmax(self._mri(base), dim=1)[0, 1].item()

            d, h, w = vol.shape
            sal = np.zeros((h, w), dtype=np.float32)
            step = 4
            mid = d // 2
            for i in range(0, h, step):
                for j in range(0, w, step):
                    occ = vol.copy()
                    occ[:, i : i + step, j : j + step] = 0.0
                    t = torch.from_numpy(occ).unsqueeze(0).unsqueeze(0).float()
                    with torch.no_grad():
                        s = F.softmax(self._mri(t), dim=1)[0, 1].item()
                    sal[i : i + step, j : j + step] = base_score - s

            out_dir = Path("./neurodegenerai/reports")
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "mri_saliency.png"
            fig, ax = plt.subplots(1, 2, figsize=(6, 3))
            ax[0].imshow(vol[mid], cmap="gray")
            ax[0].set_title("Central slice")
            ax[0].axis("off")
            ax[1].imshow(sal, cmap="hot")
            ax[1].set_title("Occlusion saliency")
            ax[1].axis("off")
            fig.tight_layout()
            fig.savefig(path, dpi=80)
            plt.close(fig)
            return [str(path)]
        except Exception as exc:  # noqa: BLE001 - heatmap is best-effort
            logger.warning(f"Could not generate MRI heatmap: {exc}")
            return []

    # -- EEG ---------------------------------------------------------------
    def _ensure_eeg(self):
        if self._eeg is not None:
            return
        from .train_eeg import EEGPredictor, train_mock_model

        cache = self.model_dir / "eeg_model.pth"
        if not cache.exists():
            logger.info("Training demo EEG model on synthetic data")
            try:
                train_mock_model(str(cache))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"EEG demo training failed, using random init: {exc}")
        self._eeg = EEGPredictor(model_path=str(cache) if cache.exists() else None)

    def predict_eeg(self, data: np.ndarray) -> dict[str, Any]:
        self._ensure_eeg()
        return self._eeg.predict(np.asarray(data, dtype=np.float32))


def get_inference_service() -> NeuroInferenceService:
    """Return the process-wide neuro inference service."""
    return NeuroInferenceService.instance()
