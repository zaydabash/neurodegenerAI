"""Tests for the self-contained neuro inference service.

These train tiny real models on synthetic data, so they require torch and
scikit-learn (installed via requirements.txt).
"""

import numpy as np
import pytest

from shared.lib.config import reload_settings

torch = pytest.importorskip("torch")


@pytest.fixture
def service(tmp_path, monkeypatch):
    from neurodegenerai.src.models.inference import NeuroInferenceService

    monkeypatch.setenv("NEURO_MODEL_DIR", str(tmp_path))
    reload_settings()
    svc = NeuroInferenceService()
    yield svc
    monkeypatch.delenv("NEURO_MODEL_DIR", raising=False)
    reload_settings()


def test_tabular_separates_high_and_low_risk(service):
    high = {
        "age": 85,
        "sex": 1,
        "apoe4": 2,
        "mmse": 18,
        "abeta": 110,
        "tau": 480,
        "ptau": 50,
    }
    low = {
        "age": 60,
        "sex": 0,
        "apoe4": 0,
        "mmse": 30,
        "abeta": 280,
        "tau": 220,
        "ptau": 15,
    }
    rh = service.predict_tabular(high)
    rl = service.predict_tabular(low)
    assert rh["prediction"] == 1
    assert rh["probability"] > rl["probability"]
    assert rh["explanation"]["top_features"]


def test_tabular_output_contract(service):
    r = service.predict_tabular({"age": 70, "sex": 0, "apoe4": 1, "mmse": 25})
    assert 0.0 <= r["probability"] <= 1.0
    assert 0.0 <= r["confidence"] <= 1.0
    assert r["model_type"] == "tabular"


def test_mri_3d_cnn_detects_lesion(service):
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, (28, 28, 28))
    lesion = noise.copy()
    lesion[11:17, 11:17, 11:17] += 4.0
    assert service.predict_mri(lesion)["prediction"] == 1
    assert service.predict_mri(noise)["prediction"] == 0


def test_mri_uses_conv3d(service):
    service._ensure_mri()
    assert any(
        isinstance(m, torch.nn.Conv3d) for m in service._mri.modules()
    ), "MRI model must use 3D convolutions"


def test_eeg_returns_valid_state(service):
    rng = np.random.default_rng(1)
    out = service.predict_eeg(rng.normal(0, 1, (8, 250)))
    assert out["prediction"] in {"Normal", "Sleep", "Anomalous"}
    assert 0.0 <= out["confidence"] <= 1.0
