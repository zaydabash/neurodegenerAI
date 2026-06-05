"""Tests for the real EEG band-power pipeline and model.

Feature extraction is tested on synthetic signals (no network/MNE-file needed);
the band-power computation itself uses MNE's Welch PSD.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("mne")

from neurodegenerai.src.data import eeg_openneuro as eo  # noqa: E402


def _sine(freq, sfreq, secs, n_ch=19):
    t = np.arange(int(sfreq * secs)) / sfreq
    sig = np.sin(2 * np.pi * freq * t)
    return np.tile(sig, (n_ch, 1)).astype(np.float32)


def test_bandpower_matrix_shape_and_normalization():
    epoch = _sine(10.0, 250, 4)  # 10 Hz -> alpha band
    bp = eo.bandpower_matrix(epoch, 250.0)
    assert bp.shape == (19, eo.N_BANDS)
    # Relative powers across bands sum to ~1 per channel.
    assert np.allclose(bp.sum(axis=1), 1.0, atol=0.05)


def test_bandpower_localizes_frequency():
    # A 10 Hz tone should put most power in the alpha band (index 2).
    bp = eo.bandpower_matrix(_sine(10.0, 250, 4), 250.0)
    assert bp[:, 2].mean() > bp[:, 0].mean()  # alpha > delta
    # A 2 Hz tone should put most power in the delta band (index 0).
    bp_delta = eo.bandpower_matrix(_sine(2.0, 250, 4), 250.0)
    assert bp_delta[:, 0].mean() > bp_delta[:, 2].mean()


def test_feature_matrix_includes_ratios():
    fm = eo.feature_matrix(_sine(10.0, 250, 4), 250.0)
    assert fm.shape == (19, eo.N_FEATURES)
    assert eo.N_FEATURES == eo.N_BANDS + 2


def test_epochs_features_splits_recording():
    data = _sine(10.0, 250, 12)  # 12s -> three 4s epochs
    feats = eo.epochs_features(data, 250.0, epoch_sec=4.0)
    assert feats.shape == (3, 19, eo.N_FEATURES)


def test_epochs_features_too_short_returns_empty():
    feats = eo.epochs_features(_sine(10.0, 250, 1), 250.0, epoch_sec=4.0)
    assert feats.shape[0] == 0


def test_channel_fitting_pads_and_truncates():
    assert eo._fit_channels(np.zeros((10, 100))).shape[0] == eo.N_CHANNELS
    assert eo._fit_channels(np.zeros((25, 100))).shape[0] == eo.N_CHANNELS


def test_model_forward_shape():
    import torch

    from neurodegenerai.src.models.eeg_real import _build_model

    model = _build_model()
    out = model(torch.randn(8, eo.N_FEATURES, eo.N_CHANNELS))
    assert out.shape == (8, 2)


def test_predictor_unavailable_without_artifact(tmp_path, monkeypatch):
    from shared.lib.config import reload_settings

    monkeypatch.setenv("NEURO_MODEL_DIR", str(tmp_path))
    reload_settings()
    from neurodegenerai.src.models.eeg_real import RealEEGPredictor

    assert RealEEGPredictor().available is False
    monkeypatch.delenv("NEURO_MODEL_DIR", raising=False)
    reload_settings()
