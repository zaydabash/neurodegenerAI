"""
Real EEG data pipeline for OpenNeuro ds004504 (Alzheimer's vs control).

Downloads the preprocessed EEG recordings (EEGLAB ``.set``) from OpenNeuro's
public S3 bucket, reads them with MNE, and turns each recording into a set of
per-epoch **band-power** feature maps suitable for training the EEG CNN.

Band power captures the hallmark EEG "slowing" of Alzheimer's disease
(increased delta/theta, decreased alpha/beta power), which is why it is the
feature of choice for AD-vs-control classification.

Data: Miltiadous, A. et al. (2023), OpenNeuro ds004504 (CC0).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from shared.lib.logging import get_logger

logger = get_logger(__name__)

S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds004504"
PARTICIPANTS_URL = f"{S3_BASE}/participants.tsv"

# Canonical frequency bands (Hz).
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_NAMES = list(BANDS.keys())
# Model input features per channel: the 5 relative band powers plus two
# ratios that are established markers of Alzheimer's EEG "slowing".
FEATURE_NAMES = BAND_NAMES + ["alpha_theta_ratio", "alpha_delta_ratio"]
N_BANDS = len(BANDS)
N_FEATURES = len(FEATURE_NAMES)
N_CHANNELS = 19  # standard 10-20 montage used throughout ds004504
EPOCH_SEC = 4.0


def _cache_dir() -> Path:
    d = Path("./neurodegenerai/data/openneuro/eeg")
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_groups(timeout: int = 30) -> dict[str, str] | None:
    """Return {participant_id: Group} from participants.tsv ('A'/'F'/'C')."""
    cache = _cache_dir().parent / "ds004504_participants.tsv"
    text: str | None = None
    if cache.exists():
        text = cache.read_text()
    else:
        try:
            import requests

            resp = requests.get(PARTICIPANTS_URL, timeout=timeout)
            resp.raise_for_status()
            text = resp.text
            cache.write_text(text)
        except Exception as exc:  # noqa: BLE001 - network optional
            logger.warning(f"Could not fetch participants.tsv: {exc}")
            return None

    import pandas as pd

    df = pd.read_csv(io.StringIO(text), sep="\t")
    return dict(zip(df["participant_id"], df["Group"], strict=False))


def download_subject(subject: str, timeout: int = 180) -> Path | None:
    """Download a subject's preprocessed EEG ``.set`` file (cached)."""
    dest = _cache_dir() / f"{subject}_eeg.set"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{S3_BASE}/derivatives/{subject}/eeg/{subject}_task-eyesclosed_eeg.set"
    try:
        import requests

        with requests.get(url, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        logger.info(f"Downloaded EEG for {subject}")
        return dest
    except Exception as exc:  # noqa: BLE001 - network optional
        logger.warning(f"Could not download EEG for {subject}: {exc}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return None


def bandpower_matrix(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """Relative band power for one epoch.

    Args:
        epoch: array of shape (channels, samples).
        sfreq: sampling frequency in Hz.

    Returns:
        Array of shape (channels, n_bands) of relative band power (rows sum ~1).
    """
    from mne.time_frequency import psd_array_welch

    n_fft = int(min(epoch.shape[1], max(64, 2 * sfreq)))
    psd, freqs = psd_array_welch(
        epoch, sfreq=sfreq, fmin=1.0, fmax=45.0, n_fft=n_fft, verbose=False
    )
    total = psd.sum(axis=1, keepdims=True) + 1e-12
    feats = np.zeros((epoch.shape[0], N_BANDS), dtype=np.float32)
    for b_idx, (lo, hi) in enumerate(BANDS.values()):
        mask = (freqs >= lo) & (freqs < hi)
        feats[:, b_idx] = (psd[:, mask].sum(axis=1) / total[:, 0]).astype(np.float32)
    return feats


def feature_matrix(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """Model input features for one epoch: band powers + slowing ratios.

    Returns array (channels, N_FEATURES) = 5 relative band powers plus
    alpha/theta and alpha/delta ratios per channel.
    """
    bp = bandpower_matrix(epoch, sfreq)  # (channels, 5)
    delta, theta, alpha = bp[:, 0], bp[:, 1], bp[:, 2]
    alpha_theta = alpha / (theta + 1e-6)
    alpha_delta = alpha / (delta + 1e-6)
    return np.concatenate(
        [bp, alpha_theta[:, None], alpha_delta[:, None]], axis=1
    ).astype(np.float32)


def epochs_features(
    data: np.ndarray, sfreq: float, epoch_sec: float = EPOCH_SEC
) -> np.ndarray:
    """Split a recording into epochs and compute features for each.

    Returns array (n_epochs, channels, N_FEATURES).
    """
    data = _fit_channels(np.asarray(data, dtype=np.float32))
    win = int(epoch_sec * sfreq)
    empty = np.empty((0, N_CHANNELS, N_FEATURES), dtype=np.float32)
    if win < 8 or data.shape[1] < win:
        return empty
    n = data.shape[1] // win
    feats = [feature_matrix(data[:, i * win : (i + 1) * win], sfreq) for i in range(n)]
    return np.stack(feats) if feats else empty


def _fit_channels(data: np.ndarray) -> np.ndarray:
    """Pad/truncate the channel dimension to N_CHANNELS."""
    if data.shape[0] == N_CHANNELS:
        return data
    if data.shape[0] > N_CHANNELS:
        return data[:N_CHANNELS]
    pad = np.zeros((N_CHANNELS - data.shape[0], data.shape[1]), dtype=data.dtype)
    return np.vstack([data, pad])


def _read_recording(path: Path) -> tuple[np.ndarray, float] | None:
    try:
        import mne

        mne.set_log_level("ERROR")
        raw = mne.io.read_raw_eeglab(str(path), preload=True)
        return raw.get_data().astype(np.float32), float(raw.info["sfreq"])
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not read {path.name}: {exc}")
        return None


def build_dataset(
    max_per_class: int = 20, epoch_sec: float = EPOCH_SEC
) -> dict[str, np.ndarray] | None:
    """Build the real AD-vs-control band-power dataset.

    Downloads up to ``max_per_class`` Alzheimer's (Group A) and control
    (Group C) subjects, extracts per-epoch band-power features, and returns
    arrays with subject groupings for leakage-free, subject-level evaluation.

    Returns dict with X (n_epochs, channels, bands), y (0=control, 1=AD),
    groups (subject id per epoch), or None if no data could be obtained.
    """
    groups_map = load_groups()
    if not groups_map:
        return None

    ad = sorted(s for s, g in groups_map.items() if g == "A")[:max_per_class]
    control = sorted(s for s, g in groups_map.items() if g == "C")[:max_per_class]
    selected = [(s, 1) for s in ad] + [(s, 0) for s in control]

    X_list, y_list, grp_list = [], [], []
    for subject, label in selected:
        path = download_subject(subject)
        if path is None:
            continue
        rec = _read_recording(path)
        if rec is None:
            continue
        data, sfreq = rec
        feats = epochs_features(data, sfreq, epoch_sec)
        if feats.shape[0] == 0:
            continue
        X_list.append(feats)
        y_list.append(np.full(feats.shape[0], label, dtype=np.int64))
        grp_list.append(np.array([subject] * feats.shape[0]))

    if not X_list:
        return None

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(grp_list)
    logger.info(
        f"Built real EEG dataset: {X.shape[0]} epochs from "
        f"{len(set(groups))} subjects (AD={int(y.sum())}, control={int((y == 0).sum())})"
    )
    return {"X": X, "y": y, "groups": groups}
