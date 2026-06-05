"""
Real clinical data loader for OpenNeuro dataset ds004504.

ds004504 - "A dataset of EEG recordings from Alzheimer's disease,
Frontotemporal dementia and Healthy subjects" (Miltiadous et al., 2023) - is
openly available on OpenNeuro's public S3 bucket (no credentials required) and
released under CC0.

The ``participants.tsv`` metadata provides real Age, Gender, MMSE and clinical
diagnosis (Group: A = Alzheimer's, F = Frontotemporal dementia, C = healthy
control) for 88 subjects. This module downloads and caches that file and turns
it into a training frame for the tabular biomarker model.

Reference:
    Miltiadous, A. et al. (2023). A Dataset of Scalp EEG Recordings of
    Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from
    Routine EEG. Data, 8(6), 95. https://doi.org/10.3390/data8060095
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from shared.lib.logging import get_logger

logger = get_logger(__name__)

DATASET_ID = "ds004504"
PARTICIPANTS_URL = "https://s3.amazonaws.com/openneuro.org/ds004504/participants.tsv"

# Real feature columns available from the clinical metadata.
REAL_TABULAR_FEATURES = ["age", "sex", "mmse"]


def _cache_path() -> Path:
    cache_dir = Path("./neurodegenerai/data/openneuro")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "ds004504_participants.tsv"


def download_participants(timeout: int = 30, force: bool = False) -> Path | None:
    """Download and cache participants.tsv. Returns the path, or None on failure."""
    cache = _cache_path()
    if cache.exists() and not force:
        return cache
    try:
        import requests

        resp = requests.get(PARTICIPANTS_URL, timeout=timeout)
        resp.raise_for_status()
        cache.write_text(resp.text)
        logger.info(f"Downloaded real clinical metadata for {DATASET_ID}")
        return cache
    except Exception as exc:  # noqa: BLE001 - network is optional
        logger.warning(f"Could not download {DATASET_ID} participants.tsv: {exc}")
        return None


def load_real_tabular(timeout: int = 30) -> pd.DataFrame | None:
    """Return a real training frame with columns age, sex, mmse, label.

    label = 1 for any dementia diagnosis (Alzheimer's or FTD), 0 for control.
    Returns None when the dataset cannot be obtained (e.g. offline).
    """
    cache = download_participants(timeout=timeout)
    if cache is None:
        # Use a stale cache if a download failed but a file exists.
        cache = _cache_path()
        if not cache.exists():
            return None

    try:
        raw = cache.read_text()
        df = pd.read_csv(io.StringIO(raw), sep="\t")
        df = df[df["MMSE"].notna()].copy()
        df["age"] = df["Age"].astype(float)
        df["sex"] = (df["Gender"].astype(str).str.upper() == "M").astype(int)
        df["mmse"] = df["MMSE"].astype(float)
        df["label"] = (df["Group"].astype(str) != "C").astype(int)
        result = df[["age", "sex", "mmse", "label"]].reset_index(drop=True)
        logger.info(
            f"Loaded {len(result)} real subjects from {DATASET_ID} "
            f"(dementia={int(result['label'].sum())}, "
            f"control={int((result['label'] == 0).sum())})"
        )
        return result
    except Exception as exc:  # noqa: BLE001 - malformed/partial file
        logger.warning(f"Could not parse {DATASET_ID} participants.tsv: {exc}")
        return None
