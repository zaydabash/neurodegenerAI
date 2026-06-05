"""Tests for the real-data (OpenNeuro ds004504) loader.

These run offline: a fixture participants.tsv is written to the cache path so
no network access is required.
"""

import pytest

from neurodegenerai.src.data import openneuro

SAMPLE_TSV = (
    "participant_id\tGender\tAge\tGroup\tMMSE\n"
    "sub-001\tF\t57\tA\t16\n"
    "sub-002\tM\t70\tA\t14\n"
    "sub-003\tF\t67\tC\t29\n"
    "sub-004\tM\t64\tF\t24\n"
    "sub-005\tM\t72\tC\t30\n"
)


@pytest.fixture
def cached_dataset(tmp_path, monkeypatch):
    cache = tmp_path / "ds004504_participants.tsv"
    cache.write_text(SAMPLE_TSV)
    monkeypatch.setattr(openneuro, "_cache_path", lambda: cache)
    return cache


def test_load_real_tabular_parses_clinical_metadata(cached_dataset):
    df = openneuro.load_real_tabular()
    assert df is not None
    assert list(df.columns) == ["age", "sex", "mmse", "label"]
    assert len(df) == 5


def test_label_maps_dementia_vs_control(cached_dataset):
    df = openneuro.load_real_tabular()
    # A (Alzheimer's) and F (FTD) -> 1; C (control) -> 0
    assert df["label"].tolist() == [1, 1, 0, 1, 0]


def test_sex_encoding(cached_dataset):
    df = openneuro.load_real_tabular()
    assert df["sex"].tolist() == [0, 1, 0, 1, 1]  # F=0, M=1


def test_returns_none_when_unavailable(tmp_path, monkeypatch):
    missing = tmp_path / "absent.tsv"
    monkeypatch.setattr(openneuro, "_cache_path", lambda: missing)
    monkeypatch.setattr(openneuro, "download_participants", lambda timeout=30: None)
    assert openneuro.load_real_tabular() is None
