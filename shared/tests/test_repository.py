"""Tests for the persistence repository layer."""

import pytest

from shared.lib.config import reload_settings
from shared.lib.database import DatabaseManager
from shared.lib.repository import (
    save_neuro_prediction,
    save_social_posts,
    save_trend_topics,
)


@pytest.fixture
def db_manager(tmp_path, monkeypatch):
    """A DatabaseManager backed by a throwaway SQLite file."""
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_file}")
    reload_settings()
    manager = DatabaseManager()
    manager.create_tables()
    yield manager
    monkeypatch.delenv("DB_URL", raising=False)
    reload_settings()


def test_save_neuro_prediction_returns_id(db_manager):
    row_id = save_neuro_prediction(
        model_type="tabular",
        prediction=1,
        probability=0.82,
        confidence=0.9,
        results_metadata={"model_name": "ensemble"},
        db_manager=db_manager,
    )
    assert isinstance(row_id, int)
    assert row_id > 0


def test_save_social_posts_counts_rows(db_manager):
    posts = [
        {"text": "hello world", "source": "reddit", "score": 3},
        {
            "text": "another post",
            "source": "twitter",
            "timestamp": "2026-01-01T00:00:00",
        },
    ]
    written = save_social_posts(posts, db_manager=db_manager)
    assert written == 2


def test_save_trend_topics_counts_rows(db_manager):
    topics = [
        {
            "topic": "AI",
            "keywords": ["ml", "deep learning"],
            "trending_score": 0.9,
            "volume": 100,
            "growth_rate": 0.2,
        }
    ]
    written = save_trend_topics(topics, db_manager=db_manager)
    assert written == 1


def test_persistence_is_best_effort_on_bad_data(db_manager):
    # Missing required "text" key should be swallowed, not raised.
    written = save_social_posts([{"source": "reddit"}], db_manager=db_manager)
    assert written == 0
