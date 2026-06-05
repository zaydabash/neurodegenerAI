"""
Persistence helpers for the Neuro-Trends Suite.

These functions provide a thin repository layer over the SQLAlchemy models
declared in :mod:`shared.lib.database`. They are intentionally best-effort:
a persistence failure is logged but never propagated to the API caller, so a
transient database problem cannot take down a prediction or search request.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from .database import (
    DatabaseManager,
    NeuroPrediction,
    SocialPost,
    TrendTopic,
    get_db_manager,
)
from .logging import get_logger

logger = get_logger(__name__)


def _session(db_manager: DatabaseManager | None) -> Session:
    return (db_manager or get_db_manager()).get_session()


def save_neuro_prediction(
    model_type: str,
    prediction: int,
    probability: float,
    confidence: float,
    results_metadata: dict[str, Any] | None = None,
    db_manager: DatabaseManager | None = None,
) -> int | None:
    """Persist a single neuro prediction. Returns the new row id, or None."""
    session = _session(db_manager)
    try:
        row = NeuroPrediction(
            model_type=model_type,
            prediction=int(prediction),
            probability=float(probability),
            confidence=float(confidence),
            results_metadata=results_metadata or {},
            timestamp=datetime.utcnow(),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id
    except Exception as exc:  # noqa: BLE001 - persistence is best-effort
        session.rollback()
        logger.error(f"Failed to persist neuro prediction: {exc}")
        return None
    finally:
        session.close()


def save_social_posts(
    posts: Iterable[dict[str, Any]],
    db_manager: DatabaseManager | None = None,
) -> int:
    """Persist a batch of social posts. Returns the number of rows written."""
    session = _session(db_manager)
    written = 0
    try:
        for post in posts:
            session.add(
                SocialPost(
                    text=post["text"],
                    source=post.get("source"),
                    url=post.get("url"),
                    author=post.get("author"),
                    score=int(post.get("score") or 0),
                    timestamp=_coerce_timestamp(post.get("timestamp")),
                )
            )
            written += 1
        session.commit()
        return written
    except Exception as exc:  # noqa: BLE001 - persistence is best-effort
        session.rollback()
        logger.error(f"Failed to persist social posts: {exc}")
        return 0
    finally:
        session.close()


def save_trend_topics(
    topics: Iterable[dict[str, Any]],
    db_manager: DatabaseManager | None = None,
) -> int:
    """Persist a batch of trending topics. Returns the number of rows written."""
    session = _session(db_manager)
    written = 0
    try:
        for topic in topics:
            session.add(
                TrendTopic(
                    topic=topic["topic"],
                    keywords=list(topic.get("keywords") or []),
                    trending_score=float(topic.get("trending_score") or 0.0),
                    volume=int(topic.get("volume") or 0),
                    growth_rate=float(topic.get("growth_rate") or 0.0),
                    timestamp=datetime.utcnow(),
                )
            )
            written += 1
        session.commit()
        return written
    except Exception as exc:  # noqa: BLE001 - persistence is best-effort
        session.rollback()
        logger.error(f"Failed to persist trend topics: {exc}")
        return 0
    finally:
        session.close()


def _coerce_timestamp(value: Any) -> datetime:
    """Best-effort conversion of a timestamp value to a ``datetime``."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.utcnow()
