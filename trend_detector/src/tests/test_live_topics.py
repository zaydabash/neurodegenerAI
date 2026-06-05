"""Tests for live topic clustering and PII-scrubbing provider."""

from trend_detector.src.ingest.provider import StreamProvider
from trend_detector.src.pipeline.live_topics import cluster_posts

CORPUS = [
    {
        "text": "new AI model breaks machine learning benchmarks",
        "timestamp": "2026-01-01T10:00:00",
    },
    {"text": "deep learning and neural networks advance AI research"},
    {"text": "machine learning powers the latest AI breakthrough"},
    {"text": "the stock market rallies as tech earnings beat estimates"},
    {"text": "investors cheer strong quarterly earnings on wall street"},
    {"text": "market volatility rises amid earnings season"},
    {"text": "new vaccine enters phase three clinical trial"},
    {"text": "researchers report progress on cancer treatment therapy"},
]


def test_cluster_posts_returns_topics():
    topics = cluster_posts(CORPUS, k=3)
    assert topics, "expected clustered topics"
    for t in topics:
        assert t["keywords"]
        assert t["volume"] >= 1
        assert 0.0 <= t["trending_score"] <= 1.0


def test_cluster_posts_too_few_returns_empty():
    assert cluster_posts([{"text": "only one"}], k=3) == []


def test_provider_scrubs_pii():
    provider = StreamProvider()
    scrubbed = provider._scrub_posts(
        [{"text": "reach me at jane@example.com", "author": "555-123-4567"}]
    )
    assert "jane@example.com" not in scrubbed[0]["text"]
    assert "[EMAIL_REDACTED]" in scrubbed[0]["text"]
    assert "[PHONE_REDACTED]" in scrubbed[0]["author"]


def test_provider_falls_back_to_demo_without_credentials():
    provider = StreamProvider()
    posts = provider.get_recent_posts(10)
    assert len(posts) > 0
    assert provider.active_sources == ["demo"]
