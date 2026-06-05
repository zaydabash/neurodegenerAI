"""
Twitter/X ingestion using the v2 API (tweepy), with exponential backoff.

Activated when ``X_BEARER_TOKEN`` is configured. Produces posts in the same
shape as :class:`RedditStream` so the two sources are interchangeable.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from shared.lib.config import get_settings
from shared.lib.logging import LoggerMixin, get_logger

logger = get_logger(__name__)

_DEFAULT_TOPICS = ["technology", "health", "science", "ai"]


class TwitterStream(LoggerMixin):
    """Fetch recent tweets via the Twitter/X v2 API."""

    def __init__(self, max_retries: int = 4) -> None:
        self.settings = get_settings()
        self.max_retries = max_retries
        self.client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        token = self.settings.x_bearer_token
        if not token:
            self.logger.info("X_BEARER_TOKEN not set; Twitter stream disabled")
            return
        try:
            import tweepy

            self.client = tweepy.Client(bearer_token=token, wait_on_rate_limit=False)
            self.logger.info("Twitter client initialized")
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self.logger.warning(f"Twitter client unavailable: {exc}")
            self.client = None

    def _search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Recent-search with exponential backoff on rate limits/errors."""
        if self.client is None:
            return []

        import tweepy

        # The v2 recent-search endpoint requires 10..100 results.
        max_results = max(10, min(limit, 100))
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                resp = self.client.search_recent_tweets(
                    query=f"{query} -is:retweet lang:en",
                    max_results=max_results,
                    tweet_fields=["created_at", "public_metrics", "author_id"],
                )
                tweets = resp.data or []
                return [self._process_tweet(t, query) for t in tweets][:limit]
            except tweepy.TooManyRequests:
                self.logger.warning(
                    f"Twitter rate limit hit, backing off {delay:.0f}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                time.sleep(delay)
                delay *= 2
            except Exception as exc:  # noqa: BLE001 - network/API errors
                self.logger.error(f"Twitter search failed: {exc}")
                return []
        self.logger.error("Twitter search exhausted retries")
        return []

    def _process_tweet(self, tweet: Any, query: str) -> dict[str, Any]:
        metrics = getattr(tweet, "public_metrics", None) or {}
        created = getattr(tweet, "created_at", None)
        timestamp = created.isoformat() if created else datetime.now(UTC).isoformat()
        return {
            "text": tweet.text,
            "source": "twitter",
            "timestamp": timestamp,
            "url": f"https://twitter.com/i/web/status/{tweet.id}",
            "author": str(getattr(tweet, "author_id", "unknown")),
            "score": int(metrics.get("like_count", 0)),
            "query": query,
            "language": "en",
        }

    def get_hot_posts(self, topic: str, limit: int = 25) -> list[dict[str, Any]]:
        """Get recent popular tweets for a topic (Reddit-compatible signature)."""
        return self._search(topic, limit)

    def search_posts(self, query: str, limit: int = 100) -> list[dict[str, Any]]:
        """Search recent tweets for a query."""
        return self._search(query, limit)

    def get_recent_posts(self, limit: int = 100) -> list[dict[str, Any]]:
        """Sample recent tweets across default topics."""
        posts: list[dict[str, Any]] = []
        for topic in _DEFAULT_TOPICS:
            posts.extend(self._search(topic, max(10, limit // len(_DEFAULT_TOPICS))))
            if len(posts) >= limit:
                break
        return posts[:limit]


def create_twitter_stream() -> TwitterStream:
    """Create a Twitter stream instance."""
    return TwitterStream()
