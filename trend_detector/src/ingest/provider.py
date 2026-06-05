"""
Stream provider that selects live or demo social data sources.

The unified API reads social posts through this provider. Live sources activate
automatically when credentials are configured:

* **Reddit** when ``REDDIT_CLIENT_ID`` / ``REDDIT_CLIENT_SECRET`` are set.
* **Twitter/X** when ``X_BEARER_TOKEN`` is set.

If no live source is available it falls back to the bundled demo stream, so the
API works out of the box. Every post returned is automatically PII-scrubbed.
"""

from __future__ import annotations

from typing import Any

from shared.lib.config import get_settings
from shared.lib.io_utils import PIIScrubber
from shared.lib.logging import get_logger

from .mock_stream import MockStream

logger = get_logger(__name__)

# Subreddits / topics sampled when no explicit query is supplied.
_DEFAULT_SUBREDDITS = ["technology", "science", "health", "news"]


class StreamProvider:
    """Serve social posts from live sources when available, else demo data."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._mock = MockStream()
        self._scrubber = PIIScrubber()
        self._reddit: Any | None = None
        self._twitter: Any | None = None
        self._reddit_tried = False
        self._twitter_tried = False

    @property
    def live(self) -> bool:
        """Whether any live source is active."""
        return self._ensure_reddit() is not None or self._ensure_twitter() is not None

    @property
    def active_sources(self) -> list[str]:
        sources = []
        if self._ensure_reddit() is not None:
            sources.append("reddit")
        if self._ensure_twitter() is not None:
            sources.append("twitter")
        return sources or ["demo"]

    def _ensure_reddit(self) -> Any | None:
        if self._reddit_tried:
            return self._reddit
        self._reddit_tried = True
        if not (
            self._settings.reddit_client_id and self._settings.reddit_client_secret
        ):
            logger.info("Reddit credentials not set; Reddit source disabled")
            return None
        try:
            from .reddit_stream import RedditStream

            self._reddit = RedditStream()
            logger.info("Live Reddit stream initialized")
        except Exception as exc:  # noqa: BLE001 - optional live dependency
            logger.warning(f"Reddit stream unavailable: {exc}")
            self._reddit = None
        return self._reddit

    def _ensure_twitter(self) -> Any | None:
        if self._twitter_tried:
            return self._twitter
        self._twitter_tried = True
        if not self._settings.x_bearer_token:
            logger.info("X_BEARER_TOKEN not set; Twitter source disabled")
            return None
        try:
            from .twitter_stream import TwitterStream

            stream = TwitterStream()
            self._twitter = stream if stream.client is not None else None
            if self._twitter is not None:
                logger.info("Live Twitter stream initialized")
        except Exception as exc:  # noqa: BLE001 - optional live dependency
            logger.warning(f"Twitter stream unavailable: {exc}")
            self._twitter = None
        return self._twitter

    def _scrub_posts(self, posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Automatically redact PII from post text and author."""
        scrubbed = []
        for post in posts:
            clean = dict(post)
            if isinstance(clean.get("text"), str):
                clean["text"] = self._scrubber.scrub(clean["text"])
            if isinstance(clean.get("author"), str):
                clean["author"] = self._scrubber.scrub(clean["author"])
            scrubbed.append(clean)
        return scrubbed

    def get_recent_posts(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent posts from live sources, or demo data (PII-scrubbed)."""
        posts: list[dict[str, Any]] = []

        reddit = self._ensure_reddit()
        if reddit is not None:
            try:
                for subreddit in _DEFAULT_SUBREDDITS:
                    posts.extend(reddit.get_hot_posts(subreddit, limit=limit))
                    if len(posts) >= limit:
                        break
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Reddit fetch failed: {exc}")

        twitter = self._ensure_twitter()
        if twitter is not None and len(posts) < limit:
            try:
                posts.extend(twitter.get_recent_posts(limit - len(posts)))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Twitter fetch failed: {exc}")

        if not posts:
            posts = self._mock.get_recent_posts(limit)

        return self._scrub_posts(posts[:limit])

    def get_trending_topics(self, window: int = 24) -> list[dict[str, Any]]:
        """Return demo trending topics (live clustering runs in ``/top``)."""
        return self._mock.get_trending_topics(window)


_provider: StreamProvider | None = None


def get_stream_provider() -> StreamProvider:
    """Return the process-wide stream provider singleton."""
    global _provider
    if _provider is None:
        _provider = StreamProvider()
    return _provider
