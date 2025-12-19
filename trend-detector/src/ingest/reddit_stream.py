"""
Reddit data ingestion using PRAW.
"""

import time
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

import praw

from shared.lib.config import get_settings
from shared.lib.logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class RedditStream(LoggerMixin):
    """Reddit data stream using PRAW."""

    def __init__(self):
        self.settings = get_settings()
        self.reddit = None
        self.subreddits = [
            "technology",
            "programming",
            "MachineLearning",
            "artificial",
            "cryptocurrency",
            "bitcoin",
            "ethereum",
            "crypto",
            "worldnews",
            "news",
            "politics",
            "science",
            "health",
            "fitness",
            "mentalhealth",
            "medicine",
            "environment",
            "climate",
            "renewable",
            "sustainability",
            "gaming",
            "movies",
            "music",
            "sports",
        ]

        # Initialize Reddit client
        self._initialize_reddit()

    def _initialize_reddit(self) -> None:
        """Initialize Reddit client."""

        try:
            if (
                self.settings.reddit_client_id
                and self.settings.reddit_client_secret
                and self.settings.reddit_user_agent
            ):
                self.reddit = praw.Reddit(
                    client_id=self.settings.reddit_client_id,
                    client_secret=self.settings.reddit_client_secret,
                    user_agent=self.settings.reddit_user_agent,
                )

                # Test connection
                self.reddit.read_only = True
                _ = self.reddit.subreddit(
                    "test"
                ).id  # This will raise an exception if credentials are invalid

                self.logger.info("Reddit client initialized successfully")
            else:
                self.logger.warning(
                    "Reddit credentials not provided, falling back to mock stream"
                )
                self.reddit = None

        except Exception as e:
            self.logger.error(f"Failed to initialize Reddit client: {e}")
            self.reddit = None

    def stream_posts(
        self, subreddits: list[str] | None = None, limit: int = 100
    ) -> Iterator[dict[str, Any]]:
        """Stream posts from Reddit."""

        if self.reddit is None:
            self.logger.warning(
                "Reddit client not available, no posts will be streamed"
            )
            return

        subreddits = subreddits or self.subreddits

        try:
            # Create multireddit
            multireddit = self.reddit.subreddit("+".join(subreddits))

            self.logger.info(f"Streaming posts from subreddits: {subreddits}")

            for submission in multireddit.new(limit=limit):
                try:
                    post = self._process_submission(submission)
                    if post:
                        yield post

                except Exception as e:
                    self.logger.warning(
                        f"Error processing submission {submission.id}: {e}"
                    )
                    continue

        except Exception as e:
            self.logger.error(f"Error streaming Reddit posts: {e}")

    def stream_comments(
        self, subreddits: list[str] | None = None, limit: int = 100
    ) -> Iterator[dict[str, Any]]:
        """Stream comments from Reddit."""

        if self.reddit is None:
            self.logger.warning(
                "Reddit client not available, no comments will be streamed"
            )
            return

        subreddits = subreddits or self.subreddits

        try:
            # Create multireddit
            multireddit = self.reddit.subreddit("+".join(subreddits))

            self.logger.info(f"Streaming comments from subreddits: {subreddits}")

            for comment in multireddit.comments(limit=limit):
                try:
                    post = self._process_comment(comment)
                    if post:
                        yield post

                except Exception as e:
                    self.logger.warning(f"Error processing comment {comment.id}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error streaming Reddit comments: {e}")

    def get_hot_posts(self, subreddit: str, limit: int = 25) -> list[dict[str, Any]]:
        """Get hot posts from a specific subreddit."""

        if self.reddit is None:
            self.logger.warning("Reddit client not available")
            return []

        try:
            posts = []
            subreddit_obj = self.reddit.subreddit(subreddit)

            for submission in subreddit_obj.hot(limit=limit):
                post = self._process_submission(submission)
                if post:
                    posts.append(post)

            self.logger.info(f"Retrieved {len(posts)} hot posts from r/{subreddit}")
            return posts

        except Exception as e:
            self.logger.error(f"Error getting hot posts from r/{subreddit}: {e}")
            return []

    def search_posts(
        self, query: str, subreddit: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Search for posts matching a query."""

        if self.reddit is None:
            self.logger.warning("Reddit client not available")
            return []

        try:
            posts = []

            if subreddit:
                # Search within specific subreddit
                subreddit_obj = self.reddit.subreddit(subreddit)
                for submission in subreddit_obj.search(query, limit=limit):
                    post = self._process_submission(submission)
                    if post:
                        posts.append(post)
            else:
                # Search across all subreddits
                for submission in self.reddit.subreddit("all").search(
                    query, limit=limit
                ):
                    post = self._process_submission(submission)
                    if post:
                        posts.append(post)

            self.logger.info(f"Found {len(posts)} posts for query: {query}")
            return posts

        except Exception as e:
            self.logger.error(f"Error searching posts for query '{query}': {e}")
            return []

    def _process_submission(self, submission) -> dict[str, Any] | None:
        """Process a Reddit submission into our format."""

        try:
            # Extract text content
            text = ""
            if hasattr(submission, "selftext") and submission.selftext:
                text = submission.selftext
            elif hasattr(submission, "title"):
                text = submission.title

            # Skip if no text content
            if not text or len(text.strip()) < 10:
                return None

            # Process timestamp
            timestamp = datetime.fromtimestamp(submission.created_utc)

            post = {
                "text": text,
                "source": "reddit",
                "timestamp": timestamp.isoformat(),
                "url": f"https://reddit.com{submission.permalink}",
                "author": str(submission.author) if submission.author else "deleted",
                "score": submission.score,
                "subreddit": submission.subreddit.display_name,
                "post_id": submission.id,
                "title": getattr(submission, "title", ""),
                "num_comments": submission.num_comments,
                "upvote_ratio": getattr(submission, "upvote_ratio", 0.0),
                "language": "en",  # Reddit is primarily English
                "domain": getattr(submission, "domain", "reddit.com"),
            }

            return post

        except Exception as e:
            self.logger.warning(f"Error processing submission: {e}")
            return None

    def _process_comment(self, comment) -> dict[str, Any] | None:
        """Process a Reddit comment into our format."""

        try:
            # Skip deleted or removed comments
            if not comment.body or comment.body in ["[deleted]", "[removed]"]:
                return None

            # Skip very short comments
            if len(comment.body.strip()) < 5:
                return None

            # Process timestamp
            timestamp = datetime.fromtimestamp(comment.created_utc)

            post = {
                "text": comment.body,
                "source": "reddit_user",
                "timestamp": timestamp.isoformat(),
                "url": f"https://reddit.com{comment.permalink}",
                "author": str(comment.author) if comment.author else "deleted",
                "score": comment.score,
                "subreddit": comment.subreddit.display_name,
                "post_id": comment.id,
                "parent_id": getattr(comment, "parent_id", None),
                "language": "en",
                "is_comment": True,
            }

            return post

        except Exception as e:
            self.logger.warning(f"Error processing comment: {e}")
            return None

    def get_subreddit_info(self, subreddit_name: str) -> dict[str, Any] | None:
        """Get information about a subreddit."""

        if self.reddit is None:
            return None

        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            info = {
                "name": subreddit.display_name,
                "title": subreddit.title,
                "description": subreddit.description,
                "subscribers": subreddit.subscribers,
                "active_users": subreddit.active_user_count,
                "created_utc": datetime.fromtimestamp(
                    subreddit.created_utc
                ).isoformat(),
                "public_description": subreddit.public_description,
                "language": subreddit.lang,
            }

            return info

        except Exception as e:
            self.logger.error(
                f"Error getting subreddit info for r/{subreddit_name}: {e}"
            )
            return None

    def monitor_subreddit(
        self, subreddit_name: str, duration_minutes: int = 60
    ) -> Iterator[dict[str, Any]]:
        """Monitor a subreddit for new posts in real-time."""

        if self.reddit is None:
            self.logger.warning("Reddit client not available")
            return

        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            self.logger.info(
                f"Monitoring r/{subreddit_name} for {duration_minutes} minutes"
            )

            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            seen_posts = set()
            retry_delay = 60

            while datetime.now() < end_time:
                try:
                    # Get new posts
                    for submission in subreddit.new(limit=10):
                        if submission.id not in seen_posts:
                            post = self._process_submission(submission)
                            if post:
                                seen_posts.add(submission.id)
                                yield post

                    # Wait before next check
                    time.sleep(30)  # Check every 30 seconds
                    retry_delay = 60  # Reset delay on success

                except Exception as e:
                    self.logger.warning(f"Error during monitoring: {e}")
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay = min(
                        retry_delay * 2, 3600
                    )  # Exponential backoff up to 1 hour

        except Exception as e:
            self.logger.error(f"Error monitoring subreddit r/{subreddit_name}: {e}")

    def get_trending_subreddits(self) -> list[dict[str, Any]]:
        """Get trending subreddits."""

        if self.reddit is None:
            return []

        try:
            trending = []

            # Get popular subreddits
            for subreddit in self.reddit.subreddits.popular(limit=50):
                info = self.get_subreddit_info(subreddit.display_name)
                if info:
                    trending.append(info)

            return trending

        except Exception as e:
            self.logger.error(f"Error getting trending subreddits: {e}")
            return []


def create_reddit_stream() -> RedditStream:
    """Create a Reddit stream instance."""
    return RedditStream()
