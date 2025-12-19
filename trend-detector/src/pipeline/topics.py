"""
Trend analysis and scoring utilities.
"""

from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from shared.lib.logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class TrendAnalyzer(LoggerMixin):
    """Analyze and score trending topics."""

    def __init__(
        self,
        novelty_weight: float = 0.3,
        volume_weight: float = 0.3,
        velocity_weight: float = 0.2,
        burstiness_weight: float = 0.2,
    ):
        self.novelty_weight = novelty_weight
        self.volume_weight = volume_weight
        self.velocity_weight = velocity_weight
        self.burstiness_weight = burstiness_weight

        # Historical data storage
        self.topic_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.centroid_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=50))

        self.logger.info("Trend analyzer initialized")

    def calculate_trend_score(
        self,
        topic_id: int,
        current_posts: list[dict[str, Any]],
        embeddings: np.ndarray,
        time_window_hours: int = 24,
    ) -> dict[str, Any]:
        """Calculate comprehensive trend score for a topic."""

        try:
            # Calculate individual components
            volume_score = self._calculate_volume_score(
                current_posts, time_window_hours
            )
            velocity_score = self._calculate_velocity_score(
                topic_id, current_posts, time_window_hours
            )
            novelty_score = self._calculate_novelty_score(topic_id, embeddings)
            burstiness_score = self._calculate_burstiness_score(
                topic_id, current_posts, time_window_hours
            )

            # Combine scores
            trend_score = (
                self.volume_weight * volume_score
                + self.velocity_weight * velocity_score
                + self.novelty_weight * novelty_score
                + self.burstiness_weight * burstiness_score
            )

            # Normalize to 0-1 range
            trend_score = max(0, min(1, trend_score))

            # Update history
            self._update_topic_history(
                topic_id,
                {
                    "timestamp": datetime.now(),
                    "volume_score": volume_score,
                    "velocity_score": velocity_score,
                    "novelty_score": novelty_score,
                    "burstiness_score": burstiness_score,
                    "trend_score": trend_score,
                    "post_count": len(current_posts),
                },
            )

            return {
                "trend_score": trend_score,
                "volume_score": volume_score,
                "velocity_score": velocity_score,
                "novelty_score": novelty_score,
                "burstiness_score": burstiness_score,
                "post_count": len(current_posts),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(
                f"Error calculating trend score for topic {topic_id}: {e}"
            )
            return {
                "trend_score": 0.0,
                "volume_score": 0.0,
                "velocity_score": 0.0,
                "novelty_score": 0.0,
                "burstiness_score": 0.0,
                "post_count": 0,
                "timestamp": datetime.now().isoformat(),
            }

    def _calculate_volume_score(
        self, posts: list[dict[str, Any]], time_window_hours: int
    ) -> float:
        """Calculate volume-based trend score."""

        if not posts:
            return 0.0

        try:
            # Count posts in time window
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)

            recent_posts = []
            for post in posts:
                try:
                    post_time = datetime.fromisoformat(
                        post.get("timestamp", "").replace("Z", "+00:00")
                    )
                    if post_time >= cutoff_time:
                        recent_posts.append(post)
                except Exception:
                    continue

            # Normalize by time window (posts per hour)
            volume_rate = len(recent_posts) / time_window_hours

            # Apply sigmoid normalization
            volume_score = 1 / (1 + np.exp(-volume_rate / 10))

            return float(volume_score)

        except Exception as e:
            self.logger.error(f"Error calculating volume score: {e}")
            return 0.0

    def _calculate_velocity_score(
        self, topic_id: int, posts: list[dict[str, Any]], time_window_hours: int
    ) -> float:
        """Calculate velocity-based trend score (rate of change)."""

        try:
            # Get historical data
            history = self.topic_history[topic_id]

            if len(history) < 2:
                return 0.5  # Default moderate score for new topics

            # Calculate rate of change
            recent_data = list(history)[-min(5, len(history)) :]

            if len(recent_data) >= 2:
                # Calculate growth rate
                old_count = recent_data[0].get("post_count", 0)
                new_count = recent_data[-1].get("post_count", 0)

                if old_count > 0:
                    growth_rate = (new_count - old_count) / old_count
                    # Normalize growth rate
                    velocity_score = 1 / (1 + np.exp(-growth_rate))
                else:
                    velocity_score = 1.0 if new_count > 0 else 0.0
            else:
                velocity_score = 0.5

            return float(velocity_score)

        except Exception as e:
            self.logger.error(f"Error calculating velocity score: {e}")
            return 0.0

    def _calculate_novelty_score(self, topic_id: int, embeddings: np.ndarray) -> float:
        """Calculate novelty score based on embedding similarity to historical centroids."""

        try:
            if len(embeddings) == 0:
                return 0.0

            # Calculate current centroid
            current_centroid = np.mean(embeddings, axis=0)

            # Get historical centroids
            centroid_history = self.centroid_history[topic_id]

            if len(centroid_history) == 0:
                # First time seeing this topic - high novelty
                self.centroid_history[topic_id].append(current_centroid)
                return 1.0

            # Calculate similarity to most recent historical centroid
            recent_centroid = centroid_history[-1]
            similarity = np.dot(current_centroid, recent_centroid) / (
                np.linalg.norm(current_centroid) * np.linalg.norm(recent_centroid)
            )

            # Novelty is inverse of similarity
            novelty_score = 1 - similarity

            # Update centroid history
            self.centroid_history[topic_id].append(current_centroid)

            return float(max(0, novelty_score))

        except Exception as e:
            self.logger.error(f"Error calculating novelty score: {e}")
            return 0.0

    def _calculate_burstiness_score(
        self, topic_id: int, posts: list[dict[str, Any]], time_window_hours: int
    ) -> float:
        """Calculate burstiness score using Kleinberg's burst detection algorithm."""

        try:
            if not posts:
                return 0.0

            # Convert posts to time series
            post_times = []
            for post in posts:
                try:
                    post_time = datetime.fromisoformat(
                        post.get("timestamp", "").replace("Z", "+00:00")
                    )
                    post_times.append(post_time)
                except Exception:
                    continue

            if len(post_times) < 2:
                return 0.0

            # Sort by time
            post_times.sort()

            # Calculate inter-arrival times
            inter_arrivals = []
            for i in range(1, len(post_times)):
                delta = (
                    post_times[i] - post_times[i - 1]
                ).total_seconds() / 3600  # Convert to hours
                inter_arrivals.append(delta)

            if len(inter_arrivals) == 0:
                return 0.0

            # Calculate burstiness using coefficient of variation
            mean_interval = np.mean(inter_arrivals)
            std_interval = np.std(inter_arrivals)

            if mean_interval == 0:
                return 1.0  # Maximum burstiness

            burstiness = (std_interval - mean_interval) / (std_interval + mean_interval)

            # Normalize to 0-1 range
            burstiness_score = max(0, min(1, (burstiness + 1) / 2))

            return float(burstiness_score)

        except Exception as e:
            self.logger.error(f"Error calculating burstiness score: {e}")
            return 0.0

    def _update_topic_history(self, topic_id: int, data: dict[str, Any]) -> None:
        """Update topic history."""

        try:
            self.topic_history[topic_id].append(data)
        except Exception as e:
            self.logger.error(f"Error updating topic history: {e}")

    def get_trending_topics(
        self,
        topics_data: dict[int, Any],
        embeddings_data: dict[int, np.ndarray],
        top_k: int = 10,
        time_window_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """Get top trending topics."""

        try:
            trending_scores = []

            for topic_id, posts in topics_data.items():
                embeddings = embeddings_data.get(topic_id, np.array([]))

                # Calculate trend score
                trend_info = self.calculate_trend_score(
                    topic_id, posts, embeddings, time_window_hours
                )

                # Extract keywords from posts
                all_text = " ".join([post.get("text", "") for post in posts])
                keywords = self._extract_keywords(all_text)

                trending_info = {
                    "topic_id": topic_id,
                    "trend_score": trend_info["trend_score"],
                    "volume_score": trend_info["volume_score"],
                    "velocity_score": trend_info["velocity_score"],
                    "novelty_score": trend_info["novelty_score"],
                    "burstiness_score": trend_info["burstiness_score"],
                    "post_count": trend_info["post_count"],
                    "keywords": keywords,
                    "representative_posts": [
                        post.get("text", "")[:100] for post in posts[:3]
                    ],
                    "timestamp": trend_info["timestamp"],
                }

                trending_scores.append(trending_info)

            # Sort by trend score
            trending_scores.sort(key=lambda x: x["trend_score"], reverse=True)

            return trending_scores[:top_k]

        except Exception as e:
            self.logger.error(f"Error getting trending topics: {e}")
            return []

    def _extract_keywords(self, text: str, top_k: int = 5) -> list[str]:
        """Extract keywords from text (simplified version)."""

        try:
            # Simple keyword extraction (in practice, you'd use more sophisticated methods)
            words = text.lower().split()

            # Filter common words
            stop_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "can",
                "this",
                "that",
                "these",
                "those",
                "i",
                "you",
                "he",
                "she",
                "it",
                "we",
                "they",
                "me",
                "him",
                "her",
                "us",
                "them",
            }

            filtered_words = [
                word for word in words if word not in stop_words and len(word) > 2
            ]

            # Count word frequencies
            word_counts = defaultdict(int)
            for word in filtered_words:
                word_counts[word] += 1

            # Get top keywords
            top_keywords = sorted(
                word_counts.items(), key=lambda x: x[1], reverse=True
            )[:top_k]

            return [word for word, count in top_keywords]

        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            return []

    def analyze_topic_evolution(
        self, topic_id: int, time_window_hours: int = 168
    ) -> dict[str, Any]:
        """Analyze how a topic evolves over time."""

        try:
            history = list(self.topic_history[topic_id])

            if len(history) < 2:
                return {
                    "topic_id": topic_id,
                    "evolution_data": [],
                    "trend_direction": "stable",
                    "peak_activity": None,
                    "growth_rate": 0.0,
                }

            # Filter to time window
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            recent_history = [h for h in history if h["timestamp"] >= cutoff_time]

            if len(recent_history) < 2:
                return {
                    "topic_id": topic_id,
                    "evolution_data": [],
                    "trend_direction": "stable",
                    "peak_activity": None,
                    "growth_rate": 0.0,
                }

            # Analyze trend direction
            first_score = recent_history[0]["trend_score"]
            last_score = recent_history[-1]["trend_score"]

            if last_score > first_score * 1.1:
                trend_direction = "increasing"
            elif last_score < first_score * 0.9:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"

            # Find peak activity
            peak_activity = max(recent_history, key=lambda x: x["trend_score"])

            # Calculate growth rate
            growth_rate = (
                (last_score - first_score) / first_score if first_score > 0 else 0.0
            )

            return {
                "topic_id": topic_id,
                "evolution_data": recent_history,
                "trend_direction": trend_direction,
                "peak_activity": peak_activity,
                "growth_rate": growth_rate,
                "time_window_hours": time_window_hours,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing topic evolution: {e}")
            return {
                "topic_id": topic_id,
                "evolution_data": [],
                "trend_direction": "unknown",
                "peak_activity": None,
                "growth_rate": 0.0,
            }

    def get_topic_statistics(self) -> dict[str, Any]:
        """Get overall statistics about tracked topics."""

        try:
            total_topics = len(self.topic_history)
            total_centroids = len(self.centroid_history)

            # Calculate average trend scores
            all_trend_scores = []
            for _topic_id, history in self.topic_history.items():
                if history:
                    latest_score = history[-1]["trend_score"]
                    all_trend_scores.append(latest_score)

            avg_trend_score = np.mean(all_trend_scores) if all_trend_scores else 0.0

            # Find most active topics
            most_active = []
            for topic_id, history in self.topic_history.items():
                if history:
                    latest_data = history[-1]
                    most_active.append(
                        {
                            "topic_id": topic_id,
                            "trend_score": latest_data["trend_score"],
                            "post_count": latest_data["post_count"],
                        }
                    )

            most_active.sort(key=lambda x: x["trend_score"], reverse=True)

            return {
                "total_topics_tracked": total_topics,
                "total_centroids": total_centroids,
                "average_trend_score": float(avg_trend_score),
                "most_active_topics": most_active[:10],
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error getting topic statistics: {e}")
            return {
                "total_topics_tracked": 0,
                "total_centroids": 0,
                "average_trend_score": 0.0,
                "most_active_topics": [],
                "last_updated": datetime.now().isoformat(),
            }
