"""
Topic clustering using BERTopic and HDBSCAN.
"""

from typing import Any

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from shared.lib.logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class TopicClusterer(LoggerMixin):
    """Topic clustering using BERTopic."""

    def __init__(
        self,
        min_topic_size: int = 10,
        n_neighbors: int = 15,
        n_components: int = 5,
        min_cluster_size: int = 10,
        min_samples: int = 5,
    ):
        self.min_topic_size = min_topic_size
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

        self.model = None
        self.topics = None
        self.probs = None
        self.topic_info = None

        # Initialize model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize BERTopic model with custom parameters."""

        try:
            # Configure UMAP for dimensionality reduction
            umap_model = UMAP(
                n_neighbors=self.n_neighbors,
                n_components=self.n_components,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            )

            # Configure HDBSCAN for clustering
            hdbscan_model = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
            )

            # Configure vectorizer for topic representation
            vectorizer_model = CountVectorizer(
                stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.95
            )

            # Initialize BERTopic
            self.model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                min_topic_size=self.min_topic_size,
                verbose=True,
            )

            self.logger.info("BERTopic model initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize BERTopic model: {e}")
            raise

    def fit_transform(self, documents: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Fit model and transform documents to topics."""

        if not documents:
            self.logger.warning("No documents provided for clustering")
            return np.array([]), np.array([])

        self.logger.info(f"Clustering {len(documents)} documents")

        try:
            # Fit and transform
            topics, probs = self.model.fit_transform(documents)

            # Store results
            self.topics = np.array(topics)
            self.probs = np.array(probs) if probs is not None else None

            # Get topic information
            self.topic_info = self.model.get_topic_info()

            self.logger.info(
                f"Clustering completed. Found {len(self.topic_info)} topics"
            )

            return self.topics, self.probs

        except Exception as e:
            self.logger.error(f"Error during clustering: {e}")
            raise

    def transform(self, documents: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Transform new documents to existing topics."""

        if self.model is None:
            raise ValueError("Model must be fitted before transform")

        if not documents:
            return np.array([]), np.array([])

        self.logger.info(f"Transforming {len(documents)} new documents")

        try:
            topics, probs = self.model.transform(documents)
            return np.array(topics), np.array(probs) if probs is not None else None

        except Exception as e:
            self.logger.error(f"Error transforming documents: {e}")
            raise

    def get_topic_info(self) -> pd.DataFrame:
        """Get topic information."""

        if self.topic_info is None:
            self.logger.warning(
                "No topic information available. Run fit_transform first."
            )
            return pd.DataFrame()

        return self.topic_info.copy()

    def get_topic_words(
        self, topic_id: int, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Get top words for a specific topic."""

        if self.model is None:
            return []

        try:
            words = self.model.get_topic(topic_id)
            return words[:top_k] if words else []

        except Exception as e:
            self.logger.error(f"Error getting words for topic {topic_id}: {e}")
            return []

    def get_document_topics(self, documents: list[str]) -> list[dict[str, Any]]:
        """Get topic assignments for documents."""

        if self.topics is None:
            return []

        results = []

        for i, (doc, topic) in enumerate(zip(documents, self.topics, strict=False)):
            result = {
                "document_index": i,
                "document": doc,
                "topic_id": int(topic),
                "topic_words": self.get_topic_words(topic),
                "probability": (
                    float(self.probs[i].max()) if self.probs is not None else 0.0
                ),
            }
            results.append(result)

        return results

    def get_topic_statistics(self) -> dict[str, Any]:
        """Get topic clustering statistics."""

        if self.topics is None:
            return {}

        try:
            unique_topics, counts = np.unique(self.topics, return_counts=True)

            stats = {
                "total_documents": len(self.topics),
                "num_topics": len(unique_topics),
                "topic_distribution": dict(
                    zip(unique_topics.tolist(), counts.tolist(), strict=False)
                ),
                "noise_documents": (
                    int(counts[unique_topics == -1][0]) if -1 in unique_topics else 0
                ),
                "avg_documents_per_topic": (
                    float(np.mean(counts[unique_topics != -1]))
                    if len(unique_topics) > 1
                    else 0.0
                ),
                "largest_topic_size": (
                    int(np.max(counts[unique_topics != -1]))
                    if len(unique_topics) > 1
                    else 0
                ),
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error computing topic statistics: {e}")
            return {}

    def visualize_topics(self, save_path: str | None = None) -> Any:
        """Visualize topics using BERTopic's built-in visualization."""

        if self.model is None:
            self.logger.warning("Model not fitted, cannot visualize topics")
            return None

        try:
            fig = self.model.visualize_topics()

            if save_path:
                fig.write_html(save_path)
                self.logger.info(f"Topic visualization saved to {save_path}")

            return fig

        except Exception as e:
            self.logger.error(f"Error visualizing topics: {e}")
            return None

    def visualize_documents(
        self, documents: list[str], save_path: str | None = None
    ) -> Any:
        """Visualize documents in topic space."""

        if self.model is None:
            self.logger.warning("Model not fitted, cannot visualize documents")
            return None

        try:
            fig = self.model.visualize_documents(documents)

            if save_path:
                fig.write_html(save_path)
                self.logger.info(f"Document visualization saved to {save_path}")

            return fig

        except Exception as e:
            self.logger.error(f"Error visualizing documents: {e}")
            return None

    def find_similar_topics(
        self, topic_id: int, top_k: int = 5
    ) -> list[tuple[int, float]]:
        """Find topics similar to the given topic."""

        if self.model is None:
            return []

        try:
            similar_topics = self.model.find_topics(
                self.model.get_topic(topic_id)[0][0],  # Use top word as query
                top_k=top_k,
            )

            return similar_topics

        except Exception as e:
            self.logger.error(f"Error finding similar topics: {e}")
            return []

    def update_topics(
        self,
        documents: list[str],
        topics: list[int],
        n_gram_range: tuple[int, int] = (1, 2),
    ) -> None:
        """Update topic representations with new data."""

        if self.model is None:
            self.logger.warning("Model not fitted, cannot update topics")
            return

        try:
            self.model.update_topics(documents, topics, n_gram_range=n_gram_range)
            self.logger.info("Topics updated successfully")

        except Exception as e:
            self.logger.error(f"Error updating topics: {e}")

    def reduce_topics(self, documents: list[str], nr_topics: int) -> None:
        """Reduce the number of topics."""

        if self.model is None:
            self.logger.warning("Model not fitted, cannot reduce topics")
            return

        try:
            self.model.reduce_topics(documents, nr_topics=nr_topics)
            self.logger.info(f"Topics reduced to {nr_topics}")

        except Exception as e:
            self.logger.error(f"Error reducing topics: {e}")


class ClusterAnalyzer(LoggerMixin):
    """Analyze clustering results and extract insights."""

    def __init__(self, clusterer: TopicClusterer):
        self.clusterer = clusterer

    def analyze_topic_evolution(
        self, documents: list[str], timestamps: list[str]
    ) -> dict[str, Any]:
        """Analyze how topics evolve over time."""

        if not documents or not timestamps:
            return {}

        try:
            # Create DataFrame
            df = pd.DataFrame(
                {
                    "document": documents,
                    "timestamp": pd.to_datetime(timestamps),
                    "topic": self.clusterer.topics,
                }
            )

            # Group by topic and time
            df["hour"] = df["timestamp"].dt.floor("H")
            topic_evolution = df.groupby(["topic", "hour"]).size().unstack(fill_value=0)

            # Calculate growth rates
            growth_rates = {}
            for topic in topic_evolution.index:
                if topic == -1:  # Skip noise
                    continue

                topic_series = topic_evolution.loc[topic]
                if len(topic_series) > 1:
                    growth_rate = (topic_series.iloc[-1] - topic_series.iloc[0]) / len(
                        topic_series
                    )
                    growth_rates[topic] = growth_rate

            return {
                "topic_evolution": topic_evolution.to_dict(),
                "growth_rates": growth_rates,
                "trending_topics": sorted(
                    growth_rates.items(), key=lambda x: x[1], reverse=True
                )[:10],
            }

        except Exception as e:
            self.logger.error(f"Error analyzing topic evolution: {e}")
            return {}

    def extract_topic_keywords(self, top_k: int = 10) -> dict[int, list[str]]:
        """Extract keywords for each topic."""

        if self.clusterer.topics is None:
            return {}

        try:
            unique_topics = np.unique(self.clusterer.topics)
            topic_keywords = {}

            for topic in unique_topics:
                if topic == -1:  # Skip noise
                    continue

                words = self.clusterer.get_topic_words(topic, top_k)
                topic_keywords[topic] = [word for word, _ in words]

            return topic_keywords

        except Exception as e:
            self.logger.error(f"Error extracting topic keywords: {e}")
            return {}

    def compute_topic_coherence(self) -> dict[int, float]:
        """Compute topic coherence scores."""

        # This is a simplified coherence measure
        # In practice, you might want to use more sophisticated measures

        if self.clusterer.topics is None:
            return {}

        try:
            unique_topics = np.unique(self.clusterer.topics)
            coherence_scores = {}

            for topic in unique_topics:
                if topic == -1:  # Skip noise
                    continue

                words = self.clusterer.get_topic_words(topic, 10)
                if len(words) >= 2:
                    # Simple coherence: average pairwise word similarity
                    word_similarities = []
                    for i, (_word1, _) in enumerate(words):
                        for _j, (_word2, _) in enumerate(words[i + 1 :], i + 1):
                            # This is a placeholder - you'd implement actual word similarity
                            similarity = 0.5  # Placeholder
                            word_similarities.append(similarity)

                    coherence_scores[topic] = (
                        np.mean(word_similarities) if word_similarities else 0.0
                    )

            return coherence_scores

        except Exception as e:
            self.logger.error(f"Error computing topic coherence: {e}")
            return {}


def create_topic_clusterer(**kwargs) -> TopicClusterer:
    """Create a topic clusterer instance."""
    return TopicClusterer(**kwargs)


def create_cluster_analyzer(clusterer: TopicClusterer) -> ClusterAnalyzer:
    """Create a cluster analyzer instance."""
    return ClusterAnalyzer(clusterer)
