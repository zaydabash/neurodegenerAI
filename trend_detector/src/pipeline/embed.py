"""
Text embedding utilities using sentence transformers.
"""

from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from shared.lib.config import get_settings
from shared.lib.logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class EmbeddingGenerator(LoggerMixin):
    """Text embedding generator using sentence transformers."""

    def __init__(self, model_name: str | None = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.embedding_model
        self.model = None
        self.embedding_cache = {}

        # Initialize model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the embedding model."""

        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")

            # Load model with device optimization
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)

            self.logger.info(f"Embedding model loaded successfully on {device}")

        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            # Fallback to a simpler model
            try:
                self.model_name = "all-MiniLM-L6-v2"
                self.model = SentenceTransformer(self.model_name)
                self.logger.info(f"Fallback model loaded: {self.model_name}")
            except Exception as e2:
                self.logger.error(f"Failed to load fallback model: {e2}")
                raise

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""

        if not texts:
            return np.array([])

        self.logger.info(f"Generating embeddings for {len(texts)} texts")

        try:
            # Remove duplicates and track indices
            unique_texts = []
            text_to_index = {}
            indices = []

            for _i, text in enumerate(texts):
                if text not in text_to_index:
                    text_to_index[text] = len(unique_texts)
                    unique_texts.append(text)
                indices.append(text_to_index[text])

            # Generate embeddings for unique texts
            unique_embeddings = self.model.encode(
                unique_texts,
                batch_size=batch_size,
                show_progress_bar=len(unique_texts) > 100,
                convert_to_numpy=True,
            )

            # Reconstruct embeddings for all texts
            embeddings = unique_embeddings[indices]

            self.logger.info(f"Generated embeddings with shape: {embeddings.shape}")

            return embeddings

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise

    def embed_single_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""

        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0]

        except Exception as e:
            self.logger.error(f"Error generating embedding for text: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""

        if self.model is None:
            return 0

        try:
            # Get embedding dimension by encoding a test text
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            return test_embedding.shape[1]

        except Exception as e:
            self.logger.error(f"Error getting embedding dimension: {e}")
            return 0

    def compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""

        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return 0.0

    def find_similar_texts(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[int, float]]:
        """Find most similar texts to a query embedding."""

        try:
            # Compute similarities
            similarities = np.dot(candidate_embeddings, query_embedding)

            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

            # Filter by threshold
            results = []
            for idx in top_indices:
                similarity = float(similarities[idx])
                if similarity >= threshold:
                    results.append((int(idx), similarity))

            return results

        except Exception as e:
            self.logger.error(f"Error finding similar texts: {e}")
            return []

    def compute_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute centroid of a set of embeddings."""

        try:
            if len(embeddings) == 0:
                return np.array([])

            centroid = np.mean(embeddings, axis=0)
            return centroid

        except Exception as e:
            self.logger.error(f"Error computing centroid: {e}")
            return np.array([])

    def compute_cluster_centroids(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> dict[int, np.ndarray]:
        """Compute centroids for each cluster."""

        try:
            centroids = {}
            unique_labels = np.unique(labels)

            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue

                cluster_embeddings = embeddings[labels == label]
                if len(cluster_embeddings) > 0:
                    centroids[label] = self.compute_centroid(cluster_embeddings)

            return centroids

        except Exception as e:
            self.logger.error(f"Error computing cluster centroids: {e}")
            return {}

    def cache_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Cache an embedding."""

        self.embedding_cache[text] = embedding.copy()

    def get_cached_embedding(self, text: str) -> np.ndarray | None:
        """Get cached embedding if available."""

        return self.embedding_cache.get(text)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""

        self.embedding_cache.clear()
        self.logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""

        return {
            "cache_size": len(self.embedding_cache),
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
        }


class EmbeddingProcessor(LoggerMixin):
    """Process embeddings for clustering and analysis."""

    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator

    def process_posts(
        self, posts: list[dict[str, Any]], batch_size: int = 32
    ) -> tuple[np.ndarray, list[str]]:
        """Process posts and generate embeddings."""

        if not posts:
            return np.array([]), []

        # Extract texts
        texts = []
        post_ids = []

        for post in posts:
            text = post.get("text", "")
            if text and len(text.strip()) > 0:
                texts.append(text)
                post_ids.append(post.get("id", len(texts)))

        if not texts:
            self.logger.warning("No valid texts found in posts")
            return np.array([]), []

        # Generate embeddings
        embeddings = self.embedding_generator.embed_texts(texts, batch_size)

        self.logger.info(f"Processed {len(texts)} posts into embeddings")

        return embeddings, post_ids

    def filter_by_similarity(
        self,
        embeddings: np.ndarray,
        threshold: float = 0.95,
        post_ids: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Filter out highly similar embeddings."""

        if len(embeddings) == 0:
            return embeddings, post_ids or []

        # Compute pairwise similarities
        similarities = np.dot(embeddings, embeddings.T)

        # Find pairs above threshold
        similar_pairs = np.where(similarities > threshold)

        # Keep track of indices to remove
        indices_to_remove = set()

        for i, j in zip(similar_pairs[0], similar_pairs[1], strict=False):
            if i != j and i not in indices_to_remove:
                indices_to_remove.add(j)

        # Filter embeddings and post_ids
        keep_indices = [i for i in range(len(embeddings)) if i not in indices_to_remove]

        filtered_embeddings = embeddings[keep_indices]
        filtered_post_ids = [post_ids[i] for i in keep_indices] if post_ids else []

        self.logger.info(
            f"Filtered {len(indices_to_remove)} similar embeddings, kept {len(filtered_embeddings)}"
        )

        return filtered_embeddings, filtered_post_ids

    def compute_diversity_score(self, embeddings: np.ndarray) -> float:
        """Compute diversity score of embeddings."""

        if len(embeddings) < 2:
            return 0.0

        try:
            # Compute pairwise distances
            from sklearn.metrics.pairwise import cosine_distances

            distances = cosine_distances(embeddings)

            # Remove diagonal (self-distances)
            mask = ~np.eye(distances.shape[0], dtype=bool)
            distances = distances[mask]

            # Diversity is the mean distance
            diversity = float(np.mean(distances))

            return diversity

        except Exception as e:
            self.logger.error(f"Error computing diversity score: {e}")
            return 0.0

    def detect_outliers(
        self, embeddings: np.ndarray, threshold: float = 2.0
    ) -> np.ndarray:
        """Detect outlier embeddings using statistical methods."""

        if len(embeddings) < 3:
            return np.array([])

        try:
            # Compute centroid and distances
            centroid = self.embedding_generator.compute_centroid(embeddings)
            distances = np.linalg.norm(embeddings - centroid, axis=1)

            # Use z-score to detect outliers
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)

            if std_dist == 0:
                return np.array([])

            z_scores = np.abs((distances - mean_dist) / std_dist)
            outliers = np.where(z_scores > threshold)[0]

            self.logger.info(f"Detected {len(outliers)} outlier embeddings")

            return outliers

        except Exception as e:
            self.logger.error(f"Error detecting outliers: {e}")
            return np.array([])


def create_embedding_generator(model_name: str | None = None) -> EmbeddingGenerator:
    """Create an embedding generator instance."""
    return EmbeddingGenerator(model_name)


def create_embedding_processor(
    embedding_generator: EmbeddingGenerator,
) -> EmbeddingProcessor:
    """Create an embedding processor instance."""
    return EmbeddingProcessor(embedding_generator)
