"""
Tests for Trend Detector clustering functionality.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from ..pipeline.cluster import TopicClusterer
from ..pipeline.embed import EmbeddingGenerator, EmbeddingProcessor
from ..pipeline.topics import TrendAnalyzer


class TestEmbeddingGenerator:
    """Test embedding generation."""

    @patch("trend_detector.src.pipeline.embed.SentenceTransformer")
    def test_embedding_generator_initialization(self, mock_transformer):
        """Test embedding generator initialization."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        generator = EmbeddingGenerator()

        assert generator.model_name == "all-MiniLM-L6-v2"
        assert generator.model is not None

    @patch("trend_detector.src.pipeline.embed.SentenceTransformer")
    def test_embed_texts(self, mock_transformer):
        """Test text embedding generation."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(3, 384)
        mock_transformer.return_value = mock_model

        generator = EmbeddingGenerator()
        texts = ["Hello world", "Test text", "Another example"]

        embeddings = generator.embed_texts(texts)

        assert embeddings.shape == (3, 384)
        assert isinstance(embeddings, np.ndarray)

    @patch("trend_detector.src.pipeline.embed.SentenceTransformer")
    def test_embed_single_text(self, mock_transformer):
        """Test single text embedding."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(1, 384)
        mock_transformer.return_value = mock_model

        generator = EmbeddingGenerator()
        text = "Hello world"

        embedding = generator.embed_single_text(text)

        assert embedding.shape == (384,)
        assert isinstance(embedding, np.ndarray)

    @patch("trend_detector.src.pipeline.embed.SentenceTransformer")
    def test_similarity_computation(self, mock_transformer):
        """Test similarity computation."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        generator = EmbeddingGenerator()

        # Create test embeddings
        embedding1 = np.array([1, 0, 0])
        embedding2 = np.array([0, 1, 0])
        embedding3 = np.array([1, 0, 0])

        # Test similarity
        sim_orthogonal = generator.compute_similarity(embedding1, embedding2)
        sim_identical = generator.compute_similarity(embedding1, embedding3)

        assert sim_orthogonal == 0.0  # Orthogonal vectors
        assert sim_identical == 1.0  # Identical vectors


class TestTopicClusterer:
    """Test topic clustering."""

    @patch("trend_detector.src.pipeline.cluster.BERTopic")
    def test_clusterer_initialization(self, mock_bertopic):
        """Test clusterer initialization."""
        mock_model = Mock()
        mock_bertopic.return_value = mock_model

        clusterer = TopicClusterer()

        assert clusterer.min_topic_size == 10
        assert clusterer.n_neighbors == 15
        assert clusterer.model is not None

    @patch("trend_detector.src.pipeline.cluster.BERTopic")
    def test_fit_transform(self, mock_bertopic):
        """Test fit and transform."""
        mock_model = Mock()
        mock_model.fit_transform.return_value = (
            np.array([0, 1, 0, 2]),
            np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]]),
        )
        mock_model.get_topic_info.return_value = Mock(
            to_dict=lambda x: [{"Topic": 0}, {"Topic": 1}, {"Topic": 2}]
        )
        mock_bertopic.return_value = mock_model

        clusterer = TopicClusterer()
        documents = ["Document 1", "Document 2", "Document 3", "Document 4"]

        topics, probs = clusterer.fit_transform(documents)

        assert len(topics) == 4
        assert probs.shape == (4, 2)
        assert clusterer.topics is not None

    @patch("trend_detector.src.pipeline.cluster.BERTopic")
    def test_get_topic_statistics(self, mock_bertopic):
        """Test topic statistics."""
        mock_model = Mock()
        mock_bertopic.return_value = mock_model

        clusterer = TopicClusterer()
        clusterer.topics = np.array([0, 1, 0, 2, 1])

        stats = clusterer.get_topic_statistics()

        assert "total_documents" in stats
        assert "num_topics" in stats
        assert "topic_distribution" in stats
        assert stats["total_documents"] == 5
        assert stats["num_topics"] == 3


class TestTrendAnalyzer:
    """Test trend analysis."""

    def test_trend_analyzer_initialization(self):
        """Test trend analyzer initialization."""
        analyzer = TrendAnalyzer()

        assert analyzer.novelty_weight == 0.3
        assert analyzer.volume_weight == 0.3
        assert analyzer.velocity_weight == 0.2
        assert analyzer.burstiness_weight == 0.2

    def test_calculate_trend_score(self):
        """Test trend score calculation."""
        analyzer = TrendAnalyzer()

        # Mock data
        posts = [
            {"text": "Test post 1", "timestamp": "2024-01-01T10:00:00Z"},
            {"text": "Test post 2", "timestamp": "2024-01-01T11:00:00Z"},
            {"text": "Test post 3", "timestamp": "2024-01-01T12:00:00Z"},
        ]

        embeddings = np.random.randn(3, 100)

        result = analyzer.calculate_trend_score(1, posts, embeddings)

        assert "trend_score" in result
        assert "volume_score" in result
        assert "velocity_score" in result
        assert "novelty_score" in result
        assert "burstiness_score" in result
        assert 0 <= result["trend_score"] <= 1

    def test_volume_score_calculation(self):
        """Test volume score calculation."""
        analyzer = TrendAnalyzer()

        # Test with posts in time window
        posts = [
            {"timestamp": "2024-01-01T10:00:00Z"},
            {"timestamp": "2024-01-01T11:00:00Z"},
            {"timestamp": "2024-01-01T12:00:00Z"},
        ]

        volume_score = analyzer._calculate_volume_score(posts, 24)

        assert 0 <= volume_score <= 1
        assert isinstance(volume_score, float)

    def test_novelty_score_calculation(self):
        """Test novelty score calculation."""
        analyzer = TrendAnalyzer()

        # Create embeddings
        embeddings = np.random.randn(5, 100)

        novelty_score = analyzer._calculate_novelty_score(1, embeddings)

        assert 0 <= novelty_score <= 1
        assert isinstance(novelty_score, float)

    def test_get_trending_topics(self):
        """Test trending topics extraction."""
        analyzer = TrendAnalyzer()

        # Mock data
        topics_data = {
            1: [
                {"text": "Topic 1 post 1", "timestamp": "2024-01-01T10:00:00Z"},
                {"text": "Topic 1 post 2", "timestamp": "2024-01-01T11:00:00Z"},
            ],
            2: [{"text": "Topic 2 post 1", "timestamp": "2024-01-01T10:00:00Z"}],
        }

        embeddings_data = {1: np.random.randn(2, 100), 2: np.random.randn(1, 100)}

        trending = analyzer.get_trending_topics(topics_data, embeddings_data, top_k=2)

        assert len(trending) <= 2
        assert all("trend_score" in topic for topic in trending)
        assert all("keywords" in topic for topic in trending)
        assert all("topic_id" in topic for topic in trending)


class TestEmbeddingProcessor:
    """Test embedding processing utilities."""

    @patch("trend_detector.src.pipeline.embed.EmbeddingGenerator")
    def test_process_posts(self, mock_generator):
        """Test post processing."""
        mock_gen = Mock()
        mock_gen.embed_texts.return_value = np.random.randn(3, 384)
        mock_generator.return_value = mock_gen

        processor = EmbeddingProcessor(mock_gen)

        posts = [
            {"text": "Post 1", "id": 1},
            {"text": "Post 2", "id": 2},
            {"text": "Post 3", "id": 3},
        ]

        embeddings, post_ids = processor.process_posts(posts)

        assert embeddings.shape == (3, 384)
        assert len(post_ids) == 3
        assert post_ids == [1, 2, 3]

    @patch("trend_detector.src.pipeline.embed.EmbeddingGenerator")
    def test_filter_by_similarity(self, mock_generator):
        """Test similarity filtering."""
        mock_gen = Mock()
        mock_generator.return_value = mock_gen

        processor = EmbeddingProcessor(mock_gen)

        # Create embeddings with high similarity
        embeddings = np.array(
            [[1, 0, 0], [1, 0, 0.1], [0, 1, 0]]  # Very similar to first  # Different
        )

        filtered_embeddings, filtered_ids = processor.filter_by_similarity(
            embeddings, threshold=0.95, post_ids=[0, 1, 2]
        )

        # Should remove one of the similar embeddings
        assert len(filtered_embeddings) < len(embeddings)
        assert len(filtered_ids) < len([0, 1, 2])


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "Machine learning is revolutionizing healthcare with AI applications",
        "Artificial intelligence helps doctors diagnose diseases more accurately",
        "Deep learning models can analyze medical images with high precision",
        "Technology is transforming the way we approach medical diagnosis",
        "AI in healthcare shows promising results for early disease detection",
    ]


def test_clustering_pipeline(sample_documents):
    """Test complete clustering pipeline."""
    with patch("trend_detector.src.pipeline.cluster.BERTopic") as mock_bertopic:
        mock_model = Mock()
        mock_model.fit_transform.return_value = (
            np.array([0, 0, 1, 1, 0]),
            np.array([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]]),
        )
        mock_model.get_topic_info.return_value = Mock(
            to_dict=lambda x: [
                {"Topic": 0, "Count": 3, "Name": "Healthcare AI"},
                {"Topic": 1, "Count": 2, "Name": "Medical Technology"},
            ]
        )
        mock_bertopic.return_value = mock_model

        # Initialize clusterer
        clusterer = TopicClusterer()

        # Fit and transform
        topics, probs = clusterer.fit_transform(sample_documents)

        # Assertions
        assert len(topics) == len(sample_documents)
        assert probs.shape[0] == len(sample_documents)
        assert probs.shape[1] == 2  # Binary classification probabilities

        # Test statistics
        stats = clusterer.get_topic_statistics()
        assert stats["total_documents"] == len(sample_documents)
        assert stats["num_topics"] >= 1
