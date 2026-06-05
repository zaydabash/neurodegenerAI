"""
Pydantic schemas for Trend Detector API.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "0.1.0"
    services: dict[str, bool] = Field(default_factory=dict)


class PostData(BaseModel):
    """Social media post data."""

    text: str = Field(..., min_length=1, description="Post text content")
    source: str = Field(..., description="Source platform (reddit, twitter, etc.)")
    timestamp: str = Field(..., description="Post timestamp (ISO format)")
    url: str | None = Field(None, description="Post URL")
    author: str | None = Field(None, description="Author username")
    score: int | None = Field(0, description="Post score/upvotes")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class ClusterData(BaseModel):
    """Topic cluster data."""

    cluster_id: int = Field(..., description="Cluster ID")
    topic: str = Field(..., description="Topic name/label")
    representative_terms: list[str] = Field(..., description="Top representative terms")
    volume: int = Field(..., description="Number of posts in cluster")
    trend_score: float = Field(..., ge=0, le=1, description="Trending score")
    timestamp: str = Field(..., description="Cluster timestamp")
    example_posts: list[str] | None = Field(
        None, description="Example posts from cluster"
    )


class TrendingTopic(BaseModel):
    """Trending topic data."""

    topic: str = Field(..., description="Topic name")
    keywords: list[str] = Field(..., description="Topic keywords")
    trending_score: float = Field(..., ge=0, le=1, description="Trending score")
    volume: int = Field(..., description="Post volume")
    growth_rate: float = Field(..., description="Growth rate")
    representative_posts: list[str] = Field(..., description="Representative posts")
    cluster_id: int | None = Field(None, description="Associated cluster ID")


class SearchRequest(BaseModel):
    """Search request."""

    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    source: str | None = Field(None, description="Filter by source platform")
    time_window: str | None = Field(None, description="Time window (e.g., '24h', '7d')")


class SearchResponse(BaseModel):
    """Search response."""

    query: str = Field(..., description="Original query")
    results: list[PostData] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_time: float = Field(..., description="Search time in seconds")


class ClustersResponse(BaseModel):
    """Clusters response."""

    clusters: list[ClusterData] = Field(..., description="List of clusters")
    total_clusters: int = Field(..., description="Total number of clusters")
    timestamp: datetime = Field(default_factory=datetime.now)


class TrendingTopicsResponse(BaseModel):
    """Trending topics response."""

    topics: list[TrendingTopic] = Field(..., description="List of trending topics")
    window_hours: int = Field(..., description="Time window in hours")
    timestamp: datetime = Field(default_factory=datetime.now)


class StreamRequest(BaseModel):
    """Stream request."""

    sources: list[str] = Field(..., description="Sources to stream from")
    duration_minutes: int = Field(
        60, ge=1, le=1440, description="Stream duration in minutes"
    )
    posts_per_minute: int = Field(
        10, ge=1, le=100, description="Target posts per minute"
    )


class StreamResponse(BaseModel):
    """Stream response."""

    stream_id: str = Field(..., description="Stream ID")
    status: str = Field(..., description="Stream status")
    posts_collected: int = Field(..., description="Number of posts collected")
    duration_minutes: int = Field(..., description="Actual duration in minutes")
    timestamp: datetime = Field(default_factory=datetime.now)


class EmbeddingRequest(BaseModel):
    """Embedding request."""

    texts: list[str] = Field(
        ..., min_length=1, max_length=1000, description="Texts to embed"
    )
    model_name: str | None = Field(None, description="Embedding model name")


class EmbeddingResponse(BaseModel):
    """Embedding response."""

    embeddings: list[list[float]] = Field(..., description="Text embeddings")
    model_name: str = Field(..., description="Model used")
    dimension: int = Field(..., description="Embedding dimension")
    processing_time: float = Field(..., description="Processing time in seconds")


class ClusteringRequest(BaseModel):
    """Clustering request."""

    texts: list[str] = Field(
        ..., min_length=10, max_length=10000, description="Texts to cluster"
    )
    min_topic_size: int = Field(10, ge=5, le=100, description="Minimum topic size")
    n_neighbors: int = Field(15, ge=5, le=50, description="UMAP neighbors")
    min_cluster_size: int = Field(10, ge=5, le=100, description="Minimum cluster size")


class ClusteringResponse(BaseModel):
    """Clustering response."""

    topics: list[int] = Field(..., description="Topic assignments")
    topic_info: list[dict[str, Any]] = Field(..., description="Topic information")
    statistics: dict[str, Any] = Field(..., description="Clustering statistics")
    processing_time: float = Field(..., description="Processing time in seconds")


class TopicAnalysisRequest(BaseModel):
    """Topic analysis request."""

    topic_id: int = Field(..., description="Topic ID to analyze")
    time_window_hours: int = Field(24, ge=1, le=168, description="Analysis time window")


class TopicAnalysisResponse(BaseModel):
    """Topic analysis response."""

    topic_id: int = Field(..., description="Topic ID")
    topic_words: list[tuple[str, float]] = Field(
        ..., description="Topic words with weights"
    )
    evolution_data: dict[str, Any] = Field(..., description="Topic evolution over time")
    similar_topics: list[tuple[int, float]] = Field(..., description="Similar topics")
    coherence_score: float = Field(..., description="Topic coherence score")


class BatchProcessingRequest(BaseModel):
    """Batch processing request."""

    posts: list[PostData] = Field(
        ..., min_length=1, max_length=1000, description="Posts to process"
    )
    process_embeddings: bool = Field(True, description="Whether to compute embeddings")
    process_clustering: bool = Field(True, description="Whether to perform clustering")
    save_to_db: bool = Field(True, description="Whether to save results to database")


class BatchProcessingResponse(BaseModel):
    """Batch processing response."""

    processed_posts: int = Field(..., description="Number of posts processed")
    embeddings_computed: int = Field(..., description="Number of embeddings computed")
    clusters_found: int = Field(..., description="Number of clusters found")
    processing_time: float = Field(..., description="Total processing time")
    results: dict[str, Any] = Field(..., description="Processing results")


class MetricsResponse(BaseModel):
    """System metrics response."""

    total_posts: int = Field(..., description="Total posts in database")
    total_clusters: int = Field(..., description="Total clusters")
    active_sources: list[str] = Field(..., description="Active data sources")
    last_update: datetime = Field(..., description="Last data update")
    system_health: dict[str, Any] = Field(..., description="System health metrics")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: str | None = Field(None, description="Request ID for tracking")


class UMAPVisualizationRequest(BaseModel):
    """UMAP visualization request."""

    texts: list[str] | None = Field(None, description="Texts to visualize")
    embeddings: list[list[float]] | None = Field(
        None, description="Pre-computed embeddings"
    )
    topics: list[int] | None = Field(None, description="Topic assignments")
    n_components: int = Field(2, ge=2, le=3, description="Number of UMAP components")


class UMAPVisualizationResponse(BaseModel):
    """UMAP visualization response."""

    coordinates: list[list[float]] = Field(..., description="UMAP coordinates")
    topics: list[int] = Field(..., description="Topic assignments")
    texts: list[str] = Field(..., description="Text samples")
    visualization_url: str | None = Field(None, description="URL to visualization")


class RealTimeUpdateRequest(BaseModel):
    """Real-time update request."""

    sources: list[str] = Field(..., description="Sources to monitor")
    update_interval: int = Field(
        60, ge=10, le=3600, description="Update interval in seconds"
    )
    max_posts: int = Field(
        1000, ge=10, le=10000, description="Maximum posts per update"
    )


class RealTimeUpdateResponse(BaseModel):
    """Real-time update response."""

    update_id: str = Field(..., description="Update ID")
    new_posts: int = Field(..., description="Number of new posts")
    new_clusters: int = Field(..., description="Number of new clusters")
    trending_changes: list[str] = Field(..., description="Changes in trending topics")
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelInfoResponse(BaseModel):
    """Model information response."""

    embedding_model: str = Field(..., description="Embedding model name")
    clustering_model: str = Field(..., description="Clustering model name")
    model_versions: dict[str, str] = Field(..., description="Model versions")
    last_training: datetime | None = Field(None, description="Last training date")
    performance_metrics: dict[str, float] = Field(
        ..., description="Model performance metrics"
    )
