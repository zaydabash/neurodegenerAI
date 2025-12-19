"""
FastAPI server for Trend Detector.
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from shared.lib.config import ensure_directories, get_settings
from shared.lib.io_utils import IOUtils
from shared.lib.logging import get_logger, log_api_request, setup_logging
from shared.lib.metrics import record_performance

from ..ingest.mock_stream import MockStream
from ..ingest.reddit_stream import RedditStream
from ..pipeline.cluster import ClusterAnalyzer, TopicClusterer
from ..pipeline.embed import EmbeddingGenerator, EmbeddingProcessor
from ..pipeline.topics import TrendAnalyzer
from .schemas import (
    ClusterData,
    ClusteringRequest,
    ClusteringResponse,
    ClustersResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PostData,
    SearchRequest,
    SearchResponse,
    StreamRequest,
    StreamResponse,
    TrendingTopic,
    TrendingTopicsResponse,
)

# Setup logging
setup_logging(service_name="trend_detector_api")
logger = get_logger(__name__)

# Global variables for models and data
mock_stream: MockStream | None = None
reddit_stream: RedditStream | None = None
embedding_generator: EmbeddingGenerator | None = None
embedding_processor: EmbeddingProcessor | None = None
topic_clusterer: TopicClusterer | None = None
cluster_analyzer: ClusterAnalyzer | None = None
trend_analyzer: TrendAnalyzer | None = None
io_utils: IOUtils | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""

    global mock_stream, reddit_stream, embedding_generator, embedding_processor
    global topic_clusterer, cluster_analyzer, trend_analyzer, io_utils

    logger.info("Starting Trend Detector API server")

    # Ensure directories exist
    ensure_directories()

    try:
        # Initialize components
        io_utils = IOUtils("trends")

        # Initialize data streams
        mock_stream = MockStream()
        reddit_stream = RedditStream()
        logger.info("Data streams initialized")

        # Initialize embedding components
        embedding_generator = EmbeddingGenerator()
        embedding_processor = EmbeddingProcessor(embedding_generator)
        logger.info("Embedding components initialized")

        # Initialize clustering components
        topic_clusterer = TopicClusterer()
        cluster_analyzer = ClusterAnalyzer(topic_clusterer)
        logger.info("Clustering components initialized")

        # Initialize trend analyzer
        trend_analyzer = TrendAnalyzer()
        logger.info("Trend analyzer initialized")

        # Seed with some initial data
        await seed_initial_data()

    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        # Continue with limited functionality

    yield

    logger.info("Shutting down Trend Detector API server")


# Create FastAPI app
app = FastAPI(
    title="Trend Detector API",
    description="Real-time trend detection from social media streams",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def seed_initial_data():
    """Seed the system with initial demo data."""

    try:
        logger.info("Seeding initial data")

        # Generate some sample posts
        if mock_stream:
            posts = mock_stream.get_recent_posts(100)

            # Process posts
            if embedding_generator and len(posts) > 10:
                texts = [post["text"] for post in posts]
                topics, probs = topic_clusterer.fit_transform(texts)

                # Store in database
                db_handler = io_utils.get_database_handler()
                if db_handler:
                    for _i, post in enumerate(posts):
                        post_id = db_handler.insert_post(post)
                        if post_id and embedding_generator:
                            embedding = embedding_generator.embed_single_text(
                                post["text"]
                            )
                            db_handler.insert_embedding(
                                post_id, embedding, embedding_generator.model_name
                            )

        logger.info("Initial data seeding completed")

    except Exception as e:
        logger.error(f"Error seeding initial data: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""

    start_time = datetime.now()

    try:
        services = {
            "mock_stream": mock_stream is not None,
            "reddit_stream": reddit_stream is not None,
            "embedding_generator": embedding_generator is not None,
            "topic_clusterer": topic_clusterer is not None,
            "trend_analyzer": trend_analyzer is not None,
        }

        all_healthy = all(services.values())
        status = "healthy" if all_healthy else "degraded"

        response = HealthResponse(status=status, services=services)

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/health", "GET", 200, duration)

        return response

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/clusters/latest", response_model=ClustersResponse)
async def get_latest_clusters(limit: int = 50):
    """Get latest clusters."""

    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        db_handler = io_utils.get_database_handler() if io_utils else None

        if db_handler:
            clusters_data = db_handler.get_latest_clusters(limit)
            clusters = []

            for cluster_data in clusters_data:
                cluster = ClusterData(
                    cluster_id=cluster_data["cluster_id"],
                    topic=cluster_data["topic"]
                    or f"Topic {cluster_data['cluster_id']}",
                    representative_terms=(
                        cluster_data["representative_terms"].split(",")
                        if cluster_data["representative_terms"]
                        else []
                    ),
                    volume=cluster_data["volume"],
                    trend_score=cluster_data["trend_score"],
                    timestamp=cluster_data["timestamp"],
                )
                clusters.append(cluster)
        else:
            # Fallback to mock data
            if mock_stream:
                trending = mock_stream.get_trending_topics()
                clusters = []

                for i, topic_data in enumerate(trending[:limit]):
                    cluster = ClusterData(
                        cluster_id=i,
                        topic=topic_data["topic"],
                        representative_terms=topic_data["keywords"][:5],
                        volume=topic_data["volume"],
                        trend_score=topic_data["trending_score"],
                        timestamp=datetime.now().isoformat(),
                    )
                    clusters.append(cluster)
            else:
                clusters = []

        response = ClustersResponse(clusters=clusters, total_clusters=len(clusters))

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/clusters/latest", "GET", 200, duration, request_id=request_id)

        return response

    except Exception as e:
        logger.error(f"Failed to get latest clusters: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/clusters/latest", "GET", 500, duration, request_id=request_id)

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/topics/top", response_model=TrendingTopicsResponse)
async def get_top_topics(window: int = 24, k: int = 10):
    """Get top trending topics."""

    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        if mock_stream:
            trending_data = mock_stream.get_trending_topics(window)
            topics = []

            for topic_data in trending_data[:k]:
                topic = TrendingTopic(
                    topic=topic_data["topic"],
                    keywords=topic_data["keywords"][:5],
                    trending_score=topic_data["trending_score"],
                    volume=topic_data["volume"],
                    growth_rate=topic_data["growth_rate"],
                    representative_posts=topic_data["representative_posts"][:3],
                )
                topics.append(topic)
        else:
            topics = []

        response = TrendingTopicsResponse(topics=topics, window_hours=window)

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/topics/top", "GET", 200, duration, request_id=request_id)

        return response

    except Exception as e:
        logger.error(f"Failed to get top topics: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/topics/top", "GET", 500, duration, request_id=request_id)

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/search", response_model=SearchResponse)
async def search_posts(request: SearchRequest):
    """Search posts by query."""

    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        db_handler = io_utils.get_database_handler() if io_utils else None

        if db_handler:
            posts_data = db_handler.search_posts(request.query, request.limit)
            results = []

            for post_data in posts_data:
                post = PostData(
                    text=post_data["text"],
                    source=post_data["source"],
                    timestamp=post_data["timestamp"],
                    url=post_data.get("url"),
                    author=post_data.get("author"),
                    score=post_data.get("score", 0),
                )
                results.append(post)
        else:
            # Fallback to mock data
            if mock_stream:
                all_posts = mock_stream.get_recent_posts(request.limit * 2)
                results = []

                for post_data in all_posts:
                    if request.query.lower() in post_data["text"].lower():
                        post = PostData(
                            text=post_data["text"],
                            source=post_data["source"],
                            timestamp=post_data["timestamp"],
                            url=post_data.get("url"),
                            author=post_data.get("author"),
                            score=post_data.get("score", 0),
                        )
                        results.append(post)

                        if len(results) >= request.limit:
                            break
            else:
                results = []

        response = SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time=(datetime.now() - start_time).total_seconds(),
        )

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/search", "POST", 200, duration, request_id=request_id)

        return response

    except Exception as e:
        logger.error(f"Search failed: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/search", "POST", 500, duration, request_id=request_id)

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for texts."""

    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        if embedding_generator is None:
            raise HTTPException(
                status_code=503, detail="Embedding generator not available"
            )

        with record_performance(
            "embedding_generation", {"num_texts": len(request.texts)}
        ):
            embeddings = embedding_generator.embed_texts(request.texts)

        response = EmbeddingResponse(
            embeddings=embeddings.tolist(),
            model_name=embedding_generator.model_name,
            dimension=embedding_generator.get_embedding_dimension(),
            processing_time=(datetime.now() - start_time).total_seconds(),
        )

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/embed", "POST", 200, duration, request_id=request_id)

        return response

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/embed", "POST", 500, duration, request_id=request_id)

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/cluster", response_model=ClusteringResponse)
async def cluster_texts(request: ClusteringRequest):
    """Cluster texts into topics."""

    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        if topic_clusterer is None:
            raise HTTPException(status_code=503, detail="Topic clusterer not available")

        with record_performance("clustering", {"num_texts": len(request.texts)}):
            topics, probs = topic_clusterer.fit_transform(request.texts)

            # Get topic information
            topic_info = topic_clusterer.get_topic_info().to_dict("records")

            # Get statistics
            statistics = topic_clusterer.get_topic_statistics()

        response = ClusteringResponse(
            topics=topics.tolist(),
            topic_info=topic_info,
            statistics=statistics,
            processing_time=(datetime.now() - start_time).total_seconds(),
        )

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/cluster", "POST", 200, duration, request_id=request_id)

        return response

    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/cluster", "POST", 500, duration, request_id=request_id)

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/stream", response_model=StreamResponse)
async def start_stream(request: StreamRequest, background_tasks: BackgroundTasks):
    """Start a data stream."""

    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        stream_id = str(uuid.uuid4())

        # Start background streaming task
        background_tasks.add_task(
            run_stream_task,
            stream_id,
            request.sources,
            request.duration_minutes,
            request.posts_per_minute,
        )

        response = StreamResponse(
            stream_id=stream_id,
            status="started",
            posts_collected=0,
            duration_minutes=request.duration_minutes,
        )

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/stream", "POST", 200, duration, request_id=request_id)

        return response

    except Exception as e:
        logger.error(f"Stream start failed: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/stream", "POST", 500, duration, request_id=request_id)

        raise HTTPException(status_code=500, detail=str(e)) from e


async def run_stream_task(
    stream_id: str, sources: list[str], duration_minutes: int, posts_per_minute: int
):
    """Background task to run data streaming."""

    try:
        logger.info(f"Starting stream task {stream_id}")

        # Use mock stream for demo
        if mock_stream:
            posts_collected = 0

            for post in mock_stream.stream_posts(duration_minutes, posts_per_minute):
                # Store post in database
                db_handler = io_utils.get_database_handler() if io_utils else None
                if db_handler:
                    post_id = db_handler.insert_post(post)
                    if post_id and embedding_generator:
                        embedding = embedding_generator.embed_single_text(post["text"])
                        db_handler.insert_embedding(
                            post_id, embedding, embedding_generator.model_name
                        )

                posts_collected += 1

                # Log progress every 10 posts
                if posts_collected % 10 == 0:
                    logger.info(
                        f"Stream {stream_id}: collected {posts_collected} posts"
                    )

            logger.info(
                f"Stream {stream_id} completed: {posts_collected} posts collected"
            )

    except Exception as e:
        logger.error(f"Stream task {stream_id} failed: {e}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_system_metrics():
    """Get system metrics."""

    start_time = datetime.now()

    try:
        db_handler = io_utils.get_database_handler() if io_utils else None

        total_posts = 0
        total_clusters = 0
        last_update = datetime.now()

        if db_handler:
            try:
                # Get post count
                posts = db_handler.get_latest_posts(1)
                total_posts = len(posts) if posts else 0

                # Get cluster count
                clusters = db_handler.get_latest_clusters(1)
                total_clusters = len(clusters) if clusters else 0

                # Get last update time
                if posts:
                    last_update = datetime.fromisoformat(posts[0]["timestamp"])
            except Exception:
                pass

        # Get active sources
        active_sources = []
        if mock_stream:
            active_sources.append("mock_stream")
        if reddit_stream and reddit_stream.reddit:
            active_sources.append("reddit")

        # System health
        system_health = {
            "embedding_generator_available": embedding_generator is not None,
            "topic_clusterer_available": topic_clusterer is not None,
            "trend_analyzer_available": trend_analyzer is not None,
            "database_available": db_handler is not None,
        }

        response = MetricsResponse(
            total_posts=total_posts,
            total_clusters=total_clusters,
            active_sources=active_sources,
            last_update=last_update,
            system_health=system_health,
        )

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/metrics", "GET", 200, duration)

        return response

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/metrics", "GET", 500, duration)

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information."""

    start_time = datetime.now()

    try:
        embedding_model = (
            embedding_generator.model_name if embedding_generator else "unknown"
        )
        clustering_model = "BERTopic" if topic_clusterer else "unknown"

        model_versions = {
            "embedding_model": embedding_model,
            "clustering_model": clustering_model,
        }

        performance_metrics = {}
        if embedding_generator:
            cache_stats = embedding_generator.get_cache_stats()
            performance_metrics.update(
                {
                    "cache_size": cache_stats.get("cache_size", 0),
                    "embedding_dimension": cache_stats.get("embedding_dimension", 0),
                }
            )

        response = ModelInfoResponse(
            embedding_model=embedding_model,
            clustering_model=clustering_model,
            model_versions=model_versions,
            performance_metrics=performance_metrics,
        )

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/model/info", "GET", 200, duration)

        return response

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/model/info", "GET", 500, duration)

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""

    logger.error(f"Unhandled exception: {exc}")

    return ErrorResponse(
        error="Internal server error", detail=str(exc), timestamp=datetime.now()
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("main:app", host="0.0.0.0", port=9002, reload=settings.env == "dev")
