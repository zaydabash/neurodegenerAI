"""
Trend detection endpoints.
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException

from core_api.src.api.v1.schemas import (
    PostData,
    SearchRequest,
    SearchResponse,
    TrendingTopic,
    TrendingTopicsResponse,
)
from shared.lib.logging import get_logger, log_api_request
from shared.lib.repository import save_social_posts, save_trend_topics

logger = get_logger(__name__)
router = APIRouter()


@router.get("/top", response_model=TrendingTopicsResponse)
async def get_top_topics(window: int = 24, k: int = 10):
    """Get top trending topics."""
    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        from trend_detector.src.ingest.provider import get_stream_provider
        from trend_detector.src.pipeline.live_topics import cluster_posts

        provider = get_stream_provider()

        # Cluster a live (or demo) corpus into topics. Fall back to the
        # provider's precomputed demo topics only if clustering yields nothing.
        corpus = provider.get_recent_posts(200)
        trending_data = cluster_posts(corpus, k=k)
        if not trending_data:
            trending_data = provider.get_trending_topics(window)

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

        save_trend_topics(topic.dict() for topic in topics)

        response = TrendingTopicsResponse(topics=topics, window_hours=window)
        log_api_request(
            "/v1/trends/top",
            "GET",
            200,
            (datetime.now() - start_time).total_seconds(),
            request_id=request_id,
        )
        return response
    except Exception as e:
        logger.error(f"Trends top topics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/search", response_model=SearchResponse)
async def search_posts(request: SearchRequest):
    """Search social media posts using semantic ranking (keyword fallback)."""
    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        from trend_detector.src.ingest.provider import get_stream_provider
        from trend_detector.src.pipeline.search import SemanticSearch

        provider = get_stream_provider()
        searcher = SemanticSearch()

        # Pull a candidate corpus, then rank it against the query.
        corpus = provider.get_recent_posts(max(request.limit * 5, 50))
        ranked = searcher.search(request.query, corpus, limit=request.limit)

        results = [
            PostData(
                text=post["text"],
                source=post["source"],
                timestamp=post["timestamp"],
                url=post.get("url"),
                author=post.get("author"),
                score=post.get("score", 0),
                metadata={"relevance": post["relevance"]}
                if "relevance" in post
                else None,
            )
            for post in ranked
        ]

        save_social_posts(ranked)

        response = SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time=(datetime.now() - start_time).total_seconds(),
        )

        log_api_request(
            "/v1/trends/search",
            "POST",
            200,
            (datetime.now() - start_time).total_seconds(),
            request_id=request_id,
        )
        return response
    except Exception as e:
        logger.error(f"Trends search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
