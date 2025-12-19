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

logger = get_logger(__name__)
router = APIRouter()

# Mock components (to be properly initialized in Phase 2)
mock_stream = None
io_utils = None


@router.get("/top", response_model=TrendingTopicsResponse)
async def get_top_topics(window: int = 24, k: int = 10):
    """Get top trending topics."""
    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        from trend_detector.src.ingest.mock_stream import MockStream

        global mock_stream
        if mock_stream is None:
            mock_stream = MockStream()

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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search_posts(request: SearchRequest):
    """Search social media posts."""
    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        from trend_detector.src.ingest.mock_stream import MockStream

        global mock_stream
        if mock_stream is None:
            mock_stream = MockStream()

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
        raise HTTPException(status_code=500, detail=str(e))
