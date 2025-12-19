"""
Unified Pydantic schemas for the consolidated Core API.
"""

from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, validator

# --- COMMON SCHEMAS ---


class HealthResponse(BaseModel):
    """Unified health check response."""

    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    db_connected: bool = False
    models_loaded: dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: str | None = None


# --- NEURO SCHEMAS ---


class TabularPredictionRequest(BaseModel):
    """Request for neurodegenerative pattern detection (Biomarkers)."""

    age: float = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)  # 0=Female, 1=Male
    apoe4: int = Field(..., ge=0, le=2)
    mmse: float = Field(..., ge=0, le=30)
    cdr: float | None = Field(None, ge=0, le=3)
    abeta: float | None = None
    tau: float | None = None
    ptau: float | None = None
    education: float | None = None
    hippocampal_volume: float | None = None
    cortical_thickness: float | None = None
    white_matter_hyperintensities: float | None = None


class MRIPredictionRequest(BaseModel):
    """Request for neurodegenerative pattern detection (MRI)."""

    volume_data: list[list[list[float]]] = Field(..., description="3D MRI volume")
    metadata: dict[str, Any] | None = None

    @validator("volume_data")
    def validate_volume(cls, v):
        arr = np.array(v)
        if len(arr.shape) != 3:
            raise ValueError("MRI data must be 3D")
        return v


class EEGPredictionRequest(BaseModel):
    """Request for EEG state decoding."""

    data: list[list[float]] = Field(
        ..., description="Multi-channel EEG (channels, length)"
    )
    sfreq: int = 250
    metadata: dict[str, Any] | None = None


class PredictionResponse(BaseModel):
    """Consolidated prediction response."""

    prediction: int = Field(..., description="Predicted class")
    label: str = Field(..., description="Human readable label")
    probability: float
    confidence: float
    model_name: str
    method_type: str
    explanation: dict[str, Any] | None = None
    heatmap_paths: list[str] | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


# --- TRENDS SCHEMAS ---


class PostData(BaseModel):
    """Social media post content."""

    text: str = Field(..., min_length=1)
    source: str
    timestamp: str
    url: str | None = None
    author: str | None = None
    score: int | None = 0
    metadata: dict[str, Any] | None = None


class TrendingTopic(BaseModel):
    """Detected trending topic."""

    topic: str
    keywords: list[str]
    trending_score: float = Field(..., ge=0, le=1)
    volume: int
    growth_rate: float
    representative_posts: list[str]


class TrendingTopicsResponse(BaseModel):
    """Trending topics report."""

    topics: list[TrendingTopic]
    window_hours: int
    timestamp: datetime = Field(default_factory=datetime.now)


class SearchRequest(BaseModel):
    """Request to search social data."""

    query: str
    limit: int = 100
    source: str | None = None
    time_window: str | None = None


class SearchResponse(BaseModel):
    """Search response."""

    query: str
    results: list[PostData]
    total_results: int
    search_time: float


# --- AUTH SCHEMAS (NEW) ---


class User(BaseModel):
    """System user schema."""

    username: str
    email: str
    full_name: str | None = None
    disabled: bool | None = None


class Token(BaseModel):
    """Auth token response."""

    access_token: str
    token_type: str
