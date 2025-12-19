"""
Unified Core API for Neuro-Trends Suite.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core_api.src.api.v1.endpoints import neuro, trends
from core_api.src.api.v1.schemas import HealthResponse
from shared.lib.config import ensure_directories, get_settings
from shared.lib.logging import get_logger, setup_logging

# Setup logging
setup_logging(service_name="core_api")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Unified application lifespan."""
    logger.info("Initializing Core API...")
    ensure_directories()

    # Initialize Database
    from shared.lib.database import get_db_manager

    db_manager = get_db_manager()
    db_manager.create_tables()

    yield
    logger.info("Core API shutting down...")


app = FastAPI(
    title="Neuro-Trends Core API",
    description="Consolidated backend for neurodegenerative analysis and trend detection.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hardened in production phase
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(neuro.router, prefix="/v1/neuro", tags=["NeuroDegenerAI"])
app.include_router(trends.router, prefix="/v1/trends", tags=["Trend Detector"])


@app.get("/health", response_model=HealthResponse)
async def health():
    """System health check."""
    return HealthResponse(status="healthy")


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host="0.0.0.0", port=8000)
