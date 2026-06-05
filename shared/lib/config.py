"""
Configuration management using Pydantic Settings.
"""

import os

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Common settings
    env: str = Field(default="dev", description="Environment (dev, staging, prod)")
    log_level: str = Field(default="INFO", description="Logging level")
    port: int = Field(default=8080, description="Default port")

    # Trend Detector settings
    reddit_client_id: str | None = Field(
        default=None, description="Reddit API client ID"
    )
    reddit_client_secret: str | None = Field(
        default=None, description="Reddit API client secret"
    )
    reddit_user_agent: str = Field(
        default="neuro-trends-suite/0.1", description="Reddit user agent"
    )
    x_bearer_token: str | None = Field(
        default=None, description="X/Twitter Bearer token"
    )
    db_url: str = Field(
        default="postgresql://trends:changeme@postgres:5432/trends",
        description="Database URL",
    )
    postgres_user: str = Field(default="trends")
    postgres_password: str = Field(default="changeme")
    postgres_db: str = Field(default="trends")
    postgres_host: str = Field(default="postgres")
    postgres_port: int = Field(default=5432)
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model"
    )

    # NeuroDegenerAI settings
    adni_data_dir: str = Field(
        default="./neurodegenerai/data/raw", description="ADNI data directory"
    )
    neuro_model_dir: str = Field(
        default="./neurodegenerai/models", description="NeuroDegenerAI model directory"
    )
    neuro_demo_mode: bool = Field(default=True, description="Run in demo mode")
    neuro_data_source: str = Field(
        default="auto",
        description="Tabular training data: 'auto' (real OpenNeuro data when "
        "reachable, else synthetic), 'real', or 'synthetic'",
    )

    # Google Cloud settings
    gcp_project_id: str | None = Field(default=None, description="GCP project ID")
    gcp_region: str = Field(default="us-central1", description="GCP region")
    gcp_sa_key: str | None = Field(default=None, description="GCP service account key")

    # Docker settings
    compose_project_name: str = Field(
        default="neuro-trends-suite", description="Docker Compose project name"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings


# Environment detection helpers
def is_development() -> bool:
    """Check if running in development mode."""
    return get_settings().env.lower() in ("dev", "development")


def is_production() -> bool:
    """Check if running in production mode."""
    return get_settings().env.lower() in ("prod", "production")


def is_demo_mode() -> bool:
    """Check if running in demo mode."""
    return get_settings().neuro_demo_mode


def get_data_dir(project: str) -> str:
    """Get data directory for a project."""
    settings = get_settings()
    if project == "neuro":
        return settings.adni_data_dir
    elif project == "trends":
        return "./trend_detector/data"
    else:
        raise ValueError(f"Unknown project: {project}")


def get_model_dir(project: str) -> str:
    """Get model directory for a project."""
    settings = get_settings()
    if project == "neuro":
        return settings.neuro_model_dir
    elif project == "trends":
        return "./trend_detector/models"
    else:
        raise ValueError(f"Unknown project: {project}")


def ensure_directories() -> None:
    """Ensure required directories exist."""
    settings = get_settings()

    # Create directories if they don't exist
    directories = [
        settings.adni_data_dir,
        settings.neuro_model_dir,
        "./trend_detector/data",
        "./trend_detector/models",
        "./neurodegenerai/reports",
        "./trend_detector/reports",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
