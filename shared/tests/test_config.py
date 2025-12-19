"""
Tests for shared configuration module.
"""

import os

from shared.lib.config import (
    Settings,
    get_settings,
    is_demo_mode,
    is_development,
    is_production,
)


def test_settings_defaults() -> None:
    """Test default settings values."""
    settings = Settings()

    assert settings.env == "dev"
    assert settings.log_level == "INFO"
    assert settings.port == 8080
    assert settings.neuro_demo_mode is True
    assert settings.embedding_model == "all-MiniLM-L6-v2"


def test_settings_from_env() -> None:
    """Test settings loaded from environment variables."""
    os.environ["ENV"] = "production"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["NEURO_DEMO_MODE"] = "false"

    settings = Settings()

    assert settings.env == "production"
    assert settings.log_level == "DEBUG"
    assert settings.neuro_demo_mode is False


def test_get_settings_singleton() -> None:
    """Test that get_settings returns singleton instance."""
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2


def test_environment_helpers() -> None:
    """Test environment helper functions."""
    settings = Settings()

    # Test development mode
    settings.env = "dev"
    assert is_development() is True
    assert is_production() is False

    # Test production mode
    settings.env = "prod"
    assert is_development() is False
    assert is_production() is True


def test_demo_mode() -> None:
    """Test demo mode detection."""
    settings = Settings()

    settings.neuro_demo_mode = True
    assert is_demo_mode() is True

    settings.neuro_demo_mode = False
    assert is_demo_mode() is False


def test_data_directory_paths() -> None:
    """Test data directory path generation."""
    from shared.lib.config import get_data_dir, get_model_dir

    # Test neuro paths
    neuro_data_dir = get_data_dir("neuro")
    neuro_model_dir = get_model_dir("neuro")

    assert "neurodegenerai" in neuro_data_dir
    assert "neurodegenerai" in neuro_model_dir

    # Test trends paths
    trends_data_dir = get_data_dir("trends")
    trends_model_dir = get_model_dir("trends")

    assert "trend-detector" in trends_data_dir
    assert "trend-detector" in trends_model_dir


def test_ensure_directories() -> None:
    """Test directory creation."""
    from shared.lib.config import ensure_directories

    # This should not raise an exception
    ensure_directories()


def test_settings_validation() -> None:
    """Test settings validation."""
    # Test valid settings
    settings = Settings()
    assert settings.port > 0
    assert settings.port < 65536

    # Test embedding model is not empty
    assert len(settings.embedding_model) > 0
