"""
Shared library for Neuro-Trends Suite.

This package provides common utilities, configurations, and components
used across both NeuroDegenerAI and Trend Detector projects.
"""

__version__ = "0.1.0"
__author__ = "Neuro-Trends Team"

from .lib.config import Settings, get_settings
from .lib.io_utils import IOUtils
from .lib.logging import get_logger, setup_logging
from .lib.metrics import MetricsCollector
from .lib.ml_utils import MLUtils
from .lib.viz import VisualizationHelper

__all__ = [
    "Settings",
    "get_settings",
    "setup_logging",
    "get_logger",
    "MetricsCollector",
    "VisualizationHelper",
    "MLUtils",
    "IOUtils",
]
