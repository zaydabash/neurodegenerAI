"""
Structured logging configuration using Loguru.
"""

import json
import sys
from typing import Any

from loguru import logger

from .config import get_settings


def setup_logging(
    level: str | None = None, structured: bool = True, service_name: str | None = None
) -> None:
    """Setup structured logging configuration."""

    settings = get_settings()
    log_level = level or settings.log_level

    # Remove default handler
    logger.remove()

    # Configure structured logging
    if structured:
        # Structured JSON logging.
        #
        # We write JSON from a sink function rather than from ``format``.
        # Loguru treats a ``format`` return value as a *template* and calls
        # ``.format_map()`` on it, so a JSON string (which contains ``{`` and
        # ``}``) would be misparsed as format fields and raise ``KeyError``.
        def json_sink(message: Any) -> None:
            record = message.record
            subset = {
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "logger": record["name"],
                "function": record["function"],
                "line": record["line"],
                "message": record["message"],
                "service": service_name or "neuro-trends-suite",
            }

            # Add extra fields if present
            if record.get("extra"):
                subset.update(record["extra"])

            print(json.dumps(subset, default=str), file=sys.stdout)

        logger.add(
            json_sink,
            level=log_level,
            backtrace=True,
            diagnose=True,
        )
    else:
        # Human-readable format for development
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        logger.add(
            sys.stdout,
            format=log_format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # Add file logging for production
    if settings.env.lower() in ("prod", "production"):
        logger.add(
            "logs/app.log",
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            level=log_level,
            serialize=structured,
        )


def get_logger(name: str) -> Any:
    """Get a logger instance for a specific module."""
    return logger.bind(name=name)


class LoggerMixin:
    """Mixin class to add logging capabilities."""

    @property
    def logger(self) -> Any:
        """Get logger for this class."""
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")


def log_function_call(func_name: str, **kwargs: Any) -> None:
    """Log function call with parameters."""
    logger.debug(f"Calling {func_name}", **kwargs)


def log_performance(operation: str, duration: float, **kwargs: Any) -> None:
    """Log performance metrics."""
    logger.info(f"Performance: {operation}", duration=duration, **kwargs)


def log_error(error: Exception, context: str | None = None, **kwargs: Any) -> None:
    """Log error with context."""
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        **kwargs,
    }
    logger.error("Error occurred", **error_info)


def log_prediction(
    model_name: str,
    prediction: Any,
    confidence: float | None = None,
    input_hash: str | None = None,
    **kwargs: Any,
) -> None:
    """Log model prediction."""
    prediction_info = {
        "model_name": model_name,
        "prediction": str(prediction),
        "confidence": confidence,
        "input_hash": input_hash,
        **kwargs,
    }
    logger.info("Model prediction", **prediction_info)


def log_api_request(
    endpoint: str,
    method: str,
    status_code: int,
    duration: float | None = None,
    **kwargs: Any,
) -> None:
    """Log API request."""
    request_info = {
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "duration": duration,
        **kwargs,
    }
    logger.info("API request", **request_info)
