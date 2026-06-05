"""
Metrics collection and monitoring utilities.
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None
    pr_auc: float | None = None
    confusion_matrix: list[list[int]] | None = None
    timestamp: str = ""
    model_name: str = ""
    data_hash: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class PerformanceMetrics:
    """Performance timing metrics."""

    operation: str
    duration: float
    timestamp: str = ""
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class MetricsCollector:
    """Centralized metrics collection and storage."""

    def __init__(self, storage_path: str = "./metrics.json"):
        self.storage_path = storage_path
        self.metrics_history: list[dict[str, Any]] = []
        self.performance_history: list[PerformanceMetrics] = []

    def record_model_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        model_name: str = "unknown",
        data_hash: str = "",
        **kwargs: Any,
    ) -> ModelMetrics:
        """Record model performance metrics."""

        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Calculate additional metrics if probabilities available
        roc_auc = None
        pr_auc = None
        if y_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_proba)
                pr_auc = average_precision_score(y_true, y_proba)
            except ValueError:
                logger.warning("Could not calculate ROC AUC or PR AUC")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred).tolist()

        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            confusion_matrix=cm,
            model_name=model_name,
            data_hash=data_hash,
            **kwargs,
        )

        # Store metrics
        self.metrics_history.append(asdict(metrics))
        self._save_metrics()

        logger.info(
            f"Recorded metrics for {model_name}",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
        )

        return metrics

    def record_performance(
        self, operation: str, duration: float, metadata: dict[str, Any] | None = None
    ) -> None:
        """Record performance timing."""

        perf_metrics = PerformanceMetrics(
            operation=operation, duration=duration, metadata=metadata or {}
        )

        self.performance_history.append(perf_metrics)
        logger.debug(f"Recorded performance: {operation} took {duration:.3f}s")

    def get_latest_metrics(self, model_name: str | None = None) -> ModelMetrics | None:
        """Get latest metrics for a model."""
        if not self.metrics_history:
            return None

        # Filter by model name if specified
        if model_name:
            filtered = [
                m for m in self.metrics_history if m.get("model_name") == model_name
            ]
            if not filtered:
                return None
            latest = filtered[-1]
        else:
            latest = self.metrics_history[-1]

        return ModelMetrics(**latest)

    def get_performance_summary(self, operation: str | None = None) -> dict[str, Any]:
        """Get performance summary for an operation."""
        if not self.performance_history:
            return {}

        # Filter by operation if specified
        filtered = (
            [p for p in self.performance_history if p.operation == operation]
            if operation
            else self.performance_history
        )

        if not filtered:
            return {}

        durations = [p.duration for p in filtered]

        return {
            "operation": operation or "all",
            "count": len(filtered),
            "mean_duration": np.mean(durations),
            "median_duration": np.median(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "std_duration": np.std(durations),
        }

    def _save_metrics(self) -> None:
        """Save metrics to disk."""
        try:
            with open(self.storage_path, "w") as f:
                json.dump(
                    {
                        "model_metrics": self.metrics_history,
                        "performance_metrics": [
                            asdict(p) for p in self.performance_history
                        ],
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def load_metrics(self) -> None:
        """Load metrics from disk."""
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
                self.metrics_history = data.get("model_metrics", [])

                perf_data = data.get("performance_metrics", [])
                self.performance_history = [PerformanceMetrics(**p) for p in perf_data]
        except FileNotFoundError:
            logger.info("No existing metrics file found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")


# Global metrics collector
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
        _metrics_collector.load_metrics()
    return _metrics_collector


def record_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    model_name: str = "unknown",
    data_hash: str = "",
    **kwargs: Any,
) -> ModelMetrics:
    """Record model metrics using global collector."""
    return get_metrics_collector().record_model_metrics(
        y_true, y_pred, y_proba, model_name, data_hash, **kwargs
    )


def record_performance(
    operation: str, duration: float, metadata: dict[str, Any] | None = None
) -> None:
    """Record performance metrics using global collector."""
    get_metrics_collector().record_performance(operation, duration, metadata)


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, operation: str, metadata: dict[str, Any] | None = None):
        self.operation = operation
        self.metadata = metadata
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        record_performance(self.operation, duration, self.metadata)
