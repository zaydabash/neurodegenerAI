"""
Visualization utilities and helpers.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .logging import get_logger

logger = get_logger(__name__)


class VisualizationHelper:
    """Helper class for creating visualizations."""

    def __init__(self, style: str = "default", figsize: tuple[int, int] = (10, 6)):
        self.style = style
        self.figsize = figsize
        self._setup_style()

    def _setup_style(self) -> None:
        """Setup matplotlib and seaborn styles."""
        plt.style.use("default")
        sns.set_palette("husl")

        # Set default figure size
        plt.rcParams["figure.figsize"] = self.figsize
        plt.rcParams["font.size"] = 12

        if self.style == "dark":
            plt.style.use("dark_background")
        elif self.style == "seaborn":
            sns.set_style("whitegrid")

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "ROC Curve",
        save_path: str | None = None,
    ) -> go.Figure:
        """Create ROC curve plot."""

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = np.trapz(tpr, fpr)

        fig = go.Figure()

        # ROC curve
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC Curve (AUC = {roc_auc:.3f})",
                line={"width": 3},
            )
        )

        # Diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line={"dash": "dash", "color": "gray"},
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1]},
            width=600,
            height=500,
        )

        if save_path:
            fig.write_image(save_path)
            logger.info(f"ROC curve saved to {save_path}")

        return fig

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Precision-Recall Curve",
        save_path: str | None = None,
    ) -> go.Figure:
        """Create precision-recall curve plot."""

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = np.trapz(precision, recall)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"PR Curve (AUC = {pr_auc:.3f})",
                line={"width": 3},
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1]},
            width=600,
            height=500,
        )

        if save_path:
            fig.write_image(save_path)
            logger.info(f"Precision-recall curve saved to {save_path}")

        return fig

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[str] | None = None,
        title: str = "Confusion Matrix",
        save_path: str | None = None,
    ) -> go.Figure:
        """Create confusion matrix heatmap."""

        cm = confusion_matrix(y_true, y_pred)

        if labels is None:
            labels = [f"Class {i}" for i in range(len(cm))]

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=600,
            height=500,
        )

        if save_path:
            fig.write_image(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        title: str = "Calibration Curve",
        save_path: str | None = None,
    ) -> go.Figure:
        """Create calibration curve plot."""

        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # Calculate calibration
        calibration_error = 0
        bin_centers = []
        bin_means = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                bin_mean = y_proba[in_bin].mean()
                bin_center = (bin_lower + bin_upper) / 2
                bin_count = in_bin.sum()

                bin_centers.append(bin_center)
                bin_means.append(bin_mean)
                bin_counts.append(bin_count)

                calibration_error += np.abs(bin_mean - bin_center) * prop_in_bin

        fig = go.Figure()

        # Perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect Calibration",
                line={"dash": "dash", "color": "gray"},
            )
        )

        # Actual calibration
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=bin_means,
                mode="markers+lines",
                name=f"Model (ECE = {calibration_error:.3f})",
                marker={"size": 8},
                line={"width": 3},
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1]},
            width=600,
            height=500,
        )

        if save_path:
            fig.write_image(save_path)
            logger.info(f"Calibration curve saved to {save_path}")

        return fig

    def plot_feature_importance(
        self,
        importance_scores: dict[str, float],
        title: str = "Feature Importance",
        top_k: int = 20,
        save_path: str | None = None,
    ) -> go.Figure:
        """Create feature importance plot."""

        # Sort features by importance
        sorted_features = sorted(
            importance_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        features, scores = zip(*sorted_features, strict=False)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(x=scores, y=features, orientation="h", marker_color="lightblue")
        )

        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(features) * 20),
            width=800,
        )

        if save_path:
            fig.write_image(save_path)
            logger.info(f"Feature importance plot saved to {save_path}")

        return fig

    def plot_topic_evolution(
        self,
        topic_data: pd.DataFrame,
        title: str = "Topic Evolution",
        save_path: str | None = None,
    ) -> go.Figure:
        """Create topic evolution timeline plot."""

        fig = go.Figure()

        # Plot each topic as a line
        for topic in topic_data["topic"].unique():
            topic_subset = topic_data[topic_data["topic"] == topic]
            fig.add_trace(
                go.Scatter(
                    x=topic_subset["timestamp"],
                    y=topic_subset["volume"],
                    mode="lines+markers",
                    name=topic,
                    line={"width": 2},
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Volume",
            hovermode="x unified",
            width=1000,
            height=600,
        )

        if save_path:
            fig.write_image(save_path)
            logger.info(f"Topic evolution plot saved to {save_path}")

        return fig

    def plot_umap_clusters(
        self,
        embeddings: np.ndarray,
        clusters: np.ndarray,
        texts: list[str] | None = None,
        title: str = "Topic Clusters (UMAP)",
        save_path: str | None = None,
    ) -> go.Figure:
        """Create UMAP cluster visualization."""

        fig = go.Figure()

        # Plot each cluster
        for cluster_id in np.unique(clusters):
            if cluster_id == -1:  # Noise points
                continue

            mask = clusters == cluster_id
            cluster_embeddings = embeddings[mask]
            cluster_texts = texts[mask] if texts is not None else None

            fig.add_trace(
                go.Scatter(
                    x=cluster_embeddings[:, 0],
                    y=cluster_embeddings[:, 1],
                    mode="markers",
                    name=f"Cluster {cluster_id}",
                    text=cluster_texts,
                    hovertemplate="%{text}<br>Cluster: %{legendgroup}<extra></extra>",
                    marker={"size": 6},
                )
            )

        # Plot noise points
        noise_mask = clusters == -1
        if noise_mask.any():
            noise_embeddings = embeddings[noise_mask]
            noise_texts = texts[noise_mask] if texts is not None else None

            fig.add_trace(
                go.Scatter(
                    x=noise_embeddings[:, 0],
                    y=noise_embeddings[:, 1],
                    mode="markers",
                    name="Noise",
                    text=noise_texts,
                    hovertemplate="%{text}<br>Noise<extra></extra>",
                    marker={"size": 4, "color": "gray"},
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            width=800,
            height=600,
        )

        if save_path:
            fig.write_image(save_path)
            logger.info(f"UMAP clusters plot saved to {save_path}")

        return fig


# Global visualization helper
_viz_helper: VisualizationHelper | None = None


def get_viz_helper() -> VisualizationHelper:
    """Get global visualization helper instance."""
    global _viz_helper
    if _viz_helper is None:
        _viz_helper = VisualizationHelper()
    return _viz_helper
