"""
Model interpretability utilities including Grad-CAM and SHAP.
"""

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression

from shared.lib.logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class GradCAMVisualizer(LoggerMixin):
    """Grad-CAM visualization for CNN models."""

    def __init__(self, model: torch.nn.Module, target_layer: str | None = None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []

        # Register hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Find target layer
        if self.target_layer is None:
            # Use the last convolutional layer
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    self.target_layer = name

        # Register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.handlers.append(module.register_forward_hook(forward_hook))
                self.handlers.append(module.register_backward_hook(backward_hook))
                break

        if not self.handlers:
            self.logger.warning("Target layer not found, using default layer")
            # Fallback: use first conv layer
            for _name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    self.handlers.append(module.register_forward_hook(forward_hook))
                    self.handlers.append(module.register_backward_hook(backward_hook))
                    break

    def generate_gradcam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
        save_path: str | None = None,
    ) -> np.ndarray:
        """Generate Grad-CAM visualization."""

        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # Backward pass
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Generate Grad-CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize to input size
        cam_np = cam.detach().cpu().numpy()
        input_size = input_tensor.shape[-2:]
        cam_resized = cv2.resize(cam_np, input_size)

        # Create visualization
        if save_path is not None:
            self._save_gradcam_visualization(
                input_tensor[0, 0].detach().cpu().numpy(), cam_resized, save_path
            )

        return cam_resized

    def _save_gradcam_visualization(
        self, original_image: np.ndarray, cam: np.ndarray, save_path: str
    ) -> None:
        """Save Grad-CAM visualization."""

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(original_image, cmap="gray")
        axes[0].set_title("Original MRI Slice")
        axes[0].axis("off")

        # Grad-CAM heatmap
        im = axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1])

        # Overlay
        axes[2].imshow(original_image, cmap="gray")
        axes[2].imshow(cam, cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Grad-CAM visualization saved to {save_path}")

    def cleanup(self) -> None:
        """Remove hooks."""
        for handler in self.handlers:
            handler.remove()
        self.handlers = []


class SHAPExplainer(LoggerMixin):
    """SHAP explanation for tabular models."""

    def __init__(self, model: Any, background_data: np.ndarray):
        self.model = model
        self.background_data = background_data

        # Initialize SHAP explainer
        if hasattr(model, "predict_proba"):
            self.explainer = shap.TreeExplainer(model)
        else:
            # Fallback to linear explainer
            self.explainer = shap.LinearExplainer(
                LinearRegression().fit(background_data, np.zeros(len(background_data))),
                background_data,
            )

    def explain_prediction(self, instance: np.ndarray) -> dict[str, Any]:
        """Explain a single prediction."""

        # Get SHAP values
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))

        # Get feature names (if available)
        feature_names = getattr(
            self.model, "feature_name_", [f"feature_{i}" for i in range(len(instance))]
        )

        # Create explanation
        explanation = {
            "shap_values": (
                shap_values[0].tolist()
                if len(shap_values.shape) > 1
                else shap_values.tolist()
            ),
            "feature_names": feature_names,
            "base_value": self.explainer.expected_value,
            "prediction": self.model.predict(instance.reshape(1, -1))[0],
            "prediction_proba": self.model.predict_proba(instance.reshape(1, -1))[
                0
            ].tolist(),
        }

        return explanation

    def explain_batch(self, instances: np.ndarray) -> list[dict[str, Any]]:
        """Explain batch of predictions."""

        explanations = []

        for i, instance in enumerate(instances):
            try:
                explanation = self.explain_prediction(instance)
                explanations.append(explanation)
            except Exception as e:
                self.logger.error(f"Error explaining instance {i}: {e}")
                explanations.append({"error": str(e), "instance_index": i})

        return explanations

    def get_feature_importance_graph(
        self, instances: np.ndarray, save_path: str | None = None
    ) -> None:
        """Generate SHAP feature importance plot."""

        # Get SHAP values for all instances
        shap_values = self.explainer.shap_values(instances)

        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, instances, show=False)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            self.logger.info(f"SHAP summary plot saved to {save_path}")
        else:
            plt.show()


class IntegratedGradients:
    """Integrated Gradients for CNN interpretability."""

    def __init__(self, model: torch.nn.Module, steps: int = 50):
        self.model = model
        self.steps = steps
        self.logger = get_logger(__name__)

    def generate_integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
        baseline: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate Integrated Gradients attribution."""

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # Generate interpolated inputs
        interpolated_inputs = []
        for i in range(self.steps):
            alpha = i / (self.steps - 1)
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated_inputs.append(interpolated)

        interpolated_inputs = torch.cat(interpolated_inputs, dim=0)
        interpolated_inputs.requires_grad_(True)

        # Forward pass
        outputs = self.model(interpolated_inputs)
        target_scores = outputs[:, target_class]

        # Backward pass
        gradients = torch.autograd.grad(
            outputs=target_scores,
            inputs=interpolated_inputs,
            grad_outputs=torch.ones_like(target_scores),
            create_graph=False,
            retain_graph=False,
        )[0]

        # Average gradients
        avg_gradients = torch.mean(gradients, dim=0, keepdim=True)

        # Integrated gradients
        integrated_gradients = (input_tensor - baseline) * avg_gradients

        return integrated_gradients

    def visualize_integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
        save_path: str | None = None,
    ) -> np.ndarray:
        """Visualize Integrated Gradients."""

        # Generate integrated gradients
        ig = self.generate_integrated_gradients(input_tensor, target_class)

        # Convert to numpy
        ig_np = ig[0, 0].detach().cpu().numpy()  # Remove batch and channel dimensions

        # Normalize
        ig_np = np.abs(ig_np)
        ig_np = (ig_np - ig_np.min()) / (ig_np.max() - ig_np.min() + 1e-8)

        # Create visualization
        if save_path is not None:
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(input_tensor[0, 0].detach().cpu().numpy(), cmap="gray")
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(ig_np, cmap="hot")
            plt.title("Integrated Gradients")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Integrated Gradients visualization saved to {save_path}")

        return ig_np


class ModelInterpretability:
    """Main class for model interpretability."""

    def __init__(self, model: Any, model_type: str = "tabular"):
        self.model = model
        self.model_type = model_type
        self.logger = get_logger(__name__)

        # Initialize appropriate explainer
        if model_type == "tabular":
            self.shap_explainer = None
        elif model_type == "cnn":
            self.gradcam_visualizer = None
            self.integrated_gradients = None

    def explain_tabular_prediction(
        self, instance: np.ndarray, background_data: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Explain tabular model prediction."""

        if self.shap_explainer is None:
            if background_data is None:
                raise ValueError("Background data required for SHAP explainer")
            self.shap_explainer = SHAPExplainer(self.model, background_data)

        return self.shap_explainer.explain_prediction(instance)

    def explain_cnn_prediction(
        self, input_tensor: torch.Tensor, target_class: int = 1, method: str = "gradcam"
    ) -> dict[str, Any]:
        """Explain CNN model prediction."""

        if method == "gradcam":
            if self.gradcam_visualizer is None:
                self.gradcam_visualizer = GradCAMVisualizer(self.model)

            gradcam = self.gradcam_visualizer.generate_gradcam(
                input_tensor, target_class
            )

            return {
                "method": "gradcam",
                "attribution": gradcam,
                "target_class": target_class,
            }

        elif method == "integrated_gradients":
            if self.integrated_gradients is None:
                self.integrated_gradients = IntegratedGradients(self.model)

            ig = self.integrated_gradients.generate_integrated_gradients(
                input_tensor, target_class
            )

            return {
                "method": "integrated_gradients",
                "attribution": ig[0, 0].detach().cpu().numpy(),
                "target_class": target_class,
            }

        else:
            raise ValueError(f"Unknown explanation method: {method}")

    def generate_explanation_report(
        self,
        data: Any,
        predictions: Any,
        save_dir: str = "./neurodegenerai/reports/interpretability",
    ) -> str:
        """Generate comprehensive explanation report."""

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.model_type == "tabular":
            # Generate SHAP plots
            if hasattr(data, "values"):
                data_array = data.values
            else:
                data_array = data

            # Summary plot
            summary_plot_path = save_path / f"shap_summary_{timestamp}.png"
            self.shap_explainer.get_feature_importance_graph(
                data_array, str(summary_plot_path)
            )

        elif self.model_type == "cnn":
            # Generate Grad-CAM visualizations
            for i, (input_tensor, prediction) in enumerate(
                zip(data, predictions, strict=False)
            ):
                save_path / f"gradcam_sample_{i}_{timestamp}.png"
                self.explain_cnn_prediction(
                    input_tensor, target_class=prediction, method="gradcam"
                )

        self.logger.info(f"Explanation report generated in {save_path}")

        return str(save_path)
