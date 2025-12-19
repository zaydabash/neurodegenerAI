"""
Prediction utilities for trained models.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from shared.lib.config import get_model_dir, get_settings
from shared.lib.logging import LoggerMixin, get_logger

from ..data.features_mri import MRIDataLoader, MRIPreprocessor
from ..data.features_tabular import TabularFeatureEngineer
from .train_cnn import ResNet18Adapted, SimpleCNN

logger = get_logger(__name__)


class ModelPredictor(LoggerMixin):
    """Model predictor for both tabular and MRI models."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name
        self.settings = get_settings()
        self.model_dir = Path(get_model_dir("neuro"))

        # Model components
        self.tabular_model = None
        self.cnn_model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.mri_preprocessor = None

        # Device for CNN models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load latest model if no specific model name provided
        if self.model_name is None:
            self.model_name = self._get_latest_model()

        self.logger.info(f"Initialized predictor with model: {self.model_name}")

    def predict_tabular(self, data: pd.DataFrame | dict[str, Any]) -> dict[str, Any]:
        """Predict using tabular model."""

        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        self.logger.info(f"Making tabular prediction for {len(data)} samples")

        # Load model if not already loaded
        if self.tabular_model is None:
            self._load_tabular_model()

        # Feature engineering
        if self.feature_engineer is None:
            self.feature_engineer = TabularFeatureEngineer()

        X_engineered = self.feature_engineer.engineer_features(data)

        # Preprocessing
        X_processed = self.preprocessor.transform(X_engineered)

        # Make prediction
        prediction = self.tabular_model.predict(X_processed)
        probability = self.tabular_model.predict_proba(X_processed)[:, 1]

        # Calibrated probability if available
        calibrated_probability = None
        if hasattr(self.tabular_model, "predict_proba"):
            calibrated_probability = self.tabular_model.predict_proba(X_processed)[:, 1]

        # Feature importance (SHAP-like explanation)
        explanation = self._get_tabular_explanation(X_processed, prediction[0])

        return {
            "prediction": int(prediction[0]),
            "probability": float(probability[0]),
            "calibrated_probability": (
                float(calibrated_probability[0])
                if calibrated_probability is not None
                else float(probability[0])
            ),
            "confidence": float(max(probability[0], 1 - probability[0])),
            "explanation": explanation,
            "model_name": self.model_name,
            "model_type": "tabular",
        }

    def predict_mri(self, volume: np.ndarray) -> dict[str, Any]:
        """Predict using CNN model."""

        self.logger.info("Making MRI prediction")

        # Load model if not already loaded
        if self.cnn_model is None:
            self._load_cnn_model()

        # Preprocess volume
        if self.mri_preprocessor is None:
            self.mri_preprocessor = MRIPreprocessor()

        processed_volume = self.mri_preprocessor.preprocess_volume(volume)

        # Extract slices
        mri_loader = MRIDataLoader()
        slices = mri_loader.extract_slices(processed_volume)

        # Convert to tensor
        slices_tensor = torch.stack(
            [
                torch.from_numpy(slice_data).unsqueeze(0)  # Add channel dimension
                for slice_data in slices
            ]
        )

        # Make prediction
        self.cnn_model.eval()
        with torch.no_grad():
            slices_tensor = slices_tensor.to(self.device)
            outputs = self.cnn_model(slices_tensor)
            probabilities = F.softmax(outputs, dim=1)

            # Average predictions across slices
            avg_probability = probabilities.mean(dim=0)
            prediction = torch.argmax(avg_probability).item()
            confidence = torch.max(avg_probability).item()

        # Generate Grad-CAM visualization
        heatmap_paths = self._generate_gradcam_visualizations(slices_tensor, slices)

        return {
            "prediction": int(prediction),
            "probability": float(avg_probability[1].item()),
            "confidence": float(confidence),
            "heatmap_paths": heatmap_paths,
            "model_name": self.model_name,
            "model_type": "cnn",
        }

    def _load_tabular_model(self) -> None:
        """Load tabular model and preprocessor."""

        # Find model file
        model_files = list(self.model_dir.glob(f"*{self.model_name}*.joblib"))

        if not model_files:
            raise FileNotFoundError(f"Tabular model not found: {self.model_name}")

        model_file = model_files[0]

        # Load model data
        save_data = joblib.load(model_file)

        self.tabular_model = save_data.get("model")
        self.preprocessor = save_data.get("preprocessor")

        if self.tabular_model is None:
            raise ValueError("Model not found in save data")

        if self.preprocessor is None:
            raise ValueError("Preprocessor not found in save data")

        self.logger.info(f"Loaded tabular model from {model_file}")

    def _load_cnn_model(self) -> None:
        """Load CNN model."""

        # Find model file
        model_files = list(self.model_dir.glob(f"*{self.model_name}*.pth"))

        if not model_files:
            raise FileNotFoundError(f"CNN model not found: {self.model_name}")

        model_file = model_files[0]

        # Load model checkpoint
        checkpoint = torch.load(model_file, map_location=self.device)

        # Initialize model
        model_type = checkpoint.get("model_type", "simple_cnn")
        if model_type == "simple_cnn":
            self.cnn_model = SimpleCNN(num_classes=2, input_channels=1)
        elif model_type == "resnet18":
            self.cnn_model = ResNet18Adapted(num_classes=2, input_channels=1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load state dict
        self.cnn_model.load_state_dict(checkpoint["model_state_dict"])
        self.cnn_model = self.cnn_model.to(self.device)

        self.logger.info(f"Loaded CNN model from {model_file}")

    def _get_tabular_explanation(
        self, X: pd.DataFrame, prediction: int
    ) -> dict[str, Any]:
        """Get feature importance explanation for tabular prediction."""

        # Get feature importance from model
        if hasattr(self.tabular_model, "feature_importances_"):
            importance_scores = dict(
                zip(X.columns, self.tabular_model.feature_importances_, strict=False)
            )
        else:
            # Fallback: use feature values as importance
            importance_scores = dict(
                zip(X.columns, X.iloc[0].abs().values, strict=False)
            )

        # Sort by importance
        sorted_features = sorted(
            importance_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Get top 10 features
        top_features = sorted_features[:10]

        return {
            "top_features": [
                {
                    "feature": feature,
                    "value": float(X.iloc[0][feature]),
                    "importance": float(importance),
                }
                for feature, importance in top_features
            ],
            "prediction_factors": {
                "positive_factors": [
                    {
                        "feature": f,
                        "value": float(X.iloc[0][f]),
                        "importance": float(imp),
                    }
                    for f, imp in top_features[:5]
                    if X.iloc[0][f] > 0
                ],
                "negative_factors": [
                    {
                        "feature": f,
                        "value": float(X.iloc[0][f]),
                        "importance": float(imp),
                    }
                    for f, imp in top_features[:5]
                    if X.iloc[0][f] < 0
                ],
            },
        }

    def _generate_gradcam_visualizations(
        self, slices_tensor: torch.Tensor, slices: list[np.ndarray]
    ) -> list[str]:
        """Generate Grad-CAM visualizations for MRI slices."""

        from .interpretability import GradCAMVisualizer

        visualizer = GradCAMVisualizer(self.cnn_model)

        heatmap_paths = []
        for i, (slice_tensor, _slice_data) in enumerate(
            zip(slices_tensor, slices, strict=False)
        ):
            try:
                heatmap_path = visualizer.generate_gradcam(
                    slice_tensor.unsqueeze(0),
                    target_class=1,  # Assume class 1 is positive
                    save_path=f"./neurodegenerai/app/assets/heatmap_slice_{i}.png",
                )
                heatmap_paths.append(heatmap_path)
            except Exception as e:
                self.logger.warning(f"Could not generate Grad-CAM for slice {i}: {e}")

        return heatmap_paths

    def _get_latest_model(self) -> str:
        """Get the latest trained model name."""

        # Look for tabular models
        tabular_models = list(self.model_dir.glob("*tabular*.joblib"))
        cnn_models = list(self.model_dir.glob("*cnn*.pth"))

        if tabular_models:
            # Return the latest tabular model
            latest_model = max(tabular_models, key=lambda x: x.stat().st_mtime)
            return latest_model.stem.split("_")[0] + "_tabular"
        elif cnn_models:
            # Return the latest CNN model
            latest_model = max(cnn_models, key=lambda x: x.stat().st_mtime)
            return latest_model.stem.split("_")[0] + "_cnn"
        else:
            raise FileNotFoundError("No trained models found")

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""

        info = {
            "model_name": self.model_name,
            "model_dir": str(self.model_dir),
            "device": str(self.device),
        }

        if self.tabular_model is not None:
            info["model_type"] = "tabular"
            info["model_class"] = type(self.tabular_model).__name__

            if hasattr(self.tabular_model, "feature_importances_"):
                info["num_features"] = len(self.tabular_model.feature_importances_)

        if self.cnn_model is not None:
            info["model_type"] = "cnn"
            info["model_class"] = type(self.cnn_model).__name__
            info["num_parameters"] = sum(p.numel() for p in self.cnn_model.parameters())

        return info

    def batch_predict_tabular(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Make batch predictions for tabular data."""

        predictions = []

        for idx, row in data.iterrows():
            try:
                prediction = self.predict_tabular(row.to_dict())
                predictions.append(prediction)
            except Exception as e:
                self.logger.error(f"Error predicting row {idx}: {e}")
                predictions.append(
                    {"prediction": -1, "probability": 0.0, "error": str(e)}
                )

        return predictions

    def batch_predict_mri(self, volumes: list[np.ndarray]) -> list[dict[str, Any]]:
        """Make batch predictions for MRI volumes."""

        predictions = []

        for i, volume in enumerate(volumes):
            try:
                prediction = self.predict_mri(volume)
                predictions.append(prediction)
            except Exception as e:
                self.logger.error(f"Error predicting volume {i}: {e}")
                predictions.append(
                    {"prediction": -1, "probability": 0.0, "error": str(e)}
                )

        return predictions


class ModelEnsemble:
    """Ensemble predictor combining tabular and CNN models."""

    def __init__(self):
        self.tabular_predictor = ModelPredictor()
        self.cnn_predictor = ModelPredictor()
        self.logger = get_logger(__name__)

    def predict_ensemble(
        self, tabular_data: pd.DataFrame, mri_volume: np.ndarray
    ) -> dict[str, Any]:
        """Make ensemble prediction using both models."""

        # Get predictions from both models
        tabular_pred = self.tabular_predictor.predict_tabular(tabular_data)
        cnn_pred = self.cnn_predictor.predict_mri(mri_volume)

        # Combine predictions (weighted average)
        tabular_weight = 0.6
        cnn_weight = 0.4

        combined_probability = (
            tabular_weight * tabular_pred["probability"]
            + cnn_weight * cnn_pred["probability"]
        )

        combined_prediction = 1 if combined_probability > 0.5 else 0
        combined_confidence = max(combined_probability, 1 - combined_probability)

        return {
            "prediction": combined_prediction,
            "probability": combined_probability,
            "confidence": combined_confidence,
            "tabular_prediction": tabular_pred,
            "cnn_prediction": cnn_pred,
            "model_type": "ensemble",
        }
