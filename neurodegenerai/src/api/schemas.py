"""
Pydantic schemas for NeuroDegenerAI API.
"""

from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, validator


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "0.1.0"
    model_loaded: bool = False


class TabularPredictionRequest(BaseModel):
    """Request schema for tabular predictions."""

    # Demographics
    age: float = Field(..., ge=0, le=120, description="Patient age")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=Female, 1=Male)")

    # Genetic
    apoe4: int = Field(
        ...,
        ge=0,
        le=2,
        description="APOE4 status (0=negative, 1=heterozygous, 2=homozygous)",
    )

    # Cognitive
    mmse: float = Field(..., ge=0, le=30, description="MMSE score")
    cdr: float | None = Field(None, ge=0, le=3, description="Clinical Dementia Rating")

    # Biomarkers
    abeta: float | None = Field(None, description="Amyloid beta levels")
    tau: float | None = Field(None, description="Tau protein levels")
    ptau: float | None = Field(None, description="Phosphorylated tau levels")

    # Additional features
    education: float | None = Field(None, ge=0, le=25, description="Years of education")
    hippocampal_volume: float | None = Field(None, description="Hippocampal volume")
    cortical_thickness: float | None = Field(None, description="Cortical thickness")
    white_matter_hyperintensities: float | None = Field(
        None, description="White matter hyperintensities"
    )

    class Config:
        schema_extra = {
            "example": {
                "age": 75.0,
                "sex": 0,
                "apoe4": 1,
                "mmse": 24.0,
                "cdr": 0.5,
                "abeta": 180.0,
                "tau": 350.0,
                "ptau": 28.0,
                "education": 16.0,
                "hippocampal_volume": 2800.0,
                "cortical_thickness": 2.3,
                "white_matter_hyperintensities": 8.5,
            }
        }


class MRIPredictionRequest(BaseModel):
    """Request schema for MRI predictions."""

    volume_data: list[list[list[float]]] = Field(..., description="3D MRI volume data")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")

    @validator("volume_data")
    def validate_volume_data(cls, v: Any) -> Any:
        """Validate volume data format."""
        if not v:
            raise ValueError("Volume data cannot be empty")

        # Check if it's a 3D array
        try:
            volume_array = np.array(v)
            if len(volume_array.shape) != 3:
                raise ValueError("Volume data must be 3D")

            if volume_array.size == 0:
                raise ValueError("Volume data cannot be empty")

        except Exception as e:
            raise ValueError(f"Invalid volume data format: {e}") from e

        return v

    class Config:
        schema_extra = {
            "example": {
                "volume_data": [
                    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                    [[1.0, 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]],
                    [[2.0, 2.1, 2.2], [2.3, 2.4, 2.5], [2.6, 2.7, 2.8]],
                ],
                "metadata": {
                    "scan_date": "2024-01-01",
                    "scanner": "3T MRI",
                    "sequence": "T1-weighted",
                },
            }
        }


class EEGPredictionRequest(BaseModel):
    """Request schema for EEG predictions."""

    data: list[list[float]] = Field(
        ..., description="Multi-channel EEG time-series data"
    )
    sfreq: int = Field(250, description="Sampling frequency")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")

    @validator("data")
    def validate_eeg_data(cls, v: Any) -> Any:
        """Validate EEG data format."""
        if not v:
            raise ValueError("EEG data cannot be empty")

        try:
            eeg_array = np.array(v)
            if len(eeg_array.shape) != 2:
                raise ValueError("EEG data must be 2D (channels, length)")
        except Exception as e:
            raise ValueError(f"Invalid EEG data format: {e}") from e

        return v

    class Config:
        schema_extra = {
            "example": {
                "data": [
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                ],
                "sfreq": 250,
                "metadata": {
                    "patient_state": "resting",
                    "num_channels": 8,
                },
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions."""

    prediction: int = Field(
        ..., description="Predicted class (0=Normal, 1=Dementia/MCI)"
    )
    probability: float = Field(..., ge=0, le=1, description="Prediction probability")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    model_name: str = Field(..., description="Model used for prediction")
    model_type: str = Field(..., description="Type of model (tabular/cnn/ensemble)")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Additional information
    explanation: dict[str, Any] | None = Field(None, description="Model explanation")
    heatmap_paths: list[str] | None = Field(
        None, description="Paths to heatmap visualizations"
    )
    error: str | None = Field(None, description="Error message if prediction failed")


class ModelMetricsResponse(BaseModel):
    """Response schema for model metrics."""

    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="Model F1 score")
    roc_auc: float | None = Field(None, description="ROC AUC score")
    pr_auc: float | None = Field(None, description="Precision-Recall AUC score")

    # Additional metrics
    confusion_matrix: list[list[int]] | None = Field(
        None, description="Confusion matrix"
    )
    classification_report: dict[str, Any] | None = Field(
        None, description="Detailed classification report"
    )

    # Model information
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    timestamp: datetime = Field(default_factory=datetime.now)
    data_hash: str = Field(..., description="Hash of evaluation data")


class FeatureImportanceResponse(BaseModel):
    """Response schema for feature importance."""

    feature_importance: dict[str, float] = Field(
        ..., description="Feature importance scores"
    )
    top_features: list[dict[str, Any]] = Field(
        ..., description="Top features with details"
    )
    model_name: str = Field(..., description="Model name")
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    data: list[dict[str, Any]] = Field(
        ..., description="List of data points for prediction"
    )
    model_type: str = Field(default="tabular", description="Type of model to use")

    @validator("data")
    def validate_data(cls, v: Any) -> Any:
        """Validate batch data."""
        if not v:
            raise ValueError("Data list cannot be empty")

        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000")

        return v

    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {
                        "age": 75.0,
                        "sex": 0,
                        "apoe4": 1,
                        "mmse": 24.0,
                        "abeta": 180.0,
                        "tau": 350.0,
                        "ptau": 28.0,
                    },
                    {
                        "age": 68.0,
                        "sex": 1,
                        "apoe4": 0,
                        "mmse": 28.0,
                        "abeta": 220.0,
                        "tau": 280.0,
                        "ptau": 22.0,
                    },
                ],
                "model_type": "tabular",
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[PredictionResponse] = Field(
        ..., description="List of predictions"
    )
    total_samples: int = Field(..., description="Total number of samples")
    successful_predictions: int = Field(
        ..., description="Number of successful predictions"
    )
    failed_predictions: int = Field(..., description="Number of failed predictions")
    model_name: str = Field(..., description="Model used for predictions")
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: str | None = Field(None, description="Request ID for tracking")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    model_class: str = Field(..., description="Model class")
    version: str = Field(..., description="Model version")

    # Training information
    training_date: str | None = Field(None, description="Training date")
    data_hash: str | None = Field(None, description="Training data hash")

    # Performance information
    accuracy: float | None = Field(None, description="Model accuracy")
    auc: float | None = Field(None, description="Model AUC")

    # Feature information
    num_features: int | None = Field(None, description="Number of features")
    feature_names: list[str] | None = Field(None, description="Feature names")

    # Model parameters
    parameters: dict[str, Any] | None = Field(None, description="Model parameters")

    timestamp: datetime = Field(default_factory=datetime.now)


class CalibrationResponse(BaseModel):
    """Calibration response schema."""

    calibration_score: float = Field(..., description="Calibration score (Brier score)")
    reliability_diagram: dict[str, Any] | None = Field(
        None, description="Reliability diagram data"
    )
    calibration_curve: list[dict[str, float]] | None = Field(
        None, description="Calibration curve points"
    )
    model_name: str = Field(..., description="Model name")
    timestamp: datetime = Field(default_factory=datetime.now)


class ExplanationRequest(BaseModel):
    """Request schema for model explanations."""

    data: dict[str, Any] = Field(..., description="Data for explanation")
    model_type: str = Field(default="tabular", description="Type of model")
    explanation_method: str = Field(default="shap", description="Explanation method")

    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "age": 75.0,
                    "sex": 0,
                    "apoe4": 1,
                    "mmse": 24.0,
                    "abeta": 180.0,
                    "tau": 350.0,
                    "ptau": 28.0,
                },
                "model_type": "tabular",
                "explanation_method": "shap",
            }
        }


class ExplanationResponse(BaseModel):
    """Response schema for model explanations."""

    explanation_method: str = Field(..., description="Explanation method used")
    explanation_data: dict[str, Any] = Field(..., description="Explanation data")
    feature_importance: dict[str, float] | None = Field(
        None, description="Feature importance scores"
    )
    visualization_path: str | None = Field(None, description="Path to visualization")
    model_name: str = Field(..., description="Model name")
    timestamp: datetime = Field(default_factory=datetime.now)
