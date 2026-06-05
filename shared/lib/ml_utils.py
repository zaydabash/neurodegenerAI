"""
Machine learning utilities and helpers.
"""

import hashlib
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from .config import get_model_dir
from .logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class DataPreprocessor(LoggerMixin):
    """Data preprocessing utilities."""

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.is_fitted = False

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Fit preprocessor and transform data."""

        self.logger.info(f"Fitting preprocessor with {len(X)} samples")

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Handle missing values
        X_clean = self._handle_missing_values(X.copy())

        # Initialize and fit scaler
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_clean),
            columns=self.feature_names,
            index=X.index,
        )

        # Handle target encoding if needed
        y_encoded = None
        if y is not None:
            if y.dtype == "object" or y.dtype.name == "category":
                self.label_encoder = LabelEncoder()
                y_encoded = pd.Series(
                    self.label_encoder.fit_transform(y), index=y.index, name=y.name
                )
                self.logger.info(f"Encoded {len(self.label_encoder.classes_)} classes")
            else:
                y_encoded = y.copy()

        self.is_fitted = True
        self.logger.info("Preprocessor fitted successfully")

        return X_scaled, y_encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""

        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        self.logger.info(f"Transforming {len(X)} samples")

        # Handle missing values
        X_clean = self._handle_missing_values(X.copy())

        # Ensure same features as training
        missing_features = set(self.feature_names) - set(X_clean.columns)
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
            for feature in missing_features:
                X_clean[feature] = 0

        # Reorder columns to match training
        X_clean = X_clean[self.feature_names]

        # Scale
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_clean), columns=self.feature_names, index=X.index
        )

        return X_scaled

    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Inverse transform encoded target values."""

        if self.label_encoder is None:
            return y_encoded

        return self.label_encoder.inverse_transform(y_encoded)

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""

        # Get numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns

        # Fill numeric columns with median
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        # Fill categorical columns with mode
        for col in categorical_cols:
            mode_val = X[col].mode()
            if len(mode_val) > 0:
                X[col] = X[col].fillna(mode_val[0])
            else:
                X[col] = X[col].fillna("Unknown")

        return X


class ModelManager(LoggerMixin):
    """Model management utilities."""

    def __init__(self, project: str):
        self.project = project
        self.model_dir = get_model_dir(project)
        self.models: dict[str, Any] = {}
        self.metadata: dict[str, dict[str, Any]] = {}

    def save_model(
        self,
        model: Any,
        name: str,
        preprocessor: DataPreprocessor | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save model with metadata."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{name}_{timestamp}.joblib"
        model_path = f"{self.model_dir}/{model_filename}"

        # Prepare save data
        save_data = {
            "model": model,
            "preprocessor": preprocessor,
            "metrics": metrics or {},
            "metadata": {
                "name": name,
                "timestamp": timestamp,
                "project": self.project,
                **(metadata or {}),
            },
        }

        # Save model
        joblib.dump(save_data, model_path)

        # Store in memory
        self.models[name] = model
        self.metadata[name] = save_data["metadata"]

        self.logger.info(f"Model {name} saved to {model_path}")

        return model_path

    def load_model(self, name: str) -> dict[str, Any]:
        """Load model from disk."""

        # Find latest model file
        import glob

        pattern = f"{self.model_dir}/{name}_*.joblib"
        model_files = glob.glob(pattern)

        if not model_files:
            raise FileNotFoundError(f"No model files found for {name}")

        # Load latest model
        latest_file = max(model_files)
        save_data = joblib.load(latest_file)

        # Store in memory
        self.models[name] = save_data["model"]
        self.metadata[name] = save_data["metadata"]

        self.logger.info(f"Model {name} loaded from {latest_file}")

        return save_data

    def list_models(self) -> list[str]:
        """List available models."""

        import glob

        pattern = f"{self.model_dir}/*.joblib"
        model_files = glob.glob(pattern)

        models = []
        for file in model_files:
            try:
                save_data = joblib.load(file)
                models.append(save_data["metadata"]["name"])
            except Exception as e:
                self.logger.warning(f"Could not load metadata from {file}: {e}")

        return list(set(models))

    def get_model_info(self, name: str) -> dict[str, Any]:
        """Get model information."""

        if name not in self.metadata:
            self.load_model(name)

        return self.metadata[name]


class FeatureSelector(LoggerMixin):
    """Feature selection utilities."""

    def __init__(self, method: str = "mutual_info"):
        self.method = method
        self.selected_features = None
        self.feature_scores = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> "FeatureSelector":
        """Fit feature selector."""

        self.logger.info(f"Fitting feature selector with method: {self.method}")

        if self.method == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif

            scores = mutual_info_classif(X, y, random_state=42)
        elif self.method == "f_score":
            from sklearn.feature_selection import f_classif

            scores, _ = f_classif(X, y)
        elif self.method == "chi2":
            from sklearn.feature_selection import chi2

            scores, _ = chi2(X, y)
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")

        # Get top k features
        feature_names = X.columns
        feature_scores = dict(zip(feature_names, scores, strict=False))

        # Sort by score
        sorted_features = sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True
        )

        self.selected_features = [f[0] for f in sorted_features[:k]]
        self.feature_scores = dict(sorted_features)
        self.is_fitted = True

        self.logger.info(f"Selected {len(self.selected_features)} features")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using selected features."""

        if not self.is_fitted:
            raise ValueError("Feature selector must be fitted before transform")

        return X[self.selected_features]

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""

        if not self.is_fitted:
            raise ValueError("Feature selector must be fitted")

        return self.feature_scores.copy()


def calculate_data_hash(data: pd.DataFrame | np.ndarray) -> str:
    """Calculate hash of dataset for reproducibility."""

    if isinstance(data, pd.DataFrame):
        # Convert to string representation
        data_str = data.to_string()
    else:
        data_str = str(data)

    return hashlib.md5(data_str.encode()).hexdigest()


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float | None = None,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame | None,
    pd.Series | None,
]:
    """Split data into train/validation/test sets."""

    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val (if requested)
    X_train, X_val, y_train, y_val = None, None, None, None
    if val_size is not None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
            stratify=y_train_val,
        )
    else:
        X_train, y_train = X_train_val, y_train_val

    return X_train, X_val, y_train, y_val, X_test, y_test


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "model"
) -> dict[str, Any]:
    """Evaluate model performance."""

    # Make predictions
    y_pred = model.predict(X_test)

    # Get probabilities if available
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    from .metrics import record_model_metrics

    metrics = record_model_metrics(
        y_true=y_test.values,
        y_pred=y_pred,
        y_proba=y_proba,
        model_name=model_name,
        data_hash=calculate_data_hash(X_test),
    )

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


class MLUtils:
    """Main ML utilities class."""

    def __init__(self, project: str):
        self.project = project
        self.preprocessor = DataPreprocessor()
        self.model_manager = ModelManager(project)
        self.feature_selector = FeatureSelector()

    def prepare_data(
        self, X: pd.DataFrame, y: pd.Series | None = None, fit: bool = True
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Prepare data for training or inference."""

        if fit:
            return self.preprocessor.fit_transform(X, y)
        else:
            return self.preprocessor.transform(X), y

    def select_features(
        self, X: pd.DataFrame, y: pd.Series, k: int = 20, fit: bool = True
    ) -> pd.DataFrame:
        """Select top features."""

        if fit:
            self.feature_selector.fit(X, y, k)

        return self.feature_selector.transform(X)

    def save_model(
        self,
        model: Any,
        name: str,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save model with all components."""

        return self.model_manager.save_model(
            model=model,
            name=name,
            preprocessor=self.preprocessor,
            metrics=metrics,
            metadata=metadata,
        )

    def load_model(self, name: str) -> dict[str, Any]:
        """Load model with all components."""

        save_data = self.model_manager.load_model(name)

        # Update preprocessor
        if save_data.get("preprocessor"):
            self.preprocessor = save_data["preprocessor"]

        return save_data
