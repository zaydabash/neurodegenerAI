"""
Training script for tabular models (LightGBM, XGBoost).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

from shared.lib.config import get_model_dir, get_settings
from shared.lib.logging import LoggerMixin, get_logger
from shared.lib.metrics import PerformanceTimer, record_model_metrics
from shared.lib.ml_utils import MLUtils, split_data

from ..data.adni_ingest import ADNIDataLoader
from ..data.features_tabular import TabularFeatureEngineer
from ..data.preprocess import TabularPreprocessor

logger = get_logger(__name__)


class TabularModelTrainer(LoggerMixin):
    """Trainer for tabular models."""

    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type
        self.settings = get_settings()
        self.model_dir = Path(get_model_dir("neuro"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_loader = ADNIDataLoader()
        self.preprocessor = TabularPreprocessor()
        self.feature_engineer = TabularFeatureEngineer()
        self.ml_utils = MLUtils("neuro")

        # Model and components
        self.model = None
        self.calibrated_model = None
        self.feature_importance = None
        self.training_metrics = None

    def train(
        self, test_size: float = 0.2, val_size: float | None = None
    ) -> dict[str, Any]:
        """Train tabular model."""

        self.logger.info(f"Starting {self.model_type} training")

        with PerformanceTimer("tabular_training"):
            # Load data
            df = self.data_loader.load_tabular_data()
            if df is None:
                raise ValueError("Could not load tabular data")

            # Prepare features and target
            X = df.drop("DIAGNOSIS", axis=1)
            y = df["DIAGNOSIS"]

            # Feature engineering
            self.logger.info("Engineering features")
            X_engineered = self.feature_engineer.engineer_features(X, y)

            # Split data
            X_train, X_val, y_train, y_val, X_test, y_test = split_data(
                X_engineered, y, test_size=test_size, val_size=val_size
            )

            # Preprocessing
            self.logger.info("Preprocessing data")
            X_train_processed, y_train_processed = self.preprocessor.fit_transform(
                X_train, y_train
            )
            X_val_processed = self.preprocessor.transform(X_val)
            X_test_processed = self.preprocessor.transform(X_test)

            # Train model
            self.logger.info(f"Training {self.model_type} model")
            self.model = self._train_model(
                X_train_processed, y_train_processed, X_val_processed, y_val
            )

            # Calibrate model
            self.logger.info("Calibrating model")
            self.calibrated_model = self._calibrate_model(
                X_train_processed, y_train_processed
            )

            # Evaluate model
            self.logger.info("Evaluating model")
            metrics = self._evaluate_model(X_test_processed, y_test)

            # Save model
            self.logger.info("Saving model")
            model_path = self._save_model()

            # Generate reports
            self._generate_reports(X_test_processed, y_test, metrics)

        self.logger.info("Training completed successfully")

        return {
            "model_path": model_path,
            "metrics": metrics,
            "feature_importance": self.feature_importance,
            "training_metrics": self.training_metrics,
        }

    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Any:
        """Train the specified model."""

        if self.model_type == "lightgbm":
            return self._train_lightgbm(X_train, y_train, X_val, y_val)
        elif self.model_type == "xgboost":
            return self._train_xgboost(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> lgb.LGBMClassifier:
        """Train LightGBM model."""

        # LightGBM parameters
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
        }

        model = lgb.LGBMClassifier(**params)

        # Train with validation set
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        # Store feature importance
        self.feature_importance = dict(
            zip(X_train.columns, model.feature_importances_, strict=False)
        )

        return model

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> xgb.XGBClassifier:
        """Train XGBoost model."""

        # XGBoost parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
        }

        model = xgb.XGBClassifier(**params)

        # Train with validation set
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Store feature importance
        self.feature_importance = dict(
            zip(X_train.columns, model.feature_importances_, strict=False)
        )

        return model

    def _calibrate_model(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> CalibratedClassifierCV:
        """Calibrate model predictions."""

        # Use isotonic regression for calibration
        calibrated_model = CalibratedClassifierCV(self.model, method="isotonic", cv=3)

        calibrated_model.fit(X_train, y_train)

        return calibrated_model

    def _evaluate_model(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, Any]:
        """Evaluate model performance."""

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Calibrated predictions
        y_proba_calibrated = self.calibrated_model.predict_proba(X_test)[:, 1]

        # Record metrics
        metrics = record_model_metrics(
            y_true=y_test.values,
            y_pred=y_pred,
            y_proba=y_proba,
            model_name=f"{self.model_type}_tabular",
            data_hash=self._calculate_data_hash(X_test),
        )

        # Additional metrics
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        confusion_mat = confusion_matrix(y_test, y_pred)

        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring="roc_auc")

        return {
            "metrics": metrics,
            "classification_report": classification_rep,
            "confusion_matrix": confusion_mat.tolist(),
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "calibrated_probabilities": y_proba_calibrated.tolist(),
        }

    def _save_model(self) -> str:
        """Save trained model and components."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.model_type}_tabular_{timestamp}"

        # Prepare save data
        save_data = {
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "preprocessor": self.preprocessor,
            "feature_importance": self.feature_importance,
            "model_type": self.model_type,
            "timestamp": timestamp,
            "training_metrics": self.training_metrics,
        }

        # Save model
        model_path = self.model_dir / f"{model_name}.joblib"
        joblib.dump(save_data, model_path)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "model_type": self.model_type,
            "timestamp": timestamp,
            "feature_importance": self.feature_importance,
            "training_metrics": self.training_metrics,
        }

        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.info(f"Model saved to {model_path}")

        return str(model_path)

    def _generate_reports(
        self, X_test: pd.DataFrame, y_test: pd.Series, metrics: dict[str, Any]
    ) -> None:
        """Generate evaluation reports."""

        reports_dir = Path("./neurodegenerai/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics
        metrics_path = reports_dir / f"tabular_metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Generate plots
        self._generate_plots(X_test, y_test, timestamp)

        self.logger.info(f"Reports generated in {reports_dir}")

    def _generate_plots(
        self, X_test: pd.DataFrame, y_test: pd.Series, timestamp: str
    ) -> None:
        """Generate evaluation plots."""

        from shared.lib.viz import get_viz_helper

        viz_helper = get_viz_helper()
        plots_dir = Path("./neurodegenerai/reports/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Get predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # ROC curve
        viz_helper.plot_roc_curve(
            y_test.values,
            y_proba,
            title=f"ROC Curve - {self.model_type.upper()} Tabular Model",
            save_path=str(plots_dir / f"roc_curve_{timestamp}.png"),
        )

        # Precision-Recall curve
        viz_helper.plot_precision_recall_curve(
            y_test.values,
            y_proba,
            title=f"Precision-Recall Curve - {self.model_type.upper()} Tabular Model",
            save_path=str(plots_dir / f"pr_curve_{timestamp}.png"),
        )

        # Confusion matrix
        viz_helper.plot_confusion_matrix(
            y_test.values,
            y_pred,
            title=f"Confusion Matrix - {self.model_type.upper()} Tabular Model",
            save_path=str(plots_dir / f"confusion_matrix_{timestamp}.png"),
        )

        # Calibration curve
        viz_helper.plot_calibration_curve(
            y_test.values,
            y_proba,
            title=f"Calibration Curve - {self.model_type.upper()} Tabular Model",
            save_path=str(plots_dir / f"calibration_curve_{timestamp}.png"),
        )

        # Feature importance
        if self.feature_importance:
            viz_helper.plot_feature_importance(
                self.feature_importance,
                title=f"Feature Importance - {self.model_type.upper()} Tabular Model",
                save_path=str(plots_dir / f"feature_importance_{timestamp}.png"),
            )

    def _calculate_data_hash(self, X: pd.DataFrame) -> str:
        """Calculate hash of dataset for reproducibility."""
        import hashlib

        data_str = X.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()


def main():
    """Main training function."""

    import argparse

    parser = argparse.ArgumentParser(description="Train tabular model")
    parser.add_argument(
        "--model",
        choices=["lightgbm", "xgboost"],
        default="lightgbm",
        help="Model type",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    parser.add_argument(
        "--val-size", type=float, default=0.1, help="Validation set size"
    )

    args = parser.parse_args()

    # Setup logging
    from shared.lib.logging import setup_logging

    setup_logging(service_name="neurodegenerai_training")

    # Train model
    trainer = TabularModelTrainer(model_type=args.model)
    results = trainer.train(test_size=args.test_size, val_size=args.val_size)

    print("Training completed successfully!")
    print(f"Model saved to: {results['model_path']}")
    print(f"Test accuracy: {results['metrics']['metrics'].accuracy:.4f}")
    print(f"Test AUC: {results['metrics']['metrics'].roc_auc:.4f}")


if __name__ == "__main__":
    main()
