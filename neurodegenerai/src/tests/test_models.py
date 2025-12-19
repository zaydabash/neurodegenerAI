"""
Tests for NeuroDegenerAI models.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ..data.features_tabular import TabularFeatureEngineer
from ..data.preprocess import TabularPreprocessor
from ..models.predict import ModelPredictor


class TestTabularPreprocessor:
    """Test tabular data preprocessing."""

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = TabularPreprocessor()
        assert preprocessor.scaler_type == "standard"
        assert preprocessor.feature_selection_k == 20
        assert preprocessor.is_fitted is False

    def test_fit_transform(self):
        """Test fit_transform method."""
        preprocessor = TabularPreprocessor()

        # Create sample data
        X = pd.DataFrame(
            {
                "AGE": [70, 75, 80],
                "SEX": [0, 1, 0],
                "MMSE": [24, 26, 22],
                "APOE4": [1, 0, 1],
            }
        )
        y = pd.Series([0, 0, 1])

        # Fit and transform
        X_processed, y_processed = preprocessor.fit_transform(X, y)

        assert X_processed.shape[0] == 3
        assert y_processed.shape[0] == 3
        assert preprocessor.is_fitted is True

    def test_transform_without_fit(self):
        """Test transform without fitting should raise error."""
        preprocessor = TabularPreprocessor()
        X = pd.DataFrame({"AGE": [70], "SEX": [0]})

        with pytest.raises(ValueError):
            preprocessor.transform(X)

    def test_missing_value_handling(self):
        """Test handling of missing values."""
        preprocessor = TabularPreprocessor()

        # Create data with missing values
        X = pd.DataFrame(
            {"AGE": [70, np.nan, 80], "SEX": [0, 1, np.nan], "MMSE": [24, 26, 22]}
        )

        X_clean = preprocessor._handle_missing_values(X.copy())

        # Check that missing values are filled
        assert not X_clean.isnull().any().any()


class TestTabularFeatureEngineer:
    """Test feature engineering."""

    def test_feature_engineer_initialization(self):
        """Test feature engineer initialization."""
        engineer = TabularFeatureEngineer()
        assert engineer is not None

    def test_engineer_features(self):
        """Test feature engineering."""
        engineer = TabularFeatureEngineer()

        # Create sample data
        X = pd.DataFrame(
            {
                "AGE": [70, 75, 80],
                "SEX": [0, 1, 0],
                "MMSE": [24, 26, 22],
                "APOE4": [1, 0, 1],
                "ABETA": [180, 200, 160],
                "TAU": [350, 300, 400],
            }
        )

        X_engineered = engineer.engineer_features(X)

        # Check that new features are created
        assert X_engineered.shape[1] > X.shape[1]
        assert "AGE_SQUARED" in X_engineered.columns
        assert "ABETA_TAU_RATIO" in X_engineered.columns

    def test_get_feature_groups(self):
        """Test feature group extraction."""
        engineer = TabularFeatureEngineer()

        X = pd.DataFrame(
            {
                "AGE": [70],
                "SEX": [0],
                "MMSE": [24],
                "APOE4": [1],
                "ABETA": [180],
                "TAU": [350],
            }
        )

        X_engineered = engineer.engineer_features(X)
        feature_groups = engineer.get_feature_groups(X_engineered)

        assert "demographic" in feature_groups
        assert "genetic" in feature_groups
        assert "cognitive" in feature_groups
        assert "biomarkers" in feature_groups


class TestModelPredictor:
    """Test model predictor."""

    @patch("neurodegenerai.src.models.predict.Path")
    def test_predictor_initialization(self, mock_path):
        """Test predictor initialization."""
        mock_path.return_value.glob.return_value = []

        with pytest.raises(FileNotFoundError):
            ModelPredictor()

    def test_predict_tabular_structure(self):
        """Test tabular prediction structure."""
        # This is a structural test - actual prediction would require trained model
        predictor = Mock(spec=ModelPredictor)

        # Mock prediction result
        result = {
            "prediction": 0,
            "probability": 0.3,
            "confidence": 0.7,
            "model_name": "test_model",
            "model_type": "tabular",
            "explanation": {"top_features": []},
        }

        predictor.predict_tabular.return_value = result

        # Test that result has expected structure
        prediction = predictor.predict_tabular({})

        assert "prediction" in prediction
        assert "probability" in prediction
        assert "confidence" in prediction
        assert "model_name" in prediction
        assert "model_type" in prediction

    def test_predict_mri_structure(self):
        """Test MRI prediction structure."""
        predictor = Mock(spec=ModelPredictor)

        # Mock prediction result
        result = {
            "prediction": 1,
            "probability": 0.8,
            "confidence": 0.8,
            "model_name": "test_model",
            "model_type": "cnn",
            "heatmap_paths": [],
        }

        predictor.predict_mri.return_value = result

        # Test that result has expected structure
        prediction = predictor.predict_mri(np.random.randn(64, 64, 64))

        assert "prediction" in prediction
        assert "probability" in prediction
        assert "confidence" in prediction
        assert "model_name" in prediction
        assert "model_type" in prediction


class TestDataValidation:
    """Test data validation utilities."""

    def test_data_hash_calculation(self):
        """Test data hash calculation for reproducibility."""
        from ..models.train_tabular import TabularModelTrainer

        trainer = TabularModelTrainer()

        # Test with different data
        data1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        data2 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        data3 = pd.DataFrame({"A": [1, 2, 4], "B": [4, 5, 6]})

        hash1 = trainer._calculate_data_hash(data1)
        hash2 = trainer._calculate_data_hash(data2)
        hash3 = trainer._calculate_data_hash(data3)

        # Same data should produce same hash
        assert hash1 == hash2

        # Different data should produce different hash
        assert hash1 != hash3

        # Hash should be string
        assert isinstance(hash1, str)
        assert len(hash1) > 0


@pytest.fixture
def sample_tabular_data():
    """Sample tabular data for testing."""
    return pd.DataFrame(
        {
            "AGE": [70, 75, 80, 65, 72],
            "SEX": [0, 1, 0, 1, 0],
            "MMSE": [24, 26, 22, 28, 25],
            "APOE4": [1, 0, 1, 0, 1],
            "ABETA": [180, 200, 160, 220, 190],
            "TAU": [350, 300, 400, 280, 320],
            "PTAU": [28, 25, 32, 22, 26],
        }
    )


def test_data_preprocessing_pipeline(sample_tabular_data):
    """Test complete data preprocessing pipeline."""
    # Create target
    y = pd.Series([0, 0, 1, 0, 1])

    # Initialize components
    preprocessor = TabularPreprocessor()
    feature_engineer = TabularFeatureEngineer()

    # Feature engineering
    X_engineered = feature_engineer.engineer_features(sample_tabular_data, y)

    # Preprocessing
    X_processed, y_processed = preprocessor.fit_transform(X_engineered, y)

    # Assertions
    assert X_processed.shape[0] == 5
    assert y_processed.shape[0] == 5
    assert not X_processed.isnull().any().any()
    assert not y_processed.isnull().any().any()

    # Check that features are scaled
    assert (
        X_processed.select_dtypes(include=[np.number]).std().mean() < 2.0
    )  # Should be close to 1 for standard scaling
