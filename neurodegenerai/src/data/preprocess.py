"""
Data preprocessing utilities for NeuroDegenerAI.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from shared.lib.logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class TabularPreprocessor(LoggerMixin):
    """Preprocessing utilities for tabular ADNI data."""

    def __init__(self, scaler_type: str = "standard", feature_selection_k: int = 20):
        self.scaler_type = scaler_type
        self.feature_selection_k = feature_selection_k
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.label_encoder = None
        self.feature_names = None
        self.is_fitted = False

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Fit preprocessor and transform data."""

        self.logger.info(f"Fitting tabular preprocessor with {len(X)} samples")

        # Store original feature names
        self.feature_names = X.columns.tolist()

        # Step 1: Handle missing values
        X_clean = self._handle_missing_values(X.copy())

        # Step 2: Feature engineering
        X_engineered = self._engineer_features(X_clean)

        # Step 3: Scaling
        X_scaled = self._scale_features(X_engineered, fit=True)

        # Step 4: Feature selection (if target available)
        if y is not None:
            X_selected = self._select_features(X_scaled, y, fit=True)
            y_encoded = self._encode_target(y)
        else:
            X_selected = X_scaled
            y_encoded = None

        self.is_fitted = True
        self.logger.info(f"Tabular preprocessing complete: {X_selected.shape}")

        return X_selected, y_encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor."""

        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        self.logger.info(f"Transforming {len(X)} samples")

        # Apply same preprocessing steps
        X_clean = self._handle_missing_values(X.copy())
        X_engineered = self._engineer_features(X_clean)
        X_scaled = self._scale_features(X_engineered, fit=False)

        # Handle missing features (add zeros for new features)
        missing_features = set(self.feature_names) - set(X_scaled.columns)
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
            for feature in missing_features:
                X_scaled[feature] = 0

        # Reorder columns to match training
        X_scaled = X_scaled.reindex(columns=self.feature_names, fill_value=0)

        # Apply feature selection if available
        if self.feature_selector is not None:
            X_selected = pd.DataFrame(
                self.feature_selector.transform(X_scaled),
                columns=self.feature_selector.get_feature_names_out(),
                index=X_scaled.index,
            )
        else:
            X_selected = X_scaled

        return X_selected

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""

        # Get numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns

        # Fill numeric columns with median
        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        # Fill categorical columns with mode
        for col in categorical_cols:
            mode_val = X[col].mode()
            if len(mode_val) > 0:
                X[col] = X[col].fillna(mode_val[0])
            else:
                X[col] = X[col].fillna("Unknown")

        return X

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features."""

        X_eng = X.copy()

        # Age-related features
        if "AGE" in X_eng.columns:
            X_eng["AGE_SQUARED"] = X_eng["AGE"] ** 2
            X_eng["AGE_LOG"] = np.log(X_eng["AGE"] + 1)

        # Cognitive score interactions
        if "MMSE" in X_eng.columns and "AGE" in X_eng.columns:
            X_eng["MMSE_AGE_INTERACTION"] = X_eng["MMSE"] * X_eng["AGE"]

        # Biomarker ratios
        if "ABETA" in X_eng.columns and "TAU" in X_eng.columns:
            X_eng["ABETA_TAU_RATIO"] = X_eng["ABETA"] / (X_eng["TAU"] + 1e-6)

        if "PTAU" in X_eng.columns and "ABETA" in X_eng.columns:
            X_eng["PTAU_ABETA_RATIO"] = X_eng["PTAU"] / (X_eng["ABETA"] + 1e-6)

        # Genetic risk interactions
        if "APOE4" in X_eng.columns and "AGE" in X_eng.columns:
            X_eng["APOE4_AGE_RISK"] = X_eng["APOE4"] * X_eng["AGE"]

        # Education level bins
        if "EDUCATION" in X_eng.columns:
            X_eng["EDUCATION_BINS"] = pd.cut(
                X_eng["EDUCATION"],
                bins=[0, 12, 16, 20, 25],
                labels=["High_School", "College", "Graduate", "PhD"],
                include_lowest=True,
            )
            # Convert to dummy variables
            edu_dummies = pd.get_dummies(X_eng["EDUCATION_BINS"], prefix="EDU")
            X_eng = pd.concat([X_eng, edu_dummies], axis=1)
            X_eng = X_eng.drop("EDUCATION_BINS", axis=1)

        # CDR severity categories
        if "CDR" in X_eng.columns:
            X_eng["CDR_SEVERE"] = (X_eng["CDR"] >= 2).astype(int)
            X_eng["CDR_MODERATE"] = ((X_eng["CDR"] >= 1) & (X_eng["CDR"] < 2)).astype(
                int
            )

        self.logger.info(f"Feature engineering complete: {X_eng.shape}")

        return X_eng

    def _scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features."""

        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return X

        if fit:
            if self.scaler_type == "standard":
                self.scaler = StandardScaler()
            elif self.scaler_type == "minmax":
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler type: {self.scaler_type}")

            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        else:
            if self.scaler is None:
                raise ValueError("Scaler must be fitted before transform")
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])

        return X

    def _select_features(
        self, X: pd.DataFrame, y: pd.Series, fit: bool = True
    ) -> pd.DataFrame:
        """Select top features using statistical tests."""

        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return X

        if fit:
            # Use mutual information for feature selection
            self.feature_selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(self.feature_selection_k, len(numeric_cols)),
            )

            X_selected = self.feature_selector.fit_transform(X[numeric_cols], y)

            # Create DataFrame with selected feature names
            selected_features = self.feature_selector.get_feature_names_out()
            X_result = pd.DataFrame(
                X_selected, columns=selected_features, index=X.index
            )

            # Add non-numeric columns back
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                X_result[col] = X[col]

            self.logger.info(
                f"Selected {len(selected_features)} features from {len(numeric_cols)}"
            )

        else:
            if self.feature_selector is None:
                raise ValueError("Feature selector must be fitted before transform")

            X_selected = self.feature_selector.transform(X[numeric_cols])
            selected_features = self.feature_selector.get_feature_names_out()

            X_result = pd.DataFrame(
                X_selected, columns=selected_features, index=X.index
            )

            # Add non-numeric columns back
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                X_result[col] = X[col]

        return X_result

    def _encode_target(self, y: pd.Series) -> pd.Series:
        """Encode target variable."""

        if y.dtype == "object" or y.dtype.name == "category":
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = pd.Series(
                    self.label_encoder.fit_transform(y), index=y.index, name=y.name
                )
            else:
                y_encoded = pd.Series(
                    self.label_encoder.transform(y), index=y.index, name=y.name
                )

            self.logger.info(f"Encoded {len(self.label_encoder.classes_)} classes")
        else:
            y_encoded = y.copy()

        return y_encoded

    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Inverse transform encoded target values."""

        if self.label_encoder is None:
            return y_encoded

        return self.label_encoder.inverse_transform(y_encoded)

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance scores."""

        if self.feature_selector is None:
            return None

        scores = self.feature_selector.scores_
        feature_names = self.feature_selector.get_feature_names_out()

        return dict(zip(feature_names, scores, strict=False))
