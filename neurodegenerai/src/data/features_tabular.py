"""
Feature engineering utilities for tabular ADNI data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from shared.lib.logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class TabularFeatureEngineer(LoggerMixin):
    """Feature engineering for tabular ADNI data."""

    def __init__(self):
        self.polynomial_features = None
        self.feature_names = None

    def engineer_features(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        """Engineer features from raw ADNI data."""

        self.logger.info(f"Engineering features from {X.shape}")

        X_eng = X.copy()

        # Basic feature engineering
        X_eng = self._add_demographic_features(X_eng)
        X_eng = self._add_biomarker_features(X_eng)
        X_eng = self._add_cognitive_features(X_eng)
        X_eng = self._add_genetic_features(X_eng)
        X_eng = self._add_interaction_features(X_eng)

        # Advanced feature engineering
        X_eng = self._add_statistical_features(X_eng)
        X_eng = self._add_ratio_features(X_eng)
        X_eng = self._add_binning_features(X_eng)

        # Polynomial features (if requested)
        if len(X_eng.columns) < 50:  # Only for smaller feature sets
            X_eng = self._add_polynomial_features(X_eng, degree=2)

        self.logger.info(f"Feature engineering complete: {X_eng.shape}")

        return X_eng

    def _add_demographic_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add demographic-based features."""

        X_eng = X.copy()

        # Age features
        if "AGE" in X_eng.columns:
            X_eng["AGE_NORMALIZED"] = (X_eng["AGE"] - X_eng["AGE"].mean()) / X_eng[
                "AGE"
            ].std()
            X_eng["AGE_BINS"] = pd.cut(
                X_eng["AGE"],
                bins=5,
                labels=["Young", "Middle", "Senior", "Elderly", "Very_Elderly"],
            )
            X_eng["IS_ELDERLY"] = (X_eng["AGE"] >= 65).astype(int)
            X_eng["IS_VERY_ELDERLY"] = (X_eng["AGE"] >= 80).astype(int)

        # Education features
        if "EDUCATION" in X_eng.columns:
            X_eng["EDUCATION_NORMALIZED"] = (
                X_eng["EDUCATION"] - X_eng["EDUCATION"].mean()
            ) / X_eng["EDUCATION"].std()
            X_eng["HIGH_EDUCATION"] = (X_eng["EDUCATION"] >= 16).astype(int)
            X_eng["LOW_EDUCATION"] = (X_eng["EDUCATION"] <= 12).astype(int)

        # Gender features
        if "SEX" in X_eng.columns:
            X_eng["IS_FEMALE"] = (X_eng["SEX"] == 0).astype(int)
            X_eng["IS_MALE"] = (X_eng["SEX"] == 1).astype(int)

        return X_eng

    def _add_biomarker_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add biomarker-based features."""

        X_eng = X.copy()

        # Amyloid beta features
        if "ABETA" in X_eng.columns:
            X_eng["ABETA_NORMALIZED"] = (
                X_eng["ABETA"] - X_eng["ABETA"].mean()
            ) / X_eng["ABETA"].std()
            X_eng["ABETA_HIGH"] = (
                X_eng["ABETA"] > X_eng["ABETA"].quantile(0.75)
            ).astype(int)
            X_eng["ABETA_LOW"] = (
                X_eng["ABETA"] < X_eng["ABETA"].quantile(0.25)
            ).astype(int)
            X_eng["ABETA_LOG"] = np.log(X_eng["ABETA"] + 1)

        # Tau features
        if "TAU" in X_eng.columns:
            X_eng["TAU_NORMALIZED"] = (X_eng["TAU"] - X_eng["TAU"].mean()) / X_eng[
                "TAU"
            ].std()
            X_eng["TAU_HIGH"] = (X_eng["TAU"] > X_eng["TAU"].quantile(0.75)).astype(int)
            X_eng["TAU_LOW"] = (X_eng["TAU"] < X_eng["TAU"].quantile(0.25)).astype(int)
            X_eng["TAU_LOG"] = np.log(X_eng["TAU"] + 1)

        # Phosphorylated tau features
        if "PTAU" in X_eng.columns:
            X_eng["PTAU_NORMALIZED"] = (X_eng["PTAU"] - X_eng["PTAU"].mean()) / X_eng[
                "PTAU"
            ].std()
            X_eng["PTAU_HIGH"] = (X_eng["PTAU"] > X_eng["PTAU"].quantile(0.75)).astype(
                int
            )
            X_eng["PTAU_LOW"] = (X_eng["PTAU"] < X_eng["PTAU"].quantile(0.25)).astype(
                int
            )
            X_eng["PTAU_LOG"] = np.log(X_eng["PTAU"] + 1)

        return X_eng

    def _add_cognitive_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add cognitive assessment features."""

        X_eng = X.copy()

        # MMSE features
        if "MMSE" in X_eng.columns:
            X_eng["MMSE_NORMALIZED"] = (X_eng["MMSE"] - X_eng["MMSE"].mean()) / X_eng[
                "MMSE"
            ].std()
            X_eng["MMSE_IMPAIRED"] = (X_eng["MMSE"] < 24).astype(int)
            X_eng["MMSE_SEVERELY_IMPAIRED"] = (X_eng["MMSE"] < 18).astype(int)
            X_eng["MMSE_QUARTILES"] = pd.qcut(
                X_eng["MMSE"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
            )

        # CDR features
        if "CDR" in X_eng.columns:
            X_eng["CDR_NORMAL"] = (X_eng["CDR"] == 0).astype(int)
            X_eng["CDR_QUESTIONABLE"] = (X_eng["CDR"] == 0.5).astype(int)
            X_eng["CDR_MILD"] = (X_eng["CDR"] == 1).astype(int)
            X_eng["CDR_MODERATE"] = (X_eng["CDR"] == 2).astype(int)
            X_eng["CDR_SEVERE"] = (X_eng["CDR"] >= 3).astype(int)
            X_eng["CDR_ANY_IMPAIRMENT"] = (X_eng["CDR"] > 0).astype(int)

        return X_eng

    def _add_genetic_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add genetic risk features."""

        X_eng = X.copy()

        # APOE4 features
        if "APOE4" in X_eng.columns:
            X_eng["APOE4_POSITIVE"] = (X_eng["APOE4"] == 1).astype(int)
            X_eng["APOE4_NEGATIVE"] = (X_eng["APOE4"] == 0).astype(int)
            # APOE4 homozygosity (if we had that data)
            X_eng["APOE4_HOMOZYGOTE"] = (X_eng["APOE4"] == 2).astype(
                int
            )  # Assuming 2 = homozygous

        return X_eng

    def _add_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features."""

        X_eng = X.copy()

        # Age-genetic interactions
        if "AGE" in X_eng.columns and "APOE4" in X_eng.columns:
            X_eng["AGE_APOE4_INTERACTION"] = X_eng["AGE"] * X_eng["APOE4"]
            X_eng["AGE_APOE4_RISK"] = X_eng["AGE"] * X_eng["APOE4_POSITIVE"]

        # Age-cognitive interactions
        if "AGE" in X_eng.columns and "MMSE" in X_eng.columns:
            X_eng["AGE_MMSE_INTERACTION"] = X_eng["AGE"] * X_eng["MMSE"]
            X_eng["COGNITIVE_RESERVE"] = (
                X_eng["MMSE"] + (X_eng["EDUCATION"] * 0.5)
                if "EDUCATION" in X_eng.columns
                else X_eng["MMSE"]
            )

        # Biomarker interactions
        if "ABETA" in X_eng.columns and "TAU" in X_eng.columns:
            X_eng["ABETA_TAU_INTERACTION"] = X_eng["ABETA"] * X_eng["TAU"]
            X_eng["ABETA_TAU_PRODUCT"] = X_eng["ABETA"] * X_eng["TAU"]

        if "ABETA" in X_eng.columns and "PTAU" in X_eng.columns:
            X_eng["ABETA_PTAU_INTERACTION"] = X_eng["ABETA"] * X_eng["PTAU"]

        # Gender-age interactions
        if "SEX" in X_eng.columns and "AGE" in X_eng.columns:
            X_eng["FEMALE_AGE_RISK"] = X_eng["IS_FEMALE"] * X_eng["AGE"]
            X_eng["MALE_AGE_RISK"] = X_eng["IS_MALE"] * X_eng["AGE"]

        return X_eng

    def _add_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""

        X_eng = X.copy()

        # Create feature groups for statistical operations
        biomarker_cols = [
            col for col in X_eng.columns if col in ["ABETA", "TAU", "PTAU"]
        ]
        cognitive_cols = [col for col in X_eng.columns if col in ["MMSE", "CDR"]]
        [col for col in X_eng.columns if col in ["AGE", "EDUCATION"]]

        # Biomarker statistics
        if len(biomarker_cols) >= 2:
            biomarker_data = X_eng[biomarker_cols]
            X_eng["BIOMARKER_MEAN"] = biomarker_data.mean(axis=1)
            X_eng["BIOMARKER_STD"] = biomarker_data.std(axis=1)
            X_eng["BIOMARKER_RANGE"] = biomarker_data.max(axis=1) - biomarker_data.min(
                axis=1
            )

        # Cognitive statistics
        if len(cognitive_cols) >= 2:
            cognitive_data = X_eng[cognitive_cols]
            X_eng["COGNITIVE_MEAN"] = cognitive_data.mean(axis=1)
            X_eng["COGNITIVE_STD"] = cognitive_data.std(axis=1)

        # Overall health score (composite)
        health_components = []
        if "MMSE" in X_eng.columns:
            health_components.append(X_eng["MMSE"] / 30)  # Normalize MMSE
        if "EDUCATION" in X_eng.columns:
            health_components.append(X_eng["EDUCATION"] / 25)  # Normalize education

        if health_components:
            X_eng["HEALTH_SCORE"] = np.mean(health_components, axis=0)

        return X_eng

    def _add_ratio_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add ratio features."""

        X_eng = X.copy()

        # Biomarker ratios
        if "ABETA" in X_eng.columns and "TAU" in X_eng.columns:
            X_eng["ABETA_TAU_RATIO"] = X_eng["ABETA"] / (X_eng["TAU"] + 1e-6)
            X_eng["TAU_ABETA_RATIO"] = X_eng["TAU"] / (X_eng["ABETA"] + 1e-6)

        if "PTAU" in X_eng.columns and "ABETA" in X_eng.columns:
            X_eng["PTAU_ABETA_RATIO"] = X_eng["PTAU"] / (X_eng["ABETA"] + 1e-6)
            X_eng["ABETA_PTAU_RATIO"] = X_eng["ABETA"] / (X_eng["PTAU"] + 1e-6)

        if "PTAU" in X_eng.columns and "TAU" in X_eng.columns:
            X_eng["PTAU_TAU_RATIO"] = X_eng["PTAU"] / (X_eng["TAU"] + 1e-6)
            X_eng["TAU_PTAU_RATIO"] = X_eng["TAU"] / (X_eng["PTAU"] + 1e-6)

        # Age-adjusted ratios
        if "AGE" in X_eng.columns and "MMSE" in X_eng.columns:
            X_eng["AGE_ADJUSTED_MMSE"] = X_eng["MMSE"] / (X_eng["AGE"] + 1e-6)

        return X_eng

    def _add_binning_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add binning features."""

        X_eng = X.copy()

        # Age bins
        if "AGE" in X_eng.columns:
            X_eng["AGE_DECILES"] = pd.qcut(
                X_eng["AGE"], q=10, labels=[f"Age_Decile_{i+1}" for i in range(10)]
            )

        # MMSE bins
        if "MMSE" in X_eng.columns:
            X_eng["MMSE_SEVERITY"] = pd.cut(
                X_eng["MMSE"],
                bins=[0, 18, 24, 30],
                labels=["Severe", "Mild", "Normal"],
                include_lowest=True,
            )

        # Biomarker quartiles
        for col in ["ABETA", "TAU", "PTAU"]:
            if col in X_eng.columns:
                X_eng[f"{col}_QUARTILES"] = pd.qcut(
                    X_eng[col], q=4, labels=[f"{col}_Q{i+1}" for i in range(4)]
                )

        return X_eng

    def _add_polynomial_features(
        self, X: pd.DataFrame, degree: int = 2
    ) -> pd.DataFrame:
        """Add polynomial features for important variables."""

        X_eng = X.copy()

        # Select important features for polynomial expansion
        important_features = []
        for col in ["AGE", "MMSE", "ABETA", "TAU", "PTAU"]:
            if col in X_eng.columns:
                important_features.append(col)

        if len(important_features) >= 2:
            try:
                poly = PolynomialFeatures(
                    degree=degree, include_bias=False, interaction_only=True
                )
                poly_features = poly.fit_transform(X_eng[important_features])

                # Create feature names
                poly_feature_names = poly.get_feature_names_out(important_features)

                # Add polynomial features
                poly_df = pd.DataFrame(
                    poly_features, columns=poly_feature_names, index=X_eng.index
                )

                # Remove original features from polynomial set to avoid duplication
                poly_df = poly_df.drop(columns=important_features, errors="ignore")

                X_eng = pd.concat([X_eng, poly_df], axis=1)

                self.logger.info(f"Added {len(poly_feature_names)} polynomial features")

            except Exception as e:
                self.logger.warning(f"Could not create polynomial features: {e}")

        return X_eng

    def get_feature_groups(self, X: pd.DataFrame) -> dict[str, list[str]]:
        """Get feature groups for analysis."""

        feature_groups = {
            "demographic": [
                col
                for col in X.columns
                if col in ["AGE", "SEX", "EDUCATION", "IS_ELDERLY", "IS_VERY_ELDERLY"]
            ],
            "genetic": [col for col in X.columns if "APOE4" in col],
            "cognitive": [
                col
                for col in X.columns
                if col in ["MMSE", "CDR"] or "MMSE_" in col or "CDR_" in col
            ],
            "biomarkers": [
                col
                for col in X.columns
                if col in ["ABETA", "TAU", "PTAU"]
                or any(x in col for x in ["ABETA_", "TAU_", "PTAU_"])
            ],
            "interactions": [
                col for col in X.columns if "INTERACTION" in col or "RISK" in col
            ],
            "ratios": [col for col in X.columns if "RATIO" in col],
            "statistical": [
                col
                for col in X.columns
                if any(x in col for x in ["MEAN", "STD", "RANGE", "SCORE"])
            ],
            "bins": [
                col
                for col in X.columns
                if any(x in col for x in ["BINS", "QUARTILES", "DECILES", "SEVERITY"])
            ],
        }

        return feature_groups

    def select_features_by_importance(
        self, X: pd.DataFrame, feature_importance: dict[str, float], top_k: int = 50
    ) -> pd.DataFrame:
        """Select top features by importance."""

        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        # Get top k feature names
        top_features = [f[0] for f in sorted_features[:top_k]]

        # Filter DataFrame
        available_features = [f for f in top_features if f in X.columns]

        self.logger.info(
            f"Selected {len(available_features)} features from {len(X.columns)} total"
        )

        return X[available_features]
