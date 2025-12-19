"""
ADNI data ingestion and loading utilities.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.lib.config import get_settings, is_demo_mode
from shared.lib.io_utils import DataLoader
from shared.lib.logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class ADNIDataLoader(LoggerMixin):
    """ADNI data loading and preprocessing."""

    def __init__(self):
        self.settings = get_settings()
        self.data_loader = DataLoader()
        self.data_dir = Path(self.settings.adni_data_dir)

    def load_tabular_data(self) -> pd.DataFrame | None:
        """Load tabular ADNI data."""

        if is_demo_mode():
            self.logger.info("Loading synthetic tabular data (demo mode)")
            return self._load_synthetic_tabular()

        # Try to load real ADNI data
        tabular_path = self.data_dir / "tabular"

        if not tabular_path.exists():
            self.logger.warning(f"Tabular data directory not found: {tabular_path}")
            self.logger.info("Falling back to synthetic data")
            return self._load_synthetic_tabular()

        # Look for CSV files
        csv_files = list(tabular_path.glob("*.csv"))

        if not csv_files:
            self.logger.warning(f"No CSV files found in {tabular_path}")
            self.logger.info("Falling back to synthetic data")
            return self._load_synthetic_tabular()

        # Load the first CSV file
        csv_file = csv_files[0]
        df = pd.read_csv(csv_file)

        # Standardize column names
        df = self._standardize_columns(df)

        # Validate data
        if not self.validate_tabular_data(df):
            self.logger.error("Data validation failed for real ADNI data")
            self.logger.info("Falling back to synthetic data")
            return self._load_synthetic_tabular()

        self.logger.info(
            f"Loaded real ADNI tabular data: {len(df)} rows, {len(df.columns)} columns"
        )

        return df

    def load_mri_data(self) -> list[np.ndarray] | None:
        """Load MRI data."""

        if is_demo_mode():
            self.logger.info("Loading synthetic MRI data (demo mode)")
            return self._load_synthetic_mri()

        # Try to load real MRI data
        mri_path = self.data_dir / "mri"

        if not mri_path.exists():
            self.logger.warning(f"MRI data directory not found: {mri_path}")
            self.logger.info("Falling back to synthetic data")
            return self._load_synthetic_mri()

        # Load MRI samples
        samples = self.data_loader.load_mri_samples(str(mri_path))

        if samples is None:
            self.logger.info("Falling back to synthetic data")
            return self._load_synthetic_mri()

        return samples

    def _load_synthetic_tabular(self) -> pd.DataFrame:
        """Load synthetic tabular data for demo mode."""

        # Generate synthetic data with realistic distributions
        np.random.seed(42)
        n_samples = 1000

        data = {
            "AGE": np.random.normal(70, 10, n_samples).astype(int),
            "SEX": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),  # More females
            "APOE4": np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            "MMSE": np.random.normal(25, 5, n_samples).astype(int),
            "ABETA": np.random.normal(200, 50, n_samples),
            "TAU": np.random.normal(300, 100, n_samples),
            "PTAU": np.random.normal(25, 10, n_samples),
            "EDUCATION": np.random.normal(14, 3, n_samples).astype(int),
            "CDR": np.random.choice([0, 0.5, 1, 2], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        }

        # Clip MMSE to valid range
        data["MMSE"] = np.clip(data["MMSE"], 0, 30)

        # Clip education to reasonable range
        data["EDUCATION"] = np.clip(data["EDUCATION"], 0, 25)

        # Generate diagnosis based on realistic risk factors
        risk_score = (
            (data["AGE"] - 70) * 0.02  # Age risk
            + (data["APOE4"] - 0.5) * 0.5  # Genetic risk
            + (30 - data["MMSE"]) * 0.1  # Cognitive risk
            + (200 - data["ABETA"]) * 0.002  # Biomarker risk
            + (data["TAU"] - 300) * 0.001
            + (data["PTAU"] - 25) * 0.01
        )

        # Add some noise
        risk_score += np.random.normal(0, 0.1, n_samples)

        # Convert to probability and generate diagnosis
        prob_disease = 1 / (1 + np.exp(-risk_score))
        data["DIAGNOSIS"] = np.random.binomial(1, prob_disease, n_samples)

        # Add some additional realistic features
        data["HIPPOCAMPAL_VOLUME"] = np.random.normal(3000, 500, n_samples)
        data["CORTICAL_THICKNESS"] = np.random.normal(2.5, 0.3, n_samples)
        data["WHITE_MATTER_HYPERINTENSITIES"] = np.random.exponential(5, n_samples)

        df = pd.DataFrame(data)

        self.logger.info(f"Generated synthetic tabular data: {len(df)} rows")
        self.logger.info(
            f"Diagnosis distribution: {df['DIAGNOSIS'].value_counts().to_dict()}"
        )

        return df

    def _load_synthetic_mri(self) -> list[np.ndarray]:
        """Load synthetic MRI data for demo mode."""

        # Generate synthetic brain-like MRI volumes
        np.random.seed(42)
        n_samples = 20
        shape = (64, 64, 64)  # Reduced size for demo

        samples = []
        for i in range(n_samples):
            # Create base brain structure
            volume = np.random.normal(0, 0.5, shape)

            # Add brain-like regions
            center = np.array(shape) // 2

            # Gray matter regions
            for z in range(shape[2]):
                y, x = np.ogrid[: shape[0], : shape[1]]
                # Hippocampal region (smaller, more intense)
                hippo_mask = ((x - center[1]) ** 2 + (y - center[0]) ** 2) < (
                    shape[0] // 6
                ) ** 2
                volume[hippo_mask, z] += np.random.normal(1.5, 0.3)

                # Cortical regions
                cortex_mask = ((x - center[1]) ** 2 + (y - center[0]) ** 2) < (
                    shape[0] // 3
                ) ** 2
                cortex_mask = cortex_mask & ~hippo_mask
                volume[cortex_mask, z] += np.random.normal(0.8, 0.2)

            # Add some pathology (for disease cases)
            if i % 3 == 0:  # 1/3 have some pathology
                # Atrophy simulation
                atrophy_mask = np.random.random(shape) < 0.1
                volume[atrophy_mask] -= np.random.normal(0.5, 0.1, atrophy_mask.sum())

            samples.append(volume.astype(np.float32))

        self.logger.info(
            f"Generated {len(samples)} synthetic MRI samples with shape {shape}"
        )

        return samples

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to expected format."""

        # Mapping of common ADNI column names to standard names
        column_mapping = {
            "Age": "AGE",
            "age": "AGE",
            "PTAGE": "AGE",
            "PTGENDER": "SEX",
            "PTETHNIC": "ETHNICITY",
            "PTRACCAT": "RACE",
            "PTEDUCAT": "EDUCATION",
            "APOE4": "APOE4",
            "MMSE": "MMSE",
            "CDR": "CDR",
            "DX": "DIAGNOSIS",
            "DXCHANGE": "DIAGNOSIS",
            "ABETA": "ABETA",
            "ABETA_UPENNBIOMK": "ABETA",
            "TAU": "TAU",
            "TAU_UPENNBIOMK": "TAU",
            "PTAU": "PTAU",
            "PTAU_UPENNBIOMK": "PTAU",
            "HIPPOCAMPAL_VOLUME": "HIPPOCAMPAL_VOLUME",
            "CORTICAL_THICKNESS": "CORTICAL_THICKNESS",
            "WHITE_MATTER_HYPERINTENSITIES": "WHITE_MATTER_HYPERINTENSITIES",
        }

        # Rename columns
        df = df.rename(columns=column_mapping)

        # Ensure required columns exist
        required_columns = ["AGE", "SEX", "APOE4", "MMSE", "DIAGNOSIS"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            self.logger.warning(f"Missing required columns: {missing_columns}")

        return df

    def validate_tabular_data(self, df: pd.DataFrame) -> bool:
        """Validate tabular data against medical research schemas."""
        try:
            # Check for data types
            if not pd.api.types.is_numeric_dtype(df["AGE"]):
                return False
            if not pd.api.types.is_numeric_dtype(df["MMSE"]):
                return False

            # Check for value ranges (Medical sanity check)
            if (df["AGE"] < 0).any() or (df["AGE"] > 120).any():
                self.logger.error("Age out of medical bounds")
                return False
            if (df["MMSE"] < 0).any() or (df["MMSE"] > 30).any():
                self.logger.error("MMSE out of bounds (0-30)")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False

    def get_data_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Get summary statistics of the loaded data."""

        summary = {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "diagnosis_distribution": (
                df["DIAGNOSIS"].value_counts().to_dict()
                if "DIAGNOSIS" in df.columns
                else {}
            ),
            "numeric_summary": (
                df.describe().to_dict()
                if len(df.select_dtypes(include=[np.number]).columns) > 0
                else {}
            ),
        }

        return summary
