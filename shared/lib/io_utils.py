"""
Input/Output utilities for data handling.
"""

import json
import pickle
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import get_settings
from .logging import LoggerMixin, get_logger


class PIIScrubber:
    """Utility to scrub Personally Identifiable Information from text."""

    def __init__(self):
        # Basic patterns for demonstration
        self.patterns = {
            "email": r"[\w\.-]+@[\w\.-]+\.\w+",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "date_of_birth": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            "social_security": r"\b\d{3}-\d{2}-\d{4}\b",
        }

    def scrub(self, text: str) -> str:
        """Replace PII with placeholders."""
        if not text:
            return text

        scrubbed = text
        for pii_type, pattern in self.patterns.items():
            scrubbed = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", scrubbed)
        return scrubbed


logger = get_logger(__name__)


class FileHandler(LoggerMixin):
    """File handling utilities."""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_json(self, data: dict[str, Any], filename: str) -> str:
        """Save data as JSON file."""

        filepath = self.base_path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info(f"Saved JSON data to {filepath}")
        return str(filepath)

    def load_json(self, filename: str) -> dict[str, Any]:
        """Load data from JSON file."""

        filepath = self.base_path / filename

        with open(filepath) as f:
            data = json.load(f)

        self.logger.info(f"Loaded JSON data from {filepath}")
        return data

    def save_pickle(self, data: Any, filename: str) -> str:
        """Save data as pickle file."""

        filepath = self.base_path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        self.logger.info(f"Saved pickle data to {filepath}")
        return str(filepath)

    def load_pickle(self, filename: str) -> Any:
        """Load data from pickle file."""

        filepath = self.base_path / filename

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.logger.info(f"Loaded pickle data from {filepath}")
        return data

    def save_csv(self, df: pd.DataFrame, filename: str, **kwargs: Any) -> str:
        """Save DataFrame as CSV file."""

        filepath = self.base_path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(filepath, **kwargs)

        self.logger.info(f"Saved CSV data to {filepath}")
        return str(filepath)

    def load_csv(self, filename: str, **kwargs: Any) -> pd.DataFrame:
        """Load CSV file as DataFrame."""

        filepath = self.base_path / filename

        df = pd.read_csv(filepath, **kwargs)

        self.logger.info(f"Loaded CSV data from {filepath}")
        return df

    def file_exists(self, filename: str) -> bool:
        """Check if file exists."""

        filepath = self.base_path / filename
        return filepath.exists()

    def list_files(self, pattern: str = "*") -> list[str]:
        """List files matching pattern."""

        files = list(self.base_path.glob(pattern))
        return [str(f.relative_to(self.base_path)) for f in files]


class DatabaseHandler(LoggerMixin):
    """Database handling utilities."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._setup_database()

    def _setup_database(self) -> None:
        """Setup database connection and tables."""

        with self.get_connection() as conn:
            # Create posts table for trend detector
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    url TEXT,
                    author TEXT,
                    score INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create clusters table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id INTEGER NOT NULL,
                    topic TEXT,
                    representative_terms TEXT,
                    volume INTEGER DEFAULT 0,
                    trend_score REAL DEFAULT 0.0,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create embeddings table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (post_id) REFERENCES posts (id)
                )
            """
            )

            conn.commit()
            self.logger.info("Database tables created successfully")

    @contextmanager
    def get_connection(self):
        """Get database connection."""

        conn = sqlite3.connect(self.db_url.replace("sqlite:///", ""))
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()

    def insert_post(self, post_data: dict[str, Any]) -> int:
        """Insert post data."""

        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO posts (text, source, timestamp, url, author, score)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    post_data["text"],
                    post_data["source"],
                    post_data["timestamp"],
                    post_data.get("url"),
                    post_data.get("author"),
                    post_data.get("score", 0),
                ),
            )

            conn.commit()
            return cursor.lastrowid

    def insert_cluster(self, cluster_data: dict[str, Any]) -> int:
        """Insert cluster data."""

        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO clusters (cluster_id, topic, representative_terms, volume, trend_score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    cluster_data["cluster_id"],
                    cluster_data["topic"],
                    cluster_data["representative_terms"],
                    cluster_data["volume"],
                    cluster_data["trend_score"],
                    cluster_data["timestamp"],
                ),
            )

            conn.commit()
            return cursor.lastrowid

    def insert_embedding(
        self, post_id: int, embedding: np.ndarray, model_name: str
    ) -> int:
        """Insert embedding data."""

        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO embeddings (post_id, embedding, model_name)
                VALUES (?, ?, ?)
            """,
                (post_id, pickle.dumps(embedding), model_name),
            )

            conn.commit()
            return cursor.lastrowid

    def get_latest_posts(
        self, limit: int = 1000, source: str | None = None
    ) -> list[dict[str, Any]]:
        """Get latest posts."""

        with self.get_connection() as conn:
            if source:
                cursor = conn.execute(
                    """
                    SELECT * FROM posts
                    WHERE source = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (source, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM posts
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (limit,),
                )

            return [dict(row) for row in cursor.fetchall()]

    def get_latest_clusters(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get latest clusters."""

        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM clusters
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def search_posts(self, query: str, limit: int = 100) -> list[dict[str, Any]]:
        """Search posts by text content."""

        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM posts
                WHERE text LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (f"%{query}%", limit),
            )

            return [dict(row) for row in cursor.fetchall()]


class DataLoader(LoggerMixin):
    """Data loading utilities for different formats."""

    def __init__(self):
        self.settings = get_settings()

    def load_adni_data(self, data_dir: str) -> pd.DataFrame | None:
        """Load ADNI data if available."""

        data_path = Path(data_dir)

        if not data_path.exists():
            self.logger.warning(f"ADNI data directory not found: {data_path}")
            return None

        # Look for CSV files
        csv_files = list(data_path.glob("*.csv"))

        if not csv_files:
            self.logger.warning(f"No CSV files found in {data_path}")
            return None

        # Load the first CSV file found
        csv_file = csv_files[0]
        df = pd.read_csv(csv_file)

        self.logger.info(
            f"Loaded ADNI data from {csv_file}: {len(df)} rows, {len(df.columns)} columns"
        )

        return df

    def generate_synthetic_data(
        self, n_samples: int = 1000, n_features: int = 10
    ) -> pd.DataFrame:
        """Generate synthetic data for demo mode."""

        np.random.seed(42)

        # Generate features with realistic distributions
        data = {}

        # Demographics
        data["AGE"] = np.random.normal(70, 10, n_samples).astype(int)
        data["SEX"] = np.random.choice([0, 1], n_samples)  # 0=Female, 1=Male

        # APOE4 status (genetic risk factor)
        data["APOE4"] = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])

        # Cognitive scores
        data["MMSE"] = np.random.normal(25, 5, n_samples).astype(int)
        data["MMSE"] = np.clip(data["MMSE"], 0, 30)

        # Biomarkers
        data["ABETA"] = np.random.normal(200, 50, n_samples)
        data["TAU"] = np.random.normal(300, 100, n_samples)
        data["PTAU"] = np.random.normal(25, 10, n_samples)

        # Additional features
        for i in range(n_features - 7):
            data[f"FEATURE_{i+1}"] = np.random.normal(0, 1, n_samples)

        # Generate diagnosis based on features (simplified model)
        risk_score = (
            (data["AGE"] - 70) * 0.1
            + (data["APOE4"] - 0.5) * 2
            + (30 - data["MMSE"]) * 0.2
            + (200 - data["ABETA"]) * 0.01
            + (data["TAU"] - 300) * 0.005
            + (data["PTAU"] - 25) * 0.05
        )

        prob_disease = 1 / (1 + np.exp(-risk_score))
        data["DIAGNOSIS"] = np.random.binomial(1, prob_disease, n_samples)

        df = pd.DataFrame(data)

        self.logger.info(
            f"Generated synthetic data: {len(df)} rows, {len(df.columns)} columns"
        )
        self.logger.info(
            f"Diagnosis distribution: {df['DIAGNOSIS'].value_counts().to_dict()}"
        )

        return df

    def load_mri_samples(self, data_dir: str) -> list[np.ndarray] | None:
        """Load MRI samples if available."""

        data_path = Path(data_dir)

        if not data_path.exists():
            self.logger.warning(f"MRI data directory not found: {data_path}")
            return None

        # Look for NIfTI files
        nifti_files = list(data_path.glob("*.nii*"))

        if not nifti_files:
            self.logger.warning(f"No NIfTI files found in {data_path}")
            return None

        samples = []
        try:
            import nibabel as nib

            for nifti_file in nifti_files[:5]:  # Limit to 5 samples for demo
                img = nib.load(nifti_file)
                data = img.get_fdata()
                samples.append(data)

            self.logger.info(f"Loaded {len(samples)} MRI samples")

        except ImportError:
            self.logger.warning("nibabel not available, cannot load MRI data")
            return None

        return samples

    def generate_synthetic_mri(
        self, n_samples: int = 10, shape: tuple = (64, 64, 64)
    ) -> list[np.ndarray]:
        """Generate synthetic MRI data for demo mode."""

        np.random.seed(42)

        samples = []
        for _i in range(n_samples):
            # Generate synthetic brain-like structure
            sample = np.random.normal(0, 1, shape)

            # Add some structure (simulate brain regions)
            center = np.array(shape) // 2
            for j in range(shape[2]):
                # Add circular regions
                y, x = np.ogrid[: shape[0], : shape[1]]
                mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 < (
                    shape[0] // 4
                ) ** 2
                sample[mask, j] += np.random.normal(2, 0.5)

            samples.append(sample)

        self.logger.info(
            f"Generated {len(samples)} synthetic MRI samples with shape {shape}"
        )

        return samples


class IOUtils:
    """Main I/O utilities class."""

    def __init__(self, project: str = "shared"):
        self.project = project
        self.file_handler = FileHandler()
        self.data_loader = DataLoader()

        # Initialize database handler for trend detector
        if project == "trends":
            settings = get_settings()
            self.db_handler = DatabaseHandler(settings.db_url)
        else:
            self.db_handler = None

    def save_data(self, data: Any, filename: str, format: str = "json") -> str:
        """Save data in specified format."""

        if format == "json":
            return self.file_handler.save_json(data, filename)
        elif format == "pickle":
            return self.file_handler.save_pickle(data, filename)
        elif format == "csv" and isinstance(data, pd.DataFrame):
            return self.file_handler.save_csv(data, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_data(self, filename: str, format: str = "json") -> Any:
        """Load data from specified format."""

        if format == "json":
            return self.file_handler.load_json(filename)
        elif format == "pickle":
            return self.file_handler.load_pickle(filename)
        elif format == "csv":
            return self.file_handler.load_csv(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_database_handler(self) -> DatabaseHandler | None:
        """Get database handler if available."""

        return self.db_handler
