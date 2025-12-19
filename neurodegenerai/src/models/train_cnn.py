"""
Training script for CNN models on MRI data.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from shared.lib.config import get_model_dir, get_settings
from shared.lib.logging import LoggerMixin, get_logger
from shared.lib.metrics import PerformanceTimer, record_model_metrics

from ..data.adni_ingest import ADNIDataLoader
from ..data.features_mri import MRIDataLoader, MRIFeatureExtractor

logger = get_logger(__name__)


class SimpleCNN(nn.Module):
    """Simple CNN for MRI slice classification."""

    def __init__(self, num_classes: int = 2, input_channels: int = 1):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet18Adapted(nn.Module):
    """Adapted ResNet18 for MRI classification."""

    def __init__(self, num_classes: int = 2, input_channels: int = 1):
        super().__init__()

        # Import torchvision ResNet
        import torchvision.models as models

        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=True)

        # Modify first layer for single channel input
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify final layer for our number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class CNNModelTrainer(LoggerMixin):
    """Trainer for CNN models on MRI data."""

    def __init__(self, model_type: str = "simple_cnn"):
        self.model_type = model_type
        self.settings = get_settings()
        self.model_dir = Path(get_model_dir("neuro"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_loader = ADNIDataLoader()
        self.mri_loader = MRIDataLoader()
        self.feature_extractor = MRIFeatureExtractor()

        # Model and training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        self.logger.info(f"Using device: {self.device}")

    def train(
        self, epochs: int = 50, batch_size: int = 16, learning_rate: float = 0.001
    ) -> dict[str, Any]:
        """Train CNN model."""

        self.logger.info(f"Starting {self.model_type} training")

        with PerformanceTimer("cnn_training"):
            # Load data
            volumes = self.data_loader.load_mri_data()
            if volumes is None:
                raise ValueError("Could not load MRI data")

            # Prepare dataset
            X, y = self._prepare_dataset(volumes)

            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)

            # Create data loaders
            train_loader = self._create_data_loader(
                X_train, y_train, batch_size, shuffle=True
            )
            val_loader = self._create_data_loader(
                X_val, y_val, batch_size, shuffle=False
            )
            test_loader = self._create_data_loader(
                X_test, y_test, batch_size, shuffle=False
            )

            # Initialize model
            self._initialize_model()

            # Train model
            self.logger.info(f"Training for {epochs} epochs")
            self._train_epochs(train_loader, val_loader, epochs)

            # Evaluate model
            self.logger.info("Evaluating model")
            metrics = self._evaluate_model(test_loader)

            # Save model
            self.logger.info("Saving model")
            model_path = self._save_model()

            # Generate reports
            self._generate_reports(test_loader, metrics)

        self.logger.info("Training completed successfully")

        return {
            "model_path": model_path,
            "metrics": metrics,
            "training_history": {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "train_accuracies": self.train_accuracies,
                "val_accuracies": self.val_accuracies,
            },
        }

    def _prepare_dataset(
        self, volumes: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare dataset from MRI volumes."""

        all_slices = []
        all_labels = []

        for i, volume in enumerate(volumes):
            # Extract slices
            slices = self.mri_loader.extract_slices(volume)

            # Create labels (simplified: some volumes have pathology)
            label = 1 if i % 3 == 0 else 0  # 1/3 have pathology

            for slice_data in slices:
                all_slices.append(slice_data)
                all_labels.append(label)

        # Convert to numpy arrays
        X = np.array(all_slices)
        y = np.array(all_labels)

        # Add channel dimension
        X = X[:, np.newaxis, :, :]  # Shape: (n_samples, 1, height, width)

        self.logger.info(f"Prepared dataset: {X.shape}, labels: {y.shape}")

        return X, y

    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> tuple:
        """Split data into train/val/test sets."""

        from sklearn.model_selection import train_test_split

        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size / (1 - test_size),
            random_state=42,
            stratify=y_train_val,
        )

        self.logger.info(
            f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _create_data_loader(
        self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True
    ) -> DataLoader:
        """Create PyTorch data loader."""

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)

        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for compatibility
        )

        return loader

    def _initialize_model(self) -> None:
        """Initialize model, optimizer, and criterion."""

        # Initialize model
        if self.model_type == "simple_cnn":
            self.model = SimpleCNN(num_classes=2, input_channels=1)
        elif self.model_type == "resnet18":
            self.model = ResNet18Adapted(num_classes=2, input_channels=1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model = self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=1e-4
        )

        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

        # Initialize criterion
        self.criterion = nn.CrossEntropyLoss()

        self.logger.info(f"Initialized {self.model_type} model")

    def _train_epochs(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int
    ) -> None:
        """Train model for specified epochs."""

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)

            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # Log progress
            if epoch % 5 == 0:
                self.logger.info(
                    f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

    def _train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Train for one epoch."""

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for _batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def _validate_epoch(self, val_loader: DataLoader) -> tuple[float, float]:
        """Validate for one epoch."""

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def _evaluate_model(self, test_loader: DataLoader) -> dict[str, Any]:
        """Evaluate model on test set."""

        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)

                _, predicted = torch.max(output, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(
                    probabilities.cpu().numpy()[:, 1]
                )  # Probability of class 1
                all_targets.extend(target.cpu().numpy())

        # Convert to numpy arrays
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        y_true = np.array(all_targets)

        # Record metrics
        metrics = record_model_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            model_name=f"{self.model_type}_cnn",
            data_hash=self._calculate_data_hash(y_true),
        )

        return {
            "metrics": metrics,
            "predictions": y_pred.tolist(),
            "probabilities": y_proba.tolist(),
            "targets": y_true.tolist(),
        }

    def _save_model(self) -> str:
        """Save trained model."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.model_type}_cnn_{timestamp}"

        # Save model state dict
        model_path = self.model_dir / f"{model_name}.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_type": self.model_type,
                "timestamp": timestamp,
                "training_history": {
                    "train_losses": self.train_losses,
                    "val_losses": self.val_losses,
                    "train_accuracies": self.train_accuracies,
                    "val_accuracies": self.val_accuracies,
                },
            },
            model_path,
        )

        # Save metadata
        metadata = {
            "model_name": model_name,
            "model_type": self.model_type,
            "timestamp": timestamp,
            "training_history": {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "train_accuracies": self.train_accuracies,
                "val_accuracies": self.val_accuracies,
            },
        }

        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.info(f"Model saved to {model_path}")

        return str(model_path)

    def _generate_reports(
        self, test_loader: DataLoader, metrics: dict[str, Any]
    ) -> None:
        """Generate evaluation reports."""

        reports_dir = Path("./neurodegenerai/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics
        metrics_path = reports_dir / f"cnn_metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Generate plots
        self._generate_training_plots(timestamp)

        self.logger.info(f"Reports generated in {reports_dir}")

    def _generate_training_plots(self, timestamp: str) -> None:
        """Generate training plots."""

        plots_dir = Path("./neurodegenerai/reports/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        ax1.plot(self.train_losses, label="Training Loss")
        ax1.plot(self.val_losses, label="Validation Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(self.train_accuracies, label="Training Accuracy")
        ax2.plot(self.val_accuracies, label="Validation Accuracy")
        ax2.set_title("Training and Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(
            plots_dir / f"training_curves_{timestamp}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _calculate_data_hash(self, y: np.ndarray) -> str:
        """Calculate hash of dataset for reproducibility."""
        import hashlib

        data_str = str(y.tolist())
        return hashlib.md5(data_str.encode()).hexdigest()


def main():
    """Main training function."""

    import argparse

    parser = argparse.ArgumentParser(description="Train CNN model")
    parser.add_argument(
        "--model",
        choices=["simple_cnn", "resnet18"],
        default="simple_cnn",
        help="Model type",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )

    args = parser.parse_args()

    # Setup logging
    from shared.lib.logging import setup_logging

    setup_logging(service_name="neurodegenerai_cnn_training")

    # Train model
    trainer = CNNModelTrainer(model_type=args.model)
    results = trainer.train(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate
    )

    print("Training completed successfully!")
    print(f"Model saved to: {results['model_path']}")
    print(f"Test accuracy: {results['metrics']['metrics'].accuracy:.4f}")
    print(f"Test AUC: {results['metrics']['metrics'].roc_auc:.4f}")


if __name__ == "__main__":
    main()
