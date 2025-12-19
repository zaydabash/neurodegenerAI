"""
MRI feature extraction and preprocessing utilities.
"""

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torchvision import transforms

from shared.lib.logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class MRIPreprocessor(LoggerMixin):
    """MRI volume preprocessing utilities."""

    def __init__(self, target_shape: tuple[int, int, int] = (64, 64, 64)):
        self.target_shape = target_shape
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
            ]
        )

    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess MRI volume."""

        # Step 1: Resize to target shape
        volume_resized = self._resize_volume(volume, self.target_shape)

        # Step 2: Normalize intensity
        volume_normalized = self._normalize_intensity(volume_resized)

        # Step 3: Remove noise
        volume_cleaned = self._denoise_volume(volume_normalized)

        return volume_cleaned.astype(np.float32)

    def extract_slices(
        self, volume: np.ndarray, num_slices: int = 16
    ) -> list[np.ndarray]:
        """Extract representative slices from volume."""

        # Get middle slices from each axis
        axial_slices = self._get_middle_slices(
            volume, axis=2, num_slices=num_slices // 3
        )
        sagittal_slices = self._get_middle_slices(
            volume, axis=0, num_slices=num_slices // 3
        )
        coronal_slices = self._get_middle_slices(
            volume, axis=1, num_slices=num_slices // 3
        )

        all_slices = axial_slices + sagittal_slices + coronal_slices

        return all_slices[:num_slices]  # Ensure we don't exceed requested number

    def _resize_volume(
        self, volume: np.ndarray, target_shape: tuple[int, int, int]
    ) -> np.ndarray:
        """Resize volume to target shape."""

        # Use scipy for 3D resizing
        zoom_factors = [target_shape[i] / volume.shape[i] for i in range(3)]

        volume_resized = ndimage.zoom(volume, zoom_factors, order=1)

        return volume_resized

    def _normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Normalize intensity values."""

        # Clip outliers
        volume = np.clip(volume, np.percentile(volume, 1), np.percentile(volume, 99))

        # Z-score normalization
        mean = np.mean(volume)
        std = np.std(volume)

        if std > 0:
            volume = (volume - mean) / std

        return volume

    def _denoise_volume(self, volume: np.ndarray) -> np.ndarray:
        """Apply denoising to volume."""

        # Apply Gaussian filter for denoising
        volume_denoised = ndimage.gaussian_filter(volume, sigma=1.0)

        return volume_denoised

    def _get_middle_slices(
        self, volume: np.ndarray, axis: int, num_slices: int
    ) -> list[np.ndarray]:
        """Get middle slices along specified axis."""

        total_slices = volume.shape[axis]
        start_idx = total_slices // 4
        end_idx = 3 * total_slices // 4

        if num_slices >= (end_idx - start_idx):
            # Take all slices in the middle region
            slice_indices = range(start_idx, end_idx)
        else:
            # Sample evenly
            slice_indices = np.linspace(start_idx, end_idx - 1, num_slices, dtype=int)

        slices = []
        for idx in slice_indices:
            if axis == 0:
                slice_data = volume[idx, :, :]
            elif axis == 1:
                slice_data = volume[:, idx, :]
            else:  # axis == 2
                slice_data = volume[:, :, idx]

            slices.append(slice_data)

        return slices


class MRIFeatureExtractor(LoggerMixin):
    """Extract handcrafted features from MRI volumes."""

    def __init__(self):
        self.preprocessor = MRIPreprocessor()

    def extract_volume_features(self, volume: np.ndarray) -> dict[str, float]:
        """Extract handcrafted features from MRI volume."""

        features = {}

        # Basic statistics
        features.update(self._extract_intensity_features(volume))

        # Shape features
        features.update(self._extract_shape_features(volume))

        # Texture features
        features.update(self._extract_texture_features(volume))

        # Regional features
        features.update(self._extract_regional_features(volume))

        return features

    def _extract_intensity_features(self, volume: np.ndarray) -> dict[str, float]:
        """Extract intensity-based features."""

        features = {
            "mean_intensity": float(np.mean(volume)),
            "std_intensity": float(np.std(volume)),
            "min_intensity": float(np.min(volume)),
            "max_intensity": float(np.max(volume)),
            "median_intensity": float(np.median(volume)),
            "skewness_intensity": float(self._calculate_skewness(volume)),
            "kurtosis_intensity": float(self._calculate_kurtosis(volume)),
        }

        # Percentile features
        for p in [10, 25, 75, 90]:
            features[f"percentile_{p}"] = float(np.percentile(volume, p))

        return features

    def _extract_shape_features(self, volume: np.ndarray) -> dict[str, float]:
        """Extract shape-based features."""

        # Binarize volume for shape analysis
        threshold = np.percentile(volume, 50)  # Use median as threshold
        binary_volume = (volume > threshold).astype(int)

        # Calculate connected components
        labeled_volume, num_components = ndimage.label(binary_volume)

        features = {
            "num_connected_components": float(num_components),
            "volume_fraction": float(np.sum(binary_volume) / binary_volume.size),
        }

        if num_components > 0:
            # Get largest component
            component_sizes = ndimage.sum(
                binary_volume, labeled_volume, range(1, num_components + 1)
            )
            largest_component_size = np.max(component_sizes)

            features.update(
                {
                    "largest_component_size": float(largest_component_size),
                    "largest_component_ratio": float(
                        largest_component_size / np.sum(binary_volume)
                    ),
                    "component_size_std": float(np.std(component_sizes)),
                }
            )

        return features

    def _extract_texture_features(self, volume: np.ndarray) -> dict[str, float]:
        """Extract texture-based features."""

        features = {}

        # Calculate gradients
        grad_x = np.gradient(volume, axis=0)
        grad_y = np.gradient(volume, axis=1)
        grad_z = np.gradient(volume, axis=2)

        # Gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        features.update(
            {
                "mean_gradient_magnitude": float(np.mean(gradient_magnitude)),
                "std_gradient_magnitude": float(np.std(gradient_magnitude)),
                "max_gradient_magnitude": float(np.max(gradient_magnitude)),
            }
        )

        # Local binary patterns (simplified)
        features.update(self._calculate_local_binary_patterns(volume))

        return features

    def _extract_regional_features(self, volume: np.ndarray) -> dict[str, float]:
        """Extract region-specific features."""

        features = {}

        # Divide volume into regions
        z_mid = volume.shape[2] // 2

        # Anterior and posterior regions
        anterior_region = volume[:, :, :z_mid]
        posterior_region = volume[:, :, z_mid:]

        features.update(
            {
                "anterior_mean": float(np.mean(anterior_region)),
                "posterior_mean": float(np.mean(posterior_region)),
                "anterior_posterior_ratio": float(
                    np.mean(anterior_region) / (np.mean(posterior_region) + 1e-6)
                ),
            }
        )

        # Superior and inferior regions
        y_mid = volume.shape[1] // 2
        superior_region = volume[:, :y_mid, :]
        inferior_region = volume[:, y_mid:, :]

        features.update(
            {
                "superior_mean": float(np.mean(superior_region)),
                "inferior_mean": float(np.mean(inferior_region)),
                "superior_inferior_ratio": float(
                    np.mean(superior_region) / (np.mean(inferior_region) + 1e-6)
                ),
            }
        )

        # Left and right regions
        x_mid = volume.shape[0] // 2
        left_region = volume[:x_mid, :, :]
        right_region = volume[x_mid:, :, :]

        features.update(
            {
                "left_mean": float(np.mean(left_region)),
                "right_mean": float(np.mean(right_region)),
                "left_right_ratio": float(
                    np.mean(left_region) / (np.mean(right_region) + 1e-6)
                ),
            }
        )

        return features

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _calculate_local_binary_patterns(self, volume: np.ndarray) -> dict[str, float]:
        """Calculate simplified local binary patterns."""

        # Sample a subset of the volume for efficiency
        step = max(1, min(volume.shape) // 16)
        sampled_volume = volume[::step, ::step, ::step]

        features = {}

        # Calculate local variance
        kernel_size = 3
        local_mean = ndimage.uniform_filter(sampled_volume, size=kernel_size)
        local_var = (
            ndimage.uniform_filter(sampled_volume**2, size=kernel_size) - local_mean**2
        )

        features.update(
            {
                "mean_local_variance": float(np.mean(local_var)),
                "std_local_variance": float(np.std(local_var)),
                "max_local_variance": float(np.max(local_var)),
            }
        )

        return features


class MRIDataLoader(LoggerMixin):
    """Data loader for MRI volumes with augmentation."""

    def __init__(self, batch_size: int = 16, num_slices: int = 16):
        self.batch_size = batch_size
        self.num_slices = num_slices
        self.preprocessor = MRIPreprocessor()
        self.feature_extractor = MRIFeatureExtractor()

        # Augmentation transforms
        self.augment_transforms = transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )

    def load_volume(self, volume_path: str) -> np.ndarray | None:
        """Load MRI volume from file."""

        try:
            if volume_path.endswith((".nii", ".nii.gz")):
                img = nib.load(volume_path)
                volume = img.get_fdata()
            else:
                # Assume it's a numpy array file
                volume = np.load(volume_path)

            return volume

        except Exception as e:
            self.logger.error(f"Error loading volume {volume_path}: {e}")
            return None

    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess volume for training."""

        return self.preprocessor.preprocess_volume(volume)

    def extract_slices(self, volume: np.ndarray) -> list[np.ndarray]:
        """Extract slices from volume."""

        return self.preprocessor.extract_slices(volume, self.num_slices)

    def extract_features(self, volume: np.ndarray) -> dict[str, float]:
        """Extract handcrafted features from volume."""

        return self.feature_extractor.extract_volume_features(volume)

    def create_batch(self, volumes: list[np.ndarray]) -> torch.Tensor:
        """Create batch tensor from volumes."""

        # Extract slices from each volume
        all_slices = []
        for volume in volumes:
            slices = self.extract_slices(volume)
            all_slices.extend(slices)

        # Convert to tensor
        slices_tensor = torch.stack(
            [
                torch.from_numpy(slice_data).unsqueeze(0)  # Add channel dimension
                for slice_data in all_slices
            ]
        )

        return slices_tensor

    def augment_volume(self, volume: np.ndarray) -> np.ndarray:
        """Apply data augmentation to volume."""

        # Extract slices
        slices = self.extract_slices(volume)

        # Apply augmentation to each slice
        augmented_slices = []
        for slice_data in slices:
            # Convert to PIL-like format for transforms
            slice_tensor = torch.from_numpy(slice_data).unsqueeze(0)

            # Apply augmentation
            augmented_slice = self.augment_transforms(slice_tensor)

            augmented_slices.append(augmented_slice.squeeze(0).numpy())

        # Reconstruct volume from augmented slices (simplified)
        # In practice, you might want to implement proper 3D augmentation
        return np.stack(augmented_slices, axis=2)
