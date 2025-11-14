"""
Data Loading and Preprocessing Utilities for Multimodal Brain Tumor Images
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import SimpleITK as sitk
from skimage import exposure, transform
from tqdm import tqdm


class MultimodalDataLoader:
    """Load and preprocess CT and MRI images for registration."""

    def __init__(self, dataset_path: str):
        """
        Initialize the data loader.

        Args:
            dataset_path: Path to the dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.ct_path = self.dataset_path / "Brain Tumor CT scan Images"
        self.mri_path = self.dataset_path / "Brain Tumor MRI images"

        # Verify paths exist
        if not self.ct_path.exists():
            raise ValueError(f"CT path not found: {self.ct_path}")
        if not self.mri_path.exists():
            raise ValueError(f"MRI path not found: {self.mri_path}")

    def load_image(self, image_path: str, as_sitk: bool = True) -> np.ndarray:
        """
        Load an image from file.

        Args:
            image_path: Path to the image file
            as_sitk: If True, return as SimpleITK image, else numpy array

        Returns:
            Image as numpy array or SimpleITK image
        """
        # Check file extension
        ext = Path(image_path).suffix.lower()

        if ext in ['.jpg', '.jpeg', '.png']:
            # Load standard image format
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            if as_sitk:
                # Convert to SimpleITK image
                sitk_img = sitk.GetImageFromArray(img)
                return sitk_img
            return img

        elif ext in ['.nii', '.nii.gz', '.dcm']:
            # Load medical image format
            sitk_img = sitk.ReadImage(str(image_path))
            if as_sitk:
                return sitk_img
            return sitk.GetArrayFromImage(sitk_img)

        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def preprocess_image(self,
                        image: np.ndarray,
                        target_size: Tuple[int, int] = (256, 256),
                        normalize: bool = True,
                        histogram_matching: bool = False,
                        reference_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Preprocess image for registration.

        Args:
            image: Input image
            target_size: Target size for resizing
            normalize: Whether to normalize intensities
            histogram_matching: Whether to apply histogram matching
            reference_image: Reference image for histogram matching

        Returns:
            Preprocessed image
        """
        # Ensure image is 2D
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize image
        if image.shape != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        # Apply histogram matching if requested
        if histogram_matching and reference_image is not None:
            image = exposure.match_histograms(image, reference_image)

        # Normalize intensities
        if normalize:
            image = self.normalize_intensity(image)

        return image

    def normalize_intensity(self, image: np.ndarray,
                           percentile_range: Tuple[float, float] = (1, 99)) -> np.ndarray:
        """
        Normalize image intensities using percentile-based normalization.

        Args:
            image: Input image
            percentile_range: Percentile range for normalization

        Returns:
            Normalized image
        """
        p_low, p_high = np.percentile(image, percentile_range)
        image_clipped = np.clip(image, p_low, p_high)

        if p_high > p_low:
            image_normalized = (image_clipped - p_low) / (p_high - p_low)
        else:
            image_normalized = image_clipped

        return image_normalized

    def load_image_pairs(self,
                        modality1: str = 'ct',
                        modality2: str = 'mri',
                        category: str = 'Healthy',
                        num_pairs: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load pairs of images from different modalities.

        Args:
            modality1: First modality ('ct' or 'mri')
            modality2: Second modality ('ct' or 'mri')
            category: 'Healthy' or 'Tumor'
            num_pairs: Number of pairs to load

        Returns:
            List of image pairs
        """
        # Get paths
        path1 = self.ct_path if modality1.lower() == 'ct' else self.mri_path
        path2 = self.ct_path if modality2.lower() == 'ct' else self.mri_path

        # Get image files
        images1 = list((path1 / category).glob('*.[jp][pn][gg]'))
        images2 = list((path2 / category).glob('*.[jp][pn][gg]'))

        # Load pairs
        pairs = []
        num_pairs = min(num_pairs, len(images1), len(images2))

        print(f"Loading {num_pairs} image pairs...")
        for i in tqdm(range(num_pairs)):
            img1 = self.load_image(str(images1[i]), as_sitk=False)
            img2 = self.load_image(str(images2[i]), as_sitk=False)

            # Preprocess
            img1 = self.preprocess_image(img1)
            img2 = self.preprocess_image(img2)

            pairs.append((img1, img2))

        return pairs

    def create_synthetic_deformation(self,
                                   image: np.ndarray,
                                   max_rotation: float = 15,
                                   max_translation: float = 20,
                                   max_scale: float = 0.1,
                                   elastic_alpha: float = 0,
                                   elastic_sigma: float = 0) -> Tuple[np.ndarray, Dict]:
        """
        Create synthetic deformation for testing registration.

        Args:
            image: Input image
            max_rotation: Maximum rotation in degrees
            max_translation: Maximum translation in pixels
            max_scale: Maximum scale factor change
            elastic_alpha: Elastic deformation alpha parameter
            elastic_sigma: Elastic deformation sigma parameter

        Returns:
            Deformed image and transformation parameters
        """
        h, w = image.shape[:2]

        # Random transformation parameters
        angle = np.random.uniform(-max_rotation, max_rotation)
        tx = np.random.uniform(-max_translation, max_translation)
        ty = np.random.uniform(-max_translation, max_translation)
        scale = np.random.uniform(1 - max_scale, 1 + max_scale)

        # Create affine transformation matrix
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        # Apply affine transformation
        deformed = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Apply elastic deformation if specified
        if elastic_alpha > 0 and elastic_sigma > 0:
            deformed = self._apply_elastic_deformation(deformed, elastic_alpha, elastic_sigma)

        # Store transformation parameters
        params = {
            'rotation': angle,
            'translation': (tx, ty),
            'scale': scale,
            'matrix': M,
            'elastic_alpha': elastic_alpha,
            'elastic_sigma': elastic_sigma
        }

        return deformed, params

    def _apply_elastic_deformation(self,
                                  image: np.ndarray,
                                  alpha: float,
                                  sigma: float) -> np.ndarray:
        """
        Apply elastic deformation to an image.

        Args:
            image: Input image
            alpha: Deformation strength
            sigma: Deformation smoothness

        Returns:
            Elastically deformed image
        """
        h, w = image.shape[:2]

        # Generate random displacement fields
        dx = np.random.randn(h, w) * alpha
        dy = np.random.randn(h, w) * alpha

        # Smooth the displacement fields
        from scipy.ndimage import gaussian_filter
        dx = gaussian_filter(dx, sigma)
        dy = gaussian_filter(dy, sigma)

        # Create mesh grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Apply displacement
        indices = (y + dy).reshape(-1), (x + dx).reshape(-1)

        # Interpolate
        from scipy.ndimage import map_coordinates
        deformed = map_coordinates(image, indices, order=1, mode='reflect').reshape(h, w)

        return deformed

    def save_preprocessed_batch(self,
                               output_dir: str,
                               modality: str = 'ct',
                               category: str = 'Healthy',
                               num_images: int = 100):
        """
        Preprocess and save a batch of images.

        Args:
            output_dir: Output directory
            modality: 'ct' or 'mri'
            category: 'Healthy' or 'Tumor'
            num_images: Number of images to process
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get source path
        source_path = self.ct_path if modality.lower() == 'ct' else self.mri_path

        # Get image files
        images = list((source_path / category).glob('*.[jp][pn][gg]'))[:num_images]

        print(f"Processing {len(images)} {modality} {category} images...")
        for img_path in tqdm(images):
            # Load and preprocess
            img = self.load_image(str(img_path), as_sitk=False)
            img = self.preprocess_image(img)

            # Save
            output_file = output_path / f"{modality}_{category}_{img_path.stem}.npy"
            np.save(output_file, img)


if __name__ == "__main__":
    # Example usage
    loader = MultimodalDataLoader("./Dataset")

    # Load sample image pairs
    pairs = loader.load_image_pairs('ct', 'mri', 'Healthy', num_pairs=5)
    print(f"Loaded {len(pairs)} image pairs")

    # Create synthetic deformation
    if pairs:
        original = pairs[0][0]
        deformed, params = loader.create_synthetic_deformation(
            original,
            max_rotation=10,
            max_translation=15,
            elastic_alpha=20,
            elastic_sigma=3
        )
        print(f"Applied deformation with params: {params}")