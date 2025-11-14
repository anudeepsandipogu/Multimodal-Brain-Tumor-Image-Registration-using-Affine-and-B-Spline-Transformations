"""
Affine Registration Module for Medical Images
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional, Callable
from scipy.optimize import minimize, differential_evolution
import SimpleITK as sitk
from mutual_information import MutualInformation


class AffineTransform:
    """
    Affine transformation implementation with various parameterizations.
    """

    def __init__(self, center: Optional[Tuple[float, float]] = None):
        """
        Initialize affine transform.

        Args:
            center: Center of rotation (if None, use image center)
        """
        self.center = center

    def params_to_matrix(self,
                         params: np.ndarray,
                         image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert parameter vector to affine transformation matrix.

        Args:
            params: Parameter vector [tx, ty, rotation, scale_x, scale_y, shear]
            image_shape: Shape of the image (height, width)

        Returns:
            3x3 affine transformation matrix
        """
        tx, ty, rotation, scale_x, scale_y, shear = params

        # Get center of rotation
        if self.center is None:
            cy, cx = image_shape[0] / 2, image_shape[1] / 2
        else:
            cx, cy = self.center

        # Convert rotation to radians
        theta = np.radians(rotation)

        # Build transformation matrix
        # Translation to origin
        T1 = np.array([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1]
        ])

        # Rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        # Scale matrix
        S = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1]
        ])

        # Shear matrix
        Sh = np.array([
            [1, shear, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        # Translation back from origin
        T2 = np.array([
            [1, 0, cx + tx],
            [0, 1, cy + ty],
            [0, 0, 1]
        ])

        # Combine transformations
        matrix = T2 @ Sh @ S @ R @ T1

        return matrix

    def apply_transform(self,
                       image: np.ndarray,
                       matrix: np.ndarray,
                       interpolation: str = 'linear') -> np.ndarray:
        """
        Apply affine transformation to an image.

        Args:
            image: Input image
            matrix: 3x3 transformation matrix
            interpolation: Interpolation method ('linear', 'cubic', 'nearest')

        Returns:
            Transformed image
        """
        # Select interpolation method
        interp_methods = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC
        }
        interp = interp_methods.get(interpolation, cv2.INTER_LINEAR)

        # Apply transformation
        h, w = image.shape[:2]
        transformed = cv2.warpAffine(
            image,
            matrix[:2, :],
            (w, h),
            flags=interp,
            borderMode=cv2.BORDER_REFLECT
        )

        return transformed

    def inverse_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute inverse of affine transformation matrix.

        Args:
            matrix: 3x3 transformation matrix

        Returns:
            Inverse transformation matrix
        """
        return np.linalg.inv(matrix)


class AffineRegistration:
    """
    Affine registration using mutual information optimization.
    """

    def __init__(self,
                 metric: str = 'mi',
                 optimizer: str = 'powell',
                 num_bins: int = 32,
                 multi_resolution: bool = True,
                 num_levels: int = 3):
        """
        Initialize affine registration.

        Args:
            metric: Similarity metric ('mi', 'nmi', 'ecc')
            optimizer: Optimization method ('powell', 'lbfgs', 'evolution')
            num_bins: Number of histogram bins for MI
            multi_resolution: Whether to use multi-resolution approach
            num_levels: Number of resolution levels
        """
        self.metric = metric
        self.optimizer = optimizer
        self.multi_resolution = multi_resolution
        self.num_levels = num_levels

        # Initialize metric calculator
        self.mi_calculator = MutualInformation(num_bins=num_bins)
        self.transform = AffineTransform()

        # Registration parameters
        self.max_iterations = 100
        self.tolerance = 1e-5

    def register(self,
                fixed_image: np.ndarray,
                moving_image: np.ndarray,
                initial_params: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None) -> Dict:
        """
        Perform affine registration.

        Args:
            fixed_image: Reference/target image
            moving_image: Image to be registered
            initial_params: Initial transformation parameters
            mask: Optional mask for region of interest

        Returns:
            Dictionary with registration results
        """
        # Ensure images are normalized
        fixed_image = self._normalize_image(fixed_image)
        moving_image = self._normalize_image(moving_image)

        if self.multi_resolution:
            # Multi-resolution registration
            params = self._multi_resolution_registration(
                fixed_image, moving_image, initial_params, mask
            )
        else:
            # Single resolution registration
            params = self._single_resolution_registration(
                fixed_image, moving_image, initial_params, mask
            )

        # Compute final transformation matrix
        matrix = self.transform.params_to_matrix(params, fixed_image.shape)

        # Apply transformation
        registered_image = self.transform.apply_transform(moving_image, matrix)

        # Compute final metric
        final_metric = self._compute_metric(fixed_image, registered_image, mask)

        return {
            'params': params,
            'matrix': matrix,
            'registered_image': registered_image,
            'metric': final_metric,
            'fixed_image': fixed_image,
            'moving_image': moving_image
        }

    def _multi_resolution_registration(self,
                                     fixed_image: np.ndarray,
                                     moving_image: np.ndarray,
                                     initial_params: Optional[np.ndarray],
                                     mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Multi-resolution registration approach.

        Args:
            fixed_image: Reference image
            moving_image: Moving image
            initial_params: Initial parameters
            mask: Optional mask

        Returns:
            Optimized parameters
        """
        # Build image pyramids
        fixed_pyramid = self._build_pyramid(fixed_image, self.num_levels)
        moving_pyramid = self._build_pyramid(moving_image, self.num_levels)

        # Initialize parameters
        if initial_params is None:
            params = np.array([0, 0, 0, 1, 1, 0], dtype=float)
        else:
            params = initial_params.copy()

        # Register at each level (coarse to fine)
        for level in range(self.num_levels - 1, -1, -1):
            # Get images at current level
            fixed_level = fixed_pyramid[level]
            moving_level = moving_pyramid[level]

            # Scale translation parameters
            if level < self.num_levels - 1:
                scale_factor = fixed_level.shape[0] / fixed_pyramid[level + 1].shape[0]
                params[0] *= scale_factor  # tx
                params[1] *= scale_factor  # ty

            # Resize mask if provided
            mask_level = None
            if mask is not None:
                mask_level = cv2.resize(
                    mask.astype(float),
                    (fixed_level.shape[1], fixed_level.shape[0])
                ) > 0.5

            print(f"Registration at level {level}, size: {fixed_level.shape}")

            # Register at current level
            params = self._single_resolution_registration(
                fixed_level, moving_level, params, mask_level
            )

        return params

    def _single_resolution_registration(self,
                                      fixed_image: np.ndarray,
                                      moving_image: np.ndarray,
                                      initial_params: Optional[np.ndarray],
                                      mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Single resolution registration.

        Args:
            fixed_image: Reference image
            moving_image: Moving image
            initial_params: Initial parameters
            mask: Optional mask

        Returns:
            Optimized parameters
        """
        # Initialize parameters
        if initial_params is None:
            params = np.array([0, 0, 0, 1, 1, 0], dtype=float)
        else:
            params = initial_params.copy()

        # Define objective function
        def objective(p):
            # Apply transformation
            matrix = self.transform.params_to_matrix(p, fixed_image.shape)
            transformed = self.transform.apply_transform(moving_image, matrix)

            # Compute negative metric (for minimization)
            metric = self._compute_metric(fixed_image, transformed, mask)
            return -metric

        # Set parameter bounds
        bounds = [
            (-50, 50),      # tx
            (-50, 50),      # ty
            (-180, 180),    # rotation
            (0.5, 1.5),     # scale_x
            (0.5, 1.5),     # scale_y
            (-0.5, 0.5)     # shear
        ]

        # Optimize
        if self.optimizer == 'powell':
            result = minimize(
                objective,
                params,
                method='Powell',
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
        elif self.optimizer == 'lbfgs':
            result = minimize(
                objective,
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
        elif self.optimizer == 'evolution':
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.max_iterations,
                tol=self.tolerance,
                seed=42
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        return result.x

    def _compute_metric(self,
                       image1: np.ndarray,
                       image2: np.ndarray,
                       mask: Optional[np.ndarray]) -> float:
        """
        Compute similarity metric.

        Args:
            image1: First image
            image2: Second image
            mask: Optional mask

        Returns:
            Metric value
        """
        if self.metric == 'mi':
            return self.mi_calculator.compute_mi(image1, image2, mask)
        elif self.metric == 'nmi':
            return self.mi_calculator.compute_nmi(image1, image2, mask)
        elif self.metric == 'ecc':
            return self.mi_calculator.compute_ecc(image1, image2, mask)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _build_pyramid(self, image: np.ndarray, levels: int) -> list:
        """
        Build image pyramid for multi-resolution registration.

        Args:
            image: Input image
            levels: Number of pyramid levels

        Returns:
            List of images at different resolutions
        """
        pyramid = [image]

        for _ in range(1, levels):
            # Downsample by factor of 2
            downsampled = cv2.pyrDown(pyramid[-1])
            pyramid.append(downsampled)

        return pyramid

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.

        Args:
            image: Input image

        Returns:
            Normalized image
        """
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        return image


class RigidRegistration(AffineRegistration):
    """
    Rigid registration (translation + rotation only).
    """

    def _single_resolution_registration(self,
                                      fixed_image: np.ndarray,
                                      moving_image: np.ndarray,
                                      initial_params: Optional[np.ndarray],
                                      mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Single resolution rigid registration.

        Args:
            fixed_image: Reference image
            moving_image: Moving image
            initial_params: Initial parameters
            mask: Optional mask

        Returns:
            Optimized parameters (with scale=1, shear=0)
        """
        # Initialize parameters for rigid transform
        if initial_params is None:
            rigid_params = np.array([0, 0, 0], dtype=float)  # tx, ty, rotation
        else:
            # Extract only rigid parameters
            rigid_params = initial_params[:3].copy()

        # Define objective function
        def objective(p):
            # Create full parameter vector with fixed scale and shear
            full_params = np.array([p[0], p[1], p[2], 1, 1, 0])

            # Apply transformation
            matrix = self.transform.params_to_matrix(full_params, fixed_image.shape)
            transformed = self.transform.apply_transform(moving_image, matrix)

            # Compute negative metric
            metric = self._compute_metric(fixed_image, transformed, mask)
            return -metric

        # Set bounds for rigid parameters
        bounds = [
            (-50, 50),      # tx
            (-50, 50),      # ty
            (-180, 180)     # rotation
        ]

        # Optimize
        result = minimize(
            objective,
            rigid_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.max_iterations}
        )

        # Return full parameter vector
        return np.array([result.x[0], result.x[1], result.x[2], 1, 1, 0])


if __name__ == "__main__":
    # Test affine registration
    np.random.seed(42)

    # Create test images
    size = (128, 128)
    fixed = np.random.randn(*size)

    # Create moving image with known transformation
    true_params = np.array([10, -5, 15, 1.1, 0.9, 0.1])
    transform = AffineTransform()
    matrix = transform.params_to_matrix(true_params, size)
    moving = transform.apply_transform(fixed, matrix)

    # Add noise
    moving += np.random.randn(*size) * 0.1

    # Register
    registration = AffineRegistration(
        metric='nmi',
        optimizer='lbfgs',
        multi_resolution=True
    )

    result = registration.register(fixed, moving)

    print(f"True params: {true_params}")
    print(f"Estimated params: {result['params']}")
    print(f"Final metric: {result['metric']:.4f}")