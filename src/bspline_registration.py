"""
B-Spline Deformable Registration Module
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional, List
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize, least_squares
from scipy.ndimage import map_coordinates
import SimpleITK as sitk
from mutual_information import MutualInformation


class BSplineTransform:
    """
    B-Spline transformation for deformable registration.
    """

    def __init__(self,
                 image_shape: Tuple[int, int],
                 grid_spacing: Tuple[int, int] = (32, 32),
                 spline_order: int = 3):
        """
        Initialize B-Spline transform.

        Args:
            image_shape: Shape of the image (height, width)
            grid_spacing: Spacing between control points
            spline_order: Order of B-spline (typically 3 for cubic)
        """
        self.image_shape = image_shape
        self.grid_spacing = grid_spacing
        self.spline_order = spline_order

        # Calculate number of control points
        self.grid_shape = (
            image_shape[0] // grid_spacing[0] + 4,
            image_shape[1] // grid_spacing[1] + 4
        )

        # Initialize control point grid
        self.control_points = self._initialize_control_points()

        # Pre-compute B-spline basis functions
        self.basis_functions = self._compute_basis_functions()

    def _initialize_control_points(self) -> np.ndarray:
        """
        Initialize regular grid of control points.

        Returns:
            Control points array of shape (grid_h, grid_w, 2)
        """
        grid_h, grid_w = self.grid_shape

        # Create regular grid
        y_coords = np.linspace(
            -self.grid_spacing[0],
            self.image_shape[0] + self.grid_spacing[0],
            grid_h
        )
        x_coords = np.linspace(
            -self.grid_spacing[1],
            self.image_shape[1] + self.grid_spacing[1],
            grid_w
        )

        # Create meshgrid
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Stack to create control points
        control_points = np.stack([xx, yy], axis=-1)

        return control_points

    def _compute_basis_functions(self) -> Dict:
        """
        Pre-compute B-spline basis functions.

        Returns:
            Dictionary of basis functions
        """
        # Cubic B-spline basis function
        def b3(t):
            """Cubic B-spline basis function."""
            t = np.abs(t)
            return np.where(
                t < 1,
                (3 * t**3 - 6 * t**2 + 4) / 6,
                np.where(
                    t < 2,
                    (-t**3 + 6 * t**2 - 12 * t + 8) / 6,
                    0
                )
            )

        return {'cubic': b3}

    def params_to_displacement_field(self,
                                    params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert parameter vector to displacement field.

        Args:
            params: Flattened control point displacements

        Returns:
            Displacement fields (dx, dy)
        """
        # Reshape parameters to control point displacements
        num_control_points = self.grid_shape[0] * self.grid_shape[1]
        dx_control = params[:num_control_points].reshape(self.grid_shape)
        dy_control = params[num_control_points:].reshape(self.grid_shape)

        # Interpolate to full resolution
        dx_field = self._interpolate_control_points(dx_control)
        dy_field = self._interpolate_control_points(dy_control)

        return dx_field, dy_field

    def _interpolate_control_points(self,
                                   control_values: np.ndarray) -> np.ndarray:
        """
        Interpolate control point values to full resolution.

        Args:
            control_values: Values at control points

        Returns:
            Interpolated field at full resolution
        """
        # Create coordinate arrays for control points
        y_control = np.linspace(0, self.image_shape[0], self.grid_shape[0])
        x_control = np.linspace(0, self.image_shape[1], self.grid_shape[1])

        # Create interpolator
        interpolator = RectBivariateSpline(
            y_control, x_control, control_values,
            kx=min(3, self.grid_shape[0] - 1),
            ky=min(3, self.grid_shape[1] - 1)
        )

        # Create full resolution coordinates
        y_full = np.arange(self.image_shape[0])
        x_full = np.arange(self.image_shape[1])

        # Interpolate
        field = interpolator(y_full, x_full)

        return field

    def apply_transform(self,
                       image: np.ndarray,
                       displacement_field: Tuple[np.ndarray, np.ndarray],
                       interpolation: str = 'linear') -> np.ndarray:
        """
        Apply B-spline transformation to an image.

        Args:
            image: Input image
            displacement_field: Displacement fields (dx, dy)
            interpolation: Interpolation method

        Returns:
            Transformed image
        """
        dx_field, dy_field = displacement_field
        h, w = image.shape[:2]

        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(
            np.arange(h), np.arange(w), indexing='ij'
        )

        # Apply displacement
        new_y = y_coords + dy_field
        new_x = x_coords + dx_field

        # Map coordinates
        order = 1 if interpolation == 'linear' else 3
        transformed = map_coordinates(
            image,
            [new_y, new_x],
            order=order,
            mode='reflect'
        )

        return transformed

    def regularization_term(self,
                           params: np.ndarray,
                           alpha: float = 0.1) -> float:
        """
        Compute regularization term for smoothness.

        Args:
            params: Control point parameters
            alpha: Regularization weight

        Returns:
            Regularization value
        """
        # Reshape parameters
        num_control_points = self.grid_shape[0] * self.grid_shape[1]
        dx_control = params[:num_control_points].reshape(self.grid_shape)
        dy_control = params[num_control_points:].reshape(self.grid_shape)

        # Compute gradients (approximation of bending energy)
        dx_grad_y = np.diff(dx_control, axis=0)
        dx_grad_x = np.diff(dx_control, axis=1)
        dy_grad_y = np.diff(dy_control, axis=0)
        dy_grad_x = np.diff(dy_control, axis=1)

        # Compute regularization
        reg = alpha * (
            np.sum(dx_grad_y**2) + np.sum(dx_grad_x**2) +
            np.sum(dy_grad_y**2) + np.sum(dy_grad_x**2)
        )

        return reg


class BSplineRegistration:
    """
    B-Spline deformable registration using mutual information.
    """

    def __init__(self,
                 grid_spacing: Tuple[int, int] = (32, 32),
                 metric: str = 'nmi',
                 optimizer: str = 'lbfgs',
                 num_bins: int = 32,
                 regularization: float = 0.01,
                 multi_resolution: bool = True,
                 num_levels: int = 3):
        """
        Initialize B-Spline registration.

        Args:
            grid_spacing: Initial grid spacing for control points
            metric: Similarity metric
            optimizer: Optimization method
            num_bins: Number of histogram bins
            regularization: Regularization weight
            multi_resolution: Whether to use multi-resolution
            num_levels: Number of resolution levels
        """
        self.grid_spacing = grid_spacing
        self.metric = metric
        self.optimizer = optimizer
        self.regularization = regularization
        self.multi_resolution = multi_resolution
        self.num_levels = num_levels

        # Initialize metric calculator
        self.mi_calculator = MutualInformation(num_bins=num_bins)

        # Registration parameters
        self.max_iterations = 50
        self.tolerance = 1e-4

    def register(self,
                fixed_image: np.ndarray,
                moving_image: np.ndarray,
                initial_displacement: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                mask: Optional[np.ndarray] = None,
                affine_params: Optional[np.ndarray] = None) -> Dict:
        """
        Perform B-spline deformable registration.

        Args:
            fixed_image: Reference image
            moving_image: Moving image
            initial_displacement: Initial displacement field
            mask: Optional mask
            affine_params: Optional affine pre-alignment parameters

        Returns:
            Registration results
        """
        # Normalize images
        fixed_image = self._normalize_image(fixed_image)
        moving_image = self._normalize_image(moving_image)

        # Apply affine pre-alignment if provided
        if affine_params is not None:
            from affine_registration import AffineTransform
            affine = AffineTransform()
            matrix = affine.params_to_matrix(affine_params, moving_image.shape)
            moving_image = affine.apply_transform(moving_image, matrix)

        if self.multi_resolution:
            # Multi-resolution registration
            displacement_field = self._multi_resolution_registration(
                fixed_image, moving_image, initial_displacement, mask
            )
        else:
            # Single resolution registration
            displacement_field = self._single_resolution_registration(
                fixed_image, moving_image, initial_displacement, mask,
                self.grid_spacing
            )

        # Apply final transformation
        transform = BSplineTransform(fixed_image.shape, self.grid_spacing)
        registered_image = transform.apply_transform(moving_image, displacement_field)

        # Compute final metric
        final_metric = self._compute_metric(fixed_image, registered_image, mask)

        return {
            'displacement_field': displacement_field,
            'registered_image': registered_image,
            'metric': final_metric,
            'fixed_image': fixed_image,
            'moving_image': moving_image,
            'grid_spacing': self.grid_spacing
        }

    def _multi_resolution_registration(self,
                                     fixed_image: np.ndarray,
                                     moving_image: np.ndarray,
                                     initial_displacement: Optional[Tuple[np.ndarray, np.ndarray]],
                                     mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-resolution B-spline registration.

        Args:
            fixed_image: Reference image
            moving_image: Moving image
            initial_displacement: Initial displacement
            mask: Optional mask

        Returns:
            Final displacement field
        """
        # Build image pyramids
        fixed_pyramid = self._build_pyramid(fixed_image, self.num_levels)
        moving_pyramid = self._build_pyramid(moving_image, self.num_levels)

        displacement_field = initial_displacement

        # Register at each level (coarse to fine)
        for level in range(self.num_levels - 1, -1, -1):
            # Get images at current level
            fixed_level = fixed_pyramid[level]
            moving_level = moving_pyramid[level]

            # Adapt grid spacing for current level
            scale_factor = 2 ** level
            grid_spacing = (
                self.grid_spacing[0] * scale_factor,
                self.grid_spacing[1] * scale_factor
            )

            # Resize mask if provided
            mask_level = None
            if mask is not None:
                mask_level = cv2.resize(
                    mask.astype(float),
                    (fixed_level.shape[1], fixed_level.shape[0])
                ) > 0.5

            # Upsample displacement field if coming from coarser level
            if displacement_field is not None and level < self.num_levels - 1:
                displacement_field = self._upsample_displacement_field(
                    displacement_field,
                    fixed_level.shape
                )

            print(f"B-spline registration at level {level}, grid spacing: {grid_spacing}")

            # Register at current level
            displacement_field = self._single_resolution_registration(
                fixed_level, moving_level, displacement_field, mask_level,
                grid_spacing
            )

        return displacement_field

    def _single_resolution_registration(self,
                                      fixed_image: np.ndarray,
                                      moving_image: np.ndarray,
                                      initial_displacement: Optional[Tuple[np.ndarray, np.ndarray]],
                                      mask: Optional[np.ndarray],
                                      grid_spacing: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single resolution B-spline registration.

        Args:
            fixed_image: Reference image
            moving_image: Moving image
            initial_displacement: Initial displacement
            mask: Optional mask
            grid_spacing: Grid spacing for this level

        Returns:
            Optimized displacement field
        """
        # Initialize B-spline transform
        transform = BSplineTransform(fixed_image.shape, grid_spacing)

        # Initialize parameters
        num_params = 2 * transform.grid_shape[0] * transform.grid_shape[1]
        if initial_displacement is not None:
            # Convert displacement field to control points
            params = self._displacement_to_params(
                initial_displacement, transform.grid_shape
            )
        else:
            params = np.zeros(num_params)

        # Define objective function
        def objective(p):
            # Get displacement field
            displacement = transform.params_to_displacement_field(p)

            # Apply transformation
            transformed = transform.apply_transform(moving_image, displacement)

            # Compute similarity metric
            similarity = self._compute_metric(fixed_image, transformed, mask)

            # Add regularization
            reg = transform.regularization_term(p, self.regularization)

            # Return negative similarity + regularization
            return -similarity + reg

        # Set bounds
        max_displacement = min(fixed_image.shape) / 4
        bounds = [(-max_displacement, max_displacement)] * num_params

        # Optimize
        if self.optimizer == 'lbfgs':
            result = minimize(
                objective,
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            optimized_params = result.x
        elif self.optimizer == 'powell':
            result = minimize(
                objective,
                params,
                method='Powell',
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            optimized_params = result.x
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        # Convert to displacement field
        displacement_field = transform.params_to_displacement_field(optimized_params)

        return displacement_field

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

    def _build_pyramid(self, image: np.ndarray, levels: int) -> List[np.ndarray]:
        """
        Build image pyramid.

        Args:
            image: Input image
            levels: Number of levels

        Returns:
            List of images at different resolutions
        """
        pyramid = [image]

        for _ in range(1, levels):
            downsampled = cv2.pyrDown(pyramid[-1])
            pyramid.append(downsampled)

        return pyramid

    def _upsample_displacement_field(self,
                                    displacement: Tuple[np.ndarray, np.ndarray],
                                    target_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Upsample displacement field to target resolution.

        Args:
            displacement: Current displacement field
            target_shape: Target shape

        Returns:
            Upsampled displacement field
        """
        dx, dy = displacement

        # Upsample and scale
        dx_up = cv2.resize(dx, (target_shape[1], target_shape[0])) * 2
        dy_up = cv2.resize(dy, (target_shape[1], target_shape[0])) * 2

        return dx_up, dy_up

    def _displacement_to_params(self,
                               displacement: Tuple[np.ndarray, np.ndarray],
                               grid_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert displacement field to control point parameters.

        Args:
            displacement: Displacement field
            grid_shape: Shape of control point grid

        Returns:
            Flattened parameter vector
        """
        dx, dy = displacement

        # Downsample to control point resolution
        dx_control = cv2.resize(dx, (grid_shape[1], grid_shape[0]))
        dy_control = cv2.resize(dy, (grid_shape[1], grid_shape[0]))

        # Flatten
        params = np.concatenate([dx_control.flatten(), dy_control.flatten()])

        return params

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


if __name__ == "__main__":
    # Test B-spline registration
    np.random.seed(42)

    # Create test images
    size = (128, 128)
    fixed = np.random.randn(*size)

    # Create deformed moving image
    transform = BSplineTransform(size, grid_spacing=(32, 32))

    # Create random deformation
    num_params = 2 * transform.grid_shape[0] * transform.grid_shape[1]
    deform_params = np.random.randn(num_params) * 5
    displacement = transform.params_to_displacement_field(deform_params)
    moving = transform.apply_transform(fixed, displacement)

    # Add noise
    moving += np.random.randn(*size) * 0.1

    # Register
    registration = BSplineRegistration(
        grid_spacing=(32, 32),
        metric='nmi',
        regularization=0.01,
        multi_resolution=True
    )

    result = registration.register(fixed, moving)

    print(f"Final metric: {result['metric']:.4f}")
    print(f"Grid spacing: {result['grid_spacing']}")