"""
Evaluation Metrics for Medical Image Registration
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
from scipy.spatial.distance import euclidean
from sklearn.metrics import confusion_matrix
import SimpleITK as sitk


class RegistrationMetrics:
    """
    Comprehensive evaluation metrics for image registration.
    """

    def __init__(self):
        """Initialize registration metrics calculator."""
        pass

    def target_registration_error(self,
                                 landmarks_fixed: np.ndarray,
                                 landmarks_moving: np.ndarray,
                                 transform_matrix: Optional[np.ndarray] = None,
                                 displacement_field: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict:
        """
        Calculate Target Registration Error (TRE) using corresponding landmarks.

        Args:
            landmarks_fixed: Landmarks in fixed image (N x 2)
            landmarks_moving: Landmarks in moving image (N x 2)
            transform_matrix: Affine transformation matrix (3 x 3)
            displacement_field: Deformation field (dx, dy)

        Returns:
            Dictionary with TRE statistics
        """
        assert landmarks_fixed.shape == landmarks_moving.shape, \
            "Fixed and moving landmarks must have same shape"

        # Transform moving landmarks
        if transform_matrix is not None:
            # Apply affine transformation
            landmarks_transformed = self._apply_affine_to_points(
                landmarks_moving, transform_matrix
            )
        elif displacement_field is not None:
            # Apply deformation field
            landmarks_transformed = self._apply_displacement_to_points(
                landmarks_moving, displacement_field
            )
        else:
            # No transformation
            landmarks_transformed = landmarks_moving

        # Calculate distances
        distances = np.sqrt(np.sum((landmarks_fixed - landmarks_transformed) ** 2, axis=1))

        # Calculate statistics
        tre_stats = {
            'mean_tre': np.mean(distances),
            'std_tre': np.std(distances),
            'median_tre': np.median(distances),
            'max_tre': np.max(distances),
            'min_tre': np.min(distances),
            'rmse': np.sqrt(np.mean(distances ** 2)),
            'distances': distances,
            'success_rate_2mm': np.mean(distances < 2.0) * 100,
            'success_rate_5mm': np.mean(distances < 5.0) * 100,
            'success_rate_10mm': np.mean(distances < 10.0) * 100
        }

        return tre_stats

    def dice_coefficient(self,
                        mask1: np.ndarray,
                        mask2: np.ndarray) -> float:
        """
        Calculate Dice Similarity Coefficient between two binary masks.

        DSC = 2 * |A ∩ B| / (|A| + |B|)

        Args:
            mask1: First binary mask
            mask2: Second binary mask

        Returns:
            Dice coefficient (0 to 1)
        """
        # Ensure binary masks
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)

        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        volume_sum = mask1.sum() + mask2.sum()

        if volume_sum == 0:
            return 1.0  # Both masks are empty

        dice = 2.0 * intersection / volume_sum
        return dice

    def jaccard_index(self,
                     mask1: np.ndarray,
                     mask2: np.ndarray) -> float:
        """
        Calculate Jaccard Index (IoU) between two binary masks.

        Jaccard = |A ∩ B| / |A ∪ B|

        Args:
            mask1: First binary mask
            mask2: Second binary mask

        Returns:
            Jaccard index (0 to 1)
        """
        # Ensure binary masks
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)

        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 1.0  # Both masks are empty

        jaccard = intersection / union
        return jaccard

    def hausdorff_distance(self,
                          mask1: np.ndarray,
                          mask2: np.ndarray,
                          percentile: float = 95.0) -> Dict:
        """
        Calculate Hausdorff distance between two binary masks.

        Args:
            mask1: First binary mask
            mask2: Second binary mask
            percentile: Percentile for robust Hausdorff distance

        Returns:
            Dictionary with Hausdorff distance metrics
        """
        # Get contour points
        points1 = self._get_contour_points(mask1)
        points2 = self._get_contour_points(mask2)

        if len(points1) == 0 or len(points2) == 0:
            return {
                'hausdorff': 0.0,
                'hausdorff_percentile': 0.0,
                'average_hausdorff': 0.0
            }

        # Calculate distances from each point in set 1 to nearest in set 2
        distances_1to2 = []
        for p1 in points1:
            min_dist = min([euclidean(p1, p2) for p2 in points2])
            distances_1to2.append(min_dist)

        # Calculate distances from each point in set 2 to nearest in set 1
        distances_2to1 = []
        for p2 in points2:
            min_dist = min([euclidean(p2, p1) for p1 in points1])
            distances_2to1.append(min_dist)

        # Combine distances
        all_distances = distances_1to2 + distances_2to1

        # Calculate metrics
        hausdorff_metrics = {
            'hausdorff': max(all_distances),
            'hausdorff_percentile': np.percentile(all_distances, percentile),
            'average_hausdorff': np.mean(all_distances),
            'modified_hausdorff': max(np.mean(distances_1to2), np.mean(distances_2to1))
        }

        return hausdorff_metrics

    def overlay_accuracy(self,
                        image1: np.ndarray,
                        image2: np.ndarray,
                        threshold: float = 0.1) -> Dict:
        """
        Calculate overlay accuracy metrics between registered images.

        Args:
            image1: First image
            image2: Second image
            threshold: Threshold for considering pixels as matching

        Returns:
            Dictionary with overlay metrics
        """
        # Normalize images
        img1_norm = self._normalize_image(image1)
        img2_norm = self._normalize_image(image2)

        # Calculate absolute difference
        diff = np.abs(img1_norm - img2_norm)

        # Calculate metrics
        overlay_metrics = {
            'mean_absolute_error': np.mean(diff),
            'root_mean_square_error': np.sqrt(np.mean(diff ** 2)),
            'peak_signal_noise_ratio': self._calculate_psnr(img1_norm, img2_norm),
            'structural_similarity': self._calculate_ssim(img1_norm, img2_norm),
            'normalized_cross_correlation': self._calculate_ncc(img1_norm, img2_norm),
            'pixel_accuracy': np.mean(diff < threshold) * 100,
            'max_error': np.max(diff),
            'median_error': np.median(diff)
        }

        return overlay_metrics

    def sensitivity_specificity(self,
                              mask_true: np.ndarray,
                              mask_pred: np.ndarray) -> Dict:
        """
        Calculate sensitivity and specificity for binary masks.

        Args:
            mask_true: Ground truth mask
            mask_pred: Predicted mask

        Returns:
            Dictionary with sensitivity and specificity metrics
        """
        # Ensure binary masks
        mask_true = mask_true.astype(bool).flatten()
        mask_pred = mask_pred.astype(bool).flatten()

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(mask_true, mask_pred).ravel()

        # Calculate metrics
        metrics = {
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }

        return metrics

    def jacobian_determinant(self,
                            displacement_field: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """
        Calculate Jacobian determinant of deformation field.
        Used to assess whether transformation preserves topology.

        Args:
            displacement_field: Displacement field (dx, dy)

        Returns:
            Dictionary with Jacobian statistics
        """
        dx, dy = displacement_field

        # Calculate gradients
        dx_dy = np.gradient(dx, axis=0)
        dx_dx = np.gradient(dx, axis=1)
        dy_dy = np.gradient(dy, axis=0)
        dy_dx = np.gradient(dy, axis=1)

        # Jacobian matrix: J = I + gradient(displacement)
        # |J| = (1 + dx/dx) * (1 + dy/dy) - (dx/dy) * (dy/dx)
        jacobian = (1 + dx_dx) * (1 + dy_dy) - dx_dy * dy_dx

        # Calculate statistics
        jacobian_stats = {
            'mean': np.mean(jacobian),
            'std': np.std(jacobian),
            'min': np.min(jacobian),
            'max': np.max(jacobian),
            'negative_percentage': np.mean(jacobian < 0) * 100,
            'folding_detected': np.any(jacobian <= 0),
            'jacobian_map': jacobian
        }

        return jacobian_stats

    def inverse_consistency_error(self,
                                 forward_field: Tuple[np.ndarray, np.ndarray],
                                 inverse_field: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        Calculate inverse consistency error between forward and backward transformations.

        Args:
            forward_field: Forward displacement field
            inverse_field: Inverse displacement field

        Returns:
            Mean inverse consistency error
        """
        # Apply forward then inverse transformation
        h, w = forward_field[0].shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Forward transformation
        y_forward = y_coords + forward_field[1]
        x_forward = x_coords + forward_field[0]

        # Interpolate inverse field at forward positions
        from scipy.ndimage import map_coordinates
        dx_inverse_at_forward = map_coordinates(inverse_field[0], [y_forward, x_forward])
        dy_inverse_at_forward = map_coordinates(inverse_field[1], [y_forward, x_forward])

        # Apply inverse transformation
        y_final = y_forward + dy_inverse_at_forward
        x_final = x_forward + dx_inverse_at_forward

        # Calculate error (should return to original positions)
        error = np.sqrt((y_final - y_coords) ** 2 + (x_final - x_coords) ** 2)

        return np.mean(error)

    def _apply_affine_to_points(self,
                               points: np.ndarray,
                               matrix: np.ndarray) -> np.ndarray:
        """
        Apply affine transformation to point coordinates.

        Args:
            points: Points to transform (N x 2)
            matrix: Affine transformation matrix (3 x 3)

        Returns:
            Transformed points
        """
        # Convert to homogeneous coordinates
        num_points = points.shape[0]
        points_hom = np.ones((num_points, 3))
        points_hom[:, :2] = points

        # Apply transformation
        points_transformed = points_hom @ matrix.T

        # Convert back to 2D
        return points_transformed[:, :2]

    def _apply_displacement_to_points(self,
                                    points: np.ndarray,
                                    displacement_field: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Apply displacement field to point coordinates.

        Args:
            points: Points to transform (N x 2)
            displacement_field: Displacement field (dx, dy)

        Returns:
            Transformed points
        """
        from scipy.ndimage import map_coordinates

        dx, dy = displacement_field
        transformed_points = np.zeros_like(points)

        for i, (x, y) in enumerate(points):
            # Interpolate displacement at point location
            dx_at_point = map_coordinates(dx, [[y], [x]])[0]
            dy_at_point = map_coordinates(dy, [[y], [x]])[0]

            # Apply displacement
            transformed_points[i] = [x + dx_at_point, y + dy_at_point]

        return transformed_points

    def _get_contour_points(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract contour points from binary mask.

        Args:
            mask: Binary mask

        Returns:
            Array of contour points
        """
        # Find contours
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return np.array([])

        # Concatenate all contour points
        all_points = np.concatenate([c.reshape(-1, 2) for c in contours])

        return all_points

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        return image

    def _calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        from skimage.metrics import structural_similarity
        return structural_similarity(img1, img2, data_range=1.0)

    def _calculate_ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Normalized Cross-Correlation."""
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()

        # Remove means
        img1_centered = img1_flat - np.mean(img1_flat)
        img2_centered = img2_flat - np.mean(img2_flat)

        # Calculate NCC
        numerator = np.sum(img1_centered * img2_centered)
        denominator = np.sqrt(np.sum(img1_centered ** 2) * np.sum(img2_centered ** 2))

        if denominator == 0:
            return 0

        return numerator / denominator


def evaluate_registration(fixed_image: np.ndarray,
                         moving_image: np.ndarray,
                         registered_image: np.ndarray,
                         transform_matrix: Optional[np.ndarray] = None,
                         displacement_field: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                         landmarks_fixed: Optional[np.ndarray] = None,
                         landmarks_moving: Optional[np.ndarray] = None,
                         masks_fixed: Optional[np.ndarray] = None,
                         masks_moving: Optional[np.ndarray] = None) -> Dict:
    """
    Comprehensive evaluation of registration results.

    Args:
        fixed_image: Fixed/reference image
        moving_image: Original moving image
        registered_image: Registered moving image
        transform_matrix: Affine transformation matrix
        displacement_field: Deformation field
        landmarks_fixed: Landmarks in fixed image
        landmarks_moving: Landmarks in moving image
        masks_fixed: Segmentation mask for fixed image
        masks_moving: Segmentation mask for moving image

    Returns:
        Dictionary with all evaluation metrics
    """
    metrics = RegistrationMetrics()
    results = {}

    # Overlay accuracy
    results['overlay'] = metrics.overlay_accuracy(fixed_image, registered_image)

    # TRE if landmarks are provided
    if landmarks_fixed is not None and landmarks_moving is not None:
        results['tre'] = metrics.target_registration_error(
            landmarks_fixed, landmarks_moving,
            transform_matrix, displacement_field
        )

    # Segmentation overlap if masks are provided
    if masks_fixed is not None and masks_moving is not None:
        # Transform moving mask
        if transform_matrix is not None:
            from affine_registration import AffineTransform
            transform = AffineTransform()
            masks_registered = transform.apply_transform(masks_moving, transform_matrix)
        elif displacement_field is not None:
            from bspline_registration import BSplineTransform
            transform = BSplineTransform(masks_moving.shape)
            masks_registered = transform.apply_transform(masks_moving, displacement_field)
        else:
            masks_registered = masks_moving

        results['dice'] = metrics.dice_coefficient(masks_fixed, masks_registered)
        results['jaccard'] = metrics.jaccard_index(masks_fixed, masks_registered)
        results['hausdorff'] = metrics.hausdorff_distance(masks_fixed, masks_registered)
        results['sensitivity_specificity'] = metrics.sensitivity_specificity(
            masks_fixed, masks_registered
        )

    # Jacobian determinant for deformable registration
    if displacement_field is not None:
        results['jacobian'] = metrics.jacobian_determinant(displacement_field)

    return results


if __name__ == "__main__":
    # Test evaluation metrics
    np.random.seed(42)

    # Create test data
    size = (128, 128)
    fixed = np.random.randn(*size)
    moving = np.random.randn(*size)
    registered = fixed + np.random.randn(*size) * 0.1

    # Create test landmarks
    num_landmarks = 10
    landmarks_fixed = np.random.rand(num_landmarks, 2) * 128
    landmarks_moving = landmarks_fixed + np.random.randn(num_landmarks, 2) * 5

    # Create test masks
    mask_fixed = np.zeros(size)
    mask_fixed[30:80, 40:90] = 1
    mask_moving = np.zeros(size)
    mask_moving[35:85, 45:95] = 1

    # Evaluate
    results = evaluate_registration(
        fixed, moving, registered,
        landmarks_fixed=landmarks_fixed,
        landmarks_moving=landmarks_moving,
        masks_fixed=mask_fixed,
        masks_moving=mask_moving
    )

    print("Evaluation Results:")
    print(f"Overlay MAE: {results['overlay']['mean_absolute_error']:.4f}")
    print(f"Overlay PSNR: {results['overlay']['peak_signal_noise_ratio']:.2f}")
    print(f"TRE Mean: {results['tre']['mean_tre']:.2f} pixels")
    print(f"Dice: {results['dice']:.4f}")
    print(f"Jaccard: {results['jaccard']:.4f}")