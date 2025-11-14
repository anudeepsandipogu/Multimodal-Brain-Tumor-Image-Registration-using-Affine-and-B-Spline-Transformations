"""
Mutual Information Metric for Image Registration
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple


class MutualInformation:
    """
    Implementation of Mutual Information and Normalized Mutual Information
    for image registration.
    """

    def __init__(self, num_bins: int = 32, sigma: float = 1.0):
        """
        Initialize Mutual Information calculator.

        Args:
            num_bins: Number of histogram bins
            sigma: Gaussian smoothing parameter for joint histogram
        """
        self.num_bins = num_bins
        self.sigma = sigma

    def compute_mi(self,
                   image1: np.ndarray,
                   image2: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Mutual Information between two images.

        MI(A,B) = H(A) + H(B) - H(A,B)

        Args:
            image1: First image
            image2: Second image
            mask: Optional mask for region of interest

        Returns:
            Mutual information value
        """
        # Ensure images are the same size
        assert image1.shape == image2.shape, "Images must have the same shape"

        # Flatten and apply mask if provided
        if mask is not None:
            image1_flat = image1[mask > 0]
            image2_flat = image2[mask > 0]
        else:
            image1_flat = image1.flatten()
            image2_flat = image2.flatten()

        # Compute joint histogram
        joint_hist = self._compute_joint_histogram(image1_flat, image2_flat)

        # Compute marginal histograms
        hist1 = np.sum(joint_hist, axis=1)
        hist2 = np.sum(joint_hist, axis=0)

        # Compute entropies
        h1 = self._entropy(hist1)
        h2 = self._entropy(hist2)
        h12 = self._joint_entropy(joint_hist)

        # Compute mutual information
        mi = h1 + h2 - h12

        return mi

    def compute_nmi(self,
                    image1: np.ndarray,
                    image2: np.ndarray,
                    mask: Optional[np.ndarray] = None,
                    method: str = 'studholme') -> float:
        """
        Compute Normalized Mutual Information.

        Args:
            image1: First image
            image2: Second image
            mask: Optional mask for region of interest
            method: Normalization method ('studholme' or 'maes')

        Returns:
            Normalized mutual information value
        """
        # Ensure images are the same size
        assert image1.shape == image2.shape, "Images must have the same shape"

        # Flatten and apply mask if provided
        if mask is not None:
            image1_flat = image1[mask > 0]
            image2_flat = image2[mask > 0]
        else:
            image1_flat = image1.flatten()
            image2_flat = image2.flatten()

        # Compute joint histogram
        joint_hist = self._compute_joint_histogram(image1_flat, image2_flat)

        # Compute marginal histograms
        hist1 = np.sum(joint_hist, axis=1)
        hist2 = np.sum(joint_hist, axis=0)

        # Compute entropies
        h1 = self._entropy(hist1)
        h2 = self._entropy(hist2)
        h12 = self._joint_entropy(joint_hist)

        # Compute normalized mutual information
        if method == 'studholme':
            # NMI = (H(A) + H(B)) / H(A,B)
            nmi = (h1 + h2) / (h12 + 1e-10)
        elif method == 'maes':
            # NMI = 2 * MI / (H(A) + H(B))
            mi = h1 + h2 - h12
            nmi = 2.0 * mi / (h1 + h2 + 1e-10)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return nmi

    def compute_ecc(self,
                    image1: np.ndarray,
                    image2: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Entropy Correlation Coefficient.

        ECC = 2 * MI / (H(A) + H(B))

        Args:
            image1: First image
            image2: Second image
            mask: Optional mask for region of interest

        Returns:
            Entropy correlation coefficient
        """
        return self.compute_nmi(image1, image2, mask, method='maes')

    def _compute_joint_histogram(self,
                                image1: np.ndarray,
                                image2: np.ndarray) -> np.ndarray:
        """
        Compute joint histogram of two images.

        Args:
            image1: First image (flattened)
            image2: Second image (flattened)

        Returns:
            Joint histogram
        """
        # Normalize images to [0, num_bins-1]
        min1, max1 = image1.min(), image1.max()
        min2, max2 = image2.min(), image2.max()

        if max1 > min1:
            image1_norm = (image1 - min1) / (max1 - min1) * (self.num_bins - 1)
        else:
            image1_norm = np.zeros_like(image1)

        if max2 > min2:
            image2_norm = (image2 - min2) / (max2 - min2) * (self.num_bins - 1)
        else:
            image2_norm = np.zeros_like(image2)

        # Compute 2D histogram
        joint_hist, _, _ = np.histogram2d(
            image1_norm,
            image2_norm,
            bins=self.num_bins,
            range=[[0, self.num_bins-1], [0, self.num_bins-1]]
        )

        # Apply Gaussian smoothing to reduce noise
        if self.sigma > 0:
            joint_hist = gaussian_filter(joint_hist, self.sigma)

        # Normalize to probability distribution
        joint_hist = joint_hist / (np.sum(joint_hist) + 1e-10)

        return joint_hist

    def _entropy(self, histogram: np.ndarray) -> float:
        """
        Compute Shannon entropy of a histogram.

        H = -sum(p * log(p))

        Args:
            histogram: Probability distribution

        Returns:
            Entropy value
        """
        # Remove zeros to avoid log(0)
        histogram = histogram[histogram > 0]

        # Compute entropy
        entropy = -np.sum(histogram * np.log(histogram + 1e-10))

        return entropy

    def _joint_entropy(self, joint_histogram: np.ndarray) -> float:
        """
        Compute joint entropy from joint histogram.

        Args:
            joint_histogram: Joint probability distribution

        Returns:
            Joint entropy value
        """
        # Flatten and remove zeros
        joint_hist_flat = joint_histogram.flatten()
        joint_hist_flat = joint_hist_flat[joint_hist_flat > 0]

        # Compute joint entropy
        joint_entropy = -np.sum(joint_hist_flat * np.log(joint_hist_flat + 1e-10))

        return joint_entropy

    def gradient(self,
                 image1: np.ndarray,
                 image2: np.ndarray,
                 transform_params: np.ndarray,
                 transform_func,
                 epsilon: float = 1e-4) -> np.ndarray:
        """
        Compute gradient of mutual information with respect to transformation parameters.

        Args:
            image1: Fixed image
            image2: Moving image
            transform_params: Current transformation parameters
            transform_func: Function that applies transformation
            epsilon: Small value for numerical differentiation

        Returns:
            Gradient vector
        """
        grad = np.zeros_like(transform_params)
        base_mi = self.compute_mi(image1, transform_func(image2, transform_params))

        # Compute gradient using finite differences
        for i in range(len(transform_params)):
            params_plus = transform_params.copy()
            params_plus[i] += epsilon

            mi_plus = self.compute_mi(image1, transform_func(image2, params_plus))
            grad[i] = (mi_plus - base_mi) / epsilon

        return grad


class LocalMutualInformation:
    """
    Local Mutual Information for capturing local image relationships.
    """

    def __init__(self,
                 window_size: int = 7,
                 num_bins: int = 32,
                 sigma: float = 1.0):
        """
        Initialize Local Mutual Information calculator.

        Args:
            window_size: Size of local window
            num_bins: Number of histogram bins
            sigma: Gaussian smoothing parameter
        """
        self.window_size = window_size
        self.mi_calculator = MutualInformation(num_bins, sigma)

    def compute_lmi_map(self,
                       image1: np.ndarray,
                       image2: np.ndarray) -> np.ndarray:
        """
        Compute Local Mutual Information map.

        Args:
            image1: First image
            image2: Second image

        Returns:
            Local MI map
        """
        h, w = image1.shape
        half_window = self.window_size // 2
        lmi_map = np.zeros_like(image1)

        # Pad images
        image1_padded = np.pad(image1, half_window, mode='reflect')
        image2_padded = np.pad(image2, half_window, mode='reflect')

        # Compute local MI for each pixel
        for i in range(h):
            for j in range(w):
                # Extract local windows
                window1 = image1_padded[i:i+self.window_size, j:j+self.window_size]
                window2 = image2_padded[i:i+self.window_size, j:j+self.window_size]

                # Compute local MI
                lmi_map[i, j] = self.mi_calculator.compute_mi(window1, window2)

        return lmi_map

    def compute_weighted_mi(self,
                           image1: np.ndarray,
                           image2: np.ndarray,
                           weight_map: Optional[np.ndarray] = None) -> float:
        """
        Compute weighted mutual information using local MI map.

        Args:
            image1: First image
            image2: Second image
            weight_map: Optional weight map for importance weighting

        Returns:
            Weighted mutual information value
        """
        # Compute local MI map
        lmi_map = self.compute_lmi_map(image1, image2)

        # Apply weights if provided
        if weight_map is not None:
            lmi_map = lmi_map * weight_map

        # Return weighted average
        return np.mean(lmi_map)


def test_mutual_information():
    """Test mutual information implementations."""
    # Create test images
    np.random.seed(42)
    size = (128, 128)

    # Similar images (high MI)
    image1 = np.random.randn(*size)
    image2 = image1 + np.random.randn(*size) * 0.1

    # Different images (low MI)
    image3 = np.random.randn(*size)

    # Create MI calculator
    mi_calc = MutualInformation(num_bins=32)

    # Compute MI values
    mi_similar = mi_calc.compute_mi(image1, image2)
    mi_different = mi_calc.compute_mi(image1, image3)
    mi_self = mi_calc.compute_mi(image1, image1)

    print(f"MI (similar images): {mi_similar:.4f}")
    print(f"MI (different images): {mi_different:.4f}")
    print(f"MI (self): {mi_self:.4f}")

    # Compute NMI values
    nmi_similar = mi_calc.compute_nmi(image1, image2)
    nmi_different = mi_calc.compute_nmi(image1, image3)
    nmi_self = mi_calc.compute_nmi(image1, image1)

    print(f"\nNMI (similar images): {nmi_similar:.4f}")
    print(f"NMI (different images): {nmi_different:.4f}")
    print(f"NMI (self): {nmi_self:.4f}")

    # Test local MI
    lmi_calc = LocalMutualInformation(window_size=7)
    lmi_map = lmi_calc.compute_lmi_map(image1, image2)
    print(f"\nLocal MI map shape: {lmi_map.shape}")
    print(f"Local MI mean: {np.mean(lmi_map):.4f}")


if __name__ == "__main__":
    test_mutual_information()