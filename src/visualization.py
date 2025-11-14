"""
Visualization Utilities for Medical Image Registration
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
from typing import Tuple, Dict, Optional, List
import cv2


class RegistrationVisualizer:
    """
    Visualization tools for registration results.
    """

    def __init__(self, figsize: Tuple[int, int] = (15, 10), dpi: int = 100):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colormap = 'gray'

    def plot_registration_results(self,
                                 fixed_image: np.ndarray,
                                 moving_image: np.ndarray,
                                 registered_image: np.ndarray,
                                 title: str = "Registration Results",
                                 save_path: Optional[str] = None) -> None:
        """
        Plot registration results in a grid.

        Args:
            fixed_image: Fixed/reference image
            moving_image: Original moving image
            registered_image: Registered moving image
            title: Figure title
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)

        # Fixed image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(fixed_image, cmap=self.colormap)
        ax1.set_title('Fixed Image')
        ax1.axis('off')

        # Moving image
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(moving_image, cmap=self.colormap)
        ax2.set_title('Moving Image')
        ax2.axis('off')

        # Registered image
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(registered_image, cmap=self.colormap)
        ax3.set_title('Registered Image')
        ax3.axis('off')

        # Checkerboard
        ax4 = fig.add_subplot(gs[1, 0])
        checkerboard = self._create_checkerboard(fixed_image, registered_image)
        ax4.imshow(checkerboard, cmap=self.colormap)
        ax4.set_title('Checkerboard')
        ax4.axis('off')

        # Difference before
        ax5 = fig.add_subplot(gs[1, 1])
        diff_before = self._compute_difference_image(fixed_image, moving_image)
        im5 = ax5.imshow(diff_before, cmap='RdBu_r', vmin=-1, vmax=1)
        ax5.set_title('Difference Before')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046)

        # Difference after
        ax6 = fig.add_subplot(gs[1, 2])
        diff_after = self._compute_difference_image(fixed_image, registered_image)
        im6 = ax6.imshow(diff_after, cmap='RdBu_r', vmin=-1, vmax=1)
        ax6.set_title('Difference After')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046)

        fig.suptitle(title, fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_overlay(self,
                    fixed_image: np.ndarray,
                    moving_image: np.ndarray,
                    registered_image: np.ndarray,
                    alpha: float = 0.5,
                    save_path: Optional[str] = None) -> None:
        """
        Plot overlay visualization of registration.

        Args:
            fixed_image: Fixed image
            moving_image: Moving image
            registered_image: Registered image
            alpha: Transparency for overlay
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Before registration overlay
        axes[0].imshow(fixed_image, cmap='gray', alpha=1.0)
        axes[0].imshow(moving_image, cmap='hot', alpha=alpha)
        axes[0].set_title('Before Registration')
        axes[0].axis('off')

        # After registration overlay
        axes[1].imshow(fixed_image, cmap='gray', alpha=1.0)
        axes[1].imshow(registered_image, cmap='hot', alpha=alpha)
        axes[1].set_title('After Registration')
        axes[1].axis('off')

        # RGB composite
        rgb_composite = self._create_rgb_composite(fixed_image, registered_image)
        axes[2].imshow(rgb_composite)
        axes[2].set_title('RGB Composite (R=Fixed, G=Registered)')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_deformation_field(self,
                             displacement_field: Tuple[np.ndarray, np.ndarray],
                             image: Optional[np.ndarray] = None,
                             spacing: int = 10,
                             scale: float = 1.0,
                             save_path: Optional[str] = None) -> None:
        """
        Visualize deformation field.

        Args:
            displacement_field: Displacement field (dx, dy)
            image: Optional background image
            spacing: Spacing between arrows
            scale: Scale factor for arrows
            save_path: Path to save figure
        """
        dx, dy = displacement_field
        h, w = dx.shape

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Magnitude plot
        magnitude = np.sqrt(dx**2 + dy**2)
        im1 = axes[0].imshow(magnitude, cmap='jet')
        axes[0].set_title('Deformation Magnitude')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # Vector field
        if image is not None:
            axes[1].imshow(image, cmap='gray', alpha=0.5)

        # Create grid for arrows
        y, x = np.mgrid[0:h:spacing, 0:w:spacing]

        # Get displacement at grid points
        dx_sparse = dx[::spacing, ::spacing]
        dy_sparse = dy[::spacing, ::spacing]

        # Plot arrows
        axes[1].quiver(x, y, dx_sparse * scale, -dy_sparse * scale,
                      angles='xy', scale_units='xy', scale=1,
                      color='red', width=0.002, headwidth=3, headlength=4)

        axes[1].set_title('Deformation Field')
        axes[1].axis('off')
        axes[1].set_xlim(0, w)
        axes[1].set_ylim(h, 0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_jacobian_determinant(self,
                                 jacobian_map: np.ndarray,
                                 save_path: Optional[str] = None) -> None:
        """
        Visualize Jacobian determinant map.

        Args:
            jacobian_map: Jacobian determinant values
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Full Jacobian map
        im1 = axes[0].imshow(jacobian_map, cmap='RdBu_r', vmin=0, vmax=2)
        axes[0].set_title('Jacobian Determinant')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # Folding regions (Jacobian <= 0)
        folding_mask = jacobian_map <= 0
        axes[1].imshow(folding_mask, cmap='hot')
        axes[1].set_title(f'Folding Regions ({np.sum(folding_mask)} pixels)')
        axes[1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_landmarks(self,
                      image: np.ndarray,
                      landmarks_before: np.ndarray,
                      landmarks_after: np.ndarray,
                      save_path: Optional[str] = None) -> None:
        """
        Visualize landmark correspondences.

        Args:
            image: Background image
            landmarks_before: Original landmark positions
            landmarks_after: Transformed landmark positions
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Display image
        ax.imshow(image, cmap='gray')

        # Plot landmarks and connections
        for i, (before, after) in enumerate(zip(landmarks_before, landmarks_after)):
            # Original position
            ax.scatter(before[0], before[1], c='red', s=50, marker='o',
                      label='Original' if i == 0 else '')

            # Transformed position
            ax.scatter(after[0], after[1], c='green', s=50, marker='x',
                      label='Transformed' if i == 0 else '')

            # Draw arrow
            arrow = FancyArrowPatch(
                before, after,
                connectionstyle="arc3,rad=0.2",
                arrowstyle='->,head_width=0.4,head_length=0.8',
                color='yellow', alpha=0.6, linewidth=1.5
            )
            ax.add_patch(arrow)

            # Add label
            ax.text(before[0], before[1] - 5, str(i), color='white',
                   fontsize=8, ha='center')

        ax.set_title('Landmark Correspondences')
        ax.legend()
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_metrics_evolution(self,
                              metrics_history: List[Dict],
                              save_path: Optional[str] = None) -> None:
        """
        Plot evolution of metrics during registration.

        Args:
            metrics_history: List of metric dictionaries over iterations
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Extract metrics
        iterations = range(len(metrics_history))
        mi_values = [m.get('mutual_information', 0) for m in metrics_history]
        mae_values = [m.get('mae', 0) for m in metrics_history]
        psnr_values = [m.get('psnr', 0) for m in metrics_history]
        ssim_values = [m.get('ssim', 0) for m in metrics_history]

        # Plot MI
        axes[0, 0].plot(iterations, mi_values, 'b-', marker='o')
        axes[0, 0].set_title('Mutual Information')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('MI')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot MAE
        axes[0, 1].plot(iterations, mae_values, 'r-', marker='s')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot PSNR
        axes[1, 0].plot(iterations, psnr_values, 'g-', marker='^')
        axes[1, 0].set_title('Peak Signal-to-Noise Ratio')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot SSIM
        axes[1, 1].plot(iterations, ssim_values, 'm-', marker='d')
        axes[1, 1].set_title('Structural Similarity')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Registration Metrics Evolution', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def create_animation(self,
                        fixed_image: np.ndarray,
                        moving_images: List[np.ndarray],
                        output_path: str,
                        fps: int = 10) -> None:
        """
        Create animation of registration process.

        Args:
            fixed_image: Fixed image
            moving_images: List of images during registration
            output_path: Path for output video
            fps: Frames per second
        """
        h, w = fixed_image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

        for moving in moving_images:
            # Create side-by-side view
            combined = np.hstack([fixed_image, moving])

            # Convert to BGR for OpenCV
            if len(combined.shape) == 2:
                combined = cv2.cvtColor(
                    (combined * 255).astype(np.uint8),
                    cv2.COLOR_GRAY2BGR
                )

            out.write(combined)

        out.release()
        print(f"Animation saved to {output_path}")

    def _create_checkerboard(self,
                            image1: np.ndarray,
                            image2: np.ndarray,
                            n_tiles: int = 8) -> np.ndarray:
        """Create checkerboard visualization."""
        h, w = image1.shape[:2]
        tile_h = h // n_tiles
        tile_w = w // n_tiles

        checkerboard = np.zeros_like(image1)

        for i in range(n_tiles):
            for j in range(n_tiles):
                y1 = i * tile_h
                y2 = min((i + 1) * tile_h, h)
                x1 = j * tile_w
                x2 = min((j + 1) * tile_w, w)

                if (i + j) % 2 == 0:
                    checkerboard[y1:y2, x1:x2] = image1[y1:y2, x1:x2]
                else:
                    checkerboard[y1:y2, x1:x2] = image2[y1:y2, x1:x2]

        return checkerboard

    def _compute_difference_image(self,
                                 image1: np.ndarray,
                                 image2: np.ndarray) -> np.ndarray:
        """Compute normalized difference image."""
        # Normalize images
        img1_norm = (image1 - image1.min()) / (image1.max() - image1.min() + 1e-8)
        img2_norm = (image2 - image2.min()) / (image2.max() - image2.min() + 1e-8)

        return img1_norm - img2_norm

    def _create_rgb_composite(self,
                             image1: np.ndarray,
                             image2: np.ndarray) -> np.ndarray:
        """Create RGB composite image."""
        # Normalize images
        img1_norm = (image1 - image1.min()) / (image1.max() - image1.min() + 1e-8)
        img2_norm = (image2 - image2.min()) / (image2.max() - image2.min() + 1e-8)

        # Create RGB image
        rgb = np.zeros((*image1.shape, 3))
        rgb[:, :, 0] = img1_norm  # Red channel
        rgb[:, :, 1] = img2_norm  # Green channel
        rgb[:, :, 2] = (img1_norm + img2_norm) / 2  # Blue channel

        return rgb


def create_comprehensive_report(results: Dict,
                               output_path: str,
                               visualizer: Optional[RegistrationVisualizer] = None) -> None:
    """
    Create comprehensive visualization report.

    Args:
        results: Registration results dictionary
        output_path: Path for output report
        visualizer: RegistrationVisualizer instance
    """
    if visualizer is None:
        visualizer = RegistrationVisualizer()

    # Create output directory
    from pathlib import Path
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main registration results
    visualizer.plot_registration_results(
        results['fixed_image'],
        results['moving_image'],
        results['final_registered'],
        title="Registration Results",
        save_path=output_dir / "registration_results.png"
    )

    # Overlay visualization
    visualizer.plot_overlay(
        results['fixed_image'],
        results['moving_image'],
        results['final_registered'],
        save_path=output_dir / "overlay.png"
    )

    # Deformation field if available
    if 'bspline' in results.get('stages', {}) and \
       'displacement_field' in results['stages']['bspline']:
        visualizer.plot_deformation_field(
            results['stages']['bspline']['displacement_field'],
            image=results['fixed_image'],
            save_path=output_dir / "deformation_field.png"
        )

    # Jacobian if available
    if 'jacobian' in results.get('metrics', {}):
        visualizer.plot_jacobian_determinant(
            results['metrics']['jacobian']['jacobian_map'],
            save_path=output_dir / "jacobian.png"
        )

    print(f"Visualization report saved to {output_path}")


if __name__ == "__main__":
    # Test visualization
    np.random.seed(42)

    # Create test images
    size = (256, 256)
    fixed = np.random.randn(*size)
    moving = np.random.randn(*size)
    registered = fixed + np.random.randn(*size) * 0.1

    # Create visualizer
    viz = RegistrationVisualizer()

    # Test plots
    viz.plot_registration_results(fixed, moving, registered)

    # Test deformation field
    dx = np.random.randn(*size) * 5
    dy = np.random.randn(*size) * 5
    viz.plot_deformation_field((dx, dy), image=fixed)

    print("Visualization tests completed")