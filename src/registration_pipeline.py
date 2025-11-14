"""
Main Registration Pipeline for Multimodal Brain Tumor Images
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
import pickle

from data_loader import MultimodalDataLoader
from mutual_information import MutualInformation
from affine_registration import AffineRegistration, RigidRegistration
from bspline_registration import BSplineRegistration
from evaluation_metrics import RegistrationMetrics, evaluate_registration


class RegistrationPipeline:
    """
    Complete registration pipeline combining affine and B-spline registration.
    """

    def __init__(self,
                 dataset_path: str,
                 output_dir: str = './results',
                 config: Optional[Dict] = None):
        """
        Initialize registration pipeline.

        Args:
            dataset_path: Path to dataset
            output_dir: Directory for saving results
            config: Configuration dictionary
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data loader
        self.data_loader = MultimodalDataLoader(dataset_path)

        # Default configuration
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        # Initialize components
        self._initialize_components()

        # Results storage
        self.results = []

    def _get_default_config(self) -> Dict:
        """Get default pipeline configuration."""
        return {
            # Preprocessing
            'target_size': (256, 256),
            'normalize': True,
            'histogram_matching': False,

            # Affine registration
            'affine': {
                'enabled': True,
                'type': 'affine',  # 'rigid' or 'affine'
                'metric': 'nmi',
                'optimizer': 'lbfgs',
                'multi_resolution': True,
                'num_levels': 3,
                'num_bins': 32
            },

            # B-spline registration
            'bspline': {
                'enabled': True,
                'grid_spacing': (32, 32),
                'metric': 'nmi',
                'optimizer': 'lbfgs',
                'regularization': 0.01,
                'multi_resolution': True,
                'num_levels': 3,
                'num_bins': 32
            },

            # Evaluation
            'evaluation': {
                'compute_tre': True,
                'compute_dice': True,
                'compute_overlay': True,
                'compute_jacobian': True
            },

            # Output
            'save_images': True,
            'save_metrics': True,
            'save_transforms': True
        }

    def _initialize_components(self):
        """Initialize registration components based on configuration."""
        # Affine registration
        if self.config['affine']['enabled']:
            affine_config = self.config['affine']
            if affine_config['type'] == 'rigid':
                self.affine_registration = RigidRegistration(
                    metric=affine_config['metric'],
                    optimizer=affine_config['optimizer'],
                    num_bins=affine_config['num_bins'],
                    multi_resolution=affine_config['multi_resolution'],
                    num_levels=affine_config['num_levels']
                )
            else:
                self.affine_registration = AffineRegistration(
                    metric=affine_config['metric'],
                    optimizer=affine_config['optimizer'],
                    num_bins=affine_config['num_bins'],
                    multi_resolution=affine_config['multi_resolution'],
                    num_levels=affine_config['num_levels']
                )

        # B-spline registration
        if self.config['bspline']['enabled']:
            bspline_config = self.config['bspline']
            self.bspline_registration = BSplineRegistration(
                grid_spacing=tuple(bspline_config['grid_spacing']),
                metric=bspline_config['metric'],
                optimizer=bspline_config['optimizer'],
                num_bins=bspline_config['num_bins'],
                regularization=bspline_config['regularization'],
                multi_resolution=bspline_config['multi_resolution'],
                num_levels=bspline_config['num_levels']
            )

        # Metrics calculator
        self.metrics_calculator = RegistrationMetrics()

    def register_pair(self,
                     fixed_image: np.ndarray,
                     moving_image: np.ndarray,
                     fixed_name: str = 'fixed',
                     moving_name: str = 'moving',
                     landmarks_fixed: Optional[np.ndarray] = None,
                     landmarks_moving: Optional[np.ndarray] = None,
                     mask_fixed: Optional[np.ndarray] = None,
                     mask_moving: Optional[np.ndarray] = None) -> Dict:
        """
        Register a single image pair.

        Args:
            fixed_image: Reference image
            moving_image: Image to register
            fixed_name: Name for fixed image
            moving_name: Name for moving image
            landmarks_fixed: Optional landmarks in fixed image
            landmarks_moving: Optional landmarks in moving image
            mask_fixed: Optional mask for fixed image
            mask_moving: Optional mask for moving image

        Returns:
            Registration results dictionary
        """
        print(f"\nRegistering {moving_name} to {fixed_name}")
        start_time = time.time()

        # Preprocess images
        fixed_processed = self.data_loader.preprocess_image(
            fixed_image,
            target_size=tuple(self.config['target_size']),
            normalize=self.config['normalize']
        )
        moving_processed = self.data_loader.preprocess_image(
            moving_image,
            target_size=tuple(self.config['target_size']),
            normalize=self.config['normalize'],
            histogram_matching=self.config['histogram_matching'],
            reference_image=fixed_processed if self.config['histogram_matching'] else None
        )

        results = {
            'fixed_name': fixed_name,
            'moving_name': moving_name,
            'fixed_image': fixed_processed,
            'moving_image': moving_processed,
            'stages': {}
        }

        current_moving = moving_processed.copy()
        cumulative_transform = None
        cumulative_displacement = None

        # Stage 1: Affine registration
        if self.config['affine']['enabled']:
            print("  Stage 1: Affine registration...")
            affine_result = self.affine_registration.register(
                fixed_processed,
                current_moving,
                mask=mask_fixed
            )

            results['stages']['affine'] = {
                'params': affine_result['params'],
                'matrix': affine_result['matrix'],
                'metric': affine_result['metric'],
                'registered_image': affine_result['registered_image']
            }

            current_moving = affine_result['registered_image']
            cumulative_transform = affine_result['matrix']

            print(f"    Affine metric: {affine_result['metric']:.4f}")

        # Stage 2: B-spline registration
        if self.config['bspline']['enabled']:
            print("  Stage 2: B-spline registration...")

            # Use affine result as initialization
            affine_params = results['stages']['affine']['params'] if 'affine' in results['stages'] else None

            bspline_result = self.bspline_registration.register(
                fixed_processed,
                moving_processed,  # Use original for B-spline
                mask=mask_fixed,
                affine_params=affine_params
            )

            results['stages']['bspline'] = {
                'displacement_field': bspline_result['displacement_field'],
                'grid_spacing': bspline_result['grid_spacing'],
                'metric': bspline_result['metric'],
                'registered_image': bspline_result['registered_image']
            }

            current_moving = bspline_result['registered_image']
            cumulative_displacement = bspline_result['displacement_field']

            print(f"    B-spline metric: {bspline_result['metric']:.4f}")

        # Final registered image
        results['final_registered'] = current_moving

        # Evaluation
        if self.config['evaluation']['compute_overlay']:
            print("  Computing overlay metrics...")
            results['metrics'] = evaluate_registration(
                fixed_processed,
                moving_processed,
                current_moving,
                transform_matrix=cumulative_transform,
                displacement_field=cumulative_displacement,
                landmarks_fixed=landmarks_fixed,
                landmarks_moving=landmarks_moving,
                masks_fixed=mask_fixed,
                masks_moving=mask_moving
            )

            # Print key metrics
            if 'overlay' in results['metrics']:
                print(f"    MAE: {results['metrics']['overlay']['mean_absolute_error']:.4f}")
                print(f"    PSNR: {results['metrics']['overlay']['peak_signal_noise_ratio']:.2f} dB")
                print(f"    SSIM: {results['metrics']['overlay']['structural_similarity']:.4f}")

            if 'tre' in results['metrics']:
                print(f"    Mean TRE: {results['metrics']['tre']['mean_tre']:.2f} pixels")
                print(f"    Success rate (<5mm): {results['metrics']['tre']['success_rate_5mm']:.1f}%")

        # Timing
        elapsed_time = time.time() - start_time
        results['elapsed_time'] = elapsed_time
        print(f"  Registration completed in {elapsed_time:.2f} seconds")

        return results

    def register_batch(self,
                      modality1: str = 'ct',
                      modality2: str = 'mri',
                      category: str = 'Healthy',
                      num_pairs: int = 10) -> List[Dict]:
        """
        Register a batch of image pairs.

        Args:
            modality1: First modality
            modality2: Second modality
            category: Image category
            num_pairs: Number of pairs to register

        Returns:
            List of registration results
        """
        print(f"\nBatch registration: {modality1.upper()} to {modality2.upper()} ({category})")
        print(f"Number of pairs: {num_pairs}")

        # Load image pairs
        pairs = self.data_loader.load_image_pairs(
            modality1, modality2, category, num_pairs
        )

        batch_results = []

        for i, (img1, img2) in enumerate(pairs):
            # Register pair
            result = self.register_pair(
                img1, img2,
                f"{modality1}_{i:03d}",
                f"{modality2}_{i:03d}"
            )

            batch_results.append(result)

            # Save intermediate results
            if self.config['save_images']:
                self._save_registration_images(result, i)

        # Compute batch statistics
        batch_stats = self._compute_batch_statistics(batch_results)

        # Save batch results
        if self.config['save_metrics']:
            self._save_batch_metrics(batch_results, batch_stats)

        print(f"\nBatch registration completed")
        self._print_batch_summary(batch_stats)

        return batch_results

    def register_with_synthetic_deformation(self,
                                          image: np.ndarray,
                                          max_rotation: float = 15,
                                          max_translation: float = 20,
                                          elastic_alpha: float = 20,
                                          elastic_sigma: float = 3) -> Dict:
        """
        Test registration with synthetic deformation.

        Args:
            image: Input image
            max_rotation: Maximum rotation angle
            max_translation: Maximum translation
            elastic_alpha: Elastic deformation strength
            elastic_sigma: Elastic deformation smoothness

        Returns:
            Registration results with known ground truth
        """
        print("\nTesting with synthetic deformation...")

        # Create synthetic deformation
        deformed, true_params = self.data_loader.create_synthetic_deformation(
            image,
            max_rotation=max_rotation,
            max_translation=max_translation,
            elastic_alpha=elastic_alpha,
            elastic_sigma=elastic_sigma
        )

        # Register
        result = self.register_pair(
            image, deformed,
            "original", "deformed"
        )

        # Add ground truth
        result['ground_truth'] = true_params

        # Compute registration error
        if 'affine' in result['stages']:
            estimated_params = result['stages']['affine']['params']
            param_error = {
                'translation_error': np.sqrt(
                    (estimated_params[0] - true_params['translation'][0])**2 +
                    (estimated_params[1] - true_params['translation'][1])**2
                ),
                'rotation_error': abs(estimated_params[2] - true_params['rotation']),
                'scale_error': abs(estimated_params[3] - true_params['scale'])
            }
            result['parameter_error'] = param_error

            print(f"\nParameter errors:")
            print(f"  Translation: {param_error['translation_error']:.2f} pixels")
            print(f"  Rotation: {param_error['rotation_error']:.2f} degrees")
            print(f"  Scale: {param_error['scale_error']:.4f}")

        return result

    def _compute_batch_statistics(self, batch_results: List[Dict]) -> Dict:
        """Compute statistics over batch of registrations."""
        stats = {
            'num_registrations': len(batch_results),
            'metrics': {},
            'timing': {}
        }

        # Collect metrics
        metric_values = {}
        for result in batch_results:
            if 'metrics' in result and 'overlay' in result['metrics']:
                for key, value in result['metrics']['overlay'].items():
                    if key not in metric_values:
                        metric_values[key] = []
                    if isinstance(value, (int, float)):
                        metric_values[key].append(value)

        # Compute statistics
        for key, values in metric_values.items():
            if values:
                stats['metrics'][key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        # Timing statistics
        times = [r['elapsed_time'] for r in batch_results if 'elapsed_time' in r]
        if times:
            stats['timing'] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'total': np.sum(times)
            }

        return stats

    def _save_registration_images(self, result: Dict, index: int):
        """Save registration images."""
        import matplotlib.pyplot as plt

        output_path = self.output_dir / 'images' / f'registration_{index:03d}'
        output_path.mkdir(parents=True, exist_ok=True)

        # Save individual images
        images = {
            'fixed': result['fixed_image'],
            'moving': result['moving_image'],
            'registered': result['final_registered']
        }

        for name, img in images.items():
            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap='gray')
            plt.title(name.capitalize())
            plt.axis('off')
            plt.savefig(output_path / f'{name}.png', dpi=100, bbox_inches='tight')
            plt.close()

    def _save_batch_metrics(self, batch_results: List[Dict], batch_stats: Dict):
        """Save batch metrics to file."""
        metrics_path = self.output_dir / 'metrics.json'

        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        # Extract metrics only
        metrics_data = {
            'batch_statistics': convert_to_serializable(batch_stats),
            'individual_results': []
        }

        for result in batch_results:
            individual = {
                'fixed': result['fixed_name'],
                'moving': result['moving_name'],
                'time': result.get('elapsed_time', None)
            }

            if 'metrics' in result:
                individual['metrics'] = convert_to_serializable(result['metrics'])

            metrics_data['individual_results'].append(individual)

        # Save to JSON
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        print(f"Metrics saved to {metrics_path}")

    def _print_batch_summary(self, batch_stats: Dict):
        """Print summary of batch registration results."""
        print("\n" + "="*50)
        print("BATCH REGISTRATION SUMMARY")
        print("="*50)

        print(f"\nNumber of registrations: {batch_stats['num_registrations']}")

        if batch_stats['metrics']:
            print("\nMetrics (mean ± std):")
            for metric, values in batch_stats['metrics'].items():
                if 'mean' in values:
                    print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")

        if batch_stats['timing']:
            print(f"\nTiming:")
            print(f"  Mean time: {batch_stats['timing']['mean']:.2f} seconds")
            print(f"  Total time: {batch_stats['timing']['total']:.2f} seconds")

    def save_configuration(self, filepath: Optional[str] = None):
        """Save pipeline configuration."""
        if filepath is None:
            filepath = self.output_dir / 'config.json'

        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"Configuration saved to {filepath}")

    def load_configuration(self, filepath: str):
        """Load pipeline configuration."""
        with open(filepath, 'r') as f:
            self.config = json.load(f)

        self._initialize_components()
        print(f"Configuration loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    pipeline = RegistrationPipeline(
        dataset_path="./Dataset",
        output_dir="./registration_results"
    )

    # Save configuration
    pipeline.save_configuration()

    # Test with synthetic deformation
    test_image = np.random.randn(256, 256)
    synthetic_result = pipeline.register_with_synthetic_deformation(test_image)

    # Register a batch of real images
    # batch_results = pipeline.register_batch(
    #     modality1='ct',
    #     modality2='mri',
    #     category='Healthy',
    #     num_pairs=5
    # )

    print("\nPipeline test completed successfully!")