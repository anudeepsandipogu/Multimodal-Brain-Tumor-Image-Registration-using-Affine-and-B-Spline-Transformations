"""
Main script for Multimodal Brain Tumor Image Registration
Implements affine + B-spline deformable registration with mutual information optimization
"""

import sys
sys.path.append('./src')

import numpy as np
import argparse
from pathlib import Path
import json
import time

from registration_pipeline import RegistrationPipeline
from visualization import RegistrationVisualizer, create_comprehensive_report
from data_loader import MultimodalDataLoader
from evaluation_metrics import RegistrationMetrics


def run_single_pair_registration(args):
    """Run registration on a single image pair."""
    print("="*60)
    print("SINGLE PAIR REGISTRATION")
    print("="*60)

    # Initialize pipeline
    config = {
        'target_size': (args.image_size, args.image_size),
        'normalize': True,
        'histogram_matching': args.histogram_matching,
        'affine': {
            'enabled': args.use_affine,
            'type': args.affine_type,
            'metric': args.metric,
            'optimizer': args.optimizer,
            'multi_resolution': args.multi_resolution,
            'num_levels': args.num_levels,
            'num_bins': args.num_bins
        },
        'bspline': {
            'enabled': args.use_bspline,
            'grid_spacing': (args.grid_spacing, args.grid_spacing),
            'metric': args.metric,
            'optimizer': args.optimizer,
            'regularization': args.regularization,
            'multi_resolution': args.multi_resolution,
            'num_levels': args.num_levels,
            'num_bins': args.num_bins
        },
        'evaluation': {
            'compute_tre': True,
            'compute_dice': True,
            'compute_overlay': True,
            'compute_jacobian': True
        },
        'save_images': True,
        'save_metrics': True,
        'save_transforms': True
    }

    pipeline = RegistrationPipeline(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        config=config
    )

    # Load images
    loader = MultimodalDataLoader(args.dataset_path)

    # Load a single pair
    pairs = loader.load_image_pairs(
        args.modality1,
        args.modality2,
        args.category,
        num_pairs=1
    )

    if not pairs:
        print("No image pairs found!")
        return

    fixed, moving = pairs[0]

    # Run registration
    result = pipeline.register_pair(
        fixed, moving,
        f"{args.modality1}_image",
        f"{args.modality2}_image"
    )

    # Create visualizations
    visualizer = RegistrationVisualizer()
    output_path = Path(args.output_dir) / 'single_registration'
    create_comprehensive_report(result, str(output_path), visualizer)

    # Print results
    print("\n" + "="*60)
    print("REGISTRATION RESULTS")
    print("="*60)

    if 'metrics' in result and 'overlay' in result['metrics']:
        metrics = result['metrics']['overlay']
        print(f"\nOverlay Metrics:")
        print(f"  MAE: {metrics['mean_absolute_error']:.4f}")
        print(f"  RMSE: {metrics['root_mean_square_error']:.4f}")
        print(f"  PSNR: {metrics['peak_signal_noise_ratio']:.2f} dB")
        print(f"  SSIM: {metrics['structural_similarity']:.4f}")
        print(f"  NCC: {metrics['normalized_cross_correlation']:.4f}")
        print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.1f}%")

    print(f"\nTotal time: {result['elapsed_time']:.2f} seconds")
    print(f"\nResults saved to: {output_path}")


def run_batch_registration(args):
    """Run registration on multiple image pairs."""
    print("="*60)
    print("BATCH REGISTRATION")
    print("="*60)

    # Initialize pipeline
    config = {
        'target_size': (args.image_size, args.image_size),
        'normalize': True,
        'histogram_matching': args.histogram_matching,
        'affine': {
            'enabled': args.use_affine,
            'type': args.affine_type,
            'metric': args.metric,
            'optimizer': args.optimizer,
            'multi_resolution': args.multi_resolution,
            'num_levels': args.num_levels,
            'num_bins': args.num_bins
        },
        'bspline': {
            'enabled': args.use_bspline,
            'grid_spacing': (args.grid_spacing, args.grid_spacing),
            'metric': args.metric,
            'optimizer': args.optimizer,
            'regularization': args.regularization,
            'multi_resolution': args.multi_resolution,
            'num_levels': args.num_levels,
            'num_bins': args.num_bins
        },
        'evaluation': {
            'compute_tre': True,
            'compute_dice': True,
            'compute_overlay': True,
            'compute_jacobian': True
        },
        'save_images': True,
        'save_metrics': True,
        'save_transforms': True
    }

    pipeline = RegistrationPipeline(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        config=config
    )

    # Save configuration
    pipeline.save_configuration()

    # Run batch registration
    results = pipeline.register_batch(
        modality1=args.modality1,
        modality2=args.modality2,
        category=args.category,
        num_pairs=args.num_pairs
    )

    print(f"\nBatch registration completed!")
    print(f"Results saved to: {args.output_dir}")


def run_synthetic_test(args):
    """Test registration with synthetic deformation."""
    print("="*60)
    print("SYNTHETIC DEFORMATION TEST")
    print("="*60)

    # Initialize pipeline
    config = {
        'target_size': (args.image_size, args.image_size),
        'normalize': True,
        'affine': {
            'enabled': True,
            'type': 'affine',
            'metric': args.metric,
            'optimizer': args.optimizer,
            'multi_resolution': True,
            'num_levels': 3,
            'num_bins': args.num_bins
        },
        'bspline': {
            'enabled': True,
            'grid_spacing': (32, 32),
            'metric': args.metric,
            'optimizer': args.optimizer,
            'regularization': 0.01,
            'multi_resolution': True,
            'num_levels': 3,
            'num_bins': args.num_bins
        }
    }

    pipeline = RegistrationPipeline(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        config=config
    )

    # Load a test image
    loader = MultimodalDataLoader(args.dataset_path)
    pairs = loader.load_image_pairs('ct', 'ct', 'Healthy', num_pairs=1)

    if not pairs:
        print("No images found for synthetic test!")
        return

    test_image = pairs[0][0]

    # Run synthetic test
    result = pipeline.register_with_synthetic_deformation(
        test_image,
        max_rotation=15,
        max_translation=20,
        elastic_alpha=20,
        elastic_sigma=3
    )

    # Visualize results
    visualizer = RegistrationVisualizer()
    output_path = Path(args.output_dir) / 'synthetic_test'
    create_comprehensive_report(result, str(output_path), visualizer)

    print(f"\nSynthetic test completed!")
    print(f"Results saved to: {output_path}")


def evaluate_dataset_performance(args):
    """Evaluate registration performance across the dataset."""
    print("="*60)
    print("DATASET PERFORMANCE EVALUATION")
    print("="*60)

    # Test different configurations
    configurations = [
        {
            'name': 'Rigid_Only',
            'affine': {'enabled': True, 'type': 'rigid'},
            'bspline': {'enabled': False}
        },
        {
            'name': 'Affine_Only',
            'affine': {'enabled': True, 'type': 'affine'},
            'bspline': {'enabled': False}
        },
        {
            'name': 'Affine_BSpline',
            'affine': {'enabled': True, 'type': 'affine'},
            'bspline': {'enabled': True, 'grid_spacing': (32, 32)}
        },
        {
            'name': 'BSpline_Only',
            'affine': {'enabled': False},
            'bspline': {'enabled': True, 'grid_spacing': (32, 32)}
        }
    ]

    results_summary = {}

    for config_update in configurations:
        print(f"\nTesting configuration: {config_update['name']}")

        # Create config
        config = {
            'target_size': (256, 256),
            'normalize': True,
            'histogram_matching': False,
            'affine': {
                'enabled': config_update['affine']['enabled'],
                'type': config_update['affine'].get('type', 'affine'),
                'metric': 'nmi',
                'optimizer': 'lbfgs',
                'multi_resolution': True,
                'num_levels': 3,
                'num_bins': 32
            },
            'bspline': {
                'enabled': config_update['bspline']['enabled'],
                'grid_spacing': config_update['bspline'].get('grid_spacing', (32, 32)),
                'metric': 'nmi',
                'optimizer': 'lbfgs',
                'regularization': 0.01,
                'multi_resolution': True,
                'num_levels': 3,
                'num_bins': 32
            },
            'evaluation': {
                'compute_overlay': True
            }
        }

        # Run pipeline
        output_dir = Path(args.output_dir) / config_update['name']
        pipeline = RegistrationPipeline(
            dataset_path=args.dataset_path,
            output_dir=str(output_dir),
            config=config
        )

        # Register batch
        results = pipeline.register_batch(
            modality1='ct',
            modality2='mri',
            category='Healthy',
            num_pairs=5
        )

        # Collect metrics
        mae_values = []
        psnr_values = []
        ssim_values = []
        time_values = []

        for result in results:
            if 'metrics' in result and 'overlay' in result['metrics']:
                mae_values.append(result['metrics']['overlay']['mean_absolute_error'])
                psnr_values.append(result['metrics']['overlay']['peak_signal_noise_ratio'])
                ssim_values.append(result['metrics']['overlay']['structural_similarity'])
            time_values.append(result.get('elapsed_time', 0))

        results_summary[config_update['name']] = {
            'mae': {'mean': np.mean(mae_values), 'std': np.std(mae_values)},
            'psnr': {'mean': np.mean(psnr_values), 'std': np.std(psnr_values)},
            'ssim': {'mean': np.mean(ssim_values), 'std': np.std(ssim_values)},
            'time': {'mean': np.mean(time_values), 'std': np.std(time_values)}
        }

    # Print comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)

    print("\n{:<20} {:<15} {:<15} {:<15} {:<15}".format(
        "Configuration", "MAE", "PSNR (dB)", "SSIM", "Time (s)"
    ))
    print("-" * 80)

    for name, metrics in results_summary.items():
        print("{:<20} {:.4f}±{:.4f}  {:.1f}±{:.1f}  {:.3f}±{:.3f}  {:.1f}±{:.1f}".format(
            name,
            metrics['mae']['mean'], metrics['mae']['std'],
            metrics['psnr']['mean'], metrics['psnr']['std'],
            metrics['ssim']['mean'], metrics['ssim']['std'],
            metrics['time']['mean'], metrics['time']['std']
        ))

    # Save summary
    summary_path = Path(args.output_dir) / 'performance_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nPerformance summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Brain Tumor Image Registration Pipeline"
    )

    # Mode selection
    parser.add_argument(
        'mode',
        choices=['single', 'batch', 'synthetic', 'evaluate'],
        help='Registration mode'
    )

    # Data parameters
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='./Dataset',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./registration_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--modality1',
        type=str,
        default='ct',
        choices=['ct', 'mri'],
        help='First modality'
    )
    parser.add_argument(
        '--modality2',
        type=str,
        default='mri',
        choices=['ct', 'mri'],
        help='Second modality'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='Healthy',
        choices=['Healthy', 'Tumor'],
        help='Image category'
    )
    parser.add_argument(
        '--num_pairs',
        type=int,
        default=10,
        help='Number of image pairs for batch mode'
    )

    # Registration parameters
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='Target image size'
    )
    parser.add_argument(
        '--use_affine',
        action='store_true',
        default=True,
        help='Use affine registration'
    )
    parser.add_argument(
        '--affine_type',
        type=str,
        default='affine',
        choices=['rigid', 'affine'],
        help='Type of affine registration'
    )
    parser.add_argument(
        '--use_bspline',
        action='store_true',
        default=True,
        help='Use B-spline registration'
    )
    parser.add_argument(
        '--grid_spacing',
        type=int,
        default=32,
        help='B-spline grid spacing'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='nmi',
        choices=['mi', 'nmi', 'ecc'],
        help='Similarity metric'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='lbfgs',
        choices=['powell', 'lbfgs', 'evolution'],
        help='Optimization method'
    )
    parser.add_argument(
        '--num_bins',
        type=int,
        default=32,
        help='Number of histogram bins for MI'
    )
    parser.add_argument(
        '--regularization',
        type=float,
        default=0.01,
        help='B-spline regularization weight'
    )
    parser.add_argument(
        '--multi_resolution',
        action='store_true',
        default=True,
        help='Use multi-resolution approach'
    )
    parser.add_argument(
        '--num_levels',
        type=int,
        default=3,
        help='Number of resolution levels'
    )
    parser.add_argument(
        '--histogram_matching',
        action='store_true',
        help='Apply histogram matching'
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run selected mode
    if args.mode == 'single':
        run_single_pair_registration(args)
    elif args.mode == 'batch':
        run_batch_registration(args)
    elif args.mode == 'synthetic':
        run_synthetic_test(args)
    elif args.mode == 'evaluate':
        evaluate_dataset_performance(args)


if __name__ == "__main__":
    main()