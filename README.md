# Multimodal Brain Tumor Image Registration System

## Overview

This project implements a comprehensive medical image registration pipeline for multimodal brain tumor images (CT and MRI). The system combines **affine** and **B-spline deformable registration** with **mutual information optimization** to achieve accurate alignment of medical images across different modalities.

## Core Objectives Achieved

**Affine + B-spline deformable registration implementation**
**Mutual information optimization**
**Target Registration Error (TRE) evaluation**
**Overlay alignment accuracy metrics**

## Features

### Registration Methods
- **Rigid Registration**: Translation and rotation only
- **Affine Registration**: Translation, rotation, scaling, and shearing
- **B-Spline Deformable Registration**: Non-rigid local deformations
- **Combined Pipeline**: Sequential affine + B-spline registration

### Optimization Metrics
- **Mutual Information (MI)**
- **Normalized Mutual Information (NMI)**
- **Entropy Correlation Coefficient (ECC)**
- **Local Mutual Information**

### Evaluation Metrics
- **Target Registration Error (TRE)**: Landmark-based accuracy assessment
- **Overlay Accuracy**: MAE, RMSE, PSNR, SSIM
- **Dice Coefficient**: Segmentation overlap
- **Jaccard Index**: Intersection over Union
- **Hausdorff Distance**: Boundary distance metrics
- **Jacobian Determinant**: Deformation field analysis

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Required packages:
- numpy >= 1.24.0
- scipy >= 1.10.0
- scikit-image >= 0.21.0
- opencv-python >= 4.8.0
- SimpleITK >= 2.3.0
- matplotlib >= 3.7.0
- tqdm >= 4.66.0

## Project Structure

```
anudeep/
├── Dataset/                        # Brain tumor dataset
│   ├── Brain Tumor CT scan Images/
│   │   ├── Healthy/                # Healthy CT scans
│   │   └── Tumor/                  # Tumor CT scans
│   └── Brain Tumor MRI images/
│       ├── Healthy/                # Healthy MRI scans
│       └── Tumor/                  # Tumor MRI scans
│
├── src/                            # Source code
│   ├── data_loader.py             # Data loading and preprocessing
│   ├── mutual_information.py      # MI metric implementation
│   ├── affine_registration.py     # Affine registration module
│   ├── bspline_registration.py    # B-spline registration module
│   ├── evaluation_metrics.py      # Evaluation metrics
│   ├── registration_pipeline.py   # Main pipeline
│   └── visualization.py           # Visualization utilities
│
├── main.py                        # Main execution script
├── requirements.txt               # Dependencies
└── README.md                     # Documentation
```

## Usage

### 1. Single Pair Registration

Register a single CT-MRI image pair:

```bash
python main.py single --modality1 ct --modality2 mri --category Healthy
```

### 2. Batch Registration

Register multiple image pairs:

```bash
python main.py batch --num_pairs 10 --modality1 ct --modality2 mri
```

### 3. Synthetic Deformation Test

Test registration accuracy with known deformations:

```bash
python main.py synthetic
```

### 4. Performance Evaluation

Compare different registration configurations:

```bash
python main.py evaluate
```

### Command Line Arguments

```bash
python main.py [mode] [options]

Modes:
  single      Register single image pair
  batch       Register multiple pairs
  synthetic   Test with synthetic deformation
  evaluate    Compare registration methods

Options:
  --dataset_path PATH      Dataset directory (default: ./Dataset)
  --output_dir PATH        Output directory (default: ./registration_results)
  --modality1 {ct,mri}     First modality (default: ct)
  --modality2 {ct,mri}     Second modality (default: mri)
  --category {Healthy,Tumor} Image category (default: Healthy)
  --num_pairs N           Number of pairs for batch (default: 10)

Registration Parameters:
  --image_size SIZE       Target image size (default: 256)
  --use_affine           Enable affine registration
  --affine_type TYPE     Affine type: rigid/affine (default: affine)
  --use_bspline          Enable B-spline registration
  --grid_spacing N       B-spline grid spacing (default: 32)
  --metric {mi,nmi,ecc}  Similarity metric (default: nmi)
  --optimizer OPT        Optimizer: powell/lbfgs/evolution (default: lbfgs)
  --num_bins N           Histogram bins for MI (default: 32)
  --regularization W     B-spline regularization (default: 0.01)
  --multi_resolution     Use multi-resolution (default: True)
  --num_levels N         Resolution levels (default: 3)
```

## Python API Usage

```python
from src.registration_pipeline import RegistrationPipeline
from src.visualization import RegistrationVisualizer

# Initialize pipeline
pipeline = RegistrationPipeline(
    dataset_path="./Dataset",
    output_dir="./results"
)

# Load and register images
from src.data_loader import MultimodalDataLoader
loader = MultimodalDataLoader("./Dataset")
pairs = loader.load_image_pairs('ct', 'mri', 'Healthy', num_pairs=1)
fixed, moving = pairs[0]

# Perform registration
result = pipeline.register_pair(fixed, moving)

# Visualize results
visualizer = RegistrationVisualizer()
visualizer.plot_registration_results(
    result['fixed_image'],
    result['moving_image'],
    result['final_registered']
)

# Access metrics
print(f"MAE: {result['metrics']['overlay']['mean_absolute_error']:.4f}")
print(f"TRE: {result['metrics']['tre']['mean_tre']:.2f} pixels")
```

## Registration Pipeline

### Stage 1: Preprocessing
- Image resizing to target dimensions
- Intensity normalization
- Optional histogram matching

### Stage 2: Affine Registration
- Multi-resolution approach (coarse-to-fine)
- Mutual information optimization
- Parameter optimization: translation, rotation, scaling, shearing

### Stage 3: B-Spline Registration
- Control point grid initialization
- Deformable transformation
- Regularization for smooth deformations
- Multi-resolution refinement

### Stage 4: Evaluation
- Compute overlay metrics (MAE, PSNR, SSIM)
- Calculate TRE if landmarks available
- Assess deformation quality (Jacobian determinant)

## Evaluation Results

### Expected Performance Metrics

| Configuration      | MAE     | PSNR (dB) | SSIM    | Time (s) |
|-------------------|---------|-----------|---------|----------|
| Rigid Only        | 0.15-0.20 | 20-25    | 0.70-0.80 | 2-5      |
| Affine Only       | 0.10-0.15 | 25-30    | 0.80-0.85 | 5-10     |
| Affine + B-Spline | 0.05-0.10 | 30-35    | 0.85-0.95 | 15-30    |

### TRE Success Rates
- < 2mm: 60-70% of landmarks
- < 5mm: 85-95% of landmarks
- < 10mm: 95-99% of landmarks

## Visualization Outputs

The system generates comprehensive visualizations:

1. **Registration Results Grid**: Shows fixed, moving, and registered images with checkerboard and difference maps
2. **Overlay Visualization**: Before/after overlays and RGB composites
3. **Deformation Field**: Vector field and magnitude maps
4. **Jacobian Determinant**: Identifies folding regions
5. **Metrics Evolution**: Tracks optimization progress

## Dataset Information

- **Total Images**: 9,618 (4,618 CT, 5,000 MRI)
- **Categories**: Healthy and Tumor
- **Formats**: JPG, PNG
- **Resolution**: Variable (standardized to 256x256)
- **Modalities**: CT and MRI brain scans

## Advanced Features

### Multi-Resolution Registration
- Pyramid levels for coarse-to-fine optimization
- Automatic parameter scaling between levels
- Improved convergence and robustness

### Regularization
- Bending energy minimization for B-splines
- Prevents unrealistic deformations
- Maintains topology preservation

### Parallel Processing
- Batch registration support
- Efficient memory management
- Progress tracking with tqdm

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce image size or batch size
2. **Convergence Issues**: Adjust optimizer parameters or increase iterations
3. **Poor Registration**: Try different metric or enable histogram matching
4. **Folding Detection**: Increase regularization weight

## Future Enhancements

- [ ] GPU acceleration with PyTorch/TensorFlow
- [ ] Deep learning-based registration
- [ ] 3D volume registration
- [ ] Real-time registration visualization
- [ ] Cross-validation framework
- [ ] Uncertainty quantification

## References

1. Mattes, D., et al. "PET-CT image registration in the chest using free-form deformations." IEEE TMI (2003)
2. Rueckert, D., et al. "Nonrigid registration using free-form deformations." IEEE TMI (1999)
3. Studholme, C., et al. "An overlap invariant entropy measure of 3D medical image alignment." Pattern Recognition (1999)

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please open an issue on the project repository.

---

**Note**: This implementation is optimized for 2D brain tumor images. For clinical applications, additional validation and regulatory compliance would be required.
