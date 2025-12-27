# YOLOv11 Knowledge Distillation Framework

A modular framework for experimenting with head-level and neck-level knowledge distillation techniques on Ultralytics YOLOv11 object detection models.

## Overview

This repository implements a flexible knowledge distillation pipeline specifically designed for YOLOv11 object detection models. Unlike traditional logit-level distillation approaches, our framework focuses on **head-level** and **neck-level** distillation, enabling more effective knowledge transfer in the critical feature extraction and prediction stages of object detection architectures.

The implementation systematically tests various distillation approaches at different network levels, with emphasis on detection head distillation and feature pyramid network (neck) distillation. The framework emphasizes reproducibility, modularity, and experimental rigor for research purposes.

## Key Features

- **Head-Level Distillation**: Specialized techniques for object detection heads (classification and regression)
- **Neck-Level Distillation**: Multi-scale feature map distillation from the feature pyramid network
- **Multi-Modal Distillation (M2D2)**: Our proposed multimodal approach for DFL head distillation
- **Modular Architecture**: Easily extensible components for new distillation methods
- **Configurable Experiments**: YAML-based configuration for reproducible experiments
- **Comprehensive Logging**: Detailed experiment tracking, loss breakdowns, and visualizations
- **Spatial Masking**: Intelligent foreground-aware masking for focused distillation

## Architecture Overview

Our distillation framework operates at two critical levels of the YOLO architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    YOLOv11 Distillation Framework            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐         ┌─────────────────────────┐   │
│  │   Teacher Model │         │      Student Model      │   │
│  │  (Pre-trained)  │         │    (Under Training)     │   │
│  └─────────────────┘         └─────────────────────────┘   │
│           │                            │                    │
│           └──────────┬─────────────────┘                    │
│                      │                                      │
│         ┌─────────────────────────────────────┐            │
│         │        Distillation Pipeline        │            │
│         ├─────────────────────────────────────┤            │
│         │                                     │            │
│         │  ┌──────────────────────────────┐  │            │
│         │  │     NECK-LEVEL DISTILLATION  │  │            │
│         │  │  • Feature Maps              │  │            │
│         │  │  • Attention Maps            │  │            │
│         │  │  • Specialized Methods       │  │            │
│         │  └──────────────────────────────┘  │            │
│         │                                     │            │
│         │  ┌──────────────────────────────┐  │            │
│         │  │    HEAD-LEVEL DISTILLATION   │  │            │
│         │  │  • Classification Head       │  │            │
│         │  │  • DFL Head                  │  │            │
│         │  │  • M2D2 (Multimodal)         │  │            │
│         │  │  • Regression Head           │  │            │
│         │  └──────────────────────────────┘  │            │
│         │                                     │            │
│         └─────────────────────────────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Mahdi-Golizadeh/moded_YOLO.git
cd moded_YOLO

# Create and activate virtual environment (recommended)
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate

# Install PyTorch with CUDA support
# Visit https://pytorch.org/get-started/locally/ for system-specific commands
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Ultralytics YOLO
pip install ultralytics

# Install additional dependencies
pip install numpy scipy scikit-learn matplotlib seaborn
pip install tensorboard  # for visualization
```

### Verification
```python
# Test installation
from ultralytics import YOLO
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Ultralytics version: {YOLO.__version__}")
```

## Quick Start Guide

### Basic Training with Distillation

```python
from ultralytics import YOLO

# Initialize student model
model = YOLO("yolo11n.pt")

# Train with comprehensive distillation
results = model.train(
    # Basic training parameters
    data="coco8.yaml",
    epochs=10,
    imgsz=640,
    
    # Teacher model configuration
    teacher_model="yolo11_2x.yaml",  # Teacher model architecture
    mask_type="pyramid",              # Mask type: "original" or "pyramid"
    
    # Head-Level Distillation
    cls_dist=True,                    # Classification head distillation
    dfl_dist=True,                    # DFL head distillation
    M2D2=True,                        # Multimodal DFL distillation
    l2_dist=True,                     # Regression head distillation
    
    # Neck-Level Distillation
    feat_distill=True,                # Enable feature distillation
    feat=True,                        # Distill feature maps
    feat_att=True,                    # Distill attention maps
    
    # Hyperparameters
    level_weights=[1.0, 1.0, 1.0],    # Feature level weights
)
```

## Head-Level Distillation

### Classification Head Distillation

Distills knowledge from the teacher's classification predictions to the student.

**Parameters:**
```python
cls_dist=True,           # Enable classification head distillation
cls_dist_kl=False,       # Use KL divergence (True) or BCE (False)
cls_fg_mask=True,        # Apply foreground spatial masking
cls_dist_t=1.0,          # Temperature for softening predictions
cls_alpha=1.0,           # Loss weight multiplier
```

**Usage Example:**
```python
# KL Divergence based distillation
results = model.train(
    data="dataset.yaml",
    epochs=50,
    teacher_model="teacher_model.pt",
    cls_dist=True,
    cls_dist_kl=True,      # Use KL divergence
    cls_dist_t=4.0,        # High temperature for soft targets
    cls_alpha=0.5,         # Moderate weight
    cls_fg_mask=True,      # Focus on foreground regions
)
```

### DFL Head Distillation

Direct distillation of the Distribution Focal Loss (DFL) head for bounding box predictions.

**Parameters:**
```python
dfl_dist=True,           # Enable DFL head distillation
dfl_fg_mask=True,        # Apply foreground masking
dfl_t=1.0,               # Temperature parameter
dfl_alpha=1.0,           # Loss weight multiplier
```

### M2D2: Multimodal DFL Distillation

Our proposed multimodal approach for DFL head distillation that combines multiple supervision signals.

**Parameters:**
```python
M2D2=True,               # Enable multimodal distillation
m2d2_t=1.0,              # Temperature parameter
m2d2_alpha=1.0,          # Loss weight multiplier
```

**Features:**
- Combines distribution matching with feature alignment
- Internally applies spatial masking
- Works synergistically with other head distillation methods

### Regression Head Distillation

Traditional regression-based bounding box distillation.

**Parameters:**
```python
l2_dist=True,            # Enable regression distillation
l2_fg_mask=False,        # Spatial masking (typically disabled for regression)
l2_alpha=1.0,            # Loss weight multiplier
```

## Neck-Level Distillation

### Feature Map Distillation

Distills multi-scale feature maps from the Feature Pyramid Network (FPN).

**Parameters:**
```python
feat_distill=True,       # Enable feature distillation
feat=True,               # Distill raw feature maps
feat_att=True,           # Distill attention maps
feat_mask=True,          # Apply spatial masking
loss_ty="cosine",        # Loss type: "cosine" or "l2"
level_weights=[1., 1., 1.],  # Weights for P3, P4, P5 features
feature_lambda=1.0,      # Global feature loss multiplier
```

### Advanced Feature Distillation Methods

Enable additional specialized distillation techniques:

```python
feat_oth=True,           # Enable advanced methods
use_cwd=True,            # Channel-Wise Distillation
use_crd=False,           # Correlation Relational Distillation
use_mmd=False,           # Maximum Mean Discrepancy
use_spatial_att=False,   # Spatial Attention Distillation
use_channel_att=False,   # Channel Attention Distillation
```

#### Channel-Wise Distillation (CWD)
```python
# CWD focuses on channel relationships
results = model.train(
    feat_distill=True,
    feat_oth=True,
    use_cwd=True,
    feat_mask=True,
    level_weights=[0.3, 0.5, 0.2],  # Emphasize middle features
)
```

#### Spatial Attention Distillation
```python
# Distill spatial attention patterns
results = model.train(
    feat_distill=True,
    feat_att=True,
    feat_oth=True,
    use_spatial_att=True,
    feat_mask=True,
    loss_ty="l2",
)
```

## Configuration Examples

### Example 1: Head-Only Distillation
```python
# Focus on detection head distillation
results = model.train(
    data="coco.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    
    # Teacher model
    teacher_model="yolo11x.pt",
    mask_type="pyramid",
    
    # Head distillation
    cls_dist=True,
    cls_dist_kl=True,
    cls_dist_t=3.0,
    cls_alpha=0.7,
    
    dfl_dist=True,
    dfl_fg_mask=True,
    dfl_alpha=0.8,
    
    M2D2=True,
    m2d2_t=2.0,
    m2d2_alpha=0.6,
    
    # No neck distillation
    feat_distill=False,
)
```

### Example 2: Comprehensive Distillation
```python
# Full distillation pipeline
results = model.train(
    data="custom_dataset.yaml",
    epochs=150,
    imgsz=640,
    
    # Teacher configuration
    teacher_model="yolo11_2x.yaml",
    mask_type="pyramid",
    
    # Head-level distillation
    cls_dist=True,
    cls_dist_kl=False,  # Use BCE
    cls_fg_mask=True,
    cls_dist_t=2.0,
    cls_alpha=0.5,
    
    dfl_dist=True,
    dfl_fg_mask=True,
    dfl_t=1.5,
    dfl_alpha=0.6,
    
    M2D2=True,
    m2d2_alpha=0.4,
    
    # Neck-level distillation
    feat_distill=True,
    feat=True,
    feat_att=True,
    feat_mask=True,
    loss_ty="cosine",
    
    feat_oth=True,
    use_cwd=True,
    use_spatial_att=True,
    
    # Hyperparameters
    level_weights=[0.4, 0.4, 0.2],
    feature_lambda=0.8,
)
```

### Example 3: Research Experiment Configuration
```python
# A/B testing different configurations
configs = [
    {
        "name": "head_only",
        "cls_dist": True,
        "dfl_dist": True,
        "feat_distill": False,
    },
    {
        "name": "neck_only",
        "cls_dist": False,
        "dfl_dist": False,
        "feat_distill": True,
        "feat": True,
        "feat_att": True,
    },
    {
        "name": "combined",
        "cls_dist": True,
        "dfl_dist": True,
        "feat_distill": True,
        "feat": True,
    }
]

for config in configs:
    print(f"Running experiment: {config['name']}")
    results = model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640,
        teacher_model="teacher.yaml",
        **{k: v for k, v in config.items() if k != 'name'}
    )
```

## Hyperparameter Tuning Guide

### Temperature Parameters
- **Low values (0.5-1.0)**: Preserve teacher's confidence
- **Medium values (1.0-3.0)**: Balanced softening
- **High values (3.0-10.0)**: Significant softening, emphasize relationships

### Loss Weights (Alpha)
```python
# Recommended starting points:
cls_alpha=0.5,      # Classification
dfl_alpha=0.7,      # DFL head
m2d2_alpha=0.3,     # M2D2 (complementary)
l2_alpha=0.4,       # Regression
feature_lambda=0.8, # Feature distillation
```

### Feature Level Weights
```python
# Different strategies:
level_weights=[1.0, 1.0, 1.0],  # Equal weighting
level_weights=[0.3, 0.5, 0.2],  # Emphasize middle features
level_weights=[0.5, 0.3, 0.2],  # Emphasize low-level features
```

## Best Practices

### 1. Teacher Model Selection
```python
# Choose appropriate teacher model
teacher_options = {
    "light": "yolo11n.pt",      # Small teacher
    "medium": "yolo11s.pt",     # Medium teacher
    "strong": "yolo11m.pt",     # Strong teacher
    "expert": "yolo11x.pt",     # Expert teacher
}

# Rule of thumb: Teacher should be 2-4x larger than student
```

### 2. Progressive Distillation
```python
# Phase 1: Neck distillation only
phase1 = model.train(epochs=30, feat_distill=True, cls_dist=False, dfl_dist=False)

# Phase 2: Add head distillation
phase2 = model.train(epochs=30, feat_distill=True, cls_dist=True, dfl_dist=True)

# Phase 3: Fine-tune with full distillation
phase3 = model.train(epochs=20, feat_distill=True, cls_dist=True, dfl_dist=True, M2D2=True)
```

### 3. Monitoring and Debugging
```python
# Enable detailed logging
results = model.train(
    # ... configuration ...
    verbose=True,           # Detailed training output
    save_period=5,         # Save checkpoints every 5 epochs
    plots=True,            # Generate training plots
    device=0,              # Use GPU 0
)
```

## Output and Results

### Training Logs
```
Training logs are saved in: runs/train/exp/
├── weights/              # Model checkpoints
├── results.png           # Training metrics
├── confusion_matrix.png  # Validation confusion
├── train_batch*.jpg      # Training batches
└── val_batch*.jpg       # Validation batches
```

### Distillation-Specific Metrics
The framework logs:
- Individual distillation loss components
- Teacher vs student performance comparison
- Feature map visualizations (when enabled)
- Attention map comparisons
- Loss weight schedules

### TensorBoard Integration
```bash
# Launch TensorBoard to visualize training
tensorboard --logdir runs/train
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
```python
# Reduce batch size
results = model.train(batch=8, ...)

# Disable some distillation methods
results = model.train(feat_att=False, feat_oth=False, ...)
```

2. **Slow Training**
```python
# Use mixed precision
results = model.train(amp=True, ...)

# Disable visualization during training
results = model.train(plots=False, ...)
```

3. **Poor Distillation Results**
```python
# Adjust temperature
results = model.train(cls_dist_t=3.0, dfl_t=2.0, ...)

# Balance loss weights
results = model.train(cls_alpha=0.3, dfl_alpha=0.5, ...)

# Try different mask type
results = model.train(mask_type="original", ...)
```

## Contributing

We welcome contributions to expand the distillation framework:

1. **New Distillation Methods**: Implement additional head or neck distillation techniques
2. **Loss Functions**: Add new distillation loss formulations
3. **Masking Strategies**: Develop advanced spatial masking approaches
4. **Optimization Techniques**: Improve training efficiency and convergence

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{yolo11_distill_framework,
  title = {YOLOv11 Head and Neck Knowledge Distillation Framework},
  author = {Mahdi Golizadeh},
  year = {2024},
  url = {https://github.com/Mahdi-Golizadeh/moded_YOLO},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This framework is under active development. New distillation methods and features are regularly added. Check the repository for the latest updates and examples.
