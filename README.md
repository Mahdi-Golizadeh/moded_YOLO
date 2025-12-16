# YOLOv11 Knowledge Distillation Framework

A modular framework for experimenting with different knowledge distillation techniques on Ultralytics YOLOv11 object detection models.

## Overview

This repository implements a flexible pipeline for knowledge distillation experiments on YOLOv11. The framework is designed to systematically test various distillation approaches at different network levels, with a current focus on logit-level and head-level distillation methods. The implementation emphasizes reproducibility and modularity for experimental research.

## Key Features

- **Multi-Level Distillation**: Support for distillation at different network levels
- **Logit-Level Methods**: Implementation of output-level knowledge transfer techniques
- **Head-Level Methods**: Specialized distillation approaches for detection heads
- **Modular Architecture**: Easily extensible components for new distillation methods
- **Configurable Experiments**: YAML-based configuration for reproducible experiments
- **Comprehensive Logging**: Detailed experiment tracking and visualization

## Current Experimental Focus

The current implementation focuses on testing and comparing:
- **Logit-level distillation**: Knowledge transfer through model outputs
- **Head-level distillation**: Specialized techniques for object detection heads

Additional distillation approaches are planned for future implementation and comparison.

## Installation

```bash
# Clone the repository
git clone https://github.com/Mahdi-Golizadeh/moded_YOLO.git
cd moded_YOLO

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support (adjust according to your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
