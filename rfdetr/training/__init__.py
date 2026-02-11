"""
RF-DETR Training Module - YOLO-style training system.

This module provides a complete training pipeline for RF-DETR with:
- Ultralytics-style augmentations (Mosaic, MixUp, CutMix, HSV, etc.)
- Automatic visualizations (batch images, metrics plots, confusion matrix)
- Reproducible experiments with seed management
- Augmentation logging
"""

from rfdetr.training.utils.seed import setup_seed, get_random_state, set_random_state
from rfdetr.training.utils.config import (
    AugmentationConfig,
    TrainingConfig,
    ModelConfig,
    ExportConfig,
    validate_config,
)
from rfdetr.training.dataset import RFDETRDataset, build_dataset, collate_fn
from rfdetr.training.trainer import RFDETRTrainer

# RFDETRValidator потребує rfdetr[plus] (platform models) — робимо імпорт опціональним
try:
    from rfdetr.training.validator import RFDETRValidator
except ImportError:
    RFDETRValidator = None

__all__ = [
    # Seed management
    "setup_seed",
    "get_random_state",
    "set_random_state",
    # Configuration
    "AugmentationConfig",
    "TrainingConfig",
    "ModelConfig",
    "ExportConfig",
    "validate_config",
    # Dataset
    "RFDETRDataset",
    "build_dataset",
    "collate_fn",
    # Trainer/Validator
    "RFDETRTrainer",
    "RFDETRValidator",
]
