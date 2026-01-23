"""
Logging module for RF-DETR training.

Provides:
- Augmentation logging (which augmentations were applied to each image)
- Training progress logging
"""

from rfdetr.training.logging.augmentation_logger import (
    AugmentationLogger,
    TrainingLogger,
)

__all__ = [
    "AugmentationLogger",
    "TrainingLogger",
]
