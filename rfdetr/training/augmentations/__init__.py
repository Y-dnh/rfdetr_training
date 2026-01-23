"""
Augmentation module for RF-DETR training.

Provides Ultralytics-style augmentations for object detection:
- Mosaic: 2x2 grid combining 4 images
- MixUp: Alpha blending of two images
- CutMix: Cutting and pasting regions between images
- RandomHSV: Hue, Saturation, Value adjustments
- RandomFlip: Horizontal and vertical flips
- RandomPerspective: Rotation, translation, scale, shear, perspective
- RandomErasing: Random rectangular region erasing
"""

from rfdetr.training.augmentations.base import (
    BaseTransform,
    Compose,
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
    RandomResize,
    SquareResize,
)
from rfdetr.training.augmentations.color import (
    RandomHSV,
    RandomBrightness,
    RandomContrast,
    RandomBlur,
    RandomNoise,
)
from rfdetr.training.augmentations.geometric import (
    RandomFlip,
    RandomPerspective,
    RandomRotate,
    RandomScale,
    LetterBox,
)
from rfdetr.training.augmentations.mosaic import Mosaic, Mosaic9
from rfdetr.training.augmentations.mixup import MixUp, CutMix
from rfdetr.training.augmentations.erasing import RandomErasing, GridErasing
from rfdetr.training.augmentations.pipeline import AugmentationPipeline, ValidationPipeline

__all__ = [
    # Base
    "BaseTransform",
    "Compose",
    "ToTensor",
    "Normalize",
    "RandomHorizontalFlip",
    "RandomResize",
    "SquareResize",
    # Color
    "RandomHSV",
    "RandomBrightness",
    "RandomContrast",
    "RandomBlur",
    "RandomNoise",
    # Geometric
    "RandomFlip",
    "RandomPerspective",
    "RandomRotate",
    "RandomScale",
    "LetterBox",
    # Mosaic
    "Mosaic",
    "Mosaic9",
    # MixUp/CutMix
    "MixUp",
    "CutMix",
    # Erasing
    "RandomErasing",
    "GridErasing",
    # Pipeline
    "AugmentationPipeline",
    "ValidationPipeline",
]
