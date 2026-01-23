"""
Base augmentation classes and wrappers for existing RF-DETR transforms.

This module provides:
- BaseTransform: Abstract base class for all augmentations
- Compose: Extended compose with logging support
- Wrappers for existing rfdetr.datasets.transforms classes
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import PIL.Image
import torchvision.transforms.functional as F

# Import existing transforms from RF-DETR
from rfdetr.datasets.transforms import (
    crop,
    hflip,
    resize,
    pad,
    RandomCrop as OrigRandomCrop,
    RandomSizeCrop as OrigRandomSizeCrop,
    CenterCrop as OrigCenterCrop,
    RandomHorizontalFlip as OrigRandomHorizontalFlip,
    RandomResize as OrigRandomResize,
    SquareResize as OrigSquareResize,
    RandomPad as OrigRandomPad,
    Pad as OrigPad,
    PILtoNdArray as OrigPILtoNdArray,
    NdArraytoPIL as OrigNdArraytoPIL,
    RandomExpand as OrigRandomExpand,
    RandomSelect as OrigRandomSelect,
    ToTensor as OrigToTensor,
    RandomErasing as OrigRandomErasing,
    Normalize as OrigNormalize,
    Compose as OrigCompose,
)


@dataclass
class AugmentationResult:
    """Result of an augmentation containing metadata for logging."""
    name: str
    applied: bool
    parameters: Dict[str, Any] = field(default_factory=dict)


class BaseTransform(ABC):
    """
    Abstract base class for all augmentations.
    
    All augmentations should inherit from this class and implement
    the __call__ method that takes (image, target) and returns (image, target).
    
    Attributes:
        p: Probability of applying the augmentation (0.0-1.0).
        name: Name of the augmentation for logging.
    """
    
    def __init__(self, p: float = 1.0, name: Optional[str] = None):
        """
        Initialize base transform.
        
        Args:
            p: Probability of applying the transform.
            name: Name for logging. If None, uses class name.
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Probability must be in [0.0, 1.0], got {p}")
        self.p = p
        self.name = name or self.__class__.__name__
        self._last_result: Optional[AugmentationResult] = None
    
    @abstractmethod
    def apply(
        self, 
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor], 
        target: Dict[str, Any]
    ) -> Tuple[Union[PIL.Image.Image, np.ndarray, torch.Tensor], Dict[str, Any]]:
        """
        Apply the augmentation to image and target.
        
        Args:
            image: Input image (PIL, numpy, or tensor).
            target: Target dict containing 'boxes', 'labels', etc.
        
        Returns:
            Tuple of (augmented_image, augmented_target).
        """
        pass
    
    def __call__(
        self, 
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor], 
        target: Dict[str, Any]
    ) -> Tuple[Union[PIL.Image.Image, np.ndarray, torch.Tensor], Dict[str, Any]]:
        """
        Call the augmentation with probability check.
        
        Args:
            image: Input image.
            target: Target dict.
        
        Returns:
            Tuple of (image, target) - possibly augmented.
        """
        if random.random() < self.p:
            image, target = self.apply(image, target)
            self._last_result = AugmentationResult(
                name=self.name,
                applied=True,
                parameters=self.get_parameters()
            )
        else:
            self._last_result = AugmentationResult(
                name=self.name,
                applied=False,
                parameters={}
            )
        return image, target
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters used for the last augmentation.
        Override in subclasses to return specific parameters.
        
        Returns:
            Dictionary of parameter names to values.
        """
        return {'p': self.p}
    
    def get_last_result(self) -> Optional[AugmentationResult]:
        """Get the result of the last augmentation call."""
        return self._last_result
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class Compose:
    """
    Compose multiple augmentations with logging support.
    
    This is an extended version of the original Compose that supports:
    - Logging of applied augmentations
    - Dynamic enable/disable of transforms
    """
    
    def __init__(
        self, 
        transforms: List[Union[BaseTransform, Callable]],
        log_augmentations: bool = False
    ):
        """
        Initialize Compose.
        
        Args:
            transforms: List of transforms to apply sequentially.
            log_augmentations: Whether to collect augmentation logs.
        """
        self.transforms = transforms
        self.log_augmentations = log_augmentations
        self._last_log: List[AugmentationResult] = []
    
    def __call__(
        self, 
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor], 
        target: Dict[str, Any]
    ) -> Tuple[Union[PIL.Image.Image, np.ndarray, torch.Tensor], Dict[str, Any]]:
        """
        Apply all transforms sequentially.
        
        Args:
            image: Input image.
            target: Target dict.
        
        Returns:
            Tuple of (augmented_image, augmented_target).
        """
        self._last_log = []
        
        for t in self.transforms:
            image, target = t(image, target)
            
            # Collect log if the transform supports it
            if self.log_augmentations and isinstance(t, BaseTransform):
                result = t.get_last_result()
                if result is not None:
                    self._last_log.append(result)
        
        return image, target
    
    def get_last_log(self) -> List[AugmentationResult]:
        """Get the log of the last augmentation call."""
        return self._last_log
    
    def get_applied_augmentations(self) -> List[AugmentationResult]:
        """Get only the augmentations that were actually applied."""
        return [r for r in self._last_log if r.applied]
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


# =============================================================================
# Wrappers for existing RF-DETR transforms
# =============================================================================

class RandomCrop(BaseTransform):
    """Wrapper for RF-DETR RandomCrop."""
    
    def __init__(self, size: Tuple[int, int], p: float = 1.0):
        super().__init__(p=p)
        self.size = size
        self._orig_transform = OrigRandomCrop(size)
    
    def apply(self, image, target):
        return self._orig_transform(image, target)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'size': self.size}


class RandomSizeCrop(BaseTransform):
    """Wrapper for RF-DETR RandomSizeCrop."""
    
    def __init__(self, min_size: int, max_size: int, p: float = 1.0):
        super().__init__(p=p)
        self.min_size = min_size
        self.max_size = max_size
        self._orig_transform = OrigRandomSizeCrop(min_size, max_size)
    
    def apply(self, image, target):
        return self._orig_transform(image, target)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'min_size': self.min_size, 'max_size': self.max_size}


class CenterCrop(BaseTransform):
    """Wrapper for RF-DETR CenterCrop."""
    
    def __init__(self, size: Tuple[int, int], p: float = 1.0):
        super().__init__(p=p)
        self.size = size
        self._orig_transform = OrigCenterCrop(size)
    
    def apply(self, image, target):
        return self._orig_transform(image, target)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'size': self.size}


class RandomHorizontalFlip(BaseTransform):
    """Wrapper for RF-DETR RandomHorizontalFlip with tracking."""
    
    def __init__(self, p: float = 0.5):
        super().__init__(p=1.0)  # Always call, internal p handles flip
        self._flip_p = p
        self._flipped = False
    
    def apply(self, image, target):
        if random.random() < self._flip_p:
            self._flipped = True
            return hflip(image, target)
        self._flipped = False
        return image, target
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'p': self._flip_p, 'flipped': self._flipped}


class RandomResize(BaseTransform):
    """Wrapper for RF-DETR RandomResize."""
    
    def __init__(self, sizes: List[int], max_size: Optional[int] = None, p: float = 1.0):
        super().__init__(p=p)
        self.sizes = sizes
        self.max_size = max_size
        self._orig_transform = OrigRandomResize(sizes, max_size)
        self._last_size = None
    
    def apply(self, image, target):
        self._last_size = random.choice(self.sizes)
        return resize(image, target, self._last_size, self.max_size)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'sizes': self.sizes, 'max_size': self.max_size, 'chosen_size': self._last_size}


class SquareResize(BaseTransform):
    """Wrapper for RF-DETR SquareResize."""
    
    def __init__(self, sizes: List[int], p: float = 1.0):
        super().__init__(p=p)
        self.sizes = sizes
        self._orig_transform = OrigSquareResize(sizes)
        self._last_size = None
    
    def apply(self, image, target):
        self._last_size = random.choice(self.sizes)
        return self._orig_transform(image, target)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'sizes': self.sizes, 'chosen_size': self._last_size}


class RandomPad(BaseTransform):
    """Wrapper for RF-DETR RandomPad."""
    
    def __init__(self, max_pad: int, p: float = 1.0):
        super().__init__(p=p)
        self.max_pad = max_pad
        self._orig_transform = OrigRandomPad(max_pad)
    
    def apply(self, image, target):
        return self._orig_transform(image, target)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'max_pad': self.max_pad}


class Pad(BaseTransform):
    """Wrapper for RF-DETR Pad."""
    
    def __init__(
        self, 
        size: Optional[Union[int, Tuple[int, int]]] = None,
        size_divisor: int = 32,
        pad_mode: int = 0,
        offsets: Optional[List[int]] = None,
        fill_value: Tuple[float, float, float] = (127.5, 127.5, 127.5),
        p: float = 1.0
    ):
        super().__init__(p=p)
        self.size = size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.offsets = offsets
        self.fill_value = fill_value
        self._orig_transform = OrigPad(size, size_divisor, pad_mode, offsets, fill_value)
    
    def apply(self, image, target):
        return self._orig_transform(image, target)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'size': self.size,
            'size_divisor': self.size_divisor,
            'pad_mode': self.pad_mode
        }


class PILtoNdArray(BaseTransform):
    """Wrapper for RF-DETR PILtoNdArray."""
    
    def __init__(self, p: float = 1.0):
        super().__init__(p=p)
        self._orig_transform = OrigPILtoNdArray()
    
    def apply(self, image, target):
        return self._orig_transform(image, target)


class NdArraytoPIL(BaseTransform):
    """Wrapper for RF-DETR NdArraytoPIL."""
    
    def __init__(self, p: float = 1.0):
        super().__init__(p=p)
        self._orig_transform = OrigNdArraytoPIL()
    
    def apply(self, image, target):
        return self._orig_transform(image, target)


class RandomExpand(BaseTransform):
    """Wrapper for RF-DETR RandomExpand."""
    
    def __init__(
        self, 
        ratio: float = 4.0, 
        prob: float = 0.5, 
        fill_value: Tuple[float, float, float] = (127.5, 127.5, 127.5),
        p: float = 1.0
    ):
        super().__init__(p=p)
        self.ratio = ratio
        self.prob = prob
        self.fill_value = fill_value
        self._orig_transform = OrigRandomExpand(ratio, prob, fill_value)
    
    def apply(self, image, target):
        return self._orig_transform(image, target)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'ratio': self.ratio, 'prob': self.prob}


class ToTensor(BaseTransform):
    """Wrapper for RF-DETR ToTensor."""
    
    def __init__(self, p: float = 1.0):
        super().__init__(p=p)
        self._orig_transform = OrigToTensor()
    
    def apply(self, image, target):
        return self._orig_transform(image, target)


class RandomErasingWrapper(BaseTransform):
    """Wrapper for RF-DETR RandomErasing."""
    
    def __init__(
        self, 
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0,
        inplace: bool = False
    ):
        super().__init__(p=1.0)  # Always call, internal p handles erasing
        self._erase_p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace
        self._orig_transform = OrigRandomErasing(
            p=p, scale=scale, ratio=ratio, value=value, inplace=inplace
        )
    
    def apply(self, image, target):
        return self._orig_transform(image, target)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'p': self._erase_p,
            'scale': self.scale,
            'ratio': self.ratio
        }


class Normalize(BaseTransform):
    """Wrapper for RF-DETR Normalize."""
    
    def __init__(
        self, 
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        p: float = 1.0
    ):
        super().__init__(p=p)
        self.mean = mean
        self.std = std
        self._orig_transform = OrigNormalize(mean, std)
    
    def apply(self, image, target):
        return self._orig_transform(image, target)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'mean': self.mean, 'std': self.std}


# =============================================================================
# Utility functions for box operations
# =============================================================================

def clip_boxes(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Clip boxes to image boundaries.
    
    Args:
        boxes: Boxes in xyxy format, shape (N, 4).
        image_size: (height, width) of the image.
    
    Returns:
        Clipped boxes.
    """
    h, w = image_size
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)  # x coordinates
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)  # y coordinates
    return boxes


def filter_small_boxes(
    boxes: torch.Tensor, 
    min_size: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter out boxes that are too small.
    
    Args:
        boxes: Boxes in xyxy format, shape (N, 4).
        min_size: Minimum width/height in pixels.
    
    Returns:
        Tuple of (filtered_boxes, keep_indices).
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (widths >= min_size) & (heights >= min_size)
    return boxes[keep], keep


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from xyxy to cxcywh format.
    
    Args:
        boxes: Boxes in xyxy format (x1, y1, x2, y2).
    
    Returns:
        Boxes in cxcywh format (cx, cy, w, h).
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([
        (x1 + x2) / 2,  # cx
        (y1 + y2) / 2,  # cy
        x2 - x1,        # w
        y2 - y1,        # h
    ], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from cxcywh to xyxy format.
    
    Args:
        boxes: Boxes in cxcywh format (cx, cy, w, h).
    
    Returns:
        Boxes in xyxy format (x1, y1, x2, y2).
    """
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([
        cx - w / 2,  # x1
        cy - h / 2,  # y1
        cx + w / 2,  # x2
        cy + h / 2,  # y2
    ], dim=-1)


def normalize_boxes(
    boxes: torch.Tensor, 
    image_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Normalize box coordinates to [0, 1] range.
    
    Args:
        boxes: Boxes in xyxy format.
        image_size: (height, width) of the image.
    
    Returns:
        Normalized boxes.
    """
    h, w = image_size
    scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    return boxes / scale


def denormalize_boxes(
    boxes: torch.Tensor, 
    image_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Denormalize box coordinates from [0, 1] range.
    
    Args:
        boxes: Normalized boxes in xyxy format.
        image_size: (height, width) of the image.
    
    Returns:
        Denormalized boxes.
    """
    h, w = image_size
    scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    return boxes * scale
