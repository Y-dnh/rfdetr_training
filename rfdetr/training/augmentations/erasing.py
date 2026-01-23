"""
Random Erasing augmentation for RF-DETR training.

Random erasing randomly selects a rectangle region and erases its pixels
with random values, helping the model become robust to occlusion.
"""

import random
from typing import Any, Dict, Tuple, Union

import numpy as np
import PIL.Image
import torch

from rfdetr.training.augmentations.base import BaseTransform


class RandomErasing(BaseTransform):
    """
    Random Erasing augmentation.
    
    Randomly selects a rectangle region in an image and erases its pixels
    with random values. This helps improve model robustness to partial occlusion.
    
    Note: This augmentation does NOT modify bounding boxes as it only
    affects pixel values within the image.
    
    Args:
        p: Probability of performing erasing.
        scale: Range of proportion of erased area against input image.
        ratio: Range of aspect ratio of erased area.
        value: Erasing value. Can be:
               - 'random': Fill with random values
               - float: Fill with single value (grayscale)
               - tuple of 3 floats: Fill with RGB value
        inplace: Whether to do erasing in-place.
    
    Reference:
        Random Erasing Data Augmentation
        https://arxiv.org/abs/1708.04896
    
    Example:
        >>> erasing = RandomErasing(p=0.5, scale=(0.02, 0.33))
        >>> image, target = erasing(image, target)
    """
    
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: Union[str, float, Tuple[float, float, float]] = 'random',
        inplace: bool = False,
    ):
        super().__init__(p=p, name='RandomErasing')
        
        if not 0.0 <= scale[0] <= scale[1] <= 1.0:
            raise ValueError(f"scale must be in range [0, 1], got {scale}")
        if ratio[0] <= 0 or ratio[1] <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace
        
        self._last_bbox = None
        self._last_value = None
    
    def _get_params(
        self, 
        img_h: int, 
        img_w: int
    ) -> Tuple[int, int, int, int]:
        """
        Get random erasing parameters.
        
        Args:
            img_h: Image height.
            img_w: Image width.
        
        Returns:
            Tuple of (y, x, h, w) for the erasing region, or None if no valid region found.
        """
        img_area = img_h * img_w
        
        for _ in range(10):  # Try up to 10 times
            erase_area = random.uniform(self.scale[0], self.scale[1]) * img_area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            h = int(round(np.sqrt(erase_area * aspect_ratio)))
            w = int(round(np.sqrt(erase_area / aspect_ratio)))
            
            if h < img_h and w < img_w:
                y = random.randint(0, img_h - h)
                x = random.randint(0, img_w - w)
                return y, x, h, w
        
        return None
    
    def apply(
        self, 
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor], 
        target: Dict[str, Any]
    ) -> Tuple[Union[PIL.Image.Image, np.ndarray, torch.Tensor], Dict[str, Any]]:
        """
        Apply random erasing to the image.
        
        Args:
            image: Input image.
            target: Target dict (unchanged).
        
        Returns:
            Tuple of (erased_image, target).
        """
        # Handle different input types
        is_pil = isinstance(image, PIL.Image.Image)
        is_tensor = isinstance(image, torch.Tensor)
        
        if is_pil:
            img = np.array(image)
        elif is_tensor:
            # Assume CxHxW tensor
            img = image.permute(1, 2, 0).numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
        else:
            img = image if self.inplace else image.copy()
        
        h, w = img.shape[:2]
        
        # Get erasing parameters
        params = self._get_params(h, w)
        
        if params is None:
            self._last_bbox = None
            return image, target
        
        y, x, eh, ew = params
        self._last_bbox = (x, y, x + ew, y + eh)
        
        # Generate erasing value
        if self.value == 'random':
            erase_value = np.random.randint(0, 256, (eh, ew, img.shape[2]), dtype=np.uint8)
            self._last_value = 'random'
        elif isinstance(self.value, (int, float)):
            erase_value = np.full((eh, ew, img.shape[2]), self.value, dtype=np.uint8)
            self._last_value = self.value
        else:
            erase_value = np.full((eh, ew, img.shape[2]), self.value, dtype=np.uint8)
            self._last_value = self.value
        
        # Apply erasing
        img[y:y + eh, x:x + ew] = erase_value
        
        # Convert back to original type
        if is_pil:
            img = PIL.Image.fromarray(img)
        elif is_tensor:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img, target
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'scale': self.scale,
            'ratio': self.ratio,
            'bbox': self._last_bbox,
            'value': self._last_value,
        }


class GridErasing(BaseTransform):
    """
    Grid-based random erasing that erases multiple small regions.
    
    Args:
        p: Probability of applying.
        num_patches: Range of number of patches to erase (min, max).
        patch_size: Range of patch size as fraction of image (min, max).
        value: Erasing value ('random' or color value).
    """
    
    def __init__(
        self,
        p: float = 0.5,
        num_patches: Tuple[int, int] = (1, 5),
        patch_size: Tuple[float, float] = (0.02, 0.1),
        value: Union[str, float] = 'random',
    ):
        super().__init__(p=p, name='GridErasing')
        
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.value = value
        self._last_patches = []
    
    def apply(
        self, 
        image: Union[PIL.Image.Image, np.ndarray], 
        target: Dict[str, Any]
    ) -> Tuple[Union[PIL.Image.Image, np.ndarray], Dict[str, Any]]:
        """Apply grid erasing."""
        is_pil = isinstance(image, PIL.Image.Image)
        
        if is_pil:
            img = np.array(image)
        else:
            img = image.copy()
        
        h, w = img.shape[:2]
        
        # Random number of patches
        n_patches = random.randint(*self.num_patches)
        self._last_patches = []
        
        for _ in range(n_patches):
            # Random patch size
            patch_scale = random.uniform(*self.patch_size)
            patch_h = int(h * patch_scale)
            patch_w = int(w * patch_scale)
            
            if patch_h > 0 and patch_w > 0:
                # Random position
                y = random.randint(0, max(0, h - patch_h))
                x = random.randint(0, max(0, w - patch_w))
                
                # Generate value
                if self.value == 'random':
                    erase_value = np.random.randint(0, 256, (patch_h, patch_w, img.shape[2]), dtype=np.uint8)
                else:
                    erase_value = np.full((patch_h, patch_w, img.shape[2]), self.value, dtype=np.uint8)
                
                img[y:y + patch_h, x:x + patch_w] = erase_value
                self._last_patches.append((x, y, x + patch_w, y + patch_h))
        
        if is_pil:
            img = PIL.Image.fromarray(img)
        
        return img, target
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'num_patches': self.num_patches,
            'patch_size': self.patch_size,
            'patches': self._last_patches,
        }
