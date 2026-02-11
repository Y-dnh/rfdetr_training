"""
Random Erasing augmentation for RF-DETR training.

Random erasing randomly selects a rectangle region and erases its pixels
with random values, helping the model become robust to occlusion.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

from rfdetr.training.augmentations.base import BaseTransform


class RandomErasing(BaseTransform):
    """
    Box-aware Random Erasing augmentation.
    
    Randomly selects a rectangle region in an image and erases its pixels
    with random values. This version is box-aware and ensures that erasing
    does not destroy too much of any bounding box.
    
    Args:
        p: Probability of performing erasing.
        scale: Range of proportion of erased area against input image.
        ratio: Range of aspect ratio of erased area.
        value: Erasing value. Can be:
               - 'random': Fill with random noise
               - 0: Fill with black
               - 128: Fill with gray (default, recommended)
               - 255: Fill with white
               - tuple of 3 floats: Fill with RGB value
        inplace: Whether to do erasing in-place.
        min_visible_ratio: Minimum ratio of box area that must remain visible (0-1).
                          If erasing would hide more than this, it will be skipped
                          or adjusted. Default 0.5 = at least 50% of box must remain.
        min_box_size: Minimum size (pixels) of remaining box dimension.
                     If box would become smaller, erasing is adjusted. Default 20.
        box_aware: If True, check overlap with boxes and adjust erasing accordingly.
                  If False, behaves like original RandomErasing.
    
    Reference:
        Random Erasing Data Augmentation
        https://arxiv.org/abs/1708.04896
    
    Example:
        >>> erasing = RandomErasing(p=0.5, scale=(0.02, 0.33), min_visible_ratio=0.5)
        >>> image, target = erasing(image, target)
    """
    
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: Union[str, float, Tuple[float, float, float]] = 128,
        inplace: bool = False,
        min_visible_ratio: float = 0.5,
        min_box_size: int = 20,
        box_aware: bool = True,
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
        self.min_visible_ratio = min_visible_ratio
        self.min_box_size = min_box_size
        self.box_aware = box_aware
        
        self._last_bbox = None
        self._last_value = None
        self._last_skipped = False
    
    def _get_params(
        self, 
        img_h: int, 
        img_w: int
    ) -> Optional[Tuple[int, int, int, int]]:
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
    
    def _check_box_overlap(
        self,
        erase_box: Tuple[int, int, int, int],  # x1, y1, x2, y2
        boxes: np.ndarray,  # Nx4 array of [x1, y1, x2, y2]
        img_h: int,
        img_w: int,
    ) -> Tuple[bool, List[int]]:
        """
        Check if erasing would destroy too much of any box.
        
        Returns:
            Tuple of (is_valid, affected_box_indices)
        """
        ex1, ey1, ex2, ey2 = erase_box
        affected_indices = []
        
        for i, box in enumerate(boxes):
            bx1, by1, bx2, by2 = box
            
            # Calculate intersection
            ix1 = max(ex1, bx1)
            iy1 = max(ey1, by1)
            ix2 = min(ex2, bx2)
            iy2 = min(ey2, by2)
            
            if ix1 < ix2 and iy1 < iy2:
                # There is overlap
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                box_area = (bx2 - bx1) * (by2 - by1)
                
                if box_area > 0:
                    overlap_ratio = intersection_area / box_area
                    
                    # Check if too much of the box would be erased
                    if overlap_ratio > (1.0 - self.min_visible_ratio):
                        affected_indices.append(i)
                    
                    # Also check if remaining box would be too small
                    remaining_w = (bx2 - bx1) - (ix2 - ix1) if ix1 <= bx1 or ix2 >= bx2 else (bx2 - bx1)
                    remaining_h = (by2 - by1) - (iy2 - iy1) if iy1 <= by1 or iy2 >= by2 else (by2 - by1)
                    
                    if remaining_w < self.min_box_size or remaining_h < self.min_box_size:
                        if overlap_ratio > 0.1:  # Only worry if significant overlap
                            affected_indices.append(i)
        
        # Remove duplicates
        affected_indices = list(set(affected_indices))
        is_valid = len(affected_indices) == 0
        
        return is_valid, affected_indices
    
    def apply(
        self, 
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor], 
        target: Dict[str, Any]
    ) -> Tuple[Union[PIL.Image.Image, np.ndarray, torch.Tensor], Dict[str, Any]]:
        """
        Apply random erasing to the image.
        
        Args:
            image: Input image.
            target: Target dict with 'boxes' key.
        
        Returns:
            Tuple of (erased_image, target).
        """
        self._last_skipped = False
        
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
        
        # Get boxes for box-aware mode
        boxes = None
        if self.box_aware and 'boxes' in target:
            boxes_data = target['boxes']
            if isinstance(boxes_data, torch.Tensor):
                boxes = boxes_data.numpy()
            else:
                boxes = np.array(boxes_data)
            
            # Convert normalized coords to pixel coords if needed
            if boxes.size > 0 and boxes.max() <= 1.0:
                boxes = boxes.copy()
                boxes[:, [0, 2]] *= w
                boxes[:, [1, 3]] *= h
        
        # Try to find valid erasing region
        max_attempts = 20 if self.box_aware and boxes is not None and len(boxes) > 0 else 10
        
        for attempt in range(max_attempts):
            params = self._get_params(h, w)
            
            if params is None:
                continue
            
            y, x, eh, ew = params
            erase_box = (x, y, x + ew, y + eh)
            
            # Check box overlap in box-aware mode
            if self.box_aware and boxes is not None and len(boxes) > 0:
                is_valid, affected = self._check_box_overlap(erase_box, boxes, h, w)
                
                if not is_valid:
                    # Try to find a region that doesn't affect boxes too much
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        # Last attempt - skip erasing
                        self._last_bbox = None
                        self._last_skipped = True
                        return image, target
            
            # Valid region found - apply erasing
            self._last_bbox = erase_box
            
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
            break
        else:
            # No valid params found
            self._last_bbox = None
            return image, target
        
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
            'box_aware': self.box_aware,
            'skipped': self._last_skipped,
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
