"""
Mosaic augmentation for RF-DETR training.

This module provides Mosaic augmentation that combines multiple images
into a single training sample, similar to Ultralytics YOLO implementation.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import PIL.Image
import torch

from rfdetr.training.augmentations.base import BaseTransform, clip_boxes, filter_small_boxes


class Mosaic(BaseTransform):
    """
    Mosaic augmentation that combines 4 images into a 2x2 grid.
    
    This augmentation:
    1. Takes the current image and 3 additional random images from dataset
    2. Places them in a 2x2 grid around a random center point
    3. Merges all bounding boxes with adjusted coordinates
    
    Args:
        dataset: Dataset object to sample additional images from.
        imgsz: Target image size (single int for square).
        p: Probability of applying mosaic.
        border: Border offset for the mosaic center.
    
    Example:
        >>> mosaic = Mosaic(dataset=train_dataset, imgsz=640, p=1.0)
        >>> image, target = mosaic(image, target, index=0)
    """
    
    def __init__(
        self,
        dataset: Any = None,
        imgsz: int = 640,
        mosaic_scale: Tuple[float, float] = (0.5, 1.5),
        min_box_size: int = 2,
        fill_color: Tuple[int, int, int] = (114, 114, 114),
        p: float = 1.0,
        border: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(p=p, name='Mosaic')
        
        self.dataset = dataset
        self.imgsz = imgsz
        self.mosaic_scale = mosaic_scale
        self.min_box_size = min_box_size
        self.fill_color = fill_color
        self.border = border if border is not None else (-imgsz // 2, -imgsz // 2)
        
        # Store info about last mosaic for logging
        self._last_indices = []
        self._last_center = (0, 0)
    
    def set_dataset(self, dataset: Any) -> None:
        """Set the dataset for sampling additional images."""
        self.dataset = dataset
    
    def _load_image(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load image and target from dataset.
        
        Args:
            index: Index of the image in dataset.
        
        Returns:
            Tuple of (image as numpy array, target dict).
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not set. Call set_dataset() first or pass dataset to __init__")
        
        # Use get_raw_item to avoid recursive augmentation calls
        if hasattr(self.dataset, 'get_raw_item'):
            img, target = self.dataset.get_raw_item(index)
        else:
            # Fallback for datasets without get_raw_item
            img, target = self.dataset[index]
            if len(img) == 3:  # (img, target, aug_log) tuple
                img, target = img[0], img[1]
        
        # Convert PIL to numpy if needed
        if isinstance(img, PIL.Image.Image):
            img = np.array(img)
        elif isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
        
        return img, target
    
    def _get_mosaic_indices(self, index: int) -> List[int]:
        """
        Get indices for mosaic images.
        
        Args:
            index: Index of the primary image.
        
        Returns:
            List of 4 indices (including the primary).
        """
        if self.dataset is None:
            return [index] * 4
        
        indices = [index]
        n = len(self.dataset)
        
        # Get 3 additional random indices
        for _ in range(3):
            idx = random.randint(0, n - 1)
            indices.append(idx)
        
        return indices
    
    def apply_with_index(
        self,
        image: Union[PIL.Image.Image, np.ndarray],
        target: Dict[str, Any],
        index: int = 0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply mosaic augmentation.
        
        Args:
            image: Input image (will be placed in one of 4 positions).
            target: Target dict with boxes, labels, etc.
            index: Index of the current image in dataset.
        
        Returns:
            Tuple of (mosaic image, merged target).
        """
        # Get 4 image indices
        indices = self._get_mosaic_indices(index)
        self._last_indices = indices
        
        # Mosaic output size
        s = self.imgsz
        
        # Random center point
        yc = int(random.uniform(s * self.mosaic_scale[0], s * self.mosaic_scale[1]))
        xc = int(random.uniform(s * self.mosaic_scale[0], s * self.mosaic_scale[1]))
        self._last_center = (xc, yc)
        
        # Initialize mosaic image and labels
        mosaic_img = np.full((s * 2, s * 2, 3), self.fill_color, dtype=np.uint8)
        mosaic_boxes = []
        mosaic_labels = []
        
        # Process 4 images
        for i, idx in enumerate(indices):
            # Load image
            if i == 0:
                # Use provided image for first position
                if isinstance(image, PIL.Image.Image):
                    img = np.array(image)
                else:
                    img = image.copy()
                img_target = target
            else:
                # Load from dataset
                try:
                    img, img_target = self._load_image(idx)
                except Exception:
                    # Fallback to provided image if loading fails
                    if isinstance(image, PIL.Image.Image):
                        img = np.array(image)
                    else:
                        img = image.copy()
                    img_target = target
            
            h, w = img.shape[:2]
            
            # Determine placement based on position
            if i == 0:  # Top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # Top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # Bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # Bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            # Place image piece in mosaic
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # Calculate padding offset
            padw = x1a - x1b
            padh = y1a - y1b
            
            # Transform boxes
            if 'boxes' in img_target and len(img_target['boxes']) > 0:
                boxes = img_target['boxes'].clone() if isinstance(img_target['boxes'], torch.Tensor) else torch.tensor(img_target['boxes'])
                labels = img_target['labels'].clone() if isinstance(img_target['labels'], torch.Tensor) else torch.tensor(img_target['labels'])
                
                # Add offset
                boxes[:, 0] += padw  # x1
                boxes[:, 1] += padh  # y1
                boxes[:, 2] += padw  # x2
                boxes[:, 3] += padh  # y2
                
                mosaic_boxes.append(boxes)
                mosaic_labels.append(labels)
        
        # Merge all boxes and labels
        if mosaic_boxes:
            final_boxes = torch.cat(mosaic_boxes, dim=0)
            final_labels = torch.cat(mosaic_labels, dim=0)
            
            # Clip to mosaic bounds
            final_boxes = clip_boxes(final_boxes, (s * 2, s * 2))
            
            # Filter small/invalid boxes
            final_boxes, keep = filter_small_boxes(final_boxes, min_size=self.min_box_size)
            final_labels = final_labels[keep]
        else:
            final_boxes = torch.zeros((0, 4), dtype=torch.float32)
            final_labels = torch.zeros((0,), dtype=torch.int64)
        
        # Crop to final size (center crop from 2s x 2s to s x s)
        # This simulates the border effect
        crop_x1 = (s * 2 - s) // 2
        crop_y1 = (s * 2 - s) // 2
        crop_x2 = crop_x1 + s
        crop_y2 = crop_y1 + s
        
        mosaic_img = mosaic_img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Adjust boxes for crop
        if len(final_boxes) > 0:
            final_boxes[:, 0] -= crop_x1
            final_boxes[:, 1] -= crop_y1
            final_boxes[:, 2] -= crop_x1
            final_boxes[:, 3] -= crop_y1
            
            # Clip again after crop
            final_boxes = clip_boxes(final_boxes, (s, s))
            final_boxes, keep = filter_small_boxes(final_boxes, min_size=self.min_box_size)
            final_labels = final_labels[keep]
        
        # Build output target
        new_target = {
            'boxes': final_boxes,
            'labels': final_labels,
            'size': torch.tensor([s, s]),
            'area': (final_boxes[:, 2] - final_boxes[:, 0]) * (final_boxes[:, 3] - final_boxes[:, 1]) if len(final_boxes) > 0 else torch.zeros(0),
        }
        
        # Copy other fields from original target
        for key in ['image_id', 'iscrowd']:
            if key in target:
                if key == 'iscrowd' and len(final_labels) > 0:
                    new_target[key] = torch.zeros(len(final_labels), dtype=torch.int64)
                else:
                    new_target[key] = target[key]
        
        return mosaic_img, new_target
    
    def apply(
        self,
        image: Union[PIL.Image.Image, np.ndarray],
        target: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply mosaic (without index - uses image 4 times).
        
        For proper mosaic with different images, use apply_with_index().
        """
        return self.apply_with_index(image, target, index=0)
    
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray],
        target: Dict[str, Any],
        index: int = 0
    ) -> Tuple[Union[PIL.Image.Image, np.ndarray], Dict[str, Any]]:
        """
        Call mosaic with optional index.
        
        Args:
            image: Input image.
            target: Target dict.
            index: Index in dataset (for sampling other images).
        
        Returns:
            Tuple of (mosaic_image, merged_target).
        """
        if random.random() < self.p:
            image, target = self.apply_with_index(image, target, index)
            self._last_result = {
                'name': self.name,
                'applied': True,
                'parameters': self.get_parameters()
            }
        else:
            self._last_result = {
                'name': self.name,
                'applied': False,
                'parameters': {}
            }
        
        return image, target
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'imgsz': self.imgsz,
            'indices': self._last_indices,
            'center': self._last_center,
        }
    
    def __getstate__(self):
        """Pickle support: remove dataset reference."""
        state = self.__dict__.copy()
        state['dataset'] = None
        return state
    
    def __setstate__(self, state):
        """Pickle support: restore state."""
        self.__dict__.update(state)


class Mosaic9(BaseTransform):
    """
    Mosaic9 augmentation that combines 9 images into a 3x3 grid.
    
    Similar to Mosaic but uses 9 images for more variety.
    
    Args:
        dataset: Dataset object to sample additional images from.
        imgsz: Target image size.
        p: Probability of applying mosaic.
    """
    
    def __init__(
        self,
        dataset: Any = None,
        imgsz: int = 640,
        min_box_size: int = 2,
        fill_color: Tuple[int, int, int] = (114, 114, 114),
        p: float = 1.0,
    ):
        super().__init__(p=p, name='Mosaic9')
        
        self.dataset = dataset
        self.imgsz = imgsz
        self.min_box_size = min_box_size
        self.fill_color = fill_color
        self._last_indices = []
    
    def set_dataset(self, dataset: Any) -> None:
        """Set the dataset for sampling additional images."""
        self.dataset = dataset
    
    def apply_with_index(
        self,
        image: Union[PIL.Image.Image, np.ndarray],
        target: Dict[str, Any],
        index: int = 0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply 3x3 mosaic augmentation."""
        s = self.imgsz
        
        # Get 9 indices
        if self.dataset is not None:
            n = len(self.dataset)
            indices = [index] + [random.randint(0, n - 1) for _ in range(8)]
        else:
            indices = [index] * 9
        
        self._last_indices = indices
        
        # Create 3x3 mosaic
        mosaic_img = np.full((s * 3, s * 3, 3), self.fill_color, dtype=np.uint8)
        mosaic_boxes = []
        mosaic_labels = []
        
        for i, idx in enumerate(indices):
            row = i // 3
            col = i % 3
            
            # Load image
            if i == 0:
                if isinstance(image, PIL.Image.Image):
                    img = np.array(image)
                else:
                    img = image.copy()
                img_target = target
            else:
                try:
                    img, img_target = self.dataset[idx]
                    if isinstance(img, PIL.Image.Image):
                        img = np.array(img)
                except Exception:
                    if isinstance(image, PIL.Image.Image):
                        img = np.array(image)
                    else:
                        img = image.copy()
                    img_target = target
            
            # Resize to fit in grid cell
            h, w = img.shape[:2]
            scale = s / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img, (new_w, new_h))
            
            # Place in grid
            y1 = row * s + (s - new_h) // 2
            x1 = col * s + (s - new_w) // 2
            y2 = y1 + new_h
            x2 = x1 + new_w
            
            mosaic_img[y1:y2, x1:x2] = img_resized
            
            # Transform boxes
            if 'boxes' in img_target and len(img_target['boxes']) > 0:
                boxes = img_target['boxes'].clone() if isinstance(img_target['boxes'], torch.Tensor) else torch.tensor(img_target['boxes'])
                labels = img_target['labels'].clone() if isinstance(img_target['labels'], torch.Tensor) else torch.tensor(img_target['labels'])
                
                # Scale and offset boxes
                boxes = boxes * scale
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1
                
                mosaic_boxes.append(boxes)
                mosaic_labels.append(labels)
        
        # Merge boxes
        if mosaic_boxes:
            final_boxes = torch.cat(mosaic_boxes, dim=0)
            final_labels = torch.cat(mosaic_labels, dim=0)
            
            # Clip to mosaic bounds
            final_boxes = clip_boxes(final_boxes, (s * 3, s * 3))
            final_boxes, keep = filter_small_boxes(final_boxes, min_size=self.min_box_size)
            final_labels = final_labels[keep]
        else:
            final_boxes = torch.zeros((0, 4), dtype=torch.float32)
            final_labels = torch.zeros((0,), dtype=torch.int64)
        
        # Center crop to final size
        crop_start = s
        mosaic_img = mosaic_img[crop_start:crop_start + s, crop_start:crop_start + s]
        
        # Adjust boxes
        if len(final_boxes) > 0:
            final_boxes[:, [0, 2]] -= crop_start
            final_boxes[:, [1, 3]] -= crop_start
            
            final_boxes = clip_boxes(final_boxes, (s, s))
            final_boxes, keep = filter_small_boxes(final_boxes, min_size=self.min_box_size)
            final_labels = final_labels[keep]
        
        new_target = {
            'boxes': final_boxes,
            'labels': final_labels,
            'size': torch.tensor([s, s]),
            'area': (final_boxes[:, 2] - final_boxes[:, 0]) * (final_boxes[:, 3] - final_boxes[:, 1]) if len(final_boxes) > 0 else torch.zeros(0),
        }
        
        return mosaic_img, new_target
    
    def apply(self, image, target):
        return self.apply_with_index(image, target, 0)
    
    def __call__(self, image, target, index=0):
        if random.random() < self.p:
            return self.apply_with_index(image, target, index)
        return image, target
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'imgsz': self.imgsz, 'indices': self._last_indices}
    
    def __getstate__(self):
        """Pickle support: remove dataset reference."""
        state = self.__dict__.copy()
        state['dataset'] = None
        return state
    
    def __setstate__(self, state):
        """Pickle support: restore state."""
        self.__dict__.update(state)
