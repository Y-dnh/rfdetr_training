"""
Geometric augmentations for RF-DETR training.

This module provides geometric transformations:
- RandomFlip: Horizontal and vertical flips
- RandomPerspective: Rotation, translation, scale, shear, perspective
- RandomAffine: Affine transformations
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF

from rfdetr.training.augmentations.base import BaseTransform, clip_boxes, filter_small_boxes


class RandomFlip(BaseTransform):
    """
    Random horizontal or vertical flip.
    
    Args:
        p: Probability of applying the flip.
        direction: 'horizontal', 'vertical', or 'both'.
    """
    
    def __init__(self, p: float = 0.5, direction: str = 'horizontal'):
        super().__init__(p=p, name=f'RandomFlip_{direction}')
        
        if direction not in ['horizontal', 'vertical', 'both']:
            raise ValueError(f"direction must be 'horizontal', 'vertical', or 'both', got {direction}")
        
        self.direction = direction
        self._last_flipped_h = False
        self._last_flipped_v = False
    
    def apply(
        self, 
        image: Union[PIL.Image.Image, np.ndarray], 
        target: Dict[str, Any]
    ) -> Tuple[Union[PIL.Image.Image, np.ndarray], Dict[str, Any]]:
        """Apply flip transformation."""
        is_pil = isinstance(image, PIL.Image.Image)
        
        if is_pil:
            w, h = image.size
        else:
            h, w = image.shape[:2]
        
        # Determine which flips to apply
        do_hflip = self.direction in ['horizontal', 'both']
        do_vflip = self.direction in ['vertical', 'both']
        
        # For 'both', randomly choose one
        if self.direction == 'both':
            do_hflip = random.random() < 0.5
            do_vflip = not do_hflip
        
        self._last_flipped_h = do_hflip
        self._last_flipped_v = do_vflip
        
        target = target.copy()
        
        if do_hflip:
            # Horizontal flip
            if is_pil:
                image = TF.hflip(image)
            else:
                image = np.ascontiguousarray(np.fliplr(image))
            
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes'].clone()
                # x1, y1, x2, y2 -> w - x2, y1, w - x1, y2
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = boxes
        
        if do_vflip:
            # Vertical flip
            if is_pil:
                image = TF.vflip(image)
            else:
                image = np.ascontiguousarray(np.flipud(image))
            
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes'].clone()
                # x1, y1, x2, y2 -> x1, h - y2, x2, h - y1
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                target['boxes'] = boxes
        
        return image, target
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'direction': self.direction,
            'flipped_h': self._last_flipped_h,
            'flipped_v': self._last_flipped_v,
        }


class RandomPerspective(BaseTransform):
    """
    Random perspective transformation including rotation, translation, scale, shear.
    
    This is similar to Ultralytics' RandomPerspective augmentation.
    
    Args:
        degrees: Maximum rotation angle in degrees.
        translate: Maximum translation as fraction of image size.
        scale: Scale range as (1 - scale, 1 + scale).
        shear: Maximum shear angle in degrees.
        perspective: Perspective distortion factor.
        border: Border size (for mosaic integration).
        p: Probability of applying the augmentation.
    """
    
    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        border: Tuple[int, int] = (0, 0),
        fill_color: Tuple[int, int, int] = (114, 114, 114),
        p: float = 1.0
    ):
        super().__init__(p=p, name='RandomPerspective')
        
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        self.fill_color = fill_color
        
        # Store last transformation matrix
        self._last_matrix = None
        self._last_params = {}
    
    def apply(
        self, 
        image: Union[PIL.Image.Image, np.ndarray], 
        target: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply perspective transformation."""
        # Convert to numpy
        if isinstance(image, PIL.Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        height, width = img.shape[:2]
        
        # Calculate new size with border
        new_height = height + self.border[0] * 2
        new_width = width + self.border[1] * 2
        
        # Center
        cx, cy = new_width / 2, new_height / 2
        
        # Perspective
        if self.perspective > 0:
            P = np.eye(3, dtype=np.float32)
            P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective
            P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective
        else:
            P = np.eye(3, dtype=np.float32)
        
        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        angle = random.uniform(-self.degrees, self.degrees)
        scale = random.uniform(1 - self.scale, 1 + self.scale)
        
        R[:2] = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        
        # Shear
        S = np.eye(3, dtype=np.float32)
        shear_x = math.tan(math.radians(random.uniform(-self.shear, self.shear)))
        shear_y = math.tan(math.radians(random.uniform(-self.shear, self.shear)))
        S[0, 1] = shear_x
        S[1, 0] = shear_y
        
        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * new_width - cx + self.border[1]
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * new_height - cy + self.border[0]
        
        # Combined transformation matrix
        M = T @ S @ R @ P
        
        self._last_matrix = M
        self._last_params = {
            'angle': angle,
            'scale': scale,
            'shear_x': shear_x,
            'shear_y': shear_y,
            'translate_x': T[0, 2],
            'translate_y': T[1, 2],
        }
        
        # Apply transformation to image
        if self.perspective > 0 or self.border != (0, 0):
            # Perspective transformation
            img = cv2.warpPerspective(
                img, M, (new_width, new_height),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.fill_color,
            )
        else:
            # Affine transformation (faster)
            img = cv2.warpAffine(
                img, M[:2], (new_width, new_height),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.fill_color,
            )
        
        # Transform boxes
        target = target.copy()
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            n = len(boxes)
            
            # Convert boxes to corner points
            # xyxy -> 4 corners (x1,y1), (x2,y1), (x2,y2), (x1,y2)
            corners = np.zeros((n, 4, 2), dtype=np.float32)
            corners[:, 0] = boxes[:, [0, 1]].numpy()  # top-left
            corners[:, 1] = boxes[:, [2, 1]].numpy()  # top-right
            corners[:, 2] = boxes[:, [2, 3]].numpy()  # bottom-right
            corners[:, 3] = boxes[:, [0, 3]].numpy()  # bottom-left
            
            # Reshape for transformation
            corners = corners.reshape(-1, 2)
            
            # Apply transformation
            ones = np.ones((corners.shape[0], 1), dtype=np.float32)
            corners_h = np.hstack([corners, ones])  # Homogeneous coordinates
            
            transformed = corners_h @ M.T
            if self.perspective > 0:
                transformed = transformed[:, :2] / transformed[:, 2:3]
            else:
                transformed = transformed[:, :2]
            
            # Reshape back to boxes
            transformed = transformed.reshape(n, 4, 2)
            
            # Get bounding box of transformed corners
            new_boxes = np.zeros((n, 4), dtype=np.float32)
            new_boxes[:, 0] = transformed[:, :, 0].min(axis=1)  # x1
            new_boxes[:, 1] = transformed[:, :, 1].min(axis=1)  # y1
            new_boxes[:, 2] = transformed[:, :, 0].max(axis=1)  # x2
            new_boxes[:, 3] = transformed[:, :, 1].max(axis=1)  # y2
            
            # Clip to image bounds
            new_boxes = torch.from_numpy(new_boxes)
            new_boxes = clip_boxes(new_boxes, (new_height, new_width))
            
            # Filter small boxes
            new_boxes, keep = filter_small_boxes(new_boxes, min_size=2)
            
            target['boxes'] = new_boxes
            
            # Filter other fields
            for key in ['labels', 'area', 'iscrowd']:
                if key in target:
                    target[key] = target[key][keep]
            
            # Update area
            if len(new_boxes) > 0:
                target['area'] = (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1])
        
        # Update size
        target['size'] = torch.tensor([new_height, new_width])
        
        return img, target
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'degrees': self.degrees,
            'translate': self.translate,
            'scale': self.scale,
            'shear': self.shear,
            'perspective': self.perspective,
            **self._last_params
        }


class RandomRotate(BaseTransform):
    """
    Random rotation augmentation.
    
    Args:
        degrees: Maximum rotation angle in degrees (rotates in range [-degrees, degrees]).
        p: Probability of applying the augmentation.
    """
    
    def __init__(self, degrees: float = 10.0, p: float = 0.5):
        super().__init__(p=p, name='RandomRotate')
        self.degrees = degrees
        self._last_angle = 0.0
    
    def apply(
        self, 
        image: Union[PIL.Image.Image, np.ndarray], 
        target: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply rotation transformation."""
        # Use RandomPerspective with only rotation
        perspective = RandomPerspective(
            degrees=self.degrees,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            p=1.0
        )
        
        image, target = perspective.apply(image, target)
        self._last_angle = perspective._last_params.get('angle', 0.0)
        
        return image, target
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'degrees': self.degrees, 'angle': self._last_angle}


class RandomScale(BaseTransform):
    """
    Random scale augmentation.
    
    Args:
        scale_range: Scale range as (min_scale, max_scale).
        p: Probability of applying the augmentation.
    """
    
    def __init__(
        self, 
        scale_range: Tuple[float, float] = (0.5, 1.5), 
        p: float = 0.5
    ):
        super().__init__(p=p, name='RandomScale')
        self.scale_range = scale_range
        self._last_scale = 1.0
    
    def apply(
        self, 
        image: Union[PIL.Image.Image, np.ndarray], 
        target: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply scale transformation."""
        # Convert to numpy
        if isinstance(image, PIL.Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        h, w = img.shape[:2]
        
        # Random scale
        scale = random.uniform(*self.scale_range)
        self._last_scale = scale
        
        # New dimensions
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize image
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Transform boxes
        target = target.copy()
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            boxes = boxes * scale
            target['boxes'] = boxes
            
            # Update area
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Update size
        target['size'] = torch.tensor([new_h, new_w])
        
        return img, target
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'scale_range': self.scale_range, 'scale': self._last_scale}


class LetterBox(BaseTransform):
    """
    Letterbox resize to maintain aspect ratio with padding.
    
    This is commonly used in YOLO models to resize images while
    maintaining aspect ratio by adding gray borders.
    
    Args:
        new_shape: Target shape (height, width) or single int for square.
        color: Padding color (default gray 114, 114, 114).
        auto: Minimum rectangle.
        scale_fill: Stretch to fill.
        scaleup: Allow scaling up.
        stride: Stride for making dimensions divisible.
        p: Probability of applying the augmentation.
    """
    
    def __init__(
        self,
        new_shape: Union[int, Tuple[int, int]] = 640,
        color: Tuple[int, int, int] = (114, 114, 114),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        stride: int = 32,
        p: float = 1.0
    ):
        super().__init__(p=p, name='LetterBox')
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        self.new_shape = new_shape
        self.color = color
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        
        self._last_ratio = (1.0, 1.0)
        self._last_pad = (0, 0)
    
    def apply(
        self, 
        image: Union[PIL.Image.Image, np.ndarray], 
        target: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply letterbox transformation."""
        # Convert to numpy
        if isinstance(image, PIL.Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        shape = img.shape[:2]  # current shape [height, width]
        
        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)
        
        # Compute padding
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]
        
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:
            dw, dh = 0.0, 0.0
            new_unpad = (self.new_shape[1], self.new_shape[0])
            ratio = self.new_shape[1] / shape[1], self.new_shape[0] / shape[0]
        
        dw /= 2
        dh /= 2
        
        self._last_ratio = ratio
        self._last_pad = (dw, dh)
        
        # Resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Add padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=self.color
        )
        
        # Transform boxes
        target = target.copy()
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            
            # Scale boxes
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio[0]
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio[1]
            
            # Add padding offset
            boxes[:, [0, 2]] += dw
            boxes[:, [1, 3]] += dh
            
            target['boxes'] = boxes
            
            # Update area
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Update size
        target['size'] = torch.tensor(img.shape[:2])
        
        return img, target
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'new_shape': self.new_shape,
            'ratio': self._last_ratio,
            'pad': self._last_pad,
        }
