"""
Augmentation Pipeline for RF-DETR training.

This module provides the main AugmentationPipeline class that composes
all augmentations in the correct order.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

from rfdetr.training.augmentations.base import BaseTransform, Compose, ToTensor, Normalize
from rfdetr.training.augmentations.color import RandomHSV
from rfdetr.training.augmentations.geometric import RandomFlip, RandomPerspective, LetterBox
from rfdetr.training.augmentations.mosaic import Mosaic
from rfdetr.training.augmentations.mixup import MixUp, CutMix
from rfdetr.training.augmentations.erasing import RandomErasing
from rfdetr.training.utils.config import AugmentationConfig


class AugmentationPipeline:
    """
    Complete augmentation pipeline for RF-DETR training.
    
    This pipeline applies augmentations in the correct order:
    1. Mosaic (combines 4 images)
    2. RandomPerspective (rotation, scale, shear, perspective)
    3. MixUp (alpha blending with another image)
    4. CutMix (cut and paste regions)
    5. RandomHSV (color augmentation)
    6. RandomFlip (horizontal/vertical)
    7. RandomErasing (random region erasing)
    8. LetterBox (resize with padding)
    9. Normalize (mean/std normalization)
    10. ToTensor (convert to tensor)
    
    Args:
        config: AugmentationConfig with all augmentation parameters.
        dataset: Dataset for mosaic/mixup/cutmix (sampling other images).
        is_train: Whether this is for training (True) or validation (False).
        log_augmentations: Whether to log applied augmentations.
    
    Example:
        >>> config = AugmentationConfig(mosaic=1.0, hsv_h=0.015, fliplr=0.5)
        >>> pipeline = AugmentationPipeline(config, dataset=train_dataset)
        >>> image, target = pipeline(image, target, index=0)
    """
    
    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        dataset: Any = None,
        is_train: bool = True,
        log_augmentations: bool = False,
    ):
        self.config = config or AugmentationConfig()
        self.dataset = dataset
        self.is_train = is_train
        self.log_augmentations = log_augmentations
        
        # Current epoch (for close_mosaic)
        self._current_epoch = 0
        self._total_epochs = 100
        
        # Build transforms
        self._build_transforms()
        
        # Last augmentation log
        self._last_log = []
    
    def _build_transforms(self) -> None:
        """Build the augmentation transforms based on config."""
        cfg = self.config
        
        if self.is_train:
            # Training augmentations
            self.mosaic = Mosaic(
                dataset=self.dataset,
                imgsz=cfg.imgsz,
                p=cfg.mosaic,
            )
            
            self.perspective = RandomPerspective(
                degrees=cfg.degrees,
                translate=cfg.translate,
                scale=cfg.scale,
                shear=cfg.shear,
                perspective=cfg.perspective,
                p=1.0 if (cfg.degrees > 0 or cfg.translate > 0 or cfg.scale > 0 
                          or cfg.shear > 0 or cfg.perspective > 0) else 0.0,
            )
            
            self.mixup = MixUp(
                dataset=self.dataset,
                alpha=32.0,
                p=cfg.mixup,
            )
            
            self.cutmix = CutMix(
                dataset=self.dataset,
                alpha=1.0,
                p=cfg.cutmix,
            )
            
            self.hsv = RandomHSV(
                h_gain=cfg.hsv_h,
                s_gain=cfg.hsv_s,
                v_gain=cfg.hsv_v,
                p=1.0 if (cfg.hsv_h > 0 or cfg.hsv_s > 0 or cfg.hsv_v > 0) else 0.0,
            )
            
            self.flip_lr = RandomFlip(
                p=cfg.fliplr,
                direction='horizontal',
            )
            
            self.flip_ud = RandomFlip(
                p=cfg.flipud,
                direction='vertical',
            )
            
            self.erasing = RandomErasing(
                p=cfg.erasing,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
            )
        
        # Common transforms (train and val)
        self.letterbox = LetterBox(
            new_shape=cfg.imgsz,
            p=1.0,
        )
        
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        
        self.to_tensor = ToTensor()
    
    def set_dataset(self, dataset: Any) -> None:
        """Set dataset for augmentations that need it."""
        self.dataset = dataset
        if self.is_train:
            self.mosaic.set_dataset(dataset)
            self.mixup.set_dataset(dataset)
            self.cutmix.set_dataset(dataset)
    
    def set_epoch(self, epoch: int, total_epochs: Optional[int] = None) -> None:
        """
        Set current epoch for close_mosaic handling.
        
        Args:
            epoch: Current epoch number.
            total_epochs: Total number of epochs (optional, uses previous value if not provided).
        """
        self._current_epoch = epoch
        if total_epochs is not None:
            self._total_epochs = total_epochs
        
        # Debug: log mosaic status change
        if self.is_train:
            remaining = self._total_epochs - self._current_epoch
            mosaic_on = remaining > self.config.close_mosaic
            if not mosaic_on:
                print(f"[Pipeline] Epoch {epoch}: Mosaic DISABLED (remaining={remaining} <= close_mosaic={self.config.close_mosaic})")
    
    def _should_apply_mosaic(self) -> bool:
        """Check if mosaic should be applied based on close_mosaic."""
        if not self.is_train:
            return False
        
        remaining_epochs = self._total_epochs - self._current_epoch
        return remaining_epochs > self.config.close_mosaic
    
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray],
        target: Dict[str, Any],
        index: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply augmentation pipeline.
        
        Args:
            image: Input image.
            target: Target dict with boxes, labels, etc.
            index: Index of image in dataset (for mosaic/mixup).
        
        Returns:
            Tuple of (augmented_tensor, augmented_target).
        """
        self._last_log = []
        
        if self.is_train:
            # 1. Mosaic
            if self._should_apply_mosaic():
                image, target = self.mosaic(image, target, index)
                self._log_augmentation('Mosaic', self.mosaic)
            
            # 2. Perspective/Affine
            image, target = self.perspective(image, target)
            self._log_augmentation('RandomPerspective', self.perspective)
            
            # 3. MixUp
            image, target = self.mixup(image, target, index)
            self._log_augmentation('MixUp', self.mixup)
            
            # 4. CutMix
            image, target = self.cutmix(image, target, index)
            self._log_augmentation('CutMix', self.cutmix)
            
            # 5. HSV
            image, target = self.hsv(image, target)
            self._log_augmentation('RandomHSV', self.hsv)
            
            # 6. Flip
            image, target = self.flip_lr(image, target)
            self._log_augmentation('FlipLR', self.flip_lr)
            
            image, target = self.flip_ud(image, target)
            self._log_augmentation('FlipUD', self.flip_ud)
        
        # 7. LetterBox (resize with padding)
        image, target = self.letterbox(image, target)
        
        # 8. Random Erasing (after resize, on final image)
        if self.is_train:
            image, target = self.erasing(image, target)
            self._log_augmentation('RandomErasing', self.erasing)
        
        # 9. ToTensor
        image, target = self.to_tensor(image, target)
        
        # 10. Normalize
        image, target = self.normalize(image, target)
        
        return image, target
    
    def _log_augmentation(self, name: str, transform: BaseTransform) -> None:
        """Log augmentation if logging is enabled."""
        if self.log_augmentations:
            result = transform.get_last_result()
            if result is not None:
                # Support both dict and object with attributes
                if isinstance(result, dict):
                    applied = result.get('applied', False)
                    parameters = result.get('parameters', {}) if applied else {}
                else:
                    applied = result.applied
                    parameters = result.parameters if applied else {}
                self._last_log.append({
                    'name': name,
                    'applied': applied,
                    'parameters': parameters,
                })
    
    def get_last_log(self) -> Optional[Dict[str, Any]]:
        """Get log of last augmentation call as a single dict."""
        if not self._last_log:
            return None
        # Return a dict with augmentations list
        return {
            'augmentations': self._last_log,
            'applied': [log for log in self._last_log if log.get('applied', False)],
        }
    
    def get_applied_augmentations(self) -> List[Dict[str, Any]]:
        """Get only augmentations that were actually applied."""
        return [log for log in self._last_log if log.get('applied', False)]
    
    def __getstate__(self):
        """Pickle support: remove dataset reference to avoid circular ref."""
        state = self.__dict__.copy()
        # Remove dataset to avoid circular reference when pickling
        state['dataset'] = None
        return state
    
    def __setstate__(self, state):
        """Pickle support: restore state and rebuild transforms."""
        self.__dict__.update(state)
        # Rebuild transforms without dataset reference
        # Dataset will be set later via set_dataset() if needed
        self._build_transforms()


class ValidationPipeline:
    """
    Simple validation pipeline without augmentations.
    
    Only applies:
    1. LetterBox (resize with padding)
    2. ToTensor
    3. Normalize
    
    Args:
        imgsz: Target image size.
    """
    
    def __init__(self, imgsz: int = 640):
        self.imgsz = imgsz
        
        self.letterbox = LetterBox(new_shape=imgsz, p=1.0)
        self.to_tensor = ToTensor()
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray],
        target: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply validation pipeline."""
        image, target = self.letterbox(image, target)
        image, target = self.to_tensor(image, target)
        image, target = self.normalize(image, target)
        return image, target
