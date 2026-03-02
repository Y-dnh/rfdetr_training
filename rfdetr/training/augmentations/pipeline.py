"""
Augmentation Pipeline for RF-DETR training.

Порядок пайплайну (train):
  1. Mosaic — або наша (mosaic.py), або A.Mosaic з albumentations.
     Якщо use_albumentations_mosaic=False (за замовчуванням): наша Mosaic, 4 зображення з датасету, 2x2 сітка.
     Якщо use_albumentations_mosaic=True: наша Mosaic не викликається; перед albu в target додається
     mosaic_metadata (3 додаткові зображення); A.Mosaic у ALBUMENTATION_CONFIG має бути з p>0 (наприклад p=0.5).
  2. MixUp — наш (mixup.py).
  3. CutMix — наш (mixup.py).
  4. Albumentations — колір, flip, геометрія, CoarseDropout тощо з ALBUMENTATION_CONFIG; при use_albumentations_mosaic
     також A.Mosaic (якщо є в списку і передано mosaic_metadata).
  5. LetterBox — наш (geometric.py).
  6. ToTensor, 7. Normalize — наші (base.py).
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

from rfdetr.training.augmentations.base import BaseTransform, ToTensor, Normalize
from rfdetr.training.augmentations.geometric import LetterBox
from rfdetr.training.augmentations.mosaic import Mosaic
from rfdetr.training.augmentations.mixup import MixUp, CutMix
from rfdetr.training.augmentations.albumentations_wrapper import AlbumentationsWrapper
from rfdetr.training.utils.config import AugmentationConfig


class AugmentationPipeline:
    """
    Complete augmentation pipeline for RF-DETR training.
    
    Order: 1) Mosaic (legacy) 2) MixUp 3) CutMix 4) AlbumentationsWrapper
    (color, flips, geometry, erasing from ALBUMENTATION_CONFIG) 5) LetterBox 6) ToTensor 7) Normalize.
    
    Args:
        config: AugmentationConfig (our custom params: mosaic, mixup, cutmix, imgsz, etc.).
        dataset: Dataset for mosaic/mixup/cutmix (sampling other images).
        albumentation_transforms: List of A.* transform instances (e.g. from get_default_albu_config()).
                                 Mosaic is excluded when building the wrapper (single-image only).
        is_train: Whether this is for training (True) or validation (False).
        log_augmentations: Whether to log applied augmentations.
    """
    
    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        dataset: Any = None,
        albumentation_transforms: Optional[List[Any]] = None,
        is_train: bool = True,
        log_augmentations: bool = False,
    ):
        self.config = config or AugmentationConfig()
        self.dataset = dataset
        self.is_train = is_train
        self.log_augmentations = log_augmentations
        # A.Mosaic потребує mosaic_metadata; якщо use_albumentations_mosaic=True — не викидаємо A.Mosaic і передаємо metadata в __call__
        self._use_albu_mosaic = bool(self.config.use_albumentations_mosaic)
        self._albu_list = self._filter_mosaic_from_albu(
            albumentation_transforms or [], keep_mosaic=self._use_albu_mosaic
        )
        
        # Current epoch (for close_mosaic)
        self._current_epoch = 0
        self._total_epochs = 100
        self._mosaic_was_enabled = None
        
        self._build_transforms()
        self._last_log: List[Dict[str, Any]] = []
    
    @staticmethod
    def _filter_mosaic_from_albu(transforms: List[Any], keep_mosaic: bool = False) -> List[Any]:
        """Exclude A.Mosaic from list (single-image wrapper), unless keep_mosaic=True (use_albumentations_mosaic)."""
        if keep_mosaic:
            return list(transforms)
        out = []
        for t in transforms:
            name = getattr(getattr(t, "__class__", None), "__name__", None)
            if name != "Mosaic":
                out.append(t)
        return out
    
    def _build_transforms(self) -> None:
        """Build the augmentation transforms based on config."""
        cfg = self.config
        
        if self.is_train:
            # Наша Mosaic (mosaic.py); при use_albumentations_mosaic використовуємо p=0 і A.Mosaic в albu з metadata
            self.mosaic = Mosaic(
                dataset=self.dataset,
                imgsz=cfg.imgsz,
                mosaic_scale=cfg.mosaic_scale,
                min_box_size=cfg.mosaic_min_box_size,
                fill_color=cfg.letterbox_color,
                p=cfg.mosaic if not self._use_albu_mosaic else 0.0,
            )
            self.mixup = MixUp(
                dataset=self.dataset,
                alpha=cfg.mixup_alpha,
                p=cfg.mixup,
            )
            self.cutmix = CutMix(
                dataset=self.dataset,
                alpha=cfg.cutmix_alpha,
                p=cfg.cutmix,
                min_visible_ratio=cfg.cutmix_min_visible,
                min_box_size=cfg.cutmix_min_box_size,
                overlap_thresh=cfg.cutmix_overlap_thresh,
            )
            # Albumentations: color, flips, erasing, etc. (from ALBUMENTATION_CONFIG)
            self.albumentations = AlbumentationsWrapper(
                transforms=self._albu_list,
                p=1.0,
                name="Albumentations",
            )
        
        # Common transforms (train and val)
        self.letterbox = LetterBox(
            new_shape=cfg.imgsz,
            color=cfg.letterbox_color,
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
        
        # One-time mosaic status message
        if self.is_train:
            mosaic_enabled = self._should_apply_mosaic() and self.config.mosaic > 0
            
            # First epoch - check if mosaic is disabled from the start
            if self._mosaic_was_enabled is None:
                self._mosaic_was_enabled = mosaic_enabled
                if not mosaic_enabled:
                    print(f"Mosaic: disabled (close_mosaic={self.config.close_mosaic})")
            # Mosaic was enabled but now disabled
            elif self._mosaic_was_enabled and not mosaic_enabled:
                self._mosaic_was_enabled = False
                remaining = self._total_epochs - self._current_epoch
                print(f"Mosaic: disabled at epoch {epoch} (remaining={remaining})")
        
    
    def _should_apply_mosaic(self) -> bool:
        """Check if mosaic should be applied based on close_mosaic."""
        if not self.is_train:
            return False
        
        remaining_epochs = self._total_epochs - self._current_epoch
        return remaining_epochs > self.config.close_mosaic

    def _build_mosaic_metadata(self, primary_index: int) -> List[Dict[str, Any]]:
        """Побудувати mosaic_metadata для A.Mosaic: 3 додаткові зображення з датасету (primary = поточне)."""
        n = len(self.dataset) if self.dataset else 0
        if n < 4:
            return []
        indices = [primary_index]
        while len(indices) < 4:
            idx = random.randint(0, n - 1)
            if idx not in indices:
                indices.append(idx)
        metadata = []
        for idx in indices[1:]:  # 3 додаткові (без primary)
            try:
                img, t = self.dataset.get_raw_item(idx)
                if isinstance(img, PIL.Image.Image):
                    img = np.array(img)
                elif isinstance(img, torch.Tensor):
                    img = img.permute(1, 2, 0).numpy()
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                boxes = t.get("boxes", torch.zeros((0, 4)))
                labels = t.get("labels", torch.zeros(0, dtype=torch.int64))
                if boxes.dim() == 1:
                    boxes = boxes.unsqueeze(0)
                bboxes_list = [tuple(float(x) for x in row) for row in boxes.cpu().numpy().tolist()]
                class_labels_list = labels.cpu().tolist()
                metadata.append({"image": np.ascontiguousarray(img), "bboxes": bboxes_list, "class_labels": class_labels_list})
            except Exception:
                continue
        return metadata

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
            # 1. Mosaic: наша (mosaic.py) або потім A.Mosaic всередині albu з metadata
            if self._should_apply_mosaic() and not self._use_albu_mosaic:
                image, target = self.mosaic(image, target, index)
                self._log_augmentation("Mosaic", self.mosaic)
            # 2. MixUp
            image, target = self.mixup(image, target, index)
            self._log_augmentation("MixUp", self.mixup)
            # 3. CutMix
            image, target = self.cutmix(image, target, index)
            self._log_augmentation("CutMix", self.cutmix)
            # Підготовка mosaic_metadata для A.Mosaic (якщо use_albumentations_mosaic)
            if self._use_albu_mosaic and self._should_apply_mosaic() and self.config.mosaic > 0:
                if "mosaic_metadata" not in target and self.dataset is not None and hasattr(self.dataset, "get_raw_item"):
                    target = dict(target)
                    target["mosaic_metadata"] = self._build_mosaic_metadata(index)
            # 4. Albumentations (color, flips, geometry, A.Mosaic якщо є metadata, тощо)
            image, target = self.albumentations(image, target)
            self._log_augmentation("Albumentations", self.albumentations)
        
        # 5. LetterBox (resize with padding)
        image, target = self.letterbox(image, target)
        # 6. ToTensor
        image, target = self.to_tensor(image, target)
        # 7. Normalize
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
