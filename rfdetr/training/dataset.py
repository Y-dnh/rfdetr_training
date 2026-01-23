"""
Dataset classes for RF-DETR training with custom augmentations.

This module provides dataset classes that integrate the augmentation pipeline
with the existing RF-DETR COCO detection framework.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.utils.data
import torchvision
from pycocotools.coco import COCO

from rfdetr.datasets.coco import CocoDetection, ConvertCoco
from rfdetr.training.augmentations.pipeline import AugmentationPipeline, ValidationPipeline
from rfdetr.training.utils.config import AugmentationConfig


class RFDETRDataset(torch.utils.data.Dataset):
    """
    Dataset for RF-DETR training with Ultralytics-style augmentations.
    
    This dataset wraps COCO-format annotations and applies the custom
    augmentation pipeline including Mosaic, MixUp, CutMix, HSV, etc.
    
    Args:
        img_folder: Path to image folder.
        ann_file: Path to COCO-format annotation JSON file.
        augmentation_config: Configuration for augmentations.
        is_train: Whether this is a training dataset.
        transforms: Additional transforms to apply after augmentation pipeline.
        log_augmentations: Whether to log applied augmentations.
    
    Example:
        >>> config = AugmentationConfig(mosaic=1.0, hsv_h=0.015)
        >>> dataset = RFDETRDataset(
        ...     img_folder='data/train',
        ...     ann_file='data/train/_annotations.coco.json',
        ...     augmentation_config=config,
        ...     is_train=True
        ... )
        >>> image, target = dataset[0]
    """
    
    def __init__(
        self,
        img_folder: Union[str, Path],
        ann_file: Union[str, Path],
        augmentation_config: Optional[AugmentationConfig] = None,
        is_train: bool = True,
        transforms: Optional[Callable] = None,
        log_augmentations: bool = False,
    ):
        self.img_folder = Path(img_folder)
        self.ann_file = Path(ann_file)
        self.is_train = is_train
        self.log_augmentations = log_augmentations
        self.transforms = transforms
        
        # Load COCO annotations
        with open(self.ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build image and annotation indices
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = list(self.images.keys())
        
        # Build annotation index (image_id -> list of annotations)
        self.annotations = {}
        for ann in self.coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        # Category mapping
        self.categories = {cat['id']: cat for cat in self.coco_data.get('categories', [])}
        self.cat_ids = list(self.categories.keys())
        
        # Remap category IDs to contiguous indices
        self.cat_id_to_label = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        self.label_to_cat_id = {idx: cat_id for cat_id, idx in self.cat_id_to_label.items()}
        
        # Set up augmentation pipeline
        self.config = augmentation_config or AugmentationConfig()
        
        if is_train:
            self.pipeline = AugmentationPipeline(
                config=self.config,
                dataset=self,  # Pass self for mosaic/mixup
                is_train=True,
                log_augmentations=log_augmentations,
            )
        else:
            self.pipeline = ValidationPipeline(imgsz=self.config.imgsz)
        
        # COCO converter for annotations
        self.prepare = ConvertCoco(include_masks=False)
        
        # Store filenames for logging
        self.filenames = [self.images[img_id]['file_name'] for img_id in self.image_ids]
        
        # Create COCO API object for evaluation
        self._coco_api = None
    
    def __setstate__(self, state):
        """Pickle support: restore dataset reference in pipeline after unpickle."""
        self.__dict__.update(state)
        # Restore self reference in pipeline for mosaic/mixup
        if self.is_train and hasattr(self.pipeline, 'set_dataset'):
            self.pipeline.set_dataset(self)
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def _load_image(self, index: int) -> PIL.Image.Image:
        """Load image at given index."""
        img_id = self.image_ids[index]
        img_info = self.images[img_id]
        img_path = self.img_folder / img_info['file_name']
        
        image = PIL.Image.open(img_path).convert('RGB')
        return image
    
    def _load_target(self, index: int) -> Dict[str, Any]:
        """Load target annotations at given index."""
        img_id = self.image_ids[index]
        img_info = self.images[img_id]
        anns = self.annotations.get(img_id, [])
        
        # Filter out crowd annotations
        anns = [ann for ann in anns if ann.get('iscrowd', 0) == 0]
        
        # Extract boxes and labels
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            # Convert from xywh to xyxy
            boxes.append([x, y, x + w, y + h])
            # Remap category ID to contiguous label
            labels.append(self.cat_id_to_label[ann['category_id']])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
            'area': torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
            'orig_size': torch.tensor([img_info['height'], img_info['width']]),
            'size': torch.tensor([img_info['height'], img_info['width']]),
        }
        
        return target
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any], Optional[Dict]]:
        """
        Get item at index.
        
        Args:
            index: Index of the item.
        
        Returns:
            Tuple of (image_tensor, target_dict, augmentation_log).
        """
        # Load image and target
        image = self._load_image(index)
        target = self._load_target(index)
        
        # Apply augmentation pipeline
        aug_log = None
        if self.is_train:
            image, target = self.pipeline(image, target, index=index)
            # Get augmentation log if enabled
            if self.log_augmentations and hasattr(self.pipeline, 'get_last_log'):
                aug_log = self.pipeline.get_last_log()
                if aug_log:
                    aug_log['image_name'] = self.get_filename(index)
        else:
            image, target = self.pipeline(image, target)
        
        # Apply additional transforms if any
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target, aug_log
    
    def get_raw_item(self, index: int) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        """
        Get raw item without augmentations (for visualization).
        
        Args:
            index: Index of the item.
        
        Returns:
            Tuple of (PIL.Image, target_dict).
        """
        image = self._load_image(index)
        target = self._load_target(index)
        return image, target
    
    def get_filename(self, index: int) -> str:
        """Get filename at index."""
        return self.filenames[index]
    
    def set_epoch(self, epoch: int, total_epochs: Optional[int] = None) -> None:
        """
        Set current epoch for close_mosaic handling.
        
        Args:
            epoch: Current epoch number.
            total_epochs: Total number of epochs (for close_mosaic calculation).
        """
        if hasattr(self.pipeline, 'set_epoch'):
            self.pipeline.set_epoch(epoch, total_epochs)
    
    def get_last_augmentation_log(self) -> List[Dict[str, Any]]:
        """Get log of last augmentation call."""
        if hasattr(self.pipeline, 'get_last_log'):
            return self.pipeline.get_last_log()
        return []
    
    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.cat_ids)
    
    @property
    def class_names(self) -> List[str]:
        """Get list of class names."""
        return [self.categories[cat_id]['name'] for cat_id in self.cat_ids]
    
    @property
    def coco(self) -> COCO:
        """
        Get COCO API object for evaluation.
        
        Creates the COCO object lazily on first access.
        
        Returns:
            COCO API object with ground truth annotations.
        """
        if self._coco_api is None:
            # Create COCO object from our annotation data
            self._coco_api = COCO()
            self._coco_api.dataset = self.coco_data
            self._coco_api.createIndex()
        return self._coco_api


def build_dataset(
    dataset_dir: Union[str, Path],
    split: str = 'train',
    augmentation_config: Optional[AugmentationConfig] = None,
    log_augmentations: bool = False,
) -> RFDETRDataset:
    """
    Build dataset from Roboflow-style directory structure.
    
    Expected structure:
        dataset_dir/
            train/
                images...
                _annotations.coco.json
            valid/
                images...
                _annotations.coco.json
            test/
                images...
                _annotations.coco.json
    
    Args:
        dataset_dir: Root directory of the dataset.
        split: One of 'train', 'valid', 'test'.
        augmentation_config: Augmentation configuration.
        log_augmentations: Whether to log augmentations.
    
    Returns:
        RFDETRDataset instance.
    """
    dataset_dir = Path(dataset_dir)
    
    split_dir = dataset_dir / split
    ann_file = split_dir / '_annotations.coco.json'
    
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")
    if not ann_file.exists():
        raise ValueError(f"Annotation file not found: {ann_file}")
    
    is_train = split == 'train'
    
    return RFDETRDataset(
        img_folder=split_dir,
        ann_file=ann_file,
        augmentation_config=augmentation_config,
        is_train=is_train,
        log_augmentations=log_augmentations,
    )


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any], Optional[Dict]]]) -> Tuple[torch.Tensor, List[Dict[str, Any]], List[Optional[Dict]]]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (image, target, aug_log) tuples.
    
    Returns:
        Tuple of (batched_images, list_of_targets, list_of_aug_logs).
    """
    images = []
    targets = []
    aug_logs = []
    
    for item in batch:
        if len(item) == 3:
            image, target, aug_log = item
        else:
            image, target = item
            aug_log = None
        images.append(image)
        targets.append(target)
        aug_logs.append(aug_log)
    
    # Stack images into batch
    images = torch.stack(images, dim=0)
    
    return images, targets, aug_logs
