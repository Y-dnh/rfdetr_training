"""
MixUp and CutMix augmentations for RF-DETR training.

These augmentations blend two images together to create more diverse
training samples and improve model generalization.
"""

import random
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import PIL.Image
import torch

from rfdetr.training.augmentations.base import BaseTransform, clip_boxes, filter_small_boxes


class MixUp(BaseTransform):
    """
    MixUp augmentation that blends two images using alpha compositing.
    
    MixUp creates a new training sample by linearly interpolating
    between two images and their labels. The mixing ratio is sampled
    from a Beta distribution.
    
    Args:
        dataset: Dataset to sample the second image from.
        alpha: Parameter for Beta distribution (default 32.0).
               Higher values produce mixing ratios closer to 0.5.
        p: Probability of applying MixUp.
    
    Reference:
        mixup: Beyond Empirical Risk Minimization
        https://arxiv.org/abs/1710.09412
    
    Example:
        >>> mixup = MixUp(dataset=train_dataset, alpha=32.0, p=0.5)
        >>> image, target = mixup(image, target, index=0)
    """
    
    def __init__(
        self,
        dataset: Any = None,
        alpha: float = 32.0,
        p: float = 0.0,  # Disabled by default in YOLO
    ):
        super().__init__(p=p, name='MixUp')
        
        self.dataset = dataset
        self.alpha = alpha
        
        self._last_ratio = 0.0
        self._last_index2 = -1
    
    def set_dataset(self, dataset: Any) -> None:
        """Set the dataset for sampling second image."""
        self.dataset = dataset
    
    def apply_with_index(
        self,
        image: Union[PIL.Image.Image, np.ndarray],
        target: Dict[str, Any],
        index: int = 0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply MixUp augmentation.
        
        Args:
            image: Primary image.
            target: Primary target dict.
            index: Index of primary image in dataset.
        
        Returns:
            Tuple of (mixed_image, merged_target).
        """
        # Convert to numpy
        if isinstance(image, PIL.Image.Image):
            img1 = np.array(image)
        else:
            img1 = image.copy()
        
        # Get second image (use get_raw_item to avoid recursive augmentation)
        if self.dataset is not None:
            n = len(self.dataset)
            index2 = random.randint(0, n - 1)
            self._last_index2 = index2
            
            try:
                if hasattr(self.dataset, 'get_raw_item'):
                    img2, target2 = self.dataset.get_raw_item(index2)
                else:
                    img2, target2 = self.dataset[index2]
                if isinstance(img2, PIL.Image.Image):
                    img2 = np.array(img2)
            except Exception:
                img2 = img1.copy()
                target2 = target
        else:
            img2 = img1.copy()
            target2 = target
            self._last_index2 = index
        
        # Sample mixing ratio from Beta distribution
        ratio = np.random.beta(self.alpha, self.alpha)
        self._last_ratio = ratio
        
        # Ensure images have same size
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if (h1, w1) != (h2, w2):
            # Resize second image to match first
            img2 = cv2.resize(img2, (w1, h1))
            
            # Scale boxes of second image
            if 'boxes' in target2 and len(target2['boxes']) > 0:
                scale_x = w1 / w2
                scale_y = h1 / h2
                target2 = target2.copy()
                boxes2 = target2['boxes'].clone() if isinstance(target2['boxes'], torch.Tensor) else torch.tensor(target2['boxes'])
                boxes2[:, [0, 2]] *= scale_x
                boxes2[:, [1, 3]] *= scale_y
                target2['boxes'] = boxes2
        
        # Mix images
        mixed_img = (img1.astype(np.float32) * ratio + img2.astype(np.float32) * (1 - ratio))
        mixed_img = np.clip(mixed_img, 0, 255).astype(np.uint8)
        
        # Merge targets - concatenate boxes and labels
        boxes1 = target['boxes'] if isinstance(target['boxes'], torch.Tensor) else torch.tensor(target['boxes'])
        labels1 = target['labels'] if isinstance(target['labels'], torch.Tensor) else torch.tensor(target['labels'])
        
        boxes2 = target2['boxes'] if isinstance(target2['boxes'], torch.Tensor) else torch.tensor(target2['boxes'])
        labels2 = target2['labels'] if isinstance(target2['labels'], torch.Tensor) else torch.tensor(target2['labels'])
        
        if len(boxes1) > 0 and len(boxes2) > 0:
            merged_boxes = torch.cat([boxes1, boxes2], dim=0)
            merged_labels = torch.cat([labels1, labels2], dim=0)
        elif len(boxes1) > 0:
            merged_boxes = boxes1
            merged_labels = labels1
        elif len(boxes2) > 0:
            merged_boxes = boxes2
            merged_labels = labels2
        else:
            merged_boxes = torch.zeros((0, 4), dtype=torch.float32)
            merged_labels = torch.zeros((0,), dtype=torch.int64)
        
        new_target = {
            'boxes': merged_boxes,
            'labels': merged_labels,
            'size': torch.tensor([h1, w1]),
            'area': (merged_boxes[:, 2] - merged_boxes[:, 0]) * (merged_boxes[:, 3] - merged_boxes[:, 1]) if len(merged_boxes) > 0 else torch.zeros(0),
        }
        
        # Copy other fields
        for key in ['image_id']:
            if key in target:
                new_target[key] = target[key]
        
        if len(merged_labels) > 0:
            new_target['iscrowd'] = torch.zeros(len(merged_labels), dtype=torch.int64)
        
        return mixed_img, new_target
    
    def apply(self, image, target):
        return self.apply_with_index(image, target, 0)
    
    def __call__(self, image, target, index=0):
        if random.random() < self.p:
            return self.apply_with_index(image, target, index)
        return image, target
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'alpha': self.alpha,
            'ratio': self._last_ratio,
            'index2': self._last_index2,
        }
    
    def __getstate__(self):
        """Pickle support: remove dataset reference."""
        state = self.__dict__.copy()
        state['dataset'] = None
        return state
    
    def __setstate__(self, state):
        """Pickle support: restore state."""
        self.__dict__.update(state)


class CutMix(BaseTransform):
    """
    CutMix augmentation that cuts a patch from one image and pastes onto another.
    
    CutMix creates a new training sample by replacing a rectangular region
    of one image with the corresponding region from another image.
    
    Args:
        dataset: Dataset to sample the second image from.
        alpha: Parameter for Beta distribution to sample cut ratio.
        p: Probability of applying CutMix.
    
    Reference:
        CutMix: Regularization Strategy to Train Strong Classifiers
        https://arxiv.org/abs/1905.04899
    
    Example:
        >>> cutmix = CutMix(dataset=train_dataset, alpha=1.0, p=0.5)
        >>> image, target = cutmix(image, target, index=0)
    """
    
    def __init__(
        self,
        dataset: Any = None,
        alpha: float = 1.0,
        p: float = 0.0,  # Disabled by default
    ):
        super().__init__(p=p, name='CutMix')
        
        self.dataset = dataset
        self.alpha = alpha
        
        self._last_bbox = (0, 0, 0, 0)
        self._last_index2 = -1
    
    def set_dataset(self, dataset: Any) -> None:
        """Set the dataset for sampling second image."""
        self.dataset = dataset
    
    def _get_cutmix_bbox(
        self, 
        h: int, 
        w: int, 
        lam: float
    ) -> Tuple[int, int, int, int]:
        """
        Get random bounding box for CutMix.
        
        Args:
            h: Image height.
            w: Image width.
            lam: Lambda value (ratio of image to keep from first image).
        
        Returns:
            Tuple of (x1, y1, x2, y2) for the cut region.
        """
        cut_ratio = np.sqrt(1 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        # Random center
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(w, cx + cut_w // 2)
        y2 = min(h, cy + cut_h // 2)
        
        return x1, y1, x2, y2
    
    def apply_with_index(
        self,
        image: Union[PIL.Image.Image, np.ndarray],
        target: Dict[str, Any],
        index: int = 0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply CutMix augmentation.
        
        Args:
            image: Primary image.
            target: Primary target dict.
            index: Index of primary image in dataset.
        
        Returns:
            Tuple of (cutmix_image, merged_target).
        """
        # Convert to numpy
        if isinstance(image, PIL.Image.Image):
            img1 = np.array(image)
        else:
            img1 = image.copy()
        
        h, w = img1.shape[:2]
        
        # Get second image (use get_raw_item to avoid recursive augmentation)
        if self.dataset is not None:
            n = len(self.dataset)
            index2 = random.randint(0, n - 1)
            self._last_index2 = index2
            
            try:
                if hasattr(self.dataset, 'get_raw_item'):
                    img2, target2 = self.dataset.get_raw_item(index2)
                else:
                    img2, target2 = self.dataset[index2]
                if isinstance(img2, PIL.Image.Image):
                    img2 = np.array(img2)
            except Exception:
                img2 = img1.copy()
                target2 = target
        else:
            img2 = img1.copy()
            target2 = target
            self._last_index2 = index
        
        # Resize second image if needed
        h2, w2 = img2.shape[:2]
        if (h, w) != (h2, w2):
            img2 = cv2.resize(img2, (w, h))
            
            if 'boxes' in target2 and len(target2['boxes']) > 0:
                scale_x = w / w2
                scale_y = h / h2
                target2 = target2.copy()
                boxes2 = target2['boxes'].clone() if isinstance(target2['boxes'], torch.Tensor) else torch.tensor(target2['boxes'])
                boxes2[:, [0, 2]] *= scale_x
                boxes2[:, [1, 3]] *= scale_y
                target2['boxes'] = boxes2
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get cut bbox
        x1, y1, x2, y2 = self._get_cutmix_bbox(h, w, lam)
        self._last_bbox = (x1, y1, x2, y2)
        
        # Apply CutMix - paste region from img2 onto img1
        mixed_img = img1.copy()
        mixed_img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
        
        # Handle boxes
        boxes1 = target['boxes'] if isinstance(target['boxes'], torch.Tensor) else torch.tensor(target['boxes'])
        labels1 = target['labels'] if isinstance(target['labels'], torch.Tensor) else torch.tensor(target['labels'])
        
        boxes2 = target2['boxes'] if isinstance(target2['boxes'], torch.Tensor) else torch.tensor(target2['boxes'])
        labels2 = target2['labels'] if isinstance(target2['labels'], torch.Tensor) else torch.tensor(target2['labels'])
        
        # Filter boxes based on overlap with cut region
        cut_tensor = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        
        # Keep boxes from img1 that are mostly outside cut region
        # Keep boxes from img2 that are mostly inside cut region
        
        final_boxes = []
        final_labels = []
        
        # Process boxes from image 1 (keep if mostly outside cut region)
        for i, box in enumerate(boxes1):
            # Calculate intersection with cut region
            inter_x1 = max(box[0].item(), x1)
            inter_y1 = max(box[1].item(), y1)
            inter_x2 = min(box[2].item(), x2)
            inter_y2 = min(box[3].item(), y2)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            else:
                inter_area = 0
            
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            
            # Keep if less than 50% overlap with cut region
            if box_area > 0 and inter_area / box_area < 0.5:
                final_boxes.append(box)
                final_labels.append(labels1[i])
        
        # Process boxes from image 2 (keep if mostly inside cut region)
        for i, box in enumerate(boxes2):
            inter_x1 = max(box[0].item(), x1)
            inter_y1 = max(box[1].item(), y1)
            inter_x2 = min(box[2].item(), x2)
            inter_y2 = min(box[3].item(), y2)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            else:
                inter_area = 0
            
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            
            # Keep if more than 50% inside cut region
            if box_area > 0 and inter_area / box_area >= 0.5:
                # Clip box to cut region
                clipped_box = torch.tensor([
                    max(box[0].item(), x1),
                    max(box[1].item(), y1),
                    min(box[2].item(), x2),
                    min(box[3].item(), y2),
                ])
                final_boxes.append(clipped_box)
                final_labels.append(labels2[i])
        
        if final_boxes:
            merged_boxes = torch.stack(final_boxes)
            merged_labels = torch.stack(final_labels) if isinstance(final_labels[0], torch.Tensor) else torch.tensor(final_labels)
        else:
            merged_boxes = torch.zeros((0, 4), dtype=torch.float32)
            merged_labels = torch.zeros((0,), dtype=torch.int64)
        
        new_target = {
            'boxes': merged_boxes,
            'labels': merged_labels,
            'size': torch.tensor([h, w]),
            'area': (merged_boxes[:, 2] - merged_boxes[:, 0]) * (merged_boxes[:, 3] - merged_boxes[:, 1]) if len(merged_boxes) > 0 else torch.zeros(0),
        }
        
        for key in ['image_id']:
            if key in target:
                new_target[key] = target[key]
        
        if len(merged_labels) > 0:
            new_target['iscrowd'] = torch.zeros(len(merged_labels), dtype=torch.int64)
        
        return mixed_img, new_target
    
    def apply(self, image, target):
        return self.apply_with_index(image, target, 0)
    
    def __call__(self, image, target, index=0):
        if random.random() < self.p:
            return self.apply_with_index(image, target, index)
        return image, target
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'alpha': self.alpha,
            'bbox': self._last_bbox,
            'index2': self._last_index2,
        }
    
    def __getstate__(self):
        """Pickle support: remove dataset reference."""
        state = self.__dict__.copy()
        state['dataset'] = None
        return state
    
    def __setstate__(self, state):
        """Pickle support: restore state."""
        self.__dict__.update(state)
