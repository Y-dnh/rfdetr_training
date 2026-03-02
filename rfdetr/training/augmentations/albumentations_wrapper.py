"""
Albumentations wrapper for RF-DETR augmentation pipeline.

Converts (image, target) to albumentations format, applies A.Compose built from
ALBUMENTATION_CONFIG (list of A.* transforms), then converts back to (image, target).
Supports spatial transforms (flips, CoarseDropout, etc.) via bbox_params.
Reference: https://explore.albumentations.ai/
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

from rfdetr.training.augmentations.base import BaseTransform, AugmentationResult, clip_boxes

# Class names that require bbox_params (spatial or dual transforms).
# From https://albumentations.ai/docs/getting_started/transforms_and_targets/
_SPATIAL_TRANSFORM_NAMES = frozenset({
    "Affine",
    "AtLeastOneBBoxRandomCrop",
    "BBoxSafeRandomCrop",
    "CenterCrop",
    "CoarseDropout",
    "ConstrainedCoarseDropout",
    "Crop",
    "CropAndPad",
    "CropNonEmptyMaskIfExists",
    "D4",
    "ElasticTransform",
    "Erasing",
    "Flip",
    "GridDistortion",
    "GridDropout",
    "HorizontalFlip",
    "LongestMaxSize",
    "MaskDropout",
    "Morphological",
    "Mosaic",
    "NoOp",
    "OpticalDistortion",
    "Pad",
    "PadIfNeeded",
    "Perspective",
    "PixelDropout",
    "RandomCrop",
    "RandomCropFromBorders",
    "RandomCropNearBBox",
    "RandomGridShuffle",
    "RandomResizedCrop",
    "RandomRotate90",
    "RandomScale",
    "RandomSizedBBoxSafeCrop",
    "RandomSizedCrop",
    "Resize",
    "Rotate",
    "SafeRotate",
    "ShiftScaleRotate",
    "SmallestMaxSize",
    "Transpose",
    "VerticalFlip",
    "XYMasking",
})


def _contains_spatial(transforms: List[Any]) -> bool:
    """Return True if any transform in the list is a spatial (bbox-affecting) transform."""
    for t in transforms:
        name = getattr(t, "__class__", None) and getattr(t.__class__, "__name__", None)
        if name and name in _SPATIAL_TRANSFORM_NAMES:
            return True
    return False


def _to_uint8_rgb(image: Union[PIL.Image.Image, np.ndarray]) -> np.ndarray:
    """Convert image to uint8 RGB numpy array (H, W, 3)."""
    if isinstance(image, PIL.Image.Image):
        img_np = np.array(image)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        return img_np
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return img


class AlbumentationsWrapper(BaseTransform):
    """
    Wrapper that applies an albumentations Compose to (image, target).

    Converts RF-DETR target (boxes in xyxy pixels, labels) to albumentations
    bboxes + class_labels, runs the transform, then converts back and updates
    target (boxes, labels, and optionally area). Handles empty bbox lists.
    """

    def __init__(
        self,
        transforms: Optional[List[Any]] = None,
        p: float = 1.0,
        name: Optional[str] = None,
    ):
        """
        Args:
            transforms: List of albumentations transform instances (e.g. from get_default_albu_config()).
                        If None or empty, the wrapper no-ops.
            p: Probability of applying the whole albumentations block.
            name: Name for logging.
        """
        super().__init__(p=p, name=name or "Albumentations")
        raw = transforms or []
        # Розгорнути тільки фабрики (function/lambda) в інстанси; інстанси A.* не викликати — вони теж callable, t() повертає dict
        resolved: List[Any] = []
        for t in raw:
            if getattr(t, "__class__", None) and type(t).__name__ == "function":
                try:
                    t = t()
                except Exception:
                    continue
            if t is not None and not isinstance(t, dict):
                resolved.append(t)
        self._transforms = resolved
        self._transform: Optional[Any] = None
        self._last_applied_names: List[str] = []  # для логу: які саме A.* трансформи спрацювали
        self._build_compose()

    def _build_compose(self) -> None:
        """Build A.Compose from the list of transforms; add bbox_params if any spatial transform."""
        if not self._transforms:
            self._transform = None
            return
        try:
            import albumentations as A
        except ImportError:
            self._transform = None
            return
        contains_spatial = _contains_spatial(self._transforms)
        # save_applied_params=True → у результаті applied_transforms для логу "які Albumentations спрацювали"
        compose_kw: Dict[str, Any] = {}
        try:
            import inspect
            if "save_applied_params" in inspect.signature(A.Compose.__init__).parameters:
                compose_kw["save_applied_params"] = True
        except Exception:
            pass
        try:
            if contains_spatial:
                self._transform = A.Compose(
                    self._transforms,
                    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
                    **compose_kw,
                )
            else:
                self._transform = A.Compose(self._transforms, **compose_kw)
        except TypeError:
            compose_kw.pop("save_applied_params", None)
            if contains_spatial:
                self._transform = A.Compose(
                    self._transforms,
                    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
                )
            else:
                self._transform = A.Compose(self._transforms)

    def apply(
        self,
        image: Union[PIL.Image.Image, np.ndarray],
        target: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply albumentations Compose to image and target.

        Args:
            image: PIL Image or numpy array (RGB).
            target: Dict with 'boxes' (N, 4) xyxy pixels, 'labels' (N,) int64.

        Returns:
            (image as np.ndarray uint8 RGB, updated target with boxes/labels/area).
        """
        if self._transform is None:
            img_np = _to_uint8_rgb(image)
            return img_np, dict(target)

        img_np = _to_uint8_rgb(image)
        h, w = img_np.shape[:2]

        # Convert target to albumentations inputs: bboxes as list [x1,y1,x2,y2], class_labels as list
        boxes_t = target.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
        labels_t = target.get("labels", torch.zeros(0, dtype=torch.int64))
        if boxes_t.dim() == 1:
            boxes_t = boxes_t.unsqueeze(0)
        n = len(boxes_t)
        if n == 0:
            bboxes_list: List[Tuple[float, float, float, float]] = []
            class_labels_list: List[int] = []
        else:
            boxes_np = boxes_t.cpu().numpy()
            # Ensure xyxy order; clip to image bounds before passing to albumentations
            boxes_np = np.clip(boxes_np, [0, 0, 0, 0], [w, h, w, h])
            bboxes_list = [tuple(float(x) for x in row) for row in boxes_np]
            class_labels_list = labels_t.cpu().tolist()

        # Optional: pass through mosaic_metadata if present (for A.Mosaic when used with 4 images)
        extra_kw: Dict[str, Any] = {}
        if "mosaic_metadata" in target:
            extra_kw["mosaic_metadata"] = target["mosaic_metadata"]

        try:
            if bboxes_list and _contains_spatial(self._transforms):
                out = self._transform(
                    image=img_np,
                    bboxes=bboxes_list,
                    class_labels=class_labels_list,
                    **extra_kw,
                )
            else:
                # Image-only or empty bboxes: albumentations may still expect bboxes/labels
                if _contains_spatial(self._transforms):
                    out = self._transform(
                        image=img_np,
                        bboxes=bboxes_list,
                        class_labels=class_labels_list,
                        **extra_kw,
                    )
                else:
                    out = self._transform(image=img_np)
                    # Image-only compose returns only 'image'; keep bboxes/labels unchanged
                    out = {"image": out["image"], "bboxes": bboxes_list, "class_labels": class_labels_list}
        except Exception:
            # If transform fails (e.g. Mosaic without metadata), return unchanged
            self._last_applied_names = []
            return img_np, dict(target)

        self._last_applied_names = [t[0] for t in out.get("applied_transforms", [])]
        out_img = out["image"]
        out_bboxes = out.get("bboxes", [])
        out_labels = out.get("class_labels", [])

        # Convert back to tensors; filter invalid boxes (non-positive width/height)
        if not out_bboxes:
            new_boxes = torch.zeros((0, 4), dtype=torch.float32)
            new_labels = torch.zeros(0, dtype=torch.int64)
        else:
            arr = np.array(out_bboxes, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            # Filter invalid: width/height > 0
            ww = arr[:, 2] - arr[:, 0]
            hh = arr[:, 3] - arr[:, 1]
            keep = (ww > 0) & (hh > 0)
            arr = arr[keep]
            labels_arr = np.array(out_labels, dtype=np.int64)
            if len(labels_arr) == len(keep):
                labels_arr = labels_arr[keep]
            else:
                labels_arr = labels_arr[: len(arr)]
            new_boxes = torch.from_numpy(arr)
            new_labels = torch.from_numpy(labels_arr)

        # Clip boxes to new image size (after spatial transform size may change)
        out_h, out_w = out_img.shape[:2]
        new_boxes = clip_boxes(new_boxes, (out_h, out_w))

        # Recompute area for kept boxes
        if len(new_boxes) > 0:
            new_area = (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1])
        else:
            new_area = torch.zeros(0, dtype=torch.float32)

        new_target = dict(target)
        new_target["boxes"] = new_boxes
        new_target["labels"] = new_labels
        new_target["area"] = new_area
        if "orig_size" in new_target:
            new_target["size"] = torch.tensor([out_h, out_w])
        return out_img, new_target

    def get_parameters(self) -> Dict[str, Any]:
        """Return parameters for logging (number of transforms, which albu transforms were applied)."""
        n = len(self._transforms) if self._transforms else 0
        out = {"num_transforms": n, "transform": self._transform is not None}
        if getattr(self, "_last_applied_names", None):
            out["applied_names"] = self._last_applied_names
        return out
