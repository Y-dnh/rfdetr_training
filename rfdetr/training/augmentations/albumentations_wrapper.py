"""
Albumentations wrapper for RF-DETR augmentation pipeline.

Converts (image, target) to albumentations format, applies A.Compose built from
ALBUMENTATION_CONFIG (list of A.* transforms), then converts back to
(image, target). Supports spatial transforms via bbox_params and transparently
injects extra targets required by RandomCropNearBBox.
"""

from __future__ import annotations

import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

from rfdetr.training.augmentations.base import BaseTransform, clip_boxes

# Class names that require bbox_params (spatial or dual transforms).
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


def _get_transform_name(transform: Any) -> Optional[str]:
    """Return the class name for an albumentations transform instance."""
    return getattr(getattr(transform, "__class__", None), "__name__", None)


def _contains_spatial(transforms: List[Any]) -> bool:
    """Return True if any transform in the list is a spatial (bbox-affecting) transform."""
    for transform in transforms:
        name = _get_transform_name(transform)
        if name and name in _SPATIAL_TRANSFORM_NAMES:
            return True
    return False


def _collect_random_crop_near_bbox_keys(transforms: List[Any]) -> List[str]:
    """Collect additional-target keys required by RandomCropNearBBox transforms."""
    keys: List[str] = []
    for transform in transforms:
        if _get_transform_name(transform) != "RandomCropNearBBox":
            continue
        key = getattr(transform, "cropping_bbox_key", "cropping_bbox")
        if isinstance(key, str) and key and key not in keys:
            keys.append(key)
    return keys


def _without_random_crop_near_bbox(transforms: List[Any]) -> List[Any]:
    """Return a transform list with RandomCropNearBBox removed."""
    return [transform for transform in transforms if _get_transform_name(transform) != "RandomCropNearBBox"]


def _warn_if_random_crop_is_not_first_spatial(transforms: List[Any]) -> None:
    """Warn when RandomCropNearBBox is preceded by another spatial transform."""
    seen_spatial = False
    for transform in transforms:
        name = _get_transform_name(transform)
        if not name:
            continue
        if name == "RandomCropNearBBox":
            if seen_spatial:
                warnings.warn(
                    "RandomCropNearBBox should appear before other spatial albumentations transforms. "
                    "Earlier spatial transforms keep cropping_bbox in the pre-transform coordinate frame "
                    "and can invalidate the crop target.",
                    stacklevel=2,
                )
            return
        if name in _SPATIAL_TRANSFORM_NAMES:
            seen_spatial = True


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


def _normalize_crop_box(
    crop_box: Any,
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Validate and clip a crop box to image bounds."""
    try:
        arr = np.asarray(crop_box, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size != 4:
        return None
    arr = np.array([
        np.floor(arr[0]),
        np.floor(arr[1]),
        np.ceil(arr[2]),
        np.ceil(arr[3]),
    ], dtype=np.int32)
    arr = np.clip(arr, [0, 0, 0, 0], [width, height, width, height])
    if arr[2] <= arr[0] or arr[3] <= arr[1]:
        return None
    return tuple(int(x) for x in arr.tolist())


def _extract_boxes_and_labels(
    target: Dict[str, Any],
    width: int,
    height: int,
) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]], List[int]]:
    """Convert RF-DETR target boxes/labels to albumentations inputs."""
    boxes_t = target.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
    labels_t = target.get("labels", torch.zeros(0, dtype=torch.int64))
    if boxes_t.dim() == 1:
        boxes_t = boxes_t.unsqueeze(0)
    if len(boxes_t) == 0:
        return np.zeros((0, 4), dtype=np.float32), [], []

    boxes_np = boxes_t.cpu().numpy()
    boxes_np = np.clip(boxes_np, [0, 0, 0, 0], [width, height, width, height]).astype(np.float32)
    widths = boxes_np[:, 2] - boxes_np[:, 0]
    heights = boxes_np[:, 3] - boxes_np[:, 1]
    keep = (widths > 0) & (heights > 0)
    boxes_np = boxes_np[keep]

    labels_np = labels_t.cpu().numpy()
    if len(labels_np) == len(keep):
        labels_np = labels_np[keep]
    else:
        labels_np = labels_np[: len(boxes_np)]

    bboxes_list = [tuple(float(x) for x in row) for row in boxes_np.tolist()]
    class_labels_list = labels_np.astype(np.int64).tolist()
    return boxes_np, bboxes_list, class_labels_list


def _build_random_crop_targets(
    target: Dict[str, Any],
    boxes_np: np.ndarray,
    width: int,
    height: int,
    crop_keys: List[str],
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Build additional targets for RandomCropNearBBox.

    If the user already provided a crop box in ``target``, it is reused.
    Otherwise a bbox is sampled from the current target using inverse-sqrt-area
    weights, which biases the crop toward smaller objects without fully
    suppressing larger ones.
    """
    if not crop_keys:
        return {}

    crop_targets: Dict[str, Tuple[int, int, int, int]] = {}
    for key in crop_keys:
        normalized = _normalize_crop_box(target.get(key), width=width, height=height)
        if normalized is not None:
            crop_targets[key] = normalized

    missing_keys = [key for key in crop_keys if key not in crop_targets]
    if not missing_keys or len(boxes_np) == 0:
        return crop_targets

    areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
    weights = 1.0 / np.sqrt(np.maximum(areas, 1.0))
    selected_index = random.choices(range(len(boxes_np)), weights=weights.tolist(), k=1)[0]
    selected_box = _normalize_crop_box(boxes_np[selected_index], width=width, height=height)
    if selected_box is None:
        return crop_targets
    for key in missing_keys:
        crop_targets[key] = selected_box
    return crop_targets


def _convert_output_boxes_and_labels(
    out_bboxes: Any,
    out_labels: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert albumentations bbox output back to tensors and drop invalid boxes."""
    if not out_bboxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.int64)

    arr = np.array(out_bboxes, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    widths = arr[:, 2] - arr[:, 0]
    heights = arr[:, 3] - arr[:, 1]
    keep = (widths > 0) & (heights > 0)
    arr = arr[keep]

    labels_arr = np.array(out_labels, dtype=np.int64)
    if len(labels_arr) == len(keep):
        labels_arr = labels_arr[keep]
    else:
        labels_arr = labels_arr[: len(arr)]

    return torch.from_numpy(arr), torch.from_numpy(labels_arr)


class AlbumentationsWrapper(BaseTransform):
    """
    Wrapper that applies an albumentations Compose to (image, target).

    Converts RF-DETR target (boxes in xyxy pixels, labels) to albumentations
    bboxes + class_labels, runs the transform, then converts back and updates
    target (boxes, labels, and optionally area). Handles empty bbox lists and
    auto-injects ``cropping_bbox`` for RandomCropNearBBox.
    """

    def __init__(
        self,
        transforms: Optional[List[Any]] = None,
        p: float = 1.0,
        name: Optional[str] = None,
    ):
        """
        Args:
            transforms: List of albumentations transform instances.
            p: Probability of applying the whole albumentations block.
            name: Name for logging.
        """
        super().__init__(p=p, name=name or "Albumentations")
        raw = transforms or []
        resolved: List[Any] = []
        for transform in raw:
            if getattr(transform, "__class__", None) and type(transform).__name__ == "function":
                try:
                    transform = transform()
                except Exception:
                    continue
            if transform is not None and not isinstance(transform, dict):
                resolved.append(transform)

        self._transforms = resolved
        self._transform: Optional[Any] = None
        self._transform_without_random_crop: Optional[Any] = None
        self._contains_spatial = False
        self._contains_spatial_without_random_crop = False
        self._random_crop_near_bbox_keys = _collect_random_crop_near_bbox_keys(self._transforms)
        self._last_applied_names: List[str] = []
        self._build_compose()

    @staticmethod
    def _build_single_compose(transforms: List[Any]) -> Tuple[Optional[Any], bool]:
        """Build one A.Compose instance and report whether it contains spatial transforms."""
        if not transforms:
            return None, False
        try:
            import albumentations as A
        except ImportError:
            return None, False

        contains_spatial = _contains_spatial(transforms)
        compose_kw: Dict[str, Any] = {}
        try:
            import inspect

            if "save_applied_params" in inspect.signature(A.Compose.__init__).parameters:
                compose_kw["save_applied_params"] = True
        except Exception:
            pass

        try:
            if contains_spatial:
                compose = A.Compose(
                    transforms,
                    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
                    **compose_kw,
                )
            else:
                compose = A.Compose(transforms, **compose_kw)
        except TypeError:
            compose_kw.pop("save_applied_params", None)
            if contains_spatial:
                compose = A.Compose(
                    transforms,
                    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
                )
            else:
                compose = A.Compose(transforms)

        return compose, contains_spatial

    def _build_compose(self) -> None:
        """Build A.Compose from the list of transforms."""
        _warn_if_random_crop_is_not_first_spatial(self._transforms)
        self._transform, self._contains_spatial = self._build_single_compose(self._transforms)

        fallback_transforms = _without_random_crop_near_bbox(self._transforms)
        if len(fallback_transforms) == len(self._transforms):
            self._transform_without_random_crop = None
            self._contains_spatial_without_random_crop = False
            return

        (
            self._transform_without_random_crop,
            self._contains_spatial_without_random_crop,
        ) = self._build_single_compose(fallback_transforms)

    @staticmethod
    def _run_compose(
        compose: Any,
        contains_spatial: bool,
        image: np.ndarray,
        bboxes_list: List[Tuple[float, float, float, float]],
        class_labels_list: List[int],
        extra_kw: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a compose object with either image-only or bbox-aware inputs."""
        if compose is None:
            return {"image": image, "bboxes": bboxes_list, "class_labels": class_labels_list}

        if contains_spatial:
            return compose(
                image=image,
                bboxes=bboxes_list,
                class_labels=class_labels_list,
                **extra_kw,
            )

        out = compose(image=image, **extra_kw)
        out.setdefault("bboxes", bboxes_list)
        out.setdefault("class_labels", class_labels_list)
        return out

    @staticmethod
    def _extract_output(
        out: Dict[str, Any],
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, List[str]]:
        """Extract image, boxes, labels, and applied transform names from compose output."""
        applied_names = [item[0] for item in out.get("applied_transforms", [])]
        out_img = out["image"]
        out_bboxes = out.get("bboxes", [])
        out_labels = out.get("class_labels", [])
        new_boxes, new_labels = _convert_output_boxes_and_labels(out_bboxes, out_labels)
        return out_img, new_boxes, new_labels, applied_names

    def apply(
        self,
        image: Union[PIL.Image.Image, np.ndarray],
        target: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply albumentations Compose to image and target.

        Args:
            image: PIL Image or numpy array (RGB).
            target: Dict with 'boxes' (N, 4) xyxy pixels and 'labels' (N,) int64.

        Returns:
            (image as np.ndarray uint8 RGB, updated target with boxes/labels/area).
        """
        if self._transform is None:
            img_np = _to_uint8_rgb(image)
            return img_np, dict(target)

        img_np = _to_uint8_rgb(image)
        height, width = img_np.shape[:2]
        boxes_np, bboxes_list, class_labels_list = _extract_boxes_and_labels(target, width=width, height=height)

        extra_kw: Dict[str, Any] = {}
        if "mosaic_metadata" in target:
            extra_kw["mosaic_metadata"] = target["mosaic_metadata"]
        extra_kw.update(
            _build_random_crop_targets(
                target=target,
                boxes_np=boxes_np,
                width=width,
                height=height,
                crop_keys=self._random_crop_near_bbox_keys,
            )
        )

        transform = self._transform
        contains_spatial = self._contains_spatial
        if (
            self._random_crop_near_bbox_keys
            and any(key not in extra_kw for key in self._random_crop_near_bbox_keys)
            and self._transform_without_random_crop is not None
        ):
            transform = self._transform_without_random_crop
            contains_spatial = self._contains_spatial_without_random_crop

        try:
            out = self._run_compose(
                compose=transform,
                contains_spatial=contains_spatial,
                image=img_np,
                bboxes_list=bboxes_list,
                class_labels_list=class_labels_list,
                extra_kw=extra_kw,
            )
        except Exception:
            if transform is self._transform and self._transform_without_random_crop is not None:
                try:
                    out = self._run_compose(
                        compose=self._transform_without_random_crop,
                        contains_spatial=self._contains_spatial_without_random_crop,
                        image=img_np,
                        bboxes_list=bboxes_list,
                        class_labels_list=class_labels_list,
                        extra_kw=extra_kw,
                    )
                    transform = self._transform_without_random_crop
                except Exception:
                    self._last_applied_names = []
                    return img_np, dict(target)
            else:
                self._last_applied_names = []
                return img_np, dict(target)

        out_img, new_boxes, new_labels, applied_names = self._extract_output(out)
        self._last_applied_names = applied_names

        should_retry_without_crop = (
            len(boxes_np) > 0
            and len(new_boxes) == 0
            and self._transform_without_random_crop is not None
            and (
                "RandomCropNearBBox" in applied_names
                or (
                    not applied_names
                    and transform is self._transform
                    and bool(self._random_crop_near_bbox_keys)
                )
            )
        )
        if should_retry_without_crop:
            try:
                fallback_out = self._run_compose(
                    compose=self._transform_without_random_crop,
                    contains_spatial=self._contains_spatial_without_random_crop,
                    image=img_np,
                    bboxes_list=bboxes_list,
                    class_labels_list=class_labels_list,
                    extra_kw=extra_kw,
                )
            except Exception:
                fallback_out = None
            if fallback_out is not None:
                out_img, new_boxes, new_labels, applied_names = self._extract_output(fallback_out)
                self._last_applied_names = applied_names

        out_h, out_w = out_img.shape[:2]
        new_boxes = clip_boxes(new_boxes, (out_h, out_w))

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

        if "mosaic_metadata" in new_target:
            new_target.pop("mosaic_metadata", None)
        for key in self._random_crop_near_bbox_keys:
            new_target.pop(key, None)

        return out_img, new_target

    def get_parameters(self) -> Dict[str, Any]:
        """Return parameters for logging."""
        out = {
            "num_transforms": len(self._transforms) if self._transforms else 0,
            "transform": self._transform is not None,
            "uses_random_crop_near_bbox": bool(self._random_crop_near_bbox_keys),
        }
        if self._last_applied_names:
            out["applied_names"] = self._last_applied_names
        return out
