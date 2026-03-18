"""
Tests for AlbumentationsWrapper integration details.
"""

import os
import sys

import numpy as np
import pytest
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

A = pytest.importorskip("albumentations")

from rfdetr.training.augmentations.albumentations_wrapper import AlbumentationsWrapper


def make_target(boxes: list[list[float]], labels: list[int], height: int, width: int) -> dict:
    """Create a minimal RF-DETR target for wrapper tests."""
    if boxes:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        areas_tensor = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
    else:
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.zeros((0,), dtype=torch.int64)
        areas_tensor = torch.zeros((0,), dtype=torch.float32)

    size_tensor = torch.tensor([height, width], dtype=torch.int64)
    return {
        "boxes": boxes_tensor,
        "labels": labels_tensor,
        "area": areas_tensor,
        "orig_size": size_tensor.clone(),
        "size": size_tensor.clone(),
    }


@pytest.fixture
def striped_image() -> np.ndarray:
    """Create an image whose left/right halves are easy to distinguish after flips."""
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :8] = [255, 0, 0]
    image[:, 8:] = [0, 0, 255]
    return image


class TestAlbumentationsWrapper:
    """Tests for RandomCropNearBBox-specific wrapper behavior."""

    def test_random_crop_near_bbox_auto_injects_crop_box(self) -> None:
        """Wrapper should auto-generate cropping_bbox from the current sample target."""
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        target = make_target([[10, 20, 30, 40]], [1], height=64, width=64)
        wrapper = AlbumentationsWrapper(
            transforms=[A.RandomCropNearBBox(max_part_shift=0.0, p=1.0)],
        )

        out_img, out_target = wrapper.apply(image, target)

        assert out_img.shape == (20, 20, 3)
        assert torch.allclose(
            out_target["boxes"],
            torch.tensor([[0.0, 0.0, 20.0, 20.0]], dtype=torch.float32),
        )
        assert out_target["size"].tolist() == [20, 20]
        assert "cropping_bbox" not in out_target

    def test_random_crop_near_bbox_supports_custom_crop_key(self) -> None:
        """Wrapper should detect custom cropping_bbox_key names from albumentations config."""
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        target = make_target([[8, 12, 24, 28]], [3], height=64, width=64)
        wrapper = AlbumentationsWrapper(
            transforms=[
                A.RandomCropNearBBox(
                    max_part_shift=0.0,
                    cropping_bbox_key="focus_bbox",
                    p=1.0,
                )
            ],
        )

        out_img, out_target = wrapper.apply(image, target)

        assert out_img.shape == (16, 16, 3)
        assert torch.allclose(
            out_target["boxes"],
            torch.tensor([[0.0, 0.0, 16.0, 16.0]], dtype=torch.float32),
        )
        assert "focus_bbox" not in out_target

    def test_random_crop_near_bbox_falls_back_when_sample_has_no_boxes(self, striped_image: np.ndarray) -> None:
        """If a sample has no boxes, wrapper should skip the crop but keep remaining transforms."""
        target = make_target([], [], height=16, width=16)
        wrapper = AlbumentationsWrapper(
            transforms=[
                A.RandomCropNearBBox(max_part_shift=0.0, p=1.0),
                A.HorizontalFlip(p=1.0),
            ],
        )

        out_img, out_target = wrapper.apply(striped_image, target)

        assert out_img.shape == striped_image.shape
        assert np.array_equal(out_img[:, 0], striped_image[:, -1])
        assert out_target["boxes"].shape == (0, 4)
        assert wrapper.get_parameters()["applied_names"] == ["HorizontalFlip"]

    def test_random_crop_near_bbox_retries_without_crop_when_crop_invalidates_boxes(
        self,
        striped_image: np.ndarray,
    ) -> None:
        """
        If crop is placed after another spatial transform and drops all boxes, wrapper should recover by
        re-running the compose without RandomCropNearBBox.
        """
        target = make_target([[1, 1, 5, 5]], [1], height=16, width=16)
        with pytest.warns(UserWarning, match="RandomCropNearBBox should appear before other spatial"):
            wrapper = AlbumentationsWrapper(
                transforms=[
                    A.HorizontalFlip(p=1.0),
                    A.RandomCropNearBBox(max_part_shift=0.0, p=1.0),
                ],
            )

        out_img, out_target = wrapper.apply(striped_image, target)

        assert out_img.shape == striped_image.shape
        assert np.array_equal(out_img[:, 0], striped_image[:, -1])
        assert torch.allclose(
            out_target["boxes"],
            torch.tensor([[11.0, 1.0, 15.0, 5.0]], dtype=torch.float32),
        )
        assert wrapper.get_parameters()["applied_names"] == ["HorizontalFlip"]
