"""
Tests for base augmentation classes and wrappers.
"""

import sys
import os

import numpy as np
import pytest
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rfdetr.training.augmentations.base import (
    BaseTransform,
    Compose,
    AugmentationResult,
    RandomHorizontalFlip,
    RandomResize,
    SquareResize,
    ToTensor,
    Normalize,
    clip_boxes,
    filter_small_boxes,
    xyxy_to_cxcywh,
    cxcywh_to_xyxy,
    normalize_boxes,
    denormalize_boxes,
)


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def sample_image():
    """Create a sample PIL image."""
    return Image.new('RGB', (640, 480), color='red')


@pytest.fixture
def sample_target():
    """Create a sample target dict with boxes and labels."""
    return {
        'boxes': torch.tensor([
            [100, 100, 200, 200],  # Box 1
            [300, 150, 400, 350],  # Box 2
        ], dtype=torch.float32),
        'labels': torch.tensor([0, 1]),
        'area': torch.tensor([10000.0, 20000.0]),
        'iscrowd': torch.tensor([0, 0]),
        'image_id': 1,
    }


# =============================================================================
# Test BaseTransform
# =============================================================================

class DummyTransform(BaseTransform):
    """Dummy transform for testing."""
    
    def apply(self, image, target):
        target = target.copy()
        target['dummy_applied'] = True
        return image, target
    
    def get_parameters(self):
        return {'dummy_param': 42}


class TestBaseTransform:
    """Tests for BaseTransform class."""
    
    def test_probability_validation(self):
        """Test that invalid probabilities raise errors."""
        with pytest.raises(ValueError):
            DummyTransform(p=-0.1)
        with pytest.raises(ValueError):
            DummyTransform(p=1.1)
    
    def test_probability_always_apply(self, sample_image, sample_target):
        """Test that p=1.0 always applies the transform."""
        transform = DummyTransform(p=1.0)
        
        # Apply 10 times, should always apply
        for _ in range(10):
            _, target = transform(sample_image, sample_target.copy())
            assert target.get('dummy_applied', False)
    
    def test_probability_never_apply(self, sample_image, sample_target):
        """Test that p=0.0 never applies the transform."""
        transform = DummyTransform(p=0.0)
        
        # Apply 10 times, should never apply
        for _ in range(10):
            _, target = transform(sample_image, sample_target.copy())
            assert not target.get('dummy_applied', False)
    
    def test_last_result_tracking(self, sample_image, sample_target):
        """Test that last result is tracked correctly."""
        transform = DummyTransform(p=1.0)
        transform(sample_image, sample_target.copy())
        
        result = transform.get_last_result()
        assert result is not None
        assert result.applied == True
        assert result.name == 'DummyTransform'
        assert result.parameters['dummy_param'] == 42
    
    def test_repr(self):
        """Test string representation."""
        transform = DummyTransform(p=0.5)
        assert 'DummyTransform' in repr(transform)
        assert '0.5' in repr(transform)


# =============================================================================
# Test Compose
# =============================================================================

class TestCompose:
    """Tests for Compose class."""
    
    def test_sequential_application(self, sample_image, sample_target):
        """Test that transforms are applied sequentially."""
        class AddOneTransform(BaseTransform):
            def apply(self, image, target):
                target = target.copy()
                target['count'] = target.get('count', 0) + 1
                return image, target
        
        compose = Compose([
            AddOneTransform(p=1.0),
            AddOneTransform(p=1.0),
            AddOneTransform(p=1.0),
        ])
        
        _, target = compose(sample_image, sample_target.copy())
        assert target['count'] == 3
    
    def test_logging_enabled(self, sample_image, sample_target):
        """Test that logging collects augmentation results."""
        compose = Compose([
            DummyTransform(p=1.0),
            DummyTransform(p=1.0),
        ], log_augmentations=True)
        
        compose(sample_image, sample_target.copy())
        log = compose.get_last_log()
        
        assert len(log) == 2
        assert all(isinstance(r, AugmentationResult) for r in log)
        assert all(r.applied for r in log)
    
    def test_logging_disabled(self, sample_image, sample_target):
        """Test that logging doesn't collect when disabled."""
        compose = Compose([
            DummyTransform(p=1.0),
        ], log_augmentations=False)
        
        compose(sample_image, sample_target.copy())
        log = compose.get_last_log()
        
        assert len(log) == 0
    
    def test_get_applied_augmentations(self, sample_image, sample_target):
        """Test filtering to only applied augmentations."""
        compose = Compose([
            DummyTransform(p=1.0),
            DummyTransform(p=0.0),
        ], log_augmentations=True)
        
        compose(sample_image, sample_target.copy())
        applied = compose.get_applied_augmentations()
        
        assert len(applied) == 1
        assert applied[0].applied == True


# =============================================================================
# Test Wrapped Transforms
# =============================================================================

class TestRandomHorizontalFlip:
    """Tests for RandomHorizontalFlip wrapper."""
    
    def test_flip_updates_boxes(self, sample_image, sample_target):
        """Test that flipping updates box coordinates correctly."""
        transform = RandomHorizontalFlip(p=1.0)
        
        original_boxes = sample_target['boxes'].clone()
        _, new_target = transform(sample_image, sample_target.copy())
        
        # After horizontal flip, x coordinates should be mirrored
        w = sample_image.size[0]  # 640
        expected_x1 = w - original_boxes[:, 2]
        expected_x2 = w - original_boxes[:, 0]
        
        assert torch.allclose(new_target['boxes'][:, 0], expected_x1)
        assert torch.allclose(new_target['boxes'][:, 2], expected_x2)
    
    def test_no_flip(self, sample_image, sample_target):
        """Test that p=0 doesn't flip."""
        transform = RandomHorizontalFlip(p=0.0)
        
        original_boxes = sample_target['boxes'].clone()
        _, new_target = transform(sample_image, sample_target.copy())
        
        assert torch.allclose(new_target['boxes'], original_boxes)


class TestRandomResize:
    """Tests for RandomResize wrapper."""
    
    def test_resize_updates_boxes(self, sample_image, sample_target):
        """Test that resizing updates box coordinates proportionally."""
        transform = RandomResize(sizes=[320], p=1.0)
        
        _, new_target = transform(sample_image, sample_target.copy())
        
        # New size should be 320
        new_size = new_target['size']
        assert new_size.min() == 320 or new_size.max() >= 320
    
    def test_multiple_sizes(self, sample_image, sample_target):
        """Test that different sizes can be selected."""
        sizes = [320, 480, 640]
        transform = RandomResize(sizes=sizes, p=1.0)
        
        selected_sizes = set()
        for _ in range(50):
            transform(sample_image, sample_target.copy())
            params = transform.get_parameters()
            selected_sizes.add(params['chosen_size'])
        
        # Should have selected multiple different sizes
        assert len(selected_sizes) >= 2


class TestSquareResize:
    """Tests for SquareResize wrapper."""
    
    def test_resize_to_square(self, sample_image, sample_target):
        """Test that image is resized to square."""
        transform = SquareResize(sizes=[512], p=1.0)
        
        _, new_target = transform(sample_image, sample_target.copy())
        new_size = new_target['size']
        
        assert new_size[0] == new_size[1] == 512


class TestToTensor:
    """Tests for ToTensor wrapper."""
    
    def test_converts_to_tensor(self, sample_image, sample_target):
        """Test that PIL image is converted to tensor."""
        transform = ToTensor(p=1.0)
        
        new_image, _ = transform(sample_image, sample_target.copy())
        
        assert isinstance(new_image, torch.Tensor)
        assert new_image.shape[0] == 3  # RGB channels


class TestNormalize:
    """Tests for Normalize wrapper."""
    
    def test_normalize_converts_boxes(self, sample_target):
        """Test that normalization converts boxes to cxcywh format."""
        # First convert to tensor
        image = torch.rand(3, 480, 640)
        transform = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        _, new_target = transform(image, sample_target.copy())
        
        # Boxes should be in cxcywh format and normalized
        boxes = new_target['boxes']
        assert boxes.max() <= 1.0
        assert boxes.min() >= 0.0


# =============================================================================
# Test Box Utility Functions
# =============================================================================

class TestBoxUtils:
    """Tests for box utility functions."""
    
    def test_clip_boxes(self):
        """Test clipping boxes to image boundaries."""
        boxes = torch.tensor([
            [-10, -20, 100, 200],  # Negative coords
            [500, 400, 700, 600],  # Outside image
        ], dtype=torch.float32)
        
        clipped = clip_boxes(boxes, (480, 640))
        
        assert clipped[0, 0] == 0  # x1 clipped
        assert clipped[0, 1] == 0  # y1 clipped
        assert clipped[1, 2] == 640  # x2 clipped
        assert clipped[1, 3] == 480  # y2 clipped
    
    def test_filter_small_boxes(self):
        """Test filtering out small boxes."""
        boxes = torch.tensor([
            [0, 0, 100, 100],  # Large box (100x100)
            [0, 0, 0.5, 0.5],  # Small box (0.5x0.5)
            [0, 0, 50, 50],   # Medium box (50x50)
        ], dtype=torch.float32)
        
        filtered, keep = filter_small_boxes(boxes, min_size=10)
        
        assert len(filtered) == 2  # Only 2 boxes kept
        assert keep.sum() == 2
    
    def test_xyxy_to_cxcywh_conversion(self):
        """Test xyxy to cxcywh conversion."""
        boxes_xyxy = torch.tensor([
            [100, 100, 200, 200],  # (100, 100) to (200, 200)
        ], dtype=torch.float32)
        
        boxes_cxcywh = xyxy_to_cxcywh(boxes_xyxy)
        
        assert boxes_cxcywh[0, 0] == 150  # cx
        assert boxes_cxcywh[0, 1] == 150  # cy
        assert boxes_cxcywh[0, 2] == 100  # w
        assert boxes_cxcywh[0, 3] == 100  # h
    
    def test_cxcywh_to_xyxy_conversion(self):
        """Test cxcywh to xyxy conversion."""
        boxes_cxcywh = torch.tensor([
            [150, 150, 100, 100],  # center (150, 150), size (100, 100)
        ], dtype=torch.float32)
        
        boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)
        
        assert boxes_xyxy[0, 0] == 100  # x1
        assert boxes_xyxy[0, 1] == 100  # y1
        assert boxes_xyxy[0, 2] == 200  # x2
        assert boxes_xyxy[0, 3] == 200  # y2
    
    def test_roundtrip_conversion(self):
        """Test that xyxy -> cxcywh -> xyxy is identity."""
        boxes_xyxy = torch.tensor([
            [100, 50, 200, 150],
            [0, 0, 640, 480],
        ], dtype=torch.float32)
        
        boxes_cxcywh = xyxy_to_cxcywh(boxes_xyxy)
        boxes_back = cxcywh_to_xyxy(boxes_cxcywh)
        
        assert torch.allclose(boxes_xyxy, boxes_back)
    
    def test_normalize_denormalize_roundtrip(self):
        """Test that normalize -> denormalize is identity."""
        boxes = torch.tensor([
            [100, 50, 200, 150],
            [320, 240, 640, 480],
        ], dtype=torch.float32)
        
        image_size = (480, 640)
        
        normalized = normalize_boxes(boxes, image_size)
        denormalized = denormalize_boxes(normalized, image_size)
        
        assert torch.allclose(boxes, denormalized)
    
    def test_normalized_boxes_in_range(self):
        """Test that normalized boxes are in [0, 1] range."""
        boxes = torch.tensor([
            [0, 0, 640, 480],  # Full image
        ], dtype=torch.float32)
        
        normalized = normalize_boxes(boxes, (480, 640))
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
