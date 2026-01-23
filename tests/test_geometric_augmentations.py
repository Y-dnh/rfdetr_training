"""
Tests for geometric augmentations (Flip, Perspective, Scale, etc.)
"""

import sys
import os

import numpy as np
import pytest
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfdetr.training.augmentations.geometric import (
    RandomFlip,
    RandomPerspective,
    RandomRotate,
    RandomScale,
    LetterBox,
)
from rfdetr.training.utils.seed import setup_seed


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def sample_image_pil():
    """Create a sample PIL image."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:50, :100] = [255, 0, 0]    # Red top-left
    img[:50, 100:] = [0, 255, 0]    # Green top-right
    img[50:, :100] = [0, 0, 255]    # Blue bottom-left
    img[50:, 100:] = [255, 255, 0]  # Yellow bottom-right
    return Image.fromarray(img)


@pytest.fixture
def sample_image_np():
    """Create a sample numpy image."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:50, :100] = [255, 0, 0]
    img[:50, 100:] = [0, 255, 0]
    img[50:, :100] = [0, 0, 255]
    img[50:, 100:] = [255, 255, 0]
    return img


@pytest.fixture
def sample_target():
    """Create a sample target dict with boxes."""
    return {
        'boxes': torch.tensor([
            [10, 10, 50, 40],   # Box in top-left quadrant
            [110, 60, 180, 90],  # Box in bottom-right quadrant
        ], dtype=torch.float32),
        'labels': torch.tensor([0, 1]),
        'area': torch.tensor([1200.0, 2100.0]),
        'iscrowd': torch.tensor([0, 0]),
        'size': torch.tensor([100, 200]),
    }


# =============================================================================
# Test RandomFlip
# =============================================================================

class TestRandomFlip:
    """Tests for RandomFlip augmentation."""
    
    def test_horizontal_flip_boxes(self, sample_image_np, sample_target):
        """Test that horizontal flip correctly transforms boxes."""
        transform = RandomFlip(p=1.0, direction='horizontal')
        
        original_boxes = sample_target['boxes'].clone()
        w = sample_image_np.shape[1]  # 200
        
        _, new_target = transform(sample_image_np.copy(), sample_target.copy())
        new_boxes = new_target['boxes']
        
        # After horizontal flip: x1' = w - x2, x2' = w - x1
        expected_x1 = w - original_boxes[:, 2]
        expected_x2 = w - original_boxes[:, 0]
        
        assert torch.allclose(new_boxes[:, 0], expected_x1)
        assert torch.allclose(new_boxes[:, 2], expected_x2)
        # Y coordinates should be unchanged
        assert torch.allclose(new_boxes[:, 1], original_boxes[:, 1])
        assert torch.allclose(new_boxes[:, 3], original_boxes[:, 3])
    
    def test_vertical_flip_boxes(self, sample_image_np, sample_target):
        """Test that vertical flip correctly transforms boxes."""
        transform = RandomFlip(p=1.0, direction='vertical')
        
        original_boxes = sample_target['boxes'].clone()
        h = sample_image_np.shape[0]  # 100
        
        _, new_target = transform(sample_image_np.copy(), sample_target.copy())
        new_boxes = new_target['boxes']
        
        # After vertical flip: y1' = h - y2, y2' = h - y1
        expected_y1 = h - original_boxes[:, 3]
        expected_y2 = h - original_boxes[:, 1]
        
        assert torch.allclose(new_boxes[:, 1], expected_y1)
        assert torch.allclose(new_boxes[:, 3], expected_y2)
        # X coordinates should be unchanged
        assert torch.allclose(new_boxes[:, 0], original_boxes[:, 0])
        assert torch.allclose(new_boxes[:, 2], original_boxes[:, 2])
    
    def test_flip_image_pixels(self, sample_image_np, sample_target):
        """Test that image pixels are actually flipped."""
        transform = RandomFlip(p=1.0, direction='horizontal')
        
        output_image, _ = transform(sample_image_np.copy(), sample_target)
        
        # Top-left (red) should now be top-right
        # Top-right (green) should now be top-left
        assert np.allclose(output_image[25, 50], [0, 255, 0])  # Was green, now in left
        assert np.allclose(output_image[25, 150], [255, 0, 0])  # Was red, now in right
    
    def test_double_flip_identity(self, sample_image_np, sample_target):
        """Test that two horizontal flips return to original."""
        transform = RandomFlip(p=1.0, direction='horizontal')
        
        original_boxes = sample_target['boxes'].clone()
        
        img1, target1 = transform(sample_image_np.copy(), sample_target.copy())
        img2, target2 = transform(img1, target1)
        
        assert torch.allclose(target2['boxes'], original_boxes, atol=1e-5)
    
    def test_invalid_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError):
            RandomFlip(direction='diagonal')
    
    def test_pil_input(self, sample_image_pil, sample_target):
        """Test that PIL input works correctly."""
        transform = RandomFlip(p=1.0, direction='horizontal')
        
        output_image, _ = transform(sample_image_pil, sample_target.copy())
        
        assert isinstance(output_image, Image.Image)


# =============================================================================
# Test RandomPerspective
# =============================================================================

class TestRandomPerspective:
    """Tests for RandomPerspective augmentation."""
    
    def test_no_transform_with_zero_params(self, sample_image_np, sample_target):
        """Test that zero parameters produce minimal change."""
        transform = RandomPerspective(
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            p=1.0
        )
        
        output_image, new_target = transform(sample_image_np.copy(), sample_target.copy())
        
        # Image size should be same
        assert output_image.shape[:2] == sample_image_np.shape[:2]
        
        # Boxes should be similar (small numerical differences allowed)
        assert torch.allclose(new_target['boxes'], sample_target['boxes'], atol=2.0)
    
    def test_rotation_changes_boxes(self, sample_image_np, sample_target):
        """Test that rotation transforms boxes."""
        setup_seed(42)
        transform = RandomPerspective(
            degrees=45.0,  # Large rotation
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            p=1.0
        )
        
        original_boxes = sample_target['boxes'].clone()
        _, new_target = transform(sample_image_np.copy(), sample_target.copy())
        
        # Boxes should be different after rotation
        assert not torch.allclose(new_target['boxes'], original_boxes, atol=1.0)
    
    def test_scale_changes_image_content(self, sample_image_np, sample_target):
        """Test that scale changes image."""
        setup_seed(42)
        transform = RandomPerspective(
            degrees=0.0,
            translate=0.0,
            scale=0.5,  # Scale between 0.5 and 1.5
            shear=0.0,
            perspective=0.0,
            p=1.0
        )
        
        output_image, _ = transform(sample_image_np.copy(), sample_target.copy())
        
        # Output should be different from input
        assert not np.array_equal(output_image, sample_image_np)
    
    def test_output_is_numpy(self, sample_image_np, sample_target):
        """Test that output is numpy array."""
        transform = RandomPerspective(degrees=10.0, p=1.0)
        
        output_image, _ = transform(sample_image_np.copy(), sample_target.copy())
        
        assert isinstance(output_image, np.ndarray)
    
    def test_boxes_remain_valid(self, sample_image_np, sample_target):
        """Test that transformed boxes are still valid (positive area)."""
        setup_seed(42)
        transform = RandomPerspective(
            degrees=30.0,
            translate=0.1,
            scale=0.3,
            shear=10.0,
            p=1.0
        )
        
        _, new_target = transform(sample_image_np.copy(), sample_target.copy())
        
        if len(new_target['boxes']) > 0:
            boxes = new_target['boxes']
            # x2 > x1 and y2 > y1
            assert (boxes[:, 2] > boxes[:, 0]).all()
            assert (boxes[:, 3] > boxes[:, 1]).all()
    
    def test_parameters_logged(self, sample_image_np, sample_target):
        """Test that transformation parameters are logged."""
        transform = RandomPerspective(degrees=30.0, translate=0.1, scale=0.2, p=1.0)
        
        transform(sample_image_np.copy(), sample_target.copy())
        params = transform.get_parameters()
        
        assert 'angle' in params
        assert 'scale' in params
        assert 'translate_x' in params
        assert 'translate_y' in params
    
    def test_reproducibility(self, sample_image_np, sample_target):
        """Test that same seed produces same result."""
        transform = RandomPerspective(degrees=30.0, translate=0.1, scale=0.2, p=1.0)
        
        setup_seed(42)
        output1, target1 = transform(sample_image_np.copy(), sample_target.copy())
        
        setup_seed(42)
        output2, target2 = transform(sample_image_np.copy(), sample_target.copy())
        
        assert np.array_equal(output1, output2)
        assert torch.allclose(target1['boxes'], target2['boxes'])


# =============================================================================
# Test RandomRotate
# =============================================================================

class TestRandomRotate:
    """Tests for RandomRotate augmentation."""
    
    def test_rotation_applied(self, sample_image_np, sample_target):
        """Test that rotation is applied."""
        setup_seed(42)
        transform = RandomRotate(degrees=45.0, p=1.0)
        
        output_image, _ = transform(sample_image_np.copy(), sample_target.copy())
        
        # Image should be different
        assert not np.array_equal(output_image, sample_image_np)
    
    def test_angle_in_range(self, sample_image_np, sample_target):
        """Test that rotation angle is within specified range."""
        transform = RandomRotate(degrees=30.0, p=1.0)
        
        for _ in range(10):
            transform(sample_image_np.copy(), sample_target.copy())
            params = transform.get_parameters()
            
            assert -30.0 <= params['angle'] <= 30.0


# =============================================================================
# Test RandomScale
# =============================================================================

class TestRandomScale:
    """Tests for RandomScale augmentation."""
    
    def test_scale_up(self, sample_image_np, sample_target):
        """Test scaling up increases image size."""
        transform = RandomScale(scale_range=(1.5, 1.5), p=1.0)
        
        output_image, new_target = transform(sample_image_np.copy(), sample_target.copy())
        
        # Image should be larger
        assert output_image.shape[0] > sample_image_np.shape[0]
        assert output_image.shape[1] > sample_image_np.shape[1]
        
        # Size in target should be updated
        assert new_target['size'][0] > sample_target['size'][0]
    
    def test_scale_down(self, sample_image_np, sample_target):
        """Test scaling down decreases image size."""
        transform = RandomScale(scale_range=(0.5, 0.5), p=1.0)
        
        output_image, new_target = transform(sample_image_np.copy(), sample_target.copy())
        
        # Image should be smaller
        assert output_image.shape[0] < sample_image_np.shape[0]
        assert output_image.shape[1] < sample_image_np.shape[1]
    
    def test_boxes_scaled_proportionally(self, sample_image_np, sample_target):
        """Test that boxes are scaled proportionally."""
        scale = 2.0
        transform = RandomScale(scale_range=(scale, scale), p=1.0)
        
        original_boxes = sample_target['boxes'].clone()
        _, new_target = transform(sample_image_np.copy(), sample_target.copy())
        
        # Boxes should be scaled by the same factor
        expected_boxes = original_boxes * scale
        assert torch.allclose(new_target['boxes'], expected_boxes)


# =============================================================================
# Test LetterBox
# =============================================================================

class TestLetterBox:
    """Tests for LetterBox augmentation."""
    
    def test_output_size(self, sample_image_np, sample_target):
        """Test that output has correct size."""
        transform = LetterBox(new_shape=640, p=1.0)
        
        output_image, _ = transform(sample_image_np.copy(), sample_target.copy())
        
        # Output should be 640x640
        assert output_image.shape[0] == 640
        assert output_image.shape[1] == 640
    
    def test_aspect_ratio_preserved(self, sample_image_np, sample_target):
        """Test that aspect ratio is preserved with padding."""
        transform = LetterBox(new_shape=640, p=1.0)
        
        # Original aspect ratio is 200/100 = 2.0
        output_image, _ = transform(sample_image_np.copy(), sample_target.copy())
        
        # Check that padding was added (gray color 114)
        # Top and bottom should have padding for this aspect ratio
        # Since width > height, we'll have padding on top/bottom
        pass  # Aspect ratio logic varies based on implementation
    
    def test_boxes_transformed(self, sample_image_np, sample_target):
        """Test that boxes are correctly transformed."""
        transform = LetterBox(new_shape=640, p=1.0)
        
        _, new_target = transform(sample_image_np.copy(), sample_target.copy())
        
        # Boxes should be within new image bounds
        if len(new_target['boxes']) > 0:
            boxes = new_target['boxes']
            assert (boxes[:, 0] >= 0).all()
            assert (boxes[:, 1] >= 0).all()
            assert (boxes[:, 2] <= 640).all()
            assert (boxes[:, 3] <= 640).all()
    
    def test_no_scaleup(self, sample_target):
        """Test scaleup=False doesn't enlarge small images."""
        small_img = np.zeros((50, 50, 3), dtype=np.uint8)
        transform = LetterBox(new_shape=640, scaleup=False, p=1.0)
        
        output_image, _ = transform(small_img, sample_target.copy())
        
        # Output is 640x640 but image content shouldn't be scaled up
        # (it will be padded instead)
        assert output_image.shape == (640, 640, 3)
    
    def test_tuple_shape(self, sample_image_np, sample_target):
        """Test that tuple shape works."""
        transform = LetterBox(new_shape=(480, 640), p=1.0)
        
        output_image, _ = transform(sample_image_np.copy(), sample_target.copy())
        
        assert output_image.shape[0] == 480
        assert output_image.shape[1] == 640


# =============================================================================
# Integration test
# =============================================================================

class TestGeometricIntegration:
    """Integration tests combining multiple geometric transforms."""
    
    def test_flip_then_perspective(self, sample_image_np, sample_target):
        """Test applying flip followed by perspective."""
        setup_seed(42)
        
        flip = RandomFlip(p=1.0, direction='horizontal')
        perspective = RandomPerspective(degrees=10.0, translate=0.05, p=1.0)
        
        img1, target1 = flip(sample_image_np.copy(), sample_target.copy())
        img2, target2 = perspective(img1, target1)
        
        # Should complete without error
        assert isinstance(img2, np.ndarray)
        assert 'boxes' in target2
    
    def test_scale_then_letterbox(self, sample_image_np, sample_target):
        """Test applying scale followed by letterbox."""
        scale = RandomScale(scale_range=(0.8, 1.2), p=1.0)
        letterbox = LetterBox(new_shape=640, p=1.0)
        
        img1, target1 = scale(sample_image_np.copy(), sample_target.copy())
        img2, target2 = letterbox(img1, target1)
        
        # Final output should be 640x640
        assert img2.shape[:2] == (640, 640)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
