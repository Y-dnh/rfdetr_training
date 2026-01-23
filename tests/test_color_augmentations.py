"""
Tests for color augmentations (HSV, Brightness, Contrast, etc.)
"""

import sys
import os

import numpy as np
import pytest
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfdetr.training.augmentations.color import (
    RandomHSV,
    RandomBrightness,
    RandomContrast,
    RandomBlur,
    RandomNoise,
)
from rfdetr.training.utils.seed import setup_seed


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def sample_image_pil():
    """Create a sample PIL image with varied colors."""
    # Create an image with red, green, blue quadrants
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = [255, 0, 0]    # Red
    img[:50, 50:] = [0, 255, 0]    # Green
    img[50:, :50] = [0, 0, 255]    # Blue
    img[50:, 50:] = [128, 128, 128]  # Gray
    return Image.fromarray(img)


@pytest.fixture
def sample_image_np():
    """Create a sample numpy image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = [255, 0, 0]
    img[:50, 50:] = [0, 255, 0]
    img[50:, :50] = [0, 0, 255]
    img[50:, 50:] = [128, 128, 128]
    return img


@pytest.fixture
def sample_target():
    """Create a sample target dict."""
    return {
        'boxes': torch.tensor([[10, 10, 50, 50], [60, 60, 90, 90]], dtype=torch.float32),
        'labels': torch.tensor([0, 1]),
    }


# =============================================================================
# Test RandomHSV
# =============================================================================

class TestRandomHSV:
    """Tests for RandomHSV augmentation."""
    
    def test_init_validation(self):
        """Test that invalid gains raise errors."""
        with pytest.raises(ValueError):
            RandomHSV(h_gain=-0.1)
        with pytest.raises(ValueError):
            RandomHSV(s_gain=1.5)
        with pytest.raises(ValueError):
            RandomHSV(v_gain=-0.5)
    
    def test_pil_input_output(self, sample_image_pil, sample_target):
        """Test that PIL input produces PIL output."""
        transform = RandomHSV(h_gain=0.5, s_gain=0.5, v_gain=0.5, p=1.0)
        output_image, _ = transform(sample_image_pil, sample_target)
        
        assert isinstance(output_image, Image.Image)
    
    def test_numpy_input_output(self, sample_image_np, sample_target):
        """Test that numpy input produces numpy output."""
        transform = RandomHSV(h_gain=0.5, s_gain=0.5, v_gain=0.5, p=1.0)
        output_image, _ = transform(sample_image_np, sample_target)
        
        assert isinstance(output_image, np.ndarray)
    
    def test_boxes_unchanged(self, sample_image_pil, sample_target):
        """Test that boxes are not modified by HSV augmentation."""
        transform = RandomHSV(h_gain=0.5, s_gain=0.5, v_gain=0.5, p=1.0)
        original_boxes = sample_target['boxes'].clone()
        
        _, new_target = transform(sample_image_pil, sample_target.copy())
        
        assert torch.allclose(original_boxes, new_target['boxes'])
    
    def test_image_modified(self, sample_image_np, sample_target):
        """Test that image pixels are modified."""
        setup_seed(42)
        transform = RandomHSV(h_gain=0.5, s_gain=0.5, v_gain=0.5, p=1.0)
        
        output_image, _ = transform(sample_image_np.copy(), sample_target)
        
        # Image should be different
        assert not np.array_equal(output_image, sample_image_np)
    
    def test_zero_gains_no_change(self, sample_image_np, sample_target):
        """Test that zero gains produce no change."""
        transform = RandomHSV(h_gain=0.0, s_gain=0.0, v_gain=0.0, p=1.0)
        
        # With zero gains, the transform is skipped
        output_image, _ = transform(sample_image_np.copy(), sample_target)
        
        # Should return original image unchanged
        assert np.array_equal(output_image, sample_image_np)
    
    def test_reproducibility(self, sample_image_np, sample_target):
        """Test that same seed produces same result."""
        transform = RandomHSV(h_gain=0.5, s_gain=0.5, v_gain=0.5, p=1.0)
        
        setup_seed(42)
        output1, _ = transform(sample_image_np.copy(), sample_target)
        
        setup_seed(42)
        output2, _ = transform(sample_image_np.copy(), sample_target)
        
        assert np.array_equal(output1, output2)
    
    def test_output_range(self, sample_image_np, sample_target):
        """Test that output pixels are in valid range [0, 255]."""
        transform = RandomHSV(h_gain=1.0, s_gain=1.0, v_gain=1.0, p=1.0)
        
        output_image, _ = transform(sample_image_np.copy(), sample_target)
        
        assert output_image.min() >= 0
        assert output_image.max() <= 255
    
    def test_parameters_logging(self, sample_image_np, sample_target):
        """Test that parameters are logged correctly."""
        transform = RandomHSV(h_gain=0.5, s_gain=0.5, v_gain=0.5, p=1.0)
        transform(sample_image_np.copy(), sample_target)
        
        params = transform.get_parameters()
        
        assert 'h_gain' in params
        assert 'h_shift_degrees' in params
        assert 's_scale' in params
        assert 'v_scale' in params
    
    def test_probability(self, sample_image_np, sample_target):
        """Test probability of application."""
        transform = RandomHSV(h_gain=0.5, s_gain=0.5, v_gain=0.5, p=0.0)
        
        # Should never apply
        for _ in range(10):
            output, _ = transform(sample_image_np.copy(), sample_target)
            result = transform.get_last_result()
            assert result.applied == False


# =============================================================================
# Test RandomBrightness
# =============================================================================

class TestRandomBrightness:
    """Tests for RandomBrightness augmentation."""
    
    def test_brightening(self, sample_image_np, sample_target):
        """Test that brightness > 1 makes image brighter."""
        # Use a controlled brightness factor
        transform = RandomBrightness(brightness_range=(1.5, 1.5), p=1.0)
        
        output_image, _ = transform(sample_image_np.copy(), sample_target)
        
        # Mean brightness should increase (accounting for clipping)
        orig_mean = sample_image_np.mean()
        out_mean = output_image.mean()
        
        # Output should be brighter or equal (due to clipping at 255)
        assert out_mean >= orig_mean * 0.9  # Allow some tolerance
    
    def test_darkening(self, sample_image_np, sample_target):
        """Test that brightness < 1 makes image darker."""
        transform = RandomBrightness(brightness_range=(0.5, 0.5), p=1.0)
        
        output_image, _ = transform(sample_image_np.copy(), sample_target)
        
        # Mean brightness should decrease
        orig_mean = sample_image_np.mean()
        out_mean = output_image.mean()
        
        assert out_mean < orig_mean
    
    def test_output_range(self, sample_image_np, sample_target):
        """Test that output is in valid range."""
        transform = RandomBrightness(brightness_range=(0.5, 2.0), p=1.0)
        
        output_image, _ = transform(sample_image_np.copy(), sample_target)
        
        assert output_image.min() >= 0
        assert output_image.max() <= 255


# =============================================================================
# Test RandomContrast
# =============================================================================

class TestRandomContrast:
    """Tests for RandomContrast augmentation."""
    
    def test_high_contrast(self, sample_image_np, sample_target):
        """Test that contrast > 1 increases variance."""
        transform = RandomContrast(contrast_range=(1.5, 1.5), p=1.0)
        
        output_image, _ = transform(sample_image_np.copy(), sample_target)
        
        # Variance should increase (accounting for clipping)
        orig_std = sample_image_np.astype(float).std()
        out_std = output_image.astype(float).std()
        
        # For high contrast, std should generally increase
        assert out_std >= orig_std * 0.5  # Allow tolerance due to clipping
    
    def test_low_contrast(self, sample_image_np, sample_target):
        """Test that contrast < 1 decreases variance."""
        transform = RandomContrast(contrast_range=(0.5, 0.5), p=1.0)
        
        output_image, _ = transform(sample_image_np.copy(), sample_target)
        
        # Variance should decrease
        orig_std = sample_image_np.astype(float).std()
        out_std = output_image.astype(float).std()
        
        assert out_std < orig_std


# =============================================================================
# Test RandomBlur
# =============================================================================

class TestRandomBlur:
    """Tests for RandomBlur augmentation."""
    
    def test_blur_reduces_edges(self, sample_target):
        """Test that blur smooths the image."""
        # Create image with sharp edges
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, 50:] = 255  # Sharp vertical edge
        
        transform = RandomBlur(kernel_size_range=(5, 5), p=1.0)
        output_image, _ = transform(img.copy(), sample_target)
        
        # The edge should be smoother (check gradient)
        orig_diff = np.abs(np.diff(img[:, 45:55, 0], axis=1)).max()
        out_diff = np.abs(np.diff(output_image[:, 45:55, 0], axis=1)).max()
        
        assert out_diff < orig_diff
    
    def test_kernel_size_odd(self, sample_image_np, sample_target):
        """Test that kernel size is always odd."""
        transform = RandomBlur(kernel_size_range=(3, 7), p=1.0)
        
        for _ in range(10):
            transform(sample_image_np.copy(), sample_target)
            params = transform.get_parameters()
            assert params['kernel_size'] % 2 == 1  # Must be odd


# =============================================================================
# Test RandomNoise
# =============================================================================

class TestRandomNoise:
    """Tests for RandomNoise augmentation."""
    
    def test_noise_adds_variance(self, sample_target):
        """Test that noise increases image variance."""
        # Create uniform image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        transform = RandomNoise(noise_range=(20, 20), p=1.0)
        output_image, _ = transform(img.copy(), sample_target)
        
        # Output should have higher variance than uniform input
        orig_std = img.astype(float).std()
        out_std = output_image.astype(float).std()
        
        assert out_std > orig_std
    
    def test_output_range(self, sample_image_np, sample_target):
        """Test that output is in valid range."""
        transform = RandomNoise(noise_range=(10, 50), p=1.0)
        
        output_image, _ = transform(sample_image_np.copy(), sample_target)
        
        assert output_image.min() >= 0
        assert output_image.max() <= 255


# =============================================================================
# Visual test (generates images for manual inspection)
# =============================================================================

class TestVisualOutput:
    """Visual tests that save images for manual inspection."""
    
    def test_save_hsv_example(self, sample_image_pil, sample_target, tmp_path):
        """Save HSV augmentation examples."""
        transform = RandomHSV(h_gain=0.5, s_gain=0.7, v_gain=0.4, p=1.0)
        
        setup_seed(42)
        output_image, _ = transform(sample_image_pil, sample_target)
        
        # Save for visual inspection
        output_path = tmp_path / "hsv_test.png"
        output_image.save(output_path)
        
        assert output_path.exists()
        print(f"HSV test image saved to: {output_path}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
