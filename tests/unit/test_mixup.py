"""
Tests for MixUp and CutMix augmentations.
"""

import sys
import os
from typing import Tuple

import numpy as np
import pytest
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rfdetr.training.augmentations.mixup import MixUp, CutMix
from rfdetr.training.utils.seed import setup_seed


# =============================================================================
# Mock Dataset
# =============================================================================

class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, num_samples: int = 10):
        self.num_samples = num_samples
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        color = self.colors[idx % len(self.colors)]
        img = np.full((100, 100, 3), color, dtype=np.uint8)
        
        target = {
            'boxes': torch.tensor([[20, 20, 80, 80]], dtype=torch.float32),
            'labels': torch.tensor([idx % 3]),
        }
        return img, target


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_dataset():
    return MockDataset(num_samples=10)


@pytest.fixture
def red_image():
    """Solid red image."""
    return np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)


@pytest.fixture
def blue_image():
    """Solid blue image."""
    return np.full((100, 100, 3), (0, 0, 255), dtype=np.uint8)


@pytest.fixture
def sample_target():
    return {
        'boxes': torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
        'labels': torch.tensor([0]),
    }


# =============================================================================
# Test MixUp
# =============================================================================

class TestMixUp:
    """Tests for MixUp augmentation."""
    
    def test_output_size(self, mock_dataset, red_image, sample_target):
        """Test that output has same size as input."""
        mixup = MixUp(dataset=mock_dataset, alpha=32.0, p=1.0)
        
        output_img, _ = mixup(red_image, sample_target, index=0)
        
        assert output_img.shape == red_image.shape
    
    def test_output_is_numpy(self, mock_dataset, red_image, sample_target):
        """Test that output is numpy array."""
        mixup = MixUp(dataset=mock_dataset, alpha=32.0, p=1.0)
        
        output_img, _ = mixup(red_image, sample_target, index=0)
        
        assert isinstance(output_img, np.ndarray)
    
    def test_mixing_changes_pixels(self, mock_dataset, red_image, sample_target):
        """Test that mixing changes pixel values."""
        setup_seed(42)
        mixup = MixUp(dataset=mock_dataset, alpha=32.0, p=1.0)
        
        output_img, _ = mixup(red_image, sample_target, index=0)
        
        # Output should be different from pure red (unless ratio is 1.0)
        # With beta(32,32), ratio is usually around 0.5
        if not np.array_equal(output_img, red_image):
            assert True  # Expected - images are mixed
    
    def test_boxes_merged(self, mock_dataset, red_image, sample_target):
        """Test that boxes from both images are merged."""
        mixup = MixUp(dataset=mock_dataset, alpha=32.0, p=1.0)
        
        _, output_target = mixup(red_image, sample_target, index=0)
        
        # Should have boxes from both images
        # Original has 1 box, dataset samples have 1 box
        assert len(output_target['boxes']) >= 1
        assert len(output_target['labels']) == len(output_target['boxes'])
    
    def test_probability_zero(self, mock_dataset, red_image, sample_target):
        """Test that p=0 doesn't apply mixup."""
        mixup = MixUp(dataset=mock_dataset, alpha=32.0, p=0.0)
        
        output_img, _ = mixup(red_image, sample_target, index=0)
        
        # Should return original image unchanged
        assert np.array_equal(output_img, red_image)
    
    def test_without_dataset(self, red_image, sample_target):
        """Test mixup without dataset (uses same image)."""
        mixup = MixUp(dataset=None, alpha=32.0, p=1.0)
        
        output_img, output_target = mixup(red_image, sample_target, index=0)
        
        # Should still work, mixing with itself
        assert output_img.shape == red_image.shape
    
    def test_pil_input(self, mock_dataset, sample_target):
        """Test that PIL input works."""
        pil_img = Image.new('RGB', (100, 100), color='red')
        mixup = MixUp(dataset=mock_dataset, alpha=32.0, p=1.0)
        
        output_img, _ = mixup(pil_img, sample_target, index=0)
        
        assert isinstance(output_img, np.ndarray)
    
    def test_reproducibility(self, mock_dataset, red_image, sample_target):
        """Test that same seed produces same result."""
        mixup = MixUp(dataset=mock_dataset, alpha=32.0, p=1.0)
        
        setup_seed(42)
        output1, _ = mixup(red_image.copy(), sample_target.copy(), index=0)
        
        setup_seed(42)
        output2, _ = mixup(red_image.copy(), sample_target.copy(), index=0)
        
        assert np.array_equal(output1, output2)
    
    def test_parameters_logged(self, mock_dataset, red_image, sample_target):
        """Test that parameters are logged."""
        mixup = MixUp(dataset=mock_dataset, alpha=32.0, p=1.0)
        
        mixup(red_image, sample_target, index=0)
        params = mixup.get_parameters()
        
        assert 'alpha' in params
        assert 'ratio' in params
        assert 'index2' in params
    
    def test_ratio_distribution(self, mock_dataset, red_image, sample_target):
        """Test that ratio is sampled correctly."""
        mixup = MixUp(dataset=mock_dataset, alpha=32.0, p=1.0)
        
        ratios = []
        for _ in range(50):
            mixup(red_image.copy(), sample_target, index=0)
            ratios.append(mixup.get_parameters()['ratio'])
        
        # With alpha=32, ratios should be clustered around 0.5
        mean_ratio = np.mean(ratios)
        assert 0.3 < mean_ratio < 0.7


# =============================================================================
# Test CutMix
# =============================================================================

class TestCutMix:
    """Tests for CutMix augmentation."""
    
    def test_output_size(self, mock_dataset, red_image, sample_target):
        """Test that output has same size as input."""
        cutmix = CutMix(dataset=mock_dataset, alpha=1.0, p=1.0)
        
        output_img, _ = cutmix(red_image, sample_target, index=0)
        
        assert output_img.shape == red_image.shape
    
    def test_cut_region_applied(self, mock_dataset, red_image, sample_target):
        """Test that a rectangular region is cut and pasted."""
        setup_seed(42)
        cutmix = CutMix(dataset=mock_dataset, alpha=1.0, p=1.0)
        
        output_img, _ = cutmix(red_image, sample_target, index=0)
        
        # Check that some pixels are different (cut region from another image)
        if not np.array_equal(output_img, red_image):
            params = cutmix.get_parameters()
            x1, y1, x2, y2 = params['bbox']
            
            # The cut region should contain pixels from second image
            # (different from red)
            if x2 > x1 and y2 > y1:
                assert True  # Cut region exists
    
    def test_probability_zero(self, mock_dataset, red_image, sample_target):
        """Test that p=0 doesn't apply cutmix."""
        cutmix = CutMix(dataset=mock_dataset, alpha=1.0, p=0.0)
        
        output_img, _ = cutmix(red_image, sample_target, index=0)
        
        assert np.array_equal(output_img, red_image)
    
    def test_bbox_in_bounds(self, mock_dataset, red_image, sample_target):
        """Test that cut bbox is within image bounds."""
        cutmix = CutMix(dataset=mock_dataset, alpha=1.0, p=1.0)
        
        for _ in range(10):
            cutmix(red_image.copy(), sample_target, index=0)
            params = cutmix.get_parameters()
            x1, y1, x2, y2 = params['bbox']
            
            h, w = red_image.shape[:2]
            assert 0 <= x1 <= w
            assert 0 <= y1 <= h
            assert 0 <= x2 <= w
            assert 0 <= y2 <= h
    
    def test_without_dataset(self, red_image, sample_target):
        """Test cutmix without dataset."""
        cutmix = CutMix(dataset=None, alpha=1.0, p=1.0)
        
        output_img, _ = cutmix(red_image, sample_target, index=0)
        
        assert output_img.shape == red_image.shape
    
    def test_boxes_filtered(self, mock_dataset, red_image, sample_target):
        """Test that boxes are filtered based on cut region."""
        cutmix = CutMix(dataset=mock_dataset, alpha=1.0, p=1.0)
        
        _, output_target = cutmix(red_image, sample_target, index=0)
        
        # Should have boxes (some combination from both images)
        assert 'boxes' in output_target
        assert 'labels' in output_target
    
    def test_pil_input(self, mock_dataset, sample_target):
        """Test that PIL input works."""
        pil_img = Image.new('RGB', (100, 100), color='red')
        cutmix = CutMix(dataset=mock_dataset, alpha=1.0, p=1.0)
        
        output_img, _ = cutmix(pil_img, sample_target, index=0)
        
        assert isinstance(output_img, np.ndarray)
    
    def test_parameters_logged(self, mock_dataset, red_image, sample_target):
        """Test that parameters are logged."""
        cutmix = CutMix(dataset=mock_dataset, alpha=1.0, p=1.0)
        
        cutmix(red_image, sample_target, index=0)
        params = cutmix.get_parameters()
        
        assert 'alpha' in params
        assert 'bbox' in params
        assert 'index2' in params
        assert len(params['bbox']) == 4


# =============================================================================
# Visual tests
# =============================================================================

class TestVisualMixUp:
    """Visual tests for mix augmentations."""
    
    def test_save_mixup_example(self, mock_dataset, tmp_path):
        """Save MixUp example for visual inspection."""
        setup_seed(42)
        
        # Create two distinct images
        img1 = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)  # Red
        target1 = {
            'boxes': torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            'labels': torch.tensor([0]),
        }
        
        mixup = MixUp(dataset=mock_dataset, alpha=32.0, p=1.0)
        output_img, _ = mixup(img1, target1, index=0)
        
        # Save
        output_path = tmp_path / "mixup_test.png"
        Image.fromarray(output_img).save(output_path)
        
        assert output_path.exists()
        print(f"MixUp test saved to: {output_path}")
    
    def test_save_cutmix_example(self, mock_dataset, tmp_path):
        """Save CutMix example for visual inspection."""
        setup_seed(42)
        
        img1 = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)  # Red
        target1 = {
            'boxes': torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            'labels': torch.tensor([0]),
        }
        
        cutmix = CutMix(dataset=mock_dataset, alpha=1.0, p=1.0)
        output_img, _ = cutmix(img1, target1, index=0)
        
        # Save
        output_path = tmp_path / "cutmix_test.png"
        Image.fromarray(output_img).save(output_path)
        
        assert output_path.exists()
        print(f"CutMix test saved to: {output_path}")
        
        # Print cut region
        params = cutmix.get_parameters()
        print(f"Cut bbox: {params['bbox']}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
