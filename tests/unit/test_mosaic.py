"""
Tests for Mosaic augmentation.
"""

import sys
import os
import json
from typing import Tuple

import numpy as np
import pytest
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rfdetr.training.augmentations.mosaic import Mosaic, Mosaic9
from rfdetr.training.utils.seed import setup_seed


# =============================================================================
# Mock Dataset for testing
# =============================================================================

class MockDataset:
    """Mock dataset for testing mosaic without real data."""
    
    def __init__(self, num_samples: int = 10, img_size: Tuple = (100, 100)):
        self.num_samples = num_samples
        self.img_size = img_size
        
        # Generate random images with different colors
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 128, 128),  # Gray
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 128, 255),  # Light blue
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Return a colored image with random boxes."""
        h, w = self.img_size
        color = self.colors[idx % len(self.colors)]
        
        # Create solid color image
        img = np.full((h, w, 3), color, dtype=np.uint8)
        
        # Add some variation
        img[h//4:3*h//4, w//4:3*w//4] = [c // 2 for c in color]
        
        # Generate random boxes
        num_boxes = np.random.randint(1, 4)
        boxes = []
        for _ in range(num_boxes):
            x1 = np.random.randint(0, w // 2)
            y1 = np.random.randint(0, h // 2)
            x2 = np.random.randint(x1 + 10, min(x1 + w // 2, w))
            y2 = np.random.randint(y1 + 10, min(y1 + h // 2, h))
            boxes.append([x1, y1, x2, y2])
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.randint(0, 3, (num_boxes,)),
            'image_id': idx,
        }
        
        return img, target


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def mock_dataset():
    """Create mock dataset."""
    return MockDataset(num_samples=10, img_size=(100, 100))


@pytest.fixture
def sample_image():
    """Create a sample image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = [255, 0, 0]
    img[:50, 50:] = [0, 255, 0]
    img[50:, :50] = [0, 0, 255]
    img[50:, 50:] = [255, 255, 0]
    return img


@pytest.fixture
def sample_target():
    """Create sample target."""
    return {
        'boxes': torch.tensor([[10, 10, 40, 40], [60, 60, 90, 90]], dtype=torch.float32),
        'labels': torch.tensor([0, 1]),
        'image_id': 0,
    }


# =============================================================================
# Test Mosaic
# =============================================================================

class TestMosaic:
    """Tests for Mosaic augmentation."""
    
    def test_output_size(self, mock_dataset, sample_image, sample_target):
        """Test that output has correct size."""
        mosaic = Mosaic(dataset=mock_dataset, imgsz=640, p=1.0)
        
        output_img, _ = mosaic(sample_image, sample_target, index=0)
        
        assert output_img.shape == (640, 640, 3)
    
    def test_output_is_numpy(self, mock_dataset, sample_image, sample_target):
        """Test that output is numpy array."""
        mosaic = Mosaic(dataset=mock_dataset, imgsz=640, p=1.0)
        
        output_img, _ = mosaic(sample_image, sample_target, index=0)
        
        assert isinstance(output_img, np.ndarray)
    
    def test_boxes_in_bounds(self, mock_dataset, sample_image, sample_target):
        """Test that output boxes are within image bounds."""
        mosaic = Mosaic(dataset=mock_dataset, imgsz=640, p=1.0)
        
        _, output_target = mosaic(sample_image, sample_target, index=0)
        
        if len(output_target['boxes']) > 0:
            boxes = output_target['boxes']
            assert (boxes[:, 0] >= 0).all()
            assert (boxes[:, 1] >= 0).all()
            assert (boxes[:, 2] <= 640).all()
            assert (boxes[:, 3] <= 640).all()
    
    def test_boxes_valid(self, mock_dataset, sample_image, sample_target):
        """Test that boxes have positive area."""
        mosaic = Mosaic(dataset=mock_dataset, imgsz=640, p=1.0)
        
        _, output_target = mosaic(sample_image, sample_target, index=0)
        
        if len(output_target['boxes']) > 0:
            boxes = output_target['boxes']
            assert (boxes[:, 2] > boxes[:, 0]).all()
            assert (boxes[:, 3] > boxes[:, 1]).all()
    
    def test_probability(self, mock_dataset, sample_image, sample_target):
        """Test that p=0 doesn't apply mosaic."""
        mosaic = Mosaic(dataset=mock_dataset, imgsz=640, p=0.0)
        
        output_img, _ = mosaic(sample_image, sample_target, index=0)
        
        # Should return original image unchanged
        assert output_img.shape == sample_image.shape
    
    def test_reproducibility(self, mock_dataset, sample_image, sample_target):
        """Test that same seed produces same result."""
        mosaic = Mosaic(dataset=mock_dataset, imgsz=640, p=1.0)
        
        setup_seed(42)
        output1, target1 = mosaic(sample_image.copy(), sample_target.copy(), index=0)
        
        setup_seed(42)
        output2, target2 = mosaic(sample_image.copy(), sample_target.copy(), index=0)
        
        assert np.array_equal(output1, output2)
    
    def test_without_dataset(self, sample_image, sample_target):
        """Test mosaic without dataset (uses same image 4 times)."""
        mosaic = Mosaic(dataset=None, imgsz=640, p=1.0)
        
        output_img, output_target = mosaic(sample_image, sample_target, index=0)
        
        assert output_img.shape == (640, 640, 3)
    
    def test_pil_input(self, mock_dataset, sample_target):
        """Test that PIL input works."""
        pil_img = Image.new('RGB', (100, 100), color='red')
        mosaic = Mosaic(dataset=mock_dataset, imgsz=640, p=1.0)
        
        output_img, _ = mosaic(pil_img, sample_target, index=0)
        
        assert isinstance(output_img, np.ndarray)
        assert output_img.shape == (640, 640, 3)
    
    def test_parameters_logged(self, mock_dataset, sample_image, sample_target):
        """Test that parameters are logged."""
        mosaic = Mosaic(dataset=mock_dataset, imgsz=640, p=1.0)
        
        mosaic(sample_image, sample_target, index=0)
        params = mosaic.get_parameters()
        
        assert 'imgsz' in params
        assert 'indices' in params
        assert 'center' in params
        assert len(params['indices']) == 4
    
    def test_labels_preserved(self, mock_dataset, sample_image, sample_target):
        """Test that labels array is created."""
        mosaic = Mosaic(dataset=mock_dataset, imgsz=640, p=1.0)
        
        _, output_target = mosaic(sample_image, sample_target, index=0)
        
        assert 'labels' in output_target
        assert len(output_target['labels']) == len(output_target['boxes'])


# =============================================================================
# Test Mosaic9
# =============================================================================

class TestMosaic9:
    """Tests for Mosaic9 (3x3) augmentation."""
    
    def test_output_size(self, mock_dataset, sample_image, sample_target):
        """Test that output has correct size."""
        mosaic = Mosaic9(dataset=mock_dataset, imgsz=640, p=1.0)
        
        output_img, _ = mosaic(sample_image, sample_target, index=0)
        
        assert output_img.shape == (640, 640, 3)
    
    def test_uses_9_images(self, mock_dataset, sample_image, sample_target):
        """Test that 9 indices are used."""
        mosaic = Mosaic9(dataset=mock_dataset, imgsz=640, p=1.0)
        
        mosaic(sample_image, sample_target, index=0)
        params = mosaic.get_parameters()
        
        assert len(params['indices']) == 9
    
    def test_boxes_valid(self, mock_dataset, sample_image, sample_target):
        """Test that output boxes are valid."""
        mosaic = Mosaic9(dataset=mock_dataset, imgsz=640, p=1.0)
        
        _, output_target = mosaic(sample_image, sample_target, index=0)
        
        if len(output_target['boxes']) > 0:
            boxes = output_target['boxes']
            assert (boxes[:, 2] > boxes[:, 0]).all()
            assert (boxes[:, 3] > boxes[:, 1]).all()


# =============================================================================
# Visual test with real images
# =============================================================================

class TestMosaicVisual:
    """Visual tests that generate images for inspection."""
    
    def test_save_mosaic_example(self, mock_dataset, tmp_path):
        """Generate and save a mosaic example."""
        setup_seed(42)
        
        mosaic = Mosaic(dataset=mock_dataset, imgsz=200, p=1.0)
        
        # Create a distinctive test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [255, 128, 0]  # Orange
        
        target = {
            'boxes': torch.tensor([[20, 20, 80, 80]], dtype=torch.float32),
            'labels': torch.tensor([0]),
        }
        
        output_img, output_target = mosaic(img, target, index=0)
        
        # Save the result
        output_path = tmp_path / "mosaic_test.png"
        Image.fromarray(output_img).save(output_path)
        
        assert output_path.exists()
        print(f"Mosaic test saved to: {output_path}")
    
    def test_mosaic_with_test_dataset(self, tmp_path):
        """Test mosaic with actual test dataset if available."""
        test_dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'test_dataset', 'train'
        )
        
        if not os.path.exists(test_dataset_path):
            pytest.skip("Test dataset not found")
        
        # Try to load annotations
        ann_file = os.path.join(test_dataset_path, '_annotations.coco.json')
        if not os.path.exists(ann_file):
            pytest.skip("Annotations not found")
        
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Load first image
        if not coco_data.get('images'):
            pytest.skip("No images in annotations")
        
        img_info = coco_data['images'][0]
        img_path = os.path.join(test_dataset_path, img_info['file_name'])
        
        if not os.path.exists(img_path):
            pytest.skip(f"Image not found: {img_path}")
        
        img = np.array(Image.open(img_path).convert('RGB'))
        
        # Get annotations for this image
        img_id = img_info['id']
        anns = [a for a in coco_data['annotations'] if a['image_id'] == img_id]
        
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels) if labels else torch.zeros(0, dtype=torch.int64),
        }
        
        # Apply mosaic (without dataset, uses same image 4 times)
        mosaic = Mosaic(dataset=None, imgsz=640, p=1.0)
        
        setup_seed(42)
        output_img, output_target = mosaic(img, target, index=0)
        
        # Save result
        output_path = tmp_path / "mosaic_real_test.png"
        Image.fromarray(output_img).save(output_path)
        
        assert output_img.shape == (640, 640, 3)
        print(f"Real mosaic test saved to: {output_path}")
        print(f"Output boxes: {len(output_target['boxes'])}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
