"""
Integration tests for RF-DETR training system using tests/test_dataset.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfdetr.training import (
    setup_seed,
    AugmentationConfig,
    RFDETRDataset,
    build_dataset,
)
from rfdetr.training.augmentations import AugmentationPipeline
from rfdetr.training.visualizations import (
    visualize_batch,
    create_labels_visualization,
    BatchVisualizer,
)


# Path to test dataset
TEST_DATASET_DIR = Path(__file__).parent / 'test_dataset'


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def test_dataset_exists():
    """Check if test dataset exists."""
    train_dir = TEST_DATASET_DIR / 'train'
    ann_file = train_dir / '_annotations.coco.json'
    
    if not train_dir.exists() or not ann_file.exists():
        pytest.skip("Test dataset not found")
    
    return True


@pytest.fixture
def augmentation_config():
    """Create augmentation config for testing."""
    return AugmentationConfig(
        mosaic=0.0,  # Disable mosaic for simpler testing
        mixup=0.0,
        cutmix=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        imgsz=640,
    )


# =============================================================================
# Test Dataset Loading
# =============================================================================

class TestDatasetLoading:
    """Tests for dataset loading functionality."""
    
    def test_build_dataset(self, test_dataset_exists, augmentation_config):
        """Test building dataset from test_dataset."""
        dataset = build_dataset(
            TEST_DATASET_DIR,
            split='train',
            augmentation_config=augmentation_config,
        )
        
        assert len(dataset) > 0
        assert dataset.num_classes > 0
        print(f"Dataset loaded: {len(dataset)} images, {dataset.num_classes} classes")
        print(f"Class names: {dataset.class_names}")
    
    def test_dataset_getitem(self, test_dataset_exists, augmentation_config):
        """Test getting items from dataset."""
        dataset = build_dataset(
            TEST_DATASET_DIR,
            split='train',
            augmentation_config=augmentation_config,
        )
        
        # Get first item
        image, target = dataset[0]
        
        # Check image
        assert isinstance(image, torch.Tensor)
        assert image.shape[0] == 3  # RGB
        assert image.shape[1] == 640  # Height
        assert image.shape[2] == 640  # Width
        
        # Check target
        assert 'boxes' in target
        assert 'labels' in target
        
        print(f"Image shape: {image.shape}")
        print(f"Number of boxes: {len(target['boxes'])}")
    
    def test_dataset_raw_item(self, test_dataset_exists, augmentation_config):
        """Test getting raw (unaugmented) items."""
        dataset = build_dataset(
            TEST_DATASET_DIR,
            split='train',
            augmentation_config=augmentation_config,
        )
        
        # Get raw item
        image, target = dataset.get_raw_item(0)
        
        assert isinstance(image, Image.Image)
        assert 'boxes' in target
        
        print(f"Raw image size: {image.size}")
        print(f"Boxes: {target['boxes']}")


# =============================================================================
# Test Augmentation Pipeline
# =============================================================================

class TestAugmentationPipeline:
    """Tests for augmentation pipeline with real images."""
    
    def test_pipeline_with_dataset(self, test_dataset_exists, augmentation_config):
        """Test augmentation pipeline with real dataset images."""
        setup_seed(42)
        
        dataset = build_dataset(
            TEST_DATASET_DIR,
            split='train',
            augmentation_config=augmentation_config,
        )
        
        # Get multiple items
        for i in range(min(5, len(dataset))):
            image, target = dataset[i]
            
            # Verify output
            assert image.shape == (3, 640, 640)
            assert len(target['boxes']) == len(target['labels'])
            
            # Check boxes are valid
            if len(target['boxes']) > 0:
                # Normalized cxcywh format after pipeline
                boxes = target['boxes']
                assert (boxes >= 0).all() and (boxes <= 1).all(), f"Boxes out of range: {boxes}"
    
    def test_reproducibility(self, test_dataset_exists, augmentation_config):
        """Test that same seed produces same augmentations."""
        dataset = build_dataset(
            TEST_DATASET_DIR,
            split='train',
            augmentation_config=augmentation_config,
        )
        
        # Get item with seed 42
        setup_seed(42)
        image1, target1 = dataset[0]
        
        # Get same item with same seed
        setup_seed(42)
        image2, target2 = dataset[0]
        
        # Should be identical
        assert torch.allclose(image1, image2)


# =============================================================================
# Test Visualizations
# =============================================================================

class TestVisualizations:
    """Tests for visualization functionality."""
    
    def test_batch_visualization(self, test_dataset_exists, augmentation_config, tmp_path):
        """Test batch visualization with real images."""
        setup_seed(42)
        
        dataset = build_dataset(
            TEST_DATASET_DIR,
            split='train',
            augmentation_config=augmentation_config,
        )
        
        # Get a batch of images
        images = []
        targets = []
        filenames = []
        
        for i in range(min(4, len(dataset))):
            img, tgt = dataset[i]
            images.append(img)
            targets.append(tgt)
            filenames.append(dataset.get_filename(i))
        
        # Stack into batch
        batch_images = torch.stack(images)
        
        # Create visualization
        mosaic = visualize_batch(
            batch_images,
            targets,
            filenames=filenames,
            class_names=dataset.class_names,
        )
        
        # Save
        output_path = tmp_path / 'batch_test.jpg'
        Image.fromarray(mosaic).save(output_path)
        
        assert output_path.exists()
        print(f"Batch visualization saved to: {output_path}")
    
    def test_labels_visualization(self, test_dataset_exists, augmentation_config, tmp_path):
        """Test labels.jpg visualization."""
        dataset = build_dataset(
            TEST_DATASET_DIR,
            split='train',
            augmentation_config=augmentation_config,
        )
        
        # Create labels visualization
        output_path = tmp_path / 'labels.jpg'
        create_labels_visualization(
            dataset,
            output_path,
            class_names=dataset.class_names,
        )
        
        assert output_path.exists()
        print(f"Labels visualization saved to: {output_path}")


# =============================================================================
# Test Full Workflow
# =============================================================================

class TestFullWorkflow:
    """Test complete training workflow (without actual training)."""
    
    def test_dataset_and_visualization_workflow(self, test_dataset_exists, tmp_path):
        """Test complete dataset loading and visualization workflow."""
        setup_seed(42)
        
        # 1. Create config
        config = AugmentationConfig(
            mosaic=0.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            fliplr=0.5,
            degrees=5.0,
            imgsz=640,
        )
        
        # 2. Load dataset
        dataset = build_dataset(
            TEST_DATASET_DIR,
            split='train',
            augmentation_config=config,
        )
        
        print(f"Loaded {len(dataset)} images with {dataset.num_classes} classes")
        
        # 3. Create batch visualizer
        visualizer = BatchVisualizer(
            tmp_path,
            class_names=dataset.class_names,
        )
        
        # 4. Get some batches and visualize
        from torch.utils.data import DataLoader
        from rfdetr.training.dataset import collate_fn
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn,
        )
        
        for batch_idx, (images, targets) in enumerate(loader):
            if batch_idx >= 2:
                break
            
            filenames = [dataset.get_filename(batch_idx * 4 + i) 
                        for i in range(len(images))]
            
            visualizer.save_train_batch(
                images, targets, batch_idx, filenames
            )
        
        # 5. Create labels visualization
        labels_path = tmp_path / 'labels.jpg'
        create_labels_visualization(
            dataset,
            labels_path,
            class_names=dataset.class_names,
        )
        
        # Verify outputs
        assert (tmp_path / 'train_batch0.jpg').exists()
        assert labels_path.exists()
        
        print(f"All visualizations saved to: {tmp_path}")
        print("Workflow test passed!")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
