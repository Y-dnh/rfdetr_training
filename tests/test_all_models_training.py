#!/usr/bin/env python
"""
Test training script for all RF-DETR model sizes.

This script trains each model size (nano, small, medium, base, large, xlarge, 2xlarge)
for 3 epochs on the test_dataset to verify that the training pipeline works correctly.

Usage:
    conda activate rfdetr_training_env
    python tests/test_all_models_training.py
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rfdetr.training.trainer import RFDETRTrainer
from rfdetr.training.utils.config import ModelConfig, TrainingConfig, AugmentationConfig


# Model configurations with adjusted batch sizes for RTX 5060 8GB
MODEL_CONFIGS = {
    'nano': {
        'size': 'n',
        'batch_size': 16,
        'resolution': 384,
    },
    # 'small': {
    #     'size': 's',
    #     'batch_size': 16,
    #     'resolution': 512,
    # },
    # 'medium': {
    #     'size': 'm',
    #     'batch_size': 8,
    #     'resolution': 576,
    # },
    # 'base': {
    #     'size': 'b',
    #     'batch_size': 8,
    #     'resolution': 560,
    # },
    # 'large': {
    #     'size': 'l',
    #     'batch_size': 4,
    #     'resolution': 704,
    # },
    # 'xlarge': {
    #     'size': 'xl',
    #     'batch_size': 2,
    #     'resolution': 700,
    #     'accept_platform_license': True,
    # },
    # '2xlarge': {
    #     'size': '2xl',
    #     'batch_size': 2,
    #     'resolution': 880,
    #     'accept_platform_license': True,
    # },
}



def train_model(model_name: str, config: dict, dataset_dir: str, output_base: str):
    """Train a single model configuration."""
    print("\n" + "=" * 80)
    print(f"TRAINING: {model_name.upper()} MODEL")
    print("=" * 80)
    
    output_dir = Path(output_base) / f"{model_name}_train_test"
    
    # Model config
    model_config = ModelConfig(
        model_size=config['size'],
        num_classes=4,  # DPSU dataset has 4 classes
        accept_platform_license=config.get('accept_platform_license', False),
    )
    
    # Training config - 2 epochs
    training_config = TrainingConfig(
        epochs=1,
        batch_size=config['batch_size'],
        workers=2, 
        lr=1e-4,
        weight_decay=1e-4,
        save_period=1,
        val_period=1,
        gradient_accumulation=2 if config['batch_size'] <= 2 else 1,
        project=str(output_dir),
        name='',
        exist_ok=True,
    )
    
    aug_config = AugmentationConfig(

    hsv_h=0.0,                       
    hsv_s=0.0,                       
    hsv_v=0.4,                       
    
    brightness=0.2,                  
    contrast=0.2,                    
    blur=0.1,                        
    noise=0.0,                       
    
    degrees=10.0,                    
    translate=0.1,                   
    scale=0.0,                       
    shear=0.0,                       
    perspective=0.0,                    
    fliplr=0.5,                      
    flipud=0.0,                       
    mosaic=1.0,                      
    mixup=0.0,                       
    cutmix=0.0,                          
    cutmix_min_visible=0.3,              
    cutmix_min_box_size=10,          
    close_mosaic=19,                 
    
    erasing=0.25,                     
    erasing_min_visible=0.5,          
    erasing_min_box_size=20,   
)      
    
    print(f"Model size: {config['size']}")
    print(f"Resolution: {config['resolution']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Output dir: {output_dir}")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        trainer = RFDETRTrainer(
            model_config=model_config,
            training_config=training_config,
            augmentation_config=aug_config,
        )
        
        trainer.train(dataset_dir=dataset_dir)
        
        elapsed = time.time() - start_time
        print(f"\n[PASS] {model_name.upper()} training completed in {elapsed/60:.1f} minutes")
        return True, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[FAIL] {model_name.upper()} training FAILED after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed


def main():
    """Run training tests for all model sizes."""
    print("=" * 80)
    print("RF-DETR MODEL TRAINING TEST")
    print("=" * 80)
    print(f"Testing all model sizes: {list(MODEL_CONFIGS.keys())}")
    print(f"Epochs: 2, Close mosaic: epoch 1")
    print("=" * 80)
    
    dataset_dir = project_root / "tests" / "test_dataset"
    output_base = project_root / "tests" / "runs"
    
    if not dataset_dir.exists():
        print(f"ERROR: Dataset not found at {dataset_dir}")
        print("Please run the dataset creation script first.")
        sys.exit(1)
    
    results = {}
    total_start = time.time()
    
    for model_name, config in MODEL_CONFIGS.items():
        success, elapsed = train_model(
            model_name=model_name,
            config=config,
            dataset_dir=str(dataset_dir),
            output_base=str(output_base),
        )
        results[model_name] = {'success': success, 'time': elapsed}
    
    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print("TRAINING TEST SUMMARY")
    print("=" * 80)
    
    for model_name, result in results.items():
        status = "[PASS]" if result['success'] else "[FAIL]"
        print(f"{model_name:>10}: {status} ({result['time']/60:.1f} min)")
    
    print("-" * 80)
    print(f"Total time: {total_time/60:.1f} minutes")
    
    passed = sum(1 for r in results.values() if r['success'])
    print(f"Results: {passed}/{len(results)} passed")
    
    if passed == len(results):
        print("\n=== All training tests PASSED! ===")
    else:
        print("\n=== Some training tests FAILED! ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
