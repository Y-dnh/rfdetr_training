#!/usr/bin/env python
"""
Test validation script for all RF-DETR model sizes.

This script validates each trained model from the training test
to verify that the validation pipeline works correctly.

Usage:
    conda activate rfdetr_training_env
    python tests/test_all_models_validation.py

Note: Run test_all_models_training.py first to generate the trained models.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rfdetr.training.validator import RFDETRValidator


# Model configurations matching training script
MODEL_CONFIGS = {
    'nano': {
        'size': 'n',
        'resolution': 384,
    },
    # 'small': {
    #     'size': 's',
    #     'resolution': 512,
    # },
    # 'medium': {
    #     'size': 'm',
    #     'resolution': 576,
    # },
    # 'base': {
    #     'size': 'b',
    #     'resolution': 560,
    # },
    # 'large': {
    #     'size': 'l',
    #     'resolution': 704,
    # },
    # 'xlarge': {
    #     'size': 'xl',
    #     'resolution': 700,
    # },
    # '2xlarge': {
    #     'size': '2xl',
    #     'resolution': 880,
    # },
}


def find_best_weights(train_dir: Path) -> Path:
    """Find the best.pt weights file in the training directory."""
    weights_dir = train_dir / "weights"
    
    # Try best.pt first
    best_path = weights_dir / "best.pt"
    if best_path.exists():
        return best_path
    
    # Try last.pt
    last_path = weights_dir / "last.pt"
    if last_path.exists():
        return last_path
    
    # Try any .pt file
    pt_files = list(weights_dir.glob("*.pt"))
    if pt_files:
        return pt_files[0]
    
    # Check in parent directory
    pt_files = list(train_dir.glob("**/*.pt"))
    if pt_files:
        return pt_files[0]
    
    return None


def validate_model(model_name: str, config: dict, runs_dir: Path, dataset_dir: Path):
    """Validate a single trained model."""
    print("\n" + "=" * 80)
    print(f"VALIDATING: {model_name.upper()} MODEL")
    print("=" * 80)
    
    train_dir = runs_dir / f"{model_name}_train_test"
    val_output_dir = runs_dir / f"{model_name}_val_test"
    
    if not train_dir.exists():
        print(f"[WARN] Training directory not found: {train_dir}")
        print("Skipping - run training test first.")
        return False, 0
    
    # Find weights
    weights_path = find_best_weights(train_dir)
    if weights_path is None:
        print(f"[WARN] No weights found in {train_dir}")
        return False, 0
    
    print(f"Model size: {config['size']}")
    print(f"Resolution: {config['resolution']}")
    print(f"Weights: {weights_path}")
    print(f"Output dir: {val_output_dir}")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        validator = RFDETRValidator(
            model_path=str(weights_path),
            model_size=config['size'],
            conf_threshold=0.25,
            iou_threshold=0.5,
            imgsz=config['resolution'],
            batch_size=8,  # Smaller batch for validation
            workers=4,
            device='cuda',
            save_dir=str(val_output_dir),
            half=True,
        )
        
        results = validator.validate(
            dataset_dir=str(dataset_dir),
            split='valid',
            save_visualizations=True,
        )
        
        elapsed = time.time() - start_time
        
        # Print key metrics
        metrics = results.get('metrics', {})
        print(f"\nResults for {model_name.upper()}:")
        print(f"  mAP@0.5:     {metrics.get('mAP50', 0):.4f}")
        print(f"  mAP@0.5:0.95:{metrics.get('mAP50-95', 0):.4f}")
        print(f"  Precision:   {metrics.get('precision', 0):.4f}")
        print(f"  Recall:      {metrics.get('recall', 0):.4f}")
        print(f"  F1 Score:    {metrics.get('f1', 0):.4f}")
        
        print(f"\n[PASS] {model_name.upper()} validation completed in {elapsed:.1f} seconds")
        return True, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[FAIL] {model_name.upper()} validation FAILED after {elapsed:.1f} seconds")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed


def main():
    """Run validation tests for all model sizes."""
    print("=" * 80)
    print("RF-DETR MODEL VALIDATION TEST")
    print("=" * 80)
    print(f"Testing all model sizes: {list(MODEL_CONFIGS.keys())}")
    print("=" * 80)
    
    dataset_dir = project_root / "test_dataset"
    runs_dir = project_root / "runs"
    
    if not dataset_dir.exists():
        print(f"ERROR: Dataset not found at {dataset_dir}")
        sys.exit(1)
    
    if not runs_dir.exists():
        print(f"ERROR: Runs directory not found at {runs_dir}")
        print("Please run training test first.")
        sys.exit(1)
    
    results = {}
    total_start = time.time()
    
    for model_name, config in MODEL_CONFIGS.items():
        success, elapsed = validate_model(
            model_name=model_name,
            config=config,
            runs_dir=runs_dir,
            dataset_dir=dataset_dir,
        )
        results[model_name] = {'success': success, 'time': elapsed}
    
    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print("VALIDATION TEST SUMMARY")
    print("=" * 80)
    
    for model_name, result in results.items():
        status = "[PASS]" if result['success'] else "[FAIL]"
        print(f"{model_name:>10}: {status} ({result['time']:.1f} sec)")
    
    print("-" * 80)
    print(f"Total time: {total_time/60:.1f} minutes")
    
    passed = sum(1 for r in results.values() if r['success'])
    print(f"Results: {passed}/{len(results)} passed")
    
    if passed == len(results):
        print("\n=== All validation tests PASSED! ===")
    else:
        print("\n=== Some validation tests FAILED! ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
