"""Test new trainer implementation."""
import sys
sys.path.insert(0, r'd:\projects_yaroslav\rfdetr_training')

print("Testing imports...")

try:
    from rfdetr.training.trainer import RFDETRTrainer
    print("RFDETRTrainer import OK")
except Exception as e:
    print(f"RFDETRTrainer import FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from rfdetr.training import (
        AugmentationConfig,
        TrainingConfig, 
        ModelConfig,
    )
    print("Config imports OK")
except Exception as e:
    print(f"Config imports FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nCreating trainer instance...")
try:
    trainer = RFDETRTrainer(
        model_config=ModelConfig(model_size='b', num_classes=3),
        training_config=TrainingConfig(epochs=2, batch_size=2, workers=0),
        augmentation_config=AugmentationConfig(mosaic=0.0),
        seed=42,
    )
    print("Trainer created OK")
except Exception as e:
    print(f"Trainer creation FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll imports and instantiation tests passed!")
