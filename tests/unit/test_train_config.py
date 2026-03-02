"""Test train.py configuration."""
import sys
sys.path.insert(0, r'd:\projects_yaroslav\rfdetr_training')

# Test imports from train.py
from rfdetr.training import (
    setup_seed,
    AugmentationConfig,
    TrainingConfig,
    ModelConfig,
    RFDETRTrainer,
)

# Test configuration creation
MODEL_CONFIG = ModelConfig(
    model_size="b",
    num_classes=3,
    pretrained_weights=None,
    freeze_encoder=False,
)
print(f"ModelConfig OK: {MODEL_CONFIG}")

TRAINING_CONFIG = TrainingConfig(
    project="runs/test",
    name="exp",
    epochs=100,
    batch_size=8,
)
print(f"TrainingConfig OK: {TRAINING_CONFIG}")

AUGMENTATION_CONFIG = AugmentationConfig(
    imgsz=640,
    mosaic=1.0,
)
print(f"AugmentationConfig OK: {AUGMENTATION_CONFIG}")

# Test trainer creation (without actually building model)
trainer = RFDETRTrainer(
    model_config=MODEL_CONFIG,
    training_config=TRAINING_CONFIG,
    augmentation_config=AUGMENTATION_CONFIG,
    seed=42,
)
print(f"RFDETRTrainer created OK")

print("\nAll train.py configs work correctly!")
