"""Quick import test."""
import sys
sys.path.insert(0, r'd:\projects_yaroslav\rfdetr_training')

from rfdetr.training.augmentations import (
    AugmentationPipeline,
    ValidationPipeline,
    RandomHSV,
    RandomFlip,
    Mosaic,
    MixUp,
    CutMix,
    RandomErasing,
)
print("Augmentations imported OK")

from rfdetr.training.dataset import RFDETRDataset, build_dataset
print("Dataset imported OK")

from rfdetr.training.visualizations import (
    visualize_batch,
    BatchVisualizer,
    MetricsLogger,
    ConfusionMatrix,
    plot_labels,
)
print("Visualizations imported OK")

# Test train.py and val.py imports
from rfdetr.training import (
    setup_seed,
    AugmentationConfig,
    TrainingConfig,
    ModelConfig,
    RFDETRTrainer,
    RFDETRValidator,
)
print("train/val components imported OK")

print("All imports successful!")
