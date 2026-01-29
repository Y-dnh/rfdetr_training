<p align="center">
  <img src="https://raw.githubusercontent.com/roboflow/rf-detr/main/docs/assets/og-card.png" alt="RF-DETR" width="100%">
</p>

<h1 align="center">RF-DETR Training Pipeline</h1>

<p align="center">
  <b>🚀 Production-Ready Training Framework with Ultralytics-Style Augmentations</b>
</p>

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-augmentations">Augmentations</a> •
  <a href="#-configuration">Configuration</a> •
  <a href="#-training-outputs">Outputs</a> •
  <a href="#-api-reference">API</a>
</p>

---

> **Extended fork of [RF-DETR by Roboflow](https://github.com/roboflow/rf-detr)** — State-of-the-art real-time object detector with a complete, customizable training pipeline inspired by Ultralytics YOLO.

---

## ✨ Key Features

| Feature | Original RF-DETR | This Fork |
|---------|------------------|-----------|
| **Augmentations** | Basic (resize, flip) | 🎨 Full Ultralytics suite: Mosaic, MixUp, CutMix, HSV, Erasing |
| **Configuration** | argparse CLI | ⚙️ Type-safe dataclasses (`ModelConfig`, `TrainingConfig`, `AugmentationConfig`) |
| **Visualizations** | TensorBoard only | 📊 Batch images, confusion matrix, PR/F1/P/R curves, metrics plots |
| **Validation** | COCO eval | 📋 Extended: markdown reports, per-class stats, detailed analysis |
| **Dataset** | CocoDetection | 📁 `RFDETRDataset` with integrated augmentation pipeline |
| **Entry Point** | `Model.train()` | 🏋️ Dedicated `RFDETRTrainer` & `RFDETRValidator` classes |
| **Box-Aware Augs** | ❌ | ✅ CutMix & RandomErasing preserve object visibility |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/rfdetr_training.git
cd rfdetr_training

# Install dependencies
pip install -e .
```

### Training

```python
from rfdetr.training import (
    setup_seed,
    ModelConfig,
    TrainingConfig,
    AugmentationConfig,
    RFDETRTrainer,
)

# Reproducibility
setup_seed(42)

# Configuration
model_config = ModelConfig(
    model_size="n",      # n, s, m, b, l, xl, 2xl
    num_classes=3,       # Auto-detected from dataset
)

training_config = TrainingConfig(
    epochs=100,
    batch_size=16,
    lr=1e-4,
    project="runs/training",
)

augmentation_config = AugmentationConfig(
    mosaic=1.0,          # 4-image mosaic
    mixup=0.3,           # Alpha blending
    cutmix=0.3,          # Box-aware cut & paste
    fliplr=0.5,          # Horizontal flip
    hsv_h=0.015,         # Hue variation
    hsv_s=0.7,           # Saturation variation
    hsv_v=0.4,           # Value variation
)

# Train
trainer = RFDETRTrainer(model_config, training_config, augmentation_config)
results = trainer.train(dataset_dir="path/to/dataset")

print(f"Best mAP: {results['best_map']:.4f}")
```

### Validation

```python
from rfdetr.training import RFDETRValidator

validator = RFDETRValidator(
    model_path="runs/training/exp/weights/best.pt",
    model_size="n",
    conf_threshold=0.5,
    iou_threshold=0.5,
)

results = validator.validate(
    dataset_dir="path/to/dataset",
    split="test",
    save_visualizations=True,
)
```

### Dataset Format

Supports both **COCO** (Roboflow export) and **YOLO** formats with auto-detection:

#### COCO Format
```
dataset/
├── train/
│   ├── image1.jpg
│   └── _annotations.coco.json
├── valid/
│   └── _annotations.coco.json
└── test/
    └── _annotations.coco.json
```

---

## 🏗 Architecture

```
rfdetr_training/
├── train.py                      # 🚀 Training entry point
├── val.py                        # 🧪 Validation entry point
└── rfdetr/
    ├── training/                 # ⭐ CUSTOM MODULE
    │   ├── trainer.py            # RFDETRTrainer (1258 lines)
    │   ├── validator.py          # RFDETRValidator (1053 lines)
    │   ├── dataset.py            # RFDETRDataset with augmentations
    │   │
    │   ├── augmentations/        # 🎨 Ultralytics-style augmentations
    │   │   ├── pipeline.py       # AugmentationPipeline orchestrator
    │   │   ├── mosaic.py         # Mosaic 4-grid & Mosaic9
    │   │   ├── mixup.py          # MixUp & CutMix (box-aware)
    │   │   ├── color.py          # HSV, Brightness, Contrast, Blur, Noise
    │   │   ├── geometric.py      # Flip, Perspective, LetterBox
    │   │   ├── erasing.py        # RandomErasing (box-aware)
    │   │   └── base.py           # BaseTransform, ToTensor, Normalize
    │   │
    │   ├── visualizations/       # 📊 YOLO-style outputs
    │   │   ├── batch_visualizer.py    # Train/Val batch images
    │   │   ├── confusion_matrix.py    # Confusion matrix
    │   │   ├── curves.py              # PR, F1, P, R curves
    │   │   ├── metrics_plotter.py     # Training metrics
    │   │   └── labels_analyzer.py     # Label distribution
    │   │
    │   ├── logging/              # 📝 Logging utilities
    │   │   └── augmentation_logger.py
    │   │
    │   └── utils/                # ⚙️ Configuration & utilities
    │       ├── config.py         # Dataclass configurations
    │       └── seed.py           # Reproducibility
    │
    ├── platform/                 # 🔒 Platform-licensed models (PML-1.0)
    │   └── models.py             # RFDETRXLarge, RFDETR2XLarge
    │
    ├── config.py                 # RF-DETR model configs (all sizes)
    ├── main.py                   # Original Model class
    ├── detr.py                   # DETR implementation
    ├── models/                   # RF-DETR architecture
    └── datasets/                 # COCO & YOLO dataset loaders
        ├── coco.py               # COCO format
        └── yolo.py               # YOLO format (NEW)
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `RFDETRTrainer` | Complete training loop with gradient accumulation, warmup, early stopping, checkpointing |
| `RFDETRValidator` | Full validation pipeline with COCO metrics, visualizations, markdown reports |
| `RFDETRDataset` | PyTorch Dataset with integrated augmentation pipeline and epoch-aware controls |
| `AugmentationPipeline` | Orchestrates all augmentations in correct order with logging |
| `BatchVisualizer` | Generates train/val batch images with ground truth and predictions |
| `ConfusionMatrix` | IoU-based confusion matrix calculation and visualization |

---

## 🎨 Augmentations

### Pipeline Order

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Mosaic        │ Combine 4 random images in 2×2 grid        │
├───────────────────┼─────────────────────────────────────────────┤
│  2. Perspective   │ Rotation, translation, scale, shear        │
├───────────────────┼─────────────────────────────────────────────┤
│  3. MixUp/CutMix  │ Blend or cut-paste between images          │
├───────────────────┼─────────────────────────────────────────────┤
│  4. HSV           │ Hue, Saturation, Value adjustments          │
├───────────────────┼─────────────────────────────────────────────┤
│  5. Color Augs    │ Brightness, Contrast, Blur, Noise          │
├───────────────────┼─────────────────────────────────────────────┤
│  6. Flip          │ Horizontal / Vertical flip                  │
├───────────────────┼─────────────────────────────────────────────┤
│  7. Erasing       │ Random rectangular region removal           │
├───────────────────┼─────────────────────────────────────────────┤
│  8. LetterBox     │ Resize with padding to target size          │
├───────────────────┼─────────────────────────────────────────────┤
│  9. Normalize     │ ImageNet mean/std normalization             │
└───────────────────┴─────────────────────────────────────────────┘
```

### Augmentation Details

#### 🧩 Mosaic
Combines 4 random images into a 2×2 grid, creating diverse training samples.

```
+--------+--------+
| Img 1  | Img 2  |
+--------+--------+
| Img 3  | Img 4  |
+--------+--------+
```

**Parameters:**
- `mosaic` — Probability (0-1)
- `close_mosaic` — Disable N epochs before training ends

---

#### 🔀 MixUp
Alpha-blends two images: `output = α × img1 + (1-α) × img2`

Where α ~ Beta(32, 32) for balanced mixing.

---

#### ✂️ CutMix (Box-Aware)
Cuts a rectangular region from one image and pastes it into another.

**Box-aware logic:**
- Preserves bounding boxes if ≥ `min_visible` portion remains
- Filters boxes smaller than `min_box_size` after cutting

---

#### 🎭 RandomErasing (Box-Aware)
Randomly erases rectangular regions while protecting object annotations.

**Box-aware logic:**
- Validates overlap with ground truth boxes
- Ensures `min_visible` ratio of each box remains visible
- Removes boxes that become smaller than `min_box_size`

---

#### 🌈 RandomHSV
Modifies color channels:
```python
H' = H × (1 ± hsv_h)
S' = S × (1 ± hsv_s)
V' = V × (1 ± hsv_v)
```

---

## ⚙️ Configuration

### Model Variants

| Size | Resolution | Decoder Layers | Parameters | License |
|------|------------|----------------|------------|---------|
| `n` (nano) | 384×384 | 2 | ~30.5M | Apache-2.0 |
| `s` (small) | 512×512 | 3 | ~32.1M | Apache-2.0 |
| `m` (medium) | 576×576 | 4 | ~33.7M | Apache-2.0 |
| `b` (base) | 560×560 | 3 | ~29M | Apache-2.0 |
| `l` (large) | 704×704 | 4 | ~33.9M | Apache-2.0 |
| `xl` (xlarge) | 700×700 | 5 | ~126.4M | ⚠️ PML-1.0 |
| `2xl` (2xlarge) | 880×880 | 5 | ~126.9M | ⚠️ PML-1.0 |

> ⚠️ **Note**: XLarge and 2XLarge models require `accept_platform_model_license=True` and an active Roboflow platform plan.

### ModelConfig

```python
ModelConfig(
    model_size="n",              # Model variant: n, s, m, b, l, xl, 2xl
    num_classes=3,               # Auto-detected from dataset
    pretrained_weights=None,     # Path or None for HuggingFace
    freeze_encoder=False,        # Freeze DINOv2 backbone
    freeze_encoder_epochs=0,     # Epochs with frozen encoder
)
```

### TrainingConfig

```python
TrainingConfig(
    # Project
    project="runs/training",     # Output directory
    name="exp",                  # Run name (auto-increments)
    
    # Training
    epochs=100,
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-4,
    
    # Scheduler
    scheduler="cosine",          # cosine, step, linear
    warmup_epochs=5,
    
    # Gradients
    gradient_accumulation=1,
    grad_clip=0.1,
    
    # Validation
    val_period=1,
    save_period=-1,              # -1 = best/last only
    early_stopping=50,
    
    # Hardware
    device="cuda",
    workers=8,
    
    # Visualizations
    vis_batches=3,               # Batches to visualize
)
```

### AugmentationConfig

```python
AugmentationConfig(
    # Mosaic
    mosaic=1.0,
    mosaic_scale=(0.5, 1.5),
    close_mosaic=10,
    
    # MixUp / CutMix
    mixup=0.0,
    cutmix=0.0,
    cutmix_min_visible=0.3,
    cutmix_min_box_size=10,
    
    # HSV
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    
    # Color
    brightness=0.0,
    contrast=0.0,
    blur=0.0,
    noise=0.0,
    
    # Geometric
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    fliplr=0.5,
    flipud=0.0,
    
    # Erasing (box-aware)
    erasing=0.0,
    erasing_min_visible=0.5,
    erasing_min_box_size=20,
)
```

---

## 📤 Training Outputs

```
runs/training/exp/
├── weights/
│   ├── best.pt                  # Best model (by mAP)
│   └── last.pt                  # Latest checkpoint
│
├── train_batch0.jpg             # Training batches with augmentations
├── train_batch1.jpg
├── train_batch_last0.jpg        # Final batches (no mosaic)
│
├── val_batch0_labels.jpg        # Validation: ground truth
├── val_batch0_pred.jpg          # Validation: predictions
│
├── confusion_matrix.png         # Per-class confusion matrix
├── PR_curve.png                 # Precision-Recall curve
├── F1_curve.png                 # F1 vs Confidence
├── P_curve.png                  # Precision vs Confidence
├── R_curve.png                  # Recall vs Confidence
├── results.png                  # Training metrics over epochs
│
├── config_model.json            # Saved configurations
├── config_training.json
├── config_augmentation.json
├── augmentations_log.json       # Per-sample augmentation log
└── training.log                 # Full training log
```

### Validation Report

The validator generates a comprehensive markdown report:

```markdown
# 🎯 RF-DETR Validation Report

## Overall Performance
| Metric        | Value  |
|---------------|--------|
| mAP@0.5       | 0.6461 |
| mAP@0.5:0.95  | 0.3765 |
| Precision     | 0.8715 |
| Recall        | 0.6508 |
| F1 Score      | 0.7451 |

## Per-Class Performance
| Class   | GT   | TP   | FP  | FN   | Precision | Recall |
|---------|------|------|-----|------|-----------|--------|
| class_0 | 2978 | 2007 | 145 | 971  | 0.933     | 0.674  |
| class_1 | 2610 | 1563 | 255 | 1047 | 0.860     | 0.599  |
| class_2 | 746  | 552  | 208 | 194  | 0.726     | 0.740  |
```

---

## 📖 API Reference

### RFDETRTrainer

```python
class RFDETRTrainer:
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        augmentation_config: AugmentationConfig,
        seed: int = 42,
    ) -> None: ...

    def train(
        self,
        dataset_dir: str,
        resume: Optional[str] = None,  # Checkpoint path for resume
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                'best_map': float,
                'epochs_trained': int,
                'save_dir': str,
            }
        """
```

### RFDETRValidator

```python
class RFDETRValidator:
    def __init__(
        self,
        model_path: str,
        model_size: str = "b",
        imgsz: Optional[int] = None,  # Auto-detect from checkpoint
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        batch_size: int = 16,
        workers: int = 8,
        device: str = "cuda",
        save_dir: Optional[str] = None,
        half: bool = True,           # FP16 inference
    ) -> None: ...

    def validate(
        self,
        dataset_dir: str,
        split: str = "valid",
        save_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                'mAP50': float,
                'mAP50-95': float,
                'precision': float,
                'recall': float,
                'f1': float,
                'save_dir': str,
            }
        """
```

### RFDETRDataset

```python
class RFDETRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        augmentation_config: AugmentationConfig,
        is_train: bool = True,
        log_augmentations: bool = False,
    ) -> None: ...

    def set_epoch(self, epoch: int, total_epochs: int): ...
    
    @property
    def num_classes(self) -> int: ...
    
    @property
    def class_names(self) -> List[str]: ...
    
    @property
    def coco(self) -> COCO: ...
```

---

## 🔬 Comparison with Ultralytics YOLO

| Feature | Ultralytics YOLO | This Project |
|---------|------------------|--------------|
| Mosaic | ✅ | ✅ |
| MixUp | ✅ | ✅ |
| CutMix | ✅ | ✅ (box-aware) |
| RandomHSV | ✅ | ✅ |
| RandomPerspective | ✅ | ✅ |
| RandomErasing | ✅ | ✅ (box-aware) |
| close_mosaic | ✅ | ✅ |
| Confusion Matrix | ✅ | ✅ |
| PR/F1 Curves | ✅ | ✅ |
| Batch Visualizations | ✅ | ✅ |
| Markdown Reports | ❌ | ✅ |
| Augmentation Logging | ✅ | ✅ |

---

## 🛠️ Command-Line Usage

### Basic Training

```bash
python train.py
```

Edit configuration at the top of `train.py`:

```python
DATASET_DIR = Path("path/to/your/dataset")
MODEL_CONFIG = ModelConfig(model_size="m", num_classes=3)
TRAINING_CONFIG = TrainingConfig(epochs=100, batch_size=16)
AUGMENTATION_CONFIG = AugmentationConfig(mosaic=1.0, mixup=0.3)
```

### Resume Training

```python
# In train.py, uncomment:
# resume_training("runs/training/exp/weights/last.pt")
```

### Validation

```bash
python val.py
```

Configure in `val.py`:

```python
MODEL_PATH = "path/to/model.pt"
MODEL_SIZE = "m"  # Must match trained model
DATASET_DIR = Path("path/to/dataset")
```

---

## 📜 License

This project is a fork of [RF-DETR](https://github.com/roboflow/rf-detr) with dual licensing:

- **Apache 2.0 License** — Core models (Nano, Small, Medium, Base, Large) and training pipeline
- **Platform Model License 1.0** — XLarge and 2XLarge models require an active Roboflow platform plan

See [LICENSE](LICENSE), [LICENSE.core](LICENSE.core), and [LICENSE.platform](LICENSE.platform) for details.

---

<p align="center">
  <sub>Built with ❤️ extending the amazing work by <a href="https://roboflow.com">Roboflow</a></sub>
</p>
