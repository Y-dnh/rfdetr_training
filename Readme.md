<p align="center">
  <img src="https://raw.githubusercontent.com/roboflow/rf-detr/main/docs/assets/og-card.png" alt="RF-DETR" width="100%">
</p>

<h1 align="center">RF-DETR Training Pipeline</h1>

<p align="center">
  <b>Production-Ready Training Framework with Ultralytics-Style Augmentations</b>
</p>

<p align="center">
  <a href="#key-features">Features</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#model-variants">Models</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#augmentations">Augmentations</a> &bull;
  <a href="#configuration">Configuration</a> &bull;
  <a href="#onnx-export">ONNX Export</a> &bull;
  <a href="#training-outputs">Outputs</a> &bull;
  <a href="#api-reference">API</a>
</p>

---

> **Extended fork of [RF-DETR by Roboflow](https://github.com/roboflow/rf-detr)** — State-of-the-art real-time object detector with a complete, customizable training pipeline inspired by Ultralytics YOLO.

---

## Key Features

| Feature | Original RF-DETR | This Fork |
|---------|------------------|-----------|
| **Augmentations** | Basic (resize, flip) | Full Ultralytics suite: Mosaic, MixUp, CutMix, HSV, Erasing |
| **Configuration** | argparse CLI | Type-safe dataclasses (`ModelConfig`, `TrainingConfig`, `AugmentationConfig`) |
| **Visualizations** | TensorBoard only | Batch images, confusion matrix, PR/F1/P/R curves, metrics plots |
| **Validation** | COCO eval | Extended: markdown reports, per-class stats, detailed analysis |
| **Dataset** | CocoDetection | `RFDETRDataset` with integrated augmentation pipeline |
| **Entry Point** | `Model.train()` | Dedicated `RFDETRTrainer` & `RFDETRValidator` classes |
| **ONNX Export** | CLI-only | Auto-detect model from checkpoint, simplified config-based script |
| **Box-Aware Augs** | No | CutMix & RandomErasing preserve object visibility |
| **PyTorch 2.6+ Fix** | No | `dynamo=False` for legacy ONNX exporter compatibility |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd rfdetr_training

# Install dependencies
pip install -e .
```

### Training

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

### Validation

```bash
python val.py
```

Configure in `val.py`:

```python
MODEL_PATH = "runs/training/exp/weights/best.pt"
MODEL_SIZE = "m"
DATASET_DIR = Path("path/to/dataset")
```

### ONNX Export

```bash
python export_onnx.py
```

Configure in `export_onnx.py`:

```python
CHECKPOINT_PATH = r"path/to/model.pt"
OUTPUT_DIR = None          # None = save next to checkpoint
SIMPLIFY = True            # onnxsim simplification
OPSET_VERSION = 17
```

Model architecture, resolution, and number of classes are **auto-detected** from the checkpoint.

### Dataset Format

Supports both **COCO** (Roboflow export) and **YOLO** formats with auto-detection:

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

## Model Variants

### Detection

| Size | Code | Resolution | Decoder Layers | Params | COCO AP50:95 | Latency (ms) | License |
|------|------|------------|----------------|--------|--------------|--------------|---------|
| Nano | `n` | 384x384 | 2 | 30.5M | 48.4 | 2.3 | Apache-2.0 |
| Small | `s` | 512x512 | 3 | 32.1M | 53.0 | 3.5 | Apache-2.0 |
| Medium | `m` | 576x576 | 4 | 33.7M | 54.7 | 4.4 | Apache-2.0 |
| Base | `b` | 560x560 | 3 | 29M | — | — | Apache-2.0 |
| Large | `l` | 704x704 | 4 | 33.9M | 56.5 | 6.8 | Apache-2.0 |
| XLarge | `xl` | 700x700 | 5 | 126.4M | 58.6 | 11.5 | PML-1.0 |
| 2XLarge | `2xl` | 880x880 | 5 | 126.9M | 60.1 | 17.2 | PML-1.0 |

> **Note**: Base is the original RF-DETR model (patch_size=14, DINOv2-Small). Nano/Small/Medium/Large are newer versions (patch_size=16). XLarge and 2XLarge require `accept_platform_model_license=True`.

> **Important**: Image resolution is **strictly tied** to the model architecture. Unlike Ultralytics YOLO/RT-DETR where `imgsz` is a free parameter, RF-DETR uses a fixed resolution per model size.

### Segmentation

| Size | Resolution | Params | COCO AP50:95 | Latency (ms) |
|------|------------|--------|--------------|--------------|
| Nano | 312x312 | 33.6M | 40.3 | 3.4 |
| Small | 384x384 | 33.7M | 43.1 | 4.4 |
| Medium | 432x432 | 35.7M | 45.3 | 5.9 |
| Large | 504x504 | 36.2M | 47.1 | 8.8 |
| XLarge | 624x624 | 38.1M | 48.8 | 13.5 |
| 2XLarge | 768x768 | 38.6M | 49.9 | 21.8 |

All latency measured on NVIDIA T4, TensorRT FP16, batch size 1.

---

## Architecture

```
rfdetr_training/
├── train.py                      # Training entry point
├── val.py                        # Validation entry point
├── export_onnx.py                # ONNX export (auto-detect model)
│
└── rfdetr/
    ├── training/                 # CUSTOM MODULE
    │   ├── trainer.py            # RFDETRTrainer — full training loop
    │   ├── validator.py          # RFDETRValidator — validation pipeline
    │   ├── dataset.py            # RFDETRDataset with augmentations
    │   │
    │   ├── augmentations/        # Ultralytics-style augmentations
    │   │   ├── pipeline.py       # AugmentationPipeline orchestrator
    │   │   ├── mosaic.py         # Mosaic 4-grid & Mosaic9
    │   │   ├── mixup.py          # MixUp & CutMix (box-aware)
    │   │   ├── color.py          # HSV, Brightness, Contrast, Blur, Noise
    │   │   ├── geometric.py      # Flip, Perspective, LetterBox
    │   │   ├── erasing.py        # RandomErasing (box-aware)
    │   │   └── base.py           # BaseTransform, ToTensor, Normalize
    │   │
    │   ├── visualizations/       # YOLO-style outputs
    │   │   ├── batch_visualizer.py    # Train/Val batch images
    │   │   ├── confusion_matrix.py    # Confusion matrix
    │   │   ├── curves.py              # PR, F1, P, R curves
    │   │   ├── metrics_plotter.py     # Training metrics
    │   │   └── labels_analyzer.py     # Label distribution
    │   │
    │   ├── logging/              # Logging utilities
    │   │   └── augmentation_logger.py
    │   │
    │   └── utils/                # Configuration & utilities
    │       ├── config.py         # Dataclass configurations
    │       └── seed.py           # Reproducibility
    │
    ├── deploy/                   # Deployment tools
    │   ├── export.py             # ONNX/TensorRT export core
    │   └── benchmark.py          # Benchmarking
    │
    ├── platform/                 # Platform-licensed models (PML-1.0)
    │   └── models.py             # RFDETRXLarge, RFDETR2XLarge
    │
    ├── config.py                 # RF-DETR model configs (all sizes)
    ├── main.py                   # Original Model class
    ├── detr.py                   # DETR implementation
    ├── models/                   # RF-DETR architecture
    └── datasets/                 # COCO & YOLO dataset loaders
        ├── coco.py
        └── yolo.py
```

---

## Augmentations

### Pipeline Order

```
Input Image
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Mosaic        │ Combine 4 random images in 2x2 grid        │
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
    │
    ▼
Model Input (resolution depends on model size)
```

### Details

**Mosaic** — combines 4 random images into a 2x2 grid. Parameters: `mosaic` (probability 0-1), `close_mosaic` (disable N epochs before end).

**MixUp** — alpha-blends two images: `output = a * img1 + (1-a) * img2`, where a ~ Beta(32, 32).

**CutMix (box-aware)** — cuts a rectangular region from one image and pastes into another. Preserves bounding boxes if >= `min_visible` portion remains. Filters boxes smaller than `min_box_size`.

**RandomErasing (box-aware)** — randomly erases rectangular regions while protecting object annotations. Validates overlap with GT boxes and ensures `min_visible` ratio.

**RandomHSV** — modifies color channels: `H' = H * (1 +/- hsv_h)`, etc.

---

## Configuration

All scripts use **config-at-the-top** style (no argparse). Edit variables at the top of each file before running.

### ModelConfig

```python
ModelConfig(
    model_size="m",              # n, s, m, b, l, xl, 2xl
    num_classes=3,               # Auto-detected from dataset
    pretrained_weights=None,     # Path or None for HuggingFace
    freeze_encoder=False,        # Freeze DINOv2 backbone
    freeze_encoder_epochs=0,     # Epochs with frozen encoder
    accept_platform_license=True,  # Required for xl/2xl
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
    vis_batches=3,
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

## ONNX Export

### Standalone export

```bash
python export_onnx.py
```

Configure in `export_onnx.py`:

```python
CHECKPOINT_PATH = r"path/to/best.pt"   # .pt or .pth — both work
OUTPUT_DIR = None                       # None = next to checkpoint
SIMPLIFY = True                         # onnxsim
OPSET_VERSION = 17
BATCH_SIZE = 1
```

The script **auto-detects** from the checkpoint:
- Model size (Nano/Small/Medium/Base/Large/XLarge/2XLarge)
- Number of classes
- Resolution
- Backbone type (DINOv2-Small vs DINOv2-Base)

### Auto-export after training

Training automatically exports ONNX when `ExportConfig.enabled=True` in `train.py`. The export uses the same `rfdetr/deploy/export.py` core.

### Export pipeline

```
PyTorch (.pt) --> ONNX (.onnx) --> [optional] onnxsim (.sim.onnx) --> [optional] TensorRT (.engine)
```

### Checkpoint format

Training saves checkpoints as `.pt` files:

```python
{
    'epoch': int,
    'model': state_dict,
    'optimizer': state_dict,
    'lr_scheduler': state_dict,
    'metrics': dict,
    'best_map': float,
    'class_names': list[str],
}
```

`.pt` and `.pth` are interchangeable — PyTorch does not distinguish between extensions.

### PyTorch 2.6+ compatibility

Starting with PyTorch 2.6, `torch.onnx.export` introduced a new `dynamo`-based exporter. Since PyTorch 2.9, `dynamo=True` is the **default**. This new exporter fails on RF-DETR's dynamic split/cat operations in the transformer decoder.

Fix applied in `rfdetr/deploy/export.py`: explicit `dynamo=False` forces the legacy TorchScript-based exporter.

---

## Training Outputs

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

```
Overall Performance
| Metric        | Value  |
|---------------|--------|
| mAP@0.5       | 0.6461 |
| mAP@0.5:0.95  | 0.3765 |
| Precision     | 0.8715 |
| Recall        | 0.6508 |
| F1 Score      | 0.7451 |

Per-Class Performance
| Class   | GT   | TP   | FP  | FN   | Precision | Recall |
|---------|------|------|-----|------|-----------|--------|
| class_0 | 2978 | 2007 | 145 | 971  | 0.933     | 0.674  |
| class_1 | 2610 | 1563 | 255 | 1047 | 0.860     | 0.599  |
| class_2 | 746  | 552  | 208 | 194  | 0.726     | 0.740  |
```

---

## API Reference

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
        resume: Optional[str] = None,
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
        half: bool = True,
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

## Comparison with Ultralytics YOLO

| Feature | Ultralytics YOLO | This Project |
|---------|------------------|--------------|
| Mosaic | Yes | Yes |
| MixUp | Yes | Yes |
| CutMix | Yes | Yes (box-aware) |
| RandomHSV | Yes | Yes |
| RandomPerspective | Yes | Yes |
| RandomErasing | Yes | Yes (box-aware) |
| close_mosaic | Yes | Yes |
| Confusion Matrix | Yes | Yes |
| PR/F1 Curves | Yes | Yes |
| Batch Visualizations | Yes | Yes |
| Markdown Reports | No | Yes |
| Augmentation Logging | Yes | Yes |
| ONNX Auto-Detect | No | Yes |

---

## License

This project is a fork of [RF-DETR](https://github.com/roboflow/rf-detr) with dual licensing:

- **Apache 2.0 License** — Core models (Nano, Small, Medium, Base, Large) and training pipeline
- **Platform Model License 1.0** — XLarge and 2XLarge models require an active Roboflow platform plan

See [LICENSE](LICENSE), [LICENSE.core](LICENSE.core), and [LICENSE.platform](LICENSE.platform) for details.

---

<p align="center">
  <sub>Built extending the work by <a href="https://roboflow.com">Roboflow</a></sub>
</p>
