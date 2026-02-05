"""
Configuration classes for RF-DETR training.

This module provides dataclasses for configuring:
- Augmentations (YOLO-style parameters)
- Training hyperparameters
- Model settings
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import warnings


@dataclass
class AugmentationConfig:
    """
    Configuration for data augmentations (YOLO-style parameters).
    
    All probability parameters are in range [0.0, 1.0].
    All gain/factor parameters follow Ultralytics conventions.
    
    Attributes:
        # Mosaic augmentation
        mosaic: Probability of applying mosaic augmentation.
        mosaic_scale: Scale range for mosaic (min, max).
        close_mosaic: Number of epochs before end to disable mosaic.
        
        # MixUp augmentation
        mixup: Probability of applying mixup augmentation.
        
        # CutMix augmentation
        cutmix: Probability of applying cutmix augmentation.
        
        # HSV augmentation
        hsv_h: Hue gain factor (0.0-1.0).
        hsv_s: Saturation gain factor (0.0-1.0).
        hsv_v: Value gain factor (0.0-1.0).
        
        # Geometric augmentations
        degrees: Maximum rotation degrees.
        translate: Maximum translation as fraction of image size.
        scale: Scale range (min, max) as fraction.
        shear: Maximum shear degrees.
        perspective: Perspective distortion factor.
        fliplr: Probability of horizontal flip.
        flipud: Probability of vertical flip.
        
        # Random erasing
        erasing: Probability of random erasing.
        
        # Additional color augmentations
        brightness: Probability of random brightness adjustment.
        contrast: Probability of random contrast adjustment.
        blur: Probability of random Gaussian blur.
        noise: Probability of random Gaussian noise.
        
        # Image size
        imgsz: Target image size (height, width) or single int for square.
    """
    # Mosaic
    mosaic: float = 1.0
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    close_mosaic: int = 10
    
    # MixUp
    mixup: float = 0.0
    
    # CutMix (box-aware)
    cutmix: float = 0.0
    cutmix_min_visible: float = 0.3   # Min ratio of box that must remain visible
    cutmix_min_box_size: int = 10     # Min box dimension after clipping
    
    # HSV
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    
    # Additional color augmentations
    brightness: float = 0.0
    contrast: float = 0.0
    blur: float = 0.0
    noise: float = 0.0
    
    # Geometric
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    fliplr: float = 0.5
    flipud: float = 0.0
    
    # Random erasing (box-aware)
    erasing: float = 0.0
    erasing_min_visible: float = 0.5   # Min ratio of box that must remain visible (0-1)
    erasing_min_box_size: int = 20     # Min box dimension after erasing (pixels)
    
    # Image size
    imgsz: int = 640
    
    def __post_init__(self):
        """Validate configuration values."""
        # Validate probabilities
        prob_fields = ['mosaic', 'mixup', 'cutmix', 'fliplr', 'flipud', 'erasing',
                       'brightness', 'contrast', 'blur', 'noise']
        for field_name in prob_fields:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in range [0.0, 1.0], got {value}")
        
        # Validate HSV gains
        hsv_fields = ['hsv_h', 'hsv_s', 'hsv_v']
        for field_name in hsv_fields:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in range [0.0, 1.0], got {value}")
        
        # Validate geometric params
        if self.degrees < 0:
            raise ValueError(f"degrees must be >= 0, got {self.degrees}")
        if not 0.0 <= self.translate <= 1.0:
            raise ValueError(f"translate must be in range [0.0, 1.0], got {self.translate}")
        if self.scale < 0:
            raise ValueError(f"scale must be >= 0, got {self.scale}")
        if self.shear < 0:
            raise ValueError(f"shear must be >= 0, got {self.shear}")
        if self.perspective < 0:
            raise ValueError(f"perspective must be >= 0, got {self.perspective}")
        
        # Validate image size
        if self.imgsz <= 0:
            raise ValueError(f"imgsz must be > 0, got {self.imgsz}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'mosaic': self.mosaic,
            'mosaic_scale': self.mosaic_scale,
            'close_mosaic': self.close_mosaic,
            'mixup': self.mixup,
            'cutmix': self.cutmix,
            'hsv_h': self.hsv_h,
            'hsv_s': self.hsv_s,
            'hsv_v': self.hsv_v,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'blur': self.blur,
            'noise': self.noise,
            'degrees': self.degrees,
            'translate': self.translate,
            'scale': self.scale,
            'shear': self.shear,
            'perspective': self.perspective,
            'fliplr': self.fliplr,
            'flipud': self.flipud,
            'erasing': self.erasing,
            'imgsz': self.imgsz,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AugmentationConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """
    Configuration for training hyperparameters.
    
    Attributes:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Initial learning rate.
        weight_decay: Weight decay for optimizer.
        warmup_epochs: Number of warmup epochs.
        scheduler: Learning rate scheduler type ('cosine', 'step').
        gradient_accumulation: Number of gradient accumulation steps.
        grad_clip: Maximum gradient norm for clipping (0 = disabled).
        early_stopping: Number of epochs without improvement before stopping.
        save_period: Save checkpoint every N epochs (-1 = only best/last).
        val_period: Validate every N epochs.
        resume: Path to checkpoint to resume from.
        pretrained: Path to pretrained weights or model name.
    """
    epochs: int = 100
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    scheduler: str = 'cosine'
    gradient_accumulation: int = 1
    grad_clip: float = 0.1
    early_stopping: int = 50
    save_period: int = -1
    val_period: int = 1
    resume: Optional[str] = None
    pretrained: Optional[str] = None
    
    # Device settings
    device: str = 'cuda'
    workers: int = 8
    
    # Output settings
    project: str = 'runs/train'
    name: str = 'exp'
    exist_ok: bool = False
    
    # Visualization settings
    vis_batches: int = 3  # Number of training batches to visualize
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
        if self.scheduler not in ['cosine', 'step', 'linear']:
            raise ValueError(f"scheduler must be 'cosine', 'step', or 'linear', got {self.scheduler}")
        if self.gradient_accumulation <= 0:
            raise ValueError(f"gradient_accumulation must be > 0, got {self.gradient_accumulation}")
        if self.workers < 0:
            raise ValueError(f"workers must be >= 0, got {self.workers}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'scheduler': self.scheduler,
            'gradient_accumulation': self.gradient_accumulation,
            'grad_clip': self.grad_clip,
            'early_stopping': self.early_stopping,
            'save_period': self.save_period,
            'val_period': self.val_period,
            'resume': self.resume,
            'pretrained': self.pretrained,
            'device': self.device,
            'workers': self.workers,
            'project': self.project,
            'name': self.name,
            'exist_ok': self.exist_ok,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    """
    Configuration for RF-DETR model.
    
    Attributes:
        model_size: Model size variant:
            - 'n' (nano): 384×384, ~30.5M params
            - 's' (small): 512×512, ~32.1M params  
            - 'm' (medium): 576×576, ~33.7M params
            - 'b' (base): 560×560, ~29M params
            - 'l' (large): 704×704, ~33.9M params
            - 'xl' (xlarge): 700×700, ~126.4M params (requires platform license)
            - '2xl' (2xlarge): 880×880, ~126.9M params (requires platform license)
        num_classes: Number of detection classes.
        pretrained_weights: Path to pretrained weights or None.
        freeze_encoder: Whether to freeze the encoder.
        freeze_encoder_epochs: Number of epochs to keep encoder frozen.
        accept_platform_license: Required for xl/2xl models (Platform Model License 1.0).
    """
    model_size: str = 'b'  # base
    num_classes: int = 80  # COCO classes
    pretrained_weights: Optional[str] = None
    freeze_encoder: bool = False
    freeze_encoder_epochs: int = 0
    accept_platform_license: bool = False  # Required for xl/2xl models
    
    def __post_init__(self):
        """Validate configuration values."""
        valid_sizes = ['n', 's', 'm', 'b', 'l', 'xl', '2xl']
        if self.model_size not in valid_sizes:
            raise ValueError(f"model_size must be one of {valid_sizes}, got {self.model_size}")
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {self.num_classes}")
        if self.freeze_encoder_epochs < 0:
            raise ValueError(f"freeze_encoder_epochs must be >= 0, got {self.freeze_encoder_epochs}")
        # Check platform license for xl/2xl
        if self.model_size in ['xl', '2xl'] and not self.accept_platform_license:
            raise ValueError(
                f"Model size '{self.model_size}' requires Platform Model License 1.0. "
                "Set accept_platform_license=True to use this model."
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_size': self.model_size,
            'num_classes': self.num_classes,
            'pretrained_weights': self.pretrained_weights,
            'freeze_encoder': self.freeze_encoder,
            'freeze_encoder_epochs': self.freeze_encoder_epochs,
            'accept_platform_license': self.accept_platform_license,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class ExportConfig:
    """
    Configuration for model export (ONNX, TensorRT).
    
    Attributes:
        enabled: Whether to export model after training.
        format: Export format ('onnx', 'tensorrt', 'both').
        simplify: Simplify ONNX model using onnxsim.
        opset_version: ONNX opset version.
        dynamic_batch: Enable dynamic batch size.
        half: Export model in FP16 (half precision).
        batch_size: Batch size for export (static batch).
        verbose: Verbose ONNX export.
    """
    enabled: bool = True               # Експортувати модель після тренування
    format: str = 'onnx'               # Формат: 'onnx', 'tensorrt', 'both'
    simplify: bool = True              # Спростити ONNX (onnxsim)
    opset_version: int = 17            # ONNX opset версія
    dynamic_batch: bool = False        # Динамічний batch size
    half: bool = False                 # FP16 (половинна точність)
    batch_size: int = 1                # Batch size для експорту
    verbose: bool = False              # Детальний вивід
    
    def __post_init__(self):
        """Validate configuration values."""
        valid_formats = ['onnx', 'tensorrt', 'both']
        if self.format not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}, got {self.format}")
        if self.opset_version < 11:
            raise ValueError(f"opset_version must be >= 11, got {self.opset_version}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'enabled': self.enabled,
            'format': self.format,
            'simplify': self.simplify,
            'opset_version': self.opset_version,
            'dynamic_batch': self.dynamic_batch,
            'half': self.half,
            'batch_size': self.batch_size,
            'verbose': self.verbose,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExportConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


def validate_config(
    augmentation_config: Optional[AugmentationConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    model_config: Optional[ModelConfig] = None,
) -> bool:
    """
    Validate configuration consistency across all configs.
    
    Args:
        augmentation_config: Augmentation configuration.
        training_config: Training configuration.
        model_config: Model configuration.
    
    Returns:
        True if all configurations are valid.
    
    Raises:
        ValueError: If configurations are inconsistent.
    """
    # Check close_mosaic vs epochs
    if augmentation_config and training_config:
        if augmentation_config.close_mosaic > training_config.epochs:
            warnings.warn(
                f"close_mosaic ({augmentation_config.close_mosaic}) > epochs ({training_config.epochs}). "
                "Mosaic will never be disabled."
            )
    
    # Check freeze_encoder_epochs vs epochs
    if model_config and training_config:
        if model_config.freeze_encoder_epochs > training_config.epochs:
            warnings.warn(
                f"freeze_encoder_epochs ({model_config.freeze_encoder_epochs}) > epochs ({training_config.epochs}). "
                "Encoder will remain frozen for entire training."
            )
    
    return True
