"""
Configuration classes for RF-DETR training.

This module provides dataclasses for configuring:
- Augmentations (YOLO-style parameters)
- Training hyperparameters
- Model settings
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
import warnings


@dataclass
class AugmentationConfig:
    """
    Configuration for data augmentations.
    
    The config is split into two sections:
    1. ОСНОВНІ ПАРАМЕТРИ — augmentation probabilities/gains (what to enable and how often)
    2. ТОНКІ НАЛАШТУВАННЯ — fine-tuning parameters (ranges, thresholds, limits)
    
    Most users only need to adjust the main parameters.
    Fine-tuning defaults work well for most use cases.
    """
    
    # =====================================================================
    # ОСНОВНІ ПАРАМЕТРИ (ймовірності та сила аугментацій)
    # =====================================================================
    
    # Image size (auto-synced from model at training time)
    # Nano=384, Small=512, Medium=576, Base=560, Large=704, XLarge=700, 2XLarge=880
    # Вказуйте вручну тільки для preview. При тренуванні перезаписується автоматично.
    imgsz: int = 640                  # Діапазон: >0 | Авто: визначається з моделі
    
    # --- Композитні аугментації (об'єднання кількох зображень) ---
    
    # Mosaic: 4 зображення → одне. Центр сітки рандомний.
    mosaic: float = 1.0               # Діапазон: [0.0–1.0] | 0=вимкнено | Рекомендовано: 0.5–1.0
    close_mosaic: int = 10            # Діапазон: [0–epochs] | Вимкнути mosaic в останніх N епохах | 0=завжди вимкнено
    
    # MixUp: альфа-блендинг 2 зображень (зазвичай слабкий).
    mixup: float = 0.0                # Діапазон: [0.0–1.0] | 0=вимкнено | Рекомендовано: 0.0–0.15
    
    # CutMix: вирізає регіон з іншого зображення і вставляє.
    cutmix: float = 0.0               # Діапазон: [0.0–1.0] | 0=вимкнено | Рекомендовано: 0.0–0.3
    
    # --- Колірні аугментації ---
    
    # HSV: gain-фактори зсуву каналів H/S/V. Не ймовірність — сила зсуву.
    hsv_h: float = 0.015              # Діапазон: [0.0–1.0] | Зсув Hue: ±h_gain*180° | Рекомендовано: 0.0–0.03
    hsv_s: float = 0.7                # Діапазон: [0.0–1.0] | Масштаб Saturation: ×(1±s_gain) | Рекомендовано: 0.0–0.9
    hsv_v: float = 0.4                # Діапазон: [0.0–1.0] | Масштаб Value: ×(1±v_gain) | Рекомендовано: 0.0–0.5
    
    # Інші колірні аугментації — ймовірності.
    brightness: float = 0.0           # Діапазон: [0.0–1.0] | 0=вимкнено | Рекомендовано: 0.0–0.3
    contrast: float = 0.0             # Діапазон: [0.0–1.0] | 0=вимкнено | Рекомендовано: 0.0–0.3
    blur: float = 0.0                 # Діапазон: [0.0–1.0] | Gaussian blur | Рекомендовано: 0.0–0.2
    noise: float = 0.0                # Діапазон: [0.0–1.0] | 0=вимкнено | Рекомендовано: 0.0–0.2
    noise_type: str = 'gaussian_mono' # Варіанти: 'gaussian_mono'(IR), 'gaussian_rgb'(RGB), 'salt_pepper'
    
    # --- Геометричні аугментації ---
    
    degrees: float = 0.0              # Діапазон: [0–180] | Поворот ±degrees | Рекомендовано: 0–15 | 0=вимкнено
    translate: float = 0.1            # Діапазон: [0.0–1.0] | Зсув як частка imgsz | Рекомендовано: 0.0–0.2
    scale: float = 0.5                # Діапазон: [0.0–0.9] | Масштаб: ×(1±scale) | Рекомендовано: 0.0–0.5 | 0=вимкнено
    shear: float = 0.0                # Діапазон: [0–90] | Зсув перспективи (градуси) | Рекомендовано: 0–5 | 0=вимкнено
    perspective: float = 0.0          # Діапазон: [0.0–0.001] | Перспективна деформація | Рекомендовано: 0.0–0.0005

    # --- Відзеркалення ---
    
    fliplr: float = 0.5               # Діапазон: [0.0–1.0] | Горизонтальний flip | Рекомендовано: 0.5
    flipud: float = 0.0               # Діапазон: [0.0–1.0] | Вертикальний flip | Рекомендовано: 0.0 (0.5 для аеро/супутник)
    
    # --- Random Erasing (box-aware видалення регіонів) ---
    
    erasing: float = 0.0              # Діапазон: [0.0–1.0] | 0=вимкнено | Рекомендовано: 0.0–0.4
    erasing_value: Union[str, float] = 128  # Варіанти: 0=чорний, 128=сірий, 255=білий, 'random'=шум
    
    # =====================================================================
    # ТОНКІ НАЛАШТУВАННЯ (зазвичай змінювати не потрібно)
    # =====================================================================
    
    # Mosaic
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)  # Діапазон: (>0, >0), max≤2.0 | Центр мозаїки = imgsz × [min, max]
    mosaic_min_box_size: int = 2      # Діапазон: ≥1 пікс | Боксы менше — відкидаються | Рекомендовано: 2–4
    
    # MixUp
    mixup_alpha: float = 32.0         # Діапазон: >0 | Beta(α,α): 1.0=рівномірний мікс, 32+=слабкий | Рекомендовано: 8–32
    
    # CutMix
    cutmix_alpha: float = 1.0         # Діапазон: >0 | Beta(α,α): 1.0=рівномірний розмір | Рекомендовано: 0.5–2.0
    cutmix_min_visible: float = 0.3   # Діапазон: [0.0–1.0] | Мін. видима частина боксу | Рекомендовано: 0.2–0.5
    cutmix_min_box_size: int = 10     # Діапазон: ≥1 пікс | Мін. розмір боксу після обрізання | Рекомендовано: 5–20
    cutmix_overlap_thresh: float = 0.1  # Діапазон: [0.0–1.0] | Нижче — бокс не змінюється | Рекомендовано: 0.05–0.2
    
    # Color
    brightness_range: Tuple[float, float] = (0.5, 1.5)  # Діапазон: (>0, >0) | Множник: <1=темніше, >1=яскравіше | Рекомендовано: (0.5, 1.5)
    contrast_range: Tuple[float, float] = (0.5, 1.5)    # Діапазон: (>0, >0) | Множник: <1=менше, >1=більше | Рекомендовано: (0.5, 1.5)
    blur_kernel_range: Tuple[int, int] = (3, 7)          # Діапазон: (≥3, ≤15) непарні | Рекомендовано: (3, 7) | Більше=сильніший blur
    noise_strength: Tuple[float, float] = (5.0, 30.0)    # Діапазон: (>0, >0) std dev | Рекомендовано: (5, 30) | Більше=сильніший шум
    salt_pepper_amount: float = 0.02                      # Діапазон: [0.0–1.0] | Частка пікселів | Рекомендовано: 0.01–0.05
    
    # Random Erasing
    erasing_min_scale: float = 0.02    # Діапазон: [0.0–1.0] < max_scale | Мін. частка площі | Рекомендовано: 0.02
    erasing_max_scale: float = 0.33    # Діапазон: [0.0–1.0] > min_scale | Макс. частка площі | Рекомендовано: 0.2–0.4
    erasing_ratio: Tuple[float, float] = (0.3, 3.3)  # Діапазон: (>0, >0) | Aspect ratio вирізу | Рекомендовано: (0.3, 3.3)
    erasing_min_visible: float = 0.5   # Діапазон: [0.0–1.0] | Мін. видима частина боксу | Рекомендовано: 0.3–0.7
    erasing_min_box_size: int = 20     # Діапазон: ≥1 пікс | Мін. розмір боксу | Рекомендовано: 10–30
    
    # LetterBox / Padding
    letterbox_color: Tuple[int, int, int] = (114, 114, 114)  # Діапазон: (0-255, 0-255, 0-255) RGB | (114,114,114)=сірий, (0,0,0)=чорний для IR
    
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
        """Convert config to dictionary (all fields)."""
        import dataclasses
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
    
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
    # --- Основні параметри навчання ---
    epochs: int = 100                 # Діапазон: ≥1 | Рекомендовано: 50–300 (fine-tune: 20–100)
    batch_size: int = 16              # Діапазон: ≥1 | Залежить від GPU VRAM | Рекомендовано: 4–32
    
    # --- Оптимізатор (AdamW) ---
    lr: float = 1e-4                  # Діапазон: >0 | Рекомендовано: 1e-5–5e-4 | Fine-tune: 1e-4
    weight_decay: float = 1e-4        # Діапазон: ≥0 | L2 регуляризація | Рекомендовано: 1e-5–1e-3
    
    # --- Learning rate scheduler ---
    warmup_epochs: int = 5            # Діапазон: ≥0 | Рекомендовано: 1–10 | 0=без warmup
    scheduler: str = 'cosine'         # Варіанти: 'cosine', 'step', 'linear' | Рекомендовано: 'cosine'
    
    # --- Градієнти ---
    gradient_accumulation: int = 1    # Діапазон: ≥1 | Ефективний batch = batch_size × accumulation | Рекомендовано: 1–8
    grad_clip: float = 0.1            # Діапазон: ≥0 | Макс. норма градієнта | 0=вимкнено | Рекомендовано: 0.05–0.5
    
    # --- Валідація та збереження ---
    early_stopping: int = 50          # Діапазон: ≥0 | Зупинка якщо mAP не росте N епох | 0=вимкнено | Рекомендовано: 10–50
    save_period: int = -1             # Діапазон: ≥1 або -1 | Checkpoint кожні N епох | -1=тільки best/last
    val_period: int = 1               # Діапазон: ≥1 | Валідація кожні N епох | Рекомендовано: 1–5
    resume: Optional[str] = None      # Шлях до checkpoint (.pt) для продовження тренування
    pretrained: Optional[str] = None  # Шлях до ваг або ім'я моделі
    
    # --- Device ---
    device: str = 'cuda'              # Варіанти: 'cuda', 'cpu', 'cuda:0', 'cuda:1'
    workers: int = 8                  # Діапазон: ≥0 | DataLoader workers | 0=основний потік (Windows)
    
    # --- Вивід ---
    project: str = 'runs/train'       # Базова папка для збереження результатів
    name: str = 'exp'                 # Назва run: exp, exp2, exp3, ...
    exist_ok: bool = False            # True=перезаписати існуючий run | False=створити новий
    
    # --- Візуалізації ---
    vis_batches: int = 3              # Діапазон: ≥0 | Кількість батчів для візуалізації | 0=вимкнено
    
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
        num_classes: NOT USED during training (auto-detected from dataset COCO JSON).
            Kept for serialization/logging purposes only.
        pretrained_weights: Path to pretrained weights or None.
        freeze_encoder: Whether to freeze the encoder.
        freeze_encoder_epochs: Number of epochs to keep encoder frozen.
        accept_platform_license: Required for xl/2xl models (Platform Model License 1.0).
    """
    model_size: str = 'b'             # Варіанти: 'n','s','m','b','l','xl','2xl' | Рекомендовано: 'b' або 'l'
    # num_classes — автоматично визначається з COCO JSON анотацій датасету.
    # Тут використовується тільки для серіалізації/логування.
    num_classes: int = 80             # Діапазон: ≥1 | Авто: з COCO JSON | Ручне значення ігнорується при тренуванні
    pretrained_weights: Optional[str] = None  # Шлях до .pt файлу або None (HuggingFace авто-завантаження)
    freeze_encoder: bool = False      # True=заморозити DINOv2 backbone | Рекомендовано: False
    freeze_encoder_epochs: int = 0    # Діапазон: [0–epochs] | Заморозити encoder на перші N епох | 0=не заморожувати
    accept_platform_license: bool = False  # Обов'язково True для xl/2xl (Platform Model License 1.0)
    
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
    enabled: bool = True               # True=експортувати після тренування | False=пропустити
    format: str = 'onnx'              # Варіанти: 'onnx', 'tensorrt', 'both' | Рекомендовано: 'onnx'
    simplify: bool = True             # True=спростити onnxsim (менший розмір, швидший інференс) | Рекомендовано: True
    opset_version: int = 17           # Діапазон: ≥11 | Рекомендовано: 16–17 | 17=найновіший стабільний
    dynamic_batch: bool = False       # True=різний batch при інференсі | False=фіксований batch_size
    half: bool = False                # True=FP16 (швидше, менше пам'яті) | False=FP32 (точніше)
    batch_size: int = 1               # Діапазон: ≥1 | Batch size при інференсі (ігнорується якщо dynamic_batch=True)
    verbose: bool = False             # True=детальний ONNX export лог | False=тихий режим
    
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
