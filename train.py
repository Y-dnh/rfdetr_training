#!/usr/bin/env python3
"""
RF-DETR Training Script
=======================
Тренування RF-DETR моделей детекції об'єктів з YOLO-style аугментаціями.
Усі параметри конфігурації знаходяться на початку файлу.

Usage:
    python train.py

Налаштуйте конфігурацію нижче перед запуском.
"""

import os
import multiprocessing
from pathlib import Path

from rfdetr.training import (
    setup_seed,
    AugmentationConfig,
    TrainingConfig,
    ModelConfig,
    ExportConfig,
    RFDETRTrainer,
)
# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ: ШЛЯХИ
# =============================================================================
SEED = 42
PROJECT_NAME = "rfdetr_large"
BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"
# У WSL задай: export RFDETR_DATASET_ROOT=/mnt/d/dataset_for_training
DATASET_ROOT = os.environ.get("RFDETR_DATASET_ROOT", "D:/dataset_for_training")
PROJECT_DIR = RUNS_DIR / PROJECT_NAME
# Датасет (COCO JSON): DATASET_ROOT містить train/, valid/, annotations/ тощо
DATASET_DIR = Path(DATASET_ROOT)

# Модель
PRETRAINED_WEIGHTS = None  # Шлях до ваг або None для завантаження з HuggingFace


# =============================================================================
# КОНФІГУРАЦІЯ МОДЕЛІ
# =============================================================================
MODEL_CONFIG = ModelConfig(
    model_size="m",                 # 'n','s','m','b','l','xl','2xl' | Рекомендовано: 'b' або 'l'
    # num_classes — автоматично визначається з COCO JSON анотацій датасету.
    # Вказане значення ігнорується при тренуванні.
    pretrained_weights=PRETRAINED_WEIGHTS,  # Шлях .pt або None (авто-завантаження з HuggingFace)
    freeze_encoder=False,           # [True/False] Заморозити DINOv2 backbone | Рекомендовано: False
    freeze_encoder_epochs=0,        # [0–epochs] Заморозити encoder на перші N епох | 0=не заморожувати
    accept_platform_license=True,   # Обов'язково True для xl/2xl (Platform Model License 1.0)
)

# Розмір зображення по моделі (для albu config і логів)
MODEL_RESOLUTIONS = {'n': 384, 's': 512, 'm': 576, 'b': 560, 'l': 704, 'xl': 700, '2xl': 880}

# =============================================================================
# КОНФІГУРАЦІЯ ТРЕНУВАННЯ
# =============================================================================
TRAINING_CONFIG = TrainingConfig(
    # -------------------------------------------------------------------------
    # Налаштування проекту (результати в PROJECT_DIR/<name>/ — Ultralytics-style)
    # -------------------------------------------------------------------------
    project=str(PROJECT_DIR),        # runs/.../ → зберігається в PROJECT_DIR/<name>/
    name="baseline",                 # Назва експерименту: runs/.../baseline/
    exist_ok=False,                  # [True/False] True=перезаписати існуючий run
    
    # -------------------------------------------------------------------------
    # Основні параметри навчання
    # -------------------------------------------------------------------------
    epochs=100,                       # [≥1] Кількість епох | Рекомендовано: 50–300 (fine-tune: 20–100)
    batch_size=4,                    # [≥1] Розмір батчу | Залежить від GPU VRAM | Рекомендовано: 4–32
    
    # -------------------------------------------------------------------------
    # Оптимізатор (AdamW)
    # -------------------------------------------------------------------------
    lr=1e-4,                         # [>0] Learning rate | Рекомендовано: 1e-5–5e-4 | Fine-tune: 1e-4
    weight_decay=1e-4,               # [≥0] L2 регуляризація | Рекомендовано: 1e-5–1e-3
    
    # -------------------------------------------------------------------------
    # Learning rate scheduler
    # -------------------------------------------------------------------------
    scheduler="cosine",              # ['cosine','step','linear'] | Рекомендовано: 'cosine'
    warmup_epochs=5,                 # [≥0] Warmup епох | Рекомендовано: 1–10 | 0=без warmup
    
    # -------------------------------------------------------------------------
    # Градієнти
    # -------------------------------------------------------------------------
    gradient_accumulation=4,         # [≥1] Ефективний batch = batch_size × accumulation | Рекомендовано: 1–8
    grad_clip=0.1,                   # [≥0] Макс. норма градієнта | 0=вимкнено | Рекомендовано: 0.05–0.5
    
    # -------------------------------------------------------------------------
    # Валідація та збереження
    # -------------------------------------------------------------------------
    val_period=1,                    # [≥1] Валідація кожні N епох | Рекомендовано: 1–5
    save_period=-1,                  # [≥1 або -1] Checkpoint кожні N епох | -1=тільки best/last
    early_stopping=30,               # [≥0] Зупинка якщо mAP не росте N епох | 0=вимкнено | Рекомендовано: 10–50
    
    # -------------------------------------------------------------------------
    # Візуалізації
    # -------------------------------------------------------------------------
    vis_batches=30,                  # [≥0] Батчів для візуалізації (train/val) | 0=вимкнено
    
    # -------------------------------------------------------------------------
    # Device та workers
    # -------------------------------------------------------------------------
    device="cuda",                   # ['cuda','cpu','cuda:0','cuda:1'] | Рекомендовано: 'cuda'
    workers=4,                       # [≥0] DataLoader workers | 0=основний потік (Windows) | Рекомендовано: 2–8
)


# =============================================================================
# ALBUMENTATION_CONFIG — тепловізія (white hot), малі об'єкти, PTZ 1280x720/1024
# Камера на башті ~30м, фіксована, горизонт рівний; детекція людей і машин.
# 50% tiny / 30% small — м'який dropout (менші діри), помірна геометрія.
# =============================================================================
import albumentations as A
ALBUMENTATION_CONFIG = [
    A.HorizontalFlip(p=0.5),

    A.OneOf([
        A.AtLeastOneBBoxRandomCrop(
            height=576,
            width=576,
            erosion_factor=0.2,
            p=0.35,
        ),
        A.RandomCropNearBBox(
            max_part_shift=(0.05, 0.2),
            p=0.45,
        ),
        A.RandomSizedCrop(
            min_max_height=(384, 512),
            size=(576, 576),
            w2h_ratio=1.0,
            p=0.20,
        ),
    ], p=0.25),

    A.Affine(
        scale=(0.95, 1.08),
        translate_percent=(0.0, 0.03),
        rotate=(-4, 4),
        shear=(-2, 2),
        p=0.20
    ),

    A.CLAHE(
        clip_limit=(1, 3),
        tile_grid_size=(8, 8),
        p=0.20
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.10,
        contrast_limit=0.12,
        p=0.25
    ),

    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=(3, 5), p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
    ], p=0.10),

    A.GaussNoise(
        std_range=(0.02, 0.05),
        p=0.08
    ),

    A.CoarseDropout(
        num_holes_range=(1, 3),
        hole_height_range=(0.01, 0.03),
        hole_width_range=(0.01, 0.03),
        p=0.06
    ),
]

# =============================================================================
# КОНФІГУРАЦІЯ АУГМЕНТАЦІЙ (пайплайн + imgsz + albu список)
# =============================================================================
AUGMENTATION_CONFIG = AugmentationConfig(
    imgsz=MODEL_RESOLUTIONS.get(MODEL_CONFIG.model_size, 576),
    mosaic=0.0,
    close_mosaic=0,
    mixup=0.0,
    cutmix=0.0,
    mosaic_scale=(0.5, 1.5),
    mosaic_min_box_size=2,
    mixup_alpha=32.0,
    cutmix_alpha=1.0,
    cutmix_min_visible=0.3,
    cutmix_min_box_size=10,
    cutmix_overlap_thresh=0.1,
    letterbox_color=(114, 114, 114),
    albumentation_transforms=ALBUMENTATION_CONFIG,
)


# =============================================================================
# КОНФІГУРАЦІЯ ЕКСПОРТУ (ONNX / TensorRT)
# =============================================================================
EXPORT_CONFIG = ExportConfig(
    # -------------------------------------------------------------------------
    # Основні налаштування
    # -------------------------------------------------------------------------
    enabled=True,                     # [True/False] Експортувати модель після тренування
    format='onnx',                    # ['onnx','tensorrt','both'] | Рекомендовано: 'onnx'
    
    # -------------------------------------------------------------------------
    # ONNX налаштування
    # -------------------------------------------------------------------------
    simplify=True,                    # [True/False] Спростити onnxsim (менший розмір) | Рекомендовано: True
    opset_version=17,                 # [≥11] ONNX opset версія | Рекомендовано: 16–17
    
    # -------------------------------------------------------------------------
    # Batch та точність
    # -------------------------------------------------------------------------
    dynamic_batch=False,              # [True/False] Різний batch при інференсі | False=фіксований
    batch_size=1,                     # [≥1] Batch size при інференсі (ігнорується якщо dynamic_batch=True)
    half=True,                       # [True/False] FP16 (швидше, менше пам'яті) | False=FP32 (точніше)
    
    # -------------------------------------------------------------------------
    # Інше
    # -------------------------------------------------------------------------
    verbose=True,                    # [True/False] Детальний ONNX export лог
)


# =============================================================================
# ГОЛОВНА ФУНКЦІЯ
# =============================================================================

def main():
    """Головна функція для запуску тренування."""
    # Налаштування seed для відтворюваності
    setup_seed(SEED)
    
    # Resolution по розміру моделі
    resolution = MODEL_RESOLUTIONS.get(MODEL_CONFIG.model_size, 560)
    
    print("\n" + "=" * 70)
    print("RF-DETR TRAINING")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Model size: {MODEL_CONFIG.model_size}")
    print(f"Image size: {resolution} (auto)")
    print(f"Epochs: {TRAINING_CONFIG.epochs}")
    print(f"Batch size: {TRAINING_CONFIG.batch_size}")
    print("=" * 70 + "\n")
    
    # Перевірка датасету
    if not DATASET_DIR.exists():
        print(f"ERROR: Датасет не знайдено: {DATASET_DIR}")
        print("Будь ласка, вкажіть правильний шлях у змінній DATASET_DIR")
        return None
    
    # Створюємо trainer
    trainer = RFDETRTrainer(
        model_config=MODEL_CONFIG,
        training_config=TRAINING_CONFIG,
        augmentation_config=AUGMENTATION_CONFIG,
        export_config=EXPORT_CONFIG,
        seed=SEED,
    )

    # Запуск тренування (resume з конфігу передається в train())
    results = trainer.train(
        dataset_dir=str(DATASET_DIR),
        resume=TRAINING_CONFIG.resume,
    )
    
    # Виведення результатів
    print("\n" + "=" * 70)
    print("[OK] ТРЕНУВАННЯ ЗАВЕРШЕНО")
    print("=" * 70)
    print(f"Результати збережено: {results['save_dir']}")
    print(f"Best mAP: {results['best_map']:.4f}")
    print(f"Epochs trained: {results['epochs_trained']}")
    if results.get('onnx_path'):
        print(f"ONNX model: {results['onnx_path']}")
    print("=" * 70 + "\n")
    
    return results


def resume_training(checkpoint_path: str):
    """
    Продовження тренування з checkpoint.
    
    Args:
        checkpoint_path: Шлях до checkpoint файлу
    """
    setup_seed(SEED)
    
    trainer = RFDETRTrainer(
        model_config=MODEL_CONFIG,
        training_config=TRAINING_CONFIG,
        augmentation_config=AUGMENTATION_CONFIG,
        export_config=EXPORT_CONFIG,
        seed=SEED,
    )

    print(f"\n[Resume] Продовження тренування з: {checkpoint_path}\n")
    results = trainer.train(dataset_dir=str(DATASET_DIR), resume=checkpoint_path)
    
    return results


if __name__ == "__main__":
    # Windows multiprocessing fix (дозволяє використовувати workers > 0)
    multiprocessing.freeze_support()
    
    # Режим 1: Базове тренування
    main()
    
    # Режим 2: Продовження з checkpoint
    # resume_training("runs/rfdetr_large/baseline/weights/last.pt")
