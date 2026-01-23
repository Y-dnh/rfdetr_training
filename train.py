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

import multiprocessing
from pathlib import Path

from rfdetr.training import (
    setup_seed,
    AugmentationConfig,
    TrainingConfig,
    ModelConfig,
    RFDETRTrainer,
)


# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ
# =============================================================================
SEED = 42
PROJECT_NAME = "rfdetr_training"

# Шляхи
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "tests" / "test_dataset"  # Змініть на ваш датасет

# Модель
PRETRAINED_WEIGHTS = None  # Шлях до ваг або None для завантаження з HuggingFace


# =============================================================================
# КОНФІГУРАЦІЯ МОДЕЛІ
# =============================================================================
MODEL_CONFIG = ModelConfig(
    model_size="b",                 # n=nano, s=small, m=medium, b=base, l=large
    num_classes=3,                  # Кількість класів (автовизначається з датасету)
    pretrained_weights=PRETRAINED_WEIGHTS,
    freeze_encoder=False,           # Заморозити encoder
)


# =============================================================================
# КОНФІГУРАЦІЯ ТРЕНУВАННЯ
# =============================================================================
TRAINING_CONFIG = TrainingConfig(
    # Налаштування проекту
    project=f"runs/{PROJECT_NAME}",
    name="exp",                     # Назва run: створює exp, exp2, exp3, ...
    exist_ok=False,                 # Перезаписувати існуючий run
    
    # Основні параметри навчання
    epochs=10,
    batch_size=8,  # Зменшено з 8 для економії GPU пам'яті
    
    # Оптимізатор
    lr=1e-4,                        # Learning rate
    weight_decay=1e-4,
    
    # Learning rate scheduler
    scheduler="cosine",             # "cosine", "step", "linear"
    
    # Валідація
    val_period=1,                   # Валідація кожні N епох
    
    # Device
    device="cuda",                  # "cuda" або "cpu"
    workers=8,                      # DataLoader workers (0 for Windows compatibility)
)


# =============================================================================
# КОНФІГУРАЦІЯ АУГМЕНТАЦІЙ (ULTRALYTICS-STYLE)
# =============================================================================
AUGMENTATION_CONFIG = AugmentationConfig(
    # Розмір зображення (має ділитися на 56 для DINOv2 backbone)
    imgsz=672,
    
    # HSV аугментації
    hsv_h=0.0,                    # HSV hue gain (0-1)
    hsv_s=0.0,                      # HSV saturation gain (0-1)
    hsv_v=0.3,                      # HSV value gain (0-1)
    
    # Геометричні трансформації
    degrees=5.0,                    # Поворот (+/- degrees)
    translate=0.1,                  # Зсув (fraction)
    scale=0.5,                      # Масштаб (+/- gain)
    shear=0.0,                      # Зсув перспективи (degrees)
    perspective=0.0,                # Перспективна трансформація (0-0.001)
    
    # Відзеркалення
    fliplr=0.5,                     # Horizontal flip probability
    flipud=0.0,                     # Vertical flip probability
    
    # Композитні аугментації
    mosaic=1.0,                     # Mosaic probability
    mixup=0.0,                      # MixUp probability
    cutmix=0.0,                     # CutMix probability
    
    # Random erasing
    erasing=0.0,                    # Random erasing probability
    
    # Close mosaic (вимкнути mosaic в останніх N епохах)
    close_mosaic=5,                # Disable mosaic last N epochs
)


# =============================================================================
# ГОЛОВНА ФУНКЦІЯ
# =============================================================================

def main():
    """Головна функція для запуску тренування."""
    # Налаштування seed для відтворюваності
    setup_seed(SEED)
    
    print("\n" + "=" * 70)
    print("RF-DETR TRAINING")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Model size: {MODEL_CONFIG.model_size}")
    print(f"Image size: {AUGMENTATION_CONFIG.imgsz}")
    print(f"Epochs: {TRAINING_CONFIG.epochs}")
    print(f"Batch size: {TRAINING_CONFIG.batch_size}")
    print("=" * 70 + "\n")
    
    # Перевірка датасету
    if not DATASET_DIR.exists():
        print(f"❌ ПОМИЛКА: Датасет не знайдено: {DATASET_DIR}")
        print("Будь ласка, вкажіть правильний шлях у змінній DATASET_DIR")
        return None
    
    # Створюємо trainer
    trainer = RFDETRTrainer(
        model_config=MODEL_CONFIG,
        training_config=TRAINING_CONFIG,
        augmentation_config=AUGMENTATION_CONFIG,
        seed=SEED,
    )
    
    # Запуск тренування
    results = trainer.train(dataset_dir=str(DATASET_DIR))
    
    # Виведення результатів
    print("\n" + "=" * 70)
    print("[OK] ТРЕНУВАННЯ ЗАВЕРШЕНО")
    print("=" * 70)
    print(f"Результати збережено: {results['save_dir']}")
    print(f"Best mAP: {results['best_map']:.4f}")
    print(f"Epochs trained: {results['epochs_trained']}")
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
    # resume_training("runs/rfdetr_training/exp/weights/last.pt")
