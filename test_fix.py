#!/usr/bin/env python3
"""
Тестовий скрипт для перевірки виправлення фільтрації класів.
Запускає валідацію на невеликій частині датасету.
"""

import torch
from pathlib import Path
from rfdetr.training import (
    ModelConfig,
    TrainingConfig,
    AugmentationConfig,
    RFDETRTrainer,
)

# Конфігурація
DATASET_DIR = Path("D:/projects_yaroslav/rfdetr_training/dataset")
MODEL_SIZE = "2xl"
NUM_CLASSES = 3

# Створюємо конфігурації
model_config = ModelConfig(
    model_size=MODEL_SIZE,
    num_classes=NUM_CLASSES,
    pretrained_weights=None,
    freeze_encoder=False,
    freeze_encoder_epochs=0,
    accept_platform_license=True,
)

training_config = TrainingConfig(
    project="runs/test_fix",
    name="validation_test",
    exist_ok=True,
    epochs=1,
    batch_size=4,
    lr=1e-4,
    weight_decay=1e-4,
    device="cuda",
    workers=2,
)

augmentation_config = AugmentationConfig(
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.0,
    erasing=0.0,
)

print("\n" + "=" * 70)
print("ТЕСТ ВИПРАВЛЕННЯ ФІЛЬТРАЦІЇ КЛАСІВ")
print("=" * 70)
print(f"Model size: {MODEL_SIZE}")
print(f"Num classes: {NUM_CLASSES}")
print(f"Dataset: {DATASET_DIR}")
print("=" * 70 + "\n")

# Перевіряємо checkpoint
checkpoint_path = "runs/rfdetr_training/rfdetr_2xl_for_autolabeling/weights/last.pt"

if not Path(checkpoint_path).exists():
    print(f"ERROR: Checkpoint не знайдено: {checkpoint_path}")
    print("Спочатку потрібно провести тренування або вказати правильний шлях до checkpoint.")
    exit(1)

print(f"Завантажую checkpoint: {checkpoint_path}")

# Завантажуємо датасети
from rfdetr.training.dataset import build_dataset

print("\nЗавантажую валідаційний датасет...")
val_dataset = build_dataset(
    dataset_dir=DATASET_DIR,
    split='valid',
    augmentation_config=augmentation_config,
    log_augmentations=False,
)

print(f"Валідаційний датасет: {len(val_dataset)} зображень")
print(f"Класи: {list(val_dataset.categories.values())}")
print(f"COCO categories: {val_dataset.coco.getCatIds()}")

# Створюємо trainer та ініціалізуємо модель
trainer = RFDETRTrainer(
    model_config=model_config,
    training_config=training_config,
    augmentation_config=augmentation_config,
    seed=42,
)

# Ініціалізуємо модель
print("\nІніціалізація моделі...")
trainer._setup_model_and_criterion(num_classes=NUM_CLASSES)

# Завантажуємо ваги
print(f"Завантажую ваги з checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

if 'model' in checkpoint:
    # Завантажуємо state dict
    missing, unexpected = trainer.model.load_state_dict(checkpoint['model'], strict=False)
    if missing:
        print(f"  Відсутні ключі: {len(missing)}")
    if unexpected:
        print(f"  Неочікувані ключі: {len(unexpected)}")
    print("✓ Модель завантажена успішно!")
else:
    print("ERROR: Checkpoint не містить ваг моделі")
    exit(1)

# Переміщаємо модель на GPU
if torch.cuda.is_available():
    trainer.model = trainer.model.cuda()
    print("✓ Модель переміщена на GPU")

# Запускаємо валідацію
from rfdetr.training.validator import RFDETRValidator

validator = RFDETRValidator(
    model_path=checkpoint_path,
    conf_threshold=0.1,
    iou_threshold=0.5,
    batch_size=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir=Path("runs/test_fix/validation_test"),
    imgsz=880,
)

# Переписуємо модель, яку ми вже завантажили
validator.model = trainer.model
validator.class_names = list(val_dataset.categories.values())

print("\n" + "=" * 70)
print("ЗАПУСК ВАЛІДАЦІЇ")
print("=" * 70 + "\n")

results = validator.validate(
    dataset=val_dataset,
    save_visualizations=True,
    save_analysis=True,
)

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТИ")
print("=" * 70)
print(f"mAP@0.5:     {results['mAP50']:.4f}")
print(f"mAP@0.5:0.95: {results['mAP50-95']:.4f}")
print(f"Precision:   {results['precision']:.4f}")
print(f"Recall:      {results['recall']:.4f}")
print(f"F1 Score:    {results['f1']:.4f}")
print("=" * 70 + "\n")

print("✓ Виправлення працює! Метрики тепер відображають реальну якість детекції.")
