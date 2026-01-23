#!/usr/bin/env python3
"""
RF-DETR Validation Script
=========================
Валідація RF-DETR моделей з генерацією візуалізацій та збереженням метрик.
Усі параметри конфігурації знаходяться на початку файлу.

Usage:
    python val.py

Налаштуйте конфігурацію нижче перед запуском.
"""

import json
from pathlib import Path
from datetime import datetime

from rfdetr.training import RFDETRValidator


# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ
# =============================================================================
PROJECT_NAME = "rfdetr_validation"

# Шляхи
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "tests" / "test_dataset"  # Змініть на ваш датасет

# Модель
MODEL_PATH = "runs/rfdetr_training/exp46/weights/best.pt"  # Шлях до навченої моделі


# =============================================================================
# КОНФІГУРАЦІЯ ВАЛІДАЦІЇ
# =============================================================================
VALIDATION_CONFIG = {
    # Параметри детекції
    "conf_threshold": 0.25,         # Confidence threshold
    "iou_threshold": 0.5,           # IoU threshold для NMS
    
    # Розмір зображення
    "imgsz": 672,
    
    # Обробка
    "batch_size": 8,
    "workers": 4,
    "device": "cuda",               # "cuda" або "cpu"
    
    # Візуалізації
    "save_visualizations": True,    # Генерувати batch візуалізації
    
    # Аналіз детекцій (2x2 grid: GT | TP / FP | FN)
    # Створює окрему папку "analysis" з візуалізаціями для кожного зображення
    "save_analysis": True,          # Генерувати per-image analysis візуалізації
}


# =============================================================================
# ГОЛОВНА ФУНКЦІЯ
# =============================================================================

def main(
    model_path: str = MODEL_PATH,
    dataset_dir: str = None,
    split: str = "test",
    save_results: bool = True,
    **kwargs
):
    """
    Головна функція для запуску валідації.
    
    Args:
        model_path: Шлях до навченої моделі
        dataset_dir: Шлях до датасету
        split: Split для валідації ("valid" або "test")
        save_results: Чи зберігати результати
        **kwargs: Додаткові параметри валідації
    """
    dataset_dir = dataset_dir or str(DATASET_DIR)
    
    print("\n" + "=" * 70)
    print("RF-DETR VALIDATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_dir}")
    print(f"Split: {split}")
    print(f"Conf threshold: {VALIDATION_CONFIG['conf_threshold']}")
    print(f"IoU threshold: {VALIDATION_CONFIG['iou_threshold']}")
    print("=" * 70 + "\n")
    
    # Перевірка датасету
    if not Path(dataset_dir).exists():
        print(f"❌ ПОМИЛКА: Датасет не знайдено: {dataset_dir}")
        print("Будь ласка, вкажіть правильний шлях у змінній DATASET_DIR")
        return None
    
    # Перевірка моделі (якщо вказано)
    if model_path and not Path(model_path).exists():
        print(f"⚠️ УВАГА: Модель не знайдено: {model_path}")
        print("Буде використано тільки ground truth візуалізації")
        model_path = None
    
    # Merge configs
    config = {**VALIDATION_CONFIG, **kwargs}
    
    # Створюємо validator
    validator = RFDETRValidator(
        model_path=model_path,
        conf_threshold=config["conf_threshold"],
        iou_threshold=config["iou_threshold"],
        imgsz=config["imgsz"],
        batch_size=config["batch_size"],
        workers=config["workers"],
        device=config["device"],
        save_dir=f"runs/{PROJECT_NAME}/val",
    )
    
    # Запуск валідації
    results = validator.validate(
        dataset_dir=dataset_dir,
        split=split,
        save_visualizations=config.get("save_visualizations", True),
        save_analysis=config.get("save_analysis", False),
    )
    
    # Збереження результатів
    if save_results and results:
        save_validation_results(results)
    
    # Виведення результатів
    print_results(results)
    
    return results


def save_validation_results(results: dict):
    """Збереження результатів у JSON."""
    output_dir = Path(results.get("save_dir", f"runs/{PROJECT_NAME}/val"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Додаємо метадані
    results["validation_date"] = datetime.now().isoformat()
    results["config"] = VALIDATION_CONFIG
    
    # Зберігаємо
    results_path = output_dir / "validation_results.json"
    
    # Конвертуємо нон-серіалізовані типи
    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=convert)
    
    print(f"\n[Results] Результати збережено: {results_path}")


def print_results(results: dict):
    """Виведення результатів у консоль."""
    if not results:
        return
    
    metrics = results.get("metrics", {})
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТИ ВАЛІДАЦІЇ")
    print("=" * 70)
    print(f"  mAP@0.5:      {metrics.get('mAP50', 0):.4f}")
    print(f"  mAP@0.5:0.95: {metrics.get('mAP50-95', 0):.4f}")
    print(f"  Precision:    {metrics.get('precision', 0):.4f}")
    print(f"  Recall:       {metrics.get('recall', 0):.4f}")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("✓ ВАЛІДАЦІЯ ЗАВЕРШЕНА")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Запуск валідації
    main()

