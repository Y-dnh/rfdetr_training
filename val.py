#!/usr/bin/env python3
"""
RF-DETR Validation Script
=========================
Валідація RF-DETR моделей з генерацією візуалізацій та збереженням метрик.

Usage:
    python val.py
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
# КОНФІГУРАЦІЯ ІНФЕРЕНСУ / ВАЛІДАЦІЇ
# =============================================================================
INFERENCE_CONFIG = {
    # -------------------------------------------------------------------------
    # Пороги детекції
    # -------------------------------------------------------------------------
    "conf_threshold": 0.25,          # Поріг впевненості (0-1), нижче = більше детекцій
    "iou_threshold": 0.5,            # IoU поріг для NMS (0-1), вище = менше фільтрації
    "max_det": 300,                  # Макс. детекцій на зображення (None = без ліміту)
    
    # -------------------------------------------------------------------------
    # Фільтрація класів
    # -------------------------------------------------------------------------
    "classes": None,                 # Фільтр класів: None, [0,1,2] або ["person","car"]
    "agnostic_nms": False,           # Class-agnostic NMS (ігнорує клас при фільтрації)
    
    # -------------------------------------------------------------------------
    # Оптимізація
    # -------------------------------------------------------------------------
    "half": False,                   # FP16 інференс (швидше, трохи менша точність)
    
    # -------------------------------------------------------------------------
    # Обробка
    # -------------------------------------------------------------------------
    "batch_size": 8,                 # Розмір батчу
    "workers": 4,                    # DataLoader workers (0 для Windows)
    "device": "cuda",                # Пристрій: "cuda", "cpu", "cuda:0"
    
    # -------------------------------------------------------------------------
    # Візуалізації
    # -------------------------------------------------------------------------
    "save_visualizations": True,     # Зберігати val_batch*_labels.jpg / val_batch*_pred.jpg
    "max_vis_batches": 3,            # Кількість батчів для візуалізації
    "save_analysis": True,           # Per-image аналіз (GT/TP/FP/FN)
    
    # -------------------------------------------------------------------------
    # Збереження результатів
    # -------------------------------------------------------------------------
    "save_json": True,               # Зберегти детекції в JSON (COCO format)
    "save_txt": False,               # Зберегти детекції в txt (YOLO format)
    
    # -------------------------------------------------------------------------
    # ПРИМІТКА: imgsz автоматично визначається з checkpoint
    # -------------------------------------------------------------------------
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
    """Запуск валідації."""
    dataset_dir = dataset_dir or str(DATASET_DIR)
    config = {**INFERENCE_CONFIG, **kwargs}
    
    print("\n" + "=" * 70)
    print("RF-DETR VALIDATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_dir}")
    print(f"Split: {split}")
    print(f"Conf threshold: {config['conf_threshold']}")
    print(f"IoU threshold: {config['iou_threshold']}")
    print(f"Max detections: {config['max_det']}")
    print(f"Half (FP16): {config['half']}")
    print("=" * 70 + "\n")
    
    # Перевірка датасету
    if not Path(dataset_dir).exists():
        print(f"ПОМИЛКА: Датасет не знайдено: {dataset_dir}")
        return None
    
    # Перевірка моделі
    if model_path and not Path(model_path).exists():
        print(f"УВАГА: Модель не знайдено: {model_path}")
        model_path = None
    
    # Створюємо validator
    validator = RFDETRValidator(
        model_path=model_path,
        conf_threshold=config["conf_threshold"],
        iou_threshold=config["iou_threshold"],
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
        save_validation_results(results, config)
    
    # Виведення результатів
    print_results(results)
    
    return results


def save_validation_results(results: dict, config: dict = None):
    """Збереження результатів у JSON."""
    output_dir = Path(results.get("save_dir", f"runs/{PROJECT_NAME}/val"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results["validation_date"] = datetime.now().isoformat()
    results["inference_config"] = config or INFERENCE_CONFIG
    
    results_path = output_dir / "validation_results.json"
    
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
    print("ВАЛІДАЦІЯ ЗАВЕРШЕНА")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
