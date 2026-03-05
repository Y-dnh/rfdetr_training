#!/usr/bin/env python3
"""
RF-DETR Validation Script
=========================
Валідація RF-DETR моделей з генерацією візуалізацій та збереженням метрик.

Usage:
    python val.py
"""

import json
import os
from pathlib import Path
from datetime import datetime

from rfdetr.training import RFDETRValidator


# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ: ШЛЯХИ
# =============================================================================
# Та сама структура, що в train.py. Валідація зберігає в PROJECT_DIR/validation/
PROJECT_NAME = "rfdetr_large"
TRAINING_RUN_NAME = "baseline"   # Назва запуску тренування (training/<name>/)
BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"
# У WSL задай: export RFDETR_DATASET_ROOT=/mnt/d/dataset_for_training
DATASET_ROOT = os.environ.get("RFDETR_DATASET_ROOT", "D:/dataset_for_training")
PROJECT_DIR = RUNS_DIR / PROJECT_NAME
# Датасет (COCO JSON)
DATASET_DIR = Path(DATASET_ROOT)

# Модель: за замовчуванням best.pt з runs/.../training/<name>/weights/
# Можна перевизначити вручну для зовнішніх checkpoint'ів
MODEL_PATH = PROJECT_DIR / "training" / TRAINING_RUN_NAME / "weights" / "best.pt"
MODEL_SIZE = "m"                # ['n','s','m','b','l','xl','2xl'] | Має відповідати checkpoint'у


# =============================================================================
# КОНФІГУРАЦІЯ ІНФЕРЕНСУ / ВАЛІДАЦІЇ
# =============================================================================
INFERENCE_CONFIG = {
    "split": "valid",
    # -------------------------------------------------------------------------
    # Пороги детекції
    # -------------------------------------------------------------------------
    "conf_threshold": 0.5,           # [0.0–1.0] Поріг впевненості | Нижче=більше детекцій | Рекомендовано: 0.25–0.7
    "iou_threshold": 0.5,            # [0.0–1.0] IoU поріг NMS | Вище=менше фільтрації | Рекомендовано: 0.4–0.7
    "max_det": 300,                  # [≥1 або None] Макс. детекцій на зображення | None=без ліміту
    
    # -------------------------------------------------------------------------
    # Фільтрація класів
    # -------------------------------------------------------------------------
    "classes": None,                 # None=всі класи | [0,1,2] або ["person","car"]=фільтр
    "agnostic_nms": False,           # [True/False] Class-agnostic NMS | True=ігнорує клас
    
    # -------------------------------------------------------------------------
    # Оптимізація
    # -------------------------------------------------------------------------
    "half": True,                    # [True/False] FP16 інференс (швидше, менше VRAM) | False=FP32
    
    # -------------------------------------------------------------------------
    # Обробка
    # -------------------------------------------------------------------------
    "batch_size": 2,                 # [≥1] Розмір батчу | Залежить від GPU VRAM | Рекомендовано: 1–16
    "workers": 2,                    # [≥0] DataLoader workers | 0=основний потік (Windows) | Рекомендовано: 2–8
    "device": "cuda",                # ['cuda','cpu','cuda:0','cuda:1'] | Рекомендовано: 'cuda'
    
    # -------------------------------------------------------------------------
    # Візуалізації
    # -------------------------------------------------------------------------
    "save_visualizations": True,     # [True/False] Зберігати візуалізації (GT/TP/FP/FN)
    "max_vis_batches": 3,            # [≥0] Батчів для візуалізації | 0=вимкнено
    
    # -------------------------------------------------------------------------
    # Збереження результатів
    # -------------------------------------------------------------------------
    "save_json": True,               # [True/False] Детекції в JSON (COCO format)
    "save_txt": False,               # [True/False] Детекції в txt (YOLO format)
    
    # -------------------------------------------------------------------------
    # ПРИМІТКА: imgsz визначається з MODEL_SIZE (n=384, s=512, m=576, b=560, l=704, xl=700, 2xl=880)
    # -------------------------------------------------------------------------
}


# =============================================================================
# ГОЛОВНА ФУНКЦІЯ
# =============================================================================

def main(
    model_path: str = MODEL_PATH,
    dataset_dir: str = None,
    split: str = None,  
    save_results: bool = True,
    **kwargs
):
    """Запуск валідації."""
    dataset_dir = dataset_dir or str(DATASET_DIR)
    config = {**INFERENCE_CONFIG, **kwargs}
    split = split or config["split"]
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
    print(f"Device: {config['device']}")
    print("=" * 70 + "\n")
    
    # Перевірка датасету
    if not Path(dataset_dir).exists():
        print(f"ПОМИЛКА: Датасет не знайдено: {dataset_dir}")
        return None
    
    # Перевірка моделі
    if model_path and not Path(model_path).exists():
        print(f"УВАГА: Модель не знайдено: {model_path}")
        model_path = None
    
    # Determine resolution based on model size
    resolutions = {'n': 384, 's': 512, 'm': 576, 'b': 560, 'l': 704, 'xl': 700, '2xl': 880}
    imgsz = resolutions.get(MODEL_SIZE, 560)
    print(f"Using resolution: {imgsz} for model size '{MODEL_SIZE}'")

    # Створюємо validator

    validation_dir = PROJECT_DIR / "validation"
    validator = RFDETRValidator(
        model_path=model_path,
        model_size=MODEL_SIZE,
        imgsz=imgsz,
        conf_threshold=config["conf_threshold"],
        iou_threshold=config["iou_threshold"],
        batch_size=config["batch_size"],
        workers=config["workers"],
        device=config["device"],
        save_dir=str(validation_dir),
    )
    
    # Запуск валідації
    results = validator.validate(
        dataset_dir=dataset_dir,
        split=split,
        save_visualizations=config.get("save_visualizations", True),
    )
    
    # Збереження результатів (додаємо model_path та imgsz для JSON)
    if save_results and results:
        results["model_path"] = model_path
        config["imgsz"] = imgsz
        config["model_size"] = MODEL_SIZE
        save_validation_results(results, config)

    # Розширений вивід у консоль (те саме, що в звіті MD)
    if results:
        print_validation_summary(results)

    return results


def print_validation_summary(results: dict) -> None:
    """Виводить у консоль розширену звітність: AP по розмірах, AR, per-class AP by area, шляхи."""
    metrics = results.get("metrics") or {}
    save_dir = Path(results.get("save_dir", PROJECT_DIR / "validation"))
    num_classes = results.get("num_classes", 0)

    print("\n" + "=" * 60)
    print("  РОЗШИРЕНИЙ ПІДСУМОК (те саме, що в validation_report.md)")
    print("=" * 60)

    # AP по розміру об'єкта
    print("\n  AP по розміру об'єкта (загалом)")
    print("  " + "-" * 40)
    print(f"  {'Small (area < 32² px)':<28} {metrics.get('AP_small', 0):.4f}")
    print(f"  {'Medium (32²–96² px)':<28} {metrics.get('AP_medium', 0):.4f}")
    print(f"  {'Large (area > 96² px)':<28} {metrics.get('AP_large', 0):.4f}")

    # AR
    print("\n  Average Recall (AR)")
    print("  " + "-" * 40)
    print(f"  {'AR @ maxDets=1':<28} {metrics.get('AR_maxDets1', 0):.4f}")
    print(f"  {'AR @ maxDets=10':<28} {metrics.get('AR_maxDets10', 0):.4f}")
    print(f"  {'AR @ maxDets=100':<28} {metrics.get('AR_maxDets100', 0):.4f}")
    print(f"  {'AR small':<28} {metrics.get('AR_small', 0):.4f}")
    print(f"  {'AR medium':<28} {metrics.get('AR_medium', 0):.4f}")
    print(f"  {'AR large':<28} {metrics.get('AR_large', 0):.4f}")

    # Per-class AP by area
    ap_by_class = metrics.get("ap_by_class_area") or []
    if ap_by_class:
        print("\n  AP по класах за розміром (Small / Medium / Large)")
        print("  " + "-" * 56)
        print(f"  {'Клас':<20} {'AP small':>10} {'AP medium':>10} {'AP large':>10}")
        print("  " + "-" * 56)
        for row in ap_by_class:
            name = (row.get("class_name") or f"class_{row.get('class_id', 0)}")[:18]
            print(f"  {name:<20} {row.get('AP_small', 0):>10.3f} {row.get('AP_medium', 0):>10.3f} {row.get('AP_large', 0):>10.3f}")

    # Шляхи до файлів
    print("\n  Збережені файли")
    print("  " + "-" * 56)
    print(f"  JSON:   {save_dir / 'validation_results.json'}")
    print(f"  Звіт:   {save_dir / 'validation_report.md'}")
    print(f"  CM:     {save_dir / 'confusion_matrix.png'}")
    print(f"  Криві:  {save_dir / 'BoxPR_curve.png'}, BoxF1_curve.png, BoxP_curve.png, BoxR_curve.png")
    print("=" * 60 + "\n")


def save_validation_results(results: dict, config: dict = None):
    """Збереження результатів у JSON у форматі як в іншому проєкті (YOLO-style)."""
    output_dir = Path(results.get("save_dir", PROJECT_DIR / "validation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = results.get("metrics") or {}
    config = config or INFERENCE_CONFIG

    # class_stats: {"0": {mAP50, mAP50-95, precision, recall, f1}, ...}
    raw_class_stats = metrics.get("class_stats") or {}
    per_class_ap = {x["class_id"]: x for x in (metrics.get("per_class_ap") or [])}
    class_stats_out = {}
    for cid, st in raw_class_stats.items():
        key = str(cid)
        gt, tp, fp, fn = st.get("gt", 0), st.get("tp", 0), st.get("fp", 0), st.get("fn", 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        ap = per_class_ap.get(int(cid) if isinstance(cid, str) and cid.isdigit() else cid, {})
        class_stats_out[key] = {
            "mAP50": ap.get("mAP50", 0.0),
            "mAP50-95": ap.get("mAP50-95", 0.0),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # ap_by_class_area: [{small, medium, large}, ...] (без class_id/class_name)
    ap_by_area = metrics.get("ap_by_class_area") or []
    ap_by_class_area_out = []
    for row in ap_by_area:
        ap_by_class_area_out.append({
            "small": row.get("AP_small", row.get("small", 0.0)),
            "medium": row.get("AP_medium", row.get("medium", 0.0)),
            "large": row.get("AP_large", row.get("large", 0.0)),
        })

    inference_fps = results.get("inference_fps") or 0.0
    inference_latency_ms = (1000.0 / inference_fps) if inference_fps > 0 else 0.0

    payload = {
        "metrics": {
            "mAP50": metrics.get("mAP50", 0.0),
            "mAP50-95": metrics.get("mAP50-95", 0.0),
            "mAP75": metrics.get("mAP75", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1": metrics.get("f1", 0.0),
            "ap_small": metrics.get("AP_small", 0.0),
            "ap_medium": metrics.get("AP_medium", 0.0),
            "ap_large": metrics.get("AP_large", 0.0),
            "ar_maxdets1": metrics.get("AR_maxDets1", 0.0),
            "ar_maxdets10": metrics.get("AR_maxDets10", 0.0),
            "ar_maxdets100": metrics.get("AR_maxDets100", 0.0),
            "ar_small": metrics.get("AR_small", 0.0),
            "ar_medium": metrics.get("AR_medium", 0.0),
            "ar_large": metrics.get("AR_large", 0.0),
            "class_stats": class_stats_out,
            "ap_by_class_area": ap_by_class_area_out,
        },
        "num_classes": results.get("num_classes", 0),
        "classes": results.get("classes", []),
        "inference_fps": inference_fps,
        "inference_latency_ms": inference_latency_ms,
        "split": results.get("split", "valid"),
        "dataset_dir": results.get("dataset_dir", ""),
        "validation_date": datetime.now().isoformat(),
        "inference_config": config,
        "model_path": results.get("model_path") or "",
    }
    results_path = output_dir / "validation_results.json"

    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=convert)

    print(f"\n[Results] Результати збережено: {results_path}")



if __name__ == "__main__":
    main()
