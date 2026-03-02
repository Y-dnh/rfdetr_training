#!/usr/bin/env python3
"""
Превью аугментацій RF-DETR
===========================
Генерує приклади аугментованих зображень з повним тренувальним пайплайном:
Mosaic (legacy) → RandomPerspective → MixUp → CutMix → Albumentations → LetterBox → ToTensor → Normalize.

Два конфіги (як у train.py):
  1. AUGMENTATION_CONFIG — наші параметри: mosaic, mixup, cutmix, perspective, erasing, imgsz тощо.
  2. ALBUMENTATION_CONFIG — список трансформ albumentations (A.*): кольори, flips, blur, CoarseDropout тощо.

Призначення: візуальна перевірка збігу bbox з аугментованими зображеннями.

Usage:
    python tests/preview_augmentations.py

Налаштуйте AUGMENTATION_CONFIG та ALBUMENTATION_CONFIG нижче.
"""

import sys
import os
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

import albumentations as A
from rfdetr.training import (
    setup_seed,
    AugmentationConfig,
    build_dataset,
)
from rfdetr.training.visualizations.batch_visualizer import (
    denormalize_image,
    draw_boxes,
)


# =============================================================================
# ШЛЯХИ ТА ЗАГАЛЬНІ НАЛАШТУВАННЯ
# =============================================================================

DATASET_DIR = Path("tests/test_dataset")
OUTPUT_DIR = Path(__file__).parent / "augmentation_preview"
NUM_EXAMPLES = 500
SEED = 42
SPLIT = "train"

# =============================================================================
# Ті самі параметри, що в train.py: тепловізія (white hot), малі об'єкти, 1024
# =============================================================================
ALBUMENTATION_CONFIG = [
    A.Blur(blur_limit=5, p=0.2),
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.35),
    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.65),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=5, val_shift_limit=35, p=0.4),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.CoarseDropout(num_holes=6, max_h_size=20, max_w_size=20, fill_value=128, p=0.35),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=5, p=0.4),
]

AUGMENTATION_CONFIG = AugmentationConfig(
    imgsz=1024,
    mosaic=0.5,
    close_mosaic=5,
    use_albumentations_mosaic=False,
    mixup=0.1,
    cutmix=0.1,
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
if not ALBUMENTATION_CONFIG:
    import sys
    print("ПОМИЛКА: get_default_albu_config() повернув порожній список (перевірте: pip install albumentations).", file=sys.stderr)
    sys.exit(1)


def _resolve_transform(t):
    """Повертає реальний інстанс трансформу (викликає тільки фабрики — function/lambda; інстанси A.* не викликати)."""
    if isinstance(t, dict):
        return None  # не трансформ
    if type(t).__name__ == "function":
        try:
            return t()
        except Exception:
            pass
    return t


def _draw_augmentation_label(img_np, text: str, max_width: int = 800) -> np.ndarray:
    """Малює текст аугментацій зверху зображення (чорна смуга + білий текст)."""
    try:
        import cv2
    except ImportError:
        return img_np
    if not text or not text.strip():
        return img_np
    h, w = img_np.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(w, h) / 800)
    thickness = max(1, int(font_scale * 2))
    # Розбити довгий текст на рядки
    words = text.replace(" | ", " ").replace(",", " ").split()
    lines = []
    current = ""
    for word in words:
        if not current:
            current = word
        else:
            test = current + " " + word
            (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
            if tw <= max_width:
                current = test
            else:
                lines.append(current)
                current = word
    if current:
        lines.append(current)
    line_height = int(30 * font_scale)
    pad = int(10 * font_scale)
    label_h = len(lines) * line_height + 2 * pad
    overlay = img_np.copy()
    cv2.rectangle(overlay, (0, 0), (w, label_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img_np, 0.3, 0, img_np)
    for i, line in enumerate(lines):
        y = pad + (i + 1) * line_height - 2
        cv2.putText(img_np, line, (pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img_np


def _albu_transform_names(transforms_list):
    """Повертає список назв класів трансформ з ALBUMENTATION_CONFIG (для виводу)."""
    names = []
    for t in transforms_list or []:
        t = _resolve_transform(t)
        if t is None:
            continue
        name = getattr(getattr(t, "__class__", None), "__name__", None)
        if name and name not in ("function", "type", "dict"):
            names.append(name)
    return names


def _albu_active_p(transforms_list):
    """Повертає список (name, p) для трансформ з p > 0 (активні в прев'ю/тренуванні)."""
    out = []
    for t in transforms_list or []:
        t = _resolve_transform(t)
        if t is None:
            continue
        name = getattr(getattr(t, "__class__", None), "__name__", None)
        if not name or name in ("function", "type", "dict"):
            continue
        p = getattr(t, "p", 0.0)
        if p > 0:
            out.append((name, p))
    return out


def _print_config_summary():
    """Друкує вибір параметрів обох конфігів."""
    cfg = AUGMENTATION_CONFIG
    albu_names = _albu_transform_names(ALBUMENTATION_CONFIG)

    print("\n" + "=" * 60)
    print("  ВИБІР ПАРАМЕТРІВ КОНФІГІВ")
    print("=" * 60)
    print("\n  AUGMENTATION_CONFIG (пайплайн):")
    print(f"    imgsz={cfg.imgsz}  mosaic={cfg.mosaic}  close_mosaic={cfg.close_mosaic}")
    print(f"    mixup={cfg.mixup}  cutmix={cfg.cutmix}")
    print(f"    (геометрія/колір/flip — у ALBUMENTATION_CONFIG)")

    print("\n  ALBUMENTATION_CONFIG (трансформи, кількість={}):".format(len(albu_names)))
    if albu_names:
        for i in range(0, len(albu_names), 8):
            chunk = albu_names[i : i + 8]
            print("    " + ", ".join(chunk))
    else:
        print("    (порожньо або всі відфільтровані)")
    active = _albu_active_p(ALBUMENTATION_CONFIG)
    if active:
        print("  Активні Albumentations (p>0), будуть помітні на зображеннях:")
        print("    " + ", ".join(f"{n}(p={p})" for n, p in active))
    print("=" * 60 + "\n")


def cxcywh_norm_to_xyxy_pixel(boxes: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    """Конвертує бокси з нормалізованого cxcywh в xyxy (пікселі)."""
    if len(boxes) == 0:
        return boxes
    xyxy = boxes.clone().float()
    xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * img_w
    xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_h
    xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * img_w
    xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * img_h
    return xyxy


def main():
    """Генерація прикладів аугментацій."""
    print("=" * 60)
    print("  RF-DETR Augmentation Preview")
    print("=" * 60)

    if not DATASET_DIR.exists():
        print(f"\n[ПОМИЛКА] Датасет не знайдено: {DATASET_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nДатасет:   {DATASET_DIR}")
    print(f"Вихід:     {OUTPUT_DIR.resolve()}")
    print(f"Прикладів: {NUM_EXAMPLES}  Seed: {SEED}  Split: {SPLIT}")

    _print_config_summary()

    if SEED is not None:
        setup_seed(SEED)

    print("Завантаження датасету...")
    dataset = build_dataset(
        DATASET_DIR,
        split=SPLIT,
        augmentation_config=AUGMENTATION_CONFIG,
        albumentation_transforms=ALBUMENTATION_CONFIG,
        log_augmentations=True,
    )

    num_images = len(dataset)
    class_names = dataset.class_names
    print(f"Завантажено: {num_images} зображень, {dataset.num_classes} класів — {class_names}")

    if num_images == 0:
        print("\n[ПОМИЛКА] Датасет порожній.")
        sys.exit(1)

    print(f"\nГенерація {NUM_EXAMPLES} прикладів...")
    print("-" * 60)

    indices = [random.randint(0, num_images - 1) for _ in range(NUM_EXAMPLES)]
    saved_count = 0

    for i, idx in enumerate(indices):
        try:
            result = dataset[idx]
            if len(result) == 3:
                image_tensor, target, aug_log = result
            else:
                image_tensor, target = result
                aug_log = None

            img_np = denormalize_image(image_tensor)
            img_h, img_w = img_np.shape[:2]
            boxes = target.get("boxes", torch.zeros((0, 4)))
            labels = target.get("labels", torch.zeros(0, dtype=torch.int64))

            if len(boxes) > 0 and boxes.max() <= 1.0:
                boxes_xyxy = cxcywh_norm_to_xyxy_pixel(boxes, img_h, img_w)
            else:
                boxes_xyxy = boxes

            img_with_boxes = draw_boxes(
                img_np,
                boxes_xyxy,
                labels,
                class_names=class_names,
                show_labels=True,
                show_conf=False,
                line_width=2,
                font_size=14,
            )

            aug_parts = []
            if aug_log and "augmentations" in aug_log:
                for a in aug_log["augmentations"]:
                    if not a.get("applied"):
                        continue
                    name = a["name"]
                    params = a.get("parameters") or {}
                    albu_names = params.get("applied_names")
                    if name == "Albumentations" and albu_names:
                        aug_parts.append("Albu: " + ", ".join(albu_names))
                    else:
                        aug_parts.append(name)
            aug_info = " | ".join(aug_parts) if aug_parts else ""
            if aug_info:
                img_with_boxes = _draw_augmentation_label(img_with_boxes, aug_info)

            original_name = Path(dataset.get_filename(idx)).stem
            output_filename = f"{i:04d}_{original_name}.jpg"
            output_path = OUTPUT_DIR / output_filename
            Image.fromarray(img_with_boxes).save(output_path, quality=95)
            saved_count += 1

            n_boxes = len(boxes)
            aug_info_log = f" | {aug_info}" if aug_info else ""
            print(f"  [{i+1:>{len(str(NUM_EXAMPLES))}}/{NUM_EXAMPLES}] {output_filename} | {n_boxes} boxes{aug_info_log}")

        except Exception as e:
            print(f"  [{i+1:>{len(str(NUM_EXAMPLES))}}/{NUM_EXAMPLES}] ПОМИЛКА (idx={idx}): {e}")

    print("-" * 60)
    print(f"\nЗбережено {saved_count}/{NUM_EXAMPLES} зображень у {OUTPUT_DIR.resolve()}")
    _print_config_summary()


if __name__ == "__main__":
    main()
