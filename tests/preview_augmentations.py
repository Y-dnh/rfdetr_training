#!/usr/bin/env python3
"""
Превью аугментацій RF-DETR
===========================
Генерує приклади аугментованих зображень із повним тренувальним пайплайном
(Mosaic, MixUp, CutMix, HSV, Flip, Erasing і т.д.) та зберігає їх у папку.

Кожне зображення зберігається з намальованими bounding box'ами та підписами класів,
щоб візуально оцінити якість та різноманітність аугментацій.

Usage:
    python tests/preview_augmentations.py

Налаштуйте конфігурацію нижче перед запуском.
"""

import sys
import os
import random
from pathlib import Path

# Додати корінь проекту до шляху
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

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
# КОНФІГУРАЦІЯ
# =============================================================================

# Шлях до датасету (COCO-формат з train/valid/test)
DATASET_DIR = Path("tests/test_dataset")

# Вихідна папка для збережених прикладів (всередині tests/)
OUTPUT_DIR = Path(__file__).parent / "augmentation_preview"

# Кількість прикладів для генерації
NUM_EXAMPLES = 200

# Seed для відтворюваності (None = випадковий кожного разу)
SEED = 42

# Split датасету для генерації ('train' для повних аугментацій)
SPLIT = "train"


# =============================================================================
# КОНФІГУРАЦІЯ АУГМЕНТАЦІЙ
# =============================================================================
# Конфіг розділений на 2 частини:
#   1. ОСНОВНІ — ймовірності та сила аугментацій (що включити і як часто)
#   2. ТОНКІ НАЛАШТУВАННЯ — діапазони, пороги, ліміти (зазвичай змінювати не потрібно)
# =============================================================================
AUGMENTATION_CONFIG = AugmentationConfig(

    # =========================================================================
    # ОСНОВНІ ПАРАМЕТРИ
    # =========================================================================

    # Розмір зображення (Nano=384, Small=512, Medium=576, Base=560, Large=704)
    imgsz=576,

    # --- Композитні (об'єднання кількох зображень) ---
    mosaic=0.5,                      # Mosaic: 4 зображення в одне (0-1)
    close_mosaic=5,                  # Вимкнути Mosaic в останніх N епохах
    mixup=0.1,                       # MixUp: альфа-блендинг 2 зображень (0-1)
    cutmix=0.2,                      # CutMix: вирізання регіону з іншого зображення (0-1)

    # --- Колірні ---
    hsv_h=0.0,                       # Зміна відтінку Hue (0-1)
    hsv_s=0.0,                       # Зміна насиченості Saturation (0-1)
    hsv_v=0.4,                       # Зміна яскравості Value (0-1)
    brightness=0.2,                  # Ймовірність зміни яскравості (0-1)
    contrast=0.2,                    # Ймовірність зміни контрасту (0-1)
    blur=0.1,                        # Ймовірність Gaussian blur (0-1)
    noise=0.2,                       # Ймовірність шуму (0-1)
    noise_type='gaussian_mono',      # Тип: 'gaussian_mono' (IR), 'gaussian_rgb' (RGB), 'salt_pepper'

    # --- Геометричні ---
    degrees=10.0,                    # Максимальний поворот (+/- градуси)
    translate=0.1,                   # Максимальний зсув (частка від розміру зображення)
    scale=0.5,                       # Масштаб (+/- scale)
    shear=0.0,                       # Зсув перспективи (градуси)
    perspective=0.0,                 # Перспективна деформація (0-0.001)

    # --- Відзеркалення ---
    fliplr=0.5,                      # Горизонтальний flip (0-1)
    flipud=0.0,                      # Вертикальний flip (0-1)

    # --- Random Erasing (box-aware видалення регіонів) ---
    erasing=0.25,                    # Ймовірність erasing (0-1)
    erasing_value=128,               # Заповнення: 0=чорний, 128=сірий, 'random'=шум

    # =========================================================================
    # ТОНКІ НАЛАШТУВАННЯ (зазвичай змінювати не потрібно)
    # =========================================================================

    # Mosaic
    mosaic_scale=(0.5, 1.5),         # Діапазон центру мозаїки (частка від imgsz)
    mosaic_min_box_size=2,           # Мін. розмір боксу після мозаїки (пікселі)

    # MixUp
    mixup_alpha=32.0,                # Beta-розподіл alpha (більше = слабше змішування)

    # CutMix
    cutmix_alpha=1.0,                # Beta-розподіл alpha (1.0 = рівномірний розмір вирізу)
    cutmix_min_visible=0.3,          # Мін. видима частина боксу (0-1)
    cutmix_min_box_size=10,          # Мін. розмір боксу після обрізання (пікселі)
    cutmix_overlap_thresh=0.1,       # Поріг перекриття нижче якого бокс не змінюється (0-1)

    # Яскравість / Контраст / Blur
    brightness_range=(0.5, 1.5),     # Діапазон множника яскравості
    contrast_range=(0.5, 1.5),       # Діапазон множника контрасту
    blur_kernel_range=(3, 7),        # Розмір ядра blur (непарні числа)

    # Шум
    noise_strength=(5.0, 30.0),      # Сила Gaussian шуму (std dev min, max)
    salt_pepper_amount=0.02,         # Частка пікселів для salt-and-pepper (0-1)

    # Random Erasing
    erasing_min_scale=0.02,          # Мін. частка площі зображення (0-1)
    erasing_max_scale=0.33,          # Макс. частка площі зображення (0-1)
    erasing_ratio=(0.3, 3.3),       # Діапазон aspect ratio вирізаного регіону
    erasing_min_visible=0.5,         # Мін. видима частина боксу (0-1)
    erasing_min_box_size=20,         # Мін. розмір боксу після erasing (пікселі)

    # LetterBox / Padding
    letterbox_color=(114, 114, 114), # Колір padding (R,G,B). (0,0,0) для IR камер
)


# =============================================================================
# ОСНОВНИЙ СКРИПТ
# =============================================================================

def cxcywh_norm_to_xyxy_pixel(boxes: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    """
    Конвертує бокси з нормалізованого cxcywh формату в xyxy (пікселі).

    Args:
        boxes: Тензор боксів (N, 4) у форматі [cx, cy, w, h], значення 0..1.
        img_h: Висота зображення в пікселях.
        img_w: Ширина зображення в пікселях.

    Returns:
        Тензор боксів (N, 4) у форматі [x1, y1, x2, y2] в пікселях.
    """
    if len(boxes) == 0:
        return boxes

    xyxy = boxes.clone().float()
    xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * img_w  # x1
    xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_h  # y1
    xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * img_w  # x2
    xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * img_h  # y2
    return xyxy


def main():
    """Генерація прикладів аугментацій."""
    print("=" * 60)
    print("  RF-DETR Augmentation Preview")
    print("=" * 60)

    # Перевірка датасету
    if not DATASET_DIR.exists():
        print(f"\n[ПОМИЛКА] Датасет не знайдено: {DATASET_DIR}")
        print("Вкажіть правильний шлях у DATASET_DIR.")
        sys.exit(1)

    # Створити вихідну папку
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nДатасет:    {DATASET_DIR}")
    print(f"Вихід:      {OUTPUT_DIR.resolve()}")
    print(f"Кількість:  {NUM_EXAMPLES}")
    print(f"Seed:       {SEED}")
    print(f"imgsz:      {AUGMENTATION_CONFIG.imgsz}")

    # Встановити seed
    if SEED is not None:
        setup_seed(SEED)

    # Побудувати датасет з аугментаціями (як при тренуванні)
    print(f"\nЗавантаження датасету (split={SPLIT})...")
    dataset = build_dataset(
        DATASET_DIR,
        split=SPLIT,
        augmentation_config=AUGMENTATION_CONFIG,
        log_augmentations=True,
    )

    num_images = len(dataset)
    class_names = dataset.class_names
    print(f"Завантажено: {num_images} зображень, {dataset.num_classes} класів")
    print(f"Класи: {class_names}")

    if num_images == 0:
        print("\n[ПОМИЛКА] Датасет порожній!")
        sys.exit(1)

    # Генерація прикладів
    print(f"\nГенерація {NUM_EXAMPLES} аугментованих прикладів...")
    print("-" * 60)

    # Випадкові індекси (з повторенням якщо NUM_EXAMPLES > num_images)
    indices = [random.randint(0, num_images - 1) for _ in range(NUM_EXAMPLES)]

    saved_count = 0
    for i, idx in enumerate(indices):
        try:
            # Отримати аугментоване зображення
            result = dataset[idx]
            if len(result) == 3:
                image_tensor, target, aug_log = result
            else:
                image_tensor, target = result
                aug_log = None

            # Денормалізувати зображення (ImageNet mean/std -> uint8 RGB)
            img_np = denormalize_image(image_tensor)
            img_h, img_w = img_np.shape[:2]

            # Конвертувати бокси з нормалізованого cxcywh в xyxy (пікселі)
            boxes = target.get('boxes', torch.zeros((0, 4)))
            labels = target.get('labels', torch.zeros(0, dtype=torch.int64))

            if len(boxes) > 0 and boxes.max() <= 1.0:
                boxes_xyxy = cxcywh_norm_to_xyxy_pixel(boxes, img_h, img_w)
            else:
                boxes_xyxy = boxes

            # Намалювати бокси
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

            # Сформувати ім'я файлу
            original_name = Path(dataset.get_filename(idx)).stem
            output_filename = f"{i:04d}_{original_name}.jpg"
            output_path = OUTPUT_DIR / output_filename

            # Зберегти
            Image.fromarray(img_with_boxes).save(output_path, quality=95)
            saved_count += 1

            # Логування застосованих аугментацій
            aug_info = ""
            if aug_log and 'augmentations' in aug_log:
                applied = [a['name'] for a in aug_log['augmentations'] if a.get('applied')]
                if applied:
                    aug_info = f" | Augs: {', '.join(applied)}"

            n_boxes = len(boxes)
            print(f"  [{i+1:>{len(str(NUM_EXAMPLES))}}/{NUM_EXAMPLES}] "
                  f"{output_filename} | {n_boxes} boxes{aug_info}")

        except Exception as e:
            print(f"  [{i+1:>{len(str(NUM_EXAMPLES))}}/{NUM_EXAMPLES}] "
                  f"ПОМИЛКА (idx={idx}): {e}")

    # Підсумок
    print("-" * 60)
    print(f"\nЗбережено {saved_count}/{NUM_EXAMPLES} зображень")
    print(f"Папка: {OUTPUT_DIR.resolve()}")

    # Інформація про конфігурацію аугментацій
    cfg = AUGMENTATION_CONFIG
    print("\nКонфігурація аугментацій:")
    print("  --- Основні ---")
    print(f"  Mosaic:      {cfg.mosaic}  (close_mosaic={cfg.close_mosaic})")
    print(f"  MixUp:       {cfg.mixup}")
    print(f"  CutMix:      {cfg.cutmix}")
    print(f"  HSV:         h={cfg.hsv_h}, s={cfg.hsv_s}, v={cfg.hsv_v}")
    print(f"  Brightness:  {cfg.brightness}  (range={cfg.brightness_range})")
    print(f"  Contrast:    {cfg.contrast}  (range={cfg.contrast_range})")
    print(f"  Blur:        {cfg.blur}  (kernel={cfg.blur_kernel_range})")
    print(f"  Noise:       {cfg.noise}  (type={cfg.noise_type}, "
          f"strength={cfg.noise_strength})")
    print(f"  Degrees:     {cfg.degrees}")
    print(f"  Translate:   {cfg.translate}")
    print(f"  Scale:       {cfg.scale}")
    print(f"  FlipLR:      {cfg.fliplr}")
    print(f"  FlipUD:      {cfg.flipud}")
    print(f"  Erasing:     {cfg.erasing}  (value={cfg.erasing_value}, "
          f"scale={cfg.erasing_min_scale}-{cfg.erasing_max_scale})")
    print(f"  imgsz:       {cfg.imgsz}")


if __name__ == '__main__':
    main()
