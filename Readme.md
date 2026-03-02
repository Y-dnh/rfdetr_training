<p align="center">
  <img src="https://raw.githubusercontent.com/roboflow/rf-detr/main/docs/assets/og-card.png" alt="RF-DETR" width="100%">
</p>

<h1 align="center">RF-DETR Training Pipeline</h1>

<p align="center">
  <b>Готовий до продакшену фреймворк тренування з аугментаціями в стилі Ultralytics та Albumentations</b>
</p>

<p align="center">
  <a href="#ключові-можливості">Можливості</a> &bull;
  <a href="#швидкий-старт">Швидкий старт</a> &bull;
  <a href="#варіанти-моделей">Моделі</a> &bull;
  <a href="#архітектура-проєкту">Архітектура</a> &bull;
  <a href="#аугментації">Аугментації</a> &bull;
  <a href="#конфігурація">Конфігурація</a> &bull;
  <a href="#експорт-onnx">Експорт ONNX</a> &bull;
  <a href="#результати-тренування">Результати</a> &bull;
  <a href="#api">API</a>
</p>

---

> **Розширений форк [RF-DETR від Roboflow](https://github.com/roboflow/rf-detr)** — детектор об'єктів у реальному часі з повним пайплайном тренування, типобезпечними конфігами та аугментаціями на базі **Albumentations** і власних композитних кроків (Mosaic, MixUp, CutMix).

---

## Ключові можливості

| Можливість | Оригінальний RF-DETR | Цей форк |
|------------|----------------------|----------|
| **Аугментації** | Базові (resize, flip) | **Два рівні:** власний пайплайн (Mosaic, MixUp, CutMix, LetterBox) + **Albumentations** (колір, flip, геометрія, blur, noise, CoarseDropout тощо). Повний список трансформ з [explore.albumentations.ai](https://explore.albumentations.ai/). |
| **Конфігурація** | argparse CLI | Типобезпечні dataclass: `ModelConfig`, `TrainingConfig`, `AugmentationConfig`; окремо **ALBUMENTATION_CONFIG** (список A.* трансформ). |
| **Mosaic** | Немає | Вибір: **наша** Mosaic (mosaic.py) або **A.Mosaic** (albumentations) через `use_albumentations_mosaic`. |
| **Візуалізації** | TensorBoard | Батчі train/val, confusion matrix, криві PR/F1/P/R, графіки метрик, превью аугментацій. |
| **Валідація** | COCO eval | Розширено: markdown-звіти, статистика по класах, детальний аналіз. |
| **Датасет** | CocoDetection | `RFDETRDataset` з інтегрованим пайплайном аугментацій та підтримкою `albumentation_transforms`. |
| **Точка входу** | `Model.train()` | Окремі класи `RFDETRTrainer` та `RFDETRValidator`. |
| **Експорт ONNX** | Тільки CLI | Автовизначення моделі з checkpoint, конфіг в одному місці. |
| **Box-aware аугментації** | Немає | CutMix та CoarseDropout з урахуванням bbox; фільтрація за видимістю та мін. розміром. |
| **PyTorch 2.6+** | Проблеми з ONNX | У `export.py` використовується `dynamo=False` для сумісності з legacy ONNX exporter. |

---

## Швидкий старт

### Встановлення

```bash
git clone <repo-url>
cd rfdetr_training

# Залежності (включає albumentations)
pip install -e .
```

Рекомендовано використовувати conda-середовище з усіма залежностями (наприклад `conda activate rfdetr_training_env`).

### Тренування

```bash
python train.py
```

Налаштування — на початку `train.py`:

```python
DATASET_DIR = Path("path/to/your/dataset")
MODEL_CONFIG = ModelConfig(model_size="m")   # num_classes з COCO JSON
TRAINING_CONFIG = TrainingConfig(epochs=100, batch_size=16)
AUGMENTATION_CONFIG = AugmentationConfig(mosaic=0.5, mixup=0.1, cutmix=0.2)
ALBUMENTATION_CONFIG = get_default_albu_config()  # або свій список A.*
```

### Валідація

```bash
python val.py
```

У `val.py` вказати шлях до моделі, розмір, датасет тощо.

### Експорт ONNX

```bash
python export_onnx.py
```

Архітектура, роздільність і кількість класів **автоматично визначаються** з checkpoint.

### Формат датасету

Підтримуються **COCO** (експорт Roboflow) та **YOLO** з автовизначенням:

```
dataset/
├── train/
│   ├── image1.jpg
│   └── _annotations.coco.json
├── valid/
│   └── _annotations.coco.json
└── test/
    └── _annotations.coco.json
```

---

## Варіанти моделей

### Детекція

| Розмір | Код | Роздільність | Параметри | COCO AP50:95 | Латентність (ms) | Ліцензія |
|--------|-----|--------------|-----------|--------------|------------------|----------|
| Nano | `n` | 384×384 | 30.5M | 48.4 | 2.3 | Apache-2.0 |
| Small | `s` | 512×512 | 32.1M | 53.0 | 3.5 | Apache-2.0 |
| Medium | `m` | 576×576 | 33.7M | 54.7 | 4.4 | Apache-2.0 |
| Base | `b` | 560×560 | 29M | — | — | Apache-2.0 |
| Large | `l` | 704×704 | 33.9M | 56.5 | 6.8 | Apache-2.0 |
| XLarge | `xl` | 700×700 | 126.4M | 58.6 | 11.5 | PML-1.0 |
| 2XLarge | `2xl` | 880×880 | 126.9M | 60.1 | 17.2 | PML-1.0 |

Роздільність зображення **жорстко прив’язана** до архітектури; при тренуванні `imgsz` підставляється з моделі. Для XLarge/2XLarge потрібно `accept_platform_license=True`.

---

## Архітектура проєкту

```
rfdetr_training/
├── train.py                          # Вхідна точка тренування
├── val.py                            # Валідація
├── export_onnx.py                    # Експорт ONNX (автовизначення з checkpoint)
│
└── rfdetr/
    ├── training/                     # Власний модуль тренування
    │   ├── trainer.py                # RFDETRTrainer — повний цикл тренування
    │   ├── validator.py              # RFDETRValidator — валідація
    │   ├── dataset.py                # RFDETRDataset з пайплайном аугментацій
    │   ├── albumentation_config.py   # get_default_albu_config() — повний список A.* трансформ
    │   │
    │   ├── augmentations/
    │   │   ├── pipeline.py           # AugmentationPipeline: Mosaic → MixUp → CutMix → Albumentations → LetterBox → ToTensor → Normalize
    │   │   ├── albumentations_wrapper.py  # AlbumentationsWrapper — застосування A.Compose до (image, target)
    │   │   ├── mosaic.py             # Mosaic 2×2, Mosaic9 (legacy; опційно A.Mosaic)
    │   │   ├── mixup.py              # MixUp та CutMix (box-aware)
    │   │   ├── geometric.py         # LetterBox, RandomPerspective (не використовується в пайплайні; геометрія через albu)
    │   │   ├── erasing.py            # RandomErasing (legacy; заміна — A.CoarseDropout в albu)
    │   │   ├── color.py              # Legacy color-трансформи (не в пайплайні)
    │   │   └── base.py               # BaseTransform, ToTensor, Normalize
    │   │
    │   ├── visualizations/           # Візуалізації в стилі YOLO
    │   │   ├── batch_visualizer.py
    │   │   ├── confusion_matrix.py
    │   │   ├── curves.py
    │   │   ├── metrics_plotter.py
    │   │   └── labels_analyzer.py
    │   │
    │   ├── logging/
    │   │   └── augmentation_logger.py
    │   │
    │   └── utils/
    │       ├── config.py             # ModelConfig, TrainingConfig, AugmentationConfig, ExportConfig
    │       └── seed.py
    │
    ├── deploy/
    ├── platform/
    ├── config.py
    ├── main.py
    ├── detr.py
    ├── models/
    └── datasets/
```

---

## Аугментації

### Два рівні конфігурації

1. **AUGMENTATION_CONFIG** (`AugmentationConfig`) — керує лише тим, що реалізовано в пайплайні власним кодом:
   - **imgsz** — роздільність (при тренуванні підставляється з моделі).
   - **mosaic**, **close_mosaic**, **use_albumentations_mosaic** — Mosaic: наша (mosaic.py) або A.Mosaic.
   - **mixup**, **cutmix** — ймовірності MixUp/CutMix.
   - Тонкі параметри: **mosaic_scale**, **mosaic_min_box_size**, **mixup_alpha**, **cutmix_alpha**, **cutmix_min_visible**, **cutmix_min_box_size**, **cutmix_overlap_thresh**, **letterbox_color**.

2. **ALBUMENTATION_CONFIG** — список екземплярів трансформ **Albumentations** (A.*). Відповідає за:
   - колір (HueSaturationValue, RandomBrightnessContrast, CLAHE тощо);
   - flip (HorizontalFlip, VerticalFlip);
   - геометрію (ShiftScaleRotate, Perspective, Affine, Rotate тощо);
   - blur/noise (GaussianBlur, GaussNoise, CoarseDropout тощо).

За замовчуванням використовується `get_default_albu_config()` з модуля `rfdetr.training.albumentation_config` — повний набір Image-Only та Dual трансформ з [explore.albumentations.ai](https://explore.albumentations.ai/). Можна підставити власний список A.* у `train.py`.

### Порядок пайплайну (train)

```
Вхідне зображення
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Mosaic                                                               │
│     • use_albumentations_mosaic=False: наша Mosaic (4 зображення з       │
│       датасету, 2×2 сітка, випадковий центр, обріз до imgsz×imgsz).       │
│     • use_albumentations_mosaic=True: наша не викликається; перед albu   │
│       в target додається mosaic_metadata (3 додаткові зображення);       │
│       A.Mosaic у ALBUMENTATION_CONFIG має бути з p>0.                    │
├─────────────────────────────────────────────────────────────────────────┤
│  2. MixUp          │ Альфа-блендинг з другим зображенням (наш mixup.py)   │
├────────────────────┼────────────────────────────────────────────────────┤
│  3. CutMix         │ Вставка регіону з другого зображення (box-aware)    │
├────────────────────┼────────────────────────────────────────────────────┤
│  4. Albumentations │ Один A.Compose з ALBUMENTATION_CONFIG: колір, flip,  │
│                    │ геометрія (ShiftScaleRotate, Perspective), blur,     │
│                    │ noise, CoarseDropout тощо; при use_albumentations_  │
│                    │ mosaic=True також A.Mosaic (якщо є в списку).      │
├────────────────────┼────────────────────────────────────────────────────┤
│  5. LetterBox      │ Resize з падінгом до imgsz (наш geometric.py)        │
├────────────────────┼────────────────────────────────────────────────────┤
│  6. ToTensor       │ Перетворення в тензор                                │
├────────────────────┼────────────────────────────────────────────────────┤
│  7. Normalize     │ ImageNet mean/std (наш base.py)                      │
└────────────────────┴────────────────────────────────────────────────────┘
    │
    ▼
Вхід моделі (тензор, imgsz залежить від розміру моделі)
```

### Mosaic: наша vs A.Mosaic

| Аспект | Наша (mosaic.py) | A.Mosaic (albumentations) |
|--------|-------------------|---------------------------|
| Джерело зображень | Пайплайн сам семплює 4 з датасету через `get_raw_item`. | Очікує `mosaic_metadata` у вхідному словнику (список dict з `image`, `bboxes`, `class_labels`). Пайплайн при `use_albumentations_mosaic=True` сам збирає metadata з датасету. |
| Параметри | imgsz, mosaic_scale, mosaic_min_box_size, fill_color, p. | grid_yx, target_size, cell_shape, center_range, fit_mode, p (див. [документацію A.Mosaic](https://explore.albumentations.ai/transform/Mosaic/docs)). |
| Вибір | За замовчуванням (`use_albumentations_mosaic=False`). | Встановити `use_albumentations_mosaic=True` і в ALBUMENTATION_CONFIG мати A.Mosaic з p>0. |

Щоб повернутися до нашої Mosaic після тестування A.Mosaic — достатньо встановити `use_albumentations_mosaic=False`.

### Деталі по кроках

- **Mosaic (наша)** — 4 випадкові зображення з датасету в сітку 2×2 навколо випадкового центру з діапазону `mosaic_scale`; обріз до центрального квадрата imgsz×imgsz. `close_mosaic` вимикає Mosaic в останніх N епохах.

- **MixUp** — `output = α·img1 + (1−α)·img2`, α ~ Beta(alpha, alpha). Більше `mixup_alpha` — слабше змішування.

- **CutMix (box-aware)** — вирізаний прямокутник з одного зображення вставляється в інше; збереження bbox за умови видимості ≥ `cutmix_min_visible`, мін. розмір боксу `cutmix_min_box_size`, поріг перекриття `cutmix_overlap_thresh`.

- **Albumentations** — усі кольорові, геометричні та інші трансформи зі списку (HueSaturationValue, HorizontalFlip, ShiftScaleRotate, Perspective, GaussianBlur, CoarseDropout тощо). Bbox передаються в pascal_voc і оновлюються після spatial-трансформ.

- **LetterBox** — зміна розміру з падінгом до `imgsz`, колір падінгу `letterbox_color` (наприклад (114,114,114) або (0,0,0) для IR).

### Превью аугментацій

Згенерувати приклади аугментованих зображень з намальованими bbox:

```bash
python tests/preview_augmentations.py
```

Конфігурація — на початку файлу (AUGMENTATION_CONFIG та ALBUMENTATION_CONFIG). Результат зберігається в `tests/augmentation_preview/`.

---

## Конфігурація

Усі скрипти використовують **конфіг на початку файлу** (без argparse).

### ModelConfig

```python
ModelConfig(
    model_size="m",              # n, s, m, b, l, xl, 2xl
    pretrained_weights=None,     # шлях до .pt або None (HuggingFace)
    freeze_encoder=False,
    freeze_encoder_epochs=0,
    accept_platform_license=True,  # потрібно для xl/2xl
)
```

`num_classes` визначається автоматично з COCO-анотацій датасету.

### TrainingConfig

```python
TrainingConfig(
    project="runs/training",
    name="exp",
    epochs=100,
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-4,
    scheduler="cosine",
    warmup_epochs=5,
    gradient_accumulation=1,
    grad_clip=0.1,
    val_period=1,
    save_period=-1,        # -1 = тільки best/last
    early_stopping=50,
    device="cuda",
    workers=8,
    vis_batches=3,
)
```

### AugmentationConfig

Керує лише пайплайн-кроками (Mosaic, MixUp, CutMix, LetterBox). Колір, flip, геометрія, blur, noise, erasing налаштовуються в **ALBUMENTATION_CONFIG**.

```python
AugmentationConfig(
    imgsz=640,                        # при тренуванні підставляється з моделі
    mosaic=1.0,
    close_mosaic=10,
    use_albumentations_mosaic=False,  # True = A.Mosaic (потрібен p>0 в ALBUMENTATION_CONFIG)
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
)
```

### ALBUMENTATION_CONFIG

Список трансформ Albumentations (стиль Ultralytics [custom albumentations](https://docs.ultralytics.com/guides/yolo-data-augmentation/#custom-albumentations-transforms-augmentations)). За замовчуванням:

```python
from rfdetr.training import get_default_albu_config
ALBUMENTATION_CONFIG = get_default_albu_config()
```

Кастомний приклад:

```python
import albumentations as A
ALBUMENTATION_CONFIG = [
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.1),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=10, p=0.5),
    A.CoarseDropout(num_holes_range=(4, 12), hole_height_range=(16, 48), hole_width_range=(16, 48), fill=128, p=0.3),
]
```

Повний перелік трансформ (Image-Only та Dual) з дефолтними параметрами — у `rfdetr/training/albumentation_config.py`; довідка: [explore.albumentations.ai](https://explore.albumentations.ai/).

---

## Експорт ONNX

### Окремий скрипт

```bash
python export_onnx.py
```

У `export_onnx.py` задати шлях до checkpoint, OUTPUT_DIR, SIMPLIFY, OPSET_VERSION тощо. З checkpoint **автоматично** визначаються розмір моделі, кількість класів, роздільність.

### Автоекспорт після тренування

При `ExportConfig.enabled=True` у `train.py` після тренування виконується експорт ONNX (і за потреби TensorRT) через `rfdetr/deploy/export.py`.

### Сумісність з PyTorch 2.6+

У `rfdetr/deploy/export.py` використовується `dynamo=False` для legacy TorchScript-based ONNX exporter, щоб уникнути помилок на динамічних операціях декодера RF-DETR.

---

## Результати тренування

```
runs/training/exp/
├── weights/
│   ├── best.pt
│   └── last.pt
├── train_batch0.jpg
├── val_batch0_labels.jpg
├── val_batch0_pred.jpg
├── confusion_matrix.png
├── PR_curve.png, F1_curve.png, P_curve.png, R_curve.png
├── results.png
├── config_*.json
├── augmentations_log.json
└── training.log
```

Валідатор формує markdown-звіт з mAP, precision, recall, F1 та таблицею по класах.

---

## API

### RFDETRTrainer

```python
RFDETRTrainer(
    model_config=ModelConfig,
    training_config=TrainingConfig,
    augmentation_config=AugmentationConfig,
    albumentation_transforms=None,  # None = get_default_albu_config()
    seed=42,
)
trainer.train(dataset_dir="path/to/dataset", resume=None)
# Повертає: {'best_map', 'epochs_trained', 'save_dir'}
```

### RFDETRValidator

```python
RFDETRValidator(model_path="...", model_size="m", ...)
validator.validate(dataset_dir="...", split="valid", save_visualizations=True)
```

### RFDETRDataset / build_dataset

```python
from rfdetr.training import build_dataset, AugmentationConfig, get_default_albu_config

dataset = build_dataset(
    "path/to/dataset",
    split="train",
    augmentation_config=AugmentationConfig(...),
    albumentation_transforms=get_default_albu_config(),  # або свій список A.*
    log_augmentations=False,
)
```

---

## Порівняння з Ultralytics YOLO

| Можливість | Ultralytics YOLO | Цей проєкт |
|------------|------------------|------------|
| Mosaic | Так | Так (наша або A.Mosaic, конфіг) |
| MixUp / CutMix | Так | Так (box-aware) |
| Колір / flip / геометрія | Власні кроки | **Albumentations** (повний список A.*) |
| RandomErasing / CoarseDropout | Обмежено | A.CoarseDropout + box-aware логіка в wrapper |
| close_mosaic | Так | Так |
| Конфігурація | YAML/CLI | Dataclass + ALBUMENTATION_CONFIG |
| Превью аугментацій | Ні | Так (`tests/preview_augmentations.py`) |
| Markdown-звіти валідації | Ні | Так |
| ONNX з автовизначенням | Ні | Так |

---

## Ліцензія

Форк [RF-DETR](https://github.com/roboflow/rf-detr):

- **Apache 2.0** — ядро моделей (Nano, Small, Medium, Base, Large) та пайплайн тренування.
- **Platform Model License 1.0** — моделі XLarge та 2XLarge потребують активного плану Roboflow.

Деталі: [LICENSE](LICENSE), [LICENSE.core](LICENSE.core), [LICENSE.platform](LICENSE.platform).

---

<p align="center">
  <sub>Проєкт побудовано на основі <a href="https://roboflow.com">Roboflow</a> RF-DETR</sub>
</p>
