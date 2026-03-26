import sys
from pathlib import Path

# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ: ШЛЯХИ
# =============================================================================

# Шлях до відеофайлу для інспекції
VIDEO_PATH = "D:/work/diff_stuff/test_videos/test_videos_dpsu/Chinese_camera.mp4"

# Шлях до COCO JSON файлу анотацій
# Очікується стандартний COCO формат: {"images": [...], "annotations": [...], "categories": [...]}
ANNOTATIONS_PATH = "D:\\work\\rfdetr_training\\tracked_videos\\rfdetr_medium_sahi_576x576\\Chinese_camera_annotations_coco.json"

# Назва датасету у FiftyOne (якщо вже існує — буде перезаписаний)
DATASET_NAME = "chinese_camera"

# =============================================================================
# ПАРАМЕТРИ FIFTYONE
# =============================================================================

# Порт FiftyOne App (за замовчуванням 5151)
FO_PORT = 5151

# True = автоматично відкрити браузер після запуску
FO_AUTO_OPEN = True

# True = заблокувати термінал (очікувати Ctrl+C), False = повернутись одразу
FO_WAIT = True

# =============================================================================
# ФІЛЬТРАЦІЯ АНОТАЦІЙ
# =============================================================================

# None = показати всі класи; або список назв класів для відображення, напр.: ["person", "car"]
FILTER_CLASSES = None

# Мінімальне значення площі боксу в пікселях (None = без фільтру)
MIN_BBOX_AREA = None


# =============================================================================
# ЛОГІКА
# =============================================================================

def load_coco_json(json_path: str) -> dict:
    """Завантажує COCO JSON і повертає словник з images, annotations, categories."""
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_category_map(coco_data: dict) -> dict:
    """Повертає {category_id: category_name} з COCO JSON."""
    return {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}


def build_image_annotations_map(coco_data: dict, category_map: dict) -> dict:
    """
    Повертає {file_name: [{"label": ..., "bbox": [x, y, w, h], "area": ...}, ...]}
    де bbox — абсолютні пікселі у форматі COCO [x_min, y_min, width, height].
    """
    ann_map: dict = {}
    images_by_id = {img["id"]: img for img in coco_data.get("images", [])}

    for ann in coco_data.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id not in images_by_id:
            continue
        img_info = images_by_id[img_id]
        file_name = img_info["file_name"]
        cat_name = category_map.get(ann.get("category_id"), "unknown")

        if FILTER_CLASSES is not None and cat_name not in FILTER_CLASSES:
            continue

        bbox = ann.get("bbox", [])
        area = ann.get("area", 0)
        if MIN_BBOX_AREA is not None and area < MIN_BBOX_AREA:
            continue

        ann_map.setdefault(file_name, []).append({
            "label": cat_name,
            "bbox": bbox,
            "area": area,
            "img_width": img_info.get("width", 0),
            "img_height": img_info.get("height", 0),
        })

    return ann_map


def coco_bbox_to_fo(bbox_coco: list, img_w: int, img_h: int):
    """
    COCO bbox [x, y, w, h] (абсолютні пікселі) → FiftyOne [rx, ry, rw, rh]
    (нормалізовані відносні координати, верхній лівий кут).
    """
    import fiftyone as fo
    x, y, w, h = bbox_coco
    if img_w <= 0 or img_h <= 0:
        return None
    return fo.Detection(
        bounding_box=[x / img_w, y / img_h, w / img_w, h / img_h],
    )


def create_fo_dataset(video_path: str, ann_map: dict, category_map: dict) -> "fo.Dataset":
    """
    Створює FiftyOne датасет з одним відео та відповідними анотаціями.

    Стратегія: якщо COCO JSON містить анотації до фреймів відео (file_name відповідає
    шаблону frame_XXXXXX.jpg або схожому), кожен фрейм отримує свої детекції.
    Якщо анотації прив'язані до імені відеофайлу напряму — детекції додаються до
    першого фрейму як fallback.
    """
    import fiftyone as fo

    # Видалити датасет якщо він вже існує
    if fo.dataset_exists(DATASET_NAME):
        fo.delete_dataset(DATASET_NAME)

    dataset = fo.Dataset(name=DATASET_NAME)

    video_file = Path(video_path)
    sample = fo.Sample(filepath=str(video_file.resolve()))

    # Перевіряємо чи є анотації прив'язані до відео безпосередньо (за іменем файлу)
    video_stem = video_file.stem
    video_name = video_file.name

    # Збираємо ключі що відповідають відео (basename або stem)
    video_keys = [k for k in ann_map if Path(k).stem == video_stem or k == video_name]

    if video_keys:
        # Анотації до всього відео → додаємо як video-level detections
        all_dets = []
        for key in video_keys:
            for ann in ann_map[key]:
                img_w = ann["img_width"]
                img_h = ann["img_height"]
                det = fo.Detection(
                    label=ann["label"],
                    bounding_box=[
                        ann["bbox"][0] / img_w if img_w > 0 else 0,
                        ann["bbox"][1] / img_h if img_h > 0 else 0,
                        ann["bbox"][2] / img_w if img_w > 0 else 0,
                        ann["bbox"][3] / img_h if img_h > 0 else 0,
                    ],
                )
                all_dets.append(det)
        if all_dets:
            sample["ground_truth"] = fo.Detections(detections=all_dets)

    else:
        # Анотації прив'язані до окремих фреймів відео через frame_labels
        frame_labels = {}
        for file_name, anns in ann_map.items():
            # Визначаємо номер фрейму з імені файлу (frame_XXXXXX.jpg → 1-indexed)
            stem = Path(file_name).stem
            frame_number = None
            for part in reversed(stem.replace("-", "_").split("_")):
                if part.isdigit():
                    frame_number = int(part)
                    break
            if frame_number is None:
                continue

            dets = []
            for ann in anns:
                img_w = ann["img_width"]
                img_h = ann["img_height"]
                if img_w <= 0 or img_h <= 0:
                    continue
                det = fo.Detection(
                    label=ann["label"],
                    bounding_box=[
                        ann["bbox"][0] / img_w,
                        ann["bbox"][1] / img_h,
                        ann["bbox"][2] / img_w,
                        ann["bbox"][3] / img_h,
                    ],
                )
                dets.append(det)

            if dets:
                frame_labels[frame_number] = fo.Detections(detections=dets)

        for frame_number, detections in frame_labels.items():
            sample.frames[frame_number]["ground_truth"] = detections

    dataset.add_sample(sample)
    dataset.save()

    return dataset


def print_dataset_summary(dataset, coco_data: dict, category_map: dict) -> None:
    """Виводить коротку статистику по датасету та анотаціях."""
    print("\n" + "=" * 70)
    print("FIFTYONE DATASET SUMMARY")
    print("=" * 70)
    print(f"Назва датасету : {dataset.name}")
    print(f"Відео          : {VIDEO_PATH}")
    print(f"Анотації       : {ANNOTATIONS_PATH}")
    print(f"Кількість зразків : {len(dataset)}")
    print(f"Категорії ({len(category_map)}): {', '.join(category_map.values())}")

    total_annotations = len(coco_data.get("annotations", []))
    print(f"Анотацій у JSON: {total_annotations}")
    if FILTER_CLASSES:
        print(f"Фільтр класів  : {FILTER_CLASSES}")
    print("=" * 70 + "\n")


def main():
    """Головна функція: завантаження, побудова датасету, відкриття FiftyOne App."""
    import fiftyone as fo

    video_file = Path(VIDEO_PATH)
    ann_file = Path(ANNOTATIONS_PATH)

    # Перевірки
    if not video_file.exists():
        print(f"[ERROR] Відео не знайдено: {VIDEO_PATH}")
        sys.exit(1)
    if not ann_file.exists():
        print(f"[ERROR] Файл анотацій не знайдено: {ANNOTATIONS_PATH}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("FIFTYONE VIDEO INSPECTOR")
    print("=" * 70)
    print(f"Відео      : {VIDEO_PATH}")
    print(f"Анотації   : {ANNOTATIONS_PATH}")
    print(f"Датасет    : {DATASET_NAME}")
    print("=" * 70)

    print("\n[1/3] Завантаження COCO JSON анотацій...")
    coco_data = load_coco_json(str(ann_file))
    category_map = build_category_map(coco_data)
    ann_map = build_image_annotations_map(coco_data, category_map)
    print(f"      Категорії: {list(category_map.values())}")
    print(f"      Зображень/фреймів з анотаціями: {len(ann_map)}")

    print("\n[2/3] Створення FiftyOne датасету...")
    dataset = create_fo_dataset(VIDEO_PATH, ann_map, category_map)
    print(f"      Датасет '{DATASET_NAME}' створено.")

    print_dataset_summary(dataset, coco_data, category_map)

    print(f"[3/3] Запуск FiftyOne App на порті {FO_PORT}...")
    session = fo.launch_app(dataset, port=FO_PORT, auto=FO_AUTO_OPEN)

    if FO_WAIT:
        print("\n[OK] FiftyOne запущено. Відкрийте браузер: http://localhost:{}\n".format(FO_PORT))
        print("     Натисніть Ctrl+C для завершення.\n")
        session.wait()
    else:
        print("\n[OK] FiftyOne App запущено у фоні.")

    return dataset


if __name__ == "__main__":
    main()
