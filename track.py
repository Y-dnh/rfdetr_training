"""
Модуль для обробки відео: детекція RF-DETR кожні N фреймів + трекінг NanoTrack,
збереження вихідного відео з накладеними боксами та ID треків.
Усі параметри конфігурації знаходяться на початку файлу.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Фікс для правильного відображення tqdm у Windows PowerShell
if sys.platform == "win32":
    os.system("")  # Включає ANSI escape sequences підтримку

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Додаємо корінь проєкту в шлях для імпорту tracking та rfdetr
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tracking import NanoTracker, TrackedObject


# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ (RF-DETR)
# =============================================================================
PROJECT_NAME = "rfdetr_track"
PROJECT_DIR = os.path.join(BASE_DIR, PROJECT_NAME)
MODEL_PATH = "D:/rfdetr_dpsu_v8.pth"  # Шлях до .pth checkpoint RF-DETR
MODEL_SIZE = "m"  # ['n','s','m','b','l','xl','2xl'] — має відповідати checkpoint'у

# --- Відео ---
VIDEO_INPUT_PATH = "E:/DPSU/dataset_videos/uzhorod/videos_to_extract/006_02.12.2025_08.20_08.40.mkv"

# --- Детекція: кожні N фреймів запускати модель; між ними — трекінг NanoTrack ---
DETECTION_INTERVAL = 10

# =============================================================================
# ПАРАМЕТРИ ІНФЕРЕНСУ RF-DETR
# =============================================================================
INFERENCE_CONFIG = {
    "conf_threshold": 0.25,
    "iou_threshold": 0.5,
    "max_det": 300,
    "half": True,
    "device": None,  # None = авто (cuda якщо є)
    "classes": None,  # Фільтр класів (None = усі)
}

# --- NanoTrack: ONNX-моделі (v2 або v3) ---
NANOTRACK_VERSION = "v2"  # "v2" або "v3"
NANOTRACK_DIR = os.path.join(BASE_DIR, "nanotrack")
if NANOTRACK_VERSION == "v3":
    NANOTRACK_BACKBONE = os.path.join(NANOTRACK_DIR, "v3", "nanotrack_backbone.onnx")
    NANOTRACK_NECKHEAD = os.path.join(NANOTRACK_DIR, "v3", "nanotrack_head.onnx")
else:
    NANOTRACK_BACKBONE = os.path.join(NANOTRACK_DIR, "v2", "nanotrack_backbone_sim.onnx")
    NANOTRACK_NECKHEAD = os.path.join(NANOTRACK_DIR, "v2", "nanotrack_head_sim.onnx")

# --- Класи та кольори (BGR): люди червоні, машини сині, вантажівка зелена ---
CLASS_NAMES = {0: "person", 1: "car", 2: "truck"}
CLASS_COLORS = [
    (0, 0, 255),    # person — червоний
    (255, 0, 0),    # car — синій
    (0, 255, 0),    # truck — зелений
]
BBOX_THICKNESS = 2
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1
LABEL_PADDING = 4
MASK_ALPHA = 0.0  # напівпрозорий overlay всередині боксу (0 = вимкнено, 0.35 як на фото)

# --- Параметри трекінгу (NanoTracker) ---
MAX_AGE = 10
MIN_HITS = 2
IOU_THRESHOLD = 0.3
CONFIRM_THRESHOLD = 5
MIN_SEC_STABLE = 1.0
USE_OPTICAL_FLOW_PREDICT = True
OPTICAL_FLOW_THRESHOLD = 8
ADAPTIVE_UPDATE = True
ADAPTIVE_THRESHOLD = 10
ENABLE_REID = True
REID_BUFFER_TIME = 20.0
REID_IOU_THRESHOLD = 0.15
REID_APPEARANCE_THRESHOLD = 0.5
REID_POSITION_WEIGHT = 0.4
REID_APPEARANCE_WEIGHT = 0.4
REID_SIZE_WEIGHT = 0.2
REID_MIN_TRACK_QUALITY = 5


# =============================================================================
# ФУНКЦІЇ
# =============================================================================
def _get_device():
    if INFERENCE_CONFIG.get("device") is not None:
        return torch.device(INFERENCE_CONFIG["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _detect_imgsz_from_checkpoint(checkpoint: dict, model_size: str) -> int:
    """Визначити imgsz з checkpoint (як у валідаторі). Fallback — з model_size."""
    size_to_resolution = {"n": 384, "s": 512, "m": 576, "b": 560, "l": 704, "xl": 700, "2xl": 880}
    if "augmentation_config" in checkpoint:
        aug = checkpoint["augmentation_config"]
        if isinstance(aug, dict) and "imgsz" in aug:
            return aug["imgsz"]
    if "model_config" in checkpoint:
        cfg = checkpoint["model_config"]
        if isinstance(cfg, dict) and "model_size" in cfg:
            return size_to_resolution.get(cfg["model_size"], 576)
    return size_to_resolution.get(model_size, 576)


def load_rfdetr_model(model_path: str, model_size: str, num_classes: int = None, class_names: dict = None):
    """Завантаження RF-DETR з checkpoint (.pth). Повертає (rfdetr, num_classes, class_names, imgsz)."""
    from rfdetr.detr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    num_classes = num_classes or checkpoint.get("num_classes") or 80
    if class_names is None and "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
    imgsz = _detect_imgsz_from_checkpoint(checkpoint, model_size)

    size_map = {
        "n": RFDETRNano, "nano": RFDETRNano,
        "s": RFDETRSmall, "small": RFDETRSmall,
        "m": RFDETRMedium, "medium": RFDETRMedium,
        "b": RFDETRBase, "base": RFDETRBase,
        "l": RFDETRLarge, "large": RFDETRLarge,
    }
    if model_size in ("xl", "xlarge", "2xl", "2xlarge"):
        try:
            from rfdetr.platform.models import RFDETRXLarge, RFDETR2XLarge
            size_map["xl"] = lambda **kw: RFDETRXLarge(accept_platform_model_license=True, **kw)
            size_map["2xl"] = lambda **kw: RFDETR2XLarge(accept_platform_model_license=True, **kw)
        except Exception:
            pass
    model_cls = size_map.get(model_size, RFDETRMedium)
    rfdetr = model_cls(pretrain_weights=model_path, num_classes=num_classes)
    rfdetr.model.device = _get_device()
    rfdetr.model.model = rfdetr.model.model.to(rfdetr.model.device)
    rfdetr.model.model.eval()
    rfdetr.optimize_for_inference(compile=False, batch_size=1)
    if class_names is not None:
        rfdetr.model.class_names = class_names
    out_names = class_names or getattr(rfdetr.model, "class_names", None) or CLASS_NAMES
    return rfdetr, num_classes, out_names, imgsz


def _letterbox_inverse_params(frame_h: int, frame_w: int, imgsz: int):
    """
    Параметри зворотного перетворення LetterBox: (r, dw, dh).
    LetterBox: scale r, pad (dw, dh) по ширині/висоті. Бокси в letterbox-пікселях
    переводяться в кадр як: frame_x = (lb_x - dw) / r, frame_y = (lb_y - dh) / r.
    """
    r = min(imgsz / frame_h, imgsz / frame_w)
    new_w = int(round(frame_w * r))
    new_h = int(round(frame_h * r))
    dw = (imgsz - new_w) / 2.0
    dh = (imgsz - new_h) / 2.0
    return r, dw, dh


def _prepare_frame_for_rfdetr(frame_bgr: np.ndarray, imgsz: int):
    """BGR кадр -> тензор для RF-DETR (LetterBox + ToTensor + Normalize)."""
    from rfdetr.training.augmentations.pipeline import ValidationPipeline

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pipeline = ValidationPipeline(imgsz=imgsz)
    tensor, _ = pipeline(frame_rgb, {})
    return tensor.unsqueeze(0)  # [1, 3, H, W]


def run_detection(rfdetr, frame: np.ndarray, frame_w: int, frame_h: int, imgsz: int, conf_threshold: float, max_det: int, classes_filter: list = None) -> list:
    """
    Інференс RF-DETR на одному кадрі.
    Модель бачить letterbox (imgsz×imgsz), тому бокси спочатку в просторі letterbox —
    перетворюємо їх у координати оригінального кадру, потім у нормалізовані (0–1).
    Повертає список dict з ключами 'box' (cx, cy, w, h у 0–1) та 'cls_id'.
    """
    device = _get_device()
    images = _prepare_frame_for_rfdetr(frame, imgsz).to(device)
    if INFERENCE_CONFIG.get("half"):
        images = images.half()

    with torch.no_grad():
        if getattr(rfdetr, "_is_optimized_for_inference", False) and rfdetr.model.inference_model is not None:
            outputs = rfdetr.model.inference_model(images)
        else:
            outputs = rfdetr.model.model(images)

    if isinstance(outputs, tuple):
        outputs = {"pred_boxes": outputs[0], "pred_logits": outputs[1]}

    # Постпроцес у просторі letterbox (imgsz×imgsz) — бокси в пікселях letterbox
    target_sizes = torch.tensor([[imgsz, imgsz]], device=device, dtype=torch.long)
    results = rfdetr.model.postprocess(outputs, target_sizes=target_sizes)
    result = results[0]

    scores = result["scores"].float().cpu()
    labels = result["labels"].long().cpu()
    boxes_xyxy_lb = result["boxes"].float().cpu()  # [N, 4] у пікселях letterbox (imgsz×imgsz)

    mask = scores > conf_threshold
    scores = scores[mask]
    labels = labels[mask]
    boxes_xyxy_lb = boxes_xyxy_lb[mask]

    if max_det is not None and len(scores) > max_det:
        top = torch.topk(scores, max_det).indices
        scores = scores[top]
        labels = labels[top]
        boxes_xyxy_lb = boxes_xyxy_lb[top]

    # Letterbox -> оригінальний кадр: frame_x = (lb_x - dw) / r
    r, dw, dh = _letterbox_inverse_params(frame_h, frame_w, imgsz)

    detections = []
    for i in range(len(scores)):
        cls_id = int(labels[i].item())
        if classes_filter is not None and cls_id not in classes_filter:
            continue
        x1_lb, y1_lb, x2_lb, y2_lb = boxes_xyxy_lb[i].tolist()
        x1 = (x1_lb - dw) / r
        y1 = (y1_lb - dh) / r
        x2 = (x2_lb - dw) / r
        y2 = (y2_lb - dh) / r
        # Кліп у межі кадру (бокс міг вийти в паддинг)
        x1 = max(0.0, min(frame_w, x1))
        y1 = max(0.0, min(frame_h, y1))
        x2 = max(0.0, min(frame_w, x2))
        y2 = max(0.0, min(frame_h, y2))
        w_n = (x2 - x1) / frame_w
        h_n = (y2 - y1) / frame_h
        cx = (x1 + x2) / 2.0 / frame_w
        cy = (y1 + y2) / 2.0 / frame_h
        conf = float(scores[i].item())
        detections.append({"box": (cx, cy, w_n, h_n), "cls_id": cls_id, "conf": conf})
    return detections


def draw_tracks(frame: np.ndarray, tracked: list, class_names: dict, colors: list) -> None:
    """
    Малювання в стилі з фото: бокс кольору класу, напівпрозорий overlay,
    лейбл у два рядки на кольоровому фоні — «ID: N» та «class (0.94)».
    """
    h, w = frame.shape[:2]
    for obj in tracked:
        bbox = getattr(obj, "bbox", None)
        if bbox is None or len(bbox) != 4:
            continue
        x1_n, y1_n, x2_n, y2_n = bbox
        x1 = int(x1_n * w)
        y1 = int(y1_n * h)
        x2 = int(x2_n * w)
        y2 = int(y2_n * h)
        cls_id = getattr(obj, "cls_id", None)
        cls_id = cls_id if cls_id is not None else 0
        color = colors[cls_id % len(colors)] if colors else (0, 0, 255)
        label = class_names.get(cls_id, f"cls_{cls_id}")
        track_id = getattr(obj, "track_id", None)
        confidence = getattr(obj, "confidence", None)
        line1 = f"ID: {track_id}" if track_id is not None else "ID: —"
        line2 = f"{label} ({confidence:.2f})" if confidence is not None else label

        # Напівпрозорий overlay всередині боксу
        if MASK_ALPHA > 0:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                overlay = roi.copy()
                overlay[:] = color
                cv2.addWeighted(overlay, MASK_ALPHA, roi, 1 - MASK_ALPHA, 0, roi)

        # Контур боксу
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)

        # Два рядки тексту
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw1, th1), _ = cv2.getTextSize(line1, font, TEXT_SCALE, TEXT_THICKNESS)
        (tw2, th2), _ = cv2.getTextSize(line2, font, TEXT_SCALE, TEXT_THICKNESS)
        label_w = max(tw1, tw2) + LABEL_PADDING * 2
        label_h = th1 + th2 + LABEL_PADDING * 3
        label_x1 = x1
        label_y1 = max(0, y1 - label_h)
        label_x2 = label_x1 + label_w
        label_y2 = label_y1 + label_h
        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, BBOX_THICKNESS)
        cv2.putText(
            frame, line1,
            (label_x1 + LABEL_PADDING, label_y1 + LABEL_PADDING + th1),
            font, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA,
        )
        cv2.putText(
            frame, line2,
            (label_x1 + LABEL_PADDING, label_y1 + LABEL_PADDING + th1 + th2 + 2),
            font, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA,
        )


def run_tracking(
    video_input_path: str,
    model_path: str = MODEL_PATH,
    detection_interval: int = DETECTION_INTERVAL,
) -> str | None:
    """
    Головний пайплайн: відео, детекція RF-DETR кожні detection_interval кадрів,
    трекінг NanoTrack на кожному кадрі, малювання, запис у tracked_videos/<модель>/.
    """
    if not video_input_path or not os.path.isfile(video_input_path):
        print(f"Помилка: не знайдено відео: {video_input_path}")
        return None
    if not os.path.isfile(model_path):
        print(f"Помилка: не знайдено модель: {model_path}")
        return None

    model_name = f"{PROJECT_NAME}_{Path(model_path).stem}"
    output_dir = os.path.join(BASE_DIR, "tracked_videos", model_name)
    os.makedirs(output_dir, exist_ok=True)
    video_stem = Path(video_input_path).stem
    output_path = os.path.join(output_dir, f"{video_stem}_tracked.mp4")
    log_path = os.path.join(output_dir, f"{video_stem}_tracked.txt")

    use_nano = os.path.isfile(NANOTRACK_BACKBONE) and os.path.isfile(NANOTRACK_NECKHEAD)
    if not use_nano:
        print(
            "Увага: моделі NanoTrack не знайдено. "
            "Буде використано режим лише детекції кожні N кадрів (без трекінгу між кадрами)."
        )

    rfdetr, num_classes, class_names, imgsz = load_rfdetr_model(model_path, MODEL_SIZE, class_names=CLASS_NAMES)
    print(f"Завантаження RF-DETR: {Path(model_path).name} (size={MODEL_SIZE}, imgsz={imgsz})")

    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print(f"Помилка: не вдалося відкрити відео: {video_input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    cfg = INFERENCE_CONFIG
    print()
    print("=" * 60)
    print("TRACK VIDEO (RF-DETR + NanoTrack)")
    print("=" * 60)
    print(f"Відео: {video_input_path}")
    print(f"Вихід: {output_path}")
    print(f"Модель: {Path(model_path).name} (RF-DETR {MODEL_SIZE})")
    print(f"Кадрів: {total_frames}, {fps:.1f} FPS, {width}x{height}")
    print(f"Детекція кожні: {detection_interval} фреймів")
    print(f"Інференс: conf={cfg.get('conf_threshold')}, max_det={cfg.get('max_det')}, imgsz={imgsz}, half={cfg.get('half')}")
    print("=" * 60)
    print()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Помилка: не вдалося створити вихідний файл: {output_path}")
        cap.release()
        return None

    tracker = None
    try:
        tracker = NanoTracker(
            class_names=class_names,
            backbone_path=NANOTRACK_BACKBONE,
            neckhead_path=NANOTRACK_NECKHEAD,
            max_age=MAX_AGE,
            min_hits=MIN_HITS,
            iou_threshold=IOU_THRESHOLD,
            confirm_threshold=CONFIRM_THRESHOLD,
            min_sec_stable=MIN_SEC_STABLE,
            use_optical_flow_predict=USE_OPTICAL_FLOW_PREDICT,
            optical_flow_threshold=OPTICAL_FLOW_THRESHOLD,
            adaptive_update=ADAPTIVE_UPDATE,
            adaptive_threshold=ADAPTIVE_THRESHOLD,
            enable_reid=ENABLE_REID,
            reid_buffer_time=REID_BUFFER_TIME,
            reid_iou_threshold=REID_IOU_THRESHOLD,
            reid_appearance_threshold=REID_APPEARANCE_THRESHOLD,
            reid_position_weight=REID_POSITION_WEIGHT,
            reid_appearance_weight=REID_APPEARANCE_WEIGHT,
            reid_size_weight=REID_SIZE_WEIGHT,
            reid_min_track_quality=REID_MIN_TRACK_QUALITY,
        )
    except Exception as e:
        print(f"Не вдалося створити NanoTracker (потрібні ONNX та OpenCV з TrackerNano): {e}")
        tracker = None

    frame_counter = 0
    last_tracked = []
    start_time = time.perf_counter()
    pbar = tqdm(total=total_frames if total_frames else None, unit="frame", desc="Track")
    conf_threshold = cfg.get("conf_threshold", 0.25)
    max_det = cfg.get("max_det", 300)
    classes_filter = cfg.get("classes")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_counter += 1
            pbar.update(1)
            frame_h, frame_w = frame.shape[:2]

            if frame_counter % detection_interval == 1 or frame_counter == 1:
                detections = run_detection(
                    rfdetr, frame, frame_w, frame_h,
                    imgsz=imgsz,
                    conf_threshold=conf_threshold,
                    max_det=max_det,
                    classes_filter=classes_filter,
                )
                if tracker is not None:
                    try:
                        last_tracked = tracker.update(detections, frame)
                    except Exception as e:
                        print(f"Помилка трекера (кадр {frame_counter}): {e}")
                        last_tracked = []
                else:
                    last_tracked = []
                    for d in detections:
                        cx, cy, w, h = d["box"]
                        bbox_n = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                        last_tracked.append(
                            type("Obj", (), {"bbox": bbox_n, "cls_id": d.get("cls_id"), "track_id": "", "confidence": d.get("conf")})()
                        )
            else:
                if tracker is not None:
                    try:
                        last_tracked = tracker.update(None, frame)
                    except Exception:
                        pass

            draw_tracks(frame, last_tracked, class_names, CLASS_COLORS)
            writer.write(frame)
    except Exception as e:
        print(f"Помилка під час обробки: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()
        cap.release()
        writer.release()

    elapsed_sec = time.perf_counter() - start_time
    fps_processed = frame_counter / elapsed_sec if elapsed_sec > 0 else 0.0

    log_lines = [
        "=" * 60,
        "РЕЗУЛЬТАТИ ОПРАЦЮВАННЯ (RF-DETR + NanoTrack)",
        "=" * 60,
        f"Дата/час: {datetime.now().isoformat()}",
        f"Вхідне відео: {video_input_path}",
        f"Вихідне відео: {output_path}",
        f"Роздільність: {width}x{height}",
        f"Кадрів у джерелі: {total_frames}",
        f"Оброблено кадрів: {frame_counter}",
        f"Час опрацювання (с): {elapsed_sec:.2f}",
        f"FPS при обробці: {fps_processed:.2f}",
        "",
        "--- МОДЕЛЬ ДЕТЕКЦІЇ (RF-DETR) ---",
        f"Модель: {model_path}",
        f"MODEL_SIZE: {MODEL_SIZE}",
        f"imgsz (з checkpoint): {imgsz}",
        "",
        "--- КОНФІГ ІНФЕРЕНСУ ---",
    ]
    for k, v in sorted(INFERENCE_CONFIG.items()):
        log_lines.append(f"  {k}: {v}")
    log_lines.extend([
        "",
        "--- ДЕТЕКЦІЯ ---",
        f"  DETECTION_INTERVAL: {detection_interval}",
        "",
        "--- NANOTRACK ---",
        f"  NANOTRACK_VERSION: {NANOTRACK_VERSION}",
        f"  NANOTRACK_BACKBONE: {NANOTRACK_BACKBONE}",
        f"  NANOTRACK_NECKHEAD: {NANOTRACK_NECKHEAD}",
        "",
        "--- ПАРАМЕТРИ ТРЕКИНГУ ---",
        f"  MAX_AGE: {MAX_AGE}",
        f"  MIN_HITS: {MIN_HITS}",
        f"  IOU_THRESHOLD: {IOU_THRESHOLD}",
        f"  ENABLE_REID: {ENABLE_REID}",
        "",
        "--- КЛАСИ ---",
        f"  CLASS_NAMES: {CLASS_NAMES}",
        "=" * 60,
    ])
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print(f"Лог збережено: {log_path}")
    except Exception as e:
        print(f"Не вдалося записати лог: {e}")

    print()
    print("Готово. Вихідне відео: " + output_path)
    return output_path


def main():
    """Головна функція для запуску обробки відео."""
    video_path = VIDEO_INPUT_PATH.strip()
    if not video_path:
        print("Задайте VIDEO_INPUT_PATH у конфігу на початку файлу.")
        return None
    return run_tracking(video_path, MODEL_PATH, DETECTION_INTERVAL)


if __name__ == "__main__":
    main()
