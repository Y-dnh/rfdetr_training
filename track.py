"""
Модуль для обробки відео: детекція RF-DETR кожні N фреймів + трекінг NanoTrack,
збереження вихідного відео з накладеними боксами та ID треків.
Усі параметри конфігурації знаходяться на початку файлу.

BENCHMARK_MODE=True: детальний профайлінг (час кожної фази, FPS), звіт *_benchmark.txt
USE_SAHI=True: Slicing Aided Hyper Inference — нарізка кадру на перекриваючі фрагменти
  для кращого виявлення дрібних об'єктів у великих кадрах.
"""

import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Tuple

# Фікс для правильного відображення tqdm у Windows PowerShell
if sys.platform == "win32":
    os.system("")  # Включає ANSI escape sequences підтримку

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Додаємо корінь проєкту в шлях для імпорту tracking та rfdetr
BASE_DIR = Path(__file__).parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tracking import NanoTracker, TrackedObject


# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ: ШЛЯХИ
# =============================================================================
# Та сама структура, що в train.py та val.py (Ultralytics-style). Модель з runs/.../<experiment>/weights/
PROJECT_NAME = "rfdetr_large"
EXPERIMENT_NAME = "baseline"   # Експеримент тренування, звідки брати модель
RUNS_DIR = BASE_DIR / "runs"
PROJECT_DIR = RUNS_DIR / PROJECT_NAME
# Модель: за замовчуванням best.pt з runs/.../<experiment>/weights/
MODEL_PATH = PROJECT_DIR / EXPERIMENT_NAME / "weights" / "inference_model.sim.engine"
MODEL_SIZE = "l"  # ['n','s','m','b','l','xl','2xl'] — має відповідати checkpoint'у
# MODEL_RESOLUTIONS = {'n': 384, 's': 512, 'm': 576, 'b': 560, 'l': 704, 'xl': 700, '2xl': 880}

# Вхідне відео або папка з відео для трекінгу.
# Якщо вказана папка — опрацьовуються всі відеофайли у ній (рекурсивно не шукаємо).
# Вихід: tracked_videos/<назва_моделі>/<ім'я_відео>_tracked.mp4 та .txt з логами.
VIDEO_INPUT_PATH = "D:\\videos_for_test\\zir"

# Benchmark: True = профайлінг (заміри по фазах, звіт _benchmark.txt)
BENCHMARK_MODE = True
BENCHMARK_CUDA_SYNC = True
BENCHMARK_WRITE_VIDEO = True   # False = тільки профайлінг, без запису відео
BENCHMARK_MAX_FRAMES = None    # None = все відео

# Розширення файлів, що вважаються відео (при вказівці папки).
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv", ".flv"}

# Як часто запускати детекцію: модель працює тільки на кадрах 1, 1+N, 1+2N, ...; між ними лише NanoTrack.
DETECTION_INTERVAL = 10

# =============================================================================
# ПАРАМЕТРИ ІНФЕРЕНСУ (RF-DETR)
# =============================================================================
INFERENCE_CONFIG = {
    "conf_threshold": 0.25,   # мінімальний confidence детекції (нижче — відкидається)
    "iou_threshold": 0.3,     # IoU поріг для NMS (об'єднання дублікатів боксів)
    "max_det": 300,           # максимум детекцій на один кадр
    "half": True,             # FP16 інференс (швидше на GPU)
    "device": None,           # None = авто (CUDA якщо є)
    "classes": None,          # фільтр класів (None = усі класи)
}

# =============================================================================
# NANOTRACK: ШЛЯХИ ДО ONNX-МОДЕЛЕЙ
# =============================================================================
NANOTRACK_VERSION = "v2"  # "v2" або "v3"
NANOTRACK_DIR = BASE_DIR / "nanotrack"
if NANOTRACK_VERSION == "v3":
    NANOTRACK_BACKBONE = NANOTRACK_DIR / "v3" / "nanotrack_backbone.onnx"
    NANOTRACK_NECKHEAD = NANOTRACK_DIR / "v3" / "nanotrack_head.onnx"
else:
    NANOTRACK_BACKBONE = NANOTRACK_DIR / "v2" / "nanotrack_backbone_sim.onnx"
    NANOTRACK_NECKHEAD = NANOTRACK_DIR / "v2" / "nanotrack_head_sim.onnx"


# =============================================================================
# ПАРАМЕТРИ NANOTRACKER (життя треків, злиття, ReID)
# =============================================================================
# Не керують тим, коли запускається детекція — лише тим, як довго живуть треки та коли їх показувати.

MAX_AGE = 20        # скільки кадрів трек може жити без оновлення детекцією; після цього видаляється
MIN_HITS = 3        # мінімум попадань детекції по треку, щоб трек почали показувати (фільтр шуму)
IOU_THRESHOLD = 0.25 # мінімальний IoU між боксом детекції та треком, щоб вважати їх одним об'єктом
CONFIRM_THRESHOLD = 3   # після скількох попадань трек вважається «підтвердженим»
MIN_SEC_STABLE = 0.25    # мінімальний час (сек) у полі зору, щоб трек став «стабільним»

# Оптичний потік: передбачення руху треків між кадрами (швидше/точніше за багато треків).
USE_OPTICAL_FLOW_PREDICT = True
OPTICAL_FLOW_THRESHOLD = 8   # від якої кількості треків увімкнути optical flow замість оновлення кожного
ADAPTIVE_UPDATE = True       # адаптивно перемикатися на optical flow при великій кількості треків
ADAPTIVE_THRESHOLD = 10      # поріг кількості треків для адаптивного режиму

# ReID: відновлення втрачених треків (наприклад, після перекриття) за зовнішнім виглядом та позицією.
ENABLE_REID = True
REID_BUFFER_TIME = 20.0           # скільки секунд зберігати «втрачені» треки для пошуку збігу
REID_IOU_THRESHOLD = 0.15         # IoU поріг для кандидатів ReID
REID_APPEARANCE_THRESHOLD = 0.5   # поріг схожості зовнішнього вигляду (0–1)
REID_POSITION_WEIGHT = 0.4        # вага позиції у скорі ReID
REID_APPEARANCE_WEIGHT = 0.4      # вага зовнішнього вигляду
REID_SIZE_WEIGHT = 0.2            # вага розміру боксу
REID_MIN_TRACK_QUALITY = 5        # мінімальна «якість» треку (наприклад, hit_streak), щоб його зберігати в ReID-буфері

# Ресайз кадру перед передачею в NanoTrack: None = без ресайзу (повний кадр),
# int = зменшити до (px, px). Зменшує навантаження на CPU при великій кількості треків.
NANO_IMAGE_RESIZE = None


# =============================================================================
# SAHI (Slicing Aided Hyper Inference)
# =============================================================================
# Розбиває зображення на перекриваючі фрагменти, запускає детекцію на кожному,
# та об'єднує результати. Ефективно для виявлення дрібних об'єктів у великих кадрах.
USE_SAHI = True                          # True = увімкнути SAHI, False = звичайна детекція

SAHI_SLICE_WIDTH = 1080                  # ширина фрагменту (px)
SAHI_SLICE_HEIGHT = 1080                 # висота фрагменту (px)
SAHI_OVERLAP_WIDTH_RATIO = 0.1           # перекриття по ширині (0.0–1.0)
SAHI_OVERLAP_HEIGHT_RATIO = 0.1          # перекриття по висоті (0.0–1.0)
SAHI_PERFORM_STANDARD_PRED = False       # додатково запустити детекцію на повному кадрі
SAHI_POSTPROCESS_TYPE = "NMS"            # "NMS" або "NMM" (Non-Maximum Merging)
SAHI_POSTPROCESS_MATCH_METRIC = "IOU"     # "IOU" або "IOS" (Intersection over Smaller)
SAHI_POSTPROCESS_MATCH_THRESHOLD = 0.5    # поріг IoU/IoS для злиття дублікатів між фрагментами
SAHI_POSTPROCESS_CLASS_AGNOSTIC = False   # True = злиття без урахування класу


# =============================================================================
# ВІЗУАЛІЗАЦІЯ: КЛАСИ ТА КОЛЬОРИ (BGR)
# =============================================================================
CLASS_NAMES = {0: "person", 1: "car", 2: "truck"}
CLASS_COLORS = [
    (0, 0, 255),    # 0 — person, червоний
    (255, 0, 0),    # 1 — car, синій
    (0, 255, 0),    # 2 — truck, зелений
]
# Стиль візуалізації як у visualization.py (та в іншому проєкті track): подвійна рамка, напівпрозорий фон мітки, HERSHEY_DUPLEX
VIS_FONT = cv2.FONT_HERSHEY_DUPLEX
VIS_TEXT_SCALE = 0.6
VIS_TEXT_THICKNESS = 2
VIS_TEXT_COLOR = (255, 255, 255)
VIS_LABEL_PADDING = 8
VIS_LINE_HEIGHT = 25
VIS_LABEL_MARGIN = 5
VIS_LABEL_BG_ALPHA = 0.8       # прозорість фону мітки (0–1)
VIS_BBOX_THICKNESS = 3         # товщина основної рамки
VIS_BBOX_INNER_THICKNESS = 1   # товщина внутрішнього «підсвіту»
MASK_ALPHA = 0.0               # напівпрозорий overlay всередині боксу (0 = вимкнено)


# =============================================================================
# TensorRT INFERENCE WRAPPER
# =============================================================================

class TRTInferenceModel:
    """
    Обгортка для TensorRT .engine файлу, сумісна з інтерфейсом rfdetr
    (model.__call__, model.postprocess).
    """

    def __init__(self, engine_path: str, device: torch.device = None):
        import tensorrt as trt

        self.device = device or _get_device()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.input_name = None
        self.output_names = []
        self.output_shapes = {}
        self.output_dtypes = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = self.engine.get_tensor_shape(name)
            else:
                self.output_names.append(name)
                self.output_shapes[name] = self.engine.get_tensor_shape(name)
                dtype = self.engine.get_tensor_dtype(name)
                self.output_dtypes[name] = trt.nptype(dtype)

        self._trt_output_to_key = {'dets': 'pred_boxes', 'labels': 'pred_logits'}
        self.model = self

        self._stream = torch.cuda.Stream(device=self.device)

        self._output_buffers = {}
        for name in self.output_names:
            shape = list(self.output_shapes[name])
            dtype_np = self.output_dtypes[name]
            dtype_torch = torch.float32 if dtype_np == np.float32 else torch.float16
            buf = torch.empty(shape, dtype=dtype_torch, device=self.device).contiguous()
            self._output_buffers[name] = buf
            self.context.set_tensor_address(name, buf.data_ptr())

    def __call__(self, images: torch.Tensor) -> dict:
        expected_h, expected_w = self.input_shape[2], self.input_shape[3]
        if images.shape[2] != expected_h or images.shape[3] != expected_w:
            raise ValueError(
                f"TRT engine очікує вхід {expected_h}x{expected_w}, "
                f"але отримано {images.shape[2]}x{images.shape[3]}. "
                f"Перевірте imgsz / MODEL_SIZE."
            )

        d_input = images.float().to(self.device).contiguous()
        self.context.set_tensor_address(self.input_name, d_input.data_ptr())

        self.context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        result = {}
        for name, buf in self._output_buffers.items():
            key = self._trt_output_to_key.get(name, name)
            result[key] = buf.clone()
        return result

    def postprocess(self, outputs: dict, target_sizes: torch.Tensor) -> list:
        from rfdetr.util import box_ops

        out_logits = outputs['pred_logits']
        out_bbox = outputs['pred_boxes']
        num_select = min(300, out_logits.shape[1] * out_logits.shape[2])

        prob = out_logits.float().sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1
        )
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox.float())
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        return [{'scores': s, 'labels': l, 'boxes': b}
                for s, l, b in zip(scores, labels, boxes)]


# =============================================================================
# ФУНКЦІЇ
# =============================================================================
def collect_videos_from_folder(folder_path: str) -> list[str]:
    """
    Повертає список шляхів до відеофайлів у вказаній папці (тільки один рівень, без підпапок).
    Відфільтровано по VIDEO_EXTENSIONS, відсортовано за іменем.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    videos = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(str(p.resolve()))
    return sorted(videos)


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


def load_trt_model(model_path: str, model_size: str, num_classes: int = None, class_names: dict = None):
    """Завантаження TensorRT engine (.engine). Повертає (wrapper, num_classes, class_names, imgsz)."""
    size_to_resolution = {"n": 384, "s": 512, "m": 576, "b": 560, "l": 704, "xl": 700, "2xl": 880}
    imgsz = size_to_resolution.get(model_size, 576)
    num_classes = num_classes or 80
    out_names = class_names or CLASS_NAMES

    trt_model = TRTInferenceModel(str(model_path))
    trt_model.inference_model = None

    class _TRTWrapper:
        """Мімікрія rfdetr інтерфейсу для сумісності з _run_detection_standard."""
        pass

    wrapper = _TRTWrapper()
    wrapper.model = trt_model
    wrapper._is_optimized_for_inference = False

    return wrapper, num_classes, out_names, imgsz


def load_rfdetr_model(model_path: str, model_size: str, num_classes: int = None, class_names: dict = None):
    """Завантаження RF-DETR. Автодетект формату: .engine → TRT, .pth/.pt → PyTorch."""
    model_path_str = str(model_path)

    if model_path_str.endswith('.engine'):
        return load_trt_model(model_path_str, model_size, num_classes, class_names)

    from rfdetr.detr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge

    checkpoint = torch.load(model_path_str, map_location="cpu", weights_only=False)
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
    rfdetr = model_cls(pretrain_weights=model_path_str, num_classes=num_classes)
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


def _run_detection_standard(rfdetr, frame: np.ndarray, frame_w: int, frame_h: int, imgsz: int, conf_threshold: float, max_det: int, classes_filter: list = None) -> list:
    """
    Стандартна детекція RF-DETR на повному кадрі.
    Модель бачить letterbox (imgsz×imgsz), тому бокси спочатку в просторі letterbox —
    перетворюємо їх у координати оригінального кадру, потім у нормалізовані (0–1).
    Повертає список dict з ключами 'box' (cx, cy, w, h у 0–1), 'cls_id', 'conf'.
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


# =============================================================================
# SAHI helper-функції
# =============================================================================

def _sahi_generate_slices(
    img_w: int, img_h: int,
    slice_w: int, slice_h: int,
    overlap_w_ratio: float, overlap_h_ratio: float,
) -> list:
    """Повертає список (x1, y1, x2, y2) координат перекриваючих фрагментів."""
    step_x = max(1, int(slice_w * (1 - overlap_w_ratio)))
    step_y = max(1, int(slice_h * (1 - overlap_h_ratio)))
    slices = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x2 = min(x + slice_w, img_w)
            y2 = min(y + slice_h, img_h)
            x1 = max(0, x2 - slice_w)
            y1 = max(0, y2 - slice_h)
            if (x1, y1, x2, y2) not in slices:
                slices.append((x1, y1, x2, y2))
            if x2 >= img_w:
                break
            x += step_x
        if y2 >= img_h:
            break
        y += step_y
    return slices


def _sahi_compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """IoU між двома боксами (x1, y1, x2, y2) у абсолютних координатах."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _sahi_compute_ios(box_a: Tuple, box_b: Tuple) -> float:
    """IoS (Intersection over Smaller) між двома боксами (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _sahi_merge_detections(
    detections: list,
    match_threshold: float,
    match_metric: str = "IOU",
    postprocess_type: str = "NMS",
    class_agnostic: bool = False,
) -> list:
    """
    Злиття детекцій після SAHI нарізки.
    NMS: залишає бокс з найвищим confidence, пригнічує решту з overlap > threshold.
    NMM: зважене середнє координат боксів, що перекриваються (вага = confidence).
    """
    if not detections:
        return []

    score_fn = _sahi_compute_iou if match_metric.upper() == "IOU" else _sahi_compute_ios
    sorted_dets = sorted(detections, key=lambda d: d["conf"], reverse=True)

    if postprocess_type.upper() == "NMS":
        kept = []
        suppressed = [False] * len(sorted_dets)
        for i, det_i in enumerate(sorted_dets):
            if suppressed[i]:
                continue
            kept.append(det_i)
            for j in range(i + 1, len(sorted_dets)):
                if suppressed[j]:
                    continue
                det_j = sorted_dets[j]
                if not class_agnostic and det_i["cls_id"] != det_j["cls_id"]:
                    continue
                score = score_fn(det_i["box_abs"], det_j["box_abs"])
                if score >= match_threshold:
                    suppressed[j] = True
        result = []
        for d in kept:
            result.append({"box": d["box"], "cls_id": d["cls_id"], "conf": d["conf"]})
        return result

    # NMM: зважене середнє
    merged_flags = [False] * len(sorted_dets)
    result = []
    for i, det_i in enumerate(sorted_dets):
        if merged_flags[i]:
            continue
        group = [det_i]
        for j in range(i + 1, len(sorted_dets)):
            if merged_flags[j]:
                continue
            det_j = sorted_dets[j]
            if not class_agnostic and det_i["cls_id"] != det_j["cls_id"]:
                continue
            score = score_fn(det_i["box_abs"], det_j["box_abs"])
            if score >= match_threshold:
                group.append(det_j)
                merged_flags[j] = True
        total_conf = sum(d["conf"] for d in group)
        if total_conf <= 0:
            result.append({"box": det_i["box"], "cls_id": det_i["cls_id"], "conf": det_i["conf"]})
            continue
        cx = sum(d["box"][0] * d["conf"] for d in group) / total_conf
        cy = sum(d["box"][1] * d["conf"] for d in group) / total_conf
        w = sum(d["box"][2] * d["conf"] for d in group) / total_conf
        h = sum(d["box"][3] * d["conf"] for d in group) / total_conf
        best_conf = group[0]["conf"]
        result.append({"box": (cx, cy, w, h), "cls_id": det_i["cls_id"], "conf": best_conf})
    return result


def _run_detection_sahi(rfdetr, frame: np.ndarray, frame_w: int, frame_h: int, imgsz: int, conf_threshold: float, max_det: int, classes_filter: list = None) -> list:
    """
    SAHI детекція: нарізка кадру на перекриваючі фрагменти → інференс на кожному →
    трансформація координат у простір повного кадру → (опціонально) інференс на повному кадрі →
    злиття через NMS/NMM.
    """
    slices = _sahi_generate_slices(
        frame_w, frame_h,
        SAHI_SLICE_WIDTH, SAHI_SLICE_HEIGHT,
        SAHI_OVERLAP_WIDTH_RATIO, SAHI_OVERLAP_HEIGHT_RATIO,
    )
    all_detections = []

    for (sx1, sy1, sx2, sy2) in slices:
        tile = frame[sy1:sy2, sx1:sx2]
        tile_w = sx2 - sx1
        tile_h = sy2 - sy1
        if tile_w <= 0 or tile_h <= 0:
            continue
        tile_dets = _run_detection_standard(rfdetr, tile, tile_w, tile_h, imgsz, conf_threshold, max_det, classes_filter)
        for d in tile_dets:
            cx_t, cy_t, w_t, h_t = d["box"]
            # нормалізовані координати тайлу → пікселі повного кадру
            cx_px = cx_t * tile_w + sx1
            cy_px = cy_t * tile_h + sy1
            w_px = w_t * tile_w
            h_px = h_t * tile_h
            # абсолютні координати для IoU/IoS
            abs_x1 = cx_px - w_px / 2
            abs_y1 = cy_px - h_px / 2
            abs_x2 = cx_px + w_px / 2
            abs_y2 = cy_px + h_px / 2
            # нормалізовані координати повного кадру
            cx_n = cx_px / frame_w
            cy_n = cy_px / frame_h
            w_n = w_px / frame_w
            h_n = h_px / frame_h
            all_detections.append({
                "box": (cx_n, cy_n, w_n, h_n),
                "cls_id": d["cls_id"],
                "conf": d["conf"],
                "box_abs": (abs_x1, abs_y1, abs_x2, abs_y2),
            })

    if SAHI_PERFORM_STANDARD_PRED:
        std_dets = _run_detection_standard(rfdetr, frame, frame_w, frame_h, imgsz, conf_threshold, max_det, classes_filter)
        for d in std_dets:
            cx, cy, w, h = d["box"]
            abs_x1 = (cx - w / 2) * frame_w
            abs_y1 = (cy - h / 2) * frame_h
            abs_x2 = (cx + w / 2) * frame_w
            abs_y2 = (cy + h / 2) * frame_h
            all_detections.append({
                "box": d["box"],
                "cls_id": d["cls_id"],
                "conf": d["conf"],
                "box_abs": (abs_x1, abs_y1, abs_x2, abs_y2),
            })

    return _sahi_merge_detections(
        all_detections,
        match_threshold=SAHI_POSTPROCESS_MATCH_THRESHOLD,
        match_metric=SAHI_POSTPROCESS_MATCH_METRIC,
        postprocess_type=SAHI_POSTPROCESS_TYPE,
        class_agnostic=SAHI_POSTPROCESS_CLASS_AGNOSTIC,
    )


def run_detection(rfdetr, frame: np.ndarray, frame_w: int, frame_h: int, imgsz: int, conf_threshold: float, max_det: int, classes_filter: list = None) -> list:
    """
    Запуск детекції на кадрі. При USE_SAHI=True використовує SAHI (нарізка на фрагменти),
    інакше — звичайний інференс на повному кадрі.
    Повертає список dict з ключами 'box' (cx, cy, w, h у 0–1), 'cls_id', 'conf'.
    """
    if USE_SAHI:
        return _run_detection_sahi(rfdetr, frame, frame_w, frame_h, imgsz, conf_threshold, max_det, classes_filter)
    return _run_detection_standard(rfdetr, frame, frame_w, frame_h, imgsz, conf_threshold, max_det, classes_filter)


def _vis_get_text_size(text: str) -> Tuple[int, int]:
    """Розмір тексту в стилі visualization.py."""
    return cv2.getTextSize(text, VIS_FONT, VIS_TEXT_SCALE, VIS_TEXT_THICKNESS)[0]


def _vis_draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = VIS_LABEL_BG_ALPHA,
) -> None:
    """Текст з напівпрозорим фоном та обводкою (стиль visualization.py)."""
    x, y = position
    tw, th = _vis_get_text_size(text)
    bg_x1 = max(0, x - VIS_LABEL_PADDING // 2)
    bg_y1 = max(0, y - th - VIS_LABEL_PADDING)
    bg_x2 = min(frame.shape[1], x + tw + VIS_LABEL_PADDING // 2)
    bg_y2 = min(frame.shape[0], y + VIS_LABEL_PADDING // 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), tuple(max(0, c - 50) for c in color), 1)
    cv2.putText(frame, text, (x, y - VIS_LABEL_PADDING // 4), VIS_FONT, VIS_TEXT_SCALE, VIS_TEXT_COLOR, VIS_TEXT_THICKNESS)


def _vis_draw_bbox(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]) -> None:
    """Подвійна рамка: основна + внутрішній підсвіт (стиль visualization.py)."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, VIS_BBOX_THICKNESS)
    inner = tuple(min(255, c + 30) for c in color)
    cv2.rectangle(frame, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), inner, VIS_BBOX_INNER_THICKNESS)


def draw_tracks(frame: np.ndarray, tracked: list, class_names: dict, colors: list) -> None:
    """
    Малювання боксів та міток у стилі visualization.py: подвійна рамка, напівпрозорий фон мітки,
    білий текст (HERSHEY_DUPLEX), два рядки — клас (confidence) та ID.
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
        cls_name = class_names.get(cls_id, f"cls_{cls_id}")
        track_id = getattr(obj, "track_id", "")
        confidence = getattr(obj, "confidence", None)
        conf_str = f"{confidence:.2f}" if confidence is not None else "—"

        # Порядок як у visualization.py: спочатку клас (conf), потім ID
        texts = [f"{cls_name} ({conf_str})", f"ID: {track_id}"]

        _vis_draw_bbox(frame, x1, y1, x2, y2, color)

        total_height = sum(_vis_get_text_size(t)[1] + VIS_LABEL_PADDING for t in texts)
        start_y = y1 - VIS_LABEL_MARGIN
        if start_y - total_height < 0:
            current_y = y1 + VIS_LINE_HEIGHT
        else:
            current_y = start_y

        for text in texts:
            tw, th = _vis_get_text_size(text)
            text_x = max(VIS_LABEL_PADDING, min(x1, w - tw - VIS_LABEL_PADDING))
            text_y = max(th + VIS_LABEL_PADDING, min(current_y, h - VIS_LABEL_PADDING))
            _vis_draw_text_with_background(frame, text, (text_x, text_y), color)
            if start_y - total_height < 0:
                current_y += VIS_LINE_HEIGHT
            else:
                current_y -= th + VIS_LABEL_PADDING


# =============================================================================
# BENCHMARK (профайлінг пайплайну)
# =============================================================================
def _benchmark_sync():
    """Синхронізація CUDA для точного виміру GPU операцій (лише при BENCHMARK_CUDA_SYNC)."""
    if BENCHMARK_CUDA_SYNC and torch.cuda.is_available():
        torch.cuda.synchronize()


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f}"


def _fmt_pct(part: float, total: float) -> str:
    if total <= 0:
        return "0.0%"
    return f"{part / total * 100:.1f}%"


class BenchmarkStats:
    """Збирач статистики по фазах для benchmark."""

    def __init__(self):
        self.frame_read_times = []
        self.detection_times = []
        self.tracker_update_times = []
        self.drawing_times = []
        self.frame_write_times = []
        self.detection_frame_count = 0
        self.tracking_only_frame_count = 0
        self.total_detections = 0
        self.total_tracks_drawn = 0
        self.warmup_time = 0.0
        self.model_load_time = 0.0

    def add_frame(self, read_t, detect_t, tracker_t, draw_t, write_t,
                  is_detection_frame, num_detections, num_tracks):
        self.frame_read_times.append(read_t)
        self.detection_times.append(detect_t)
        self.tracker_update_times.append(tracker_t)
        self.drawing_times.append(draw_t)
        self.frame_write_times.append(write_t)
        if is_detection_frame:
            self.detection_frame_count += 1
        else:
            self.tracking_only_frame_count += 1
        self.total_detections += num_detections
        self.total_tracks_drawn += num_tracks

    @property
    def n(self):
        return len(self.frame_read_times)


def _generate_benchmark_report(stats, pipeline_total: float, model_path: str, video_path: str, device_name: str, gpu_name: str, width: int, height: int, fps_video: float, total_frames: int, cfg: dict, has_tracker: bool) -> str:
    n = stats.n
    if n == 0: return "No frames processed.\n"
    phases = {"Frame Read (I/O)": stats.frame_read_times, "Detection (GPU)": stats.detection_times, "Tracker (CPU)": stats.tracker_update_times, "Drawing (CPU)": stats.drawing_times, "Frame Write (I/O)": stats.frame_write_times}
    phase_totals = {name: sum(arr) for name, arr in phases.items()}
    measured_total = sum(phase_totals.values())
    overhead = pipeline_total - measured_total
    effective_fps = n / pipeline_total if pipeline_total > 0 else 0
    det_only = [stats.detection_times[i] for i in range(n) if stats.detection_times[i] > 0]
    track_only_times = [stats.tracker_update_times[i] for i in range(n) if stats.detection_times[i] == 0]
    track_with_det_times = [stats.tracker_update_times[i] for i in range(n) if stats.detection_times[i] > 0]
    lines = [
        "=" * 72, "  BENCHMARK REPORT: TRACKING PIPELINE PROFILING", "=" * 72,
        f"  Date:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Model:            {MODEL_PATH}",
        f"  Device:           {device_name}",
        f"  GPU:              {gpu_name}",
        f"  Video:            {video_path}",
        f"  Resolution:       {width}x{height} @ {fps_video:.1f} FPS",
        f"  Frames processed: {n} / {total_frames}",
        f"  Detection every:  {DETECTION_INTERVAL} frames",
        f"  Tracker:          {'NanoTrack v' + NANOTRACK_VERSION if has_tracker else 'None'}",
        f"  Tracker resize:   {f'{NANO_IMAGE_RESIZE}x{NANO_IMAGE_RESIZE}' if NANO_IMAGE_RESIZE else 'OFF (full frame)'}",
        f"  SAHI:             {'ON' if USE_SAHI else 'OFF'}",
        f"  CUDA Sync:        {BENCHMARK_CUDA_SYNC}",
    ]
    if USE_SAHI:
        n_slices = len(_sahi_generate_slices(width, height, SAHI_SLICE_WIDTH, SAHI_SLICE_HEIGHT, SAHI_OVERLAP_WIDTH_RATIO, SAHI_OVERLAP_HEIGHT_RATIO))
        lines.extend(["", f"  SAHI config:", f"    Slice size:      {SAHI_SLICE_WIDTH}x{SAHI_SLICE_HEIGHT}", f"    Overlap:         {SAHI_OVERLAP_WIDTH_RATIO:.0%} x {SAHI_OVERLAP_HEIGHT_RATIO:.0%}", f"    Slices count:    {n_slices}", f"    Full-frame pred: {SAHI_PERFORM_STANDARD_PRED}", f"    Postprocess:     {SAHI_POSTPROCESS_TYPE} ({SAHI_POSTPROCESS_MATCH_METRIC}, thr={SAHI_POSTPROCESS_MATCH_THRESHOLD})", f"    Class agnostic:  {SAHI_POSTPROCESS_CLASS_AGNOSTIC}"])
    lines.extend(["", "  Inference config:"])
    for k in ["imgsz", "conf", "iou", "half", "max_det", "device"]:
        if k in cfg: lines.append(f"    {k:16s} = {cfg[k]}")
    lines.extend([
        "", "-" * 72, "  TOTAL PIPELINE", "-" * 72,
        f"  Wall-clock time:    {pipeline_total:.3f} s",
        f"  Measured phases:    {measured_total:.3f} s",
        f"  Overhead (loop):    {overhead:.3f} s ({_fmt_pct(overhead, pipeline_total)})",
        f"  Effective FPS:      {effective_fps:.1f}",
        f"  Avg frame time:     {_fmt_ms(pipeline_total / n)} ms",
        "", "-" * 72, "  TIME BREAKDOWN BY PHASE", "-" * 72,
        f"  {'Phase':<24s} {'Total (s)':>10s} {'% of total':>10s} {'Avg (ms)':>10s} {'Min (ms)':>10s} {'Max (ms)':>10s}",
        "  " + "-" * 68,
    ])
    for name, arr in phases.items():
        t_total = sum(arr); t_avg = t_total / n if n > 0 else 0; t_min = min(arr) if arr else 0; t_max = max(arr) if arr else 0
        lines.append(f"  {name:<24s} {t_total:>10.3f} {_fmt_pct(t_total, pipeline_total):>10s} {_fmt_ms(t_avg):>10s} {_fmt_ms(t_min):>10s} {_fmt_ms(t_max):>10s}")
    lines.extend(["  " + "-" * 68, f"  {'SUM (measured)':<24s} {measured_total:>10.3f} {_fmt_pct(measured_total, pipeline_total):>10s} {_fmt_ms(measured_total / n):>10s}", "", "-" * 72, "  DETECTION FRAMES (GPU inference)", "-" * 72, f"  Detection frames:     {stats.detection_frame_count} / {n}", f"  Tracking-only frames: {stats.tracking_only_frame_count} / {n}", f"  Total detections:     {stats.total_detections}"])
    if stats.detection_frame_count > 0: lines.append(f"  Avg detections/frame: {stats.total_detections / stats.detection_frame_count:.1f}")
    if det_only:
        det_total = phase_totals["Detection (GPU)"]
        det_fps = stats.detection_frame_count / det_total if det_total > 0 else 0
        lines.extend([f"  Detection time (det frames only):", f"    Avg: {_fmt_ms(sum(det_only) / len(det_only))} ms", f"    Min: {_fmt_ms(min(det_only))} ms", f"    Max: {_fmt_ms(max(det_only))} ms", f"  Detection speed:      {det_fps:.1f} inf/s  (інференсів на секунду)"])
    lines.extend(["", "-" * 72, "  TRACKER UPDATE (CPU)", "-" * 72, f"  Total tracks drawn:   {stats.total_tracks_drawn}"])
    if n > 0: lines.append(f"  Avg tracks/frame:     {stats.total_tracks_drawn / n:.1f}")
    track_total = phase_totals["Tracker (CPU)"]; track_avg_ms = (track_total / n * 1000) if n > 0 and track_total > 0 else 0; track_fps = 1000 / track_avg_ms if track_avg_ms > 0 else 0
    if track_with_det_times: lines.extend([f"  Tracker on det frames:", f"    Avg: {_fmt_ms(sum(track_with_det_times) / len(track_with_det_times))} ms"])
    if track_only_times: lines.extend([f"  Tracker on non-det frames:", f"    Avg: {_fmt_ms(sum(track_only_times) / len(track_only_times))} ms"])
    if track_total > 0: lines.append(f"  Tracker speed:        {track_avg_ms:.1f} ms/frame  ({track_fps:.0f} frames/s по трекеру)")
    io_total = phase_totals["Frame Read (I/O)"] + phase_totals["Frame Write (I/O)"]; compute_total = phase_totals["Detection (GPU)"] + phase_totals["Tracker (CPU)"]; compute_fps = n / compute_total if compute_total > 0 else 0
    lines.extend(["", "-" * 72, "  I/O", "-" * 72, f"  Total I/O time:   {io_total:.3f} s ({_fmt_pct(io_total, pipeline_total)})", "", "-" * 72, "  INITIALIZATION", "-" * 72, f"  Model load:       {_fmt_ms(stats.model_load_time)} ms", f"  Warmup:           {_fmt_ms(stats.warmup_time)} ms", "", "-" * 72, "  SUMMARY", "-" * 72, f"  Effective FPS (pipeline):   {effective_fps:.1f}  (з I/O та малюванням)", f"  Compute FPS (det+track):    {compute_fps:.1f}  (тільки детекція + трекінг, без I/O)", f"  CUDA Sync:                  {BENCHMARK_CUDA_SYNC}", "", "=" * 72])
    return "\n".join(lines)


def run_tracking(video_input_path: str, model_path: str = MODEL_PATH, detection_interval: int = DETECTION_INTERVAL, benchmark_mode: bool = False, output_base_dir: str = None) -> dict | None:
    import json
    if not video_input_path or not os.path.isfile(video_input_path): return None
    if not os.path.isfile(model_path): return None

    video_stem = Path(video_input_path).stem
    if output_base_dir is None:
        stem = Path(model_path).stem
        model_name = PROJECT_NAME
        if USE_SAHI:
            model_name += f"_sahi_{SAHI_SLICE_WIDTH}x{SAHI_SLICE_HEIGHT}"
        else:
            model_name += "_no_sahi"
        output_dir = os.path.join(BASE_DIR, "tracked_videos", model_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = output_base_dir

    output_path = os.path.join(output_dir, f"{video_stem}_tracked.mp4")
    log_path = os.path.join(output_dir, f"{video_stem}_tracked.txt")
    benchmark_log_path = os.path.join(output_dir, f"{video_stem}_benchmark.txt")

    benchmark_stats = BenchmarkStats() if benchmark_mode else None
    use_nano = os.path.isfile(NANOTRACK_BACKBONE) and os.path.isfile(NANOTRACK_NECKHEAD)
    t0 = time.perf_counter()
    rfdetr, num_classes, class_names, imgsz = load_rfdetr_model(model_path, MODEL_SIZE, class_names=CLASS_NAMES)
    if benchmark_stats is not None: benchmark_stats.model_load_time = time.perf_counter() - t0

    print(f"\n[INFO] Модель: {Path(model_path).name}")
    print(f"[INFO] Вхідний розмір: {imgsz}x{imgsz}")
    
    cfg = INFERENCE_CONFIG.copy()
    conf_threshold = cfg.get("conf_threshold", 0.25)
    max_det = cfg.get("max_det", 300)
    classes_filter = cfg.get("classes", None)
    
    is_trt = type(getattr(rfdetr, "model", None)).__name__ == "TRTInferenceModel"
    warmup_sz = imgsz if (getattr(rfdetr.model, "inference_model", None) is not None or is_trt) else 64
    dummy = np.zeros((warmup_sz, warmup_sz, 3), dtype=np.uint8)
    if benchmark_stats is not None: _benchmark_sync(); t0 = time.perf_counter()
    run_detection(rfdetr, dummy, warmup_sz, warmup_sz, imgsz=warmup_sz, conf_threshold=conf_threshold, max_det=max_det, classes_filter=None)
    if benchmark_stats is not None: _benchmark_sync(); benchmark_stats.warmup_time = time.perf_counter() - t0

    device = _get_device(); gpu_name = "N/A"
    try:
        device_name = f"{device}"
        if device.type == "cuda": gpu_name = torch.cuda.get_device_name(device.index); device_name = f"CUDA:{device.index} ({gpu_name})"
    except Exception:
        device_name = "TensorRT (GPU)"
    print(f"[INFO] Пристрій: {device_name}")

    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened(): return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if USE_SAHI:
        slices = _sahi_generate_slices(width, height, SAHI_SLICE_WIDTH, SAHI_SLICE_HEIGHT, SAHI_OVERLAP_WIDTH_RATIO, SAHI_OVERLAP_HEIGHT_RATIO)
        print(f"[INFO] Відео: {width}x{height} | SAHI: ON | Розмір слайса: {SAHI_SLICE_WIDTH}x{SAHI_SLICE_HEIGHT} | Оверлап: {SAHI_OVERLAP_WIDTH_RATIO} | Слайсів: {len(slices)}")
    else:
        print(f"[INFO] Відео: {width}x{height} | SAHI: OFF")
    
    frames_to_process = min(total_frames, BENCHMARK_MAX_FRAMES) if (benchmark_mode and BENCHMARK_MAX_FRAMES) else total_frames
    
    writer = None
    if not benchmark_mode or BENCHMARK_WRITE_VIDEO:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened(): cap.release(); return None

    tracker = None
    if use_nano:
        try: tracker = NanoTracker(class_names=class_names, backbone_path=NANOTRACK_BACKBONE, neckhead_path=NANOTRACK_NECKHEAD, max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD, confirm_threshold=CONFIRM_THRESHOLD, min_sec_stable=MIN_SEC_STABLE, use_optical_flow_predict=USE_OPTICAL_FLOW_PREDICT, optical_flow_threshold=OPTICAL_FLOW_THRESHOLD, adaptive_update=ADAPTIVE_UPDATE, adaptive_threshold=ADAPTIVE_THRESHOLD, enable_reid=ENABLE_REID, reid_buffer_time=REID_BUFFER_TIME, reid_iou_threshold=REID_IOU_THRESHOLD, reid_appearance_threshold=REID_APPEARANCE_THRESHOLD, reid_position_weight=REID_POSITION_WEIGHT, reid_appearance_weight=REID_APPEARANCE_WEIGHT, reid_size_weight=REID_SIZE_WEIGHT, reid_min_track_quality=REID_MIN_TRACK_QUALITY)
        except Exception: tracker = None

    frame_counter = 0; last_tracked = []; detection_counts = defaultdict(int); track_durations = {}; track_frames = {}
    coco_data = {"categories": [{"id": k, "name": v} for k, v in class_names.items()], "images": [], "annotations": []}; coco_ann_idx = 1
    start_time = time.perf_counter(); pbar = tqdm(total=frames_to_process if frames_to_process else None, unit="frame", desc="Benchmark" if benchmark_mode else "Track")
    
    try:
        while True:
            if benchmark_mode and frames_to_process and frame_counter >= frames_to_process: break
            t_read_s = time.perf_counter() if benchmark_stats else 0
            ret, frame = cap.read()
            t_read = time.perf_counter() - t_read_s if benchmark_stats else 0
            if not ret or frame is None: break
            frame_counter += 1; pbar.update(1)

            coco_data["images"].append({"id": frame_counter, "width": width, "height": height, "file_name": f"frame_{frame_counter:06d}.jpg"})
            tracker_frame = cv2.resize(frame, (NANO_IMAGE_RESIZE, NANO_IMAGE_RESIZE)) if NANO_IMAGE_RESIZE is not None and tracker is not None else frame
            
            is_det_frame = (frame_counter % detection_interval == 1 or frame_counter == 1)
            num_detections = 0; t_detect = 0.0

            if is_det_frame:
                if benchmark_stats: _benchmark_sync(); t_det_s = time.perf_counter()
                detections = run_detection(rfdetr, frame, width, height, imgsz=imgsz, conf_threshold=conf_threshold, max_det=max_det, classes_filter=classes_filter)
                if benchmark_stats: _benchmark_sync(); t_detect = time.perf_counter() - t_det_s
                num_detections = len(detections)
                for d in detections:
                    cid = d.get("cls_id")
                    if cid is not None: detection_counts[cid] += 1
                if benchmark_stats: t_track_s = time.perf_counter()
                if tracker is not None:
                    try: last_tracked = tracker.update(detections, tracker_frame)
                    except Exception: last_tracked = []
                else:
                    last_tracked = []
                    for d in detections:
                        cx, cy, w, h = d["box"]; bbox_n = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                        last_tracked.append(type("Obj", (), {"bbox": bbox_n, "cls_id": d.get("cls_id"), "track_id": "", "confidence": d.get("conf", 0.0)})())
                if benchmark_stats: t_track = time.perf_counter() - t_track_s
            else:
                t_track = 0.0
                if benchmark_stats: t_track_s = time.perf_counter()
                if tracker is not None:
                    try: last_tracked = tracker.update(None, tracker_frame)
                    except Exception: pass
                if benchmark_stats: t_track = time.perf_counter() - t_track_s

            num_tracks = len(last_tracked)
            if benchmark_stats: t_draw_s = time.perf_counter()
            for obj in last_tracked:
                cid = getattr(obj, "cls_id", None); cid = cid if cid is not None else 0; tid = getattr(obj, "track_id", "")
                if tid:
                    first = getattr(obj, "first_seen", None); last = getattr(obj, "last_seen", None)
                    if first is not None and last is not None: track_durations[(tid, cid)] = last - first
                    if (tid, cid) not in track_frames: track_frames[(tid, cid)] = [frame_counter, frame_counter]
                    else: track_frames[(tid, cid)][1] = frame_counter
                
                bbox = getattr(obj, "bbox", None)
                if bbox and len(bbox) == 4 and tid:
                    x1_n, y1_n, x2_n, y2_n = bbox
                    x1 = max(0, int(x1_n * width)); y1 = max(0, int(y1_n * height)); x2 = min(width, int(x2_n * width)); y2 = min(height, int(y2_n * height))
                    w_box = x2 - x1; h_box = y2 - y1
                    coco_data["annotations"].append({"id": coco_ann_idx, "image_id": frame_counter, "category_id": cid, "bbox": [x1, y1, w_box, h_box], "area": w_box * h_box, "track_id": tid, "iscrowd": 0})
                    coco_ann_idx += 1

            draw_tracks(frame, last_tracked, class_names, CLASS_COLORS)
            if benchmark_stats: t_draw = time.perf_counter() - t_draw_s
            if writer:
                if benchmark_stats: t_write_s = time.perf_counter()
                writer.write(frame)
                if benchmark_stats: t_write = time.perf_counter() - t_write_s
            else: t_write = 0.0 if benchmark_stats else 0
            if benchmark_stats: benchmark_stats.add_frame(t_read, t_detect, t_track, t_draw, t_write, is_det_frame, num_detections, num_tracks)
    finally:
        pbar.close(); cap.release()
        if writer: writer.release()

    elapsed_sec = time.perf_counter() - start_time
    fps_processed = frame_counter / elapsed_sec if elapsed_sec > 0 else 0.0

    coco_path = os.path.join(output_dir, f"{video_stem}_annotations_coco.json")
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)
    
    return {
        "output_path": output_path if writer else None,
        "video_path": video_input_path,
        "video_stem": video_stem,
        "benchmark_stats": benchmark_stats,
        "elapsed_sec": elapsed_sec,
        "total_detections": sum(detection_counts.values()) if detection_counts else 0,
        "total_tracks": len(track_frames),
        "track_frames": track_frames,
        "frames_processed": frame_counter,
        "fps_processed": fps_processed
    }

def _generate_global_reports(results: list, output_dir: str):
    bench_lines = ["# Усі Відео - Профайлінг (Benchmark)\n", "## Навігація"]
    for r in results:
        bs = r.get("benchmark_stats")
        if bs: bench_lines.append(f"- [{r['video_stem']}](#{r['video_stem'].lower().replace(' ', '-')})")
    
    total_time = sum(r.get('elapsed_sec', 0) for r in results); total_frames = sum(r.get('frames_processed', 0) for r in results)
    global_fps = total_frames / total_time if total_time > 0 else 0
    
    global_read_times = []; global_det_times = []; global_trk_times = []; global_draw_times = []; global_write_times = []
    global_det_frames = 0; global_trk_only_frames = 0; global_total_detections = 0; global_tracks_drawn = 0
    
    for r in results:
        bs = r.get("benchmark_stats")
        if bs and bs.n > 0:
            global_read_times.extend(bs.frame_read_times); global_det_times.extend(bs.detection_times); global_trk_times.extend(bs.tracker_update_times); global_draw_times.extend(bs.drawing_times); global_write_times.extend(bs.frame_write_times)
            global_det_frames += bs.detection_frame_count; global_trk_only_frames += bs.tracking_only_frame_count; global_total_detections += bs.total_detections; global_tracks_drawn += bs.total_tracks_drawn

    bench_lines.extend(["\n## Глобальна статистика (по всій папці)", f"- **Оброблено кадрів загалом:** {total_frames}", f"- **Загальний час:** {total_time:.2f} с", f"- **Середній FPS:** {global_fps:.2f} кадрів/с\n"])
    
    global_n = len(global_read_times)
    if global_n > 0:
        global_read = sum(global_read_times); global_det = sum(global_det_times); global_trk = sum(global_trk_times); global_draw = sum(global_draw_times); global_write = sum(global_write_times)
        measured_total = global_read + global_det + global_trk + global_draw + global_write
        overhead = total_time - measured_total
        
        def _fmt_ms(s): return f"{s*1000:.2f}"
        def _fmt_pct(p, t): return f"{p/t*100:.1f}%" if t>0 else "0.0%"
        
        det_only = [t for t in global_det_times if t > 0]
        trk_with_det = [global_trk_times[i] for i in range(global_n) if global_det_times[i] > 0]
        trk_only = [global_trk_times[i] for i in range(global_n) if global_det_times[i] == 0]
        
        io_total = global_read + global_write; compute_total = global_det + global_trk
        compute_fps = global_n / compute_total if compute_total > 0 else 0
        
        txt = ["```text", "-" * 72, "  TOTAL PIPELINE (GLOBAL)", "-" * 72, f"  Wall-clock time:    {total_time:.3f} s", f"  Measured phases:    {measured_total:.3f} s", f"  Overhead (loop):    {overhead:.3f} s ({_fmt_pct(overhead, total_time)})", f"  Effective FPS:      {global_fps:.1f}", f"  Avg frame time:     {_fmt_ms(total_time / global_n)} ms", "", "-" * 72, "  TIME BREAKDOWN BY PHASE", "-" * 72, f"  {'Phase':<24s} {'Total (s)':>10s} {'% of total':>10s} {'Avg (ms)':>10s} {'Min (ms)':>10s} {'Max (ms)':>10s}", "  " + "-" * 68]
        
        phases = {"Frame Read (I/O)": global_read_times, "Detection (GPU)": global_det_times, "Tracker (CPU)": global_trk_times, "Drawing (CPU)": global_draw_times, "Frame Write (I/O)": global_write_times}
        for name, arr in phases.items():
            t_tot = sum(arr); t_avg = t_tot / global_n; t_min = min(arr) if arr else 0; t_max = max(arr) if arr else 0
            txt.append(f"  {name:<24s} {t_tot:>10.3f} {_fmt_pct(t_tot, total_time):>10s} {_fmt_ms(t_avg):>10s} {_fmt_ms(t_min):>10s} {_fmt_ms(t_max):>10s}")
            
        txt.extend(["  " + "-" * 68, f"  {'SUM (measured)':<24s} {measured_total:>10.3f} {_fmt_pct(measured_total, total_time):>10s} {_fmt_ms(measured_total / global_n):>10s}", "", "-" * 72, "  DETECTION FRAMES (GPU inference)", "-" * 72, f"  Detection frames:     {global_det_frames} / {global_n}", f"  Tracking-only frames: {global_trk_only_frames} / {global_n}", f"  Total detections:     {global_total_detections}"])
        if global_det_frames > 0: txt.append(f"  Avg detections/frame: {global_total_detections / global_det_frames:.1f}")
        if det_only:
            det_fps = global_det_frames / sum(det_only) if sum(det_only) > 0 else 0
            txt.extend([f"  Detection time (det frames only):", f"    Avg: {_fmt_ms(sum(det_only) / len(det_only))} ms", f"    Min: {_fmt_ms(min(det_only))} ms", f"    Max: {_fmt_ms(max(det_only))} ms", f"  Detection speed:      {det_fps:.1f} inf/s  (інференсів на секунду)"])
            
        txt.extend(["", "-" * 72, "  TRACKER UPDATE (CPU)", "-" * 72, f"  Total tracks drawn:   {global_tracks_drawn}"])
        if global_n > 0: txt.append(f"  Avg tracks/frame:     {global_tracks_drawn / global_n:.1f}")
        
        track_avg_ms = (global_trk / global_n * 1000) if global_n > 0 else 0
        if trk_with_det: txt.extend([f"  Tracker on det frames:", f"    Avg: {_fmt_ms(sum(trk_with_det) / len(trk_with_det))} ms"])
        if trk_only: txt.extend([f"  Tracker on non-det frames:", f"    Avg: {_fmt_ms(sum(trk_only) / len(trk_only))} ms"])
        if global_trk > 0: txt.append(f"  Tracker speed:        {track_avg_ms:.1f} ms/frame  ({1000/track_avg_ms:.0f} frames/s)")
            
        txt.extend(["", "-" * 72, "  I/O", "-" * 72, f"  Total I/O time:   {io_total:.3f} s ({_fmt_pct(io_total, total_time)})", "", "-" * 72, "  SUMMARY", "-" * 72, f"  Effective FPS (pipeline):   {global_fps:.1f}  (з I/O та малюванням)", f"  Compute FPS (det+track):    {compute_fps:.1f}  (тільки детекція + трекінг, без I/O)", "```\n"])
        bench_lines.extend(txt)
    
    bench_lines.extend(["\n## Загальна статистика по кожному відео (Summary)", "| Відео | Час обробки (с) | FPS | Детекція (ms/фрейм) | Трекінг (ms/фрейм) |", "|---|---|---|---|---|"])
    for r in results:
        bs = r.get("benchmark_stats")
        if bs and bs.n > 0: bench_lines.append(f"| {r['video_stem']} | {r['elapsed_sec']:.2f} | {r['fps_processed']:.1f} | {sum(bs.detection_times) / bs.n * 1000:.1f} | {sum(bs.tracker_update_times) / bs.n * 1000:.1f} |")
    
    for r in results:
        bs = r.get("benchmark_stats")
        if not bs or bs.n == 0: continue
        bench_lines.extend([f"\n### {r['video_stem']}", "| Фаза | Загальний час (с) | % від загального | Середній (ms) | Мін (ms) | Макс (ms) |", "|---|---|---|---|---|---|"])
        measured_total = sum(bs.frame_read_times) + sum(bs.detection_times) + sum(bs.tracker_update_times) + sum(bs.drawing_times) + sum(bs.frame_write_times)
        phases = {"Frame Read (I/O)": bs.frame_read_times, "Detection (GPU)": bs.detection_times, "Tracker (CPU)": bs.tracker_update_times, "Drawing (CPU)": bs.drawing_times, "Frame Write (I/O)": bs.frame_write_times}
        for name, arr in phases.items():
            t_tot = sum(arr); pct = t_tot / measured_total * 100 if measured_total > 0 else 0; t_avg = t_tot / bs.n * 1000; t_min = min(arr) * 1000 if arr else 0; t_max = max(arr) * 1000 if arr else 0
            bench_lines.append(f"| {name} | {t_tot:.3f} | {pct:.1f}% | {t_avg:.1f} | {t_min:.1f} | {t_max:.1f} |")

    with open(os.path.join(output_dir, "benchmark.md"), "w", encoding="utf-8") as f: f.write("\n".join(bench_lines))

    det_lines = ["# Життєвий цикл об'єктів (Detection Result)\n"]
    for r in results:
        det_lines.append(f"### {r['video_stem']}")
        det_lines.append(f"- **Оригінальні детекції:** {r.get('total_detections', 0)}")
        det_lines.append(f"- **Загальна кількість треків:** {r.get('total_tracks', 0)}")
        frames = [frames_list[1] - frames_list[0] + 1 for frames_list in r.get("track_frames", {}).values()]
        if frames: det_lines.extend([f"- **Характеристики життя треків (в кадрах):**", f"  - Середнє: {sum(frames)/len(frames):.1f}", f"  - Мінімальне: {min(frames)}", f"  - Максимальне: {max(frames)}"])
        else: det_lines.append(f"- **Характеристики життя треків (в кадрах):** Немає треків")
        det_lines.append("\n")
        
    with open(os.path.join(output_dir, "detection_result.md"), "w", encoding="utf-8") as f: f.write("\n".join(det_lines))

def main():
    input_path = VIDEO_INPUT_PATH.strip()
    if not input_path: print("Задайте VIDEO_INPUT_PATH."); return
    path = Path(input_path)
    stem = Path(MODEL_PATH).stem
    model_name = PROJECT_NAME
    if USE_SAHI:
        model_name += f"_sahi_{SAHI_SLICE_WIDTH}x{SAHI_SLICE_HEIGHT}"
    else:
        model_name += "_no_sahi"
    output_dir = os.path.join(BASE_DIR, "tracked_videos", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            lines = f.readlines()
            config_lines = lines[35:161]
        with open(os.path.join(output_dir, "config.txt"), "w", encoding="utf-8") as f: f.writelines(config_lines)
    except: pass

    results = []
    videos = collect_videos_from_folder(input_path) if path.is_dir() else [input_path]
    if not videos:
        raise FileNotFoundError(f"Вказане джерело відео не знайдено або порожнє: {input_path}")
    
    for i, video_path in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Обробка: {Path(video_path).name}")
        out = run_tracking(video_path, model_path=MODEL_PATH, output_base_dir=output_dir, detection_interval=DETECTION_INTERVAL, benchmark_mode=BENCHMARK_MODE)
        if out: results.append(out)
            
    if results: _generate_global_reports(results, output_dir)

if __name__ == "__main__":
    main()
