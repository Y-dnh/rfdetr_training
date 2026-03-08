#!/usr/bin/env python3
"""
RF-DETR Export Script
=====================
Експорт RF-DETR моделі в ONNX та/або TensorRT формати.
Автоматичне визначення архітектури з checkpoint.

Обгортка навколо rfdetr/deploy/export.py з конфігурацією
у стилі train.py / val.py.

Usage:
    python export_onnx.py

Налаштуйте конфігурацію нижче перед запуском.
"""

import math
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch

from rfdetr.training import ExportConfig


# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ: ШЛЯХИ
# =============================================================================
# Та сама структура, що в train.py та val.py (Ultralytics-style).
PROJECT_NAME = "rfdetr_dpsu_v8"
EXPERIMENT_NAME = "baseline"          # Експеримент тренування, звідки брати модель
BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"
PROJECT_DIR = RUNS_DIR / PROJECT_NAME

# Модель: за замовчуванням best.pt з runs/.../<experiment>/weights/
CHECKPOINT_PATH = PROJECT_DIR / EXPERIMENT_NAME / "weights" / "best.pth"
OUTPUT_DIR = None                     # None = поруч з checkpoint


# =============================================================================
# КОНФІГУРАЦІЯ ЕКСПОРТУ (ONNX / TensorRT)
# =============================================================================
EXPORT_CONFIG = ExportConfig(
    # -------------------------------------------------------------------------
    # Основні налаштування
    # -------------------------------------------------------------------------
    enabled=True,                     # [True/False] Виконати експорт
    format='both',                    # ['onnx','tensorrt','both'] | Рекомендовано: 'onnx'

    # -------------------------------------------------------------------------
    # ONNX налаштування
    # -------------------------------------------------------------------------
    simplify=True,                    # [True/False] Спростити onnxsim (менший розмір) | Рекомендовано: True
    opset_version=17,                 # [≥11] ONNX opset версія | Рекомендовано: 16–17

    # -------------------------------------------------------------------------
    # Batch та точність
    # -------------------------------------------------------------------------
    dynamic_batch=False,              # [True/False] Динамічний batch при інференсі | False=фіксований
    batch_size=1,                     # [≥1] Batch size (ігнорується якщо dynamic_batch=True)
    half=True,                        # [True/False] FP16 (TensorRT) | False=FP32

    # -------------------------------------------------------------------------
    # Інше
    # -------------------------------------------------------------------------
    verbose=True,                     # [True/False] Детальний лог експорту

    # -------------------------------------------------------------------------
    # TensorRT (trtexec) — тільки якщо format='tensorrt' або 'both'
    # -------------------------------------------------------------------------
    trt_profile=False,                # [True/False] nsys профілювання при конвертації
    trt_dry_run=False,                # [True/False] Показати trtexec команду без виконання
)


# =============================================================================
# ТАБЛИЦЯ АРХІТЕКТУР RF-DETR
# =============================================================================
MODEL_ARCHITECTURES = {
    (384, 16, 384): 'n',     # Nano
    (384, 16, 512): 's',     # Small
    (384, 16, 576): 'm',     # Medium
    (384, 14, 560): 'b',     # Base
    (384, 16, 704): 'l',     # Large
    (768, 20, 700): 'xl',    # XLarge
    (768, 20, 880): '2xl',   # 2XLarge
}

MODEL_NAMES = {
    'n': 'Nano', 's': 'Small', 'm': 'Medium', 'b': 'Base',
    'l': 'Large', 'xl': 'XLarge', '2xl': '2XLarge',
}

MODEL_RESOLUTIONS = {
    'n': 384, 's': 512, 'm': 576, 'b': 560,
    'l': 704, 'xl': 700, '2xl': 880,
}


# =============================================================================
# АВТОДЕТЕКТ МОДЕЛІ
# =============================================================================

def load_state_dict(checkpoint_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Завантажити state_dict з checkpoint файлу.

    Підтримує формати:
    - {'model': state_dict, ...} (training checkpoint .pt/.pth)
    - bare state_dict
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    metadata: Dict[str, Any] = {}

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        metadata['epoch'] = checkpoint.get('epoch')
        metadata['best_map'] = checkpoint.get('best_map')
        metadata['class_names'] = checkpoint.get('class_names')
        if 'args' in checkpoint and hasattr(checkpoint['args'], 'class_names'):
            metadata['class_names'] = checkpoint['args'].class_names
    elif isinstance(checkpoint, dict) and any(k.startswith('backbone.') for k in checkpoint):
        state_dict = checkpoint
    else:
        raise ValueError(
            f"Невідомий формат checkpoint: {checkpoint_path}\n"
            f"Очікується dict з ключем 'model' або bare state_dict"
        )

    return state_dict, metadata


def detect_architecture(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Автоматично визначити архітектуру моделі з state_dict.

    Returns:
        Dict з ключами: model_size, model_name, num_classes, resolution,
                        embedding_dim, hidden_dim, patch_size, dec_layers, pos_enc_size
    """
    proj_key = 'backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight'
    if proj_key not in state_dict:
        raise ValueError(f"Ключ '{proj_key}' не знайдено. Це не RF-DETR модель?")

    proj_weight = state_dict[proj_key]
    embedding_dim = proj_weight.shape[0]
    patch_size = proj_weight.shape[2]

    hidden_dim = state_dict['class_embed.weight'].shape[1]
    num_classes = state_dict['class_embed.bias'].shape[0]

    pos_key = 'backbone.0.encoder.encoder.embeddings.position_embeddings'
    num_tokens = state_dict[pos_key].shape[1]
    pos_enc_size = int(math.sqrt(num_tokens - 1))
    resolution = pos_enc_size * patch_size

    dec_layers = 0
    for key in state_dict:
        if key.startswith('transformer.decoder.layers.'):
            layer_num = int(key.split('.')[3])
            dec_layers = max(dec_layers, layer_num + 1)

    arch_key = (embedding_dim, patch_size, resolution)
    model_size = MODEL_ARCHITECTURES.get(arch_key, 'unknown')

    return {
        'model_size': model_size,
        'model_name': MODEL_NAMES.get(model_size, f'Unknown ({arch_key})'),
        'num_classes': num_classes,
        'resolution': resolution,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'patch_size': patch_size,
        'dec_layers': dec_layers,
        'pos_enc_size': pos_enc_size,
    }


def get_rfdetr_class(model_size: str):
    """Повернути відповідний клас RFDETR для заданого розміру моделі."""
    from rfdetr.detr import (
        RFDETRNano, RFDETRSmall, RFDETRMedium,
        RFDETRBase, RFDETRLarge,
    )

    class_map = {
        'n': RFDETRNano,
        's': RFDETRSmall,
        'm': RFDETRMedium,
        'b': RFDETRBase,
        'l': RFDETRLarge,
    }

    if model_size in ('xl', '2xl'):
        from rfdetr.platform.models import RFDETRXLarge, RFDETR2XLarge
        class_map['xl'] = RFDETRXLarge
        class_map['2xl'] = RFDETR2XLarge

    if model_size not in class_map:
        raise ValueError(
            f"Невідомий розмір моделі: '{model_size}'\n"
            f"Доступні: {list(class_map.keys())}"
        )

    return class_map[model_size]


def print_arch_info(arch: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Вивести інформацію про архітектуру моделі."""
    print(f"  Модель:      RF-DETR {arch['model_name']}")
    print(f"  Resolution:  {arch['resolution']}x{arch['resolution']}")
    print(f"  Classes:     {arch['num_classes']}")
    print(f"  Hidden dim:  {arch['hidden_dim']}")
    backbone_name = 'Base' if arch['embedding_dim'] == 768 else 'Small'
    print(f"  Backbone:    DINOv2 {backbone_name} "
          f"(dim={arch['embedding_dim']}, patch={arch['patch_size']})")
    print(f"  Decoder:     {arch['dec_layers']} layers")
    if metadata.get('class_names'):
        print(f"  Class names: {metadata['class_names']}")
    if metadata.get('best_map') is not None:
        print(f"  Best mAP:    {metadata['best_map']:.4f}")
    if metadata.get('epoch') is not None:
        print(f"  Epoch:       {metadata['epoch']}")


# =============================================================================
# TensorRT КОНВЕРТАЦІЯ (Python API)
# =============================================================================

def _build_trt_engine(
    onnx_path: str,
    fp16: bool = True,
    verbose: bool = False,
    workspace_mb: int = 4096,
) -> Optional[str]:
    """
    Конвертація ONNX → TensorRT engine через Python API.
    Не потребує trtexec у PATH.
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("\n  ERROR: tensorrt не встановлено.")
        print("  pip install --no-cache-dir tensorrt tensorrt-cu12")
        print("  ONNX модель збережена — можна конвертувати пізніше.")
        return None

    engine_path = onnx_path.replace('.onnx', '.engine')

    severity = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(severity)

    print(f"  TensorRT:  {trt.__version__}")
    print(f"  Workspace: {workspace_mb} MB")

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16:      enabled (GPU supports fast FP16)")
    elif fp16:
        print("  FP16:      requested but GPU has no fast FP16, using FP32")

    print("\n  Будування engine (це може зайняти кілька хвилин)...")

    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except Exception as e:
        print(f"\n  ERROR: Не вдалося зібрати engine: {e}")
        return None

    if engine_bytes is None:
        print("\n  ERROR: builder.build_serialized_network повернув None.")
        print("  Можливо, не вистачає GPU пам'яті або ONNX модель несумісна.")
        return None

    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)

    print(f"  Engine збережено: {engine_path}")
    return engine_path


# =============================================================================
# ГОЛОВНА ФУНКЦІЯ
# =============================================================================

def main():
    """Головна функція для експорту моделі."""
    from rfdetr.deploy.export import (
        export_onnx as _export_onnx_raw,
        onnx_simplify as _onnx_simplify,
        make_infer_image,
    )

    checkpoint_path = str(Path(CHECKPOINT_PATH).resolve())

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint не знайдено: {checkpoint_path}")
        return None

    output_dir = OUTPUT_DIR or str(Path(checkpoint_path).parent)
    os.makedirs(output_dir, exist_ok=True)

    export_format = EXPORT_CONFIG.format
    do_onnx = export_format in ('onnx', 'both')
    do_trt = export_format in ('tensorrt', 'both')

    format_label = {'onnx': 'ONNX', 'tensorrt': 'TensorRT', 'both': 'ONNX + TensorRT'}
    total_steps = 2 + int(do_onnx or do_trt) + int(do_trt)

    # ---- Banner ----
    print("\n" + "=" * 70)
    print(f"RF-DETR EXPORT — {format_label.get(export_format, export_format)}")
    print("=" * 70)

    # ---- 1. Аналіз checkpoint ----
    step = 1
    print(f"\n[{step}/{total_steps}] Аналіз checkpoint: {checkpoint_path}")
    state_dict, metadata = load_state_dict(checkpoint_path)
    arch = detect_architecture(state_dict)
    print_arch_info(arch, metadata)

    if arch['model_size'] == 'unknown':
        print(f"\nERROR: Не вдалося визначити модель. Архітектура: "
              f"emb={arch['embedding_dim']}, patch={arch['patch_size']}, res={arch['resolution']}")
        return None

    # ---- 2. Створення моделі ----
    step += 1
    api_num_classes = arch['num_classes'] - 1
    print(f"\n[{step}/{total_steps}] Створення RF-DETR {arch['model_name']} "
          f"(num_classes={api_num_classes})...")

    rfdetr_cls = get_rfdetr_class(arch['model_size'])
    init_kwargs = {
        'pretrain_weights': checkpoint_path,
        'num_classes': api_num_classes,
    }
    if arch['model_size'] in ('xl', '2xl'):
        init_kwargs['accept_platform_model_license'] = True

    detector = rfdetr_cls(**init_kwargs)
    model = detector.model.model
    model.eval()
    model.cpu()

    resolution = arch['resolution']
    input_tensors = make_infer_image(
        infer_dir=None,
        shape=(resolution, resolution),
        batch_size=EXPORT_CONFIG.batch_size,
        device='cpu',
    )

    input_names = ['input']
    output_names = ['dets', 'labels']
    dynamic_axes = None
    if EXPORT_CONFIG.dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch'},
            'dets': {0: 'batch'},
            'labels': {0: 'batch'},
        }

    onnx_path: Optional[str] = None
    engine_path: Optional[str] = None

    # ---- 3. ONNX Export ----
    if do_onnx or do_trt:
        step += 1
        print(f"\n[{step}/{total_steps}] Експорт в ONNX...")
        print(f"  Output:   {output_dir}")
        print(f"  Opset:    {EXPORT_CONFIG.opset_version}")
        print(f"  Simplify: {EXPORT_CONFIG.simplify}")
        print(f"  Batch:    {EXPORT_CONFIG.batch_size} "
              f"{'(dynamic)' if EXPORT_CONFIG.dynamic_batch else '(static)'}")

        onnx_path = _export_onnx_raw(
            output_dir=output_dir,
            model=model,
            input_names=input_names,
            input_tensors=input_tensors,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=EXPORT_CONFIG.verbose,
            opset_version=EXPORT_CONFIG.opset_version,
        )

        if EXPORT_CONFIG.simplify:
            try:
                onnx_path = _onnx_simplify(
                    onnx_dir=onnx_path,
                    input_names=input_names,
                    input_tensors=input_tensors,
                    force=True,
                )
            except Exception as e:
                print(f"  УВАГА: Спрощення ONNX не вдалось: {e}")

    # ---- 4. TensorRT Export ----
    if do_trt and onnx_path:
        step += 1
        print(f"\n[{step}/{total_steps}] Конвертація в TensorRT engine...")
        print(f"  Source:  {onnx_path}")
        print(f"  FP16:    {EXPORT_CONFIG.half}")

        engine_path = _build_trt_engine(
            onnx_path=onnx_path,
            fp16=EXPORT_CONFIG.half,
            verbose=EXPORT_CONFIG.verbose,
        )

    # ---- Результат ----
    print("\n" + "=" * 70)
    print("[OK] ЕКСПОРТ ЗАВЕРШЕНО")
    print("=" * 70)

    if onnx_path and os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  ONNX:      {onnx_path}  ({size_mb:.1f} MB)")

        base_onnx = os.path.join(output_dir, 'inference_model.onnx')
        sim_onnx = os.path.join(output_dir, 'inference_model.sim.onnx')
        if EXPORT_CONFIG.simplify and os.path.exists(sim_onnx) and os.path.exists(base_onnx):
            print(f"  ONNX base: {base_onnx}")
            print(f"  ONNX sim:  {sim_onnx} (рекомендовано)")

    if engine_path and os.path.exists(engine_path):
        size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"  Engine:    {engine_path}  ({size_mb:.1f} MB)")

    if not do_onnx and do_trt:
        print("  (ONNX збережено як проміжний артефакт для TensorRT)")

    print("=" * 70 + "\n")

    return onnx_path or engine_path


if __name__ == "__main__":
    main()
