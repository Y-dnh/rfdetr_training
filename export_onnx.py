#!/usr/bin/env python3
"""
RF-DETR ONNX Export Script
===========================
Експорт RF-DETR моделі в ONNX формат з автоматичним визначенням архітектури.

Автоматично визначає з checkpoint:
- Розмір моделі (Nano/Small/Medium/Base/Large/XLarge/2XLarge)
- Кількість класів
- Resolution

Usage:
    python export_onnx.py

Налаштуйте конфігурацію нижче перед запуском.
"""

import math
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch


# =============================================================================
# КОНФІГУРАЦІЯ
# =============================================================================

# Шлях до checkpoint (.pt або .pth)
CHECKPOINT_PATH = r"D:\rfdetr_dpsu_v8.pth"

# Папка для збереження ONNX (None = поруч з checkpoint)
OUTPUT_DIR = None

# Параметри експорту
SIMPLIFY = True                     # Спростити ONNX (onnxsim)
OPSET_VERSION = 17                  # ONNX opset version
BATCH_SIZE = 1                      # Batch size для експорту
DYNAMIC_BATCH = False               # Динамічний batch size
VERBOSE = False                     # Детальний вивід ONNX


# =============================================================================
# ТАБЛИЦЯ АРХІТЕКТУР RF-DETR
# =============================================================================
# Кожна модель однозначно ідентифікується за: (embedding_dim, patch_size, resolution)
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


# =============================================================================
# АВТОДЕТЕКТ МОДЕЛІ
# =============================================================================

def load_state_dict(checkpoint_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Завантажити state_dict з checkpoint файлу.
    
    Підтримує формати:
    - {'model': state_dict, ...} (training checkpoint .pt/.pth)
    - bare state_dict
    
    Returns:
        (state_dict, metadata)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    metadata = {}
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        metadata['epoch'] = checkpoint.get('epoch')
        metadata['best_map'] = checkpoint.get('best_map')
        metadata['class_names'] = checkpoint.get('class_names')
        # Формат оригінального rfdetr API
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
        Dict: model_size, num_classes, resolution, embedding_dim,
              hidden_dim, patch_size, dec_layers, pos_enc_size
    """
    # 1. Backbone embedding dim та patch size
    proj_key = 'backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight'
    if proj_key not in state_dict:
        raise ValueError(f"Ключ '{proj_key}' не знайдено. Це не RF-DETR модель?")
    
    proj_weight = state_dict[proj_key]
    embedding_dim = proj_weight.shape[0]   # 384 (ViT-S) або 768 (ViT-B)
    patch_size = proj_weight.shape[2]      # 14, 16, або 20
    
    # 2. Hidden dim та кількість класів
    hidden_dim = state_dict['class_embed.weight'].shape[1]
    num_classes = state_dict['class_embed.bias'].shape[0]
    
    # 3. Resolution з positional embeddings
    pos_key = 'backbone.0.encoder.encoder.embeddings.position_embeddings'
    num_tokens = state_dict[pos_key].shape[1]  # 1 (CLS) + num_patches
    pos_enc_size = int(math.sqrt(num_tokens - 1))
    resolution = pos_enc_size * patch_size
    
    # 4. Кількість decoder layers
    dec_layers = 0
    for key in state_dict:
        if key.startswith('transformer.decoder.layers.'):
            layer_num = int(key.split('.')[3])
            dec_layers = max(dec_layers, layer_num + 1)
    
    # 5. Ідентифікація моделі
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


# =============================================================================
# ГОЛОВНА ФУНКЦІЯ
# =============================================================================

def main():
    """Головна функція для експорту."""
    checkpoint_path = str(Path(CHECKPOINT_PATH).resolve())
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint не знайдено: {checkpoint_path}")
        return None
    
    # Визначити output_dir
    output_dir = OUTPUT_DIR
    if output_dir is None:
        output_dir = str(Path(checkpoint_path).parent)
    os.makedirs(output_dir, exist_ok=True)
    
    # ---- Аналіз checkpoint ----
    print("\n" + "=" * 70)
    print("RF-DETR ONNX EXPORT")
    print("=" * 70)
    
    print(f"\n[1/4] Аналіз checkpoint: {checkpoint_path}")
    state_dict, metadata = load_state_dict(checkpoint_path)
    arch = detect_architecture(state_dict)
    
    print(f"  Модель:      RF-DETR {arch['model_name']}")
    print(f"  Resolution:  {arch['resolution']}x{arch['resolution']}")
    print(f"  Classes:     {arch['num_classes']}")
    print(f"  Hidden dim:  {arch['hidden_dim']}")
    print(f"  Backbone:    DINOv2 {'Base' if arch['embedding_dim'] == 768 else 'Small'} "
          f"(dim={arch['embedding_dim']}, patch={arch['patch_size']})")
    print(f"  Decoder:     {arch['dec_layers']} layers")
    
    if metadata.get('class_names'):
        print(f"  Class names: {metadata['class_names']}")
    if metadata.get('best_map') is not None:
        print(f"  Best mAP:    {metadata['best_map']:.4f}")
    if metadata.get('epoch') is not None:
        print(f"  Epoch:       {metadata['epoch']}")
    
    if arch['model_size'] == 'unknown':
        print(f"\nERROR: Не вдалося визначити модель. Архітектура: "
              f"emb={arch['embedding_dim']}, patch={arch['patch_size']}, res={arch['resolution']}")
        return None
    
    # ---- Створити модель та завантажити ваги ----
    # RF-DETR API: num_classes + 1 = кількість виходів class_embed
    api_num_classes = arch['num_classes'] - 1
    
    print(f"\n[2/4] Створення RF-DETR {arch['model_name']} (num_classes={api_num_classes})...")
    
    rfdetr_cls = get_rfdetr_class(arch['model_size'])
    
    init_kwargs = {
        'pretrain_weights': checkpoint_path,
        'num_classes': api_num_classes,
    }
    if arch['model_size'] in ('xl', '2xl'):
        init_kwargs['accept_platform_model_license'] = True
    
    detector = rfdetr_cls(**init_kwargs)
    
    # ---- Експорт ----
    print(f"\n[3/4] Експорт в ONNX...")
    print(f"  Output:   {output_dir}")
    print(f"  Opset:    {OPSET_VERSION}")
    print(f"  Simplify: {SIMPLIFY}")
    print(f"  Batch:    {BATCH_SIZE} {'(dynamic)' if DYNAMIC_BATCH else '(static)'}")
    
    detector.model.export(
        output_dir=output_dir,
        simplify=SIMPLIFY,
        opset_version=OPSET_VERSION,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE,
    )
    
    # ---- Результат ----
    onnx_path = os.path.join(output_dir, 'inference_model.onnx')
    sim_path = os.path.join(output_dir, 'inference_model.sim.onnx')
    final_path = sim_path if SIMPLIFY and os.path.exists(sim_path) else onnx_path
    
    print(f"\n[4/4] Результат:")
    if os.path.exists(final_path):
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        print(f"  ONNX:   {final_path}")
        print(f"  Розмір: {size_mb:.1f} MB")
    
    print("\n" + "=" * 70)
    print("[OK] ONNX ЕКСПОРТ ЗАВЕРШЕНО")
    print("=" * 70)
    if os.path.exists(onnx_path):
        print(f"  Base:       {onnx_path}")
    if SIMPLIFY and os.path.exists(sim_path):
        print(f"  Simplified: {sim_path} (рекомендовано)")
    print("=" * 70 + "\n")
    
    return final_path


if __name__ == "__main__":
    main()
