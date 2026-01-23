#!/usr/bin/env python3
"""
Test to debug box formats in validation visualization.

Prints sample values from:
1. Ground truth labels (from dataset)
2. Model predictions (raw output)
3. Postprocessed predictions
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rfdetr.training import (
    setup_seed,
    AugmentationConfig,
    TrainingConfig,
    ModelConfig,
    RFDETRTrainer,
)
from rfdetr.training.dataset import build_dataset, collate_fn
from torch.utils.data import DataLoader


def test_box_formats():
    """Test to understand box formats."""
    setup_seed(42)
    
    # Paths
    dataset_dir = project_root / "tests" / "test_dataset"
    
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configs
    aug_config = AugmentationConfig(imgsz=672)
    
    # Build validation dataset
    val_dataset = build_dataset(
        dataset_dir=dataset_dir,
        split='valid',
        augmentation_config=aug_config,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    # Build model directly using RFDETRBase
    from rfdetr import RFDETRBase
    rfdetr = RFDETRBase()
    model = rfdetr.model.model.to(device)
    model.eval()
    
    print(f"\nModel bbox_reparam: {model.bbox_reparam}")
    
    # Get one batch
    images, targets, _ = next(iter(val_loader))
    images = images.to(device)
    targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
               for k, v in t.items()} for t in targets]
    
    print("\n" + "=" * 70)
    print("BOX FORMAT ANALYSIS")
    print("=" * 70)
    
    # 1. Analyze ground truth labels
    print("\n1. GROUND TRUTH LABELS (from dataset after transforms)")
    print("-" * 50)
    for i, target in enumerate(targets):
        boxes = target['boxes']
        print(f"\nImage {i}:")
        print(f"  orig_size: {target['orig_size'].tolist()}")
        print(f"  num_boxes: {len(boxes)}")
        if len(boxes) > 0:
            print(f"  boxes shape: {boxes.shape}")
            print(f"  boxes dtype: {boxes.dtype}")
            print(f"  boxes min: {boxes.min().item():.6f}")
            print(f"  boxes max: {boxes.max().item():.6f}")
            print(f"  Sample boxes (first 3):")
            for j, box in enumerate(boxes[:3]):
                print(f"    Box {j}: {box.tolist()}")
                if boxes.max() <= 1.0:
                    print(f"           -> Looks like NORMALIZED cxcywh")
                else:
                    print(f"           -> Looks like PIXEL coordinates")
    
    # 2. Run model forward pass
    print("\n2. MODEL PREDICTIONS (raw output)")
    print("-" * 50)
    
    from rfdetr.util.misc import NestedTensor
    b, c, h, w = images.shape
    mask = torch.zeros((b, h, w), dtype=torch.bool, device=device)
    samples = NestedTensor(images, mask)
    
    with torch.no_grad():
        outputs = model(samples, targets)
    
    pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes]
    pred_boxes = outputs['pred_boxes']    # [B, num_queries, 4]
    
    print(f"\npred_logits shape: {pred_logits.shape}")
    print(f"pred_boxes shape: {pred_boxes.shape}")
    print(f"pred_boxes dtype: {pred_boxes.dtype}")
    print(f"pred_boxes min: {pred_boxes.min().item():.6f}")
    print(f"pred_boxes max: {pred_boxes.max().item():.6f}")
    
    # Check if values are in [0,1] range
    if pred_boxes.min() >= 0 and pred_boxes.max() <= 1:
        print(f"\n-> pred_boxes ARE in [0, 1] range - NORMALIZED")
    elif pred_boxes.min() >= -0.5 and pred_boxes.max() <= 1.5:
        print(f"\n-> pred_boxes mostly in [0, 1] with some overflow")
    else:
        print(f"\n-> pred_boxes NOT in [0, 1] range!")
        print(f"   This might need sigmoid or other transformation!")
    
    # Show some high-confidence predictions
    print("\n  Predictions analysis:")
    probs = pred_logits.sigmoid()
    max_scores, labels = probs.max(dim=-1)
    
    for i in range(len(images)):
        print(f"\n  Image {i}:")
        print(f"    All scores: min={max_scores[i].min():.4f}, max={max_scores[i].max():.4f}")
        
        # Check box coordinate statistics
        boxes_i = pred_boxes[i]
        print(f"    Box cx: min={boxes_i[:, 0].min():.4f}, max={boxes_i[:, 0].max():.4f}")
        print(f"    Box cy: min={boxes_i[:, 1].min():.4f}, max={boxes_i[:, 1].max():.4f}")
        print(f"    Box w:  min={boxes_i[:, 2].min():.4f}, max={boxes_i[:, 2].max():.4f}")
        print(f"    Box h:  min={boxes_i[:, 3].min():.4f}, max={boxes_i[:, 3].max():.4f}")
        
        # Show top 5 predictions by score
        topk = torch.topk(max_scores[i], min(5, len(max_scores[i])))
        print(f"    Top 5 predictions:")
        for j, (score, idx) in enumerate(zip(topk.values, topk.indices)):
            box = boxes_i[idx]
            cx, cy, bw, bh = box.tolist()
            lbl = labels[i][idx].item()
            print(f"      #{j+1}: score={score:.4f}, label={lbl}, cx={cx:.4f}, cy={cy:.4f}, w={bw:.4f}, h={bh:.4f}")
    
    # 3. Postprocessed results
    print("\n3. POSTPROCESSED PREDICTIONS")
    print("-" * 50)
    
    from rfdetr.models.lwdetr import PostProcess
    postprocessor = PostProcess(num_select=100)
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    
    results = postprocessor(outputs, orig_target_sizes)
    
    for i, result in enumerate(results):
        print(f"\nImage {i}:")
        print(f"  orig_size: {orig_target_sizes[i].tolist()}")
        boxes = result['boxes']
        scores = result['scores']
        
        # Filter by confidence
        high_conf = scores > 0.3
        if high_conf.sum() > 0:
            boxes = boxes[high_conf][:5]
            scores = scores[high_conf][:5]
            print(f"  High-confidence boxes (postprocessed, in orig_size coords):")
            for j, (box, score) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = box.tolist()
                print(f"    Score {score:.3f}: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
        else:
            print("  No predictions with score > 0.3")
    
    # 4. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
GT boxes range: [{targets[0]['boxes'].min():.4f}, {targets[0]['boxes'].max():.4f}]
Pred boxes range: [{pred_boxes.min():.4f}, {pred_boxes.max():.4f}]

Expected for visualization:
- GT: normalized cxcywh [0, 1]
- Pred: normalized cxcywh [0, 1]

If pred_boxes max >> 1, we need to apply sigmoid!
If pred_boxes min << 0, we need to apply sigmoid!
""")
    
    print("=" * 70)


if __name__ == "__main__":
    test_box_formats()
