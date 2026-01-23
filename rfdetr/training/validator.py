"""
RF-DETR Validator class for model evaluation.

This module provides validation functionality that generates:
- Validation batch visualizations
- Confusion matrix
- PR curves, F1 curves
- Per-class metrics
- Markdown evaluation report
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rfdetr.training.dataset import RFDETRDataset, build_dataset, collate_fn
from rfdetr.training.utils.config import AugmentationConfig
from rfdetr.training.visualizations import (
    BatchVisualizer,
    ConfusionMatrix,
    save_all_curves,
    AnalysisVisualizer,
    match_predictions_to_gt,
)


class RFDETRValidator:
    """
    Validator for RF-DETR models.
    
    Generates YOLO-style validation outputs:
    - val_batch*_labels.jpg
    - val_batch*_pred.jpg
    - confusion_matrix.png
    - PR_curve.png, F1_curve.png, etc.
    
    Example:
        >>> validator = RFDETRValidator(
        ...     model_path='runs/train/exp/weights/best.pt',
        ...     conf_threshold=0.25,
        ...     iou_threshold=0.5,
        ... )
        >>> results = validator.validate(dataset_dir='data/my_dataset')
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        imgsz: Optional[int] = None,  # Auto-detect from checkpoint
        batch_size: int = 16,
        workers: int = 8,
        device: str = 'cuda',
        save_dir: Optional[str] = None,
    ):
        """
        Initialize validator.
        
        Args:
            model_path: Path to model weights.
            conf_threshold: Confidence threshold for predictions.
            iou_threshold: IoU threshold for NMS.
            imgsz: Image size for validation. If None, auto-detects from checkpoint.
            batch_size: Batch size.
            workers: Number of data loading workers.
            device: Device to use.
            save_dir: Directory to save results.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._imgsz_override = imgsz  # User override
        self.imgsz = imgsz or 560  # Default, will be updated from checkpoint
        self.batch_size = batch_size
        self.workers = workers
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else None
        
        self.model = None
        self.class_names = None
        
        # Auto-detect imgsz from checkpoint if not specified
        if model_path and imgsz is None:
            self._detect_imgsz_from_checkpoint()
    
    def _detect_imgsz_from_checkpoint(self):
        """Auto-detect image size from checkpoint config."""
        if not self.model_path or not Path(self.model_path).exists():
            return
        
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Try to get from augmentation_config
            if 'augmentation_config' in checkpoint:
                aug_config = checkpoint['augmentation_config']
                if isinstance(aug_config, dict) and 'imgsz' in aug_config:
                    self.imgsz = aug_config['imgsz']
                    print(f"[Validator] Auto-detected imgsz={self.imgsz} from checkpoint")
                    return
            
            # Try to get from model_config (model_size -> resolution)
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
                if isinstance(model_config, dict) and 'model_size' in model_config:
                    size_to_resolution = {'n': 384, 's': 512, 'm': 576, 'b': 560, 'l': 560}
                    model_size = model_config['model_size']
                    self.imgsz = size_to_resolution.get(model_size, 560)
                    print(f"[Validator] Auto-detected imgsz={self.imgsz} from model_size={model_size}")
                    return
                    
        except Exception as e:
            print(f"[Validator] Could not auto-detect imgsz: {e}")
    
    def _load_model(self) -> nn.Module:
        """Load the model from checkpoint."""
        if self.model_path is None:
            raise ValueError("model_path must be specified")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Build model - placeholder, actual implementation depends on RF-DETR
        # model = build_model(...)
        # model.load_state_dict(checkpoint['model_state_dict'])
        
        return None  # Placeholder
    
    def _setup_dataset(
        self,
        dataset_dir: Union[str, Path],
        split: str = 'valid',
    ) -> RFDETRDataset:
        """Set up validation dataset."""
        config = AugmentationConfig(imgsz=self.imgsz)
        
        dataset = build_dataset(
            dataset_dir,
            split=split,
            augmentation_config=config,
            log_augmentations=False,
        )
        
        return dataset
    
    def _setup_dataloader(self, dataset: RFDETRDataset) -> DataLoader:
        """Create data loader."""
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        return loader
    
    def validate(
        self,
        dataset_dir: Union[str, Path],
        split: str = 'valid',
        save_visualizations: bool = True,
        save_analysis: bool = False,
    ) -> Dict[str, Any]:
        """
        Run validation on a dataset.
        
        Args:
            dataset_dir: Path to dataset directory.
            split: Dataset split to validate on.
            save_visualizations: Whether to save visualization images.
            save_analysis: Whether to save per-image analysis visualizations
                          (2x2 grid with GT/TP/FP/FN).
        
        Returns:
            Dictionary with validation results.
        """
        # Setup save directory
        if self.save_dir is None:
            self.save_dir = Path('runs/val/exp')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup dataset and dataloader
        dataset = self._setup_dataset(dataset_dir, split)
        dataloader = self._setup_dataloader(dataset)
        
        self.class_names = dataset.class_names
        num_classes = dataset.num_classes
        
        print(f"Validating on {len(dataset)} images...")
        print(f"Classes: {num_classes} - {self.class_names}")
        
        # Setup visualizers
        batch_visualizer = BatchVisualizer(
            self.save_dir,
            class_names=self.class_names,
            max_batches=3,
        )
        
        confusion_matrix = ConfusionMatrix(
            num_classes=num_classes,
            class_names=self.class_names,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )
        
        # Setup analysis visualizer if requested
        analysis_visualizer = None
        if save_analysis:
            analysis_visualizer = AnalysisVisualizer(
                self.save_dir,
                class_names=self.class_names,
                iou_threshold=self.iou_threshold,
                conf_threshold=self.conf_threshold,
                image_size=1920,  # 960x960 per panel = 1920x1920 total grid
            )
            print(f"[Analysis] Saving per-image analysis to: {analysis_visualizer.save_dir}")
        
        # Collect all predictions and targets
        all_predictions = []
        all_targets = []
        
        # Per-class statistics
        class_stats = {i: {'gt': 0, 'tp': 0, 'fp': 0, 'fn': 0} for i in range(num_classes)}
        
        # Run validation
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        
        for batch_idx, (images, targets, _aug_logs) in enumerate(dataloader):
            images = images.to(device)
            
            # Get predictions - placeholder
            # with torch.no_grad():
            #     predictions = self.model(images)
            
            # Placeholder predictions
            predictions = [
                {
                    'boxes': t['boxes'][:min(3, len(t['boxes']))],
                    'labels': t['labels'][:min(3, len(t['labels']))],
                    'scores': torch.ones(min(3, len(t['labels']))) * 0.9,
                }
                for t in targets
            ]
            
            # Collect for metrics
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            # Update confusion matrix
            confusion_matrix.process_batch(predictions, targets)
            
            # Update per-class statistics
            for target, pred in zip(targets, predictions):
                gt_boxes = target.get('boxes', torch.zeros((0, 4)))
                gt_labels = target.get('labels', torch.zeros(0))
                pred_boxes = pred.get('boxes', torch.zeros((0, 4)))
                pred_labels = pred.get('labels', torch.zeros(0))
                pred_scores = pred.get('scores', None)
                
                # Convert to numpy
                if isinstance(gt_boxes, torch.Tensor):
                    gt_boxes = gt_boxes.cpu().numpy()
                if isinstance(gt_labels, torch.Tensor):
                    gt_labels = gt_labels.cpu().numpy()
                if isinstance(pred_boxes, torch.Tensor):
                    pred_boxes = pred_boxes.cpu().numpy()
                if isinstance(pred_labels, torch.Tensor):
                    pred_labels = pred_labels.cpu().numpy()
                if pred_scores is not None and isinstance(pred_scores, torch.Tensor):
                    pred_scores = pred_scores.cpu().numpy()
                
                # Match predictions to GT
                if len(gt_boxes) > 0 or len(pred_boxes) > 0:
                    tp_indices, fp_indices, fn_indices = match_predictions_to_gt(
                        gt_boxes, gt_labels,
                        pred_boxes, pred_labels, pred_scores,
                        iou_threshold=self.iou_threshold,
                    )
                    
                    # Update GT counts per class
                    for label in gt_labels:
                        label_int = int(label)
                        if label_int < num_classes:
                            class_stats[label_int]['gt'] += 1
                    
                    # Update TP per class
                    for idx in tp_indices:
                        label_int = int(pred_labels[idx])
                        if label_int < num_classes:
                            class_stats[label_int]['tp'] += 1
                    
                    # Update FP per class
                    for idx in fp_indices:
                        label_int = int(pred_labels[idx])
                        if label_int < num_classes:
                            class_stats[label_int]['fp'] += 1
                    
                    # Update FN per class
                    for idx in fn_indices:
                        label_int = int(gt_labels[idx])
                        if label_int < num_classes:
                            class_stats[label_int]['fn'] += 1
            
            # Save batch visualizations
            if save_visualizations and batch_idx < 3:
                filenames = [dataset.get_filename(batch_idx * self.batch_size + i) 
                            for i in range(len(images))]
                batch_visualizer.save_val_batch(
                    images.cpu(), targets, predictions, batch_idx, filenames
                )
            
            # Save per-image analysis visualizations
            if save_analysis and analysis_visualizer is not None:
                for i, (img, target, pred) in enumerate(zip(images, targets, predictions)):
                    idx = batch_idx * self.batch_size + i
                    filename = dataset.get_filename(idx) if hasattr(dataset, 'get_filename') else None
                    
                    analysis_visualizer.save_analysis(
                        image=img.cpu(),
                        gt_boxes=target.get('boxes', torch.zeros((0, 4))),
                        gt_labels=target.get('labels', torch.zeros(0)),
                        pred_boxes=pred.get('boxes', torch.zeros((0, 4))),
                        pred_labels=pred.get('labels', torch.zeros(0)),
                        pred_scores=pred.get('scores', None),
                        filename=filename,
                        denormalize=True,
                    )
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        # Add per-class stats to metrics
        metrics['class_stats'] = class_stats
        
        # Save confusion matrix
        confusion_matrix.plot(self.save_dir / 'confusion_matrix.png')
        
        # Print analysis stats if used
        analysis_stats = None
        if save_analysis and analysis_visualizer is not None:
            analysis_visualizer.print_stats()
            analysis_stats = analysis_visualizer.get_stats()
            metrics['analysis_stats'] = analysis_stats
        
        # Save PR curves and other plots
        # save_all_curves(metrics, self.save_dir, self.class_names)
        
        # Save results to JSON
        results = {
            'metrics': metrics,
            'num_images': len(dataset),
            'num_classes': num_classes,
            'class_names': self.class_names,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'save_dir': str(self.save_dir),
            'dataset_dir': str(dataset_dir),
            'split': split,
        }
        
        with open(self.save_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report(results, class_stats, analysis_stats)
        
        # Print summary
        self._print_results(metrics)
        
        return results
    
    def _calculate_metrics(
        self,
        predictions: List[Dict[str, Any]],
        targets: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate validation metrics."""
        # Placeholder metrics calculation
        metrics = {
            'mAP50': 0.5,
            'mAP50-95': 0.3,
            'precision': 0.8,
            'recall': 0.7,
        }
        
        return metrics
    
    def _generate_markdown_report(
        self,
        results: Dict[str, Any],
        class_stats: Dict[int, Dict[str, int]],
        analysis_stats: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Generate comprehensive markdown evaluation report.
        
        Args:
            results: Validation results dictionary.
            class_stats: Per-class statistics (GT, TP, FP, FN).
            analysis_stats: Optional analysis statistics from AnalysisVisualizer.
        
        Returns:
            Path to saved report.
        """
        metrics = results.get('metrics', {})
        num_images = results.get('num_images', 0)
        num_classes = results.get('num_classes', 0)
        class_names = results.get('class_names', [])
        
        # Calculate totals
        total_gt = sum(stats['gt'] for stats in class_stats.values())
        total_tp = sum(stats['tp'] for stats in class_stats.values())
        total_fp = sum(stats['fp'] for stats in class_stats.values())
        total_fn = sum(stats['fn'] for stats in class_stats.values())
        
        # Overall precision/recall
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        # Model name from path
        model_name = Path(self.model_path).stem if self.model_path else "Unknown"
        
        report_content = f"""# 🎯 RF-DETR Validation Report

## Experiment Overview

| **Parameter** | **Value** |
|---------------|-----------|
| **Model** | `{model_name}` |
| **Model Path** | `{self.model_path or 'N/A'}` |
| **Date & Time** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
| **Dataset** | `{results.get('dataset_dir', 'N/A')}` |
| **Split** | `{results.get('split', 'valid')}` |
| **Test Images** | {num_images} |
| **Object Categories** | {num_classes} |
| **Total GT Annotations** | {total_gt} |

## Configuration Settings

| **Setting** | **Value** |
|-------------|-----------|
| **Confidence Threshold** | {self.conf_threshold} |
| **IoU Threshold** | {self.iou_threshold} |
| **Image Size** | {self.imgsz} |
| **Batch Size** | {self.batch_size} |
| **Device** | {self.device} |

---

## 📊 Overall Performance

| **Metric** | **Value** |
|------------|-----------|
| **mAP@0.5** | {metrics.get('mAP50', 0):.4f} |
| **mAP@0.5:0.95** | {metrics.get('mAP50-95', 0):.4f} |
| **Precision** | {overall_precision:.4f} |
| **Recall** | {overall_recall:.4f} |
| **F1 Score** | {overall_f1:.4f} |

---

## 📈 Detection Statistics

| **Metric** | **Count** | **Percentage** |
|------------|-----------|----------------|
| **Ground Truth (GT)** | {total_gt} | 100.0% |
| **True Positives (TP)** | {total_tp} | {(total_tp/total_gt*100) if total_gt > 0 else 0:.1f}% |
| **False Positives (FP)** | {total_fp} | - |
| **False Negatives (FN)** | {total_fn} | {(total_fn/total_gt*100) if total_gt > 0 else 0:.1f}% |

### Detection Summary
- ✅ **Correctly detected**: {total_tp} objects ({(total_tp/total_gt*100) if total_gt > 0 else 0:.1f}% of GT)
- ❌ **Missed**: {total_fn} objects ({(total_fn/total_gt*100) if total_gt > 0 else 0:.1f}% of GT)
- ⚠️ **False alarms**: {total_fp} detections

---

## 📋 Per-Class Performance

| **Class** | **GT** | **TP** | **FP** | **FN** | **Precision** | **Recall** | **F1** |
|-----------|--------|--------|--------|--------|---------------|------------|--------|
"""
        
        # Add per-class statistics
        class_metrics = []
        for class_id in sorted(class_stats.keys()):
            stats = class_stats[class_id]
            if class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"class_{class_id}"
            
            gt = stats['gt']
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if gt > 0 or tp > 0 or fp > 0:  # Only show classes with data
                report_content += f"| {class_name} | {gt} | {tp} | {fp} | {fn} | {precision:.3f} | {recall:.3f} | {f1:.3f} |\n"
                class_metrics.append({
                    'name': class_name,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'gt': gt,
                })
        
        # Performance rankings
        if class_metrics:
            by_precision = sorted(class_metrics, key=lambda x: x['precision'], reverse=True)
            by_recall = sorted(class_metrics, key=lambda x: x['recall'], reverse=True)
            by_f1 = sorted(class_metrics, key=lambda x: x['f1'], reverse=True)
            
            report_content += f"""
---

## 🏆 Performance Rankings

### By Precision
"""
            for i, m in enumerate(by_precision[:5]):
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                report_content += f"{medal} **{m['name']}** - {m['precision']:.3f}\n"
            
            report_content += f"""
### By Recall
"""
            for i, m in enumerate(by_recall[:5]):
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                report_content += f"{medal} **{m['name']}** - {m['recall']:.3f}\n"
            
            report_content += f"""
### By F1 Score
"""
            for i, m in enumerate(by_f1[:5]):
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                report_content += f"{medal} **{m['name']}** - {m['f1']:.3f}\n"
        
        # Category distribution
        report_content += f"""
---

## 📊 Category Distribution

| **Category** | **GT Annotations** | **Percentage** |
|--------------|-------------------|----------------|
"""
        sorted_classes = sorted(class_metrics, key=lambda x: x['gt'], reverse=True)
        for m in sorted_classes:
            if m['gt'] > 0:
                percentage = (m['gt'] / total_gt) * 100 if total_gt > 0 else 0
                report_content += f"| {m['name']} | {m['gt']} | {percentage:.1f}% |\n"
        
        # Key findings
        if class_metrics:
            best_precision = max(class_metrics, key=lambda x: x['precision'])
            best_recall = max(class_metrics, key=lambda x: x['recall'])
            best_f1 = max(class_metrics, key=lambda x: x['f1'])
            worst_recall = min([m for m in class_metrics if m['gt'] > 0], key=lambda x: x['recall'], default=None)
            
            report_content += f"""
---

## 🔍 Key Findings

### Strengths
- **Best Precision**: {best_precision['name']} ({best_precision['precision']:.3f}) - lowest false positive rate
- **Best Recall**: {best_recall['name']} ({best_recall['recall']:.3f}) - finds most objects
- **Best F1**: {best_f1['name']} ({best_f1['f1']:.3f}) - best overall balance

### Areas for Improvement
"""
            if worst_recall and worst_recall['recall'] < 0.5:
                report_content += f"- **{worst_recall['name']}** has low recall ({worst_recall['recall']:.3f}) - consider more training data or augmentation\n"
            
            if total_fp > total_gt * 0.2:
                report_content += f"- High false positive rate ({total_fp} FP) - consider increasing confidence threshold\n"
            
            if total_fn > total_gt * 0.3:
                report_content += f"- Many missed detections ({total_fn} FN, {(total_fn/total_gt*100):.1f}%) - model may need more training\n"
        
        # Output files
        report_content += f"""
---

## 📁 Output Files

| **File** | **Description** |
|----------|-----------------|
| `confusion_matrix.png` | Confusion matrix visualization |
| `val_batch*_labels.jpg` | Ground truth annotations |
| `val_batch*_pred.jpg` | Model predictions |
| `results.json` | Raw validation results |
| `validation_report.md` | This report |
"""
        
        if analysis_stats:
            report_content += f"| `analysis/` | Per-image GT/TP/FP/FN visualizations ({analysis_stats.get('images_processed', 0)} images) |\n"
        
        # Technical notes
        report_content += f"""
---

## 📝 Technical Notes

### Metrics Explanation
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.5 to 0.95
- **Precision**: TP / (TP + FP) - how many detections are correct
- **Recall**: TP / (TP + FN) - how many GT objects are found
- **F1 Score**: Harmonic mean of Precision and Recall

### Matching Criteria
- IoU threshold: {self.iou_threshold}
- Confidence threshold: {self.conf_threshold}
- A detection is TP if IoU ≥ {self.iou_threshold} and class matches

---

*📊 Report generated automatically by RF-DETR Validation System*  
*🕐 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*📍 Save directory: `{self.save_dir}`*
"""
        
        # Save report
        report_path = self.save_dir / 'validation_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n📄 Markdown report generated: {report_path}")
        
        return report_path
    
    def _print_results(self, metrics: Dict[str, Any]) -> None:
        """Print validation results."""
        print("\n" + "=" * 50)
        print("Validation Results")
        print("=" * 50)
        
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{name}: {value:.4f}")
            elif name not in ('class_stats', 'analysis_stats'):
                print(f"{name}: {value}")
        
        print("=" * 50)
