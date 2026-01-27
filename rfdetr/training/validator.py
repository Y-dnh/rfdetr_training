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
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rfdetr.training.dataset import RFDETRDataset, build_dataset, collate_fn
from rfdetr.training.utils.config import AugmentationConfig
from rfdetr.util import box_ops
from pycocotools.cocoeval import COCOeval
from rfdetr.training.visualizations import (
    BatchVisualizer,
    ConfusionMatrix,
    save_all_curves,
    AnalysisVisualizer,
    match_predictions_to_gt,
)
from rfdetr.detr import RFDETRBase, RFDETRLarge, RFDETRNano, RFDETRSmall, RFDETRMedium


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
        model_size: Optional[str] = None,  # Manual override
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        imgsz: Optional[int] = None,  # Auto-detect from checkpoint
        batch_size: int = 16,
        workers: int = 8,
        device: str = 'cuda',
        save_dir: Optional[str] = None,
        half: bool = True,
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
            half: Whether to use half precision (FP16).
        """
        self.model_path = model_path
        self.model_size_override = model_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._imgsz_override = imgsz  # User override
        self.imgsz = imgsz or 560  # Default, will be updated from checkpoint
        self.batch_size = batch_size
        self.workers = workers
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else None
        self.half = half
        
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
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
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
    
    def _load_model(self, num_classes_override: int = None):
        """Load the RF-DETR model from checkpoint."""
        if self.model_path is None:
            raise ValueError("model_path must be specified")
        
        print(f"[Validator] Loading model from {self.model_path}...")
        
        # Load checkpoint to determine model size
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        # Determine model size from checkpoint or override
        model_size = self.model_size_override or 'm'  # default medium or override
        
        if 'model_config' in checkpoint and not self.model_size_override:
            model_config = checkpoint['model_config']
            if isinstance(model_config, dict) and 'model_size' in model_config:
                model_size = model_config['model_size']
        elif not self.model_size_override:
            print(f"[Validator] Warning: 'model_config' not found in checkpoint. Using default model_size='{model_size}'")
            print(f"[Validator] Checkpoint keys: {list(checkpoint.keys())}")
        else:
             print(f"[Validator] Using manual model_size='{model_size}'")
        
        # Get number of classes
        num_classes = num_classes_override or 80  # Prefer override from dataset
        
        if num_classes_override is None:
            if 'num_classes' in checkpoint:
                num_classes = checkpoint['num_classes']
            elif 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
                if isinstance(model_config, dict) and 'num_classes' in model_config:
                    num_classes = model_config['num_classes']
            elif 'args' in checkpoint and hasattr(checkpoint['args'], 'num_classes'):
                 # Try to get from args if available
                 num_classes = checkpoint['args'].num_classes

        
        # Get class names
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        
        print(f"[Validator] Model size: {model_size}, num_classes: {num_classes}")
        
        # Create model based on size
        # Create model based on size with error handling
        try:
            if model_size in ['n', 'nano']:
                self.rfdetr = RFDETRNano(pretrain_weights=self.model_path, num_classes=num_classes)
            elif model_size in ['s', 'small']:
                self.rfdetr = RFDETRSmall(pretrain_weights=self.model_path, num_classes=num_classes)
            elif model_size in ['m', 'medium']:
                self.rfdetr = RFDETRMedium(pretrain_weights=self.model_path, num_classes=num_classes)
            elif model_size in ['l', 'large']:
                self.rfdetr = RFDETRLarge(pretrain_weights=self.model_path, num_classes=num_classes)
            else:
                self.rfdetr = RFDETRBase(pretrain_weights=self.model_path, num_classes=num_classes)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print("\n" + "!" * 80)
                print(f"[ERROR] Size mismatch detected for model_size='{model_size}'!")
                print("The checkpoint seems to have a different architecture than selected.")
                print(f"Original error: {e}")
                print("-" * 80)
                print("SUGGESTION: Try forcing a different model size (e.g., 'base' or 'small').")
                print("You can modify the default 'model_size' in validator.py (around line 136).")
                print("!" * 80 + "\n")
                raise e
            else:
                raise e
        
        # Move model to device
        self.rfdetr.model.device = torch.device(self.device)
        self.rfdetr.model.model = self.rfdetr.model.model.to(self.device)
        self.rfdetr.model.model.eval()
        
        # Optimize for inference
        self.rfdetr.optimize_for_inference(compile=False, batch_size=self.batch_size)
        
        # Verify device
        param_device = next(self.rfdetr.model.model.parameters()).device
        print(f"[Validator] Model loaded successfully! Weights are on: {param_device}")
        return self.rfdetr
    
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
    ) -> Dict[str, Any]:
        """
        Run validation on a dataset.
        
        Args:
            dataset_dir: Path to dataset directory.
            split: Dataset split to validate on.
            save_visualizations: Whether to save visualizations (batch + per-image GT/TP/FP/FN).
        
        Returns:
            Dictionary with validation results.
        """
        # Setup save directory
        if self.save_dir is None:
            self.save_dir = Path('runs')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup dataset and dataloader
        dataset = self._setup_dataset(dataset_dir, split)
        dataloader = self._setup_dataloader(dataset)
        
        self.class_names = dataset.class_names
        num_classes = dataset.num_classes
        
        print(f"Validating on {len(dataset)} images...")
        print(f"Classes: {num_classes} - {self.class_names}")
        
        # Load model (pass num_classes from dataset)
        self._load_model(num_classes_override=num_classes)
        
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
            imgsz=self.imgsz,
        )
        
        # Setup analysis visualizer if requested
        analysis_visualizer = None
        if save_visualizations:
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
        
        # FPS tracking (pure inference only)
        total_inference_time = 0.0
        total_images = 0
        
        pbar = tqdm(dataloader, desc="Validating", unit="batch")
        for batch_idx, (images, targets, _aug_logs) in enumerate(pbar):
            images = images.to(device)
            batch_size = images.size(0)
            
            # --- INFERENCE TIMING START ---
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_start = time.perf_counter()
            
            # Real RF-DETR inference (low-level to bypass checks and re-normalization)
            with torch.no_grad():
                # Forward pass
                if self.rfdetr._is_optimized_for_inference:
                   outputs = self.rfdetr.model.inference_model(images)
                else:
                   outputs = self.rfdetr.model.model(images)

                # Postprocess
                # We need original sizes, but since we are validating fixed size batches, 
                # we can assume "original size" is the current tensor size for metric calculation relative to input
                # OR retrieve 'orig_size' from targets if available. 
                # RF-DETR postprocess needs target_sizes to scale boxes back to original image
                
                # Use current inference image size for postprocessing to match visualization coordinate system
                # This ensures predictions are in the same scale as the validation image tensor (e.g. 576x576)
                # matching the normalized GT boxes which AnalysisVisualizer scales to this size.
                target_sizes = torch.tensor([images.shape[-2:]] * batch_size, device=device)

                # Handle tuple output from inference model (boxes, logits)
                if isinstance(outputs, tuple):
                    outputs = {
                        'pred_boxes': outputs[0],
                        'pred_logits': outputs[1]
                    }

                results = self.rfdetr.model.postprocess(outputs, target_sizes=target_sizes)

            # Convert to our format
            predictions = []
            for result in results:
                # RF-DETR postprocess returns dict with 'scores', 'labels', 'boxes'
                # Filter by threshold
                scores = result['scores']
                mask = scores > self.conf_threshold
                
                pred = {
                    'boxes': result['boxes'][mask].cpu(),
                    'labels': result['labels'][mask].cpu(),
                    'scores': result['scores'][mask].cpu()
                }
                predictions.append(pred)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = time.perf_counter() - inference_start
            # --- INFERENCE TIMING END ---
            
            total_inference_time += inference_time
            total_images += batch_size
            
            # Update progress bar with current FPS
            current_fps = batch_size / inference_time if inference_time > 0 else 0
            pbar.set_postfix({'FPS': f'{current_fps:.1f}'})
            
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
                
                # Convert GT boxes from normalized cxcywh to absolute xyxy (same as predictions)
                if isinstance(gt_boxes, torch.Tensor):
                    gt_boxes = gt_boxes.cpu()
                    if len(gt_boxes) > 0 and gt_boxes.max() <= 1.0:
                        # Convert normalized cxcywh to xyxy and scale to imgsz
                        gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
                        gt_boxes = gt_boxes * self.imgsz
                    gt_boxes = gt_boxes.numpy()
                
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
            if save_visualizations and analysis_visualizer is not None:
                for i, (img, target, pred) in enumerate(zip(images, targets, predictions)):
                    idx = batch_idx * self.batch_size + i
                    filename = dataset.get_filename(idx) if hasattr(dataset, 'get_filename') else None
                    
                    analysis_visualizer.save_visualization(
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
        metrics = self._calculate_metrics(all_predictions, all_targets, dataset)
        
        # Add per-class stats to metrics
        metrics['class_stats'] = class_stats
        
        # Save confusion matrix
        confusion_matrix.plot(self.save_dir / 'confusion_matrix.png')
        
        # Collect analysis stats if used
        analysis_stats = None
        if save_visualizations and analysis_visualizer is not None:
            analysis_stats = analysis_visualizer.get_stats()
            metrics['analysis_stats'] = analysis_stats
        
        # Save PR curves and other plots
        self._generate_curves(all_predictions, all_targets, num_classes)
        
        # Calculate average FPS
        avg_fps = total_images / total_inference_time if total_inference_time > 0 else 0
        avg_latency_ms = (total_inference_time / total_images) * 1000 if total_images > 0 else 0
        
        print(f"\n📊 Inference Speed: {avg_fps:.1f} FPS ({avg_latency_ms:.2f} ms/image)")
        
        # Prepare results dict for report generation
        results = {
            'metrics': metrics,
            'num_images': len(dataset),
            'num_classes': num_classes,
            'inference_fps': round(avg_fps, 2),
            'split': split,
            'dataset_dir': str(dataset_dir),
        }
        
        # Generate markdown report
        self._generate_markdown_report(results, class_stats, analysis_stats)
        
        # --- Compact Final Output ---
        print("\n" + "="*50)
        print("RF-DETR VALIDATION SUMMARY")
        print("="*50)
        
        # 1. Analysis Stats (extended)
        if analysis_stats:
            print(f"Images:      {len(dataset)}")
            print(f"Total GT:    {analysis_stats.get('total_gt', 0)}")
            print(f"TP:          {analysis_stats.get('total_tp', 0)}")
            print(f"FP:          {analysis_stats.get('total_fp', 0)}")
            print(f"FN:          {analysis_stats.get('total_fn', 0)}")
            # print metrics from analysis (simple P/R)
            p_simple = analysis_stats.get('precision', 0)
            r_simple = analysis_stats.get('recall', 0)
            print(f"P (Approx):  {p_simple:.4f}")
            print(f"R (Approx):  {r_simple:.4f}")
            print("-" * 50)

        # 2. Main Metrics (COCO)
        print(f"mAP@0.5:     {metrics['mAP50']:.4f}")
        print(f"mAP@0.5:0.95:{metrics['mAP50-95']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1 Score:    {metrics['f1']:.4f}")
        
        # 3. Speed
        print("-" * 50)
        print(f"Speed:       {avg_fps:.1f} FPS ({avg_latency_ms:.2f} ms/img)")
        
        # 4. Saved Results
        print("-" * 50)
        print(f"JSON Results: {self.save_dir / 'results.json'}")
        print(f"MD Report:    {self.save_dir / 'validation_report.md'}") 
        print("="*50 + "\n")
        
        # self._print_results(metrics) # OLD - remove or comment out
        
        return results
    
    def _calculate_metrics(
        self,
        predictions: List[Dict[str, Any]],
        targets: List[Dict[str, Any]],
        dataset: Optional[RFDETRDataset] = None,
    ) -> Dict[str, float]:
        """
        Calculate validation metrics using COCO API.
        
        Args:
            predictions: List of prediction dicts.
            targets: List of target dicts.
            dataset: Dataset object (needed for COCO API).
            
        Returns:
            Dictionary with metrics (mAP50, mAP50-95, precision, recall, f1).
        """
        if dataset is None or not hasattr(dataset, 'coco'):
            print("Warning: Dataset does not support COCO eval. Using simplified metrics.")
            return self._calculate_simple_metrics(predictions, targets)
            
        # Prepare predictions for COCO eval
        coco_results = []
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            # Image ID (needed for COCO)
            image_id = target['image_id'].item() if isinstance(target['image_id'], torch.Tensor) else target['image_id']
            if isinstance(image_id, list):
                image_id = image_id[0]
                
            # Predictions
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']
            
            if isinstance(pred_boxes, torch.Tensor):
                pred_boxes = pred_boxes.cpu().numpy()
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.cpu().numpy()
            if isinstance(pred_labels, torch.Tensor):
                pred_labels = pred_labels.cpu().numpy()
                
            if len(pred_boxes) == 0:
                continue
                
            # Get original size for this image
            orig_h, orig_w = target['orig_size'].tolist() if isinstance(target['orig_size'], torch.Tensor) else target['orig_size']
            
            # Current size used for inference (imgsz)
            cur_h, cur_w = self.imgsz, self.imgsz
            
            # Rescale boxes to original image size (Un-Letterbox)
            # 1. Calculate the scale and padding that was applied
            r = min(cur_h / orig_h, cur_w / orig_w)
            
            # Compute padding (same logic as LetterBox)
            # new_unpad = int(round(orig_w * r)), int(round(orig_h * r))
            pad_w = (cur_w - int(round(orig_w * r))) / 2
            pad_h = (cur_h - int(round(orig_h * r))) / 2
            
            # 2. Subtract padding
            formatted_boxes = pred_boxes.copy()
            formatted_boxes[:, 0] -= pad_w
            formatted_boxes[:, 2] -= pad_w
            formatted_boxes[:, 1] -= pad_h
            formatted_boxes[:, 3] -= pad_h
            
            # 3. Scale back to original size
            formatted_boxes[:, :4] /= r
            
            # 4. Clip to original image bounds
            formatted_boxes[:, 0] = np.clip(formatted_boxes[:, 0], 0, orig_w)
            formatted_boxes[:, 2] = np.clip(formatted_boxes[:, 2], 0, orig_w)
            formatted_boxes[:, 1] = np.clip(formatted_boxes[:, 1], 0, orig_h)
            formatted_boxes[:, 3] = np.clip(formatted_boxes[:, 3], 0, orig_h)
            
            # Convert xyxy to xywh for COCO
            formatted_boxes[:, 2] -= formatted_boxes[:, 0]
            formatted_boxes[:, 3] -= formatted_boxes[:, 1]
            
            for box, score, label in zip(formatted_boxes, pred_scores, pred_labels):
                if label < 0: continue
                # Convert label index to COCO category ID
                if hasattr(dataset, 'label_to_cat_id'):
                    cat_id = dataset.label_to_cat_id.get(int(label), int(label))
                else:
                    cat_id = int(label)
                    
                coco_results.append({
                    'image_id': int(image_id),
                    'category_id': int(cat_id),
                    'bbox': box.tolist(),
                    'score': float(score)
                })
        
        # Run COCO Eval
        if not coco_results:
            print("No predictions generated.")
            return {'mAP50': 0.0, 'mAP50-95': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
        coco_dt = dataset.coco.loadRes(coco_results)
        coco_eval = COCOeval(dataset.coco, coco_dt, 'bbox')
        
        # Suppress print
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        
        # Extract metrics
        map50_95 = coco_eval.stats[0]
        map50 = coco_eval.stats[1]
        
        # We can also get simplified P/R from our own counter (which is robust to scaling issues if we matched in 576 space)
        # But COCO eval is authoritative. 
        # Note: COCOeval doesn't output single P/R values easily (it has arrays).
        # Let's use our manual calculation for P/R/F1 to be consistent with what we see in visualization
        # and use COCO for mAP.
        
        simple_metrics = self._calculate_simple_metrics(predictions, targets)
        
        metrics = {
            'mAP50': map50,
            'mAP50-95': map50_95,
            'precision': simple_metrics['precision'],
            'recall': simple_metrics['recall'],
            'f1': simple_metrics['f1'],
        }
        
        return metrics

    def _calculate_simple_metrics(self, predictions, targets):
        """Legacy simplified metric calculation (P/R/F1 only)."""
        total_tp = 0
        total_fp = 0
        total_gt = 0
        
        for pred, target in zip(predictions, targets):
            # Predictions are absolute xyxy (imgsz scale)
            pred_boxes = pred['boxes'].numpy() if isinstance(pred['boxes'], torch.Tensor) else pred['boxes']
            pred_labels = pred['labels'].numpy() if isinstance(pred['labels'], torch.Tensor) else pred['labels']
            pred_scores = pred['scores'].numpy() if isinstance(pred['scores'], torch.Tensor) else pred['scores']
            
            # Targets (norm cxcywh) -> absolute xyxy (imgsz scale)
            gt_boxes = target['boxes']
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu()
            
            if len(gt_boxes) > 0 and gt_boxes.max() <= 1.0:
                 # Convert normalized cxcywh to xyxy
                 gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
                 # Scale to imgsz
                 gt_boxes = gt_boxes * self.imgsz
            
            gt_boxes = gt_boxes.numpy()
            gt_labels = target['labels'].cpu().numpy() if isinstance(target['labels'], torch.Tensor) else target['labels']
            
            total_gt += len(gt_boxes)
            
            tp_idx, fp_idx, fn_idx = match_predictions_to_gt(
                gt_boxes, gt_labels,
                pred_boxes, pred_labels, pred_scores,
                iou_threshold=self.iou_threshold
            )
            total_tp += len(tp_idx)
            total_fp += len(fp_idx)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
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
        
        # Use self.class_names which is populated from dataset
        class_names = self.class_names or []
        
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
        
        # Inference speed
        inference_fps = results.get('inference_fps', 0)
        latency_ms = 1000 / inference_fps if inference_fps > 0 else 0
        
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
| **Half Precision (FP16)** | {self.half} |

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
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                })
        
        # Category distribution
        if class_metrics:
            report_content += """
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
        
        # Inference speed section (always add if we have data)
        report_content += f"""
---

## ⚡ Inference Speed

| **Metric** | **Value** |
|------------|-----------|
| **FPS** | {inference_fps:.1f} |
| **Latency** | {latency_ms:.2f} ms/image |
| **Batch Size** | {self.batch_size} |
| **Device** | {self.device} |

---

*📊 Report generated by RF-DETR Validation System*  
*🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        report_path = self.save_dir / 'validation_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_path
    
    def _generate_curves(
        self,
        all_predictions: List[Dict[str, Any]],
        all_targets: List[Dict[str, Any]],
        num_classes: int,
    ) -> None:
        """
        Generate PR/F1/P/R curves from accumulated predictions.
        
        Uses the curves.py module for plotting.
        """
        try:
            from rfdetr.training.visualizations.curves import (
                plot_pr_curve, plot_f1_curve, plot_confidence_curve
            )
            from rfdetr.util import box_ops
            
            # Confidence thresholds for curve sampling
            conf_thresholds = np.linspace(0.0, 1.0, 101)
            
            # Per-class curves storage
            class_precision = {i: [] for i in range(num_classes)}
            class_recall = {i: [] for i in range(num_classes)}
            class_f1 = {i: [] for i in range(num_classes)}
            
            # Compute metrics at each confidence threshold
            for conf_thresh in conf_thresholds:
                class_tp = {i: 0 for i in range(num_classes)}
                class_fp = {i: 0 for i in range(num_classes)}
                class_fn = {i: 0 for i in range(num_classes)}
                
                for pred, target in zip(all_predictions, all_targets):
                    pred_boxes = pred['boxes'].cpu().numpy() if hasattr(pred['boxes'], 'cpu') else pred['boxes']
                    pred_labels = pred['labels'].cpu().numpy() if hasattr(pred['labels'], 'cpu') else pred['labels']
                    pred_scores = pred['scores'].cpu().numpy() if hasattr(pred['scores'], 'cpu') else pred['scores']
                    
                    gt_boxes = target['boxes']
                    if hasattr(gt_boxes, 'cpu'):
                        gt_boxes = gt_boxes.cpu()
                    
                    # Convert GT boxes if normalized
                    if len(gt_boxes) > 0 and gt_boxes.max() <= 1.0:
                        gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
                        gt_boxes = gt_boxes * self.imgsz
                    gt_boxes = gt_boxes.numpy() if hasattr(gt_boxes, 'numpy') else gt_boxes
                    gt_labels = target['labels'].cpu().numpy() if hasattr(target['labels'], 'cpu') else target['labels']
                    
                    # Filter predictions by confidence
                    mask = pred_scores >= conf_thresh
                    filtered_boxes = pred_boxes[mask]
                    filtered_labels = pred_labels[mask]
                    filtered_scores = pred_scores[mask]
                    
                    # Match predictions to GT
                    tp_idx, fp_idx, fn_idx = match_predictions_to_gt(
                        gt_boxes, gt_labels,
                        filtered_boxes, filtered_labels, filtered_scores,
                        iou_threshold=self.iou_threshold,
                    )
                    
                    # Count per class
                    for idx in tp_idx:
                        cls = int(filtered_labels[idx])
                        if cls < num_classes:
                            class_tp[cls] += 1
                    
                    for idx in fp_idx:
                        cls = int(filtered_labels[idx])
                        if cls < num_classes:
                            class_fp[cls] += 1
                    
                    for idx in fn_idx:
                        cls = int(gt_labels[idx])
                        if cls < num_classes:
                            class_fn[cls] += 1
                
                # Calculate P/R/F1 for each class at this threshold
                for cls in range(num_classes):
                    tp = class_tp[cls]
                    fp = class_fp[cls]
                    fn = class_fn[cls]
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    class_precision[cls].append(precision)
                    class_recall[cls].append(recall)
                    class_f1[cls].append(f1)
            
            # Convert to numpy arrays and create dicts for curves.py functions
            precision_dict = {}
            recall_dict = {}
            f1_dict = {}
            ap_dict = {}
            
            for cls in range(num_classes):
                precision_dict[cls] = np.array(class_precision[cls])
                recall_dict[cls] = np.array(class_recall[cls])
                f1_dict[cls] = np.array(class_f1[cls])
                # Compute AP as area under PR curve
                ap_dict[cls] = np.trapz(precision_dict[cls], recall_dict[cls])
            
            # Use curves.py plotting functions
            plot_pr_curve(
                precision_dict, recall_dict, ap_dict,
                self.save_dir / 'BoxPR_curve.png',
                class_names=self.class_names
            )
            
            plot_f1_curve(
                f1_dict, conf_thresholds,
                self.save_dir / 'BoxF1_curve.png',
                class_names=self.class_names
            )
            
            plot_confidence_curve(
                precision_dict, conf_thresholds,
                self.save_dir / 'BoxP_curve.png',
                ylabel='Precision',
                title='Precision-Confidence Curve',
                class_names=self.class_names
            )
            
            plot_confidence_curve(
                recall_dict, conf_thresholds,
                self.save_dir / 'BoxR_curve.png',
                ylabel='Recall',
                title='Recall-Confidence Curve',
                class_names=self.class_names
            )
            
        except Exception as e:
            print(f"Warning: Failed to generate curves: {e}")
    
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

