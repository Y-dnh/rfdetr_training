"""
Confusion matrix visualization for RF-DETR evaluation.

Creates YOLO-style confusion matrix (confusion_matrix.png).
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def compute_confusion_matrix(
    predictions: List[int],
    ground_truths: List[int],
    num_classes: int,
    include_background: bool = True,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: List of predicted class indices.
        ground_truths: List of ground truth class indices.
        num_classes: Number of classes.
        include_background: Whether to include background class.
    
    Returns:
        Confusion matrix of shape (num_classes + bg, num_classes + bg).
    """
    if include_background:
        n = num_classes + 1  # +1 for background
    else:
        n = num_classes
    
    matrix = np.zeros((n, n), dtype=np.int64)
    
    for pred, gt in zip(predictions, ground_truths):
        if include_background:
            # Background is the last class
            pred = pred if pred < num_classes else num_classes
            gt = gt if gt < num_classes else num_classes
        matrix[pred, gt] += 1
    
    return matrix


def plot_confusion_matrix(
    matrix: np.ndarray,
    save_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    include_background: bool = True,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        matrix: Confusion matrix array.
        save_path: Path to save the plot.
        class_names: List of class names.
        include_background: Whether background class is included.
        normalize: Whether to normalize values.
        title: Plot title.
    """
    # Prepare labels
    n = matrix.shape[0]
    if class_names is None:
        labels = [str(i) for i in range(n)]
        if include_background:
            labels[-1] = 'background'
    else:
        labels = list(class_names)
        if include_background and len(labels) == n - 1:
            labels.append('background')
    
    # Normalize if requested
    if normalize:
        matrix = matrix.astype(np.float32)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        matrix = matrix / row_sums
    
    # Create figure
    fig_size = max(8, n * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # Create heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.0f' if not normalize else '.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=True,
        square=True,
        annot_kws={'size': 8} if n > 10 else {'size': 10},
    )
    
    ax.set_xlabel('True', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Rotate x labels if many classes
    if n > 5:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


class ConfusionMatrix:
    """
    Confusion matrix accumulator for evaluation.
    
    Example:
        >>> cm = ConfusionMatrix(num_classes=3, class_names=['cat', 'dog', 'bird'])
        >>> cm.process_batch(predictions, ground_truths)
        >>> cm.plot('confusion_matrix.png')
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        include_background: bool = True,
        imgsz: int = 640,
    ):
        self.num_classes = num_classes
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.include_background = include_background
        self.imgsz = imgsz
        
        # Initialize matrix
        n = num_classes + 1 if include_background else num_classes
        self.matrix = np.zeros((n, n), dtype=np.int64)
    
    def process_batch(
        self,
        predictions: List[dict],
        targets: List[dict],
    ) -> None:
        """
        Process a batch of predictions and targets.
        
        Args:
            predictions: List of prediction dicts with 'boxes', 'labels', 'scores'.
            targets: List of target dicts with 'boxes', 'labels'.
        """
        for pred, target in zip(predictions, targets):
            self._process_single(pred, target)
    
    def _process_single(self, pred: dict, target: dict) -> None:
        """Process single image predictions."""
        pred_boxes = pred.get('boxes', np.zeros((0, 4)))
        pred_labels = pred.get('labels', np.zeros(0))
        pred_scores = pred.get('scores', np.ones(len(pred_labels)))
        
        gt_boxes = target.get('boxes', np.zeros((0, 4)))
        gt_labels = target.get('labels', np.zeros(0))
        
        # Convert to numpy
        if hasattr(pred_boxes, 'numpy'):
            pred_boxes = pred_boxes.cpu().numpy()
        if hasattr(pred_labels, 'numpy'):
            pred_labels = pred_labels.cpu().numpy()
        if hasattr(pred_scores, 'numpy'):
            pred_scores = pred_scores.cpu().numpy()
        if hasattr(gt_boxes, 'numpy'):
            gt_boxes = gt_boxes.cpu().numpy()
        if hasattr(gt_labels, 'numpy'):
            gt_labels = gt_labels.cpu().numpy()
        
        # Convert GT boxes from normalized cxcywh to xyxy if needed
        # and scale to match prediction coordinates using imgsz
        if len(gt_boxes) > 0 and np.max(gt_boxes) <= 1.0:
            # Use imgsz for scaling normalized boxes to absolute coordinates
            scale = self.imgsz
            
            # Assume normalized cxcywh format - convert to xyxy and scale
            cx, cy, w, h = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
            gt_boxes_xyxy = np.zeros_like(gt_boxes)
            gt_boxes_xyxy[:, 0] = (cx - w / 2) * scale  # x1
            gt_boxes_xyxy[:, 1] = (cy - h / 2) * scale  # y1
            gt_boxes_xyxy[:, 2] = (cx + w / 2) * scale  # x2
            gt_boxes_xyxy[:, 3] = (cy + h / 2) * scale  # y2
            gt_boxes = gt_boxes_xyxy


        
        # Filter by confidence
        mask = pred_scores >= self.conf_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]
        pred_scores = pred_scores[mask]
        
        # Match predictions to ground truths
        bg_idx = self.num_classes  # Background class index
        matched_gt = set()
        
        for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
            best_iou = 0
            best_gt_idx = -1
            
            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if j in matched_gt:
                    continue
                
                iou = self._compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                gt_label = int(gt_labels[best_gt_idx])
                matched_gt.add(best_gt_idx)
                self.matrix[int(pred_label), gt_label] += 1
            else:
                # False positive (prediction with no matching GT)
                if self.include_background:
                    self.matrix[int(pred_label), bg_idx] += 1
        
        # Count false negatives (GT with no matching prediction)
        for j in range(len(gt_labels)):
            if j not in matched_gt:
                gt_label = int(gt_labels[j])
                if self.include_background:
                    self.matrix[bg_idx, gt_label] += 1
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def plot(self, save_path: Union[str, Path], normalize: bool = False) -> None:
        """Plot and save confusion matrix."""
        plot_confusion_matrix(
            self.matrix,
            save_path,
            class_names=self.class_names,
            include_background=self.include_background,
            normalize=normalize,
        )
    
    def get_matrix(self) -> np.ndarray:
        """Get the confusion matrix."""
        return self.matrix.copy()
    
    def reset(self) -> None:
        """Reset the confusion matrix."""
        self.matrix.fill(0)
