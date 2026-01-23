"""
Performance curves for RF-DETR evaluation.

Creates YOLO-style evaluation curves:
- PR_curve.png: Precision-Recall curve
- F1_curve.png: F1-Confidence curve
- P_curve.png: Precision-Confidence curve
- R_curve.png: Recall-Confidence curve
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Default colors for classes
CLASS_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Yellow-green
    '#17becf',  # Cyan
]


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision using 101-point interpolation.
    
    Args:
        recall: Recall values.
        precision: Precision values.
    
    Returns:
        Average Precision value.
    """
    # Add sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    
    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def plot_pr_curve(
    precision: Dict[str, np.ndarray],
    recall: Dict[str, np.ndarray],
    ap: Dict[str, float],
    save_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Plot Precision-Recall curve.
    
    Args:
        precision: Dict mapping class name/id to precision values.
        recall: Dict mapping class name/id to recall values.
        ap: Dict mapping class name/id to AP values.
        save_path: Path to save the plot.
        class_names: Optional list of class names.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each class
    for i, (cls_id, prec) in enumerate(precision.items()):
        rec = recall[cls_id]
        cls_ap = ap.get(cls_id, 0.0)
        
        color = CLASS_COLORS[i % len(CLASS_COLORS)]
        
        # Get class name
        if class_names is not None and isinstance(cls_id, int) and cls_id < len(class_names):
            label = f"{class_names[cls_id]} {cls_ap:.3f}"
        else:
            label = f"{cls_id} {cls_ap:.3f}"
        
        ax.plot(rec, prec, color=color, linewidth=2, label=label)
    
    # Plot all classes (mean)
    if len(precision) > 1:
        mean_ap = np.mean(list(ap.values()))
        ax.plot([0, 1], [mean_ap, mean_ap], 'k--', linewidth=3, 
                label=f'all classes {mean_ap:.3f} mAP@0.5')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_f1_curve(
    f1_scores: Dict[str, np.ndarray],
    confidence: np.ndarray,
    save_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Plot F1-Confidence curve.
    
    Args:
        f1_scores: Dict mapping class name/id to F1 scores at different confidences.
        confidence: Confidence thresholds.
        save_path: Path to save the plot.
        class_names: Optional list of class names.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Find best F1 across all classes
    all_f1 = np.zeros_like(confidence)
    
    # Plot each class
    for i, (cls_id, f1) in enumerate(f1_scores.items()):
        color = CLASS_COLORS[i % len(CLASS_COLORS)]
        
        if class_names is not None and isinstance(cls_id, int) and cls_id < len(class_names):
            label = class_names[cls_id]
        else:
            label = str(cls_id)
        
        ax.plot(confidence, f1, color=color, linewidth=2, label=label)
        all_f1 = np.maximum(all_f1, f1)
    
    # Plot all classes mean
    if len(f1_scores) > 1:
        mean_f1 = np.mean([f1 for f1 in f1_scores.values()], axis=0)
        best_idx = np.argmax(mean_f1)
        best_conf = confidence[best_idx]
        best_f1 = mean_f1[best_idx]
        
        ax.plot(confidence, mean_f1, 'k-', linewidth=3, 
                label=f'all classes {best_f1:.2f} at {best_conf:.3f}')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('F1', fontsize=12)
    ax.set_title('F1-Confidence Curve', fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_confidence_curve(
    values: Dict[str, np.ndarray],
    confidence: np.ndarray,
    save_path: Union[str, Path],
    ylabel: str = "Value",
    title: str = "Confidence Curve",
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Plot generic confidence curve (Precision or Recall vs Confidence).
    
    Args:
        values: Dict mapping class to values at different confidences.
        confidence: Confidence thresholds.
        save_path: Path to save the plot.
        ylabel: Y-axis label.
        title: Plot title.
        class_names: Optional list of class names.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, (cls_id, vals) in enumerate(values.items()):
        color = CLASS_COLORS[i % len(CLASS_COLORS)]
        
        if class_names is not None and isinstance(cls_id, int) and cls_id < len(class_names):
            label = class_names[cls_id]
        else:
            label = str(cls_id)
        
        ax.plot(confidence, vals, color=color, linewidth=2, label=label)
    
    # Plot mean
    if len(values) > 1:
        mean_vals = np.mean([v for v in values.values()], axis=0)
        ax.plot(confidence, mean_vals, 'k-', linewidth=3, label='all classes')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_all_curves(
    results: Dict[str, Any],
    save_dir: Union[str, Path],
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Save all evaluation curves from results dict.
    
    Expected keys in results:
    - precision: Dict of precision arrays per class
    - recall: Dict of recall arrays per class
    - ap: Dict of AP values per class
    - f1: Dict of F1 arrays per class (at different confidences)
    - confidence: Confidence thresholds array
    
    Args:
        results: Results dictionary.
        save_dir: Directory to save plots.
        class_names: Optional class names.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    confidence = results.get('confidence', np.linspace(0, 1, 100))
    
    # PR curve
    if 'precision' in results and 'recall' in results and 'ap' in results:
        plot_pr_curve(
            results['precision'],
            results['recall'],
            results['ap'],
            save_dir / 'PR_curve.png',
            class_names=class_names,
        )
    
    # F1 curve
    if 'f1' in results:
        plot_f1_curve(
            results['f1'],
            confidence,
            save_dir / 'F1_curve.png',
            class_names=class_names,
        )
    
    # Precision curve
    if 'precision_conf' in results:
        plot_confidence_curve(
            results['precision_conf'],
            confidence,
            save_dir / 'P_curve.png',
            ylabel='Precision',
            title='Precision-Confidence Curve',
            class_names=class_names,
        )
    
    # Recall curve
    if 'recall_conf' in results:
        plot_confidence_curve(
            results['recall_conf'],
            confidence,
            save_dir / 'R_curve.png',
            ylabel='Recall',
            title='Recall-Confidence Curve',
            class_names=class_names,
        )
