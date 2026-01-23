"""
Metrics plotting for RF-DETR training.

Creates YOLO-style training results visualization (results.png) with:
- Training and validation losses
- Precision, Recall, mAP metrics
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def smooth_data(data: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """
    Exponential moving average smoothing.
    
    Args:
        data: Input data array.
        weight: Smoothing weight (0-1). Higher = smoother.
    
    Returns:
        Smoothed data array.
    """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = weight * smoothed[i-1] + (1 - weight) * data[i]
    return smoothed


def plot_results(
    results: Dict[str, List[float]],
    save_path: Union[str, Path],
    title: str = "Training Results",
    smooth_factor: float = 0.6,
) -> None:
    """
    Plot training results in YOLO style (2 rows x 5 cols).
    
    Expected keys in results dict:
    - train/box_loss
    - train/cls_loss
    - train/dfl_loss (optional, can be general loss)
    - metrics/precision
    - metrics/recall
    - val/box_loss
    - val/cls_loss
    - val/dfl_loss (optional)
    - metrics/mAP50
    - metrics/mAP50-95
    
    Args:
        results: Dictionary mapping metric names to lists of values.
        save_path: Path to save the plot.
        title: Plot title.
        smooth_factor: Smoothing factor for curves.
    """
    # Define subplot layout (2 rows x 5 cols)
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    fig.suptitle(title, fontsize=14)
    
    # Define metrics to plot
    metrics_layout = [
        # Row 1: Training metrics
        ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'metrics/precision(B)', 'metrics/recall(B)'],
        # Row 2: Validation metrics
        ['val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'],
    ]
    
    # Alternative keys (for flexibility)
    alt_keys = {
        'train/dfl_loss': ['train/loss', 'train/total_loss'],
        'val/dfl_loss': ['val/loss', 'val/total_loss'],
        'metrics/precision(B)': ['metrics/precision', 'precision'],
        'metrics/recall(B)': ['metrics/recall', 'recall'],
        'metrics/mAP50(B)': ['metrics/mAP50', 'mAP50', 'metrics/mAP@0.5'],
        'metrics/mAP50-95(B)': ['metrics/mAP50-95', 'mAP50-95', 'metrics/mAP@0.5:0.95', 'metrics/mAP'],
    }
    
    def get_data(key: str) -> Optional[np.ndarray]:
        """Get data for a metric key, trying alternatives."""
        if key in results:
            return np.array(results[key])
        
        # Try alternative keys
        for alt_key in alt_keys.get(key, []):
            if alt_key in results:
                return np.array(results[alt_key])
        
        return None
    
    for row_idx, row_metrics in enumerate(metrics_layout):
        for col_idx, metric_key in enumerate(row_metrics):
            ax = axes[row_idx, col_idx]
            
            data = get_data(metric_key)
            
            if data is not None and len(data) > 0:
                x = np.arange(len(data))
                
                # Plot raw data
                ax.plot(x, data, 'o-', markersize=2, alpha=0.7, label='results')
                
                # Plot smoothed data
                if len(data) > 1:
                    smoothed = smooth_data(data, smooth_factor)
                    ax.plot(x, smoothed, '--', linewidth=2, color='orange', alpha=0.8, label='smooth')
                
                ax.set_xlabel('Epoch')
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            
            # Set title (metric name)
            ax.set_title(metric_key, fontsize=10)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_lr_schedule(
    lr_values: List[float],
    save_path: Union[str, Path],
    title: str = "Learning Rate Schedule",
) -> None:
    """
    Plot learning rate schedule.
    
    Args:
        lr_values: List of learning rates per epoch.
        save_path: Path to save the plot.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(len(lr_values))
    ax.plot(epochs, lr_values, 'b-', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


class MetricsLogger:
    """
    Logger for collecting and plotting training metrics.
    
    Example:
        >>> logger = MetricsLogger(save_dir='runs/exp1')
        >>> logger.log('train/box_loss', 0.5, epoch=0)
        >>> logger.log('train/cls_loss', 0.3, epoch=0)
        >>> logger.save_plots()
    """
    
    def __init__(self, save_dir: Union[str, Path]):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: Dict[str, List[float]] = {}
        self.lr_values: List[float] = []
    
    def log(self, name: str, value: float, epoch: Optional[int] = None) -> None:
        """
        Log a metric value.
        
        Args:
            name: Metric name.
            value: Metric value.
            epoch: Epoch number (unused, for API compatibility).
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def log_lr(self, lr: float) -> None:
        """Log learning rate."""
        self.lr_values.append(lr)
    
    def log_epoch(self, epoch_metrics: Dict[str, float]) -> None:
        """
        Log all metrics for an epoch.
        
        Args:
            epoch_metrics: Dictionary of metric name to value.
        """
        for name, value in epoch_metrics.items():
            self.log(name, value)
    
    def save_plots(self) -> None:
        """Save all plots."""
        # Save results.png
        if self.metrics:
            plot_results(
                self.metrics,
                save_path=self.save_dir / 'results.png',
                title='Training Results'
            )
        
        # Save LR schedule
        if self.lr_values:
            plot_lr_schedule(
                self.lr_values,
                save_path=self.save_dir / 'lr_schedule.png'
            )
    
    def save_csv(self) -> None:
        """Save metrics to CSV file."""
        if not self.metrics:
            return
        
        csv_path = self.save_dir / 'results.csv'
        
        # Get all metric names and max length
        metric_names = list(self.metrics.keys())
        max_len = max(len(v) for v in self.metrics.values())
        
        with open(csv_path, 'w') as f:
            # Header
            f.write('epoch,' + ','.join(metric_names) + '\n')
            
            # Data rows
            for i in range(max_len):
                row = [str(i)]
                for name in metric_names:
                    if i < len(self.metrics[name]):
                        row.append(f"{self.metrics[name][i]:.6f}")
                    else:
                        row.append('')
                f.write(','.join(row) + '\n')
    
    def get_best_metrics(self) -> Dict[str, Tuple[int, float]]:
        """
        Get best (max) value and epoch for each metric.
        
        Returns:
            Dictionary mapping metric name to (best_epoch, best_value).
        """
        best = {}
        for name, values in self.metrics.items():
            if values:
                # For losses, best is minimum
                if 'loss' in name.lower():
                    best_idx = int(np.argmin(values))
                else:
                    best_idx = int(np.argmax(values))
                best[name] = (best_idx, values[best_idx])
        return best
