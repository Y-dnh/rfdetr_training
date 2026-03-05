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
    Plot training results in YOLO style (6 rows x 5 cols): train losses, val losses,
    P/R/F1, mAP, AR by size, AP by size.
    """
    metrics_layout = [
        ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'train/class_error', None],
        ['val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'val/class_error', None],
        ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/F1(B)', None, None],
        ['metrics/mAP50(B)', 'metrics/mAP75(B)', 'metrics/mAP50-95(B)', None, None],
        ['metrics/AR_small(B)', 'metrics/AR_medium(B)', 'metrics/AR_large(B)', None, None],
        ['metrics/AP_small(B)', 'metrics/AP_medium(B)', 'metrics/AP_large(B)', None, None],
    ]
    alt_keys = {
        'train/dfl_loss': ['train/loss', 'train/total_loss'],
        'val/dfl_loss': ['val/loss', 'val/total_loss'],
        'metrics/precision(B)': ['metrics/precision', 'precision'],
        'metrics/recall(B)': ['metrics/recall', 'recall'],
        'metrics/F1(B)': ['metrics/f1', 'metrics/F1', 'f1'],
        'metrics/mAP50(B)': ['metrics/mAP50', 'mAP50', 'metrics/mAP@0.5'],
        'metrics/mAP75(B)': ['metrics/mAP75', 'mAP75'],
        'metrics/mAP50-95(B)': ['metrics/mAP50-95', 'mAP50-95', 'metrics/mAP@0.5:0.95', 'metrics/mAP'],
        'metrics/AR_small(B)': ['metrics/AR_small', 'AR_small'],
        'metrics/AR_medium(B)': ['metrics/AR_medium', 'AR_medium'],
        'metrics/AR_large(B)': ['metrics/AR_large', 'AR_large'],
        'metrics/AP_small(B)': ['metrics/AP_small', 'AP_small'],
        'metrics/AP_medium(B)': ['metrics/AP_medium', 'AP_medium'],
        'metrics/AP_large(B)': ['metrics/AP_large', 'AP_large'],
    }

    def get_data(key: str) -> Optional[np.ndarray]:
        if key is None:
            return None
        if key in results:
            return np.array(results[key])
        for alt_key in alt_keys.get(key, []):
            if alt_key in results:
                return np.array(results[alt_key])
        return None

    fig, axes = plt.subplots(6, 5, figsize=(15, 14))
    fig.suptitle(title, fontsize=14)

    for row_idx, row_metrics in enumerate(metrics_layout):
        for col_idx, metric_key in enumerate(row_metrics):
            ax = axes[row_idx, col_idx]
            if metric_key is None:
                ax.axis('off')
                continue
            data = get_data(metric_key)
            if data is not None and len(data) > 0:
                x = np.arange(len(data))
                ax.plot(x, data, 'o-', markersize=2, alpha=0.7, label='results')
                if len(data) > 1:
                    smoothed = smooth_data(data, smooth_factor)
                    ax.plot(x, smoothed, '--', linewidth=2, color='orange', alpha=0.8, label='smooth')
                ax.set_xlabel('Epoch')
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric_key, fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
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
    
    def load_from_csv(self, csv_path: Union[str, Path]) -> None:
        """
        Load full history from results.csv (e.g. on resume) into self.metrics and self.lr_values.
        Column names are normalized by stripping the '(B)' suffix so new epochs append to the same lists.
        """
        import csv
        csv_path = Path(csv_path)
        if not csv_path.exists():
            return
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return
        # Normalize key: remove (B) suffix for consistency with log_epoch keys
        def norm_key(k: str) -> str:
            if k.endswith('(B)'):
                return k[:-3].strip()
            return k
        fieldnames = [fn for fn in reader.fieldnames if fn and fn not in ('epoch', 'time')]
        for name in fieldnames:
            norm = norm_key(name)
            if norm not in self.metrics:
                self.metrics[norm] = []
        for row in rows:
            for name in fieldnames:
                norm = norm_key(name)
                val = row.get(name, '').strip()
                try:
                    self.metrics[norm].append(float(val) if val else 0.0)
                except ValueError:
                    self.metrics[norm].append(0.0)
        # lr_values from lr/pg0 (one per row)
        lr_key = 'lr/pg0'
        if lr_key in (reader.fieldnames or []):
            for row in rows:
                val = row.get(lr_key, '').strip()
                if val:
                    try:
                        self.lr_values.append(float(val))
                    except ValueError:
                        pass
    
    def save_csv(self) -> None:
        """Save metrics to CSV file."""
        if not self.metrics:
            return
        
        csv_path = self.save_dir / 'metrics.csv'
        
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
