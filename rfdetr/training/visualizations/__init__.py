"""
Visualization module for RF-DETR training.

Provides YOLO-style visualizations:
- Batch visualizations (train_batch*.jpg, val_batch*_labels.jpg, val_batch*_pred.jpg)
- Training metrics plots (results.png)
- Precision-Recall curves (PR_curve.png)
- F1 confidence curves (F1_curve.png)
- Confusion matrix (confusion_matrix.png)
- Dataset label analysis (labels.jpg)
"""

from rfdetr.training.visualizations.batch_visualizer import (
    visualize_batch,
    save_batch_visualization,
    BatchVisualizer,
    draw_boxes,
    denormalize_image,
    create_analysis_grid,
    AnalysisVisualizer,
    match_predictions_to_gt,
)
from rfdetr.training.visualizations.metrics_plotter import (
    plot_results,
    plot_lr_schedule,
    MetricsLogger,
)
from rfdetr.training.visualizations.curves import (
    plot_pr_curve,
    plot_f1_curve,
    plot_confidence_curve,
    save_all_curves,
)
from rfdetr.training.visualizations.confusion_matrix import (
    plot_confusion_matrix,
    compute_confusion_matrix,
    ConfusionMatrix,
)
from rfdetr.training.visualizations.labels_analyzer import (
    analyze_labels,
    plot_labels,
    create_labels_visualization,
)

__all__ = [
    # Batch visualization
    "visualize_batch",
    "save_batch_visualization",
    "BatchVisualizer",
    "draw_boxes",
    "denormalize_image",
    # Analysis visualization (GT/TP/FP/FN grid)
    "create_analysis_grid",
    "AnalysisVisualizer",
    "match_predictions_to_gt",
    # Metrics plotting
    "plot_results",
    "plot_lr_schedule",
    "MetricsLogger",
    # Curves
    "plot_pr_curve",
    "plot_f1_curve",
    "plot_confidence_curve",
    "save_all_curves",
    # Confusion matrix
    "plot_confusion_matrix",
    "compute_confusion_matrix",
    "ConfusionMatrix",
    # Labels analysis
    "analyze_labels",
    "plot_labels",
    "create_labels_visualization",
]
