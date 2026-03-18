"""
Dataset label analysis and visualization for RF-DETR.

Creates YOLO-style labels.jpg with:
- Class distribution bar chart
- Bounding box overlay visualization
- Center point heatmap
- Width vs Height scatter plot
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm


def analyze_labels(
    boxes: List[np.ndarray],
    labels: List[np.ndarray],
    image_sizes: List[Tuple[int, int]],
    class_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Analyze dataset labels.
    
    Args:
        boxes: List of box arrays in xyxy format, one per image.
        labels: List of label arrays, one per image.
        image_sizes: List of (height, width) tuples for each image.
        class_names: Optional class names.
    
    Returns:
        Dictionary with analysis results.
    """
    # Collect all boxes and labels
    all_boxes = []
    all_labels = []
    all_centers_normalized = []
    all_sizes_normalized = []
    
    for img_boxes, img_labels, (h, w) in zip(boxes, labels, image_sizes):
        if len(img_boxes) == 0:
            continue
        
        # Normalize boxes
        boxes_norm = img_boxes.copy()
        boxes_norm[:, [0, 2]] /= w
        boxes_norm[:, [1, 3]] /= h
        
        # Calculate centers and sizes
        centers_x = (boxes_norm[:, 0] + boxes_norm[:, 2]) / 2
        centers_y = (boxes_norm[:, 1] + boxes_norm[:, 3]) / 2
        widths = boxes_norm[:, 2] - boxes_norm[:, 0]
        heights = boxes_norm[:, 3] - boxes_norm[:, 1]
        
        all_boxes.append(boxes_norm)
        all_labels.extend(img_labels.tolist())
        all_centers_normalized.extend(list(zip(centers_x, centers_y)))
        all_sizes_normalized.extend(list(zip(widths, heights)))
    
    # Concatenate
    all_boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 4))
    all_labels = np.array(all_labels)
    all_centers = np.array(all_centers_normalized)
    all_sizes = np.array(all_sizes_normalized)
    
    # Count classes
    if len(all_labels) > 0:
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        class_counts = dict(zip(unique_labels.tolist(), counts.tolist()))
    else:
        class_counts = {}
    
    return {
        'boxes': all_boxes,
        'labels': all_labels,
        'centers': all_centers,
        'sizes': all_sizes,
        'class_counts': class_counts,
        'num_boxes': len(all_labels),
        'num_images': len(boxes),
    }


def plot_labels(
    analysis: Dict[str, np.ndarray],
    save_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Create YOLO-style labels.jpg visualization.
    
    Args:
        analysis: Analysis results from analyze_labels().
        save_path: Path to save the plot.
        class_names: Optional class names.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. Class distribution (top-left)
    ax1 = axes[0, 0]
    class_counts = analysis['class_counts']
    
    if class_counts:
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        
        if class_names:
            x_labels = [class_names[c] if c < len(class_names) else str(c) for c in classes]
        else:
            x_labels = [str(c) for c in classes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        bars = ax1.bar(range(len(classes)), counts, color=colors)
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(x_labels, rotation=45 if len(classes) > 5 else 0)
        ax1.set_ylabel('instances')
        ax1.set_title('Class Distribution')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    str(count), ha='center', va='bottom', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No labels', ha='center', va='center')
        ax1.set_title('Class Distribution')
    
    # 2. Bounding box visualization (top-right)
    ax2 = axes[0, 1]
    boxes = analysis['boxes']
    
    if len(boxes) > 0:
        # Sample boxes for visualization (max 1000)
        n_sample = min(1000, len(boxes))
        indices = np.random.choice(len(boxes), n_sample, replace=False)
        
        for idx in indices:
            box = boxes[idx]
            x, y, x2, y2 = box
            w, h = x2 - x, y2 - y
            rect = Rectangle((x, y), w, h, 
                            linewidth=0.5, edgecolor='cyan', facecolor='none', alpha=0.5)
            ax2.add_patch(rect)
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(1, 0)  # Invert y-axis
        ax2.set_aspect('equal')
    
    ax2.set_title('Bounding Boxes')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    
    # 3. Center point heatmap (bottom-left)
    ax3 = axes[1, 0]
    centers = analysis['centers']
    
    if len(centers) > 0:
        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            centers[:, 0], centers[:, 1], 
            bins=50, range=[[0, 1], [0, 1]]
        )
        
        im = ax3.imshow(heatmap.T, extent=[0, 1, 1, 0], 
                       cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax3, label='count')
    
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Box Centers')
    
    # 4. Width vs Height scatter plot (bottom-right)
    ax4 = axes[1, 1]
    sizes = analysis['sizes']
    
    if len(sizes) > 0:
        # Sample for scatter plot (max 2000 points)
        n_sample = min(2000, len(sizes))
        indices = np.random.choice(len(sizes), n_sample, replace=False)
        sampled_sizes = sizes[indices]
        
        ax4.scatter(sampled_sizes[:, 0], sampled_sizes[:, 1], 
                   alpha=0.3, s=5, c='steelblue')
        ax4.set_xlim(0, 0.5)  # Most boxes are small
        ax4.set_ylim(0, 0.35)
    
    ax4.set_xlabel('width')
    ax4.set_ylabel('height')
    ax4.set_title('Box Sizes')
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_labels_visualization(
    dataset,
    save_path: Union[str, Path],
    max_images: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    random_seed: int = 42,
) -> None:
    """
    Create labels visualization from a dataset.
    
    Args:
        dataset: Dataset object with __getitem__ returning (image, target).
        save_path: Path to save labels.jpg.
        max_images: Maximum number of images to analyze. ``None`` means use the
                    full dataset. If smaller than the dataset, images are sampled
                    uniformly at random across the full split.
        class_names: Optional class names.
        random_seed: Seed for reproducible dataset sampling.
    """
    boxes_list = []
    labels_list = []
    sizes_list = []
    
    total_images = len(dataset)
    n_samples = total_images if max_images is None else min(total_images, max_images)
    if n_samples == 0:
        analysis = analyze_labels(boxes_list, labels_list, sizes_list, class_names)
        plot_labels(analysis, save_path, class_names)
        return analysis

    if n_samples < total_images:
        rng = np.random.default_rng(random_seed)
        sample_indices = rng.choice(total_images, size=n_samples, replace=False).tolist()
    else:
        sample_indices = list(range(n_samples))

    progress_desc = "Creating labels.jpg"
    for i in tqdm(sample_indices, desc=progress_desc, total=len(sample_indices)):
        try:
            # Try to get raw item if available
            if hasattr(dataset, 'get_raw_item'):
                img, target = dataset.get_raw_item(i)
                h, w = img.size[1], img.size[0]  # PIL: width, height
            else:
                img, target = dataset[i]
                if hasattr(img, 'shape'):
                    if len(img.shape) == 3 and img.shape[0] == 3:
                        h, w = img.shape[1], img.shape[2]
                    else:
                        h, w = img.shape[:2]
                else:
                    h, w = 640, 640  # Default
            
            # Get boxes and labels
            boxes = target.get('boxes', np.zeros((0, 4)))
            labels = target.get('labels', np.zeros(0))
            
            if hasattr(boxes, 'numpy'):
                boxes = boxes.numpy()
            if hasattr(labels, 'numpy'):
                labels = labels.numpy()
            
            boxes_list.append(boxes)
            labels_list.append(labels)
            sizes_list.append((h, w))
            
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue
    
    # Analyze
    analysis = analyze_labels(
        boxes_list, labels_list, sizes_list, class_names
    )
    
    # Plot
    plot_labels(analysis, save_path, class_names)
    
    return analysis
