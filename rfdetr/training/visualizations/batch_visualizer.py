"""
Batch visualization for RF-DETR training.

Creates YOLO-style batch visualizations:
- train_batch*.jpg: Training batches with ground truth boxes
- val_batch*_labels.jpg: Validation batches with ground truth
- val_batch*_pred.jpg: Validation batches with predictions
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# Default colors for different classes (YOLO style - vibrant colors)
COLORS = [
    (255, 56, 56),    # Red
    (255, 157, 151),  # Pink
    (255, 112, 31),   # Orange
    (255, 178, 29),   # Yellow-Orange
    (207, 210, 49),   # Yellow-Green
    (72, 249, 10),    # Green
    (146, 204, 23),   # Lime
    (61, 219, 134),   # Teal
    (26, 147, 52),    # Dark Green
    (0, 212, 187),    # Cyan
    (44, 153, 168),   # Blue-Cyan
    (0, 194, 255),    # Light Blue
    (52, 69, 147),    # Dark Blue
    (100, 115, 255),  # Blue
    (0, 24, 236),     # Bright Blue
    (132, 56, 255),   # Purple
    (82, 0, 133),     # Dark Purple
    (203, 56, 255),   # Magenta
    (255, 149, 200),  # Light Pink
    (255, 55, 199),   # Hot Pink
]


def get_color(class_idx: int) -> Tuple[int, int, int]:
    """Get color for a class index."""
    return COLORS[class_idx % len(COLORS)]


def denormalize_image(
    image: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """
    Denormalize image tensor to numpy array.
    
    Args:
        image: Normalized image tensor (C, H, W).
        mean: Normalization mean.
        std: Normalization std.
    
    Returns:
        Denormalized numpy array (H, W, C) in uint8 format.
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Handle both CHW and HWC formats
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    
    # Denormalize
    mean = np.array(mean)
    std = np.array(std)
    image = image * std + mean
    
    # Clip and convert to uint8
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image


def draw_boxes(
    image: np.ndarray,
    boxes: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray, List[int]],
    scores: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
    class_names: Optional[List[str]] = None,
    show_labels: bool = True,
    show_conf: bool = True,
    line_width: int = 2,
    font_size: int = 12,
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Image as numpy array (H, W, C) in RGB format.
        boxes: Bounding boxes in xyxy format (N, 4).
        labels: Class labels (N,).
        scores: Confidence scores (N,). Optional.
        class_names: List of class names. If None, uses label indices.
        show_labels: Whether to show class labels.
        show_conf: Whether to show confidence scores.
        line_width: Width of bounding box lines.
        font_size: Size of label font.
    
    Returns:
        Image with drawn boxes.
    """
    image = image.copy()
    
    # Convert tensors to numpy
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    # Convert to PIL for drawing
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.astype(int)
        color = get_color(int(label))
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        
        # Prepare label text
        if show_labels:
            if class_names is not None and int(label) < len(class_names):
                label_text = class_names[int(label)]
            else:
                label_text = str(int(label))
            
            if show_conf and scores is not None:
                label_text = f"{label_text} {scores[i]:.2f}"
            
            # Get text size
            bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # Draw label background
            draw.rectangle(
                [x1, y1 - text_h - 4, x1 + text_w + 4, y1],
                fill=color
            )
            
            # Draw label text
            draw.text((x1 + 2, y1 - text_h - 2), label_text, fill=(255, 255, 255), font=font)
    
    return np.array(pil_image)


def create_batch_mosaic(
    images: List[np.ndarray],
    filenames: Optional[List[str]] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    max_images: int = 16,
    image_size: int = 640,
) -> np.ndarray:
    """
    Create a mosaic grid of images.
    
    Args:
        images: List of images as numpy arrays.
        filenames: List of filenames to overlay.
        grid_size: Grid size (rows, cols). If None, auto-calculated.
        max_images: Maximum number of images to include.
        image_size: Size of each image in the grid.
    
    Returns:
        Mosaic image as numpy array.
    """
    n_images = min(len(images), max_images)
    
    if grid_size is None:
        # Calculate grid size (prefer square-ish grid)
        cols = int(math.ceil(math.sqrt(n_images)))
        rows = int(math.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    # Create canvas
    mosaic_h = rows * image_size
    mosaic_w = cols * image_size
    mosaic = np.full((mosaic_h, mosaic_w, 3), 114, dtype=np.uint8)  # Gray background
    
    for i in range(n_images):
        row = i // cols
        col = i % cols
        
        img = images[i]
        h, w = img.shape[:2]
        
        # Resize to fit in grid cell
        scale = min(image_size / h, image_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Calculate position (center in cell)
        y_offset = row * image_size + (image_size - new_h) // 2
        x_offset = col * image_size + (image_size - new_w) // 2
        
        # Place in mosaic
        mosaic[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
        
        # Add filename overlay
        if filenames is not None and i < len(filenames):
            filename = filenames[i]
            # Draw filename in top-left corner of cell
            cell_x = col * image_size
            cell_y = row * image_size
            
            # Convert to PIL for text drawing
            pil_mosaic = Image.fromarray(mosaic)
            draw = ImageDraw.Draw(pil_mosaic)
            
            try:
                font = ImageFont.truetype("arial.ttf", 10)
            except:
                font = ImageFont.load_default()
            
            # Draw text with background
            draw.text((cell_x + 2, cell_y + 2), filename, fill=(0, 255, 255), font=font)
            mosaic = np.array(pil_mosaic)
    
    return mosaic


def visualize_batch(
    images: Union[torch.Tensor, List[torch.Tensor], List[np.ndarray]],
    targets: List[Dict[str, Any]],
    predictions: Optional[List[Dict[str, Any]]] = None,
    filenames: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    max_images: int = 16,
    image_size: int = 640,
    show_labels: bool = True,
    denormalize: bool = True,
) -> np.ndarray:
    """
    Visualize a batch of images with bounding boxes.
    
    Args:
        images: Batch of images (B, C, H, W) or list of images.
        targets: List of target dicts with 'boxes' and 'labels'.
        predictions: Optional list of prediction dicts with 'boxes', 'labels', 'scores'.
        filenames: List of filenames to overlay.
        class_names: List of class names.
        max_images: Maximum number of images.
        image_size: Size of each image in mosaic.
        show_labels: Whether to show labels on boxes.
        denormalize: Whether to denormalize images.
    
    Returns:
        Mosaic image as numpy array.
    """
    # Convert batch tensor to list
    if isinstance(images, torch.Tensor):
        images = [images[i] for i in range(images.shape[0])]
    
    processed_images = []
    
    for i, (img, target) in enumerate(zip(images[:max_images], targets[:max_images])):
        # Denormalize if needed
        if denormalize:
            img_np = denormalize_image(img)
        elif isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy()
            if img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img
        
        # Get boxes and labels
        boxes = target.get('boxes', torch.zeros((0, 4)))
        labels = target.get('labels', torch.zeros(0))
        
        # Handle normalized boxes (cxcywh format after Normalize)
        if len(boxes) > 0:
            # Check if boxes are normalized (values <= 1)
            if isinstance(boxes, torch.Tensor) and boxes.max() <= 1.0:
                h, w = img_np.shape[:2]
                # Convert from cxcywh to xyxy
                boxes_xyxy = boxes.clone()
                boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w  # x1
                boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h  # y1
                boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w  # x2
                boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h  # y2
                boxes = boxes_xyxy
        
        # Use predictions if provided
        if predictions is not None and i < len(predictions):
            pred = predictions[i]
            boxes = pred.get('boxes', boxes)
            labels = pred.get('labels', labels)
            scores = pred.get('scores', None)
        else:
            scores = None
        
        # Draw boxes
        img_with_boxes = draw_boxes(
            img_np, boxes, labels,
            scores=scores,
            class_names=class_names,
            show_labels=show_labels,
            show_conf=scores is not None,
        )
        
        processed_images.append(img_with_boxes)
    
    # Create mosaic
    mosaic = create_batch_mosaic(
        processed_images,
        filenames=filenames[:max_images] if filenames else None,
        max_images=max_images,
        image_size=image_size,
    )
    
    return mosaic


def save_batch_visualization(
    images: Union[torch.Tensor, List[torch.Tensor]],
    targets: List[Dict[str, Any]],
    save_path: Union[str, Path],
    predictions: Optional[List[Dict[str, Any]]] = None,
    filenames: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    **kwargs
) -> None:
    """
    Visualize batch and save to file.
    
    Args:
        images: Batch of images.
        targets: List of target dicts.
        save_path: Path to save the visualization.
        predictions: Optional predictions.
        filenames: Optional filenames.
        class_names: Optional class names.
        **kwargs: Additional arguments for visualize_batch.
    """
    mosaic = visualize_batch(
        images, targets,
        predictions=predictions,
        filenames=filenames,
        class_names=class_names,
        **kwargs
    )
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(mosaic).save(save_path, quality=95)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in xyxy format.
    
    Args:
        box1: First box [x1, y1, x2, y2].
        box2: Second box [x1, y1, x2, y2].
    
    Returns:
        IoU value.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def match_predictions_to_gt(
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: Optional[np.ndarray] = None,
    iou_threshold: float = 0.5,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Match predictions to ground truth boxes.
    
    Args:
        gt_boxes: Ground truth boxes (N, 4) in xyxy format.
        gt_labels: Ground truth labels (N,).
        pred_boxes: Prediction boxes (M, 4) in xyxy format.
        pred_labels: Prediction labels (M,).
        pred_scores: Prediction scores (M,). Optional.
        iou_threshold: IoU threshold for matching.
    
    Returns:
        Tuple of (tp_pred_indices, fp_pred_indices, fn_gt_indices).
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return [], [], []
    
    if len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes)))
    
    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), []
    
    # Sort predictions by score if available
    if pred_scores is not None:
        sorted_indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_indices]
        pred_labels = pred_labels[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
    else:
        sorted_indices = np.arange(len(pred_boxes))
    
    matched_gt = set()
    tp_pred_indices = []
    fp_pred_indices = []
    
    for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_idx in matched_gt:
                continue
            
            # Check class match
            if int(pred_label) != int(gt_label):
                continue
            
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        original_idx = sorted_indices[pred_idx]
        
        if best_iou >= iou_threshold:
            tp_pred_indices.append(original_idx)
            matched_gt.add(best_gt_idx)
        else:
            fp_pred_indices.append(original_idx)
    
    # False negatives are unmatched ground truths
    fn_gt_indices = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    
    return tp_pred_indices, fp_pred_indices, fn_gt_indices


def create_analysis_grid(
    image: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    iou_threshold: float = 0.5,
    image_size: int = 640,
) -> np.ndarray:
    """
    Create a 2x2 analysis grid visualization.
    
    Layout:
        +---------------+---------------+
        | Ground Truth  | True Positive |
        +---------------+---------------+
        | False Positive| False Negative|
        +---------------+---------------+
    
    Args:
        image: Input image (H, W, C) in RGB format.
        gt_boxes: Ground truth boxes (N, 4) in xyxy format.
        gt_labels: Ground truth labels (N,).
        pred_boxes: Prediction boxes (M, 4) in xyxy format.
        pred_labels: Prediction labels (M,).
        pred_scores: Prediction scores (M,). Optional.
        class_names: List of class names.
        iou_threshold: IoU threshold for matching.
        image_size: Size of each panel in the grid.
    
    Returns:
        Grid image as numpy array.
    """
    # Ensure numpy arrays
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
    
    # Match predictions to ground truth
    tp_indices, fp_indices, fn_indices = match_predictions_to_gt(
        gt_boxes, gt_labels,
        pred_boxes, pred_labels, pred_scores,
        iou_threshold=iou_threshold,
    )
    
    # Resize image to fit in grid
    h, w = image.shape[:2]
    scale = min(image_size / h, image_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (new_w, new_h))
    
    # Scale boxes
    def scale_boxes(boxes):
        if len(boxes) == 0:
            return boxes
        scaled = boxes.copy()
        scaled[:, [0, 2]] *= scale
        scaled[:, [1, 3]] *= scale
        return scaled
    
    gt_boxes_scaled = scale_boxes(gt_boxes)
    pred_boxes_scaled = scale_boxes(pred_boxes)
    
    # Scale-dependent sizes
    box_font_size = max(14, image_size // 40)      # Box label font
    title_font_size = max(18, image_size // 30)    # Panel title font
    box_line_width = max(2, image_size // 240)     # Box outline width
    separator_width = max(1, image_size // 320)    # Line between panels
    
    # Gray background color (like YOLO)
    bg_color = 128
    
    # Create 4 panels
    def create_panel(img_base, boxes, labels, scores, title, box_color=None):
        """Create a single panel with boxes and title overlay."""
        # Create canvas with gray background
        panel = np.full((image_size, image_size, 3), bg_color, dtype=np.uint8)
        
        # Center image in panel (preserve aspect ratio)
        y_offset = (image_size - new_h) // 2
        x_offset = (image_size - new_w) // 2
        panel[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_base
        
        # Convert to PIL for drawing
        pil_panel = Image.fromarray(panel)
        draw = ImageDraw.Draw(pil_panel)
        
        # Load fonts
        try:
            box_font = ImageFont.truetype("arial.ttf", box_font_size)
            title_font = ImageFont.truetype("arial.ttf", title_font_size)
        except:
            try:
                box_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", box_font_size)
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", title_font_size)
            except:
                box_font = ImageFont.load_default()
                title_font = ImageFont.load_default()
        
        # Draw title overlay (top-left, white text)
        draw.text((5, 2), title, fill=(255, 255, 255), font=title_font)
        
        # Draw boxes
        if len(boxes) > 0:
            for i, (box, label) in enumerate(zip(boxes, labels)):
                x1, y1, x2, y2 = box.astype(int)
                x1 += x_offset
                x2 += x_offset
                y1 += y_offset
                y2 += y_offset
                
                if box_color is not None:
                    color = box_color
                else:
                    color = get_color(int(label))
                
                draw.rectangle([x1, y1, x2, y2], outline=color, width=box_line_width)
                
                # Label text: "class conf" or just "class 1.0" for GT
                if class_names is not None and int(label) < len(class_names):
                    label_text = class_names[int(label)]
                else:
                    label_text = str(int(label))
                
                if scores is not None and i < len(scores):
                    label_text = f"{label_text} {scores[i]:.1f}"
                else:
                    label_text = f"{label_text} 1.0"  # GT always 1.0
                
                bbox = draw.textbbox((x1, y1), label_text, font=box_font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                padding = 2
                # Draw label background and text
                draw.rectangle([x1, y1 - text_h - padding * 2, x1 + text_w + padding * 2, y1], fill=color)
                draw.text((x1 + padding, y1 - text_h - padding), label_text, fill=(255, 255, 255), font=box_font)
        
        return np.array(pil_panel)
    
    # Panel 1: Ground Truth (per-class colors)
    gt_panel = create_panel(
        image_resized.copy(),
        gt_boxes_scaled, gt_labels, None,
        "Ground Truth",
    )
    
    # Panel 2: True Positives (matched predictions)
    if len(tp_indices) > 0:
        tp_boxes = pred_boxes_scaled[tp_indices]
        tp_labels = pred_labels[tp_indices]
        tp_scores = pred_scores[tp_indices] if pred_scores is not None else None
    else:
        tp_boxes = np.zeros((0, 4))
        tp_labels = np.zeros(0)
        tp_scores = None
    
    tp_panel = create_panel(
        image_resized.copy(),
        tp_boxes, tp_labels, tp_scores,
        "True Positives",
    )
    
    # Panel 3: False Positives (unmatched predictions)
    if len(fp_indices) > 0:
        fp_boxes = pred_boxes_scaled[fp_indices]
        fp_labels = pred_labels[fp_indices]
        fp_scores = pred_scores[fp_indices] if pred_scores is not None else None
    else:
        fp_boxes = np.zeros((0, 4))
        fp_labels = np.zeros(0)
        fp_scores = None
    
    fp_panel = create_panel(
        image_resized.copy(),
        fp_boxes, fp_labels, fp_scores,
        "False Positives",
    )
    
    # Panel 4: False Negatives (unmatched GT)
    if len(fn_indices) > 0:
        fn_boxes = gt_boxes_scaled[fn_indices]
        fn_labels = gt_labels[fn_indices]
    else:
        fn_boxes = np.zeros((0, 4))
        fn_labels = np.zeros(0)
    
    fn_panel = create_panel(
        image_resized.copy(),
        fn_boxes, fn_labels, None,
        "False Negatives",
    )
    
    # Combine panels into 2x2 grid with separator lines
    panel_h, panel_w = gt_panel.shape[:2]
    grid_h = panel_h * 2 + separator_width
    grid_w = panel_w * 2 + separator_width
    
    grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)
    
    # Place panels
    grid[0:panel_h, 0:panel_w] = gt_panel                                    # Top-left
    grid[0:panel_h, panel_w + separator_width:] = tp_panel                   # Top-right
    grid[panel_h + separator_width:, 0:panel_w] = fp_panel                   # Bottom-left
    grid[panel_h + separator_width:, panel_w + separator_width:] = fn_panel  # Bottom-right
    
    return grid


class AnalysisVisualizer:
    """
    Helper class for saving per-image analysis visualizations.
    
    Creates 2x2 grid visualizations showing:
    - Ground Truth
    - True Positives
    - False Positives  
    - False Negatives
    
    Args:
        save_dir: Directory to save visualizations.
        class_names: List of class names.
        iou_threshold: IoU threshold for matching.
        conf_threshold: Confidence threshold for predictions.
        image_size: Size of each panel in the grid.
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        class_names: Optional[List[str]] = None,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.25,
        image_size: int = 320,
    ):
        self.save_dir = Path(save_dir) / "visualizations"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.image_size = image_size
        
        self._count = 0
        self._stats = {
            'total_gt': 0,
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
        }
    
    def save_visualization(
        self,
        image: Union[torch.Tensor, np.ndarray],
        gt_boxes: Union[torch.Tensor, np.ndarray],
        gt_labels: Union[torch.Tensor, np.ndarray],
        pred_boxes: Union[torch.Tensor, np.ndarray],
        pred_labels: Union[torch.Tensor, np.ndarray],
        pred_scores: Optional[Union[torch.Tensor, np.ndarray]] = None,
        filename: Optional[str] = None,
        denormalize: bool = True,
    ) -> Path:
        """
        Save analysis visualization for a single image.
        
        Args:
            image: Input image (C, H, W) or (H, W, C).
            gt_boxes: Ground truth boxes (N, 4) in xyxy format.
            gt_labels: Ground truth labels (N,).
            pred_boxes: Prediction boxes (M, 4) in xyxy format.
            pred_labels: Prediction labels (M,).
            pred_scores: Prediction scores (M,). Optional.
            filename: Original filename for naming.
            denormalize: Whether to denormalize the image.
        
        Returns:
            Path to saved visualization.
        """
        # Process image
        if denormalize:
            img_np = denormalize_image(image)
        elif isinstance(image, torch.Tensor):
            img_np = image.cpu().numpy()
            if img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = image.copy()
        
        # Convert tensors to numpy
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
        
        # Filter predictions by confidence
        if pred_scores is not None and len(pred_scores) > 0:
            mask = pred_scores >= self.conf_threshold
            pred_boxes = pred_boxes[mask]
            pred_labels = pred_labels[mask]
            pred_scores = pred_scores[mask]
        
        # Handle normalized boxes (cxcywh format)
        h, w = img_np.shape[:2]
        
        if len(gt_boxes) > 0 and np.max(gt_boxes) <= 1.0:
            # Convert from cxcywh to xyxy
            gt_boxes_xyxy = np.zeros_like(gt_boxes)
            gt_boxes_xyxy[:, 0] = (gt_boxes[:, 0] - gt_boxes[:, 2] / 2) * w
            gt_boxes_xyxy[:, 1] = (gt_boxes[:, 1] - gt_boxes[:, 3] / 2) * h
            gt_boxes_xyxy[:, 2] = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2) * w
            gt_boxes_xyxy[:, 3] = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2) * h
            gt_boxes = gt_boxes_xyxy
        
        if len(pred_boxes) > 0 and np.max(pred_boxes) <= 1.0:
            pred_boxes_xyxy = np.zeros_like(pred_boxes)
            pred_boxes_xyxy[:, 0] = (pred_boxes[:, 0] - pred_boxes[:, 2] / 2) * w
            pred_boxes_xyxy[:, 1] = (pred_boxes[:, 1] - pred_boxes[:, 3] / 2) * h
            pred_boxes_xyxy[:, 2] = (pred_boxes[:, 0] + pred_boxes[:, 2] / 2) * w
            pred_boxes_xyxy[:, 3] = (pred_boxes[:, 1] + pred_boxes[:, 3] / 2) * h
            pred_boxes = pred_boxes_xyxy
        
        # Create analysis grid
        grid = create_analysis_grid(
            img_np,
            gt_boxes, gt_labels,
            pred_boxes, pred_labels, pred_scores,
            class_names=self.class_names,
            iou_threshold=self.iou_threshold,
            image_size=self.image_size,
        )
        
        # Update stats
        tp_indices, fp_indices, fn_indices = match_predictions_to_gt(
            gt_boxes, gt_labels,
            pred_boxes, pred_labels, pred_scores,
            iou_threshold=self.iou_threshold,
        )
        self._stats['total_gt'] += len(gt_boxes)
        self._stats['total_tp'] += len(tp_indices)
        self._stats['total_fp'] += len(fp_indices)
        self._stats['total_fn'] += len(fn_indices)
        
        # Save
        if filename:
            stem = Path(filename).stem
            save_path = self.save_dir / f"{stem}_analysis.jpg"
        else:
            save_path = self.save_dir / f"image_{self._count:04d}_analysis.jpg"
        
        Image.fromarray(grid).save(save_path, quality=95)
        self._count += 1
        
        return save_path
    
    def get_stats(self) -> Dict[str, Any]:
        """Get accumulated statistics."""
        total_pred = self._stats['total_tp'] + self._stats['total_fp']
        precision = self._stats['total_tp'] / total_pred if total_pred > 0 else 0
        recall = self._stats['total_tp'] / self._stats['total_gt'] if self._stats['total_gt'] > 0 else 0
        
        return {
            'images_processed': self._count,
            'total_gt': self._stats['total_gt'],
            'total_tp': self._stats['total_tp'],
            'total_fp': self._stats['total_fp'],
            'total_fn': self._stats['total_fn'],
            'precision': precision,
            'recall': recall,
        }
    
    def print_stats(self) -> None:
        """Print accumulated statistics."""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("Analysis Statistics")
        print("=" * 50)
        print(f"  Images processed: {stats['images_processed']}")
        print(f"  Total GT:         {stats['total_gt']}")
        print(f"  True Positives:   {stats['total_tp']}")
        print(f"  False Positives:  {stats['total_fp']}")
        print(f"  False Negatives:  {stats['total_fn']}")
        print(f"  Precision:        {stats['precision']:.4f}")
        print(f"  Recall:           {stats['recall']:.4f}")
        print("=" * 50)


class BatchVisualizer:
    """
    Helper class for saving batch visualizations during training.
    
    Args:
        save_dir: Directory to save visualizations.
        class_names: List of class names.
        max_batches: Maximum number of batches to save.
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        class_names: Optional[List[str]] = None,
        max_batches: int = 3,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        self.max_batches = max_batches
        
        self._train_batch_count = 0
        self._val_batch_count = 0
    
    def save_train_batch(
        self,
        images: torch.Tensor,
        targets: List[Dict[str, Any]],
        batch_idx: int,
        filenames: Optional[List[str]] = None,
        is_last: bool = False,
    ) -> Optional[Path]:
        """
        Save training batch visualization.
        
        Args:
            images: Batch of images.
            targets: List of targets.
            batch_idx: Batch index.
            filenames: Optional filenames.
            is_last: Whether this is one of the last batches.
        
        Returns:
            Path to saved file or None if not saved.
        """
        # Save first few batches and last batches
        if batch_idx < self.max_batches or is_last:
            save_path = self.save_dir / f"train_batch{batch_idx}.jpg"
            save_batch_visualization(
                images, targets,
                save_path=save_path,
                filenames=filenames,
                class_names=self.class_names,
                show_labels=False,  # Train batches show class index only
            )
            return save_path
        return None
    
    def save_val_batch(
        self,
        images: torch.Tensor,
        targets: List[Dict[str, Any]],
        predictions: Optional[List[Dict[str, Any]]] = None,
        batch_idx: int = 0,
        filenames: Optional[List[str]] = None,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Save validation batch visualizations (labels and predictions).
        
        Args:
            images: Batch of images.
            targets: List of targets.
            predictions: Optional predictions.
            batch_idx: Batch index.
            filenames: Optional filenames.
        
        Returns:
            Tuple of (labels_path, predictions_path).
        """
        if batch_idx >= self.max_batches:
            return None, None
        
        # Save ground truth labels
        labels_path = self.save_dir / f"val_batch{batch_idx}_labels.jpg"
        save_batch_visualization(
            images, targets,
            save_path=labels_path,
            filenames=filenames,
            class_names=self.class_names,
            show_labels=True,
        )
        
        # Save predictions if available
        pred_path = None
        if predictions is not None:
            pred_path = self.save_dir / f"val_batch{batch_idx}_pred.jpg"
            save_batch_visualization(
                images, targets,  # Use targets for image, but predictions for boxes
                predictions=predictions,
                save_path=pred_path,
                filenames=filenames,
                class_names=self.class_names,
                show_labels=True,
            )
        
        return labels_path, pred_path
