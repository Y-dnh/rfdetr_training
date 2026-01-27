"""
Convert YOLO format annotations to COCO JSON format.

Usage:
    python -m rfdetr.util.convert_yolo_to_coco --labels path/to/labels --images path/to/images
    
    # Or from util directory:
    python convert_yolo_to_coco.py --labels ../../dataset/test/labels --images ../../dataset/test/images
"""

import argparse
import json
from pathlib import Path
from PIL import Image


def yolo_to_coco(labels_dir: str, images_dir: str, output_name: str = "_annotations.coco.json"):
    """
    Convert YOLO format labels to COCO JSON format.
    
    Args:
        labels_dir: Path to folder with .txt YOLO labels
        images_dir: Path to folder with images (output will be saved here)
        output_name: Name of output JSON file
    """
    
    labels_path = Path(labels_dir)
    images_path = Path(images_dir)
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_path}")
    
    # COCO format structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Get unique classes first
    classes = set()
    for txt_file in labels_path.glob("*.txt"):
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
    
    # Create categories
    for class_id in sorted(classes):
        coco["categories"].append({
            "id": class_id,
            "name": f"class_{class_id}",
            "supercategory": "object"
        })
    
    annotation_id = 1
    image_id = 1
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    for txt_file in sorted(labels_path.glob("*.txt")):
        basename = txt_file.stem
        
        # Find corresponding image
        image_file = None
        for ext in image_extensions:
            candidate = images_path / f"{basename}{ext}"
            if candidate.exists():
                image_file = candidate
                break
        
        if image_file is None:
            print(f"Warning: No image found for {txt_file.name}")
            continue
        
        # Get image dimensions
        try:
            with Image.open(image_file) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error reading {image_file}: {e}")
            continue
        
        # Add image entry
        coco["images"].append({
            "id": image_id,
            "file_name": image_file.name,
            "width": width,
            "height": height
        })
        
        # Read YOLO annotations
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center, y_center = float(parts[1]), float(parts[2])
                w, h = float(parts[3]), float(parts[4])
                
                # Convert YOLO (normalized) to COCO (pixels)
                x = (x_center - w/2) * width
                y = (y_center - h/2) * height
                w_px, h_px = w * width, h * height
                
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x, y, w_px, h_px],
                    "area": w_px * h_px,
                    "iscrowd": 0
                })
                annotation_id += 1
        
        image_id += 1
    
    # Save COCO JSON
    output_file = images_path / output_name
    with open(output_file, 'w') as f:
        json.dump(coco, f, indent=2)
    
    print(f"Saved: {output_file}")
    print(f"Images: {len(coco['images'])}, Annotations: {len(coco['annotations'])}, Categories: {len(coco['categories'])}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO annotations to COCO JSON format")
    parser.add_argument("--labels", "-l", required=True, help="Path to YOLO labels folder")
    parser.add_argument("--images", "-i", required=True, help="Path to images folder (output saved here)")
    parser.add_argument("--output", "-o", default="_annotations.coco.json", help="Output filename")
    
    args = parser.parse_args()
    yolo_to_coco(args.labels, args.images, args.output)


if __name__ == "__main__":
    import sys
    
    # ============================================================
    # QUICK RUN: Change PATH below and run file directly in IDE
    # ============================================================
    PATH = "dataset/test"  # <- Change this! e.g. "dataset/train", "dataset/valid"
    # ============================================================
    
    if len(sys.argv) == 1:  # No CLI arguments - use PATH variable
        # Get project root (2 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        
        labels_dir = project_root / PATH / "labels"
        images_dir = project_root / PATH / "images"
        
        print(f"Converting: {PATH}")
        print(f"  Labels: {labels_dir}")
        print(f"  Images: {images_dir}")
        
        yolo_to_coco(str(labels_dir), str(images_dir))
    else:
        main()
