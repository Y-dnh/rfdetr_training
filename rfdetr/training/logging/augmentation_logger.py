"""
Augmentation logging for RF-DETR training.

Logs which augmentations were applied to each image during training,
useful for debugging and experiment reproducibility.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class AugmentationLogger:
    """
    Logger for tracking augmentations applied during training.
    
    Logs are saved as JSON files containing:
    - Image filename
    - Applied augmentations with parameters
    - Timestamp
    
    Example:
        >>> logger = AugmentationLogger(save_dir='runs/exp1/logs')
        >>> logger.log_batch(
        ...     batch_idx=0,
        ...     filenames=['img1.jpg', 'img2.jpg'],
        ...     augmentations=[
        ...         {'mosaic': {'indices': [0, 5, 10, 15]}, 'hsv': {'h': 0.01}},
        ...         {'mosaic': {'indices': [1, 3, 7, 9]}, 'hsv': {'h': -0.02}},
        ...     ]
        ... )
        >>> logger.save()
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        log_every_n_batches: int = 100,
        max_entries: int = 10000,
    ):
        """
        Initialize augmentation logger.
        
        Args:
            save_dir: Directory to save log files.
            log_every_n_batches: Only log every N batches to reduce overhead.
            max_entries: Maximum number of entries to keep in memory.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_every_n_batches = log_every_n_batches
        self.max_entries = max_entries
        
        self.entries: List[Dict[str, Any]] = []
        self.batch_count = 0
        self.epoch = 0
    
    def set_epoch(self, epoch: int) -> None:
        """Set current epoch number."""
        self.epoch = epoch
    
    def log_batch(
        self,
        batch_idx: int,
        filenames: List[str],
        augmentations: List[Dict[str, Any]],
    ) -> None:
        """
        Log augmentations for a batch.
        
        Args:
            batch_idx: Batch index.
            filenames: List of image filenames in the batch.
            augmentations: List of augmentation dicts, one per image.
        """
        self.batch_count += 1
        
        # Only log every N batches
        if self.batch_count % self.log_every_n_batches != 0:
            return
        
        timestamp = datetime.now().isoformat()
        
        entry = {
            'epoch': self.epoch,
            'batch_idx': batch_idx,
            'timestamp': timestamp,
            'images': {},
        }
        
        for filename, aug in zip(filenames, augmentations):
            # Filter out None values and empty dicts
            filtered_aug = {k: v for k, v in aug.items() if v is not None and v != {}}
            entry['images'][filename] = filtered_aug
        
        self.entries.append(entry)
        
        # Trim entries if too many
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
    
    def log_image(
        self,
        filename: str,
        augmentations: Dict[str, Any],
        batch_idx: Optional[int] = None,
    ) -> None:
        """
        Log augmentations for a single image.
        
        Args:
            filename: Image filename.
            augmentations: Dict of augmentation name to parameters.
            batch_idx: Optional batch index.
        """
        entry = {
            'epoch': self.epoch,
            'batch_idx': batch_idx,
            'timestamp': datetime.now().isoformat(),
            'images': {filename: augmentations},
        }
        
        self.entries.append(entry)
        
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
    
    def log(self, aug_log: Dict[str, Any]) -> None:
        """
        Log augmentation data from pipeline.
        
        Args:
            aug_log: Dictionary with 'image_name', 'augmentations', 'applied' keys.
        """
        if aug_log is None:
            return
        
        image_name = aug_log.get('image_name', 'unknown')
        augmentations = aug_log.get('augmentations', [])
        applied = aug_log.get('applied', [])
        
        # Convert list format to dict format
        aug_dict = {}
        for aug in applied:
            name = aug.get('name', 'unknown')
            params = aug.get('parameters', {})
            aug_dict[name] = params
        
        if aug_dict:
            self.log_image(image_name, aug_dict)
    
    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save log to JSON file.
        
        Args:
            filename: Optional filename. If None, uses default.
        
        Returns:
            Path to saved file.
        """
        if filename is None:
            filename = 'augmentation_log.json'
        
        save_path = self.save_dir / filename
        
        with open(save_path, 'w') as f:
            json.dump({
                'metadata': {
                    'total_entries': len(self.entries),
                    'last_epoch': self.epoch,
                    'saved_at': datetime.now().isoformat(),
                },
                'entries': self.entries,
            }, f, indent=2)
        
        return save_path
    
    def clear(self) -> None:
        """Clear all logged entries."""
        self.entries = []
        self.batch_count = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of logged augmentations.
        
        Returns:
            Dictionary with augmentation statistics.
        """
        aug_counts = {}
        
        for entry in self.entries:
            for img_augs in entry.get('images', {}).values():
                for aug_name in img_augs.keys():
                    aug_counts[aug_name] = aug_counts.get(aug_name, 0) + 1
        
        return {
            'total_logged_batches': len(self.entries),
            'augmentation_counts': aug_counts,
        }


class TrainingLogger:
    """
    Simple training logger for console and file output.
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        name: str = 'training',
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.save_dir / f'{name}.log'
        self.start_time = datetime.now()
    
    def log(self, message: str, level: str = 'INFO') -> None:
        """Log a message."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted = f"[{timestamp}] [{level}] {message}"
        
        print(formatted)
        
        with open(self.log_file, 'a') as f:
            f.write(formatted + '\n')
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.log(message, 'INFO')
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.log(message, 'WARNING')
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.log(message, 'ERROR')
    
    def epoch_summary(
        self,
        epoch: int,
        metrics: Dict[str, float],
        lr: float,
    ) -> None:
        """Log epoch summary."""
        metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.info(f"Epoch {epoch} | LR: {lr:.6f} | {metrics_str}")
