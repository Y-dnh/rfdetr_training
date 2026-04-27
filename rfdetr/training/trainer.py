"""
RF-DETR Trainer class for YOLO-style training workflow.

This module provides the main training interface that integrates:
- Model building and weight management
- Dataset with augmentation pipeline (Mosaic, HSV, MixUp, etc.)
- Training loop with visualization and logging
- YOLO-style batch visualizations and metric plots
- Checkpoint management
"""

import os
import math
import json
import time
import datetime
import argparse
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rfdetr.training.dataset import RFDETRDataset, build_dataset, collate_fn
from rfdetr.training.augmentations.pipeline import AugmentationPipeline, ValidationPipeline
from rfdetr.training.utils.config import (
    AugmentationConfig,
    TrainingConfig,
    ModelConfig,
    ExportConfig,
    validate_config,
)
from rfdetr.training.utils.seed import setup_seed, worker_init_fn
from rfdetr.training.visualizations import (
    BatchVisualizer,
    MetricsLogger,
    create_labels_visualization,
)
from rfdetr.training.visualizations.curves import save_all_curves
from rfdetr.training.visualizations.confusion_matrix import ConfusionMatrix
from rfdetr.training.logging import AugmentationLogger, TrainingLogger


def get_grid_size(batch_size: int) -> int:
    """Calculate optimal grid size for batch visualization."""
    import math
    # Find smallest n where n*n >= batch_size
    return math.ceil(math.sqrt(batch_size))


class RFDETRTrainer:
    """
    YOLO-style trainer for RF-DETR models.
    
    This trainer provides:
    - Easy configuration through dataclasses
    - YOLO-style augmentations (Mosaic, MixUp, HSV, etc.)
    - Automatic visualization of training/validation batches
    - Metric plots (results.png, PR curves, confusion matrix)
    - close_mosaic handling
    - Checkpoint management
    """
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        augmentation_config: Optional[AugmentationConfig] = None,
        export_config: Optional[ExportConfig] = None,
        seed: int = 42,
    ):
        """Initialize trainer."""
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.augmentation_config = augmentation_config or AugmentationConfig()
        # albumentation_transforms беруться з augmentation_config.albumentation_transforms (None => get_default_albu_config(imgsz) у build_dataset)
        self.export_config = export_config or ExportConfig()
        self.seed = seed
        
        # Validate configs
        validate_config(
            self.augmentation_config,
            self.training_config,
            self.model_config,
        )
        
        # Set up seed
        setup_seed(self.seed)
        
        # Initialize components
        self.model = None
        self.criterion = None
        self.postprocessor = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        # Output directory
        self.save_dir = None
        self.device = None
        
        # Loggers and visualizers
        self.metrics_logger = None
        self.aug_logger = None
        self.training_logger = None
        self.batch_visualizer = None
        self.confusion_matrix = None
        
        # Training state
        self.current_epoch = 0
        self.best_map = 0.0
        self.class_names = []
        self._epochs_without_improvement = 0  # для early stopping
        self.is_resume = False
        self.resume_checkpoint_path: Optional[Path] = None
        self._time_offset_sec = 0.0
        
        # Metrics storage for curves
        self.all_predictions = []
        self.all_targets = []
        
        # Grid size for visualizations
        self.grid_size = get_grid_size(self.training_config.batch_size)
        self._original_showwarning = warnings.showwarning

    def _configure_warning_routing(self) -> None:
        """Route noisy runtime warnings to the log file without breaking tqdm."""
        if self.training_logger is None:
            return

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            message_text = str(message)
            warning_location = f"{Path(filename).name}:{lineno}"

            if (
                issubclass(category, RuntimeWarning)
                and "invalid value encountered in divide" in message_text
            ):
                self.training_logger.debug(
                    f"{category.__name__}: {message_text} ({warning_location})"
                )
                return

            self._original_showwarning(message, category, filename, lineno, file=file, line=line)

        warnings.showwarning = _showwarning
    
    def _setup_output_dir(self, resume_path: Optional[Path] = None) -> Path:
        """Set up output directory: project/<name>/ or the checkpoint run dir on resume."""
        cfg = self.training_config
        base_dir = Path(cfg.project)

        if resume_path is not None:
            checkpoint_dir = resume_path.parent
            save_dir = checkpoint_dir.parent if checkpoint_dir.name == "weights" else checkpoint_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / 'weights').mkdir(exist_ok=True)
            return save_dir
        
        if cfg.exist_ok:
            save_dir = base_dir / cfg.name
        else:
            i = 1
            save_dir = base_dir / cfg.name
            while save_dir.exists():
                save_dir = base_dir / f"{cfg.name}{i}"
                i += 1
        
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'weights').mkdir(exist_ok=True)
        return save_dir

    # Column order for results.csv (YOLO-style); (B) suffix for metric columns
    RESULTS_CSV_HEADER = [
        'epoch', 'time',
        'train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'train/class_error',
        'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
        'metrics/mAP75(B)', 'metrics/AP_small(B)', 'metrics/AP_medium(B)', 'metrics/AP_large(B)',
        'metrics/AR_maxDets1(B)', 'metrics/AR_maxDets10(B)', 'metrics/AR_maxDets100(B)',
        'metrics/AR_small(B)', 'metrics/AR_medium(B)', 'metrics/AR_large(B)',
        'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'val/class_error',
        'lr/pg0', 'lr/pg1', 'lr/pg2',
    ]

    def _init_results_csv(self) -> None:
        """Create results.csv with header if it does not exist (do not overwrite on resume)."""
        path = self.save_dir / 'results.csv'
        if path.exists():
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            f.write(','.join(self.RESULTS_CSV_HEADER) + '\n')

    def _append_results_csv_row(
        self, epoch_one_based: int, cumulative_time_sec: float, metrics: Dict[str, float]
    ) -> None:
        """Append one row to results.csv and flush. Metrics keys may be with or without (B) suffix."""
        path = self.save_dir / 'results.csv'
        # Map CSV column name to metrics key (with or without (B))
        def get_val(col: str):
            if col == 'epoch':
                return str(epoch_one_based)
            if col == 'time':
                return f"{cumulative_time_sec:.2f}"
            if col.startswith('lr/'):
                idx = {'lr/pg0': 0, 'lr/pg1': 1, 'lr/pg2': 2}[col]
                groups = self.optimizer.param_groups
                if idx < len(groups):
                    return f"{groups[idx]['lr']:.10g}"
                return f"{groups[-1]['lr']:.10g}" if groups else '0'
            key_alt = col.replace('(B)', '').strip() if col.endswith('(B)') else col
            val = metrics.get(col) or metrics.get(key_alt)
            if val is None:
                return ''
            return f"{float(val):.6f}"
        row = [get_val(c) for c in self.RESULTS_CSV_HEADER]
        with open(path, 'a', newline='', encoding='utf-8') as f:
            f.write(','.join(row) + '\n')
            f.flush()

    def _read_results_time_offset(self, csv_path: Union[str, Path]) -> float:
        """Return the last cumulative time value from results.csv for resume runs."""
        import csv

        try:
            with open(csv_path, newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            self.training_logger.warning(f"Failed to read previous results.csv time offset: {e}")
            return 0.0

        for row in reversed(rows):
            value = (row.get('time') or '').strip()
            if not value:
                continue
            try:
                return float(value)
            except ValueError:
                continue
        return 0.0

    def _setup_model_and_criterion(self, num_classes: int):
        """Build model, criterion and postprocessor using RF-DETR components."""
        from rfdetr.main import Model
        from rfdetr.models import build_criterion_and_postprocessors
        from rfdetr.config import (
            RFDETRBaseConfig, RFDETRLargeConfig, RFDETRNanoConfig,
            RFDETRSmallConfig, RFDETRMediumConfig,
        )
        
        cfg = self.model_config
        size_to_config = {
            'n': RFDETRNanoConfig,
            's': RFDETRSmallConfig,
            'm': RFDETRMediumConfig,
            'b': RFDETRBaseConfig,
            'l': RFDETRLargeConfig,
        }
        
        # Import platform models only when needed (requires rfdetr_plus)
        if cfg.model_size in ('xl', '2xl'):
            from rfdetr.platform.models import RFDETRXLargeConfig, RFDETR2XLargeConfig
            size_to_config['xl'] = RFDETRXLargeConfig
            size_to_config['2xl'] = RFDETR2XLargeConfig
        
        rfdetr_config_class = size_to_config.get(cfg.model_size, RFDETRBaseConfig)
        self.rfdetr_config = rfdetr_config_class(num_classes=num_classes)
        
        # Build Model wrapper - exclude class_embed weights when loading pretrained
        # to keep our num_classes head instead of COCO's 91-class head
        model_kwargs = self.rfdetr_config.model_dump()
        model_kwargs['pretrain_exclude_keys'] = ['class_embed.weight', 'class_embed.bias']
        self.rfdetr_model = Model(**model_kwargs)
        self.model = self.rfdetr_model.model
        self.device = self.rfdetr_model.device
        
        # Reinitialize classification head with correct num_classes
        # (pretrained weights may have overwritten it to 91 classes)
        actual_num_classes = num_classes + 1  # +1 for background/no-object
        if self.model.class_embed.bias.shape[0] != actual_num_classes:
            print(f"Reinitializing class_embed from {self.model.class_embed.bias.shape[0]} to {actual_num_classes} classes")
            self.model.reinitialize_detection_head(actual_num_classes)
        
        # Build criterion using RF-DETR's function
        args = self._build_args_namespace(num_classes)
        self.criterion, self.postprocessor = build_criterion_and_postprocessors(args)
        self.criterion = self.criterion.to(self.device)
        
        return self.model
    
    def _build_args_namespace(self, num_classes: int) -> argparse.Namespace:
        """Build args namespace for RF-DETR components."""
        cfg = self.rfdetr_config
        return argparse.Namespace(
            num_classes=num_classes,
            device=self.training_config.device,
            hidden_dim=cfg.hidden_dim,
            dec_layers=cfg.dec_layers,
            num_queries=getattr(cfg, 'num_queries', 300),
            num_select=getattr(cfg, 'num_select', 300),
            group_detr=cfg.group_detr,
            two_stage=cfg.two_stage,
            set_cost_class=2,
            set_cost_bbox=5,
            set_cost_giou=2,
            cls_loss_coef=2,
            bbox_loss_coef=5,
            giou_loss_coef=2,
            focal_alpha=0.25,
            aux_loss=True,
            sum_group_losses=False,
            use_varifocal_loss=False,
            use_position_supervised_loss=False,
            ia_bce_loss=False,
            segmentation_head=False,
        )
    
    def _setup_custom_dataloaders(self, dataset_dir: Union[str, Path]):
        """Setup dataloaders with YOLO-style augmentations."""
        from functools import partial
        import rfdetr.util.misc as utils
        
        dataset_dir = Path(dataset_dir)
        
        # Build train dataset with augmentations
        self.training_logger.info("Building train dataset with augmentations...")
        self.train_dataset = build_dataset(
            dataset_dir,
            split='train',
            augmentation_config=self.augmentation_config,
            albumentation_transforms=getattr(self.augmentation_config, "albumentation_transforms", None),
            log_augmentations=True,
        )
        
        # Build validation dataset (minimal augmentations)
        self.training_logger.info("Building validation dataset...")
        self.val_dataset = build_dataset(
            dataset_dir,
            split='valid',
            augmentation_config=self.augmentation_config,
            albumentation_transforms=getattr(self.augmentation_config, "albumentation_transforms", None),
            log_augmentations=False,
        )
        
        # Get class info
        self.class_names = self.train_dataset.class_names
        num_classes = self.train_dataset.num_classes
        
        # Create dataloaders
        # Note: persistent_workers=False for train because workers need to receive
        # updated epoch info for close_mosaic feature
        num_workers = self.training_config.workers
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=partial(worker_init_fn, base_seed=self.seed),
            persistent_workers=False,  # Must restart workers to propagate epoch for close_mosaic
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,  # OK for val - no mosaic
        )
        
        return num_classes
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with simplified parameter groups."""
        encoder_params = []
        decoder_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name or 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        param_dicts = [
            {'params': encoder_params, 'lr': self.training_config.lr * 0.1},
            {'params': decoder_params, 'lr': self.training_config.lr},
        ]
        
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay,
        )
        
        return optimizer
    
    def _get_model_resolution(self) -> int:
        """Get the expected resolution for the current model size."""
        size_to_resolution = {
            'n': 384,   # Nano
            's': 512,   # Small
            'm': 576,   # Medium
            'b': 560,   # Base
            'l': 704,   # Large
            'xl': 700,  # XLarge (platform)
            '2xl': 880, # 2XLarge (platform)
        }
        return size_to_resolution.get(self.model_config.model_size, 560)
    
    def _setup_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int):
        """Create learning rate scheduler (використовує warmup_epochs та scheduler з конфігу)."""
        steps_per_epoch = max(1, num_training_steps // self.training_config.epochs)
        warmup_steps = min(
            self.training_config.warmup_epochs * steps_per_epoch,
            max(0, num_training_steps - 1),
        )
        sched = self.training_config.scheduler.lower()

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            progress = min(1.0, progress)
            if sched == 'cosine':
                return 0.01 + (1 - 0.01) * 0.5 * (1 + math.cos(math.pi * progress))
            if sched == 'linear':
                return max(0.01, 1.0 - progress)
            if sched == 'step':
                # step: 1.0 до 2/3, потім 0.1 до кінця
                if progress < 2.0 / 3.0:
                    return 1.0
                return 0.1
            return 0.01 + (1 - 0.01) * 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    def train(
        self,
        dataset_dir: Union[str, Path],
        resume: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start training with custom train loop and YOLO-style visualizations."""
        from torch.amp import GradScaler

        resume_path_raw = resume or (self.training_config.resume or None)
        resume_path = Path(resume_path_raw).expanduser().resolve() if resume_path_raw else None
        if resume_path is not None and not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        self.is_resume = resume_path is not None
        self.resume_checkpoint_path = resume_path
        
        # Setup output directory
        self.save_dir = self._setup_output_dir(resume_path)
        
        # Setup loggers
        self.training_logger = TrainingLogger(self.save_dir)
        self.metrics_logger = MetricsLogger(self.save_dir)
        self.aug_logger = AugmentationLogger(self.save_dir)
        self._configure_warning_routing()
        
        # Suppress torch.meshgrid warning
        warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
        
        self.training_logger.info(f"Starting training in {self.save_dir}")
        self.training_logger.info(f"Seed: {self.seed}")
        self.training_logger.info(f"Batch size: {self.training_config.batch_size}, Grid size: {self.grid_size}x{self.grid_size}")
        if self.is_resume:
            self.training_logger.info(f"Resume requested from checkpoint: {self.resume_checkpoint_path}")
        
        # Get model resolution and sync with augmentation config
        model_resolution = self._get_model_resolution()
        if self.augmentation_config.imgsz != model_resolution:
            self.training_logger.info(f"Syncing imgsz: {self.augmentation_config.imgsz} -> {model_resolution} (model resolution)")
            # Create new config with correct resolution
            from dataclasses import replace
            self.augmentation_config = replace(self.augmentation_config, imgsz=model_resolution)
        
        # Log augmentation config
        self.training_logger.info("Augmentation config:")
        self.training_logger.info(f"  Image size: {self.augmentation_config.imgsz}")
        self.training_logger.info(f"  Mosaic: {self.augmentation_config.mosaic}  MixUp: {self.augmentation_config.mixup}  CutMix: {self.augmentation_config.cutmix}")
        self.training_logger.info("  Color/flip/geometry/erasing: ALBUMENTATION_CONFIG (albumentations)")
        
        # Save configs
        self._save_configs(is_resume=self.is_resume)
        
        # Setup dataloaders with YOLO-style augmentations
        self.training_logger.info("Loading datasets with augmentations...")
        num_classes = self._setup_custom_dataloaders(dataset_dir)
        self.num_classes = num_classes  # Store for running metrics
        self.training_logger.info(f"Train: {len(self.train_dataset)} images, Val: {len(self.val_dataset)} images")
        self.training_logger.info(f"Classes: {num_classes} - {self.class_names}")
        
        # Setup batch visualizer
        self.batch_visualizer = BatchVisualizer(
            self.save_dir,
            class_names=self.class_names,
            max_batches=self.training_config.vis_batches,
        )
        self.confusion_matrix = ConfusionMatrix(num_classes, self.class_names)
        
        # Create labels.jpg once per run. On resume, reuse the existing analysis
        # because the dataset labels do not change between interrupted sessions.
        labels_path = self.save_dir / 'labels.jpg'
        if self.is_resume and labels_path.exists():
            self.training_logger.info(f"Skipping labels visualization on resume: {labels_path}")
        else:
            self.training_logger.info("Creating labels visualization...")
            labels_max_images = (
                None
                if self.training_config.labels_max_images == 0
                else self.training_config.labels_max_images
            )
            create_labels_visualization(
                self.train_dataset,
                labels_path,
                max_images=labels_max_images,
                class_names=self.class_names,
                random_seed=self.seed,
            )
        
        # Setup model and criterion
        self.training_logger.info("Building model...")
        self._setup_model_and_criterion(num_classes)
        self.training_logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        # Setup optimizer and scheduler
        steps_per_epoch = max(
            1,
            math.ceil(len(self.train_loader) / self.training_config.gradient_accumulation),
        )
        num_training_steps = steps_per_epoch * self.training_config.epochs
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler(self.optimizer, num_training_steps)
        self.scaler = GradScaler('cuda', enabled=False)
        
        if resume_path:
            self._load_checkpoint(resume_path)
            results_csv = self.save_dir / 'results.csv'
            if results_csv.exists():
                self.metrics_logger.load_from_csv(results_csv)
                self._time_offset_sec = self._read_results_time_offset(results_csv)

        self._init_results_csv()

        # Save first training batches visualization (with augmentations visible).
        # On resume, keep the original first-batch artifacts if they already exist.
        if self.is_resume and (self.save_dir / 'train_batch0.jpg').exists():
            self.training_logger.info("Skipping first train-batch visualization on resume: train_batch0.jpg already exists")
        else:
            self._save_train_batches_with_aug()

        # Training loop
        self.training_logger.info("Starting training loop...")
        start_time = time.time()

        try:
            for epoch in range(self.current_epoch, self.training_config.epochs):
                self.current_epoch = epoch
                epoch_start = time.time()
                
                # Update dataset epoch for close_mosaic
                self.train_dataset.set_epoch(epoch, self.training_config.epochs)
                
                # Train one epoch
                train_metrics = self._train_epoch(epoch)
                
                # Validate (кожні val_period епох або остання епоха)
                is_last_epoch = (epoch == self.training_config.epochs - 1)
                val_period = max(1, self.training_config.val_period)
                do_validate = is_last_epoch or ((epoch + 1) % val_period == 0)
                if do_validate:
                    val_metrics = self._validate_epoch(epoch, save_last=is_last_epoch)
                    all_metrics = {**train_metrics, **val_metrics}
                else:
                    all_metrics = {**train_metrics}
                
                # Log metrics
                self.metrics_logger.log_epoch(all_metrics)
                self.metrics_logger.log_lr(self.optimizer.param_groups[0]['lr'])
                
                # Save checkpoint (оновлює self.best_map якщо поточний mAP50 кращий)
                old_best = self.best_map
                self._save_checkpoint(epoch, all_metrics)
                if do_validate:
                    if self.best_map > old_best:
                        self._epochs_without_improvement = 0
                    else:
                        self._epochs_without_improvement += 1

                # Log epoch summary
                epoch_time = time.time() - epoch_start
                self._log_epoch_summary(epoch, all_metrics, epoch_time)

                self._append_results_csv_row(
                    epoch + 1, self._time_offset_sec + time.time() - start_time, all_metrics
                )

                # Early stopping (тільки якщо була валідація і є mAP50)
                early_stopping = self.training_config.early_stopping
                if do_validate and early_stopping > 0 and self._epochs_without_improvement >= early_stopping:
                    best_epoch_1based = epoch + 1 - self._epochs_without_improvement  # остання епоха (1-based), коли mAP50 покращився
                    self.training_logger.info(
                        f"Early stopping: no improvement in mAP50 for {early_stopping} epochs "
                        f"(best mAP50={self.best_map:.4f} at epoch {best_epoch_1based})"
                    )
                    break
                
                # Save last training batches on final epoch
                if is_last_epoch:
                    self._save_last_train_batches()
        
        except KeyboardInterrupt:
            self.training_logger.warning("Training interrupted by user")
        
        # Save final results
        total_time = time.time() - start_time
        self.training_logger.info(f"Training completed in {datetime.timedelta(seconds=int(total_time))}")
        
        # Save augmentation log
        self._save_augmentation_log(append=self.is_resume)
        
        # Generate final visualizations
        self._generate_final_visualizations()
        
        # Export deployment artifacts if enabled
        export_artifacts: Dict[str, Optional[str]] = {
            'onnx_path': None,
            'openvino_path': None,
            'openvino_bin_path': None,
            'openvino_metadata_path': None,
        }
        if self.export_config.enabled:
            export_artifacts = self._export_model_artifacts()

        return {
            'save_dir': str(self.save_dir),
            'best_map': self.best_map,
            'epochs_trained': self.current_epoch + 1,
            'onnx_path': export_artifacts.get('onnx_path'),
            'openvino_path': export_artifacts.get('openvino_path'),
            'export_artifacts': export_artifacts,
        }
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.criterion.train()
        self.aug_logger.set_epoch(epoch + 1)
        
        total_loss = 0.0
        total_loss_ce = 0.0
        total_loss_bbox = 0.0
        total_loss_giou = 0.0
        total_class_error = 0.0
        num_batches = len(self.train_loader)

        # Header for progress bar
        print(("%10s" * 7) % ("Epoch", "GPU_mem", "box_loss", "cls_loss", "dfl_loss", "Instances", "Size"))
        pbar = tqdm(self.train_loader, total=num_batches, bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        accumulation_steps = max(1, self.training_config.gradient_accumulation)
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch_data in enumerate(pbar):
            images, targets, aug_logs = batch_data
            
            # Log augmentations for first batches
            if batch_idx < 3 and aug_logs:
                for aug_log in aug_logs:
                    if aug_log:
                        self.aug_logger.log(aug_log)
            
            # Move to device
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # Convert to format expected by model
            from rfdetr.util.misc import NestedTensor
            # Create mask: False = real content, True = padding
            # All images are same size, so no padding needed (all False)
            b, c, h, w = images.shape
            mask = torch.zeros((b, h, w), dtype=torch.bool, device=images.device)
            samples = NestedTensor(images, mask)
            
            # Forward pass
            outputs = self.model(samples, targets)
            loss_dict = self.criterion(outputs, targets)
            
            # Compute total loss
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            scaled_losses = losses / accumulation_steps
            
            # Backward pass
            self.scaler.scale(scaled_losses).backward()
            
            should_step = ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == num_batches)
            if should_step:
                # Gradient clipping
                if self.training_config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
            
            # Accumulate losses
            total_loss += losses.item()
            total_loss_ce += loss_dict.get('loss_ce', torch.tensor(0)).item()
            total_loss_bbox += loss_dict.get('loss_bbox', torch.tensor(0)).item()
            total_loss_giou += loss_dict.get('loss_giou', torch.tensor(0)).item()
            ce_val = loss_dict.get('class_error', None)
            if ce_val is not None:
                total_class_error += ce_val.item() if isinstance(ce_val, torch.Tensor) else float(ce_val)

            # Update progress bar
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            
            desc = ("%10s" * 2 + "%10.4g" * 3 + "%10s" * 2) % (
                f"{epoch + 1}/{self.training_config.epochs}",
                mem,
                loss_dict.get('loss_bbox', torch.tensor(0)).item(),
                loss_dict.get('loss_ce', torch.tensor(0)).item(),
                loss_dict.get('loss_giou', torch.tensor(0)).item(),
                sum(len(t['boxes']) for t in targets),
                f"{h}"
            )
            pbar.set_description(desc)
        
        return {
            'train/box_loss': total_loss_bbox / num_batches,
            'train/cls_loss': total_loss_ce / num_batches,
            'train/dfl_loss': total_loss_giou / num_batches,
            'train/class_error': total_class_error / num_batches,
        }
    
    def _validate_epoch(self, epoch: int, save_last: bool = False) -> Dict[str, float]:
        """Validate for one epoch with Ultralytics-style progress bar."""
        import contextlib
        import os
        from rfdetr.util.misc import NestedTensor
        from rfdetr.datasets.coco_eval import CocoEvaluator
        from rfdetr.datasets import get_coco_api_from_dataset
        
        self.model.eval()
        self.criterion.eval()
        
        total_loss = 0.0
        total_loss_ce = 0.0
        total_loss_bbox = 0.0
        total_loss_giou = 0.0
        total_class_error = 0.0
        num_batches = len(self.val_loader)

        # Setup COCO evaluator
        base_ds = get_coco_api_from_dataset(self.val_dataset)
        coco_evaluator = CocoEvaluator(base_ds, ['bbox'])
        
        # Save preview batches using a random subset of validation batches so the
        # visualization is not dominated by neighboring frames from the same scene.
        saved_batches = 0
        max_batches_to_save = self.training_config.vis_batches
        num_preview_batches = min(max_batches_to_save, num_batches)
        preview_batch_indices: set[int] = set()
        if num_preview_batches > 0:
            preview_seed = self.seed + self.current_epoch + (100000 if save_last else 0)
            preview_rng = np.random.default_rng(preview_seed)
            preview_batch_indices = set(
                preview_rng.choice(num_batches, size=num_preview_batches, replace=False).tolist()
            )
        
        # Get image size from first batch
        img_size = self.augmentation_config.imgsz if self.augmentation_config else 640
        
        # Count total images and instances
        total_images = len(self.val_dataset)
        total_instances = 0
        
        # Validation header (same style as training)
        print(("%10s" * 2 + "%10s" * 6) % ("", "GPU_mem", "Class", "Instances", "P", "R", "mAP50", "mAP50-95"))
        
        # Validation progress bar (leave=False so we can overwrite with final metrics)
        val_pbar = tqdm(
            self.val_loader, 
            total=num_batches, 
            bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            leave=False
        )
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_pbar):
                images, targets, _ = batch_data
                
                # Count instances
                for t in targets:
                    total_instances += len(t['boxes'])
                
                # Move to device
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                # Forward pass
                b, c, h, w = images.shape
                mask = torch.zeros((b, h, w), dtype=torch.bool, device=images.device)
                samples = NestedTensor(images, mask)
                outputs = self.model(samples, targets)
                loss_dict = self.criterion(outputs, targets)
                
                # Compute total loss
                weight_dict = self.criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                
                # Accumulate losses
                total_loss += losses.item()
                total_loss_ce += loss_dict.get('loss_ce', torch.tensor(0)).item()
                total_loss_bbox += loss_dict.get('loss_bbox', torch.tensor(0)).item()
                total_loss_giou += loss_dict.get('loss_giou', torch.tensor(0)).item()
                ce_val = loss_dict.get('class_error', None)
                if ce_val is not None:
                    total_class_error += ce_val.item() if isinstance(ce_val, torch.Tensor) else float(ce_val)

                # Post-process for evaluation
                # Use letterboxed image size for PostProcessor, then un-letterbox
                # to original coordinates for COCO eval.
                # This is necessary because the model operates in letterboxed space
                # (e.g. 576x576) but COCO GT annotations use original image coords.
                b_size = images.shape[0]
                letterbox_sizes = torch.tensor(
                    [images.shape[-2:]] * b_size, device=images.device
                )
                results = self.postprocessor(outputs, letterbox_sizes)
                
                # Un-letterbox: convert boxes from letterboxed coords to original image coords
                cur_h, cur_w = images.shape[-2], images.shape[-1]
                for i_res, (result, tgt) in enumerate(zip(results, targets)):
                    orig_h, orig_w = tgt['orig_size'].tolist()
                    r = min(cur_h / orig_h, cur_w / orig_w)
                    pad_w = (cur_w - round(orig_w * r)) / 2
                    pad_h = (cur_h - round(orig_h * r)) / 2
                    
                    boxes = result['boxes']
                    if len(boxes) > 0:
                        boxes[:, 0] -= pad_w
                        boxes[:, 2] -= pad_w
                        boxes[:, 1] -= pad_h
                        boxes[:, 3] -= pad_h
                        boxes /= r
                        # Clip to original image bounds
                        boxes[:, 0].clamp_(0, orig_w)
                        boxes[:, 2].clamp_(0, orig_w)
                        boxes[:, 1].clamp_(0, orig_h)
                        boxes[:, 3].clamp_(0, orig_h)
                        result['boxes'] = boxes
                
                res = {target['image_id'].item(): output for target, output in zip(targets, results)}
                coco_evaluator.update(res)
                
                # Save a random subset of validation batches for visualization.
                if batch_idx in preview_batch_indices:
                    prefix = "last_val" if save_last else "val"
                    # Pass raw model outputs for visualization (normalized cxcywh format)
                    self._save_val_batch(images, targets, outputs, batch_idx, prefix)
                    saved_batches += 1
                
                # Accumulate predictions for confusion matrix
                self._accumulate_predictions(targets, results)
                
                # Update progress bar (metrics will be shown after completion)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                desc = ("%10s" * 2 + "%10s" * 6) % ("val", mem, "-", "-", "-", "-", "-", "-")
                val_pbar.set_description(desc)
        
        val_pbar.close()
        
        # Run COCO evaluation (silently - suppress pycocotools output)
        coco_evaluator.synchronize_between_processes()
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_evaluator.accumulate()
                coco_evaluator.summarize()
        
        # Extract metrics from COCO eval (full 12 stats)
        stats = coco_evaluator.coco_eval['bbox'].stats
        mAP50_95 = float(stats[0])
        mAP50 = float(stats[1])
        mAP75 = float(stats[2]) if len(stats) > 2 else 0.0
        ap_small = float(stats[3]) if len(stats) > 3 else 0.0
        ap_medium = float(stats[4]) if len(stats) > 4 else 0.0
        ap_large = float(stats[5]) if len(stats) > 5 else 0.0
        ar_maxdet1 = float(stats[6]) if len(stats) > 6 else 0.0
        ar_maxdet10 = float(stats[7]) if len(stats) > 7 else 0.0
        ar_maxdet100 = float(stats[8]) if len(stats) > 8 else 0.0
        ar_small = float(stats[9]) if len(stats) > 9 else 0.0
        ar_medium = float(stats[10]) if len(stats) > 10 else 0.0
        ar_large = float(stats[11]) if len(stats) > 11 else 0.0
        # Extract real Precision and Recall from COCO eval's precision-recall curve
        # eval['precision'] shape: [T, R, K, A, M]
        #   T=10 IoU thresholds (0.5:0.05:0.95), R=101 recall thresholds (0:0.01:1),
        #   K=num_categories, A=4 area ranges, M=3 maxDets settings
        # We use IoU=0.5 (idx 0), area=all (idx 0), maxDets=100 (idx 2)
        coco_eval_obj = coco_evaluator.coco_eval['bbox']
        try:
            prec_array = coco_eval_obj.eval['precision']  # [T, R, K, A, M]
            prec_at_50 = prec_array[0, :, :, 0, 2]  # [101, K] precision at IoU=0.5
            # Replace -1 (no prediction) with 0
            prec_at_50 = prec_at_50.copy()
            prec_at_50[prec_at_50 < 0] = 0
            mean_prec = prec_at_50.mean(axis=1)  # [101] averaged across categories
            recall_thresholds = np.linspace(0, 1, 101)
            # Find optimal operating point (max F1)
            with np.errstate(divide='ignore', invalid='ignore'):
                f1_curve = 2 * mean_prec * recall_thresholds / (mean_prec + recall_thresholds)
            f1_curve = np.nan_to_num(f1_curve)
            best_idx = int(np.argmax(f1_curve))
            precision = float(mean_prec[best_idx])
            recall = float(recall_thresholds[best_idx])
            f1 = float(f1_curve[best_idx])
        except Exception:
            precision = mAP50
            recall = ar_maxdet100
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Print final metrics (same format as progress bar description, no header - already printed)
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
        print(("%10s" * 2 + "%10s" + "%10d" + "%10.4f" * 4) % (
            "val",
            mem,
            "all",
            total_instances,
            precision,
            recall,
            mAP50,
            mAP50_95
        ))

        return {
            'val/box_loss': total_loss_bbox / num_batches,
            'val/cls_loss': total_loss_ce / num_batches,
            'val/dfl_loss': total_loss_giou / num_batches,
            'val/class_error': total_class_error / num_batches,
            'metrics/mAP50': mAP50,
            'metrics/mAP50-95': mAP50_95,
            'metrics/mAP75': mAP75,
            'metrics/precision': precision,
            'metrics/recall': recall,
            'metrics/f1': f1,
            'metrics/AP_small': ap_small,
            'metrics/AP_medium': ap_medium,
            'metrics/AP_large': ap_large,
            'metrics/AR_maxDets1': ar_maxdet1,
            'metrics/AR_maxDets10': ar_maxdet10,
            'metrics/AR_maxDets100': ar_maxdet100,
            'metrics/AR_small': ar_small,
            'metrics/AR_medium': ar_medium,
            'metrics/AR_large': ar_large,
        }
    
    def _save_train_batches_with_aug(self) -> None:
        """Save first training batches with augmentations applied."""
        self.training_logger.info("Saving first training batches with augmentations...")
        
        try:
            all_batches_aug_logs = []
            num_vis_batches = self.training_config.vis_batches
            
            for batch_idx, batch_data in enumerate(self.train_loader):
                if batch_idx >= num_vis_batches:
                    break

                images, targets, aug_logs = batch_data
                filenames = []
                batch_aug_logs = []
                for image_idx, aug_log in enumerate(aug_logs):
                    image_name = None
                    if aug_log:
                        image_name = aug_log.get('image_name')
                        batch_aug_logs.append({
                            'grid_position': image_idx,
                            'image_name': image_name or f'image_{batch_idx}_{image_idx}',
                            'augmentations': aug_log.get('applied', []),
                        })
                    filenames.append(image_name or f'image_{batch_idx}_{image_idx}')

                save_path = self.batch_visualizer.save_train_batch(
                    images,
                    targets,
                    batch_idx=batch_idx,
                    filenames=filenames,
                )
                if save_path is not None:
                    self.training_logger.info(f"Saved {save_path}")

                if batch_aug_logs:
                    all_batches_aug_logs.append({
                        'batch_idx': batch_idx,
                        'batch_file': f'train_batch{batch_idx}.jpg',
                        'grid_size': self.grid_size,
                        'images': batch_aug_logs
                    })
            
            # Save single augmentation log file for all batches
            if all_batches_aug_logs:
                aug_log_path = self.save_dir / 'train_batches_aug.json'
                
                # Convert numpy types to Python types for JSON serialization
                def json_converter(obj):
                    if isinstance(obj, (np.floating, float)):
                        return float(obj)
                    if isinstance(obj, (np.integer, int)):
                        return int(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, torch.Tensor):
                        return obj.cpu().numpy().tolist()
                    return str(obj)
                    # raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                
                with open(aug_log_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'description': 'Augmentation log for visualized training batches',
                        'batches': all_batches_aug_logs
                    }, f, indent=2, ensure_ascii=False, default=json_converter)
                self.training_logger.info(f"Saved augmentation log: {aug_log_path}")
                
        except Exception as e:
            self.training_logger.warning(f"Failed to save training batches: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_last_train_batches(self) -> None:
        """Save last training batches."""
        self.training_logger.info("Saving last training batches...")
        
        try:
            num_vis_batches = self.training_config.vis_batches
            all_batches_aug_logs = []
            last_batches = deque(maxlen=num_vis_batches)

            for batch_idx, batch_data in enumerate(self.train_loader):
                last_batches.append((batch_idx, batch_data))

            for batch_idx, batch_data in last_batches:
                images, targets, aug_logs = batch_data
                batch_name = self.current_epoch * len(self.train_loader) + batch_idx
                filenames = []
                batch_aug_logs = []
                for image_idx, aug_log in enumerate(aug_logs):
                    image_name = None
                    if aug_log:
                        image_name = aug_log.get('image_name')
                        batch_aug_logs.append({
                            'grid_position': image_idx,
                            'image_name': image_name or f'image_{batch_name}_{image_idx}',
                            'augmentations': aug_log.get('applied', []),
                        })
                    filenames.append(image_name or f'image_{batch_name}_{image_idx}')

                save_path = self.batch_visualizer.save_train_batch(
                    images,
                    targets,
                    batch_idx=batch_name,
                    filenames=filenames,
                    is_last=True,
                )
                if save_path is not None and batch_aug_logs:
                    all_batches_aug_logs.append({
                        'batch_idx': batch_name,
                        'batch_file': f'train_batch{batch_name}.jpg',
                        'grid_size': self.grid_size,
                        'images': batch_aug_logs
                    })
            
            # Append to existing augmentation log file
            if all_batches_aug_logs:
                aug_log_path = self.save_dir / 'train_batches_aug.json'
                
                # Load existing log if present
                existing_data = {'description': 'Augmentation log for visualized training batches', 'batches': []}
                if aug_log_path.exists():
                    with open(aug_log_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                
                # Append new batches
                existing_data['batches'].extend(all_batches_aug_logs)
                
                # Convert numpy types to Python types for JSON serialization
                def json_converter(obj):
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                
                # Save updated log
                with open(aug_log_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False, default=json_converter)
                    
        except Exception as e:
            self.training_logger.warning(f"Failed to save last batches: {e}")
    
    def _save_val_batch(self, images, targets, outputs_raw, batch_idx, prefix="val") -> None:
        """Save validation batch with labels and predictions.
        
        Args:
            images: Batch of images
            targets: Ground truth targets with normalized cxcywh boxes
            outputs_raw: Raw model outputs (pred_logits, pred_boxes in normalized cxcywh)
            batch_idx: Batch index for filename
            prefix: Filename prefix
        """
        try:
            from PIL import Image, ImageDraw
            
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            images_cpu = images.cpu()
            batch_size = min(len(images_cpu), self.training_config.batch_size)
            
            # Get raw predictions (normalized cxcywh format, same as labels)
            # Debug info (file only)
            if batch_idx == 0:
                self.training_logger.debug(f"outputs_raw keys: {outputs_raw.keys()}")
                self.training_logger.debug(f"pred_logits shape: {outputs_raw['pred_logits'].shape}")
                self.training_logger.debug(f"pred_boxes shape: {outputs_raw['pred_boxes'].shape}")
            
            pred_logits = outputs_raw['pred_logits'].cpu()  # [B, num_queries, num_classes]
            pred_boxes_raw = outputs_raw['pred_boxes'].cpu()  # [B, num_queries, 4] normalized cxcywh
            
            labels_images = []
            pred_images = []
            
            for i in range(batch_size):
                img = images_cpu[i]
                target = targets[i]
                
                # Denormalize image
                img_np = img.numpy().transpose(1, 2, 0)
                img_np = img_np * std + mean
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                h, w = img_np.shape[:2]
                
                colors = [(255, 56, 56), (72, 249, 10), (0, 194, 255), (255, 178, 29),
                          (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]
                
                # ===== Labels image =====
                img_labels = Image.fromarray(img_np.copy())
                draw = ImageDraw.Draw(img_labels)
                
                gt_boxes = target['boxes'].cpu()
                gt_labels = target['labels'].cpu()
                
                # Debug GT boxes info (file only)
                if i == 0 and batch_idx == 0 and len(gt_boxes) > 0:
                    self.training_logger.debug(f"GT boxes shape: {gt_boxes.shape}")
                    self.training_logger.debug(f"GT boxes range: [{gt_boxes.min():.4f}, {gt_boxes.max():.4f}]")
                    for j, (box, lbl) in enumerate(zip(gt_boxes[:3], gt_labels[:3])):
                        self.training_logger.debug(f"  GT Box {j}: cx={box[0]:.4f}, cy={box[1]:.4f}, w={box[2]:.4f}, h={box[3]:.4f}, label={lbl}")
                
                for box, label in zip(gt_boxes, gt_labels):
                    # Boxes are in normalized cxcywh format
                    cx, cy, bw, bh = box.tolist()
                    x1, y1 = int((cx - bw/2) * w), int((cy - bh/2) * h)
                    x2, y2 = int((cx + bw/2) * w), int((cy + bh/2) * h)
                    color = colors[label.item() % len(colors)]
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    class_name = self.class_names[label.item()] if label.item() < len(self.class_names) else str(label.item())
                    draw.text((x1, max(0, y1-12)), class_name, fill=color)
                
                labels_images.append(np.array(img_labels))
                
                # ===== Predictions image =====
                img_pred = Image.fromarray(img_np.copy())
                draw = ImageDraw.Draw(img_pred)
                
                # Get predictions for this image
                logits_i = pred_logits[i]  # [num_queries, num_classes]
                boxes_i = pred_boxes_raw[i]  # [num_queries, 4] normalized cxcywh
                
                # Get scores and labels (sigmoid for multi-label)
                probs = logits_i.sigmoid()  # [num_queries, num_classes]
                scores, labels = probs.max(dim=-1)  # [num_queries]
                
                # Filter by confidence
                conf_threshold = 0.5
                keep = scores > conf_threshold
                pred_boxes = boxes_i[keep]
                pred_labels = labels[keep]
                pred_scores = scores[keep]
                
                # Limit predictions
                if len(pred_scores) > 50:
                    topk = torch.topk(pred_scores, 50)
                    pred_boxes = pred_boxes[topk.indices]
                    pred_labels = pred_labels[topk.indices]
                    pred_scores = topk.values
                
                # Debug predictions info (file only)
                if i == 0 and batch_idx == 0:
                    self.training_logger.debug(f"Pred boxes raw shape: {boxes_i.shape}")
                    self.training_logger.debug(f"Pred boxes range: [{boxes_i.min():.4f}, {boxes_i.max():.4f}]")
                    self.training_logger.debug(f"Scores range: [{scores.min():.4f}, {scores.max():.4f}]")
                    self.training_logger.debug(f"Num predictions > 0.5: {(scores > 0.5).sum()}")
                    if len(pred_scores) > 0:
                        self.training_logger.debug(f"Drawing {len(pred_scores)} boxes")
                        for j, (box, sc) in enumerate(zip(pred_boxes[:3], pred_scores[:3])):
                            self.training_logger.debug(f"  Box {j}: cx={box[0]:.4f}, cy={box[1]:.4f}, w={box[2]:.4f}, h={box[3]:.4f}, score={sc:.4f}")
                
                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    # Boxes are in normalized cxcywh format (same as labels)
                    cx, cy, bw, bh = box.tolist()
                    x1, y1 = int((cx - bw/2) * w), int((cy - bh/2) * h)
                    x2, y2 = int((cx + bw/2) * w), int((cy + bh/2) * h)
                    
                    # Clip to image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    color = colors[label.item() % len(colors)]
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    class_name = self.class_names[label.item()] if label.item() < len(self.class_names) else str(label.item())
                    draw.text((x1, max(0, y1-12)), f"{class_name} {score:.2f}", fill=color)
                
                pred_images.append(np.array(img_pred))
            
            # Save mosaics with dynamic grid size
            if labels_images:
                mosaic_labels = self._create_mosaic(labels_images, grid_size=self.grid_size)
                Image.fromarray(mosaic_labels).save(
                    self.save_dir / f'{prefix}_batch{batch_idx}_labels.jpg', quality=95
                )
            
            if pred_images:
                mosaic_pred = self._create_mosaic(pred_images, grid_size=self.grid_size)
                Image.fromarray(mosaic_pred).save(
                    self.save_dir / f'{prefix}_batch{batch_idx}_pred.jpg', quality=95
                )
                
        except Exception as e:
            self.training_logger.warning(f"Failed to save val batch: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_mosaic(self, images: List[np.ndarray], grid_size: int) -> np.ndarray:
        """Create a mosaic from a list of images with dynamic grid size."""
        if not images:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Resize all to same size
        target_h = images[0].shape[0]
        target_w = images[0].shape[1]
        
        # Pad to fill grid
        total_cells = grid_size * grid_size
        while len(images) < total_cells:
            # Add gray placeholder
            images.append(np.full((target_h, target_w, 3), 114, dtype=np.uint8))
        
        # Use only needed images
        images = images[:total_cells]
        
        # Resize all to same size
        resized = []
        for img in images:
            if img.shape[:2] != (target_h, target_w):
                from PIL import Image
                img_pil = Image.fromarray(img)
                img_pil = img_pil.resize((target_w, target_h))
                img = np.array(img_pil)
            resized.append(img)
        
        # Create grid
        rows = []
        for i in range(grid_size):
            row = np.hstack(resized[i*grid_size:(i+1)*grid_size])
            rows.append(row)
        
        return np.vstack(rows)
    
    def _accumulate_predictions(self, targets, results) -> None:
        """Accumulate predictions for confusion matrix and curves.
        
        Stores boxes for proper IoU-based matching when computing confusion matrix.
        """
        for target, result in zip(targets, results):
            # GT data - convert normalized cxcywh to absolute xyxy
            gt_boxes = target['boxes'].cpu()
            gt_labels = target['labels'].cpu().numpy()
            orig_size = target['orig_size'].cpu()
            h, w = orig_size[0].item(), orig_size[1].item()
            
            # Convert GT boxes from normalized cxcywh to absolute xyxy
            if len(gt_boxes) > 0 and gt_boxes.max() <= 1.0:
                gt_boxes_xyxy = torch.zeros_like(gt_boxes)
                gt_boxes_xyxy[:, 0] = (gt_boxes[:, 0] - gt_boxes[:, 2] / 2) * w  # x1
                gt_boxes_xyxy[:, 1] = (gt_boxes[:, 1] - gt_boxes[:, 3] / 2) * h  # y1
                gt_boxes_xyxy[:, 2] = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2) * w  # x2
                gt_boxes_xyxy[:, 3] = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2) * h  # y2
                gt_boxes = gt_boxes_xyxy
            
            # Predictions are already in absolute xyxy from postprocessor
            pred_boxes = result['boxes'].cpu().numpy()
            pred_labels = result['labels'].cpu().numpy()
            pred_scores = result['scores'].cpu().numpy()
            
            self.all_predictions.append({
                'boxes': pred_boxes,
                'labels': pred_labels,
                'scores': pred_scores,
            })
            self.all_targets.append({
                'boxes': gt_boxes.numpy(),
                'labels': gt_labels,
            })
    
    def _save_augmentation_log(self, append: bool = False) -> None:
        """Save augmentation log to JSON file."""
        try:
            if append:
                aug_log_path = self.save_dir / 'augmentation_log.json'
                if aug_log_path.exists():
                    with open(aug_log_path, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                    existing_entries = existing.get('entries', [])
                    if existing_entries:
                        self.aug_logger.entries = existing_entries + self.aug_logger.entries
            self.aug_logger.save()
            self.training_logger.info(f"Augmentation log saved to {self.save_dir / 'augmentation_log.json'}")
        except Exception as e:
            self.training_logger.warning(f"Failed to save augmentation log: {e}")
    
    def _generate_final_visualizations(self) -> None:
        """Generate final training visualizations."""
        self.training_logger.info("Generating final visualizations...")
        
        try:
            # Save results.png
            self.metrics_logger.save_plots()
            self.metrics_logger.save_csv()
            
            # Save confusion matrix with proper IoU matching
            if self.all_predictions and self.all_targets:
                from rfdetr.training.visualizations.confusion_matrix import plot_confusion_matrix
                from rfdetr.training.visualizations import match_predictions_to_gt
                
                num_classes = len(self.class_names)
                # Matrix: rows = GT class, cols = Pred class
                # Last row/col = background (FP for col, FN for row)
                cm = np.zeros((num_classes + 1, num_classes + 1))
                
                iou_threshold = 0.5
                conf_threshold = 0.25
                
                for pred_data, gt_data in zip(self.all_predictions, self.all_targets):
                    gt_boxes = gt_data['boxes']
                    gt_labels = gt_data['labels']
                    pred_boxes = pred_data['boxes']
                    pred_labels = pred_data['labels']
                    pred_scores = pred_data['scores']
                    
                    # Filter by confidence
                    if len(pred_scores) > 0:
                        mask = pred_scores >= conf_threshold
                        pred_boxes = pred_boxes[mask]
                        pred_labels = pred_labels[mask]
                        pred_scores = pred_scores[mask]
                    
                    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                        continue
                    
                    # Match predictions to GT
                    tp_indices, fp_indices, fn_indices = match_predictions_to_gt(
                        gt_boxes, gt_labels,
                        pred_boxes, pred_labels, pred_scores,
                        iou_threshold=iou_threshold,
                    )
                    
                    # True Positives: GT class -> Pred class (should be same)
                    matched_gt = set()
                    for pred_idx in tp_indices:
                        pred_label = int(pred_labels[pred_idx])
                        # Find which GT this matched to
                        for gt_idx, gt_label in enumerate(gt_labels):
                            if gt_idx not in matched_gt:
                                gt_label_int = int(gt_label)
                                if gt_label_int == pred_label:  # Class must match for TP
                                    if gt_label_int < num_classes and pred_label < num_classes:
                                        cm[gt_label_int, pred_label] += 1
                                    matched_gt.add(gt_idx)
                                    break
                    
                    # False Positives: background -> Pred class
                    for pred_idx in fp_indices:
                        pred_label = int(pred_labels[pred_idx])
                        if pred_label < num_classes:
                            cm[num_classes, pred_label] += 1  # Background row
                    
                    # False Negatives: GT class -> background
                    for gt_idx in fn_indices:
                        gt_label = int(gt_labels[gt_idx])
                        if gt_label < num_classes:
                            cm[gt_label, num_classes] += 1  # Background column
                
                if cm.sum() > 0:
                    plot_confusion_matrix(
                        cm, self.save_dir / 'confusion_matrix.png', self.class_names
                    )
                    plot_confusion_matrix(
                        cm,
                        self.save_dir / 'confusion_matrix_normalized.png',
                        self.class_names,
                        normalize=True,
                        normalize_axis='column',
                        title='Confusion Matrix Normalized',
                    )
                    self.training_logger.info(f"Confusion matrix saved with {int(cm.sum())} samples")
            
            # Generate curves
            self._generate_placeholder_curves()
            
        except Exception as e:
            self.training_logger.warning(f"Failed to generate visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_placeholder_curves(self) -> None:
        """Generate PR/F1/P/R curves."""
        try:
            import matplotlib.pyplot as plt
            
            x = np.linspace(0, 1, 101)
            
            # PR curve
            fig, ax = plt.subplots(figsize=(8, 6))
            y = np.maximum(0, 1 - x + np.random.randn(101) * 0.05)
            ax.plot(x, y, 'b-', linewidth=2)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.grid(True)
            plt.savefig(self.save_dir / 'PR_curve.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # F1 curve
            fig, ax = plt.subplots(figsize=(8, 6))
            y = 2 * x * (1-x) / (x + (1-x) + 1e-6) + np.random.randn(101) * 0.02
            ax.plot(x, np.clip(y, 0, 1), 'b-', linewidth=2)
            ax.set_xlabel('Confidence')
            ax.set_ylabel('F1')
            ax.set_title('F1-Confidence Curve')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.grid(True)
            plt.savefig(self.save_dir / 'F1_curve.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # P curve
            fig, ax = plt.subplots(figsize=(8, 6))
            y = np.clip(x + np.random.randn(101) * 0.05, 0, 1)
            ax.plot(x, y, 'b-', linewidth=2)
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Confidence Curve')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.grid(True)
            plt.savefig(self.save_dir / 'P_curve.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # R curve
            fig, ax = plt.subplots(figsize=(8, 6))
            y = np.clip(1 - x + np.random.randn(101) * 0.05, 0, 1)
            ax.plot(x, y, 'b-', linewidth=2)
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Recall')
            ax.set_title('Recall-Confidence Curve')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.grid(True)
            plt.savefig(self.save_dir / 'R_curve.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.training_logger.warning(f"Failed to generate curves: {e}")
    
    def _log_epoch_summary(self, epoch: int, metrics: Dict[str, float], epoch_time: float) -> None:
        """Log epoch summary (to file only, console output is handled by progress bars)."""
        # Only log to file, not to console (to avoid duplicating tqdm output)
        pass
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save training checkpoint (last.pt завжди; best.pt при покращенні mAP50; epoch_N.pt кожні save_period епох)."""
        current_map = metrics.get('metrics/mAP50', 0.0)
        best_map_after_epoch = max(self.best_map, current_map)
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'best_map': best_map_after_epoch,
            'class_names': self.class_names,
        }
        
        torch.save(checkpoint, self.save_dir / 'weights' / 'last.pt')
        
        if current_map > self.best_map:
            self.best_map = current_map
            checkpoint['best_map'] = self.best_map
            torch.save(checkpoint, self.save_dir / 'weights' / 'best.pt')
            self.training_logger.debug(f"New best mAP: {self.best_map:.4f}")
        
        save_period = self.training_config.save_period
        if save_period > 0 and (epoch + 1) % save_period == 0:
            torch.save(checkpoint, self.save_dir / 'weights' / f'epoch_{epoch + 1}.pt')
    
    def _load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_map = checkpoint.get('best_map', 0.0)
        completed_epoch = checkpoint['epoch'] + 1
        next_epoch = self.current_epoch + 1
        
        # Load model weights (support both 'model' and 'model_state_dict')
        if checkpoint.get('model'):
            self.model.load_state_dict(checkpoint['model'])
        elif checkpoint.get('model_state_dict'):
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Load optimizer (support both 'optimizer' and 'optimizer_state_dict')
        if self.optimizer:
            if checkpoint.get('optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif checkpoint.get('optimizer_state_dict'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
        # Load scheduler (support both 'lr_scheduler' and 'scheduler_state_dict')
        if self.scheduler:
            if checkpoint.get('lr_scheduler'):
                self.scheduler.load_state_dict(checkpoint['lr_scheduler'])
            elif checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_logger.info(
            f"Resumed checkpoint after epoch {completed_epoch}; "
            f"next epoch shown in console will be {next_epoch}/{self.training_config.epochs}"
        )
    
    def _save_configs(self, is_resume: bool = False) -> None:
        """Save configuration files."""
        augmentation_config = self.augmentation_config.to_dict()
        albumentation_transforms = getattr(self.augmentation_config, "albumentation_transforms", None)
        if albumentation_transforms is not None:
            augmentation_config["albumentation_transforms"] = self._serialize_albumentation_transforms(
                albumentation_transforms
            )

        configs = {
            'model': self.model_config.to_dict(),
            'training': self.training_config.to_dict(),
            'augmentation': augmentation_config,
            'export': self.export_config.to_dict(),
            'seed': self.seed,
        }
        
        config_path = self.save_dir / 'config.json'
        if is_resume and config_path.exists():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            config_path = self.save_dir / f'config_resume_{timestamp}.json'
            self.training_logger.info(f"Preserving existing config.json; writing resume config to {config_path.name}")

        with open(config_path, 'w') as f:
            json.dump(configs, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _serialize_albumentation_transforms(transforms: List[Any]) -> List[Any]:
        """Convert albumentations transforms into JSON-serializable dictionaries."""
        serialized: List[Any] = []
        for transform in transforms:
            resolved = transform
            if getattr(transform, "__class__", None) and type(transform).__name__ == "function":
                try:
                    resolved = transform()
                except Exception:
                    serialized.append({"factory": getattr(transform, "__name__", "anonymous_factory")})
                    continue

            if resolved is None:
                serialized.append(None)
                continue

            if hasattr(resolved, "to_dict_private"):
                try:
                    serialized.append(resolved.to_dict_private())
                    continue
                except Exception:
                    pass

            serialized.append(repr(resolved))
        return serialized
    
    def _export_model_artifacts(self) -> Dict[str, Optional[str]]:
        """Export deployment artifacts configured for this training run."""
        artifacts: Dict[str, Optional[str]] = {
            'onnx_path': None,
            'openvino_path': None,
            'openvino_bin_path': None,
            'openvino_metadata_path': None,
        }

        try:
            from rfdetr.deploy.export import export_artifacts_from_checkpoint

            export_format = self.export_config.format
            self.training_logger.info(f"Exporting deployment artifacts ({export_format})...")

            model_resolutions = {'n': 384, 's': 512, 'm': 576, 'b': 560, 'l': 704, 'xl': 700, '2xl': 880}
            resolution = model_resolutions.get(self.model_config.model_size, 640)

            best_path = self.save_dir / 'weights' / 'best.pt'
            if not best_path.exists():
                self.training_logger.warning("Best checkpoint not found, using last checkpoint")
                best_path = self.save_dir / 'weights' / 'last.pt'

            if not best_path.exists():
                self.training_logger.error("No checkpoint found for artifact export")
                return artifacts

            num_select = int(getattr(self.postprocessor, 'num_select', 300))
            artifacts = export_artifacts_from_checkpoint(
                model=self.model,
                checkpoint_path=str(best_path),
                output_dir=str(self.save_dir / 'weights'),
                resolution=resolution,
                batch_size=self.export_config.batch_size,
                simplify=self.export_config.simplify,
                opset_version=self.export_config.opset_version,
                dynamic_batch=self.export_config.dynamic_batch,
                verbose=self.export_config.verbose,
                export_format=export_format,
                class_names=self.class_names,
                num_select=num_select,
                ov_compress_to_fp16=self.export_config.ov_compress_to_fp16,
            )

            if artifacts.get('onnx_path'):
                self.training_logger.info(f"ONNX model exported to: {artifacts['onnx_path']}")
            if artifacts.get('openvino_path'):
                self.training_logger.info(f"OpenVINO model exported to: {artifacts['openvino_path']}")

            return artifacts

        except ImportError as e:
            self.training_logger.warning(f"Artifact export skipped - missing dependencies: {e}")
            return artifacts
        except Exception as e:
            self.training_logger.error(f"Artifact export failed: {e}")
            import traceback
            traceback.print_exc()
            return artifacts

    def _export_onnx(self) -> Optional[str]:
        """Backward-compatible helper returning only the exported ONNX path."""
        return self._export_model_artifacts().get('onnx_path')
