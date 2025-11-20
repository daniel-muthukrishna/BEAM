"""
Weights & Biases callback for BEAM.

This module provides a callback for logging metrics, images,
and other artifacts to Weights & Biases during training.
"""

import os
from typing import Dict, Any
import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

from beam.training.callbacks import Callback
from beam.utils.wandb_utils import (
    init_wandb, log_metrics, log_sample_images,
    log_figure, log_model, finish_run
)
from beam.utils.visualization import plot_loss_history


class WeightsAndBiasesCallback(Callback):
    """
    Callback for logging to Weights & Biases.
    
    Args:
        project_name: W&B project name
        model_name: Name of the model for W&B
        log_freq: Frequency (in batches) for logging batch metrics
        log_samples_freq: Frequency (in epochs) for logging sample images
        log_model_freq: Frequency (in epochs) for logging model weights
        use_wandb: Whether to use W&B (set to False to disable)
    """
    
    def __init__(
        self, 
        project_name: str = "beam-tess",
        model_name: str = None,
        log_freq: int = 10,
        log_samples_freq: int = 5, 
        log_model_freq: int = 20,
        use_wandb: bool = True,
        mode: str = "online",
        mean: float = 0.01978608595999928,
        std: float = 0.08876927679677006,
    ):
        super().__init__()
        self.project_name = project_name
        self.model_name = model_name
        self.log_freq = log_freq
        self.log_samples_freq = log_samples_freq
        self.log_model_freq = log_model_freq
        self.use_wandb = use_wandb
        self.mode = mode if use_wandb else "disabled"
        self.batch_count = 0
        self.current_step = 0  # Track the current step for consistent logging
        
    def on_train_begin(self, trainer: Any) -> None:
        """Initialize W&B at the start of training."""
        if not self.use_wandb:
            return
            
        # Generate a model name if not provided
        if self.model_name is None:
            # Convert timestamp to formatted time string
            time_str = datetime.datetime.fromtimestamp(trainer.start_training_time).strftime("%Y%m%d_%H%M%S")
            model_name = config.get('wandb_model_name', 'beam-tess-model')
            self.model_name = f"{model_name}_{time_str}"
        
        # Add run name to config
        config = trainer.config.copy()
        config["run_name"] = self.model_name
        
        # Initialize W&B
        init_wandb(config, self.project_name, self.mode)

        print(f"Initialized W&B logging with run name: {self.model_name}")

        # Calculate steps per epoch correctly for distributed training
        self.steps_per_epoch = len(trainer.train_dataset) // trainer.config['training_batch_size']
        
        # Set initial step based on current epoch (in case of resuming training)
        if trainer.current_epoch > 0:
            # Estimate the step based on the epoch * batch size
            self.current_step = trainer.current_epoch * self.steps_per_epoch
        
    def on_train_end(self, trainer: Any) -> None:
        """Finish W&B run at the end of training."""
        if not self.use_wandb:
            return
            
        # Get base model (remove DistributedDataParallel wrapper if needed)
        model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        
        # Log final model
        try:
            log_model(model, name="final_model")
        except Exception as e:
            print(f"Warning: Failed to log final model to W&B: {e}")
        
        # Finish the run
        finish_run()
        
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Reset batch counter at the beginning of each epoch."""
        self.batch_count = 0
        
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict) -> None:
        """Log metrics and samples at the end of an epoch."""
        if not self.use_wandb:
            return
        
        # Update current step
        self.current_step = max(self.current_step, epoch * self.steps_per_epoch + self.batch_count)
            
        # Log epoch metrics
        try:
            metrics = {
                "epoch": epoch,
                "train_loss": trainer.loss_history_train[-1],
                "valid_loss": trainer.loss_history_valid[-1],
                "time_hours": trainer.time_history[-1] if trainer.time_history else 0,
            }
            
            # Add learning rate
            if hasattr(trainer, 'optimizer'):
                metrics["learning_rate"] = trainer.optimizer.param_groups[0]['lr']
                
            log_metrics(metrics, step=self.current_step)
        except Exception as e:
            print(f"Warning: Failed to log epoch metrics to W&B: {e}")
        
        # Log samples periodically
        if epoch % self.log_samples_freq == 0 or epoch == trainer.config['training_n_epoch'] - 1:
            try:
                # Get base model (remove DistributedDataParallel wrapper if needed)
                model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
                
                # Log training samples
                log_sample_images(
                    model=model,
                    dataloader=trainer.train_loader,
                    device=trainer.device,
                    n_sample=1,
                    n_datapoint=1,
                    image_shape=trainer.config['data_image_shape'] if trainer.config['data_patch_size'] is None else trainer.config.get('data_patch_size'),
                    step=self.current_step,
                    name="Training Set",
                    ema=trainer.ema,
                    mean=self.mean,
                    std=self.std
                )
                
                # Log validation samples
                log_sample_images(
                    model=model,
                    dataloader=trainer.valid_loader,
                    device=trainer.device,
                    n_sample=1,
                    n_datapoint=1,
                    image_shape=trainer.config['data_image_shape'] if trainer.config['data_patch_size'] is None else trainer.config.get('data_patch_size'),
                    step=self.current_step,
                    name="Validation Set",
                    ema=trainer.ema,
                    mean=self.mean,
                    std=self.std
                )
            except Exception as e:
                print(f"Warning: Failed to log sample images to W&B: {e}")
        
        # Log model weights periodically
        if epoch % self.log_model_freq == 0 or epoch == trainer.config['training_n_epoch'] - 1:
            try:
                # Get base model (remove DistributedDataParallel wrapper if needed)
                model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
                
                # Log model weights
                log_model(model, name=f"model_epoch_{epoch}", ema=trainer.ema)
                
            except Exception as e:
                print(f"Warning: Failed to log model weights to W&B: {e}")
        
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict) -> None:
        """Log batch metrics periodically."""
        # Increment batch counter even if not logging
        self.batch_count += 1
        
        # Update current step
        self.current_step = max(self.current_step, trainer.current_epoch * self.steps_per_epoch + self.batch_count)
        
        if not self.use_wandb or batch % self.log_freq != 0:
            return
        
        # Only log every log_freq batches and avoid overwhelming the server
        try:
            # Log batch metrics
            metrics = {
                "batch": self.batch_count,
                "batch_loss": logs.get("batch_loss", 0),
            }
            
            # Use a try-except block to handle connection issues
            log_metrics(metrics, step=self.current_step)
        except Exception as e:
            print(f"Warning: Failed to log batch metrics to W&B: {e}")
            # Continue training even if logging fails
            pass