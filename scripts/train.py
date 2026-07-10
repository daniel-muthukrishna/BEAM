#!/usr/bin/env python
"""
BEAM: Background Elimination with Advanced Machine learning - Training Script

This script trains a conditional diffusion model on TESS (Transiting Exoplanet
Survey Satellite) Full Frame Images.
"""

import os
import argparse
import datetime
import torch
import torch.multiprocessing as mp

from beam.data.datasets import TESSDataset, StarDataset, create_train_valid_datasets_by_orbit
from beam.models.scores import VESDE, SBD
from beam.training.trainer import SMTrainer
from beam.training.distributed import run_distributed
from beam.utils.config import load_config, prepare_training_config
from beam.training.wandb_callback import WeightsAndBiasesCallback
from beam.training.callbacks import ModelCheckpoint, EarlyStopping


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a diffusion model on TESS images")

    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--wandb', type=str, default='online',
                        help='W&B logging mode: online, offline, or disabled')

    return parser.parse_args()


def prepare_dataset(config):
    """
    Prepare the TESS dataset for training and validation.

    Args:
        config: Training configuration

    Returns:
        Tuple of (training dataset, validation dataset)
    """
    # Create dataset
    print(f"Creating TESSDataset...")
    tess_dataset = TESSDataset(
        angle_path=config['data_angle_path'],
        ccd_folder=config['data_ccd_folder'],
        image_shape=config['data_image_shape'],
        patch_size=config.get('data_patch_size', None),
        repeat_factor=config.get('data_repeat_factor', 1),
        mean=config.get('data_mean', 0),
        std=config.get('data_std', 1),
        camera_number=config.get('data_camera_number', '3'),
    )
    # Create training and validation splits based on orbit number (fast, metadata only)
    print("Creating training and validation splits by orbit...")
    train_dataset, valid_dataset = create_train_valid_datasets_by_orbit(
        tess_dataset,
        config["data_orbit_threshold"],
        config["data_orbit_max"],
    )

    return train_dataset, valid_dataset


def create_callbacks(config, args, rank):
    """
    Create training callbacks.

    Args:
        config: Training configuration
        args: Command line arguments
        rank: GPU rank

    Returns:
        List of callbacks
    """
    callbacks = []

    # Only add callbacks on the main process (rank 0)
    if rank == 0:
        # Add ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(config['paths_save_dir'], 'checkpoint_epoch_{epoch}_{datetime}.pth'),
            monitor='valid_loss',
            save_best_only=config.get('checkpoint_save_best_only', True),
            save_weights_only=config.get('checkpoint_save_weights_only', False),
            mode='min',
            period=config.get('checkpoint_epoch_checkpoint', 100)
        )
        callbacks.append(checkpoint_callback)

        # Add EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='valid_loss',
            patience=config['training_patience'],
            min_delta=0.0001,
            mode='min'
        )
        callbacks.append(early_stopping)
        # Add Weights & Biases callback if enabled
        if args.wandb != 'disabled':
            # Get model name from config and append timestamp to create a unique run name
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = config.get('wandb_model_name', 'beam-tess-model')
            run_name = f"{model_name}_{timestamp}"
            wandb_callback = WeightsAndBiasesCallback(
                project_name=config.get('wandb_project', 'beam-tess'),
                model_name=run_name,
                log_freq=config.get('wandb_log_freq', 10),
                log_samples_freq=config.get('wandb_log_samples_freq', 5),
                log_model_freq=config.get('wandb_log_model_freq', 20),
                use_wandb=True,
                mode=args.wandb,
                MEAN=config.get('data_mean', 0),
                STD=config.get('data_std', 1)
            )
            callbacks.append(wandb_callback)

    return callbacks


def train_worker(rank, world_size, config, train_dataset, valid_dataset, args, resume_path=None, device=None):
    """
    Worker function for training on a single GPU.

    Args:
        rank: GPU rank
        world_size: Total number of GPUs
        config: Training configuration
        train_dataset: Training dataset (pre-created in main process)
        valid_dataset: Validation dataset (pre-created in main process)
        args: Command line arguments
        resume_path: Path to checkpoint to resume from
        device: Device to train on
    """
    # Set device if not provided
    if device is None:
        device = torch.device(f'cuda:{rank}')

    print(f"GPU {rank}: Starting worker with {len(train_dataset) // world_size} training samples "
          f"and {len(valid_dataset) // world_size} validation samples on device {device}")

    # Create callbacks
    callbacks = create_callbacks(config, args, rank)

    # Create trainer
    trainer = SMTrainer(
        config=config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        rank=rank,
        world_size=world_size,
        device=device,
        callbacks=callbacks  # Pass callbacks to trainer
    )

    # Resume from checkpoint if specified
    if resume_path:
        trainer.load_checkpoint(resume_path)
        print(f"GPU {rank}: Resumed training from {resume_path}")

    # Train the model
    trainer.train(
        checkpoint_freq=config['checkpoint_epoch_checkpoint'],
        save_model=config['checkpoint_save_model']
    )

    print(f"GPU {rank}: Training complete")


def main():
    """Main function to set up and launch distributed training."""
    print("Beginning Training")
    # Parse arguments
    args = parse_args()

    # Load and process configuration
    config = load_config(args.config)
    flat_config = prepare_training_config(config)

    # Get world size
    world_size = flat_config['distributed_world_size']
    print(f"Using {world_size} GPUs for training")

    # Prepare dataset once in the main process before spawning workers
    print("Preparing dataset in main process...")
    train_dataset, valid_dataset = prepare_dataset(flat_config)

    # Run distributed training with pre-created datasets
    run_distributed(
        train_worker,
        world_size=world_size,
        args=(flat_config, train_dataset, valid_dataset, args, args.resume),
    )

    print("Training complete!")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
