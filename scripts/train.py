#!/usr/bin/env python
"""
"""

import os
import argparse
import datetime
import torch
import torch.multiprocessing as mp
import time
import pickle

from beam.data.datasets import TESSDataset, create_train_valid_datasets_by_orbit, compute_image_stats
from beam.training.trainer import SMTrainer
from beam.training.distributed import run_distributed
from beam.utils.config import load_config, prepare_training_config
from beam.training.wandb_callback import WeightsAndBiasesCallback
from beam.training.callbacks import ModelCheckpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a diffusion model on TESS images")

    parser.add_argument('--config', type=str, default='configs/sm_config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--wandb', type=str, default='online',
                        help='W&B logging mode: online, offline, or disabled')

    return parser.parse_args()


def prepare_dataset(config):
    """
    Prepare the TESS dataset for training and validation.
    """
    print(f"Creating TESSDataset...")
    # Now computing mean/std at runtime
    tess_dataset = TESSDataset(
        angle_path=config['data_angle_path'],
        ccd_folder=config['data_ccd_folder'],
        image_shape=config['data_image_shape'],
        patch_size=config.get('data_patch_size', None),
        repeat_factor=config.get('data_repeat_factor', 1),
        mean=0.0,
        std=1.0,
        camera_number=config.get('data_camera_number', '3'),
        #conditioning encoding
        add_exposure_flag=config.get('data_add_exposure_flag', True),
        normalize_context=config.get('data_normalize_context', True),
        # sky-normalized / unconditional variants (star prior). flatten_config
        # explodes the yaml's data.sky_by_sector mapping into one key per
        # sector ("data_sky_by_sector_<N>"), so reassemble it here.
        sky_by_sector=({int(k.rsplit('_', 1)[1]): float(v)
                        for k, v in config.items()
                        if k.startswith('data_sky_by_sector_')} or None),
        zero_context=config.get('data_zero_context', False),
    )

    print("Creating training and validation splits by orbit...")
    train_dataset, valid_dataset = create_train_valid_datasets_by_orbit(
        tess_dataset,
        config["data_valid_orbits"],
    )

    if config.get('data_compute_stats', False):
        mean, std = compute_image_stats(
            tess_dataset,
            train_dataset.indices,
            max_images=config.get('data_stats_max_images', None),
        )
        print(f"Computed image stats over train split: mean={mean:.6g}, std={std:.6g}")
    elif config.get('data_stats_path'):
        with open(config['data_stats_path'], 'rb') as f:
            stats = pickle.load(f)
        mean, std = float(stats['mean']), float(stats['std'])
        print(f"Loaded image stats from {config['data_stats_path']}: mean={mean:.6g}, std={std:.6g}")
    else:
        raise ValueError(
            "No image normalization source: set data.compute_stats: true or provide "
            "data.stats_path (an image_stats.pkl from a previous run). Hand-set "
            "data.mean/data.std are no longer supported."
        )

    tess_dataset.MEAN = mean
    tess_dataset.STD = std
    config['data_mean'] = mean
    config['data_std'] = std
    stats_path = os.path.join(config['paths_save_dir'], 'image_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump({
            'mean': mean, 'std': std,
            'definition': 'normalized = (flux - mean) / std, flux as stored (30-minute exposure units)',
        }, f)
    print(f"Saved image stats to {stats_path}")

    return train_dataset, valid_dataset


def create_callbacks(config, args, rank):
    """
    Create training callbacks.
    """
    callbacks = []

    # Only add callbacks on the main process (rank 0)
    if rank == 0:
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(config['paths_save_dir'], 'checkpoint_epoch_{epoch}_{datetime}.pth'),
            monitor='valid_loss',
            save_best_only=config.get('checkpoint_save_best_only', True),
            save_weights_only=config.get('checkpoint_save_weights_only', False),
            mode='min',
            period=config.get('checkpoint_epoch_checkpoint', 100)
        )
        callbacks.append(checkpoint_callback)



        if args.wandb != 'disabled':
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
    Worker function for training on a single GPU. (Unchanged apart from the
    V2 trainer import.)
    """
    if device is None:
        device = torch.device(f'cuda:{rank}')

    print(f"GPU {rank}: Starting worker with {len(train_dataset) // world_size} training samples "
          f"and {len(valid_dataset) // world_size} validation samples on device {device}")

    callbacks = create_callbacks(config, args, rank)

    trainer = SMTrainer(
        config=config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        rank=rank,
        world_size=world_size,
        device=device,
        callbacks=callbacks
    )

    if resume_path:
        trainer.load_checkpoint(resume_path)
        print(f"GPU {rank}: Resumed training from {resume_path}")

    trainer.train(
        checkpoint_freq=config['checkpoint_epoch_checkpoint'],
        save_model=config['checkpoint_save_model']
    )

    print(f"GPU {rank}: Training complete")


def main():
    """Main function to set up and launch distributed training."""
    print("Beginning Training")
    args = parse_args()

    config = load_config(args.config)
    flat_config = prepare_training_config(config)

    world_size = flat_config['distributed_world_size']
    print(f"Using {world_size} GPUs for training")

    print("Preparing dataset in main process...")
    train_dataset, valid_dataset = prepare_dataset(flat_config)

    run_distributed(
        train_worker,
        world_size=world_size,
        args=(flat_config, train_dataset, valid_dataset, args, args.resume),
    )

    print("Training complete!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
