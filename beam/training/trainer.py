"""
Training functionality for score/flow based diffusion models.

This module contains classes and functions for training and evaluating
diffusion models on TESS data.
"""

import os
import time
import pickle
from typing import Dict, List, Tuple, Optional, Union, Callable
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, utils, transforms
from tqdm.auto import tqdm

from beam.models.unet import ContextUnet
from beam.utils.visualization import plot_samples, plot_loss_history
from beam.utils.wandb_utils import log_sample_images
from beam.models.probabilitypath import GaussianProbabilityPath, OTAlpha, OTBeta, VPBeta
from beam.models.interpolant import ScoreMatch, EMA
from beam.models.scores import VESDE, SBD

class SMTrainer:
    """
    Trainer class for flow/score matching models.
    """
    def __init__(
        self,
        config: Dict,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        rank: int = 0,
        world_size: int = 1,
        device: Optional[torch.device] = None,
        callbacks: List = None  # Add callbacks parameter
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.rank = rank
        self.world_size = world_size
        self.device = device or torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        # Enforce CUDA for distributed training
        if self.world_size > 1 and self.device.type != "cuda":
            raise RuntimeError("Distributed training with multiple GPUs requires CUDA devices. Please run on a machine with GPUs and CUDA available.")
        
        
        # Initialize callbacks
        self.callbacks = callbacks or []
        
        # Create data loaders
        self.train_loader, self.valid_loader = self._create_data_loaders()
        
        # Create model
        self.model = self._create_model() 
         # self.model = self._create_sde()

        # Choose best convolution algorithm
        torch.backends.cudnn.benchmark = True
        # Create EMA copy of model weights
        self.ema = self._create_ema(beta = config['ema_beta'], update_after_step = config['ema_update_after_step'])
        self.mp_dtype = torch.bfloat16 
        self.use_amp = config['training_mixed_precision']
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.use_amp and self.mp_dtype == torch.float16))
     

        self.max_steps = config['training_n_epoch'] *  len(self.train_loader)
            
        # Create optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['optimizer_max_lr'], weight_decay=config['optimizer_weight_decay'], fused=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_steps, eta_min=config['optimizer_min_lr'])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = config['optimizer_max_lr'] if config['optimizer_warmup_steps'] == 0 else self.config['optimizer_min_lr']
        
        # Initialize training state
        self.loss_history_train = []
        self.loss_history_valid = []
        self.time_history = []
        self.best_valid_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        self.current_step = 0
        self.start_training_time = None
        self.stop_training = False  # Flag for early stopping


        
        # Create checkpoint directory
        self.save_dir = config['paths_save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save dataset info
        if rank == config['checkpoint_checkpoint_gpu']:
            print(f"GPU {rank}: Saving dataset splits and model info...")
            self._save_dataset_splits()
            self._save_model_info()
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for training and validation.
        
        Returns:
            Tuple of (training data loader, validation data loader)
        """
        # Create distributed samplers if using multiple GPUs
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                self.train_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank, 
                shuffle=True
            )
            
            valid_sampler = DistributedSampler(
                self.valid_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank, 
                shuffle=False
            )
        else:
            train_sampler = None
            valid_sampler = None
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config['training_batch_size'], 
            pin_memory=True, 
            num_workers=2,
            persistent_workers=True,
            drop_last=True, 
            sampler=train_sampler,
            shuffle=(train_sampler is None)
        )
        
        valid_loader = DataLoader(
            self.valid_dataset, 
            batch_size=self.config['training_batch_size'], 
            pin_memory=True,
            num_workers=2, 
            persistent_workers=True,
            drop_last=True, 
            sampler=valid_sampler,
            shuffle=False
        )
        
        return train_loader, valid_loader
    
    def _create_model(self) -> ScoreMatch:
        """
        Create and initialize diffusion model.
        
        Returns:
            Initialized DDPM model
        """
        # Get conditioning dimension from dataset
        sample = self.train_dataset[0]
        in_dim = sample['x'].shape[1]  # Dimension of the conditioning vector
        
        # Create U-Net model
        unet = ContextUnet(
            in_feats=1, 
            context_len=in_dim, 
            n_feat=self.config['model_n_feat'],
            channel_mults=self.config['model_channel_mults'],
            heads_at=self.config['model_heads_at'],
            time_dim = 64, #16 for 1024
            context_dim = 64,
            num_res = self.config['model_num_res']

        )
        
        # Create score matching model
        
        fm = ScoreMatch(
            nn_model=unet, 
            probability_path=GaussianProbabilityPath(alpha=OTAlpha(), beta=OTBeta() if self.config['model_architecture'] == 'flow' else VPBeta()),
            device=self.device,
            drop_prob=self.config['model_drop_prob'],
            epsilon=self.config['model_epsilon'],
            architecture=self.config['model_architecture']
        )
        fm.to(self.device)

        # Wrap model for distributed training if using multiple GPUs
        if self.world_size > 1:
            fm = nn.parallel.DistributedDataParallel(
                fm, 
                device_ids=[self.rank],
                static_graph=True,
                find_unused_parameters=False,
            )
        return fm

    def _create_ema(self, beta: float = 0.999, update_after_step: int = 0) -> EMA:
        """
        Create exponential moving average of model parameters.
        """
        return EMA(self.model, beta=beta, update_after_step=update_after_step)
    
    def _save_dataset_splits(self) -> None:
        """
        Save information about training and validation dataset splits.
        """
        # Only save if this is the main process
        if self.rank != 0:
            return
        # Helper function to extract FFI numbers from a dataset
        def get_ffi_nums(dataset):
            if isinstance(dataset, Subset):
                ffi_nums = [dataset.dataset.ffi_nums[idx] for idx in dataset.indices]
                return ffi_nums
            else:
                return list(dataset.ffi_nums)
        
        # Extract FFI numbers for each split
        training_dataset_ffis = get_ffi_nums(self.train_dataset)
        validation_dataset_ffis = get_ffi_nums(self.valid_dataset)
        
        # Save to pickle files
        with open(os.path.join(self.save_dir, 'training_dataset_ffinumbers.pkl'), 'wb') as file:
            pickle.dump(training_dataset_ffis, file)
            print(f"Saved training dataset FFI numbers to {file.name}")
            
        with open(os.path.join(self.save_dir, 'validation_dataset_ffinumbers.pkl'), 'wb') as file:
            pickle.dump(validation_dataset_ffis, file)
            print(f"Saved validation dataset FFI numbers to {file.name}")
    
    def _save_model_info(self) -> None:
        """
        Save model configuration information to a text file.
        """
        # Only save if this is the main process 
        if self.rank != 0:
            return
            
        with open(os.path.join(self.save_dir, 'model_info.txt'), 'w') as f:
            f.write("MODEL AND DATA INFORMATION\n")
            f.write(f"Max number of epochs\t\t\t{self.config['training_n_epoch']}\n")
            f.write(f"Batch size\t\t\t\t\t\t{self.config['training_batch_size']}\n")
            f.write(f"Training/validation ratio\t\t{len(self.train_dataset) / (len(self.train_dataset) + len(self.valid_dataset)):.2f}\n")
            f.write(f"n_feat (CNN num of features)\t{self.config['model_n_feat']}\n")
            f.write(f"Learning rate\t\t\t\t\t{self.config['optimizer_max_lr']}\n")
            f.write(f"Image size\t\t\t\t\t\t{self.config['data_image_shape']}\n")
            f.write(f"GPUs used\t\t\t\t\t\t{self.world_size}\n")
            f.write(f"Early stopping patience\t\t{self.config['training_patience']}\n")
            f.write(f"Architecture\t\t\t\t\t{self.config['model_architecture']}\n")
            f.write(f"Mixed precision\t\t\t\t{self.config['training_mixed_precision']}\n")
            f.write(f"Patch size\t\t\t\t\t{self.config['data_patch_size']}\n")
        
        print(f"Model info saved to {os.path.join(self.save_dir, 'model_info.txt')}")
    
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get base model (remove DistributedDataParallel wrapper if needed)
        model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        
        # Load model and optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.ema.load_state_dict(checkpoint['ema_state_dict'])

        self.ema.to(self.device)
        
        
        # Load training state
        self.current_epoch = checkpoint['epoch'] + 1
        self.loss_history_train = checkpoint['train_loss']
        self.loss_history_valid = checkpoint['valid_loss']
        if checkpoint['scaler_state_dict'] is not None and self.config['training_mixed_precision']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.time_history = checkpoint['time_history']
        self.best_valid_loss = checkpoint['best_valid_loss']
        self.epochs_without_improvement = checkpoint['epochs_without_improvement']
        
  
    def lr_schedule(self, step: int) -> float:
        if step <= self.config['optimizer_warmup_steps']:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config['optimizer_max_lr'] * step / self.config['optimizer_warmup_steps']
        else:
            self.scheduler.step()
        
    
    def train_epoch(self) -> float:
        """
        Train the model for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        
        # Set epoch for distributed sampling
        if self.world_size > 1 and isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(self.current_epoch)
        
        # Apply learning rate decay
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] = self.config['optimizer_max_lr'] * (
        #         1 - self.current_epoch / self.config['training_n_epoch']
        #     )
        # if self.config['optimizer_warmup_steps'] > 0:
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = self.config['optimizer_max_lr'] * (
        #             1 - self.current_epoch / self.config['training_n_epoch']
        #         )
        ema_weight = self.config['ema_beta']
        # Training loop
        loss_ema_train = None
        
        for batch_idx, data_batch in enumerate(tqdm(self.train_loader, desc=f"Training epoch {self.current_epoch}")):

            # Call on_batch_begin callbacks
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch_idx)
                
            # Prepare batch logs
            batch_logs = {}
            
            self.optimizer.zero_grad()
            
            # Move data to device
            x_train = data_batch['y'].to(self.device, non_blocking=True)
            c_train = data_batch['x'].to(self.device, non_blocking=True)

            
            # Forward pass and loss calculation
            with torch.autocast('cuda', enabled=self.use_amp, dtype=self.mp_dtype):
                loss_train = self.model(x_train, c_train)

            self.scaler.scale(loss_train).backward()
            self.scaler.unscale_(self.optimizer)
            if self.config['optimizer_grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['optimizer_grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
   
            self.ema.update(self.model)

            #Update learning rate
            self.lr_schedule(self.current_step)
            self.current_step += 1
                
            # Update exponential moving average of loss
            if loss_ema_train is None:
                loss_ema_train = loss_train.item()
            else:
                loss_ema_train = ema_weight * loss_ema_train + (1 - ema_weight) * loss_train.item()
            
            # Update batch logs
            batch_logs["batch_loss"] = loss_train.item()
            batch_logs["batch_loss_ema"] = loss_ema_train
            
            # Call on_batch_end callbacks
            for callback in self.callbacks:
                callback.on_batch_end(self, batch_idx, batch_logs)
        
        return loss_ema_train
    
    def validate_epoch(self) -> float:
        """
        Validate the model on the validation dataset.
        
        Returns:
            Average validation loss
        """
        # Set to evaluation mode
        self.model.eval()
        
        # Validation loop
        loss_ema_valid = None
        ema_weight = self.config['ema_beta']
        with torch.no_grad():
            for data_batch in tqdm(self.valid_loader, desc=f"Validation epoch {self.current_epoch}"):
                # Move data to device
                x_valid = data_batch['y'].to(self.device, non_blocking=True)
                c_valid = data_batch['x'].to(self.device, non_blocking=True)
                
                # Forward pass and loss calculation
                with torch.autocast('cuda', enabled=self.use_amp, dtype=self.mp_dtype):
                    loss_valid = self.model(x_valid, c_valid)
                
                # Update exponential moving average of loss
                if loss_ema_valid is None:
                    loss_ema_valid = loss_valid.item()
                else:
                    loss_ema_valid = ema_weight * loss_ema_valid + (1 - ema_weight) * loss_valid.item()
        return loss_ema_valid
    
    
    def run_checkpoint(self, save_model: bool) -> bool:
        """
        Run checkpoint operations (saving model, generating samples, plotting)
        
        Args:
            save_model: Whether to save the model weights
            
        Returns:
            True if early stopping should be triggered, False otherwise
        """
        # Only run on the designated GPU
        if self.rank != self.config['checkpoint_checkpoint_gpu']:
            return False
            
        print(f'Running checkpoint at epoch {self.current_epoch} on GPU {self.rank}')
        
        try:                  
            # Save loss plots
            plot_loss_history(
                train_losses=self.loss_history_train,
                valid_losses=self.loss_history_valid,
                save_dir=self.save_dir,
                show_last_n=50
            )
            
            # Save loss history to text file
            self._save_loss_history()
            
            
            # Check for early stopping
            early_stop = (self.epochs_without_improvement >= self.config['training_patience'] and 
                        self.current_epoch >= self.config['training_patience'])
            
            return early_stop
            
        except Exception as e:
            print(f"Error in checkpoint on GPU {self.rank}: {e}")
            traceback.print_exc()
            # Don't stop training just because checkpoint failed
            return False
    
    def _save_loss_history(self) -> None:
        """
        Save loss history to a text file.
        """
        with open(os.path.join(self.save_dir, 'loss_history.txt'), 'w') as f:
            f.write("Epoch\tTraining MSE Loss\tValidation MSE Loss\tTime(Hrs)\n")
            
            for i in range(len(self.loss_history_valid)):
                f.write(
                    f"{i}\t\t{self.loss_history_train[i]:.4e}\t\t"
                    f"{self.loss_history_valid[i]:.4e}\t\t{self.time_history[i]}\n"
                )
            
        print(f"Loss history saved to {os.path.join(self.save_dir, 'loss_history.txt')}")
    
    def train(
        self, 
        n_epoch: Optional[int] = None, 
        checkpoint_freq: int = 10,
        save_model: bool = False
    ) -> Tuple[List[float], List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            n_epoch: Number of epochs to train for (None = use config value)
            checkpoint_freq: Run checkpoint operations every N epochs
            save_model: Whether to save model weights at checkpoints
            
        Returns:
            Tuple of (training loss history, validation loss history)
        """
        # Set number of epochs
        n_epoch = n_epoch or self.config['training_n_epoch']

        if self.config['load_checkpoint']:
            self.load_checkpoint(self.config['load_checkpoint'])
        
        # Initialize timing
        self.start_training_time = time.time()
        
        # Call on_train_begin callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)

        if self.rank == 0:
            print(f"Logging training samples on GPU {self.rank}")
            model = self.model.module if hasattr(self.model, 'module') else self.model
                
            # Log training samples
            log_sample_images(
                model=model,
                dataloader=self.train_loader,
                device=self.device,
                n_sample=1,
                n_datapoint=1,
                image_shape=self.config['data_image_shape'] if self.config['data_patch_size'] is None else self.config.get('data_patch_size'),
                name="Training Set",
                ema=self.ema
            )
            
        #     # Log validation samples
        #     log_sample_images(
        #         model=model,
        #         dataloader=self.valid_loader,
        #         device=self.device,
        #         n_sample=1,
        #         n_datapoint=1,
        #         image_shape=self.config['data_image_shape'] if self.config['data_patch_size'] is None else self.config.get('data_patch_size'),
        #         name="Validation Set",
        #         ema=self.ema
        #     )
        # Training loop
        for epoch in range(self.current_epoch, n_epoch):
            self.current_epoch = epoch
            print(f'Epoch {epoch}/{n_epoch}, GPU {self.rank}')
            
            # Call on_epoch_begin callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)
            
            # Train for one epoch
            train_loss = self.train_epoch()
            self.loss_history_train.append(train_loss)
            
            # Validate for one epoch
            valid_loss = self.validate_epoch() 
            self.loss_history_valid.append(valid_loss)

            # Memory cleanup here
            torch.cuda.empty_cache()
            
            # Update time history
            current_time = round((time.time() - self.start_training_time) / 3600, 3)  # Hours
            self.time_history.append(current_time)
            
            # Check for improvement
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Create logs dictionary for callbacks
            logs = {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "time_hours": current_time,
                "lr": self.optimizer.param_groups[0]['lr'],
                "best_valid_loss": self.best_valid_loss,
                "epochs_without_improvement": self.epochs_without_improvement
            }
            
            # Call on_epoch_end callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, logs)
            

        
            # Print status
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Valid Loss = {valid_loss:.6f}, "
                  f"Time = {current_time:.2f} hours")
            
            # Run checkpoint operations
            run_checkpoint = (epoch % checkpoint_freq == 0 or 
                             epoch == n_epoch - 1 or 
                             self.epochs_without_improvement >= self.config['training_patience'])
            
            if run_checkpoint:
                early_stop = self.run_checkpoint(save_model)
                
                # Synchronize early stopping across all GPUs
                if self.world_size > 1:
                    early_stop_tensor = torch.tensor([1 if early_stop else 0], device=self.device)
                    dist.all_reduce(early_stop_tensor, op=dist.ReduceOp.MAX)
                    early_stop = early_stop_tensor.item() == 1
                
                if early_stop or self.stop_training:
                    print(f"Early stopping triggered after {epoch} epochs on GPU {self.rank}")
                    break
        
        # Call on_train_end callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        return self.loss_history_train, self.loss_history_valid



        