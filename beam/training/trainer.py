"""
Training functionality for score/flow based diffusion models — V2.
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
from tqdm.auto import tqdm
import wandb

# UPDATE: V2 model modules
from beam.models.unet import ContextUnet
from beam.models.interpolant import ScoreMatch, EMA
from beam.models.probabilitypath import GaussianProbabilityPath, OTAlpha, OTBeta, VPBeta
from beam.utils.visualization import plot_samples, plot_loss_history
from beam.utils.wandb_utils import log_sample_images

E_PER_S_PER_UNIT = 633118 / 1425.6
BRIGHTNESS_EDGES = [-float('inf'), 1.0, 10.0, 100.0, 1000.0, float('inf')]
BRIGHTNESS_LABELS = ['0-1', '1-10', '10-100', '100-1000', '1000+']


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
        callbacks: List = None
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

        # Choose best convolution algorithm
        torch.backends.cudnn.benchmark = True
        # Create EMA copy of model weights
        self.ema = self._create_ema(beta=config['ema_beta'], update_after_step=config['ema_update_after_step'])
        self.mp_dtype = torch.bfloat16
        self.use_amp = config['training_mixed_precision']
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.use_amp and self.mp_dtype == torch.float16))

        self.max_steps = config['training_n_epoch'] * len(self.train_loader)

        # Create optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['optimizer_max_lr'], weight_decay=config['optimizer_weight_decay'], foreach=True)
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
        Create data loaders for training and validation. (Unchanged.)
        """
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

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training_batch_size'],
            pin_memory=True,
            num_workers=2,
            persistent_workers=True,
            drop_last=True,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
        )

        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config['training_batch_size'],
            pin_memory=True,
            num_workers=1,
            persistent_workers=True,
            drop_last=True,
            sampler=valid_sampler,
            shuffle=False,
        )

        return train_loader, valid_loader

    def _create_model(self) -> ScoreMatch:
        """
        Create and initialize diffusion model.


        """
        # Get conditioning dimension from dataset
        sample = self.train_dataset[0]
        in_dim = sample['x'].shape[1]  # Dimension of the conditioning vector

        unet = ContextUnet(
            in_feats=1,
            context_len=in_dim,
            n_feat=self.config['model_n_feat'],
            channel_mults=self.config['model_channel_mults'],
            heads_at=self.config['model_heads_at'],
            time_dim=self.config.get('model_time_dim', 128),
            context_dim=self.config.get('model_context_dim', 64),
            emb_dim=self.config.get('model_emb_dim', 256),
            num_res=self.config['model_num_res'],
            use_cross_attn=self.config.get('model_use_cross_attn', True),
        )

        fm = ScoreMatch(
            nn_model=unet,
            probability_path=GaussianProbabilityPath(alpha=OTAlpha(), beta=OTBeta() if self.config['model_architecture'] == 'flow' else VPBeta()),
            device=self.device,
            drop_prob=self.config['model_drop_prob'],
            epsilon=self.config['model_epsilon'],
            architecture=self.config['model_architecture'],
            t_sampler=self.config['model_t_sampler'],
            # Ut-distribution shape knobs (see interpolant._sample_t).
            logit_normal_mean=self.config.get('model_t_logit_mean', 0.0),
            logit_normal_std=self.config.get('model_t_logit_std', 1.0),
            low_t_boost_prob=self.config.get('model_t_low_boost_prob', 0.0),
            low_t_boost_min=self.config.get('model_t_low_boost_min', 0.001),
            low_t_boost_max=self.config.get('model_t_low_boost_max', 0.1),
            #  floor of the uniform-sigma draw;
            # only used when model_t_sampler == 'uniform_sigma'.
            sigma_min=self.config.get('model_sigma_min', 0.002),
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

        Also saves the conditioning-normalization statistics
        """
        # Only save if this is the main process
        if self.rank != 0:
            return

        def get_base(dataset):
            return dataset.dataset if isinstance(dataset, Subset) else dataset

        def get_ffi_nums(dataset):
            base = get_base(dataset)
            if not hasattr(base, 'ffi_nums'):
                return None
            if isinstance(dataset, Subset):
                return [base.ffi_nums[idx] for idx in dataset.indices]
            return list(base.ffi_nums)

        base = get_base(self.train_dataset)
        if getattr(base, 'context_mean', None) is not None:
            stats_path = os.path.join(self.save_dir, 'context_stats.pkl')
            with open(stats_path, 'wb') as file:
                pickle.dump(base.context_stats_dict(), file)
            print(f"Saved conditioning normalization stats to {stats_path}")

        # Extract FFI numbers for each split
        training_dataset_ffis = get_ffi_nums(self.train_dataset)
        validation_dataset_ffis = get_ffi_nums(self.valid_dataset)

        # Skip if no FFI
        if training_dataset_ffis is None or validation_dataset_ffis is None:
            print("Datasets have no FFI numbers; skipping dataset-split save.")
            return

        # Save to pickle files
        with open(os.path.join(self.save_dir, 'training_dataset_ffinumbers.pkl'), 'wb') as file:
            pickle.dump(training_dataset_ffis, file)
            print(f"Saved training dataset FFI numbers to {file.name}")

        with open(os.path.join(self.save_dir, 'validation_dataset_ffinumbers.pkl'), 'wb') as file:
            pickle.dump(validation_dataset_ffis, file)
            print(f"Saved validation dataset FFI numbers to {file.name}")

        # The same FFI number appears once per camera, so also save (FFI number,
        # camera) pairs, one per frame
        def get_frames(dataset):
            base = get_base(dataset)
            indices = dataset.indices if isinstance(dataset, Subset) else range(len(base.ffi_nums))
            return list(dict.fromkeys((base.ffi_nums[idx], base.cameras[idx]) for idx in indices))

        if hasattr(base, 'cameras'):
            for split, dataset in (('training', self.train_dataset), ('validation', self.valid_dataset)):
                with open(os.path.join(self.save_dir, f'{split}_dataset_frames.pkl'), 'wb') as file:
                    pickle.dump(get_frames(dataset), file)
                    print(f"Saved {split} dataset (FFI number, camera) pairs to {file.name}")

    def _save_model_info(self) -> None:
        """
        Save model configuration information to a text file. (Unchanged apart
        from the two new fields at the bottom.)
        """
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
            f.write(f"t sampler\t\t\t\t\t{self.config.get('model_t_sampler', 'logit_normal')}\n")
            f.write(f"Cross attention\t\t\t\t{self.config.get('model_use_cross_attn', True)}\n")

        print(f"Model info saved to {os.path.join(self.save_dir, 'model_info.txt')}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a model checkpoint. (Unchanged.)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.ema.load_state_dict(checkpoint['ema_state_dict'])

        self.ema.to(self.device)

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
            EMA of the training loss for the epoch (display only)
        """
       
        self.model.train()

        # Set epoch for distributed sampling
        if self.world_size > 1 and isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(self.current_epoch)

        # Display EMA decoupled from the weight-EMA decay. 
        ema_weight = self.config.get('training_loss_display_ema', 0.98)
        loss_ema_train = None

        for batch_idx, data_batch in enumerate(tqdm(self.train_loader, desc=f"Training epoch {self.current_epoch}")):

            for callback in self.callbacks:
                callback.on_batch_begin(self, batch_idx)

            batch_logs = {}

            self.optimizer.zero_grad()

            x_train = data_batch['y'].to(self.device, non_blocking=True)
            c_train = data_batch['x'].to(self.device, non_blocking=True)
            cam_train = data_batch['cam'].to(self.device, non_blocking=True)

            with torch.autocast('cuda', enabled=self.use_amp, dtype=self.mp_dtype):
                loss_train = self.model(x_train, c_train, cam=cam_train)

            self.scaler.scale(loss_train).backward()
            self.scaler.unscale_(self.optimizer)
            if self.config['optimizer_grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['optimizer_grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update(self.model)

            self.lr_schedule(self.current_step)
            self.current_step += 1

            if loss_ema_train is None:
                loss_ema_train = loss_train.item()
            else:
                loss_ema_train = ema_weight * loss_ema_train + (1 - ema_weight) * loss_train.item()

            batch_logs["batch_loss"] = loss_train.item()
            batch_logs["batch_loss_ema"] = loss_ema_train

            for callback in self.callbacks:
                callback.on_batch_end(self, batch_idx, batch_logs)

        return loss_ema_train

    def validate_epoch(self) -> Tuple[float, List[float], List[float], List[float], List[float]]:
        """
        Validate the model on the validation dataset.

        Returns:
            Tuple of (mean validation loss, per-t-bin mean losses,
            per-t-bin mean flux errors, per-sigma-bin mean losses,
            per-sigma-bin mean flux errors — last two empty unless VP path +
            uniform_sigma, flux error by frame brightness as a list of
            (mean absolute error in e-/s, mean relative error, frame count)
            per BRIGHTNESS_LABELS bin)
        """
        self.model.eval()

        #Load EMA weights
        self.ema.store(self.model)
        self.ema.copy_to(self.model)

    
        seed = int(self.config.get('training_valid_seed', 42))
        gen = torch.Generator()
        gen.manual_seed(seed)

        eps = float(self.config['model_epsilon'])
        num_bins = int(self.config.get('training_valid_t_bins', 10))

        total_loss = torch.zeros(1, device=self.device)
        total_count = torch.zeros(1, device=self.device)
        bin_sums = torch.zeros(num_bins, device=self.device)
        bin_counts = torch.zeros(num_bins, device=self.device)

        # (flux metric): log-spaced bins at low t for the flux-error
        flux_edges = torch.tensor([0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0],
                                  device=self.device)
        flux_sums = torch.zeros(len(flux_edges) - 1, device=self.device)
        flux_counts = torch.zeros(len(flux_edges) - 1, device=self.device)

        # (sigma-binned validation): sigma = beta(t) for VP path
        sigma_mode = (self.config['model_architecture'] != 'flow'
                      and self.config.get('model_t_sampler', 'logit_normal') == 'uniform_sigma')
        sigma_min = float(self.config.get('model_sigma_min', 0.002))
        sigma_edges = torch.tensor([0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0],
                                   device=self.device)
        sigma_loss_sums = torch.zeros(len(sigma_edges) - 1, device=self.device)
        sigma_loss_counts = torch.zeros(len(sigma_edges) - 1, device=self.device)
        sigma_flux_sums = torch.zeros(len(sigma_edges) - 1, device=self.device)
        sigma_flux_counts = torch.zeros(len(sigma_edges) - 1, device=self.device)

        bright_edges = torch.tensor(BRIGHTNESS_EDGES, device=self.device)
        bright_abs = torch.zeros(len(BRIGHTNESS_LABELS), device=self.device)
        bright_rel = torch.zeros(len(BRIGHTNESS_LABELS), device=self.device)
        bright_counts = torch.zeros(len(BRIGHTNESS_LABELS), device=self.device)
        data_mean, data_std = float(self.config['data_mean']), float(self.config['data_std'])

        with torch.no_grad():
            for data_batch in tqdm(self.valid_loader, desc=f"Validation epoch {self.current_epoch}"):
                x_valid = data_batch['y'].to(self.device, non_blocking=True)
                c_valid = data_batch['x'].to(self.device, non_blocking=True)
                cam_valid = data_batch['cam'].to(self.device, non_blocking=True)
                B = x_valid.shape[0]

               
                t = (torch.arange(B, dtype=torch.float32) + torch.rand(B, generator=gen)) / B
                t = t.clamp(eps, 1.0 - eps)
                # sigma-binned validation:
                n_sig = B // 2 if sigma_mode else 0
                if n_sig > 0:
                    u = (torch.arange(n_sig, dtype=torch.float32)
                         + torch.rand(n_sig, generator=gen)) / n_sig
                    sig = sigma_min ** (1.0 - u)
                    t[1::2] = torch.sqrt(1.0 - sig ** 2).clamp_min(eps)
                t = t.view(B, 1, 1, 1).to(self.device)

                noise = torch.randn(x_valid.shape, generator=gen).to(self.device)

                mask = torch.zeros(B, 1, c_valid.shape[2], device=self.device, dtype=c_valid.dtype)

                with torch.autocast('cuda', enabled=self.use_amp, dtype=self.mp_dtype):
                    per_sample, t_used, flux_err = self.model(
                        x_valid, c_valid, cam=cam_valid,
                        t=t, noise=noise, context_mask=mask,
                        reduction='none',
                    )

                per_sample = per_sample.float()
                total_loss += per_sample.sum()
                total_count += B

                #per t bin accumulation for diagnostics
                idx = (t_used.float() * num_bins).long().clamp(0, num_bins - 1)
                bin_sums.index_add_(0, idx, per_sample)
                bin_counts.index_add_(0, idx, torch.ones_like(per_sample))

                # (flux metric): same accumulation on log-spaced bins.
                fidx = (torch.bucketize(t_used.float(), flux_edges) - 1).clamp(0, len(flux_edges) - 2)
                flux_sums.index_add_(0, fidx, flux_err.float())
                flux_counts.index_add_(0, fidx, torch.ones_like(flux_err.float()))

                # (sigma-binned validation): same accumulations keyed
                # by the VP noise level (both grid halves contribute)
                if sigma_mode:
                    sigma_used = torch.sqrt((1.0 - t_used.float() ** 2).clamp_min(0.0))
                    sidx = (torch.bucketize(sigma_used, sigma_edges) - 1).clamp(0, len(sigma_edges) - 2)
                    sigma_loss_sums.index_add_(0, sidx, per_sample)
                    sigma_loss_counts.index_add_(0, sidx, torch.ones_like(per_sample))
                    sigma_flux_sums.index_add_(0, sidx, flux_err.float())
                    sigma_flux_counts.index_add_(0, sidx, torch.ones_like(flux_err.float()))

                # The model's flux for each frame, predicted from pure noise
                with torch.autocast('cuda', enabled=self.use_amp, dtype=self.mp_dtype):
                    _, _, start_flux_err = self.model(
                        x_valid, c_valid, cam=cam_valid,
                        t=torch.full_like(t, eps), noise=noise, context_mask=mask,
                        reduction='none',
                    )
                true_flux = (x_valid.float().mean(dim=(1, 2, 3)) * data_std + data_mean) * E_PER_S_PER_UNIT
                abs_err = start_flux_err.float().sqrt() * data_std * E_PER_S_PER_UNIT
                bidx = (torch.bucketize(true_flux, bright_edges) - 1).clamp(0, len(BRIGHTNESS_LABELS) - 1)
                bright_abs.index_add_(0, bidx, abs_err)
                bright_rel.index_add_(0, bidx, abs_err / true_flux.abs().clamp_min(1.0))
                bright_counts.index_add_(0, bidx, torch.ones_like(abs_err))

        self.ema.restore(self.model)

        # Sync across GPUs (sums, so the result is the true global mean).
        if self.world_size > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(bin_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(bin_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(flux_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(flux_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(bright_abs, op=dist.ReduceOp.SUM)
            dist.all_reduce(bright_rel, op=dist.ReduceOp.SUM)
            dist.all_reduce(bright_counts, op=dist.ReduceOp.SUM)
            if sigma_mode:
                dist.all_reduce(sigma_loss_sums, op=dist.ReduceOp.SUM)
                dist.all_reduce(sigma_loss_counts, op=dist.ReduceOp.SUM)
                dist.all_reduce(sigma_flux_sums, op=dist.ReduceOp.SUM)
                dist.all_reduce(sigma_flux_counts, op=dist.ReduceOp.SUM)

        valid_loss = (total_loss / total_count.clamp(min=1)).item()
        bin_means = (bin_sums / bin_counts.clamp(min=1)).cpu().tolist()
        flux_means = (flux_sums / flux_counts.clamp(min=1)).cpu().tolist()
        if sigma_mode:
            sigma_means = (sigma_loss_sums / sigma_loss_counts.clamp(min=1)).cpu().tolist()
            sigma_flux_means = (sigma_flux_sums / sigma_flux_counts.clamp(min=1)).cpu().tolist()
        else:
            sigma_means, sigma_flux_means = [], []
        n = bright_counts.clamp(min=1)
        brightness = list(zip((bright_abs / n).cpu().tolist(), (bright_rel / n).cpu().tolist(),
                              bright_counts.cpu().long().tolist()))
        return valid_loss, bin_means, flux_means, sigma_means, sigma_flux_means, brightness

    def run_checkpoint(self, save_model: bool) -> bool:
        """
        Run checkpoint operations (saving model, generating samples, plotting).
        (Unchanged.)
        """
        if self.rank != self.config['checkpoint_checkpoint_gpu']:
            return False

        print(f'Running checkpoint at epoch {self.current_epoch} on GPU {self.rank}')

        try:
            plot_loss_history(
                train_losses=self.loss_history_train,
                valid_losses=self.loss_history_valid,
                save_dir=self.save_dir,
                show_last_n=50
            )

            self._save_loss_history()

            early_stop = (self.epochs_without_improvement >= self.config['training_patience'] and
                          self.current_epoch >= self.config['training_patience'])

            return early_stop

        except Exception as e:
            print(f"Error in checkpoint on GPU {self.rank}: {e}")
            traceback.print_exc()
            return False

    def _save_loss_history(self) -> None:
        """
        Save loss history to a text file. (Unchanged.)
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
        """
        n_epoch = n_epoch or self.config['training_n_epoch']

        if self.config['load_checkpoint']:
            self.load_checkpoint(self.config['load_checkpoint'])

        self.start_training_time = time.time()

        for callback in self.callbacks:
            callback.on_train_begin(self)

        # Sample logging needs an active W&B run; there is none when W&B is disabled
        if self.rank == 0 and wandb.run is not None:
            print(f"Logging training samples on GPU {self.rank}")
            model = self.model.module if hasattr(self.model, 'module') else self.model

            log_sample_images(
                model=model,
                dataloader=self.train_loader,
                device=self.device,
                n_sample=1,
                n_datapoint=1,
                image_shape=self.config['data_image_shape'] if self.config['data_patch_size'] is None else self.config.get('data_patch_size'),
                name="Training Set",
                ema=self.ema,
                MEAN=self.config['data_mean'],
                STD=self.config['data_std']
            )
        # Training loop
        for epoch in range(self.current_epoch, n_epoch):
            self.current_epoch = epoch
            print(f'Epoch {epoch}/{n_epoch}, GPU {self.rank}')

            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)

            train_loss = self.train_epoch()
            self.loss_history_train.append(train_loss)

            (valid_loss, valid_t_bins, valid_flux_bins,
             valid_sigma_bins, valid_sigma_flux_bins, valid_brightness) = self.validate_epoch()
            self.loss_history_valid.append(valid_loss)

            torch.cuda.empty_cache()

            current_time = round((time.time() - self.start_training_time) / 3600, 3)  # Hours
            self.time_history.append(current_time)

            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            logs = {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "time_hours": current_time,
                "lr": self.optimizer.param_groups[0]['lr'],
                "best_valid_loss": self.best_valid_loss,
                "epochs_without_improvement": self.epochs_without_improvement
            }
            #validation metrics to give to wandb
            for i, v in enumerate(valid_t_bins):
                logs[f"valid_loss_t_bin_{i}"] = v
            
            for i, v in enumerate(valid_flux_bins):
                logs[f"valid_flux_err_t_bin_{i}"] = v
      
            for i, v in enumerate(valid_sigma_bins):
                logs[f"valid_loss_sigma_bin_{i}"] = v
            for i, v in enumerate(valid_sigma_flux_bins):
                logs[f"valid_flux_err_sigma_bin_{i}"] = v
            for label, low, (abs_err, rel_err, count) in zip(BRIGHTNESS_LABELS, BRIGHTNESS_EDGES, valid_brightness):
                if count:
                    logs[f"valid_flux_abs_err_{label}"] = abs_err
                    if low >= 1:
                        logs[f"valid_flux_rel_err_{label}"] = rel_err

            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, logs)

            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Valid Loss = {valid_loss:.6f}, "
                  f"Time = {current_time:.2f} hours")
            print("  Validation flux error by frame brightness in e-/s: " + " | ".join(
                f"{label}: " + (f"{abs_err:.2f}" + (f" ({rel_err:.0%})" if low >= 1 else "") + f", n={count}"
                                if count else "no frames")
                for label, low, (abs_err, rel_err, count) in zip(BRIGHTNESS_LABELS, BRIGHTNESS_EDGES, valid_brightness)))

            run_checkpoint = (epoch % checkpoint_freq == 0 or
                              epoch == n_epoch - 1 or
                              self.epochs_without_improvement >= self.config['training_patience'])

            if run_checkpoint:
                early_stop = self.run_checkpoint(save_model)

                if self.world_size > 1:
                    early_stop_tensor = torch.tensor([1 if early_stop else 0], device=self.device)
                    dist.all_reduce(early_stop_tensor, op=dist.ReduceOp.MAX)
                    early_stop = early_stop_tensor.item() == 1

                if early_stop or self.stop_training:
                    print(f"Early stopping triggered after {epoch} epochs on GPU {self.rank}")
                    break

        for callback in self.callbacks:
            callback.on_train_end(self)

        return self.loss_history_train, self.loss_history_valid
