#!/usr/bin/env python3
"""Train a simple conditional CNN on 256x256 scattered-light images.

This script is intentionally minimal:
- loads full 256x256 samples from ``TESSDataset`` with no patching
- trains on a single device only
- saves best/last checkpoints
- exports validation predictions and summary statistics
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from beam.data.datasets import TESSDataset, create_train_valid_datasets_by_orbit


DEFAULT_ANGLE_PATH = "/pdo/users/jlupoiii/TESS/data/angles/angles_O11-54_data_dic.pkl"
DEFAULT_CCD_FOLDER = "/pdo/users/jlupoiii/TESS/data/processed_images_im256x256"
DEFAULT_IMAGE_SHAPE = (256, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple CNN from orbital conditioning to 256x256 scattered-light images"
    )
    parser.add_argument("--angle-path", type=str, default=DEFAULT_ANGLE_PATH)
    parser.add_argument("--ccd-folder", type=str, default=DEFAULT_CCD_FOLDER)
    parser.add_argument("--image-shape", type=int, nargs=2, default=list(DEFAULT_IMAGE_SHAPE))
    parser.add_argument("--camera-number", type=str, default="3")
    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--repeat-factor", type=int, default=1)
    parser.add_argument("--orbit-threshold", type=int, default=45)
    parser.add_argument("--orbit-max", type=int, default=54)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["mse", "l1", "huber"], default="mse")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--min-channels", type=int, default=32)
    parser.add_argument("--start-shape", type=int, nargs=2, default=[4, 4])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-save-images", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _group_count(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = _group_count(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        return x + residual


class SimpleConditioningCNN(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        output_shape: Tuple[int, int],
        hidden_dim: int = 256,
        base_channels: int = 128,
        min_channels: int = 32,
        start_shape: Tuple[int, int] = (4, 4),
    ):
        super().__init__()
        target_h, target_w = output_shape
        start_h = max(1, min(start_shape[0], target_h))
        start_w = max(1, min(start_shape[1], target_w))

        self.output_shape = (target_h, target_w)
        self.start_shape = (start_h, start_w)
        self.base_channels = base_channels

        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, base_channels * start_h * start_w),
        )

        blocks = []
        stage_sizes = []
        current_h, current_w = self.start_shape
        channels = base_channels
        while current_h < target_h or current_w < target_w:
            next_h = min(current_h * 2, target_h)
            next_w = min(current_w * 2, target_w)
            next_channels = max(min_channels, channels // 2)
            stage_sizes.append((next_h, next_w))
            blocks.append(UpsampleBlock(channels, next_channels))
            current_h, current_w = next_h, next_w
            channels = next_channels

        self.stage_sizes = stage_sizes
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        conditioning = conditioning.flatten(start_dim=1).float()
        x = self.cond_net(conditioning)
        x = x.view(conditioning.shape[0], self.base_channels, *self.start_shape)

        for size, block in zip(self.stage_sizes, self.blocks):
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            x = block(x)

        if x.shape[-2:] != self.output_shape:
            x = F.interpolate(x, size=self.output_shape, mode="bilinear", align_corners=False)

        return self.head(x)


def build_dataset(args: argparse.Namespace):
    image_shape = tuple(args.image_shape)
    dataset = TESSDataset(
        angle_path=args.angle_path,
        ccd_folder=args.ccd_folder,
        image_shape=image_shape,
        mean=args.mean,
        std=args.std,
        patch_size=None,
        repeat_factor=args.repeat_factor,
        camera_number=args.camera_number,
    )
    train_dataset, valid_dataset = create_train_valid_datasets_by_orbit(
        dataset,
        orbit_threshold=args.orbit_threshold,
        max_orbit=args.orbit_max,
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise RuntimeError("Empty train or validation split. Check dataset paths and orbit bounds.")
    return train_dataset, valid_dataset


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def make_loss(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    return nn.SmoothL1Loss()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
    desc: str = "",
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0

    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=True)
        y = batch["y"].to(device=device, dtype=torch.float32, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        loss = loss_fn(pred, y)

        if training:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = y.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_items += batch_size
        progress.set_postfix(loss=f"{loss.detach().item():.4f}")

    return total_loss / max(total_items, 1)


def denormalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return x * std + mean


def save_preview(target: np.ndarray, pred: np.ndarray, output_path: str, title: str) -> None:
    residual = pred - target
    vmin, vmax = 0, 1
    rmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(target, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Target")
    axes[1].imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    axes[2].imshow(residual, cmap="coolwarm", vmin=-rmax, vmax=rmax)
    axes[2].set_title("Residual")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


class RunningMean:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(dtype=torch.float64)
        batch_count = values.numel()
        if batch_count == 0:
            return

        batch_mean = values.mean().item()
        total = self.count + batch_count
        self.mean += (batch_mean - self.mean) * (batch_count / float(total))
        self.count = total


class RunningMoments:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(dtype=torch.float64)
        batch_count = values.numel()
        if batch_count == 0:
            return

        batch_mean = values.mean().item()
        batch_m2 = torch.square(values - batch_mean).sum().item()

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / float(total)
        self.mean += delta * batch_count / float(total)
        self.count = total

    @property
    def std(self) -> float:
        if self.count == 0:
            return 0.0
        return math.sqrt(max(self.m2 / float(self.count), 0.0))


@torch.no_grad()
def export_validation_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: str,
    mean: float,
    std: float,
    num_save_images: int,
) -> Dict[str, float]:
    model.eval()
    image_dir = os.path.join(output_dir, "generated_images")
    array_dir = os.path.join(output_dir, "generated_arrays")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(array_dir, exist_ok=True)

    sq_error = RunningMean()
    abs_error = RunningMean()
    pred_stats = RunningMoments()
    target_stats = RunningMoments()
    residual_stats = RunningMoments()
    example_count = 0
    saved = 0

    for batch in tqdm(loader, desc="export", leave=False):
        x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=True)
        y = batch["y"].to(device=device, dtype=torch.float32, non_blocking=True)
        ffi_nums = batch["ffi_num"]

        pred = model(x)
        y = denormalize(y, mean, std)
        pred = denormalize(pred, mean, std)
        residual = pred - y

        sq_error.update(torch.square(residual))
        abs_error.update(torch.abs(residual))
        pred_stats.update(pred)
        target_stats.update(y)
        residual_stats.update(residual)
        example_count += y.shape[0]

        for i in range(y.shape[0]):
            if saved >= num_save_images:
                break
            ffi_num = str(ffi_nums[i])
            target_np = y[i, 0].detach().cpu().numpy()
            pred_np = pred[i, 0].detach().cpu().numpy()
            np.save(os.path.join(array_dir, f"{saved:03d}_{ffi_num}_target.npy"), target_np)
            np.save(os.path.join(array_dir, f"{saved:03d}_{ffi_num}_prediction.npy"), pred_np)
            save_preview(
                target_np,
                pred_np,
                os.path.join(image_dir, f"{saved:03d}_{ffi_num}.png"),
                title=f"FFI {ffi_num}",
            )
            saved += 1

    mse = sq_error.mean
    mae = abs_error.mean

    stats = {
        "num_examples": example_count,
        "num_pixels": pred_stats.count,
        "valid_mse": mse,
        "valid_rmse": math.sqrt(mse),
        "valid_mae": mae,
        "prediction_mean": pred_stats.mean,
        "prediction_std": pred_stats.std,
        "target_mean": target_stats.mean,
        "target_std": target_stats.std,
        "residual_mean": residual_stats.mean,
        "residual_std": residual_stats.std,
    }

    with open(os.path.join(output_dir, "stats.json"), "w") as handle:
        json.dump(stats, handle, indent=2)

    return stats


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("model_outputs", f"simple_cnn_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "run_args.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)

    print(f"Using device: {device}")
    print("Building dataset...")
    train_dataset, valid_dataset = build_dataset(args)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    valid_loader = make_loader(valid_dataset, args.batch_size, False, args.num_workers)

    sample = train_dataset[0]
    cond_dim = sample["x"].numel()
    output_shape = tuple(sample["y"].shape[-2:])
    print(f"Condition dim: {cond_dim}")
    print(f"Output shape: {output_shape}")
    print(f"Train samples: {len(train_dataset)} | Valid samples: {len(valid_dataset)}")

    model = SimpleConditioningCNN(
        cond_dim=cond_dim,
        output_shape=output_shape,
        hidden_dim=args.hidden_dim,
        base_channels=args.base_channels,
        min_channels=args.min_channels,
        start_shape=tuple(args.start_shape),
    ).to(device)
    loss_fn = make_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"train_loss": [], "valid_loss": []}
    best_valid_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            desc=f"train {epoch}",
        )
        with torch.no_grad():
            valid_loss = run_epoch(
                model=model,
                loader=valid_loader,
                loss_fn=loss_fn,
                device=device,
                optimizer=None,
                grad_clip=None,
                desc=f"valid {epoch}",
            )

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        with open(os.path.join(output_dir, "history.json"), "w") as handle:
            json.dump(history, handle, indent=2)

        print(f"  train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_valid_loss": best_valid_loss,
                    "history": history,
                    "args": vars(args),
                },
                os.path.join(output_dir, "best.pt"),
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Stopping early after {epoch} epochs")
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_valid_loss": best_valid_loss,
            "history": history,
            "args": vars(args),
        },
        os.path.join(output_dir, "last.pt"),
    )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    stats = export_validation_outputs(
        model=model,
        loader=valid_loader,
        device=device,
        output_dir=output_dir,
        mean=args.mean,
        std=args.std,
        num_save_images=args.num_save_images,
    )

    print("Finished")
    print(f"Best validation loss: {best_valid_loss:.6f}")
    print(f"Validation MSE: {stats['valid_mse']:.6f}")
    print(f"Validation MAE: {stats['valid_mae']:.6f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
