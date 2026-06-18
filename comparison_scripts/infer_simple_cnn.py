#!/usr/bin/env python3
"""Run inference for a trained simple CNN checkpoint.

Loads a checkpoint saved by train_simple_cnn.py, rebuilds the model from
the stored args, and generates preview images + arrays for requested FFIs
(or the first N validation samples).
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from beam.data.datasets import TESSDataset, create_train_valid_datasets_by_orbit

# ---------------------------------------------------------------------------
# Reuse model definition from training script
# ---------------------------------------------------------------------------
from scripts.train_simple_cnn import SimpleConditioningCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for a trained simple CNN")
    parser.add_argument("--model-path", type=str, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-save-images", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--ffi-nums", type=str, nargs="+", default=['00034520','00033682', '00033637'],
        help="Only generate for these FFI numbers. Alternatively pass a path "
             "to a directory of '<idx>_<ffi>.png' files to extract FFIs from.",
    )
    return parser.parse_args()


def resolve_ffi_nums(raw: List[str]) -> Set[str]:
    if len(raw) == 1 and os.path.isdir(raw[0]):
        ffis: Set[str] = set()
        for fname in os.listdir(raw[0]):
            m = re.match(r"\d+_(\d+)\.png$", fname)
            if m:
                ffis.add(m.group(1))
        if not ffis:
            raise ValueError(f"No '<idx>_<ffi>.png' files found in {raw[0]}")
        return ffis
    return set(raw)


def get_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_dataset(train_args: Dict):
    image_shape = tuple(train_args["image_shape"])
    dataset = TESSDataset(
        angle_path=train_args["angle_path"],
        ccd_folder=train_args["ccd_folder"],
        image_shape=image_shape,
        mean=train_args.get("mean", 0.0),
        std=train_args.get("std", 1.0),
        patch_size=None,
        repeat_factor=train_args.get("repeat_factor", 1),
        camera_number=train_args.get("camera_number", "3"),
    )
    _, valid_dataset = create_train_valid_datasets_by_orbit(
        dataset,
        orbit_threshold=train_args.get("orbit_threshold", 45),
        max_orbit=train_args.get("orbit_max", 54),
    )
    return valid_dataset


def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


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


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
    num_save_images: int,
    output_dir: str,
    target_ffis: Optional[Set[str]] = None,
) -> None:
    model.eval()
    image_dir = os.path.join(output_dir, "generated_images")
    array_dir = os.path.join(output_dir, "generated_arrays")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(array_dir, exist_ok=True)

    saved = 0
    needed = set(target_ffis) if target_ffis is not None else None

    for batch in tqdm(loader, desc="infer", leave=False):
        if needed is not None and len(needed) == 0:
            break
        if needed is None and saved >= num_save_images:
            break

        ffi_nums = batch["ffi_num"]

        if needed is not None:
            hits = [i for i, f in enumerate(ffi_nums) if str(f) in needed]
            if not hits:
                continue
            idx = torch.tensor(hits, dtype=torch.long)
            x = batch["x"][idx].to(device=device, dtype=torch.float32, non_blocking=True)
            y = batch["y"][idx].to(device=device, dtype=torch.float32, non_blocking=True)
            ffi_nums = [str(ffi_nums[i]) for i in hits]
        else:
            x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=True)
            y = batch["y"].to(device=device, dtype=torch.float32, non_blocking=True)
            ffi_nums = [str(f) for f in ffi_nums]

        pred = model(x)
        y = denormalize(y, mean, std)
        pred = denormalize(pred, mean, std)

        for i in range(y.shape[0]):
            if needed is None and saved >= num_save_images:
                break
            ffi_num = ffi_nums[i]
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
            if needed is not None:
                needed.discard(ffi_num)

    if needed:
        print(f"Warning: FFIs not found in validation set: {needed}")
    print(f"Saved {saved} preview images to {image_dir}")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    checkpoint = torch.load(args.model_path, map_location=device)
    train_args = checkpoint["args"]

    output_dir = args.output_dir or os.path.join(
        "model_outputs",
        "cnn_inference_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using device: {device}")
    valid_dataset = build_dataset(train_args)
    valid_loader = make_loader(valid_dataset, args.batch_size, args.num_workers)

    sample = valid_dataset[0]
    cond_dim = sample["x"].numel()
    output_shape = tuple(sample["y"].shape[-2:])
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Output shape: {output_shape}")

    model = SimpleConditioningCNN(
        cond_dim=cond_dim,
        output_shape=output_shape,
        hidden_dim=train_args.get("hidden_dim", 256),
        base_channels=train_args.get("base_channels", 128),
        min_channels=train_args.get("min_channels", 32),
        start_shape=tuple(train_args.get("start_shape", [4, 4])),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    target_ffis = resolve_ffi_nums(args.ffi_nums) if args.ffi_nums else None
    if target_ffis:
        print(f"Target FFIs: {sorted(target_ffis)}")

    run_inference(
        model=model,
        loader=valid_loader,
        device=device,
        mean=train_args.get("mean", 0.0),
        std=train_args.get("std", 1.0),
        num_save_images=args.num_save_images,
        output_dir=output_dir,
        target_ffis=target_ffis,
    )

    print("Finished")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
