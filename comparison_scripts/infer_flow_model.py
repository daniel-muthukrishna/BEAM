#!/usr/bin/env python3
"""Run validation-set inference for a trained flow model.

This mirrors the CNN inference workflow:
- loads a trained flow checkpoint
- runs one generation per validation conditioning vector
- saves preview images and raw arrays
"""

from __future__ import annotations

import argparse
import json
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

from beam.data.datasets import TESSDataset, StarDataset, create_train_valid_datasets_by_orbit
from beam.models.interpolant import EMA, ScoreMatch
from beam.models.probabilitypath import GaussianProbabilityPath, OTAlpha, OTBeta, VPBeta
from beam.models.unet import ContextUnet
from beam.utils.config import flatten_config, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for a trained BEAM flow model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--config", type=str, default="/pdo/users/djtufto/BEAM/model_outputs/cam3_11-54/config.yaml", help="Training config used for the model")
    parser.add_argument("--device", type=str, default="cuda:2", help="Single device to use")
    parser.add_argument("--dataset-class", choices=["auto", "TESSDataset", "StarDataset"], default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=400)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--orbit-threshold", type=int, default=45)
    parser.add_argument("--orbit-max", type=int, default=54)
    parser.add_argument("--num-save-images", type=int, default=1)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--ffi-nums", type=str, nargs="+", default=['00034520', '00033682', '00033637'],
        help="Only generate for these FFI numbers. Alternatively pass a path "
             "to a directory of '<idx>_<ffi>.png' files to extract FFIs from.",
    )
    return parser.parse_args()


def resolve_ffi_nums(raw: List[str]) -> Set[str]:
    """Accept explicit FFI numbers or a single directory path containing
    ``<idx>_<ffi>.png`` files (the format produced by save_preview)."""
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


def build_dataset(config: Dict, dataset_class: str):
    resolved_class = dataset_class
    if resolved_class == "auto":
        resolved_class = "StarDataset" if config.get("data_orbit_skip") is not None else "TESSDataset"

    common_kwargs = dict(
        angle_path=config["data_angle_path"],
        ccd_folder=config["data_ccd_folder"],
        image_shape=tuple(config["data_image_shape"]),
        mean=config.get("data_mean", 0.0),
        std=config.get("data_std", 1.0),
        patch_size=config.get("data_patch_size"),
        repeat_factor=config.get("data_repeat_factor", 1),
        camera_number=config.get("data_camera_number", "3"),
    )

    if resolved_class == "StarDataset":
        dataset = StarDataset(
            **common_kwargs,
            orbit_skip=config.get("data_orbit_skip", []),
        )
    else:
        dataset = TESSDataset(**common_kwargs)

    train_dataset, valid_dataset = create_train_valid_datasets_by_orbit(
        dataset,
        orbit_threshold=45,
        max_orbit=54,
    )
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


def build_model(config: Dict, sample: Dict[str, torch.Tensor], device: torch.device) -> ScoreMatch:
    unet = ContextUnet(
        in_feats=1,
        context_len=sample["x"].shape[1],
        n_feat=config["model_n_feat"],
        channel_mults=config["model_channel_mults"],
        heads_at=config["model_heads_at"],
        time_dim=64,
        context_dim=64,
        num_res=config["model_num_res"],
    )

    model = ScoreMatch(
        nn_model=unet,
        probability_path=GaussianProbabilityPath(
            alpha=OTAlpha(),
            beta=OTBeta() if config["model_architecture"] == "flow" else VPBeta(),
        ),
        device=device,
        drop_prob=config.get("model_drop_prob", 0.1),
        epsilon=config.get("model_epsilon", 1e-4),
        architecture=config["model_architecture"],
    )
    return model.to(device)


def load_model(model: ScoreMatch, checkpoint_path: str, use_ema: bool, device: torch.device) -> ScoreMatch:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if use_ema and "ema_state_dict" in checkpoint:
        ema = EMA(model=model)
        ema.load_state_dict(checkpoint["ema_state_dict"])
        ema.copy_to(model.nn_model)

    model.eval()
    return model


def denormalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return x * std + mean


def save_preview(target: np.ndarray, pred: np.ndarray, output_path: str, title: str) -> None:
    residual = pred - target
    combined = np.concatenate([target.ravel(), pred.ravel()])
    vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(target, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Target")
    axes[1].imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Generation")
    axes[2].imshow(residual, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[2].set_title("Residual")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def run_inference(
    model: ScoreMatch,
    loader: DataLoader,
    device: torch.device,
    image_shape: Tuple[int, int],
    mean: float,
    std: float,
    num_steps: int,
    guidance_scale: float,
    epsilon: float,
    num_save_images: int,
    output_dir: str,
    target_ffis: Optional[Set[str]] = None,
) -> None:
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

        pred, _, _ = model.simulate(
            c_i=x,
            n_sample=1,
            size=image_shape,
            device=device,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            num_save=1,
            epsilon=epsilon,
        )

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
    raw_config = load_config(args.config)
    config = flatten_config(raw_config)

    if args.orbit_threshold is not None:
        config["data_orbit_threshold"] = args.orbit_threshold
    if args.orbit_max is not None:
        config["data_orbit_max"] = args.orbit_max

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or os.path.join(
        "model_outputs",
        "flow_inference_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "run_args.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)

    print("Using device: {}".format(device))
    _, valid_dataset = build_dataset(config, args.dataset_class)
    valid_loader = make_loader(valid_dataset, args.batch_size, False, args.num_workers)
    sample = valid_dataset[0]
    image_shape = tuple(sample["y"].shape[-2:])

    print("Validation samples: {}".format(len(valid_dataset)))
    print("Image shape: {}".format(image_shape))

    model = build_model(config, sample, device)
    model = load_model(model, args.model_path, args.use_ema, device)

    target_ffis = resolve_ffi_nums(args.ffi_nums) if args.ffi_nums else None
    if target_ffis:
        print("Target FFIs: {}".format(sorted(target_ffis)))

    run_inference(
        model=model,
        loader=valid_loader,
        device=device,
        image_shape=image_shape,
        mean=config.get("data_mean", 0.0),
        std=config.get("data_std", 1.0),
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        epsilon=args.epsilon if args.epsilon is not None else config.get("model_epsilon", 1e-4),
        num_save_images=args.num_save_images,
        output_dir=output_dir,
        target_ffis=target_ffis,
    )

    print("Finished")
    print("Outputs: {}".format(output_dir))


if __name__ == "__main__":
    main()
