#!/usr/bin/env python3
"""Generate TESS clean-image products with the simple conditioning CNN.

This mirrors the observed-FITS workflow in scripts/generate.py, but replaces
the likelihood/Langevin sampler with one direct CNN prediction of scattered
light from the orbital angles.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.train_simple_cnn import SimpleConditioningCNN


DEFAULT_MODEL_DIR = "model_outputs/simple_cnn_20260414_005047"
DEFAULT_OBS_DIR = ["/pdo/users/roland/SL_data/orbit-67"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CNN-only scattered-light generation on observed TESS FITS files"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing run_args.json and best.pt/last.pt",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Explicit checkpoint path. If omitted, searches --model-dir for best.pt then last.pt.",
    )
    parser.add_argument(
        "--run-args-path",
        type=str,
        default=None,
        help="Path to run_args.json. Defaults to <model-dir>/run_args.json.",
    )
    parser.add_argument(
        "--angle-path",
        type=str,
        default=None,
        help="Angle dictionary path. Defaults to the path saved in the CNN training args.",
    )
    parser.add_argument(
        "--obs_dir",
        type=str,
        nargs="+",
        default=DEFAULT_OBS_DIR,
        help="One or more directories containing observed FITS files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to model_outputs/cnn_generation_<timestamp>.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to cuda:0 when available, otherwise cpu.",
    )
    parser.add_argument(
        "--ffi-nums",
        type=str,
        nargs="+",
        default=None,
        help="Only generate for these FFI numbers. If omitted, runs on all observed FITS files.",
    )
    parser.add_argument(
        "--camera-number",
        type=str,
        default=None,
        help="Camera number. Defaults to the value saved in the CNN training args.",
    )
    parser.add_argument(
        "--observed-scale",
        type=float,
        default=633118.0,
        help="Value used to normalize observed FITS pixels before subtraction.",
    )
    return parser.parse_args()


def get_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested if torch.cuda.is_available() or "cuda" not in requested else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:1")
    return torch.device("cpu")


def resolve_checkpoint(model_dir: str, model_path: Optional[str]) -> str:
    if model_path is not None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        return model_path

    candidates = [
        os.path.join(model_dir, "best.pt"),
        os.path.join(model_dir, "last.pt"),
        os.path.join(model_dir, "checkpoint.pt"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "No CNN checkpoint found. Expected one of "
        f"{', '.join(candidates)}, or pass --model-path explicitly."
    )


def load_train_args(run_args_path: str, checkpoint: Dict) -> Dict:
    if "args" in checkpoint:
        return checkpoint["args"]
    if os.path.isfile(run_args_path):
        with open(run_args_path, "r") as handle:
            return json.load(handle)
    raise FileNotFoundError(
        f"No training args found in checkpoint and run_args.json is missing: {run_args_path}"
    )


def load_angles(angle_path: str, camera_number: str) -> Dict[str, torch.Tensor]:
    with open(angle_path, "rb") as handle:
        angles_dic = pickle.load(handle)

    ffi_to_angles: Dict[str, torch.Tensor] = {}
    for ffi_num, angles in angles_dic.items():
        params = np.array(
            [
                angles["1/ED"],
                angles["1/MD"],
                angles["1/ED^2"],
                angles["1/MD^2"],
                angles["Eel"],
                angles["Eaz"],
                angles["Mel"],
                angles["Maz"],
                angles["E" + camera_number + "el"],
                angles["E" + camera_number + "az"],
                angles["M" + camera_number + "el"],
                angles["M" + camera_number + "az"],
            ],
            dtype=np.float32,
        )
        ffi_to_angles[str(ffi_num)] = torch.from_numpy(params).view(1, 12)
    return ffi_to_angles


def find_observations(
    obs_dirs: Sequence[str], camera: str, requested: Optional[Set[str]]
) -> Sequence[Tuple[str, str]]:
    obs_entries = []
    for obs_dir in obs_dirs:
        if not os.path.isdir(obs_dir):
            print(f"WARNING: obs_dir does not exist, skipping: {obs_dir}")
            continue
        for fname in os.listdir(obs_dir):
            if not fname.endswith(".fits.gz"):
                continue
            parts = fname.split("-")
            if len(parts) < 3 or parts[2] != camera:
                continue
            if requested is not None and fname[18:26] not in requested:
                continue
            obs_entries.append((obs_dir, fname))

    obs_entries.sort(key=lambda item: item[1])
    return obs_entries


def warn_missing_requested(requested: Optional[Set[str]], obs_entries: Iterable[Tuple[str, str]]) -> None:
    if requested is None:
        return
    found = {fname[18:26] for _, fname in obs_entries}
    missing = requested - found
    if missing:
        print(f"WARNING: requested FFIs not found in observations: {sorted(missing)}")


def load_observed_fits(path: str, observed_scale: float) -> np.ndarray:
    columns_to_delete = (
        list(range(0, 44)) + list(range(2092, 2180)) + list(range(4228, 4272))
    )
    rows_to_delete = range(2048, 2108)

    with fits.open(path) as hdul:
        image = hdul[0].data / observed_scale
    image = np.delete(image, columns_to_delete, axis=1)
    image = np.delete(image, rows_to_delete, axis=0)
    return np.ascontiguousarray(image, dtype=np.float32)


def denormalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return x * std + mean


def save_image_grid(
    panels: Sequence[Tuple[str, np.ndarray]],
    output_path: str,
    title: str,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for ax, (panel_title, image) in zip(np.atleast_1d(axes), panels):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(panel_title)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_model(train_args: Dict, checkpoint: Dict, device: torch.device) -> SimpleConditioningCNN:
    output_shape = tuple(train_args["image_shape"])
    model = SimpleConditioningCNN(
        cond_dim=12,
        output_shape=output_shape,
        hidden_dim=train_args.get("hidden_dim", 256),
        base_channels=train_args.get("base_channels", 128),
        min_channels=train_args.get("min_channels", 32),
        start_shape=tuple(train_args.get("start_shape", [4, 4])),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_light_fullres(
    model: torch.nn.Module,
    conditioning: torch.Tensor,
    output_shape: Tuple[int, int],
    mean: float,
    std: float,
    device: torch.device,
) -> torch.Tensor:
    pred = model(conditioning.to(device=device, dtype=torch.float32))
    pred = denormalize(pred, mean, std)
    if pred.shape[-2:] != output_shape:
        pred = F.interpolate(pred, size=output_shape, mode="bilinear", align_corners=False)
    return pred


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    model_dir = args.model_dir
    run_args_path = args.run_args_path or os.path.join(model_dir, "run_args.json")
    model_path = resolve_checkpoint(model_dir, args.model_path)

    checkpoint = torch.load(model_path, map_location=device)
    train_args = load_train_args(run_args_path, checkpoint)
    camera = args.camera_number or train_args.get("camera_number", "3")
    angle_path = args.angle_path or train_args["angle_path"]
    output_dir = args.output_dir or os.path.join(
        "model_outputs", f"cnn_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    cnn_dir = os.path.join(output_dir, "cnn")
    os.makedirs(cnn_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loaded CNN checkpoint: {model_path}")
    print(f"Loading angles: {angle_path}")

    model = build_model(train_args, checkpoint, device)
    ffi_to_angles = load_angles(angle_path, camera)

    requested = set(args.ffi_nums) if args.ffi_nums else None
    obs_entries = find_observations(args.obs_dir, camera, requested)
    warn_missing_requested(requested, obs_entries)
    print(f"Found {len(obs_entries)} observation FITS files across {args.obs_dir}")

    mean = train_args.get("mean", 0.0)
    std = train_args.get("std", 1.0)

    for obs_dir, obs_file in tqdm(obs_entries, desc="Processing FITS", unit="file"):
        ffi_num = obs_file[18:26]
        basename = obs_file.replace(".fits.gz", "")
        print(f"\n{'=' * 60}")
        print(f"Processing {obs_file}  (ffi={ffi_num}) from {obs_dir}")

        if ffi_num not in ffi_to_angles:
            print(f"  WARNING: no angles for ffi={ffi_num}, skipping")
            continue

        x_obs_np = load_observed_fits(os.path.join(obs_dir, obs_file), args.observed_scale)
        x_obs = torch.from_numpy(x_obs_np).to(device).view(1, 1, *x_obs_np.shape)

        conditioning = ffi_to_angles[ffi_num].unsqueeze(0)
        light_full = predict_light_fullres(
            model=model,
            conditioning=conditioning,
            output_shape=x_obs_np.shape,
            mean=mean,
            std=std,
            device=device,
        )
        corrected = (x_obs - light_full)[0, 0].cpu().numpy()
        light_np = light_full[0, 0].cpu().numpy()
        corrected_save = corrected * args.observed_scale
        light_save = light_np * args.observed_scale

        light_fits_path = os.path.join(cnn_dir, f"{basename}_cnn_scattered_light.fits")
        fits.writeto(light_fits_path, light_save.astype(np.float32), overwrite=True)
        print(f"  Saved {light_fits_path}")

        clean_fits_path = os.path.join(cnn_dir, f"{basename}_cnn.fits")
        fits.writeto(clean_fits_path, corrected_save.astype(np.float32), overwrite=True)
        print(f"  Saved {clean_fits_path}")

        comparison_path = os.path.join(output_dir, f"{basename}_cnn_comparison.png")
        save_image_grid(
            panels=[("Observed", x_obs_np), ("CNN Clean", corrected)],
            output_path=comparison_path,
            title=obs_file,
        )
        print(f"  Saved {comparison_path}")

        light_png = os.path.join(output_dir, f"{basename}_cnn_light_stages.png")
        save_image_grid(
            panels=[("CNN Scattered Light", light_np)],
            output_path=light_png,
            title=f"Light Estimate - {obs_file}",
        )
        print(f"  Saved {light_png}")

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
