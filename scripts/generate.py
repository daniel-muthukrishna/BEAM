#!/usr/bin/env python
"""Generate scattered-light images from a trained BEAM model.

Usage:
    python scripts/generate.py --config configs/generation_config.yaml --sector 10

Every setting has exactly one source:
  - the generation config (--config): model folder, angle file, output folder,
    sampler and device settings. Every key in GENERATION_KEYS is required.
  - the command line: which FFIs to generate (--sector or --ffi-nums, --limit)
    and sharding (--num-shards, --shard-index).
  - the model folder: architecture and image settings from its config.yaml
    (REQUIRED_CONFIG_KEYS and OPTIONAL_CONFIG_KEYS only), the weights and the
    training statistics.

A model folder always holds the files named in MODEL_FILES, under exactly those
names. The folder itself can be named anything and unpacked anywhere.
"""
import os
import pickle
import argparse

import yaml

from astropy.io import fits
import numpy as np
import torch
import torch.multiprocessing as mp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# This script must mirror beam/training/trainer.py's model
# construction exactly or the checkpoint will not load / the encoding drifts.
from beam.models.unet import ContextUnet
from beam.models.interpolant import ScoreMatch, EMA
from beam.models.probabilitypath import GaussianProbabilityPath, OTAlpha, OTBeta, VPBeta
from beam.models.simulator import ODE, SDE, ODEIntegrator, EulerMaruyama
from beam.utils.config import load_config, flatten_config
from beam.data.datasets import TESSDataset_angles_only

# Training files are (FFI - background) / 633118, scaled to a 30-minute exposure
# (1425.6 s effective), so this one factor converts them to e-/s for every cadence.
E_PER_S_PER_UNIT = 633118 / 1425.6

# The only keys read from the model config. All are required; none has a default,
# so a missing key fails loudly instead of silently building the wrong model.
REQUIRED_CONFIG_KEYS = [
    "model_architecture",
    "model_n_feat",
    "model_channel_mults",
    "model_heads_at",
    "model_num_res",
    "model_time_dim",
    "model_context_dim",
    "model_emb_dim",
    "model_drop_prob",
    "model_epsilon",
    "data_image_shape",
    "data_camera_number",
]

# Optional architecture keys and their defaults. Cross-attention is off in this
# version of the model, so a config may leave it out.
OPTIONAL_CONFIG_KEYS = {
    "model_use_cross_attn": False,
}

# Fixed file names inside every model folder. scripts/clean.py uses these too.
MODEL_FILES = {
    "config": "config.yaml",
    "checkpoint": "checkpoint_epoch_final.pth",
    "context_stats": "context_stats.pkl",
    "image_stats": "image_stats.pkl",
}


def model_files(model_dir, needed=tuple(MODEL_FILES)):
    """Paths of the fixed-name files in model_dir. Raises if the folder or any needed file is missing."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model folder not found: {model_dir}")
    paths = {k: os.path.join(model_dir, MODEL_FILES[k]) for k in needed}
    missing = [MODEL_FILES[k] for k, p in paths.items() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"{model_dir} is missing {missing}; a model folder must contain "
                                f"{[MODEL_FILES[k] for k in needed]}")
    return paths


def sector_to_orbits(sector: int) -> set:
    """sector 1 = orbits 9,10"""
    return {2 * sector + 7, 2 * sector + 8}


def orbit_to_sector(orbit: int) -> int:
    return (int(orbit) - 7) // 2


# Every key of the generation config. All are required and no others are allowed,
# so a typo or a missing setting fails loudly instead of falling back to a default.
GENERATION_KEYS = {
    "model_dir": "folder holding " + ", ".join(MODEL_FILES.values()),
    "angle_path": "Earth/Moon angle file (.pkl)",
    "output_dir": "folder for the generated FITS files",
    "ema": "true = sample with EMA weights",
    "guidance_scale": "classifier-free guidance scale",
    "num_sde_steps": "Euler-Maruyama steps (SDE architectures only; 'flow' uses adaptive dopri5)",
    "batch_size": "FFIs per forward pass, per device",
    "devices": "list of devices, one worker each, e.g. [cuda:0, cuda:1] or [cpu]",
    "seed": "seed for the initial noise; null = nondeterministic",
}


def load_generation_config(path):
    """Read and check the generation config. Returns its settings as a dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if set(cfg) != {"generation"}:
        raise ValueError(f"{path} must have exactly one top-level section, 'generation:'")
    gen = cfg["generation"] or {}
    missing = [k for k in GENERATION_KEYS if k not in gen]
    unknown = [k for k in gen if k not in GENERATION_KEYS]
    if missing or unknown:
        raise KeyError(f"{path}: missing keys {missing}, unknown keys {unknown}. "
                       f"Expected exactly: {list(GENERATION_KEYS)}")

    def check(ok, key, what):
        if not ok:
            raise TypeError(f"{path}: generation.{key} must be {what}, got {gen[key]!r}")

    for key in ("model_dir", "angle_path", "output_dir"):
        check(isinstance(gen[key], str) and gen[key], key, "a path")
        gen[key] = os.path.expanduser(gen[key])
    check(isinstance(gen["ema"], bool), "ema", "true or false")
    check(isinstance(gen["guidance_scale"], (int, float)) and not isinstance(gen["guidance_scale"], bool),
          "guidance_scale", "a number")
    for key in ("num_sde_steps", "batch_size"):
        check(isinstance(gen[key], int) and not isinstance(gen[key], bool) and gen[key] > 0,
              key, "a positive integer")
    check(isinstance(gen["devices"], list) and gen["devices"] and all(isinstance(d, str) for d in gen["devices"]),
          "devices", "a non-empty list such as [cuda:0]")
    check(gen["seed"] is None or (isinstance(gen["seed"], int) and not isinstance(gen["seed"], bool)),
          "seed", "an integer or null")
    return gen


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate scattered-light images from a trained BEAM model. "
                    "Paths, sampler and device settings live in the generation config; "
                    "the command line picks which FFIs to generate."
    )
    parser.add_argument("--config", required=True, help="generation config, e.g. configs/generation_config.yaml")

    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument("--sector", type=int, help="Generate every FFI of this sector in the angle file")
    select.add_argument("--ffi-nums", type=str, nargs="+", help="Generate these FFIs")
    parser.add_argument("--limit", type=int, default=None, help="Only generate the first N selected FFIs")
    parser.add_argument("--num-shards", type=int, default=1, help="Split the selected FFIs into this many jobs")
    parser.add_argument("--shard-index", type=int, default=0, help="Which of those jobs this run is (0-based)")
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")
    try:
        gen = load_generation_config(args.config)
    except (OSError, ValueError, KeyError, TypeError) as e:
        parser.error(f"--config: {e}")
    try:
        files = model_files(gen["model_dir"])
    except FileNotFoundError as e:
        parser.error(f"generation.model_dir: {e}")
    if not os.path.isfile(gen["angle_path"]):
        parser.error(f"generation.angle_path: file not found: {gen['angle_path']}")
    for d in gen["devices"]:
        if d.startswith("cuda") and not torch.cuda.is_available():
            parser.error(f"generation.devices has {d} but CUDA is not available; set devices: [cpu]")

    # One flat namespace for the rest of the script; each value came from exactly one place.
    args.files = files
    args.model_dir = gen["model_dir"]
    args.angles = gen["angle_path"]
    args.output_dir = gen["output_dir"]
    args.use_ema = gen["ema"]
    args.guidance_scale = float(gen["guidance_scale"])
    args.num_steps = gen["num_sde_steps"]
    args.batch_size = gen["batch_size"]
    args.devices = list(gen["devices"])
    args.seed = gen["seed"]
    return args


def load_model_config(path):
    config = flatten_config(load_config(path))
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")
    # Keep only the architecture keys so nothing else in the file can be used.
    arch = {k: config[k] for k in REQUIRED_CONFIG_KEYS}
    for k, default in OPTIONAL_CONFIG_KEYS.items():
        arch[k] = config.get(k, default)
    return arch


def load_model(checkpoint_path, config, context_len, device, use_ema=True):
    unet = ContextUnet(
        in_feats=1,
        context_len=context_len,
        n_feat=config["model_n_feat"],
        channel_mults=config["model_channel_mults"],
        heads_at=config["model_heads_at"],
        time_dim=config["model_time_dim"],
        context_dim=config["model_context_dim"],
        emb_dim=config["model_emb_dim"],
        num_res=config["model_num_res"],
        # .get so scripts/clean.py, which passes its own config, gets the same default
        use_cross_attn=config.get("model_use_cross_attn", OPTIONAL_CONFIG_KEYS["model_use_cross_attn"]),
    )
    # Two schedules hardcoded
    model = ScoreMatch(
        nn_model=unet,
        probability_path=GaussianProbabilityPath(
            alpha=OTAlpha(),
            beta=OTBeta() if config["model_architecture"] == "flow" else VPBeta(),
        ),
        device=device,
        drop_prob=config["model_drop_prob"],
        epsilon=config["model_epsilon"],
        architecture=config["model_architecture"],
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if use_ema:
        if "ema_state_dict" not in checkpoint:
            raise KeyError(f"No ema_state_dict in {checkpoint_path}; rerun with --no-ema")
        ema = EMA(model=model)
        ema.load_state_dict(checkpoint["ema_state_dict"])
        ema.to(device)
        ema.copy_to(model)
        print(f"Loaded EMA weights from {checkpoint_path}")
    else:
        print(f"Loaded raw weights from {checkpoint_path}")

    model.eval()
    return model


def build_simulator(model, guidance_scale, num_steps, epsilon, device, cam=None):
    # cam: (B,) 0-based camera indices, one per sample in the batch
    if model.architecture == "flow":
        t = torch.linspace(epsilon, 1.0 - epsilon, 2, device=device).unsqueeze(0)
        return ODEIntegrator(ODE(model.nn_model, guidance_scale, cam=cam), t, num_save=1)
    t = torch.linspace(epsilon, 1.0 - epsilon, num_steps, device=device).unsqueeze(0)
    sde = SDE(
        nn_model=model.nn_model,
        probability_path=model.probability_path,
        guidance_value=guidance_scale,
        architecture=model.architecture,
        cam=cam,
    )
    return EulerMaruyama(sde, t, num_save=1)


def run_worker(worker_idx, run):
    entries = run["worker_entries"][worker_idx]
    if not entries:
        return
    device_str = run["devices"][worker_idx]
    args = run["args"]
    config = run["config"]

    device = torch.device(device_str)
    context_len = entries[0][2].shape[-1]
    model = load_model(args.files["checkpoint"], config, context_len, device, use_ema=args.use_ema)
    cam_idx = int(run["camera"]) - 1

    # record seed + make sure shards dont have same idx
    if args.seed is not None:
        torch.manual_seed(int(args.seed) + 1000 * args.shard_index + worker_idx)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    preview = []
    batches = range(0, len(entries), args.batch_size)
    for start in tqdm(batches, desc=f"[{device_str}] generating", position=worker_idx):
        chunk = entries[start : start + args.batch_size]
        c = torch.from_numpy(np.stack([ctx for (_, _, ctx) in chunk])).to(device)
        x0 = torch.randn(len(chunk), 1, *run["image_shape"], device=device)
        cam = torch.full((len(chunk),), cam_idx, dtype=torch.long, device=device)
        simulator = build_simulator(
            model, args.guidance_scale, args.num_steps, run["epsilon"], device, cam=cam
        )

        with torch.no_grad():
            xs, _ = simulator.simulate(x0, c)
        gen = xs[-1].float().cpu().numpy()[:, 0]  # (B, H, W)

        for i, (ffi, orbit, _) in enumerate(chunk):
            sector = orbit_to_sector(orbit)
            # denormalize, then convert to e-/s
            raw = (gen[i] * run["img_mean_std"][1] + run["img_mean_std"][0]) * E_PER_S_PER_UNIT

            header = fits.Header()
            header["FFI"] = ffi
            header["ORBIT"] = orbit
            header["SECTOR"] = sector
            header["CAMERA"] = run["camera"]
            header["ARCH"] = model.architecture
            header["GUIDANCE"] = args.guidance_scale
            header["NSTEPS"] = (
                args.num_steps if model.architecture != "flow" else -1,
                "-1 = adaptive dopri5",
            )
            header["IMGMEAN"] = run["img_mean_std"][0]
            header["IMGSTD"] = run["img_mean_std"][1]
            header["BUNIT"] = "electron/s"
            header["MODELDIR"] = os.path.basename(os.path.normpath(args.model_dir))
            header["EMA"] = args.use_ema

            fits_path = os.path.join(
                output_dir,
                f"generated_light_s{sector:04d}-{ffi}-cam{run['camera']}.fits",
            )
            fits.writeto(fits_path, raw.astype(np.float32), header, overwrite=True)

            if worker_idx == 0 and len(preview) < 16:
                preview.append((f"FFI {ffi} (o{orbit})", raw))

    if preview:
        ncols = min(4, len(preview))
        nrows = (len(preview) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        for ax, (title, img) in zip(np.atleast_1d(axes).ravel(), preview):
            ax.imshow(img, cmap="viridis", origin="lower")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        for ax in np.atleast_1d(axes).ravel()[len(preview) :]:
            ax.axis("off")
        label = f"sector {args.sector}" if args.sector is not None else "requested FFIs"
        fig.suptitle(f"Generated light — {label} (cam{run['camera']})", fontsize=12)
        plt.tight_layout()
        png_path = os.path.join(output_dir, "generation_preview.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"Saved preview {png_path}")

    print(f"[{device_str}] done: {len(entries)} generations")


def main():
    args = parse_args()

    config = load_model_config(args.files["config"])
    with open(args.files["context_stats"], "rb") as f:
        context_stats = pickle.load(f)
    with open(args.files["image_stats"], "rb") as f:
        image_stats = pickle.load(f)
    img_mean, img_std = float(image_stats["mean"]), float(image_stats["std"])
    print(f"Image stats: mean={img_mean:.6g}, std={img_std:.6g}")

    camera = str(config["data_camera_number"])
    image_shape = tuple(config["data_image_shape"])
    epsilon = float(config["model_epsilon"])

    angles_ds = TESSDataset_angles_only(
        angle_path=args.angles,
        camera_number=camera,
        context_stats=context_stats,
    )
    ffi_to_idx = {ffi: i for i, ffi in enumerate(angles_ds.ffi_nums)}
    if args.sector is not None:
        orbits = sector_to_orbits(args.sector)
        selected = [
            ffi
            for ffi in angles_ds.ffi_nums
            if int(angles_ds.angles_dic[ffi]["orbit"]) in orbits
        ]
        if not selected:
            raise ValueError(
                f"No FFIs for sector {args.sector} (orbits {sorted(orbits)}) in {args.angles}"
            )
        print(f"Sector {args.sector} (orbits {sorted(orbits)}): {len(selected)} FFIs")
    else:
        selected, missing = [], []
        for ffi in args.ffi_nums:
            (selected if ffi in ffi_to_idx else missing).append(ffi)
        if missing:
            print(f"WARNING: FFIs not in angle file, skipping: {missing}")
        if not selected:
            raise ValueError(f"None of the requested FFIs are in {args.angles}")
        print(f"Generating for {len(selected)} requested FFIs")

    selected = sorted(selected)
    if args.limit is not None:
        selected = selected[: args.limit]
        print(f"Limiting to first {len(selected)} FFIs")

    if args.num_shards > 1:
        selected = selected[args.shard_index :: args.num_shards]
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(selected)} FFIs")

    entries = [
        (
            ffi,
            int(angles_ds.angles_dic[ffi]["orbit"]),
            angles_ds[ffi_to_idx[ffi]]["x"].numpy(),
        )
        for ffi in selected
    ]
    devices = args.devices[: max(1, min(len(args.devices), len(entries)))]
    worker_entries = [entries[i :: len(devices)] for i in range(len(devices))]

    sampler_name = (
        "ODE (dopri5)"
        if config["model_architecture"] == "flow"
        else f"SDE (Euler-Maruyama, {args.num_steps} steps)"
    )
    print(
        f"Architecture '{config['model_architecture']}' -> {sampler_name}, "
        f"guidance={args.guidance_scale}"
    )
    print(
        f"Devices: {devices} "
        f"({', '.join(str(len(w)) for w in worker_entries)} FFIs per worker)"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    run = {
        "args": args,
        "config": config,
        "devices": devices,
        "worker_entries": worker_entries,
        "img_mean_std": (img_mean, img_std),
        "camera": camera,
        "image_shape": image_shape,
        "epsilon": epsilon,
    }

    if len(devices) == 1:
        run_worker(0, run)
    else:
        mp.spawn(run_worker, args=(run,), nprocs=len(devices), join=True)

    print(f"\n{len(entries)} generations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
