#!/usr/bin/env python
"""
BEAM: Background Elimination with Advanced Machine learning - Generation Script

Compares Likelihood Langevin vs Posterior Sampler on observed TESS images.
For each observation, runs both methods and saves the clean (star) image
as a .fits file along with a side-by-side comparison .png.
"""

import os
import re
import pickle
import argparse

from astropy.io import fits
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from beam.models.simulator import LikelihoodLangevin
from beam.models.unet import ContextUnet
from beam.models.probabilitypath import GaussianProbabilityPath, OTAlpha, OTBeta, VPBeta
from beam.models.interpolant import ScoreMatch, EMA
from beam.utils.config import load_config, flatten_config
from beam.data.datasets import TESSDataset_angles_only
from torch.utils.data import DataLoader


_TICA_RE = re.compile(
    r"hlsp_tica_tess_ffi_s(\d+)-(?:o\d+-)?(\d{8})-cam(\d)-ccd(\d)"
)


def parse_tica_filename(fits_filename):
    """Returns (ffi_num, sector, camera, ccd) or None if no TICA pattern match."""
    match = _TICA_RE.search(fits_filename)
    if not match:
        return None
    sector, ffi_num, camera, ccd = match.groups()
    return ffi_num, int(sector), int(camera), int(ccd)


def load_tica_observation(sector_dir, ccd1_filename, camera,
                          rows_to_delete, columns_to_delete):
    """Load all 4 TICA CCDs for a given FFI from <sector_dir>/cam{N}-ccd{1..4}/
    subfolders, stitch them following preprocess_all.py, then trim to 4096x4096.

    Returns a (4096, 4096) float32 array (un-normalized; caller divides by 633118).
    """
    parsed = parse_tica_filename(ccd1_filename)
    if parsed is None:
        raise ValueError(f"Filename does not match TICA pattern: {ccd1_filename}")
    ffi_num, _, parsed_cam, parsed_ccd = parsed
    if int(parsed_cam) != int(camera):
        raise ValueError(
            f"Camera mismatch: filename has cam{parsed_cam}, expected cam{camera}"
        )

    ccd_arrays = []
    for ccd in range(1, 5):
        ccd_folder = os.path.join(sector_dir, f"cam{camera}-ccd{ccd}")
        ccd_filename = ccd1_filename.replace(
            f"-cam{camera}-ccd{parsed_ccd}_",
            f"-cam{camera}-ccd{ccd}_",
        )
        ccd_path = os.path.join(ccd_folder, ccd_filename)
        if not os.path.exists(ccd_path):
            if not os.path.isdir(ccd_folder):
                raise FileNotFoundError(f"Missing CCD folder: {ccd_folder}")
            matches = [
                name
                for name in os.listdir(ccd_folder)
                if f"-{ffi_num}-cam{camera}-ccd{ccd}_" in name
                and name.endswith((".fits", ".fits.gz"))
            ]
            if not matches:
                raise FileNotFoundError(
                    f"No CCD {ccd} FITS for FFI {ffi_num} in {ccd_folder}"
                )
            ccd_path = os.path.join(ccd_folder, sorted(matches)[0])
        ccd_arrays.append(fits.getdata(ccd_path, ext=0))

    stitched = np.block([
        [ccd_arrays[2], ccd_arrays[3]],
        [np.flip(ccd_arrays[1]), np.flip(ccd_arrays[0])],
    ])

    if stitched.shape[0] - len(rows_to_delete) == 4096:
        stitched = np.delete(stitched, rows_to_delete, axis=0)
    if stitched.shape[1] - len(columns_to_delete) == 4096:
        stitched = np.delete(stitched, columns_to_delete, axis=1)

    if stitched.shape != (4096, 4096):
        raise ValueError(
            f"Stitched+trimmed shape {stitched.shape} != (4096, 4096) for FFI {ffi_num}"
        )
    return stitched.astype(np.float32)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Likelihood Langevin vs Posterior Sampler"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/generation_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--obs_dir",
        type=str,
        nargs="+",
        default=["/pdo/qlp-data/tica-delivery/s0030"],
        help="One or more TICA sector folders, each containing "
             "cam{N}-ccd{1..4} subfolders with per-CCD FITS files. The 4 CCDs "
             "are stitched per FFI to form a 4096x4096 observation.",
    )
    parser.add_argument(
        "--init_dir",
        type=str,
        nargs="+",
        default=None,
        help="One or more directories containing preprocessed .pkl light inits "
             "(defaults to config data.ccd_folder)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to config generation.output_dir)",
    )
    parser.add_argument(
        "--num_sde_steps",
        type=int,
        default=500,
        help="Number of Euler-Maruyama steps for the prior pass",
    )
    parser.add_argument(
        "--num_corrections",
        type=int,
        default=3,
        help="Number of posterior Langevin corrections",
    )
    parser.add_argument(
        "--corr_step_size",
        type=float,
        default=0.01,
        help="Step size for posterior Langevin corrections",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1"

    )
    parser.add_argument(
        "--ffi-nums",
        type=str,
        nargs="+",
        default=None,
        help="Only generate for these FFI numbers (e.g. 00034520 00033682). "
             "If omitted, runs on all observation FITS files in --obs_dir.",
    )
    return parser.parse_args()


def load_model(model_path, config, device, train_loader, context_len=None):
    """
    Load a trained model from checkpoint.

    Builds a ContextUnet + ScoreMatch wrapper, loads weights (optionally EMA),
    and returns the full ScoreMatch model.
    """
    batch = next(iter(train_loader))
    unet = ContextUnet(
        in_feats=1,
        context_len=batch["x"].shape[2] if context_len is None else context_len,
        n_feat=config["model_n_feat"],
        channel_mults=config["model_channel_mults"],
        heads_at=config["model_heads_at"],
        num_res=config["model_num_res"],
        time_dim=64,
        context_dim=64,
    )

    model = ScoreMatch(
        nn_model=unet,
        probability_path=GaussianProbabilityPath(
            alpha=OTAlpha(),
            beta=OTBeta() if config["model_architecture"] == "flow" else VPBeta(),
        ),
        device=device,
        architecture=config["model_architecture"],
    )

    ema = EMA(model=model)
    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if config["generation_ema"]:
        ema.load_state_dict(checkpoint["ema_state_dict"])
        ema.copy_to(model.nn_model)

    print(f"Loaded model from {model_path}")
    return model


def main():
    args = parse_args()
    config = load_config(args.config)
    config = flatten_config(config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    STARMEAN = config["generation_star_mean"]
    STARSTD = config["generation_star_std"]
    LIGHTMEAN = config["generation_light_mean"]
    LIGHTSTD = config["generation_light_std"]

    num_langevin_steps = config.get("generation_num_langevin_steps", 100)
    langevin_step_size = config["generation_langevin_step_size"]
    guidance_scale = config.get("generation_guidance_scale", 2.0)
    epsilon = config.get("model_epsilon", 1e-4)

  
    num_avg_snapshots = int(config.get("generation_num_avg_snapshots", 1))
    avg_burn_in = int(config.get("generation_avg_burn_in", 0))

    lazy_start_step = config.get("generation_lazy_start_step", None)
    if lazy_start_step is not None:
        lazy_start_step = int(lazy_start_step)
    lazy_interval = int(config.get("generation_lazy_interval", 1))

    if args.init_dir:
        init_dirs = args.init_dir if isinstance(args.init_dir, list) else [args.init_dir]
    else:
        cfg_ccd = config["data_ccd_folder"]
        init_dirs = cfg_ccd if isinstance(cfg_ccd, list) else [cfg_ccd]
    output_dir = args.output_dir or config["generation_output_dir"]

    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir = os.path.join(output_dir, timestamp)
    ll_dir = os.path.join(output_dir, "likelihood_langevin")
    os.makedirs(ll_dir, exist_ok=True)

    # Build a minimal loader so load_model can infer conditioning shape
    angles_ds = TESSDataset_angles_only(
        angle_path=config["data_angle_path"],
        camera_number=config.get("data_camera_number", "3"),
    )
    train_loader = DataLoader(angles_ds, batch_size=1, shuffle=False)

    # Build angles lookup
    ffi_to_angles = {}
    for i in range(len(angles_ds)):
        sample = angles_ds[i]
        ffi_to_angles[angles_ds.ffi_nums[i]] = sample["x"]

    star_model = load_model(
        config["generation_model_path_star"],
        config,
        device,
        train_loader,
        context_len=12,
    )

    star_unet = star_model.nn_model.to(device).eval()
    # light_unet = light_model.nn_model.to(device).eval()

    columns_to_delete = (
        list(range(0, 44)) + list(range(2092, 2180)) + list(range(4228, 4272))
    )
    rows_to_delete = range(2048, 2108)

    camera = config.get("data_camera_number", "3")

    obs_dirs = args.obs_dir if isinstance(args.obs_dir, list) else [args.obs_dir]

    obs_entries = []  # list of (sector_dir, ccd1_filename, ffi_num)
    for d in obs_dirs:
        if not os.path.isdir(d):
            print(f"WARNING: obs_dir does not exist, skipping: {d}")
            continue
        ccd1_dir = os.path.join(d, f"cam{camera}-ccd1")
        if not os.path.isdir(ccd1_dir):
            print(f"WARNING: missing cam{camera}-ccd1 subfolder, skipping: {d}")
            continue
        for f in os.listdir(ccd1_dir):
            if not f.endswith((".fits", ".fits.gz")):
                continue
            parsed = parse_tica_filename(f)
            if parsed is None:
                continue
            ffi_num, _, parsed_cam, _ = parsed
            if int(parsed_cam) != int(camera):
                continue
            obs_entries.append((d, f, ffi_num))

    if args.ffi_nums:
        requested = set(args.ffi_nums)
    else:
        requested = set()
        for d in init_dirs:
            if not os.path.isdir(d):
                print(f"WARNING: init_dir does not exist, skipping: {d}")
                continue
            for f in os.listdir(d):
                parsed = parse_tica_filename(f)
                if parsed is not None:
                    requested.add(parsed[0])
        print(f"No --ffi-nums given; using {len(requested)} FFIs from {init_dirs}")

    obs_entries = [e for e in obs_entries if e[2] in requested]
    found = {ffi for (_, _, ffi) in obs_entries}
    missing = requested - found
    if missing:
        print(f"WARNING: requested FFIs not found in {obs_dirs}: {sorted(missing)}")

    obs_entries.sort(key=lambda t: (t[2], t[1]))
    num_obs_files = len(obs_entries)
    print(f"Found {num_obs_files} observation FFIs across {obs_dirs}")

    for sector_dir, ccd1_filename, ffi_num in obs_entries:
        basename = ccd1_filename.replace(
            f"-cam{camera}-ccd1_", f"-cam{camera}-ccdALL_"
        )
        if basename.endswith(".fits.gz"):
            basename = basename[:-len(".fits.gz")]
        elif basename.endswith(".fits"):
            basename = basename[:-len(".fits")]

        print(f"\n{'='*60}")
        print(f"Processing FFI {ffi_num} (4 quads stitched) from {sector_dir}")

        # stitching 
        x_obs_np = load_tica_observation(
            sector_dir,
            ccd1_filename,
            camera=camera,
            rows_to_delete=rows_to_delete,
            columns_to_delete=columns_to_delete,
        ) / 633118
        x_obs_np = np.ascontiguousarray(x_obs_np, dtype=np.float32)
        x_obs = torch.from_numpy(x_obs_np).to(device).view(1, 1, 4096, 4096)

        # Likelihood Only -- search init across all init_dirs
        init_path = None
        for d in init_dirs:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                parsed = parse_tica_filename(f)
                if parsed is not None and parsed[0] == ffi_num:
                    init_path = os.path.join(d, f)
                    break
            if init_path is not None:
                break
        if init_path is None:
            print(f"  WARNING: no init found for ffi={ffi_num} in {init_dirs}, skipping")
            continue
        ll_init_light = None
        print("  Running Likelihood Langevin ...")
        with open(init_path, "rb") as f:
            init_np = pickle.load(f)
        init_np = np.ascontiguousarray(init_np, dtype=np.float32)
        X0 = torch.from_numpy(init_np).to(device).view(1, 1, 4096, 4096)

        ll_init_light = X0[0, 0].cpu().numpy()

        ll_sampler = LikelihoodLangevin(
            star_score=star_unet,
            x_obs=x_obs,
            sigma=VPBeta(),
            star_mean=STARMEAN,
            star_std=STARSTD,
            tile_size=256,
            num_steps=num_langevin_steps,
            step_size=langevin_step_size,
            num_save=num_avg_snapshots,
            star_tile_microbatch=160,
            context_len=12,
            save_start_step=avg_burn_in,
            lazy_start_step=lazy_start_step,
            lazy_interval=lazy_interval,
        )
        with torch.no_grad():
            Xs_saved, save_indices = ll_sampler.simulate(X0)

        # Average the snapshots collected along a single chain.
        ll_avg = Xs_saved.mean(dim=0)
        print(
            f"  Averaged {Xs_saved.shape[0]} snapshots at steps {save_indices}"
        )

        ll_corrected = (x_obs - ll_avg)[0, 0].cpu().numpy()
        ll_corrected_light = ll_avg[0, 0].cpu().numpy()

        ll_light_fits_path = os.path.join(
            ll_dir, f"{basename}_likelihood_scattered_light.fits"
        )
        fits.writeto(
            ll_light_fits_path, ll_corrected_light.astype(np.float32), overwrite=True
        )
        print(f"  Saved {ll_light_fits_path}")

        ll_fits_path = os.path.join(ll_dir, f"{basename}_likelihood.fits")
        fits.writeto(ll_fits_path, ll_corrected.astype(np.float32), overwrite=True)
        print(f"  Saved {ll_fits_path}")
       

        # Save comparison png
        panels = [("Observed", x_obs_np)]
        panels.append(("Preprocessed Method", x_obs_np - ll_init_light))
        if ll_corrected is not None:
            panels.append(("Likelihood Langevin", ll_corrected))
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
        for ax, (title, img) in zip(np.atleast_1d(axes), panels):
            ax.imshow(img, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"FFI {ffi_num} (cam{camera}, stitched)", fontsize=12)
        plt.tight_layout()
        png_path = os.path.join(output_dir, f"{basename}_comparison.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {png_path}")

        # Save 4-panel light estimate comparison
        light_panels = []
        if ll_init_light is not None:
            light_panels.append(("Preprocessed Init", ll_init_light))
        if ll_corrected_light is not None:
            light_panels.append(("After Likelihood Langevin", ll_corrected_light))
        # if ps_pre_light is not None:
        #     light_panels.append(("Prior SDE Output", ps_pre_light))
        # if ps_corrected_light is not None:
        #     light_panels.append(("After Posterior Correction", ps_corrected_light))

        if light_panels:
            fig, axes = plt.subplots(
                1, len(light_panels), figsize=(5 * len(light_panels), 5)
            )
            for ax, (title, img) in zip(np.atleast_1d(axes), light_panels):
                ax.imshow(img, cmap="viridis", vmin=0, vmax=1)
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle(
                f"Light Estimates — FFI {ffi_num} (cam{camera}, stitched)",
                fontsize=12,
            )
            plt.tight_layout()
            light_png = os.path.join(output_dir, f"{basename}_light_stages.png")
            fig.savefig(light_png, dpi=150)
            plt.close(fig)
            print(f"  Saved {light_png}")

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
