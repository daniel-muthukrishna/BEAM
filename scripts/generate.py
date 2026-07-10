#!/usr/bin/env python
"""
BEAM: Generation Script
"""

import os
import re
import pickle
import argparse

from astropy.io import fits
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from beam.models.simulator import LikelihoodLangevin
from beam.models.unet import ContextUnet
from beam.models.probabilitypath import GaussianProbabilityPath, OTAlpha, OTBeta, VPBeta
from beam.models.interpolant import ScoreMatch, EMA
from beam.utils.config import load_config, flatten_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Likelihood Langevin vs Posterior Sampler"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/generation_config.yaml",
    )
    parser.add_argument(
        "--obs_dir",
        type=str,
        nargs="+",
        default=["/pdo/qlp-data/tica-delivery/s0030"],
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
        "--device",
        type=str,
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=1
    )

    parser.add_argument(
        "--shard-id",
        type=int,
        default=0
    )

    parser.add_argument(
        "--ffi-nums",
        type=str,
        nargs="+",
        default=None,
        help="Only generate for these FFI numbers. "
             "If omitted, runs on all observation FITS files in --obs_dir.",
    )
    return parser.parse_args()


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
    """Load all 4 TICA CCDs, stich and trim
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



def load_model(model_path, config, device, context_len=12):
    """
    Load a trained model from checkpoint.

    Builds a ContextUnet + ScoreMatch wrapper, loads weights (optionally EMA),
    and returns the full ScoreMatch model.
    """
    unet = ContextUnet(
        in_feats=1,
        context_len= context_len,
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


    num_langevin_steps = config.get("generation_num_langevin_steps", 100)
    langevin_step_size = config["generation_langevin_step_size"]

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
 
    star_model = load_model(
        config["generation_model_path_star"],
        config,
        device,
        context_len=12,
    )

    star_unet = star_model.nn_model.to(device).eval()

    columns_to_delete = (
        list(range(0, 44)) + list(range(2092, 2180)) + list(range(4228, 4272))
    )
    rows_to_delete = range(2048, 2108)

    camera = config.get("data_camera_number", "3")

    # raw tica data from tica_delivery
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
    obs_entries = obs_entries[args.shard_id::args.num_shards]
    num_obs_files = len(obs_entries)
    print(f"Shard {args.shard_id}: Found {num_obs_files} observation FFIs across {obs_dirs}")

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
        ) / 633118 / 5.3
        x_obs_np = np.ascontiguousarray(x_obs_np, dtype=np.float32)
        x_obs = torch.from_numpy(x_obs_np).to(device).view(1, 1, 4096, 4096)

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
            star_tile_microbatch=120,
            context_len=12,
        )
        with torch.no_grad():
            ll_light = ll_sampler.simulate(X0)

        clean_m = (x_obs - ll_light)[0,0].cpu().numpy()   # model scale for plotting
        light_m =  ll_light[0,0].cpu().numpy()

        # rescale to original units
        ll_corrected = clean_m * 633118 * 5.3
        ll_corrected_light = light_m * 633118 * 5.3 

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
        panels = [("Observed", x_obs_np), ("Preprocessed Method", x_obs_np - ll_init_light), ("Likelihood Langevin", clean_m)]
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

        # Save  light estimate comparison
        light_panels = [("Preprocessed Init", ll_init_light), ("After Likelihood Langevin", light_m)]

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
