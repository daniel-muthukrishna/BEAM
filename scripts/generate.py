#!/usr/bin/env python
"""
BEAM: Background Elimination with Advanced Machine learning - Generation Script

Compares Likelihood Langevin vs Posterior Sampler on observed TESS images.
For each observation, runs both methods and saves the clean (star) image
as a .fits file along with a side-by-side comparison .png.
"""

import os
import pickle
import argparse
import datetime

from astropy.io import fits
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from beam.models.simulator import LikelihoodLangevin, PosteriorSDE
from beam.models.unet import ContextUnet
from beam.models.probabilitypath import GaussianProbabilityPath, OTAlpha, OTBeta, VPBeta
from beam.models.interpolant import ScoreMatch, EMA
from beam.utils.config import load_config, flatten_config
from beam.data.datasets import TESSDataset_angles_only
from torch.utils.data import DataLoader


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
        default="/pdo/users/roland/SL_data/O16_data",
        help="Directory containing observed FITS files",
    )
    parser.add_argument(
        "--init_dir",
        type=str,
        default=None,
        help="Directory containing preprocessed .pkl light inits (defaults to config data.ccd_folder)",
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
        default="cuda:1",
        help="Device to use (e.g. cuda:0, cuda:1, cpu)",
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
    langevin_step_size = config.get("generation_langevin_step_size", 0.01)
    guidance_scale = config.get("generation_guidance_scale", 2.0)
    epsilon = config.get("model_epsilon", 1e-4)

    init_dir = args.init_dir or config["data_ccd_folder"]
    output_dir = args.output_dir or config["generation_output_dir"]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
    ll_dir = os.path.join(output_dir, "likelihood_langevin")
    ps_dir = os.path.join(output_dir, "posterior_sampler")
    os.makedirs(ll_dir, exist_ok=True)
    os.makedirs(ps_dir, exist_ok=True)

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

    # Load both models
    star_model = load_model(
        config["generation_model_path_star"],
        config,
        device,
        train_loader,
        context_len=24,
    )
    light_model = load_model(
        config["generation_model_path_light"], config, device, train_loader
    )
    star_unet = star_model.nn_model.to(device).eval()
    # light_unet = light_model.nn_model.to(device).eval()

    columns_to_delete = (
        list(range(0, 44)) + list(range(2092, 2180)) + list(range(4228, 4272))
    )
    rows_to_delete = range(2048, 2108)

    camera = config.get("data_camera_number", "3")
    
    obs_files = [f for f in os.listdir(args.obs_dir) if f.endswith(".fits.gz")]
    print(f"Found {len(obs_files)} observation FITS files in {args.obs_dir}")

    for obs_file in obs_files:
        for i in range(30):
            ll_avg = np.zeros((4096, 4096))
            ll_clean_avg = np.zeros((4096, 4096))
            ffi_num = obs_file[18:26]
            basename = obs_file.replace(".fits.gz", "")
            print(f"\n{'='*60}")
            print(f"Processing {obs_file}  (ffi={ffi_num})")

            # Load observed image
            with fits.open(os.path.join(args.obs_dir, obs_file)) as hdul:
                x_obs_np = hdul[0].data / 633118
            x_obs_np = np.delete(x_obs_np, columns_to_delete, axis=1)
            x_obs_np = np.delete(x_obs_np, rows_to_delete, axis=0)
            x_obs_np = np.ascontiguousarray(x_obs_np, dtype=np.float32)
            x_obs = torch.from_numpy(x_obs_np).to(device).view(1, 1, 4096, 4096)

            # Likelihood Only
            init_candidates = [
                f
                for f in os.listdir(init_dir)
                if f[18:26] == ffi_num and f.endswith(".pkl")
            ]
            init_path = (
                os.path.join(init_dir, init_candidates[0]) if init_candidates else None
            )
            ll_init_light = None
            ll_corrected_light = None
            if init_path is not None and os.path.exists(init_path):
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
                    num_save=1,
                    star_tile_microbatch=160,
                )
                with torch.no_grad():
                    Xs_saved, _ = ll_sampler.simulate(X0)

                X_final = Xs_saved[-1]  # (1, 1, 4096, 4096) light estimate
                ll_corrected_light = X_final[0, 0].cpu().numpy()
                ll_clean = (x_obs - X_final)[0, 0].cpu().numpy()

                ll_avg += ll_corrected_light
                ll_clean_avg += ll_clean

            ll_avg /= 30
            ll_clean_avg /= 30

            ll_light_fits_path = os.path.join(
                ll_dir, f"{basename}_likelihood_scattered_light.fits"
            )
            fits.writeto(
                ll_light_fits_path, ll_avg.astype(np.float32), overwrite=True
            )
            print(f"  Saved {ll_light_fits_path}")

            ll_fits_path = os.path.join(ll_dir, f"{basename}_likelihood.fits")
            fits.writeto(ll_fits_path, ll_clean_avg.astype(np.float32), overwrite=True)
            print(f"  Saved {ll_fits_path}")
        else:
            ll_clean = None
            print(
                f"  WARNING: no init found for ffi={ffi_num} in {init_dir}, skipping Likelihood Langevin"
            )

        # Posterior Sampler
        ps_pre_light = None
        ps_corrected_light = None
        if ffi_num not in ffi_to_angles:
            print(f"  WARNING: no angles for ffi={ffi_num}, skipping Posterior Sampler")
            ps_clean = None
        else:
            print("  Running Posterior Sampler ...")
            c_obs = ffi_to_angles[ffi_num].to(device).unsqueeze(0)  # (1, 1, 12)

            t = torch.linspace(epsilon, 1 - epsilon, args.num_sde_steps, device=device)
            t = t.unsqueeze(0).expand(1, args.num_sde_steps)

            posterior = PosteriorSDE(
                star_score=star_unet,
                light_score=light_unet,
                t=t,
                x_obs=x_obs,
                sigma=VPBeta(),
                guidance_value=guidance_scale,
                upscale_factor=16,
                tile_size=256,
                num_save=1,
                num_corrections=args.num_corrections,
                corr_step_size=args.corr_step_size,
                star_tile_microbatch=160,
            )

            y0 = torch.randn(1, 1, 256, 256, device=device)
            with torch.no_grad():
                y_saved, _, y_pre_correction = posterior.simulate(c=c_obs, y0=y0)

            # Pre-correction: light prior output (before posterior Langevin)
            y_pre_denorm = y_pre_correction * LIGHTSTD + LIGHTMEAN
            ps_pre_light = posterior.U(y_pre_denorm)[0, 0].cpu().numpy()

            # Post-correction: after posterior Langevin refinement
            y_final = y_saved[-1] * LIGHTSTD + LIGHTMEAN
            light_full = posterior.U(y_final)
            ps_corrected_light = light_full[0, 0].cpu().numpy()
            ps_clean = (x_obs - light_full)[0, 0].cpu().numpy()

            ps_light_fits_path = os.path.join(
                ps_dir, f"{basename}_posterior_scattered_light.fits"
            )
            fits.writeto(
                ps_light_fits_path, ps_corrected_light.astype(np.float32), overwrite=True
            )
            print(f"  Saved {ps_light_fits_path}")

            ps_fits_path = os.path.join(ps_dir, f"{basename}_posterior.fits")
            fits.writeto(ps_fits_path, ps_clean.astype(np.float32), overwrite=True)
            print(f"  Saved {ps_fits_path}")

        # Save comparison png
        panels = [("Observed", x_obs_np)]
        panels.append(("Preprocessed Image", x_obs_np - ll_init_light))
        if ll_clean is not None:
            panels.append(("Likelihood Langevin", ll_clean))
        if ps_clean is not None:
            panels.append(("Posterior Sampler", ps_clean))

        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
        for ax, (title, img) in zip(np.atleast_1d(axes), panels):
            ax.imshow(img, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"{obs_file}", fontsize=12)
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
        if ps_pre_light is not None:
            light_panels.append(("Prior SDE Output", ps_pre_light))
        if ps_corrected_light is not None:
            light_panels.append(("After Posterior Correction", ps_corrected_light))

        if light_panels:
            fig, axes = plt.subplots(
                1, len(light_panels), figsize=(5 * len(light_panels), 5)
            )
            for ax, (title, img) in zip(np.atleast_1d(axes), light_panels):
                ax.imshow(img, cmap="viridis", vmin=0, vmax=1)
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle(f"Light Estimates — {obs_file}", fontsize=12)
            plt.tight_layout()
            light_png = os.path.join(output_dir, f"{basename}_light_stages.png")
            fig.savefig(light_png, dpi=150)
            plt.close(fig)
            print(f"  Saved {light_png}")

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
