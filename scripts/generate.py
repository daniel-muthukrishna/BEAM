#!/usr/bin/env python
"""
BEAM: Background Elimination with Advanced Machine learning - Generation Script

Likelihood-only Langevin sampling at 4096x4096.
Loads a preprocessed light estimate, refines it via Langevin dynamics
using the star model score on the residual (obs - X).
"""

import os
import pickle
import argparse
import datetime
from astropy.io import fits
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from beam.models.simulator import LikelihoodLangevin
from beam.models.unet import ContextUnet
from beam.models.probabilitypath import GaussianProbabilityPath, OTAlpha, VPBeta
from beam.models.interpolant import ScoreMatch, EMA
from beam.utils.config import load_config, flatten_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Likelihood-only Langevin generation")
    parser.add_argument('--config', type=str, default='configs/generation_config.yaml',
                        help='Path to configuration YAML file')
    return parser.parse_args()


def load_star_model(model_path, config, device):
    """
    Load the star U-Net from a checkpoint.

    Returns the raw U-Net in eval mode (not the ScoreMatch wrapper).
    """
    unet = ContextUnet(
        in_feats=1,
        context_len=24,
        n_feat=config['model_n_feat'],
        channel_mults=config['model_channel_mults'],
        heads_at=config['model_heads_at'],
        num_res=config['model_num_res'],
        time_dim=64,
        context_dim=64,
    )

    model = ScoreMatch(
        nn_model=unet,
        probability_path=GaussianProbabilityPath(
            alpha=OTAlpha(),
            beta=VPBeta()
        ),
        device=device,
        architecture=config['model_architecture']
    )

    ema = EMA(model=model)
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    if config['generation_ema']:
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.copy_to(model.nn_model)

    print(f"Loaded star model from {model_path}")
    return model.nn_model.to(device).eval()


def main():
    args = parse_args()
    config = load_config(args.config)
    config = flatten_config(config)

    os.makedirs(config['generation_output_dir'], exist_ok=True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    STARMEAN = config['generation_star_mean']
    STARSTD = config['generation_star_std']

    star_unet = load_star_model(config['generation_model_path_star'], config, device)

    # Load the observed 4096x4096 image and the preprocessed light estimate.
    light_file = 'tess2018305145302-00009383-3-crm-ffi_dehoc_processed_im4096x4096.pkl'
    obs_file = 'tess2018305145302-00009383-3-crm-ffi_dehoc.fits.gz'
    with fits.open('/pdo/users/roland/SL_data/O15_data/' + obs_file) as hdul:
        x_obs_np = hdul[0].data/633118
    columns_to_delete = (
            list(range(0, 44)) + list(range(2092, 2180)) + list(range(4228, 4272))
        )
    rows_to_delete = range(2048, 2108)
    x_obs_np = np.delete(x_obs_np, columns_to_delete, axis=1)
    x_obs_np = np.delete(x_obs_np, rows_to_delete, axis=0)
    x_obs_np = np.ascontiguousarray(x_obs_np, dtype=np.float32)
    x_obs = torch.from_numpy(x_obs_np).to(device).view(1, 1, 4096, 4096)

    # Use the same preprocessed file as the initial light estimate X0.
    # In practice this is a background-subtracted image; Langevin will refine it.
    with open(os.path.join(config['data_ccd_folder'], light_file), "rb") as f:
        light_np = pickle.load(f)
    light_np = np.ascontiguousarray(light_np, dtype=np.float32)
    X0 = torch.from_numpy(light_np).to(device).view(1, 1, 4096, 4096)

    num_steps = config.get('generation_num_langevin_steps', 100)
    step_size = config.get('generation_langevin_step_size', 0.01)

    sampler = LikelihoodLangevin(
        star_score=star_unet,
        x_obs=x_obs,
        sigma=VPBeta(),
        star_mean=STARMEAN,
        star_std=STARSTD,
        tile_size=256,
        num_steps=num_steps,
        step_size=step_size,
        num_save=config.get('generation_num_timesteps', 10),
        star_tile_microbatch=160,
    )

    with torch.no_grad():
        Xs_saved, step_indices = sampler.simulate(X0)

    X_final = Xs_saved[-1]
    stars = x_obs - X_final

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(config['generation_output_dir'], 'likelihood_langevin', timestamp)
    os.makedirs(out_dir, exist_ok=True)
    print(torch.max(torch.abs(X0 - X_final)))

    # Save component decomposition plot.
    B = x_obs.shape[0]
    for idx in range(B):
        observed = x_obs[idx, 0].detach().cpu().numpy()
        light_img = X_final[idx, 0].detach().cpu().numpy()
        star_img = stars[idx, 0].detach().cpu().numpy()

        entries = [
            ("Observed", observed),
            ("Light (refined)", light_img),
            ("Star Residual", star_img),
        ]
        fig, axes = plt.subplots(1, len(entries), figsize=(3 * len(entries), 3))
        for ax, (title, img) in zip(np.atleast_1d(axes), entries):
            ax.imshow(img, cmap='viridis', vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'components_sample{idx}.png'))
        plt.close(fig)

    # Save trajectory snapshots (downsampled to 256x256 for display).
    # if Xs_saved.numel() > 0:
    #     n_snaps = Xs_saved.shape[0]
    #     fig, axes = plt.subplots(1, n_snaps, figsize=(2 * n_snaps, 3))
    #     axes = np.atleast_1d(axes)
    #     for i, (ax, step) in enumerate(zip(axes, step_indices)):
    #         img = Xs_saved[i, 0, 0].detach().cpu().numpy()
    #         # Downsample for display
    #         ax.imshow(img, cmap='viridis', vmin=0, vmax=1)
    #         ax.axis('off')
    #         if i == 0:
    #             ax.set_title("Init", fontsize=10)
    #         elif i == n_snaps - 1:
    #             ax.set_title("Final", fontsize=10)
    #         else:
    #             ax.set_title(f"Step {step}", fontsize=10)

    #     fig.suptitle("Likelihood Langevin Trajectory", fontsize=14)
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(out_dir, 'trajectory.png'))
    #     plt.close(fig)

    print(f"Likelihood Langevin outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
