#!/usr/bin/env python
"""
BEAM: Background Elimination with Advanced Machine learning - Generation Script

"""

import os
import pickle
import argparse
import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from beam.models.simulator import PosteriorSDE
from beam.models.unet import ContextUnet
from beam.models.probabilitypath import GaussianProbabilityPath, OTAlpha, OTBeta, VPBeta
from beam.models.interpolant import ScoreMatch, EMA
from beam.utils.config import load_config, flatten_config
from beam.data.datasets import TESSDataset, create_train_valid_datasets_by_orbit, TESSDataset_angles_only
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate images with the trained diffusion model")
    parser.add_argument('--config', type=str, default='configs/generation_config.yaml',
                        help='Path to configuration YAML file')
    
    return parser.parse_args()


def load_model(model_path, config, device, train_loader, context_len=None):
    """
    Load a trained diffusion model.
    
    Args:
        model_path: Path to model checkpoint
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded DDPM model
    """
    # Build the same U-Net shape used during training so checkpoint weights load cleanly.
    # We infer the default conditioning length from one training batch unless overridden.
    batch = next(iter(train_loader))
    unet = ContextUnet(
        in_feats=1, 
        context_len=batch['x'].shape[2] if context_len is None else context_len,
        n_feat=config['model_n_feat'],
        channel_mults=config['model_channel_mults'],
        heads_at=config['model_heads_at'],
        num_res=config['model_num_res'],
        time_dim=64 ,
        context_dim=64,
    )
    
    # Wrap the U-Net in the training/sampling helper that knows how to interpret
    # the network output (flow / score / v-parameterization depending on config).
    model = ScoreMatch(
        nn_model=unet, 
        probability_path=GaussianProbabilityPath(
            alpha=OTAlpha(),
            beta=OTBeta() if config['model_architecture'] == "flow" else VPBeta()
        ),
        device=device, 
        architecture=config['model_architecture']
    )

    ema = EMA(
        model=model,
    )

    # Load state from checkpoint
    checkpoint = torch.load(model_path, map_location=device)

  
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])


    # Sampling usually uses the EMA-smoothed weights instead of raw training weights.
    if config['generation_ema']:
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.copy_to(model.nn_model)
  
    print(f"Loaded model from {model_path}")
    return model


def plot_real_param_and_generated(
    x_real,
    param_vec,
    x_gen,
    save_path,
    title="Samples",
    MEAN=0.1154092,
    STD=0.2346011,
    orbit=None,
    ffi=None,
):
    """
    x_real: [1, H, W] or [C, H, W] torch.Tensor
    param_vec: [12] torch.Tensor
    x_gen: [n_samples, 1, H, W] torch.Tensor
    """
    print(MEAN, STD)
    if len(x_gen.shape) == 3:
        x_gen = x_gen.unsqueeze(1)

    n_sample = x_gen.shape[0]
    image_shape = x_real.shape[-2:]
    ncols = n_sample + 1  # real image + generated samples

    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 5))
    header_parts = []
    if ffi is not None:
        header_parts.append(f"FFI {ffi}")
    if orbit is not None:
        header_parts.append(f"Orbit {orbit}")
    if header_parts:
        fig.suptitle(" | ".join(header_parts), fontsize=14)
    print(x_real.shape)

    # Real image
    axes[0].imshow(((x_real[0].cpu().numpy()/3)*STD + MEAN)*3, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title("Preprocessed Light")
    axes[0].axis('off')
  

    # Generated samples
    for j in range(n_sample):
        axes[j+1].imshow(x_gen[j][0].cpu().numpy()*STD + MEAN*3, cmap='viridis', vmin=0, vmax=1)
        axes[j+1].set_title(f"Generated Light {j+1}")
        axes[j+1].axis('off')
    
    # Param vector as text
    param_text = "\n".join([f"{v:.3f}" for v in param_vec.cpu().numpy().flatten()])


                
    param_text = (
                f"1/ED: {param_vec[0]:.3f} | 1/MD: {param_vec[1]:.3f} | 1/ED²: {param_vec[2]:.3f} | 1/MD²: {param_vec[3]:.3f}\n"
                f"E_el/az: {param_vec[4]:.1f}/{param_vec[5]:.1f} | M_el/az: {param_vec[6]:.1f}/{param_vec[7]:.1f}\n"
                f"E3_el/az: {param_vec[8]:.1f}/{param_vec[9]:.1f} | M3_el/az: {param_vec[10]:.1f}/{param_vec[11]:.1f}")
    
    fig.text(0.5, 0.05, param_text, fontsize=10, ha='center', family='monospace', weight='normal')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main_posterior_sde():
    # This is the active generation entrypoint.
    # It runs a posterior-guided sampling demo:
    #   1) warm up a low-res light image with the light prior model
    #   2) apply posterior score corrections using both light + star models
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    config = flatten_config(config)

    # `flatten_config` converts nested yaml keys like `generation.output_dir`
    # into flat keys like `generation_output_dir`.
    # Create output directory
    os.makedirs(config['generation_output_dir'], exist_ok=True)

    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Normalization constants for the two image domains used in the posterior:
    # - LIGHT: low-res diffuse/background light model domain (256x256)
    # - STAR: full-res residual/star model domain (4096x4096 tiles)
    LIGHTMEAN = config['generation_light_mean']
    LIGHTSTD = config['generation_light_std']
    STARMEAN = config['generation_star_mean']
    STARSTD = config['generation_star_std']


    loader_batch_size = config.get('generation_batch_size', 1)

    # Build a loader so `load_model(...)` can infer the conditioning shape from a batch.
    # The posterior demo below does not sample from this loader directly.
    angle_path = config['data_angle_path']
    ccd_folder = config['data_ccd_folder']
    camera_number = config['data_camera_number']
    background_path = config.get('data_background_path')

    if background_path is None:
        shape_dataset = TESSDataset_angles_only(
            angle_path=angle_path,
            camera_number=camera_number,
        )
        train_loader = DataLoader(shape_dataset, batch_size=loader_batch_size, shuffle=True)
    else:
        image_shape = tuple(config['data_image_shape'])
        full_dataset = TESSDataset(
            angle_path=angle_path,
            ccd_folder=ccd_folder,
            image_shape=image_shape,
            background_path=background_path,
            patch_size=config['data_patch_size'],
            repeat_factor=config['data_repeat_factor'],
            camera_number=camera_number,
            mean=config['data_mean'],
            std=config['data_std'],
        )
        train_dataset, _ = create_train_valid_datasets_by_orbit(full_dataset)
        train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True)

    # Load two checkpoints:
    # - `light_model`: predicts the low-res light/background component conditioned on 12 angles.
    # - `star_model`: predicts score/v on full-res residual tiles (used tile-wise on x_obs - U(y)).
    #
    # We pass the raw U-Nets into PosteriorSDE. That class handles CFG and converts the
    # v-parameterized outputs into scores/vector fields internally.
    star_model = load_model(config['generation_model_path_star'], config, device, train_loader, context_len=24)
    light_model = load_model(config['generation_model_path'], config, device, train_loader)
    star_unet = star_model.nn_model.to(device).eval()
    light_unet = light_model.nn_model.to(device).eval()

    # Demo path: pick one observed frame and recover a low-res light image `y` such that
    # x_obs ~= U(y) + residual, where U is bilinear upsampling to full resolution.
    obs_file =  'tess2018305145302-00009383-3-crm-ffi_dehoc_processed_im4096x4096.pkl.npy'
    light_file =  'tess2018305145302-00009383-3-crm-ffi_dehoc_processed_im256x256.pkl'

    x_obs = np.load("/pdo/users/djtufto/data/data_tess_4096_full/" + obs_file, mmap_mode="r")
    # The preprocessed low-res file is loaded here for inspection/debug parity, but the
    # sampler below starts from noise and does not directly initialize from this file.
    with open("/pdo/users/jlupoiii/TESS/data/processed_images_im256x256/" + light_file, "rb") as f:
        preprocessed_image = pickle.load(f)
    x_obs = torch.tensor(x_obs, dtype=torch.float32).to(device).view(1, 1, 4096, 4096)
    ffis = obs_file[18:26]

    # Look up the conditioning vector (12 orbital/geometry values) for this exact FFI.
    with open(config['data_angle_path'], "rb") as f:
        angle_dict = pickle.load(f)
    angles = angle_dict[ffis]
    # 12-D orbital/geometry conditioning vector expected by the light model.
    params = np.array([
        angles['1/ED'], angles['1/MD'], 
        angles['1/ED^2'], angles['1/MD^2'], 
        angles['Eel'], angles['Eaz'], 
        angles['Mel'], angles['Maz'], 
        angles['E3el'], angles['E3az'], 
        angles['M3el'], angles['M3az']
    ])
    params = torch.tensor(params, dtype=torch.float32).to(device)
    c_obs = params.view(1, 1, 12)
  

    B, C, H, W = x_obs.shape
    # PosteriorSDE evolves a low-res variable y (256x256 by default) and uses:
    # - U(y): upsampled light estimate in observed-image space
    # - UT(.): downprojection of full-res score terms back to low-res
    low_res_cfg = config.get('posterior_low_res_size', 256)
    low_res_size = int(low_res_cfg) if low_res_cfg is not None else 256
    low_res_size = min(low_res_size, H, W)
    if H % low_res_size != 0 or W % low_res_size != 0:
        raise ValueError(f"Image resolution {H}x{W} not divisible by low-res size {low_res_size}")
    upscale_factor = H // low_res_size
    stride = int(config.get('posterior_stride', low_res_size))
    # Time grid for the predictor/corrector dynamics (avoid exactly t=0 and t=1 for stability).
    t=torch.linspace(config['model_epsilon'], 1 - config['model_epsilon'], 100)
    t = t.unsqueeze(0).expand(1, 100).to(device)
    # PosteriorSDE implements the hybrid sampler:
    # - predictor phase: prior-driven warmup in low-res light space
    # - correction phase: posterior Langevin-style refinement using light + star terms
    posterior = PosteriorSDE(
        star_score=star_unet,
        light_score=light_unet,
        t=t,
        sigma=VPBeta(),
        guidance_value=config['generation_guidance_scale'],
        x_obs=x_obs,
        num_save=config['generation_num_timesteps'],
        num_corrections=3,
        corr_step_size=0.01,
        upscale_factor=upscale_factor,
        stride=stride,
        tile_size=low_res_size,
    )

    # Initialize low-res light image from Gaussian noise.
    y0 = torch.randn(1, 1, 256, 256).to(device)
    with torch.no_grad():
        # Returns saved trajectory states (not just final sample) and corresponding saved times.
        y_low, timesteps = posterior.simulate(
            c=c_obs,
            y0=y0,
        )
    # Convert the saved low-res trajectory back to image space for plotting.
    y_low = y_low*LIGHTSTD + LIGHTMEAN
    y_final = y_low[-1]
    # Upsample final low-res light estimate back to observed-image resolution.
    light_full = posterior.U(y_final)
    # Residual (star-like/high-frequency component) implied by the recovered light image.
    reconstruction = x_obs - light_full

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    posterior_dir = os.path.join(config['generation_output_dir'], 'posterior_sde', timestamp)
    os.makedirs(posterior_dir, exist_ok=True)
  

    # Save side-by-side diagnostics for the inferred decomposition x_obs ≈ light + residual.
    for idx in range(B):
        observed = x_obs[idx].detach().cpu().numpy()[0]
        light_img = light_full[idx].detach().cpu().numpy()[0]
        star_img = reconstruction[idx].detach().cpu().numpy()[0]

        entries = [
            ("Observed", observed, True),
            ("Light (U(y))", light_img, True),
            ("Star Residual", star_img, True),
        ]

        fig, axes = plt.subplots(1, len(entries), figsize=(3 * len(entries), 3))
        for ax, (title, img, clamp) in zip(np.atleast_1d(axes), entries):
            if clamp:
                ax.imshow(img, cmap='viridis', vmin=0, vmax=1)
            else:
                im = ax.imshow(img, cmap='viridis')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(posterior_dir, f'posterior_components_sample{idx}.png'))
        plt.close(fig)

    # Save a coarse visualization of the low-res trajectory from noise to final sample.
    # `y_low` has shape [num_saved_steps, batch, channel, H, W].
    if y_low.numel() > 0:
        traj_np = y_low.detach().cpu().numpy()
        time_np = timesteps.detach().cpu().numpy()
        total_steps = traj_np.shape[0]
        num_ts_cfg = config.get('posterior_num_timesteps', min(total_steps, 10))
        num_ts = int(num_ts_cfg) if num_ts_cfg is not None else min(total_steps, 10)
        num_ts = max(1, min(num_ts, total_steps))

        if num_ts >= total_steps:
            idx = np.arange(total_steps, dtype=int)
        else:
            idx = np.linspace(0, total_steps - 1, num_ts, dtype=int)

        if time_np.shape[0] == total_steps:
            disp_times = time_np[idx]
        elif time_np.shape[0] == idx.shape[0]:
            disp_times = time_np
        else:
            disp_times = np.linspace(0.0, 1.0, len(idx))

        fig, axes = plt.subplots(1, len(idx), figsize=(2 * len(idx), 3))
        axes = np.atleast_1d(axes)
        sample_idx = 0
        for ax, step_idx, t_val in zip(axes, idx, disp_times):
            ax.imshow(
                traj_np[step_idx, sample_idx, 0],
                cmap='viridis',
                vmin=0,
                vmax=1,
            )
            ax.axis('off')
            if step_idx == 0:
                ax.set_title("Start (noise)", fontsize=10)
            elif step_idx == total_steps - 1:
                ax.set_title("Final", fontsize=10)
            else:
                ax.set_title(f"Step {t_val:.3f}", fontsize=10)

        fig.suptitle("Posterior SDE Trajectory", fontsize=14)
        plt.tight_layout()
        fig.savefig(os.path.join(posterior_dir, 'posterior_trajectory.png'))
        plt.close(fig)

    print(f"Posterior SDE outputs saved to {posterior_dir}")

if __name__ == "__main__":
    main_posterior_sde()
