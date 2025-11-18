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
import matplotlib.pyplot as plt
from beam.models.simulator import ODEIntegrator, EulerMaruyama, PosteriorSDE

from beam.models.unet import ContextUnet
from beam.models.probabilitypath import GaussianProbabilityPath, OTAlpha, OTBeta, VPBeta
from beam.models.interpolant import ScoreMatch, EMA
from beam.utils.config import load_config, flatten_config
from beam.utils.visualization import plot_samples   
from beam.data.datasets import TESSDataset, create_train_valid_datasets_by_orbit
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
    # Create model architecture
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
    
    # Create model
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


    # Load EMA weights
    if config['generation_ema']:
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.copy_to(model.nn_model)
  
    print(f"Loaded model from {model_path}")
    return model

def load_params(params_file, n_samples=5):
    """
    Load orbital parameters from file.
    
    Args:
        params_file: Path to orbital parameters file
        n_samples: Number of random parameters to use if file not provided
        
    Returns:
        Tensor of parameters
    """
    if params_file and os.path.exists(params_file):
        with open(params_file, 'rb') as f:
            params = pickle.load(f)
            return torch.tensor(params, dtype=torch.float32)
    else:
        # Generate random parameters for testing
        print(f"No parameters file found, using {n_samples} random parameters")
        return torch.rand((n_samples, 1, 12))


def plot_real_param_and_generated(x_real, param_vec, x_gen, save_path, title="Samples", MEAN=0.1154092, STD=0.2346011):
    """
    x_real: [1, H, W] or [C, H, W] torch.Tensor
    param_vec: [12] torch.Tensor
    x_gen: [n_samples, 1, H, W] torch.Tensor
    """
    if len(x_gen.shape) == 3:
        x_gen = x_gen.unsqueeze(1)

    n_sample = x_gen.shape[0]
    image_shape = x_real.shape[-2:]
    ncols = n_sample + 1  # real image + generated samples

    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3))

    # Real image
    axes[0].imshow(x_real[0].cpu().numpy()*STD + MEAN, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # # Param vector as text
    # param_text = "\n".join([f"{v:.3f}" for v in param_vec.cpu().numpy().flatten()])
    # axes[1].text(0.5, 0.5, param_text, fontsize=10, ha='center', va='center', family='monospace')
    # axes[1].set_title("Params")
    # axes[1].axis('off')

    # Generated samples
    for j in range(n_sample):
        axes[j+1].imshow(x_gen[j][0].cpu().numpy()*STD + MEAN, cmap='viridis', vmin=0, vmax=1)
        axes[j+1].set_title(f"Sample {j+1}")
        axes[j+1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
 


def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config = flatten_config(config)
    
    # Create output directory
    os.makedirs(config['generation_output_dir'], exist_ok=True)
    
    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
 
    # MEAN = 0.65187982
    # STD =  1.10163832
    # MEAN = 0.1154092
    # STD =  0.2346011
    # MEAN = 0.1099859
    # STD =  0.8313040
    MEAN = 0.01978608595999928
    STD =  0.08876927679677006
    # MEAN = 0
    # STD = 1
    # Load parameters
    params = load_params(config['generation_params_file'], config['generation_n_datapoint'])
    params = params.to(device)

    # Load dataset for real images
    angle_path = config['data_angle_path']
    ccd_folder = config['data_ccd_folder']
    image_shape = tuple(config['generation_image_shape'])
    background_path = config['data_background_path']
    full_dataset = TESSDataset(angle_path=angle_path, 
                               ccd_folder=ccd_folder, 
                               image_shape=image_shape, 
                               background_path=background_path, 
                               patch_size=config['data_patch_size'], 
                               repeat_factor=config['data_repeat_factor']
                               )
    train_dataset, valid_dataset = create_train_valid_datasets_by_orbit(full_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config['generation_n_datapoint'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['generation_n_datapoint'], shuffle=True)
  

    model = load_model(config['generation_model_path'], config, device, train_loader)
    model.eval()
    # Generate and plot for training set
    # train_batch = next(iter(train_loader))
    for batch in train_loader:
        if batch['y'].mean() > 0.3:
            print
            train_batch = batch
            break
    print(train_batch['x'].shape)
    print(train_batch['y'].shape)
    valid_batch = next(iter(valid_loader))

    # Generate samples for training set
    params_train = train_batch['x'].to(device)
    x_real_train = train_batch['y'].to(device)
    print("Generating training set")
    for i in range(x_real_train.shape[0]):
        param = params_train[i].unsqueeze(0)  # [1, 12]
        with torch.no_grad():
            x_gen, x_gen_intermediate, timesteps = model.simulate(
                c_i=param,
                n_sample=config['generation_n_sample'],
                size=image_shape,
                device=device,
                simulator=ODEIntegrator if config['model_architecture'] == "flow" else EulerMaruyama,
                guidance_scale=config['generation_guidance_scale'],
                num_save=config['generation_num_timesteps'],
                num_steps=200,
                epsilon=config['model_epsilon']
            )
        save_path = os.path.join(
            config['generation_output_dir'],
            f"train_real_param_gen_{i+1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
    
        plot_real_param_and_generated(x_real_train[i], param[0], x_gen, save_path, title="train", MEAN=MEAN, STD=STD)
        fig, axes = plt.subplots(1, x_gen_intermediate.shape[0], figsize=(3 * x_gen_intermediate.shape[0], 3))
        for j in range(x_gen_intermediate.shape[0]):
            timestep = timesteps[j]
            axes[j].imshow(x_gen_intermediate[j][0][0]*STD + MEAN, cmap='viridis', vmin=0, vmax=1)
            axes[j].set_title(f"t={timestep}")
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(
            config['generation_output_dir'],
            f"train_real_param_gen_{i+1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_intermediate.png"))
    

    # # Generate and plot for validation set
    # params_valid = valid_batch['x'].to(device)
    # x_real_valid = valid_batch['y'].to(device)
    
    # for i in range(x_real_valid.shape[0]):
    #     param = params_valid[i].unsqueeze(0)  # [1, 12]
    #     with torch.no_grad():
    #         x_gen, _, _ = model.simulate(
    #             c_i=param,
    #             n_sample=config['generation_n_sample'],
    #             size=image_shape,
    #             device=device,
    #             simulator=ODEIntegrator if config['model_architecture'] == "flow" else EulerMaruyama,
    #             guidance_scale=config.get('generation_guidance_scale', 1.0),
    #         )
        # save_path = os.path.join(
        #     config['generation_output_dir'],
        #     f"valid_real_param_gen_{i+1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        # )
        # plot_real_param_and_generated(x_real_valid[i], param[0], x_gen, save_path, title="valid")

def main_posterior_sde():
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    config = flatten_config(config)

    # Create output directory
    os.makedirs(config['generation_output_dir'], exist_ok=True)

    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Normalisation constants
    LIGHTMEAN = 0.01978608595999928
    LIGHTSTD = 0.08876927679677006
    STARMEAN = 0.08837062429384838 
    STARSTD =  0.8176902707359797


    # Load parameters (currently unused but retained for parity with main())
    params = load_params(config['generation_params_file'], config['generation_n_datapoint']).to(device)

    # Load dataset for real images
    angle_path = config['data_angle_path']
    ccd_folder = config['data_ccd_folder']
    image_shape = tuple(config['generation_image_shape'])
    background_path = config['data_background_path']
    full_dataset = TESSDataset(
        angle_path=angle_path,
        ccd_folder=ccd_folder,
        image_shape=image_shape,
        background_path=background_path,
        patch_size=config['data_patch_size'],
        repeat_factor=config['data_repeat_factor'],
    )
    train_dataset, valid_dataset = create_train_valid_datasets_by_orbit(full_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config['generation_n_datapoint'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['generation_n_datapoint'], shuffle=True)

    # Load models and extract the underlying UNets used for scoring
    star_model = load_model(config['generation_model_path_star'], config, device, train_loader, context_len=24)
    light_model = load_model(config['generation_model_path'], config, device, train_loader)
    star_unet = star_model.nn_model.to(device).eval()
    light_unet = light_model.nn_model.to(device).eval()

    # Choose a representative batch
    # train_batch = None
    # for batch in train_loader:
    #     if batch['y'].mean() > 0.3:
    #         train_batch = batch
    #         break
    # if train_batch is None:
    #     train_batch = next(iter(train_loader))

    # obs_files = sorted(os.listdir("/pdo/users/djtufto/data/data_tess_4096_full/"))
    # light_files = sorted(os.listdir("/pdo/users/jlupoiii/TESS/data/processed_images_im256x256/"))
    # rdx = random.randint(0, len(obs_files)-1)
    # obs_file = obs_files[rdx]
    # light_file =  obs_file[:-17] + '256x256.pkl'
    obs_file =  'tess2018305145302-00009383-3-crm-ffi_dehoc_processed_im4096x4096.pkl.npy'
    light_file =  'tess2018305145302-00009383-3-crm-ffi_dehoc_processed_im256x256.pkl'

    x_obs = np.load("/pdo/users/djtufto/data/data_tess_4096_full/" + obs_file, mmap_mode="r")
    with open("/pdo/users/jlupoiii/TESS/data/processed_images_im256x256/" + light_file, "rb") as f:
        preprocessed_image = pickle.load(f)
    x_obs = torch.tensor(x_obs, dtype=torch.float32).to(device).view(1, 1, 4096, 4096)
    y0 = torch.randn(1, 1, 256, 256).to(device)
    ffis = obs_file[18:26]

    with open(config['data_angle_path'], "rb") as f:
        angle_dict = pickle.load(f)
    ffis = obs_file[18:26]
    angles = angle_dict[ffis]
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
    low_res_cfg = config.get('posterior_low_res_size', 256)
    low_res_size = int(low_res_cfg) if low_res_cfg is not None else 256
    low_res_size = min(low_res_size, H, W)
    if H % low_res_size != 0 or W % low_res_size != 0:
        raise ValueError(f"Image resolution {H}x{W} not divisible by low-res size {low_res_size}")
    upscale_factor = H // low_res_size
    stride = int(config.get('posterior_stride', low_res_size))
    t=torch.linspace(config['model_epsilon'], 1 - config['model_epsilon'], 100)
    t = t.unsqueeze(0).expand(1, 100).to(device)
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

    y0 = torch.randn(1, 1, 256, 256).to(device)
    with torch.no_grad():
        y_low, timesteps = posterior.simulate(
            c=c_obs,
            y0=y0,
        )
    y_low = y_low*LIGHTSTD + LIGHTMEAN
    y_final = y_low[-1]
    light_full = posterior.U(y_final)
    reconstruction = x_obs - light_full

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    posterior_dir = os.path.join(config['generation_output_dir'], 'posterior_sde', timestamp)
    os.makedirs(posterior_dir, exist_ok=True)
  

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
