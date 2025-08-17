#!/usr/bin/env python
"""
BEAM: Background Elimination with Advanced Machine learning - Generation Script

This script uses a trained diffusion model to generate TESS images conditioned
on orbital parameters.
"""

import os
import pickle
import argparse
import time
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from beam.models.simulator import ODEIntegrator, EulerMaruyama
from beam.models.unet import ContextUnet
from beam.models.probabilitypath import GaussianProbabilityPath, OTAlpha, OTBeta, VPBeta
from beam.models.interpolant import ScoreMatch, EMA
from beam.utils.config import load_config, flatten_config
from beam.utils.visualization import plot_samples, plot_generation_process
from beam.data.datasets import TESSDataset, create_train_valid_datasets_by_orbit
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate images with the trained diffusion model")
    parser.add_argument('--config', type=str, default='configs/generation_config.yaml',
                        help='Path to configuration YAML file')
    
    return parser.parse_args()


def load_model(model_path, config, device, train_loader):
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
        context_len=batch['x'].shape[2],
        n_feat=config['model_n_feat'],
        channel_mults=config['model_channel_mults'],
        heads_at=config['model_heads_at'],
        num_res=config['model_num_res'],
        time_dim=16,
        context_dim=16
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

    print(checkpoint.keys())
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
    
@torch.no_grad()
def one_step_denoise_check(net, x, c, t_min=0.001, t_max=0.1, w=1.0):
    """
    x: clean held-out batch in the same scaling used for training (e.g., [-1,1])
    c: conditioning tensor of shape (B, 1, D)
    w: CFG weight (1.0 = pure conditional)
    """
    net.eval()

    # sample t and build forward corruption
    B = x.shape[0]
    t = torch.rand(B, 1, 1, 1, device=x.device) * (t_max - t_min) + t_min
    beta = torch.sqrt((1 - t**2).clamp_min(1e-12))
    eps_true = torch.randn_like(x)
    x_t = t * x + beta * eps_true

    # pure conditional vs unconditional inputs (per-example masks)
    mask_c = torch.zeros(B, 1, c.shape[-1], device=c.device, dtype=c.dtype)
    mask_u = torch.ones (B, 1, c.shape[-1], device=c.device, dtype=c.dtype)
    c_u = torch.zeros_like(c)

    # two passes share the same (x_t, t)
    eps_c = net(x_t, c,  t, mask_c)
    eps_u = net(x_t, c_u, t, mask_u)
    eps_g = (1 - w) * eps_u + w * eps_c        # CFG in ε-space

    # reconstruct x0
    x0_pred = (x_t - beta * eps_g) / t.clamp_min(1e-3)

    # quick diagnostics
    mse_x0 = torch.mean((x0_pred - x)**2).item()
    mse_eps = torch.mean((eps_c - eps_true)**2).item()
    return x0_pred, {"mse_x0": mse_x0, "mse_eps_vs_true": mse_eps, "t_mean": t.mean().item()}




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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
 

    # MEAN = 0.1154092
    # STD =  0.2346011
    MEAN = 0.1099859
    STD =  0.8313040
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
        if batch['y'].mean() > 0.07:
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
                guidance_scale=config.get('generation_guidance_scale', 1.0),
                num_save=config['generation_num_timesteps'],
                num_steps=1500,
                epsilon=config['model_epsilon']
            )
            print(x_gen.shape)
            print(x_gen_intermediate.shape)
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


if __name__ == "__main__":
    main()
