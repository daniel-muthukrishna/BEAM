#!/usr/bin/env python
"""
BEAM: Background Elimination with Advanced Machine learning - Generation Script

This script uses a trained diffusion model to generate TESS images conditioned
on orbital parameters.
"""

import os
import pickle
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from beam.models.unet import ContextUnet
from beam.models.probabilitypath import GaussianProbabilityPath, LinearAlpha, LinearBeta
from beam.models.scorematch import ScoreMatch, EMA
from beam.utils.config import load_config, flatten_config
from beam.utils.visualization import plot_samples, plot_generation_process


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate images with the trained diffusion model")
    parser.add_argument('--config', type=str, default='configs/generation_config.yaml',
                        help='Path to configuration YAML file')
    
    return parser.parse_args()


def load_model(model_path, config, device):
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
    unet = ContextUnet(
        in_channels=1, 
        in_dim=12,  # Fixed dimension for orbital parameters
        n_feat=config['model_n_feat']
    )
    
    # Create model
    flow_model = ScoreMatch(
        nn_model=unet, 
        probability_path=GaussianProbabilityPath(
            alpha=LinearAlpha(),
            beta=LinearBeta()
        ),
        device=device, 
        architecture=config['model_architecture']
    )
    
    # Load state from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        flow_model.load_state_dict(checkpoint['model_state_dict'])
  
    
    print(f"Loaded model from {model_path}")
    return flow_model


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
    model = load_model(config['generation_model_path'], config, device)
    model.eval()
    
    #Optionally use EMA weights for sampling
    if config['generation_ema']:
        model.ema.copy_to(model.nn_model)
    
    # Load parameters
    params = load_params(config['generation_params_file'], config['generation_n_sample'])
    params = params.to(device)


    
    # Generate samples
    image_shape = config['generation_image_shape']
    with torch.no_grad():
        if config['generation_guidance_scale'] > 0:
            # Generate with guidance
            x_gen, x_gen_store, timesteps_store = model.simulate(
                c_i=params,
                n_sample=params.shape[0],
                size=(image_shape[0], image_shape[1]),
                device=device,
                guidance_scale=config['generation_guidance_scale'],
                num_save=config['generation_num_timesteps']
            )
    
    # Plot and save results
    for i in range(params.shape[0]):
        # Plot generation process
        fig = plot_generation_process(
            x_gen_store=x_gen_store,
            sample_idx=i,
            num_timesteps=8,
            title=f"Generation Process (Sample {i+1})",
            save_path=os.path.join(config['generation_output_dir'], f"process_sample_{i+1}.png")
        )
        plt.close(fig)
        
        # Save generated image
        plt.figure(figsize=(6, 6))
        plt.imshow(x_gen[i][0].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f"Generated Sample {i+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(config['generation_output_dir'], f"sample_{i+1}.png"))
        plt.close()
    
    print(f"Generated {params.shape[0]} samples in {config['generation_output_dir']}")


if __name__ == "__main__":
    main()
