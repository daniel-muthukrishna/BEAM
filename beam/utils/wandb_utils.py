"""
Weights & Biases integration for BEAM.

This module provides utility functions for logging metrics, images,
and other artifacts to Weights & Biases during training.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
from torch.utils.data import DataLoader
from PIL import Image
import io

# Import Weights & Biases
import wandb


from beam.models.interpolant import ScoreMatch, EMA

def init_wandb(config: Dict[str, Any], project_name: str = "beam-tess", mode: str = "online") -> None:
    """
    Initialize Weights & Biases for the current run.
    
    Args:
        config: Configuration dictionary
        project_name: W&B project name
        mode: W&B run mode ("online", "offline", "disabled")
    """
    # Initialize W&B with our config
    wandb.init(
        project=project_name,
        config=config,
        name=config.get("run_name", None),
        mode=mode
    )
    
    # Log the entire config
    wandb.config.update(config)


def log_metrics(metrics: Dict[str, float], step: int = None) -> None:
    """
    Log metrics to W&B.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number (e.g., epoch)
    """
    try:
        # Clean any problematic values before logging
        cleaned_metrics = {}
        for k, v in metrics.items():
            # Skip None values or invalid metrics
            if v is None:
                continue
            # Convert numpy types or tensors to Python native types
            if hasattr(v, 'item'):
                v = v.item()  # Handle PyTorch tensors
            elif hasattr(v, 'tolist'):
                v = v.tolist()  # Handle NumPy arrays
            cleaned_metrics[k] = v
            
        wandb.log(cleaned_metrics, step=step)
    except Exception as e:
        print(f"Warning: Error logging metrics to W&B: {e}")
        # Continue even if logging fails


def fig_to_image(fig: plt.Figure) -> np.ndarray:
    """
    Convert a matplotlib figure to a numpy array.
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        Image as a numpy array
    """
    # Save figure to a buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Open with PIL and convert to numpy array
    img = Image.open(buf)
    return np.array(img)


def log_images(
    tag: str, 
    images: Union[np.ndarray, List[np.ndarray], torch.Tensor], 
    step: int = None, 
    caption: Optional[Union[str, List[str]]] = None
) -> None:
    """
    Log images to W&B.
    
    Args:
        tag: Tag/name for the images
        images: Image or list of images (numpy array or torch tensor)
        step: Optional step number (e.g., epoch)
        caption: Optional caption or list of captions
    """
    # Process torch tensors
    if isinstance(images, torch.Tensor):
        images = images.cpu().detach().numpy()
    
    # Handle single image
    if isinstance(images, np.ndarray) and images.ndim in [2, 3]:
        if images.ndim == 3 and images.shape[0] in [1, 3]:  # CHW format
            images = np.transpose(images, (1, 2, 0))  # Convert to HWC
        # Add batch dimension if missing
        if images.ndim == 2 or (images.ndim == 3 and images.shape[2] in [1, 3]):
            images = np.expand_dims(images, axis=0)
    
    # Create captions list if a single caption is provided
    if caption is not None and isinstance(caption, str):
        caption = [caption] * len(images)
    
    # Log images with wandb
    wandb.log({tag: [wandb.Image(img, caption=cap) for img, cap in zip(images, caption or [None] * len(images))]}, step=step)


def log_sample_images(
    model: ScoreMatch,
    dataloader: DataLoader,
    device: torch.device,
    n_sample: int = 3,
    n_datapoint: int = 1,
    image_shape: Tuple[int, int] = (64, 64),
    step: int = None,
    name: str = "Training Set",  # "Training" or "Validation"
    ema: Optional[EMA] = None
) -> None:
    """
    Generate samples and log them to W&B with original and generated images side by side.
    
    Args:
        model: The diffusion model
        dataloader: DataLoader containing real data
        device: Device to generate on
        n_sample: Number of samples per conditioning vector
        n_datapoint: Number of datapoints to visualize
        image_shape: Shape of the images
        step: Optional step number (e.g., epoch)
        name: Name for the log (e.g., "Training Set", "Validation Set")
    """
    MEAN = 0.1154092
    STD =  0.2346011
    if ema is not None:
        ema.store(model)
        ema.copy_to(model)

    # Get a batch of data
    data_batch = next(iter(dataloader))
    # Extract data (limited to n_datapoint)
    x_real = data_batch['y'][:n_datapoint].to(device) if data_batch['y'].shape[1] == 1 else data_batch['y'].permute(1, 0, 2, 3)[:n_datapoint].to(device)
    c_real = data_batch['x'][:n_datapoint].to(device) if data_batch['x'].shape[1] == 1 else data_batch['x'].permute(1, 0, 2)[:n_datapoint].to(device)
    ffi_nums = data_batch['ffi_num'][:n_datapoint] if data_batch['x'].shape[1] == 1 else data_batch['ffi_num']*n_datapoint
    orbits = data_batch['orbit'][:n_datapoint] if data_batch['x'].shape[1] == 1 else data_batch['orbit']*n_datapoint
    # Generate samples
    with torch.no_grad():
        x_gen, x_gen_store, timesteps_store = model.simulate(
            c_real, 
            n_sample, 
            (image_shape[0], image_shape[1]), 
            device
        )

    # print(f"x_gen: {x_gen.shape}, x_gen_store: {x_gen_store.shape}, timesteps_store: {timesteps_store.shape if isinstance(timesteps_store, np.ndarray) else len(timesteps_store)}")
    # Create side-by-side comparison for each datapoint
    comparison_images = []
    comparison_captions = []
    
    for i in range(n_datapoint):
        # Create a figure with n_sample + 1 subplots (original + generated samples)
        fig = plt.figure(figsize=(3*(n_sample+1), 3))
        
        # Add original image
        plt.subplot(1, n_sample+1, 1)
        plt.imshow((x_real[i][0].cpu().detach().numpy()*STD + MEAN), cmap='viridis', vmin=0, vmax=1)
        plt.title(f"Original\nOrbit {orbits[i]}, FFI {ffi_nums[i]}")
        plt.axis('off')
        # Add generated samples
        for j in range(n_sample):
            plt.subplot(1, n_sample+1, j+2)
            plt.imshow((x_gen[i*n_sample + j][0].cpu().detach().numpy()*STD + MEAN), cmap='viridis', vmin=0, vmax=1)
            plt.title(f"Sample {j+1}")
            plt.axis('off')
        plt.tight_layout()
        
        # Convert figure to image array
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        
        comparison_images.append(img_array)
        comparison_captions.append(f"Orbit {orbits[i]}, FFI {ffi_nums[i]}")
    # Log the comparison images
    log_images(f"{name} Original vs Generated Samples", comparison_images, step=step, caption=comparison_captions)
    # Log generation process for one sample
    if len(timesteps_store) > 8:
        indices = -np.logspace(0.2, np.log10(len(timesteps_store)), 8, dtype=int) # log scale so that earlier diffusion steps are more frequent
        display_timesteps = [timesteps_store[i] for i in indices][::-1]
        display_samples = [x_gen_store[i] for i in indices][::-1]
    else:
        display_timesteps = timesteps_store
        display_samples = x_gen_store   

    # Create figure for generation process
    fig = plt.figure(figsize=(16, 3))
    for idx, (ts, sample) in enumerate(zip(display_timesteps, display_samples)):
        plt.subplot(1, len(display_timesteps), idx+1)
        plt.imshow((sample[0, 0]*STD + MEAN), cmap='viridis', vmin=0, vmax=1)
        
        if idx == 0:
            plt.title(f"t={ts}\n(Pure Noise)")
        elif idx == len(display_timesteps) - 1:
            plt.title(f"t={ts}\n(Final Image)")
        else:
            plt.title(f"t={ts}")
            
        plt.axis('off')
    plt.tight_layout()

    # Convert figure to image array
    fig.canvas.draw()
    process_img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    
    # Log the generation process
    log_images(f"{name} Generation Process", [process_img], step=step, caption=["Denoising steps"])
    if ema is not None:
        ema.restore(model)


def log_figure(tag: str, fig: plt.Figure, step: int = None) -> None:
    """
    Log a matplotlib figure to W&B.
    
    Args:
        tag: Tag/name for the figure
        fig: Matplotlib figure to log
        step: Optional step number (e.g., epoch)
    """
    image = fig_to_image(fig)
    log_images(tag, image, step=step)


def log_model(model: torch.nn.Module, name: str = "model") -> None:
    """
    Log a PyTorch model to W&B.
    
    Args:
        model: PyTorch model
        name: Name for the model
    """
    # Get model state dict, handling DistributedDataParallel
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    ema_state = model.ema.state_dict() if hasattr(model, 'ema') else None
    
    # Save model to a temporary file
    tmp_path = f"{name}.pt"
    tmp_ema_path = f"{name}_ema.pt"
    torch.save(model_state, tmp_path)
    torch.save(ema_state, tmp_ema_path)
    
    # Log model file
    wandb.save(tmp_path)
    wandb.save(tmp_ema_path)
    # Clean up
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    if os.path.exists(tmp_ema_path):
        os.remove(tmp_ema_path)

def finish_run() -> None:
    """Finish the current W&B run."""
    wandb.finish()