import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from tqdm import tqdm
import argparse

# Import the model and dataset classes from the training script
from train_angle_to_image import AngleToImageDataset, AngleToImageModel


def create_prediction_gifs(model_path, angles_pkl, images_dir, output_dir, device='cuda'):
    """
    Create GIFs of predictions and ground truth for the test set

    Args:
        model_path: Path to the trained model checkpoint
        angles_pkl: Path to angles pickle file
        images_dir: Directory containing images
        output_dir: Directory to save output GIFs
        device: Device to run inference on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test dataset (orbits 53 and 54)
    print("Loading test dataset (orbits 53 and 54)...")
    test_dataset = AngleToImageDataset(angles_pkl, images_dir, filter_orbits=[53, 54])
    print(f"Test dataset size: {len(test_dataset)}")

    # Load model
    print(f"Loading model from {model_path}...")
    model = AngleToImageModel(num_angle_features=test_dataset.num_features, image_size=256)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Generate predictions for all test samples
    print("Generating predictions...")
    predictions = []
    ground_truths = []
    ffi_ids = []
    angle_metadata = []

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            angles, true_image, ffi_id = test_dataset[i]

            # Get the angle dictionary for metadata
            _, angle_dict, _ = test_dataset.samples[i]

            # Predict
            angles_batch = angles.unsqueeze(0).to(device)
            pred_image = model(angles_batch).cpu().squeeze().numpy()
            true_image = true_image.squeeze().numpy()

            predictions.append(pred_image)
            ground_truths.append(true_image)
            ffi_ids.append(ffi_id)
            angle_metadata.append(angle_dict)

    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    print(f"Generated {len(predictions)} predictions")

    # Create individual GIFs
    print("Creating GIF of predictions...")
    create_single_gif(predictions, output_dir / 'predictions.gif',
                     title='Predicted Scattered Light (Orbits 53-54)',
                     ffi_ids=ffi_ids, metadata=angle_metadata)

    print("Creating GIF of ground truth...")
    create_single_gif(ground_truths, output_dir / 'ground_truth.gif',
                     title='True Scattered Light (Orbits 53-54)',
                     ffi_ids=ffi_ids, metadata=angle_metadata)

    # Create side-by-side comparison GIF
    print("Creating side-by-side comparison GIF...")
    create_sidebyside_gif(ground_truths, predictions, output_dir / 'comparison.gif',
                         ffi_ids=ffi_ids, metadata=angle_metadata)

    # Create static comparison plot with several samples
    print("Creating static comparison plot...")
    create_comparison_plot(ground_truths, predictions, ffi_ids,
                          output_dir / 'comparison_samples.png', num_samples=10)

    print(f"All visualizations saved to {output_dir}")


def create_single_gif(images, output_path, title='', fps=10, vmin=0, vmax=1,
                     ffi_ids=None, metadata=None):
    """Create a GIF from a sequence of images"""
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    # Create initial image and colorbar
    im = ax.imshow(images[0], cmap='gray', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add text for FFI ID (above image)
    ffi_text = fig.text(0.5, 0.95, '', ha='center', va='top', fontsize=12, weight='bold')

    # Add text for parameters (below image)
    params_text = fig.text(0.5, 0.02, '', ha='center', va='bottom', fontsize=10,
                          family='monospace')

    def update(frame):
        im.set_data(images[frame])
        ax.set_title(f'{title}\nFrame {frame+1}/{len(images)}', fontsize=14)
        ax.axis('off')

        # Update FFI ID
        if ffi_ids is not None:
            ffi_text.set_text(f'FFI: {ffi_ids[frame]}')

        # Update parameters
        if metadata is not None:
            meta = metadata[frame]

            # Format values safely
            def fmt_float(val):
                return f"{val:.2f}" if isinstance(val, (int, float)) else str(val)

            def fmt_exp(val):
                return f"{val:.2e}" if isinstance(val, (int, float)) else str(val)

            params_str = (
                f"Eel={fmt_float(meta.get('Eel', 'N/A'))}° | "
                f"Eaz={fmt_float(meta.get('Eaz', 'N/A'))}° | "
                f"Mel={fmt_float(meta.get('Mel', 'N/A'))}° | "
                f"Maz={fmt_float(meta.get('Maz', 'N/A'))}° | "
                f"ED={fmt_exp(meta.get('ED', 'N/A'))} | "
                f"MD={fmt_exp(meta.get('MD', 'N/A'))} | "
                f"Below sunshade={meta.get('below_sunshade', 'N/A')}"
            )
            params_text.set_text(params_str)

        return im, ffi_text, params_text

    anim = animation.FuncAnimation(fig, update, frames=len(images),
                                  interval=1000/fps, blit=False)
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    print(f"Saved GIF to {output_path}")


def create_sidebyside_gif(ground_truths, predictions, output_path, fps=10, vmin=0, vmax=1,
                          ffi_ids=None, metadata=None):
    """Create a side-by-side comparison GIF"""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Create initial images and colorbars
    im0 = axes[0].imshow(ground_truths[0], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth', fontsize=14)
    axes[0].axis('off')
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(predictions[0], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title('Prediction', fontsize=14)
    axes[1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    diff0 = np.abs(ground_truths[0] - predictions[0])
    im2 = axes[2].imshow(diff0, cmap='hot', vmin=0, vmax=vmax)
    axes[2].set_title('Absolute Difference', fontsize=14)
    axes[2].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Add text for FFI ID (above)
    ffi_text = fig.text(0.5, 0.96, '', ha='center', va='top', fontsize=12, weight='bold')

    # Add text for parameters (below)
    params_text = fig.text(0.5, 0.02, '', ha='center', va='bottom', fontsize=9,
                          family='monospace')

    def update(frame):
        # Update images
        im0.set_data(ground_truths[frame])
        im1.set_data(predictions[frame])
        diff = np.abs(ground_truths[frame] - predictions[frame])
        im2.set_data(diff)

        # Update FFI ID
        if ffi_ids is not None:
            ffi_text.set_text(f'FFI: {ffi_ids[frame]} | Frame {frame+1}/{len(ground_truths)}')

        # Update parameters
        if metadata is not None:
            meta = metadata[frame]

            # Format values safely
            def fmt_float(val):
                return f"{val:.2f}" if isinstance(val, (int, float)) else str(val)

            def fmt_exp(val):
                return f"{val:.2e}" if isinstance(val, (int, float)) else str(val)

            params_str = (
                f"Eel={fmt_float(meta.get('Eel', 'N/A'))}° | "
                f"Eaz={fmt_float(meta.get('Eaz', 'N/A'))}° | "
                f"Mel={fmt_float(meta.get('Mel', 'N/A'))}° | "
                f"Maz={fmt_float(meta.get('Maz', 'N/A'))}° | "
                f"ED={fmt_exp(meta.get('ED', 'N/A'))} | "
                f"MD={fmt_exp(meta.get('MD', 'N/A'))} | "
                f"Below sunshade={meta.get('below_sunshade', 'N/A')}"
            )
            params_text.set_text(params_str)

        return im0, im1, im2, ffi_text, params_text

    anim = animation.FuncAnimation(fig, update, frames=len(ground_truths),
                                  interval=1000/fps, blit=False)
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    print(f"Saved side-by-side comparison GIF to {output_path}")


def create_comparison_plot(ground_truths, predictions, ffi_ids, output_path,
                          num_samples=10, vmin=0, vmax=1):
    """Create a static plot comparing multiple samples"""
    # Sample evenly across the test set
    indices = np.linspace(0, len(ground_truths)-1, num_samples, dtype=int)

    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 5*num_samples))

    for i, idx in enumerate(indices):
        # Ground truth
        im0 = axes[i, 0].imshow(ground_truths[idx], cmap='gray', vmin=vmin, vmax=vmax)
        if i == 0:
            axes[i, 0].set_title('Ground Truth', fontsize=14)
        axes[i, 0].set_ylabel(f'FFI: {ffi_ids[idx][:10]}...', fontsize=10)
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # Prediction
        im1 = axes[i, 1].imshow(predictions[idx], cmap='gray', vmin=vmin, vmax=vmax)
        if i == 0:
            axes[i, 1].set_title('Prediction', fontsize=14)
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Difference
        diff = np.abs(ground_truths[idx] - predictions[idx])
        im2 = axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=vmax)
        if i == 0:
            axes[i, 2].set_title('Absolute Difference', fontsize=14)
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize test set predictions')
    parser.add_argument('--model_path', type=str,
                       default='angle_to_image_outputs/checkpoints/checkpoint_final.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--angles_pkl', type=str,
                       default='/pdo/users/djtufto/data/data_tess_4096_raw/angles/tess_angles_O11-54_data_dic.pkl',
                       help='Path to angles pickle file')
    parser.add_argument('--images_dir', type=str,
                       default='/pdo/users/jlupoiii/TESS/data/processed_images_im256x256/',
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str,
                       default='test_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for GIFs')

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model not found at {args.model_path}")
        print("Please train the model first or specify a different model path")
        return

    create_prediction_gifs(
        model_path=args.model_path,
        angles_pkl=args.angles_pkl,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
