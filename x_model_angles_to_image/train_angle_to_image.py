import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

class AngleToImageDataset(Dataset):
    """Dataset that loads angles and corresponding TESS images"""

    def __init__(self, angles_pkl_path, images_dir, transform=None, filter_orbits=None):
        """
        Args:
            angles_pkl_path: Path to the angles pickle file
            images_dir: Directory containing the .pkl image files
            transform: Optional transform to be applied on images
            filter_orbits: List of orbit numbers to include (None includes all orbits)
        """
        # Load angles data
        with open(angles_pkl_path, 'rb') as f:
            self.angles_data = pickle.load(f)

        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filter_orbits = filter_orbits

        # Build a mapping of FFI ID to image file path (much faster than globbing for each entry)
        print("Building image file index...")
        image_files = {}
        for filepath in self.images_dir.iterdir():
            if filepath.suffix in ['.pkl', '.npy']:
                # Extract FFI ID from filename (format: *-XXXXXXXX-*)
                parts = filepath.stem.split('-')
                if len(parts) >= 2:
                    ffi_id = parts[1]  # FFI ID is typically the second part
                    image_files[ffi_id] = filepath

        # Get list of valid samples (where both angle and image exist)
        print("Matching angles with images...")
        self.samples = []
        for ffi_id, angle_dict in self.angles_data.items():
            if ffi_id in image_files:
                # Extract orbit number if filtering is enabled
                if self.filter_orbits is not None:
                    orbit = self._extract_orbit(ffi_id, angle_dict)
                    if orbit is None or orbit not in self.filter_orbits:
                        continue
                self.samples.append((ffi_id, angle_dict, image_files[ffi_id]))

        if self.filter_orbits is not None:
            print(f"Found {len(self.samples)} valid samples for orbits {self.filter_orbits}")
        else:
            print(f"Found {len(self.samples)} valid samples")

        # Extract feature names (excluding metadata)
        self.feature_names = ['1/ED', '1/MD', '1/ED^2', '1/MD^2',
                             'Eel', 'Eaz', 'Mel', 'Maz',
                             'E3el', 'E3az', 'M3el', 'M3az']
        self.num_features = len(self.feature_names)

    def _extract_orbit(self, ffi_id, angle_dict):
        """Extract orbit number from angle_dict"""
        if 'orbit' in angle_dict:
            return int(angle_dict['orbit'])
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ffi_id, angle_dict, image_path = self.samples[idx]

        # Extract angle features
        angles = torch.tensor([angle_dict[name] for name in self.feature_names],
                             dtype=torch.float32)

        # Load image (handle both .pkl and .npy formats)
        if image_path.suffix == '.pkl':
            with open(image_path, 'rb') as f:
                image = pickle.load(f)
        else:
            image = np.load(image_path)

        image = torch.from_numpy(image).float()

        # Images are already preprocessed, just normalize to [0, 1] range
        # Using a robust normalization approach
        image_min = image.min()
        image_max = image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = torch.zeros_like(image)

        # Add channel dimension: (1, H, W)
        image = image.unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return angles, image, ffi_id


class AngleToImageModel(nn.Module):
    """Neural network that predicts images from angle inputs"""

    def __init__(self, num_angle_features=12, image_size=256, hidden_dim=1024):
        super(AngleToImageModel, self).__init__()

        self.num_angle_features = num_angle_features
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        # Encoder for angles
        self.angle_encoder = nn.Sequential(
            nn.Linear(num_angle_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # Decoder to generate image
        # Start with small spatial resolution and upsample
        self.initial_size = 32  # Start at 32x32
        self.fc_to_conv = nn.Linear(hidden_dim, 512 * (self.initial_size // 8) * (self.initial_size // 8))

        # Transposed convolutions to upsample to 256x256
        self.decoder = nn.Sequential(
            # (512, 4, 4) -> (256, 8, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # (256, 8, 8) -> (128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # (128, 16, 16) -> (64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # (64, 32, 32) -> (32, 64, 64)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # (32, 64, 64) -> (16, 128, 128)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # (16, 128, 128) -> (1, 256, 256)
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, angles):
        # Encode angles
        x = self.angle_encoder(angles)

        # Project to conv feature map
        x = self.fc_to_conv(x)

        # Reshape to (batch, channels, height, width)
        batch_size = x.size(0)
        x = x.view(batch_size, 512, self.initial_size // 8, self.initial_size // 8)

        # Decode to image
        x = self.decoder(x)

        return x


def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4, device='cuda',
                output_dir='outputs', resume_checkpoint=None, use_multi_gpu=True):
    """Train the model

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        lr: Learning rate
        device: Device to train on
        output_dir: Directory to save outputs (checkpoints, plots, etc.)
        resume_checkpoint: Path to checkpoint file to resume from (optional)
        use_multi_gpu: Whether to use DataParallel for multi-GPU training
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Multi-GPU setup
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses = []
    val_losses = []
    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint)

        # Handle DataParallel models - load state dict properly
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')

        for angles, images, _ in train_pbar:
            angles, images = angles.to(device), images.to(device)

            optimizer.zero_grad()
            outputs = model(angles)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for angles, images, _ in val_loader:
                angles, images = angles.to(device), images.to(device)
                outputs = model(angles)
                loss = criterion(outputs, images)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            # Save model state dict (handle DataParallel)
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')

    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / 'checkpoint_final.pth'
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, final_checkpoint_path)
    print(f'Saved final checkpoint to {final_checkpoint_path}')

    return train_losses, val_losses


def plot_predictions(model, dataset, num_samples=5, device='cuda', output_dir='outputs'):
    """Plot example predictions

    Args:
        model: The trained model
        dataset: Dataset to sample from
        num_samples: Number of samples to visualize
        device: Device to run inference on
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model = model.to(device)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            angles, true_image, ffi_id = dataset[idx]

            # Predict
            angles_batch = angles.unsqueeze(0).to(device)
            pred_image = model(angles_batch).cpu().squeeze()
            true_image = true_image.squeeze()

            # Plot with colorbar
            im0 = axes[i, 0].imshow(true_image, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title(f'True Image (FFI: {ffi_id})')
            axes[i, 0].axis('off')
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

            im1 = axes[i, 1].imshow(pred_image, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title(f'Predicted Image')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

            # Difference
            diff = torch.abs(true_image - pred_image)
            im2 = axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Absolute Difference')
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    predictions_path = output_dir / 'predictions.png'
    plt.savefig(predictions_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved predictions to {predictions_path}")


def main():
    # Configuration
    angles_pkl = '/pdo/users/djtufto/data/data_tess_4096_raw/angles/tess_angles_O11-54_data_dic.pkl'
    images_dir = '/pdo/users/jlupoiii/TESS/data/processed_images_im256x256/'
    batch_size = 16
    num_epochs = 50
    lr = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Output directory and checkpoint resumption
    output_dir = 'angle_to_image_outputs'
    resume_checkpoint = None  # Set to checkpoint path to resume training, e.g., 'angle_to_image_outputs/checkpoints/checkpoint_epoch_20.pth'
    use_multi_gpu = False  # Set to False to use single GPU

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Create datasets
    print("Loading datasets...")

    # Test set: only orbits 53 and 54
    test_dataset = AngleToImageDataset(angles_pkl, images_dir, filter_orbits=[53, 54])

    # Train/Val set: all orbits except 53 and 54
    # First get all orbits in the dataset
    temp_dataset = AngleToImageDataset(angles_pkl, images_dir)
    all_orbits = set()
    for _, angle_dict, _ in temp_dataset.samples:
        orbit = temp_dataset._extract_orbit(None, angle_dict)
        if orbit is not None:
            all_orbits.add(orbit)

    train_val_orbits = sorted(all_orbits - {53, 54})
    print(f"Train/Val orbits: {train_val_orbits}")
    print(f"Test orbits: [53, 54]")

    # Create train/val dataset excluding test orbits
    train_val_dataset = AngleToImageDataset(angles_pkl, images_dir, filter_orbits=train_val_orbits)

    # Split train/val dataset into train and val
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    # Create model
    print("Creating model...")
    model = AngleToImageModel(num_angle_features=train_val_dataset.num_features, image_size=256)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        num_epochs=num_epochs, lr=lr, device=device,
        output_dir=output_dir, resume_checkpoint=resume_checkpoint,
        use_multi_gpu=use_multi_gpu
    )

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    curves_path = output_dir_path / 'training_curves.png'
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {curves_path}")

    # Plot predictions on test set
    print("Plotting predictions on test set...")
    # Extract base model if using DataParallel
    plot_model = model.module if isinstance(model, nn.DataParallel) else model
    plot_predictions(plot_model, test_dataset, num_samples=5, device=device, output_dir=output_dir)

    # Save final model weights
    model_path = output_dir_path / 'angle_to_image_model.pth'
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(model_state, model_path)
    print(f"Saved model weights to {model_path}")


if __name__ == '__main__':
    main()
