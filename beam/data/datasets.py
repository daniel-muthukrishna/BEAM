"""
Dataset classes for TESS image data.

This module contains dataset implementations for loading and processing
TESS (Transiting Exoplanet Survey Satellite) image data.  
"""

import os
import time
import pickle
from typing import Dict, Tuple, Optional, Union, List

import numpy as np
import math
import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from PIL import Image
from astropy.io import fits


class TESSDataset(Dataset):
    """
    Dataset for TESS (Transiting Exoplanet Survey Satellite) images and their
    corresponding orbital parameters.
    """
    def __init__(
        self,
        angle_path: str,
        ccd_folder: str,
        image_shape: Tuple[int, int],
        mean: float,
        std: float,
        patch_size: Optional[Tuple[int, int]] = None,
        repeat_factor: int = 1,
        camera_number: str = '3',
    ):
        start_time = time.time()
        # Define paths and parameters
        self.ccd_folders = list(ccd_folder) if isinstance(ccd_folder, (list, tuple)) else [ccd_folder]
        self.image_shape = image_shape
        self.angle_path = angle_path

        self.length = 0
        self.patch_size = patch_size
        self.repeat_factor = repeat_factor

        self.camera_number = camera_number

     
        if self.patch_size is not None:
            assert self.image_shape[0] % self.patch_size[0] == 0, "Image shape must be divisible by patch size"

            self.ph, self.pw = self.patch_size
            self.patch_x = self.image_shape[0]//self.pw
            self.patch_y = self.image_shape[1]//self.ph
            self.num_patches = self.patch_x * self.patch_y #assume square images and patches so y = x
            self.embed_dim = 6
            self.row_embeds = embed_patch(torch.arange(self.patch_x), self.embed_dim)
            self.col_embeds = embed_patch(torch.arange(self.patch_y), self.embed_dim)
      
        
    
        # Load orbital parameter dictionary
        self.angles_dic = pickle.load(open(self.angle_path, "rb"))
        self.MEAN = mean
        self.STD = std
        
        # Find all valid image files that have corresponding angle data
        # store files for use in __getitem__
        self.files = []        # full image paths
        self.ffi_nums = []
        self.cameras = []      
        for folder in self.ccd_folders:
            print(len(os.listdir(folder)))
            for filename in os.listdir(folder):
                ffi_num, camera = parse_ffi_camera(filename)
                if ffi_num in self.angles_dic.keys():
                    self.files.append(os.path.join(folder, filename))
                    self.ffi_nums.append(ffi_num)
                    self.cameras.append(camera)
                    self.length += 1
      
        
        end_time = time.time()
        print(f"Dataset built with {self.length} samples in {end_time - start_time:.2f} seconds")

    def _get_mmap(self, path, cache):
        mm = cache.get(path)
        if mm is None:
            mm = np.load(path, mmap_mode='r', allow_pickle=True)  
            cache[path] = mm
        return mm

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.length * self.repeat_factor 
        
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary with:
                - x: Orbital parameters tensor
                - y: Image tensor
                - ffi_num: FFI identification number
                - orbit: Orbit number
        """
        file_path = self.files[idx]  # already a full path
        ffi_num = self.ffi_nums[idx]
        camera = self.cameras[idx]   #"1".."4"

        # Load image data
        angles = self.angles_dic[ffi_num]
        orbit = angles['orbit']

        # Prepare orbital parameters (12 values), using this sample's camera columns
        params = np.array([
            angles['1/ED'], angles['1/MD'],
            angles['1/ED^2'], angles['1/MD^2'],
            angles['Eel'], angles['Eaz'],
            angles['Mel'], angles['Maz'],
            angles['E' + camera + 'el'], angles['E' + camera + 'az'],
            angles['M' + camera + 'el'], angles['M' + camera + 'az']
        ])
        
        image_arr = pickle.load(open(file_path, "rb")) if self.patch_size is None else np.load(file_path, mmap_mode='r', allow_pickle=True)            

        # Apply patch if patch_size is not None
        if self.patch_size is not None:
            patch_idx = torch.randint(self.num_patches, (1,))
            patch_row, patch_col = divmod(patch_idx.item(), self.patch_x)
            top, left = patch_row*self.ph, patch_col*self.pw
            image_arr = image_arr[top:top+self.ph, left:left+self.pw]
            conditioning_loc = torch.cat([self.row_embeds[patch_row], self.col_embeds[patch_col]], dim=-1).unsqueeze(0)

        ffi_image = Image.fromarray(image_arr.flatten())
        angles_image = Image.fromarray(params)

        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            lambda s: s.reshape(1, angles_image.size[1])  # Reshape to 1×N tensor
        ])
        target_transform = transforms.Compose([
            lambda s: np.array(s),
            lambda s: s.reshape(self.image_shape if self.patch_size is None else self.patch_size),  # Reshape to the target image size
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
        ])
     

        # Apply transformations
        angles_image = transform(angles_image)
        ffi_image = target_transform(ffi_image)

        if int(orbit) > 61:
            ffi_image.mul_(3)

        # angles_image = torch.zeros_like(angles_image)
        # angles_image = torch.cat([angles_image, conditioning_loc], dim=-1)
        return {
            "x": angles_image,       # Orbital parameters (1×12 vector)
            "y": ffi_image,          # Image (64×64 or other size)
            "ffi_num": ffi_num,      # FFI identification number
            "orbit": orbit, # Orbit number
            "cam": torch.tensor(int(camera) - 1, dtype=torch.long),  # 0-based camera index
            }

    

def parse_ffi_camera(filename: str) -> Tuple[str, str]:
    """
    Parse the FFI number and camera from a TICA filename, e.g.
    'hlsp_tica_tess_ffi_s0002-o1-00006084-cam1-ccd1_tess_v01_img_processed_im256x256'
    -> ffi_num='00006084', camera='1'.

    The fields are dash-separated: [..., orbit, ffi_num, camN, ccdM, ...].
    """
    parts = filename.split('-')
    ffi_num = parts[2]
    camera = parts[3].replace('cam', '')
    return ffi_num, camera


def embed_patch(prow, embed_dim):
    """
    Embed a patch location into a vector of dimension embed_dim. (1 row and col embedding)
    Args:
        prow: 1d tensor of patch row indices
        embed_dim: Dimension of the embedding
    Returns:
        Embedding of the patch location with dimension embed_dim. 
    """

    half = embed_dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
    )  
    prow = prow.unsqueeze(1) # [N, 1]
    row_scaled = prow * freqs[None] # [N, half]

    return torch.cat([
        torch.sin(row_scaled),
        torch.cos(row_scaled),
    ], dim=-1)



def create_train_valid_datasets_by_orbit(dataset: TESSDataset, orbit_threshold: int = 90, max_orbit: int = 100):
    """
    Create training and validation datasets from a TESSDataset using orbit number as the split criterion.
    Orbits <= orbit_threshold go to training, orbits > orbit_threshold go to validation.
    
    Args:
        dataset: The complete TESSDataset
        orbit_threshold: Orbit number threshold for splitting
    
    Returns:
        Tuple of (training dataset, validation dataset)
    """
    train_indices = [
        idx for idx, ffi_num in enumerate(dataset.ffi_nums)
        if int(dataset.angles_dic[ffi_num]["orbit"]) <= orbit_threshold
    ]
    valid_indices = [
        idx for idx, ffi_num in enumerate(dataset.ffi_nums)
        if int(dataset.angles_dic[ffi_num]["orbit"]) > orbit_threshold and int(dataset.angles_dic[ffi_num]["orbit"]) <= max_orbit
    ]
    train_dataset = Subset(dataset, train_indices * dataset.repeat_factor)
    valid_dataset = Subset(dataset, valid_indices * dataset.repeat_factor)
    return train_dataset, valid_dataset




