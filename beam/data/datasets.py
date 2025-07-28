"""
Dataset classes for TESS image data.

This module contains dataset implementations for loading and processing
TESS (Transiting Exoplanet Survey Satellite) image data.
"""

import os
import time
import pickle
import multiprocessing
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
        background_path: str,
        image_shape: Tuple[int, int],
        patch_size: Optional[Tuple[int, int]] = None,
        repeat_factor: int = 1
    ):
        start_time = time.time()
        # Define paths and parameters
        self.ccd_folder = ccd_folder
        self.image_shape = image_shape
        self.angle_path = angle_path
        self.background_path = background_path

        self.length = 0
        self.patch_size = patch_size
        self.repeat_factor = repeat_factor

     
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
        self.background_dic = {int(path[1:-19]): path for path in os.listdir(self.background_path) if path.endswith('.npy')}

        self.MEAN = 0.1154092
        self.STD =  0.2346011
        
        # Find all valid image files that have corresponding angle data
        # store files for use in __getitem__
        self.files = []
        self.ffi_nums = []

        # Load all files in the ccd_folder that have corresponding angle data 
        for filename in os.listdir(self.ccd_folder):
            ffi_num = filename[18:18+8]
            if ffi_num in self.angles_dic.keys():
                self.files.append(filename)
                self.ffi_nums.append(ffi_num)
                self.length += 1

        # Convert to tuples to prevent reordering
        self.files = tuple(self.files)
        self.ffi_nums = tuple(self.ffi_nums)
        
        end_time = time.time()
        print(f"Dataset built with {self.length} samples in {end_time - start_time:.2f} seconds")

    def _get_cache(self):
        if self._cache is None:
            # one cache per worker process
            self._cache = {}
        return self._cache

    def _mmap(self, path):
        cache = self._get_cache()
        arr = cache.get(path)
        if arr is None:
            # load once per worker, keep the memmap open
            arr = np.load(path, mmap_mode='r')
            cache[path] = arr
        return arr

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
        file_path = os.path.join(self.ccd_folder, self.files[idx])
        ffi_num = self.ffi_nums[idx]
        

        # Load image data
        # image_arr = pickle.load(open(file_path, "rb"))
        angles = self.angles_dic[ffi_num]
        orbit = angles['orbit']
        background_path = os.path.join(self.background_path, self.background_dic[int(orbit)])
        # Prepare orbital parameters (12 values)
        params = np.array([
            angles['1/ED'], angles['1/MD'], 
            angles['1/ED^2'], angles['1/MD^2'], 
            angles['Eel'], angles['Eaz'], 
            angles['Mel'], angles['Maz'], 
            angles['E3el'], angles['E3az'], 
            angles['M3el'], angles['M3az']
        ])
        

        # Convert to images for consistent processing
        # angles_image = torch.from_numpy(params).unsqueeze(0)

        
        # ffi_image = Image.fromarray(image_arr.flatten())
        # Define transformations
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     lambda s: s.reshape(1, angles_image.size[1])  # Reshape to 1×N tensor
        # ])
        # print()
        # target_transform = transforms.Compose([
        #     lambda s: np.array(s),
        #     lambda s: s.reshape(self.image_shape),  # Reshape to the target image size
        #     transforms.ToTensor(),
        # ])

        # # Apply transformations
        # angles_image = transform(angles_image)
   
        if self.patch_size is not None:
            background_arr = np.load(background_path, mmap_mode='r')
            image_arr = np.load(file_path, mmap_mode='r')
            params = params.astype(np.float32, copy=False)
            angles_image = torch.from_numpy(params).unsqueeze(0)
            # Return one random patch:
            patch_idx = torch.randint(self.num_patches, (1,))
            # patch_idx = torch.tensor(0)
            
            patch_row, patch_col = divmod(patch_idx.item(), self.patch_x)
            conditioning_loc = torch.cat([self.row_embeds[patch_row], self.col_embeds[patch_col]], dim=-1).unsqueeze(0)
            top, left = patch_row*self.ph, patch_col*self.pw
            ffi_image = image_arr[top:top+self.ph, left:left+self.pw]
            angles_image = torch.cat([angles_image, conditioning_loc], dim=1)
            background_image = background_arr[top:top+self.ph, left:left+self.pw].astype(np.float32, copy=False)
            background_image = torch.from_numpy(1/633118 * background_image).unsqueeze(0)
            ffi_image = torch.from_numpy(ffi_image).unsqueeze(0) - background_image
            ffi_image = (ffi_image-self.MEAN)/self.STD
            # #Compute all patches
            # ffi_image = torch.from_numpy(image_arr).unsqueeze(0)
            # num_patches = (self.image_shape[0]//self.patch_size[0])**2 #assume square images and patches
            # angles_image = angles_image.expand(num_patches, -1) # N x 12
            # locs = 1/num_patches*torch.arange(1, num_patches+ 1).unsqueeze(1) # N x 1 with values in [0, 1]
            # angles_image = torch.cat([angles_image, locs], dim=1)
            
            # ph, pw = self.patch_size
            # sh, sw = self.patch_size 

            # # # Unfold the image into patches 
            # patched_ffi = ffi_image.unfold(1, ph, sh).unfold(2,pw,sw)
            # patch_shape = patched_ffi.shape #Need for reshaping back 

            # ffi_image = patched_ffi.contiguous().view(-1, ph, pw) # 1 x num_pixels/(ph*pw) x ph x pw; 
        else:
            image_arr = pickle.load(open(file_path, "rb"))
            ffi_image = Image.fromarray(image_arr.flatten())
            angles_image = Image.fromarray(params)
            # Define transformations
            transform = transforms.Compose([
                transforms.ToTensor(),
                lambda s: s.reshape(1, angles_image.size[1])  # Reshape to 1×N tensor
            ])
            target_transform = transforms.Compose([
                lambda s: np.array(s),
                lambda s: s.reshape(self.image_shape),  # Reshape to the target image size
                transforms.ToTensor(),
            ])

            # Apply transformations
            angles_image = transform(angles_image)
            ffi_image = target_transform(ffi_image)

        return {
            "x": angles_image,       # Orbital parameters (1×12 vector)
            "y": ffi_image,          # Image (64×64 or other size)
            "ffi_num": ffi_num,      # FFI identification number
            "orbit": orbit, # Orbit number
            }

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



#CURRENTLY NOT USED
# def create_train_valid_datasets(
#     dataset: Dataset, 
#     train_split_criteria: callable, 
#     valid_split_criteria: callable
# ) -> Tuple[Subset, Subset]:
#     """
#     Create training and validation datasets from a full dataset.
    
#     Args:
#         dataset: The complete dataset
#         train_split_criteria: Function to determine if a sample belongs to the training set
#         valid_split_criteria: Function to determine if a sample belongs to the validation set
        
#     Returns:
#         Tuple of (training dataset, validation dataset)
#     """
#     train_indices = [
#         idx for idx in range(len(dataset)) 
#         if train_split_criteria(dataset[idx])
#     ]
    
#     valid_indices = [
#         idx for idx in range(len(dataset)) 
#         if valid_split_criteria(dataset[idx])
#     ]
    
#     train_dataset = Subset(dataset, train_indices)
#     valid_dataset = Subset(dataset, valid_indices)
    
#     return train_dataset, valid_dataset


def create_train_valid_datasets_by_orbit(dataset: TESSDataset, orbit_threshold: int = 47):
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
        if int(dataset.angles_dic[ffi_num]["orbit"]) > orbit_threshold
    ]
    train_dataset = Subset(dataset, train_indices * dataset.repeat_factor)
    valid_dataset = Subset(dataset, valid_indices * dataset.repeat_factor)
    return train_dataset, valid_dataset


class TESS_4096_original_images(Dataset):
    """
    Dataset for raw original 4096x4096 TESS images.
    """
    def __init__(self):
        self.ffi_to_fits_filepath = {}
        
        # Build index of fits files
        fits_folder_paths = [f"/pdo/users/roland/SL_data/O{i}_data/" for i in range(9, 63)]

        for fits_folder_path in fits_folder_paths:
            orbit = fits_folder_path[27:29] if fits_folder_path[29] == '_' else fits_folder_path[27:28]
            for fits_filename in os.listdir(fits_folder_path):
                # Only include camera 3 FITS files
                if (len(fits_filename) > 40 and 
                    fits_filename[-7:] == 'fits.gz' and 
                    fits_filename[27] == '3'):
                    ffi_num = fits_filename[18:26]
                    self.ffi_to_fits_filepath[ffi_num] = os.path.join(fits_folder_path, fits_filename)

    def __len__(self) -> int:
        return len(self.ffi_to_fits_filepath)

    def __getitem__(self, ffi_num: str) -> torch.Tensor:
        # Rows and columns to remove from the image (black areas)
        rows_to_delete = range(2048, 2108)
        columns_to_delete = list(range(0, 44)) + list(range(2092, 2180)) + list(range(4228, 4272))
        
        # Load and process image
        image = fits.getdata(self.ffi_to_fits_filepath[ffi_num], ext=0)
        image = np.delete(image, rows_to_delete, axis=0)
        image = np.delete(image, columns_to_delete, axis=1)
        image = image.astype(np.float64)
        image *= 1/633118  # Scale values to reasonable range
        
        return torch.tensor(image)


class TESS_4096_processed_images(Dataset):
    """
    Dataset for pre-processed 4096x4096 TESS images.
    """
    def __init__(self):
        self.ffi_to_pkl_filepath = {}
        
        pkl_folder_path = "/pdo/users/jlupoiii/TESS/data/processed_images_im4096x4096/"
        for pkl_filename in os.listdir(pkl_folder_path):
            ffi_num = pkl_filename[18:26]
            self.ffi_to_pkl_filepath[ffi_num] = os.path.join(pkl_folder_path, pkl_filename)

    def __len__(self) -> int:
        return len(self.ffi_to_pkl_filepath)

    def __getitem__(self, ffi_num: str) -> torch.Tensor:
        with open(self.ffi_to_pkl_filepath[ffi_num], 'rb') as file:
            return pickle.load(file)



