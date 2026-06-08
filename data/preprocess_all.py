import argparse
from typing import List, Optional
import os
import numpy as np
import pickle
from astropy.io import fits
import matplotlib.pyplot as plt
import yaml
import multiprocessing
from tqdm import tqdm
import re

SECTOR_TO_ORBIT = {
    1: [9, 10], 2: [11, 12], 3: [13, 14], 4: [15, 16], 5: [17, 18],
    6: [19, 20], 7: [21, 22], 8: [23, 24], 9: [25, 26], 10: [27, 28],
    11: [29, 30], 12: [31, 32], 13: [33, 34], 14: [35, 36], 15: [37, 38],
    16: [39, 40], 17: [41, 42], 18: [43, 44], 19: [45, 46], 20: [47, 48],
    21: [49, 50], 22: [51, 52], 23: [53, 54], 24: [55, 56], 25: [57, 58],
    26: [59, 60], 27: [61, 62], 28: [63, 64], 29: [65, 66], 30: [67, 68],
    31: [69, 70], 32: [71, 72], 33: [73, 74], 34: [75, 76], 35: [77, 78],
    36: [79, 80], 37: [81, 82], 38: [83, 84], 39: [85, 86], 40: [87, 88],
    41: [89, 90], 42: [91, 92], 43: [93, 94], 44: [95, 96], 45: [97, 98],
    46: [99, 100], 47: [101, 102], 48: [103, 104], 49: [105, 106], 50: [107, 108],
    51: [109, 110], 52: [111, 112], 53: [113, 114], 54: [115, 116], 55: [117, 118],
    56: [119, 120], 57: [121, 122], 58: [123, 124], 59: [125, 126], 60: [127, 128],
    61: [129, 130], 62: [131, 132], 63: [133, 134], 64: [135, 136], 65: [137, 138],
    66: [139, 140], 67: [141, 142], 68: [143, 144], 69: [145, 146], 70: [147, 148],
    71: [149, 150], 72: [151, 152], 73: [153, 154], 74: [155, 156], 75: [157, 158],
    76: [159, 160], 77: [161, 162], 78: [163, 164], 79: [165, 166], 80: [167, 168],
    81: [169, 170], 82: [171, 172], 83: [173, 174], 84: [175, 176], 85: [177, 178],
    86: [179, 180], 87: [181, 182], 88: [183, 184], 89: [185, 186], 90: [187, 188],
    91: [189, 190], 92: [191, 192], 93: [193, 194], 94: [195, 196], 95: [197, 198],
    96: [199, 200], 97: [201, 202, 203, 204], 98: [205, 206, 207, 208],
}

ORBIT_TO_SECTOR = {o: s for s, orbits in SECTOR_TO_ORBIT.items() for o in orbits}


def show_img(
    arr: np.ndarray,
    title: str = "",
    vmin: float = 0,
    vmax: float = 1,
    save_path: Optional[str] = None,
):
    """
    Args: the numpy array we want to show
            the title of the image (optional)
            the minimum value of the colorbar (optional)
            the maximum value of the colorbar (optional)
            the path to save the image (optional)
    """
    fig, ax = plt.subplots()
    im = ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax)
    plt.grid(visible=False)
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()





def get_fits_folders_from_root(
    root_folder: str,
    sector_start: int,
    sector_end: int,
    skip_sectors: Optional[List[int]] = None,
) -> List[str]:
    """
    Returns a list of all desired folders containing .fits files in the given folder.
    """
    skip_set = set(skip_sectors or [])
    folders = []
    for i in range(sector_start, sector_end + 1):
        if i in skip_set:
            continue
        sector_folder = os.path.join(root_folder, f"s{i:04d}")
        if not os.path.isdir(sector_folder):
            sector_folder = os.path.join(root_folder, f"sector{i}")
        folders.append(sector_folder)
    return folders


def parse_tica_filename(fits_filename: str) -> Optional[tuple[str, int, int, int]]:
    match = re.search(
        r"hlsp_tica_tess_ffi_s(\d+)-(?:o\d+-)?(\d{8})-cam(\d)-ccd(\d)",
        fits_filename,
    )
    if not match:
        return None
    sector, ffi_num, camera, ccd = match.groups()
    return ffi_num, int(sector), int(camera), int(ccd)


class Preprocessing:
    def __init__(
        self,
        fits_root_folder,
        angle_folder,
        ccd_folder,
        background_ccd_folder,
        angles_dic,
        display_images_folder,
        image_size,
        num_workers=1,
        sector_start=9,
        sector_end=64,
        camera_number='3',
        skip_sectors: Optional[List[int]] = None,
    ):
        self.skip_sectors = {int(o) for o in (skip_sectors or [])}
        self.fits_root_folder = fits_root_folder
        self.angle_folder = angle_folder
        self.ccd_folder = ccd_folder
        self.background_ccd_folder = background_ccd_folder
        self.angles_dic = angles_dic
        self.display_images_folder = display_images_folder
        self.image_size = image_size
        self.num_workers = num_workers
        self.data_dic = {}  # Dictionary of ffi->orbit->data stored in the angle file
        self.sector_start = sector_start
        self.sector_end = sector_end
        self.camera_number = camera_number
        self.skipped_sectors = set()
        # creates folders if they don't exist
        os.makedirs(self.ccd_folder, exist_ok=True)
        if self.display_images_folder:
            os.makedirs(self.display_images_folder, exist_ok=True)
        # os.makedirs(self.angle_folder, exist_ok=True)
        os.makedirs(self.background_ccd_folder, exist_ok=True)

        # Generate fits folders
        if self.skip_sectors:
            print(f"Skipping sectors: {sorted(self.skip_sectors)}")
        self.fits_folder_paths = get_fits_folders_from_root(
            self.fits_root_folder, self.sector_start, self.sector_end, skip_sectors=self.skip_sectors
        )
        if not self.fits_folder_paths:
            print(
                f"Warning: No folders containing .fits files found in {self.fits_root_folder}!"
            )

    def get_arr(self, fits_filename: str, fits_folder_path: str) -> np.ndarray:
        """
        Args: the name of the fits file we need to get info out of
                the path to the fodler where the fits file is in
        Returns: numpy array of four ccd images
        """
        return fits.getdata(os.path.join(fits_folder_path, fits_filename), ext=0)

    def get_camera_arr(self, fits_filename: str, sector_folder_path: str) -> np.ndarray:
        parsed = parse_tica_filename(fits_filename)
        if parsed is None:
            return self.get_arr(fits_filename, sector_folder_path)

        ffi_num, _, camera, _ = parsed
        ccd_arrays = []
        for ccd in range(1, 5):
            ccd_folder = os.path.join(sector_folder_path, f"cam{camera}-ccd{ccd}")
            ccd_filename = fits_filename.replace(
                f"-cam{camera}-ccd{parsed[3]}_",
                f"-cam{camera}-ccd{ccd}_",
            )
            ccd_path = os.path.join(ccd_folder, ccd_filename)
            if not os.path.exists(ccd_path):
                matches = [
                    name
                    for name in os.listdir(ccd_folder)
                    if f"-{ffi_num}-cam{camera}-ccd{ccd}_" in name
                    and name.endswith((".fits", ".fits.gz"))
                ]
                if not matches:
                    raise FileNotFoundError(
                        f"No CCD {ccd} FITS for FFI {ffi_num} in {ccd_folder}"
                    )
                ccd_filename = sorted(matches)[0]
            ccd_arrays.append(self.get_arr(ccd_filename, ccd_folder))

        ims = np.block([[ccd_arrays[2], ccd_arrays[3]], [np.flip(ccd_arrays[1]), np.flip(ccd_arrays[0])]]) 
        return ims

    def trim_and_downsample(
        self,
        arr: np.ndarray,
        rows_to_delete,
        columns_to_delete,
    ) -> np.ndarray:
        if arr.shape[0] - len(rows_to_delete) == 4096:
            arr = np.delete(arr, rows_to_delete, axis=0)
        if arr.shape[1] - len(columns_to_delete) == 4096:
            arr = np.delete(arr, columns_to_delete, axis=1)

        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Expected square image after trimming, got {arr.shape}")
        if arr.shape[0] % self.image_size != 0:
            raise ValueError(
                f"Image shape {arr.shape} is not divisible by image_size={self.image_size}"
            )

        arr = arr.astype(np.float32)
        block = arr.shape[0] // self.image_size
        return np.median(
            arr.reshape(self.image_size, block, self.image_size, block), axis=(1, 3)
        )

    def load_angle_data(self) -> None:
        angle_path = os.path.join(self.angle_folder, self.angles_dic)
        with open(angle_path, "rb") as file:
            self.data_dic = pickle.load(file)
        print(f"Loaded angle file from: {angle_path}")

    # BACKGROUND CREATION
    def create_backgrounds_by_sector(self) -> None:
        """
        Args: None
        Returns: None
        Parallelizes over sectors via calls to background_sector_worker
        """
        # Same as in the notebook, but parallelized over sectors
        rows_to_delete = range(2048, 2108)
        columns_to_delete = (
            list(range(0, 44)) + list(range(2092, 2180)) + list(range(4228, 4272))
        )
        sector_to_folder = {}
        for i, folder_path in enumerate(self.fits_folder_paths):
            sector = int(os.path.basename(os.path.normpath(folder_path)).lstrip("sS").replace("ector", ""))
            sector_to_folder[sector] = (folder_path, i)
        sector_args = []
        for sector, (folder_path, i) in sector_to_folder.items():
            sector_args.append(
                (self, folder_path, sector, i, rows_to_delete, columns_to_delete)
            )
        with multiprocessing.Pool(self.num_workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(Preprocessing.background_sector_worker, sector_args),
                total=len(sector_args),
                desc="Creating Backgrounds (by sector)",
            ):
                continue

    @staticmethod
    def background_sector_worker(args) -> None:
        """
        Args: the path to the folder where the fits files are in
                the sector number
                the index of the folder
                the rows to delete
                the columns to delete
        Returns: None
        Processes background images in a given sector sequentially
        """
        self, folder_path, sector, i, rows_to_delete, columns_to_delete = args
        # Collect all FFI numbers for this sector that are below the sunshade
        ffis_in_sector_below_sunshade = [
            ffi
            for ffi, data in self.data_dic.items()
            if ORBIT_TO_SECTOR[int(data["orbit"])] == int(sector) and data["below_sunshade"]
        ]
        if not ffis_in_sector_below_sunshade:
            print(f"No images below sunshade in sector {sector} (ffis_in_sector_below_sunshade)")
            return
        args = []
        ccd1_folder = os.path.join(folder_path, f"cam{self.camera_number}-ccd1")
        if not os.path.isdir(ccd1_folder):
            print(f"Missing folder for sector {sector}: {ccd1_folder}")
            return
        for fits_filename in os.listdir(ccd1_folder):
            parsed = parse_tica_filename(fits_filename)
            if parsed is None:
                continue
            ffi_num, _, camera, _ = parsed
            if (
                ffi_num in ffis_in_sector_below_sunshade
                and str(camera) == str(self.camera_number)
            ):
                args.append(
                    (
                        self,
                        fits_filename,
                        folder_path,
                        rows_to_delete,
                        columns_to_delete,
                    )
                )
        if not args:
            print(f"No images below sunshade in sector {sector} (args)")
            return
        # Sequentially process background images in this sector
        print(f"Number of images below sunshade in sector {sector}: {len(args)}")
        running_sum = None
        count = 0
        for arg in args:
            result = Preprocessing.background_image_worker(arg)
            if result is not None:
                if running_sum is None:
                    running_sum = result.astype(np.float64)
                else:
                    running_sum += result
                count += 1
        if count == 0:
            print(f"No images below sunshade in sector {sector} (len 0)")
            return
        arr_average = running_sum / count
        out_path = os.path.join(
            self.background_ccd_folder, f"S{int(sector):04d}_background_ccd.pkl"
        )
        with open(out_path, "wb") as file:
            pickle.dump(arr_average, file)
            print(f"Saved average background image to {str(file.name)}")
        print(f"{count} images averaged in sector {sector}")
        if self.display_images_folder and i == 0:
            show_img(
                arr_average,
                title=f"Average Sector {sector} image below sunshade",
                vmin=0,
                vmax=633118,
                save_path=os.path.join(
                    self.display_images_folder, f"avg_sector_{int(sector):04d}_below_sunshade.png"
                ),
            )

    @staticmethod
    def background_image_worker(args) -> np.ndarray:
        """
        Args: the name of the fits file we need to get info out of
                the path to the fodler where the fits file is in
                the rows to delete
                the columns to delete
        Returns: None
        Saves the processed image to the ccd_folder
        """
        self, fits_filename, folder_path, rows_to_delete, columns_to_delete = args
        if fits_filename.endswith((".fits", ".fits.gz")):
            arr = self.get_camera_arr(fits_filename, folder_path)
            return self.trim_and_downsample(arr, rows_to_delete, columns_to_delete)
        # else:
        #     print(f"{fits_filename} is not a fits file") 

    # IMAGE PROCESSING
    def images_processing_by_sector(self) -> None:
        """
        Args: None
        Returns: None
        Creates a process for each sector to process the images in that sector in parallel
        via calls to image_sector_worker
        """
        print(f"Processing images in {len(self.fits_folder_paths)} sectors")
        rows_to_delete = range(2048, 2108)
        columns_to_delete = (
            list(range(0, 44)) + list(range(2092, 2180)) + list(range(4228, 4272))
        )
        sector_to_folder = {}
        skipped_sectors = set()

        for i, fits_folder in enumerate(self.fits_folder_paths):
            sector = int(os.path.basename(os.path.normpath(fits_folder)).lstrip("sS").replace("ector", ""))
            sector_to_folder[sector] = (fits_folder, i)
        sector_args = []
        for sector, (folder_path, i) in sector_to_folder.items():
            sector_args.append(
                (self, folder_path, sector, i, rows_to_delete, columns_to_delete)
            )
        with multiprocessing.Pool(self.num_workers) as pool:
            for result in pool.imap_unordered(Preprocessing.image_sector_worker, sector_args):
                if result is not None:
                    skipped_sectors.add(result)
        print(f"Skipped sectors: {sorted(skipped_sectors)}")

    @staticmethod
    def image_sector_worker(args) -> None:
        """
        Args: the path to the folder where the fits files are in
                the sector number
                the index of the folder
                the rows to delete
                the columns to delete
        Returns: None
        Processes images in a given sector sequentially (no threading)
        """
        self, folder_path, sector, i, rows_to_delete, columns_to_delete = args
        image_args = []
        bg_path = os.path.join(
            self.background_ccd_folder, f"S{int(sector):04d}_background_ccd.pkl"
        )
        if not os.path.exists(bg_path):
            print(f"Warning: No background for sector {sector}. Skipping.")
            return sector
        ccd1_folder = os.path.join(folder_path, f"cam{self.camera_number}-ccd1")
        if not os.path.isdir(ccd1_folder):
            print(f"Missing folder for sector {sector}: {ccd1_folder}")
            return sector
        # Only process one representative file per FFI; image_worker loads ccd1..ccd4.
        for fits_filename in os.listdir(ccd1_folder):
            parsed = parse_tica_filename(fits_filename)
            if parsed is None:
                continue
            _, _, camera, _ = parsed
            if str(camera) == str(self.camera_number):
                image_args.append(
                    (
                        self,
                        fits_filename,
                        folder_path,
                        rows_to_delete,
                        columns_to_delete,
                    )
                )
        if not image_args:
            return None
        # Sequentially process images in this sector
        for arg in tqdm(
            image_args,
            total=len(image_args),
            desc=f"Processing Sector {sector} Images",
            position=i,
            leave=False,
        ):
            Preprocessing.image_worker(arg)
        

    @staticmethod
    def image_worker(args) -> None:
        """
        Args: the name of the fits file we need to get info out of
                the path to the fodler where the fits file is in
                the rows to delete
                the columns to delete
        Returns: None
        Saves the processed image to the ccd_folder
        """
        self, fits_filename, folder_path, rows_to_delete, columns_to_delete = args
        try:
            parsed = parse_tica_filename(fits_filename)
            if parsed is None:
                print(f"Could not parse FITS filename {fits_filename}")
                return
            ffi_num, sector, _, _ = parsed
            # Only process if ffi_num is in self.data_dic
            if ffi_num not in self.data_dic:
                print(f"No angle data for {ffi_num}")
                return  # Skip processing if no angle data
            # if self.data_dic[ffi_num]["below_sunshade"] == False:
            #     return  # Skip processing if image is not below sunshade
            # elif self.data_dic[ffi_num]["below_sunshade"] == False:
            #     print(f"Image {ffi_num} is not below sunshade")
            #     return  # Skip processing if image is not below sunshade
            arr = self.get_camera_arr(fits_filename, folder_path)
            arr = self.trim_and_downsample(arr, rows_to_delete, columns_to_delete)

            bg_path = os.path.join(
                self.background_ccd_folder, f"S{int(sector):04d}_background_ccd.pkl"
            )
            with open(bg_path, "rb") as file:
                background_avg_image = pickle.load(file)
            arr = arr - background_avg_image
                
            # Pixel scaling
            scale_factor = 1 / 633118 / 5.3
            arr *= scale_factor
            # Save debug images if requested
            if self.display_images_folder and not os.listdir(
                self.display_images_folder
            ):
                show_img(
                    arr,
                    title=f"{ffi_num} Processed",
                    vmin=0,
                    vmax=1,
                    save_path=os.path.join(
                        self.display_images_folder,
                        f"{ffi_num}_im{self.image_size}x{self.image_size}.png",
                    ),
                )
            # Save as pickle
            out_path = os.path.join(
                self.ccd_folder,
                f"{fits_filename.rsplit('.', 2)[0]}_processed_im{self.image_size}x{self.image_size}",
            )

            with open(out_path, "wb") as file:
                pickle.dump(arr, file)
            # np.save(out_path, arr, allow_pickle=False)
        except Exception as e:
            print(f"Error processing {fits_filename}: {e}")

    def run(self, process_images=True, make_background=True):
        if make_background or process_images:
            self.load_angle_data()
        if make_background:
            self.create_backgrounds_by_sector()
        if process_images:
            self.images_processing_by_sector()


def main():
    parser = argparse.ArgumentParser(
        description="Consolidated TESS preprocessing script for all image sizes (YAML config version, with multiprocessing support)."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--no_images", action="store_true", help="Skip image processing"
    )
    parser.add_argument(
        "--no_backgrounds",
        action="store_true",
        help="Do not create new background images, use existing ones only",
    )
    args = parser.parse_args()

    # Load config from YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    preproc = Preprocessing(
        fits_root_folder=config["fits_folder"],
        angle_folder=config["angle_folder"],
        ccd_folder=config["ccd_folder"],
        background_ccd_folder=config["background_ccd_folder"],
        angles_dic=config["angles_dic"],
        display_images_folder=config.get("display_images_folder", ""),
        image_size=config["image_size"],
        num_workers=config.get("num_workers", 1),
        sector_start=config.get("sector_start", config.get("orbit_start", 1)),
        sector_end=config.get("sector_end", config.get("orbit_end", 32)),
        camera_number=config.get("camera_number", "1"),
        skip_sectors=config.get("skip_sectors", config.get("skip_orbits", [])),
    )
    # CLI flags
    preproc.run(
        make_background=not args.no_backgrounds,
        process_images=not args.no_images,
    )


if __name__ == "__main__":
    main()
