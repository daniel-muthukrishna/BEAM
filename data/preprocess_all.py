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


def get_angle_files_from_folder(
    folder: str, orbit_start: int, orbit_end: int, skip_orbits: Optional[List[int]] = None
) -> List[str]:
    """
    Returns a list of all desired angle files in the given folder.
    """
    skip_set = set(skip_orbits or [])
    return [
        os.path.join(folder, f"O{i}_altaz.out")
        for i in range(orbit_start, orbit_end + 1)
        if i not in skip_set
    ]


def get_fits_folders_from_root(
    root_folder: str,
    orbit_start: int,
    orbit_end: int,
    skip_orbits: Optional[List[int]] = None,
) -> List[str]:
    """
    Returns a list of all desired folders containing .fits files in the given folder.
    """
    skip_set = set(skip_orbits or [])
    return [
        os.path.join(root_folder, f"orbit-{i}/")
        for i in range(orbit_start, orbit_end + 1)
        if i not in skip_set
    ]


class Preprocessing:
    def __init__(
        self,
        fits_root_folder,
        angle_folder,
        ccd_folder,
        background_ccd_folder,
        raw_angles_folder,
        angle_file,
        display_images_folder,
        image_size,
        num_workers=1,
        orbit_start=9,
        orbit_end=64,
        camera_number='1',
        skip_orbits: Optional[List[int]] = None,
    ):
        self.skip_orbits = {int(o) for o in (skip_orbits or [])}
        self.fits_root_folder = fits_root_folder
        self.angle_folder = angle_folder
        self.ccd_folder = ccd_folder
        self.background_ccd_folder = background_ccd_folder
        self.raw_angles_folder = raw_angles_folder
        self.angle_file = angle_file
        self.display_images_folder = display_images_folder
        self.image_size = image_size
        self.num_workers = num_workers
        self.data_dic = {}  # Dictionary of ffi->orbit->data stored in the angle file
        self.orbit_start = orbit_start
        self.orbit_end = orbit_end
        self.camera_number = camera_number
        # creates folders if they don't exist
        os.makedirs(self.ccd_folder, exist_ok=True)
        if self.display_images_folder:
            os.makedirs(self.display_images_folder, exist_ok=True)
        os.makedirs(self.angle_folder, exist_ok=True)
        os.makedirs(self.background_ccd_folder, exist_ok=True)

        # Generate the raw angle files + fits folders
        if self.skip_orbits:
            print(f"Skipping orbits: {sorted(self.skip_orbits)}")
        self.fits_folder_paths = get_fits_folders_from_root(
            self.fits_root_folder, self.orbit_start, self.orbit_end, skip_orbits=self.skip_orbits
        )
        self.raw_angles_file_paths = get_angle_files_from_folder(
            self.raw_angles_folder, self.orbit_start, self.orbit_end, skip_orbits=self.skip_orbits
        )
        if not self.raw_angles_file_paths:
            print(f"Warning: No .out files found in {self.raw_angles_folder}!")
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

    # ANGLE PROCESSING
    def process_angles(self, save=True) -> None:
        """
        Args: whether to save the angle file
        Returns: None
        Processes the angles in the angle file and saves the data to the class
        """
        Eel_idx = 7 + (int(self.camera_number) - 1) * 2
        Eaz_idx = 8 + (int(self.camera_number) - 1) * 2
        Mel_idx = 5 + (int(self.camera_number) - 1) * 2
        Maz_idx = 6 + (int(self.camera_number) - 1) * 2
        # loop through all the angle files (by orbit)
        for file_path in self.raw_angles_file_paths:
            orbit = file_path.split("/")[-1].split("_")[0][1:]
            with open(file_path, "r") as file:
                # get the data from the file
                for line in file.read().split("\n")[1:]:
                    arr = line.strip().split()
                    if len(arr) < 2:
                        break
                    arr = [
                        float(arr[i]) if i > 0 else str(arr[i]) for i in range(len(arr))
                    ]
                    ffi = arr[0]

                    # fucntion that edits values and rounds to 5 digits
                    deg_to_rad = lambda x: round(x * np.pi / 180, 5)
                    dist_scaled_inverse = lambda x: round(1 / (x / 50), 5)
                    dist_scaled_inverse_sqrd = lambda x: round(1 / (x / 50) ** 2, 5)

                    # creates dictionary of values for each ffi

                   
                    data_dic = {}
                    data_dic["ffi"] = ffi
                    data_dic["orbit"] = str(orbit)
                    data_dic["1/ED"] = dist_scaled_inverse(arr[1])
                    data_dic["1/MD"] = dist_scaled_inverse(arr[2])
                    data_dic["1/ED^2"] = dist_scaled_inverse_sqrd(arr[1])
                    data_dic["1/MD^2"] = dist_scaled_inverse_sqrd(arr[2])
                    data_dic["Eel"] = deg_to_rad(arr[3])
                    data_dic["Eaz"] = deg_to_rad(arr[4])
                    data_dic["Mel"] = deg_to_rad(arr[5])
                    data_dic["Maz"] = deg_to_rad(arr[6])
                    data_dic["E" + self.camera_number + "el"] = deg_to_rad(arr[Eel_idx])
                    data_dic["E" + self.camera_number + "az"] = deg_to_rad(arr[Eaz_idx])
                    data_dic["M" + self.camera_number + "el"] = deg_to_rad(arr[Mel_idx])
                    data_dic["M" + self.camera_number + "az"] = deg_to_rad(arr[Maz_idx])
                    data_dic["below_sunshade"] = arr[3] < -5.0 and arr[5] < -5.0

                

                    # saves dictionary of dictionaries to the class
                    self.data_dic[ffi] = data_dic

        print(f"Dataset size: {len(self.data_dic)}")

        # saves dictionary to computer
        if save:
            with open(os.path.join(self.angle_folder, self.angle_file), "wb") as file:
                pickle.dump(self.data_dic, file)
                print(f"Saved angle file to: {file.name}")
        print("Finished processing angles.")

    # BACKGROUND CREATION
    def create_backgrounds_by_orbit(self) -> None:
        """
        Args: None
        Returns: None
        Parallelizes over orbits via calls to background_orbit_worker
        """
        # Same as in the notebook, but parallelized over orbits
        rows_to_delete = range(2048, 2108)
        columns_to_delete = (
            list(range(0, 44)) + list(range(2092, 2180)) + list(range(4228, 4272))
        )
        # Group FFIs by orbit using self.data_dic
        orbit_to_folder = {}
        for i, folder_path in enumerate(self.fits_folder_paths):
            orbit = folder_path.split("-")[-1][:-1]
                #folders renamed from O11-54 to orbit-11-54
                # (
                # folder_path[27:29]
                # if len(folder_path) > 29 and folder_path[29] == "_"
                # else folder_path[27:28]
                #)
                
            
            orbit_to_folder[orbit] = (folder_path, i)
        # Prepare args for each orbit
        orbit_args = []
        for orbit, (folder_path, i) in orbit_to_folder.items():
            orbit_args.append(
                (self, folder_path, orbit, i, rows_to_delete, columns_to_delete)
            )
        with multiprocessing.Pool(self.num_workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(Preprocessing.background_orbit_worker, orbit_args),
                total=len(orbit_args),
                desc="Creating Backgrounds (by orbit)",
            ):
                continue

    @staticmethod
    def background_orbit_worker(args) -> None:
        """
        Args: the path to the folder where the fits files are in
                the orbit number
                the index of the folder
                the rows to delete
                the columns to delete
        Returns: None
        Processes background images in a given orbit sequentially
        """
        self, folder_path, orbit, i, rows_to_delete, columns_to_delete = args
        # Collect all FFI numbers for this orbit that are below the sunshade
        ffis_in_orbit_below_sunshade = [
            ffi
            for ffi, data in self.data_dic.items()
            if data["orbit"] == orbit and data["below_sunshade"]
        ]
        if not ffis_in_orbit_below_sunshade:
            print(f"No images below sunshade in orbit {orbit} (ffis_in_orbit_below_sunshade)")
            return
        args = []
        for fits_filename in os.listdir(folder_path):
            if fits_filename.endswith(".fits") or fits_filename.endswith(".fits.gz"):
                ffi_num = fits_filename[18:26]
                if ffi_num in ffis_in_orbit_below_sunshade:
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
            print(f"No images below sunshade in orbit {orbit} (args)")
            return
        # Sequentially process background images in this orbit
        print(f"Number of images below sunshade in orbit {orbit}: {len(args)}")
        orbit_images_below_sunshade = [
            Preprocessing.background_image_worker(arg) for arg in args
        ]
        orbit_images_below_sunshade = np.array(
            [k for k in orbit_images_below_sunshade if k is not None]
        )
        if len(orbit_images_below_sunshade) == 0:
            print(f"No images below sunshade in orbit {orbit} (len 0)")
            return
        arr_average = np.average(np.array(orbit_images_below_sunshade), axis=0)
        out_path = os.path.join(
            self.background_ccd_folder, f"O{orbit}_background_ccd.pkl"
        )
        with open(out_path, "wb") as file:
            pickle.dump(arr_average, file)
            print(f"Saved average background image to {str(file.name)}")
        print(f"{len(orbit_images_below_sunshade)} images averaged in orbit {orbit}")
        if self.display_images_folder and i == 0:
            show_img(
                arr_average,
                title=f"Average Orbit {orbit} image below sunshade",
                vmin=0,
                vmax=633118,
                save_path=os.path.join(
                    self.display_images_folder, f"avg_orbit_{orbit}_below_sunshade.png"
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
        if (
            len(fits_filename) > 40
            and fits_filename.endswith((".fits", ".fits.gz"))
            and fits_filename[27] == self.camera_number #user defined camera number
        ):
            

            arr = self.get_arr(fits_filename, folder_path)
            arr = np.delete(arr, rows_to_delete, axis=0)
            arr = np.delete(arr, columns_to_delete, axis=1)
            downsampled_arr = arr.astype(np.float32)
            # Downsample with median
            block = 4096 // self.image_size
            downsampled_arr = np.median(
                arr.reshape(self.image_size, block, self.image_size, block), axis=(1, 3)
            )

            return downsampled_arr
        # else:
        #     print(f"{fits_filename} is not a fits file") 

    # IMAGE PROCESSING
    def images_processing_by_orbit(self) -> None:
        """
        Args: None
        Returns: None
        Creates a process for each orbit to process the images in that orbit in parallel
        via calls to image_orbit_worker
        """
        print(f"Processing images in {len(self.fits_folder_paths)} orbits")
        rows_to_delete = range(2048, 2108)
        columns_to_delete = (
            list(range(0, 44)) + list(range(2092, 2180)) + list(range(4228, 4272))
        )
        orbit_to_folder = {}

        for i, fits_folder in enumerate(self.fits_folder_paths):
            # orbit = ( # old orbit naming convention
            #     fits_folder[27:29]
            #     if len(fits_folder) > 29 and fits_folder[29] == "_"
            #     else fits_folder[27:28]
            # )
            orbit = fits_folder.split("-")[-1][:-1]
            orbit_to_folder[orbit] = (fits_folder, i)
        orbit_args = []
        for orbit, (folder_path, i) in orbit_to_folder.items():
            orbit_args.append(
                (self, folder_path, orbit, i, rows_to_delete, columns_to_delete)
            )
        with multiprocessing.Pool(self.num_workers) as pool:
            for _ in pool.imap_unordered(Preprocessing.image_orbit_worker, orbit_args):
                continue

    @staticmethod
    def image_orbit_worker(args) -> None:
        """
        Args: the path to the folder where the fits files are in
                the orbit number
                the index of the folder
                the rows to delete
                the columns to delete
        Returns: None
        Processes images in a given orbit sequentially (no threading)
        """
        self, folder_path, orbit, i, rows_to_delete, columns_to_delete = args
        image_args = []
        # Only process files from camera x
        for fits_filename in os.listdir(folder_path):
            if (
                len(fits_filename) > 40
                and fits_filename.endswith((".fits", ".fits.gz"))
                and fits_filename[27] == self.camera_number
            ):
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
        # Sequentially process images in this orbit
        for arg in tqdm(
            image_args,
            total=len(image_args),
            desc=f"Processing Orbit {orbit} Images",
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
            ffi_num = fits_filename[18:26]
            # Only process if ffi_num is in self.data_dic
            if ffi_num not in self.data_dic:
                print(f"No angle data for {ffi_num}, orbit {orbit}")
                return  # Skip processing if no angle data
            arr = self.get_arr(fits_filename, folder_path)
            # Remove black bars
            arr = np.delete(arr, rows_to_delete, axis=0)
            arr = np.delete(arr, columns_to_delete, axis=1)
            # Downsample with median
            block = 4096 // self.image_size
            arr = arr.astype(np.float32)
            arr = np.median(
                arr.reshape(self.image_size, block, self.image_size, block), axis=(1, 3)
            )

            # Orbit mapping from notebook
            orbit = self.data_dic[ffi_num]["orbit"]
            orbit_to_subtract = orbit
            if orbit == "10":
                orbit_to_subtract = "9"
            if orbit == "11":
                orbit_to_subtract = "12"
            if orbit == "13":
                orbit_to_subtract = "14"
            if orbit == "15":
                orbit_to_subtract = "16"
            if orbit == "28":
                orbit_to_subtract = "27"
            if orbit == "30":
                orbit_to_subtract = "29"
            if orbit == "32":
                orbit_to_subtract = "31"
            if orbit == "34":
                orbit_to_subtract = "33"
            if orbit == "37":
                orbit_to_subtract = "38"
            if orbit == "56":
                orbit_to_subtract = "55"
            if orbit == "60":
                orbit_to_subtract = "59"
            if orbit == "61":
                orbit_to_subtract = "62"
            if orbit == "63":
                orbit_to_subtract = "64"
            if orbit == "80":
                orbit_to_subtract = "79"
            if orbit == "82":
                orbit_to_subtract = "81"
            if orbit == "84":   
                orbit_to_subtract = "83"
            if orbit == "85":
                orbit_to_subtract = "86"
            if orbit == "91":
                orbit_to_subtract = "92"
            if orbit == "100":
                orbit_to_subtract = "99"
            


            bg_path = os.path.join(
                self.background_ccd_folder, f"O{orbit_to_subtract}_background_ccd.pkl"
            )
            if os.path.exists(bg_path):
                background_avg_image = pickle.load(open(bg_path, "rb"))
                arr = arr - background_avg_image
            else:
                print(
                    f"Warning: Background file {bg_path} not found. Skipping subtraction for {fits_filename}."
                )
            # Pixel scaling
            scale_factor = 1 / 633118 
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
                f"{fits_filename[:42]}_processed_im{self.image_size}x{self.image_size}.pkl",
            )

            with open(out_path, "wb") as file:
                pickle.dump(arr, file)
        except Exception as e:
            print(f"Error processing {fits_filename}: {e}")

    def run(self, process_angles=True, process_images=True, make_background=True):
        if process_angles:
            self.process_angles()
        if make_background:
            self.create_backgrounds_by_orbit()
        if process_images:
            self.images_processing_by_orbit()


def main():
    parser = argparse.ArgumentParser(
        description="Consolidated TESS preprocessing script for all image sizes (YAML config version, with multiprocessing support)."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--no_angles", action="store_true", help="Skip angle processing"
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
        raw_angles_folder=config["raw_angles_folder"],
        angle_file=config["angle_file"],
        display_images_folder=config.get("display_images_folder", ""),
        image_size=config["image_size"],
        num_workers=config.get("num_workers", 1),
        orbit_start=config.get("orbit_start", 11),
        orbit_end=config.get("orbit_end", 54),
        camera_number=config.get("camera_number", "1"),
        skip_orbits=config.get("skip_orbits", []),
    )
    # CLI flags
    preproc.run(
        process_angles=not args.no_angles,
        make_background=not args.no_backgrounds,
        process_images=not args.no_images,
    )


if __name__ == "__main__":
    main()
