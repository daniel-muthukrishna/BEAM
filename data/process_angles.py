import argparse
import yaml
from typing import List, Optional
import os
import numpy as np
import pickle


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


def process_angles(
    raw_angles_folder: str,
    angle_folder: str,
    angle_file: str,
    orbit_start: int,
    orbit_end: int,
    camera_number: str = "3",
    skip_orbits: Optional[List[int]] = None,
    save: bool = True,
) -> dict:
    """
    Process raw angle files into a dictionary keyed by FFI.
    """
    os.makedirs(angle_folder, exist_ok=True)
    raw_angles_file_paths = get_angle_files_from_folder(
        raw_angles_folder,
        orbit_start,
        orbit_end,
        skip_orbits=skip_orbits,
    )
    if not raw_angles_file_paths:
        print(f"Warning: No .out files found in {raw_angles_folder}!")

    camera_number = str(camera_number)
    eel_idx = 7 + (int(camera_number) - 1) * 2
    eaz_idx = 8 + (int(camera_number) - 1) * 2
    mel_idx = 5 + (int(camera_number) - 1) * 2
    maz_idx = 6 + (int(camera_number) - 1) * 2
    data_dic = {}

    def deg_to_rad(value):
        return round(value * np.pi / 180, 5)

    def dist_scaled_inverse(value):
        return round(1 / (value / 50), 5)

    def dist_scaled_inverse_sqrd(value):
        return round(1 / (value / 50) ** 2, 5)

    for file_path in raw_angles_file_paths:
        orbit = os.path.basename(file_path).split("_")[0][1:]
        with open(file_path, "r") as file:
            for line in file.read().split("\n")[1:]:
                arr = line.strip().split()
                if len(arr) < 2:
                    break
                arr = [float(arr[i]) if i > 0 else str(arr[i]) for i in range(len(arr))]
                ffi = arr[0]

                data_dic[ffi] = {
                    "ffi": ffi,
                    "orbit": str(orbit),
                    "1/ED": dist_scaled_inverse(arr[1]),
                    "1/MD": dist_scaled_inverse(arr[2]),
                    "1/ED^2": dist_scaled_inverse_sqrd(arr[1]),
                    "1/MD^2": dist_scaled_inverse_sqrd(arr[2]),
                    "Eel": deg_to_rad(arr[3]),
                    "Eaz": deg_to_rad(arr[4]),
                    "Mel": deg_to_rad(arr[5]),
                    "Maz": deg_to_rad(arr[6]),
                    "E" + camera_number + "el": deg_to_rad(arr[eel_idx]),
                    "E" + camera_number + "az": deg_to_rad(arr[eaz_idx]),
                    "M" + camera_number + "el": deg_to_rad(arr[mel_idx]),
                    "M" + camera_number + "az": deg_to_rad(arr[maz_idx]),
                    "below_sunshade": arr[3] < -5.0 and arr[5] < -5.0,
                }

    print(f"Dataset size: {len(data_dic)}")

    if save:
        with open(os.path.join(angle_folder, angle_file), "wb") as file:
            pickle.dump(data_dic, file)
            print(f"Saved angle file to: {file.name}")
    print("Finished processing angles.")
    return data_dic


def _config_value(config: dict, args: argparse.Namespace, name: str, default=None):
    value = getattr(args, name)
    if value is not None:
        return value
    return config.get(name, default)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process raw TESS angle .out files into an angle dictionary pickle."
    )
    parser.add_argument("--config", type=str, help="Optional YAML config file")
    parser.add_argument("--raw_angles_folder", default="/pdo/users/djtufto/altazzes_new/altazzes/", type=str, help="Folder containing O*_altaz.out files")
    parser.add_argument("--angle_folder", type=str, help="Output folder for processed angle data")
    parser.add_argument(
        "--angle_file",
        default="angles_O11-116_cam_3_data_dic_full.pkl",
        type=str,
        help="Output pickle filename. Falls back to config key 'angles_dic'.",
    )
    parser.add_argument("--orbit_start", type=int, help="First orbit number to process")
    parser.add_argument("--orbit_end", type=int, help="Last orbit number to process")
    parser.add_argument("--camera_number", type=str, default="3", help="Camera number, 1 through 4")
    parser.add_argument(
        "--skip_orbits",
        type=int,
        nargs="*",
        help="Orbit numbers to skip",
    )
    parser.add_argument("--no_save", action="store_true", help="Return data without writing pickle")
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file) or {}

    raw_angles_folder = _config_value(config, args, "raw_angles_folder")
    angle_folder = _config_value(config, args, "angle_folder")
    angle_file = _config_value(config, args, "angle_file", config.get("angles_dic"))
    orbit_start = _config_value(config, args, "orbit_start")
    orbit_end = _config_value(config, args, "orbit_end")
    camera_number = _config_value(config, args, "camera_number", "1")
    skip_orbits = _config_value(config, args, "skip_orbits", [])

    missing = [
        name
        for name, value in {
            "raw_angles_folder": raw_angles_folder,
            "angle_folder": angle_folder,
            "angle_file": angle_file,
            "orbit_start": orbit_start,
            "orbit_end": orbit_end,
        }.items()
        if value is None
    ]
    if missing:
        parser.error(
            "Missing required value(s): "
            + ", ".join(missing)
            + ". Provide them as CLI args or in --config."
        )

    process_angles(
        raw_angles_folder=raw_angles_folder,
        angle_folder=angle_folder,
        angle_file=angle_file,
        orbit_start=int(orbit_start),
        orbit_end=int(orbit_end),
        camera_number=str(camera_number),
        skip_orbits=skip_orbits,
        save=not args.no_save,
    )


if __name__ == "__main__":
    main()