import os
import re
import glob
import pickle
import multiprocessing as mp
from collections import Counter

import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    processed_folder = "/pdo/users/djtufto/cam3/tica_11_116"
    background_folder = "/pdo/users/djtufto/cam3/bg4096"
    angle_path = "/pdo/users/djtufto/cam3/angle_full/angles_O11-116_cam_3_data_dic_full.pkl"
    output_folder = "/pdo/users/djtufto/cam3/tica_below_sunshade"

    os.makedirs(output_folder, exist_ok=True)

    scale_factor = 1 / 633118 / 5.3
    num_workers = 20

    with open(angle_path, "rb") as f:
        data_dic = pickle.load(f)

    pattern = re.compile(
        r"hlsp_tica_tess_ffi_s(?P<sector>\d{4})-o\d+-(?P<ffi>\d{8})-cam\d-ccd\d"
    )

    _worker_data_dic = None
    _worker_background_cache = None


    def init_worker(worker_data_dic):
        global _worker_data_dic, _worker_background_cache
        _worker_data_dic = worker_data_dic
        _worker_background_cache = {}


    def process_one_image(img_path):
        filename = os.path.basename(img_path)

        match = pattern.search(filename)
        if match is None:
            return {"status": "parse_failed", "filename": filename}

        sector = int(match.group("sector"))
        ffi_num = match.group("ffi")

        if ffi_num not in _worker_data_dic:
            return {"status": "missing_angle", "ffi_num": ffi_num}

        if not _worker_data_dic[ffi_num].get("below_sunshade", False):
            return {"status": "not_below_sunshade"}

        if sector not in _worker_background_cache:
            bg_path = os.path.join(background_folder, f"S{sector:04d}_background_ccd.pkl")
            if not os.path.exists(bg_path):
                return {"status": "missing_background", "sector": sector}
            with open(bg_path, "rb") as f:
                _worker_background_cache[sector] = pickle.load(f).astype(np.float32)

        with open(img_path, "rb") as f:
            processed = pickle.load(f).astype(np.float32)

        # processed = (trimmed_observation - background) * scale_factor
        # bg_added = trimmed_observation * scale_factor
        bg_added = processed + _worker_background_cache[sector] * scale_factor
        bg_added = bg_added.astype(np.float32)

        out_filename = filename.replace("_processed_", "_bss_") + ".pkl"
        out_path = os.path.join(output_folder, out_filename)
        with open(out_path, "wb") as f:
            pickle.dump(bg_added, f)

        arr64 = bg_added.astype(np.float64)
        return {
            "status": "saved",
            "n_pixels": arr64.size,
            "sum_pixels": arr64.sum(),
            "sum_sq_pixels": np.square(arr64).sum(),
        }


    img_paths = sorted(glob.glob(os.path.join(processed_folder, "*_processed_im4096x4096")))

    n_pixels = 0
    sum_pixels = 0.0
    sum_sq_pixels = 0.0
    status_counts = Counter()

    # Workers only read inputs and write unique output files. Mean/std are aggregated here
    # in the parent process to avoid races on shared counters.
    ctx = mp.get_context("fork")
    with ctx.Pool(num_workers, initializer=init_worker, initargs=(data_dic,)) as pool:
        for result in tqdm(pool.imap_unordered(process_one_image, img_paths), total=len(img_paths)):
            status = result["status"]
            status_counts[status] += 1

            if status == "saved":
                n_pixels += result["n_pixels"]
                sum_pixels += result["sum_pixels"]
                sum_sq_pixels += result["sum_sq_pixels"]
            elif status == "parse_failed":
                print(f"Could not parse filename: {result['filename']}")
            elif status == "missing_angle":
                print(f"No angle data for {result['ffi_num']}; skipping")
            elif status == "missing_background":
                print(f"No background for sector {result['sector']}; skipping")

    if n_pixels == 0:
        raise RuntimeError("No below-sunshade images were saved, so mean/std cannot be computed.")

    mean = sum_pixels / n_pixels
    std = np.sqrt(max((sum_sq_pixels / n_pixels) - mean**2, 0.0))

    print(f"Saved bg-added below-sunshade images: {status_counts['saved']}")
    print(f"Skipped not below sunshade: {status_counts['not_below_sunshade']}")
    print(f"Skipped missing angle data: {status_counts['missing_angle']}")
    print(f"Skipped parse failures: {status_counts['parse_failed']}")
    print(f"Skipped missing backgrounds: {status_counts['missing_background']}")
    print(f"Dataset mean: {mean}")
    print(f"Dataset std: {std}")