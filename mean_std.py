import os, pickle, numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_file(args):
    path, angles_dic = args
    ffi_num = path.split("/")[-1].split("_")[0][18:18+8]
    if ffi_num in angles_dic:
        angles = angles_dic[ffi_num]
    else:
        print(f"FFI number {ffi_num} not found in angles dictionary")
        return np.nan, np.nan, np.nan

    with open(path, "rb") as f:
        image = pickle.load(f)
    orbit = angles['orbit']
    if orbit in ['43', '44']:
        return np.nan, np.nan, np.nan
    if int(orbit) >= 60:
        image = image*3
    return image.sum(dtype=np.float64), (image**2).sum(dtype=np.float64), image.size

if __name__ == "__main__":
    files = [os.path.join("/pdo/users/djtufto/cam2/ccd/", e) for e in os.listdir("/pdo/users/djtufto/cam2/ccd/") if e.endswith(".pkl")]
    print(f"Processing {len(files)} files")
    angles_dic = pickle.load(open("/pdo/users/djtufto/cam2/angles/angles_O11-116_cam_2data_dic.pkl", "rb"))
    print(type(angles_dic))
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(process_file, [(file, angles_dic) for file in files])
        sums, sq_sums, totals = zip(*tqdm(results, total=len(files)))
    total_sum = np.nansum(sums)
    total_sq_sum = np.nansum(sq_sums)
    total_pixels = np.nansum(totals)
    mean = total_sum / total_pixels
    std = np.sqrt(total_sq_sum / total_pixels - mean**2)
    print(mean, std)
