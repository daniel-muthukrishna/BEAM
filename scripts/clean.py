#!/usr/bin/env python
"""
BEAM background cleaning for camera-3 TICA full-frame images.

Usage:
    python scripts/clean.py --config configs/clean_config.yaml --frames frames.txt

The frames file lists one frame per line as "sector ffi" (e.g. "22 00032936"); lines starting with # are ignored.
TICA files are found by file name anywhere under data.tica_dir. Requires a CUDA GPU.
"""
import os
import sys
import re
import math
import time
import pickle
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]
from beam.utils.config import load_config, flatten_config  # noqa: E402
from generate import load_model  # noqa: E402

FINE, COARSE, PATCH = 4096, 256, 256   
K = FINE // COARSE # coarse cell num pixels          
BLOCK = 16 #mask bg size                            
# CCD layout and trimming of data/preprocess_all.py
ROWS_TO_DELETE = range(2048, 2108)
COLS_TO_DELETE = list(range(0, 44)) + list(range(2092, 2180)) + list(range(4228, 4272))
TICA_NAME = re.compile(r"_s(\d{4})-o\d+-(\d{8})-cam3-ccd([1-4])_.*\.fits$")   # sector, FFI, CCD


def tica_index(root):
    """Map (sector, ffi, ccd) -> path for every camera-3 TICA FFI under root, searched recursively."""
    index = {}
    for dirpath, _, names in os.walk(root):
        for name in names:
            m = TICA_NAME.search(name)
            if m:
                key, path = tuple(int(g) for g in m.groups()), os.path.join(dirpath, name)
                if key in index:
                    raise ValueError(f"two TICA files for sector {key[0]}, FFI {key[1]:08d}, CCD {key[2]}: "
                                     f"{index[key]} and {path}")
                index[key] = path
    return index


def camera_image(ccd_paths, file_unit):
    """Stitch and trim CCD 1-4 into a 4096x4096 camera image """
    im = np.block([[c3, c4], [np.flip(c2), np.flip(c1)]])
    im = np.delete(np.delete(im, ROWS_TO_DELETE, axis=0), COLS_TO_DELETE, axis=1)
    return (im / file_unit).astype(np.float32)


def load_frame(data, index, sector, ffi):
    """Loads 4096x4096 observation and corresponding preprocessed omage."""
    tica = [index[(sector, ffi, c)] for c in range(1, 5)]
    exptime = fits.getval(tica[0], "EXPTIME")
    if exptime != data["exptime"]:
        raise ValueError(f"{tica[0]}: EXPTIME {exptime} s, but the units assume {data['exptime']} s")
    start_name = os.path.basename(tica[0]).replace(".fits", "_processed_im256x256")
    start = np.asarray(np.load(os.path.join(data["start_dir"], start_name), allow_pickle=True), dtype=np.float32)
    return camera_image(tica, data["file_unit"]), start


def U(w):
    """Bilinear Upsample"""
    return F.interpolate(w, size=(FINE, FINE), mode="bilinear", align_corners=False)


def U_T(x):
    """Adjoint of U"""
    w = torch.zeros(x.shape[0], 1, COARSE, COARSE, device=x.device, requires_grad=True)
    with torch.enable_grad():
        return torch.autograd.grad(U(w), w, grad_outputs=x)[0]


def smooth(d, k):
    for _ in range(2):
        d = F.avg_pool2d(F.pad(d, (k // 2,) * 4, mode="reflect"), k, stride=1)
    return d


def shifted_tiles(d):
    cuts = sorted({0, FINE, *range(d, FINE, PATCH)})
    return [(a, b, min(a, FINE - PATCH)) for a, b in zip(cuts[:-1], cuts[1:])]


@torch.no_grad()
def score(net, u, t, shift, model):
    """Score s(u, t) = grad log p_t(u), evaluated patchwise on a tiling offset by shift = (dy, dx).
    ASSUMES V PARAM TRAINED MODEL
    """
    R = u.shape[0]
    grid = [(r, c) for r in shifted_tiles(shift[0]) for c in shifted_tiles(shift[1])]
    x = torch.cat([u[:, :, r0:r0 + PATCH, c0:c0 + PATCH] for (_, _, r0), (_, _, c0) in grid])  
    s = torch.empty_like(x)
    sigma = math.sqrt(1 - t ** 2)
    for i in range(0, len(x), model["patch_batch"]):
        xb = x[i:i + model["patch_batch"]]
        B = len(xb)
        ctx = torch.zeros(B, 1, model["context_len"], device=u.device)
        v = net(xb, ctx, torch.full((B, 1, 1, 1), t, device=u.device), torch.zeros_like(ctx),
                torch.full((B,), model["camera"], device=u.device)).float()
        s[i:i + B] = -(xb + t * v / sigma) 
    for j, ((a, b, r0), (c, d, c0)) in enumerate(grid):
        out[:, :, a:b, c:d] = s[j * R:(j + 1) * R, :, a - r0:b - r0, c - c0:d - c0]
    return out


def block_median(img, b):
    n = img.shape[0] // b
    return np.median(img.reshape(n, b, n, b).swapaxes(1, 2).reshape(n, n, b * b), axis=-1)

def star_mask(y):
   """Masks pixels past 3 sigma"""
    r = y - U(torch.from_numpy(block_median(y, BLOCK))[None, None])[0, 0].numpy()
    n = FINE // PATCH
    rp = r.reshape(n, PATCH, n, PATCH).swapaxes(1, 2).reshape(n, n, PATCH * PATCH)
    med = np.median(rp, axis=-1, keepdims=True)
    sig = 1.4826 * np.median(np.abs(rp - med), axis=-1, keepdims=True)
    star = (np.abs(rp - med) > 3 * sig).reshape(n, n, PATCH, PATCH).swapaxes(1, 2).reshape(1, 1, FINE, FINE)
    star = F.max_pool2d(torch.from_numpy(star.astype(np.float32)), 5, stride=1, padding=2)
    return 1 - star


def clean(net, y, w, mask, lo, hi, sigmas, cfg, release):
    """Main Cleaning algorithm. Hold keeps pixels near init as init is good there.
    Release lets BEAM correct image where it needs correcting
    """
    lad, hold_c = cfg["ladder"], cfg["hold"]["c"]
    rng = np.random.default_rng(cfg["seed"]) # tiling offsets
    w0 = w.clone()
    start = time.time()
    for sigma in sigmas:
        t = math.sqrt(1 - sigma ** 2)
        eta = lad["step"] * sigma ** 2 / (t ** 2 * K ** 2)
        tau = hold_c * sigma / (t * K)
        # MAP estimate as a efficient approx of averaging N chains
        for _ in range(lad["steps_per_level"]):
            u = t * (y - U(w)).clamp(lo, hi) # clamp residual to remove preprocessing artifacts                      
            shift = tuple(rng.integers(PATCH, size=2).tolist())
            s = score(net, u, t, shift, cfg["model"]) * mask # mask score at stars
            g = t * U_T(s)
            g = g - smooth(g, cfg["highpass"]) # smooth part of init accurate
            w = w - eta * g
            d = w - w0 #hold                                              
            w = w0 + d / (1 + (eta / tau ** 2) / (1 + (d / release) ** 2)) #release
        print(f"  level sigma = {sigma:.4f} done ({time.time() - start:.0f} s)", flush=True)
    return w


def main():
    parser = argparse.ArgumentParser(description="BEAM background cleaning for a batch of camera-3 TICA frames.")
    parser.add_argument("--config", required=True, help="cleaning config, e.g. configs/clean_config.yaml")
    parser.add_argument("--frames", required=True, help='text file with one frame per line: "sector ffi"')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    frames = []
    with open(args.frames) as f:
        for line in f:
            if line.strip() and not line.lstrip().startswith("#"):
                sector, ffi = line.split()
                frames.append((int(sector), int(ffi)))

    model, data = cfg["model"], cfg["data"]
    index = tica_index(data["tica_dir"])
    incomplete = [f"{s} {f:08d}" for s, f in frames if any((s, f, c) not in index for c in range(1, 5))]
    if incomplete:
        raise FileNotFoundError(f"no complete set of CCD 1-4 TICA files under {data['tica_dir']} for: {incomplete}")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True # fixed input sizes
    model_cfg = load_config(os.path.join(model["dir"], "config.yaml"))
    net = load_model(os.path.join(model["dir"], model["checkpoint"]), flatten_config(model_cfg),
                     model["context_len"], device).nn_model
                     
    with open(os.path.join(model["dir"], "image_stats.pkl"), "rb") as f:
        stats = pickle.load(f)

    sky_by_sector = model_cfg["data"]["sky_by_sector"]
    missing = sorted({s for s, _ in frames if s not in sky_by_sector})
    if missing:
        raise KeyError(f"sectors {missing} have no sky constant in the model's config.yaml")
    e_per_s = data["file_unit"] / data["exptime"] # e-/s per file unit
    release = cfg["hold"]["release"] / (e_per_s * stats["std"])
    lad = cfg["ladder"]
    sigmas = np.geomspace(lad["sigma_top"], lad["sigma_bottom"], lad["levels"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    for i in range(0, len(frames), cfg["batch_size"]):
        batch = frames[i:i + cfg["batch_size"]]
        print(f"frames {i + 1}-{i + len(batch)} of {len(frames)}", flush=True)
        ys, starts, masks = [], [], []

        for sector, ffi in batch:
            y, start = load_frame(data, index, sector, ffi)
            ys.append(y)
            starts.append(start)
            masks.append(star_mask(y))

        y = torch.from_numpy(np.stack(ys))[:, None].to(device)
        sky = torch.tensor([sky_by_sector[s] for s, _ in batch], device=device).view(-1, 1, 1, 1)
        y_model = (y - sky - stats["mean"]) / stats["std"]
        w0 = torch.from_numpy(np.stack(starts))[:, None].to(device) / stats["std"]
        mask = torch.cat(masks).to(device)
        c = cfg["clip"] / e_per_s
        lo, hi = (-c - sky - stats["mean"]) / stats["std"], (c - sky - stats["mean"]) / stats["std"]

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=model["bf16"]):
            w = clean(net, y_model, w0, mask, lo, hi, sigmas, cfg, release)

        background = (U(w) * stats["std"]).cpu().numpy()[:, 0] * e_per_s

        for j, (sector, ffi) in enumerate(batch):
            header = fits.Header({"BUNIT": "e-/s", "SECTOR": sector, "FFI": f"{ffi:08d}", "CAMERA": 3})
            for name, img in [("background", background[j]), ("cleaned", ys[j] * e_per_s - background[j])]:
                path = os.path.join(cfg["output_dir"], f"s{sector:04d}_{ffi:08d}_cam3_beam_{name}.fits")
                fits.PrimaryHDU(img.astype(np.float32), header).writeto(path, overwrite=True)
            print(f"  wrote s{sector:04d}_{ffi:08d}", flush=True)


if __name__ == "__main__":
    main()
