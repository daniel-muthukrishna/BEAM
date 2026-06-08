#!/usr/bin/env python3
"""Regenerate CNN preview images from saved arrays using viridis colormap."""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARRAY_DIR = "model_outputs/simple_cnn_20260414_005047/generated_arrays"
IMAGE_DIR = "model_outputs/simple_cnn_20260414_005047/generated_images"

for fname in sorted(os.listdir(ARRAY_DIR)):
    if not fname.endswith("_target.npy"):
        continue
    prefix = fname.replace("_target.npy", "")
    m = re.match(r"(\d+)_(\d+)", prefix)
    idx, ffi = m.group(1), m.group(2)

    target = np.load(os.path.join(ARRAY_DIR, prefix + "_target.npy"))
    pred = np.load(os.path.join(ARRAY_DIR, prefix + "_prediction.npy"))
    residual = pred - target
    combined = np.concatenate([target.ravel(), pred.ravel()])
    vmin, vmax = 0, 1
    rmax = 1.0
    if not np.isfinite(rmax) or rmax == 0:
        rmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(target, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Target")
    axes[1].imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    axes[2].imshow(residual, cmap="coolwarm", vmin=-rmax, vmax=rmax)
    axes[2].set_title("Residual")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("FFI " + ffi)
    out = os.path.join(IMAGE_DIR, idx + "_" + ffi + ".png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("Wrote " + out)
