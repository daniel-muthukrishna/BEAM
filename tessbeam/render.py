"""
Images -> PNG / GIF / FITS.

Layout: the 256x256 image is the camera's 4-CCD mosaic, built in training as
np.block([[ccd3, ccd4], [flip(ccd2), flip(ccd1)]]) and shown with origin='lower':
CCD3 bottom-left, CCD4 bottom-right, CCD2 top-left, CCD1 top-right.
"""

import numpy as np
from matplotlib import colormaps, patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image

CBAR_LABEL = "e-/s per pixel (stray light above quiet background)"
ZOOM = 20         # image pixels on a side of the zoom around a target
MIN_TOP = 10.0    # e-/s: the colour bar never ends lower, so a quiet image isn't stretched into noise


def gif_palette():
    """The one 256-colour palette every GIF frame is drawn with, as a Pillow 'P' image."""
    viridis = colormaps["viridis"](np.linspace(0, 1, 223), bytes=True)[:, :3]   # image, colour bar
    red = [[255, 0, 0]]                                                         # the marker
    greys = np.linspace(0, 255, 32).round()[:, None].repeat(3, axis=1)          # black..white: text
    colours = np.vstack([viridis, red, greys]).astype(np.uint8)                 # 223 + 1 + 32 = 256
    pal = Image.new("P", (1, 1))
    pal.putpalette(colours.tobytes())
    return pal


GIF_PALETTE = gif_palette()


def zoom_window(ix, iy, size=ZOOM, n=256):
    """(x0, x1, y0, y1): a size x size block of image pixels around pixel (ix, iy), moved inward
    at the image's edges so that it is always the full size."""
    x0 = min(max(ix - size // 2, 0), n - size)
    y0 = min(max(iy - size // 2, 0), n - size)
    return x0, x0 + size, y0, y0 + size


def _crop(images, window):
    """The part of an image (or a stack of them) inside a zoom window, for its colour scale."""
    if window is None:
        return images
    x0, x1, y0, y1 = window
    return np.asarray(images)[..., y0:y1, x0:x1]


def color_norm(images):
    """One linear colour scale for a whole image set, so frames are comparable."""
    v = np.asarray(images, dtype=np.float64)
    frames = v.reshape(-1, v.shape[-2] * v.shape[-1])
    lo = min(0.0, float(np.percentile(v, 1)))
    # the top comes from the brightest frame, so that its glare is not one flat saturated patch
    hi = max(float(np.percentile(frames, 99.5, axis=1).max()), MIN_TOP)
    return Normalize(vmin=lo, vmax=hi)


def render_frame(img, norm, title="", marker=None, grid=None, label=None, window=None):
    """One image -> RGB uint8 array (with CCD grid, labels and colour bar).

    marker: None, or (x, y) in image pixels to mark with a red cross.
    grid: None, or RA/Dec lines [(x, y, text), ...] in image pixels (tessbeam.sky.grid_lines).
    label: None, or text to write beside the marker (the target's RA/Dec).
    window: None, or (x0, x1, y0, y1) to show only that block of image pixels (a zoom). Everything
        is drawn in full-image coordinates and the axes are then narrowed to the window, so the
        cross, the grid and the CCD boundary need no conversion.
    Uses matplotlib's object API (no pyplot), so it never touches the caller's backend
    or open figures.
    """
    fig = Figure(figsize=(5.6, 5.9), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.06, 0.05, 0.76, 0.84])
    cax = fig.add_axes([0.84, 0.05, 0.035, 0.84])
    im = ax.imshow(img, origin="lower", cmap="viridis", norm=norm, interpolation="nearest")
    n = img.shape[0]
    for p in (n / 2 - 0.5,):
        ax.axhline(p, color="#c8c8c8", lw=0.6)
        ax.axvline(p, color="#c8c8c8", lw=0.6)
    if window is None:                    # in a zoom these corner labels would fall outside the view
        kw = dict(color="w", fontsize=8, alpha=0.8)
        ax.text(3, 3, "CCD3", va="bottom", **kw)
        ax.text(n / 2 + 3, 3, "CCD4", va="bottom", **kw)
        ax.text(3, n - 4, "CCD2", va="top", **kw)
        ax.text(n / 2 + 3, n - 4, "CCD1", va="top", **kw)
    x0, x1, y0, y1 = window or (0, n, 0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    # dark outline, so white text stays readable on the bright yellow of strong glare
    outline = [patheffects.withStroke(linewidth=1.5, foreground="black")]
    for x, y, text in grid or ():
        # opaque light grey, not translucent white: a GIF frame's fixed palette has that grey, but
        # not every blend of white with the image under it (those came out green)
        ax.plot(x, y, color="#c8c8c8", lw=0.5, scalex=False, scaley=False, zorder=3)
        # the label goes at the middle of the part of the line that is in view
        seen = np.flatnonzero((x >= x0 - 0.5) & (x <= x1 - 0.5) & (y >= y0 - 0.5) & (y <= y1 - 0.5))
        if len(seen):
            m = seen[len(seen) // 2]
            ax.text(x[m], y[m], text, color="white", fontsize=6, ha="center", va="center", zorder=4,
                    path_effects=outline, clip_on=True)
    if label is not None and marker is not None:
        # beside the cross, on whichever side of the view has room; 6 pixels of the full image
        # (scaled down in a zoom, so the label stays next to the cross)
        x, y = marker
        gap = 6 * (x1 - x0) / n
        right, top = x < x0 + 0.75 * (x1 - x0), y < y0 + 0.9 * (y1 - y0)
        ax.text(x + (gap if right else -gap), y + (gap if top else -gap), label, color="white", fontsize=7,
                ha="left" if right else "right", va="bottom" if top else "top", zorder=7,
                path_effects=outline)
    if marker is not None:
        # Pure red has almost no contrast against the green/teal middle of viridis, so the
        # cross gets a thin white outline. scalex/scaley=False keep the axes exactly on the
        # image (a target on the very edge must not rescale it), and clip_on=False draws
        # the whole cross even there.
        x, y = marker
        ax.plot([x], [y], marker="+", color="red", markersize=14, markeredgewidth=1.6,
                linestyle="none", scalex=False, scaley=False, clip_on=False, zorder=6,
                path_effects=[patheffects.Stroke(linewidth=3.4, foreground="white"),
                              patheffects.Normal()])
    if window is not None:
        ax.set_xlim(x0 - 0.5, x1 - 0.5)
        ax.set_ylim(y0 - 0.5, y1 - 0.5)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(CBAR_LABEL, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def to_png(img, path, title="", marker=None, grid=None, label=None, window=None):
    # a zoom gets its own colour scale, from the pixels in view
    Image.fromarray(render_frame(img, color_norm(_crop(img, window)), title, marker, grid, label, window)).save(path)
    return path


def to_gif(images, titles, path, fps=8, marker=None, grid=None, label=None, window=None):
    """One GIF frame per image; `marker` (if any) is the same in every frame."""
    norm = color_norm(_crop(images, window))                    # a zoom: from the pixels in view
    # A GIF frame holds at most 256 colours. Given RGB frames, Pillow picks a new set of
    # 256 for every frame, and in some frames none of them is red: the cross then flickers
    # between red and dark crimson. So each frame is mapped onto one fixed palette (without
    # dithering, which would speckle the flat colours) as soon as it is rendered - which
    # also means only the small palette frames are held in memory, not the RGB ones.
    frames = [Image.fromarray(render_frame(im, norm, t, marker, grid, label, window))
              .quantize(palette=GIF_PALETTE, dither=Image.Dither.NONE)
              for im, t in zip(images, titles)]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    return path


def to_lightcurve(times_utc, values, path, title=""):
    """Stray light at one place, frame by frame -> PNG: an approximate light curve of the glare."""
    from datetime import datetime
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter, date2num

    t = date2num([datetime.fromisoformat(str(u)[:19]) for u in times_utc])
    v = np.asarray(values, dtype=np.float64)
    # Break the line where frames are missing (such as the week of sector 1751 inside sector 99),
    # instead of drawing a straight line across the gap as if the light had been measured there.
    if len(t) > 2:
        cut = np.flatnonzero(np.diff(t) > 1.5 * np.median(np.diff(t))) + 1
        t, v = np.insert(t, cut, np.nan), np.insert(v, cut, np.nan)
    fig = Figure(figsize=(5.6, 3.4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.13, 0.15, 0.83, 0.72])
    ax.plot(t, v, "-", color="C0", lw=1.2)
    ax.plot(t, v, ".", color="C0", ms=4)
    ax.xaxis_date()
    loc = AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
    ax.set_ylim(bottom=min(0.0, float(np.nanmin(v))))           # linear, from zero, like the images
    ax.set_ylabel("e-/s per pixel", fontsize=9)
    ax.set_xlabel("UTC", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=9)
    canvas.draw()
    Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3].copy()).save(path)
    return path


def to_fits(images, table, header, path):
    """images: (H, W) or (T, H, W); table: dict of equal-length columns; header: dict of cards."""
    from astropy.io import fits
    from astropy.table import Table

    prim = fits.PrimaryHDU(np.asarray(images, dtype=np.float32))
    prim.header["BUNIT"] = "electron/s"
    for k, v in header.items():
        prim.header[k] = v
    tab = fits.table_to_hdu(Table(table))
    tab.name = "FRAMES"
    fits.HDUList([prim, tab]).writeto(path, overwrite=True)
    return path
