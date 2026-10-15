"""
RA/Dec across the stray-light image, for drawing a sky grid on it. Precomputed and kept on disk.
    ccd_wcs(sector, ccd)    -> astropy WCS for that CCD (pixel 0,0 = the first column of the FFI file)
    image_sky(sector)       -> (ra, dec): degrees at the centre of every pixel of the 256x256 image
    grid_lines(sector)      -> [(x, y, label), ...]: RA and Dec lines in image coordinates

"""

import json
import math
import os
import warnings
from functools import lru_cache

import numpy as np

N = 256                       # image pixels along each axis
BIN = 16                      # detector pixels per image pixel
SNAPSHOT = os.path.join(os.path.dirname(__file__), "data", "sky_wcs.json")   # shipped with the package
_STORE = None                 # {sector: {"pointing": [ra, dec, roll], "camera": 3, "ccds": {ccd: header}}}


def _cache_file():
    # read the environment now, not at import (the same folder the sector table uses)
    cache = os.environ.get("TESSBEAM_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "tessbeam")
    return os.path.join(cache, "sky_wcs.json")


def _store():
    """Every stored fit: the packaged ones, then those fitted on this machine (which win)."""
    global _STORE
    if _STORE is None:
        _STORE = {}
        for path in (SNAPSHOT, _cache_file()):
            try:
                with open(path) as f:
                    _STORE.update(json.load(f))
            except FileNotFoundError:
                pass
            except Exception as e:                         # a damaged file must not stop tessbeam
                warnings.warn(f"tessbeam ignored its stored sky fits {path}: can't read it ({e})")
    return _STORE


def _pointing_key(t2p, sector):
    """tess-point's own pointing for the sector: the fit is only valid while this is unchanged."""
    table = t2p.TESS_Spacecraft_Pointing_Data
    i = [int(s) for s in table.sectors].index(int(sector))
    return [round(float(table.ras[i]), 6), round(float(table.decs[i]), 6), round(float(table.rolls[i]), 6)]


def _fit(t2p, sector, ccd, camera):
    """Fit an astropy WCS for one CCD to tess-point's camera model (see the module docstring)."""
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import fit_wcs_from_points

    cols, rows = np.meshgrid(np.linspace(45, 2092, 9), np.linspace(1, 2048, 9))   # tess-point pixels
    sky = [t2p.tess_stars2px_reverse_function_entry(int(sector), camera, ccd, c, r)[:2]
           for c, r in zip(cols.ravel(), rows.ravel())]
    ra, dec = np.array(sky, dtype=np.float64).T
    # WCS pixel coordinates are 0-based: tess-point's 1-based (col, row) minus one
    return fit_wcs_from_points((cols.ravel() - 1, rows.ravel() - 1), SkyCoord(ra, dec, unit="deg"),
                               projection="TAN", sip_degree=3)


def _record(key, camera, wcs):
    """A sector's four fits as JSON-ready text (each WCS as its FITS header, SIP terms included)."""
    return {"pointing": key, "camera": camera,
            "ccds": {str(c): w.to_header(relax=True).tostring(sep="\n") for c, w in wcs.items()}}


def sector_wcs(sector, camera=3):
    """{ccd: astropy WCS} for one sector's four CCDs: stored on disk, or fitted now and stored."""
    return _sector_wcs(int(sector), int(camera))      # one cache entry, however the call is written


@lru_cache(maxsize=None)
def _sector_wcs(sector, camera):
    from astropy.io import fits
    from astropy.wcs import WCS
    from tessbeam.target import tess_point

    t2p = tess_point(sector)
    key = _pointing_key(t2p, sector)
    rec = _store().get(str(int(sector)))
    if rec and rec["pointing"] == key and rec["camera"] == camera:
        return {int(c): WCS(fits.Header.fromstring(h, sep="\n")) for c, h in rec["ccds"].items()}
    wcs = {ccd: _fit(t2p, sector, ccd, camera) for ccd in (1, 2, 3, 4)}
    _store()[str(int(sector))] = _record(key, camera, wcs)
    _save_cache(int(sector))
    return wcs


def _save_cache(sector):
    """Add one sector's fits to the cache file (atomic: a temporary file, then a rename)."""
    path = _cache_file()
    try:
        try:
            with open(path) as f:
                kept = json.load(f)
        except FileNotFoundError:
            kept = {}
        kept[str(sector)] = _store()[str(sector)]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(kept, f)
        os.replace(tmp, path)
    except Exception as e:                                 # the fit still works, it's just not kept
        warnings.warn(f"tessbeam could not keep the sky fits of sector {sector} in {path} ({e})")


def ccd_wcs(sector, ccd, camera=3):
    """The astropy WCS of one CCD in one sector."""
    return sector_wcs(int(sector), camera)[ccd]


def _quadrants():
    """(CCD, image rows, image columns) of the four CCDs in the 256x256 image (see tessbeam.target)."""
    h = N // 2
    return [(3, slice(0, h), slice(0, h)), (4, slice(0, h), slice(h, N)),
            (2, slice(h, N), slice(0, h)), (1, slice(h, N), slice(h, N))]


def image_sky(sector, camera=3):
    """RA and Dec (degrees) at the centre of every pixel of the 256x256 image."""
    if camera != 3:
        raise NotImplementedError("the mosaic layout is only verified for camera 3 so far")
    ra, dec = np.full((N, N), np.nan), np.full((N, N), np.nan)
    iy, ix = np.mgrid[0:N, 0:N]
    C, R = (ix + 0.5) * BIN - 0.5, (iy + 0.5) * BIN - 0.5        # image pixel centre in the mosaic
    for ccd, rows, cols in _quadrants():
        c, r = C[rows, cols], R[rows, cols]
        # the inverse of target._mosaic: mosaic (C, R) -> 0-based science pixel (xs, ys) of this CCD
        xs, ys = {3: (c, r), 4: (c - 2048, r), 2: (2047 - c, 4095 - r), 1: (4095 - c, 4095 - r)}[ccd]
        s = sector_wcs(int(sector), camera)[ccd].pixel_to_world(xs + 44, ys)    # 44 collateral columns
        ra[rows, cols], dec[rows, cols] = s.ra.deg, s.dec.deg
    return ra, dec


def _step(span, lines=4):
    """A round grid spacing (degrees) giving at most `lines` lines across `span` degrees."""
    for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 15, 20, 30, 45, 60, 90):
        if span / s <= lines:
            return s
    return 90


def _label(level, step, ra):
    """'-56.5°' or '45°': only as many decimals as the spacing needs; RA from 0 to 360."""
    d = 0 if step >= 1 else 1 if round(step * 10, 6) == int(round(step * 10, 6)) else 2
    return f"{level % 360:.{d}f}°" if ra else f"{level:+.{d}f}°"


def grid_lines(sector, camera=3, window=None):
    """RA and Dec lines across the image: a list of (x, y, label), x and y in image pixels.

    Drawn CCD by CCD, so that lines stop at the gaps between CCDs, as the sky does on the detector.
    window: None for the whole image, or (x0, x1, y0, y1) for a zoom: lines only inside it, spaced
    for what it shows (a 20-pixel zoom is about 2 degrees across, so the lines are 0.5 or 1 degree apart).
    """
    window = None if window is None else tuple(int(v) for v in window)
    return _grid_lines(int(sector), int(camera), window)      # one cache entry, however the call is written


@lru_cache(maxsize=None)
def _grid_lines(sector, camera, window):
    import contourpy

    ra, dec = image_sky(sector, camera)
    x = np.arange(N, dtype=np.float64)
    x0, x1, y0, y1 = window or (0, N, 0, N)
    view = (slice(y0, y1), slice(x0, x1))
    dstep = _step(np.nanmax(dec[view]) - np.nanmin(dec[view]))
    # the RA span grows towards a pole (all 360 degrees around it), so the spacing grows with it
    rstep = _step(float(np.ptp(((ra[view] - np.nanmedian(ra[view]) + 180) % 360) - 180)))
    shortest = 10 if window is None else 3              # skip slivers at CCD corners
    out = []
    for _, rows, cols in _quadrants():
        rows = slice(max(rows.start, y0), min(rows.stop, y1))      # the part of this CCD in view
        cols = slice(max(cols.start, x0), min(cols.stop, x1))
        if rows.stop - rows.start < 2 or cols.stop - cols.start < 2:
            continue
        D, A = dec[rows, cols], ra[rows, cols]
        centre = A[A.shape[0] // 2, A.shape[1] // 2]
        A = ((A - centre + 180) % 360) - 180 + centre              # no jump at RA 0/360 inside one CCD
        A = np.ma.masked_where(np.abs(D) > 88, A)                  # RA has no meaning at the pole
        for Z, step, is_ra in ((D, dstep, False), (A, rstep, True)):
            gen = contourpy.contour_generator(x=x[cols], y=x[rows], z=Z)
            lo, hi = float(np.ma.min(Z)), float(np.ma.max(Z))
            for level in np.arange(math.ceil(lo / step) * step, hi, step):
                for line in gen.lines(level):
                    if len(line) > shortest:
                        out.append((line[:, 0], line[:, 1], _label(level, step, is_ra)))
    return out


def precompute(numbers, camera=3):
    """Grid lines for all these sectors now, so that no request has to wait for them.

    Sectors tess-point has no pointing for are skipped. Returns how many sectors are ready.
    """
    ready = 0
    for n in numbers:
        try:
            grid_lines(int(n), camera)
            ready += 1
        except (ValueError, NotImplementedError):          # e.g. a sector newer than tess-point's table
            pass
    return ready


def build(path=SNAPSHOT, camera=3):
    """Fit every sector in tess-point's table afresh and write them to `path` (the packaged copy)."""
    from tessbeam.target import tess_point

    t2p = tess_point(1)
    out = {}
    for n in [int(s) for s in t2p.TESS_Spacecraft_Pointing_Data.sectors]:
        out[str(n)] = _record(_pointing_key(t2p, n), camera, {c: _fit(t2p, n, c, camera) for c in (1, 2, 3, 4)})
        print(f"\rsector {n}", end="", flush=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)
    print(f"\nwrote {len(out)} sectors to {path}")


if __name__ == "__main__":
    build()
