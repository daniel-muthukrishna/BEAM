"""
Where a sky position falls in the stray-light image.
    locate(ra, dec, sector)            -> [(camera, ccd, col, row), ...]   every detector hit
    to_image(camera, ccd, col, row)    -> (x, y, ix, iy)                   position in the image
    find(ra, dec, sector, camera=3)    -> Target                           both steps, one camera

Pixel conventions:
  col, row   tess-point's detector pixel: 1-based, pixel centres at whole numbers, with the
             44 leading collateral columns counted, so the science pixels are columns
             45..2092 and rows 1..2048. The sector's mid-time aberration is applied
             (about 0.2 px from real stars; without it about 1 px).
  x, y       position in the 256x256 image, pixel centres at whole numbers. This is what
             matplotlib's imshow(origin='lower') uses, so ax.plot(x, y, 'r+') marks the target.
  ix, iy     images[..., iy, ix] is the image pixel (a 16x16 block of detector pixels)
             that holds the target.


"""

import math
import numbers
from dataclasses import dataclass

BIN = 16                    # detector pixels per image pixel, along each axis
POINTING_TOL_DEG = 5e-4     # pointing tables are quoted to 4 decimals; this allows for rounding


@dataclass(frozen=True)
class Target:
    ra: float               # deg, as given
    dec: float
    sector: int
    camera: int
    ccd: int
    col: float              # tess-point detector pixel (see the module docstring)
    row: float
    x: float                # position in the 256x256 image, pixel centres at whole numbers
    y: float
    ix: int                 # images[..., iy, ix] holds the target
    iy: int


def _is_number(v):
    return isinstance(v, numbers.Real) and math.isfinite(v)


def _check(ra, dec, sector):
    """Raise a ValueError naming the bad field.

    tess-point would not: it wraps an RA outside 0-360 and answers "not observed" for NaN,
    and either would hide a typo.
    """
    if not (_is_number(ra) and _is_number(dec)):
        raise ValueError(f"target RA and Dec must be numbers in degrees, got RA {ra!r}, Dec {dec!r}")
    if not 0 <= ra <= 360:
        raise ValueError(f"target RA must be between 0 and 360 degrees, got {ra}")
    if not -90 <= dec <= 90:
        raise ValueError(f"target Dec must be between -90 and +90 degrees, got {dec}")
    if not (_is_number(sector) and sector == int(sector)):
        raise ValueError(f"sector must be a whole number, got {sector!r}")


def _no_pointing(sector, tess_stars2px):
    """The error for a sector the installed tess-point does not know."""
    extra = [str(s) for s in tess_stars2px.TESS_Spacecraft_Pointing_Data.sectors
             if s > tess_stars2px.max_sector]                 # one-off pointings such as 1751
    covers = " and ".join([f"1-{tess_stars2px.max_sector}"] + extra)
    return ValueError(f"the installed tess-point has no pointing for sector {sector} "
                      f"(its table covers sectors {covers}); upgrade tess-point")


def tess_point(sector):
    """The tess-point module, once it is known to be installed and to have a pointing for `sector`."""
    try:
        import tess_stars2px        # here, not at the top: only targets and the sky grid need it
    except ImportError:
        raise ImportError("marking a target needs the tess-point package, which is not installed "
                          "(pip install tess-point)")
    if int(sector) not in [int(s) for s in tess_stars2px.TESS_Spacecraft_Pointing_Data.sectors]:
        raise _no_pointing(int(sector), tess_stars2px)
    return tess_stars2px


def _angle_gap(a, b):
    """Difference between two angles in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def locate(ra, dec, sector, pointing=None):
    """Every detector the target falls on in this sector: a list of (camera, ccd, col, row).

    Usually one entry; two in the thin strips where neighbouring cameras overlap; empty if
    the target is outside the field of view or in a gap between CCDs.

    pointing: optional (ra, dec, roll) the caller is using for this sector. tess-point
    places the target with its own built-in pointing, so if the two differ the mark would
    not belong to the image it is drawn on; that is reported as an error.
    """
    _check(ra, dec, sector)
    sector = int(sector)
    tess_stars2px = tess_point(sector)
    table = tess_stars2px.TESS_Spacecraft_Pointing_Data
    known = [int(s) for s in table.sectors]

    if pointing is not None:
        ours = tuple(pointing)
        if len(ours) != 3:
            raise ValueError(f"pointing must be (RA, Dec, roll) in degrees, got {pointing!r}")
        i = known.index(sector)
        theirs = (float(table.ras[i]), float(table.decs[i]), float(table.rolls[i]))
        # All three are compared ignoring whole turns: RA and roll need it (a roll may be
        # quoted as -148 or 212), and it changes nothing for Dec, which never wraps.
        if not all(_angle_gap(a, b) <= POINTING_TOL_DEG for a, b in zip(ours, theirs)):
            show = lambda p: "RA {:.4f}, Dec {:.4f}, roll {:.4f}".format(*p)
            raise ValueError(f"the sector {sector} pointing in use ({show(ours)}) and the installed "
                             f"tess-point's ({show(theirs)}) disagree; one of them is out of date")

    try:
        # never pass scInfo: tess-point then ignores trySector and searches every sector
        out = tess_stars2px.tess_stars2px_function_entry(0, ra, dec, trySector=sector, aberrate=True)
    except SystemExit:              # how tess-point reacts to a sector it does not know
        raise _no_pointing(sector, tess_stars2px)
    # a target tess-point cannot see comes back as one row of -1, not as an error
    return [(int(cam), int(ccd), float(col), float(row))
            for sec, cam, ccd, col, row in zip(*out[3:8]) if sec != -1]


def _mosaic(ccd, xs, ys):
    """Science pixel (xs, ys) of one CCD, 0-based -> (column, row) in the 4096x4096 mosaic.

    CCDs 3 and 4 sit in the mosaic as they are read out. CCDs 2 and 1 went in through
    np.flip, which reverses both axes, so their pixels count down from the far corner.
    """
    if ccd == 3:
        return xs, ys
    if ccd == 4:
        return 2048 + xs, ys
    if ccd == 2:
        return 2047 - xs, 4095 - ys
    if ccd == 1:
        return 4095 - xs, 4095 - ys
    raise ValueError(f"CCD must be 1, 2, 3 or 4, got {ccd!r}")


def to_image(camera, ccd, col, row):
    """tess-point's (camera, ccd, col, row) -> (x, y, ix, iy) in the 256x256 image."""
    if camera != 3:
        raise NotImplementedError("the mosaic layout is only verified for camera 3 so far")
    xs, ys = col - 45.0, row - 1.0          # 0-based science pixel, centres at whole numbers

    # tess-point accepts positions up to half a pixel off the silicon; those are pinned to
    # the CCD's edge so the mark stays inside the image.
    C, R = _mosaic(ccd, min(max(xs, -0.5), 2047.5), min(max(ys, -0.5), 2047.5))
    # +0.5 measures from the mosaic's outer edge, where binning is a plain division;
    # -0.5 goes back to counting from the centre of the first image pixel.
    x, y = (C + 0.5) / BIN - 0.5, (R + 0.5) / BIN - 0.5

    # The index is worked out in whole pixels inside the target's own CCD. Rounding x and y
    # instead would give -1, 256 or the neighbouring CCD for a target in that half-pixel band.
    px = min(max(math.floor(xs + 0.5), 0), 2047)
    py = min(max(math.floor(ys + 0.5), 0), 2047)
    C, R = _mosaic(ccd, px, py)
    return float(x), float(y), C // BIN, R // BIN


def find(ra, dec, sector, camera=3, pointing=None):
    """The target on one camera's image, or an error that says where it is instead."""
    hits = locate(ra, dec, sector, pointing)
    sector = int(sector)
    if not hits:
        raise ValueError(f"target RA {ra:.4f}, Dec {dec:.4f} is not on any TESS detector in sector "
                         f"{sector} (outside the field of view, or not located on a science pixel)")
    here = [h for h in hits if h[0] == camera]
    if not here:
        where = " and ".join(f"camera {cam} (CCD {ccd})" for cam, ccd, _, _ in hits)
        raise NotImplementedError(f"target RA {ra:.4f}, Dec {dec:.4f} is on {where} in sector "
                                  f"{sector}; only camera {camera} is implemented so far")
    cam, ccd, col, row = here[0]            # the CCDs of one camera never overlap: one hit
    x, y, ix, iy = to_image(cam, ccd, col, row)
    return Target(float(ra), float(dec), sector, cam, ccd, col, row, x, y, ix, iy)
