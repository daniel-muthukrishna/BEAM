"""
Api for tessbeam. Users call one of these functions:

    predict(time, target=None)                              -> Prediction   one frame
    predict_series(start, end, step="30min", target=None)   -> Series       frames from start to end
    predict_sector(sector, step="6h", target=None)          -> Series       a whole sector
    compute_geometry(ra, dec, roll, times)                  -> dict of arrays (low level)

time: ISO string in UTC ("2019-05-09T05:15:00") or a float TJD (= JD_TDB - 2457000),
target: optional (RA, Dec) in degrees.

"""

import numbers
import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import numpy as np
from astropy.time import Time

from tessbeam import features, geometry, render, sectors, sky
from tessbeam.ephem import HorizonsEphemeris, moon_position
from tessbeam.target import Target, find as find_target

# The current model was trained with geometry computed at each FFI's nominal
# cadence time, ~1248 s before the 30-min FFI's mid-exposure. To predict the
# frame centred on time t, evaluate the geometry at t - GEOM_LEAD_S.
GEOM_LEAD_S = 1248.0
TJD_ZERO = 2457000.0

_EPHEM = None
_MODEL = None


def get_ephemeris():
    global _EPHEM
    if _EPHEM is None:
        _EPHEM = HorizonsEphemeris()
    return _EPHEM


def get_model(model_dir=None, device=None):
    """Load the model once per process (importing torch takes a while)."""
    global _MODEL
    if _MODEL is None:
        from tessbeam.model import DEFAULT_MODEL_DIR, StrayLightModel
        _MODEL = StrayLightModel(model_dir or DEFAULT_MODEL_DIR, device=device)
    return _MODEL


def to_tjd(t):
    """ISO UTC string or TJD number -> TJD (TDB)."""
    if isinstance(t, (int, float, np.integer, np.floating)):
        return float(t)
    t = str(t).strip()
    if re.fullmatch(r"[-+]?\d+(\.\d*)?", t):
        return float(t)
    try:
        return float(Time(t, scale="utc").tdb.jd - TJD_ZERO)
    except ValueError:
        raise ValueError(f"can't read time {t!r}; use ISO UTC like 2019-05-09T05:15:00, or a TJD number")


def tjd_to_utc(tjd):
    return Time(np.asarray(tjd) + TJD_ZERO, format="jd", scale="tdb", precision=0).utc.isot   # to the nearest second


def step_days(step):
    """'30min', '2h', '1d', '600s', or a plain number (minutes) -> days (> 0)."""
    if isinstance(step, (numbers.Real, np.number)):
        days = float(step) / 1440.0
    else:
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(s|sec|m|min|h|hr|hour|d|day)?s?\s*", str(step).lower())
        if not m:
            raise ValueError(f"can't parse step {step!r}; use e.g. '30min', '2h', '1d'")
        unit = {"s": 86400, "sec": 86400, "m": 1440, "min": 1440, "h": 24, "hr": 24,
                "hour": 24, "d": 1, "day": 1}[m.group(2) or "min"]
        days = float(m.group(1)) / unit
    if not np.isfinite(days) or days <= 0:
        raise ValueError(f"step must be positive, got {step!r}")
    return days


def _camera(camera):
    """Validate the camera number (1-4) and return it as an int."""
    try:
        c = float(camera)
    except (TypeError, ValueError):
        raise ValueError(f"camera must be 1, 2, 3 or 4, got {camera!r}")
    if c != int(c) or int(c) not in (1, 2, 3, 4):
        raise ValueError(f"camera must be 1, 2, 3 or 4, got {camera!r}")
    return int(c)


def _geometry(ra, dec, roll, tjd):
    """Geometry for frames centred on tjd, plus a mask of frames with a TESS position."""
    jd_geom = tjd + TJD_ZERO - GEOM_LEAD_S / 86400.0
    tess = get_ephemeris().position(jd_geom)
    covered = ~np.isnan(tess).any(axis=1)
    if not covered.any():
        raise ValueError(f"no TESS position for the requested time(s): "
                         f"{get_ephemeris().why_missing(jd_geom[0])}")
    g = geometry.emi(ra, dec, roll, tess[covered], moon_position(jd_geom[covered]))
    return g, covered


def compute_geometry(ra, dec, roll, times):
    """Geometry values (see tessbeam.geometry.emi) for frames centred on `times`."""
    tjd = np.atleast_1d([to_tjd(t) for t in np.atleast_1d(times)]).astype(np.float64)
    g, covered = _geometry(ra, dec, roll, tjd)
    if not covered.all():
        i = int(np.argmin(covered))
        raise ValueError(f"no TESS position for {tjd_to_utc(tjd[i])[:16]} UTC: "
                         f"{get_ephemeris().why_missing(tjd[i] + TJD_ZERO - GEOM_LEAD_S / 86400.0)}")
    return g


def _angle_keys(camera):
    c = camera
    return ["ED", "MD", "Eel", "Eaz", "Mel", "Maz", f"E{c}el", f"E{c}az", f"M{c}el", f"M{c}az"]


@dataclass
class Series:
    images: np.ndarray                 # (T, 256, 256) e-/s per pixel
    tjd: np.ndarray                    # (T,) frame mid-times, TJD (TDB)
    times_utc: List[str]
    ra: float                          # spacecraft pointing used (the sector's), deg
    dec: float
    roll: float
    camera: int
    seed: int
    angles: Dict[str, np.ndarray]      # geometry values for this camera, (T,) each
    out_of_range: Optional[List[List[str]]] = None   # per frame: features outside the training range (kept, not shown)
    skipped: Optional[List[str]] = None              # requested times with no TESS position (UTC)
    sector: Optional[int] = None       # None only for a low-level run with a hand-given pointing
    planned: bool = False              # sector not flown yet: provisional pointing, dates good to ~a day
    target: Optional[Target] = None    # where the requested sky position falls in this image

    def __post_init__(self):
        n = len(self.tjd)
        if self.out_of_range is None:
            self.out_of_range = [[] for _ in range(n)]
        if self.skipped is None:
            self.skipped = []
        lengths = {len(self.images), len(self.times_utc), len(self.out_of_range),
                   *(len(v) for v in self.angles.values())}
        if lengths != {n}:
            raise ValueError(f"Series fields have inconsistent frame counts: {sorted(lengths)} vs {n}")

    @property
    def target_flux(self):
        """Predicted stray light (e-/s per pixel) in the 16x16-pixel block holding the target, per frame."""
        return None if self.target is None else self.images[:, self.target.iy, self.target.ix]

    @property
    def marker(self):
        """Where to draw the red cross: the target's (x, y) in the image, or None."""
        return None if self.target is None else (self.target.x, self.target.y)

    @property
    def marker_label(self):
        """The target's RA/Dec, written beside the cross."""
        return None if self.target is None else f"RA {self.target.ra:.3f}°\nDec {self.target.dec:+.3f}°"

    @property
    def zoom(self):
        """The block of image pixels shown in the zoom around the target, or None without one."""
        return None if self.target is None else render.zoom_window(self.target.ix, self.target.iy)

    def zoom_titles(self):
        z = render.ZOOM
        return [t.split("\n")[0] + f"\nzoom: {z} x {z} pixels around the target ({z * 16} TESS pixels, "
                f"{z * 16 * 21 / 3600:.1f}° across)" for t in self.titles()]

    @property
    def grid(self):
        """RA/Dec lines for this sector's image (tessbeam.sky), or None if they can't be made."""
        if self.sector is None:
            return None
        return self._grid()

    @property
    def zoom_grid(self):
        """Finer RA/Dec lines for the zoom around the target, or None."""
        return None if self.target is None else self._grid(self.zoom)

    def _grid(self, window=None):
        try:
            return sky.grid_lines(self.sector, self.camera, window)
        except (ImportError, ValueError, NotImplementedError):   # no tess-point, or no pointing for it
            return None

    def titles(self):
        c = self.camera
        where = "" if self.sector is None else f"sector {self.sector}{' (planned)' if self.planned else ''}  "
        return [f"{u[:16]} UTC  {where}cam {c}\nED {ed:.1f} Re   Earth {e:.0f}° / Moon {m:.0f}° from boresight"
                for u, ed, e, m in zip(self.times_utc, self.angles["ED"], self.angles[f"E{c}el"], self.angles[f"M{c}el"])]

    def table(self):
        t = {"TJD": self.tjd, "UTC": np.array(self.times_utc)}
        t.update({k: np.asarray(v) for k, v in self.angles.items()})
        if self.target is not None:
            t["TARGFLUX"] = np.asarray(self.target_flux, dtype=np.float32)     # e-/s per pixel
        return t

    def header(self):
        h = {"RA_PNT": self.ra, "DEC_PNT": self.dec, "ROLL_PNT": self.roll, "CAMERA": self.camera,
             "SEED": self.seed, "GEOMLEAD": (GEOM_LEAD_S, "s, geometry epoch before frame mid-time")}
        if self.sector is not None:
            h["SECTOR"] = self.sector
            h["PLANNED"] = (self.planned, "sector not flown yet: provisional pointing")
            h["TBLDATE"] = (sectors.built()[:19], "sector table build time (UTC)")
        if self.target is not None:
            t = self.target
            h.update({"TARG_RA": (t.ra, "deg"), "TARG_DEC": (t.dec, "deg"),
                      "TARGCAM": t.camera, "TARGCCD": t.ccd,
                      "TARGCOL": (round(t.col, 3), "FFI column: 1-based, 44 collateral included"),
                      "TARGROW": (round(t.row, 3), "FFI row: 1-based (tess-point, aberrated)"),
                      "TARGX": (round(t.x, 3), "x in this image, 0-based (ds9 shows +1)"),
                      "TARGY": (round(t.y, 3), "y in this image, 0-based (ds9 shows +1)")})
        return h

    def to_gif(self, path, fps=8):
        return render.to_gif(self.images, self.titles(), path, fps=fps, marker=self.marker,
                             grid=self.grid, label=self.marker_label)

    def to_fits(self, path):
        return render.to_fits(self.images, self.table(), self.header(), path)

    def to_zoom_gif(self, path, fps=8):
        """GIF of the 20 x 20 pixels around the target, on their own colour scale (needs a target)."""
        if self.target is None:
            raise ValueError("a zoom needs a target: give its RA and Dec")
        return render.to_gif(self.images, self.zoom_titles(), path, fps=fps, marker=self.marker,
                             grid=self.zoom_grid, label=self.marker_label, window=self.zoom)

    def to_lightcurve(self, path):
        """PNG of the stray light at the target, frame by frame (needs a target)."""
        if self.target is None:
            raise ValueError("a light curve needs a target: give its RA and Dec")
        t = self.target
        title = (f"Stray light at RA {t.ra:.4f}, Dec {t.dec:.4f} "
                 f"(camera {t.camera}, CCD {t.ccd}), sector {self.sector}")
        return render.to_lightcurve(self.times_utc, self.target_flux, path, title)

    def frame(self, i):
        return Prediction(image=self.images[i], tjd=float(self.tjd[i]), time_utc=self.times_utc[i],
                          ra=self.ra, dec=self.dec, roll=self.roll, camera=self.camera, seed=self.seed,
                          angles={k: float(v[i]) for k, v in self.angles.items()},
                          out_of_range=self.out_of_range[i],
                          sector=self.sector, planned=self.planned, target=self.target)


@dataclass
class Prediction:
    image: np.ndarray                  # (256, 256) e-/s per pixel
    tjd: float
    time_utc: str
    ra: float
    dec: float
    roll: float
    camera: int
    seed: int
    angles: Dict[str, float]
    out_of_range: List[str]
    sector: Optional[int] = None
    planned: bool = False
    target: Optional[Target] = None

    @property
    def target_flux(self):
        """Predicted stray light (e-/s per pixel) in the 16x16-pixel block holding the target."""
        return None if self.target is None else float(self.image[self.target.iy, self.target.ix])

    def _series(self):
        return Series(images=self.image[None], tjd=np.array([self.tjd]), times_utc=[self.time_utc],
                      ra=self.ra, dec=self.dec, roll=self.roll, camera=self.camera, seed=self.seed,
                      angles={k: np.array([v]) for k, v in self.angles.items()},
                      out_of_range=[self.out_of_range],
                      sector=self.sector, planned=self.planned, target=self.target)

    def to_png(self, path):
        s = self._series()
        return render.to_png(self.image, path, s.titles()[0], marker=s.marker, grid=s.grid, label=s.marker_label)

    def to_zoom_png(self, path):
        """PNG of the 20 x 20 pixels around the target, on their own colour scale (needs a target)."""
        s = self._series()
        if s.target is None:
            raise ValueError("a zoom needs a target: give its RA and Dec")
        return render.to_png(self.image, path, s.zoom_titles()[0], marker=s.marker, grid=s.zoom_grid,
                             label=s.marker_label, window=s.zoom)

    def to_fits(self, path):
        s = self._series()
        return render.to_fits(self.image, s.table(), s.header(), path)


def predict(time, *, target=None, camera=3, seed=0):
    """One frame centred on `time`."""
    tjd = np.array([to_tjd(time)])
    return _predict(tjd, _one_sector(tjd), target, camera, seed).frame(0)


def predict_series(start, end, step="30min", *, target=None, camera=3, seed=0,
                   max_frames=None, progress=None):
    """Frames from start to end every `step`; they must all lie in one sector. Times with no
    TESS position (after the end of JPL's predicted orbit) are skipped and listed in Series.skipped."""
    tjd = _frame_times(to_tjd(start), to_tjd(end), step, max_frames)
    return _predict(tjd, _one_sector(tjd), target, camera, seed, progress, skip_missing=True)


def predict_sector(sector, step="6h", *, target=None, camera=3, seed=0,
                   max_frames=None, progress=None):
    """Frames over a whole sector, from its start to its end every `step`."""
    spans = sectors.spans(sector)                       # raises for an unknown or flagged sector
    tjd = _frame_times(spans[0][0], spans[-1][1], step, max_frames)
    rows = sectors.table()
    # sector 99 pauses for a week at another pointing (sector 1751): keep only this sector's frames
    mine = np.array([i >= 0 and rows[i].sector == int(sector) for i in sectors.find(tjd)])
    return _predict(tjd[mine], int(sector), target, camera, seed, progress, skip_missing=True)


def _frame_times(t0, t1, step, max_frames):
    """Times from t0 to t1 every `step` (TJD); refuses more than max_frames."""
    if t1 < t0:
        raise ValueError("end is before start")
    dt = step_days(step)
    # Count first, so a tiny step can't allocate. The 0.1 s of slack keeps the last frame of
    # "2 days at 1h": a JD held in one float is only good to 40 microseconds, and TDB drifts
    # against UTC by up to 30 microseconds a day, so such a span comes out a hair short.
    n = int(np.floor((t1 - t0 + 1e-6) / dt)) + 1
    if max_frames is not None and n > max_frames:
        fits = int(np.ceil((t1 - t0) * 1440 / max(max_frames - 1, 1)))
        raise ValueError(f"{n} frames requested; the limit is {max_frames} "
                         f"(use a step of at least {fits}min)")
    return t0 + dt * np.arange(n)


def _sector_dates(sector):
    """'sector N: start to end UTC', read from the table rows (works for a flagged sector too)."""
    rows = [r for r in sectors.table() if r.sector == sector]
    utc = lambda t: str(tjd_to_utc(t))[:16].replace("T", " ")
    return f"sector {sector}: {utc(rows[0].start)} to {utc(rows[-1].end)} UTC"


def _one_sector(tjd):
    """The sector that every one of these frame times lies in. The pointing is fixed per
    sector, so a request that leaves the sectors or spans two of them is refused, saying why."""
    idx = sectors.find(tjd)
    if (idx < 0).any():
        raise ValueError(sectors.why(tjd[int(np.argmax(idx < 0))]))
    rows = sectors.table()
    found = list(dict.fromkeys(rows[i].sector for i in idx))       # in time order, no repeats
    if len(found) > 1:
        raise ValueError(f"the requested times cover sectors {' and '.join(map(str, found))}, which have "
                         "different pointings; keep a series inside one sector ("
                         + "; ".join(_sector_dates(s) for s in found) + ")")
    return found[0]


def _predict(tjd, sector, target, camera, seed, progress=None, skip_missing=False):
    """Frames at `tjd`, all in `sector`: look up its pointing and the target's pixel, then run."""
    camera = _camera(camera)
    ra, dec, roll = sectors.pointing(sector)            # raises for a sector the table flags
    where = None
    if target is not None:                              # checked before the slow model load
        try:
            target_ra, target_dec = target
        except (TypeError, ValueError):
            raise ValueError(f"target must be a pair (RA, Dec) in degrees, got {target!r}")
        where = find_target(target_ra, target_dec, sector, camera, pointing=(ra, dec, roll))
    s = _run(ra, dec, roll, tjd, camera, seed, progress, skip_missing)
    return replace(s, sector=int(sector), planned=sectors.planned(sector), target=where)


def _run(ra, dec, roll, tjd, camera, seed, progress=None, skip_missing=False):
    camera = _camera(camera)
    ra, dec, roll, seed = float(ra), float(dec), float(roll), int(seed)
    g, covered = _geometry(ra, dec, roll, tjd)
    if not covered.all() and not skip_missing:
        compute_geometry(ra, dec, roll, tjd)          # raises, saying why there's no position
    skipped = [u[:19] for u in tjd_to_utc(tjd[~covered])] if (~covered).any() else []
    tjd = tjd[covered]
    model = get_model()
    if camera not in model.trained_cameras:
        raise ValueError(f"camera {camera} is not covered by this model "
                         f"(trained on camera {', '.join(map(str, sorted(model.trained_cameras)))})")
    ctx = features.context_vectors(g, camera, model.context_stats)
    images = model.sample(ctx, camera=camera, seed=seed, progress=progress)
    return Series(images, tjd, [u[:19] for u in tjd_to_utc(tjd)], ra, dec, roll, camera, seed,
                  {k: g[k] for k in _angle_keys(camera)},
                  features.out_of_range(ctx, model.model_dir, camera), skipped)
