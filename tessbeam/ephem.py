"""
Where TESS and the Moon are at a given time.

TESS: JPL Horizons (target -95), queried through astroquery as geocentric state vectors
on equatorial ICRF axes, in 30-day chunks at 30-minute steps. Each chunk is cached on
disk and interpolated. Horizonscovers launch (2018-04-18) to the end of JPL's predicted orbit.
Moon: erfa.moon98 .
All positions are geocentric ICRS in km; times are JD (TDB).
"""

import os
import re
import time
import warnings
from datetime import datetime

import erfa
import numpy as np
import requests
from astropy.time import Time
from scipy.interpolate import CubicHermiteSpline

AU_KM = 149597870.7
DEFAULT_CACHE = os.environ.get(
    "TESSBEAM_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "tessbeam", "horizons"))


class HorizonsEphemeris:
    """TESS geocentric ICRF position (km) from JPL Horizons, cached on disk in 30-day chunks."""

    target = "-95"                     # TESS's Horizons id
    chunk_days = 30.0
    step_minutes = 30
    origin_jd = 2458200.5              # chunk grid origin (2018-03-21, before launch)
    version = "v1"                     # bump if the query settings change (invalidates the cache)
    refresh_days = 7.0                 # chunks reaching past today hold JPL *predictions*: re-fetch weekly

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or DEFAULT_CACHE
        self._chunks = {}              # chunk index -> (times, spline or None, message)

    # ---- public -----------------------------------------------------------------
    def position(self, jd_tdb):
        """Geocentric ICRS position of TESS (km), shape (N, 3); NaN where Horizons has none."""
        jd = np.atleast_1d(np.asarray(jd_tdb, dtype=np.float64))
        out = np.full((len(jd), 3), np.nan)
        ks = self._chunk_index(jd)
        for k in np.unique(ks):
            t, spline, _ = self._chunk(int(k))
            if spline is None:
                continue
            sel = (ks == k) & (jd >= t[0]) & (jd <= t[-1])
            out[sel] = spline(jd[sel])
        return out

    def why_missing(self, jd_tdb):
        """Reason a time has no position (it lies outside TESS's ephemeris at JPL)."""
        jd = float(np.atleast_1d(jd_tdb)[0])
        t, _, msg = self._chunk(int(self._chunk_index(np.array([jd]))[0]))
        utc = lambda x: Time(x, format="jd", scale="tdb").utc.isot[:16]
        if msg:                                  # Horizons had nothing for this whole chunk
            return msg
        if len(t) and jd < t[0]:
            return f"before the start of JPL's TESS ephemeris ({utc(t[0])} UTC)"
        if len(t) and jd > t[-1]:
            return f"after the end of JPL's TESS ephemeris ({utc(t[-1])} UTC)"
        return "JPL Horizons returned no TESS position for this time"

    # ---- internals --------------------------------------------------------------
    def _chunk_index(self, jd):
        return np.floor((jd - self.origin_jd) / self.chunk_days).astype(int)

    def _chunk(self, k):
        """Times, Hermite spline and message for chunk k: from memory, disk cache or Horizons."""
        if k in self._chunks:
            return self._chunks[k]
        path = os.path.join(self.cache_dir,
                            f"tess_{self.version}_{self.chunk_days:g}d_{self.step_minutes}m_{k}.npz")
        start, stop = self.origin_jd + k * self.chunk_days, self.origin_jd + (k + 1) * self.chunk_days
        cached = os.path.exists(path)
        predicted = stop > Time.now().jd               # chunk reaches past today
        stale = cached and predicted and (time.time() - os.path.getmtime(path) > self.refresh_days * 86400)
        if cached and not stale:
            t, pos, vel, msg = self._load(path)
        else:
            try:
                t, pos, vel, msg = self._download(start, stop)
            except RuntimeError as e:                  # offline: an old prediction beats nothing
                if not cached:
                    raise
                warnings.warn(f"{e}; using cached positions from {time.ctime(os.path.getmtime(path))}")
                t, pos, vel, msg = self._load(path)
            else:
                os.makedirs(self.cache_dir, exist_ok=True)
                tmp = path + ".tmp.npz"
                np.savez(tmp, jd_tdb=t, pos_km=pos, vel_kms=vel, message=np.array(msg))
                os.replace(tmp, path)                  # atomic: no half-written cache files
        spline = (CubicHermiteSpline(t, pos, vel * 86400.0, axis=0)   # velocity km/s -> km/day
                  if len(t) >= 2 else None)
        self._chunks[k] = (t, spline, msg)
        return self._chunks[k]

    @staticmethod
    def _load(path):
        d = np.load(path)
        return d["jd_tdb"], d["pos_km"], d["vel_kms"], str(d["message"])

    def _download(self, start, stop):
        """Fetch [start, stop]; if it runs past TESS's ephemeris, retry clipped to its limits."""
        msg = ""
        for _ in range(2):
            vectors, msg = self._query(start, stop)
            if vectors is not None and len(vectors):
                return vectors[:, 0], vectors[:, 1:4], vectors[:, 4:7], ""
            m = re.search(r"(prior to|after) A\.D\. (\S+ \S+) TD", msg)   # astroquery trims "TDB" to "TD"
            if not m:
                break
            bound = datetime.strptime(m.group(2).split(".")[0], "%Y-%b-%d %H:%M:%S")
            bound_jd = Time(bound, scale="tdb").jd
            if m.group(1) == "prior to" and bound_jd + 1 / 86400 < stop:
                start = bound_jd + 1 / 86400
            elif m.group(1) == "after" and bound_jd - 1 / 86400 > start:
                stop = bound_jd - 1 / 86400
            else:
                break
        return np.empty(0), np.empty((0, 3)), np.empty((0, 3)), msg

    def _query(self, start, stop):
        """One Horizons query -> (array of [jd, x, y, z, vx, vy, vz] in km, km/s, ""), or (None, message)."""
        from astroquery.jplhorizons import Horizons
        obj = Horizons(id=self.target, location="500@399",          # seen from Earth's centre
                       epochs={"start": f"JD{start:.6f}", "stop": f"JD{stop:.6f}",
                               "step": f"{self.step_minutes}m"})
        try:
            v = obj.vectors(refplane="earth",        # equatorial ICRF axes (astroquery's default is ecliptic)
                            aberrations="geometric",  # true positions, no light-time correction
                            cache=False)              # cached by us, per chunk
        except ValueError as e:                       # Horizons error text, e.g. outside the ephemeris
            msg = str(e).replace("Horizons Error:", "").strip()
            return None, msg + "B" if msg.endswith(" TD") else msg   # astroquery trims "TDB"
        except requests.RequestException as e:
            raise RuntimeError(f"could not get TESS positions from JPL Horizons ({e}); "
                               "this needs internet access to ssd.jpl.nasa.gov") from e
        col = lambda name: np.asarray(v[name], dtype=np.float64)
        jd = col("datetime_jd")                                          # TDB
        pos = np.column_stack([col(c) for c in ("x", "y", "z")]) * AU_KM             # AU -> km
        vel = np.column_stack([col(c) for c in ("vx", "vy", "vz")]) * AU_KM / 86400   # AU/day -> km/s
        return np.column_stack([jd, pos, vel]), ""


def moon_position(jd_tdb):
    """Geocentric position of the Moon (km), shape (N, 3)."""
    jd = np.atleast_1d(np.asarray(jd_tdb, dtype=np.float64))
    return erfa.moon98(2400000.5, jd - 2400000.5)["p"] * AU_KM
