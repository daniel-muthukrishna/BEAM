"""
Which sector TESS is in at a given time, and where the spacecraft points.

A snapshot of the table ships with the package (tessbeam/data/sectors.csv). TESS keeps
flying, so once the table in use is a week old it is rebuilt from the sources and kept as
<cache>/sectors.csv, where <cache> is $TESSBEAM_CACHE or ~/.cache/tessbeam. A refresh
that fails only warns. Set TESSBEAM_SECTOR_REFRESH=0 to never download.

"""

import os
import time
import warnings
from collections import namedtuple

import numpy as np
from astropy.time import Time

from tessbeam import build_sectors

TJD_ZERO = 2457000.0
# Real frames begin up to 9 minutes before the listed start of their sector, and the
# smallest real gap between two stretches is 16.7 min, so 10 min never reaches a neighbour.
START_TOL_MIN = 10
REFRESH_DAYS = 7       # rebuild the table once it is this old
RETRY_HOURS = 24       # after a refresh attempt (failed or not), wait this long before the next
SNAPSHOT = build_sectors.SNAPSHOT

Row = namedtuple("Row", "sector start end ra dec roll planned problem")   # start, end: TJD (TDB)

_TABLE = None          # tuple of Row in time order, loaded on first use
_BUILT = ""            # its "# built:" stamp
_BUILT_UNIX = 0.0      # the same moment as time.time() counts it, to tell the table's age cheaply
_LAST_TRY = None       # time.time() of this process's last refresh attempt


# ---- the table --------------------------------------------------------------------
def table():
    """Every stretch as a tuple of Row, in time order (refreshed first if it is old)."""
    refresh()
    return _TABLE


def built():
    """The "# built:" stamp (UTC) of the table in use."""
    table()
    return _BUILT


def numbers():
    """The sector numbers in the table, sorted."""
    return sorted({r.sector for r in table()})


def refresh(force=False):
    """Rebuild the table from the published sources. True if a new table is now in use.

    Without `force` this happens only when the table in use is more than REFRESH_DAYS old,
    this process has not tried within the last RETRY_HOURS, and TESSBEAM_SECTOR_REFRESH is
    not "0". It never raises: when anything goes wrong it warns and the table in use stays.
    """
    global _LAST_TRY
    if _TABLE is None:
        _use_newest()
    if not force:
        fresh = time.time() - _BUILT_UNIX <= REFRESH_DAYS * 86400
        tried_lately = _LAST_TRY is not None and time.time() - _LAST_TRY < RETRY_HOURS * 3600
        if fresh or tried_lately or os.environ.get("TESSBEAM_SECTOR_REFRESH") == "0":
            return False
    _LAST_TRY = time.time()
    try:
        rows = build_sectors.build(build_sectors.fetch(build_sectors.MIT_URL),
                                   build_sectors.fetch(build_sectors.SCHEDULE_URL))
        # A cut-off download still builds a table, so compare it with the one in use: no
        # sector may vanish, and none MIT listed before may fall back to being planned.
        new = {r["sector"]: r["planned"] for r in rows}
        lost = sorted({r.sector for r in _TABLE if r.sector not in new or (new[r.sector] and not r.planned)})
        if lost:
            raise ValueError(f"the downloaded sources no longer list sector {lost[0]}"
                             + (f" and {len(lost) - 1} more" if len(lost) > 1 else ""))
        build_sectors.write(rows, _cache_file(), build_sectors.utc_now())
        _use(*_load(_cache_file()))
    except Exception as e:      # offline, a source changed its layout, cache not writable, ...
        reason = " ".join(str(e).split())
        warnings.warn(f"tessbeam could not refresh its sector table ({reason}); "
                      f"still using the one built {_BUILT} UTC")
        return False
    return True


def _cache_file():
    # read the environment now, not at import: a program may set it after importing tessbeam
    cache = os.environ.get("TESSBEAM_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "tessbeam")
    return os.path.join(cache, "sectors.csv")


def _load(path):
    """A table file -> (tuple of Row in time order, built stamp)."""
    rows, stamp = build_sectors.read(path)
    Time(stamp, scale="utc")                               # a stamp that is not a date = a broken file
    # UTC strings -> TJD once, here, so that every later lookup only compares numbers
    tjd = lambda col: Time([r[col] for r in rows], scale="utc").tdb.jd - TJD_ZERO
    deg = lambda x: float("nan") if x is None else x       # no pointing (such a row has a problem)
    out = [Row(r["sector"], float(a), float(b), deg(r["ra_deg"]), deg(r["dec_deg"]), deg(r["roll_deg"]),
               bool(r["planned"]), r["problem"])
           for r, a, b in zip(rows, tjd("start_utc"), tjd("end_utc"))]
    return tuple(sorted(out, key=lambda r: r.start)), stamp


def _use(rows, stamp):
    """Make this table the one in use."""
    global _TABLE, _BUILT, _BUILT_UNIX
    _TABLE, _BUILT, _BUILT_UNIX = rows, stamp, Time(stamp, scale="utc").unix


def _use_newest():
    """Load the newer of the packaged snapshot and the refreshed copy in the cache folder."""
    _use(*_load(SNAPSHOT))
    copy = _cache_file()
    if not os.path.exists(copy):
        return
    try:
        rows, stamp = _load(copy)
    except Exception as e:                                 # a damaged copy must not stop tessbeam
        warnings.warn(f"tessbeam ignored its refreshed sector table {copy}: can't read it ({e})")
        return
    if stamp > _BUILT:                                     # ISO stamps sort like the dates they are
        _use(rows, stamp)


# ---- by sector number ---------------------------------------------------------------
def pointing(sector):
    """(ra, dec, roll) of the spacecraft boresight in degrees, tess-point convention."""
    r = _stretches(sector)[0]
    return r.ra, r.dec, r.roll


def spans(sector):
    """[(start_tjd, end_tjd), ...]: one per stretch (sector 99 has two)."""
    return [(r.start, r.end) for r in _stretches(sector)]


def planned(sector):
    """True if any part of the sector is still a plan (its dates come from the schedule)."""
    return any(r.planned for r in _stretches(sector))


def _stretches(sector):
    """The rows of one sector; ValueError if it is not a sector tessbeam can use."""
    try:
        n = int(float(sector))
        whole = n == float(sector)
    except (TypeError, ValueError, OverflowError):
        whole = False
    if not whole:
        raise ValueError(f"sector must be a whole number, got {sector!r}")
    rows = [r for r in table() if r.sector == n]
    if not rows:
        atlas = build_sectors.ATLAS_SECTOR
        usual = [s for s in numbers() if s != atlas]
        raise ValueError(f"sector {n} is not in tessbeam's sector table (sectors {usual[0]}-{usual[-1]}"
                         + (f" and {atlas}" if atlas in numbers() else "") + f"; built {built()[:10]})")
    for r in rows:
        if r.problem:
            raise ValueError(r.problem)
    return rows


# ---- by time --------------------------------------------------------------------------
def find(tjd):
    """Row index in table() for each time, as an int array; -1 where no stretch matches.

    A time matches the stretch it lies in. Failing that, it matches a stretch that starts
    within the next START_TOL_MIN minutes, unless that stretch is only planned. A flagged
    stretch is found like any other (at() is the one that refuses it).
    """
    rows = table()
    t = np.atleast_1d(np.asarray(tjd, dtype=np.float64))
    start = np.array([r.start for r in rows])
    end = np.array([r.end for r in rows])
    exact = np.array([not r.planned for r in rows])
    i = np.searchsorted(start, t, side="right") - 1        # the last stretch starting at or before t
    inside = (i >= 0) & (t <= end[np.maximum(i, 0)])
    nxt = np.minimum(i + 1, len(rows) - 1)                 # the stretch after it, if there is one
    early = ~inside & (i + 1 < len(rows)) & exact[nxt] & (start[nxt] - t <= START_TOL_MIN / 1440.0)
    return np.where(inside, i, np.where(early, nxt, -1))


def at(tjd):
    """The Row for ONE time. ValueError saying why if there is none, or if it is flagged."""
    i = int(find(float(tjd))[0])
    if i < 0:
        raise ValueError(why(tjd))
    row = table()[i]
    if row.problem:
        raise ValueError(row.problem)
    return row


def why(tjd):
    """The reason a time has no stretch, as a sentence for the user."""
    rows, t = table(), float(tjd)
    if not np.isfinite(t):
        return f"{tjd!r} is not a time"
    i = int(find(t)[0])
    if i >= 0:
        return f"{_utc(t)} UTC is in sector {rows[i].sector}"
    # the neighbours are the rows next to the time, not the sector numbers next to each
    # other: in time the order is ... 98, 99, 1751, 99, 100 ...
    before = [r for r in rows if r.end < t]
    after = [r for r in rows if r.start > t]
    if not before:
        return (f"{_utc(t)} UTC is before TESS's first sector "
                f"(sector {rows[0].sector} began {_utc(rows[0].start)} UTC)")
    if not after:
        return (f"{_utc(t)} UTC is after the last sector in tessbeam's table "
                f"(sector {rows[-1].sector} ends {_utc(rows[-1].end)} UTC; table built {built()[:10]})")
    a, b = before[-1], after[0]
    return (f"{_utc(t)} UTC is not inside a TESS sector: sector {a.sector} ended {_utc(a.end)} UTC "
            f"and sector {b.sector} begins {_utc(b.start)} UTC")


def _utc(tjd):
    """TJD -> 'YYYY-MM-DD HH:MM' in UTC, for messages."""
    # precision=0 rounds to the second before the cut, so a listed 06:40:00 never reads 06:39
    return Time(tjd + TJD_ZERO, format="jd", scale="tdb", precision=0).utc.iso[:16]


if __name__ == "__main__":
    flagged = sorted({r.sector for r in table() if r.problem})
    print(f"sector table built {built()} UTC: {len(table())} stretches, sectors "
          f"{', '.join(map(str, numbers()[:2]))}, ... {', '.join(map(str, numbers()[-2:]))}; "
          f"{sum(r.planned for r in table())} planned; flagged: {flagged or 'none'}")
