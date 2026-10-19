"""
Build the angle file used for training: Earth, Moon and Sun geometry for every FFI of an
existing angle file, computed at each frame's mid-exposure time (MIDTJD in the TICA
headers) with tessbeam's geometry code.
"""

import argparse
import os
import pickle
import re
import sys
from multiprocessing import Pool

import erfa
import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tessbeam import features, geometry, sectors
from tessbeam.ephem import AU_KM, HorizonsEphemeris, moon_position

TICA = "/pdo/qlp-data/tica-delivery"
TJD_ZERO = 2457000.0


def sector_of(orbit):
    return (int(orbit) - 7) // 2


def nominal_tjd(ffi, orbit):
    cadence = 1800 if int(orbit) <= 60 else 600
    return 1227.454802 + int(ffi) * cadence / 86400


def header_files(sector):
    folder = f"{TICA}/s{sector:04d}/cam1-ccd1"
    if not os.path.isdir(folder):
        folder = f"{TICA}/primary-mission/s{sector:04d}/cam1-ccd1"
    found = {}
    for name in os.listdir(folder):
        match = re.search(r"-(\d{8})-cam1-ccd1_", name)
        if match and name.endswith(".fits"):
            found[match.group(1)] = os.path.join(folder, name)
    return found


def read_midtjd(path):
    return fits.getval(path, "MIDTJD", ext=0)


def mid_exposure_times(ffis, orbits, per_sector, workers):
    """
    MIDTJD for every FFI. Its offset from the nominal cadence time drifts by well under a
    second across a sector, so read per_sector headers in each sector and fit a line.
    """
    sector = np.array([sector_of(o) for o in orbits])
    fin = ffis.astype(int)
    nominal = np.array([nominal_tjd(f, o) for f, o in zip(ffis, orbits)])

    picks, paths = {}, []
    for s in np.unique(sector):
        files = header_files(s)
        rows = [i for i in np.where(sector == s)[0] if ffis[i] in files]
        picks[s] = [rows[j] for j in np.unique(np.linspace(0, len(rows) - 1, per_sector).astype(int))]
        paths += [files[ffis[i]] for i in picks[s]]
    with Pool(workers) as pool:
        mids = iter(pool.map(read_midtjd, paths, chunksize=4))

    tjd = np.empty(len(ffis))
    worst = 0.0
    for s, rows in picks.items():
        lead = np.array([next(mids) for _ in rows]) - nominal[rows]
        fit = np.polyfit(fin[rows], lead, 1)
        worst = max(worst, np.abs(lead - np.polyval(fit, fin[rows])).max())
        tjd[sector == s] = nominal[sector == s] + np.polyval(fit, fin[sector == s])
    print(f"MIDTJD fitted per sector from {len(paths)} headers; largest fit residual {worst * 86400:.2f} s")
    return tjd


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def angles_for_sector(sector, tjd, ephemeris):
    jd = tjd + TJD_ZERO
    ra, dec, roll = sectors.pointing(sector)
    tess = ephemeris.position(jd)
    moon = moon_position(jd)
    sun = -erfa.epv00(2400000.5, jd - 2400000.5)[0]["p"] * AU_KM

    g = geometry.emi(ra, dec, roll, tess, moon)
    out = {}
    for camera in geometry.CAMERAS:
        out.update(features.training_angles(g, camera))

    sun_body = unit((sun - tess) @ geometry.body_matrix(ra, dec, roll).T)
    out["Sel"] = np.round(np.arcsin(np.clip(sun_body[:, 2], -1, 1)), 5)
    out["Saz"] = np.round(np.arctan2(sun_body[:, 1], sun_body[:, 0]), 5)
    out["Ephase"] = np.round(np.arccos(np.clip((unit(sun) * unit(tess)).sum(1), -1, 1)), 5)
    out["Mphase"] = np.round(np.arccos(np.clip((unit(sun - moon) * unit(tess - moon)).sum(1), -1, 1)), 5)
    return out


def main():
    parser = argparse.ArgumentParser(description="Recompute the training angle file at mid-exposure, with Sun angles")
    parser.add_argument("--old", required=True, help="existing angle file (FFI list, orbits, below_sunshade flags)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--nominal", action="store_true",
                        help="use the nominal cadence times the old file used, to check against it")
    parser.add_argument("--headers-per-sector", type=int, default=40)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    old = pickle.load(open(args.old, "rb"))
    ffis = np.array(sorted(old))
    orbits = np.array([int(old[f]["orbit"]) for f in ffis])
    if args.nominal:
        tjd = np.array([nominal_tjd(f, o) for f, o in zip(ffis, orbits)])
    else:
        tjd = mid_exposure_times(ffis, orbits, args.headers_per_sector, args.workers)

    ephemeris = HorizonsEphemeris()
    sector = np.array([sector_of(o) for o in orbits])
    new = {}
    for s in np.unique(sector):
        rows = np.where(sector == s)[0]
        values = angles_for_sector(s, tjd[rows], ephemeris)
        for i, row in enumerate(rows):
            ffi = ffis[row]
            entry = {"ffi": ffi, "orbit": old[ffi]["orbit"], "below_sunshade": old[ffi]["below_sunshade"],
                     "tjd": float(tjd[row])}
            entry.update({key: float(v[i]) for key, v in values.items()})
            new[ffi] = entry

    with open(args.out, "wb") as file:
        pickle.dump(new, file)
    print(f"wrote {len(new)} FFIs to {args.out}")


if __name__ == "__main__":
    main()
