"""
Command line:

    python -m tessbeam --time 2019-05-09T05:15:00 -o frame.png
    python -m tessbeam --start 2019-05-08T05:00 --end 2019-05-10T05:00 --step 1h -o movie.gif
    python -m tessbeam --sector 11 -o sector11.gif
    python -m tessbeam --time 2020-10-05T12:00 --target 67.136829 -56.515645 -o toi2645.png

The time decides the sector, and the sector decides the spacecraft pointing.
--target RA DEC (degrees) marks that sky position with a red cross and prints the
stray light predicted there.
Output format follows the file extension (.png/.fits for one time, .gif/.fits for a series).
"""

import argparse
import sys

import numpy as np

from tessbeam import api


def main():
    ap = argparse.ArgumentParser(description="Predict TESS stray light for a time, a time range or a sector")
    ap.add_argument("--time", help="frame mid-time: ISO UTC or TJD")
    ap.add_argument("--start", help="series start: ISO UTC or TJD")
    ap.add_argument("--end", help="series end: ISO UTC or TJD")
    ap.add_argument("--sector", type=int, help="whole sector, from its start to its end")
    ap.add_argument("--step", help="series step, e.g. 30min, 2h, 1d (default 30min; 6h for --sector)")
    ap.add_argument("--target", nargs=2, type=float, metavar=("RA", "DEC"),
                    help="sky position to mark (degrees)")
    ap.add_argument("--camera", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    #check the request and output format before the model load
    single, whole = args.time is not None, args.sector is not None
    ranged = bool(args.start or args.end)
    if single + whole + ranged != 1 or (ranged and not (args.start and args.end)):
        ap.error("give exactly one of --time, --sector, or --start with --end")
    out = args.output.lower()
    is_fits = out.endswith((".fits", ".fit", ".fits.gz", ".fit.gz"))
    if not is_fits and not out.endswith(".png" if single else ".gif"):
        ap.error(f"output must end in {'.png' if single else '.gif'} or .fits, got {args.output!r}")

    opts = dict(target=args.target, camera=args.camera, seed=args.seed)
    progress = lambda i, n: print(f"\r{i}/{n} frames", end="", flush=True)
    try:
        if single:
            r = api.predict(args.time, **opts)
        elif whole:
            r = api.predict_sector(args.sector, args.step or "6h", progress=progress, **opts)
        else:
            r = api.predict_series(args.start, args.end, args.step or "30min", progress=progress, **opts)
    except (ValueError, NotImplementedError, ImportError) as e:      #bad request
        sys.exit(f"error: {e}")
    if not single:
        print()
    if is_fits:
        r.to_fits(args.output)
    else:
        (r.to_png if single else r.to_gif)(args.output)

    print(f"sector {r.sector}{' (planned: provisional pointing and dates)' if r.planned else ''}: "
          f"pointing RA {r.ra:.4f}, Dec {r.dec:.4f}, roll {r.roll:.4f}")
    if r.target is not None:
        t, f = r.target, r.target_flux
        value = f"{f:.1f}" if single else f"{f.min():.1f} to {f.max():.1f} (median {np.median(f):.1f})"
        print(f"target: camera {t.camera}, CCD {t.ccd}, column {t.col:.2f}, row {t.row:.2f}; "
              f"stray light there {value} e-/s per pixel")
    if not single and r.skipped:
        print(f"skipped {len(r.skipped)} times with no TESS position "
              f"({r.skipped[0]} to {r.skipped[-1]} UTC)")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
